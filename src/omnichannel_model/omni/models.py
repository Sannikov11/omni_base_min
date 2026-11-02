from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnichannel_model.omni.blocks import *
from omnichannel_model.losses import *
from omnichannel_model.omni.configs import OmnichannelModelArtifacts, OmnichannelModelConfig
from omnichannel_model.base_classes.base_model import BaseModelClass

from omnichannel_model.omni.heads import BaseHead, CRHead, PairwiseHead, CTRHead
from omnichannel_model.omni.multitask import MultiTaskConfig, MultiTaskHead, MultiTaskLoss
from typing import Literal

DataFormat = Literal["raw", "precomputed"]

def migrate_legacy_state_dict(state_dict: dict, target_head: str = 'cr') -> dict:
    """
    Переименовывает старые ключи под новую структуру с heads:
      predict_ffn.* → heads.<target_head>.predict_ffn.*
      final_layer.* → heads.<target_head>.final_layer.*
    """
    legacy_prefixes = ('predict_ffn.', 'final_layer.')
    migrated = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in legacy_prefixes):
            migrated[f'heads.{target_head}.{k}'] = v
        else:
            migrated[k] = v
    
    return migrated

def migrate_old_core_to_new_state_dict(state_dict: dict) -> dict:  
    
    migrated = {}
    for k, v in state_dict.items():
        if k.startswith('dcn.'):
            new_key = 'core.'+'.'.join(k.split('.')[1:]) 
            migrated[new_key] = v
        else:
            migrated[k] = v
    return migrated
    
class OmnichannelModel(BaseModelClass):
    def __init__(self,
                 config: OmnichannelModelConfig,
                 artifacts: OmnichannelModelArtifacts,
                 dlt_model: nn.Module,
                 evk_model: nn.Module,
                 activation: nn.Module = nn.GELU(),
                 loss: nn.Module = None,
                 head_kinds: Optional[List[str]] = None,
                 mt_config: Optional[MultiTaskConfig] = None,
                 **kwargs) -> None:
        super().__init__(config, artifacts)
        self.config = config
        self.artifacts = artifacts
        self.dlt_model = dlt_model
        self.evk_model = evk_model
        
        assert self.config.tab_norm_type in ['feature', 'source'], 'tab_norm_type should be either feature or source'
        
        self._version = config.model_version
        self._model_name = config.model_name
        self.loss = loss
        self.activation = activation
        
        
        if config.use_emb_flags is None:
            self.config.use_emb_flags = {
                'romashka_emb': True,
                'evk_emb': True,
                'prod_hier_emb': True, 
                'prod_desc_emb': True
            }
       
        if config.emb_dropout is None:
            self.config.emb_dropout = {
                'ple': 0.15,
                'low_cat': 0.3,
                'high_cat': 0.1,
                'romashka_emb': 0.05,
                'evk_emb': 0.05,
                'prod_hier_emb': 0.1, 
                'prod_desc_emb': 0.1
            }
            
        # -------------Инициализация табличного блока-------------
        
        # считаем маски и оффсеты для эмбеддинг слоев категориальных фичей
        self.tab_cat_low_card_mask = torch.tensor(artifacts.tab_cat_cardinalities) <= config.tab_cat_card_threshold
        self.tab_cat_high_card_mask = torch.tensor(artifacts.tab_cat_cardinalities) > config.tab_cat_card_threshold
        self.tab_cat_low_card = torch.tensor(artifacts.tab_cat_cardinalities)[self.tab_cat_low_card_mask]
        self.tab_cat_low_offset = torch.cat([torch.tensor([0]), self.tab_cat_low_card[:-1]], dim=0).cumsum(0)
        self.tab_cat_high_card = torch.tensor(artifacts.tab_cat_cardinalities)[self.tab_cat_high_card_mask]
        self.tab_cat_high_offset = torch.cat([torch.tensor([0]), self.tab_cat_high_card[:-1]], dim=0).cumsum(0)
 
        # Создаем слои для табличных данных
        self.ple = PiecewiseLinearEmbeddings(artifacts.tab_num_bins, d_embedding=config.tab_num_emb_size)
        self.ple_dropout = nn.Dropout1d(config.emb_dropout['ple'])
        
        self.low_cat_emb = nn.Embedding(sum(self.tab_cat_low_card), config.tab_cat_low_emb_size)
        nn.init.normal_(self.low_cat_emb.weight, mean=0.0, std=1/config.tab_cat_low_emb_size)
        self.low_cat_norm = nn.LayerNorm(config.tab_cat_low_emb_size)
        self.low_cat_dropout = nn.Dropout1d(config.emb_dropout['low_cat'])
        
        self.high_cat_emb = nn.Embedding(sum(self.tab_cat_high_card), config.tab_cat_high_emb_size)
        nn.init.normal_(self.high_cat_emb.weight, mean=0.0, std=1/config.tab_cat_high_emb_size)
        self.high_cat_norm = nn.LayerNorm(config.tab_cat_high_emb_size)
        self.high_cat_dropout = nn.Dropout1d(config.emb_dropout['high_cat'])
        
        # считаем размерность табличных эмбеддингов
        self.tab_dim = (config.tab_cat_low_emb_size * self.tab_cat_low_card.shape[0]) 
        self.tab_dim += (config.tab_cat_high_emb_size * self.tab_cat_high_card.shape[0])
        self.tab_dim += (config.tab_num_emb_size * len(self.artifacts.tab_num_bins))
        # нормализуем эмбеддинги
        self.tab_norm = nn.LayerNorm(self.tab_dim)
        
        self.combined_dim = self.tab_dim
        
        # --------------Инициализация эмбеддингов--------------
        if config.use_emb_flags.get('romashka_emb', False):
            # Инициализируем токены пропущенного эмбеддинга
            self.romashka_miss = nn.Parameter(torch.zeros(config.romashka_emb_size))
            d_rsqrt = config.romashka_emb_size ** -0.5
            nn.init.uniform_(self.romashka_miss, -d_rsqrt, d_rsqrt)
            
            self.rom_projection = EmbeddingProjection(
                input_dim=config.romashka_emb_size, 
                output_dim=64, 
                hidden_dims=[config.romashka_hid_size], 
                dropout=config.emb_dropout['romashka_emb'], 
                activation=activation
                )
            rom_size = self.rom_projection.output_dim
            self.combined_dim += rom_size
        else:
            rom_size = 0
            
        if config.use_emb_flags.get('evk_emb', False):
            # Инициализируем токены пропущенного эмбеддинга
            self.evk_miss = nn.Parameter(torch.zeros(config.evk_emb_size))
            d_rsqrt = config.evk_emb_size ** -0.5
            nn.init.uniform_(self.evk_miss, -d_rsqrt, d_rsqrt)
            
            self.evk_projection = EmbeddingProjection(
                input_dim=config.evk_emb_size, 
                output_dim=64, 
                hidden_dims=[config.evk_hid_size], 
                dropout=config.emb_dropout['evk_emb'], 
                activation=activation
                )
            evk_size = self.evk_projection.output_dim
            self.combined_dim += evk_size
        else:
            evk_size = 0
            
        if config.use_emb_flags.get('prod_hier_emb', False):
            # Инициализируем токены пропущенного эмбеддинга
            self.prod_hier_miss = nn.Parameter(torch.zeros(config.frida_emb_size))
            d_rsqrt = self.config.frida_emb_size ** -0.5
            nn.init.uniform_(self.prod_hier_miss, -d_rsqrt, d_rsqrt)
            
            self.prod_hier_projection = EmbeddingProjection(
                input_dim=config.frida_emb_size, 
                output_dim=256, 
                hidden_dims=[config.prod_hier_hid_size], 
                dropout=config.emb_dropout['prod_hier_emb'], 
                activation=activation
                )
            prod_hier_size = self.prod_hier_projection.output_dim
            self.combined_dim += prod_hier_size
        else:
            prod_hier_size = 0
            
        if config.use_emb_flags.get('prod_desc_emb', False):
            # Инициализируем токены пропущенного эмбеддинга
            self.prod_desc_miss = nn.Parameter(torch.zeros(config.prod_desc_emb_size))
            d_rsqrt = self.config.prod_desc_emb_size ** -0.5
            nn.init.uniform_(self.prod_desc_miss, -d_rsqrt, d_rsqrt)
            
            self.prod_desc_projection = EmbeddingProjection(
                input_dim=config.prod_desc_emb_size, 
                output_dim=256, 
                hidden_dims=[config.prod_desc_hid_size], 
                dropout=config.emb_dropout['prod_desc_emb'], 
                activation=activation
                )
            prod_desc_size = self.prod_desc_projection.output_dim
            self.combined_dim += prod_desc_size
        else:
            prod_desc_size = 0
        
        # -------------Инициализация DeepCrossNetwork-------------
        self.core = DeepCrossNetwork(
            input_dim=self.tab_dim if config.dcn_tab_only else self.combined_dim, 
            dcn_type=config.dcn_type, 
            cross_layer_num=config.cross_layer_num, 
            cross_type=config.cross_type, 
            cross_kwargs=artifacts.cross_kwargs, 
            deep_type=config.deep_type, 
            deep_kwargs=artifacts.deep_kwargs
        )
        self.combined_norm = nn.LayerNorm(self.core.output_dim + rom_size + evk_size + prod_hier_size + prod_desc_size if config.dcn_tab_only else self.combined_dim)

        self.encode_output_dim = (
            self.core.output_dim
            if not config.dcn_tab_only
            else self.core.output_dim + rom_size + evk_size + prod_hier_size + prod_desc_size
        )

        if head_kinds is None:
            head_kinds = [] if mt_config is not None else ['cr']
        if mt_config is not None and {'cr', 'ctr'} & set(head_kinds):
            raise ValueError("при mt_config головы 'cr'/'ctr' задаются через MultiTaskHead")
        heads = self.built_heads(head_kinds)
        self.heads = nn.ModuleDict(heads)
        self.mt_config = mt_config
        self.mt = MultiTaskHead(self.encode_output_dim, mt_config, activation=type(self.activation)()) if mt_config is not None else None
        self.mt_loss = MultiTaskLoss(mt_config) if mt_config is not None else None

    
    def built_heads(self, kinds: List[str]) -> Dict[str, BaseHead]:
        heads = {}
        for kind in kinds:
            if kind == 'cr':
                heads['cr'] = CRHead(
                    input_dim=self.encode_output_dim,
                    hidden_dims=[2048, 1024],
                    output_dim=512,
                    dropout=self.config.dropout_prob,
                    activation=type(self.activation)(),
                    loss_fn=self.loss,
                )
            elif kind == 'pairwise':
                heads['pairwise'] = PairwiseHead(
                    input_dim=self.encode_output_dim,
                    hidden_dims=[2048, 1024],
                    output_dim=512,
                    dropout=self.config.dropout_prob,
                    activation=type(self.activation)(),
                )
            elif kind == 'ctr':
                heads['ctr'] = CTRHead(
                    input_dim=self.encode_output_dim,
                    hidden_dims=[2048, 1024],
                    output_dim=512,
                    dropout=self.config.dropout_prob,
                    activation=type(self.activation)(),
                    loss_fn=self.loss,
                )
            else:
                raise ValueError(f"Unknown head kind: {kind}")
        return heads
    
    def encode_user(self,
               input_batch: Dict[str, torch.Tensor],
               data_format: DataFormat) -> torch.Tensor:
        """
        Прогон фич через все эмбеддинги + DCN.
        Возвращает (B, encode_output_dim) — то, что подаётся в любую голову.

        inference=False — режим обучения (romashka/evk идут через dlt_model/evk_model).
        inference=True  — режим инференса (в батче готовые эмбеддинги romashka/evk).
        """
        x_num = input_batch['num_feats']
        x_cat = input_batch['cat_feats']

        embeds_to_concat = []

        # ── Числовые ───────────────────────────────────────
        x_num = self.ple(x_num)
        if self.config.tab_norm_type == 'feature':
            x_num = self.ple_dropout(x_num).flatten(1)
        else:
            x_num = self.ple_dropout(x_num.flatten(1))

        # ── Категориальные ────────────────────────────────
        x_cat_low = self.low_cat_emb(
            x_cat[:, self.tab_cat_low_card_mask.to(x_cat.device)]
            + self.tab_cat_low_offset[None].to(x_cat.device)
        )
        if self.config.tab_norm_type == 'feature':
            x_cat_low = self.low_cat_dropout(x_cat_low).flatten(1)
        else:
            x_cat_low = self.low_cat_dropout(x_cat_low.flatten(1))

        x_cat_high = self.high_cat_emb(
            x_cat[:, self.tab_cat_high_card_mask.to(x_cat.device)]
            + self.tab_cat_high_offset[None].to(x_cat.device)
        )
        if self.config.tab_norm_type == 'feature':
            x_cat_high = self.high_cat_dropout(x_cat_high).flatten(1)
        else:
            x_cat_high = self.high_cat_dropout(x_cat_high.flatten(1))

        # ── Tab concat + опциональный DCN ─────────────────
        x_tab = torch.cat([x_num, x_cat_low, x_cat_high], dim=1)
        x_tab = self.tab_norm(x_tab)

        if self.config.dcn_tab_only:
            tab_emb = self.core(x_tab)
            embeds_to_concat.append(tab_emb)
        else:
            embeds_to_concat.append(x_tab)

        # ── Romashka ──────────────────────────────────────
        if self.config.use_emb_flags.get('romashka_emb', False):
            rom_miss = input_batch['romashka_mask']
            if data_format == "raw": 
                dlt_data = input_batch['romashka']
                dlt_ts = input_batch['romashka_ts']
                _, encoded_patches, masks = self.dlt_model(dlt_data, dlt_ts)
                patch_mask = masks[0]
                out_pos = patch_mask.argmin(-1) - 1
                x_rom = encoded_patches[
                    torch.arange(encoded_patches.shape[0], device=encoded_patches.device),
                    out_pos,
                ]
            else:
                x_rom = input_batch['romashka']

            x_rom = torch.where(rom_miss[..., None], self.romashka_miss[None, :], x_rom)
            x_rom = self.rom_projection(x_rom)
            embeds_to_concat.append(x_rom)

        # ── EVK ───────────────────────────────────────────
        if self.config.use_emb_flags.get('evk_emb', False):
            evk_miss = input_batch['evk_mask']
            if data_format == "raw": 
                evk_data = input_batch['evk']
                evk_positions = input_batch['evk_positions']
                evk_out, _ = self.evk_model(evk_data, evk_positions)
                x_evk = evk_out[:, 0, :]
            else:
                x_evk = input_batch['evk']

            x_evk = torch.where(evk_miss[..., None], self.evk_miss[None, :], x_evk)
            x_evk = self.evk_projection(x_evk)
            embeds_to_concat.append(x_evk)

        user_emb = torch.cat(embeds_to_concat, dim=1)

        return user_emb
    
    def encode_prod(self, input_batch: Dict[str, torch.Tensor], side: Optional[str] = None) -> torch.Tensor:
        desc_key = f'prod_desc_{side}' if side else 'prod_desc'
        mask_key = f'{desc_key}_mask' if side else 'prod_desc_mask'
        hier_key = f'prod_hier_{side}' if side else 'prod_hier'
             
        x_hier = input_batch[hier_key]

        embeds_to_concat = []

        # ── Prod desc ─────────────────────────────────────
        if self.config.use_emb_flags.get('prod_desc_emb', False):
            x_desc = input_batch[desc_key]
            desc_miss = input_batch[mask_key]

            x_desc = torch.where(desc_miss[..., None], self.prod_desc_miss[None, :], x_desc)
            x_desc = self.prod_desc_projection(x_desc)
            embeds_to_concat.append(x_desc)

        # ── Prod hier ─────────────────────────────────────
        x_hier = self.prod_hier_projection(x_hier)
        embeds_to_concat.append(x_hier)

        prod_emb = torch.cat(embeds_to_concat, dim=1)

        return prod_emb
    
    def process(self, 
                user_emb: torch.Tensor,
                prod_emb: torch.Tensor) -> torch.Tensor:
        
        combined_emb = torch.cat([user_emb, prod_emb], dim = 1)
        combined_emb = self.combined_norm(combined_emb)
        
        if not self.config.dcn_tab_only:
            combined_emb = self.core(combined_emb)

        return combined_emb

    def forward(
        self, 
        batch: Dict[str, torch.Tensor], 
        data_format: DataFormat,
        head_names: Optional[List[str]] = None,
        )-> Dict[str, torch.Tensor]:
        names = head_names if head_names is not None else list(self.heads.keys())
        want_mt = self.mt is not None and (head_names is None or 'mt' in head_names)
        names = [n for n in names if n != 'mt']
        outputs = {}
        
        user_emb = self.encode_user(batch, data_format=data_format)
        comb_pointwise = None
        comb_pairwise = None
        
        for name in names:
            head = self.heads[name]
            if head.kind == 'pairwise' and self.training:
                
                if comb_pairwise is None:
                    prod_pos_emb = self.encode_prod(batch, side = 'pos')
                    prod_neg_emb = self.encode_prod(batch, side = 'neg')

                    user_emb_double = torch.cat([user_emb, user_emb], dim = 0)
                    prod_emb_double = torch.cat([prod_pos_emb, prod_neg_emb], dim = 0)

                    comb_pairwise = self.process(user_emb_double, prod_emb_double)
                
                out_double = head(comb_pairwise).squeeze(-1)
                
                s_pos, s_neg = out_double.chunk(2, dim=0)
                
                outputs[name] = {
                        's_pos': s_pos,
                        's_neg': s_neg,
                    }
                
            else:
                if comb_pointwise is None:
                    prod_emb = self.encode_prod(batch, side = None)
                    comb_pointwise = self.process(user_emb, prod_emb)
                outputs[name] = head(comb_pointwise)
                
        if want_mt:
            if comb_pointwise is None:
                comb_pointwise = self.process(user_emb, self.encode_prod(batch, side=None))
            groups = batch.get('groups')
            channel = None if groups is None else groups[:, self.mt_config.channel_col]
            outputs['mt'] = self.mt(comb_pointwise, channel)

        return outputs
            

    def _resolve_head(self, head_name: Optional[str] = None) -> nn.Module:
        if head_name is None:
            if len(self.heads) == 1:
                return next(iter(self.heads.values()))
            raise ValueError(
                f"Модель имет несколько голов: {len(self.heads)}, "
                f"Доступны: {list(self.heads.keys())}"
            )
        if head_name not in self.heads:
            raise ValueError(
                f"Голова{head_name} не зарегистрирована,"
                f"Доступны: {list(self.heads.keys())}"
            )
        return self.heads[head_name]

    def calculate_mt_loss(self, outputs, targets, sample_weights=None):
        return self.mt_loss(outputs['mt'], targets, sample_weights)


    def calculate_loss(self,
                       outputs,
                       targets = None,
                       head_name: Optional[str] = None, 
                       sample_weights: torch.Tensor = None,) -> torch.Tensor:
        
        if head_name == 'mt':
            return self.calculate_mt_loss(outputs, targets, sample_weights).total

        head = self._resolve_head(head_name)
        if head_name is None:
            if len(self.heads) == 1:
                name = next(iter(self.heads.keys()))
            else:
                raise ValueError("Укажите head_name для calculate_loss, так как модель имеет несколько голов.")
        else:
            name = head_name
        head_outputs = outputs[name]
        return head.calculate_loss(outputs=head_outputs, targets=targets, sample_weights=sample_weights) # см ком-+мент к head.py про batch


    @torch.inference_mode()
    def predict(self, 
                batch: Dict[str, torch.Tensor],
                data_format: DataFormat, 
                head_names: Optional[Union[str, List[str]]] = None) -> Dict[str, torch.Tensor]:
        self.eval() 
        logits = self.forward(batch, data_format=data_format, head_names=head_names)
        preds = {}
        for name, out in logits.items():
            if name == 'mt':
                preds.update(self.mt.predict(out))
                continue

            head = self.heads[name]
            if head.kind == 'pointwise':
                preds[name] = head.predict_activation(out)
            else:
                preds[name] = out.squeeze(-1)
        return preds

