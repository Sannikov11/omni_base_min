
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Union, List, Optional
from dataclasses import asdict, field
from omnichannel_model.base_classes.base_configs import BaseModelArtifactsClass, BaseModelConfigClass, final_dataclass

@final_dataclass
class OmnichannelModelConfig(BaseModelConfigClass):
    tab_cat_card_threshold: int = 20
    tab_cat_low_emb_size: int = 4
    tab_cat_high_emb_size: int = 16
    tab_num_emb_size: int = 8
    romashka_emb_size: int = 512
    romashka_hid_size: int = 256
    evk_emb_size: int = 512
    evk_hid_size: int = 256
    fm_ft_emb_size: int = 448
    fm_ft_hid_size: int = 256
    fm_no_ft_emb_size: int = 256
    fm_no_ft_hid_size: int = 128
    frida_emb_size: int = 1536
    prod_hier_hid_size: int = 512
    prod_desc_emb_size: int = 2560
    prod_desc_hid_size: int = 512
    prod_desc_emb_size: int = 2560
    prod_desc_hid_size: int = 512
    dcn_type: str = 'stacked'
    cross_layer_num: int = 3
    cross_type: str = 'lowrank'
    deep_type: str = 'densenet'
    dcn_tab_only: bool = False
    dropout_prob: float = 0.2
    use_emb_flags: Optional[Dict[str, bool]] = None
    emb_dropout: Optional[Dict[str, float]] = None
    tab_norm_type: str = 'source'


@final_dataclass
class OmnichannelModelArtifacts(BaseModelArtifactsClass):
    tab_num_bins: List[torch.Tensor]
    tab_cat_cardinalities: List[int]
    deep_kwargs: Dict[str, Union[int, float, bool, nn.Module]] = field(
        default_factory = lambda : {
        'num_layers': 4,
        'growth_rate': 64,
        'dropout_rate': 0.2,
        'activation': nn.GELU()
    })
    cross_kwargs: Dict[str, Union[str, int, float, bool, nn.Module]] = field(
        default_factory = lambda : {
            'rank': 64, 
            'num_experts': 4, 
            'use_nonlinear': False, 
            'moe_gate_type': 'sigmoid', 
            'moe_activation': nn.Tanh()
    })
    
    
# ---------------------------------------------------------------------------
# SequenceConfig — архитектурные размерности (должны совпадать с весами модели)
# ---------------------------------------------------------------------------

@dataclass
class SequenceConfig:
    """
    Размерности последовательностей EVK и ромашки.
    Хранится рядом с OmnichannelModelArtifacts — меняется вместе с весами.
    """
    evk_seq_len: int = 256
    evk_n_mods: int = 8
    romashka_seq_len: int = 2048
    romashka_n_mods: int = 7
    rom_pad_value: int = 0
    # Scoring-режим: ромашка и EVK приходят как готовые float32-эмбеддинги.
    # Если у ромашки и EVK разные размерности — задайте их явно через
    # romashka_scoring_emb_len / evk_scoring_emb_len; иначе оба возьмут scoring_emb_len.
    scoring_emb_len: int = 512
    romashka_scoring_emb_len: Optional[int] = None
    evk_scoring_emb_len: Optional[int] = None

    @property
    def evk_len(self) -> int:
        return self.evk_seq_len * self.evk_n_mods

    @property
    def romashka_len(self) -> int:
        return self.romashka_seq_len * self.romashka_n_mods

    def get_scoring_emb_len(self, key: str) -> int:
        """Возвращает размерность scoring-эмбеддинга для конкретного ключа."""
        override = {"romashka": self.romashka_scoring_emb_len,
                    "evk":      self.evk_scoring_emb_len}
        return override.get(key) or self.scoring_emb_len
