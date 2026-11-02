from typing import List, Optional, Literal

import warnings
import torch
import torch.nn as nn
from omnichannel_model.omni.utils import _check_bins
from omnichannel_model.base_classes.base_blocks import *

class _NLinear(nn.Module):
    """ N *separate* linear layers for N feature embeddings """
    def __init__(self, n: int, in_features: int, out_features: int, bias:bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()
   
    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)
   
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('_NLinear supports only inputs with exactly one batch dimension, so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING).')
        
        x = x.transpose(0,1)
        x = x @ self.weight
        x = x.transpose(0,1)
        if self.bias is not None:
            x = x + self.bias
        return x
    
class LinearEmbeddings(nn.Module):
    """ input: ``(*, n_features)``
        output: ``(*, n_features, d_embedding)``
    """
    def __init__(self, n_features: int, d_embedding: int) -> None:
        assert n_features > 0 and d_embedding > 0, f'n_features and d_embedding must be positive, however {n_features=}, {d_embedding=}'
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.bias = nn.Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()
       
    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addcmul(self.bias, self.weight, x[..., None])

class _PiecewiseLinearEncodingImpl(nn.Module):
    """ copied from rtdl-num-embeddings
        input: ``(*, n_features)``
        output: ``(*, n_features, max_n_bins)``
    """
    weight: torch.Tensor
    bias: torch.Tensor
    single_bin_mask: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    def __init__(self, bins: List[torch.Tensor]) -> None:
        assert len(bins) > 0
        super().__init__()
       
        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)
        
        self.max_n_bins = max_n_bins
        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))
       
        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer('single_bin_mask', single_bin_mask if single_bin_mask.any() else None)
       
        if all(len(x) == len(bins[0]) for x in bins):
            _m = None
        else:
            _m = torch.row_stack([torch.cat([torch.ones((len(x) - 1) - 1, dtype = torch.bool),
                                             torch.zeros(max_n_bins - (len(x) - 1), dtype = torch.bool),
                                             torch.ones(1, dtype = torch.bool)]) for x in bins])
        self.register_buffer('mask', _m)
       
        for i,bin_edges in enumerate(bins):
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
           
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
           
            self.weight[i, :n_bins[i] - 1] = w[:-1]
            self.bias[i, :n_bins[i] - 1] = b[:-1]
           
    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if self.max_n_bins > 1:
            x = torch.cat([
                            x[...,:1].clamp_max(1.0),
                            x[...,1:-1].clamp(0.0, 1.0),
                            (x[...,-1:].clamp_min(0.0) if self.single_bin_mask is None else torch.where(self.single_bin_mask[..., None], x[..., -1:], x[..., -1:].clamp_min(0.0)))
                            ], dim=-1)
        return x

class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, bins: List[torch.Tensor]) -> None:
        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl.mask is None else x[:, self.impl.mask]

class PiecewiseLinearEmbeddings(nn.Module):
    """Piecewise-linear embeddings.
 
    **Shape**
 
    - Input: ``(batch_size, n_features)``
    - Output: ``(batch_size, n_features, d_embedding)``
    """
 
    def __init__(
        self,
        bins: List[torch.Tensor],
        d_embedding: int,
        *,
        activation: bool = False,
        version: Literal[None, 'A', 'B'] = 'B',
        handle_missings: bool = True
    ) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
            d_embedding: the embedding size.
            activation: if True, the ReLU activation is additionally applied in the end.
            version: the preset for various implementation details, such as
                parametrization and initialization. See README for details.
        """
        if d_embedding <= 0:
            raise ValueError(
                f'd_embedding must be a positive integer, however: {d_embedding=}'
            )
        _check_bins(bins)
        if version is None:
            warnings.warn(
                'The `version` argument is not provided, so version="A" will be used'
                ' for backward compatibility.'
                ' See README for recommendations regarding `version`.'
                ' In future, omitting this argument will result in an exception.'
            )
            version = 'A'
 
        super().__init__()
        n_features = len(bins)
        # NOTE[DIFF]
        # version="B" was introduced in a different paper (about the TabM model).
        is_version_B = version == 'B'
 
        self.linear0 = (
            LinearEmbeddings(n_features, d_embedding) if is_version_B else None
        )
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            d_embedding,
            # For the version "B", the bias is already presented in self.linear0.
            bias=not is_version_B,
        )
        if is_version_B:
            # Because of the following line, at initialization,
            # the whole embedding behaves like a linear embedding.
            # The piecewise-linear component is incrementally learnt during training.
            nn.init.zeros_(self.linear.weight)
        self.activation = nn.ReLU() if activation else None
       
        if handle_missings: #getting a learnable embedding for every features missings
            self.missing_embeddings = nn.Parameter(torch.zeros(len(bins), d_embedding))
            d_rsqrt = d_embedding ** -0.5
            nn.init.uniform_(self.missing_embeddings, -d_rsqrt, d_rsqrt)
        else:
            self.missing_embeddings = None
 
    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        n_features = self.linear.weight.shape[0]
        d_embedding = self.linear.weight.shape[2]
        return torch.Size((n_features, d_embedding))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim != 2:
            raise ValueError(
                'For now, only inputs with exactly one batch dimension are supported.'
            )
        missing_mask = None
        if self.missing_embeddings is not None:
            missing_mask = torch.isnan(x)
            x = torch.where(missing_mask, torch.zeros_like(x), x)
       
        x_linear = None if self.linear0 is None else self.linear0(x)
 
        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)
        if self.activation is not None:
            x_ple = self.activation(x_ple)
           
        if missing_mask is not None:
            x_ple = torch.where(missing_mask[...,None], self.missing_embeddings[None, :, :], x_ple)
           
        return x_ple if x_linear is None else x_linear + x_ple
    
class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: nn.Module=nn.ReLU(), dropout_prob: float=0.2, norm: bool=True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        if norm:
            self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self,'norm'):
            x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float, activation: nn.Module = nn.GELU()):
        super(MLPNet, self).__init__()
        assert len(hidden_dims) > 0, "hidden_dims should be a non-empty list"
        self.layers = nn.ModuleList([])
        
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(MLPLayer(input_dim=current_dim, output_dim=hidden_dim, dropout_prob=dropout_rate, activation=activation, norm=True))
            current_dim = hidden_dim
        
        self.output_dim = current_dim
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x   
    
class AttentionFusion(nn.Module):
    def __init__(self, emb_dim_1=512, emb_dim_2=512) -> None:
        super().__init__()
        self.query = nn.Sequential(
        nn.Linear(emb_dim_1 + emb_dim_2, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 2),
        nn.Softmax(dim=-1)
        )
       
        self.combine = nn.Sequential(
        nn.Linear(emb_dim_1 + emb_dim_2, emb_dim_1),
        nn.ReLU(),
        nn.Dropout(0.1)
        )
   
    def forward(self, x_rom: torch.Tensor, x_evk: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([x_rom, x_evk], dim=-1)
       
        weights = self.query(concat)
       
        weighted = weights[:, 0].unsqueeze(1) * x_rom + weights[:, 1].unsqueeze(1) * x_evk
        combined = self.combine(concat)
       
        return weighted + combined * 0.5
    
class ModalGating(nn.Module):
    def __init__(self, tab_size: int = 256, rom_size: int = 512, evk_size: int = 512, prod_hier_size: int = 256, output_size: int = 256):
        super().__init__()
        self.tab_proj = nn.Linear(tab_size, output_size)
        self.rom_proj = nn.Linear(rom_size, output_size)
        self.evk_proj = nn.Linear(evk_size, output_size)
        self.prod_proj = nn.Linear(prod_hier_size, output_size)
       
        total_size = tab_size + rom_size + evk_size + prod_hier_size
        self.gate = nn.Sequential(
            nn.Linear(total_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )
   
    def forward(self, tab: torch.Tensor, rom: torch.Tensor, evk: torch.Tensor, prod_hier: torch.Tensor):
        tab_proj = self.tab_proj(tab)
        rom_proj = self.rom_proj(rom)
        evk_proj = self.evk_proj(evk)
        prod_proj = self.prod_proj(prod_hier)
       
        concat = torch.cat([tab, rom, evk, prod_hier], dim=-1)
        weights = self.gate(concat)
        output = (weights[:, 0:1] * tab_proj + weights[:, 1:2] * rom_proj + weights[:, 2:3] * evk_proj + weights[:, 3:4] * prod_proj)
       
        return output

class DenseNet(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, growth_rate: int, dropout_rate: float = 0.1, activation: nn.Module = nn.ReLU()):
        super(DenseNet, self).__init__()
        self.layers = nn.ModuleList([])
        
        current_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_dim, growth_rate, dropout_rate, activation))
            current_dim += growth_rate
        
        self.output_dim = current_dim
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EmbeddingProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1, activation: nn.Module = nn.GELU()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
 
        layers = []
        current_dim = input_dim
 
        if hidden_dims:
            for h_dim in hidden_dims:
                layer = [
                    nn.Linear(current_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    activation,
                    nn.Dropout(dropout)
                    ]
                layers.extend(layer)
                current_dim = h_dim
        
        layer = [nn.Linear(current_dim, output_dim), nn.LayerNorm(output_dim)]
        layers.extend(layer)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class CrossNet(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 cross_layer_num: int, 
                 layer_type: str,
                 rank: Optional[int], 
                 num_experts: Optional[int], 
                 use_nonlinear: Optional[bool], 
                 moe_gate_type: Optional[str] = 'softmax', 
                 moe_activation: Optional[nn.Module] = nn.Tanh()
                ):
        super().__init__()
        self.input_dim = input_dim
        self.cross_layer_num = cross_layer_num
        assert layer_type in ['matrix', 'lowrank', 'moe'], "layer_type should be one of ['matrix', 'lowrank', 'moe']"
        if layer_type != 'matrix':
            assert rank is not None, "for type LowRank and MoE, rank must be specified!"
            if layer_type == 'moe':
                assert num_experts is not None and use_nonlinear is not None and moe_gate_type is not None and moe_activation is not None, "For MoE you should specify num_experts, use_nonlinear, moe_gate_type, moe_activation"
        self.layers = nn.ModuleList()
 
        for _ in range(cross_layer_num):
            if layer_type == 'lowrank':
                layer = LowRankCrossLayer(input_dim, rank)
            elif layer_type == 'matrix':
                layer = CrossLayer(input_dim)
            elif layer_type == 'moe':
                layer = MoECrossLayer(input_dim, num_experts, rank, moe_gate_type, use_nonlinear, moe_activation)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer in self.layers:
            x = layer(x0, x)
        return x

class DeepCrossNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 dcn_type:str = 'parallel', 
                 cross_layer_num: int = 3,
                 cross_type: str = 'lowrank',
                 cross_kwargs: dict = {
                     'rank': 64, 
                     'num_experts': 4, 
                     'use_nonlinear': False, 
                     'moe_gate_type': 'sigmoid', 
                     'moe_activation': nn.Tanh()
                 },
                 deep_type: str = 'densenet', 
                 deep_kwargs: dict = {
                     'num_layers': 4, 
                     'growth_rate': 64, 
                     'dropout_rate': 0.2, 
                     'activation': nn.GELU()
                 }
                ):
        super().__init__()
        assert dcn_type in ['parallel', 'stacked'], "DCN type should be either parallel or stacked!"
        assert cross_type in ['matrix', 'lowrank', 'moe'], "Cross type should be either matrix, lowrank or moe!"
        assert deep_type in ['mlp', 'densenet'], "Deep type should be either mlp or densenet!"
        
        self.input_dim = input_dim
        self.dcn_type = dcn_type
        self.cross_type = cross_type
        self.deep_type = deep_type
        self.cross_net = CrossNet(input_dim=input_dim, cross_layer_num=cross_layer_num, layer_type=cross_type, **cross_kwargs)
        if self.deep_type == 'mlp':
            self.deep_net = MLPNet(input_dim, **deep_kwargs)
        elif self.deep_type == 'densenet':
            self.deep_net = DenseNet(input_dim, **deep_kwargs)
        self.output_dim = self.deep_net.output_dim
        if self.dcn_type == 'parallel':
            # crossnet output_dim is equal to input_dim
            self.output_dim += input_dim
    
    def forward(self, x):
        cross_output = self.cross_net(x)
        if self.dcn_type == 'parallel':
            deep_output = self.deep_net(x)
            return torch.cat([cross_output, deep_output], dim=1)
        elif self.dcn_type == 'stacked':
            deep_output = self.deep_net(cross_output)
            return deep_output
        
class AttentionFusionV2(nn.Module):
    def __init__(self, evk_dim, rom_dim, emb_dim, num_heads=4, dropout=0.1) -> None:
        super().__init__()

        self.evk_proj = nn.Linear(evk_dim, emb_dim)
        self.rom_proj = nn.Linear(rom_dim, emb_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln = nn.LayerNorm(emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, evk_proj, rom_proj) -> torch.Tensor:
        batch_size = evk_proj.shape[0]

        evk_p = self.evk_proj(evk_proj)
        rom_p = self.rom_proj(rom_proj)

        cls = self.cls_token.expand(batch_size, -1, -1)

        tokens = torch.cat([cls, evk_p.unsqueeze(1), rom_p.unsqueeze(1)], dim=1)

        attended, attn_weights = self.attention(
            tokens, tokens, tokens,
            need_weights=False,
            average_attn_weights=True
        )

        cls_out = self.ln(attended[:, 0, :] + cls.squeeze(1))

        fused = self.ln2(cls_out + self.ffn(cls_out))

        return fused

class CrossAttentionWithTabular(nn.Module):
    def __init__(self, tabular_dim, evk_dim, rom_dim, emb_dim=256, num_heads=4) -> None:
        super().__init__()

        self.query_proj = nn.Linear(tabular_dim, emb_dim)
        self.evk_key_proj = nn.Linear(evk_dim, emb_dim)
        self.evk_value_proj = nn.Linear(evk_dim, emb_dim)
        self.rom_key_proj = nn.Linear(rom_dim, emb_dim)
        self.rom_value_proj = nn.Linear(rom_dim, emb_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.ln = nn.LayerNorm(emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        self.ln2 = nn.LayerNorm(emb_dim)


    def forward(self, tabular_emb, evk_emb, rom_emb) -> torch.Tensor:
        batch_size = tabular_emb.shape[0]

        Q = self.query_proj(tabular_emb).unsqueeze(1)

        K_evk = self.evk_key_proj(evk_emb).unsqueeze(1)
        V_evk = self.evk_value_proj(evk_emb).unsqueeze(1)
        K_rom = self.rom_key_proj(rom_emb).unsqueeze(1)
        V_rom = self.rom_value_proj(rom_emb).unsqueeze(1)

        K = torch.cat([K_evk, K_rom], dim=1)
        V = torch.cat([V_evk, V_rom], dim=1)

        attended, attn_weights = self.cross_attention(
            Q, K, V,
            need_weights=False,
            average_attn_weights=True
        )

        fused = attended.squeeze(1)

        fused = self.ln(fused)

        fused = self.ln2(fused + self.ffn(fused))

        return fused