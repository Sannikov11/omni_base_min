import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Базовые компоненты модели трансформера.
"""

class AttentionModule(nn.Module):
    def __init__(self, n_embed, n_head, drop_rate=0.1):
        super().__init__()
        assert n_embed % n_head == 0
        self.q_layer = nn.Linear(n_embed, n_embed)
        self.k_layer = nn.Linear(n_embed, n_embed)
        self.v_layer = nn.Linear(n_embed, n_embed)
        self.proj_layer = nn.Linear(n_embed, n_embed)
        self.attn_dropout = nn.Dropout(drop_rate)
        self.resid_dropout = nn.Dropout(drop_rate)
        self.drop_rate = drop_rate
        self.n_head = n_head
        
    def forward(self, x_0, x_1, attn_mask = None):
        batch, seq_0, dim = x_0.size()
        _, seq_1, _ = x_1.size()
        
        head_dim = dim // self.n_head
        q = self.q_layer(x_0).view(batch, seq_0, self.n_head, head_dim).transpose(1,2)
        k = self.k_layer(x_1).view(batch, seq_1, self.n_head, head_dim).transpose(1,2)
        v = self.v_layer(x_1).view(batch, seq_1, self.n_head, head_dim).transpose(1,2)
        
        att_score = (q @ k.transpose(-2,-1)) * (1. / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att_score = att_score + attn_mask
        att_score = self.attn_dropout(F.softmax(att_score, dim=-1))
        
        out = att_score @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_0, dim)
        out = self.resid_dropout(self.proj_layer(out))
    
        return out


class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, drop_rate=0.1, mlp_linear_coef=4, **kwargs):
        super().__init__()
        self.ln_attn = nn.LayerNorm(n_embed)
        self.self_attn = AttentionModule(n_embed=n_embed, n_head=n_head, drop_rate=drop_rate)
        self.ln_mlp = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, mlp_linear_coef * n_embed), 
            nn.GELU(),
            nn.Linear(mlp_linear_coef * n_embed, n_embed),
            nn.Dropout(drop_rate),
        )
        
    def forward(self, x, attn_mask=None):
        
        mha = self.self_attn(
            self.ln_attn(x), 
            self.ln_attn(x), 
            attn_mask = attn_mask
        )
        mha = mha[0]
        x = x + mha
        x = x + self.mlp(self.ln_mlp(x))

        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embed, n_head, drop_rate=0.1, mlp_linear_coef=4):
        super().__init__()
        self.ln_context_cross_attn = nn.LayerNorm(n_embed)
        self.ln_memory_cross_attn = nn.LayerNorm(n_embed)
        self.ln_context_self_attn = nn.LayerNorm(n_embed)
        self.ln_mlp = nn.LayerNorm(n_embed) 
        self.self_attention = AttentionModule(n_embed=n_embed, n_head=n_head, drop_rate=drop_rate)
        self.cross_attention = AttentionModule(n_embed=n_embed, n_head=n_head, drop_rate=drop_rate)       
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, mlp_linear_coef * n_embed),
            nn.GELU(),
            nn.Linear(mlp_linear_coef * n_embed, n_embed),
            nn.Dropout(drop_rate),
        )
        
    def forward(self, x_context, x_memory, attn_mask=None):
        # маска на данные из будущего
        T = x_context.shape[1]
        casual_mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        casual_mask = (1 - casual_mask) * torch.finfo(x_context.dtype).min
        casual_mask = casual_mask.to(x_context.device)
        casual_mask = casual_mask.to(x_context.dtype)
        # маскируем decoder
        self_att_output = self.self_attention(
            self.ln_context_self_attn(x_context), 
            self.ln_context_self_attn(x_context), 
            attn_mask = casual_mask
        )
        self_att_output = self_att_output[0]
        x = x_context + self_att_output
        cross_att_output = self.cross_attention(
            self.ln_context_cross_attn(x), 
            self.ln_memory_cross_attn(x_memory), 
            attn_mask = attn_mask
        )
        cross_att_output = cross_att_output[0]
        x = x + cross_att_output
        x = x + self.mlp(self.ln_mlp(x))
        return x
    

class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self._init_weights()
 
    def _init_weights(self):
        nn.init.normal_(self.linear.weight, std=0.01)
    
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        output = x0 * self.linear(xl) + xl
        return output

class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, rank: int = 32):
        super().__init__()
        self.V = nn.Linear(input_dim, rank, bias=False)
        self.U = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.normal_(self.U.weight, std=0.01)
    
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        x_v = self.V(xl)
        x_uv = self.U(x_v) + self.bias
        output =  x0 * x_uv + xl
        return output

class MoECrossLayer(nn.Module):
    def __init__(self, input_dim: int, num_experts: int = 4, rank: int = 32, gate_type: str = 'softmax', use_nonlinear: bool = False, activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        if gate_type == 'softmax':
            self.gate_type = nn.Softmax(dim=-1)
        elif gate_type == 'sigmoid':
            self.gate_type = nn.Sigmoid()
        self.rank = rank
        self.use_nonlinear = use_nonlinear
        self.activation = activation
 
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
 
        self.U = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        self.V = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        self.bias = nn.Parameter(torch.zeros(num_experts, input_dim))
 
        if self.use_nonlinear:
            self.C = nn.Parameter(torch.empty(num_experts, input_dim, rank))
        else:
            self.C = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.U, std=0.01)
        nn.init.normal_(self.V, std=0.01)
        nn.init.xavier_uniform_(self.gate.weight)
 
        if self.use_nonlinear:
            nn.init.normal_(self.C, std=0.01)
        
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        #bs - batch_size, K - num_experts, d - input_dim, r -rank
        gate_logits = self.gate(xl)
        gate_scores = self.gate_type(gate_logits)
        
        # V.T @ xl - общее начало как для линейного, так и нелинейного MoE
        x_proj = torch.einsum('bd, kdr -> bkr', xl, self.V)
 
        if not self.use_nonlinear:
            # Перемножаем с U
            expert_out = torch.einsum('bkr, kdr -> bkd', x_proj, self.U)
        else:
            # Активация на перемножение первых матриц
            x_proj = self.activation(x_proj)
            # Умножаем на C по формуле из статьи
            x_mixed = torch.einsum('bkr, krz -> bkz', x_proj, self.C)
            # Активация на перемножение с C
            x_mixed = self.activation(x_mixed)
            # Перемножаем с U
            expert_out = torch.einsum('bkz, kdz -> bkd', x_mixed, self.U)
        
        expert_out = expert_out + self.bias
        weighted_experts = torch.einsum('bk, bkd -> bd', gate_scores, expert_out)
        output = x0 * weighted_experts + xl
        return output
    
class DenseLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int = 64, dropout_rate: float=0.1, activation: nn.Module = nn.ReLU()):
        super(DenseLayer, self).__init__()
        self.norm = nn.LayerNorm(in_features)
        self.activation = activation
        self.linear = nn.Linear(in_features, growth_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x) -> torch.Tensor:
        out = self.norm(x)
        out = self.activation(out)
        out = self.linear(out)
        out = self.dropout(out)
        
        return torch.cat([x, out], dim=1)
    
