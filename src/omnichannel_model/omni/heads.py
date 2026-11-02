from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from omnichannel_model.omni.blocks import EmbeddingProjection
from omnichannel_model.losses import FocalLoss, BPRLoss


class BaseHead(nn.Module):

    kind: str = 'pointwise'

    def calculate_loss(self, outputs, targets = None, sample_weights=None):
        if self.kind == 'pointwise':
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            if targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
            if sample_weights is not None:
                return self.loss_fn(outputs.float(), targets.float(), sample_weights = sample_weights)
            return self.loss_fn(outputs.float(), targets.float())
        if self.kind == 'pairwise':
            if sample_weights is not None:
                return self.loss_fn(outputs['s_pos'], outputs['s_neg'], sample_weights = sample_weights)
            return self.loss_fn(outputs['s_pos'], outputs['s_neg'])
        raise ValueError(f'Unknown head kind: {self.kind}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        if self.kind == 'pointwise':
            return self.predict_activation(logits)
        if self.kind == 'pairwise':
            return logits.squeeze(-1)
        raise ValueError(f'Неизвестный тип головы: {self.kind}')
    
    def get_parameter_names(self) -> List[str]:
        params = [name for name, _ in self.named_parameters()]
        return params
    
    def get_layer_names(self) -> List[str]:
        layers = [name for name, _ in self.named_children()]
        return layers

class CRHead(BaseHead):
    """Pointwise CR-голова: encode_repr -> логит -> (для predict) sigmoid."""
    kind = 'pointwise'

    def __init__(self,
                 input_dim: int,
                 hidden_dims=(2048, 1024),
                 output_dim: int = 512,
                 dropout: float = 0.1,
                 activation: nn.Module = None,
                 loss_fn: nn.Module = None,
                 predict_activation: nn.Module = None):
        super().__init__()
        self.activation = activation
        self.predict_ffn = EmbeddingProjection(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            activation=activation,
        )
        self.final_layer = nn.Sequential(
            nn.Dropout(dropout),
            self.activation,  
            nn.Linear(output_dim, 1),
        )
        self.loss_fn = loss_fn if loss_fn is not None else FocalLoss(
            alpha=1.0, gamma=0.0, reduction='mean', logits=True
        )
        self.predict_activation = predict_activation if predict_activation is not None else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.predict_ffn(x)
        x = self.final_layer(x)
        return x

class CTRHead(BaseHead):
    """Pointwise CTR-голова: encode_repr -> логит -> (для predict) sigmoid."""
    kind = 'pointwise'

    def __init__(self,
                 input_dim: int,
                 hidden_dims=(2048, 1024),
                 output_dim: int = 512,
                 dropout: float = 0.1,
                 activation: nn.Module = None,
                 loss_fn: nn.Module = None,
                 predict_activation: nn.Module = None):
        super().__init__()
        self.activation = activation
        self.predict_ffn = EmbeddingProjection(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            activation=activation,
        )
        self.final_layer = nn.Sequential(
            nn.Dropout(dropout),
            self.activation,  
            nn.Linear(output_dim, 1),
        )
        self.loss_fn = loss_fn if loss_fn is not None else FocalLoss(
            alpha=1.0, gamma=0.0, reduction='mean', logits=True
        )
        self.predict_activation = predict_activation if predict_activation is not None else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.predict_ffn(x)
        x = self.final_layer(x)
        return x

class PairwiseHead(BaseHead):
    """Pairwise-голова: обучается через два encode (pos/neg), BPR loss."""
    kind = 'pairwise'

    def __init__(self,
                 input_dim: int,
                 hidden_dims=(2048, 1024),
                 output_dim: int = 512,
                 dropout: float = 0.1,
                 activation: nn.Module = None,
                 loss_fn: nn.Module = None,
                 predict_activation: nn.Module = None):
        super().__init__()
        self.activation = activation
        self.predict_ffn = EmbeddingProjection(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=list(hidden_dims),
                dropout=dropout,
                activation=activation,
            )
        self.final_layer = nn.Sequential(
            nn.Dropout(dropout),
            self.activation,
            nn.Linear(output_dim, 1),
        )
        self.loss_fn = loss_fn if loss_fn is not None else BPRLoss()
        self.predict_activation = predict_activation if predict_activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.predict_ffn(x)
        x = self.final_layer(x)
        return x
