import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal, Union, Dict, Tuple
 
import numpy as np
 

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = 'mean', logits: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits
       
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        logits = inputs.squeeze()
        targets = targets.squeeze()
       
        if not self.logits:
            logits = torch.logit(logits.clamp(1e-8, 1 - 1e-8))
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        
        focal_factor = self.alpha * (1 - p_t) ** self.gamma
        loss_vec = focal_factor * bce_loss
       
        if sample_weight is not None:
            sample_weight = sample_weight.squeeze()
            loss_vec = loss_vec * sample_weight
       
        if self.reduction == 'mean':
            denom = sample_weight.sum() if sample_weight is not None else loss_vec.numel()
            return loss_vec.sum() / denom
        if self.reduction == 'sum':
            return loss_vec.sum()
        
        return loss_vec
    
class ClassBalancedLoss(nn.Module):
    def __init__(self, n_samples, beta=0.9999):
        super().__init__()
        n_pos = int(n_samples * 0.098)
        n_neg = n_samples - n_pos
        
        effective_pos = (1 - beta**n_pos) / (1 - beta)
        effective_neg = (1 - beta**n_neg) / (1 - beta)
        
        weight_pos = (1 - beta) / effective_pos
        weight_neg = (1 - beta) / effective_neg
        
        total = weight_pos + weight_neg
        self.weight_pos = weight_pos / total * 2
        self.weight_neg = weight_neg / total * 2
        
        
    def forward(self, logits, targets):
        weights = torch.where(
        targets == 1,
        torch.tensor(self.weight_pos, device=logits.device),
        torch.tensor(self.weight_neg, device=logits.device)
        )
        
        loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(),
        targets,
        reduction='none'
        )
        
        return (weights * loss).mean()

class LDAMLoss(nn.Module):
    def __init__(self, n_samples, max_m=0.5, s=30):
        super().__init__()
        n_pos = int(n_samples * 0.098)
        n_neg = n_samples - n_pos
        
        cls_num_list = [n_neg, n_pos]
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        
        self.m_neg = torch.tensor(m_list[0])
        self.m_pos = torch.tensor(m_list[1])
        self.s = s
        
        
    def forward(self, logits, targets):
        margins = torch.where(
        targets == 1,
        self.m_pos.to(logits.device),
        self.m_neg.to(logits.device)
        )
        
        logits_m = torch.where(
        targets == 1,
        logits.squeeze() - margins,
        logits.squeeze() + margins
        )
        
        logits_m = self.s * logits_m
        
        loss = F.binary_cross_entropy_with_logits(
        logits_m,
        targets
        )
        
        return loss
    
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos_scores = pos_scores.squeeze()
        neg_scores = neg_scores.squeeze()
        
        bpr = -F.logsigmoid(pos_scores - neg_scores)
        
        if sample_weight is not None:
            sample_weight = sample_weight.squeeze()
            bpr = bpr * sample_weight
            denom = sample_weight.sum().clamp(1e-8)
        else:
            denom = bpr.numel()
            
        return bpr.sum() / denom