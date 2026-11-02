from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnichannel_model.omni.blocks import EmbeddingProjection


@dataclass
class MultiTaskConfig:
    tasks: Tuple[str, ...] = ("cr", "ctr")
    channels_per_task: Dict[str, int] = field(default_factory=lambda: {"cr": 4, "ctr": 1})
    # позиция в списке = индекс башни; значение = eval_channel (mass=0, phygital=1, premier=2, sbol=3)
    channel_ids: Dict[str, List[int]] = field(default_factory=lambda: {"cr": [0, 1, 2, 3]})
    channel_col: int = 1
    target_cols: Dict[str, int] = field(default_factory=lambda: {"cr": 0, "ctr": 1})
    n_task_experts: Dict[str, int] = field(default_factory=lambda: {"cr": 2, "ctr": 2})
    n_shared_experts: int = 2
    expert_hidden: int = 512
    expert_dim: int = 256
    expert_dropout: float = 0.2
    gate_type: Literal["softmax", "sigmoid"] = "softmax"
    gate_temperature: float = 1.0
    gate_expert_dropout: float = 0.0
    # "none": у задач с несколькими каналами нет канальных башен (нужны general_tasks); башни одноканальных задач остаются
    tower_type: Literal["per_channel", "shared", "none"] = "per_channel"
    tower_hidden: int = 128
    tower_dropout: float = 0.1
    task_weights: Dict[str, float] = field(default_factory=lambda: {"cr": 1.0, "ctr": 1.0})
    channel_weights: Optional[Dict[str, List[float]]] = None
    pos_weight: Optional[Dict[str, List[float]]] = None
    focal_gamma: float = 0.0
    lb_coef: float = 0.0
    # общая башня: учится на всех строках задачи без маршрутизации по каналу
    general_tasks: Tuple[str, ...] = ()
    general_weight: Dict[str, float] = field(default_factory=dict)
    general_pos_weight: Dict[str, float] = field(default_factory=dict)
    # канальная поправка поверх общей башни: logit_c = base_logit + delta_c
    adapter_tasks: Tuple[str, ...] = ()
    adapter_emb_dim: int = 8
    adapter_hidden: int = 32
    # isolated: z и base_logit отсоединены (поправка не влияет на базу);
    # to_features: градиент идёт в z, но не в голову базы; joint: сквозное обучение
    adapter_grad: Literal["isolated", "to_features", "joint"] = "isolated"
    adapter_loss_weight: float = 1.0
    adapter_reg: float = 0.01
    adapter_reg_weights: Dict[str, List[float]] = field(default_factory=dict)

    general_tower_hidden_dims: Tuple[int, ...] = ()

    def __post_init__(self):
        self.tasks = tuple(self.tasks)
        self.general_tasks = tuple(self.general_tasks)
        self.adapter_tasks = tuple(self.adapter_tasks)
        self.general_tower_hidden_dims = tuple(int(h) for h in self.general_tower_hidden_dims)
        if any(h <= 0 for h in self.general_tower_hidden_dims):
            raise ValueError("general_tower_hidden_dims: размеры слоёв должны быть > 0")

        for t in self.tasks:
            k = self.channels_per_task[t]
            for name in ("pos_weight", "channel_weights"):
                vals = (getattr(self, name) or {}).get(t)
                if vals is not None and len(vals) != k:
                    raise ValueError(f"{name}[{t}] должен содержать {k} значений")
            if k > 1 and len(self.channel_ids.get(t, [])) != k:
                raise ValueError(f"channel_ids[{t}] должен содержать {k} значений")
            for name in ("n_task_experts", "task_weights", "target_cols"):
                if t not in getattr(self, name):
                    raise ValueError(f"{name} не содержит задачу {t}")
        if self.n_shared_experts + min(self.n_task_experts.values()) < 1:
            raise ValueError("у каждой задачи должен быть минимум один эксперт")
        for t in self.general_tasks:
            if t not in self.tasks:
                raise ValueError(f"general_tasks содержит неизвестную задачу {t}")
        for name in ("general_weight", "general_pos_weight"):
            bad = set(getattr(self, name)) - set(self.general_tasks)
            if bad:
                raise ValueError(f"{name}: задач {sorted(bad)} нет в general_tasks")
        if self.adapter_grad not in ("isolated", "to_features", "joint"):
            raise ValueError(f"adapter_grad: неизвестный режим {self.adapter_grad}")
        for t in self.adapter_tasks:
            if t not in self.general_tasks:
                raise ValueError(f"adapter_tasks: задача {t} должна быть и в general_tasks (поправка строится над общей башней)")
            if self.channels_per_task[t] <= 1:
                raise ValueError(f"adapter_tasks: у задачи {t} один канал, поправка не нужна")
        bad = set(self.adapter_reg_weights) - set(self.adapter_tasks)
        if bad:
            raise ValueError(f"adapter_reg_weights: задач {sorted(bad)} нет в adapter_tasks")
        for t, vals in self.adapter_reg_weights.items():
            if len(vals) != self.channels_per_task[t]:
                raise ValueError(f"adapter_reg_weights[{t}] должен содержать {self.channels_per_task[t]} значений")
        if self.tower_type == "none":
            for t in self.tasks:
                if self.channels_per_task[t] > 1 and t not in self.general_tasks:
                    raise ValueError(f"tower_type='none': задаче {t} нужна общая башня (general_tasks), иначе у неё нет выходов")

    def has_towers(self, task: str) -> bool:
        return not (self.tower_type == "none" and self.channels_per_task[task] > 1)

    def loss_keys(self) -> Dict[str, Tuple[str, str]]:
        keys: Dict[str, Tuple[str, str]] = {}
        for t in self.tasks:
            if self.has_towers(t):
                keys[t] = (t, "tower")
        for t in self.general_tasks:
            keys[f"{t}_general"] = (t, "general")
        for t in self.adapter_tasks:
            keys[f"{t}_adj"] = (t, "adj")
        return keys

    @classmethod
    def from_dict(cls, d: dict) -> "MultiTaskConfig":
        return cls(**d)


class ExpertGroup(nn.Module):
    def __init__(self, in_dim: int, n_experts: int, hidden: int, out_dim: int,
                 dropout: float, activation: nn.Module):
        super().__init__()
        self.out_dim = out_dim
        self.experts = nn.ModuleList([
            EmbeddingProjection(in_dim, out_dim, hidden_dims=[hidden], dropout=dropout,
                                activation=type(activation)())
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 0:
            return x.new_zeros(x.shape[0], 0, self.out_dim)
        return torch.stack([e(x) for e in self.experts], dim=1)


class TaskGate(nn.Module):
    def __init__(self, in_dim: int, n_experts: int, gate_type: str,
                 temperature: float, expert_dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_experts)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate_type = gate_type
        self.temperature = temperature
        self.expert_dropout = expert_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x).float() / self.temperature
        if self.training and self.expert_dropout > 0.0:
            drop = torch.rand_like(logits) < self.expert_dropout
            drop = drop & ~drop.all(dim=1, keepdim=True)
            logits = logits.masked_fill(drop, float("-inf"))
        if self.gate_type == "softmax":
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)


class TaskTower(nn.Module):
    def __init__(self, in_dim: int, hidden: Union[int, Sequence[int]], n_channels: int, tower_type: str,
                  dropout: float, activation: nn.Module):
        super().__init__()
        self.n_channels = n_channels
        self.per_channel = tower_type == "per_channel" and n_channels > 1
        self.activation = type(activation)()
        self.dropout = nn.Dropout(dropout)
        dims = [int(hidden)] if isinstance(hidden, int) else [int(h) for h in hidden]
        if self.per_channel and len(dims) != 1:
            raise ValueError("per_channel башня поддерживает ровно один скрытый слой")
        self.deep = None
        if len(dims) > 1:
            # несколько скрытых слоёв: Linear -> act -> Dropout на каждом, затем выходной Linear
            layers, cur = [], in_dim
            for h in dims:
                layers += [nn.Linear(cur, h), type(activation)(), nn.Dropout(dropout)]
                cur = h
            layers.append(nn.Linear(cur, n_channels))
            self.deep = nn.Sequential(*layers)
            return
        hidden = dims[0]
        if self.per_channel:
            k = n_channels
            bound = in_dim ** -0.5
            self.weight_1 = nn.Parameter(torch.empty(k, in_dim, hidden).uniform_(-bound, bound))
            self.bias_1 = nn.Parameter(torch.zeros(k, hidden))
            bound2 = hidden ** -0.5
            self.weight_2 = nn.Parameter(torch.empty(k, hidden).uniform_(-bound2, bound2))
            self.bias_2 = nn.Parameter(torch.zeros(k))
        else:
            self.fc_1 = nn.Linear(in_dim, hidden)
            self.fc_2 = nn.Linear(hidden, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deep is not None:
            return self.deep(x)
        if self.per_channel:
            h = torch.einsum("bd,kdh->bkh", x, self.weight_1.to(x.dtype)) + self.bias_1.to(x.dtype)
            h = self.dropout(self.activation(h))
            return (h * self.weight_2.to(h.dtype)).sum(-1) + self.bias_2.to(h.dtype)
        return self.fc_2(self.dropout(self.activation(self.fc_1(x))))


class ChannelAdapter(nn.Module):
    """Поправка delta_k(z) для каждого из K каналов; веса MLP общие, канал задаётся обучаемым эмбеддингом."""

    def __init__(self, z_dim: int, n_channels: int, emb_dim: int, hidden: int, activation: nn.Module):
        super().__init__()
        self.z_dim = z_dim
        self.emb = nn.Embedding(n_channels, emb_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=1.0 / emb_dim)
        self.fc_1 = nn.Linear(z_dim + emb_dim, hidden)
        self.fc_2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc_2.weight)
        nn.init.zeros_(self.fc_2.bias)
        self.bias_c = nn.Parameter(torch.zeros(n_channels))
        self.activation = type(activation)()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # первый слой над [z || e_k] раскладывается на W_z z + W_e e_k: без материализации [B, K, z+e]
        w = self.fc_1.weight
        h_z = F.linear(z, w[:, :self.z_dim])
        h_e = F.linear(self.emb.weight, w[:, self.z_dim:])
        h = self.activation(h_z[:, None, :] + h_e[None, :, :] + self.fc_1.bias)
        delta = self.fc_2(h).squeeze(-1) + self.bias_c
        return delta.float()


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim: int, cfg: MultiTaskConfig, activation: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        act = activation if activation is not None else nn.GELU()
        self.shared = ExpertGroup(in_dim, cfg.n_shared_experts, cfg.expert_hidden,
                                  cfg.expert_dim, cfg.expert_dropout, act)
        self.task_experts = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.towers = nn.ModuleDict()
        self.general_towers = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        for t in cfg.tasks:
            n_t = cfg.n_task_experts[t]
            self.task_experts[t] = ExpertGroup(in_dim, n_t, cfg.expert_hidden,
                                               cfg.expert_dim, cfg.expert_dropout, act)
            self.gates[t] = TaskGate(in_dim, n_t + cfg.n_shared_experts, cfg.gate_type,
                                     cfg.gate_temperature, cfg.gate_expert_dropout)
            if cfg.has_towers(t):
                self.towers[t] = TaskTower(cfg.expert_dim, cfg.tower_hidden, cfg.channels_per_task[t],
                                           cfg.tower_type, cfg.tower_dropout, act)
            if t in cfg.general_tasks:
                self.general_towers[t] = TaskTower(cfg.expert_dim,
                                                   cfg.general_tower_hidden_dims or cfg.tower_hidden, 1,
                                                   "shared", cfg.tower_dropout, act)
            if t in cfg.adapter_tasks:
                self.adapters[t] = ChannelAdapter(cfg.expert_dim, cfg.channels_per_task[t],
                                                  cfg.adapter_emb_dim, cfg.adapter_hidden, act)
            if cfg.channels_per_task[t] > 1:
                ids = cfg.channel_ids[t]
                lut = torch.full((max(ids) + 1,), -1, dtype=torch.long)
                for pos, ch in enumerate(ids):
                    lut[ch] = pos
                self.register_buffer(f"lut_{t}", lut, persistent=False)

    def _route(self, task: str, channel: Optional[torch.Tensor], batch: int,
               device: torch.device) -> torch.Tensor:
        if self.cfg.channels_per_task[task] == 1:
            return torch.zeros(batch, dtype=torch.long, device=device)
        if channel is None:
            if self.training:
                raise ValueError(f"для задачи '{task}' нужен канал строки (batch['groups'][:, channel_col])")
            return torch.full((batch,), -1, dtype=torch.long, device=device)
        lut = getattr(self, f"lut_{task}")
        ch = channel.long()
        ok = (ch >= 0) & (ch < lut.numel())
        return torch.where(ok, lut[ch.clamp(0, lut.numel() - 1)], torch.full_like(ch, -1))

    def forward(self, x: torch.Tensor, channel: Optional[torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        shared = self.shared(x)
        out = {}
        for t in self.cfg.tasks:
            experts = torch.cat([self.task_experts[t](x), shared], dim=1)
            w = self.gates[t](x)
            mix = torch.einsum("be,bed->bd", w.to(experts.dtype), experts)
            entry = {"index": self._route(t, channel, x.shape[0], x.device), "gate": w}
            if t in self.towers:
                entry["logits_all"] = self.towers[t](mix).float()
            if t in self.general_towers:
                entry["logits_general"] = self.general_towers[t](mix).float()
            if t in self.adapters:
                mode = self.cfg.adapter_grad
                z_in = mix.detach() if mode == "isolated" else mix
                base = entry["logits_general"]
                base_in = base if mode == "joint" else base.detach()
                delta = self.adapters[t](z_in)
                entry["delta_all"] = delta
                entry["logits_adj_all"] = base_in + delta
            out[t] = entry
        return out

    @staticmethod
    def predict(out: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        preds = {}
        nan = float("nan")
        for t, o in out.items():
            idx = o["index"]
            known = idx >= 0
            gi = idx.clamp_min(0)[:, None]
            if "logits_all" in o:
                p_all = torch.sigmoid(o["logits_all"])
                sel = p_all.gather(1, gi)[:, 0]
                preds[t] = torch.where(known, sel, torch.full_like(sel, nan))[:, None]
                if p_all.shape[1] > 1:
                    preds[f"{t}_by_channel"] = p_all
            if "logits_general" in o:
                preds[f"{t}_general"] = torch.sigmoid(o["logits_general"])
                preds[f"{t}_general_logit"] = o["logits_general"]
            if "logits_adj_all" in o:
                p_adj = torch.sigmoid(o["logits_adj_all"])
                sel = p_adj.gather(1, gi)[:, 0]
                preds[f"{t}_adj"] = torch.where(known, sel, torch.full_like(sel, nan))[:, None]
                preds[f"{t}_adj_by_channel"] = p_adj
        return preds


@dataclass
class MTLossOutput:
    total: torch.Tensor
    per_task: Dict[str, torch.Tensor]
    per_channel: Dict[str, torch.Tensor]
    n_valid: Dict[str, torch.Tensor]
    gate_entropy: Dict[str, torch.Tensor]
    gate_util: Dict[str, torch.Tensor]
    task_terms: Dict[str, torch.Tensor] = field(default_factory=dict)
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg
        for t in cfg.tasks:
            k = cfg.channels_per_task[t]
            pw = (cfg.pos_weight or {}).get(t, [1.0] * k)
            cw = (cfg.channel_weights or {}).get(t, [1.0] * k)
            self.register_buffer(f"pos_weight_{t}", torch.tensor(pw, dtype=torch.float32), persistent=False)
            self.register_buffer(f"chan_weight_{t}", torch.tensor(cw, dtype=torch.float32), persistent=False)

    def _bce(self, logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        if self.cfg.focal_gamma > 0.0:
            bce = bce * (1.0 - torch.exp(-bce)) ** self.cfg.focal_gamma
        return bce

    @staticmethod
    def _mean(row: torch.Tensor, mask: torch.Tensor, sw: Optional[torch.Tensor]) -> torch.Tensor:
        if sw is not None:
            return (row * mask * sw).sum() / (mask * sw).sum().clamp_min(1e-8)
        return (row * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, out: Dict[str, Dict[str, torch.Tensor]], target: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> MTLossOutput:
        cfg = self.cfg
        total = 0.0
        per_task, per_channel, n_valid, ent, util, extras = {}, {}, {}, {}, {}, {}
        terms = {}
        sw = None if sample_weights is None else sample_weights.reshape(-1).float()
        for t in cfg.tasks:
            o = out[t]
            k = cfg.channels_per_task[t]
            tgt = target[:, cfg.target_cols[t]]
            idx = o["index"]
            valid = ((tgt >= 0) & (idx >= 0)).float()
            vg = (tgt >= 0).float()
            gi = idx.clamp_min(0)
            y = tgt.clamp_min(0).float()
            onehot = F.one_hot(gi, k).float() * valid[:, None]
            cnt = onehot.sum(0)
            n_valid[t] = cnt
            task_obj = 0.0

            if "logits_all" in o:
                logit = o["logits_all"].gather(1, gi[:, None])[:, 0]
                bce = self._bce(logit, y)
                pw = getattr(self, f"pos_weight_{t}")[gi]
                cw = getattr(self, f"chan_weight_{t}")[gi]
                row = bce * torch.where(y > 0.5, pw, torch.ones_like(pw)) * cw
                task_loss = self._mean(row, valid, sw)
                if cfg.lb_coef > 0.0:
                    n_exp = o["gate"].shape[1]
                    p = (o["gate"] * valid[:, None]).sum(0)
                    p = p / p.sum().clamp_min(1e-8)
                    task_loss = task_loss + cfg.lb_coef * (n_exp * (p * p).sum() - 1.0)
                total = total + cfg.task_weights[t] * task_loss
                terms[t] = cfg.task_weights[t] * task_loss
                with torch.no_grad():
                    per_channel[t] = (onehot * bce.detach()[:, None]).sum(0) / cnt.clamp_min(1.0)
                    per_task[t] = task_loss.detach()

            with torch.no_grad():
                g = o["gate"].detach()
                gn = g / g.sum(1, keepdim=True).clamp_min(1e-8)
                h = -(gn * gn.clamp_min(1e-8).log()).sum(1)
                nv = valid.sum().clamp_min(1.0)
                ent[t] = (h * valid).sum() / nv
                util[t] = (gn * valid[:, None]).sum(0) / nv

            if "logits_general" in o:
                bce_g = self._bce(o["logits_general"][:, 0], y)
                pw_g = torch.full_like(bce_g, cfg.general_pos_weight.get(t, 1.0))
                row_g = bce_g * torch.where(y > 0.5, pw_g, torch.ones_like(pw_g))
                loss_g = self._mean(row_g, vg, sw)
                w_g = cfg.general_weight.get(t, 1.0)
                total = total + w_g * loss_g
                gk = f"{t}_general"
                terms[gk] = w_g * loss_g
                task_obj = task_obj + w_g * loss_g
                with torch.no_grad():
                    ng = vg.sum().reshape(1)
                    per_task[gk] = loss_g.detach()
                    n_valid[gk] = ng
                    per_channel[gk] = ((bce_g.detach() * vg).sum() / ng.clamp_min(1.0)).reshape(1)

            if "logits_adj_all" in o:
                la = o["logits_adj_all"].gather(1, gi[:, None])[:, 0]
                bce_a = self._bce(la, y)
                loss_adj = self._mean(bce_a, valid, sw)
                delta = o["delta_all"]
                rw = delta.new_tensor(cfg.adapter_reg_weights.get(t, [1.0] * k))
                reg = (delta.pow(2) * rw[None, :] * vg[:, None]).sum() / (vg.sum().clamp_min(1.0) * k)
                term_a = cfg.adapter_loss_weight * loss_adj + cfg.adapter_reg * reg
                total = total + term_a
                ak = f"{t}_adj"
                terms[ak] = term_a
                task_obj = task_obj + term_a
                with torch.no_grad():
                    per_task[ak] = loss_adj.detach()
                    n_valid[ak] = cnt
                    per_channel[ak] = (onehot * bce_a.detach()[:, None]).sum(0) / cnt.clamp_min(1.0)
                    extras[f"{ak}/reg"] = reg.detach()
                    extras[f"{ak}/delta_mean"] = (delta.detach() * vg[:, None]).sum(0) / vg.sum().clamp_min(1.0)

            if t not in terms:
                terms[t] = task_obj
        return MTLossOutput(total, per_task, per_channel, n_valid, ent, util, terms, extras)
