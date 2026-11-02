from __future__ import annotations

import os
import torch

import pandas as pd
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Tuple, Union

from omnichannel_model.omni.models import OmnichannelModel
from omnichannel_model.base_classes.base_calculator import BaseCalculatorClass


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Переносит все тензоры батча на указанный device in-place."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def normalize_weights(
    weights: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Нормируем веса так, чтобы их сумма равнялась 1.
    
    Отрицательные веса обнуляются. Если сумма близка к нулю,
    возвращает равномерное распределение.
    """
    weights = torch.clamp(weights, min=0.0)
    weights_sum = torch.sum(weights)

    if weights_sum < eps:
        return torch.ones_like(weights) / weights.size(0)

    return weights/weights_sum

def _compute_grad_norm(parameters) -> float:
    """Вычисляет L2-норму градиентов по всем параметрам модели."""
    total = sum(
        p.grad.data.norm(2).item() ** 2
        for p in parameters
        if p.grad is not None
    )
    return total ** 0.5


def _compute_weight_change(
    named_parameters,
    prev_weights: Dict[str, torch.Tensor],
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Считает суммарное изменение весов относительно предыдущего снапшота.
    Возвращает величину изменения и обновлённый снапшот.
    """
    new_weights: Dict[str, torch.Tensor] = {}
    change = 0.0
    for name, param in named_parameters:
        new_weights[name] = param.data.detach().clone()
        if name in prev_weights:
            change += torch.norm(param.data.detach() - prev_weights[name]).item()
    return change, new_weights


def _extract_sample_weight(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Возвращает нормированные веса сэмплов из батча, если они есть."""
    if "sample_weight" in batch:
        return normalize_weights(batch["sample_weight"])
    return None

def _masked_head_loss(model, outputs, head_name, tgt_col, n_valid: int):
    """Среднее по валидным строкам головы; метка -1 в лосс не попадает."""
    out = outputs[head_name]
    if n_valid == 0:
        return out.sum() * 0.0
    valid = tgt_col >= 0
    return model.module.calculate_loss(
        {head_name: out[valid]}, targets=tgt_col[valid], head_name=head_name)


# ---------------------------------------------------------------------------
# Train on 1 GPU functions
# ---------------------------------------------------------------------------

def train_cr_head(
    model: OmnichannelModel,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int
) -> float:
    model.train()
    total_loss = 0
    pbar = tqdm(loader)
    prev_weights = None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    for batch_idx, data in enumerate(pbar):
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        if "sample_weight" in input_batch:
            sample_weight = normalize_weights(input_batch["sample_weight"])
        else:
            sample_weight = None

        optimizer.zero_grad()

        outputs = model(input_batch, head_names = ["cr"], data_format=data_format)
        if torch.isnan(outputs['cr']).any():
            print("Warning:NaN detected in outputs")

        loss = model.calculate_loss(outputs, targets = target, head_name = "cr", sample_weights = sample_weight)

        loss.backward()

        if batch_idx % 100 == 0:
            norm_before = _compute_grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.parameters())

            current_lr = optimizer.param_groups[0]["lr"]

            if prev_weights is None:
                prev_weights = {}
                for name, param in model.named_parameters():
                    prev_weights[name] = param.data.detach().clone()

            weight_change, prev_weights = _compute_weight_change(
                model.named_parameters(), prev_weights
            )

            pbar.set_description(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss/(batch_idx+1):.4f}, "
                                 f"Grad Norm Before: {norm_before:.4f}, After: {norm_after:.4f}, "
                                 f"LR: {current_lr:.2e}, WeightChange: {weight_change:.6f}")
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        writer.add_scalar(
            "Train Loss",
            total_loss / (batch_idx + 1),
            epoch * len(loader) + batch_idx + 1,
        )
    return total_loss / (batch_idx + 1)

def train_ctr_head(
    model: OmnichannelModel,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int
) -> float:
    model.train()
    total_loss = 0
    pbar = tqdm(loader)
    prev_weights = None

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch_idx, data in enumerate(pbar):
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        if "sample_weight" in input_batch:
            sample_weight = normalize_weights(input_batch["sample_weight"])
        else:
            sample_weight = None

        optimizer.zero_grad()

        outputs = model(input_batch, head_names = ["ctr"], data_format=data_format)
        if torch.isnan(outputs["ctr"]).any():
            print("Warning:NaN detected in outputs")

        loss = model.calculate_loss(outputs, targets = target, head_name = "ctr", sample_weights = sample_weight)

        loss.backward()

        if batch_idx % 100 == 0:
            norm_before = _compute_grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.parameters())

            current_lr = optimizer.param_groups[0]["lr"]

            if prev_weights is None:
                prev_weights = {}
                for name, param in model.named_parameters():
                    prev_weights[name] = param.data.detach().clone()

            weight_change, prev_weights = _compute_weight_change(
                model.named_parameters(), prev_weights
            )

            pbar.set_description(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss/(batch_idx+1):.4f}, "
                                 f"Grad Norm Before: {norm_before:.4f}, After: {norm_after:.4f}, "
                                 f"LR: {current_lr:.2e}, WeightChange: {weight_change:.6f}")
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        writer.add_scalar(
            "Train Loss",
            total_loss / (batch_idx + 1),
            epoch * len(loader) + batch_idx + 1,
        )
    return total_loss / (batch_idx + 1)

def train_pairwise_head(
    model: OmnichannelModel,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int
) -> float:
    model.train()
    total_loss = 0
    pair_acc_sum = 0.0
    margin_sum = 0.0
    pbar = tqdm(loader)
    prev_weights = None
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch_idx, data in enumerate(pbar):
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        if "sample_weight" in input_batch:
            sample_weight = normalize_weights(input_batch["sample_weight"])
        else:
            sample_weight = None

        optimizer.zero_grad()

        outputs = model(input_batch, head_names = ["pairwise"], data_format=data_format)
        if torch.isnan(outputs["pairwise"]["s_pos"]).any() or torch.isnan(outputs["pairwise"]["s_neg"]).any():
            print("Warning:NaN detected in outputs")

        loss = model.calculate_loss(outputs, targets = None, head_name = "pairwise", sample_weights = sample_weight)

        loss.backward()
        
        with torch.no_grad():
            diff = outputs["pairwise"]["s_pos"] - outputs["pairwise"]["s_neg"]
            pair_acc = (diff > 0).float().mean().item()
            margin = diff.mean().item()
            pair_acc_sum += pair_acc
            margin_sum += margin

        if batch_idx % 100 == 0:
            norm_before = _compute_grad_norm(model.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.parameters())

            current_lr = optimizer.param_groups[0]["lr"]

            if prev_weights is None:
                prev_weights = {}
                for name, param in model.named_parameters():
                    prev_weights[name] = param.data.detach().clone()

            weight_change, prev_weights = _compute_weight_change(
                model.named_parameters(), prev_weights
            )

            pbar.set_description(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss/(batch_idx+1):.4f}, '
                                 f'PairAcc: {pair_acc_sum / (batch_idx + 1):.4f}, '
                                 f'Margin: {margin_sum / (batch_idx + 1):.4f}, '
                                 f'Grad Norm Before: {norm_before:.4f}, After: {norm_after:.4f}, '
                                 f'LR: {current_lr:.2e}, WeightChange: {weight_change:.6f}')
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        
        step = epoch * len(loader) + batch_idx + 1
        writer.add_scalar('Train Loss', total_loss/(batch_idx+1), step)
        writer.add_scalar('Train PairAcc', pair_acc_sum/(batch_idx+1), step)   # ← НОВОЕ
        writer.add_scalar('Train Margin', margin_sum/(batch_idx+1), step) 
    return total_loss / (batch_idx + 1)


# ---------------------------------------------------------------------------
# Train DPP functions
# ---------------------------------------------------------------------------
def train_cr_head_osiris(
    model: OmnichannelModel,
    accelerator: Accelerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int
) -> float:
    """
    Одна эпоха обучения в распределённом режиме (multi-GPU) через Accelerate.

    Использует ручную итерацию с dist.all_reduce для синхронизации конца эпохи
    между ранками — это предотвращает NCCL-дедлок, когда размеры датасетов
    на разных GPU не совпадают.
    Возвращает среднее значение loss за эпоху.
    """
    model.module.train()
    device = accelerator.device
    total_loss = 0.0
    prev_weights = None
    
    
    
    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)  #// accelerator.num_processes 
    except Exception:
        total_len = None
       
    pbar = tqdm(loader,
                total=total_len,
                disable=not accelerator.is_local_main_process)
    
    
    
    
    data_iter = iter(pbar)  # ← вручную, чтобы поймать StopIteration
    batch_idx = 0
    while True:
        # --- Синхронизация окончания эпохи ---
        # Если хотя бы один rank исчерпал данные — все останавливаются.
        # Это должно стоять ДО любого вызова accelerator.backward,
        # потому что backward вызывает NCCL all_reduce по градиентам.
        has_data = torch.ones(1, device=device)
        try:
            data = next(data_iter)
        except StopIteration:
            has_data.zero_()

        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        if has_data.item() < 0.5:
            break
        # --- конец блока синхронизации --
            
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        sample_weight = _extract_sample_weight(input_batch)

        optimizer.zero_grad()

        outputs = model(input_batch, head_names = ["cr"], data_format=data_format)
        if torch.isnan(outputs["cr"]).any():
            print("Warning:NaN detected in outputs")

        loss = model.module.calculate_loss(outputs, targets = target, head_name = "cr", sample_weights=sample_weight)

        accelerator.backward(loss)

        if batch_idx % 100 == 0:
            norm_before = _compute_grad_norm(model.module.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.module.parameters())

            current_lr = optimizer.param_groups[0]["lr"]

            if prev_weights is None:
                prev_weights = {}
                for name, param in model.module.named_parameters():
                    prev_weights[name] = param.data.detach().clone()

            weight_change, prev_weights = _compute_weight_change(
                model.module.named_parameters(), prev_weights
            )

            pbar.set_description(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss/(batch_idx+1):.4f}, "
                                 f"Grad Norm Before: {norm_before:.4f}, After: {norm_after:.4f}, "
                                 f"LR: {current_lr:.2e}, WeightChange: {weight_change:.6f}")
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        
        if writer is not None:
            writer.add_scalar(
                "Train Loss",
                total_loss / (batch_idx + 1),
                epoch * len(loader) + batch_idx + 1,
            )
        batch_idx += 1

    return total_loss / max(1, batch_idx)

def train_ctr_head_osiris(
    model: OmnichannelModel,
    accelerator: Accelerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int
) -> float:

    model.module.train()
    device = accelerator.device
    total_loss = 0.0
    prev_weights = None
    
    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)  #// accelerator.num_processes 
    except Exception:
        total_len = None
       
    pbar = tqdm(
        loader,
        total=total_len,
        disable=not accelerator.is_local_main_process
    )
    
    
    
    
    data_iter = iter(pbar)  # ← вручную, чтобы поймать StopIteration
    batch_idx = 0
    while True:
        # --- Синхронизация окончания эпохи ---
        # Если хотя бы один rank исчерпал данные — все останавливаются.
        # Это должно стоять ДО любого вызова accelerator.backward,
        # потому что backward вызывает NCCL all_reduce по градиентам.
        has_data = torch.ones(1, device=device)
        try:
            data = next(data_iter)
        except StopIteration:
            has_data.zero_()

        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        if has_data.item() < 0.5:
            break
        # --- конец блока синхронизации --
            
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        sample_weight = _extract_sample_weight(input_batch)

        optimizer.zero_grad()

        outputs = model(input_batch, head_names = ["ctr"], data_format=data_format)
        if torch.isnan(outputs["ctr"]).any():
            print("Warning:NaN detected in outputs")

        loss = model.module.calculate_loss(outputs, targets = target, head_name = "ctr", sample_weights=sample_weight)

        accelerator.backward(loss)

        if batch_idx % 100 == 0:
            norm_before = _compute_grad_norm(model.module.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.module.parameters())

            current_lr = optimizer.param_groups[0]["lr"]

            if prev_weights is None:
                prev_weights = {}
                for name, param in model.module.named_parameters():
                    prev_weights[name] = param.data.detach().clone()

            weight_change, prev_weights = _compute_weight_change(
                model.module.named_parameters(), prev_weights
            )

            pbar.set_description(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {total_loss/(batch_idx+1):.4f}, "
                                 f"Grad Norm Before: {norm_before:.4f}, After: {norm_after:.4f}, "
                                 f"LR: {current_lr:.2e}, WeightChange: {weight_change:.6f}")
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        
        if writer is not None:
            writer.add_scalar(
                "Train Loss",
                total_loss / (batch_idx + 1),
                epoch * len(loader) + batch_idx + 1,
            )
        batch_idx += 1

    return total_loss / max(1, batch_idx)

def train_pairwise_head_osiris(
    model: OmnichannelModel,
    accelerator: Accelerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int
) -> float:
    """
    Одна эпоха обучения в распределённом режиме (multi-GPU) через Accelerate.

    Использует ручную итерацию с dist.all_reduce для синхронизации конца эпохи
    между ранками — это предотвращает NCCL-дедлок, когда размеры датасетов
    на разных GPU не совпадают.
    Возвращает среднее значение loss за эпоху.
    """
    model.module.train()
    device = accelerator.device
    total_loss = 0.0
    pair_acc_sum = 0.0                           # [BPR] накопитель pair accuracy
    margin_sum   = 0.0                           # [BPR] накопитель margin
    prev_weights = None
    
    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)  #// accelerator.num_processes 
    except Exception:
        total_len = None
       
    pbar = tqdm(
        loader,
        total=total_len,
        disable=not accelerator.is_local_main_process
    )
    
    
    
    
    data_iter = iter(pbar)  # ← вручную, чтобы поймать StopIteration
    batch_idx = 0
    while True:
        # --- Синхронизация окончания эпохи ---

        has_data = torch.ones(1, device=device)
        try:
            data = next(data_iter)
        except StopIteration:
            has_data.zero_()

        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        if has_data.item() < 0.5:
            break
        # --- конец блока синхронизации --
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
            
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        sample_weight = _extract_sample_weight(input_batch)

        optimizer.zero_grad()

        outputs = model(input_batch, head_names = ["pairwise"], data_format=data_format)
        if torch.isnan(outputs["pairwise"]["s_pos"]).any() or torch.isnan(outputs["pairwise"]["s_neg"]).any():
            print("Warning:NaN detected in outputs")

        loss = model.module.calculate_loss(outputs, targets = None, head_name = "pairwise", sample_weights=sample_weight)

        accelerator.backward(loss)

        if batch_idx % 100 == 0:
            norm_before = _compute_grad_norm(model.module.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.module.parameters())

            current_lr = optimizer.param_groups[0]["lr"]

            if prev_weights is None:
                prev_weights = {}
                for name, param in model.module.named_parameters():
                    if not param.requires_grad:
                        continue
                    prev_weights[name] = param.data.detach().clone()

            weight_change, prev_weights = _compute_weight_change(
                model.module.named_parameters(), prev_weights
            )

            pbar.set_description(
                f"Epoch {epoch+1}, Batch {batch_idx+1}, "
                f"Loss: {total_loss/(batch_idx+1):.4f}, "
                f"PairAcc: {pair_acc_sum/(batch_idx+1):.4f}, "          # [BPR]
                f"Margin: {margin_sum/(batch_idx+1):.4f}, "             # [BPR]
                f"GradNorm B/A: {norm_before:.4f}/{norm_after:.4f}, "
                f"LR: {current_lr:.2e}, WChange: {weight_change:.6f}"
            )
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # ── Step ────────────────────────────────────────
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # ── Логирование ──────────────────────────────────
        if writer is not None:                                  # [BPR] защита от None на не-main rank
            step = epoch * (total_len or 0) + batch_idx + 1
            writer.add_scalar("Train Loss",    total_loss/(batch_idx+1), step)
            writer.add_scalar("Train PairAcc", pair_acc_sum/(batch_idx+1), step)  # [BPR]
            writer.add_scalar("Train Margin",  margin_sum/(batch_idx+1), step)    # [BPR]

        batch_idx += 1

    return total_loss / max(1, batch_idx)

def train_multihead_osiris(
    model: OmnichannelModel,
    accelerator: Accelerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int,
    head_targets: Dict[str, int],
    head_weights: Optional[Dict[str, float]] = None,
    log_every: int = 10,
    diag_every: int = 100,
    step_offset: int = 0,
) -> Dict[str, float]:
    
    """
    Совместное обучение pointwise-голов за один проход по данным.
    target имеет форму (B, k), -1 = метка головы не определена.
    Лосс = sum_h w_h * mean_{valid_h}(BCE). Возвращает средний лосс эпохи по головам (без весов).
    """
    
    model.module.train()
    device = accelerator.device
    heads = list(head_targets)
    cols = [head_targets[h] for h in heads]
    w = head_weights or {h: 1.0 for h in heads}
    sums = {h: torch.zeros((), device=device) for h in heads}
    steps_h = {h: 0 for h in heads}
    total_loss = torch.zeros((), device=device)
    prev_weights, diag_str, batch_idx = None, "", 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)
    except Exception:
        total_len = None
    pbar = tqdm(loader, total=total_len, disable=not accelerator.is_local_main_process)
    data_iter = iter(pbar)

    while True:
        # --- синхронизация конца эпохи между рангами (как в train_*_head_osiris) ---
        has_data = torch.ones(1, device=device)
        try:
            data = next(data_iter)
        except StopIteration:
            has_data.zero_()
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        if has_data.item() < 0.5:
            break

        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        if "sample_weight" in input_batch:
            raise RuntimeError("sample_weight не поддержан в train_multihead_osiris")
        target = input_batch["target"]                              # (B, k) Int32
        n_valid = (target[:, cols] >= 0).sum(0).tolist()            # одна синхронизация на шаг

        optimizer.zero_grad()
        outputs = model(input_batch, head_names=heads, data_format=data_format)

        loss = 0.0
        for h, n in zip(heads, n_valid):
            l_h = _masked_head_loss(model, outputs, h, target[:, head_targets[h]], n)
            loss = loss + w[h] * l_h
            if n > 0:
                sums[h] += l_h.detach()
                steps_h[h] += 1

        accelerator.backward(loss)

        if batch_idx % diag_every == 0:
            norm_before = _compute_grad_norm(model.module.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            norm_after = _compute_grad_norm(model.module.parameters())
            named_trainable = [(n, p) for n, p in model.module.named_parameters() if p.requires_grad]
            if prev_weights is None:
                prev_weights = {n: p.data.detach().clone() for n, p in named_trainable}
            weight_change, prev_weights = _compute_weight_change(named_trainable, prev_weights)
            diag_str = (f", GradNorm B/A: {norm_before:.4f}/{norm_after:.4f}, "
                        f"WChange: {weight_change:.6f}")
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        if batch_idx % log_every == 0:
            if not torch.isfinite(loss.detach()):
                print(f"Warning: non-finite loss, batch {batch_idx}", flush=True)
            lr_now = optimizer.param_groups[0]["lr"]
            means = {h: (sums[h] / max(1, steps_h[h])).item() for h in heads}
            pbar.set_description(
                f"Epoch {epoch+1}, Batch {batch_idx+1}, "
                f"Loss: {(total_loss/(batch_idx+1)).item():.4f}, "
                + ", ".join(f"{h}: {means[h]:.4f}" for h in heads)
                + f", LR: {lr_now:.2e}" + diag_str)
            if writer is not None:
                step = step_offset + batch_idx
                for h in heads:
                    writer.add_scalar(f"Train Loss/{h}", means[h], step)
                writer.add_scalar("LR/emb", lr_now, step)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach()
        batch_idx += 1

    return {h: float(sums[h] / max(1, steps_h[h])) for h in heads}

def train_several_models(
    model_dict: Dict[
        Union[str, int],
        Tuple[
            OmnichannelModel,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler
        ]
    ],
    device: torch.device,
    loader: DataLoader,
    epoch: int
) -> Dict[str, float]:
    losses = {k: 0 for k in model_dict}
    pbar = tqdm(loader)

    for batch_idx, data in enumerate(pbar):
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]

        sample_weight = _extract_sample_weight(input_batch)

        for k, (model, optimizer, scheduler) in model_dict.items():
            optimizer.zero_grad()

            outputs = model(input_batch)
            if torch.isnan(outputs).any():
                print("Warning:NaN detected in outputs")

            loss = model.calculate_loss(outputs, target, sample_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            losses[k] += loss.item()

    for k in losses:
        losses[k] = losses[k] / (batch_idx + 1)

    return losses

# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: OmnichannelModel,
    device: torch.device,
    loader: DataLoader,
    calculator: BaseCalculatorClass,
) -> Any:
    pbar = tqdm(loader)
    for data in pbar:
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]
        eval_groups = data["groups"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model.predict(input_batch, data_format=data_format)
        calculator.update(preds, target, eval_groups[:, 0], eval_groups[:, 1])

    return calculator.compute()



@torch.no_grad()
def evaluate_all_epochs(
    model_dict: Dict[
        Union[str, int],
        Tuple[
            OmnichannelModel,
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler,
        ],
    ],
    device: torch.device,
    loader: DataLoader,
    calculator: BaseCalculatorClass,
) -> Dict[Union[str, int], Any]:
    
    calc_dict = {k: deepcopy(calculator) for k in model_dict}
    pbar = tqdm(loader)

    for data in pbar:
        input_batch = to_device(data, device)
        data_format = input_batch.pop("data_format")
        target = input_batch["target"]

        eval_groups = data["groups"]

        for k in model_dict:
            model = model_dict[k][0] if len(model_dict[k]) > 1 else model_dict[k]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model.predict(input_batch, data_format=data_format)
            calc_dict[k].update(preds, target, eval_groups[:,0], eval_groups[:,1])

    res = {}
    for k in model_dict:
        res[k] = calc_dict[k].compute()
    return res


@torch.no_grad()
def evaluate_osiris(
    model: OmnichannelModel,
    accelerator: Accelerator,
    loader: torch.utils.data.DataLoader,
    calculator: BaseCalculatorClass, 
    head_name: str,
    num_feats_ind: List[int] = None,
    cat_feats_ind: List[int] = None,
    target_idx: int = 0,
) -> Dict[str, Any]:
    """
    Распределенный инференс с защитой от NCCL таймаутов (Early Stopping).
    Если у одного GPU заканчиваются данные, остальные синхронно прерывают цикл.
    """
    device = accelerator.device
    rank = accelerator.process_index
    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)
    except Exception:
        total_len = None
    pbar = tqdm(total=total_len, disable=not accelerator.is_local_main_process)

    local_preds, local_target, local_groups = [], [], []
    local_epk, local_dt = [], []          

    loader_iter = iter(loader)
    exhausted = False
    status = torch.zeros(1, dtype=torch.int32, device=device)

    while True:
        data = None
        if not exhausted:
            try:
                data = next(loader_iter)
                status[0] = 1
            except StopIteration:
                exhausted = True
                status[0] = 0
        else:
            status[0] = 0

        global_status = accelerator.gather(status)

        if data is not None:
            if num_feats_ind is not None:
                data['num_feats'] = data['num_feats'][:, num_feats_ind]
            if cat_feats_ind is not None:
                data['cat_feats'] = data['cat_feats'][:, cat_feats_ind]

            input_batch = to_device(data, device)
            data_format = input_batch.pop("data_format")
            target = input_batch["target"][:, target_idx:target_idx + 1] 
            eval_groups = input_batch["groups"]

            with accelerator.autocast():
                preds = model.predict(input_batch, data_format=data_format)
            preds = preds[head_name]
            
            valid = target[:, 0] >= 0                          
            if valid.any():
                local_preds.append(preds[valid].cpu())
                local_target.append(target[valid].cpu())
                local_groups.append(eval_groups[valid].cpu())

            pbar.update(1)

        if global_status.max() == 0:
            break
    pbar.close()

    accelerator.wait_for_everyone()

    local_result = {
        "preds":  torch.cat(local_preds)  if local_preds  else torch.tensor([], dtype=torch.float32),
        "target": torch.cat(local_target) if local_target else torch.tensor([], dtype=torch.int8),
        "groups": torch.cat(local_groups) if local_groups else torch.tensor([], dtype=torch.int32).reshape(0, 2),
    }

    gathered = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, local_result)

    if accelerator.is_main_process:
        ok = [d for d in gathered if d["preds"].numel() > 0]
        if not ok:
            print("[WARN] Нет валидных строк для расчёта метрик", flush=True)
            return {}
        all_preds  = torch.cat([d["preds"]  for d in ok])
        all_target = torch.cat([d["target"] for d in ok])
        all_groups = torch.cat([d["groups"] for d in ok])
        
        # if dump_path:
        #     torch.save({"preds": all_preds, "target": all_target, "groups": all_groups}, dump_path) 

        channel_ids = all_groups[:,1].to(device) if all_groups.shape[1] > 1 else None
        calculator.update(
            preds=all_preds.to(device),
            targets=all_target.to(device),
            product_ids=all_groups[:, 0].to(device),
            channel_ids=channel_ids,
        )
        return calculator.compute()
    return None

@torch.no_grad()
def evaluate_osiris_multi(
    model: OmnichannelModel,
    accelerator: Accelerator,
    loader: torch.utils.data.DataLoader,
    calculators: Dict[str, BaseCalculatorClass],
    head_targets: Dict[str, int],
    filter_valid: bool = True,
    dump_dir: Optional[str] = None,
    dump_tag: str = "",
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Один проход по loader для всех голов (head_targets: голова -> индекс колонки target).
    Конец итерации синхронизируется как в evaluate_osiris.
    Возвращает {голова: метрики} на main-ранге, None на остальных.
    """
    device = accelerator.device
    heads = list(head_targets)
    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)
    except Exception:
        total_len = None
    pbar = tqdm(total=total_len, disable=not accelerator.is_local_main_process)

    local = {h: {"preds": [], "target": [], "groups": []} for h in heads}
    loader_iter = iter(loader)
    exhausted = False
    status = torch.zeros(1, dtype=torch.int32, device=device)

    while True:
        data = None
        if not exhausted:
            try:
                data = next(loader_iter)
                status[0] = 1
            except StopIteration:
                exhausted = True
                status[0] = 0
        else:
            status[0] = 0

        global_status = accelerator.gather(status)

        if data is not None:
            input_batch = to_device(data, device)
            data_format = input_batch.pop("data_format")
            target = input_batch["target"]
            eval_groups = input_batch["groups"]

            with accelerator.autocast():
                preds = model.predict(input_batch, data_format=data_format, head_names=heads)

            for h in heads:
                t = target[:, head_targets[h]]
                p = preds[h].reshape(-1)
                g = eval_groups
                if filter_valid:
                    valid = t >= 0
                    p, t, g = p[valid], t[valid], g[valid]
                if p.numel() == 0:
                    continue
                local[h]["preds"].append(p.float().cpu())
                local[h]["target"].append(t.cpu())
                local[h]["groups"].append(g.cpu())

            pbar.update(1)

        if global_status.max() == 0:
            break
    pbar.close()

    accelerator.wait_for_everyone()

    local_result = {
        h: {
            "preds": torch.cat(b["preds"]) if b["preds"] else torch.empty(0, dtype=torch.float32),
            "target": torch.cat(b["target"]) if b["target"] else torch.empty(0, dtype=torch.int32),
            "groups": torch.cat(b["groups"]) if b["groups"] else torch.empty((0, 2), dtype=torch.int32),
        }
        for h, b in local.items()
    }
    del local

    gathered = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, local_result)
    del local_result

    if not accelerator.is_main_process:
        return None

    results = {}
    for h in heads:
        ok = [d[h] for d in gathered if d[h]["preds"].numel() > 0]
        if not ok:
            print(f"[WARN] голова '{h}': нет валидных строк для расчёта метрик", flush=True)
            results[h] = {}
            continue
        all_preds = torch.cat([d["preds"] for d in ok])
        all_target = torch.cat([d["target"] for d in ok])
        all_groups = torch.cat([d["groups"] for d in ok])
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            torch.save({"preds": all_preds, "target": all_target, "groups": all_groups},
                       os.path.join(dump_dir, f"preds_{dump_tag}_{h}.pt"))
        bad = int(((all_target < 0) | (all_target > 1)).sum())
        if bad:
            raise ValueError(f"голова '{h}': {bad} таргетов вне {{0,1}}")
        calculators[h].update(
            preds=all_preds,
            targets=all_target,
            product_ids=all_groups[:, 0],
            channel_ids=all_groups[:, 1] if all_groups.shape[1] > 1 else None,
        )
        results[h] = calculators[h].compute()
        calculators[h].reset()
    return results



@torch.no_grad()
def score_osiris(
    model: torch.nn.Module,
    accelerator: "Accelerator",
    loader: torch.utils.data.DataLoader,
    output_dir: str,
    decode_map: dict = None,
    save_every_n_batches: int = 100
):
    device = accelerator.device
    rank = accelerator.process_index
    
    # Создаем локальную директорию на главной видеокарте, остальные ждут
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)
    except Exception:
        total_len = None
        
    pbar = tqdm(
        total=total_len,
        desc="Scoring OOT",
        disable=not accelerator.is_local_main_process,
    )
    
    loader_iter = iter(loader)
    
    buf_epk = []
    buf_dt = []
    buf_prod = []
    buf_score = []
    file_counter = 0
    
    def flush_buffer() -> None:
        nonlocal buf_epk, buf_dt, buf_prod, buf_score, file_counter
        if not buf_score:
            return
            
        # 1. Создаем плоскую таблицу
        df = pd.DataFrame({
            "epk_id": buf_epk,
            "report_dt": buf_dt,
            "eval_prod": buf_prod,
            "score": buf_score,
        })
        
        buf_epk.clear(); buf_dt.clear(); buf_prod.clear(); buf_score.clear()
        
        # 2. Чистим дубликаты и пустые ID
        df = df.drop_duplicates(["epk_id", "report_dt", "eval_prod"])
        df = df.dropna(subset=["epk_id", "score"])
        
        # 3. Декодируем продукты
        if decode_map is not None:
            safe_decode_map = {str(k): str(v) for k, v in decode_map.items()}
            df["eval_prod"] = df["eval_prod"].astype(str).map(safe_decode_map).fillna(df["eval_prod"]).astype(str)
        else:
            df["eval_prod"] = df["eval_prod"].astype(str)
            
        # 4. Приводим типы
        df["epk_id"] = df["epk_id"].astype("int64")
        df["report_dt"] = df["report_dt"].astype(str)
        df["score"] = df["score"].astype("float32")
        
        if df.empty:
            return
            
        out_path = os.path.abspath(os.path.join(output_dir, f"additional_scores_rank_{rank}_part_{file_counter}.parquet"))
        
        # 5. Сохраняем как ПРОСТУЮ таблицу ЛОКАЛЬНО
        try:
            df.to_parquet(out_path, index = False, engine = "pyarrow")
        except Exception as e:
            print(f"[Rank {rank}] ОШИБКА ЗАПИСИ НА ДИСК: Не могу сохранить {out_path}. Причина: {e}")
            return

        file_counter += 1
    # =========================================================================
    # ОСНОВНОЙ ЦИКЛ ИНФЕРЕНСА
    # =========================================================================
    batch_idx = 0
    while True:
        try:
            data = next(loader_iter)
        except StopIteration:
            # Локальные файлы для ЭТОЙ видеокарты просто закончились. Выходим.
            print(f"\n[Rank {rank}] Данные закончились, выход из цикла.", flush=True)
            break
        if batch_idx % 500 == 0:
            print(f"\n[Rank {rank}] batch_idx = {batch_idx}, buf_size = {len(buf_score)}", flush=True)
            
        input_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model.spark_inference(input_batch)
            
        # Забираем данные на CPU
        preds_cpu = preds.float().cpu().view(-1).numpy()
        prods_cpu = data["groups"][:, 0].cpu().numpy()  # Индекс 0 = eval_prod
        
        buf_epk.extend(data["epk_id"])
        buf_dt.extend(data["report_dt"])
        buf_prod.extend(prods_cpu)
        buf_score.extend(preds_cpu)
        
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        batch_idx += 1
        if batch_idx % save_every_n_batches == 0:
            flush_buffer()
            
        pbar.update(1)
        
    # Обязательно сбрасываем остаток буфера после выхода из цикла
    flush_buffer()
    pbar.close()
    
    # =========================================================================
    # ФИНАЛИЗАЦИЯ И СИНХРОНИЗАЦИЯ
    # =========================================================================
    
    print(f"[Rank {rank}] Готов. Ожидаю завершения остальных видеокарт...", flush=True)
    
    # Барьер: ждем, пока все видеокарты допишут свои последние файлы
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("\n[INFO] Инференс полностью завершен на всех GPU! Все файлы сохранены локально в:", output_dir, flush=True)
