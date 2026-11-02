from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from omnichannel_model.omni.cycles import to_device


def _log_names(cfg, channel_names: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    base = {}
    for t in cfg.tasks:
        k = cfg.channels_per_task[t]
        names = (channel_names or {}).get(t)
        if names is None:
            names = [str(c) for c in cfg.channel_ids[t]] if k > 1 else ["all"]
        if len(names) != k:
            raise ValueError(f"channel_names[{t}] должен содержать {k} имён")
        base[t] = list(names)
    return {key: (["all"] if kind == "general" else base[t]) for key, (t, kind) in cfg.loss_keys().items()}


def _validate_first_batch(batch: Dict[str, torch.Tensor], cfg) -> str:
    if "groups" not in batch:
        raise RuntimeError(
            "В train-батче нет 'groups': передайте --evaluate-cols (например eval_prod,eval_channel); "
            "маршрутизация CR по каналу невозможна без этой колонки."
        )
    tgt, grp = batch["target"], batch["groups"]
    if tgt.shape[1] <= max(cfg.target_cols.values()):
        raise RuntimeError(f"target имеет {tgt.shape[1]} колонок, а target_cols={cfg.target_cols}")
    if grp.shape[1] <= cfg.channel_col:
        raise RuntimeError(f"groups имеет {grp.shape[1]} колонок, а channel_col={cfg.channel_col}")
    parts = []
    for t in cfg.tasks:
        col = tgt[:, cfg.target_cols[t]]
        lab = col >= 0
        info = f"{t}: labeled={int(lab.sum())}/{col.numel()}, pos={int((col == 1).sum())}"
        if cfg.channels_per_task[t] > 1:
            vals, cnts = grp[:, cfg.channel_col][lab].unique(return_counts=True)
            known = set(cfg.channel_ids[t])
            info += f", channels={dict(zip(vals.tolist(), cnts.tolist()))}"
            unknown = [v for v in vals.tolist() if v not in known]
            if unknown:
                info += f" !!! неизвестные каналы {unknown}"
        parts.append(info)
    return " | ".join(parts)


def _cos(ga, gb) -> Optional[torch.Tensor]:
    pairs = [(a.float(), b.float()) for a, b in zip(ga, gb) if a is not None and b is not None]
    if not pairs:
        return None
    dot = sum((a * b).sum() for a, b in pairs)
    na = sum((a * a).sum() for a, _ in pairs).sqrt()
    nb = sum((b * b).sum() for _, b in pairs).sqrt()
    return dot / (na * nb).clamp_min(1e-12)


def _grad_conflict(L, params, pairs: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
    names = sorted({n for pair in pairs for n in pair})
    grads = {n: torch.autograd.grad(L.task_terms[n], params, retain_graph=True, allow_unused=True) for n in names}
    res = {}
    for a, b in pairs:
        c = _cos(grads[a], grads[b])
        if c is None:
            continue
        both = (L.n_valid[a].sum() > 0) & (L.n_valid[b].sum() > 0)
        res[f"{a}_{b}"] = torch.where(both, c, torch.full_like(c, float("nan")))
    return res


def train_multitask_osiris(
    model,
    accelerator,
    loader,
    optimizer,
    scheduler,
    writer,
    epoch: int,
    log_every: int = 10,
    gc_every: int = 0,
    step_offset: int = 0,
    channel_names: Optional[Dict[str, List[str]]] = None,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    core = model.module
    core.train()
    cfg = core.mt_config
    if cfg is None:
        raise RuntimeError("train_multitask_osiris: у модели нет mt_config — OmnichannelModel создана без mt_config=...")
    tasks = list(cfg.tasks)
    names = _log_names(cfg, channel_names)
    keys = list(names)
    device = accelerator.device
    trainable = [p for p in model.parameters() if p.requires_grad]
    adapter_ids = {id(p) for p in core.mt.adapters.parameters()}
    adapter_params = [p for p in trainable if id(p) in adapter_ids]
    base_params = [p for p in trainable if id(p) not in adapter_ids]
    shared_params = [p for p in core.mt.shared.parameters() if p.requires_grad]

    conflict_pairs = [(a, b) for i, a in enumerate(tasks) for b in tasks[i + 1:]]
    conflict_pairs += [(f"{t}_general", f"{t}_adj") for t in cfg.adapter_tasks if cfg.adapter_grad != "isolated"]

    zero = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)
    sum_t = {k: zero(()) for k in keys}
    cnt_t = {k: zero(()) for k in keys}
    sum_c = {k: zero(len(names[k])) for k in keys}
    cnt_c = {k: zero(len(names[k])) for k in keys}
    unk = {t: zero(()) for t in tasks}
    lab = {t: zero(()) for t in tasks}
    total_loss = zero(())
    gnorm = zero(())
    gnorm_a = zero(())

    try:
        total_len = int(len(loader) // accelerator.num_processes * 1.1)
    except Exception:
        total_len = None
    pbar = tqdm(loader, total=total_len, disable=not accelerator.is_local_main_process)
    data_iter = iter(pbar)
    batch_idx = 0

    while True:
        has_data = torch.ones(1, device=device)
        try:
            data = next(data_iter)
        except StopIteration:
            has_data.zero_()
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
        if has_data.item() < 0.5:
            break

        batch = to_device(data, device)
        data_format = batch.pop("data_format")
        if batch_idx == 0:
            info = _validate_first_batch(batch, cfg)
            if accelerator.is_local_main_process:
                print(f"[MT] первый батч: {info}", flush=True)
        target = batch["target"]
        sample_weights = batch.get("sample_weight")

        optimizer.zero_grad()
        outputs = model(batch, head_names=["mt"], data_format=data_format)
        L = core.calculate_mt_loss(outputs, target, sample_weights)

        gc_val = None
        if gc_every > 0 and batch_idx % gc_every == 0 and conflict_pairs and shared_params:
            gc_val = _grad_conflict(L, shared_params, conflict_pairs)

        accelerator.backward(L.total)
        gnorm = torch.nn.utils.clip_grad_norm_(base_params, max_grad_norm)
        if adapter_params:
            gnorm_a = torch.nn.utils.clip_grad_norm_(adapter_params, max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            for k in keys:
                nv = L.n_valid[k]
                n = nv.sum()
                sum_t[k] += L.per_task[k] * n
                cnt_t[k] += n
                sum_c[k] += L.per_channel[k] * nv
                cnt_c[k] += nv
            for t in tasks:
                col = target[:, cfg.target_cols[t]]
                has_lab = col >= 0
                lab[t] += has_lab.sum()
                unk[t] += (has_lab & (outputs["mt"][t]["index"] < 0)).sum()
            total_loss += L.total.detach()

        step = step_offset + batch_idx
        if batch_idx % log_every == 0 and accelerator.is_local_main_process:
            items = [("total", total_loss / (batch_idx + 1)), ("gnorm", gnorm), ("gnorm_a", gnorm_a)]
            for k in keys:
                items.append((f"loss/{k}", sum_t[k] / cnt_t[k].clamp_min(1.0)))
                for i, nm in enumerate(names[k]):
                    items.append((f"loss/{k}/{nm}", (sum_c[k] / cnt_c[k].clamp_min(1.0))[i]))
                    items.append((f"nvalid/{k}/{nm}", L.n_valid[k][i]))
            for t in tasks:
                items.append((f"ent/{t}", L.gate_entropy[t]))
                items.append((f"unk/{t}", unk[t] / lab[t].clamp_min(1.0)))
                for i in range(L.gate_util[t].shape[0]):
                    items.append((f"util/{t}/e{i}", L.gate_util[t][i]))
            for xk, xv in L.extras.items():
                flat = xv.detach().float().reshape(-1)
                labels = names.get(xk.split("/")[0]) if flat.numel() > 1 else None
                if labels is None or len(labels) != flat.numel():
                    labels = [str(i) for i in range(flat.numel())]
                for i in range(flat.numel()):
                    items.append((f"x/{xk}" + (f"/{labels[i]}" if flat.numel() > 1 else ""), flat[i]))
            vals = dict(zip([k for k, _ in items],
                            torch.stack([v.detach().float().reshape(()) for _, v in items]).tolist()))
            if not math.isfinite(vals["total"]):
                print(f"Warning: non-finite loss, batch {batch_idx}", flush=True)
            pbar.set_description(
                f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {vals['total']:.4f}, "
                + ", ".join(f"{k}: {vals[f'loss/{k}']:.4f}" for k in keys)
                + f", GradNorm: {vals['gnorm']:.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            if writer is not None:
                writer.add_scalar("Train Loss/total", vals["total"], step)
                writer.add_scalar("Grad/norm", vals["gnorm"], step)
                if adapter_params:
                    writer.add_scalar("Grad/norm_adapter", vals["gnorm_a"], step)
                for k in keys:
                    writer.add_scalar(f"Train Loss/{k}", vals[f"loss/{k}"], step)
                    for nm in names[k]:
                        writer.add_scalar(f"Train Loss/{k}/{nm}", vals[f"loss/{k}/{nm}"], step)
                        writer.add_scalar(f"Batch/n_valid/{k}/{nm}", vals[f"nvalid/{k}/{nm}"], step)
                for t in tasks:
                    n_exp = L.gate_util[t].shape[0]
                    writer.add_scalar(f"Gate/entropy/{t}", vals[f"ent/{t}"], step)
                    if n_exp > 1:
                        writer.add_scalar(f"Gate/entropy_norm/{t}", vals[f"ent/{t}"] / math.log(n_exp), step)
                    writer.add_scalar(f"Batch/unknown_channel_frac/{t}", vals[f"unk/{t}"], step)
                    for i in range(n_exp):
                        writer.add_scalar(f"Gate/util/{t}/e{i}", vals[f"util/{t}/e{i}"], step)
                for tag, v in vals.items():
                    if tag.startswith("x/"):
                        writer.add_scalar("Adapter/" + tag[2:], v, step)
                for gi, g in enumerate(optimizer.param_groups):
                    if g["params"]:
                        writer.add_scalar(f"LR/{g.get('name', gi)}", g["lr"], step)

        if gc_val is not None and writer is not None:
            for k, v in gc_val.items():
                v = v.item()
                if not math.isnan(v):
                    writer.add_scalar(f"GradConflict/cos_{k}", v, step)

        batch_idx += 1

    if batch_idx == 0:
        raise RuntimeError("train_multitask_osiris: 0 батчей за эпоху — проверьте колонки evaluate_cols/target, тип eval_channel и логи [Error processing file]")

    res = {}
    for k in keys:
        res[k] = float(sum_t[k] / cnt_t[k].clamp_min(1.0))
        mean_c = (sum_c[k] / cnt_c[k].clamp_min(1.0)).tolist()
        for nm, v in zip(names[k], mean_c):
            res[f"{k}/{nm}"] = v
    return res
