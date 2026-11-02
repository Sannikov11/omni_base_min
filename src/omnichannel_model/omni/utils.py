import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional

import warnings


def build_optimizer(
    model: nn.Module,
    lrs: List[float] = [2e-4, 5e-4, 1e-3, 1e-5, 1e-5],
    wds: List[float] = [1e-5, 1e-4, 1e-2, 1e-5, 1e-5],
    mt_lrs: Optional[Dict[str, float]] = None,
    mt_wds: Optional[Dict[str, float]] = None,

) -> torch.optim.Optimizer:
    """
    Создает AdamW оптимизатор с разделением параметров на 4 группы:
    1. Embeddings (PLE, Cat, Proj Embeddings) -> Low LR, Low WD
    2. DCN Cross Layers (U, V, W, Gates)      -> Mid LR, Mid WD
    3. Main Body (Deep MLP, Heads, Projections)-> High LR, High WD
    4. No Decay (Bias, LayerNorm, BatchNorm)   -> High LR, 0.0 WD

    Fine-tuning:
    5. DLT weights                 -> lrs[3], wds[3]
    6. DLT no-decay (bias, norm)   -> lrs[3], wd=0.0
    7. EVK weights                 -> lrs[4], wds[4]
    8. EVK no-decay (bias, norm)   -> lrs[4], wd=0.0
    
    Args:
        model: nn.Module
        lrs: [lr_emb, lr_cross, lr_main, lr_dlt, lr_evk] (default: [2e-4, 5e-4, 1e-3, 1e-5, 1e-5])
        wds: [wd_emb, wd_cross, wd_main, wd_dlt, wd_evk] (default: [1e-5, 1e-4, 1e-2, 1e-5, 1e-5])
        
    """

    # Множества для идентификации параметров (используем id объекта, так как параметры хэшируемы)
    embed_ids = set()
    norm_bias_ids = set()
    cross_ids = set()

    # 1. Проход по модулям для классификации (Embeddings, Norms, CrossNet)
    for name, m in model.named_modules():
        cls_name = m.__class__.__name__.lower()

        # --- Группа 1: Эмбеддинги ---
        # Ищем nn.Embedding, nn.EmbeddingBag или классы со словом "embedding"
        if isinstance(m, (nn.Embedding, nn.EmbeddingBag)) or ("embedding" in cls_name and "projection" not in cls_name):
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    embed_ids.add(p)

        # --- Группа 4 (часть): Norm Layers ---
        # Ищем BatchNorm, LayerNorm, GroupNorm или классы со словом "norm"
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                            nn.GroupNorm)) or ("norm" in cls_name):
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    norm_bias_ids.add(p)

        # --- Группа 2: Cross Network (DCN) ---
        # Ищем модуль с именем атрибута "cross_network" или класс, содержащий "crossnet"
        # Важно: параметры U, V, W внутри CrossNet обычно nn.Parameter, а не подмодули
        elif "cross_net" in name or "cross" in cls_name or "projection"  in cls_name:
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    # Bias внутри DCN тоже не должен иметь WD, но 
                    # мы обработаем это ниже по имени параметра (".bias")
                    cross_ids.add(p)

    # 2. Распределение параметров по спискам для оптимизатора
    group_embed = [] # lr[0], wd[0]
    group_cross = [] # lr[1], wd[1]
    group_main = [] # lr[2], wd[2]
    group_nowd = [] # lr[2], 0.0
    group_dlt = [] # lr[3], wd[3]
    group_dlt_nowd = [] # lr[3], 0
    group_evk = [] # lr[4], wd[4]
    group_evk_nowd = [] # lr[4], 0
    mt_groups = {(k, nw): [] for k in ("experts", "gates", "towers", "adapter") for nw in (False, True)}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if mt_lrs is not None and name.startswith("mt."):
            kind = "gates" if name.startswith("mt.gates.") else ("towers" if name.startswith(("mt.towers.", "mt.general_towers.")) else ("adapter" if name.startswith("mt.adapters.") else "experts"))
            mt_groups[(kind, (p in norm_bias_ids) or "bias" in name)].append(p)
            continue


        is_bias_or_norm = (p in norm_bias_ids) or name.endswith(".bias") or "bias" in name
        
        if name.startswith("dlt_model."):
            group_dlt_nowd.append(p) if is_bias_or_norm else group_dlt.append(p)
            continue

        if name.startswith("evk_model."):
            group_evk_nowd.append(p) if is_bias_or_norm else group_evk.append(p)
            continue
            
        # Bias и Norm (No Decay)
        if is_bias_or_norm:
            group_nowd.append(p)
            continue

        # Эмбеддинги
        if p in embed_ids:
            group_embed.append(p)
            continue

        # Cross Network
        if (p in cross_ids) or ("cross_network" in name):
            group_cross.append(p)
            continue

        # Все остальное
        # Сюда попадут веса Linear слоев (MLP, Projections, Heads)
        group_main.append(p)

    # 3. Создание оптимизатора
    optimizer = torch.optim.AdamW(
        [
            # 1. Embeddings (PLE, Cat) -> Аккуратное обучение
            {"params": group_embed, "lr": lrs[0], "weight_decay": wds[0]}, 

            # 2. Cross Network -> Средний темп
            {"params": group_cross, "lr": lrs[1], "weight_decay": wds[1]}, 

            # 3. Main Body (Deep MLP) -> Стандартный темп и регуляризация
            {"params": group_main,  "lr": lrs[2], "weight_decay": wds[2]}, 

            # 4. Bias & Norm -> Без регуляризации, темп как у Main
            {"params": group_nowd,  "lr": lrs[2], "weight_decay": 0.0}, 
            
            {"params": group_dlt,      "lr": lrs[3], "weight_decay": wds[3]},
            
            {"params": group_dlt_nowd, "lr": lrs[3], "weight_decay": 0.0},
            
            {"params": group_evk,      "lr": lrs[4], "weight_decay": wds[4]},
            
            {"params": group_evk_nowd, "lr": lrs[4], "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    if mt_lrs is not None:
        wds_mt = mt_wds or {}
        for (kind, nowd), ps in mt_groups.items():
            if ps:
                optimizer.add_param_group({
                    "params": ps,
                    "lr": mt_lrs.get(kind, lrs[2]),
                    "weight_decay": 0.0 if nowd else wds_mt.get(kind, wds[2]),
                    "name": f"mt_{kind}{'_nowd' if nowd else ''}",
                })


    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    num_epochs: int,
    warmup_ratio: float = 0.05,
    eta_min: float = 1e-6
) -> torch.optim.lr_scheduler.LRScheduler:

    num_training_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_ratio * num_training_steps))

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,     # начинаем с 10% от базового LR
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_training_steps - warmup_steps),
        eta_min=eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    return scheduler


def _check_bins(bins: List[torch.Tensor]) -> None:
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, torch.Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        # if len(feature_bins) == 2:
        #     warnings.warn(
        #         f'The {i}-th feature has just two bin edges, which means only one bin.'
        #         ' Strictly speaking, using a single bin for the'
        #         ' piecewise-linear encoding should not break anything,'
        #         ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
        #     )


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    handle_missings: bool = True
) -> List[torch.Tensor]:
    """Compute the bin boundaries for `PiecewiseLinearEncoding` and `PiecewiseLinearEmbeddings`.
 
    **Usage**
 
    Compute bins using quantiles (Section 3.2.1 in the paper):
 
    >>> X_train = torch.randn(10000, 2)
    >>> bins = compute_bins(X_train)
 
    Args:
        X: the training features.
        n_bins: the number of bins.
        handle_missings: flag if NaNs should be eliminated from calculation
 
    Returns:
        A list of bin edges for all features. For one feature:
 
        - the maximum possible number of bin edges is ``n_bins + 1``.
        - the minimum possible number of bin edges is ``1``.
    """  # noqa: E501
    if not isinstance(X, torch.Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')

    if handle_missings:
        finite_mask = X.isfinite()
        if not finite_mask.any():
            raise ValueError('X contains only nan/inf/-inf values.')
    else:
        if not X.isfinite().all():
            raise ValueError('X must not contain nan/inf/-inf.')

    if (X == X[0]).all(dim=0).any():
        raise ValueError(
            'All columns of X must have at least two distinct values.'
            ' However, X contains columns with just one distinct value.'
        )
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            'n_bins must be more than 1, but less than len(X), however:'
            f' {n_bins=}, {len(X)=}'
        )

    _upper = 2**24  # 16_777_216
    if len(X) > _upper:
        warnings.warn(
            f'Computing quantile-based bins for more than {_upper} million objects'
            ' may not be possible due to the limitation of PyTorch'
            ' (for details, see https://github.com/pytorch/pytorch/issues/64947;'
            ' if that issue is successfully resolved, this warning may be irrelevant).'  # noqa
            ' As a workaround, subsample the data, i.e. instead of'
            '\ncompute_bins(X, ...)'
            '\ndo'
            '\ncompute_bins(X[torch.randperm(len(X), device=X.device)[:16_777_216]], ...)'  # noqa
            '\nOn CUDA, the computation can still fail with OOM even after'
            ' subsampling. If this is the case, try passing features by groups:'
            '\nbins = sum('
            '\n    compute_bins(X[:, idx], ...)'
            '\n    for idx in torch.arange(len(X), device=X.device).split(group_size),'  # noqa
            '\n    start=[]'
            '\n)'
            '\nAnother option is to perform the computation on CPU:'
            '\ncompute_bins(X.cpu(), ...)'
       )
    del _upper
    bins = []
    for j in tqdm(range(X.shape[1])):
        col = X[:, j]
        if handle_missings:
            col = col[col.isfinite()]
            if len(col) == 0:
                bins.append(torch.tensor([0.0, 1.0], device=X.device, dtype=X.dtype))
                continue
            q = torch.quantile(col, torch.linspace(0.0, 1.0, n_bins + 1).to(col), dim=0)
            bins.append(q.unique())

    _check_bins(bins)
    return bins