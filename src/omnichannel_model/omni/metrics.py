import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_auroc, binary_average_precision, binary_calibration_error
from functools import partial
from typing import Dict, List, Any, Union
import pandas as pd

def lift_at_k(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: float = 0.1
) -> float:
    """
    Считает Lift на топ-K процентах выборки.
    Lift = (Response Rate in top K%) / (Global Response Rate)

    Args:
        preds: Вероятности (0..1)
        targets: Бинарные метки (0 или 1)
        k: Доля выборки (например, 0.1 для топ-10%)
    """
    # Гарантируем float и правильную размерность
    preds = preds.float().view(-1)
    targets = targets.float().view(-1)

    n_samples = preds.shape[0]
    n_top_k = int(n_samples * k)

    if n_top_k == 0:
        return 0.0

    # Глобальный Response Rate
    global_rate = targets.mean().item()

    if global_rate == 0:
        return 0.0

    # Сортируем по убыванию скора
    _, indices = torch.sort(preds, descending=True)
    top_k_indices = indices[:n_top_k]

    # Response Rate в топе
    top_k_rate = targets[top_k_indices].mean().item()

    return top_k_rate / global_rate


def cumulative_gain_at_k(
    preds: torch.Tensor,
    targets: torch.Tensor,
    k: float = 0.1
) -> float:
    """
    Считает Cumulative Gain (Recall) на топ-K процентах.
    CG@K = (Кол-во единиц в топ-K) / (Всего единиц в выборке)

    Это показывает, какую долю от всех целевых клиентов мы "поймали",
    взяв только топ-K процентов базы.
    """
    preds = preds.float().view(-1)
    targets = targets.float().view(-1)

    total_positives = targets.sum().item()
    if total_positives == 0:
        return 0.0

    n_samples = preds.shape[0]
    n_top_k = int(n_samples * k)

    if n_top_k == 0:
        return 0.0

    # Сортируем
    _, indices = torch.sort(preds, descending=True)
    top_k_indices = indices[:n_top_k]

    # Сколько единиц поймали
    captured_positives = targets[top_k_indices].sum().item()

    return captured_positives / total_positives


calc_metrics_dict = {
    "gini": lambda p, t: 200 * binary_auroc(p, t).item() - 100,
    "pr_auc": lambda p, t: binary_average_precision(p, t).item(),
    "avg_cr": lambda p, t: torch.mean(t.float()).item(),
    "sales": lambda p, t: torch.sum(t).item(),
    "communications": lambda p, t: t.shape[0],
    "logloss": lambda p, t: nn.functional.binary_cross_entropy(p.float(), t.float()).item(),
    "ece": lambda p, t: binary_calibration_error(p, t, n_bins=20).item(),
    "lift@10%": partial(lift_at_k, k=0.1),
    "lift@30%": partial(lift_at_k, k=0.3),
    "lift@50%": partial(lift_at_k, k=0.5),
    "cgain@5%": partial(cumulative_gain_at_k, k=0.05),
    "cgain@10%": partial(cumulative_gain_at_k, k=0.1),
    "cgain@50%": partial(cumulative_gain_at_k, k=0.5)
}


def flatten_metrics_dict(
    metrics_dict: Union[
        Dict[str, 
             Dict[str, 
                  Dict[str, 
                       float]
                 ]
            ], 
        Dict[str, 
             Dict[str, 
                  Dict[str, 
                       Dict[str, 
                            float]
                      ]
                 ]
            ]
    ],
    slice_types: List[str] = None,
    sep: str = '_'
):
    """
    Превращает вложенный словарь метрик в плоский список записей.
    Предполагается структура:
    Модель -> Тип среза (Общие/Каналы/...) -> Имя среза (Канал_1) -> Метрики
    """
    rows = []

    # 1. Уровень: Модель (ModelName)
    for model_name, model_data in metrics_dict.items():

        # 2. Уровень: Тип среза (General, Channels, Products, ...)
        for slice_type, slice_data in model_data.items():
            if slice_types is not None:
                if slice_type not in slice_types:
                    continue
            # Обработка "Общих" метрик (у них нет подуровня "Имя среза")
            # Если словарь сразу содержит числа -> это метрики "General"
            if all(isinstance(v, (int, float)) for v in slice_data.values()):
                # Пример: {'gini': 0.5, 'auc': 0.8}
                row = {
                    'model': model_name,
                    'slice_type': slice_type,  # Например "general"
                    'slice_name': 'All',
                }
                # Добавляем сами метрики
                row.update(slice_data)
                rows.append(row)
                continue

            # 3. Уровень: Имя конкретного среза (Канал_1, Продукт_А)
            for slice_entity, metrics in slice_data.items():

                # Собираем строку
                row = {
                    'model': model_name,
                    'slice_type': slice_type,  # Например "channel"
                    'slice_name': slice_entity,  # Например "massvsp"
                }

                # 4. Уровень: Сами метрики (gini, gain, ...)
                if isinstance(metrics, dict):
                    row.update(metrics)
                    rows.append(row)
                else:
                    # Если вдруг структура нарушена
                    print(f"Warning: Unexpected structure at {model_name}->{slice_type}->{slice_entity}")

    return pd.DataFrame(rows)
