import torch
from abc import ABC
from typing import Union, List, Dict, Optional, Any, Callable, Tuple

MetricFunc = Callable[[torch.Tensor, torch.Tensor], float]


class BaseCalculatorClass(ABC):
    """Единый класс для расчета метрик при батчевом evaluate.
    Работает с режимами general, product, channel и product-in-channel
    Принимает функции для расчета метрик, что позволяет считать несколько метрик, храня скоры и таргеты единожды
    """

    def __init__(
        self,
        metrics: Dict[str, MetricFunc],
        modes: Union[str, List[str]], 
        device: torch.device = torch.device('cpu'),
        decode_maps: Optional[Dict[str, Dict[int, str]]] = None
    ) -> None:
        self.metrics_funcs = metrics
        self.device = device
        self.decode_maps = decode_maps or {}

        if isinstance(modes, str):
            modes = [modes]
        self.modes = list(set(modes))

        # Единый буфер: список кортежей (preds, targets, pids, cids)
        # Храним всё на CPU, чтобы не занимать GPU память во время эпохи
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = []

    def reset(self) -> None:
        self.buffer = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        product_ids: Optional[torch.Tensor] = None,
        channel_ids: Optional[torch.Tensor] = None
    ) -> None:
        """
        Сохраняет предсказания и таргеты для дальнейшего расчета.
        """
        # Detach и CPU обязательны для экономии памяти
        p_cpu = preds.detach().cpu()
        t_cpu = targets.detach().cpu()
        pid_cpu = product_ids.detach().cpu() if product_ids is not None else None
        cid_cpu = channel_ids.detach().cpu() if channel_ids is not None else None

        self.buffer.append((p_cpu, t_cpu, pid_cpu, cid_cpu))

    def compute(self) -> Dict[str, Any]:
        """
        Финальная сборка и расчет всех переданных метрик.
        """
        if not self.buffer:
            return {}

        # 1. Склеиваем всё в огромные тензоры один раз
        # zip(*self.buffer) транспонирует список кортежей в кортеж списков
        preds_list, targets_list, pids_list, cids_list = zip(*self.buffer)

        all_preds = torch.cat(preds_list)
        all_targets = torch.cat(targets_list)

        # Для ID нужно проверить, не None ли они
        all_pids = torch.cat(pids_list) if pids_list[0] is not None else None
        all_cids = torch.cat(cids_list) if cids_list[0] is not None else None

        results = {}

        # 2. Проходим по режимам и фильтруем один общий тензор
        # TO DO: убрать hardcode режимов, чтобы можно было переиспользовать для evk, cs и тд

        # --- GENERAL ---
        if "general" in self.modes:
            # Считаем на всем массиве
            results["general"] = self._calc_metrics_on_slice(all_preds, all_targets)

        # --- PRODUCT ---
        if "product" in self.modes and all_pids is not None:
            results["product"] = self._calc_grouped_metrics(
                all_preds, all_targets, groups=all_pids, map_name="product"
            )

        # --- CHANNEL ---
        if "channel" in self.modes and all_cids is not None:
            results["channel"] = self._calc_grouped_metrics(
                all_preds, all_targets, groups=all_cids, map_name="channel"
            )

        # --- PRODUCT_CHANNEL ---
        if "product_channel" in self.modes and all_pids is not None and all_cids is not None:
            results["product_channel"] = self._calc_double_grouped_metrics(
                all_preds, all_targets, all_pids, all_cids
            )

        return results

    def _calc_metrics_on_slice(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Расчет метрик на конкретном срезе данных."""
        # Переносим на девайс только для расчета
        if self.device.type != 'cpu':
            preds = preds.to(self.device)
            targets = targets.to(self.device)

        res = {}
        for name, func in self.metrics_funcs.items():
            try:
                val = func(preds, targets)
                res[name] = val.item() if isinstance(val, torch.Tensor) else val
            except Exception:
                res[name] = 0.0
        return res

    def _calc_grouped_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor,
        map_name: str
    ) -> Dict[str, Dict[str, float]]:
        """Группировка по одному ID (product или channel)."""
        res = {}
        unique_groups = torch.unique(groups)

        for g_id in unique_groups:
            mask = groups == g_id
            metrics = self._calc_metrics_on_slice(preds[mask], targets[mask])

            # Декодинг имени
            real_name = self.decode_maps.get(map_name, {}).get(int(g_id.item()), int(g_id.item()))
            res[real_name] = metrics
        return res

    def _calc_double_grouped_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        pids: torch.Tensor,
        cids: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Группировка по паре (product, channel)."""
        res = {}
        # Находим уникальные пары
        pairs = torch.unique(torch.stack([pids, cids], dim=1), dim=0)

        for pair in pairs:
            pid, cid = pair[0], pair[1]
            mask = (pids == pid) & (cids == cid)
            metrics = self._calc_metrics_on_slice(preds[mask], targets[mask])

            # Декодинг имени
            p_name = self.decode_maps.get("product", {}).get(int(pid.item()), int(pid.item()))
            c_name = self.decode_maps.get("channel", {}).get(int(cid.item()), int(cid.item()))
            key_name = f"{p_name}_{c_name}"

            res[key_name] = metrics
        return res