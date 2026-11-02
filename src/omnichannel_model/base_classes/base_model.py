import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type, TypeVar
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime
import warnings
from omnichannel_model.base_classes.base_configs import BaseModelArtifactsClass, BaseModelConfigClass

logger = logging.getLogger(__name__)
MetricFunc = Callable[[torch.Tensor, torch.Tensor], float]


class BaseModelClass(nn.Module, ABC):
    """
    Абстрактный базовый класс для всех ML моделей.
    
    Обеспечивает единый интерфейс для:
    - Обучения и инференса
    - Сохранения/загрузки весов, конфигураций и артефактов
    - Версионирования моделей
    """
    
    # Можно переопределить в наследниках для указания конкретных классов конфига/артефактов
    ConfigClass = BaseModelConfigClass 
    ArtifactsClass = BaseModelArtifactsClass

    def __init__(
        self, 
        config: Optional[BaseModelConfigClass] = None, 
        artifacts: Optional[BaseModelArtifactsClass] = None
    ):
        super().__init__()
        self.config = config
        self.artifacts = artifacts
        self._version = config.model_version
        self._device = torch.device("cpu")
    
    def _init_weights(self, module):
        """Инициализация весов модели, которые не относятся к специальным блокам.
        Используется через self.apply(self._init_weights())
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        
    @abstractmethod
    def forward(self, input_batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Прямой проход через модель.
        Args:
            input_batch: Словарь с входными данными для модели
            **kwargs: Дополнительные параметры
        Returns:
            Выходной тензор
        """
        raise NotImplementedError("Метод forward должен быть реализован в дочернем классе")
    
    @abstractmethod
    def calculate_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Вычисление функции потерь.
        Args:
            outputs: Предсказания модели
            targets: Целевые значения
            **kwargs: Дополнительные параметры
        Returns:
            Значение функции потерь
        """
        raise NotImplementedError("Метод calculate_loss должен быть реализован в дочернем классе")

    @abstractmethod
    def predict(self, input_batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        pass
    
    # ==================== Методы управления весами и чекпоинтами ====================
    
    def save_checkpoint(
        self, 
        path: Union[str, Path], 
        metadata: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        save_artifacts: bool = True,
        **extra_data
    ) -> None:
        """
        Сохранение полного чекпоинта модели.
        
        Args:
            path: Путь для сохранения
            metadata: Дополнительная метаинформация
            optimizer: Оптимизатор для сохранения его состояния
            epoch: Номер эпохи
            save_artifacts: Сохранять ли артефакты внутри чекпоинта (может быть тяжело)
            **extra_data: Любые дополнительные данные
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(), # Сохраняем словарь конфига
            "model_version": self._version,
            "model_class": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Опционально сохраняем артефакты прямо в чекпоинт
        # Альтернатива: сохранять их отдельно через artifacts.save()
        if save_artifacts and self.artifacts is not None:
            checkpoint["artifacts"] = self.artifacts # Pickle сериализует dataclass
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint["epoch"] = epoch
        
        if metadata is not None:
            checkpoint["metadata"] = metadata
        
        checkpoint.update(extra_data)
        
        torch.save(checkpoint, path)
        logger.info(f"Чекпоинт сохранен в {path}")
    
    def load_checkpoint(
        self, 
        path: Union[str, Path],
        strict: bool = True,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Загрузка весов и состояния оптимизатора из чекпоинта в текущую модель.
        Внимание: этот метод НЕ пересоздает модель с новым конфигом. 
        Для создания модели из чекпоинта используйте from_checkpoint.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")
        
        checkpoint = torch.load(path, map_location=self._device)
        
        # Валидация
        if checkpoint.get("model_class") != self.__class__.__name__:
            logger.warning(
                f"Класс модели не совпадает: "
                f"ожидается {self.__class__.__name__}, "
                f"получен {checkpoint.get('model_class')}"
            )
        
        # Загрузка весов
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        # Загрузка оптимизатора
        if load_optimizer and optimizer is not None:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                logger.warning("Состояние оптимизатора не найдено в чекпоинте")
        
        logger.info(f"Веса загружены из {path}")
        
        return {
            "epoch": checkpoint.get("epoch"),
            "timestamp": checkpoint.get("timestamp"),
            "metadata": checkpoint.get("metadata")
        }

    @classmethod
    def from_checkpoint(
        cls, 
        path: Union[str, Path],
        device: str = "cpu"
    ) -> "BaseModelClass":
        """
        Фабричный метод: создает полную модель из файла чекпоинта.
        Восстанавливает конфиг, артефакты и веса.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")
            
        checkpoint = torch.load(path, map_location=device)
        
        # 1. Восстанавливаем конфиг
        config_data = checkpoint.get("config")
        if not config_data:
            raise ValueError("В чекпоинте отсутствует конфигурация")
        # Используем from_dict базового класса конфига
        config = cls.ConfigClass.from_dict(config_data)
        
        # 2. Восстанавливаем артефакты (если они были сохранены)
        artifacts = checkpoint.get("artifacts")
        
        # Если артефактов нет в чекпоинте, пробуем создать модель без них
        model = cls(config=config, artifacts=artifacts)
        
        # 3. Грузим веса
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model._device = torch.device(device)
        
        return model

    # ==================== Методы сохранения отдельных частей ====================

    def save_weights(self, path: Union[str, Path]) -> None:
        """Сохранение только весов модели (state_dict)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Веса модели сохранены в {path}")
    
    def load_weights(self, path: Union[str, Path], strict: bool = True) -> None:
        """Загрузка только весов модели."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл с весами не найден: {path}")
        self.load_state_dict(torch.load(path, map_location=self._device), strict=strict)
        logger.info(f"Веса модели загружены из {path}")

    def save_config(self, path: Union[str, Path]) -> None:
        """Сохранение конфига модели.
        Используются методы базового класса конфигов
        """
        self.config.save(path)

    def save_artifacts(self, path: Union[str, Path]) -> None:
        """Сохранение артефактов модели.
        Используются методы базового класса артефактов"""
        if self.artifacts:
            self.artifacts.save(path)
        else:
            logger.warning("Попытка сохранить артефакты, но self.artifacts is None")

    # ==================== Инфо и Утилиты ====================

    def get_version(self) -> str:
        return self._version
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "model_class": self.__class__.__name__,
            "model_version": self._version,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "device": str(self._device),
            "config": self.config.to_dict(), 
        }
        return info

    def summary(self, input_size: Optional[Tuple[int, ...]] = None) -> str:
        """Человекочитаемое представление архитектуры."""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append(f"Model: {self.__class__.__name__}")
        summary_lines.append(f"Version: {self._version}")
        summary_lines.append("=" * 80)
        
        info = self.get_model_info()
        summary_lines.append(f"Total parameters: {info['total_parameters']:,}")
        summary_lines.append(f"Trainable parameters: {info['trainable_parameters']:,}")
        summary_lines.append(f"Non-trainable parameters: {info['non_trainable_parameters']:,}")
        summary_lines.append(f"Device: {info['device']}")
        summary_lines.append("=" * 80)
        
        summary_lines.append("\nModel Architecture:")
        summary_lines.append(str(self))
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)

    # Методы freeze/unfreeze без изменений
    def freeze_layers(self, layer_names: Optional[List[str]] = None) -> None:
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = False
            logger.info("Все слои модели заморожены")
        else:
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = False
            logger.info(f"Заморожены слои: {layer_names}")
    
    def unfreeze_layers(self, layer_names: Optional[List[str]] = None) -> None:
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = True
            logger.info("Все слои модели разморожены")
        else:
            for name, module in self.named_modules():
                if name in layer_names:
                    for param in module.parameters():
                        param.requires_grad = True
            logger.info(f"Разморожены слои: {layer_names}")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]