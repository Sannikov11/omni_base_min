#TO DO: переписать все под pydantic.dataclasses
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union, Type, TypeVar

from pathlib import Path
import json
import yaml
from datetime import datetime
import warnings
from typing import Dict, Any

try:
    # Пытаемся заимпортить pydantic, так как он автоматизирует проверку типов в датаклассе
    from pydantic.dataclasses import dataclass
    from pydantic import ConfigDict
    config = ConfigDict(arbitrary_types_allowed=True, strict=True)
    final_dataclass = dataclass(config=config)
except:
    # Fallback - классический датакласс
    from dataclasses import dataclass
    final_dataclass = dataclass
    
from dataclasses import asdict, field
import logging

T = TypeVar("T", bound="BaseModelArtifactsClass")

@final_dataclass
class BaseModelConfigClass:
    """Базовый класс для конфигурации модели.
    
    Обеспечивает единый интерфейс для cохранения и загрузки конфигов из словаря,yaml и json
    """

    model_name: Optional[str] = 'Omni'
    model_version: Optional[str] = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация конфига в словарь."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseModelConfigClass":
        """Создание конфига из словаря."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Сохраняет конфигурацию в файл. Формат определяется по расширению.
        Поддерживает: .json, .yaml, .yml
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()

        with open(path, 'w', encoding='utf-8') as f:
            if ext == '.json':
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            elif ext in ['.yaml', '.yml']:
                yaml.dump(
                    self.to_dict(), 
                    f, 
                    default_flow_style=False, 
                    sort_keys=False, 
                    allow_unicode=True
                )
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}. Используйте .json или .yaml")
        
        print(f"Конфигурация сохранена в {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModelConfigClass":
        """
        Загружает конфигурацию из файла. Формат определяется по расширению.
        Поддерживает: .json, .yaml, .yml
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
            
        ext = path.suffix.lower()
        
        with open(path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                data = json.load(f)
            elif ext in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}. Используйте .json или .yaml")
        
        return cls.from_dict(data)
    
@final_dataclass
class BaseModelArtifactsClass:
    """Базовый класс для артефактов модели (списки, массивы, другие объекты)
    
    Обеспечивает единый интерфейс сохранения и загрузки артефактов через torch.save

    Любой наследник должен быть декорарирован @dataclass.
    """

    def save(self, path: Union[str, Path]) -> None:
        """
        Сохранение всех полей класса в бинарный файл.
        Использует torch.save, так как он лучше оптимизирован.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # asdict рекурсивно превращает dataclass в словарь
        data = asdict(self)
        
        # Сохраняем словарь через torch (под капотом pickle, но оптимизированный для тензоров)
        torch.save(data, path)
        print(f"Артефакты сохранены в {path}")

    @classmethod
    def load(cls: Type[T], path: Union[str, Path], device: str = 'cpu') -> T:
        """
        Загрузка полей из бинарного файла. Создает экземпляр конкретного класса-наследника.
        
        Args:
            path: Путь к файлу
            device: Куда перенести тензоры при загрузке (cpu/cuda)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл артефактов не найден: {path}")
            
        # Загружаем словарь данных
        data = torch.load(path, map_location=device)
        
        # Рекурсивный перенос всех тензоров в словаре на нужное устройство
        # (на случай если внутри списки тензоров или вложенные словари)
        data = cls._move_to_device(data, device)
        
        # Создаем экземпляр класса, распаковывая словарь аргументов
        return cls(**data)

    @staticmethod
    def _move_to_device(data: Any, device: str) -> Any:
        """Вспомогательный метод для рекурсивного переноса данных на устройство."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: BaseModelArtifactsClass._move_to_device(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            return [BaseModelArtifactsClass._move_to_device(v, device) for v in data]
        elif isinstance(data, tuple):
            return tuple(BaseModelArtifactsClass._move_to_device(v, device) for v in data)
        return data