import os 
from collections import OrderedDict
from typing import List, Optional
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from omnichannel_model.omni.dataset import OmniTrainDataset
from omnichannel_model.base_classes.base_dataset import HDFSConfig

from camomile_models.models.dlt.DLTmodel import DLTmodelBase
from omnichannel_model.evk.models import CommunicationModelFull, CommunicationModelConfig


# ---------------------------------------------------------------------------
# Подготовка названий полей с эмбеддингами для передачи в OmniTrainDataset
# ---------------------------------------------------------------------------
def resolve_emb_dataset_kwargs(
    use_emb_flags: dict,
    artifacts_path: str,
    prod_desc_name: str = "prod_desc_emb.pt",
    prod_hier_name: str = "prod_hier_emb.pt",
) -> dict:
    """Транслирует use_emb_flags модели в аргументы конструктора OmniTrainDataset.
    Начиная с новой версии dataset.py флаг romashka_emb/evk_emb/... не поддерживается
    внутри класса, поэтому колонки нужно заранее обнулять снаружи."""
    return {
        "romashka_col": "romashka" if use_emb_flags.get("romashka_emb", True) else None,
        "evk_col": "evk_emb" if use_emb_flags.get("evk_emb", True) else None,
        
        "prod_desc_col": "prod_desc" if use_emb_flags.get("prod_desc_emb", True) else None,
        "prod_desc_name": (
            os.path.join(artifacts_path, prod_desc_name)
            if use_emb_flags.get("prod_desc_emb", True) else None
        ),
        
        "prod_hier_col": "prod_hier" if use_emb_flags.get("prod_hier_emb", True) else None,
        "prod_hier_name": (
            os.path.join(artifacts_path, prod_hier_name)
            if use_emb_flags.get("prod_hier_emb", True) else None
        ),
    }

# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataset(
    config_omni: dict,
    data_path: str,
    num_feats: list,
    cat_feats: list,
    target_columns: list,
    evaluate_cols: list,
    artifacts_path: str,
    cat_encoder_name: str,
    quantile_transformer_name: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool,
    unk_emb_prob: float,
    file_fraction: float,
    process_idx: int,
    num_processes: int,
    hdfs_config: HDFSConfig,
    hdfs_data_path: str,
    data_format: str,
) -> OmniTrainDataset:
    """
    Instantiate OmniTrainDataset.
    romashka/evk/prod_desc/prodhier columns are passed only when the
    corresponding emb_flag is True — matching the model config.
    """

    # Получаем конфиг с включенными эмбеддингами модели 
    emb_kwargs = resolve_emb_dataset_kwargs(
        use_emb_flags=config_omni.use_emb_flags,
        artifacts_path=artifacts_path,
    )
    return OmniTrainDataset(
        data_path=data_path,
        num_features=num_feats,
        cat_features=cat_feats,
        target_columns=target_columns,
        weight_col=None,
        cat_encoder_name=artifacts_path + cat_encoder_name,
        imputer_name=None,
        preprocessor_name=artifacts_path + quantile_transformer_name,
        evaluate_cols=evaluate_cols,
        batch_size=batch_size,
        shuffle=shuffle,
        random_seed=seed,
        drop_last=drop_last,
        unk_emb_prob=unk_emb_prob,
        file_fraction=file_fraction,
        process_idx=process_idx,
        num_processes=num_processes,
        hdfs_config=hdfs_config,
        hdfs_data_path=hdfs_data_path,
        data_format=data_format,
        prefetch_ahead=100,
        max_cache_bytes=200 * 1024 ** 3,
        delete_after_processing=True,
        **emb_kwargs,
    )


def create_dataloader(
    ds: OmniTrainDataset,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=None,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

# ---------------------------------------------------------------------------
# Submodels preparation
# ---------------------------------------------------------------------------
def prepare_submodel(
    model: nn.Module,
    weights_path: Optional[str] = None,
    trainable_patterns: Optional[List[str]] = None,
    strip_prefix: Optional[str] = None,
) -> nn.Module:
    """
    Единая подготовка саб-модуля (EVK-энкодер, DLT-модель) под три режима обучения.

    weights_path=None        -> обучение с нуля: веса не грузятся, все параметры
                                 обучаемы (дефолт PyTorch после инициализации).
    weights_path задан,
        trainable_patterns=None  -> предобученные веса грузятся, ВСЕ параметры замораживаются.
    weights_path задан,
        trainable_patterns=[...] -> предобученные веса грузятся, обучаемы только параметры,
                                     чьё имя содержит одну из подстрок (fine-tuning).

    strip_prefix — опционально обрезает префикс в ключах чекпоинта (например "base_model.").
    """
    if weights_path is None:
        print(f"[{model.__class__.__name__}] weights_path не задан — обучение с нуля")
        return model

    state_dict = torch.load(weights_path, map_location="cpu")
    if strip_prefix:
        state_dict = OrderedDict(
            (k[len(strip_prefix):] if k.startswith(strip_prefix) else k, v)
            for k, v in state_dict.items()
        )

    miss, unexp = model.load_state_dict(state_dict, strict=False)
    print(f"[{model.__class__.__name__}] Загружено из {weights_path}: "
          f"missing={len(miss)}, unexpected={len(unexp)}")

    if not trainable_patterns:
        for p in model.parameters():
            p.requires_grad = False
    else:
        for name, p in model.named_parameters():
            p.requires_grad = any(pattern in name for pattern in trainable_patterns)

    return model


def load_evk_model(
    config_path: str,
    weights_path: Optional[str] = None,
    trainable_patterns: Optional[List[str]] = None,
) -> nn.Module:
    evk_cnf = CommunicationModelConfig.from_dict(json.load(open(config_path)))
    evk_model = CommunicationModelFull(evk_cnf).encoder
    return prepare_submodel(evk_model, weights_path, trainable_patterns)


def load_dlt_model(
    config_path: str,
    weights_path: Optional[str] = None,
    trainable_patterns: Optional[List[str]] = None,
) -> nn.Module:
    
    rom_cnf = json.load(open(config_path))
    dlt_model = DLTmodelBase(**rom_cnf)
    return prepare_submodel(dlt_model, weights_path, trainable_patterns, strip_prefix="base_model.")