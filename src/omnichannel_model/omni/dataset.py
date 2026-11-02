"""OmniDataset — единый класс, работающий и в local, и в streaming режиме.

Изменения относительно оригинала МИНИМАЛЬНЫ:
- Импорт BaseIterableDataset → из base_dataset_v2 (тот же класс, расширенный)
- _estimate_size() — добавлена ветка для streaming (чтение метаданных через HDFS)

Всё остальное — без единого изменения.

Использование (LOCAL — как было):
    dataset = OmniDataset(
        data_path="/home/datalab/nfs/.../INPUT_DATA",
        num_features=num_features,
        ...
    )

Использование (STREAMING — добавляем 2 параметра):
    from base_dataset_v2 import HDFSConfig

    dataset = OmniDataset(
        data_path="/home/datalab/nfs/.../INPUT_DATA",   # используется как cache_dir
        num_features=num_features,
        ...
        hdfs_config=HDFSConfig(kerb_ticket=kerb_ticket),
        hdfs_data_path="/user/team/.../omnichannel_evk_scoring_data",
        prefetch_ahead=30,
    )
"""

from __future__ import annotations

import pickle
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Union

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from sklearn.exceptions import InconsistentVersionWarning

from omnichannel_model.base_classes.base_dataset import BaseIterableDataset
from omnichannel_model.omni.configs import SequenceConfig

from typing import Literal
DataFormat = Literal["raw", "precomputed"]

warnings.filterwarnings("ignore", category=pl.exceptions.PolarsWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=pl.exceptions.PerformanceWarning)


# ---------------------------------------------------------------------------
# Вспомогательная функция UNK-dropout
# ---------------------------------------------------------------------------

def _build_unk_dropout_exprs(
    cat_features: List[str],
    cat_encoders: dict,
    unk_emb_prob: float,
) -> List[pl.Expr]:
    """
    Детерминированный UNK-dropout через хэш от (epk_id, report_dt, col, value).
    Воспроизводим между перезапусками.
    """
    exprs = []
    for col in cat_features:
        if col not in cat_encoders:
            continue
        unk_token = cat_encoders[col].get("UNK", 0)
        key = pl.concat_str(
            [
                pl.col("epk_id").cast(pl.Utf8),
                pl.col("report_dt").cast(pl.Utf8),
                pl.lit(col).cast(pl.Utf8),
                pl.col(col).cast(pl.Utf8),
            ],
            separator="|",
        )
        uniform = (
            key.hash().cast(pl.UInt64).mod(1 << 53).cast(pl.Float64) / float(1 << 53)
        )
        exprs.append(
            pl.when(uniform < unk_emb_prob)
            .then(pl.lit(unk_token))
            .otherwise(pl.col(col))
            .alias(col)
        )
    return exprs


# ---------------------------------------------------------------------------
# Реестр scoring-эмбеддингов для OmniScoringDataset
#
# Каждая запись описывает один user-side scoring-эмбеддинг:
#   flag_key  — ключ в use_emb_flags
#   col_attr  — атрибут датасета с именем колонки в parquet (уже None если флаг False)
#   emb_key   — ключ размерности в SequenceConfig.get_scoring_emb_len()
#   batch_key — ключ в батче ("romashka", "evk", ...)
#
# Чтобы добавить новый эмбеддинг:
#   1. Добавить строку в _SCORING_EMB_REGISTRY
#   2. Добавить col_attr в OmniDataset.__init__ (аргумент + self.xxx_col)
#   3. Добавить emb_key в SequenceConfig.get_scoring_emb_len() при нестандартной размерности
#
# ---------------------------------------------------------------------------
_SCORING_EMB_REGISTRY: List[tuple] = [
    # (flag_key,       col_attr,       emb_key,      batch_key)
    ("romashka_emb",  "romashka_col",  "romashka",   "romashka"),
    ("evk_emb",       "evk_col",       "evk",        "evk"),
]


# Технические константы
_DEFAULT_CHUNK_SIZE = 65_536
_SECONDS_PER_DAY = 60 * 60 * 24


# ---------------------------------------------------------------------------
# OmniDataset — общий базовый класс
# ---------------------------------------------------------------------------

class OmniDataset(BaseIterableDataset):
    """
    Базовый класс для всех omni-датасетов.

    Параметры
    ---------
    data_path         : папка с .parquet-файлами
    num_features      : числовые признаки
    cat_features      : категориальные признаки
    seq_config        : размерности EVK/ромашки (совпадают с весами модели)
    weight_col        : опциональная колонка весов
    cat_encoder_name  : путь к pkl с {col: {value: int}}
    romashka_col      : имя колонки ромашки в parquet
    evk_col           : имя колонки EVK в parquet
    prod_desc_col     : имя колонки индекса product description
    prod_desc_name    : путь к .pt с embedding-матрицей prod_desc
    prod_hier_col     : имя колонки индекса product hierarchy
    prod_hier_name    : путь к .pt с embedding-матрицей prod_hier
    imputer_name      : путь к pkl с fitted SimpleImputer
    preprocessor_name : путь к pkl с fitted QuantileTransformer
    unk_emb_prob      : вероятность замены категории на UNK
    chunk_size        : строк за один iter_slices (performance tuning)
    **kwargs          : пробрасываются в BaseIterableDataset
                        (batch_size, shuffle, hdfs_config, hdfs_data_path, ...)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        num_features: List[str],
        cat_features: List[str],
        seq_config: Optional[SequenceConfig] = None,
        data_format: DataFormat = "raw",  
        weight_col: Optional[str] = None,
        cat_encoder_name: Optional[str] = None,
        romashka_col: Optional[str] = "romashka",
        evk_col: Optional[str] = "evk_emb",
        prod_desc_col: Optional[str] = "prod_desc",
        prod_desc_name: Optional[str] = "prod_desc_emb.pt",
        prod_hier_col: Optional[str] = "prod_hier",
        prod_hier_name: Optional[str] = "prod_hier_emb.pt",
        imputer_name: Optional[str] = None,
        preprocessor_name: Optional[str] = None,
        unk_emb_prob: float = 0.1,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        **kwargs,
    ):
        super().__init__(data_path=data_path, file_pattern="*.parquet", **kwargs)

        self.seq_config = seq_config or SequenceConfig()
        self.data_format: DataFormat = data_format

        self.num_features = num_features
        self.cat_features = cat_features
        self.weight_col = weight_col
        self.cat_encoder_name = cat_encoder_name
        self.romashka_col = romashka_col
        self.evk_col = evk_col
        self.prod_desc_col = prod_desc_col
        self.prod_desc_name = prod_desc_name
        self.prod_hier_col = prod_hier_col
        self.prod_hier_name = prod_hier_name
        self.imputer_name = imputer_name
        self.preprocessor_name = preprocessor_name
        self.unk_emb_prob = unk_emb_prob
        self.chunk_size = chunk_size

        # Ленивые ресурсы
        self.imputer = None
        self.preprocessor = None
        self.cat_encoders = None
        self.prod_desc_emb = None
        self.prod_hier_emb = None
        self._expressions_initialized = False

        self.all_columns = self._build_all_columns()
        self._estimate_size()

    def _extra_columns(self) -> List[str]:
        return []

    def _build_all_columns(self) -> List[str]:
        # None-атрибуты (отключённые флагом) автоматически отсеиваются filter(None)
        optional = [
            self.weight_col, self.evk_col,
            self.prod_desc_col, self.prod_hier_col, self.romashka_col,
        ]
        cols = (
            self.num_features
            + self.cat_features
            + ["epk_id", "report_dt"]
            + [c for c in optional if c]
            + self._extra_columns()
        )
        return list(set(filter(None, cols)))

    # ------------------------------------------------------------------
    # Размер датасета
    # ------------------------------------------------------------------

    def _estimate_size(self) -> None:
        self.total_rows = 0
        if not self.files:
            return
        try:
            if self._streaming:
                fs = self._hdfs_config.create_filesystem()
                with fs.open_input_file(self._hdfs_files[0]) as f:
                    meta = pq.read_metadata(f)
            else:
                meta = pq.read_metadata(self.files[0])
            self.total_rows = int(len(self.files) * meta.num_rows * self.file_fraction)
        except Exception:
            pass

    def __len__(self) -> int:
        return max(1, self.total_rows // self.batch_size)

    def _build_seq_exprs(self) -> None:
        """Polars-выражения для romashka/evk — конкретная формула зависит от self.data_format."""
        self._seq_exprs: Dict[str, pl.Expr] = {}
        cfg = self.seq_config

        for col, key in ((self.romashka_col, "romashka"), (self.evk_col, "evk")):
            if col is None:
                continue

            if self.data_format == "raw":
                if key != "romashka":
                    continue  
                expr = (
                    pl.concat_list([
                        pl.lit([cfg.rom_pad_value] * cfg.romashka_len, dtype=pl.List(pl.Int16)),
                        pl.col(col).fill_null([]),
                    ])
                    .list.tail(cfg.romashka_len)
                    .alias(col)
                )
            else:  # precomputed
                emb_len = cfg.get_scoring_emb_len(key)
                fill = pl.lit([float("nan")] * emb_len, dtype=pl.Array(pl.Float32, emb_len))
                expr = pl.col(col).fill_null(fill).cast(pl.Array(pl.Float32, emb_len)).alias(col)

            self._seq_exprs[col] = expr

    def _apply_seq_exprs(self, df: pl.DataFrame) -> pl.DataFrame:
        applicable = [expr for col, expr in self._seq_exprs.items() if col in df.columns]
        return df.with_columns(applicable) if applicable else df

    def _convert_seq_columns(self, batch_df: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """Тензоры romashka/evk — конкретная логика зависит от self.data_format."""
        out: Dict[str, torch.Tensor] = {}
        cfg = self.seq_config
        B = batch_df.height

        if self.romashka_col is not None:
            if self.data_format == "raw":
                rom = batch_df[self.romashka_col].explode().to_numpy().astype(np.int16).reshape(B, cfg.romashka_len)
                mask = torch.from_numpy(rom.sum(axis=1) == 0)
                tensor = torch.from_numpy(rom).reshape(B, cfg.romashka_seq_len, cfg.romashka_n_mods)
                out["romashka_ts"] = tensor[:, :, 0].to(torch.int64) * _SECONDS_PER_DAY
                out["romashka"] = tensor[:, :, 1:].to(torch.int32)
                out["romashka_mask"] = mask
            else:
                tensor = batch_df.select(self.romashka_col).to_torch()
                out["romashka"] = tensor
                out["romashka_mask"] = torch.isnan(tensor).any(dim=1)

        if self.evk_col is not None:
            if self.data_format == "raw":
                evk = batch_df[self.evk_col].explode().to_numpy().astype(np.int16).reshape(B, cfg.evk_len)
                mask = torch.from_numpy(evk.sum(axis=1) == 0)
                tensor = torch.from_numpy(evk).reshape(B, cfg.evk_seq_len, cfg.evk_n_mods)
                out["evk"] = tensor.to(torch.int32)
                out["evk_positions"] = torch.arange(cfg.evk_seq_len).repeat(B, 1)
                out["evk_mask"] = mask
            else:
                tensor = batch_df.select(self.evk_col).to_torch()
                out["evk"] = tensor
                out["evk_mask"] = torch.isnan(tensor).any(dim=1)

        return out
    # ------------------------------------------------------------------
    # Ленивая инициализация общих артефактов
    # ------------------------------------------------------------------

    def _lazy_init_processors(self) -> None:
        if self._expressions_initialized:
            return

        if self.imputer_name and not self.imputer:
            with open(self.imputer_name, "rb") as f:
                self.imputer = pickle.load(f)

        if self.preprocessor_name and not self.preprocessor:
            with open(self.preprocessor_name, "rb") as f:
                self.preprocessor = pickle.load(f)

        if self.cat_encoder_name and not self.cat_encoders:
            with open(self.cat_encoder_name, "rb") as f:
                self.cat_encoders = pickle.load(f)

        # prod_desc_name / prod_hier_name == None если флаг False → torch.load не вызывается
        if self.prod_desc_name and not self.prod_desc_emb:
            self.prod_desc_emb = torch.load(self.prod_desc_name)

        if self.prod_hier_name and not self.prod_hier_emb:
            self.prod_hier_emb = torch.load(self.prod_hier_name)

        self.unk_exprs = (
            _build_unk_dropout_exprs(
                self.cat_features,
                self.cat_encoders,
                self.unk_emb_prob,
            )
            if self.unk_emb_prob > 0.0 and self.cat_encoders
            else []
        )

        self._init_sequence_exprs()
        self._expressions_initialized = True

    def _init_sequence_exprs(self):
        self._build_seq_exprs()

    # ------------------------------------------------------------------
    # Обработка чанка
    # ------------------------------------------------------------------

    def _process_chunk_expressions(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.unk_exprs:
            df = df.with_columns(self.unk_exprs)
        return df

    def _convert_to_tensor_dict(
        self, df: pl.DataFrame, indices: Optional[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Общий _process_file
    # ------------------------------------------------------------------

    def _process_file(
        self,
        file_path: Path,
        remainder_df: Optional[pl.DataFrame] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        self._lazy_init_processors()
        logger = self._get_logger()

        try:
            ds_file = pl.read_parquet(
                file_path,
                columns=self.all_columns,
                rechunk=False,
                low_memory=False,
                hive_partitioning=True,
            )
            for chunk in ds_file.iter_slices(n_rows=self.chunk_size):
                try:
                    df = self._process_chunk_expressions(chunk)
                except Exception as e:
                    logger.error(f"Ошибка chunk {file_path}: {e}")
                    traceback.print_exc()
                    continue

                if remainder_df is not None:
                    df = pl.concat([remainder_df, df])
                    del remainder_df
                    remainder_df = None

                if df.height == 0:
                    continue

                n_rows = df.height
                indices = (
                    self._np_gen.permutation(n_rows)
                    if self.shuffle and self._is_train
                    else np.arange(n_rows)
                )

                for i in range(0, n_rows, self.batch_size):
                    batch_indices = indices[i: i + self.batch_size]
                    if len(batch_indices) < self.batch_size:
                        remainder_df = df[batch_indices]
                        self._remainder_df = remainder_df
                        break
                    yield self._convert_to_tensor_dict(df, batch_indices)
                del df

            self._remainder_df = remainder_df

        except Exception as e:
            self._remainder_df = remainder_df
            logger.error(f"Error processing file {file_path}: {e}")
            traceback.print_exc()
            return

    # ------------------------------------------------------------------
    # Хелпер: числовые признаки
    # ------------------------------------------------------------------

    def _transform_num_feats(self, batch_df: pl.DataFrame) -> np.ndarray:
        num_feats = batch_df.select(self.num_features).to_numpy()
        if self.imputer:
            num_feats = self.imputer.transform(num_feats)
        if self.preprocessor:
            num_feats = self.preprocessor.transform(num_feats)
        return num_feats

    def _prod_emb(self, batch_df: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """
        prod_desc / prod_hier добавляются в батч только если флаг включён.
        prod_desc_emb / prod_hier_emb == None когда флаг False → блок пропускается.
        """
        out = {}
        if self.prod_desc_emb is not None:
            idx = (
                batch_df.select(pl.col(self.prod_desc_col).fill_null(0))
                .to_torch().squeeze(1).to(torch.int32)
            )
            out["prod_desc"] = self.prod_desc_emb[idx].detach().float()
            out["prod_desc_mask"] = (
                batch_df.select(pl.col(self.prod_desc_col).is_null())
                .to_torch().squeeze(1)
            )

        if self.prod_hier_emb is not None:
            idx = (
                batch_df.select(pl.col(self.prod_hier_col).fill_null(0))
                .to_torch().squeeze(1).to(torch.int32)
            )
            out["prod_hier"] = self.prod_hier_emb[idx].detach().float()

        return out


# ---------------------------------------------------------------------------
# OmniTrainDataset — train/eval режим (ромашка = List[Int16] токены)
# ---------------------------------------------------------------------------

class OmniTrainDataset(OmniDataset):
    """
    Датасет для обучения и оффлайн-валидации.
    Ромашка и EVK хранятся как List[Int16]-токены.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        num_features: List[str],
        cat_features: List[str],
        target_columns: List[str],
        evaluate_cols: List[str],
        **kwargs,
    ):
        self._target_columns = target_columns
        self._evaluate_cols = evaluate_cols
        super().__init__(
            data_path=data_path,
            num_features=num_features,
            cat_features=cat_features,
            **kwargs,
        )

    def _extra_columns(self) -> List[str]:
        return self._target_columns + self._evaluate_cols


    def _process_chunk_expressions(self, df):
        df = super()._process_chunk_expressions(df)   # UNK-dropout
        return self._apply_seq_exprs(df)

    def _convert_to_tensor_dict(
        self, df: pl.DataFrame, indices: Optional[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        batch_df = df[indices] if indices is not None else df
        cfg = self.seq_config

        num_feats = self._transform_num_feats(batch_df)

        batch: Dict[str, torch.Tensor] = {
            "num_feats": torch.from_numpy(num_feats).float(),
            "cat_feats": batch_df.select(self.cat_features).to_torch(dtype=pl.Int32),
            "target":    batch_df.select(self._target_columns).to_torch(dtype=pl.Int32),
        }

        batch.update(self._convert_seq_columns(batch_df))
        batch.update(self._prod_emb(batch_df))
        

        if self._evaluate_cols:
            batch["groups"] = batch_df.select(self._evaluate_cols).to_torch(dtype=pl.Int32)

        if self.weight_col:
            batch["sample_weight"] = batch_df.select(self.weight_col).to_torch(dtype=pl.Float32)
        
        batch["data_format"] = self.data_format
        return batch


# ---------------------------------------------------------------------------
# OmniScoringDataset — scoring режим (ромашка/EVK = Float32 эмбеддинги)
# ---------------------------------------------------------------------------

class OmniScoringDataset(OmniDataset):
    """
    Датасет для скоринга.
    Ромашка и EVK хранятся в parquet как готовые float32-эмбеддинги.

    Scoring-эмбеддинги описаны в _SCORING_EMB_REGISTRY.
    Чтобы добавить новый — достаточно добавить строку в реестр (см. выше).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        num_features: List[str],
        cat_features: List[str],
        evaluate_cols: Optional[List[str]] = None,
        data_format: DataFormat = "precomputed",
        **kwargs,
    ):
        self._evaluate_cols = evaluate_cols or []
        super().__init__(
            data_path=data_path,
            num_features=num_features,
            cat_features=cat_features,
            data_format=data_format,
            **kwargs,
        )

    def _extra_columns(self) -> List[str]:
        return list(self._evaluate_cols)

    def _init_sequence_exprs(self) -> None:

        super()._init_sequence_exprs() # romashka/evk -> self._seq_exprs, по self
        
        col_attr_map = {"romashka_col": self.romashka_col, "evk_col": self.evk_col}
        self._scoring_exprs: Dict[str, tuple] = {}
        for flag_key, col_attr, emb_key, batch_key in _SCORING_EMB_REGISTRY:
            if batch_key in ("romashka", "evk"):
                continue # теперь через self._seq_exprs
            col = col_attr_map.get(col_attr)
            if col is None:
                continue
            emb_len = self.seq_config.get_scoring_emb_len(emb_key)
            fill = pl.lit([float("nan")] * emb_len, dtype=pl.Array(pl.Float32, emb_len))
            expr = pl.col(col).fill_null(fill).cast(pl.Array(pl.Float32, emb_len)).alias(col)
            self._scoring_exprs[batch_key] = (expr, emb_len)        

    def _process_chunk_expressions(self, df: pl.DataFrame) -> pl.DataFrame:
        df = super()._process_chunk_expressions(df)
        if self.prod_hier_col:
            df = df.filter(pl.col(self.prod_hier_col).is_not_null())
        df = self._apply_seq_exprs(df)
        if self._scoring_exprs:
            df = df.with_columns([expr for expr, _ in self._scoring_exprs.values()])
        return df

    def _convert_to_tensor_dict(
        self, df: pl.DataFrame, indices: Optional[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        batch_df = df[indices] if indices is not None else df

        num_feats = self._transform_num_feats(batch_df)

        batch: Dict[str, torch.Tensor] = {
            "num_feats": torch.from_numpy(num_feats).float(),
            "cat_feats": batch_df.select(self.cat_features).to_torch(dtype=pl.Int32),
            "epk_id":    batch_df.select("epk_id")
                         .to_numpy().astype(str).flatten().tolist(),
            "report_dt": batch_df.select("report_dt")
                         .to_numpy().astype(str).flatten().tolist(),
        }
        batch.update(self._convert_seq_columns(batch_df)) 
        # Итерируемся по тому же реестру что и _init_sequence_exprs —
        # ровно те ключи, для которых был построен expr, попадают в батч.
        # Нет рассинхрона между "что обработали" и "что положили в батч".
        col_attr_map = {
            "romashka_col": self.romashka_col,
            "evk_col":      self.evk_col,
        }
        for flag_key, col_attr, emb_key, batch_key in _SCORING_EMB_REGISTRY:
            if batch_key not in self._scoring_exprs:
                continue
            col = col_attr_map[col_attr]
            tensor = batch_df.select(col).to_torch()
            batch[batch_key]              = tensor
            batch[f"{batch_key}_mask"]    = torch.isnan(tensor).any(dim=1)

        batch.update(self._prod_emb(batch_df))

        if self._evaluate_cols:
            batch["groups"] = batch_df.select(self._evaluate_cols).to_torch(dtype=pl.Int32)

        if self.weight_col:
            batch["sample_weight"] = batch_df.select(self.weight_col).to_torch(dtype=pl.Float32)

        batch["data_format"] = self.data_format
        return batch


# ---------------------------------------------------------------------------
# OmniInferenceDataset — inference (scoring + join с prod_data по report_dt)
# ---------------------------------------------------------------------------

class OmniInferenceDataset(OmniScoringDataset):
    """
    Датасет для инференса: scoring + join с prod_data по report_dt.
    """

    def __init__(
        self,
        prod_data_path: Union[str, Path],
        expansion_factor: float = 100.0,
        process_idx: int = 0,
        num_processes: int = 1,
        *args,
        **kwargs,
    ):
        self.prod_data_path = Path(prod_data_path)
        self.prod_data_df = None
        self.expansion_factor = expansion_factor
        self._log_rank = process_idx
        self._log_total = num_processes

        kwargs["shuffle"] = False
        kwargs["evaluate_cols"] = []
        super().__init__(*args, **kwargs)

        prod_schema = pl.read_parquet_schema(self.prod_data_path)
        prod_cols = set(prod_schema.keys()) - {"report_dt"}
        self.all_columns = [c for c in self.all_columns if c not in prod_cols]

        self._diag_batch_count = 0
        self._diag_file_count = 0

    def _lazy_init_processors(self) -> None:
        super()._lazy_init_processors()
        if self.prod_data_df is None:
            self.prod_data_df = pl.read_parquet(self.prod_data_path)

    def _process_chunk_expressions(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.join(self.prod_data_df, on="report_dt", how="inner")
        return super()._process_chunk_expressions(df)

    def _process_file(
        self,
        file_path: Path,
        remainder_df: Optional[pl.DataFrame] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        self._lazy_init_processors()
        logger = self._get_logger()
        self._remainder_df = None
        self._diag_file_count += 1

        try:
            ds_file = pl.read_parquet(
                file_path,
                columns=self.all_columns,
                rechunk=False,
                low_memory=False,
                hive_partitioning=True,
            )
            for chunk in ds_file.iter_slices(n_rows=self.chunk_size):
                try:
                    df = self._process_chunk_expressions(chunk)
                except Exception as e:
                    logger.error(
                        f"[Rank {self._log_rank}] Ошибка chunk, файл={file_path.name}: {e}"
                    )
                    traceback.print_exc()
                    continue

                if remainder_df is not None:
                    df = pl.concat([remainder_df, df])
                    del remainder_df
                    remainder_df = None

                if df.height == 0:
                    continue

                n_rows = df.height
                indices = np.arange(n_rows)

                for i in range(0, n_rows, self.batch_size):
                    batch_indices = indices[i: i + self.batch_size]
                    if len(batch_indices) < self.batch_size:
                        remainder_df = df[batch_indices]
                        self._remainder_df = remainder_df
                        break
                    try:
                        self._diag_batch_count += 1
                        yield self._convert_to_tensor_dict(df, batch_indices)
                    except Exception as e:
                        logger.error(
                            f"[Rank {self._log_rank}] Ошибка convert, файл={file_path.name}: {e}"
                        )
                        traceback.print_exc()
                        continue
                del df

            self._remainder_df = remainder_df

            if self._diag_file_count % 10 == 0:
                logger.info(
                    f"[Rank {self._log_rank}] "
                    f"файлов={self._diag_file_count}, батчей={self._diag_batch_count}"
                )

        except Exception as e:
            self._remainder_df = remainder_df
            logger.error(f"[Rank {self._log_rank}] Ошибка файла {file_path.name}: {e}")
            traceback.print_exc()
            return

    def __len__(self) -> int:
        if not getattr(self, "total_rows", 0):
            return 1
        return max(1, int(self.total_rows * self.expansion_factor) // self.batch_size)