import os
import time
import logging
import subprocess
import threading
import numpy as np
import pyarrow.fs
from pathlib import Path
from queue import Queue, Empty, Full
from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional, Union, Iterator, Any, Dict, Tuple
from dataclasses import dataclass, field
import fcntl

# Добавить после импортов, перед классами:
_KERB_MARKERS = ('kerberos', 'tgt', 'gssexception', 'errno 255', 'no valid credentials')
_PER_ATTEMPT_DL_TIMEOUT = 150  # секунд на одну попытку скачивания


# Вспомогательная функция для работы с зависаниями 
def _hdfs_with_timeout(fn, timeout: float, logger, label: str):
    result = {'value': None, 'error': None}

    def _run(res=result):
        try:
            res['value'] = fn()
        except Exception as e:
            res['error'] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        logger.error(f"[HANG] {label} завис после {timeout}s")
        raise TimeoutError(f"{label} timeout after {timeout}s")

    if result['error'] is not None:
        raise result['error']

    return result['value']



# ---------------------------------------------------------------------------
# HDFSConfig — параметры подключения к HDFS (нужен только для streaming)
# ---------------------------------------------------------------------------

@dataclass
class HDFSConfig:
    """
    Параметры подключения к HDFS.

    Используется только при streaming-режиме.
    Если не передан в BaseIterableDataset — dataset работает как раньше.
    """
    hdfs_url: str = "hdfs://arnsdpsbx"
    kerb_ticket: Optional[str] = None
    user: Optional[str] = None
    extra_conf: Dict[str, str] = field(default_factory=lambda: {
        'hadoop.security.authentication': 'kerberos'
    })
    max_download_workers: int = 16
    max_retries: int = 3

    def create_filesystem(self) -> pyarrow.fs.HadoopFileSystem:
        return pyarrow.fs.HadoopFileSystem(
            self.hdfs_url,
            kerb_ticket=self.kerb_ticket,
            user=self.user,
            extra_conf=self.extra_conf,
        )


# ---------------------------------------------------------------------------
# Вспомогательные классы для streaming (приватные, не часть публичного API)
# ---------------------------------------------------------------------------

class _LocalStorageManager:
    """Управление дисковым кэшем: учёт места, удаление обработанных файлов."""

    def __init__(self, cache_dir: Path, max_cache_bytes: int):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_bytes
        self._lock = threading.Lock()
        self._current_bytes = 0
        self._logger = logging.getLogger("StreamingStorage")
        self._scan_existing()

    def _scan_existing(self):
        total = sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())
        self._current_bytes = total
        self._logger.info(
            f"Кэш: {self.cache_dir}, "
            f"занято {self._current_bytes / 1024**3:.1f} GB / "
            f"{self.max_cache_bytes / 1024**3:.1f} GB"
        )

    def has_space(self) -> bool:
        with self._lock:
            return self._current_bytes < self.max_cache_bytes

    def register_downloaded(self, size_bytes: int):
        with self._lock:
            self._current_bytes += size_bytes

    def release_file(self, local_path: Path):
        try:
            size = local_path.stat().st_size if local_path.exists() else 0
            local_path.unlink(missing_ok=True)
            with self._lock:
                self._current_bytes = max(0, self._current_bytes - size)
                
            # parent = local_path.parent
            # while parent != self.cache_dir:
            #     try:
            #         parent.rmdir()
            #         parent = parent.parent
            #     except OSError:
            #         break
        except OSError as e:
            self._logger.warning(f"Не удалось удалить {local_path} или папки: {e}")

    def get_local_path(self, hdfs_path: str) -> Path:
        # """HDFS-путь → плоский локальный путь (partition + filename)."""
        # parts = hdfs_path.rstrip("/").split("/")
        # local_name = parts[-2] + "_" + parts[-1] if len(parts) >= 2 else parts[-1]
        # return self.cache_dir / local_name
        """HDFS-путь -> локальный путь с сохранением иерархии"""
        clean_path = hdfs_path
        if '://' in clean_path:
            clean_path = clean_path.split('://', 1)[1]
            if "/" in clean_path:
                clean_path = clean_path.split("/", 1)[1]
        
        parts = [p for p in clean_path.split('/') if p]
        
        if len(parts) >= 3 and parts[0] == 'user' and parts[1] == 'team':
            start_idx = 3
            if len(parts) > start_idx and parts[start_idx] == 'hive':
                start_idx += 1

            rel_parts = parts[start_idx:]

            if not rel_parts:
                rel_parts = [parts[-1]]
        else:
            rel_parts = parts[-2:] if len(parts) >= 2 else parts[-1:]
        
        local_path = self.cache_dir.joinpath(*rel_parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        return local_path
        

    @property
    def current_usage_gb(self) -> float:
        with self._lock:
            return self._current_bytes / 1024**3


class _HDFSPrefetcher:
    """
    Фоновый скачиватель файлов из HDFS (producer-consumer).

    Скачивает файлы наперёд, пока текущий обрабатывается.
    Семафор ограничивает кол-во скачанных-но-не-обработанных файлов.
    """

    def __init__(
        self,
        hdfs_config: HDFSConfig,
        storage: _LocalStorageManager,
        prefetch_ahead: int,
    ):
        self.hdfs_config = hdfs_config
        self.storage = storage
        self.prefetch_ahead = prefetch_ahead
        self._logger = logging.getLogger("HDFSPrefetcher")

        # Queue maxsize = prefetch_ahead + 1: +1 для sentinel (None),
        # чтобы producer всегда мог положить sentinel даже при полной очереди
        self._ready_queue: Queue[Optional[Tuple[str, Path]]] = Queue(maxsize=prefetch_ahead + 1)
        self._fs: Optional[pyarrow.fs.HadoopFileSystem] = None
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._disk_semaphore = threading.Semaphore(prefetch_ahead)

    # def start(self, hdfs_paths: List[str]):
    #     self._stop_event.clear()
    #     self._fs = self.hdfs_config.create_filesystem()
    #     self._producer_thread = threading.Thread(
    #         target=self._producer_loop, args=(hdfs_paths,), daemon=True,
    #     )
    #     self._producer_thread.start()
    def start(self, hdfs_paths: List[str]):
        self._stop_event.clear()
        self._logger.info("[FS] Создаём HDFS filesystem...")
        MAX_FS_RETRIES = 3
        for fs_attempt in range(MAX_FS_RETRIES):
            try:
                self._fs = _hdfs_with_timeout(
                    self.hdfs_config.create_filesystem,
                    timeout=60,
                    logger=self._logger,
                    label=f"create_filesystem() attempt {fs_attempt + 1}/{MAX_FS_RETRIES}",
                )
                self._logger.info(f"[FS] OK на попытке {fs_attempt + 1}")
                break
            except Exception as e:
                self._logger.error(
                    f"[FS FAIL] attempt {fs_attempt + 1}/{MAX_FS_RETRIES}: {e}"
                )
                if fs_attempt < MAX_FS_RETRIES - 1:
                    self._logger.info("[FS] Ждём 30s перед следующей попыткой...")
                    time.sleep(30)
        else:
            self._logger.error("[FS FAIL] Все попытки исчерпаны → sentinel")
            self._safe_put(None)
            return

        self._producer_thread = threading.Thread(
            target=self._producer_loop, args=(hdfs_paths,), daemon=True,
        )
        self._producer_thread.start()

    
    

    def stop(self):
        self._stop_event.set()
        # Drain queue чтобы разблокировать producer если он завис на put()
        self._drain_queue()
        if self._producer_thread:
            self._producer_thread.join(timeout=30)
            if self._producer_thread.is_alive():
                self._logger.warning(
                    "Producer thread не завершился за 30с "
                    "(возможно, заблокирован на HDFS I/O)"
                )

    def _drain_queue(self):
        """Вычитываем всё из очереди, чтобы разблокировать producer на put()."""
        while True:
            try:
                self._ready_queue.get_nowait()
            except Empty:
                break

    # def get_next_file(self, timeout: float = 600) -> Optional[Tuple[str, Path]]:
    #     """Блокирующе ждёт готовый файл. None = файлы кончились."""
    #     try:
    #         return self._ready_queue.get(timeout=timeout)
    #     except Empty:
    #         self._logger.warning("Таймаут ожидания файла из HDFS")
    #         return None
    def get_next_file(self, timeout: float = 300) -> Optional[Tuple[str, Path]]:
        while True:
            try:
                return self._ready_queue.get(timeout=timeout)
            except Empty:
                if self._producer_thread is not None and self._producer_thread.is_alive():
                    self._logger.warning(
                        f"[QUEUE WAIT] Producer жив, но файл не появился за {timeout}s — ждём ещё"
                    )
                    continue
                else:
                    self._logger.warning(
                        "[QUEUE TIMEOUT] Producer завершился, файлов больше нет"
                    )
                    return None

    
    def notify_file_processed(self):
        """Вызвать ПОСЛЕ удаления обработанного файла — освобождает слот."""
        self._disk_semaphore.release()

    # --- private ---

    def _producer_loop(self, hdfs_paths: List[str]):
        try:
            for hdfs_path in hdfs_paths:
                if self._stop_event.is_set():
                    break

                # Ждём свободного слота
                while not self._stop_event.is_set():
                    if self._disk_semaphore.acquire(timeout=1.0):
                        break
                if self._stop_event.is_set():
                    break

                # Ждём места на диске
                _disk_wait_start = time.monotonic()
                _last_disk_log   = 0.0
                while not self.storage.has_space() and not self._stop_event.is_set():
                    elapsed = time.monotonic() - _disk_wait_start
                    if elapsed - _last_disk_log >= 60:
                        self._logger.warning(
                            f"[DISK WAIT] {elapsed:.0f}s, "
                            f"usage={self.storage.current_usage_gb:.1f} GB / "
                            f"{self.storage.max_cache_bytes / 1024**3:.1f} GB — "
                            f"consumer не успевает?"
                        )
                        _last_disk_log = elapsed
                    time.sleep(0.5)
                if self._stop_event.is_set():
                    break

                local_path = self._download_single(hdfs_path)
                if local_path is not None:
                    self._safe_put((hdfs_path, local_path))
                else:
                    self._disk_semaphore.release()

        except Exception as e:
            self._logger.error(f"Producer loop failed: {e}")
        finally:
            # Sentinel ДОЛЖЕН быть доставлен — используем safe_put
            self._safe_put(None)

    def _safe_put(self, item, timeout_per_attempt: float = 1.0):
        """put() с проверкой stop_event, чтобы не блокироваться навсегда."""
        while not self._stop_event.is_set():
            try:
                self._ready_queue.put(item, timeout=timeout_per_attempt)
                return
            except Full:
                continue
        # Если stop_event установлен — делаем последнюю попытку для sentinel
        if item is None:
            try:
                self._ready_queue.put_nowait(None)
            except Full:
                pass  # consumer уже не читает, это нормально

    def _download_single(self, hdfs_path: str) -> Optional[Path]:
        local_path = self.storage.get_local_path(hdfs_path)

        # Уже скачан ранее?
        if local_path.exists():
            try:
                expected = _hdfs_with_timeout(
                    lambda: self._fs.get_file_info(hdfs_path).size,
                    timeout=30,
                    logger=self._logger,
                    label=f"get_file_info cache-check {Path(hdfs_path).name}",
                )
                if local_path.stat().st_size == expected:
                    self.storage.register_downloaded(expected)
                    return local_path
            except Exception:
                local_path.unlink(missing_ok=True)

        for attempt in range(self.hdfs_config.max_retries):
            tmp_path = local_path.with_suffix(f'.tmp{attempt}')
            result   = {'size': None}
            error    = {'val': None}
            if self._fs is None:
                try:
                    self._fs = _hdfs_with_timeout(
                        self.hdfs_config.create_filesystem,
                        timeout=60,
                        logger=self._logger,
                        label=f"create_filesystem() reconnect retry {attempt+1}/{self.hdfs_config.max_retries}",
                    )
                except Exception as e:
                    self._logger.warning(f"[FS RECONNECT RETRY FAIL] attempt {attempt+1}: {e}")
                    time.sleep(10)
                    continue   # ← нет fs → пропускаем эту попытку

            fs_snap = self._fs   # ← теперь гарантированно не None (или уже сделали continue)


            def _do_download(tmp=tmp_path, fs=fs_snap, res=result, err=error):
                try:
                    expected = fs.get_file_info(hdfs_path).size
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with fs.open_input_stream(hdfs_path) as reader:
                        with open(tmp, 'wb') as writer:
                            while True:
                                chunk = reader.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                writer.write(chunk)
                    res['size'] = expected
                except Exception as e:
                    err['val'] = e

            t = threading.Thread(target=_do_download, daemon=True)
            t.start()
            t.join(timeout=_PER_ATTEMPT_DL_TIMEOUT)

            # --- Поток завис ---
            if t.is_alive():
                self._logger.warning(
                    f"[DL TIMEOUT] {Path(hdfs_path).name}: завис после "
                    f"{_PER_ATTEMPT_DL_TIMEOUT}s, "
                    f"attempt {attempt + 1}/{self.hdfs_config.max_retries}"
                )
                tmp_path.unlink(missing_ok=True)
                try:
                    self._logger.info("[FS RECONNECT] Пересоздаём _fs после таймаута...")
                    self._fs = _hdfs_with_timeout(
                        self.hdfs_config.create_filesystem,
                        timeout=30,
                        logger=self._logger,
                        label="create_filesystem() after DL TIMEOUT",
                    )
                    self._logger.info("[FS RECONNECT] OK")
                except Exception as fs_err:
                    self._logger.error(f"[FS RECONNECT FAIL] {fs_err}")
                    self._fs = None
                continue

            # --- Поток завершился с ошибкой ---
            if error['val'] is not None:
                err_str = str(error['val']).lower()
                if any(k in err_str for k in _KERB_MARKERS):
                    self._logger.error(
                        f"[KERB ERROR] {Path(hdfs_path).name}: "
                        f"{error['val']}, "
                        f"attempt {attempt + 1}/{self.hdfs_config.max_retries}"
                    )
                else:
                    self._logger.warning(
                        f"[DL ERROR] {Path(hdfs_path).name}: {error['val']}, "
                        f"attempt {attempt + 1}/{self.hdfs_config.max_retries}"
                    )
                tmp_path.unlink(missing_ok=True)
                continue

            # --- Проверка размера ---
            actual   = tmp_path.stat().st_size if tmp_path.exists() else 0
            expected = result['size']
            if expected is not None and actual == expected:
                tmp_path.rename(local_path)
                self.storage.register_downloaded(actual)
                self._logger.debug(
                    f"[DL OK] {Path(hdfs_path).name} ({actual / 1024**2:.1f} MB)"
                )
                return local_path
            else:
                tmp_path.unlink(missing_ok=True)
                self._logger.warning(
                    f"[DL SIZE MISMATCH] {Path(hdfs_path).name}: "
                    f"expected={expected}, actual={actual}, "
                    f"attempt {attempt + 1}/{self.hdfs_config.max_retries}"
                )

        self._logger.error(
            f"[DL FAILED] {Path(hdfs_path).name}: "
            f"не удалось скачать после {self.hdfs_config.max_retries} попыток"
        )
        return None




# ---------------------------------------------------------------------------
# BaseIterableDataset — единый класс с двумя режимами
# ---------------------------------------------------------------------------

class BaseIterableDataset(IterableDataset, ABC):
    """
    Универсальный базовый класс для iterable-датасетов.

    Два режима работы (переключается одним параметром):

    1. LOCAL (по умолчанию):
       data_path → glob файлов → распределение по воркерам → _process_file()
       Все файлы уже на диске. Поведение идентично текущему.

    2. STREAMING (hdfs_config + hdfs_data_path):
       Список файлов берётся из HDFS.
       Фоновый поток скачивает файлы наперёд (prefetch_ahead штук).
       После обработки файл удаляется с диска.
       На диске одновременно не более prefetch_ahead файлов.

    Наследник (OmniDataset и т.д.) реализует ТОЛЬКО _process_file().
    Ему не нужно знать, откуда взялся файл — он всегда получает локальный Path.

    Отвечает за:
    1. Распределение файлов между workerами;
    2. Управление Random Seed и детерминизмом (Epoch-based);
    3. Логирование;
    4. Стратегии выборки файлов (subset/fraction);
    5. [NEW] Streaming из HDFS с prefetch и управлением дисковым кэшем.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int = 128,
        shuffle: bool = True,
        random_seed: int = 42,
        worker_id_shift: int = 0,
        drop_last: bool = False,
        file_fraction: float = 1.0,
        file_subset_strategy: str = 'random',
        log_level: str = 'INFO',
        file_pattern: str = '*.parquet',
        process_idx: int = 0,
        num_processes: int = 1,
        num_workers: int = 1,
        # ---- Streaming параметры (все Optional → если не заданы, LOCAL mode) ----
        hdfs_config: Optional[HDFSConfig] = None,
        hdfs_data_path: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_cache_bytes: int = 200 * 1024**3,
        prefetch_ahead: int = 20,
        delete_after_processing: bool = True,
        kerb_refresh_hours: float = 7.0,
        kerb_refresh_cmd: str = "kinit -R",
    ):
        super().__init__()

        # --- Общие параметры ---
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.worker_id_shift = worker_id_shift
        self.drop_last = drop_last
        self.file_fraction = file_fraction
        self.file_subset_strategy = file_subset_strategy
        self.log_level = log_level
        self.process_idx = process_idx
        self.num_processes = num_processes
        self.num_workers = num_workers
        self._logger = None
        self._epoch = 0
        self._is_train = True
        self._np_gen = None
        self._remainder_df = None

        # --- Определяем режим ---
        self._streaming = (hdfs_config is not None and hdfs_data_path is not None)

        if self._streaming:
            # STREAMING MODE
            self.data_path = Path(cache_dir or data_path)
            self.data_path.mkdir(parents=True, exist_ok=True)

            self._hdfs_config = hdfs_config
            self._hdfs_data_path = hdfs_data_path
            self._max_cache_bytes = max_cache_bytes
            self._prefetch_ahead = prefetch_ahead
            self._delete_after_processing = delete_after_processing
            self._kerb_refresh_hours = kerb_refresh_hours
            self._kerb_refresh_cmd = kerb_refresh_cmd

            # Список файлов = из HDFS (строки с путями, не Path)
            self._hdfs_files: Optional[List[str]] = None
            self.files: Optional[List[str]] = None

        else:
            # LOCAL MODE — как было
            self.data_path = Path(data_path)

            if self.data_path.is_file():
                self.files = [self.data_path]
            else:
                self.files = sorted(list(self.data_path.rglob(file_pattern)))

            if not self.files:
                self._get_logger().warning(
                    f"Не найдено файлов в {self.data_path} с шаблоном {file_pattern}"
                )

    # ======================================================================
    # Публичный API (без изменений)
    # ======================================================================

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def set_epoch(self, epoch: int) -> None:
        """Важно для DDP и корректного шаффла между эпохами."""
        self._epoch = int(epoch)

    @property
    def is_streaming(self) -> bool:
        """Текущий режим работы."""
        return self._streaming

    # ======================================================================
    # Абстрактный метод (без изменений)
    # ======================================================================

    @abstractmethod
    def _process_file(self, file_path: Path) -> Iterator[Any]:
        """
        Метод-генератор, который должен реализовать наследник.
        Принимает путь к ЛОКАЛЬНОМУ файлу, возвращает итератор батчей.
        В обоих режимах file_path — всегда локальный путь.
        """
        pass

    # ======================================================================
    # __iter__ — единая точка входа
    # ======================================================================

    def __iter__(self) -> Iterator[Any]:
        """
        Главный цикл итерации.
        Автоматически выбирает стратегию по self._streaming.
        """
        self._init_rng()

        if self._streaming:
            yield from self._iter_streaming()
        else:
            yield from self._iter_local()

    # ======================================================================
    # LOCAL режим (оригинальная логика, без изменений)
    # ======================================================================

    def _iter_local(self) -> Iterator[Any]:
        """Итерация по локальным файлам — оригинальное поведение."""
        # 1. Выбираем активные файлы (fraction)
        active_files = self._select_active_files()

        # 2. Шаффл порядка файлов
        if self.shuffle and self._is_train:
            active_files = list(self._np_gen.permutation(active_files))

        # 3. Распределяем файлы этому воркеру
        my_files = self._distribute_files(active_files)

        self._get_logger().info(f"[LOCAL] Обработаем {len(my_files)} файлов")

        # 4. Итерируемся
        remainder_df = None
        for file_path in my_files:
            try:
                for batch in self._process_file(file_path, remainder_df=remainder_df):
                    yield batch
                remainder_df = self._remainder_df
            except Exception as e:
                self._get_logger().error(f"Не удалось обработать {file_path}: {e}")
                remainder_df = None
                continue
        
        if remainder_df is not None and not self.drop_last and remainder_df.height > 0:
            indices = np.arange(remainder_df.height)
            yield self._convert_to_tensor_dict(remainder_df, indices)

    # ======================================================================
    # STREAMING режим (новая логика)
    # ======================================================================

    def _iter_streaming(self) -> Iterator[Any]:
        self._ensure_file_list()
        active_files = self._select_active_files_hdfs()

        if self.shuffle and self._is_train:
            active_files = list(self._np_gen.permutation(active_files))

        my_files = self._distribute_files(active_files)

        logger = self._get_logger()
        logger.info(f"[STREAMING] Обработаем {len(my_files)} файлов из HDFS")

        info = get_worker_info()
        if info is not None:
            nw          = info.num_workers if self.num_workers == 1 else self.num_workers
            total_parts = self.num_processes * nw
            wid         = (self.process_idx * nw) + info.id
        else:
            total_parts = self.num_processes
            wid         = self.process_idx

        worker_cache_dir = self.data_path / f"_stream_w{wid}"
        per_worker_cache = self._max_cache_bytes // max(1, total_parts)

        storage    = _LocalStorageManager(worker_cache_dir, per_worker_cache)
        prefetcher = _HDFSPrefetcher(self._hdfs_config, storage, self._prefetch_ahead)

        logger.info("[STREAMING] Запускаем prefetcher (возможны retry create_filesystem)...")
        prefetcher.start(my_files)
        logger.info("[STREAMING] prefetcher.start() вернул управление")

        start_time      = time.monotonic()
        files_processed = 0
        remainder_df = None

        try:
            while True:
                # Kerberos ПЕРЕД get_next_file — срабатывает даже при простое producer
                elapsed_hours = (time.monotonic() - start_time) / 3600
                if elapsed_hours >= self._kerb_refresh_hours:
                    self._refresh_kerberos(prefetcher=prefetcher)
                    start_time = time.monotonic()

                result = prefetcher.get_next_file(timeout=300)
                if result is None:
                    break

                hdfs_path, local_path = result

                try:
                    for batch in self._process_file(local_path, remainder_df=remainder_df):
                        yield batch
                    remainder_df = self._remainder_df
                    files_processed += 1
                    if files_processed % 100 == 0:
                        logger.info(
                            f"[STREAMING] {files_processed}/{len(my_files)}, "
                            f"кэш: {storage.current_usage_gb:.1f} GB"
                        )
                except Exception as e:
                    logger.error(f"Ошибка обработки {local_path}: {e}")
                    remainder_df = None
                finally:
                    if self._delete_after_processing:
                        storage.release_file(local_path)
                    prefetcher.notify_file_processed()

                if remainder_df is not None and not self.drop_last and remainder_df.height > 0:
                    indices = np.arange(remainder_df.height)
                    yield self._convert_to_tensor_dict(remainder_df, indices)

        finally:
            prefetcher.stop()
            if files_processed == 0 and len(my_files) > 0:
                logger.error(
                    f"[STREAMING FAIL] Обработано 0 файлов из {len(my_files)} — "
                    f"проверь логи [FS FAIL] выше"
                )
            else:
                logger.info(f"[STREAMING] Завершено: {files_processed}/{len(my_files)} файлов")


    # ======================================================================
    # Внутренние методы (общие / shared)
    # ======================================================================

    def _get_logger(self) -> logging.Logger:
        if self._logger is None:
            pid = os.getpid()
            info = get_worker_info()
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            name = f'Data.R{rank}.W{info.id}.P{pid}' if info else f"Data.Main.R{rank}.P{pid}"


            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(getattr(logging, self.log_level.upper()))
            self._logger = logger
        return self._logger

    def _init_rng(self) -> None:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        seed_seq = np.random.SeedSequence(
            [self.random_seed, self._epoch, wid, self.worker_id_shift]
        )
        self._np_gen = np.random.Generator(np.random.PCG64(seed_seq))

    def _select_active_files(self) -> list:
        """Выбор подмножества файлов для текущей эпохи (работает в обоих режимах)."""
        if self._streaming and self._hdfs_files is None:
            self._ensure_file_list()
        files = self.files
        if self.file_fraction >= 1.0 or not files:
            return files

        k = max(1, int(np.ceil(len(files) * self.file_fraction)))

        if self.file_subset_strategy == 'head':
            return files[:k]
        elif self.file_subset_strategy == 'random':
            rng = np.random.RandomState(self.random_seed + self._epoch)
            if isinstance(files[0], str):
                # streaming mode: list of strings
                indices = rng.choice(len(files), size=k, replace=False)
                return [files[i] for i in indices]
            else:
                return list(rng.choice(files, size=k, replace=False))
        return files

    # В streaming-режиме _select_active_files_hdfs просто делегирует
    # _select_active_files — список файлов уже в self.files
    _select_active_files_hdfs = _select_active_files

    def _distribute_files(self, files: list) -> list:
        """Распределение файлов между воркерами (работает в обоих режимах)."""
        info = get_worker_info()
        if info is None:
            return files[self.process_idx::self.num_processes]

        num_workers = info.num_workers if self.num_workers == 1 else self.num_workers
        total_parts = self.num_processes * num_workers
        part_id = (self.process_idx * num_workers) + info.id

        return files[part_id::total_parts]

    # ======================================================================
    # Streaming-only внутренние методы
    # ======================================================================

    def _ensure_file_list(self) -> List[str]:
        """
        Ленивая загрузка списка файлов из HDFS

        Вызывается при первом обращении, результат кэшируется.
        Важно: не вызывать в __init__ - сломает Pytorch Dataloader
        """
        if self._hdfs_files is None:
            self._hdfs_files = self._fetch_hdfs_file_list()
            self.files = self._hdfs_files
        return self._hdfs_files

#     def _fetch_hdfs_file_list(self) -> List[str]:
#         """Получает полный список parquet-файлов из HDFS."""
#         fs = self._hdfs_config.create_filesystem()
#         selector = pyarrow.fs.FileSelector(self._hdfs_data_path, recursive=True)
#         all_files = fs.get_file_info(selector)

#         result = sorted([
#             f.path for f in all_files
#             if f.is_file and f.size > 0 and f.path.endswith('.parquet')
#         ])

#         self._get_logger().info(
#             f"HDFS: {len(result)} parquet-файлов в {self._hdfs_data_path}"
#         )
#         return result

    def _fetch_hdfs_file_list(self) -> List[str]:
        logger = self._get_logger()
        logger.info("[FILELIST] Создаём FS для сканирования файлов...")
        try:
            fs = _hdfs_with_timeout(
                self._hdfs_config.create_filesystem,
                timeout=60,
                logger=logger,
                label="create_filesystem() in _fetch_hdfs_file_list",
            )
        except Exception as e:
            logger.error(f"[FILELIST FAIL] create_filesystem(): {e}")
            raise

        logger.info(f"[FILELIST] Сканируем {self._hdfs_data_path}...")
        selector = pyarrow.fs.FileSelector(self._hdfs_data_path, recursive=True)
        try:
            all_files = _hdfs_with_timeout(
                lambda: fs.get_file_info(selector),
                timeout=120,
                logger=logger,
                label="get_file_info(selector) in _fetch_hdfs_file_list",
            )
        except Exception as e:
            logger.error(f"[FILELIST FAIL] get_file_info(selector): {e}")
            raise

        result = sorted([
            f.path for f in all_files
            if f.is_file and f.size > 0 and f.path.endswith('.parquet')
        ])
        logger.info(f"[FILELIST OK] {len(result)} parquet-файлов в {self._hdfs_data_path}")
        return result


    # def _refresh_kerberos(self):
    #     try:
    #         subprocess.run(
    #             self._kerb_refresh_cmd.split(),
    #             check=True, capture_output=True, timeout=30,
    #         )
    #         result = subprocess.run(
    #             ["klist"], capture_output=True, text=True, timeout=10,
    #         )
    #         new_ticket = result.stdout.split('\n')[0].split(':')[-1].strip()
    #         self._hdfs_config.kerb_ticket = new_ticket
    #         self._get_logger().info("Kerberos тикет обновлён")
    #     except Exception as e:
    #         self._get_logger().warning(f"Не удалось обновить kerberos: {e}")
    def _refresh_kerberos(self, prefetcher: '_HDFSPrefetcher' = None):
        logger      = self._get_logger()
        ccache_path = os.environ.get('KRB5CCNAME', '')
        lock_path   = (ccache_path + '.lock') if ccache_path else '/tmp/krb5cc_refresh.lock'

        logger.info(
            f"[KERB] Запускаем обновление тикета. "
           f"KRB5CCNAME={ccache_path or 'не задан'}, "
            f"UID={os.getuid()}, PID={os.getpid()}"
        )

        try:
            lock_file = open(lock_path, 'w')
        except OSError as e:
            logger.error(
                f"[KERB] Не удалось открыть lock-файл {lock_path}: {e} — пропускаем refresh"
            )
            return

        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info(f"[KERB] Lock получен: {lock_path}")

        except BlockingIOError:
            logger.info("[KERB] Другой воркер держит lock — ждём завершения его refresh...")
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                logger.info(
                    "[KERB] Lock освобождён другим воркером — пропускаем kinit, reconnect _fs"
                )
                if prefetcher is not None:
                    try:
                        self._hdfs_config.kerb_ticket = None
                        prefetcher._fs = _hdfs_with_timeout(
                            self._hdfs_config.create_filesystem,
                            timeout=60,
                            logger=logger,
                            label="create_filesystem() in skip-branch kerberos refresh",
                        )
                        logger.info("[KERB] _fs переподключён в skip-ветке ✓")
                    except Exception as fs_err:
                        logger.warning(f"[KERB] Ошибка reconnect _fs в skip-ветке: {fs_err}")
            except Exception as e:
                logger.warning(f"[KERB] Не удалось дождаться lock: {e}")
            finally:
                lock_file.close()
            return


        try:
            # Состояние тикета ДО
            try:
                before = subprocess.run(["klist"], capture_output=True, text=True, timeout=10)
                logger.info(f"[KERB BEFORE]\n{before.stdout.strip()}")
            except Exception as e:
                logger.warning(f"[KERB] klist before failed: {e}")

            # kinit -R
            try:
                r = subprocess.run(
                    self._kerb_refresh_cmd.split(),
                    check=True, capture_output=True, text=True, timeout=30,
                )
                logger.info(
                    f"[KERB kinit OK] stdout={r.stdout.strip()!r} stderr={r.stderr.strip()!r}"
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"[KERB kinit FAIL] returncode={e.returncode}, "
                    f"stderr={e.stderr.strip()!r} — тикет НЕ обновлён"
                )
                return
            except Exception as e:
                logger.error(f"[KERB kinit FAIL] {e} — тикет НЕ обновлён")
                return

            # Состояние тикета ПОСЛЕ
            try:
                after = subprocess.run(["klist"], capture_output=True, text=True, timeout=10)
                logger.info(f"[KERB AFTER]\n{after.stdout.strip()}")
            except Exception as e:
                logger.warning(f"[KERB] klist after failed: {e}")

            # kerb_ticket=None → pyarrow подхватит /home/datalab/nfs/krb5cc из KRB5CCNAME
            self._hdfs_config.kerb_ticket = None

            # Переподключаем _fs — Java держит старый Delegation Token
            if prefetcher is not None:
                logger.info("[KERB] Переподключаем _fs в prefetcher...")
                try:
                    prefetcher._fs = _hdfs_with_timeout(
                        self._hdfs_config.create_filesystem,
                        timeout=60,
                        logger=logger,
                        label="create_filesystem() after kerberos refresh",
                    )
                    logger.info("[KERB] _fs переподключён успешно")
                except Exception as fs_err:
                    logger.error(
                        f"[KERB] _fs reconnect fail: {fs_err} — "
                        f"старый Java Delegation Token до следующего start()"
                    )
            else:
                logger.warning(
                    "[KERB] prefetcher не передан → _fs держит старый "
                    "Java Delegation Token до следующего start()"
                )

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            logger.info(f"[KERB] Lock освобождён: {lock_path}")
