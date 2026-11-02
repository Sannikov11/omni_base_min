"""
base_dataset.py — production IterableDataset с LOCAL и STREAMING режимами.

STREAMING: скачивает Parquet-файлы из HDFS (PyArrow HadoopFileSystem + Kerberos)
в фоновом потоке, по одному выдаёт consumer-у, удаляет после обработки.

Критические фиксы, заложенные в архитектуру:
  • При AccessControlException (Kerberos) — немедленный reconnect _fs
  • _refresh_kerberos не делает ранний return при kinit fail, всегда reconnect
  • Доступ к _fs через _get_fs()/_set_fs() с Lock (write из двух потоков)
"""
from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pyarrow.fs
from torch.utils.data import IterableDataset, get_worker_info

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KERB_MARKERS: Tuple[str, ...] = (
    'kerberos', 'tgt', 'gssexception', 'errno 255', 'no valid credentials',
)
_DL_TIMEOUT = 150        # секунд на одну попытку скачивания
_FS_TIMEOUT = 60         # секунд на create_filesystem()
_FS_INIT_RETRIES = 3     # попыток при старте prefetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_with_timeout(fn, timeout: float, logger: logging.Logger, label: str) -> Any:
    """Запускает fn() в daemon-треде; бросает TimeoutError если завис."""
    result: Dict[str, Any] = {'value': None, 'error': None}

    def _worker():
        try:
            result['value'] = fn()
        except Exception as exc:
            result['error'] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.error(f"[HANG] {label} завис после {timeout}s")
        raise TimeoutError(f"{label} timeout after {timeout}s")
    if result['error'] is not None:
        raise result['error']
    return result['value']


# ---------------------------------------------------------------------------
# HDFSConfig  (публичный API — не менять)
# ---------------------------------------------------------------------------

@dataclass
class HDFSConfig:
    """Параметры подключения к HDFS. Передаётся в BaseIterableDataset для streaming."""

    hdfs_url: str = "hdfs://arnsdpsbx"
    kerb_ticket: Optional[str] = None
    user: Optional[str] = None
    extra_conf: Dict[str, str] = field(default_factory=lambda: {
        'hadoop.security.authentication': 'kerberos',
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
# _LocalStorageManager
# ---------------------------------------------------------------------------

class _LocalStorageManager:
    """Учёт дискового кэша: место, регистрация скачанного, удаление файлов."""

    def __init__(self, cache_dir: Path, max_cache_bytes: int) -> None:
        self.cache_dir = cache_dir
        self.max_cache_bytes = max_cache_bytes
        self._lock = threading.Lock()
        self._logger = logging.getLogger("StreamingStorage")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._current_bytes: int = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
        self._logger.info(
            f"Cache {cache_dir}: {self._current_bytes / 1024**3:.1f} GB / "
            f"{max_cache_bytes / 1024**3:.1f} GB"
        )

    @property
    def current_usage_gb(self) -> float:
        with self._lock:
            return self._current_bytes / 1024**3

    def has_space(self) -> bool:
        with self._lock:
            return self._current_bytes < self.max_cache_bytes

    def register_downloaded(self, size_bytes: int) -> None:
        with self._lock:
            self._current_bytes += size_bytes

    def release_file(self, local_path: Path) -> None:
        try:
            size = local_path.stat().st_size if local_path.exists() else 0
            local_path.unlink(missing_ok=True)
            with self._lock:
                self._current_bytes = max(0, self._current_bytes - size)
        except OSError as exc:
            self._logger.warning(f"Не удалось удалить {local_path}: {exc}")

    def local_path_for(self, hdfs_path: str) -> Path:
        """HDFS-путь → локальный путь с сохранением иерархии."""
        clean = hdfs_path.split('://', 1)[-1] if '://' in hdfs_path else hdfs_path
        clean = clean.split('/', 1)[-1] if '/' in clean else clean
        parts = [p for p in clean.split('/') if p]

        if len(parts) >= 3 and parts[0] == 'user' and parts[1] == 'team':
            start = 3 + (1 if len(parts) > 3 and parts[3] == 'hive' else 0)
            rel = parts[start:] or [parts[-1]]
        else:
            rel = parts[-2:] if len(parts) >= 2 else parts[-1:]

        local = self.cache_dir.joinpath(*rel)
        local.parent.mkdir(parents=True, exist_ok=True)
        return local


# ---------------------------------------------------------------------------
# _HDFSPrefetcher
# ---------------------------------------------------------------------------

class _HDFSPrefetcher:
    """
    Producer-consumer: фоновый поток скачивает файлы из HDFS наперёд.

    Семафор _disk_semaphore ограничивает кол-во одновременно лежащих на диске файлов.
    Sentinel None в очереди сигнализирует об окончании потока.
    """

    def __init__(
        self,
        hdfs_config: HDFSConfig,
        storage: _LocalStorageManager,
        prefetch_ahead: int,
    ) -> None:
        self.hdfs_config = hdfs_config
        self.storage = storage
        self._log = logging.getLogger("HDFSPrefetcher")
        self._queue: Queue[Optional[Tuple[str, Path]]] = Queue(maxsize=prefetch_ahead + 1)
        self._stop = threading.Event()
        self._semaphore = threading.Semaphore(prefetch_ahead)
        self._thread: Optional[threading.Thread] = None
        # _fs доступен из двух потоков → lock обязателен
        self._fs: Optional[pyarrow.fs.HadoopFileSystem] = None
        self._fs_lock = threading.Lock()

    # --- thread-safe доступ к _fs ---

    def _get_fs(self) -> Optional[pyarrow.fs.HadoopFileSystem]:
        with self._fs_lock:
            return self._fs

    def _set_fs(self, fs: Optional[pyarrow.fs.HadoopFileSystem]) -> None:
        with self._fs_lock:
            self._fs = fs

    def _reconnect_fs(self, label: str, timeout: int = _FS_TIMEOUT) -> bool:
        """Пересоздаёт _fs; возвращает True при успехе."""
        try:
            fs = _run_with_timeout(self.hdfs_config.create_filesystem, timeout, self._log, label)
            self._set_fs(fs)
            self._log.info(f"[FS RECONNECT OK] {label}")
            return True
        except Exception as exc:
            self._log.error(f"[FS RECONNECT FAIL] {label}: {exc}")
            self._set_fs(None)
            return False

    # --- lifecycle ---

    def start(self, hdfs_paths: List[str]) -> None:
        self._stop.clear()
        for attempt in range(1, _FS_INIT_RETRIES + 1):
            label = f"create_filesystem() attempt {attempt}/{_FS_INIT_RETRIES}"
            if self._reconnect_fs(label):
                break
            if attempt < _FS_INIT_RETRIES:
                time.sleep(30)
            else:
                self._log.error("[FS FAIL] Все попытки исчерпаны → sentinel")
                self._safe_put(None)
                return
        self._thread = threading.Thread(
            target=self._producer_loop, args=(hdfs_paths,), daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._drain()
        if self._thread:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                self._log.warning("Producer thread не завершился за 30с")

    # --- consumer API ---

    def get_next_file(self, timeout: float = 300) -> Optional[Tuple[str, Path]]:
        while True:
            try:
                return self._queue.get(timeout=timeout)
            except Empty:
                if self._thread and self._thread.is_alive():
                    self._log.warning(f"[QUEUE WAIT] нет файла за {timeout}s, producer жив — ждём")
                else:
                    return None

    def notify_file_processed(self) -> None:
        self._semaphore.release()

    # --- producer internals ---

    def _producer_loop(self, hdfs_paths: List[str]) -> None:
        try:
            for hdfs_path in hdfs_paths:
                if self._stop.is_set():
                    break
                if not self._acquire_slot():
                    break
                if not self._wait_for_disk_space():
                    break
                local = self._download_single(hdfs_path)
                if local is not None:
                    self._safe_put((hdfs_path, local))
                else:
                    self._semaphore.release()
        except Exception as exc:
            self._log.error(f"Producer loop failed: {exc}")
        finally:
            self._safe_put(None)

    def _acquire_slot(self) -> bool:
        """Ждёт свободного слота семафора; возвращает False при stop."""
        while not self._stop.is_set():
            if self._semaphore.acquire(timeout=1.0):
                return True
        return False

    def _wait_for_disk_space(self) -> bool:
        """Ждёт пока на диске есть место; логирует раз в минуту."""
        start = time.monotonic()
        last_log = 0.0
        while not self.storage.has_space() and not self._stop.is_set():
            elapsed = time.monotonic() - start
            if elapsed - last_log >= 60:
                self._log.warning(
                    f"[DISK WAIT] {elapsed:.0f}s, "
                    f"{self.storage.current_usage_gb:.1f} GB / "
                    f"{self.storage.max_cache_bytes / 1024**3:.1f} GB"
                )
                last_log = elapsed
            time.sleep(0.5)
        return not self._stop.is_set()

    def _safe_put(self, item: Optional[Tuple[str, Path]]) -> None:
        """put() без блокировки навсегда; последняя попытка для sentinel."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=1.0)
                return
            except Full:
                continue
        if item is None:
            try:
                self._queue.put_nowait(None)
            except Full:
                pass

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _download_single(self, hdfs_path: str) -> Optional[Path]:
        local = self.storage.local_path_for(hdfs_path)

        # Проверяем уже скачанный файл
        if local.exists():
            try:
                expected = _run_with_timeout(
                    lambda: self._get_fs().get_file_info(hdfs_path).size,
                    30, self._log, f"get_file_info cache-check {local.name}",
                )
                if local.stat().st_size == expected:
                    self.storage.register_downloaded(expected)
                    return local
            except Exception:
                local.unlink(missing_ok=True)

        for attempt in range(self.hdfs_config.max_retries):
            if self._get_fs() is None:
                ok = self._reconnect_fs(
                    f"create_filesystem() retry {attempt + 1}/{self.hdfs_config.max_retries}"
                )
                if not ok:
                    time.sleep(10)
                    continue

            result = self._try_download(hdfs_path, local, attempt)
            if result == 'ok':
                return local
            if result == 'kerb':
                # Немедленный reconnect — старый _fs держит протухший Delegation Token
                self._reconnect_fs(f"create_filesystem() after KERB ERROR attempt {attempt + 1}")
            # 'error' / 'timeout' / 'mismatch' → reconnect уже сделан внутри, просто retry

        self._log.error(f"[DL FAILED] {local.name}: не удалось после {self.hdfs_config.max_retries} попыток")
        return None

    def _try_download(self, hdfs_path: str, local: Path, attempt: int) -> str:
        """
        Запускает одну попытку скачивания в daemon-треде.
        Возвращает: 'ok' | 'kerb' | 'error' | 'timeout' | 'mismatch'
        """
        tmp = local.with_suffix(f".tmp{attempt}")
        err: Dict[str, Optional[Exception]] = {'val': None}
        size: Dict[str, Optional[int]] = {'val': None}
        fs_snap = self._get_fs()

        def _worker():
            try:
                expected = fs_snap.get_file_info(hdfs_path).size
                local.parent.mkdir(parents=True, exist_ok=True)
                with fs_snap.open_input_stream(hdfs_path) as src, open(tmp, 'wb') as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        dst.write(chunk)
                size['val'] = expected
            except Exception as exc:
                err['val'] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=_DL_TIMEOUT)

        fname = local.name

        if t.is_alive():
            self._log.warning(f"[DL TIMEOUT] {fname} attempt {attempt + 1}")
            tmp.unlink(missing_ok=True)
            self._reconnect_fs(f"create_filesystem() after DL TIMEOUT attempt {attempt + 1}", timeout=30)
            return 'timeout'

        if err['val'] is not None:
            err_str = str(err['val']).lower()
            is_kerb = any(k in err_str for k in _KERB_MARKERS)
            level = self._log.error if is_kerb else self._log.warning
            tag = "[KERB ERROR]" if is_kerb else "[DL ERROR]"
            level(f"{tag} {fname}: {err['val']} attempt {attempt + 1}")
            tmp.unlink(missing_ok=True)
            return 'kerb' if is_kerb else 'error'

        actual = tmp.stat().st_size if tmp.exists() else 0
        expected = size['val']
        if expected is not None and actual == expected:
            tmp.rename(local)
            self.storage.register_downloaded(actual)
            self._log.debug(f"[DL OK] {fname} ({actual / 1024**2:.1f} MB)")
            return 'ok'

        tmp.unlink(missing_ok=True)
        self._log.warning(f"[DL SIZE MISMATCH] {fname}: expected={expected}, actual={actual} attempt {attempt + 1}")
        return 'mismatch'


# ---------------------------------------------------------------------------
# BaseIterableDataset  (публичный API — не менять)
# ---------------------------------------------------------------------------

class BaseIterableDataset(IterableDataset, ABC):
    """
    Базовый IterableDataset с двумя режимами:

    LOCAL    — файлы уже на диске, data_path → glob → _process_file()
    STREAMING — файлы в HDFS, фоновый prefetcher скачивает их по одному
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
        hdfs_config: Optional[HDFSConfig] = None,
        hdfs_data_path: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_cache_bytes: int = 200 * 1024**3,
        prefetch_ahead: int = 20,
        delete_after_processing: bool = True,
        kerb_refresh_hours: float = 7.0,
        kerb_refresh_cmd: str = "kinit -R",
    ) -> None:
        super().__init__()
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
        self._epoch = 0
        self._is_train = True
        self._np_gen: Optional[np.random.Generator] = None
        self._remainder_df = None
        self._logger: Optional[logging.Logger] = None

        self._streaming = hdfs_config is not None and hdfs_data_path is not None

        if self._streaming:
            self.data_path = Path(cache_dir or data_path)
            self.data_path.mkdir(parents=True, exist_ok=True)
            self._hdfs_config = hdfs_config
            self._hdfs_data_path = hdfs_data_path
            self._max_cache_bytes = max_cache_bytes
            self._prefetch_ahead = prefetch_ahead
            self._delete_after_processing = delete_after_processing
            self._kerb_refresh_hours = kerb_refresh_hours
            self._kerb_refresh_cmd = kerb_refresh_cmd
            self._hdfs_files: Optional[List[str]] = None
            self.files: Optional[List[str]] = None
        else:
            self.data_path = Path(data_path)
            self.files = (
                [self.data_path] if self.data_path.is_file()
                else sorted(self.data_path.rglob(file_pattern))
            )
            if not self.files:
                self._get_logger().warning(f"Нет файлов в {self.data_path} с паттерном {file_pattern}")

    # --- публичный API ---

    def train(self) -> None:
        self._is_train = True

    def eval(self) -> None:
        self._is_train = False

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @abstractmethod
    def _process_file(self, file_path: Path, **kwargs) -> Iterator[Any]:
        ...

    # --- итерация ---

    def __iter__(self) -> Iterator[Any]:
        self._init_rng()
        yield from self._iter_streaming() if self._streaming else self._iter_local()

    def _iter_local(self) -> Iterator[Any]:
        my_files = self._distribute_files(self._get_active_files())
        self._get_logger().info(f"[LOCAL] {len(my_files)} файлов")
        remainder_df = None
        for fp in my_files:
            try:
                for batch in self._process_file(fp, remainder_df=remainder_df):
                    yield batch
                remainder_df = self._remainder_df
            except Exception as exc:
                self._get_logger().error(f"Ошибка обработки {fp}: {exc}")
                remainder_df = None
        if remainder_df is not None and not self.drop_last and remainder_df.height > 0:
            yield self._convert_to_tensor_dict(remainder_df, np.arange(remainder_df.height))

    def _iter_streaming(self) -> Iterator[Any]:
        self._ensure_file_list()
        my_files = self._distribute_files(self._get_active_files())
        log = self._get_logger()
        log.info(f"[STREAMING] {len(my_files)} файлов")

        wid, total_parts = self._worker_identity()
        storage = _LocalStorageManager(
            self.data_path / f"_stream_w{wid}",
            self._max_cache_bytes // max(1, total_parts),
        )
        prefetcher = _HDFSPrefetcher(self._hdfs_config, storage, self._prefetch_ahead)
        prefetcher.start(my_files)

        kerb_deadline = time.monotonic() + self._kerb_refresh_hours * 3600
        files_done = 0
        remainder_df = None
        try:
            while True:
                if time.monotonic() >= kerb_deadline:
                    self._refresh_kerberos(prefetcher)
                    kerb_deadline = time.monotonic() + self._kerb_refresh_hours * 3600

                item = prefetcher.get_next_file(timeout=300)
                if item is None:
                    break
                hdfs_path, local = item
                try:
                    for batch in self._process_file(local, remainder_df=remainder_df):
                        yield batch
                    remainder_df = self._remainder_df
                    files_done += 1
                    if files_done % 100 == 0:
                        log.info(f"[STREAMING] {files_done}/{len(my_files)}, cache {storage.current_usage_gb:.1f} GB")
                except Exception as exc:
                    log.error(f"Ошибка обработки {local}: {exc}")
                    remainder_df = None
                finally:
                    if self._delete_after_processing:
                        storage.release_file(local)
                    prefetcher.notify_file_processed()

            if remainder_df is not None and not self.drop_last and remainder_df.height > 0:
                yield self._convert_to_tensor_dict(remainder_df, np.arange(remainder_df.height))
        finally:
            prefetcher.stop()
            if files_done == 0 and my_files:
                log.error(f"[STREAMING FAIL] 0/{len(my_files)} файлов — проверь логи [FS FAIL]")
            else:
                log.info(f"[STREAMING] done: {files_done}/{len(my_files)}")

    # --- вспомогательные методы ---

    def _worker_identity(self) -> Tuple[int, int]:
        """Возвращает (worker_id, total_parts) для текущего DataLoader-воркера."""
        info = get_worker_info()
        if info is None:
            return self.process_idx, self.num_processes
        nw = info.num_workers if self.num_workers == 1 else self.num_workers
        return (self.process_idx * nw) + info.id, self.num_processes * nw

    def _get_logger(self) -> logging.Logger:
        if self._logger is None:
            info = get_worker_info()
            try:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            name = (
                f"Data.R{rank}.W{info.id}.P{os.getpid()}" if info
                else f"Data.Main.R{rank}.P{os.getpid()}"
            )
            logger = logging.getLogger(name)
            if not logger.handlers:
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(h)
                logger.setLevel(getattr(logging, self.log_level.upper()))
            self._logger = logger
        return self._logger

    def _init_rng(self) -> None:
        info = get_worker_info()
        wid = info.id if info else 0
        seed = np.random.SeedSequence([self.random_seed, self._epoch, wid, self.worker_id_shift])
        self._np_gen = np.random.Generator(np.random.PCG64(seed))

    def _get_active_files(self) -> list:
        files = self.files
        if not files or self.file_fraction >= 1.0:
            return files

        k = max(1, int(np.ceil(len(files) * self.file_fraction)))
        if self.file_subset_strategy == 'head':
            return files[:k]

        rng = np.random.RandomState(self.random_seed + self._epoch)
        if isinstance(files[0], str):
            idx = rng.choice(len(files), size=k, replace=False)
            return [files[i] for i in idx]
        return list(rng.choice(files, size=k, replace=False))

    # Alias для совместимости с вызовом в _iter_streaming
    _select_active_files = _get_active_files
    _select_active_files_hdfs = _get_active_files

    def _distribute_files(self, files: list) -> list:
        info = get_worker_info()
        if info is None:
            return files[self.process_idx::self.num_processes]
        nw = info.num_workers if self.num_workers == 1 else self.num_workers
        part = (self.process_idx * nw) + info.id
        return files[part::self.num_processes * nw]

    def _ensure_file_list(self) -> List[str]:
        """Ленивая загрузка списка файлов из HDFS. Не вызывать в __init__."""
        if self._hdfs_files is None:
            self._hdfs_files = self._fetch_hdfs_file_list()
            self.files = self._hdfs_files
        return self._hdfs_files

    def _fetch_hdfs_file_list(self) -> List[str]:
        log = self._get_logger()
        fs = _run_with_timeout(
            self._hdfs_config.create_filesystem, 60, log,
            "create_filesystem() in _fetch_hdfs_file_list",
        )
        selector = pyarrow.fs.FileSelector(self._hdfs_data_path, recursive=True)
        all_files = _run_with_timeout(
            lambda: fs.get_file_info(selector), 120, log,
            "get_file_info(selector) in _fetch_hdfs_file_list",
        )
        result = sorted(
            f.path for f in all_files
            if f.is_file and f.size > 0 and f.path.endswith('.parquet')
        )
        log.info(f"[FILELIST] {len(result)} файлов в {self._hdfs_data_path}")
        return result

    def _refresh_kerberos(self, prefetcher: Optional[_HDFSPrefetcher] = None) -> None:
        log = self._get_logger()
        ccache = os.environ.get('KRB5CCNAME', '')
        lock_path = f"{ccache}.lock" if ccache else '/tmp/krb5cc_refresh.lock'

        try:
            lock_file = open(lock_path, 'w')
        except OSError as exc:
            log.error(f"[KERB] Не удалось открыть lock {lock_path}: {exc}")
            return

        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Другой воркер уже делает kinit — ждём и переподключаем _fs
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                log.info("[KERB] Другой воркер завершил refresh — переподключаем _fs")
                self._kerb_reconnect_fs(prefetcher, log)
            except Exception as exc:
                log.warning(f"[KERB] Не удалось дождаться lock: {exc}")
            finally:
                lock_file.close()
            return

        try:
            self._run_kinit(log)
            self._hdfs_config.kerb_ticket = None
            self._kerb_reconnect_fs(prefetcher, log)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def _run_kinit(self, log: logging.Logger) -> bool:
        """Запускает kinit; возвращает True при успехе. Не бросает исключений."""
        try:
            r = subprocess.run(
                self._kerb_refresh_cmd.split(),
                check=True, capture_output=True, text=True, timeout=30,
            )
            log.info(f"[KERB] kinit OK: {r.stderr.strip() or r.stdout.strip()!r}")
            return True
        except subprocess.CalledProcessError as exc:
            log.error(f"[KERB] kinit FAIL rc={exc.returncode}: {exc.stderr.strip()!r}")
        except Exception as exc:
            log.error(f"[KERB] kinit FAIL: {exc}")
        # Не делаем ранний return — _kerb_reconnect_fs всё равно попробует текущий ccache
        return False

    def _kerb_reconnect_fs(
        self, prefetcher: Optional[_HDFSPrefetcher], log: logging.Logger,
    ) -> None:
        """Переподключает _fs в prefetcher-е (kerb_ticket уже сброшен в None)."""
        if prefetcher is None:
            log.warning("[KERB] prefetcher не передан — _fs не обновлён")
            return
        ok = prefetcher._reconnect_fs("create_filesystem() after kerberos refresh")
        if not ok:
            log.error("[KERB] _fs reconnect fail — старый Delegation Token до следующего start()")
