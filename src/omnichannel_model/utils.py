import os
import subprocess
import logging
import pickle
import pyarrow as pa
import pyarrow.fs
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
from typing import List, Optional, Any
from functools import partial
from pathlib import Path
from getpass import getpass
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


def load_pickle(file_path: str) -> Any:
    """Загрузка объектов из Pickle"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def make_kinit(user: Optional[str] = None):
    """Делает kinit к нужной учетной записи

    Args:
        user: Имя пользователя (principal). Если None, kinit выполнится для текущего пользователя
    """
    cmd = ["kinit"]
    if user:
        cmd.append(user)

    prompt = f"Пароль для {user}: " if user else "Пароль для kinit: "
    password = getpass(prompt=prompt)

    try:
        result = subprocess.run(
            cmd,
            input=password.encode(), # kinit ждет байты
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info(f"Успешно создали Kerberos тикет для {user if user else 'текущего пользователя'}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode().strip()
        logger.error(f"Не удалось получить тикет: {error_message}")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")


def get_kerberos_ticket() -> str:
    """Получает текущий Kerberos тикет"""
    try:
        output = subprocess.check_output(['klist'], text=True)
        lines = output.split('\n')
        ticket = lines[0].split(':')[-1].strip()
        return ticket
    except Exception as e:
        logger.warning(f"Не удалось получить тикет Kerberos: {e}")


def get_hdfs_fs(node_path: str, user: Optional[str] = None) -> pyarrow.fs.FileSystem:
    """Создает подключение к HDFS"""
    ticket = get_kerberos_ticket()

    fs = pyarrow.fs.HadoopFileSystem(node_path, kerb_ticket=ticket, user=user, extra_conf={'hadoop.security.authentication':'kerberos'})

    return fs


def _download_file(
    file_info: pa.fs.FileInfo, 
    fs: pa.fs.FileSystem, 
    local_root: Path,
    hdfs_root: str
):
    """Внутренняя функция для загрузки одного файла из HDFS

    Args:
        file_info: Информация о файле из pyarrow.fs
        fs: Объект файловой системы HDFS
        local_root: Локальная директория, куда сохранять файл
        hdfs_root: HDFS корневая директория, для вычисления относительного пути
    """
    try:
        hdfs_path = file_info.path

        clean_hdfs_root = hdfs_root.lstrip('/')
        clean_file_path = hdfs_path.lstrip('/')

        rel_path = Path(clean_file_path).relative_to(clean_hdfs_root)

        local_file_path = local_root / rel_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        with fs.open_input_file(hdfs_path) as f_src:
            with open(local_file_path, 'wb') as f_dst:
                f_dst.write(f_src.read())
    except Exception as e:
        logger.error(f"Не удалось загрузить {file_info.path}: {e}")
        raise e


def download_table_from_hdfs(table_name: str, hdfs_base_path: str, local_base_path: str, n_workers: int, file_extension: str = 'parquet'):
    """Основная функция для скачивания таблицы (папки) из HDFS

    Args:
        table_name: Имя таблицы из HDFS
        hdfs_base_path: Базовый путь в HDFS (до таблицы)
        local_base_path: Куда сохранять локально
        n_workers: количество потоков
        file_extension: Фильтр по расширению
    """
    local_save_path = Path(local_base_path) / table_name

    if local_save_path.exists() and any(local_save_path.iterdir()):
        logger.info(f"Директория {local_save_path} уже существует. Пропускаем.")
        return

    logger.info(f"Начинаем загрузку {table_name}...")

    full_hdfs_path = os.path.join(hdfs_base_path, table_name)

    path_parts = full_hdfs_path.split('/')
    node = '/'.join(path_parts[:3])
    hdfs_root_relative = '/' + '/'.join(path_parts[3:])

    try:
        fs = get_hdfs_fs(node)
    except Exception as e:
        logger.error(f"Не удалось подключиться к HDFS: {e}")
        return

    try:
        file_selector = pyarrow.fs.FileSelector(hdfs_root_relative, recursive=True)
        all_files = fs.get_file_info(file_selector)

        # Фильтрация по расширению
        files_to_download = [f for f in all_files if f.is_file
                             and f.path.endswith(f'.{file_extension}')
                             and f.size > 0]

        logger.info(f"Нашлось {len(files_to_download)} файлов для загрузки.")
    except Exception as e:
        logger.error(f"Не удалось получить список файлов из {hdfs_root_relative}: {e}")
        return

    parent_of_table = str(Path(hdfs_root_relative).parent)
    download_func = partial(_download_file, fs=fs, local_root=Path(local_base_path), hdfs_root=parent_of_table)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(
            pool.map(download_func, files_to_download),
            total=len(files_to_download),
            desc=f"Загружаем {table_name}"
        ))
    logger.info(f"Загрузка завершена: {local_save_path}")


def send_email_with_results(
    omega_login: str,
    omega_pass: str = None,
    sender_email: str = "",
    recipients: List[str] = [], 
    subject: str = f"Omni experiments {str(datetime.now().date())}",
    file_path_to_attach: str = None
):
    if omega_pass is None:
        omega_pass = getpass("Введите пароль от Omega: ")

    msg = MIMEMultipart('mixed')
    msg['From'] = sender_email
    msg['To'] = ",".join(recipients)
    msg['Subject'] = subject

    if file_path_to_attach is not None:
        with open(file_path_to_attach, 'rb') as f:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(f.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename= {file_path_to_attach}')
        msg.attach(attachment)

    server = smtplib.SMTP("smtp.omega.sbrf.ru", 2525)
    server.starttls()
    server.login(omega_login, omega_pass)
    for r in recipients:
        server.sendmail(msg['From'], r, msg.as_string())
    server.close()