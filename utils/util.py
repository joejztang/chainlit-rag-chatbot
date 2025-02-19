import shutil
from typing import Any

LOCAL_STORAGE_PREFIX = "./pdfs"


def upload_file_to_blob(file: Any, path):
    raise NotImplementedError("This function is not implemented yet.")


def save_file_to_disk(from_: str, to_: str):
    shutil.copy2(from_, to_)
