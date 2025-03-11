import shutil
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory

from models.model import InMemoryHistory

LOCAL_STORAGE_PREFIX = "./pdfs"


def upload_file_to_blob(file: Any, path):
    raise NotImplementedError("This function is not implemented yet.")


def save_file_to_disk(from_: str, to_: str):
    shutil.copy2(from_, to_)


def get_by_session_id(session_id: str, store: dict) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]
