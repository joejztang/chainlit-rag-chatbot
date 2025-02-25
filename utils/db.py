import asyncio
import os

from langchain.indexes import SQLRecordManager
from sqlalchemy.ext.asyncio import create_async_engine

a_pgvector_engine = create_async_engine(os.getenv("PGVECTOR_CONNECTION_STRING"))
a_record_engine = create_async_engine(os.getenv("RECORDMANAGER_CONNECTION_STRING"))


class LocalRecordManager:
    """Singleton class for managing local record managers.

    Returns:
        SQLRecordmanager: return the instance of the SQLRecordManager.
    """

    _instance = dict()

    def __new__(cls, namespace: str) -> SQLRecordManager:
        if cls._instance.get(namespace, None) is None:
            sqlRecordManager = SQLRecordManager(namespace, engine=a_record_engine)
            asyncio.run(sqlRecordManager.acreate_schema())
            cls._instance[namespace] = sqlRecordManager

        return cls._instance[namespace]
