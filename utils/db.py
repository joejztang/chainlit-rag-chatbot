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
            try:
                asyncio.run(sqlRecordManager.acreate_schema())
            except Exception as e:
                # schema already exists, avoid crashing.
                print(f"failed to create schema, schema exists.\n{e}")
            cls._instance[namespace] = sqlRecordManager

        return cls._instance[namespace]
