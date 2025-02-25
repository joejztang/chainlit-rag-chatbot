import os
from pathlib import Path
from typing import Any, List

from langchain.indexes import aindex
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_postgres.vectorstores import PGVector

from utils.db import LocalRecordManager, a_pgvector_engine

chunk_size = 1024
chunk_overlap = 50


async def process_pdfs(
    uid: str,
    pdfs: List[Any],
    vectordb: Any,
) -> PGVector:
    # pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for element in pdfs:
        loader = PyMuPDFLoader(element.path)
        documents = loader.load()
        docs += text_splitter.split_documents(documents)
        # "source" will be used as group_id in upsertion_record table
        for doc in docs:
            doc.metadata["source"] = os.path.join(uid, element.name)

    record_manager = LocalRecordManager(uid)

    index_result = await aindex(
        docs,
        record_manager,
        vectordb,
        cleanup="incremental",
        source_id_key="source",
    )
    print(f"Indexing stats: {index_result}")


async def _cleanup(uid: str, vectordb: PGVector):
    record_manager = LocalRecordManager(uid)
    await aindex(
        [],
        record_manager,
        vectordb,
        cleanup="full",
    )
