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
    pdf_storage_path: str,
    collection_name: str,
    embeddings_model: Any,
) -> PGVector:
    pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for pdf_path in pdf_directory.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)
        for doc in docs:
            doc.metadata["source"] = pdf_path.stem

    doc_search = await PGVector.afrom_documents(
        docs,
        embeddings_model,
        collection_name=collection_name,
        connection=a_pgvector_engine,
        use_jsonb=True,
    )

    namespace = "pgvector/my_documents"
    record_manager = LocalRecordManager(namespace)
    # record_manager = SQLRecordManager(namespace, engine=a_record_engine)
    # await record_manager.acreate_schema()

    index_result = await aindex(
        docs,
        record_manager,
        doc_search,
        cleanup="full",
        source_id_key="source",
    )
    print(f"Indexing stats: {index_result}")

    return doc_search
