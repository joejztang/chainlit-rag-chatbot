# sqlite3 version bug https://github.com/chroma-core/chroma/issues/1985#issuecomment-2055963683
# async pysqlite https://stackoverflow.com/a/71756224/4308025
# __import__("aiosqlite")
# import sys

# sys.modules["sqlite3"] = sys.modules.pop("aiosqlite")

import os
from pathlib import Path
from typing import List

import aiosqlite
import chainlit as cl
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.indexes import SQLRecordManager, aindex, index
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import Runnable, RunnableConfig, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.ext.asyncio import create_async_engine

from utils.store import PostgresByteStore

chunk_size = 1024
chunk_overlap = 50

load_dotenv()

embeddings_model = OpenAIEmbeddings()

PDF_STORAGE_PATH = "./pdfs"


async def process_pdfs(pdf_storage_path: str, collection_name: str):
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

    a_pgvector_engine = create_async_engine(os.getenv("POSTGRES_CONNECTION_STRING"))
    doc_search = await PGVector.afrom_documents(
        docs,
        embeddings_model,
        collection_name=collection_name,
        connection=a_pgvector_engine,
        use_jsonb=True,
    )

    namespace = "pgvector/my_documents"
    a_record_engine = create_async_engine(
        "sqlite+aiosqlite:///record_manager_cache.sql"
    )
    record_manager = SQLRecordManager(namespace, engine=a_record_engine)
    await record_manager.acreate_schema()

    index_result = await aindex(
        docs,
        record_manager,
        doc_search,
        cleanup="full",
        # TODO: source id key likely wrong so that error finding index
        source_id_key="source",
    )
    print(f"Indexing stats: {index_result}")

    # store = PostgresByteStore(os.getenv("POSTGRES_CONNECTION_STRING"), collection_name)
    # id_key = "doc_id"

    # retriever = MultiVectorRetriever(
    #     vectorstore=doc_search,
    #     docstore=store,
    #     id_key=id_key,
    # )

    return doc_search

    # doc_search = Chroma.from_documents(docs, embeddings_model)

    # namespace = "chromadb/my_documents"
    # record_manager = SQLRecordManager(
    #     namespace, db_url="sqlite:///record_manager_cache.sql"
    # )
    # record_manager.create_schema()

    # index_result = index(
    #     docs,
    #     record_manager,
    #     vectorstore,
    #     cleanup="incremental",
    #     source_id_key="source",
    # )

    # print(f"Indexing stats: {index_result}")

    # return vectorstore


model = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    doc_search = await process_pdfs(PDF_STORAGE_PATH, "my_documents")

    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever()

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    # # TODO: how to handle doc upload
    # print(len(message.elements))
    # if message.elements:
    #     await cl.Message(content=f"Received {message.elements[0]} image(s)").send()
    #     await cl.Message(content=f"Received {len(message.elements)} doc(s)").send()
    #     return

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata["source"], d.metadata["page"])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join(
                    [f"{source}#page={page}" for source, page in self.sources]
                )
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    # TODO: pgvector async stream available?
    # async for chunk in runnable.astream(
    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)]
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()
