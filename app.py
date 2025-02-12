import asyncio
import os
from pathlib import Path
from typing import List

import chainlit as cl
from chainlit import run_sync
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.indexes import SQLRecordManager, aindex
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import Runnable, RunnableConfig, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.ext.asyncio import create_async_engine

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
        source_id_key="source",
    )
    print(f"Indexing stats: {index_result}")

    return doc_search


model = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)

doc_search = asyncio.run(process_pdfs(PDF_STORAGE_PATH, "my_documents"))


@cl.on_chat_start
async def on_chat_start():

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

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)]
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()
