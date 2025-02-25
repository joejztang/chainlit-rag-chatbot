import os

import chainlit as cl
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableConfig, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from utils.db import a_pgvector_engine
from utils.handler import PostMessageHandler
from utils.processor import _cleanup, process_pdfs
from utils.util import save_file_to_disk

load_dotenv()
USER_ID = os.getenv("USER_ID", "test")
PDF_STORAGE_PATH = "./pdfs"

embeddings_model = OpenAIEmbeddings()
model = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)
vectordb = PGVector(embeddings=embeddings_model, connection=a_pgvector_engine)


@cl.on_chat_start
async def on_chat_start():
    """Prepare for chat."""
    template = """Quesition: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    cl.user_session.set("prompt", prompt)

    file_template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    file_prompt = ChatPromptTemplate.from_template(file_template)
    cl.user_session.set("file_prompt", file_prompt)

    vectordb = PGVector(
        embeddings_model,
        collection_name=f"{USER_ID}/{cl.user_session.get("id")}",
        connection=a_pgvector_engine,
        use_jsonb=True,
    )
    await vectordb.__apost_init__()
    cl.user_session.set("vectordb", vectordb)

    cl.user_session.set("uid", USER_ID)


@cl.on_message
async def on_message(message: cl.Message):
    """Logic for handling the user's message.

    Args:
        message (cl.Message): Response to the user's message.
    """

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    file_prompt = cl.user_session.get("file_prompt")  # type: Runnable
    prompt = cl.user_session.get("prompt")
    uid = cl.user_session.get("uid")

    msg = cl.Message(content="")

    runnable = {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()
    if message.elements:
        print(message.elements)

        collections = set()
        for element in message.elements:
            await cl.make_async(save_file_to_disk)(
                element.path, os.path.join(PDF_STORAGE_PATH, element.name)
            )
            collections.add(os.path.join(uid, element.name))
        cl.user_session.set("collections", collections)

        vectordb = cl.user_session.get("vectordb")
        await process_pdfs(
            uid,
            message.elements,
            vectordb,
        )

        retriever = vectordb.as_retriever()
        runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | file_prompt
            | model
            | StrOutputParser()
        )

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)]
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()


@cl.on_chat_end
async def on_chat_end():
    """Delete the index and vectors."""
    uid = cl.user_session.get("uid")
    vectordb = cl.user_session.get("vectordb")
    await _cleanup(uid, vectordb)
    print("Done deleting index and vectors.")
