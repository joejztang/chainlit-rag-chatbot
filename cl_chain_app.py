import os
from functools import partial
from operator import itemgetter

import chainlit as cl
from chainlit.input_widget import Switch
from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from utils.db import LocalRecordManager, a_pgvector_engine
from utils.handler import PostMessageHandler
from utils.processor import _cleanup, process_pdfs
from utils.prompt import file_prompt, file_with_history_prompt, simple_prompt
from utils.util import get_by_session_id

load_dotenv()
USER_ID = os.getenv("USER_ID", "test")

embeddings_model = OpenAIEmbeddings()
model = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    """Prepare for chat."""
    vectordb = PGVector(
        embeddings_model,
        collection_name=f"{USER_ID}/{cl.user_session.get("id")}",
        connection=a_pgvector_engine,
        use_jsonb=True,
    )
    await vectordb.__apost_init__()
    cl.user_session.set("vectordb", vectordb)

    recordmanager = LocalRecordManager(f"{USER_ID}/{cl.user_session.get("id")}")
    cl.user_session.set("recordmanager", recordmanager)

    cl.user_session.set("uid", USER_ID)
    cl.user_session.set("store", {})
    settings = await cl.ChatSettings(
        [
            Switch(
                id="sticky_rag",
                label="Chat about documents within the whole session",
                initial=False,
            ),
        ]
    ).send()
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


@cl.on_message
async def on_message(message: cl.Message):
    """Logic for handling the user's message.

    Args:
        message (cl.Message): Response to the user's message.
    """

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # file_prompt = cl.user_session.get("file_prompt")  # type: Runnable
    # prompt = cl.user_session.get("prompt")
    uid = cl.user_session.get("uid")

    msg = cl.Message(content="")

    chain = (
        {"question": RunnablePassthrough()} | simple_prompt | model | StrOutputParser()
    )
    if message.elements or cl.user_session.get("settings").get("sticky_rag"):

        vectordb = cl.user_session.get("vectordb")
        recordmanager = cl.user_session.get("recordmanager")

        if message.elements:
            await process_pdfs(uid, message.elements, vectordb, recordmanager)

        retriever = vectordb.as_retriever()
        context = itemgetter("question") | retriever | format_docs
        first_step = RunnablePassthrough.assign(context=context)
        chain = first_step | file_with_history_prompt | model | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        partial(get_by_session_id, store=cl.user_session.get("store")),
        input_messages_key="question",
        history_messages_key="history",
    )

    async for chunk in chain_with_history.astream(
        {"question": message.content},
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()


@cl.on_chat_end
async def on_chat_end():
    """Delete the index and vectors."""
    uid = cl.user_session.get("uid")
    vectordb = cl.user_session.get("vectordb")
    recordmanager = cl.user_session.get("recordmanager")
    await _cleanup(uid, vectordb, recordmanager)
    # del cl.user_session["store"]
    print("Done deleting index and vectors.")
