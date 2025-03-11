import os
from operator import itemgetter

import chainlit as cl
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableConfig, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from graphs.singlenode import SingleNodeGraph
from utils.db import LocalRecordManager, a_pgvector_engine
from utils.handler import PostMessageHandler
from utils.processor import _cleanup, process_pdfs

load_dotenv()
USER_ID = os.getenv("USER_ID", "test")

embeddings_model = OpenAIEmbeddings()
model = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)
# vectordb = PGVector(embeddings=embeddings_model, connection=a_pgvector_engine)


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

    recordmanager = LocalRecordManager(f"{USER_ID}/{cl.user_session.get("id")}")
    cl.user_session.set("recordmanager", recordmanager)

    cl.user_session.set("uid", USER_ID)


# This construct a graph.
@cl.on_message
async def graph_on_message(message: cl.Message):
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

    chain = {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()
    if message.elements:
        print(message.elements)

        vectordb = cl.user_session.get("vectordb")
        recordmanager = cl.user_session.get("recordmanager")
        await process_pdfs(uid, message.elements, vectordb, recordmanager)

        retriever = vectordb.as_retriever()
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | file_prompt
            | model
            | StrOutputParser()
        )

    # TODO: currently only support one pdf chatting with memory. supprot multiple pdfs later.
    if not cl.user_session.get("graph"):
        graph = SingleNodeGraph(chain=chain, mem=True).get_graph()
        cl.user_session.set("graph", graph)
    graph = cl.user_session.get("graph")
    # print(chain.input_schema.model_json_schema())

    async for chunk in graph.astream(
        # message.content,
        {"messages": [{"role": "user", "content": message.content}]},
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)],
            configurable=dict(thread_id=cl.user_session.get("id")),
        ),
        # stream_mode="messages",
    ):
        # print(chunk)
        # print(chunk["chain"]["messages"][-1]["content"])
        # tok = chunk["chain"]["messages"][-1].content
        # print(tok)
        await msg.stream_token(chunk["chain"]["messages"][-1]["content"])

    await msg.send()


@cl.on_chat_end
async def on_chat_end():
    """Delete the index and vectors."""
    uid = cl.user_session.get("uid")
    vectordb = cl.user_session.get("vectordb")
    recordmanager = cl.user_session.get("recordmanager")
    await _cleanup(uid, vectordb, recordmanager)
    print("Done deleting index and vectors.")
