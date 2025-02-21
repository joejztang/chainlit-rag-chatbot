import os

import chainlit as cl
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableConfig, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from snowflake import SnowflakeGenerator

from utils.handler import PostMessageHandler
from utils.processor import process_pdfs
from utils.util import save_file_to_disk

load_dotenv()
MACHINE_ID = os.getenv("MACHINE_ID", 1)
PDF_STORAGE_PATH = "./pdfs"

sfgen = SnowflakeGenerator(MACHINE_ID)
embeddings_model = OpenAIEmbeddings()
model = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)
# doc_search = asyncio.run(
#     process_pdfs(PDF_STORAGE_PATH, "my_documents", embeddings_model)
# )


@cl.on_chat_start
async def on_chat_start():
    """Set the prompt for the user."""
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    sid = next(sfgen)
    cl.user_session.set("prompt", prompt)
    cl.user_session.set("sid", sid)


@cl.on_message
async def on_message(message: cl.Message):
    """Logic for handling the user's message.

    Args:
        message (cl.Message): Response to the user's message.
    """

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    prompt = cl.user_session.get("prompt")  # type: Runnable

    msg = cl.Message(content="")

    if message.elements:
        print(message.elements)
        for element in message.elements:
            await cl.make_async(save_file_to_disk)(
                element.path, os.path.join(PDF_STORAGE_PATH, element.name)
            )
        doc_search = await process_pdfs(
            PDF_STORAGE_PATH, "my_documents", embeddings_model
        )
        retriever = doc_search.as_retriever()
        runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
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
    else:
        await cl.Message("no file uploaded").send()


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")
