from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# simple_template = """Quesition: {question}"""
# file_template = """Answer the question based only on the following context:

# {context}

# Question: {question}
# """

simple_prompt = ChatPromptTemplate.from_template("""Quesition: {question}""")

file_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question based only on the following context:\n{context}\n",
        ),
        ("human", "{question}"),
    ]
)

file_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question based only on the following context:\n{context}\n",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
