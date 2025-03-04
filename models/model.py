from typing import TypedDict

from pydantic import BaseModel


class SimpleState(TypedDict):
    question: str

    def __str__(self):
        return self.question
