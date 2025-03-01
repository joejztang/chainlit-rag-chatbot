"""Base class for graphs."""

from abc import ABC, abstractmethod

from langgraph.graph import Graph


class BaseGraph(ABC):
    @abstractmethod
    def get_graph(self, *args, **kwargs) -> Graph:
        pass
