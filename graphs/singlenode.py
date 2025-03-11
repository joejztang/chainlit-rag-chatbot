"""Single node graph."""

from typing import Optional

from langchain.schema.runnable import RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, Graph, MessagesState, StateGraph

from .basegraph import BaseGraph

# TODO: fix relative import issue


class SingleNodeGraph(BaseGraph):
    """Single Node Graph."""

    def __init__(self, chain: Optional[RunnableSerializable] = None, mem: bool = False):
        self.graph = StateGraph(MessagesState)
        self.mem = mem
        self.chain = chain

    async def a_call_chain(self, state: MessagesState):
        """Call the chain."""
        ret = await self.chain.ainvoke(state["messages"][-1].content)
        return {"messages": [{"role": "assistant", "content": ret}]}

    def build(self):
        """Build the graph."""
        self.graph.add_node("chain", self.a_call_chain)
        self.graph.add_edge(START, "chain")

    def get_graph(self) -> Graph:
        """Get the graph."""
        if not self.chain:
            raise ValueError("Chain not set.")

        self.build()
        if self.mem:
            return self.graph.compile(checkpointer=MemorySaver())
        return self.graph.compile()
