"""Assemble the LangGraph workflow.

        ┌─────────┐
        │ router  │  classify intent + detect language
        └────┬────┘
   ┌─────────┼──────────┐
   ▼         ▼          ▼
 medical   action     other
 (RAG)   (verified    (persona
          payloads)    smalltalk)
   └─────────┼──────────┘
             ▼
            END

State is checkpointed per thread (MemorySaver), so multi-turn context —
including the clinical context used for booking slot-filling — persists
across turns of the same session.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import Config, get_config
from ..db import Repository, get_repository
from ..llm import ChatLLM, EmbeddingClient, OpenAICompatLLM
from ..vectorstore import ProtocolRetriever
from .nodes import ChatbotEngine
from .state import ChatState


def build_graph(
    repo: Repository | None = None,
    llm: ChatLLM | None = None,
    embedder: EmbeddingClient | None = None,
    config: Config | None = None,
):
    config = config or get_config()
    repo = repo or get_repository(config)
    llm = llm or OpenAICompatLLM(config)
    embedder = embedder if embedder is not None else EmbeddingClient(config)
    retriever = ProtocolRetriever(repo, embedder)
    engine = ChatbotEngine(repo, retriever, llm, config)

    graph = StateGraph(ChatState)
    graph.add_node("router", engine.router_node)
    graph.add_node("medical", engine.medical_node)
    graph.add_node("action", engine.action_node)
    graph.add_node("other", engine.other_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {"medical": "medical", "action": "action", "other": "other"},
    )
    graph.add_edge("medical", END)
    graph.add_edge("action", END)
    graph.add_edge("other", END)

    return graph.compile(checkpointer=MemorySaver())


def chat_turn(compiled_graph, thread_id: str, user_message: str) -> dict:
    """Run one conversation turn; returns the structured response dict."""
    result = compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["response"]
