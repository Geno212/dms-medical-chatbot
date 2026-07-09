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

import sqlite3

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import Config, get_config
from ..db import Repository, get_repository
from ..llm import ChatLLM, EmbeddingClient, OpenAICompatLLM
from ..logging_setup import configure_logging, get_logger
from ..vectorstore import ProtocolRetriever
from .nodes import ChatbotEngine
from .state import ChatState

log = get_logger()


def _make_checkpointer(config: Config):
    """Durable conversation memory: LangGraph state (transcript + clinical
    context) is checkpointed so a chat thread picks up exactly where it left
    off even across process restarts. The checkpoint store follows the data
    backend: Supabase/Postgres when DB_BACKEND=postgres, SQLite otherwise.
    Falls back gracefully (Postgres -> SQLite -> memory) so a checkpointing
    problem can never take the chatbot down."""
    if config.checkpoint_db == ":memory:":
        return MemorySaver()
    if config.db_backend == "postgres":
        try:
            from psycopg import Connection
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = Connection.connect(config.database_url, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()  # idempotent: creates checkpoint tables on first run
            log.info("conversation checkpoints -> Postgres (same database as hospital data)")
            return saver
        except Exception as exc:
            log.warning("Postgres checkpointer unavailable (%s: %s) -> falling back to SQLite", type(exc).__name__, exc)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        log.warning("langgraph-checkpoint-sqlite not installed -> conversations are in-memory only")
        return MemorySaver()
    conn = sqlite3.connect(config.checkpoint_db, check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(
    repo: Repository | None = None,
    llm: ChatLLM | None = None,
    embedder: EmbeddingClient | None = None,
    config: Config | None = None,
):
    config = config or get_config()
    configure_logging(config)
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

    return graph.compile(checkpointer=_make_checkpointer(config))


def chat_turn(compiled_graph, thread_id: str, user_message: str) -> dict:
    """Run one conversation turn; returns the structured response dict."""
    result = compiled_graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["response"]
