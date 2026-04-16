# state.py
# ─────────────────────────────────────────────────────────────────────────────
# Defines the shared LangGraph state schema for the Regulatory Intelligence Agent.
#
# AgentState is the single source of truth that flows between every node in
# the graph. Each key is explicitly typed to enforce data contracts between
# agents and to support LangGraph's reducer/merge mechanics.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The canonical state object passed between all nodes in the LangGraph graph.

    Fields
    ------
    messages : Annotated[list[BaseMessage], add_messages]
        The full conversation history (HumanMessage, AIMessage, ToolMessage, etc.).
        The ``add_messages`` reducer *appends* new messages rather than replacing
        the list, ensuring the entire context window is preserved across nodes.

    rag_context : str
        Raw policy text retrieved from ChromaDB (the RBI/SEBI circular store).
        Populated by the RAG agent and consumed by the analyst/report-writer agent.
        Defaults to an empty string; an empty value signals no relevant circulars
        were found for the current query.

    sql_context : str
        Stringified query results from the SQLite startup-metrics database,
        retrieved via the Model Context Protocol (MCP) tool.
        Populated by the SQL agent and consumed by the analyst/report-writer agent.

    sql_error : str
        Error message emitted when an SQLite query fails (e.g., malformed SQL,
        missing table). This key powers the A2A (Agent-to-Agent) feedback loop:
        the supervisor reads this field and re-routes control back to the SQL
        agent with a corrective prompt, enabling self-healing query retries.
        An empty string indicates the last query succeeded.

    next_agent : str
        Routing directive written by the supervisor node to indicate which agent
        node should execute next. The supervisor's conditional edge reads this
        value to dispatch control. Possible values should be defined as string
        constants in the supervisor module (e.g., "rag_agent", "sql_agent",
        "analyst_agent", "report_writer", "END").
    """

    messages: Annotated[list[BaseMessage], add_messages]
    rag_context: str
    sql_context: str
    sql_error: str
    next_agent: str
    a2a_iterations: int