# graph.py
# ─────────────────────────────────────────────────────────────────────────────
# Assembles the compiled LangGraph StateGraph for the Regulatory Intelligence
# Agent. This file is the single source of truth for graph topology — all
# node registrations, edge definitions, and compilation live here.
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    GRAPH TOPOLOGY (Full Picture)                        ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║                                                                          ║
# ║   ┌───────┐                                                              ║
# ║   │ START │                                                              ║
# ║   └───┬───┘                                                              ║
# ║       │ (always)                                                         ║
# ║       ▼                                                                  ║
# ║  ┌─────────────────┐   ◄─────────────────────────────────────────┐      ║
# ║  │ supervisor_node │ ◄──────────────────────────────────────┐    │      ║
# ║  └────────┬────────┘                                        │    │      ║
# ║           │                                                 │    │      ║
# ║    [conditional]  reads state["next_agent"]                 │    │      ║
# ║    ┌──────┴────────────────────────┐                        │    │      ║
# ║    │                               │                        │    │      ║
# ║    ▼ "policy_rag_node"             ▼ "data_analyst_node"    │    │      ║
# ║  ┌──────────────────┐   ┌──────────────────────┐           │    │      ║
# ║  │  policy_rag_node │   │  data_analyst_node   │           │    │      ║
# ║  └────────┬─────────┘   └──────────┬───────────┘           │    │      ║
# ║           │ (always)               │ (always)               │    │      ║
# ║           └──────────────┬─────────┘                        │    │      ║
# ║                          └──────────────────────────────────┘    │      ║
# ║                     [both loop back to supervisor]                │      ║
# ║                                                                   │      ║
# ║           ▼ "synthesis_node"  (from supervisor)                   │      ║
# ║  ┌──────────────────┐                                             │      ║
# ║  │  synthesis_node  │                                             │      ║
# ║  └────────┬─────────┘                                             │      ║
# ║           │                                                       │      ║
# ║    [conditional]  reads state["next_agent"]                       │      ║
# ║    ┌──────┴────────────────────────────┐                          │      ║
# ║    │ "data_analyst_node"               │ "report_writer_node"     │      ║
# ║    │  (A2A error loop)                 │  (success path)          │      ║
# ║    └──────────────────────────────────►┘                          │      ║
# ║                    │                  ▼                            │      ║
# ║          ┌─────────┘      ┌────────────────────────┐              │      ║
# ║          │ loops back ────►  report_writer_node    │              │      ║
# ║          │                └────────────┬───────────┘              │      ║
# ║          │                             │ (always)                  │      ║
# ║          │                             ▼                           │      ║
# ║          │                         ┌─────┐                         │      ║
# ║          │                         │ END │                         │      ║
# ║          │                         └─────┘                         │      ║
# ║          └───────────────── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘      ║
# ║           (data_analyst_node loops back to supervisor, not synthesis)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Key design principles:
#   1. supervisor_node is the SOLE entry point from START and the central hub
#      that all worker nodes report back to (except synthesis on success).
#   2. Conditional edges ONLY exist on supervisor_node and synthesis_node.
#      All other edges are deterministic (add_edge).
#   3. The MemorySaver checkpointer gives the graph short-term thread memory,
#      enabling multi-turn conversations and interrupted run resumption.
# ─────────────────────────────────────────────────────────────────────────────

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState

# ── Node functions ────────────────────────────────────────────────────────────
from nodes import (
    supervisor_node,
    policy_rag_node,
    data_analyst_node,
    synthesis_node,
    report_writer_node,
)

# ── Routing string constants ──────────────────────────────────────────────────
# These are the exact strings stored in AgentState.next_agent by nodes.py.
# They MUST match the names passed to workflow.add_node() below.
# Centralising them here and importing from nodes.py prevents silent
# routing failures caused by string typos.
from nodes import (
    ROUTE_POLICY_RAG,     # "policy_rag_node"
    ROUTE_DATA_ANALYST,   # "data_analyst_node"
    ROUTE_SYNTHESIS,      # "synthesis_node"
    ROUTE_REPORT_WRITER,  # "report_writer_node"
    ROUTE_END,            # "END" — safety catch for supervisor path_map
)

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Local constant — supervisor has no ROUTE_ export in nodes.py since it is
# never a *destination* written into next_agent by another node. It is only
# the *source* of conditional edges and the fixed START target.
# ─────────────────────────────────────────────────────────────────────────────
_SUPERVISOR_NODE = "supervisor_node"


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def route_supervisor(state: AgentState) -> str:
    """
    Routing function for the conditional edge leaving supervisor_node.

    Simply reads and returns state["next_agent"], which supervisor_node
    populates based on its state-inspection logic. LangGraph uses the
    returned string as a key into the path_map to resolve the destination node.

    Possible return values and their destinations:
      "policy_rag_node"   → policy_rag_node   (cold start / RAG retrieval)
      "data_analyst_node" → data_analyst_node  (initial SQL pass)
      "synthesis_node"    → synthesis_node     (A2A error path or success)
      "END"               → END                (safety catch — graph terminates)
    """
    destination = state.get("next_agent", "")
    logger.debug("route_supervisor | next_agent=%r", destination)
    return destination


def route_synthesis(state: AgentState) -> str:
    """
    Routing function for the conditional edge leaving synthesis_node.

    synthesis_node writes one of two values into next_agent:
      • ROUTE_DATA_ANALYST  → A2A error loop: route back to data_analyst_node
                              with a correction message in the conversation.
      • ROUTE_REPORT_WRITER → Success path: route to report_writer_node
                              to persist the completed Markdown report.

    Possible return values and their destinations:
      "data_analyst_node"  → data_analyst_node  (A2A retry with critique)
      "report_writer_node" → report_writer_node  (report persistence)
    """
    destination = state.get("next_agent", "")
    logger.debug("route_synthesis | next_agent=%r", destination)
    return destination


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

# Instantiate the StateGraph with our typed state schema.
# AgentState (TypedDict) serves as both the type annotation and the structural
# contract — LangGraph will merge each node's returned dict into this schema.
workflow = StateGraph(AgentState)

# ─────────────────────────────────────────────────────────────────────────────
# Node Registration
# ─────────────────────────────────────────────────────────────────────────────
workflow.add_node(_SUPERVISOR_NODE,    supervisor_node)     # Sync  — pure state logic
workflow.add_node(ROUTE_POLICY_RAG,    policy_rag_node)     # Sync  — RAG retrieval
workflow.add_node(ROUTE_DATA_ANALYST,  data_analyst_node)   # Async — LLM + MCP SQL tool
workflow.add_node(ROUTE_SYNTHESIS,     synthesis_node)      # Async — LLM critic / drafter
workflow.add_node(ROUTE_REPORT_WRITER, report_writer_node)  # Async — MCP file write

logger.info("Graph nodes registered: %d nodes", 5)

# ─────────────────────────────────────────────────────────────────────────────
# Edge Definitions
# ─────────────────────────────────────────────────────────────────────────────

# ── Entry point ───────────────────────────────────────────────────────────────
workflow.add_edge(START, _SUPERVISOR_NODE)

# ── Deterministic (unconditional) edges ───────────────────────────────────────
workflow.add_edge(ROUTE_POLICY_RAG,    _SUPERVISOR_NODE)
workflow.add_edge(ROUTE_DATA_ANALYST,  _SUPERVISOR_NODE)
workflow.add_edge(ROUTE_REPORT_WRITER, END)

# ── Conditional edge: supervisor_node ─────────────────────────────────────────
workflow.add_conditional_edges(
    _SUPERVISOR_NODE,
    route_supervisor,
    {
        ROUTE_POLICY_RAG:   ROUTE_POLICY_RAG,    # Cold start → RAG retrieval
        ROUTE_DATA_ANALYST: ROUTE_DATA_ANALYST,  # RAG done   → SQL analysis
        ROUTE_SYNTHESIS:    ROUTE_SYNTHESIS,     # Both ready → synthesise
        ROUTE_END:          END,                 # Safety catch → terminate
    },
)

# ── Conditional edge: synthesis_node ──────────────────────────────────────────
workflow.add_conditional_edges(
    ROUTE_SYNTHESIS,
    route_synthesis,
    {
        ROUTE_DATA_ANALYST:  ROUTE_DATA_ANALYST,   # A2A retry loop
        ROUTE_REPORT_WRITER: ROUTE_REPORT_WRITER,  # Report persistence
    },
)

logger.info(
    "Graph edges defined | standard=%d | conditional=%d",
    4,  # START→supervisor, policy_rag→supervisor, data_analyst→supervisor, report_writer→END
    2,  # supervisor conditional, synthesis conditional
)

# ─────────────────────────────────────────────────────────────────────────────
# Compilation
# ─────────────────────────────────────────────────────────────────────────────
memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

logger.info("Regulatory Intelligence Agent graph compiled successfully ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Public export
# ─────────────────────────────────────────────────────────────────────────────
__all__ = ["app"]


# ─────────────────────────────────────────────────────────────────────────────
# Developer utilities (only executed when this file is run directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage

    print("\n" + "─" * 72)
    print("GRAPH TOPOLOGY — Mermaid Diagram")
    print("─" * 72)
    print(app.get_graph().draw_mermaid())

    async def smoke_test() -> None:
        print("\n" + "─" * 72)
        print("SMOKE TEST — Streaming graph execution")
        print("─" * 72)

        run_config = {
            "configurable": {
                "thread_id": "smoke-test-001",
            }
        }

        # ── FIX 2: a2a_iterations added to initial_state ──────────────────────
        # Now that AgentState includes a2a_iterations, it must be initialised
        # here to 0. Without this, MemorySaver may raise a validation error on
        # first checkpoint because the key is missing from the state snapshot.
        initial_state = {
            "messages": [HumanMessage(
                content=(
                    "Analyse our startup's compliance posture against RBI's cash reserve "
                    "requirements for entities handling foreign remittances. Retrieve the "
                    "latest metrics and flag any breaches."
                )
            )],
            "rag_context":    "",
            "sql_context":    "",
            "sql_error":      "",
            "next_agent":     "",
            "a2a_iterations": 0,   # ← NEW: must match AgentState TypedDict
        }

        async for event in app.astream(initial_state, config=run_config):
            for node_name, node_output in event.items():
                updated_keys = list(node_output.keys())
                print(f"  [{node_name}] → updated keys: {updated_keys}")

                if "messages" in node_output and node_output["messages"]:
                    last_msg = node_output["messages"][-1]
                    preview  = str(getattr(last_msg, "content", last_msg))[:160]
                    print(f"    └─ message preview: {preview!r}")

        print("\n✓ Smoke test complete.")

    asyncio.run(smoke_test())