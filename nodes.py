# nodes.py
# ─────────────────────────────────────────────────────────────────────────────
# All LangGraph node functions for the Regulatory Intelligence Agent.
#
# Node execution order (happy path):
#   supervisor_node
#       └─► policy_rag_node
#               └─► supervisor_node
#                       └─► data_analyst_node  ◄─────────────────────┐
#                               └─► supervisor_node                  │
#                                       └─► synthesis_node           │
#                                               ├─(A2A error)────────┘
#                                               └─(success)─► report_writer_node
#
# Node types:
#   Sync  : supervisor_node, policy_rag_node  (no I/O — pure state inspection)
#   Async : data_analyst_node, synthesis_node, report_writer_node (LLM / tool I/O)
# ─────────────────────────────────────────────────────────────────────────────

import logging
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import LLM
from state import AgentState
from mcp_tools import execute_local_sql, write_markdown_report

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Routing sentinel constants
# These must exactly match the node names registered in graph.py's add_node()
# calls. Centralising them here prevents silent routing failures from typos.
# ─────────────────────────────────────────────────────────────────────────────
ROUTE_POLICY_RAG    = "policy_rag_node"
ROUTE_DATA_ANALYST  = "data_analyst_node"
ROUTE_SYNTHESIS     = "synthesis_node"
ROUTE_REPORT_WRITER = "report_writer_node"
ROUTE_END           = "END"

# ─────────────────────────────────────────────────────────────────────────────
# TPM / Context-size guard constants
#
# Groq free-tier limits:
#   llama-3.3-70b-versatile : 12 000 TPM
#   llama-3.1-8b-instant    :  6 000 TPM
#
# To stay safely under both limits we:
#   1. Keep only the FIRST message (original user query) + last 3 messages
#      from state["messages"] before every LLM call.
#   2. Hard-truncate rag_context and sql_context to MAX_CONTEXT_CHARS each
#      before injecting them into the synthesis system prompt.
# ─────────────────────────────────────────────────────────────────────────────
_MAX_HISTORY_MESSAGES = 3    # keep first + last N from state["messages"]
_MAX_CONTEXT_CHARS    = 1200 # ~300 tokens per context block
_MAX_A2A_ITERATIONS   = 2    # hard cap on A2A correction loop retries


def _trim_messages(messages: list, keep_last: int = _MAX_HISTORY_MESSAGES) -> list:
    """
    Returns a trimmed copy of the message list to control LLM context size.

    Strategy:
      - Always preserve messages[0] (the original HumanMessage / user query).
      - Keep only the last `keep_last` messages from the remainder.
      - This prevents the A2A correction loop from inflating the context
        window across retries and hitting Groq's TPM rate limit (413).

    Example (keep_last=3, 8 messages in history):
      IN : [H, A, A, A, A, A, A, A]   (H = original query, A = agent msgs)
      OUT: [H, A, A, A]               (first + last 3)
    """
    if len(messages) <= keep_last + 1:
        return list(messages)
    return [messages[0]] + list(messages[-(keep_last):])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

# ── Data Analyst system prompt ────────────────────────────────────────────────
_SQL_ANALYST_SYSTEM_TEMPLATE = """You are an expert SQL analyst for an Indian startup regulatory intelligence platform.

Your task is to query a local SQLite database to retrieve relevant startup financial and operational data.

DATABASE SCHEMA:
  Table : startup_metrics
  Columns:
    - startup_name           TEXT    : Name of the startup entity
    - revenue_inr            REAL    : Monthly revenue in Indian Rupees
    - burn_rate_inr          REAL    : Monthly operational burn rate in INR
    - headcount              INTEGER : Total number of employees
    - foreign_remittance_usd REAL    : Monthly inward foreign remittances in USD
    - cash_reserve_ratio     REAL    : Current cash reserve as a decimal (e.g., 0.35 = 35%)
    - month                  TEXT    : Reporting month in YYYY-MM format

INSTRUCTIONS:
  - Use the `execute_local_sql` tool to run your query.
  - Generate ONLY a valid SQLite SELECT statement.
  - Retrieve ALL relevant columns — the synthesis node will need the full picture.
  - Do NOT use markdown code blocks in your query; pass raw SQL only.

{error_context}"""

# ── A2A error context injected when a previous SQL attempt failed ─────────────
_SQL_ERROR_CONTEXT_TEMPLATE = """
PREVIOUS QUERY FAILED — CORRECTION REQUIRED:

Error: {sql_error}

Fix guide:
  1. "no such table"  → run: SELECT name FROM sqlite_master WHERE type='table';
  2. "no such column" → run: PRAGMA table_info(startup_metrics);
  3. Syntax error     → correct the SQL and retry.

Execute the corrected query now via the execute_local_sql tool.
"""

# ── Synthesis system prompt (success path) ────────────────────────────────────
# NOTE: kept intentionally short to avoid hitting TPM limits.
# The actual context is injected via {rag_context} and {sql_context},
# both of which are truncated to _MAX_CONTEXT_CHARS before formatting.
_SYNTHESIS_SYSTEM_TEMPLATE = """You are a Senior Regulatory Intelligence Analyst specialising in Indian startup compliance (RBI, SEBI, FEMA).

REGULATORY CONTEXT:
{rag_context}

STARTUP METRICS:
{sql_context}

Write a concise Markdown compliance report with:
  1. Executive Summary (2-3 sentences)
  2. Compliance Status Table (requirement | actual value | PASS/BREACH/WARNING)
  3. Risk Findings (bullet points, cite circular numbers)
  4. Recommendations (bullet points, specific and actionable)

Be precise. Cite circular numbers and exact metric values. Keep the total report under 600 words."""


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — supervisor_node  (SYNC)
# ─────────────────────────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    """
    Central orchestrator. Inspects the current state and writes `next_agent`
    to direct the conditional edges in the compiled LangGraph graph.

    Routing decision matrix:
    ┌──────────────┬──────────────┬───────────────┬──────────────────────────┐
    │  rag_context │  sql_context │   sql_error   │       Destination        │
    ├──────────────┼──────────────┼───────────────┼──────────────────────────┤
    │    empty     │    empty     │     any       │  policy_rag_node         │
    │  populated   │    empty     │    empty      │  data_analyst_node       │
    │  populated   │    empty     │  populated    │  synthesis_node (A2A ↑)  │
    │  populated   │  populated   │    empty      │  synthesis_node (✓)      │
    └──────────────┴──────────────┴───────────────┴──────────────────────────┘
    """
    rag_ctx = state.get("rag_context", "").strip()
    sql_ctx = state.get("sql_context", "").strip()
    sql_err = state.get("sql_error",   "").strip()

    logger.info(
        "supervisor_node | rag=%s | sql=%s | err=%s",
        bool(rag_ctx), bool(sql_ctx), bool(sql_err),
    )

    if not rag_ctx:
        logger.info("supervisor_node ──► %s  [cold start]", ROUTE_POLICY_RAG)
        return {"next_agent": ROUTE_POLICY_RAG}

    if rag_ctx and not sql_ctx and not sql_err:
        logger.info("supervisor_node ──► %s  [initial SQL pass]", ROUTE_DATA_ANALYST)
        return {"next_agent": ROUTE_DATA_ANALYST}

    if rag_ctx and sql_err and not sql_ctx:
        logger.info("supervisor_node ──► %s  [A2A error path]", ROUTE_SYNTHESIS)
        return {"next_agent": ROUTE_SYNTHESIS}

    if rag_ctx and sql_ctx and not sql_err:
        logger.info("supervisor_node ──► %s  [success path]", ROUTE_SYNTHESIS)
        return {"next_agent": ROUTE_SYNTHESIS}

    logger.warning(
        "supervisor_node | Unrecognised state — restarting pipeline. "
        "rag=%r | sql=%r | err=%r", bool(rag_ctx), bool(sql_ctx), bool(sql_err)
    )
    return {"next_agent": ROUTE_POLICY_RAG}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — policy_rag_node  (SYNC)
# ─────────────────────────────────────────────────────────────────────────────

def policy_rag_node(state: AgentState) -> dict:
    """
    Retrieves regulatory policy context relevant to the user's query.
    Currently a mock implementation — replace with real ChromaDB retrieval
    in production.
    """
    logger.info("policy_rag_node | Loading regulatory context (mock ChromaDB retrieval)")

    # Kept intentionally short to avoid contributing to TPM overruns.
    mock_rag_context = (
        "[RBI/2025-26/47]: Startups handling foreign remittances must maintain "
        "a minimum Cash Reserve Ratio (CRR) of 30% (0.30). Maximum allowed "
        "burn rate is 40% of liquid assets.\n"
        "[SEBI/HO/CFD/DIL2/CIR/2025/001]: Startups with headcount > 50 must "
        "file quarterly SMEG-Q disclosures. FDI limit capped at 74%."
    )

    logger.info(
        "policy_rag_node | Mock context loaded | length=%d chars",
        len(mock_rag_context),
    )

    return {
        "rag_context": mock_rag_context,
        "messages": [
            AIMessage(
                content=(
                    "✅ Regulatory retrieval complete. "
                    "Loaded 2 relevant circulars from the knowledge base:\n"
                    "  • RBI/2025-26/47  — Payment Aggregators & Cross-Border Remittances\n"
                    "  • SEBI/HO/CFD/DIL2/CIR/2025/001 — SME Emerge Disclosure Norms\n"
                    "Proceeding to startup metrics data retrieval."
                )
            )
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — data_analyst_node  (ASYNC)
# ─────────────────────────────────────────────────────────────────────────────

async def data_analyst_node(state: AgentState) -> dict:
    """
    SQL expert agent. Generates and executes a SQL query against the local
    SQLite startup-metrics database via the MCP execute_local_sql tool.

    FIX: Uses _trim_messages() before every LLM call to prevent the growing
    A2A correction history from exceeding Groq's TPM limit.
    """
    sql_error = state.get("sql_error", "").strip()

    # ── Build system prompt ───────────────────────────────────────────────────
    if sql_error:
        error_context = _SQL_ERROR_CONTEXT_TEMPLATE.format(sql_error=sql_error[:300])
        logger.info(
            "data_analyst_node | A2A retry — injecting error context | err=%.120s",
            sql_error,
        )
    else:
        error_context = (
            "No previous errors. Generate your best initial query "
            "to fetch all relevant metrics from startup_metrics."
        )

    system_prompt = _SQL_ANALYST_SYSTEM_TEMPLATE.format(error_context=error_context)

    # ── Bind the MCP SQL tool to the LLM ─────────────────────────────────────
    llm_with_sql_tool = LLM.bind_tools([execute_local_sql])

    # FIX: Trim message history before passing to LLM.
    # Without this, every A2A retry appends the full correction message
    # (~500-800 tokens) to the context, quickly blowing past Groq's TPM cap.
    trimmed_history = _trim_messages(state["messages"])

    messages_for_llm = [SystemMessage(content=system_prompt)] + trimmed_history

    logger.info(
        "data_analyst_node | Invoking LLM | history_depth=%d (trimmed from %d)",
        len(trimmed_history), len(state["messages"]),
    )

    # ── LLM invocation ────────────────────────────────────────────────────────
    response: AIMessage = await llm_with_sql_tool.ainvoke(messages_for_llm)

    logger.debug(
        "data_analyst_node | LLM response | has_tool_calls=%s",
        bool(response.tool_calls),
    )

    # ── Guard: LLM declined to call the tool ─────────────────────────────────
    if not response.tool_calls:
        logger.warning(
            "data_analyst_node | No tool_calls in response. Content: %.200s",
            response.content,
        )
        return {
            "messages":  [response],
            "sql_error": (
                "SQL_ERROR: LLM did not generate a SQL tool call. "
                f"Model response was: {response.content[:200]}"
            ),
        }

    # ── Extract the generated SQL query ──────────────────────────────────────
    tool_call      = response.tool_calls[0]
    sql_query: str = tool_call["args"].get("query", "").strip()

    if not sql_query:
        logger.error("data_analyst_node | tool_call present but 'query' arg is empty")
        return {
            "messages":  [response],
            "sql_error": (
                "SQL_ERROR: LLM generated a tool call but the query string was empty."
            ),
        }

    logger.info("data_analyst_node | Executing SQL | query=%.200s", sql_query)

    # ── Execute the MCP SQL tool ──────────────────────────────────────────────
    tool_output: str = await execute_local_sql.ainvoke({"query": sql_query})

    logger.debug(
        "data_analyst_node | Tool output (first 300 chars): %.300s", tool_output
    )

    # ── Route based on output sentinel ───────────────────────────────────────
    if tool_output.startswith("SQL_ERROR"):
        logger.warning(
            "data_analyst_node | SQL execution failed | error=%.200s", tool_output
        )
        return {
            "messages":  [response],
            "sql_error": tool_output,
        }

    logger.info(
        "data_analyst_node | SQL succeeded | output_length=%d chars",
        len(tool_output),
    )
    return {
        "messages":    [response],
        "sql_context": tool_output,
        "sql_error":   "",  # ← explicitly clear; critical for A2A loop exit
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — synthesis_node  (ASYNC)
# ─────────────────────────────────────────────────────────────────────────────

async def synthesis_node(state: AgentState) -> dict:
    """
    Dual-purpose node: A2A Critic on error, Report Drafter on success.

    PATH A — A2A Critic (sql_error populated):
      Hard-capped at _MAX_A2A_ITERATIONS. Generates a concise correction
      message and routes back to data_analyst_node.

    PATH B — Report Drafter (no error):
      Truncates rag_context and sql_context to _MAX_CONTEXT_CHARS each
      before injecting into the synthesis prompt, preventing TPM overruns.
    """
    sql_error      = state.get("sql_error",      "").strip()
    rag_context    = state.get("rag_context",    "").strip()
    sql_context    = state.get("sql_context",    "").strip()
    a2a_iterations = state.get("a2a_iterations", 0)

    # ─── PATH A: A2A Error Critique ───────────────────────────────────────────
    if sql_error:

        # ── Hard cap: stop the loop after _MAX_A2A_ITERATIONS ─────────────────
        if a2a_iterations >= _MAX_A2A_ITERATIONS:
            logger.warning(
                "synthesis_node | A2A cap reached (%d/%d) — routing to report_writer "
                "with partial context.",
                a2a_iterations, _MAX_A2A_ITERATIONS,
            )
            fallback_message = AIMessage(
                content=(
                    f"⚠️ **Data Retrieval Unavailable**\n\n"
                    f"The SQL data analyst failed after {a2a_iterations} correction "
                    f"attempts. Final error:\n```\n{sql_error}\n```\n\n"
                    f"Report will be generated using regulatory context only."
                )
            )
            return {
                "messages":       [fallback_message],
                "sql_error":      "",  # clear so PATH B runs on next synthesis call
                "a2a_iterations": a2a_iterations,
                "next_agent":     ROUTE_REPORT_WRITER,
            }

        logger.info(
            "synthesis_node | PATH A — A2A critique | iteration=%d/%d | error=%.120s",
            a2a_iterations + 1, _MAX_A2A_ITERATIONS, sql_error,
        )

        # Self-contained prompt — does NOT use state["messages"] so there
        # is no risk of history inflation hitting the TPM limit here.
        critique_messages = [
            SystemMessage(content=(
                "You are a Senior SQL Architect. A SQL query failed. "
                "Respond with ONLY: one sentence diagnosis + the corrected SQL query. "
                "No Python, no lists, no headers. Just diagnosis + SQL."
            )),
            HumanMessage(content=(
                f"Error:\n```\n{sql_error[:300]}\n```\n\n"
                f"Database: SQLite. Table: startup_metrics. "
                f"Columns: startup_name, revenue_inr, burn_rate_inr, headcount, "
                f"foreign_remittance_usd, cash_reserve_ratio, month.\n\n"
                f"Provide the corrected SQL query only."
            )),
        ]

        critique_response: AIMessage = await LLM.ainvoke(critique_messages)

        logger.info(
            "synthesis_node | Critique generated (%d chars) | routing ──► %s",
            len(critique_response.content), ROUTE_DATA_ANALYST,
        )

        correction_message = AIMessage(
            content=(
                f"🔄 **A2A Correction — Iteration {a2a_iterations + 1}/{_MAX_A2A_ITERATIONS}**\n\n"
                f"Error: `{sql_error[:200]}`\n\n"
                f"Corrected query:\n\n{critique_response.content}"
            )
        )

        return {
            "messages":       [correction_message],
            "a2a_iterations": a2a_iterations + 1,  # ← increment the counter
            "next_agent":     ROUTE_DATA_ANALYST,
        }

    # ─── PATH B: Report Drafting ──────────────────────────────────────────────
    logger.info(
        "synthesis_node | PATH B — Report synthesis "
        "| rag_len=%d | sql_len=%d | a2a_iterations=%d",
        len(rag_context), len(sql_context), a2a_iterations,
    )

    # Recover the original user question to anchor the report.
    original_query = "Provide a complete regulatory compliance assessment."
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            original_query = msg.content.strip()
            break

    # FIX: Truncate both contexts before injecting into the system prompt.
    # Full rag_context + sql_context + template can easily exceed 9k tokens
    # on Groq's free tier. Capping at _MAX_CONTEXT_CHARS each keeps the
    # total request well under the 6k TPM floor (llama-3.1-8b-instant).
    rag_trimmed = rag_context[:_MAX_CONTEXT_CHARS]
    sql_trimmed = sql_context[:_MAX_CONTEXT_CHARS]

    if len(rag_context) > _MAX_CONTEXT_CHARS:
        logger.warning(
            "synthesis_node | rag_context truncated %d → %d chars",
            len(rag_context), _MAX_CONTEXT_CHARS,
        )
    if len(sql_context) > _MAX_CONTEXT_CHARS:
        logger.warning(
            "synthesis_node | sql_context truncated %d → %d chars",
            len(sql_context), _MAX_CONTEXT_CHARS,
        )

    synthesis_messages = [
        SystemMessage(
            content=_SYNTHESIS_SYSTEM_TEMPLATE.format(
                rag_context=rag_trimmed,
                sql_context=sql_trimmed,
            )
        ),
        HumanMessage(
            content=(
                f"Generate the compliance report for this query:\n\n"
                f"**{original_query[:300]}**\n\n"
                f"Cite circular numbers and exact metric values. Keep it under 600 words."
            )
        ),
    ]

    logger.info("synthesis_node | Invoking LLM for report draft")
    report_draft: AIMessage = await LLM.ainvoke(synthesis_messages)

    logger.info(
        "synthesis_node | Report draft complete | length=%d chars | routing ──► %s",
        len(report_draft.content), ROUTE_REPORT_WRITER,
    )

    return {
        "messages":       [report_draft],
        "a2a_iterations": a2a_iterations,
        "next_agent":     ROUTE_REPORT_WRITER,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — report_writer_node  (ASYNC)
# ─────────────────────────────────────────────────────────────────────────────

async def report_writer_node(state: AgentState) -> dict:
    """
    Terminal persistence node. Writes the synthesised Markdown report to the
    WORKSPACE_DIR via the MCP Filesystem tool. No LLM call — pure file I/O.
    """
    DRAFT_MIN_LENGTH = 300
    report_content   = ""

    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and len(msg.content.strip()) >= DRAFT_MIN_LENGTH:
            report_content = msg.content.strip()
            break

    if not report_content:
        logger.error(
            "report_writer_node | No substantive AIMessage found (min_len=%d). "
            "History has %d entries.",
            DRAFT_MIN_LENGTH, len(state["messages"]),
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "WRITE_ERROR: No report draft found in message history. "
                        "synthesis_node may not have completed successfully."
                    )
                )
            ],
            "next_agent": ROUTE_END,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"regulatory_report_{timestamp}.md"

    logger.info(
        "report_writer_node | Writing report | filename=%s | size=%d chars",
        filename, len(report_content),
    )

    tool_result: str = await write_markdown_report.ainvoke(
        {"filename": filename, "content": report_content}
    )

    logger.info("report_writer_node | MCP write result: %.150s", tool_result)

    if tool_result.startswith("WRITE_ERROR"):
        final_message = (
            f"❌ **Report persistence failed.**\n\n"
            f"**Error:** `{tool_result}`\n\n"
            f"The report content is preserved in the message history."
        )
        logger.error("report_writer_node | Write failed: %s", tool_result)
    else:
        final_message = (
            f"✅ **Regulatory Intelligence Report — Complete**\n\n"
            f"{tool_result}\n\n"
            f"**Report covers:**\n"
            f"  • RBI/SEBI circular compliance analysis\n"
            f"  • Startup metrics vs regulatory thresholds\n"
            f"  • Actionable remediation recommendations\n\n"
            f"Ready for review in the workspace directory."
        )
        logger.info("report_writer_node | Pipeline complete ✓ | file=%s", filename)

    return {
        "messages":   [AIMessage(content=final_message)],
        "next_agent": ROUTE_END,
    }