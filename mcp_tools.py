# mcp_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# Async LangChain tools that communicate with external data services through
# the Model Context Protocol (MCP). Each tool spawns a dedicated MCP server
# subprocess, performs a single operation over a short-lived stdio session,
# and then cleanly shuts down — following the "connect → initialize → call →
# close" lifecycle that is the canonical pattern for MCP Python clients.
#
# Dependencies (add to requirements.txt / pyproject.toml):
#   mcp>=1.4.0
#   langchain-core>=0.3.0
#   uvx       (for mcp-server-sqlite  — install via: pip install uv)
#   Node.js   (for @modelcontextprotocol/server-filesystem — npx is bundled)
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import os
from typing import Any

from langchain_core.tools import tool

# ── MCP Python SDK ────────────────────────────────────────────────────────────
# StdioServerParameters  : Describes the subprocess to spawn as the MCP server.
# stdio_client           : Async context manager that launches the subprocess
#                          and yields (read_stream, write_stream) stdio pipes.
# ClientSession          : Async context manager that wraps the raw streams in
#                          the full MCP JSON-RPC session protocol.
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# ── Project config ────────────────────────────────────────────────────────────
from config import SQLITE_DB_PATH, WORKSPACE_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Error sentinels
# The supervisor node in our LangGraph graph pattern-matches on these prefixes
# to decide whether to trigger the A2A self-correction loop or proceed.
# ─────────────────────────────────────────────────────────────────────────────
SQL_ERROR_PREFIX   = "SQL_ERROR"
WRITE_ERROR_PREFIX = "WRITE_ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_content(content_items: list[Any]) -> str:
    """
    Extract a single human-readable string from an MCP CallToolResult.content list.

    MCP tool results return a heterogeneous list of content objects:
      • TextContent  → has a `.text`  attribute (most common for SQL/file tools)
      • BlobContent  → has a `.data`  attribute (binary payloads)
      • ImageContent → has a `.data`  attribute (base64 encoded)

    We concatenate all text parts so callers receive a unified string,
    regardless of how many content blocks the MCP server chose to emit.
    """
    parts: list[str] = []
    for item in content_items:
        if hasattr(item, "text") and isinstance(item.text, str):
            parts.append(item.text.strip())
        elif hasattr(item, "data"):
            # Gracefully handle non-text blobs by stringifying them
            parts.append(str(item.data))
        else:
            parts.append(repr(item))
    return "\n".join(parts) if parts else "(empty response)"


def _get_subprocess_env() -> dict[str, str]:
    """
    Return a full copy of the current process environment for subprocesses.

    Passing os.environ.copy() (rather than None) is explicit and avoids
    subtle failures on systems where PATH is not inherited by default,
    ensuring `uvx` and `npx` are discoverable inside spawned processes.
    """
    return os.environ.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — execute_local_sql
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def execute_local_sql(query: str) -> str:
    """
    Execute a SQL query against the local SQLite startup-metrics database
    using the official MCP SQLite server (mcp-server-sqlite).

    Use this tool to answer questions that require retrieving structured data
    such as revenue figures, burn rate, headcount, or any other operational
    metrics stored in the local database. Supports SELECT statements.

    For INSERT / UPDATE / DELETE operations, this tool will automatically
    route to the server's write tool ('write-query') instead of 'read-query'.

    Returns query results as a formatted string on success.
    Returns a string beginning with "SQL_ERROR:" on failure — this sentinel
    is intentionally preserved for the A2A supervisor self-correction loop.

    Args:
        query: A valid SQL statement to execute against the startup metrics DB.
    """
    # ── 1. Server configuration ───────────────────────────────────────────────
    # StdioServerParameters describes HOW to launch the MCP server subprocess.
    #   command : The executable to run ("uvx" manages Python tool environments).
    #   args    : CLI arguments forwarded to the subprocess.
    #             "mcp-server-sqlite" is the package name; "--db-path" points to
    #             our SQLite file. uvx downloads + runs it in an isolated venv.
    #   env     : Full copy of the parent environment so PATH is always resolved.
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-sqlite", "--db-path", str(SQLITE_DB_PATH)],
        env=_get_subprocess_env(),
    )

    # Determine the correct MCP tool name based on the SQL statement type.
    # The official mcp-server-sqlite exposes two distinct query tools:
    #   read-query  → SELECT statements (read-only, safe)
    #   write-query → INSERT / UPDATE / DELETE (mutating)
    # Note: the official tool names use HYPHENS, not underscores.
    query_upper = query.strip().upper()
    mcp_tool_name = (
        "read-query"
        if query_upper.startswith("SELECT") or query_upper.startswith("WITH")
        else "write-query"
    )

    logger.info(
        "execute_local_sql | Spawning MCP SQLite server | tool=%s | query=%.120s",
        mcp_tool_name,
        query,
    )

    try:
        # ── 2. MCP Session Lifecycle ──────────────────────────────────────────
        #
        # The session lifecycle is managed entirely through nested async context
        # managers. Each layer has a clear responsibility:
        #
        # Layer A — stdio_client(server_params)
        #   • Spawns the `uvx mcp-server-sqlite` subprocess.
        #   • Connects to its stdin/stdout as bidirectional async byte streams.
        #   • Yields a (read_stream, write_stream) tuple.
        #   • On __aexit__: sends EOF and waits for the subprocess to terminate.
        #
        # Layer B — ClientSession(read_stream, write_stream)
        #   • Wraps the raw byte streams in the MCP JSON-RPC 2.0 framing layer.
        #   • Manages request/response correlation via message IDs.
        #   • On __aexit__: sends a graceful MCP shutdown notification.
        #
        # Layer C — session.initialize()
        #   • Performs the MCP capability handshake:
        #     client → { "method": "initialize", clientInfo, capabilities }
        #     server → { protocolVersion, serverInfo, capabilities }
        #   • Must be called exactly once before any tool calls.
        #   • After this, session.list_tools() / session.call_tool() are valid.
        #
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                # ── Layer C: Handshake ────────────────────────────────────────
                await session.initialize()
                logger.debug("execute_local_sql | MCP session initialized (mcp-server-sqlite)")

                # ── Layer D: Tool Invocation ──────────────────────────────────
                # session.call_tool() sends a JSON-RPC "tools/call" request and
                # awaits the server's response. The server executes the SQL query
                # against the SQLite file and returns results as CallToolResult.
                #
                # CallToolResult structure:
                #   .content  : list[TextContent | ImageContent | BlobContent]
                #   .isError  : bool — True when the server itself caught an error
                #                     (e.g., invalid SQL, missing table).
                result = await session.call_tool(
                    mcp_tool_name,
                    arguments={"query": query},
                )

                # ── Layer E: Application-layer error check ────────────────────
                # isError=True means the MCP server executed successfully as a
                # process but the SQL query itself was rejected at runtime.
                # We surface this as our SQL_ERROR sentinel rather than raising,
                # so the LangGraph supervisor can inspect and self-correct.
                if result.isError:
                    error_text = _extract_text_from_content(result.content)
                    logger.warning(
                        "execute_local_sql | Server returned isError=True: %s", error_text
                    )
                    return f"{SQL_ERROR_PREFIX}: {error_text}"

                # ── Layer F: Success path ─────────────────────────────────────
                output = _extract_text_from_content(result.content)
                logger.info(
                    "execute_local_sql | Query succeeded | response_length=%d chars", len(output)
                )
                return output

    except Exception as exc:
        # ── A2A Self-Correction Sentinel ──────────────────────────────────────
        # ALL exceptions are caught here and returned as a structured error
        # string rather than being re-raised. This is CRITICAL for graph health:
        #
        # The supervisor node reads AgentState.sql_error. If it starts with
        # "SQL_ERROR:", the supervisor re-routes to the SQL agent, injects the
        # error text as corrective context, and triggers a retry cycle.
        # Raising here would crash the entire graph run instead.
        #
        # Exception categories covered:
        #   • FileNotFoundError   → uvx or mcp-server-sqlite not installed
        #   • ConnectionError     → subprocess crashed before session init
        #   • McpError            → MCP protocol-level failure
        #   • Exception           → Any unexpected runtime error
        error_msg = f"{SQL_ERROR_PREFIX}: {type(exc).__name__}: {exc}"
        logger.error(
            "execute_local_sql | Exception during MCP session: %s",
            error_msg,
            exc_info=True,
        )
        return error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — write_markdown_report
# ─────────────────────────────────────────────────────────────────────────────

@tool
async def write_markdown_report(filename: str, content: str) -> str:
    """
    Persist a Markdown intelligence report to the agent's workspace directory
    using the official MCP Filesystem server (@modelcontextprotocol/server-filesystem).

    Use this tool as the FINAL step in the report-writing agent to durably
    save the generated compliance analysis, risk assessments, or circular
    summaries as a Markdown file. Files are written inside WORKSPACE_DIR only;
    the MCP Filesystem server enforces this boundary at the OS level.

    Args:
        filename : Name of the file to create (e.g., "rbi_circular_q1_2025.md").
                   Include the .md extension. Do NOT include a path prefix —
                   the tool resolves the full path automatically.
        content  : Complete Markdown text to write. Existing files are overwritten.

    Returns:
        A success message including the absolute file path on success.
        A string beginning with "WRITE_ERROR:" on failure.
    """
    # ── 1. Resolve target path ─────────────────────────────────────────────────
    # We anchor the caller-supplied filename to WORKSPACE_DIR to prevent path
    # traversal attacks (e.g., filename="../../etc/passwd") and to guarantee
    # all reports land in the correct directory regardless of the calling agent's
    # working directory.
    target_path = WORKSPACE_DIR / filename

    # ── 2. Server configuration ────────────────────────────────────────────────
    # StdioServerParameters for the official Node.js Filesystem MCP server.
    #   command : "npx" — Node Package Execute, bundled with Node.js (≥v16).
    #   args[0] : "-y"  — Auto-confirm package installation (no interactive prompt).
    #   args[1] : The NPM package name for the official MCP Filesystem server.
    #   args[2] : The ALLOWED ROOT DIRECTORY. The server refuses all file
    #             operations outside this path, providing a sandboxed workspace.
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(WORKSPACE_DIR),   # Sandbox root — only this directory is accessible
        ],
        env=_get_subprocess_env(),
    )

    logger.info(
        "write_markdown_report | Spawning MCP Filesystem server | target=%s | content_length=%d",
        target_path,
        len(content),
    )

    try:
        # ── 3. MCP Session Lifecycle ───────────────────────────────────────────
        #
        # Identical three-layer pattern as execute_local_sql:
        #
        # Layer A — stdio_client(server_params)
        #   • Spawns `npx @modelcontextprotocol/server-filesystem <WORKSPACE_DIR>`.
        #   • Node.js downloads the package on first run, then caches it locally.
        #   • Yields (read_stream, write_stream) connected to the server's stdio.
        #   • On __aexit__: terminates the Node.js process cleanly.
        #
        # Layer B — ClientSession(read_stream, write_stream)
        #   • MCP JSON-RPC session layer (same as execute_local_sql above).
        #   • Handles message framing, correlation, and graceful shutdown.
        #
        # Layer C — session.initialize()
        #   • MCP capability handshake. The Filesystem server reports which
        #     tools it supports (write_file, read_text_file, list_directory, etc.).
        #   • Since we do NOT advertise roots capability in our ClientSession,
        #     the server falls back to using the command-line argument (WORKSPACE_DIR)
        #     as its sole allowed directory — which is exactly what we want.
        #
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                # ── Layer C: Handshake ────────────────────────────────────────
                await session.initialize()
                logger.debug(
                    "write_markdown_report | MCP session initialized "
                    "(server-filesystem | root=%s)",
                    WORKSPACE_DIR,
                )

                # ── Layer D: Tool Invocation ──────────────────────────────────
                # The official @modelcontextprotocol/server-filesystem server
                # exposes the "write_file" tool with two required parameters:
                #   path    (str): Absolute path to the file. Must be inside the
                #                  allowed root directory (WORKSPACE_DIR).
                #   content (str): UTF-8 string content to write. Overwrites any
                #                  existing file at that path atomically.
                #
                # Reference: modelcontextprotocol/servers · src/filesystem/README.md
                result = await session.call_tool(
                    "write_file",
                    arguments={
                        "path":    str(target_path),
                        "content": content,
                    },
                )

                # ── Layer E: Application-layer error check ────────────────────
                if result.isError:
                    error_text = _extract_text_from_content(result.content)
                    logger.error(
                        "write_markdown_report | Server returned isError=True: %s", error_text
                    )
                    return f"{WRITE_ERROR_PREFIX}: {error_text}"

                # ── Layer F: Success path ─────────────────────────────────────
                # Build a human-readable success message. The report-writer agent
                # uses this string to populate AgentState.messages, confirming
                # the file was persisted before the graph transitions to END.
                success_msg = (
                    f"Report successfully written.\n"
                    f"  File : {target_path}\n"
                    f"  Size : {len(content):,} characters\n"
                    f"  Lines: {content.count(chr(10)) + 1:,}"
                )
                logger.info("write_markdown_report | %s", success_msg.replace("\n", " | "))
                return success_msg

    except Exception as exc:
        # Return a WRITE_ERROR sentinel rather than raising, mirroring the
        # defensive pattern used in execute_local_sql. The supervisor can
        # inspect this and decide whether to retry or surface the error to
        # the user via the messages channel.
        error_msg = f"{WRITE_ERROR_PREFIX}: {type(exc).__name__}: {exc}"
        logger.error(
            "write_markdown_report | Exception during MCP session: %s",
            error_msg,
            exc_info=True,
        )
        return error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# Import this list into your LangGraph node definitions or ToolNode:
#
#   from mcp_tools import MCP_TOOLS
#   tool_node = ToolNode(MCP_TOOLS)
# ─────────────────────────────────────────────────────────────────────────────
MCP_TOOLS: list = [execute_local_sql, write_markdown_report]