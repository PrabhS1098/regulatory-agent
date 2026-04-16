"""
api.py — FastAPI SSE backend for the Regulatory Intelligence Agent
Serves index.html from /static and streams LangGraph node updates via SSE.
"""

import json
import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# ── Import your compiled LangGraph app ──────────────────────────────────────
from graph import app as langgraph_app

# ── App setup ────────────────────────────────────────────────────────────────
api = FastAPI(
    title="Regulatory Intelligence Agent API",
    description="SSE-streaming LangGraph backend",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html lives here)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response models ─────────────────────────────────────────────────
class AgentRequest(BaseModel):
    prompt: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_payload(node_name: str, node_output: dict) -> dict:
    """
    Converts a single LangGraph node update dict into a clean SSE payload.

    Expected fields in the SSE payload:
        node      – name of the node that just ran
        content   – any assistant/AI message text produced
        sql_error – non-empty string if a SQL error occurred
        report    – path/name of the generated report file (if any)
        done      – True only on the final "end" sentinel
    """
    payload: dict = {
        "node": node_name,
        "content": "",
        "sql_error": "",
        "report": "",
        "done": False,
    }

    # ── Pull the latest AI message content, if present ──────────────────────
    messages = node_output.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # LangGraph may return BaseMessage objects or plain dicts
        if hasattr(last_msg, "content"):
            payload["content"] = last_msg.content or ""
        elif isinstance(last_msg, dict):
            payload["content"] = last_msg.get("content", "")

    # ── SQL error forwarding ─────────────────────────────────────────────────
    sql_error = node_output.get("sql_error", "")
    if sql_error:
        payload["sql_error"] = sql_error

    # ── Report file detection ────────────────────────────────────────────────
    # Convention: report_writer node stores the filename in node_output["report"]
    # or embeds a markdown path inside its message content.
    report_path = node_output.get("report", "")
    if report_path:
        payload["report"] = report_path
    elif node_name == "report_writer" and payload["content"]:
        # Heuristic: last line may be "Report saved to: reports/xxx.md"
        for line in reversed(payload["content"].splitlines()):
            line = line.strip()
            if line.endswith(".md") or line.endswith(".pdf"):
                payload["report"] = line.split()[-1]
                break

    return payload


async def _sse_generator(prompt: str) -> AsyncGenerator[str, None]:
    """
    Async generator that streams LangGraph node updates as SSE data frames.

    Each frame is a JSON object terminated with a double-newline, conforming
    to the EventSource protocol:
        data: {...}\n\n
    """
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "rag_context": "",
        "sql_context": "",
        "sql_error": "",
        "next_agent": "",
        "a2a_iterations": 0,
    }

    # Generate a unique thread ID for this specific run
    run_config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    try:
        # Pass the config with the thread_id into astream
        async for event in langgraph_app.astream(
            initial_state,
            config=run_config,
            stream_mode="updates",
        ):
            # `event` is { node_name: node_output_dict }
            for node_name, node_output in event.items():
                if not isinstance(node_output, dict):
                    continue

                payload = _extract_payload(node_name, node_output)
                yield f"data: {json.dumps(payload)}\n\n"

                # Small yield to keep the event loop cooperative
                await asyncio.sleep(0)

    except Exception as exc:
        error_payload = {
            "node": "error",
            "content": f"Fatal agent error: {exc}",
            "sql_error": "",
            "report": "",
            "done": False,
        }
        yield f"data: {json.dumps(error_payload)}\n\n"

    finally:
        # Send a terminal "done" frame so the frontend knows to stop
        done_payload = {
            "node": "__done__",
            "content": "",
            "sql_error": "",
            "report": "",
            "done": True,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────
@api.get("/", response_class=HTMLResponse)
async def root():
    """Redirect root to the static index page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Place index.html inside the /static directory.</h1>",
        status_code=404,
    )


@api.post("/api/run_agent")
async def run_agent(request: AgentRequest):
    """
    Accepts a JSON body: { "prompt": "<user query>" }
    Returns a text/event-stream SSE response that emits one JSON payload
    per completed LangGraph node.
    """
    return StreamingResponse(
        _sse_generator(request.prompt),
        media_type="text/event-stream",
        headers={
            # Prevent buffering at any proxy layer
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@api.get("/health")
async def health():
    return {"status": "ok"}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:api",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )