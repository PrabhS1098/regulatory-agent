
# Regulatory Intelligence Agent

Autonomous multi-agent system that analyses startup compliance posture against RBI/SEBI regulatory circulars and internal metrics data.

## Architecture

5-node LangGraph StateGraph with a central supervisor orchestrating:

- **Supervisor** — routes state across the pipeline
- **Policy RAG** — retrieves relevant RBI/SEBI circulars
- **Data Analyst** — generates and executes SQL against startup metrics DB
- **Synthesis** — drafts the compliance report with A2A self-correction loop
- **Report Writer** — persists the final Markdown report via MCP filesystem tool

## Key Features

- A2A self-correction loop — when SQL fails, Synthesis critiques the error and routes back to Data Analyst with a fix, capped at 2 iterations
- Real-time SSE streaming — every node update pushed to the frontend as it happens
- Animated dark-mode UI — agent nodes pulse and glow as they execute, red flash when A2A loop triggers
- MCP tool integration — SQL execution and file writes via Model Context Protocol
- Context-aware trimming — message history trimmed before every LLM call to stay within free-tier token limits

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Google Gemini 2.0 Flash |
| Backend | FastAPI + uvicorn |
| Streaming | Server-Sent Events (SSE) |
| Database | SQLite |
| Frontend | Vanilla JS + Tailwind CSS |
| Tools | MCP (Model Context Protocol) |

## Setup

```bash
git clone https://github.com/PrabhS1098/regulatory-agent.git
cd regulatory-agent
pip install -r requirements.txt
echo "google_api_key=your_key_here" > .env
python api.py
```

Open http://localhost:8000

## Project Structure
