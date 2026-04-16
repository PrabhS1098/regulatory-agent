# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralized configuration for the Regulatory Intelligence Agent.
# Handles environment loading, LLM initialization, and path management.
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

GOOGLE_API_KEY: str = os.environ.get("google_api_key", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "google_api_key is not set. "
        "Please add it to your .env file: google_api_key=your_key_here"
    )

# ── LLM ───────────────────────────────────────────────────────────────────────
# Single, shared Gemini instance used across all agents in the graph.
# temperature=0 ensures deterministic, factual outputs for regulatory analysis.
# gemini-2.0-flash has 1M TPM on the free tier — no rate limit issues.
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)
logger.info("ChatGoogleGenerativeAI LLM initialized | model=gemini-2.0-flash")

# ── Local Data Paths ──────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).parent.resolve()

SQLITE_DB_PATH: Path = _PROJECT_ROOT / "data" / "startup_metrics.db"
CHROMADB_DIR:   Path = _PROJECT_ROOT / "data" / "chroma_db"
WORKSPACE_DIR:  Path = _PROJECT_ROOT / "workspace"

# ── Directory Bootstrap ───────────────────────────────────────────────────────
for _dir in (SQLITE_DB_PATH.parent, CHROMADB_DIR, WORKSPACE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

logger.info("Workspace directories verified/created.")
logger.info("  SQLITE_DB_PATH : %s", SQLITE_DB_PATH)
logger.info("  CHROMADB_DIR   : %s", CHROMADB_DIR)
logger.info("  WORKSPACE_DIR  : %s", WORKSPACE_DIR)