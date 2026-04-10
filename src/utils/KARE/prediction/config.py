import os
from pathlib import Path

ROOT_DIR         = Path(__file__).resolve().parent
DATA_DIR         = ROOT_DIR / "data"
RESULTS_DIR      = ROOT_DIR / "results"
LOGS_DIR         = ROOT_DIR / "logs"

PATIENTS_FILE       = DATA_DIR / "train_wei.jsonl"
KG_FILE             = DATA_DIR / "kg.json"
KG_COMMUNITIES_FILE = DATA_DIR / "kg_communities.json"
BASE_CONTEXT_FILE   = DATA_DIR / "base_contexts.jsonl"
EMBEDDINGS_FILE     = DATA_DIR / "embeddings.npz"
SIM_PATIENTS_FILE   = DATA_DIR / "sim_patients.json"

TEST_FILE               = DATA_DIR / "test_wei.jsonl"
TEST_BASE_CONTEXT_FILE  = DATA_DIR / "test_base_contexts.jsonl"
TEST_AUG_CONTEXT_FILE   = DATA_DIR / "test_augmented_contexts.jsonl"

PREDICT_INPUT_FILE  = DATA_DIR / "predict_input.jsonl"
PREDICT_OUTPUT_FILE = RESULTS_DIR / "predictions.jsonl"
EVAL_OUTPUT_FILE    = RESULTS_DIR / "eval_results.json"

EMB_MODEL_NAME   = "moka-ai/m3e-base"

TOP_K_SIM        = 5
TOP_K_KG         = 12
TOP_K_COMMUNITY  = 3
TOP_K_OUT        = 10
MIN_HERB_FREQ    = 2
COMMUNITY_MIN_SIZE = 3

# ── API ───────────────────────────────────────
API_URL   = os.environ.get(
    "OPENAI_API_URL",
    "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
)
API_KEY   = os.environ.get(
    "OPENAI_API_KEY",
    ""
)
LLM_MODEL        = "gpt-4"
LLM_TEMP_REASON  = 0.2
LLM_TEMP_PREDICT = 0.1
LLM_MAX_TOKENS   = 1200
API_MAX_RETRY    = 3
API_RETRY_WAIT   = 5

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
