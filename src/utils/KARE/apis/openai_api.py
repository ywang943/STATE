# apis/openai_api.py
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
import sys
import time
import logging
import requests
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import API_URL, API_KEY, API_MAX_RETRY, API_RETRY_WAIT, LLM_MODEL

logger = logging.getLogger(__name__)


def call_llm(messages: list,
             model: str = None,
             temperature: float = 0.1,
             max_tokens: int = 1200,
             max_retry: int = None) -> str:
    """
    统一LLM调用接口。返回 assistant message 内容字符串。

    注意：Authorization 直接传 API Key，不加 'Bearer ' 前缀，
    以兼容 HKUST GPT（https://gpt-api.hkust-gz.edu.cn）接口。
    """
    if model is None:
        model = LLM_MODEL
    if max_retry is None:
        max_retry = API_MAX_RETRY

    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY,
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    last_exc = None
    for attempt in range(max_retry):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=500,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_exc = e
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retry}): {e}"
            )
            if attempt < max_retry - 1:
                sleep_s = API_RETRY_WAIT * (attempt + 1)
                logger.info(f"  Retrying in {sleep_s}s ...")
                time.sleep(sleep_s)

    raise RuntimeError(
        f"LLM call failed after {max_retry} attempts. Last error: {last_exc}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    print("Testing LLM connection...")
    reply = call_llm(
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
        max_tokens=100,
    )
    print("LLM reply:", reply)
