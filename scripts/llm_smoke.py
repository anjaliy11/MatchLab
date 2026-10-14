"""Send one minimal request to the explicitly configured OpenAI-compatible endpoint."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matchlab.adapters.openai_compat import CompletionError, OpenAICompatibleAdapter


def main() -> int:
    base_url = (os.environ.get("LLM_BASE_URL") or "").strip()
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if not base_url or not model:
        print("Outcome: bad_request — configure LLM_BASE_URL and LLM_MODEL; no default host is used.")
        return 2
    try:
        adapter = OpenAICompatibleAdapter(base_url=base_url, model=model)
        adapter.complete(
            [{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_tokens=4,
        )
    except CompletionError as exc:
        print(f"Outcome: {exc.outcome} — {exc}")
        return 1
    except ValueError as exc:
        print(f"Outcome: bad_request — {exc}")
        return 2
    print("Outcome: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
