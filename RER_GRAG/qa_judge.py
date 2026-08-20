from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI


def judge_answer(
    question: str,
    prediction: str,
    gold_answer: str,
    *,
    prompt_path: str,
    model: str = "gpt-5-mini",
) -> int:
    if not prediction or not str(prediction).strip():
        return 0

    template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = (
        template.replace("{{question}}", str(question))
        .replace("{{gold_answer}}", str(gold_answer))
        .replace("{{model_answer}}", str(prediction))
    )

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )

    text = ""
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            value = getattr(part, "text", None)
            if value:
                text += value

    try:
        return int(json.loads(text.strip()).get("flag", 0))
    except Exception:
        return 0
