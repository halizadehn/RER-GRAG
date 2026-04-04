import os
from pathlib import Path
import pandas as pd
from openai import OpenAI
import json
import numpy as np
import ast
import re
client = OpenAI()


def judge_qa_with_llm(input_csv, output_csv, prompt_path="QA_judge.md", model_name="gpt-5-mini"):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    def judge_row(row):
        q = str(row.get("question", ""))
        gold = str(row.get("gold_answer", ""))
        model_ans = str(row.get("grag_answer", ""))

        if model_ans.strip() == "":
            return 0

        user_content = (prompt_template
                        .replace("{{question}}", q)
                        .replace("{{gold_answer}}", gold)
                        .replace("{{model_answer}}", model_ans))

        r = client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": user_content}],
        )

        text = ""
        if getattr(r, "output", None):
            for item in r.output:
                for p in (getattr(item, "content", None) or []):
                    t = getattr(p, "text", None)
                    if t:
                        text += t
        text = text.strip()

        try:
            flag = int(json.loads(text).get("flag", 0))
        except:
            flag = 0
        print(f"gold: {gold} model: {model_ans} -> flag: {flag}")
        return flag

    df["llm_flag"] = df.apply(judge_row, axis=1)
    df.to_csv(output_csv, index=False)
