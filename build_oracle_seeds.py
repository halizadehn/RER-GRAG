from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return [str(x) for x in value.tolist()]

    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [text]


def read_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True, dtype={"qid": str})
    if path.suffix.lower() == ".json":
        return pd.read_json(path, dtype={"qid": str})
    return pd.read_csv(path, dtype={"qid": str})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build gold-informed oracle Top-K entity lists from a fixed Top-T "
            "candidate pool. Supporting entities are prioritized without injecting "
            "entities outside the original candidate pool."
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--supporting-entities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qid-column", default="qid")
    parser.add_argument("--candidate-column", default="candidate_entity_ids")
    parser.add_argument("--top-t", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k > args.top_t:
        raise ValueError("--top-k cannot exceed --top-t.")

    candidates = read_records(Path(args.candidates))
    supporting = read_records(Path(args.supporting_entities))

    if args.qid_column not in candidates.columns:
        raise ValueError(f"Candidate file must contain '{args.qid_column}'.")
    if args.candidate_column not in candidates.columns:
        raise ValueError(f"Candidate file must contain '{args.candidate_column}'.")
    if "supporting_entity_ids" not in supporting.columns:
        raise ValueError("Supporting-entity file must contain 'supporting_entity_ids'.")

    support_map = {
        str(row[args.qid_column]): set(parse_list(row["supporting_entity_ids"]))
        for _, row in supporting.iterrows()
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_cases = 0
    n_with_support_top_t = 0

    with output.open("w", encoding="utf-8") as handle:
        for _, row in candidates.iterrows():
            qid = str(row[args.qid_column])
            ranked = parse_list(row[args.candidate_column])[: args.top_t]
            support = support_map.get(qid, set())

            supporting_in_pool = [eid for eid in ranked if eid in support]
            non_supporting_in_pool = [eid for eid in ranked if eid not in support]

            # Preserve original retrieval order within each group.
            oracle_topk = (
                supporting_in_pool + non_supporting_in_pool
            )[: args.top_k]

            if supporting_in_pool:
                n_with_support_top_t += 1

            record = {
                "qid": qid,
                "query": row.get("query", row.get("perturbed_question", row.get("question", ""))),
                "perturbation": row.get("perturbation", ""),
                "level": row.get("level", ""),
                "candidate_entity_ids": ranked,
                "supporting_entity_ids_in_top_t": supporting_in_pool,
                "oracle_topk_entity_ids": oracle_topk,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_cases += 1

    print(f"Oracle seed cases: {n_cases}")
    print(f"Cases with supporting entities in Top-{args.top_t}: {n_with_support_top_t}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
