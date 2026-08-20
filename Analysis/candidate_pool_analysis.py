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
            "Compute document-grounded supporting-entity coverage in a ranked "
            "candidate pool, including Recall@5/10/15/20 and ranks 6-20 statistics."
        )
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help=(
            "CSV/JSON/JSONL with qid and ranked candidate_entity_ids. "
            "Optional columns: level, perturbation, query."
        ),
    )
    parser.add_argument(
        "--supporting-entities",
        required=True,
        help="JSONL created by extract_supporting_entities.py.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qid-column", default="qid")
    parser.add_argument("--candidate-column", default="candidate_entity_ids")
    args = parser.parse_args()

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

    rows = []
    for _, row in candidates.iterrows():
        qid = str(row[args.qid_column])
        ranked = parse_list(row[args.candidate_column])
        support = support_map.get(qid, set())

        if not support:
            # Keep the case for transparency, but recall is undefined.
            recall_values = {k: None for k in (5, 10, 15, 20)}
        else:
            recall_values = {
                k: 100.0 * len(set(ranked[:k]) & support) / len(support)
                for k in (5, 10, 15, 20)
            }

        support_6_20 = set(ranked[5:20]) & support
        support_top20 = set(ranked[:20]) & support

        result = {
            "qid": qid,
            "perturbation": row.get("perturbation", ""),
            "level": row.get("level", ""),
            "n_supporting_entities": len(support),
            "recall_at_5": recall_values[5],
            "recall_at_10": recall_values[10],
            "recall_at_15": recall_values[15],
            "recall_at_20": recall_values[20],
            "n_supporting_entities_ranks_6_20": len(support_6_20),
            "has_supporting_entity_ranks_6_20": int(bool(support_6_20)),
            "omits_all_supporting_entities_top20": int(
                bool(support) and not bool(support_top20)
            ),
        }
        rows.append(result)

    case_df = pd.DataFrame(rows)
    if case_df.empty:
        raise ValueError("No candidate-pool records were analyzed.")

    valid_df = case_df[case_df["n_supporting_entities"] > 0].copy()
    if valid_df.empty:
        raise ValueError(
            "No cases had document-grounded supporting entities. "
            "Check the evidence-to-KG mapping."
        )

    group_column = "level" if valid_df["level"].astype(str).str.len().gt(0).any() else None

    summary_rows = []
    groups = (
        valid_df.groupby(group_column, dropna=False)
        if group_column
        else [("All Cases", valid_df)]
    )

    for name, group in groups:
        summary_rows.append(
            {
                "level": name,
                "n_cases": len(group),
                "R@5": group["recall_at_5"].mean(),
                "R@10": group["recall_at_10"].mean(),
                "R@15": group["recall_at_15"].mean(),
                "R@20": group["recall_at_20"].mean(),
                "supporting_entities_ranks_6_20_mean": (
                    group["n_supporting_entities_ranks_6_20"].mean()
                ),
                "cases_with_supporting_entities_ranks_6_20_pct": (
                    100.0
                    * group["has_supporting_entity_ranks_6_20"].mean()
                ),
                "top20_complete_omission_count": int(
                    group["omits_all_supporting_entities_top20"].sum()
                ),
                "top20_complete_omission_pct": (
                    100.0
                    * group["omits_all_supporting_entities_top20"].mean()
                ),
            }
        )

    # Weighted/all-case row: direct aggregation over all cases.
    summary_rows.append(
        {
            "level": "All Levels",
            "n_cases": len(valid_df),
            "R@5": valid_df["recall_at_5"].mean(),
            "R@10": valid_df["recall_at_10"].mean(),
            "R@15": valid_df["recall_at_15"].mean(),
            "R@20": valid_df["recall_at_20"].mean(),
            "supporting_entities_ranks_6_20_mean": (
                valid_df["n_supporting_entities_ranks_6_20"].mean()
            ),
            "cases_with_supporting_entities_ranks_6_20_pct": (
                100.0 * valid_df["has_supporting_entity_ranks_6_20"].mean()
            ),
            "top20_complete_omission_count": int(
                valid_df["omits_all_supporting_entities_top20"].sum()
            ),
            "top20_complete_omission_pct": (
                100.0 * valid_df["omits_all_supporting_entities_top20"].mean()
            ),
        }
    )

    summary_df = pd.DataFrame(summary_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / "candidate_pool_cases.csv"
    summary_path = output_dir / "candidate_pool_summary.csv"

    case_df.to_csv(case_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(summary_df.round(2).to_string(index=False))
    print(f"\nSaved: {case_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
