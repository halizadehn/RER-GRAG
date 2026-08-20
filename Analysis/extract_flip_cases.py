from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_flag(value) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value == 1)

    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "correct", "pass", "accepted"}:
        return 1
    if value in {"0", "false", "no", "incorrect", "fail", "rejected"}:
        return 0
    raise ValueError(f"Unknown correctness value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract perturbation-induced flip cases: questions answered correctly "
            "for the clean query but incorrectly for the perturbed query."
        )
    )
    parser.add_argument("--clean", required=True, help="Judged clean-result CSV.")
    parser.add_argument("--perturbed", required=True, help="Judged perturbed-result CSV.")
    parser.add_argument("--output", required=True, help="Output CSV containing flip cases.")
    parser.add_argument("--qid-column", default="qid")
    parser.add_argument("--flag-column", default="llm_flag")
    parser.add_argument("--query-column", default="question")
    parser.add_argument(
        "--perturbed-query-column",
        default="perturbed_question",
        help="Column containing the perturbed query when available.",
    )
    args = parser.parse_args()

    clean = pd.read_csv(args.clean, dtype={args.qid_column: str})
    perturbed = pd.read_csv(args.perturbed, dtype={args.qid_column: str})

    required_clean = {args.qid_column, args.flag_column}
    required_perturbed = {args.qid_column, args.flag_column}
    if not required_clean.issubset(clean.columns):
        raise ValueError(f"Clean file must contain {sorted(required_clean)}.")
    if not required_perturbed.issubset(perturbed.columns):
        raise ValueError(f"Perturbed file must contain {sorted(required_perturbed)}.")

    clean = clean.copy()
    perturbed = perturbed.copy()
    clean["clean_correct"] = clean[args.flag_column].apply(parse_flag)
    perturbed["perturbed_correct"] = perturbed[args.flag_column].apply(parse_flag)

    keep_clean = [args.qid_column, "clean_correct"]
    if args.query_column in clean.columns:
        keep_clean.append(args.query_column)

    keep_pert = [args.qid_column, "perturbed_correct"]
    if args.perturbed_query_column in perturbed.columns:
        keep_pert.append(args.perturbed_query_column)
    elif args.query_column in perturbed.columns:
        keep_pert.append(args.query_column)

    merged = clean[keep_clean].merge(
        perturbed[keep_pert],
        on=args.qid_column,
        how="inner",
        suffixes=("_clean", "_perturbed"),
        validate="one_to_one",
    )

    flips = merged[
        (merged["clean_correct"] == 1)
        & (merged["perturbed_correct"] == 0)
    ].copy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    flips.to_csv(output, index=False)

    print(f"Matched questions: {len(merged)}")
    print(f"Perturbation-induced flip cases: {len(flips)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
