import argparse
import random
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

import character_level
import sentence_level
import word_level

REQUIRED_COLUMNS = {"qid", "question", "gold_answer"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    character_level.set_seed(seed)
    word_level.set_seed(seed)


def build_registry(model: str, temperature: float, char_pct_words_to_swap: float):
    return {
        "character": {
            "RandSwap": lambda q: character_level.rand_swap(q, char_pct_words_to_swap),
            "NeighborSwap": lambda q: character_level.neighbor_swap(q, char_pct_words_to_swap),
            "AdjacentSwap": lambda q: character_level.adjacent_swap(q, char_pct_words_to_swap),
            "RandDeletion": lambda q: character_level.rand_deletion(q, char_pct_words_to_swap),
        },
        "word": {
            "SynonymSwap": lambda q: word_level.synonym_swap(q, model, temperature),
            "NeighborSwap": word_level.neighbor_swap,
            "NeutralInsertion": word_level.neutral_insertion,
            "NeutralDeletion": word_level.neutral_deletion,
        },
        "sentence": {
            "Paraphrase": lambda q: sentence_level.paraphrase(q, model, temperature),
            "Formality": lambda q: sentence_level.formality(q, model, temperature),
            "Politeness": lambda q: sentence_level.politeness(q, model, temperature),
            "Readability": lambda q: sentence_level.readability(q, model, temperature),
        },
    }


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")


def select_jobs(registry, level: str, perturbation_type: Optional[str]):
    if level == "all":
        if perturbation_type:
            raise ValueError("--type can only be used with character, word, or sentence level.")
        for current_level, functions in registry.items():
            for name, fn in functions.items():
                yield current_level, name, fn
        return

    functions = registry[level]
    if perturbation_type:
        if perturbation_type not in functions:
            raise ValueError(
                f"Unknown type '{perturbation_type}' for level '{level}'. "
                f"Valid values: {', '.join(functions.keys())}"
            )
        yield level, perturbation_type, functions[perturbation_type]
        return

    for name, fn in functions.items():
        yield level, name, fn


def generate_file(df: pd.DataFrame, fn: Callable[[str], str], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists. Use --overwrite to replace it.")
        return

    perturbed_questions = []
    for row_number, question in enumerate(df["question"], start=1):
        question = str(question)
        try:
            perturbed_questions.append(fn(question))
        except Exception as exc:
            print(f"[warning] row {row_number}: {exc}; original question retained.")
            perturbed_questions.append(question)

    out_df = pd.DataFrame({
        "qid": df["qid"].values,
        "question": df["question"].values,
        "gold_answer": df["gold_answer"].values,
        "perturbed_question": perturbed_questions,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[saved] {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate query perturbations used in the GraphRAG robustness study.")
    parser.add_argument("--input", required=True, help="CSV with qid, question, and gold_answer columns.")
    parser.add_argument("--output-dir", required=True, help="Dataset-specific output directory.")
    parser.add_argument("--level", required=True, choices=["all", "character", "word", "sentence"])
    parser.add_argument("--type", default=None, help="Optional single perturbation type, e.g. RandSwap or Paraphrase.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--char-pct-words-to-swap", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.input)
    validate_input(df)

    registry = build_registry(args.model, args.temperature, args.char_pct_words_to_swap)
    jobs = list(select_jobs(registry, args.level, args.type))

    print(f"Loaded {len(df)} questions from {args.input}")
    print(f"Generating {len(jobs)} perturbation file(s).")

    output_dir = Path(args.output_dir)
    for level, name, fn in jobs:
        print(f"[{level}/{name}]")
        generate_file(df, fn, output_dir / level / f"{name}.csv", args.overwrite)


if __name__ == "__main__":
    main()
