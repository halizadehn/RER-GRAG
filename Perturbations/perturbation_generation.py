import os
import sys
import pandas as pd

PERTURBATION_DIR = "."          
sys.path.insert(0, PERTURBATION_DIR)

import importlib

char_level     = importlib.import_module("char-level")
word_level     = importlib.import_module("word-level")
sentence_level = importlib.import_module("sentence-level")

PERTURBATIONS = {
    # char-level.py
    "RandDelete":         lambda q: char_level.rand_delete(q),
    "RandSub":            lambda q: char_level.rand_sub(q),
    "SwapNeighbor":       lambda q: char_level.swap_neighbor(q),
    "SwapAdjacent":       lambda q: char_level.swap_adjacent_keyboard(q),
    # word-level.py
    "SynonymReplacement": lambda q: word_level.synonym_replacement(q),
    "WordShuffle":        lambda q: word_level.word_order(q),
    "SingularPluralVerb": lambda q: word_level.singular_plural_verb(q),
    "WordDeletion":       lambda q: word_level.word_delete(q),
    "NeutralInsertion":   lambda q: word_level.word_add(q),
    # sentence-level.py
    "Paraphrase":         lambda q: sentence_level.paraphrase(q),
    "Formality":          lambda q: sentence_level.formality(q),
    "Readability":        lambda q: sentence_level.readability(q),
    "Politeness":         lambda q: sentence_level.politeness(q),
}


def run_pipeline(input_csv: str = INPUT_CSV, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    required = {"qid", "question", "gold_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    print(f"Loaded {len(df)} rows from '{input_csv}'")
    print(f"Generating {len(PERTURBATIONS)} perturbation files into '{output_dir}/'\n")

    for name, fn in PERTURBATIONS.items():
        print(f"  [{name}] ...", end=" ", flush=True)
        perturbed_questions = []
        for question in df["question"]:
            try:
                perturbed_questions.append(fn(str(question)))
            except Exception as e:
                print(f"\n    WARNING: failed on '{str(question)[:50]}': {e}")
                perturbed_questions.append(question)

        out_df = pd.DataFrame({
            "qid":                df["qid"].values,
            "question":           df["question"].values,
            "gold_answer":        df["gold_answer"].values,
            "perturbed_question": perturbed_questions,
        })

        out_path = os.path.join(output_dir, f"{name}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"saved → {out_path}")
        