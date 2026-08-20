from __future__ import annotations

import argparse
import asyncio
import inspect
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from graphrag_adapter import install_entity_reranking, remove_entity_reranking
from pipeline import build_search_engine
from qa_judge import judge_answer
from reranker import OutcomeSupervisedEntityReranker


REQUIRED_COLUMNS = {"qid", "question", "gold_answer"}


def parse_final_answer(response: str) -> str:
    match = re.search(r"FINAL_ANSWER\s*=\s*([^\n\r]+)", str(response))
    return match.group(1).strip() if match else str(response).strip()


async def run_search(search_engine: Any, query: str):
    if hasattr(search_engine, "asearch"):
        return await search_engine.asearch(query)
    result = search_engine.search(query)
    if inspect.isawaitable(result):
        return await result
    return result


async def main_async(args) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(args.input)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if args.query_column not in df.columns:
        raise ValueError(f"Query column '{args.query_column}' is not present.")

    rer_cfg = config["reranker"]
    tr_cfg = config["training"]
    grag_cfg = config["graphrag"]
    judge_cfg = config["judge"]

    reranker = OutcomeSupervisedEntityReranker(
        base_model=rer_cfg["base_model"],
        device=args.device,
        max_length=rer_cfg["max_length"],
        pair_batch_size=rer_cfg["pair_batch_size"],
        learning_rate=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
        fp16=rer_cfg["fp16"],
        replay_buffer_size=tr_cfg["replay_buffer_size"],
        replay_probability=tr_cfg["replay_probability"],
        replay_balance=tr_cfg["replay_balance"],
        top_k_pool=grag_cfg["selected_entities"],
        soft_topk_temperature=rer_cfg["soft_topk_temperature"],
        gradient_clip=tr_cfg["gradient_clip"],
        seed=tr_cfg["seed"],
    )
    reranker.set_training(True)

    install_entity_reranking(
        reranker,
        oversampling_factor=rer_cfg["oversampling_factor"],
    )
    search_engine = build_search_engine(args.kg_dir, config)

    records = []
    try:
        for idx, row in df.iterrows():
            query = str(row[args.query_column])
            gold = str(row["gold_answer"])

            result = await run_search(search_engine, query)
            prediction = parse_final_answer(result.response)

            flag = judge_answer(
                query,
                prediction,
                gold,
                prompt_path=judge_cfg["prompt"],
                model=judge_cfg["model"],
            )
            loss = reranker.update_last_bag(flag)

            records.append(
                {
                    "qid": row["qid"],
                    "query": query,
                    "gold_answer": gold,
                    "prediction": prediction,
                    "correct": flag,
                    "training_loss": loss,
                }
            )
            print(
                f"[{idx + 1}/{len(df)}] qid={row['qid']} "
                f"correct={flag} loss={loss}"
            )
    finally:
        remove_entity_reranking()

    reranker.save(args.output_dir)
    log_path = Path(args.output_dir) / "training_predictions.csv"
    pd.DataFrame(records).to_csv(log_path, index=False)

    print(f"Saved trained reranker to: {args.output_dir}")
    print(f"Saved training log to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tr-RER-GRAG using outcome-level answer correctness."
    )
    parser.add_argument("--config", default="RER_GRAG/config.yaml")
    parser.add_argument("--kg-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-column", default="question")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
