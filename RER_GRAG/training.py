from __future__ import annotations
import csv
import os
import re
import pandas as pd
import asyncio
import ast
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_text_units,
)
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

INPUT_DIR = "./HotpotGRAG/output"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
COVARIATE_TABLE = "covariates"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 0

entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")

entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)

description_embedding_store = LanceDBVectorStore(
    vector_store_schema_config=VectorStoreSchemaConfig(
        index_name="default-entity-description"
    )
)
description_embedding_store.connect(db_uri=LANCEDB_URI)

print(f"Entity count: {len(entity_df)}")

relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

print(f"Relationship count: {len(relationship_df)}")

text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)

print(f"Text unit records: {len(text_unit_df)}")

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer


chat_config = LanguageModelConfig(
    api_key=my_api_key,
    type=ModelType.Chat,
    model_provider="openai",
    model="gpt-4o-mini",
    max_retries=20,
)
chat_model = ModelManager().get_or_create_chat_model(
    name="local_search",
    model_type=ModelType.Chat,
    config=chat_config,
)

embedding_config = LanguageModelConfig(
    api_key=my_api_key,
    type=ModelType.Embedding,
    model_provider="openai",
    model="text-embedding-3-small",
    max_retries=20,
)

text_embedder = ModelManager().get_or_create_embedding_model(
    name="local_search_embedding",
    model_type=ModelType.Embedding,
    config=embedding_config,
)

tokenizer = get_tokenizer(chat_config)

context_builder = LocalSearchMixedContext(
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,  
    text_embedder=text_embedder,
    tokenizer=tokenizer,
)


local_context_params = {
    "text_unit_prop": 0.8,
    "community_prop": 0.0,
    "conversation_history_max_turns": 0,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 5,
    "top_k_relationships": 5,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  
    "max_tokens": 6_000,  
}

model_params = {
    "max_tokens": 2_000,  
    "temperature": 0.0,
}

new_response_type = (
    "Provide the final answer in the following exact format on a single line:\n"
    "FINAL_ANSWER=<answer1>\n"
    "Rules:\n"
    "- Output must be exactly one line.\n"
    "- Each answer must be 1–5 words.\n"
    "- Do NOT add spaces before or after the semicolon.\n"
    "- Do NOT use commas, slashes, 'or', 'and', or parentheses.\n"
    "- No explanation, citations, or extra punctuation.\n"
    "- If unknown, output exactly: FINAL_ANSWER=Unknown"
)

search_engine = LocalSearch(
    model=chat_model,
    context_builder=context_builder,
    tokenizer=tokenizer,
    model_params=model_params,
    context_builder_params=local_context_params,
    response_type=new_response_type,
)

from openai import OpenAI
client = OpenAI()

def judge_qa_with_llm(question, prediction_answer, gold_answer, prompt_path="QA_judge.md", model_name="gpt-5-mini"):

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

        if prediction_answer.strip() == "":
            return 0

        user_content = (prompt_template
                        .replace("{{question}}", question)
                        .replace("{{gold_answer}}", gold_answer)
                        .replace("{{model_answer}}", prediction_answer))

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

        print(f"gold: {gold_answer} model: {prediction_answer} -> flag: {flag}")
        return flag


async def train(question_file:str):

    df = pd.read_csv(question_file)
    questions = df["question"].tolist()  
    gold_answers = df["answer"].tolist()
    qids = df["qid"].tolist()

    i = 0
    for question in questions:
        print(i)
        result = await search_engine.search(question)
        tmp_answer = result.response.replace("\n", " ")
        match = re.search(r'FINAL_ANSWER\s*=\s*([^\n\r]+)', tmp_answer)
        tmp_answer = match.group(1).strip() if match else None
        accuracy_flag = judge_qa_with_llm(question, tmp_answer, gold_answers[i])
        reranker.update_last_bag(int(accuracy_flag))
    reranker.save(output_dir)
  
from online_ce_bag_reranker import OnlineCEBagReranker
from graphrag.query.context_builder import entity_extraction


reranker = OnlineCEBagReranker(
    entities_parquet=enttiy_file_path,
    model_dir=None,
    device="cpu",
    max_len=256,
    pair_batch_size=16,
    lr=2e-5,
    replay_size=400,
    replay_p=0.3,
    replay_balance=0.5,
    topk_pool=5,
    pool_mode="soft_topk",
    soft_topk_tau=0.7,
    neg_update_min_prob=0.0,
    debug=True,
    flush_every=1,
)
reranker.set_train_mode(True)
entity_extraction.set_global_seed_reranker(reranker)
asyncio.run(get_responses(train_data_file)) 
