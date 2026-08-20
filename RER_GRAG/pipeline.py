from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_text_units,
)
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag.vector_stores.lancedb import LanceDBVectorStore


REQUIRED_KG_FILES = (
    "entities.parquet",
    "communities.parquet",
    "relationships.parquet",
    "text_units.parquet",
)


def _require_kg_files(kg_dir: Path) -> None:
    missing = [name for name in REQUIRED_KG_FILES if not (kg_dir / name).exists()]
    if not (kg_dir / "lancedb").exists():
        missing.append("lancedb/")
    if missing:
        raise FileNotFoundError(
            f"Missing GraphRAG artifact(s) under {kg_dir}: {', '.join(missing)}"
        )


def build_search_engine(kg_dir: str, config: Dict[str, Any]) -> LocalSearch:
    """
    Build the Microsoft GraphRAG local-search pipeline from released KG artifacts.

    The API layout follows the GraphRAG version used by the original project
    code. Pin that exact GraphRAG version in the repository environment file.
    """
    kg_path = Path(kg_dir)
    _require_kg_files(kg_path)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    grag = config["graphrag"]
    community_level = int(grag.get("community_level", 0))

    entity_df = pd.read_parquet(kg_path / "entities.parquet")
    community_df = pd.read_parquet(kg_path / "communities.parquet")
    relationship_df = pd.read_parquet(kg_path / "relationships.parquet")
    text_unit_df = pd.read_parquet(kg_path / "text_units.parquet")

    entities = read_indexer_entities(entity_df, community_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    text_units = read_indexer_text_units(text_unit_df)

    description_embedding_store = LanceDBVectorStore(
        vector_store_schema_config=VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
    )
    description_embedding_store.connect(db_uri=str(kg_path / "lancedb"))

    chat_config = LanguageModelConfig(
        api_key=api_key,
        type=ModelType.Chat,
        model_provider="openai",
        model=str(grag["llm_model"]),
        max_retries=20,
    )
    chat_model = ModelManager().get_or_create_chat_model(
        name="rer_grag_local_search",
        model_type=ModelType.Chat,
        config=chat_config,
    )

    embedding_config = LanguageModelConfig(
        api_key=api_key,
        type=ModelType.Embedding,
        model_provider="openai",
        model=str(grag["embedding_model"]),
        max_retries=20,
    )
    text_embedder = ModelManager().get_or_create_embedding_model(
        name="rer_grag_embedding",
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

    context_builder_params = {
        "text_unit_prop": float(grag["text_unit_prop"]),
        "community_prop": float(grag["community_prop"]),
        "conversation_history_max_turns": 0,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": int(grag["selected_entities"]),
        "top_k_relationships": int(grag["top_k_relationships"]),
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": int(grag["max_context_tokens"]),
    }

    model_params = {
        "max_tokens": int(grag["max_answer_tokens"]),
        "temperature": float(grag["temperature"]),
    }

    response_type = (
        "Provide the final answer in the following exact format on a single line:\n"
        "FINAL_ANSWER=<answer>\n"
        "Rules:\n"
        "- Output exactly one FINAL_ANSWER line.\n"
        "- Give only the answer, without explanation or citations.\n"
        "- If unknown, output FINAL_ANSWER=Unknown."
    )

    return LocalSearch(
        model=chat_model,
        context_builder=context_builder,
        tokenizer=tokenizer,
        model_params=model_params,
        context_builder_params=context_builder_params,
        response_type=response_type,
    )
