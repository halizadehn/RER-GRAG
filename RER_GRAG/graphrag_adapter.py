from __future__ import annotations

from typing import Any, Callable

from graphrag.query.structured_search.local_search import mixed_context


_ORIGINAL_MAP_QUERY_TO_ENTITIES: Callable[..., Any] = mixed_context.map_query_to_entities
_INSTALLED = False


def install_entity_reranking(
    reranker: Any,
    *,
    oversampling_factor: int = 4,
) -> None:
    """
    Install the RER-GRAG entity-selection modification.

    GraphRAG remains configured with the final entity budget K. This adapter
    retrieves T = alpha*K candidates with the original GraphRAG entity mapper,
    reranks the T candidates, and returns only the reranked top-K entities to
    the unchanged graph-expansion and context-building stages.
    """
    global _INSTALLED

    if _INSTALLED:
        remove_entity_reranking()

    alpha = int(oversampling_factor)
    if alpha <= 1:
        raise ValueError("oversampling_factor must be > 1.")

    def patched_map_query_to_entities(*args: Any, **kwargs: Any):
        if "k" not in kwargs:
            raise RuntimeError(
                "GraphRAG called map_query_to_entities without keyword argument 'k'."
            )

        final_k = int(kwargs["k"])
        candidate_k = alpha * final_k

        expanded_kwargs = dict(kwargs)
        expanded_kwargs["k"] = candidate_k

        candidates = _ORIGINAL_MAP_QUERY_TO_ENTITIES(*args, **expanded_kwargs)

        query = kwargs.get("query")
        if query is None:
            raise RuntimeError("GraphRAG entity mapping did not provide the query.")

        return reranker.rerank(str(query), list(candidates), final_k)

    mixed_context.map_query_to_entities = patched_map_query_to_entities
    _INSTALLED = True


def remove_entity_reranking() -> None:
    """Restore the original GraphRAG entity-mapping function."""
    global _INSTALLED
    mixed_context.map_query_to_entities = _ORIGINAL_MAP_QUERY_TO_ENTITIES
    _INSTALLED = False
