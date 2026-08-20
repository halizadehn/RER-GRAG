from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def parse_list(value: Any) -> list[str]:
    """Parse list-valued CSV/Parquet cells robustly."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return [str(x) for x in value.tolist() if x is not None]

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed if x is not None]
        except Exception:
            pass

    # Final fallback for simple delimiter-separated fields.
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def choose_column(df: pd.DataFrame, candidates: Iterable[str], table_name: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find a suitable column in {table_name}. "
        f"Tried {list(candidates)}; available columns: {list(df.columns)}"
    )


def read_evidence(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path, dtype={"qid": str})


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive document-grounded supporting KG entities from benchmark gold "
            "supporting documents: gold documents -> GraphRAG documents -> text units -> entities."
        )
    )
    parser.add_argument(
        "--evidence",
        required=True,
        help=(
            "CSV/JSON/JSONL with qid and a list-valued supporting-title column "
            "(default column: supporting_titles)."
        ),
    )
    parser.add_argument(
        "--kg-dir",
        required=True,
        help="Directory containing GraphRAG parquet outputs.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--qid-column", default="qid")
    parser.add_argument("--supporting-title-column", default="supporting_titles")
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    kg_dir = Path(args.kg_dir)

    documents_path = kg_dir / "documents.parquet"
    text_units_path = kg_dir / "text_units.parquet"
    entities_path = kg_dir / "entities.parquet"

    for path in (documents_path, text_units_path, entities_path):
        if not path.exists():
            raise FileNotFoundError(f"Required GraphRAG file not found: {path}")

    evidence = read_evidence(evidence_path)
    if args.qid_column not in evidence.columns:
        raise ValueError(f"Evidence file must contain '{args.qid_column}'.")
    if args.supporting_title_column not in evidence.columns:
        raise ValueError(
            f"Evidence file must contain '{args.supporting_title_column}' as a list "
            "or JSON-encoded list of gold supporting document titles."
        )

    documents = pd.read_parquet(documents_path)
    text_units = pd.read_parquet(text_units_path)
    entities = pd.read_parquet(entities_path)

    doc_id_col = choose_column(documents, ["id", "document_id"], "documents.parquet")
    doc_title_col = choose_column(
        documents, ["title", "name", "document_title"], "documents.parquet"
    )
    tu_id_col = choose_column(text_units, ["id", "text_unit_id"], "text_units.parquet")
    tu_doc_col = choose_column(
        text_units, ["document_ids", "document_id"], "text_units.parquet"
    )
    tu_entity_col = choose_column(
        text_units, ["entity_ids", "entities"], "text_units.parquet"
    )
    entity_id_col = choose_column(entities, ["id", "entity_id"], "entities.parquet")

    entity_title_col = next(
        (c for c in ["title", "name", "description"] if c in entities.columns),
        None,
    )

    document_lookup: dict[str, list[str]] = {}
    for _, row in documents.iterrows():
        key = normalize_title(row[doc_title_col])
        document_lookup.setdefault(key, []).append(str(row[doc_id_col]))

    text_units_by_document: dict[str, list[tuple[str, list[str]]]] = {}
    for _, row in text_units.iterrows():
        text_unit_id = str(row[tu_id_col])
        document_ids = parse_list(row[tu_doc_col])
        entity_ids = parse_list(row[tu_entity_col])
        for document_id in document_ids:
            text_units_by_document.setdefault(str(document_id), []).append(
                (text_unit_id, entity_ids)
            )

    entity_name_lookup = {}
    if entity_title_col:
        entity_name_lookup = {
            str(row[entity_id_col]): str(row[entity_title_col])
            for _, row in entities.iterrows()
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    unmatched_titles = 0
    records_written = 0

    with output.open("w", encoding="utf-8") as handle:
        for _, row in evidence.iterrows():
            qid = str(row[args.qid_column])
            supporting_titles = parse_list(row[args.supporting_title_column])

            supporting_document_ids: list[str] = []
            missing_titles: list[str] = []

            for title in supporting_titles:
                matches = document_lookup.get(normalize_title(title), [])
                if not matches:
                    missing_titles.append(title)
                    unmatched_titles += 1
                supporting_document_ids.extend(matches)

            supporting_document_ids = list(dict.fromkeys(supporting_document_ids))

            supporting_text_unit_ids: list[str] = []
            supporting_entity_ids: list[str] = []

            for document_id in supporting_document_ids:
                for text_unit_id, entity_ids in text_units_by_document.get(document_id, []):
                    supporting_text_unit_ids.append(text_unit_id)
                    supporting_entity_ids.extend(entity_ids)

            supporting_text_unit_ids = list(dict.fromkeys(supporting_text_unit_ids))
            supporting_entity_ids = list(dict.fromkeys(supporting_entity_ids))

            record = {
                "qid": qid,
                "supporting_titles": supporting_titles,
                "supporting_document_ids": supporting_document_ids,
                "supporting_text_unit_ids": supporting_text_unit_ids,
                "supporting_entity_ids": supporting_entity_ids,
                "supporting_entity_names": [
                    entity_name_lookup.get(entity_id, "")
                    for entity_id in supporting_entity_ids
                ],
                "unmatched_supporting_titles": missing_titles,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    print(f"Questions processed: {records_written}")
    print(f"Unmatched supporting-document titles: {unmatched_titles}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
