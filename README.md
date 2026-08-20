# Query Perturbation Robustness in GraphRAG: Analysis and Mitigation through Entity Reranking

This repository provides the implementation and experimental resources for studying the robustness of Graph-based Retrieval-Augmented Generation (GraphRAG) under query perturbations and mitigating perturbation-induced failures through entity reranking. The study evaluates character-, word-, and sentence-level query variations on HotpotQA and 2WikiMultiHopQA and introduces RER-GRAG and its outcome-supervised trained variant, Tr-RER-GRAG, which rerank an expanded candidate entity pool before graph expansion. 

## Query Perturbation Generation

The experiments use 12 query perturbations organized into three linguistic levels:

- **Character:** `NeighborSwap`, `RandSwap`, `AdjacentSwap`, `RandDeletion`
- **Word:** `SynonymSwap`, `NeutralInsertion`, `NeighborSwap`, `NeutralDeletion`
- **Sentence:** `Paraphrase`, `Formality`, `Politeness`, `Readability`

Generate all perturbations for a dataset:

```bash
python Perturbations/perturbation_generation.py \
  --input Data/<DATASET>/questions_answers.csv \
  --output-dir Perturbations/<DATASET> \
  --level all
```

Generate a single perturbation level:

```bash
python Perturbations/perturbation_generation.py \
  --input Data/<DATASET>/questions_answers.csv \
  --output-dir Perturbations/<DATASET> \
  --level character
```

Generate a single perturbation type:

```bash
python Perturbations/perturbation_generation.py \
  --input Data/<DATASET>/questions_answers.csv \
  --output-dir Perturbations/<DATASET> \
  --level character \
  --type RandSwap
```

## RER-GRAG and Tr-RER-GRAG

RER-GRAG modifies only the entity-selection stage of GraphRAG. The adapter in `RER_GRAG/graphrag_adapter.py` intercepts entity mapping, retrieves an expanded pool of `T = alpha*K ` entities using the original GraphRAG mapper, applies the cross-encoder reranker, and returns the reranked top-5 entities to the unchanged graph-expansion and context-building stages.

Tr-RER-GRAG uses the same inference pipeline and further optimizes the cross-encoder using outcome-level answer correctness. The candidate pool is treated as a multiple-instance bag, the highest-scoring `K` candidates are aggregated with soft top-K pooling, and the reranker is updated using binary cross-entropy loss.

The default configuration follows the manuscript settings: `K=5`, oversampling factor `alpha=4`, `BAAI/bge-reranker-base`, learning rate `1e-6`, replay buffer size `200`, and replay probability `0.3`.

```bash
python RER_GRAG/train.py \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/train.csv 
```

### Evaluate

Evaluate the original GraphRAG pipeline:

```bash
python RER_GRAG/evaluate.py \
  --method graphrag \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/test.csv 
```

Evaluate RER-GRAG:

```bash
python RER_GRAG/evaluate.py \
  --method rer-grag \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/test.csv \
  --output Results/<DATASET>/RER_GRAG_clean.csv
```

Evaluate Tr-RER-GRAG:

```bash
python RER_GRAG/evaluate.py \
  --method tr-rer-grag \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/test.csv \
  --checkpoint Models/<DATASET>/Tr-RER-GRAG \
  --output Results/<DATASET>/Tr_RER_GRAG_clean.csv
```

