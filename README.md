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
