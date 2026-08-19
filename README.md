# Query Perturbation Robustness in GraphRAG: Analysis and Mitigation through Entity Reranking

This repository provides the implementation and experimental resources for studying the robustness of Graph-based Retrieval-Augmented Generation (GraphRAG) under query perturbations and mitigating perturbation-induced failures through entity reranking. The study evaluates character-, word-, and sentence-level query variations on HotpotQA and 2WikiMultiHopQA and introduces RER-GRAG and its outcome-supervised trained variant, Tr-RER-GRAG, which rerank an expanded candidate entity pool before graph expansion. The repository includes the datasets and fixed train/test splits, GraphRAG knowledge-graph artifacts, perturbation-generation scripts and generated perturbation files, reranker training and evaluation code, and scripts required to reproduce the analyses reported in the paper.

## Query Perturbation Generation

The experiments use 12 query perturbations organized into three linguistic levels:

- **Character:** `RandSwap`, `NeighborSwap`, `AdjacentSwap`, `RandDeletion`
- **Word:** `SynonymSwap`, `NeighborSwap`, `NeutralInsertion`, `NeutralDeletion`
- **Sentence:** `Paraphrase`, `Formality`, `Politeness`, `Readability`

