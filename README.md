# Query Perturbation Robustness in GraphRAG: Analysis and Mitigation through Entity Reranking

This repository provides the implementation and experimental resources for studying the robustness of Graph-based Retrieval-Augmented Generation (GraphRAG) under query perturbations and mitigating perturbation-induced failures through entity reranking. The study evaluates character-, word-, and sentence-level query variations on HotpotQA and 2WikiMultiHopQA and introduces RER-GRAG and its outcome-supervised trained variant, Tr-RER-GRAG, which rerank an expanded candidate entity pool before graph expansion. 

## Repository Structure

```text
├── Data/
│   ├── HotpotQA/
│   │   ├── KG/
│   │   ├── Splits/
│   │   └── questions_answers.csv
│   └── 2WikiMultiHopQA/
│       ├── KG/
│       ├── Splits/
│       └── questions_answers.csv
│
├── Perturbation/
│   ├── character_level.py
│   ├── word_level.py
│   ├── sentence_level.py
│   ├── perturbation_generation.py
│   ├── HotpotQA/
│   └── 2WikiMultiHopQA/
│
├── RER_GRAG/
│   ├── config.yaml
│   ├── GRAG_settings.yaml
│   ├── reranker.py
│   ├── graphrag_adapter.py
│   ├── pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   ├── qa_judge.py
│   └── QA_judge.md
│
├── Analysis/
│   ├── extract_flip_cases.py
│   ├── extract_supporting_entities.py
│   ├── candidate_pool_analysis.py
│   └── build_oracle_seeds.py
│   └── Files/
│
└── README.md
```

## Query Perturbation Generation

The experiments use 12 query perturbations organized into three linguistic levels:

- **Character:** `NeighborSwap`, `RandSwap`, `AdjacentSwap`, `RandDeletion`
- **Word:** `SynonymSwap`, `NeutralInsertion`, `NeighborSwap`, `NeutralDeletion`
- **Sentence:** `Paraphrase`, `Formality`, `Politeness`, `Readability`

Generate all perturbations for a dataset:

```bash
python Perturbation/perturbation_generation.py \
  --input Data/<DATASET>/questions_answers.csv \
  --output-dir Perturbation/<DATASET> \
  --level all
```

Generate a single perturbation level:

```bash
python Perturbation/perturbation_generation.py \
  --input Data/<DATASET>/questions_answers.csv \
  --output-dir Perturbation/<DATASET> \
  --level character
```

Generate a single perturbation type:

```bash
python Perturbation/perturbation_generation.py \
  --input Data/<DATASET>/questions_answers.csv \
  --output-dir Perturbation/<DATASET> \
  --level character \
  --type RandSwap
```

## RER-GRAG and Tr-RER-GRAG

RER-GRAG modifies only the entity-selection stage of GraphRAG. The adapter in `RER_GRAG/graphrag_adapter.py` intercepts entity mapping, retrieves an expanded entity pool using the original GraphRAG mapper, applies the cross-encoder reranker, and returns the reranked top-K entities to the unchanged graph-expansion and context-building stages.

Tr-RER-GRAG uses the same inference pipeline and further optimizes the cross-encoder using outcome-level answer correctness. The candidate pool is treated as a multiple-instance bag, the highest-scoring `K` candidates are aggregated with soft top-K pooling, and the reranker is updated using binary cross-entropy loss.

Set the OpenAI API key before running the pipeline:

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="<YOUR_API_KEY>"
```

### Train Tr-RER-GRAG

```bash
python RER_GRAG/train.py \
  --config RER_GRAG/config.yaml \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/train_with_perturbations.csv \
  --output-dir Models/<DATASET>/Tr-RER-GRAG
```

### Evaluate

Evaluate the original GraphRAG pipeline:

```bash
python RER_GRAG/evaluate.py \
  --method graphrag \
  --config RER_GRAG/config.yaml \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/test.csv \
  --output Results/<DATASET>/GraphRAG_clean.csv
```

Evaluate RER-GRAG:

```bash
python RER_GRAG/evaluate.py \
  --method rer-grag \
  --config RER_GRAG/config.yaml \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/test.csv \
  --output Results/<DATASET>/RER_GRAG_clean.csv
```

Evaluate Tr-RER-GRAG:

```bash
python RER_GRAG/evaluate.py \
  --method tr-rer-grag \
  --config RER_GRAG/config.yaml \
  --kg-dir Data/<DATASET>/KG \
  --input Data/<DATASET>/Splits/test.csv \
  --checkpoint Models/<DATASET>/Tr-RER-GRAG \
  --output Results/<DATASET>/Tr_RER_GRAG_clean.csv
```

The same command can evaluate a released perturbation file. When `perturbed_question` is present, it is selected automatically:

```bash
python RER_GRAG/evaluate.py \
  --method tr-rer-grag \
  --config RER_GRAG/config.yaml \
  --kg-dir Data/<DATASET>/KG \
  --input Perturbation/<DATASET>/character/RandSwap_test.csv \
  --checkpoint Models/<DATASET>/Tr-RER-GRAG \
  --output Results/<DATASET>/Tr_RER_GRAG_RandSwap.csv
```

Replace `<DATASET>` with `HotpotQA` or `2WikiMultiHopQA`. 

## Reproducibility

For exact reproduction of the reported results, use the released:

- fixed train/test splits in `Data/<DATASET>/Splits/`;
- GraphRAG knowledge-graph artifacts in `Data/<DATASET>/KG/`;
- fixed perturbation files in `Perturbation/<DATASET>/`;
- method and GraphRAG configuration files in `RER_GRAG/`.
