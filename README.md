# Understanding and Mitigating Query Perturbation Failures in Graph-based RAG


## 📌 Overview

Graph-based Retrieval-Augmented Generation (GraphRAG) has emerged as a promising framework for multi-hop question answering by integrating graph-structured retrieval with large language model (LLM) generation. 
Nevertheless, its robustness to query perturbations remains underexplored, even though real-world information needs are frequently expressed through noisy, reformulated, or stylistically diverse queries. 
This research is driven by two central objectives:
- to quantify the impact of different types of query perturbations on GraphRAG performance, and
- to mitigate perturbation-induced failures specific to the GraphRAGpipeline by addressing them at early stages, before they propagate to downstream components such as graph expansion.


## 🚀 Key Contributions

- Comprehensive robustness evaluation under:
  - Character-level perturbations  
  - Word-level perturbations  
  - Sentence-level perturbations  

- A **cross-encoder-based reranking approach** for improving entity selection  

- An **outcome-supervised training strategy** for reranker optimization  

- Extensive experiments on:
  - **HotpotQA**
  - **2WikiMultiHopQA**


## ⚙️ Method Overview

#### 🔹 RER-GRAG
- Expands candidate entity pool (oversampling)
- Applies cross-encoder reranking 
- Selects top-$K$ entities for graph expansion

#### 🔹 Tr-RER-GRAG
- Extends RER-GRAG with outcome-supervised training
- Optimizes reranker using answer correctness signals


## 📊 Evaluation

### Datasets
- HotpotQA 
- 2WikiMultiHopQA

### Perturbation Types
- Character-level (e.g., RandSwap, NeighborSwap)
- Word-level (e.g., SynonymSwap, NeutralInsertion)
- Sentence-level (e.g., Paraphrase, Formality)

### Metrics
- Accuracy (ACC)
