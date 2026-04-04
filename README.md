# Understanding and Mitigating Query Perturbation Failures in Graph-based RAG


## 📌 Overview

Graph-based Retrieval-Augmented Generation (GraphRAG) has emerged as a promising framework for multi-hop question answering by integrating graph-structured retrieval with large language model (LLM) generation. 
Nevertheless, its robustness to query perturbations remains underexplored, even though real-world information needs are frequently expressed through noisy, reformulated, or stylistically diverse queries. 
This research is driven by two central objectives:
- (1) to quantify the impact of different types of query perturbations on GraphRAG performance, and
- (2) to mitigate perturbation-induced failures specific to the GraphRAGpipeline by addressing them at early stages, before they propagate to downstream components such as graph expansion.


## 🚀 Key Contributions

- Comprehensive robustness evaluation under:
  - Character-level perturbations  
  - Word-level perturbations  
  - Sentence-level perturbations  

- A **cross-encoder-based reranking framework** for improving entity selection  

- An **outcome-supervised training strategy** for reranker optimization  

- Extensive experiments on:
  - **HotpotQA**
  - **2WikiMultiHopQA**

---

## 📂 Repository Structure
