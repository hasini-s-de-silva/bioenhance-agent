# Evaluation results

- Backend: **local LLM (qwen2.5:7b via Ollama)**
- Retriever: **sentence-transformers/all-MiniLM-L6-v2 + faiss**
- Cases: **12**

| System | Citation accuracy | Unsupported claims | Structured-output success | Uncertainty reported | Retrieval hit | Risk agreement | Fabricated citations |
|---|---|---|---|---|---|---|---|
| LLM alone | n/a | 100% | 100% | 100% | 17% | 67% | 0 |
| LLM + retrieval | 96% | 50% | 100% | 100% | 58% | 33% | 1 |
| Retrieval + descriptors + LLM | 100% | 25% | 100% | 92% | 58% | 75% | 0 |

`n/a` citation accuracy means the configuration cited nothing at all, which is undefined rather than perfect.

Top-strategy stability across repeated runs: **100%** mean agreement.
