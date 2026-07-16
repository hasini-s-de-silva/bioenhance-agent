# Evaluation results

- Backend: **rule-based baseline (no LLM)**
- Retriever: **tf-idf (sparse fallback)**
- Cases: **12**

> **These numbers come from the deterministic rule-based backend, not an LLM.**
> Read them as a floor and a harness self-test, not as a result about LLM behaviour:
>
> - The rule-based backend cites by construction (strategy -> tag -> retrieved id),
>   so its citation accuracy is trivially 100% and says nothing about whether an
>   LLM would fabricate sources.
> - It always uses descriptors, so `LLM + retrieval` and `Retrieval + descriptors + LLM`
>   are the same system here and their rows are necessarily identical.
> - It is deterministic, so stability is trivially 100%.
>
> Set `ANTHROPIC_API_KEY` and re-run to populate the LLM rows for real:
> `python -m scripts.run_evaluation --backend anthropic`

| System | Citation accuracy | Unsupported claims | Structured-output success | Uncertainty reported | Retrieval hit | Risk agreement | Fabricated citations |
|---|---|---|---|---|---|---|---|
| LLM alone | n/a | 92% | 100% | 100% | 17% | 75% | 0 |
| LLM + retrieval | 100% | 42% | 100% | 100% | 83% | 75% | 0 |
| Retrieval + descriptors + LLM | 100% | 42% | 100% | 100% | 83% | 75% | 0 |

`n/a` citation accuracy means the configuration cited nothing at all, which is undefined rather than perfect.

Top-strategy stability across repeated runs: **100%** mean agreement (trivially 100% for a deterministic backend).
