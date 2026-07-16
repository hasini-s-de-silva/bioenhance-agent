# HANDOFF — BioEnhance Agent

Working notes for picking this project back up. Not part of the deliverable; delete it
whenever it stops being useful.

**Repo:** https://github.com/hasini-s-de-silva/bioenhance-agent (public)
**Local:** `~/Desktop/GITHUB/bioenhance-agent`
**Last updated:** 2026-07-16 (evening)

---

## Where things stand

The project is **complete, tested, and pushed**. Everything below is verified, not assumed.

| Piece | State |
|---|---|
| RDKit descriptor tool | Done. Verified against known values (aspirin, caffeine, ibuprofen). |
| ESOL solubility model | Done. Trained on real Delaney data (n=1128). Test RMSE **0.758**, R² **0.878**. |
| Evidence library | Done. **50 real PubMed open-access records**, every PMID verified to resolve. |
| Retrieval | Done. sentence-transformers + FAISS, with a TF-IDF fallback. Now thread-safe (see below). |
| Structured LLM output | Done. Pydantic-validated, with retry on schema failure. |
| Grounding guardrail | Done. `check_grounding()` catches fabricated citations. Tested against a deliberately fabricating model. |
| Streamlit app | Done. Verified in a real browser; `assets/demo.png` is a real screenshot. |
| Tests | **66 pass** offline (68 with `--run-network`). |
| Evaluation harness | Done, with 3 ablations + 12 cases incl. 2 negative controls. |
| **LLM evaluation numbers** | **Done.** Real qwen2.5:7b results are in `results/evaluation.md` and the README. |

## The evaluation is finished — real results are in

`results/evaluation.md` and the README's Evaluation section now hold real qwen2.5:7b
results (local, free, no API key), not the old rule-based placeholder.

Headline findings, measured:

- **`llm_only` calls every one of the 12 compounds "high solubility risk,"** including
  paracetamol and metformin — the two negative controls, which are freely soluble. It scores
  67% risk agreement only because 8 of 12 test compounds genuinely are high-risk. That is
  accuracy with no calibration — the measured version of the README's "Why LLM-only systems
  are insufficient" argument.
- **`llm_rag` (retrieval, no descriptors) made risk agreement *worse*** (33%, down from 67%)
  and produced the run's only fabricated citation, on metformin. Evidence without grounding
  descriptors gave the model something to talk around, not reason from.
- **`full` (descriptors + retrieval + LLM) is the only configuration that correctly calls
  paracetamol "low."** Risk agreement 75%, zero fabricated citations. This is the clean,
  quantified demonstration that the scientific tooling — not the LLM — is what makes the
  system trustworthy.

Caveat to keep: qwen2.5:7b is a small local model, so this shows what *this* model does
ungrounded. Do not overclaim it as a universal law about LLMs.

Reproduce with:

```bash
cd ~/Desktop/GITHUB/bioenhance-agent
source .venv/bin/activate
ollama pull qwen2.5:7b                                    # if not already pulled
python -m scripts.run_evaluation --backend ollama --repeats 2 --workers 3
```

`finish.sh` was removed — it was a one-off wrapper for the hand-off step above, not part of
the project.

---

## Design decisions worth not re-litigating

- **The evidence library is never LLM-written.** Every document is harvested from PubMed
  E-utilities by `scripts/harvest_evidence.py`, with a real PMID and a real abstract.
  Records lacking an abstract/year/journal are rejected at harvest. A formulation tool whose
  citations don't resolve is worse than no tool.
- **The local model is the documented default**, not the paid API. An evaluation that needs a
  credit card is a claim; one anyone can reproduce on a laptop is a result.
- **Recommending nothing is a valid answer.** For a soluble compound the correct output is an
  empty strategy list. Over-recommending for a freely soluble drug is the false positive that
  would most damage trust.
- **Known model limitations are documented, not tuned away.** Metformin is misclassified
  because the ESOL model predicts neutral-species solubility and has no concept of ionisation
  (metformin is a strong base, pKa ~12.4, soluble cation). The thresholds were deliberately
  NOT fitted to the 12-compound test set — that would measure nothing.
- **`n/a` is not 100%.** A system that cites nothing has *undefined* citation accuracy. An
  earlier version scored the ungrounded baseline 100% by giving it a free pass; that was a
  metric bug, and it is now explicitly `n/a`.
- **RDKit's caffeine TPSA (61.82) differs from PubChem's (58.44)** by exactly 3.38 Å² — RDKit
  perceives the two amide ring nitrogens as aromatic. Both are "right"; the test pins RDKit's
  value and documents why. Don't "fix" it.

## Bugs already found and fixed (don't reintroduce)

1. `.env` was only loaded by `app.py`, so the eval silently fell back to rule-based and
   reported it as an LLM run. `load_dotenv()` now lives in `src/llm_agent.py`.
2. Setup errors were swallowed into per-case results, producing a table of 0% scores labelled
   "Backend: LLM" — which overwrote real results. `save()` now refuses to write when zero runs
   succeeded, and `ConfigurationError` aborts instead of being caught.
3. Terminal API errors (401/403/404/billing-400) were retried 3x, tripling failed calls and
   burying the cause under "failed to return schema-valid JSON". Rule now: the *only*
   retryable failure is our own JSON parsing; every `APIStatusError` is terminal.
4. The rule-based backend over-triggered, recommending cocrystals for paracetamol.
5. Small local models ignore an *implied* citation requirement — qwen2.5:7b cited nothing
   until the prompt demanded citations explicitly (measured: 0 → S01+S30).
6. `get_index()` was a lazy singleton with no lock — the evaluation harness calls it from a
   6-worker thread pool, so concurrent callers raced to construct their own
   `SentenceTransformer` instance. Reproducibly segfaulted the process. Fixed with a
   double-checked lock, plus building the index once on the main thread before the pool
   starts.
7. `EvidenceIndex._embed_query()` called the shared model's `.encode()` from multiple
   threads with no synchronisation — also reproducibly segfaulted (this one crashed even
   after fix #6, since the model was already built; the encode *call itself* isn't
   thread-safe here). Fixed with a lock around query embedding.
8. With #6 and #7 fixed, 6 concurrent client requests against Ollama (which serialises
   without `OLLAMA_NUM_PARALLEL` set server-side — see Gotchas) queued long enough to exceed
   the 300s per-request timeout. Fixed by raising `BIOENHANCE_OLLAMA_TIMEOUT` to 900 and
   defaulting `--workers` to a lower count for local-model runs.

## Gotchas

- **Ollama parallelism**: `OLLAMA_NUM_PARALLEL` is a *server* setting. Setting it client-side
  does nothing — requests stay serialised. To actually parallelise, restart the Ollama server
  with it set.
- **`.env` must never be committed.** It's gitignored, and `.githooks/pre-commit` blocks it
  plus any real `sk-ant-` key. Enable once per clone: `git config core.hooksPath .githooks`.
- The trained model (`data/solubility_model.joblib`, 43 MB) is gitignored and rebuilds in
  ~1 min via `python -m scripts.train_solubility`.
- On this Mac use `gtimeout`, not `timeout`.

## Things deliberately NOT done

Per the spec's "features to avoid": no user accounts, no database, no multi-agent
orchestration, no large corpus, no custom-trained foundation model, no bioequivalence claims.
Keep it narrow and scientifically honest.

## Next steps, in value order

1. Add pKa prediction — the single largest source of error, and why metformin fails.
2. Melting point / glass-transition prediction to make ASD recommendations defensible.
3. Expand the library beyond 50 documents; add a cross-encoder reranker.
4. Replace tag-overlap retrieval relevance with expert-annotated judgements.
