"""Evaluation harness.

Measures the things that actually matter for an evidence-grounded system:

  citation_accuracy   - of the source ids cited, how many were genuinely retrieved?
                        A fabricated id scores 0. This is the anti-hallucination metric.
  unsupported_rate    - share of ranked strategies asserted with no supporting source.
  schema_success      - did the backend return output satisfying the Pydantic contract?
  uncertainty_report  - did it declare missing information AND next experiments?
  retrieval_hit       - did top-k contain a document tagged with the compound's
                        literature-documented strategy? (weak relevance signal)
  risk_agreement      - did the solubility risk call match the expected BCS-based band?

The three system configurations are ablations, so the table shows what retrieval and
descriptors each contribute rather than asserting a single score:

  llm_only  - no evidence, no descriptors
  llm_rag   - evidence, no descriptors
  full      - evidence + descriptors + solubility model
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .llm_agent import ConfigurationError, FormulationAgent
from .retrieval import get_index

ROOT = Path(__file__).resolve().parents[1]
CASES_CSV = ROOT / "data" / "evaluation_cases.csv"
RESULTS_DIR = ROOT / "results"

CONFIGS = ["llm_only", "llm_rag", "full"]


def load_cases(path: Path = CASES_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@dataclass
class CaseOutcome:
    name: str
    config: str
    schema_ok: bool
    # None when the system cited nothing at all. A system that makes no citations has
    # undefined citation accuracy, not perfect citation accuracy — averaging a free 1.0
    # into the LLM-only row would make the ungrounded baseline look flawless.
    citation_accuracy: float | None
    n_cited: int
    n_hallucinated: int
    unsupported_rate: float
    uncertainty_ok: bool
    retrieval_hit: bool
    risk_agreement: bool
    predicted_risk: str = ""
    expected_risk: str = ""
    top_strategy: str = ""
    error: str = ""


def _retrieval_hit(retrieved, reference_strategy: str) -> bool:
    """Did we surface at least one document tagged with the documented strategy?"""
    if reference_strategy == "none required":
        return True  # nothing to hit; not a retrieval failure
    tags = {t for r in retrieved for t in r.doc.tags}
    # 'nanosuspension' and 'particle size reduction' are the same intervention family.
    families = {
        "particle size reduction": {"particle size reduction", "nanosuspension"},
        "nanosuspension": {"nanosuspension", "particle size reduction"},
        "supersaturating formulation": {
            "supersaturating formulation",
            "precipitation inhibition",
        },
    }
    wanted = families.get(reference_strategy, {reference_strategy})
    return bool(tags & wanted)


def evaluate_case(agent: FormulationAgent, case: dict, config: str) -> CaseOutcome:
    try:
        result = agent.run(smiles=case["smiles"], name=case["name"], prompt_mode=config)
    except ConfigurationError:
        # Never swallow a setup error into a per-case result. Doing so turns "your key
        # is wrong" into a plausible-looking table of 0% scores attributed to the model.
        raise
    except Exception as exc:  # noqa: BLE001 - a failed run is a real result
        return CaseOutcome(
            name=case["name"],
            config=config,
            schema_ok=False,
            citation_accuracy=None,
            n_cited=0,
            n_hallucinated=0,
            unsupported_rate=1.0,
            uncertainty_ok=False,
            retrieval_hit=False,
            risk_agreement=False,
            error=str(exc)[:200],
        )

    a, g = result.assessment, result.grounding
    strategies = a.ranked_strategies

    return CaseOutcome(
        name=case["name"],
        config=config,
        schema_ok=True,  # reaching here means Pydantic validation passed
        citation_accuracy=g.citation_accuracy if g.cited_ids else None,
        n_cited=len(g.cited_ids),
        n_hallucinated=len(g.hallucinated_ids),
        # No strategies is a legitimate answer for a soluble compound, not 100%
        # unsupported. Only count strategies that were actually asserted.
        unsupported_rate=(
            len(g.uncited_strategies) / len(strategies) if strategies else 0.0
        ),
        uncertainty_ok=bool(
            a.missing_information and a.recommended_experiments and a.overall_uncertainty
        ),
        retrieval_hit=_retrieval_hit(result.retrieved, case["reference_strategy"]),
        risk_agreement=a.compound_summary.solubility_risk.value == case["expected_risk"],
        predicted_risk=a.compound_summary.solubility_risk.value,
        expected_risk=case["expected_risk"],
        top_strategy=strategies[0].strategy if strategies else "",
    )


@dataclass
class ConfigSummary:
    config: str
    n: int
    citation_accuracy: float | None  # None when no run in this config cited anything
    n_cases_citing: int
    unsupported_rate: float
    schema_success: float
    uncertainty_report: float
    retrieval_hit: float
    risk_agreement: float
    total_hallucinated: int
    outcomes: list[CaseOutcome] = field(default_factory=list)


def summarise(outcomes: list[CaseOutcome], config: str) -> ConfigSummary:
    rows = [o for o in outcomes if o.config == config]
    ok = [o for o in rows if o.schema_ok]
    citing = [o for o in ok if o.citation_accuracy is not None]
    return ConfigSummary(
        config=config,
        n=len(rows),
        citation_accuracy=mean([o.citation_accuracy for o in citing]) if citing else None,
        n_cases_citing=len(citing),
        unsupported_rate=mean([o.unsupported_rate for o in ok]) if ok else 1.0,
        schema_success=len(ok) / len(rows) if rows else 0.0,
        uncertainty_report=mean([o.uncertainty_ok for o in ok]) if ok else 0.0,
        retrieval_hit=mean([o.retrieval_hit for o in ok]) if ok else 0.0,
        risk_agreement=mean([o.risk_agreement for o in ok]) if ok else 0.0,
        total_hallucinated=sum(o.n_hallucinated for o in rows),
        outcomes=rows,
    )


def stability_check(agent: FormulationAgent, case: dict, repeats: int = 3) -> dict:
    """Do repeated runs agree on the top-ranked strategy?

    'No strategy recommended' is a real, and for a soluble compound correct, answer —
    it is recorded as <none> so that consistently recommending nothing counts as
    stable rather than as a failure to agree.
    """
    tops: list[str] = []
    for _ in range(repeats):
        try:
            r = agent.run(smiles=case["smiles"], name=case["name"], prompt_mode="full")
            strategies = r.assessment.ranked_strategies
            tops.append(strategies[0].strategy if strategies else "<none>")
        except Exception:  # noqa: BLE001
            tops.append("<error>")
    if not tops:
        return {"name": case["name"], "agreement": 0.0, "modal": "", "runs": []}
    modal, count = Counter(tops).most_common(1)[0]
    return {
        "name": case["name"],
        "agreement": count / len(tops),
        "modal": modal,
        "runs": tops,
    }


def run_evaluation(
    backend: str = "auto",
    configs: list[str] | None = None,
    stability_repeats: int = 3,
) -> dict:
    configs = configs or CONFIGS
    agent = FormulationAgent(backend=backend)
    cases = load_cases()

    outcomes: list[CaseOutcome] = []
    for config in configs:
        print(f"\n[{config}]", flush=True)
        for case in cases:
            outcome = evaluate_case(agent, case, config)
            outcomes.append(outcome)
            flag = "ok " if outcome.schema_ok else "ERR"
            cite = (
                "n/a " if outcome.citation_accuracy is None else f"{outcome.citation_accuracy:.2f}"
            )
            print(
                f"  {flag} {case['name']:15s} cite={cite} "
                f"halluc={outcome.n_hallucinated} risk={outcome.predicted_risk}"
                f"{'' if outcome.risk_agreement else ' (exp ' + outcome.expected_risk + ')'}",
                flush=True,
            )

    summaries = [summarise(outcomes, c) for c in configs]

    print("\n[stability]", flush=True)
    stability = [stability_check(agent, c, stability_repeats) for c in cases]
    for s in stability:
        print(f"  {s['name']:15s} agreement={s['agreement']:.2f}  top={s['modal']}", flush=True)

    return {
        "backend": agent.mode_label,
        "retriever": get_index().backend,
        "n_cases": len(cases),
        "summaries": [
            {k: v for k, v in vars(s).items() if k != "outcomes"} for s in summaries
        ],
        "stability_mean_agreement": mean([s["agreement"] for s in stability]) if stability else 0.0,
        "stability": stability,
        "outcomes": [vars(o) for o in outcomes],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def to_markdown(report: dict) -> str:
    is_rulebased = "rule-based" in report["backend"]

    lines = [
        "# Evaluation results",
        "",
        f"- Backend: **{report['backend']}**",
        f"- Retriever: **{report['retriever']}**",
        f"- Cases: **{report['n_cases']}**",
        "",
    ]

    if is_rulebased:
        lines += [
            "> **These numbers come from the deterministic rule-based backend, not an LLM.**",
            "> Read them as a floor and a harness self-test, not as a result about LLM behaviour:",
            ">",
            "> - The rule-based backend cites by construction (strategy -> tag -> retrieved id),",
            ">   so its citation accuracy is trivially 100% and says nothing about whether an",
            ">   LLM would fabricate sources.",
            "> - It always uses descriptors, so `LLM + retrieval` and `Retrieval + descriptors + LLM`",
            ">   are the same system here and their rows are necessarily identical.",
            "> - It is deterministic, so stability is trivially 100%.",
            ">",
            "> Set `ANTHROPIC_API_KEY` and re-run to populate the LLM rows for real:",
            "> `python -m scripts.run_evaluation --backend anthropic`",
            "",
        ]

    lines += [
        "| System | Citation accuracy | Unsupported claims | Structured-output success | "
        "Uncertainty reported | Retrieval hit | Risk agreement | Fabricated citations |",
        "|---|---|---|---|---|---|---|---|",
    ]

    labels = {
        "llm_only": "LLM alone",
        "llm_rag": "LLM + retrieval",
        "full": "Retrieval + descriptors + LLM",
    }
    for s in report["summaries"]:
        lines.append(
            f"| {labels.get(s['config'], s['config'])} "
            f"| {_pct(s['citation_accuracy'])} "
            f"| {_pct(s['unsupported_rate'])} "
            f"| {_pct(s['schema_success'])} "
            f"| {_pct(s['uncertainty_report'])} "
            f"| {_pct(s['retrieval_hit'])} "
            f"| {_pct(s['risk_agreement'])} "
            f"| {s['total_hallucinated']} |"
        )

    lines += [
        "",
        "`n/a` citation accuracy means the configuration cited nothing at all, which is "
        "undefined rather than perfect.",
        "",
        f"Top-strategy stability across repeated runs: "
        f"**{_pct(report['stability_mean_agreement'])}** mean agreement"
        + (" (trivially 100% for a deterministic backend)." if is_rulebased else "."),
        "",
    ]
    return "\n".join(lines)


def save(report: dict) -> None:
    """Write results, refusing to publish a run in which nothing actually worked.

    A run where every case errored produces a table of 0% scores that reads as a
    finding about the model. It is not — it is a broken run, and overwriting good
    results with it destroys real data.
    """
    succeeded = sum(1 for o in report["outcomes"] if o["schema_ok"])
    if succeeded == 0:
        first_error = next(
            (o["error"] for o in report["outcomes"] if o.get("error")), "unknown error"
        )
        raise RuntimeError(
            f"Every one of {len(report['outcomes'])} runs failed — refusing to overwrite "
            f"results/ with a table of zeros.\nFirst error: {first_error}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2))
    (RESULTS_DIR / "evaluation.md").write_text(to_markdown(report))
