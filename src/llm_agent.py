"""The formulation agent: tools -> retrieval -> LLM -> validated, grounded output.

Two backends:

  anthropic  - a real LLM call (requires ANTHROPIC_API_KEY).
  rulebased  - a deterministic, non-LLM baseline used for tests, CI and offline demos.

The rule-based backend is NOT an LLM and never pretends to be. It exists so the
repository runs end-to-end without an API key, and so the test suite is deterministic.
Its outputs are labelled as such everywhere they surface.

Every run — whichever backend — is passed through `check_grounding`, which verifies
that the model cited only source ids that were genuinely retrieved. That check is what
makes "evidence-grounded" a property we test rather than a claim we make.
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from pydantic import ValidationError

# Load .env here rather than only in app.py, so every entry point — the Streamlit
# app, the evaluation harness, a bare `python -c` — sees the key. Without this the
# harness silently falls back to the rule-based backend and reports results that
# look like LLM results but are not.
load_dotenv()

from .descriptors import compute_descriptors, resolve_input  # noqa: E402
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retrieval import build_query, retrieve_relevant_evidence
from .schemas import (
    AgentResult,
    Confidence,
    CompoundSummary,
    FormulationAssessment,
    GroundingReport,
    MolecularDescriptors,
    RankedStrategy,
    RetrievedEvidence,
    SolubilityPrediction,
    SolubilityRisk,
)
from .solubility_model import get_predictor

DEFAULT_MODEL = os.environ.get("BIOENHANCE_MODEL", "claude-sonnet-5")
MAX_TOKENS = 4000

# A real key is a long opaque string. The .env.example placeholder is short and
# contains an ellipsis; copying the example without editing it is the single most
# likely setup mistake, and a 401 on every one of 72 calls is an expensive way to
# discover it.
MIN_KEY_LENGTH = 40


class ConfigurationError(RuntimeError):
    """Setup is wrong in a way no retry can fix — bad key, bad model, no access."""


# --------------------------------------------------------------------------
# Grounding guardrail
# --------------------------------------------------------------------------


def check_grounding(
    assessment: FormulationAssessment, retrieved: list[RetrievedEvidence]
) -> GroundingReport:
    """Verify every citation points at a document we actually supplied."""
    retrieved_ids = {r.doc.id for r in retrieved}
    cited = assessment.cited_ids()

    return GroundingReport(
        cited_ids=sorted(cited),
        retrieved_ids=sorted(retrieved_ids),
        hallucinated_ids=sorted(cited - retrieved_ids),
        uncited_strategies=[
            s.strategy for s in assessment.ranked_strategies if not s.supporting_sources
        ],
    )


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating fences and stray prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, depth = text.find("{"), 0
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in response: {text[:200]}")


# --------------------------------------------------------------------------
# Rule-based backend
# --------------------------------------------------------------------------

# Maps a formulation strategy to the library tags whose documents support it.
_STRATEGY_TAGS = {
    "Amorphous solid dispersion": ["amorphous solid dispersion"],
    "Lipid-based formulation": ["lipid-based formulation"],
    "Cyclodextrin complexation": ["cyclodextrin"],
    "Salt formation": ["salt formation"],
    "Cocrystal": ["cocrystal"],
    "Particle-size reduction": ["particle size reduction", "nanosuspension"],
    "Supersaturating formulation with precipitation inhibitor": [
        "supersaturating formulation",
        "precipitation inhibition",
    ],
}


def _sources_for(strategy: str, retrieved: list[RetrievedEvidence], limit: int = 2) -> list[str]:
    tags = _STRATEGY_TAGS.get(strategy, [])
    hits = [r.doc.id for r in retrieved if any(t in r.doc.tags for t in tags)]
    return hits[:limit]


def _rule_based_assessment(
    desc: MolecularDescriptors,
    sol: SolubilityPrediction,
    retrieved: list[RetrievedEvidence],
) -> FormulationAssessment:
    """A transparent heuristic ranking. Not an LLM — a defensible baseline."""
    drivers: list[str] = []
    if sol.risk.value in {"high", "moderate"}:
        drivers.append(f"Predicted log S {sol.log_s:.2f} (+/- {sol.log_s_std:.2f})")
    if desc.clogp >= 3:
        drivers.append(f"High calculated lipophilicity (cLogP {desc.clogp:.2f})")
    if desc.molecular_weight > 500:
        drivers.append(f"Molecular weight {desc.molecular_weight:.0f} above 500")
    if desc.h_bond_donors >= 3:
        drivers.append(f"{desc.h_bond_donors} H-bond donors may stabilise the crystal lattice")
    if desc.aromatic_rings >= 3 and desc.fraction_csp3 < 0.3:
        drivers.append("Planar, aromatic-rich, low sp3 fraction — dense crystal packing likely")
    if not drivers:
        drivers.append("No strong solubility-limiting descriptor flags")

    # A freely soluble compound does not need an enabling formulation. Recommending one
    # anyway is over-triggering, and it is the failure mode that would most damage
    # trust in the tool, so it is handled explicitly rather than falling through the
    # ranking heuristics below.
    if sol.risk == SolubilityRisk.LOW:
        return FormulationAssessment(
            compound_summary=CompoundSummary(solubility_risk=sol.risk, main_drivers=drivers),
            ranked_strategies=[],
            missing_information=[
                "intestinal permeability (may be the actual exposure limit)",
                "dose and required exposure",
                "chemical and solid-state stability",
            ],
            recommended_experiments=[
                "confirm equilibrium solubility across the physiological pH range",
                "permeability assay (Caco-2 or PAMPA) to test whether absorption is permeability-limited",
            ],
            overall_uncertainty=(
                f"Predicted log S {sol.log_s:.2f} (+/- {sol.log_s_std:.2f}) indicates adequate "
                "aqueous solubility, so no solubility-enabling formulation is indicated on these "
                "descriptors alone. If exposure is poor in vivo, the limit is more likely "
                "permeability, first-pass metabolism or dose, none of which this prototype "
                "assesses. Generated by the deterministic rule-based backend, not an LLM."
            ),
        )

    candidates: list[tuple[str, str, Confidence, list[str]]] = []

    if desc.clogp >= 3 and sol.risk.value in {"high", "moderate"}:
        candidates.append(
            (
                "Amorphous solid dispersion",
                f"High lipophilicity (cLogP {desc.clogp:.2f}) with predicted low aqueous "
                f"solubility (log S {sol.log_s:.2f}) is the profile ASDs are most often "
                "applied to; removing crystal lattice energy raises apparent solubility.",
                Confidence.MEDIUM,
                [
                    "Physical stability and recrystallisation on storage are unresolved without solid-state data",
                    "Requires a polymer screen; no melting point or glass transition data supplied",
                ],
            )
        )
        candidates.append(
            (
                "Lipid-based formulation",
                f"cLogP {desc.clogp:.2f} suggests the compound may partition into lipid "
                "excipients, presenting the drug in a pre-solubilised state.",
                Confidence.MEDIUM,
                [
                    "Depends on unmeasured solubility in the excipients themselves",
                    "Digestion-driven precipitation risk cannot be assessed from descriptors",
                ],
            )
        )

    if desc.h_bond_donors >= 1 or desc.h_bond_acceptors >= 4:
        candidates.append(
            (
                "Cocrystal",
                f"{desc.h_bond_donors} donors / {desc.h_bond_acceptors} acceptors provide "
                "hydrogen-bonding sites for coformer selection.",
                Confidence.LOW,
                [
                    "Coformer screening is empirical; no pKa or crystallinity data supplied",
                    "Improves dissolution but may not raise equilibrium solubility",
                ],
            )
        )

    candidates.append(
        (
            "Particle-size reduction",
            "Increasing surface area raises dissolution rate where absorption is "
            "dissolution-rate limited.",
            Confidence.LOW,
            [
                "Does not overcome equilibrium-solubility limits",
                "Cannot distinguish dissolution-rate-limited from solubility-limited absorption without permeability data",
            ],
        )
    )

    if desc.molecular_weight < 600 and desc.aromatic_rings >= 1:
        candidates.append(
            (
                "Cyclodextrin complexation",
                f"Molecular weight {desc.molecular_weight:.0f} and aromatic content are "
                "geometrically compatible with cyclodextrin cavity inclusion.",
                Confidence.LOW,
                [
                    "Complexation efficiency is compound-specific and must be measured",
                    "Dose and excipient load may be limiting",
                ],
            )
        )

    # Only keep strategies the retrieved evidence can actually support.
    strategies: list[RankedStrategy] = []
    for rank, (name, rationale, conf, limits) in enumerate(candidates[:5], start=1):
        sources = _sources_for(name, retrieved)
        strategies.append(
            RankedStrategy(
                strategy=name,
                rank=rank,
                rationale=rationale,
                supporting_sources=sources,
                confidence=conf if sources else Confidence.LOW,
                limitations=limits
                + ([] if sources else ["No retrieved document directly supports this strategy"]),
            )
        )

    return FormulationAssessment(
        compound_summary=CompoundSummary(solubility_risk=sol.risk, main_drivers=drivers),
        ranked_strategies=strategies,
        missing_information=[
            "melting point / glass transition temperature",
            "pKa and ionisation state at physiological pH",
            "crystallinity and polymorph landscape",
            "intestinal permeability (Caco-2 or PAMPA)",
            "precipitation tendency on dilution",
            "measured equilibrium and kinetic solubility",
            "dose and required exposure",
        ],
        recommended_experiments=[
            "kinetic and equilibrium solubility assay in biorelevant media (FaSSIF/FeSSIF)",
            "solid-state characterisation (DSC, XRPD)",
            "pH-solubility profile and pKa determination",
            "excipient and polymer compatibility screen",
            "in vitro dissolution with precipitation monitoring",
        ],
        overall_uncertainty=(
            f"Ranking is driven by calculated descriptors and a baseline ESOL model whose "
            f"held-out RMSE is {sol.model_rmse:.2f} log units — the log S estimate of "
            f"{sol.log_s:.2f} should be read as a band of roughly "
            f"{sol.log_s - sol.model_rmse:.1f} to {sol.log_s + sol.model_rmse:.1f}. "
            "No experimental solid-state, ionisation or permeability data were supplied, so "
            "these are hypotheses for screening, not a formulation decision. Generated by the "
            "deterministic rule-based backend, not an LLM."
        ),
    )


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


class FormulationAgent:
    def __init__(self, backend: str = "auto", model: str = DEFAULT_MODEL):
        self.model = model
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        if backend == "auto":
            backend = "anthropic" if self._key_looks_real(key) else "rulebased"
        elif backend == "anthropic":
            # Explicitly asking for the LLM with a missing or unedited key is a mistake
            # worth shouting about up front. Falling through would burn one 401 per
            # request and then report rule-based-looking zeros as an LLM result.
            self._validate_key(key)

        self.backend = backend
        self._client = None

    @staticmethod
    def _key_looks_real(key: str) -> bool:
        return bool(key) and "..." not in key and len(key) >= MIN_KEY_LENGTH

    @classmethod
    def _validate_key(cls, key: str) -> None:
        if not key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set.\n"
                "  1. cp .env.example .env\n"
                "  2. put your real key in .env  ->  ANTHROPIC_API_KEY=sk-ant-...\n"
                "  3. re-run from the repository root\n"
                "Or use --backend rulebased for the deterministic no-LLM baseline."
            )
        if "..." in key or len(key) < MIN_KEY_LENGTH:
            raise ConfigurationError(
                f"ANTHROPIC_API_KEY looks like the unedited .env.example placeholder "
                f"(length {len(key)}; a real key is ~100 characters and has no '...').\n"
                "Open .env, replace the whole placeholder with your real key, and SAVE "
                "the file before re-running."
            )

    @property
    def mode_label(self) -> str:
        if self.backend == "anthropic":
            return f"LLM ({self.model})"
        return "rule-based baseline (no LLM)"

    def _anthropic(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _call_llm(self, system: str, user: str, temperature: float) -> str:
        resp = self._anthropic().messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def _generate(
        self,
        *,
        prompt_mode: str,
        compound_label: str,
        desc: MolecularDescriptors,
        sol: SolubilityPrediction,
        retrieved: list[RetrievedEvidence],
        temperature: float,
        **context,
    ) -> FormulationAssessment:
        if self.backend == "rulebased":
            # The rule-based baseline has no notion of prompt ablation; it always uses
            # descriptors. Evidence is withheld to mirror the llm_only condition.
            return _rule_based_assessment(
                desc, sol, [] if prompt_mode == "llm_only" else retrieved
            )

        user = build_user_prompt(
            mode=prompt_mode,
            compound_label=compound_label,
            desc=desc,
            sol=sol,
            evidence=retrieved,
            **context,
        )

        import anthropic

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                raw = self._call_llm(SYSTEM_PROMPT, user, temperature)
                return FormulationAssessment.model_validate(_extract_json(raw))
            except (
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
                anthropic.NotFoundError,
            ) as exc:
                # Bad key, no access, or a model that doesn't exist. Retrying cannot
                # help — it just turns one clear failure into three noisy ones and
                # buries the cause under a misleading "failed validation" message.
                raise ConfigurationError(
                    f"{type(exc).__name__}: {exc}\n"
                    "This is a configuration problem, not a model problem. Check "
                    "ANTHROPIC_API_KEY in .env, that the key is active with credit at "
                    "console.anthropic.com, and that BIOENHANCE_MODEL names a model "
                    "your account can reach."
                ) from exc
            except (ValueError, ValidationError) as exc:
                # Malformed or schema-invalid JSON — the one failure worth retrying,
                # since the feedback often fixes it.
                last_error = exc
                user += (
                    f"\n\nYour previous response failed validation: {exc}. "
                    "Return ONLY the valid JSON object."
                )
        raise RuntimeError(
            f"Model failed to return schema-valid JSON after 3 attempts: {last_error}"
        )

    def run(
        self,
        *,
        name: str | None = None,
        smiles: str | None = None,
        dose: str | None = None,
        ph: str | None = None,
        dosage_form: str | None = None,
        constraints: str | None = None,
        top_k: int = 6,
        prompt_mode: str = "full",
        temperature: float = 0.0,
    ) -> AgentResult:
        """Run the full pipeline for one compound."""
        resolved_smiles, resolved_name = resolve_input(name, smiles)

        desc = compute_descriptors(resolved_smiles)
        sol = get_predictor().predict(desc)

        query = build_query(desc, sol, constraints=constraints, dosage_form=dosage_form)
        retrieved = (
            [] if prompt_mode == "llm_only" else retrieve_relevant_evidence(query, top_k=top_k)
        )

        label = resolved_name or resolved_smiles
        assessment = self._generate(
            prompt_mode=prompt_mode,
            compound_label=label,
            desc=desc,
            sol=sol,
            retrieved=retrieved,
            temperature=temperature,
            dose=dose,
            ph=ph,
            dosage_form=dosage_form,
            constraints=constraints,
        )

        return AgentResult(
            query_name=resolved_name,
            descriptors=desc,
            solubility=sol,
            retrieved=retrieved,
            assessment=assessment,
            grounding=check_grounding(assessment, retrieved),
            mode=f"{self.mode_label} / {prompt_mode}",
        )
