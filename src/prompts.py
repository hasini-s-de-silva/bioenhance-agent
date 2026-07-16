"""Prompt construction for the formulation agent.

Three prompt configurations exist so the evaluation harness can ablate the system and
measure what retrieval and descriptors are actually buying us:

    llm_only  - compound identity only. No tools, no evidence.
    llm_rag   - retrieved evidence, but no calculated descriptors.
    full      - calculated descriptors + solubility model + retrieved evidence.
"""

from __future__ import annotations

import json

from .schemas import MolecularDescriptors, RetrievedEvidence, SolubilityPrediction

SYSTEM_PROMPT = """You are an evidence-grounded pharmaceutical formulation research assistant.

You may use ONLY:
1. the calculated molecular descriptors supplied to you;
2. the retrieved scientific evidence supplied to you.

Do not invent experimental values, publications or citations. You must never cite a
source identifier that does not appear in the supplied evidence block. If the evidence
does not support a strategy, say so and lower your confidence rather than reaching for
a source that is not there.

Clearly distinguish:
- calculated facts (descriptors and model predictions supplied to you);
- evidence-supported conclusions (traceable to a supplied source id);
- hypotheses requiring experimental validation.

Rank formulation strategies conservatively. Prefer fewer, well-supported
recommendations over a long speculative list. A calculated descriptor is a property of
the molecule, not evidence that a formulation strategy will work — only the supplied
literature can support that claim.

For every recommendation provide: rationale, supporting source identifiers,
confidence, and limitations. For the assessment as a whole provide missing
information, recommended next experiments, and an overall uncertainty statement.

Return ONLY valid JSON matching the requested schema. No markdown fences, no prose
outside the JSON object."""


OUTPUT_SCHEMA = {
    "compound_summary": {
        "solubility_risk": "one of: low | moderate | high",
        "main_drivers": ["short strings citing calculated descriptors"],
    },
    "ranked_strategies": [
        {
            "strategy": "name of the formulation strategy",
            "rank": 1,
            "rationale": "why this strategy suits this molecule",
            "supporting_sources": ["S01", "S07"],
            "confidence": "one of: low | medium | high",
            "limitations": ["what this strategy does not solve"],
        }
    ],
    "missing_information": ["experimental data needed to firm up the assessment"],
    "recommended_experiments": ["concrete next experiments"],
    "overall_uncertainty": "one honest paragraph on how much to trust this assessment",
}


def format_evidence(evidence: list[RetrievedEvidence]) -> str:
    """Render retrieved documents as an explicitly delimited, id-tagged block."""
    if not evidence:
        return "(no evidence retrieved)"
    blocks = []
    for item in evidence:
        d = item.doc
        blocks.append(
            f"[{d.id}] {d.title}\n"
            f"  Journal: {d.source} ({d.year})   PMID: {d.pmid}\n"
            f"  Retrieval score: {item.score:.3f}\n"
            f"  Abstract: {d.text}"
        )
    return "\n\n".join(blocks)


def format_descriptors(desc: MolecularDescriptors, sol: SolubilityPrediction) -> str:
    lines = [
        "CALCULATED MOLECULAR DESCRIPTORS (RDKit — these are computed facts):",
        f"  Canonical SMILES        : {desc.canonical_smiles}",
        f"  Molecular weight        : {desc.molecular_weight:.2f}",
        f"  Calculated logP (cLogP) : {desc.clogp:.2f}",
        f"  TPSA                    : {desc.tpsa:.2f} A^2",
        f"  H-bond donors           : {desc.h_bond_donors}",
        f"  H-bond acceptors        : {desc.h_bond_acceptors}",
        f"  Rotatable bonds         : {desc.rotatable_bonds}",
        f"  Aromatic rings          : {desc.aromatic_rings}",
        f"  Fraction sp3 carbons    : {desc.fraction_csp3:.3f}",
        f"  Molar refractivity      : {desc.molar_refractivity:.2f}",
        f"  Lipinski violations     : {desc.lipinski_violations} "
        f"({'pass' if desc.lipinski_pass else 'fail'})",
        f"  Veber rule              : {'pass' if desc.veber_pass else 'fail'}",
        "",
        "BASELINE SOLUBILITY MODEL (random forest trained on Delaney ESOL):",
        f"  Predicted log S         : {sol.log_s:.2f} (mol/L, log10)",
        f"  Ensemble spread (1 s.d.): +/- {sol.log_s_std:.2f} log units",
        f"  Approx. solubility      : {sol.mg_per_ml:.4f} mg/mL",
        f"  Solubility class        : {sol.classification}",
        f"  Solubility risk         : {sol.risk.value}",
        f"  Model held-out RMSE     : {sol.model_rmse:.2f} log units "
        f"(the prediction is worth no more than this)",
    ]
    if sol.applicability_warning:
        lines.append(f"  APPLICABILITY WARNING   : {sol.applicability_warning}")
    return "\n".join(lines)


def build_user_prompt(
    *,
    mode: str = "full",
    compound_label: str,
    desc: MolecularDescriptors | None = None,
    sol: SolubilityPrediction | None = None,
    evidence: list[RetrievedEvidence] | None = None,
    dose: str | None = None,
    ph: str | None = None,
    dosage_form: str | None = None,
    constraints: str | None = None,
) -> str:
    sections: list[str] = [f"COMPOUND: {compound_label}"]

    context_bits = [
        f"  Dose: {dose}" if dose else None,
        f"  Target pH: {ph}" if ph else None,
        f"  Intended dosage form: {dosage_form}" if dosage_form else None,
        f"  Development constraints: {constraints}" if constraints else None,
    ]
    context_bits = [b for b in context_bits if b]
    if context_bits:
        sections.append("DEVELOPMENT CONTEXT:\n" + "\n".join(context_bits))

    if mode == "full" and desc and sol:
        sections.append(format_descriptors(desc, sol))
    elif mode in {"llm_only", "llm_rag"}:
        sections.append(
            "NOTE: No calculated descriptors are supplied in this configuration. Do not "
            "state numeric physicochemical values you cannot derive from the supplied "
            "information."
        )

    if mode in {"llm_rag", "full"}:
        sections.append(
            "RETRIEVED SCIENTIFIC EVIDENCE — you may cite ONLY these source ids:\n\n"
            + format_evidence(evidence or [])
        )
    else:
        sections.append(
            "NO EVIDENCE IS SUPPLIED in this configuration. You have no source ids "
            "available, so `supporting_sources` must be an empty list for every strategy."
        )

    sections.append(
        "TASK:\n"
        "Assess the risk that poor aqueous solubility limits oral exposure for this "
        "compound, then rank the bioenhancement strategies that the supplied evidence "
        "actually supports. Rank conservatively — three to five strategies is usually "
        "enough. Every strategy must carry its own limitations.\n\n"
        "If the compound is adequately soluble, return an EMPTY ranked_strategies list "
        "and say so in overall_uncertainty. Recommending an enabling formulation for a "
        "freely soluble drug is a failure, not a safe default.\n\n"
        "Return JSON exactly matching this schema:\n"
        + json.dumps(OUTPUT_SCHEMA, indent=2)
    )

    return "\n\n".join(sections)
