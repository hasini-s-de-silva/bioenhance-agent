"""Pydantic models for every value that crosses a module boundary.

The LLM-facing models (`FormulationAssessment` and below) double as the contract we
validate generated JSON against, so a malformed or hallucinated response fails loudly
rather than reaching the user.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SolubilityRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# --------------------------------------------------------------------------
# Scientific tool outputs — computed, never generated
# --------------------------------------------------------------------------


class MolecularDescriptors(BaseModel):
    """RDKit-calculated descriptors. Every field here is a computed fact."""

    smiles: str
    canonical_smiles: str
    molecular_weight: float
    clogp: float = Field(description="Wildman-Crippen calculated logP")
    tpsa: float = Field(description="Topological polar surface area (A^2)")
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    fraction_csp3: float
    molar_refractivity: float

    # Rule-based flags derived from the above
    lipinski_violations: int
    lipinski_pass: bool
    veber_pass: bool

    def summary_lines(self) -> list[str]:
        return [
            f"Molecular weight: {self.molecular_weight:.1f}",
            f"Calculated logP: {self.clogp:.2f}",
            f"Topological polar surface area: {self.tpsa:.1f}",
            f"H-bond donors: {self.h_bond_donors}",
            f"H-bond acceptors: {self.h_bond_acceptors}",
            f"Rotatable bonds: {self.rotatable_bonds}",
            f"Aromatic rings: {self.aromatic_rings}",
            f"Fraction sp3 carbons: {self.fraction_csp3:.2f}",
            f"Lipinski violations: {self.lipinski_violations}",
        ]


class SolubilityPrediction(BaseModel):
    """Baseline ESOL-model output plus its own honest error bars."""

    log_s: float = Field(description="Predicted log10 aqueous solubility (mol/L)")
    log_s_std: float = Field(description="Std dev across the ensemble (mol/L, log10)")
    mg_per_ml: float
    risk: SolubilityRisk
    classification: str = Field(description="Human-readable solubility band")
    model_rmse: float = Field(description="Held-out test RMSE of the backing model")
    applicability_warning: str | None = Field(
        default=None,
        description="Set when the query sits outside the ESOL training distribution",
    )

    model_config = {"protected_namespaces": ()}


class EvidenceDoc(BaseModel):
    """One real PubMed record from the curated library."""

    id: str
    title: str
    source: str
    year: int
    url: str
    doi: str = ""
    pmid: str = ""
    text: str
    tags: list[str] = Field(default_factory=list)

    def citation(self) -> str:
        return f"[{self.id}] {self.title}. {self.source} ({self.year}). PMID {self.pmid}"


class RetrievedEvidence(BaseModel):
    doc: EvidenceDoc
    score: float = Field(description="Cosine similarity to the query embedding")


# --------------------------------------------------------------------------
# LLM output contract
# --------------------------------------------------------------------------


class CompoundSummary(BaseModel):
    solubility_risk: SolubilityRisk
    main_drivers: list[str] = Field(
        default_factory=list,
        description="Descriptor-grounded reasons for the risk call",
    )


class RankedStrategy(BaseModel):
    strategy: str
    rank: int = Field(ge=1)
    rationale: str
    supporting_sources: list[str] = Field(
        default_factory=list, description="Evidence ids, e.g. ['S03', 'S17']"
    )
    confidence: Confidence
    limitations: list[str] = Field(default_factory=list)

    @field_validator("supporting_sources")
    @classmethod
    def _normalise_ids(cls, v: list[str]) -> list[str]:
        # Models like to write "[S03]" or "Source 3"; store the bare id.
        out = []
        for raw in v:
            token = raw.strip().strip("[]() ")
            if token.lower().startswith("source"):
                digits = "".join(c for c in token if c.isdigit())
                token = f"S{int(digits):02d}" if digits else token
            out.append(token.upper())
        return out


class FormulationAssessment(BaseModel):
    """The full structured response. This is the schema the LLM must satisfy."""

    compound_summary: CompoundSummary
    ranked_strategies: list[RankedStrategy]
    missing_information: list[str] = Field(default_factory=list)
    recommended_experiments: list[str] = Field(default_factory=list)
    overall_uncertainty: str

    @field_validator("ranked_strategies")
    @classmethod
    def _sorted_ranks(cls, v: list[RankedStrategy]) -> list[RankedStrategy]:
        return sorted(v, key=lambda s: s.rank)

    def cited_ids(self) -> set[str]:
        return {sid for s in self.ranked_strategies for sid in s.supporting_sources}


class GroundingReport(BaseModel):
    """Post-hoc check that the model cited only what it was actually given.

    This is the guardrail that makes the 'evidence-grounded' claim testable rather
    than aspirational.
    """

    cited_ids: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    hallucinated_ids: list[str] = Field(
        default_factory=list, description="Cited but never retrieved — a fabrication"
    )
    uncited_strategies: list[str] = Field(
        default_factory=list, description="Strategies asserted with no supporting source"
    )

    @property
    def is_grounded(self) -> bool:
        return not self.hallucinated_ids

    @property
    def citation_accuracy(self) -> float:
        """Share of citations that point at a genuinely retrieved document."""
        if not self.cited_ids:
            return 0.0
        valid = len(set(self.cited_ids) - set(self.hallucinated_ids))
        return valid / len(set(self.cited_ids))


class AgentResult(BaseModel):
    """Everything one run produces, for the UI and the evaluation harness."""

    query_name: str | None = None
    descriptors: MolecularDescriptors
    solubility: SolubilityPrediction
    retrieved: list[RetrievedEvidence]
    assessment: FormulationAssessment
    grounding: GroundingReport
    mode: str = Field(description="Which LLM backend produced this")
