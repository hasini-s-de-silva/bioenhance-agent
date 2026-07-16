"""Structured-output and grounding tests.

These are the tests that back the project's central claim. `check_grounding` is what
turns "evidence-grounded" from marketing into something falsifiable, so it is tested
against a deliberately fabricating model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm_agent import (
    ConfigurationError,
    FormulationAgent,
    _explain_api_error,
    _extract_json,
    check_grounding,
)
from src.schemas import (
    CompoundSummary,
    Confidence,
    EvidenceDoc,
    FormulationAssessment,
    RankedStrategy,
    RetrievedEvidence,
    SolubilityRisk,
)

PARACETAMOL = "CC(=O)NC1=CC=C(C=C1)O"
ITRACONAZOLE_LIKE = "CCCCCCCCc1ccc(cc1)C(=O)Nc1ccc(cc1)c1ccc(cc1)C(=O)Nc1ccccc1"


def _doc(doc_id: str) -> EvidenceDoc:
    return EvidenceDoc(
        id=doc_id,
        title=f"Study {doc_id}",
        source="J Test",
        year=2023,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/",
        pmid="12345678",
        text="An abstract about solubility enhancement.",
        tags=["amorphous solid dispersion"],
    )


def _retrieved(*ids: str) -> list[RetrievedEvidence]:
    return [RetrievedEvidence(doc=_doc(i), score=0.5) for i in ids]


def _assessment(sources: list[str]) -> FormulationAssessment:
    return FormulationAssessment(
        compound_summary=CompoundSummary(
            solubility_risk=SolubilityRisk.HIGH, main_drivers=["high clogp"]
        ),
        ranked_strategies=[
            RankedStrategy(
                strategy="Amorphous solid dispersion",
                rank=1,
                rationale="high lipophilicity",
                supporting_sources=sources,
                confidence=Confidence.MEDIUM,
                limitations=["stability unknown"],
            )
        ],
        missing_information=["pKa"],
        recommended_experiments=["kinetic solubility"],
        overall_uncertainty="Hypothesis only.",
    )


class TestGroundingGuardrail:
    def test_valid_citation_is_grounded(self):
        report = check_grounding(_assessment(["S01"]), _retrieved("S01", "S02"))
        assert report.is_grounded
        assert report.hallucinated_ids == []
        assert report.citation_accuracy == 1.0

    def test_fabricated_citation_is_caught(self):
        """The core failure mode: citing a source that was never supplied."""
        report = check_grounding(_assessment(["S99"]), _retrieved("S01", "S02"))
        assert not report.is_grounded
        assert report.hallucinated_ids == ["S99"]
        assert report.citation_accuracy == 0.0

    def test_partial_fabrication_is_caught(self):
        report = check_grounding(_assessment(["S01", "S99"]), _retrieved("S01"))
        assert not report.is_grounded
        assert report.hallucinated_ids == ["S99"]
        assert report.citation_accuracy == 0.5

    def test_strategy_with_no_source_is_flagged(self):
        report = check_grounding(_assessment([]), _retrieved("S01"))
        assert report.uncited_strategies == ["Amorphous solid dispersion"]

    def test_citing_nothing_is_not_scored_as_perfect(self):
        report = check_grounding(_assessment([]), _retrieved("S01"))
        assert report.citation_accuracy == 0.0


class TestSchemaValidation:
    def test_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            RankedStrategy(
                strategy="x",
                rank=0,
                rationale="r",
                confidence=Confidence.LOW,
            )

    def test_confidence_must_be_in_enum(self):
        with pytest.raises(ValidationError):
            RankedStrategy(
                strategy="x",
                rank=1,
                rationale="r",
                confidence="extremely-sure",
            )

    def test_strategies_are_sorted_by_rank(self):
        a = FormulationAssessment(
            compound_summary=CompoundSummary(solubility_risk=SolubilityRisk.HIGH),
            ranked_strategies=[
                RankedStrategy(strategy="b", rank=2, rationale="", confidence=Confidence.LOW),
                RankedStrategy(strategy="a", rank=1, rationale="", confidence=Confidence.LOW),
            ],
            overall_uncertainty="",
        )
        assert [s.rank for s in a.ranked_strategies] == [1, 2]

    def test_source_ids_are_normalised(self):
        """Models write '[S03]' or 'Source 3'; both must normalise to 'S03'."""
        s = RankedStrategy(
            strategy="x",
            rank=1,
            rationale="",
            confidence=Confidence.LOW,
            supporting_sources=["[S03]", "Source 7", "s12"],
        )
        assert s.supporting_sources == ["S03", "S07", "S12"]


class TestKeyValidation:
    """A bad key must fail loudly at setup, never as a plausible-looking 0% result."""

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="not set"):
            FormulationAgent(backend="anthropic")

    def test_unedited_placeholder_is_rejected(self, monkeypatch):
        """The .env.example placeholder is the most likely setup mistake."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
        with pytest.raises(ConfigurationError, match="placeholder"):
            FormulationAgent(backend="anthropic")

    def test_truncated_key_is_rejected(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc123")
        with pytest.raises(ConfigurationError, match="placeholder"):
            FormulationAgent(backend="anthropic")

    def test_auto_falls_back_rather_than_using_a_bad_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
        assert FormulationAgent(backend="auto").backend == "rulebased"

    def test_plausible_key_is_accepted(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 95)
        assert FormulationAgent(backend="anthropic").backend == "anthropic"


class TestApiErrorExplanations:
    """An API error must name the actual next action, not dump a raw dict."""

    @staticmethod
    def _error(status: int, message: str):
        import anthropic
        import httpx

        body = {"type": "error", "error": {"message": message}}
        response = httpx.Response(
            status,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            json=body,
        )
        cls = {
            400: anthropic.BadRequestError,
            401: anthropic.AuthenticationError,
            404: anthropic.NotFoundError,
            429: anthropic.RateLimitError,
        }[status]
        return cls(message=message, response=response, body=body)

    def test_no_credit_is_explained_as_billing(self):
        exc = self._error(400, "Your credit balance is too low to access the Anthropic API.")
        assert "no credit" in _explain_api_error(exc)
        assert "Plans & Billing" in _explain_api_error(exc)

    def test_bad_key_is_explained_as_auth(self):
        exc = self._error(401, "invalid x-api-key")
        assert "key was rejected" in _explain_api_error(exc)

    def test_unknown_model_names_the_model_setting(self):
        exc = self._error(404, "model not found")
        assert "BIOENHANCE_MODEL" in _explain_api_error(exc)

    def test_billing_error_is_terminal_not_retried(self, monkeypatch):
        """A 400 must abort immediately — retrying it 3x buries the cause."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 95)
        agent = FormulationAgent(backend="anthropic")

        calls = {"n": 0}

        def _boom(*a, **k):
            calls["n"] += 1
            raise self._error(400, "Your credit balance is too low.")

        monkeypatch.setattr(agent, "_call_llm", _boom)
        with pytest.raises(ConfigurationError, match="no credit"):
            agent.run(smiles=PARACETAMOL, prompt_mode="full")
        assert calls["n"] == 1, f"terminal error retried {calls['n']}x — must be 1"


class TestSaveRefusesBrokenRuns:
    def test_all_failed_run_is_not_written(self, tmp_path, monkeypatch):
        """A run where nothing worked must not overwrite good results with zeros."""
        from src import evaluation

        monkeypatch.setattr(evaluation, "RESULTS_DIR", tmp_path)
        report = {
            "outcomes": [
                {"schema_ok": False, "error": "401 invalid x-api-key"},
                {"schema_ok": False, "error": "401 invalid x-api-key"},
            ]
        }
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            evaluation.save(report)
        assert not (tmp_path / "evaluation.md").exists()


class TestJsonExtraction:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        assert _extract_json('Here you go:\n{"a": 1}\nHope that helps!') == {"a": 1}

    def test_nested_braces(self):
        assert _extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("there is no json here")


class TestEndToEnd:
    """Runs the real pipeline on the deterministic backend (no network, no API key)."""

    @pytest.fixture(scope="class")
    def agent(self):
        return FormulationAgent(backend="rulebased")

    def test_poorly_soluble_compound_gets_recommendations(self, agent):
        r = agent.run(smiles=ITRACONAZOLE_LIKE)
        assert r.assessment.ranked_strategies
        assert r.solubility.risk == SolubilityRisk.HIGH

    def test_output_is_always_grounded(self, agent):
        r = agent.run(smiles=ITRACONAZOLE_LIKE)
        assert r.grounding.is_grounded, r.grounding.hallucinated_ids

    def test_uncertainty_is_always_reported(self, agent):
        r = agent.run(smiles=ITRACONAZOLE_LIKE)
        assert r.assessment.missing_information
        assert r.assessment.recommended_experiments
        assert r.assessment.overall_uncertainty

    def test_soluble_compound_gets_no_enabling_formulation(self, agent):
        """Over-recommending for a soluble drug is the key false-positive failure."""
        r = agent.run(smiles=PARACETAMOL)
        assert r.solubility.risk == SolubilityRisk.LOW
        assert r.assessment.ranked_strategies == []

    def test_ranks_are_contiguous_from_one(self, agent):
        r = agent.run(smiles=ITRACONAZOLE_LIKE)
        ranks = [s.rank for s in r.assessment.ranked_strategies]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_every_strategy_declares_limitations(self, agent):
        r = agent.run(smiles=ITRACONAZOLE_LIKE)
        for s in r.assessment.ranked_strategies:
            assert s.limitations, f"{s.strategy} claims no limitations"

    def test_llm_only_mode_retrieves_nothing(self, agent):
        r = agent.run(smiles=ITRACONAZOLE_LIKE, prompt_mode="llm_only")
        assert r.retrieved == []
        assert r.grounding.is_grounded  # cannot fabricate what it never cites

    def test_mode_is_labelled_as_non_llm(self, agent):
        r = agent.run(smiles=PARACETAMOL)
        assert "rule-based" in r.mode
