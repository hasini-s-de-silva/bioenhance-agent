"""Retrieval and evidence-library integrity tests.

The library-integrity tests matter more than the ranking tests: if a document's
citation metadata is wrong, the system is confidently pointing a formulation
scientist at a paper that does not say what it claims.
"""

from __future__ import annotations

import re

import pytest

from src.retrieval import EvidenceIndex, build_query, load_library
from src.schemas import MolecularDescriptors, SolubilityPrediction, SolubilityRisk


@pytest.fixture(scope="module")
def library():
    return load_library()


@pytest.fixture(scope="module")
def index(library):
    # Pin the sparse backend: tests must not depend on a model download.
    return EvidenceIndex(docs=library, prefer_dense=False)


class TestLibraryIntegrity:
    def test_library_is_populated(self, library):
        assert 20 <= len(library) <= 60, "spec calls for roughly 20-40 documents"

    def test_ids_are_unique(self, library):
        ids = [d.id for d in library]
        assert len(ids) == len(set(ids))

    def test_pmids_are_unique(self, library):
        pmids = [d.pmid for d in library]
        assert len(pmids) == len(set(pmids))

    def test_every_doc_has_citable_metadata(self, library):
        """A document we cannot cite honestly must not be in the library."""
        for d in library:
            assert d.title.strip(), f"{d.id} has no title"
            assert d.source.strip(), f"{d.id} has no journal"
            assert 1990 <= d.year <= 2030, f"{d.id} has implausible year {d.year}"
            assert d.pmid.isdigit(), f"{d.id} has a non-numeric PMID {d.pmid!r}"
            assert d.text.strip(), f"{d.id} has no abstract text"

    def test_urls_point_at_the_stated_pmid(self, library):
        """The URL must resolve to the same record as the PMID field."""
        for d in library:
            assert d.url == f"https://pubmed.ncbi.nlm.nih.gov/{d.pmid}/"

    def test_abstracts_are_substantial(self, library):
        for d in library:
            assert len(d.text) >= 400, f"{d.id} abstract is too short to ground a claim"

    def test_ids_follow_convention(self, library):
        for d in library:
            assert re.fullmatch(r"S\d{2}", d.id), f"{d.id} breaks the S## convention"

    def test_every_doc_is_tagged(self, library):
        for d in library:
            assert d.tags, f"{d.id} has no tags"

    def test_all_strategy_families_are_covered(self, library):
        tags = {t for d in library for t in d.tags}
        for required in [
            "amorphous solid dispersion",
            "lipid-based formulation",
            "salt formation",
            "cocrystal",
            "cyclodextrin",
            "nanosuspension",
            "particle size reduction",
            "supersaturating formulation",
            "precipitation inhibition",
            "poorly soluble oral drugs",
        ]:
            assert required in tags, f"library has no evidence for {required!r}"


class TestRetrieval:
    def test_returns_requested_number(self, index):
        assert len(index.search("poorly soluble drug", top_k=5)) == 5

    def test_scores_are_descending(self, index):
        scores = [r.score for r in index.search("amorphous solid dispersion", top_k=6)]
        assert scores == sorted(scores, reverse=True)

    def test_topic_query_retrieves_matching_topic(self, index):
        """A cyclodextrin query must surface cyclodextrin evidence."""
        hits = index.search("cyclodextrin inclusion complex solubility", top_k=5)
        assert any("cyclodextrin" in r.doc.tags for r in hits)

    def test_lipid_query_retrieves_lipid_evidence(self, index):
        hits = index.search(
            "self-emulsifying lipid based formulation oral bioavailability", top_k=5
        )
        assert any("lipid-based formulation" in r.doc.tags for r in hits)

    def test_top_k_larger_than_library_is_safe(self, index, library):
        hits = index.search("solubility", top_k=len(library) + 25)
        assert len(hits) <= len(library)

    def test_results_are_deterministic(self, index):
        a = [r.doc.id for r in index.search("nanosuspension dissolution", top_k=5)]
        b = [r.doc.id for r in index.search("nanosuspension dissolution", top_k=5)]
        assert a == b


def _descriptors(**overrides) -> MolecularDescriptors:
    base = dict(
        smiles="CCO",
        canonical_smiles="CCO",
        molecular_weight=400.0,
        clogp=4.5,
        tpsa=60.0,
        h_bond_donors=1,
        h_bond_acceptors=4,
        rotatable_bonds=5,
        aromatic_rings=2,
        heavy_atoms=30,
        fraction_csp3=0.4,
        molar_refractivity=100.0,
        lipinski_violations=0,
        lipinski_pass=True,
        veber_pass=True,
    )
    base.update(overrides)
    return MolecularDescriptors(**base)


def _solubility(risk=SolubilityRisk.HIGH, log_s=-5.0) -> SolubilityPrediction:
    return SolubilityPrediction(
        log_s=log_s,
        log_s_std=0.4,
        mg_per_ml=0.001,
        risk=risk,
        classification="low (poorly soluble)",
        model_rmse=0.76,
    )


class TestQueryConstruction:
    def test_high_logp_query_mentions_lipophilicity(self):
        q = build_query(_descriptors(clogp=6.0), _solubility())
        assert "lipophilicity" in q.lower()

    def test_low_risk_query_differs_from_high_risk(self):
        high = build_query(_descriptors(), _solubility(SolubilityRisk.HIGH, -5.0))
        low = build_query(_descriptors(), _solubility(SolubilityRisk.LOW, -1.0))
        assert high != low
        assert "poorly water soluble" in high.lower()

    def test_constraints_are_included(self):
        q = build_query(_descriptors(), _solubility(), constraints="avoid organic solvents")
        assert "avoid organic solvents" in q

    def test_dosage_form_is_included(self):
        q = build_query(_descriptors(), _solubility(), dosage_form="oral tablet")
        assert "oral tablet" in q
