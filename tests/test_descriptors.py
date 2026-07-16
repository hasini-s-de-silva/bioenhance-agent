"""Descriptor calculations are checked against independently known values.

If these drift, the numbers the LLM reasons over are wrong, and every downstream
claim is unsafe. Reference values are RDKit's Wildman-Crippen logP and standard
TPSA, cross-checkable against PubChem's computed properties.
"""

from __future__ import annotations

import pytest

from src.descriptors import (
    InvalidSmilesError,
    compute_descriptors,
    parse_molecule,
    resolve_input,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


class TestDescriptorAccuracy:
    def test_aspirin_matches_known_values(self):
        d = compute_descriptors(ASPIRIN)
        assert d.molecular_weight == pytest.approx(180.16, abs=0.05)
        assert d.tpsa == pytest.approx(63.6, abs=0.5)
        assert d.h_bond_donors == 1
        assert d.h_bond_acceptors == 3
        assert d.aromatic_rings == 1

    def test_caffeine_matches_known_values(self):
        d = compute_descriptors(CAFFEINE)
        assert d.molecular_weight == pytest.approx(194.19, abs=0.05)
        assert d.h_bond_donors == 0

    def test_caffeine_tpsa_follows_rdkit_aromaticity_model(self):
        """RDKit gives caffeine TPSA 61.82; PubChem publishes 58.44. Both are 'right'.

        RDKit perceives the two amide ring nitrogens as aromatic (Ertl contribution
        4.93 A^2 each); the value PubChem reports treats them as non-aromatic tertiary
        nitrogens (3.24 A^2 each). 2 x (4.93 - 3.24) = 3.38, which is exactly the gap.

        Pinning RDKit's value documents the convention this project computes in, so a
        reviewer comparing against PubChem knows why the numbers differ.
        """
        d = compute_descriptors(CAFFEINE)
        assert d.tpsa == pytest.approx(61.82, abs=0.1)

    def test_ibuprofen_matches_known_values(self):
        d = compute_descriptors(IBUPROFEN)
        assert d.molecular_weight == pytest.approx(206.28, abs=0.05)
        assert d.h_bond_donors == 1
        assert d.clogp > 3  # ibuprofen is lipophilic

    def test_canonical_smiles_is_stable(self):
        """Different input spellings of one molecule give one canonical form."""
        a = compute_descriptors("c1ccccc1C(=O)O")
        b = compute_descriptors("OC(=O)c1ccccc1")
        assert a.canonical_smiles == b.canonical_smiles


class TestLipinski:
    def test_aspirin_passes(self):
        d = compute_descriptors(ASPIRIN)
        assert d.lipinski_violations == 0
        assert d.lipinski_pass is True

    def test_large_lipophilic_molecule_violates(self):
        d = compute_descriptors(
            "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)O"
        )
        assert d.lipinski_violations >= 2
        assert d.lipinski_pass is False

    def test_violation_count_is_consistent_with_flags(self):
        d = compute_descriptors(IBUPROFEN)
        expected = sum(
            [
                d.molecular_weight > 500,
                d.clogp > 5,
                d.h_bond_donors > 5,
                d.h_bond_acceptors > 10,
            ]
        )
        assert d.lipinski_violations == expected


class TestInvalidInput:
    def test_unparseable_smiles_raises(self):
        with pytest.raises(InvalidSmilesError):
            compute_descriptors("this-is-not-a-molecule")

    def test_broken_valence_raises(self):
        with pytest.raises(InvalidSmilesError):
            parse_molecule("C(C)(C)(C)(C)(C)C")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            resolve_input(None, None)

    def test_smiles_wins_over_name(self):
        """An explicit SMILES must not trigger a network lookup."""
        smiles, name = resolve_input("aspirin", ASPIRIN)
        assert smiles == ASPIRIN
        assert name == "aspirin"


@pytest.mark.network
class TestNameResolution:
    def test_resolves_aspirin(self):
        from src.descriptors import resolve_name_to_smiles

        d = compute_descriptors(resolve_name_to_smiles("aspirin"))
        assert d.molecular_weight == pytest.approx(180.16, abs=0.05)

    def test_nonsense_name_raises(self):
        from src.descriptors import NameResolutionError, resolve_name_to_smiles

        with pytest.raises(NameResolutionError):
            resolve_name_to_smiles("zzzznotarealdrugzzzz")
