"""Build data/evaluation_cases.csv with real SMILES resolved from PubChem.

The reference strategies below are well-documented, textbook formulation facts (e.g.
Sporanox is an amorphous solid dispersion; Tricor and Emend use nanocrystal
technology). They are used as a weak sanity signal for retrieval relevance, NOT as a
gold-standard label for "the correct formulation" — real formulation choices depend on
dose, stability and manufacturability that this prototype never sees.

Two freely soluble compounds are included as negative controls: a system that
recommends enabling formulation for metformin is over-triggering.

Usage:
    python -m scripts.build_eval_cases
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from src.descriptors import resolve_name_to_smiles

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluation_cases.csv"

CASES = [
    {
        "name": "itraconazole",
        "bcs_class": "II",
        "expected_risk": "high",
        "reference_strategy": "amorphous solid dispersion",
        "notes": "Sporanox is formulated as an ASD coated on sugar spheres; very high logP.",
    },
    {
        "name": "ritonavir",
        "bcs_class": "IV",
        "expected_risk": "high",
        "reference_strategy": "amorphous solid dispersion",
        "notes": "Norvir tablet uses melt-extruded ASD; the 1998 polymorph conversion is the classic solid-state cautionary tale.",
    },
    {
        "name": "fenofibrate",
        "bcs_class": "II",
        "expected_risk": "high",
        "reference_strategy": "particle size reduction",
        "notes": "Tricor uses NanoCrystal technology; also a common lipid-formulation model compound.",
    },
    {
        "name": "aprepitant",
        "bcs_class": "IV",
        "expected_risk": "high",
        "reference_strategy": "nanosuspension",
        "notes": "Emend capsules use nanocrystal particle-size reduction.",
    },
    {
        "name": "griseofulvin",
        "bcs_class": "II",
        "expected_risk": "high",
        "reference_strategy": "particle size reduction",
        "notes": "Classic micronisation case; ultramicrosize formulations improve exposure.",
    },
    {
        "name": "carbamazepine",
        "bcs_class": "II",
        "expected_risk": "moderate",
        "reference_strategy": "cocrystal",
        "notes": "The most studied pharmaceutical cocrystal model compound.",
    },
    {
        "name": "celecoxib",
        "bcs_class": "II",
        "expected_risk": "high",
        "reference_strategy": "amorphous solid dispersion",
        "notes": "Widely studied for ASD and cyclodextrin approaches.",
    },
    {
        "name": "cinnarizine",
        "bcs_class": "II",
        "expected_risk": "high",
        "reference_strategy": "supersaturating formulation",
        "notes": "Weak base; standard model compound for SEDDS and precipitation studies.",
    },
    {
        "name": "danazol",
        "bcs_class": "II",
        "expected_risk": "high",
        "reference_strategy": "lipid-based formulation",
        "notes": "Classic lipid-formulation and cyclodextrin model compound.",
    },
    {
        "name": "ibuprofen",
        "bcs_class": "II",
        "expected_risk": "moderate",
        "reference_strategy": "salt formation",
        "notes": "Marketed as lysine and sodium salts for faster onset.",
    },
    # --- negative controls: freely soluble, should NOT trigger enabling formulation ---
    {
        "name": "metformin",
        "bcs_class": "III",
        "expected_risk": "low",
        "reference_strategy": "none required",
        "notes": "Negative control. Highly soluble, permeability-limited.",
    },
    {
        "name": "paracetamol",
        "bcs_class": "I",
        "expected_risk": "low",
        "reference_strategy": "none required",
        "notes": "Negative control. Freely soluble and permeable.",
    },
]

FIELDS = ["name", "smiles", "bcs_class", "expected_risk", "reference_strategy", "notes"]


def main() -> None:
    rows = []
    for case in CASES:
        smiles = resolve_name_to_smiles(case["name"])
        rows.append({**case, "smiles": smiles})
        print(f"  {case['name']:16s} {smiles[:56]}")
        time.sleep(0.35)  # be polite to PubChem

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} cases -> {OUT}")


if __name__ == "__main__":
    main()
