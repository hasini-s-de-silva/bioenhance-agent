"""Baseline aqueous-solubility model trained on the Delaney ESOL dataset.

This is deliberately a small, inspectable model rather than a deep net. Its job is to
give the LLM a calculated solubility estimate *with an honest error bar*, not to be a
state-of-the-art solubility predictor.

The model reports:
  - a point estimate of log10 S (mol/L),
  - an ensemble standard deviation (spread across the forest's trees),
  - the held-out test RMSE, so the UI can show what the model is actually worth,
  - an applicability-domain warning when a query sits outside the training range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from .descriptors import compute_descriptors, parse_molecule
from .schemas import MolecularDescriptors, SolubilityPrediction, SolubilityRisk

ROOT = Path(__file__).resolve().parents[1]
ESOL_CSV = ROOT / "data" / "esol.csv"
MODEL_PATH = ROOT / "data" / "solubility_model.joblib"
METRICS_PATH = ROOT / "data" / "solubility_metrics.json"

RANDOM_STATE = 42

# Delaney's original descriptor set, plus a few cheap RDKit additions.
FEATURES = [
    "molecular_weight",
    "clogp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "aromatic_rings",
    "fraction_csp3",
    "molar_refractivity",
    "heavy_atoms",
    "aromatic_proportion",
]

# Solubility bands. These follow the conventional pharmaceutical reading of log S,
# where roughly < -4 is the "poorly soluble, needs enabling formulation" regime.
POOR_LOG_S = -4.0
MODERATE_LOG_S = -2.0


def _feature_row(desc: MolecularDescriptors) -> dict[str, float]:
    mol = parse_molecule(desc.canonical_smiles)
    heavy = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    row = {f: getattr(desc, f) for f in FEATURES if f != "aromatic_proportion"}
    row["aromatic_proportion"] = aromatic_atoms / heavy if heavy else 0.0
    return row


def build_training_frame(csv_path: Path = ESOL_CSV) -> pd.DataFrame:
    """Featurise ESOL from SMILES with our own RDKit pipeline.

    We recompute descriptors rather than using the columns shipped in the CSV, so the
    features at training time are produced by exactly the same code path as at
    inference time.
    """
    raw = pd.read_csv(csv_path)
    target_col = "measured log solubility in mols per litre"

    rows, targets = [], []
    for _, rec in raw.iterrows():
        try:
            desc = compute_descriptors(rec["smiles"])
        except Exception:  # noqa: BLE001 - a handful of ESOL SMILES fail to parse
            continue
        rows.append(_feature_row(desc))
        targets.append(float(rec[target_col]))

    frame = pd.DataFrame(rows)
    frame["log_s"] = targets
    return frame


def train(csv_path: Path = ESOL_CSV, save: bool = True) -> dict:
    """Train the forest, evaluate honestly on a held-out split, and persist."""
    import joblib

    frame = build_training_frame(csv_path)
    X = frame[FEATURES].to_numpy()
    y = frame["log_s"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    cv = cross_val_score(
        model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    metrics = {
        "n_compounds": int(len(frame)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_r2": float(r2_score(y_test, pred)),
        "cv_rmse_mean": float(-cv.mean()),
        "cv_rmse_std": float(cv.std()),
        "feature_importance": {
            f: float(i) for f, i in zip(FEATURES, model.feature_importances_)
        },
        # Applicability domain: the 1st-99th percentile box of the training features.
        "train_ranges": {
            f: [float(np.percentile(X[:, i], 1)), float(np.percentile(X[:, i], 99))]
            for i, f in enumerate(FEATURES)
        },
    }

    if save:
        model.fit(X, y)  # refit on everything for the shipped artefact
        joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


class SolubilityPredictor:
    """Loads the trained forest and turns descriptors into a solubility call."""

    def __init__(self, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH):
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run: python -m scripts.train_solubility"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self.rmse = float(self.metrics.get("test_rmse", float("nan")))
        self.ranges = self.metrics.get("train_ranges", {})

    def _applicability(self, row: dict[str, float]) -> str | None:
        outside = [
            f
            for f, (lo, hi) in self.ranges.items()
            if f in row and not (lo <= row[f] <= hi)
        ]
        if not outside:
            return None
        return (
            "Query sits outside the ESOL training distribution for: "
            + ", ".join(outside)
            + ". Treat the solubility estimate as indicative only."
        )

    def predict(self, desc: MolecularDescriptors) -> SolubilityPrediction:
        row = _feature_row(desc)
        X = np.array([[row[f] for f in self.features]])

        # Per-tree predictions give a cheap, honest spread for the ensemble.
        per_tree = np.array([t.predict(X)[0] for t in self.model.estimators_])
        log_s = float(per_tree.mean())
        log_s_std = float(per_tree.std())

        if log_s <= POOR_LOG_S:
            risk, label = SolubilityRisk.HIGH, "low (poorly soluble)"
        elif log_s <= MODERATE_LOG_S:
            risk, label = SolubilityRisk.MODERATE, "moderate"
        else:
            risk, label = SolubilityRisk.LOW, "high (freely soluble)"

        return SolubilityPrediction(
            log_s=round(log_s, 2),
            log_s_std=round(log_s_std, 2),
            mg_per_ml=round(float(10**log_s) * desc.molecular_weight, 4),
            risk=risk,
            classification=label,
            model_rmse=round(self.rmse, 2),
            applicability_warning=self._applicability(row),
        )


_PREDICTOR: SolubilityPredictor | None = None


def get_predictor() -> SolubilityPredictor:
    """Process-wide singleton so Streamlit reruns don't reload the forest."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = SolubilityPredictor()
    return _PREDICTOR
