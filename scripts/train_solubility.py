"""Train the baseline ESOL solubility model and print its honest metrics.

Usage:
    python -m scripts.train_solubility
"""

from __future__ import annotations

import json

from src.solubility_model import train


def main() -> None:
    print("Featurising ESOL with RDKit (this takes ~30s)...")
    metrics = train()

    print(f"\nCompounds featurised : {metrics['n_compounds']}")
    print(f"Train / test split   : {metrics['n_train']} / {metrics['n_test']}")
    print("\nHeld-out test performance")
    print(f"  RMSE : {metrics['test_rmse']:.3f} log units")
    print(f"  MAE  : {metrics['test_mae']:.3f} log units")
    print(f"  R^2  : {metrics['test_r2']:.3f}")
    print(f"  5-fold CV RMSE : {metrics['cv_rmse_mean']:.3f} +/- {metrics['cv_rmse_std']:.3f}")

    print("\nTop feature importances")
    top = sorted(metrics["feature_importance"].items(), key=lambda kv: -kv[1])[:6]
    for name, imp in top:
        print(f"  {name:22s} {imp:.3f}")

    print(f"\nSaved model + metrics to data/. Metrics:\n{json.dumps(metrics['feature_importance'], indent=2)[:0]}")


if __name__ == "__main__":
    main()
