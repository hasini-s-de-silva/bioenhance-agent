"""Run the evaluation harness and write results/ artefacts.

Usage:
    python -m scripts.run_evaluation                 # auto backend (LLM if key present)
    python -m scripts.run_evaluation --backend rulebased
    python -m scripts.run_evaluation --repeats 5
"""

from __future__ import annotations

import argparse

from src.evaluation import run_evaluation, save, to_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", choices=["auto", "anthropic", "rulebased"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--configs", nargs="*", default=None, help="subset of: llm_only llm_rag full"
    )
    args = parser.parse_args()

    report = run_evaluation(
        backend=args.backend, configs=args.configs, stability_repeats=args.repeats
    )
    save(report)

    print("\n" + "=" * 78)
    print(to_markdown(report))
    print("Wrote results/evaluation.json and results/evaluation.md")


if __name__ == "__main__":
    main()
