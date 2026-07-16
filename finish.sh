#!/usr/bin/env bash
# Waits for the evaluation to finish, then commits and pushes the real results.
# Run it and walk away:   ./finish.sh
set -uo pipefail
cd "$(dirname "$0")"

echo "Waiting for the evaluation to finish (it may already be done)..."
while pgrep -f "run_evaluation" >/dev/null; do sleep 15; done

if ! grep -q "Backend" results/evaluation.md 2>/dev/null; then
    echo "No results were written — the run failed. Nothing pushed."
    tail -5 /tmp/eval2.log 2>/dev/null
    exit 1
fi

BACKEND=$(grep -m1 '^- Backend' results/evaluation.md)
echo "Results present: $BACKEND"

if grep -q "rule-based" results/evaluation.md; then
    echo "These are the rule-based results, not the local-LLM run. Not pushing."
    exit 1
fi

git add results/ README.md
git commit -q -m "Add local-LLM evaluation results (qwen2.5:7b via Ollama)

Generated locally at zero cost; reproducible with:
  ollama pull qwen2.5:7b
  python -m scripts.run_evaluation --backend ollama" || { echo "nothing to commit"; exit 0; }
git push -q origin main && echo "Pushed. Table is live at:"
echo "  https://github.com/hasini-s-de-silva/bioenhance-agent/blob/main/results/evaluation.md"
