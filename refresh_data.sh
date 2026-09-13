#!/usr/bin/env bash
# Runs the same data/model refresh pipeline the GitHub Actions workflow used to run.
# Moved local because the CI runner's IP gets every FastF1 request rejected -
# see the comment in .github/workflows/weekly_refresh.yml for details.
#
# Usage: ./refresh_data.sh
# Run this from your own machine every so often (e.g. weekly) to catch up on
# new race data. It does NOT commit or push - review the changes yourself
# with `git status` / `git diff --stat` and commit when you're happy.

set -e
cd "$(dirname "$0")"

if [ -d venv ]; then
  source venv/bin/activate
fi

echo "== 1/5 Fetch latest race results =="
python etl_process.py

echo "== 2/5 Generate quali/practice training features =="
python generate_features.py

echo "== 3/5 Retrain qualifying model =="
python train_model.py

echo "== 4/5 Generate master feature store =="
python generate_master_features.py

echo "== 5/5 Retrain race model suite =="
python train_master_model.py

echo ""
echo "Done. Review changes with:"
echo "  git status"
echo "  git diff --stat"
echo "Then commit + push when ready."
