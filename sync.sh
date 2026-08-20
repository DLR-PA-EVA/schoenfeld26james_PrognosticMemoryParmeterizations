#!/bin/bash
# Copy tracked files in this publication repo from their live HPC working
# directories, so you can review and commit the changes deliberately.
#
# Usage:
#   ./sync.sh            # sync both experiments
#   ./sync.sh DYAMOND     # sync just one
#
# New files are NOT picked up automatically: `cp` the file into the repo
# and `git add` it once, and future syncs will keep it up to date.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -A LIVE_DIR=(
  [L96]="/work/bd1179/b309297/L96"
  [DYAMOND]="/work/bd1179/b309297/DYAMOND_experiment/DYAMOND"
)

TARGETS=("${!LIVE_DIR[@]}")
if [ $# -gt 0 ]; then
  TARGETS=("$@")
fi

cd "$REPO_ROOT"
for exp in "${TARGETS[@]}"; do
  live="${LIVE_DIR[$exp]:-}"
  if [ -z "$live" ]; then
    echo "unknown target: $exp (known: ${!LIVE_DIR[*]})" >&2
    exit 1
  fi
  echo "== $exp  <-  $live =="
  git ls-files "$exp" | grep -v -e '/\.gitignore$' -e '^README.md$' | while read -r repo_path; do
    rel="${repo_path#"$exp"/}"
    src="$live/$rel"
    if [ -f "$src" ]; then
      cp "$src" "$repo_path"
    else
      echo "  WARNING: $src no longer exists in live dir, leaving $repo_path untouched" >&2
    fi
  done
done

echo
echo "Done. Review changes with: git status / git diff"
