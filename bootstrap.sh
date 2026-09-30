#!/usr/bin/env bash
# bootstrap.sh — create the project folder, agent rules, first commit and GitHub repo.
# Usage:  ./bootstrap.sh [repo-name] [public|private]
# Run it from the folder that contains README.md, AGENTS.md and PROMPTS.md.

set -euo pipefail

REPO_NAME="${1:-hybrid-nwp-blend}"
VISIBILITY="${2:-public}"
SRC_DIR="$(pwd)"

# ---- 1. Pre-flight checks ---------------------------------------------------
for cmd in git gh; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' is not installed."; exit 1; }
done

gh auth status >/dev/null 2>&1 || {
  echo "ERROR: GitHub CLI is not logged in. Run: gh auth login && gh auth setup-git"
  exit 1
}

for f in README.md AGENTS.md PROMPTS.md; do
  [ -f "$SRC_DIR/$f" ] || { echo "ERROR: $f not found in $SRC_DIR"; exit 1; }
done

if [ -z "$(git config --global user.name || true)" ] || [ -z "$(git config --global user.email || true)" ]; then
  echo "ERROR: set your git identity first:"
  echo '  git config --global user.name  "Your Name"'
  echo '  git config --global user.email "you@example.com"'
  exit 1
fi

# ---- 2. Create project folder ----------------------------------------------
PROJECT_DIR="$SRC_DIR/$REPO_NAME"
if [ -e "$PROJECT_DIR" ]; then
  echo "ERROR: $PROJECT_DIR already exists. Choose another name or remove it."
  exit 1
fi
mkdir -p "$PROJECT_DIR"
cp "$SRC_DIR/README.md" "$SRC_DIR/AGENTS.md" "$SRC_DIR/PROMPTS.md" "$PROJECT_DIR/"
cd "$PROJECT_DIR"

mkdir -p .agent/rules .agent/workflows configs src/nwpblend scripts tests docs/figures \
         dashboard docker .github/workflows data

# Agent rules: same content as AGENTS.md
cp AGENTS.md .agent/rules/project.md

# Reusable workflow: /commit-and-push
cat > .agent/workflows/commit-and-push.md <<'EOF'
---
description: Run checks, commit the current step with a Conventional Commit message, and push to GitHub
---

1. Run `ruff check . && ruff format . && pytest -q`. If anything fails, stop and fix it first.
2. Run `git status` and confirm no data files, secrets, or virtualenv folders are staged.
3. Stage only the files that belong to this step with `git add <paths>`.
4. Commit with a Conventional Commit message (feat, fix, test, docs, chore, refactor) describing the change.
5. Run `git push origin main`. If rejected, run `git pull --rebase origin main`, re-run the tests, and push again. Never force-push.
6. Report the commit hash, files changed, and tests run.
EOF

# ---- 3. .gitignore, .env.example -------------------------------------------
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
build/
dist/

# Secrets and credentials
.env
.cdsapirc
.netrc
*.pem
*.key

# Data and models (large / regenerable)
data/
!data/.gitkeep
models/*.pt
models/*.pkl
*.zarr/
*.nc
*.grib
*.grib2
*.grb
*.idx

# Notebooks / OS / editors
.ipynb_checkpoints/
.DS_Store
.idea/
.vscode/
EOF
touch data/.gitkeep

cat > .env.example <<'EOF'
# Copy to .env and fill in. NEVER commit .env
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=
# NASA Earthdata (IMERG): store in ~/.netrc instead of here if you can
EARTHDATA_USER=
EARTHDATA_PASS=
# Optional: local path to NCMRWF NCUM-G / NEPS GRIB2 files
NCMRWF_DATA_DIR=
EOF

# ---- 4. Git init, first commit, GitHub repo, push --------------------------
git init -b main
git add .
git commit -m "docs: add project README, agent rules and build prompts"

gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin --push \
  --description "Hybrid AI-NWP multi-model forecast blending system (SIH 2026, PS 26081, MoES-NCMRWF)"

echo
echo "Done."
echo "  Local folder : $PROJECT_DIR"
echo "  GitHub repo  : $(gh repo view --json url -q .url)"
echo
echo "Next: open '$PROJECT_DIR' in Antigravity and paste Prompt 0 from PROMPTS.md."
