# AGENTS.md — Instructions for the AI coding agent

You are building **Hybrid AI–NWP Forecast Blending System** (SIH 2026, PS 26081, MoES–NCMRWF).
Read `README.md` first. It is the source of truth for scope, data sources, layout and metrics.

## 1. Mission

Build a pipeline that ingests forecasts from physical NWP, ensembles and AI weather models, bias-corrects them, tags the weather regime, learns adaptive per-grid-point blend weights, calibrates probabilities, outputs extreme-event probabilities, verifies against observations, and publishes results via files, an API and a dashboard.

## 2. Ground rules

1. **Work in small steps.** One step from `PROMPTS.md` at a time. Do not start the next step until the current one passes its acceptance checks.
2. **Plan before coding.** For each step, write a short plan (files to create, functions, tests) before editing anything.
3. **Never fabricate results.** Any metric, skill score or improvement figure must come from code that was actually run. If a target (e.g. "10–15% lower RMSE") is not reproduced, say so plainly. Do not put invented numbers in the README, docs or dashboard.
4. **Never fabricate data sources.** If a download fails or needs credentials, fall back to `--demo` synthetic data and log it. Do not silently substitute.
5. **No secrets in git.** Never commit API keys, `.cdsapirc`, `.netrc`, `.env`, or raw data. Use `.env.example` for placeholders.
6. **Demo mode must always work.** `python scripts/run_daily.py --demo` must run end to end offline in under 5 minutes on a laptop CPU. Protect this in every step.
7. **Ask instead of guessing** when a requirement is ambiguous and the choice is hard to reverse. Otherwise choose a sensible default, document it in the code, and continue.

## 3. Tech and code standards

- Python 3.11, package `nwpblend` under `src/`, installable with `pip install -e ".[dev]"`.
- Data: xarray, Zarr, xESMF, cfgrib, dask. Models: LightGBM, PyTorch, statsmodels, scikit-learn, SHAP.
- Type hints on all public functions. Docstrings state units, dimensions and coordinate names.
- Format and lint with `ruff`. Tests with `pytest`. Keep files under ~300 lines.
- Configuration in `configs/*.yaml`. No hard-coded paths, thresholds or model lists in code.
- Log with the `logging` module, not `print`.
- Canonical data schema (do not deviate):
  - Dims: `time` (init), `lead` (timedelta or hours), `lat`, `lon`, plus `member` for ensembles and `model` when stacked.
  - Variables: `precip` (mm per 24 h), `t2m` (°C), `wind10m` (m s⁻¹), `gust10m` (m s⁻¹).
  - Grid: 0.25° regular lat/lon over the configured domain. Rename model-specific variable names at ingest.
- Preserve units and attributes in xarray. Convert kelvin to °C and metres to mm at ingest and record it in `attrs`.

## 4. Scientific guardrails

- **No leakage.** Training must only use data available before the forecast issue time. Cross-validate by year or season, never by random split.
- **Precipitation** is zero-inflated and skewed. Handle it with the two-stage model (occurrence, then amount) and quantile mapping. Do not use plain RMSE alone; report FSS and CSI too.
- **Weights** are non-negative and sum to 1 per grid point (softmax output). If a model is missing, renormalise.
- **Probabilities** must be calibrated. Check reliability diagrams and Brier score, not just accuracy.
- **Thresholds** (heavy rain, heatwave, gusts) live in `configs/thresholds.yaml`. Verify the IMD definitions before finalising and cite the source in a comment.
- **Comparisons must be fair.** Every claim of improvement is against: each single model, the equal-weight mean, EWA and BMA, on the same data, same period, same grid.

## 5. Testing requirements

- Every module gets at least one unit test in `tests/` using tiny synthetic arrays (fast, offline).
- Add an end-to-end smoke test that runs the pipeline in demo mode.
- Run before every commit: `ruff check . && ruff format --check . && pytest -q`

## 6. Git and GitHub workflow (mandatory)

The repo already has a `main` branch and a GitHub remote named `origin`. Before starting, run `git remote -v` and `gh auth status` to confirm both.

**After each completed step:**

1. Run the checks: `ruff check . && ruff format . && pytest -q`. Do not commit if they fail. Fix first.
2. `git status` to review. Make sure no data files, secrets or `.venv` are staged.
3. Stage only files belonging to the step: `git add <paths>` (avoid blind `git add -A` unless `.gitignore` is verified).
4. Commit using **Conventional Commits**, one logical commit per step (more if the step is large):
   - `feat(ingest): add ECMWF open data downloader`
   - `fix(regrid): handle longitude wrap at 0/360`
   - `test(blend): add BMA weight-sum test`
   - `docs: update README roadmap`
   - `chore(ci): add GitHub Actions workflow`
5. `git push origin main`. If the push is rejected, run `git pull --rebase origin main`, re-run tests, then push again. Never force-push.
6. After each **phase** (1 to 4 in the README roadmap), tag it: `git tag -a v0.<phase> -m "<summary>" && git push origin --tags`, and tick the box in the README roadmap.
7. Report to the user: commit hash, files changed, tests run, and anything left undone.

If `git push` fails because of authentication, **stop and tell the user**. Do not try to work around it, store tokens in files, or change credentials.

## 7. Definition of done for each step

- Code implemented and matches the schema in section 3.
- Tests written and passing.
- `run_daily.py --demo` still works.
- Docstrings and any config keys documented.
- Committed and pushed with a Conventional Commit message.
- A 3–5 line summary posted to the user with what was built, how to run it, and known limitations.

## 8. What to avoid

- Big-bang commits containing many unrelated changes.
- Adding heavy dependencies without need. Justify each new one.
- Downloading large datasets inside tests or CI.
- Claiming operational readiness. This is a research prototype until validated against NCMRWF data.
- Editing `README.md` result tables with numbers that were not produced by `scripts/run_benchmark.py`.
