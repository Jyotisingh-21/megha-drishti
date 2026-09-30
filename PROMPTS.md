# PROMPTS.md — Build guide for Google Antigravity

Follow the parts in order. Part A and B you do by hand once. Part C is the set of prompts you paste into Antigravity's agent panel, one at a time.

---

## Part A — One-time setup on your machine (10 min)

1. **Install tools:** Git, Python 3.11+, [GitHub CLI (`gh`)](https://cli.github.com/), and Google Antigravity.
2. **Log in to GitHub** so the agent can push without you handing it a token:
   ```bash
   gh auth login          # choose GitHub.com → HTTPS → Login with browser
   gh auth setup-git      # lets git use the gh credential
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```
3. **Create the project and GitHub repo** by running the bootstrap script (it creates the folder, `.gitignore`, agent rules, the first commit, and a new GitHub repo, then pushes):
   ```bash
   chmod +x bootstrap.sh
   ./bootstrap.sh hybrid-nwp-blend public      # or "private"
   ```
   Put `README.md`, `AGENTS.md` and `PROMPTS.md` in the same folder as `bootstrap.sh` before you run it. The script copies them in.
4. **Confirm:** open `https://github.com/<you>/hybrid-nwp-blend`. You should see the README and an initial commit.

## Part B — Antigravity setup (5 min)

1. Open Antigravity → **Open Folder** → select `hybrid-nwp-blend`.
2. Use **Planning mode** so the agent shows a plan and task list before it edits. Pick the strongest model available to you.
3. **Rules:** the script placed the agent rules at `.agent/rules/project.md` (a copy of `AGENTS.md`). Confirm they are visible in the Antigravity rules / customizations panel. If your version doesn't pick up that path, paste the contents of `AGENTS.md` into the workspace rules there, or keep `AGENTS.md` at the repo root and start each session by telling the agent to read it. (Antigravity's settings UI changes between versions, so check its current docs if a menu is missing.)
4. **Terminal permissions:** allow the agent to run `git`, `gh`, `python`, `pip`, `pytest` and `ruff` in the terminal. Prefer an allow-list or "request review" policy over fully autonomous mode, so you see every push.
5. **Workflow shortcut:** the script also created `.agent/workflows/commit-and-push.md`. You can invoke it with `/commit-and-push` after each step.
6. **Smoke test:** paste this into the agent to confirm git works before any real work:
   ```
   Run `git remote -v`, `git log --oneline -3` and `gh auth status`. Report the output. Do not change anything.
   ```

## How to use the prompts

- Paste **one** prompt, read the plan Antigravity produces, approve it, let it run.
- When it finishes, check the summary, run the acceptance command yourself, and confirm that a commit appeared on GitHub.
- Only then paste the next prompt. If something breaks, use the "Fix" prompt in Part D.

> Every prompt ends with the same commit instruction. That instruction is what makes the code land on GitHub after each step.

---

## Part C — The prompts

### Prompt 0 — Read the brief and scaffold the project

```
Read README.md and AGENTS.md fully. Then scaffold the project exactly as described in the README "Repository layout":

- pyproject.toml (package `nwpblend` in src/, Python >=3.11; core deps: numpy, pandas, xarray, zarr, dask, netcdf4, cfgrib, scipy, scikit-learn, statsmodels, lightgbm, torch, shap, minisom, pyyaml, matplotlib, fastapi, uvicorn, streamlit, requests, s3fs, gcsfs, ecmwf-opendata, imdlib; optional extra `regrid` for xesmf; dev extra: pytest, ruff, pytest-cov)
- empty package folders with __init__.py, tests/ folder, configs/default.yaml and configs/thresholds.yaml with sensible placeholder values and comments
- .env.example, LICENSE (MIT), scripts/run_daily.py stub with argparse flags --demo, --date, --domain
- .github/workflows/ci.yml that installs the dev extra, runs ruff and pytest (no data downloads)
- one trivial passing test

xESMF needs ESMF from conda, so make the regridder pluggable: try xesmf, and fall back to xarray's interp (bilinear) with a logged warning.

Show me the plan first. When finished, verify with: pip install -e ".[dev]" && ruff check . && pytest -q

Then follow AGENTS.md section 6: commit with message "chore: scaffold project structure" and push to origin main.
```

**Acceptance:** tests pass, CI file exists, commit visible on GitHub.

---

### Prompt 1 — Synthetic demo data generator (build this first so everything else can be tested offline)

```
Implement src/nwpblend/ingest/synthetic.py and the canonical schema described in AGENTS.md section 3.

Generate a realistic-enough synthetic dataset over a small India domain (default lat 6–37, lon 68–98, coarsened to ~1° for demo speed, configurable to 0.25°) for ~90 days:

- "truth": daily precip (zero-inflated gamma with a monsoon-like spatial pattern, heavier over the Western Ghats and north-east), t2m, wind10m, gust10m.
- 6 model forecasts for leads 1–10 days: ncum_g, ecmwf_ifs, gfs, aifs, graphcast, pangu. Build each as truth plus structured error, with deliberately DIFFERENT strengths so a good blender has something to learn: e.g. AI models smooth extremes and are good at large scale/timing; NWP models are better over orography and at extremes; error grows with lead; each model is worst in a different region or regime.
- an ensemble (10 members) for ecmwf_ens and neps.
- a hidden "regime" label per day (monsoon-active / monsoon-break / western-disturbance / other) that modulates which model is best. Store it separately as ground truth for testing the regime detector later.

Save as Zarr under data/demo/. Add scripts/download_demo_data.py --demo to generate it, with a fixed random seed. Add tests checking dims, variable names, units, no NaNs where not expected, and that the models really do differ in skill by region and lead.

Acceptance: `python scripts/download_demo_data.py --demo` finishes in under 60 seconds and the tests pass.

Follow AGENTS.md section 6: commit "feat(ingest): add synthetic demo data generator" and push.
```

---

### Prompt 2 — Real data ingest (open sources)

```
Implement real-data ingest under src/nwpblend/ingest/, each with a common interface `fetch(date, domain, leads, variables) -> xarray.Dataset` in the canonical schema:

1. ecmwf.py: ECMWF Open Data (IFS HRES 0.25° and ENS) via the `ecmwf-opendata` client. Variables: 2t, tp, 10u, 10v, 10fg (if available). Convert units (K→°C, m→mm, accumulate tp to 24 h, wind speed from u,v).
2. gfs.py: NOAA GFS from the anonymous AWS bucket noaa-gfs-bdp-pds via s3fs + cfgrib, same variables.
3. aifs.py: ECMWF AIFS single from Open Data, same variables. Add an optional graphcast/pangu path using the `ai-models` package, guarded by an import check and a `--gpu` flag; if it's not installed just log and skip.
4. ncmrwf_adapter.py: a thin adapter that reads NCUM-G / NEPS GRIB2 files from a local directory path in the config, maps their variable names to the schema, and raises a clear error if the folder is empty. I do not have this data; make it testable with a tiny fixture.
5. truth.py: ERA5 from the ARCO-ERA5 anonymous Google Cloud Zarr (primary), IMD gridded rainfall/temperature via imdlib (secondary), and a stub for GPM-IMERG that reads credentials from ~/.netrc and skips gracefully if absent.

Requirements: retries with backoff, local caching in data/raw/, each source failing independently without crashing the run (return None and log), and a `sources_available()` helper. Do NOT download data in tests; mock the network and use small fixtures. Verify real endpoints/variable names by reading the current package docs rather than guessing, and tell me anything you could not verify.

Acceptance: unit tests pass offline; `python scripts/download_demo_data.py --date 2024-07-30 --sources ecmwf,gfs --dry-run` prints the plan.

Follow AGENTS.md section 6: commit "feat(ingest): add ECMWF, GFS, AIFS, NCMRWF adapter and truth loaders" and push.
```

---

### Prompt 3 — Harmonise and store

```
Implement src/nwpblend/harmonise/regrid.py and store.py.

- regrid.py: regrid any Dataset to the common 0.25° grid over the configured domain using xESMF conservative regridding for precip and bilinear for other variables; fall back to xarray interp when xesmf is unavailable (log it). Handle longitude conventions (0–360 vs −180–180).
- store.py: write/append/read a consolidated Zarr store with chunking (time=1, lead=all, lat/lon=64), and a function that stacks all models into one Dataset with a `model` dimension, aligned by (time, lead, lat, lon). Missing models become NaN-filled entries plus a boolean `available` mask (time, model).
- pipeline.py: a first `build_dataset(date_range, models, truth)` function that ties ingest, regrid and store together.

Tests: regridding conserves the area-mean precip within tolerance on a synthetic field; longitude wrap works; missing model produces the mask correctly.

Follow AGENTS.md section 6: commit "feat(harmonise): add regridding and Zarr store" and push.
```

---

### Prompt 4 — Bias correction

```
Implement src/nwpblend/biascorrect/.

- quantile_map.py: empirical quantile mapping per model, per lead-time bin, per season, per grid point (or per small region to avoid overfitting; make that configurable). For precip use a wet-day threshold (0.1 mm) and map only wet-day quantiles, keeping the dry-day frequency corrected by a separate frequency adjustment. Fit on a training window, apply to any period. Extrapolate tails with a constant or scaled extension, never producing negative rain.
- lapse_rate.py: correct t2m for elevation mismatch between model orography and truth-grid orography using a configurable lapse rate (default 6.5 K/km); accept an orography field and, if none is provided, skip with a warning.
- A `BiasCorrector` class with fit/transform/save/load, following the sklearn convention.

Use only training data before the validation period (no leakage). Tests: after correction on synthetic demo data the mean bias and the distribution of each model is closer to truth than before; no negative precip; the transform is monotone.

Also add a small plotting helper that saves a before/after quantile-quantile figure to docs/figures/.

Acceptance: on demo data, report the RMSE and bias of each model before and after correction in a table printed by `python -m nwpblend.biascorrect.report --demo`.

Follow AGENTS.md section 6: commit "feat(biascorrect): quantile mapping and lapse-rate correction" and push.
```

---

### Prompt 5 — Baseline blenders (equal-weight, EWA, BMA) and verification metrics

```
Implement src/nwpblend/blend/baselines.py and src/nwpblend/verify/metrics.py.

baselines.py:
- equal_weight(models)
- ewa: exponentially weighted averaging where weights ∝ exp(-eta * recent_error) over a sliding window (window length and eta configurable), computed per grid point, per lead.
- bma: Bayesian model averaging using EM to fit per-model weights and variances (Gaussian for temperature/wind; for precip use the gamma-mixture form or a documented simplified variant), sliding training window.
All must renormalise weights if a model is unavailable (use the `available` mask).

metrics.py (vectorised, xarray-friendly, with tests against hand-computed values):
RMSE, MAE, bias, ACC (against a climatology argument), CRPS (ensemble and Gaussian closed form), Brier score, reliability-diagram data, FSS (neighbourhood fractions, configurable window and threshold), and categorical scores POD, FAR, CSI from a contingency table.

Add scripts/run_benchmark.py that, on demo data, evaluates: each single model, equal-weight, EWA, BMA, and writes results to docs/results_demo.md as a table. Label it clearly as DEMO (synthetic) data.

Acceptance: pytest passes; EWA and BMA beat equal-weight on demo data (if they do not, investigate and report honestly rather than tuning the data).

Follow AGENTS.md section 6: commit "feat(blend): baselines and verification metrics" and push. Tag v0.1 (Phase 1 complete), tick Phase 1 in the README roadmap.
```

---

### Prompt 6 — Weather regime detection

```
Implement src/nwpblend/regimes/.

- indices.py: load or compute regime predictors. For demo mode use the synthetic regime label as a proxy for the ground truth. For real mode: MJO (RMM index from BoM/NOAA via a documented download, cached), ENSO (Niño3.4) and IOD (DMI) monthly indices, a monsoon active/break flag from a simple rainfall-anomaly index over central India, and a western-disturbance flag from 500 hPa geopotential trough detection or a documented simple proxy. If a download is not possible, expose the function with a clear NotImplemented path and a config switch instead of faking data.
- cluster.py: cluster daily ERA5 fields (850 hPa winds, 500 hPa geopotential height, precipitable water; reduced with PCA) using k-means (k configurable, default 4–6) and optionally a SOM via minisom. Provide fit / predict / save / load and a function `tag_day(date) -> regime_id + soft probabilities`.

Tests: on demo data the clusters recover the hidden regime labels with adjusted Rand index above a threshold you choose and justify; predict is deterministic with a fixed seed.

Follow AGENTS.md section 6: commit "feat(regimes): clustering and climate indices" and push.
```

---

### Prompt 7 — Adaptive gating network (mixture of experts) + SHAP

```
Implement src/nwpblend/blend/gating.py and explain.py.

gating.py: a PyTorch mixture-of-experts blender.
- Inputs per grid point/day: lead time (sin/cos or embedding), lat, lon, elevation (if available), season (sin/cos of day-of-year), regime soft probabilities, recent per-model skill features (e.g. trailing 7-day and 30-day error per model), and optionally each model's forecast value and spread.
- Output: softmax weights over models (respecting the `available` mask by masking logits before the softmax so weights always sum to 1 and missing models get 0).
- Loss: for continuous variables MSE or CRPS on the blended forecast; for precip use a weighted loss that up-weights heavy-rain cases (weights configurable) so extremes are not ignored.
- Training: year-wise / season-wise cross-validation with early stopping; CPU-friendly (target: trains in minutes on demo data); save/load checkpoints under models/.
- Add a LightGBM stacking baseline for comparison.

explain.py: SHAP for the gating network (DeepExplainer or KernelExplainer on a sample), functions to output (a) global feature importance per model, (b) weight maps: for a given date/lead/variable, an xarray of weights per model on the grid, saved as PNG map figures to docs/figures/.

Tests: weights sum to 1 and are non-negative; missing model gets 0 weight; a tiny overfit test where one model is perfect and the gate learns to pick it.

Acceptance: on demo data, gating blend beats EWA and BMA (report honestly; if not, diagnose and explain rather than tweaking the synthetic data), and weight maps show regionally different winners.

Follow AGENTS.md section 6: commit "feat(blend): mixture-of-experts gating network with SHAP" and push.
```

---

### Prompt 8 — EMOS probabilistic calibration

```
Implement src/nwpblend/blend/emos.py.

- Non-homogeneous Gaussian regression (EMOS) for t2m and wind10m: mean = a + b·blend_mean, variance = c + d·ensemble_spread², coefficients fitted by minimum CRPS (closed form for Gaussian) with scipy.optimize, per region/lead-bin and a sliding training window.
- Precipitation: a censored/shifted gamma (or the documented alternative) EMOS with a point mass at zero; provide the probability of exceeding arbitrary thresholds.
- Output an xarray Dataset with calibrated mean, spread, quantiles (10/50/90) and a function `exceedance_prob(threshold)`.

Tests: on synthetic ensemble data that is deliberately under-dispersive, EMOS improves CRPS and the reliability curve is closer to the diagonal; PIT histogram helper added.

Follow AGENTS.md section 6: commit "feat(blend): EMOS probabilistic calibration" and push.
```

---

### Prompt 9 — Extreme event module

```
Implement src/nwpblend/extremes/.

- thresholds.py: load configs/thresholds.yaml. Populate it with IMD heavy-rain categories (heavy, very heavy, extremely heavy, in mm/24 h), heatwave criteria (absolute and departure-from-normal thresholds for plains, coastal and hill regions), and gust thresholds. Look up the current official IMD definitions and cite the source in a YAML comment; tell me if you could not verify any number.
- rain2stage.py: two-stage rainfall model: stage 1 classifier for rain/no-rain (LightGBM, calibrated probabilities), stage 2 amount regression conditional on rain (log or gamma target, event-weighted), combined into a full predictive distribution. Compare against the single-stage blend on heavy-rain CSI and FSS.
- exceedance probabilities: for each threshold, combine EMOS distribution and the two-stage model into P(exceed) maps for Day 1–10, plus a district-level aggregation (accept a district shapefile/GeoJSON path in config; if absent, aggregate to coarse boxes and log it).
- cyclone.py: a simple track consensus: given track positions from several models (input as a small table), compute the mean track, spread and a cone of uncertainty. Keep it lightweight and well documented; do not attempt full vortex tracking.

Tests: thresholds load and are ordered; exceedance probabilities are monotone in threshold; two-stage output is non-negative; cyclone consensus spread is zero when all tracks agree.

Follow AGENTS.md section 6: commit "feat(extremes): two-stage rain model, exceedance probabilities, cyclone consensus" and push. Tag v0.2, tick Phase 2 in README.
```

---

### Prompt 10 — Verification, ablation and event replay

```
Implement src/nwpblend/verify/ablation.py and replay.py, and complete scripts/run_benchmark.py.

- ablation.py: run the ladder raw best single model → equal-weight → + bias correction → + regimes → + gating network → + EMOS/probabilistic layer, using identical data and CV folds, and output a table of RMSE, ACC, CRPS, Brier, FSS, POD, FAR, CSI plus per-region skill maps (PNG).
- replay.py: replay named events with configurable date windows: Wayanad heavy rain (Jul 2024), Cyclone Biparjoy (Jun 2023), Cyclone Remal (May 2024), Delhi heatwave (May–Jun 2024). For each, produce a short report: blend vs each model vs observation, exceedance probabilities issued at Day 1/3/5, and an honest hit/miss note. If real data is unavailable, run the code path on demo data and label it as such.
- Write outputs to docs/results.md (real data) or docs/results_demo.md (synthetic), always stating which data was used, the periods, folds and the baseline definitions.
- Add a "model drift" check: flag a model when its trailing 14-day skill falls below a configurable fraction of its 90-day mean.

Do not put any improvement percentage in README.md unless docs/results.md contains a reproducible number from real data.

Follow AGENTS.md section 6: commit "feat(verify): ablation study, event replay, drift flags" and push. Tag v0.3, tick Phase 3.
```

---

### Prompt 11 — Export + FastAPI

```
Implement src/nwpblend/export/ and src/nwpblend/api/main.py.

- export: write blended forecast, weights and exceedance probabilities to NetCDF (CF-compliant attrs, compression) and to GRIB2 (via eccodes if available, otherwise skip with a clear message and NetCDF only). Include metadata: model versions, training window, issue time, data sources actually used.
- api: FastAPI app with endpoints
  GET /health
  GET /forecast?var=&lead=&date=  (JSON grid or GeoJSON) 
  GET /weights?var=&lead=&date=&model=
  GET /probability?event=heavy_rain|heatwave|high_wind&lead=&date=&district=
  GET /skill?model=&region=&metric=
  GET /drift
  Add pydantic response models, input validation, OpenAPI docs, and read from the latest published output directory.
- Add tests using FastAPI TestClient on demo output.

Follow AGENTS.md section 6: commit "feat(api): NetCDF/GRIB2 export and FastAPI service" and push.
```

---

### Prompt 12 — Dashboard

```
Build dashboard/app.py in Streamlit, reading from the API (or directly from the output directory if the API is down).

Pages:
1. Forecast: map of blended forecast for chosen variable/lead/date, with a comparison toggle to each individual model.
2. Weight maps: which model is trusted where (map + legend), lead-time slider, regime shown for the day.
3. Extreme guidance: probability maps for heavy rain, heatwave and high wind; a district table sorted by highest probability; colour-coded warning levels.
4. Skill and drift: skill trends by model and by region, drift flags, ablation table from docs/results*.md.
5. Explainability: SHAP feature-importance plots per model.
6. About: data sources, limitations, and a clear banner showing DEMO vs REAL data mode.

Make it work in demo mode with no internet. Use plotly or folium/pydeck for maps. Keep the design clean and readable. Add English and Hindi labels for the alert text (a simple i18n dict, structured so more languages can be added).

Follow AGENTS.md section 6: commit "feat(dashboard): Streamlit dashboard with weight maps and extreme guidance" and push.
```

---

### Prompt 13 — Daily workflow, Docker, CI/CD

```
Complete the operational workflow.

- src/nwpblend/pipeline.py: `run_daily(date, domain, demo)` executing ingest → harmonise → bias-correct → regime tag → blend → EMOS → extremes → export → verify yesterday's forecast and update rolling skill/weights. Each stage logs timing and failures; a missing model feed renormalises weights and adds a warning to the run report; write a run_report.json.
- scripts/run_daily.py wired to it, with --demo, --date, --domain, --skip-download.
- A Prefect flow (optional extra) OR a documented cron/`schedule` alternative; pick the simpler and document it.
- docker/Dockerfile (slim, multi-stage) and docker-compose.yml with services: pipeline (one-shot), api, dashboard, sharing a data volume.
- CI: extend .github/workflows/ci.yml to also run an end-to-end demo-mode smoke test and build the Docker image (no push to a registry).
- Update README quick-start to match what really works. Run every command in it to verify.

Follow AGENTS.md section 6: commit "feat(ops): daily pipeline, Docker, scheduler, CI" and push. Tag v0.4, tick Phase 4.
```

---

### Prompt 14 — Polish for the hackathon demo

```
Final polish.

1. Run the whole thing from a clean clone in a fresh virtual environment following only the README. Fix any step that fails.
2. Generate final figures into docs/figures/ (weight maps, skill maps, reliability diagram, ablation bar chart, one event replay) and reference them from the README.
3. Add docs/ARCHITECTURE.md (diagram and data flow), docs/LIMITATIONS.md (honest list: demo vs real data, NCMRWF data not tested, targets not verified, etc.) and docs/DEMO_SCRIPT.md (a 5-minute demo walkthrough for the jury).
4. Update README: replace placeholders, add badges for CI, keep the "Targets vs results" section honest.
5. Run `ruff`, `pytest --cov`, and report coverage.

Follow AGENTS.md section 6: commit "docs: final polish and demo script" and push. Tag v1.0.0.
```

---

## Part D — Helper prompts

**Fix a failing step**
```
The last step failed with the error below. Diagnose the root cause first (don't guess), explain it in two lines, fix it, re-run ruff + pytest, and commit with a `fix(...)` Conventional Commit message and push. 
<paste error here>
```

**Commit and push manually at any time**
```
Run ruff and pytest. If they pass, show me `git status`, stage only the intended files, commit with a Conventional Commit message that describes the change, run `git push origin main`, and show me the resulting commit hash. If the push is rejected, pull --rebase, re-test, push again. Never force-push.
```

**Review before continuing**
```
Act as a strict reviewer of the last step. Check for: data leakage, hard-coded paths or thresholds, unit errors (K vs °C, m vs mm), weights not summing to 1, missing tests, claims in docs not backed by output. List issues by severity. Fix the critical ones.
```

**Sanity check that nothing is fabricated**
```
List every number (percentages, skill scores, improvements) that appears in README.md and docs/. For each, tell me the exact script and output file that produced it. Remove or clearly label any that you cannot trace.
```

**Recover context in a new session**
```
Read README.md, AGENTS.md and `git log --oneline -20`. Tell me which step of PROMPTS.md is complete, which is next, and whether the working tree is clean. Do not change anything yet.
```

---

## Part E — Practical notes

- **Time budget:** if you have a 36-hour final, do Prompts 0, 1, 3, 4, 5, 7, 9 (rain only), 11, 12 first. That gives a working end-to-end demo. Do 2, 6, 8, 10, 13, 14 as time allows. Prompt 1 (synthetic data) is what keeps you unblocked while real data downloads are slow.
- **Real vs demo results:** the jury will care that skill numbers are honest. Present demo-data results as a functional test of the pipeline, and present real-data results (even a small case such as JJAS 2024 over a few states) as the evidence.
- **Domain size:** start with a sub-domain (e.g. Kerala + Karnataka for Western Ghats rain, plus Delhi–UP for heat) before running all-India at 0.25°.
- **GPU:** you do not need one. ECMWF publishes AIFS output as open data, so fetch that instead of running GraphCast or Pangu yourself.
- **Costs and rate limits:** Antigravity usage limits vary by plan. Keep prompts one step at a time and don't let the agent loop on failing downloads.
- **Never paste tokens** (GitHub, CDS, Earthdata) into a prompt or file. Use `gh auth login`, `~/.cdsapirc` and `~/.netrc`.
