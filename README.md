# Megha-Drishti: Hybrid AI–NWP Forecast Blending System

[![CI Status](https://github.com/Jyotisingh-21/megha-drishti/actions/workflows/ci.yml/badge.svg)](https://github.com/Jyotisingh-21/megha-drishti/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

> **SIH 2026 · PS 26081 · MoES – NCMRWF**
> An adaptive, explainable framework that learns *where, when and for which weather regime* each forecast model is trustworthy, and blends physical NWP, ensembles and AI weather models into one calibrated forecast with extreme-event probabilities.

---

## 1. Problem

No single forecast system wins everywhere. Skill changes with region, season, lead time and weather regime. A model that is best for Day-2 monsoon rain over the Western Ghats may be weak for a Day-6 heatwave over the Indo-Gangetic Plain. Forecasters today compare models by hand or use fixed-weight averages that cannot adapt.

| Source family | Examples | Strength | Weakness |
|---|---|---|---|
| Physical NWP | NCUM-G, ECMWF IFS, GFS | Physics, orography, cyclone dynamics | Displacement errors, expensive |
| Ensembles | NEPS, ECMWF ENS | Uncertainty and probabilities | Under-dispersive, need calibration |
| AI models | GraphCast, Pangu, AIFS, NeuralGCM | Fast, skilful large-scale patterns | Smooth fields, under-predict extremes |

## 2. What this repo delivers

1. **Dynamic blend**: one best combined forecast for rainfall, 2 m temperature and 10 m wind.
2. **Weight maps**: which model to trust, by grid point, lead time, season and regime.
3. **Higher skill**: benchmarked against every individual model and the equal-weight mean.
4. **Extreme guidance**: exceedance probabilities for heavy rain, heatwave and high wind.
5. **Operational workflow**: one command runs the daily pipeline; a FastAPI service and a dashboard publish the results.

## 3. Pipeline

```
 ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────┐
 │ 1 Ingest  │→ │ 2 Bias-    │→ │ 3 Regime │→ │ 4 Adaptive │→ │ 5 Extreme│→ │ 6 Verify &    │
 │ harmonise │  │   correct  │  │   detect │  │   blend    │  │   module │  │   learn       │
 └───────────┘  └────────────┘  └──────────┘  └────────────┘  └──────────┘  └───────────────┘
  xESMF, Zarr    quantile map,   k-means/SOM   EWA/BMA →       2-stage rain,  score vs obs,
  0.25° grid     lapse-rate      + MJO/ENSO    regime-aware →   thresholds,    update weights,
                                 /IOD/WD       gating net+SHAP  cyclone track  drift flags,
                                               + EMOS calib.    consensus      GRIB2/NetCDF/API
```

**Three layers of adaptive weighting**

1. *Baseline*: exponentially weighted averaging (EWA) and Bayesian model averaging (BMA) over a sliding recent-skill window.
2. *Regime-aware*: separate weights for monsoon-active, monsoon-break and western-disturbance days.
3. *Learned gating*: a mixture-of-experts network outputs per-grid-point weights, explained with SHAP.

## 4. Data sources

| Role | Source | Access | Notes |
|---|---|---|---|
| NWP + ensemble | ECMWF Open Data (IFS HRES, ENS) | `ecmwf-opendata` | Free, 0.25° |
| NWP | NOAA GFS | AWS `noaa-gfs-bdp-pds`, anonymous | Free |
| AI model | ECMWF AIFS | ECMWF Open Data | No GPU needed |
| AI model | GraphCast, Pangu | `ai-models` package | GPU recommended |
| AI model | NeuralGCM | `neuralgcm` package | Optional |
| Domestic NWP | NCUM-G / NEPS | Adapter reading GRIB2 | Restricted; open-data fallback uses the identical schema |
| Truth | ERA5 | ARCO-ERA5 (Google Cloud, anonymous) or CDS API | Training and regimes |
| Truth | IMD 0.25° rain, 1° temperature | `imdlib` | Rainfall and temperature verification |
| Truth | GPM-IMERG | NASA Earthdata login | Rainfall |
| Truth | IMDAA | NCMRWF / NCAR RDA | Optional |

> **Fallback rule:** if any feed is unavailable, the pipeline renormalises weights over the remaining models and logs a warning. If *all* real data is unavailable, `--demo` mode uses a synthetic generator with the same schema so the whole stack still runs.

## 5. Tech stack

- **Data:** Python 3.11, xarray, Zarr, xESMF, cfgrib, dask
- **Models:** LightGBM (baseline), PyTorch (gating network), statsmodels / scipy (EMOS), scikit-learn + minisom (regimes), SHAP
- **Serving:** FastAPI, Docker, Prefect or cron
- **Front end:** Streamlit (fast) or React + MapLibre (stretch)
- **Quality:** pytest, ruff, GitHub Actions

## 6. Repository layout

```
.
├── README.md
├── AGENTS.md                  # rules for the AI coding agent
├── PROMPTS.md                 # step-by-step build prompts
├── pyproject.toml
├── configs/
│   ├── default.yaml           # domain, models, variables, lead times
│   └── thresholds.yaml        # IMD heavy-rain / heatwave / wind thresholds
├── src/nwpblend/
│   ├── ingest/                # ecmwf.py, gfs.py, aifs.py, ncmrwf_adapter.py, truth.py, synthetic.py
│   ├── harmonise/             # regrid.py, store.py
│   ├── biascorrect/           # quantile_map.py, lapse_rate.py
│   ├── regimes/               # cluster.py, indices.py
│   ├── blend/                 # baselines.py, gating.py, emos.py, explain.py
│   ├── extremes/              # rain2stage.py, thresholds.py, cyclone.py
│   ├── verify/                # metrics.py, ablation.py, replay.py
│   ├── export/                # grib2.py, netcdf.py
│   ├── api/                   # main.py
│   └── pipeline.py            # daily orchestration
├── dashboard/app.py
├── scripts/                   # run_daily.py, download_demo_data.py
├── tests/
├── docker/Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## 7. Quick start

```bash
# 1. Clone and set up
git clone https://github.com/Jyotisingh-21/megha-drishti.git
cd megha-drishti
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Run the full pipeline on synthetic demo data (no downloads, no credentials)
python scripts/run_daily.py --demo

# 3. Launch the API and dashboard
uvicorn src.nwpblend.api.main:app --reload         # http://localhost:8000/docs
streamlit run dashboard/app.py                 # http://localhost:8501

# 4. Run on real open data for the latest available cycle
python scripts/run_daily.py --date latest

# 5. Schedule operational runs (Windows)
# Right-click scripts\schedule_windows.ps1 -> Run with PowerShell
#
# Linux Cron equivalent (UTC times):
# 30 08,20 * * * cd /path/to/megha-drishti && /path/to/.venv/bin/python scripts/run_daily.py --date latest >> data/logs/cron.log 2>&1
```

### Screenshots

See the UI in action (generated automatically):
- [Command Center](docs/figures/dashboard_command_center.png)
- [Forecast Map](docs/figures/dashboard_forecast_map.png)
- [Point Explorer](docs/figures/dashboard_point_explorer.png)
- [Extreme Guidance](docs/figures/dashboard_extreme_guidance.png)
- [Event Replay](docs/figures/dashboard_event_replay.png)
- [Weight Maps](docs/figures/dashboard_weight_maps.png)
- [Skill & Drift](docs/figures/dashboard_skill_and_drift.png)

### Credentials (only for real data)

| Service | Needed for | Setup |
|---|---|---|
| CDS API | ERA5 (if not using ARCO-ERA5) | `~/.cdsapirc` |
| NASA Earthdata | IMERG | `~/.netrc` |
| None | ECMWF Open Data, GFS, ARCO-ERA5, IMD via `imdlib` | – |

## 8. Validation plan

- **Periods:** monsoon JJAS 2023–24 and summer 2024.
- **Cross-validation:** year-wise / leave-one-season-out, so nothing leaks from the future.
- **Baselines:** each individual model, equal-weight mean, EWA, BMA.
- **Metrics:** RMSE, ACC, CRPS, Brier score, FSS (handles the rainfall double penalty), POD, FAR, CSI.
- **Ablation:** raw → + bias correction → + regimes → + gating → + EMOS, showing what each part adds.
- **Event replay:** Wayanad landslides 2024, Cyclone Biparjoy 2023, Cyclone Remal 2024, Delhi heatwave 2024.

> **Targets vs results.** The design targets (10-15% lower rainfall RMSE than the best single model, 15-25% higher CSI for heavy rain) are design objectives. **The metrics available in this repository (docs/results.md) are generated using verified real-world meteorological data from ECMWF and GFS, evaluated against IMD and ARCO-ERA5.** Note that due to limited training history and S3 rate limits during ingestion, the current blending weights may not significantly outperform the best single model. See the results document for exact figures.

## 8.5 Scheduling

For Windows, run scripts\schedule_windows.ps1 to register a daily Task Scheduler job.

For Linux, add the following to your crontab (crontab -e):
`ash
0 6 * * * cd /path/to/megha-drishti && /path/to/venv/bin/python scripts/run_daily.py --date latest >> data/logs/cron.log 2>&1
`

## 9. Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 · Data + baseline | Ingest, regrid, bias-correct; equal-weight, EWA and BMA baselines | ✅ |
| 2 · Adaptive + probabilistic | Regimes, gating network, EMOS calibration | ✅ |
| 3 · Extremes + verification | Threshold probabilities, event replay, ablation | ✅ |
| 4 · Deploy | Dashboard, FastAPI, GRIB2 export, Docker, scheduler | ✅ |

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| NCMRWF data restricted | Open-data fallback with identical schema; thin GRIB2 adapter |
| A model feed goes missing | Automatic weight renormalisation |
| Rare extremes, overfitting | Year-wise CV, event-weighted loss |
| Compute cost | Daily cadence, cached AI runs, chunked Zarr |
| Trust | SHAP explanations, skill dashboards |

## 11. References

1. Raftery et al. (2005). Using Bayesian model averaging to calibrate forecast ensembles. *Mon. Wea. Rev.* 133, 1155–1174.
2. Gneiting et al. (2005). Calibrated probabilistic forecasting using EMOS and minimum CRPS estimation. *Mon. Wea. Rev.* 133, 1098–1118.
3. Lam et al. (2023). GraphCast. *Science* 382, 1416–1421.
4. Bi et al. (2023). Pangu-Weather. *Nature* 619, 533–538.
5. Lang et al. (2024). AIFS. arXiv:2406.01465.
6. Kochkov et al. (2024). Neural general circulation models. *Nature* 632, 1060–1066.
7. Hersbach et al. (2020). The ERA5 global reanalysis. *QJRMS* 146, 1999–2049.
8. Rani et al. (2021). IMDAA reanalysis. *J. Climate* 34, 5109–5133.
9. Pai et al. (2014). 0.25° daily gridded rainfall over India. *MAUSAM* 65, 1–18.
10. Roberts and Lean (2008). Scale-selective verification of rainfall accumulations. *Mon. Wea. Rev.* 136, 78–97.
11. Jacobs et al. (1991). Adaptive mixtures of local experts. *Neural Computation* 3, 79–87.
12. Lundberg and Lee (2017). A unified approach to interpreting model predictions. *NeurIPS* 30.

## 12. License and acknowledgements

MIT License (see `LICENSE` file). Built for Smart India Hackathon 2026, PS 26081, Ministry of Earth Sciences (NCMRWF).

**Data Attribution**:
- **ECMWF Open Data**: Contains modified ECMWF information. This information is published by ECMWF under a CC-BY-4.0 licence.
- **NOAA GFS**: Accessed via the NOAA Big Data Program.
- **Copernicus ARCO-ERA5**: Contains modified Copernicus Climate Change Service information (CC-BY-4.0).
- **IMD**: Data provided by the India Meteorological Department.
- **DataMeet**: Boundary data sourced from the DataMeet community (CC-BY-2.5-IN).

See [docs/DATA_LICENCES.md](docs/DATA_LICENCES.md) for full licensing details.
 
 # #   K n o w n   L i m i t a t i o n s  
 -   * * E C M W F   0 . 1 �   L a y o u t * * :   P r e p a r e d   b u t   n o t   v e r i f i e d   a g a i n s t   l i v e   f i l e s .   ( T O D O :   R e - t e s t   a r e a - a v e r a g i n g   c o n s e r v a t i v e   r e g r i d d i n g   w h e n   E C M W F   f u l l y   s w i t c h e s   t o   0 . 1 �   O p e n   D a t a ) .  
 