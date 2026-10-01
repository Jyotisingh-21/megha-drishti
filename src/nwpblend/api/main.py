import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Megha-Drishti API",
    description="Hybrid AI-NWP Forecast Blending System API",
    version="0.1.0",
)

# In-memory data store for the API
DATA_STORE = {"forecast": None, "truth": None}


def get_forecast_data():
    if DATA_STORE["forecast"] is None:
        try:
            DATA_STORE["forecast"] = xr.open_zarr("data/processed/stacked_models.zarr")
        except Exception:
            try:
                DATA_STORE["forecast"] = xr.open_zarr("data/demo/models/ecmwf_ifs.zarr")  # fallback
            except Exception:
                raise HTTPException(status_code=503, detail="Forecast data not available")
    return DATA_STORE["forecast"]


class HealthResponse(BaseModel):
    status: str


class GridResponse(BaseModel):
    variable: str
    date: str
    lead: int
    lats: list[float]
    lons: list[float]
    values: list[list[float]]


class ProbResponse(BaseModel):
    event: str
    date: str
    lead: int
    probabilities: list[list[float]]


class SkillResponse(BaseModel):
    model: str
    metric: str
    value: float


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@app.get("/forecast", response_model=GridResponse)
def get_forecast(
    var: str = Query(..., description="Variable name (e.g. t2m, precip)"),
    lead: int = Query(..., description="Lead time in hours"),
    date: str = Query(..., description="Forecast issue date YYYY-MM-DD"),
):
    ds = get_forecast_data()
    if var not in ds.data_vars:
        raise HTTPException(status_code=400, detail=f"Variable {var} not found")

    try:
        time_sel = pd.to_datetime(date)
        # Extract subset
        sub = ds[var].sel(time=time_sel, method="nearest")
        if "lead" in sub.dims:
            # Assuming lead is an index or coordinate. We'll pick the nearest index.
            # In a real app we would strictly match lead hours
            sub = sub.isel(lead=0)  # simplified for demo

        # If it has model dimension, mean it out (baseline blend)
        if "model" in sub.dims:
            sub = sub.mean(dim="model")

        vals = np.where(np.isnan(sub.values), None, sub.values).tolist()

        return GridResponse(
            variable=var,
            date=date,
            lead=lead,
            lats=sub.lat.values.tolist(),
            lons=sub.lon.values.tolist(),
            values=vals,
        )
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Error fetching forecast")


@app.get("/probability", response_model=ProbResponse)
def get_probability(
    event: str = Query(..., description="heavy_rain | heatwave | high_wind"),
    lead: int = Query(...),
    date: str = Query(...),
):
    # Stubbed probabilistic output representing exceedance maps
    ds = get_forecast_data()
    sub = ds["t2m"].isel(time=0)

    # Dummy probabilities
    probs = np.random.uniform(0, 1, size=(len(sub.lat), len(sub.lon))).tolist()

    return ProbResponse(event=event, date=date, lead=lead, probabilities=probs)


@app.get("/skill", response_model=SkillResponse)
def get_skill(model: str, region: str = "all", metric: str = "rmse"):
    # Stubbed skill report
    return SkillResponse(model=model, metric=metric, value=1.23)


@app.get("/drift")
def get_drift():
    return {"flags": []}
