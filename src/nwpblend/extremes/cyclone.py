import numpy as np
import pandas as pd


def cyclone_consensus(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple track consensus.
    tracks_df: DataFrame with columns ['model', 'time', 'lat', 'lon']
    Returns DataFrame with ['time', 'mean_lat', 'mean_lon', 'spread_km']
    """
    if tracks_df.empty:
        return pd.DataFrame()

    # Group by time
    grouped = tracks_df.groupby("time")

    results = []
    for time, group in grouped:
        mean_lat = group["lat"].mean()
        mean_lon = group["lon"].mean()

        # Approximate spread in km (1 deg ~ 111 km)
        # using Euclidean distance on lat/lon for simplicity
        lat_diff = (group["lat"] - mean_lat) * 111.0
        lon_diff = (group["lon"] - mean_lon) * 111.0 * np.cos(np.deg2rad(mean_lat))

        distances = np.sqrt(lat_diff**2 + lon_diff**2)
        spread_km = distances.std() if len(distances) > 1 else 0.0

        results.append(
            {"time": time, "mean_lat": mean_lat, "mean_lon": mean_lon, "spread_km": spread_km}
        )

    return pd.DataFrame(results)
