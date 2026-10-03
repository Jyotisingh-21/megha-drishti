import numpy as np
import xarray as xr

# Mock a 0.1 degree dataset
lats = np.arange(38, 7.9, -0.1)
lons = np.arange(68, 98.1, 0.1)
data = np.ones((len(lats), len(lons)))
ds = xr.Dataset({"precip": (["lat", "lon"], data)}, coords={"lat": lats, "lon": lons})

# Target grid: 0.25 degree
# Domain: 38 N to 8 N, 68 E to 98 E
target_lats = np.arange(38, 7.9, -0.25)
target_lons = np.arange(68, 98.1, 0.25)

lat_bins = np.append(target_lats + 0.125, target_lats[-1] - 0.125)
# Ensure lat_bins is strictly monotonically increasing or decreasing. Since target_lats is descending, lat_bins is descending.
# groupby_bins requires strictly increasing bins!
lat_bins = np.sort(lat_bins)
lon_bins = np.append(target_lons - 0.125, target_lons[-1] + 0.125)

print("lat_bins:", lat_bins[:5])
print("lon_bins:", lon_bins[:5])

grouped_lon = ds.groupby_bins("lon", lon_bins, labels=target_lons).mean()
grouped = grouped_lon.groupby_bins("lat", lat_bins, labels=target_lats[::-1]).mean()

# Reverse lat back to descending
grouped = grouped.sortby("lat_bins", ascending=False)
grouped = grouped.rename({"lat_bins": "lat", "lon_bins": "lon"})
print(grouped)
print("Original mean:", ds["precip"].mean().values)
print("Regridded mean:", grouped["precip"].mean().values)
