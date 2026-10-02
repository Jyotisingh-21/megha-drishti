import argparse
import datetime
import os
import warnings

import s3fs
import xarray as xr
import yaml
from ecmwf.opendata import Client as ECMWFClient

warnings.filterwarnings("ignore")

def check_ecmwf(source="aws"):
    client = ECMWFClient(source=source)
    # Get today's 00Z or yesterday's 12Z
    target = f"scratch/ecmwf_{source}_test.grib2"
    os.makedirs("scratch", exist_ok=True)
    try:
        _ = client.retrieve(
            time=0,
            step=24,
            type="fc",
            param=["2t", "tp", "10u", "10v", "10fg"],
            target=target
        )
        ds = xr.open_dataset(target, engine="cfgrib")
        
        # Sanity check
        if "t2m" in ds:
            val = ds["t2m"].mean().values
            # K to C
            val_c = val - 273.15
            if not (-50 < val_c < 60):
                return False, f"t2m {val_c:.1f}C out of bounds"
        if "tp" in ds:
            val = ds["tp"].mean().values
            if val < 0:
                return False, "Precipitation is negative"
                
        return True, "Passed"
    except Exception as e:
        return False, str(e)[:100]

def check_gfs():
    fs = s3fs.S3FileSystem(anon=True)
    today = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
    path = f"noaa-gfs-bdp-pds/gfs.{today}/00/atmos/gfs.t00z.pgrb2.0p25.f024"
    if not fs.exists(path):
        return False, "File not found on AWS"
    
    target = "scratch/gfs_test.grib2"
    try:
        fs.get(path, target)
        # Note: loading the whole GFS file is huge! We should use filter_by_keys or rely on the existence check + partial read
        # For this script we will just check if file downloaded and has size > 0
        sz = os.path.getsize(target)
        if sz < 1000000:
            return False, f"File too small: {sz} bytes"
        
        # xarray cfgrib filtering to just 2t
        ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
        val = ds["t2m"].mean().values - 273.15
        if not (-50 < val < 60):
            return False, f"t2m {val:.1f}C out of bounds"
        return True, "Passed"
    except Exception as e:
        return False, str(e)[:100]

def main():
    print("Testing Real Data Sources...")
    print(f"{'Source':<15} | {'Status':<10} | {'Details'}")
    print("-" * 50)
    
    for src in ["aws", "google", "ecmwf"]:
        status, details = check_ecmwf(src)
        stat_str = "PASS" if status else "FAIL"
        print(f"ECMWF {src:<9} | {stat_str:<10} | {details}")
        
    gfs_stat, gfs_det = check_gfs()
    print(f"NOAA GFS aws    | {'PASS' if gfs_stat else 'FAIL':<10} | {gfs_det}")

if __name__ == "__main__":
    main()
