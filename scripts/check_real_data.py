import datetime
import os
import sys
import time

# Suppress warnings for clean output
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import imdlib
import requests
import s3fs
import xarray as xr

warnings.filterwarnings("ignore")


def run_with_timeout(func, args=(), kwargs=None, timeout=90):
    if kwargs is None:
        kwargs = {}
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        return False, "Timeout", f"{timeout:.1f}s", "N/A", f"Exceeded {timeout}s limit"


def check_ecmwf(source):
    start = time.time()
    try:
        from ecmwf.opendata import Client

        client = Client(
            source=source,
            model="ifs",
            maximum_retries=3,
            retry_after=5,
            use_server_retry_after=False,
        )

        # Get URLs to show
        request_args = {"time": 0, "step": 24, "type": "fc", "param": ["2t"]}
        result = client._get_urls(request_args, target="dummy", use_index=False)
        exact_url = result.urls[0] if result.urls else "Unknown"

        target = f"scratch/ecmwf_{source}_test.grib2"
        os.makedirs("scratch", exist_ok=True)
        if os.path.exists(target):
            os.remove(target)

        client.retrieve(target=target, **request_args)

        ds = xr.open_dataset(target, engine="cfgrib")
        val = ds["t2m"].mean().values
        val_c = val - 273.15

        elapsed = time.time() - start

        if not (-50 < val_c < 60):
            return False, "200", f"{elapsed:.1f}s", exact_url, f"t2m {val_c:.1f}C out of bounds"

        return True, "200", f"{elapsed:.1f}s", exact_url, "Passed"
    except Exception as e:
        elapsed = time.time() - start
        status = "Err"
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            status = str(e.response.status_code)
        elif "503" in str(e):
            status = "503"
        elif "Timeout" in str(e):
            status = "Timeout"
        return (
            False,
            status,
            f"{elapsed:.1f}s",
            exact_url if "exact_url" in locals() else "N/A",
            str(e)[:100].replace("\n", " "),
        )


def check_gfs():
    start = time.time()
    try:
        fs = s3fs.S3FileSystem(anon=True)
        today = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")
        path = f"noaa-gfs-bdp-pds/gfs.{today}/00/atmos/gfs.t00z.pgrb2.0p25.f024"
        exact_url = f"s3://{path}"

        if not fs.exists(path):
            elapsed = time.time() - start
            return False, "404", f"{elapsed:.1f}s", exact_url, "File not found on AWS"

        target = "scratch/gfs_test.grib2"
        idx_path = path + ".idx"
        if fs.exists(idx_path):
            idx_data = fs.cat(idx_path).decode("utf-8").splitlines()
            byte_start, byte_end = None, None
            for i, line in enumerate(idx_data):
                if ":TMP:2 m above ground:" in line:
                    byte_start = int(line.split(":")[1])
                    if i + 1 < len(idx_data):
                        byte_end = int(idx_data[i + 1].split(":")[1]) - 1
                    break

            if byte_start is not None:
                with fs.open(path, "rb", fill_cache=False) as f_in:
                    f_in.seek(byte_start)
                    chunk = f_in.read(byte_end - byte_start + 1 if byte_end else 1000000)
                with open(target, "wb") as f_out:
                    f_out.write(chunk)

                ds = xr.open_dataset(target, engine="cfgrib")
                val = ds["t2m"].mean().values - 273.15
                elapsed = time.time() - start
                if not (-50 < val < 60):
                    return (
                        False,
                        "200",
                        f"{elapsed:.1f}s",
                        exact_url,
                        f"t2m {val:.1f}C out of bounds",
                    )
                return True, "200", f"{elapsed:.1f}s", exact_url, "Passed"

        elapsed = time.time() - start
        return False, "Err", f"{elapsed:.1f}s", exact_url, "Could not extract variable via idx"
    except Exception as e:
        elapsed = time.time() - start
        return (
            False,
            "Err",
            f"{elapsed:.1f}s",
            exact_url if "exact_url" in locals() else "N/A",
            str(e)[:100].replace("\n", " "),
        )


def check_arco():
    start = time.time()
    exact_url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    try:
        import zarr

        store = xr.open_zarr(exact_url, chunks=None)
        val = store["2m_temperature"].isel(time=-1, latitude=500, longitude=500).values - 273.15
        elapsed = time.time() - start
        if not (-50 < float(val) < 60):
            return (
                False,
                "200",
                f"{elapsed:.1f}s",
                exact_url,
                f"t2m {float(val):.1f}C out of bounds",
            )
        return True, "200", f"{elapsed:.1f}s", exact_url, "Passed"
    except Exception as e:
        elapsed = time.time() - start
        return False, "Err", f"{elapsed:.1f}s", exact_url, str(e)[:100].replace("\n", " ")


def check_imd():
    start = time.time()
    exact_url = "https://imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data/tmax.nc"
    try:
        import imdlib as imd

        data = imd.get_data("tmax", 2023, 2023, fn_format="yearwise", file_dir="scratch/")
        ds = data.get_xarray()
        val = float(ds["tmax"].mean().values)
        elapsed = time.time() - start
        if val > 60 or val < -50:
            return False, "200", f"{elapsed:.1f}s", exact_url, f"tmax {val:.1f}C out of bounds"
        return True, "200", f"{elapsed:.1f}s", exact_url, "Passed"
    except Exception as e:
        elapsed = time.time() - start
        return False, "Err", f"{elapsed:.1f}s", exact_url, str(e)[:100].replace("\n", " ")


def run_check(name, func, timeout=90):
    print(f"Testing {name}...", end=" ", flush=True)
    status, http_stat, elapsed, url, err = run_with_timeout(func, timeout=timeout)

    stat_str = "PASS" if status else "FAIL"
    print(f"{stat_str} [{http_stat}] ({elapsed}) - {err}")
    return stat_str, http_stat, elapsed, url, err


def main():
    print("Testing Real Data Sources with Fail-Fast")
    print("-" * 120)
    print(
        f"{'Source (Mirror)':<20} | {'Status':<6} | {'HTTP':<5} | {'Time':<6} | {'URL':<30} | {'Error'}"
    )
    print("-" * 120)

    results = []

    for src in ["ecmwf", "aws", "google", "azure"]:
        stat_str, http_stat, elapsed, url, err = run_check(
            f"ECMWF ({src})", lambda s=src: check_ecmwf(s)
        )
        results.append((f"ECMWF ({src})", stat_str, http_stat, elapsed, url, err))

    stat_str, http_stat, elapsed, url, err = run_check("NOAA GFS (aws)", check_gfs)
    results.append(("NOAA GFS (aws)", stat_str, http_stat, elapsed, url, err))

    stat_str, http_stat, elapsed, url, err = run_check("ARCO-ERA5 (gcp)", check_arco)
    results.append(("ARCO-ERA5 (gcp)", stat_str, http_stat, elapsed, url, err))

    stat_str, http_stat, elapsed, url, err = run_check("IMD (imdlib)", check_imd)
    results.append(("IMD (imdlib)", stat_str, http_stat, elapsed, url, err))

    print("\n" + "=" * 120)
    print(
        f"{'Source (Mirror)':<20} | {'Status':<6} | {'HTTP':<5} | {'Time':<6} | {'URL':<30} | {'Error'}"
    )
    print("-" * 120)
    for row in results:
        url_trunc = row[4][:27] + "..." if len(row[4]) > 30 else row[4]
        print(
            f"{row[0]:<20} | {row[1]:<6} | {row[2]:<5} | {row[3]:<6} | {url_trunc:<30} | {row[5]}"
        )


if __name__ == "__main__":
    main()
