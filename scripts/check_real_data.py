import datetime
import os
import sys
import time
import warnings
import argparse
import json
import subprocess

warnings.filterwarnings("ignore")

def check_ecmwf(mirror):
    start = time.time()
    try:
        from ecmwf.opendata import Client
        client = Client(source=mirror, model="ifs", maximum_retries=1)
        request_args = {"time": 0, "step": 24, "type": "fc", "param": ["2t"]}
        result = client._get_urls(request_args, target="dummy", use_index=False)
        exact_url = result.urls[0] if result.urls else "Unknown"

        target = f"ecmwf_{mirror}_test.grib2"
        if os.path.exists(target):
            os.remove(target)

        client.retrieve(target=target, **request_args)
        
        import xarray as xr
        ds = xr.open_dataset(target, engine="cfgrib")
        val = ds["t2m"].mean().values - 273.15
        elapsed = time.time() - start
        
        if os.path.exists(target):
            os.remove(target)
            
        if not (-50 < val < 60):
            return {"status": False, "http": "200", "time": elapsed, "url": exact_url, "err": f"t2m {val:.1f}C out of bounds"}
        return {"status": True, "http": "200", "time": elapsed, "url": exact_url, "err": "Passed"}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": False, "http": "Err", "time": elapsed, "url": exact_url if 'exact_url' in locals() else "N/A", "err": str(e)[:100].replace("\n", " ")}


def check_gfs():
    start = time.time()
    try:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        today = datetime.datetime.now(datetime.UTC)
        # Probe backwards up to 10 days to find a valid run
        valid_path = None
        for i in range(40):
            cand = today - datetime.timedelta(hours=i*6)
            dstr = cand.strftime("%Y%m%d")
            hstr = f"{cand.hour:02d}"
            path = f"noaa-gfs-bdp-pds/gfs.{dstr}/{hstr}/atmos/gfs.t{hstr}z.pgrb2.0p25.f024"
            if fs.exists(path + ".idx"):
                valid_path = path
                break
        
        if not valid_path:
            elapsed = time.time() - start
            return {"status": False, "http": "404", "time": elapsed, "url": "N/A", "err": "No index found in last 10 days"}

        exact_url = f"s3://{valid_path}"
        idx_path = valid_path + ".idx"
        idx_data = fs.cat(idx_path).decode("utf-8").splitlines()
        
        byte_start, byte_end = None, None
        for i, line in enumerate(idx_data):
            if ":TMP:2 m above ground:" in line:
                byte_start = int(line.split(":")[1])
                if i + 1 < len(idx_data):
                    byte_end = int(idx_data[i + 1].split(":")[1]) - 1
                break

        if byte_start is not None:
            length = byte_end - byte_start + 1 if byte_end else 1000000
            with fs.open(valid_path, "rb", fill_cache=False) as f_in:
                f_in.seek(byte_start)
                chunk = f_in.read(length)
                
            target = "gfs_test.grib2"
            with open(target, "wb") as f_out:
                f_out.write(chunk)

            import xarray as xr
            ds = xr.open_dataset(target, engine="cfgrib")
            val = ds["t2m"].mean().values - 273.15
            elapsed = time.time() - start
            
            if os.path.exists(target):
                os.remove(target)
                
            if not (-50 < val < 60):
                return {"status": False, "http": "200", "time": elapsed, "url": exact_url, "err": f"Downloaded {length} bytes. t2m {val:.1f}C out of bounds"}
            return {"status": True, "http": "200", "time": elapsed, "url": exact_url, "err": f"Downloaded {length} bytes. Passed"}

        elapsed = time.time() - start
        return {"status": False, "http": "Err", "time": elapsed, "url": exact_url, "err": "Variable missing in index"}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": False, "http": "Err", "time": elapsed, "url": "N/A", "err": str(e)[:100].replace("\n", " ")}


def check_arco():
    start = time.time()
    exact_url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    try:
        import xarray as xr
        import pandas as pd
        store = xr.open_zarr(exact_url, chunks=None, consolidated=True, storage_options={"token": "anon"})
        
        # Check last available timestamp
        last_time = pd.to_datetime(store.time.values[-1])
        now = pd.Timestamp.utcnow()
        lag = now - last_time
        warning = ""
        if lag.days > 5:
            warning = f"(Lag {lag.days} days > ERA5T 5 days)"

        # Fetch tiny slice
        val = store["2m_temperature"].isel(time=-1, latitude=slice(0,1), longitude=slice(0,1)).values - 273.15
        elapsed = time.time() - start
        
        if not (-50 < float(val) < 60):
            return {"status": False, "http": "200", "time": elapsed, "url": exact_url, "err": f"t2m {float(val):.1f}C out of bounds {warning}"}
        return {"status": True, "http": "200", "time": elapsed, "url": exact_url, "err": f"Passed. Last time: {last_time.strftime('%Y-%m-%d')} {warning}"}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": False, "http": "Err", "time": elapsed, "url": exact_url, "err": str(e)[:100].replace("\n", " ")}


def check_imd():
    start = time.time()
    exact_url = "https://imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data/Data/Rainfall_0.25/Rainfall_0.25_2025.nc"
    try:
        import imdlib as imd
        
        data = imd.get_data("tmax", 2024, 2024, fn_format="yearwise", file_dir="scratch/")
        ds = data.get_xarray()
        val = float(ds["tmax"].mean().values)
        elapsed = time.time() - start
        
        # Document caching behavior in error string
        if val > 60 or val < -50:
            return {"status": False, "http": "200", "time": elapsed, "url": exact_url, "err": f"tmax {val:.1f}C out of bounds (imdlib downloads whole years and caches)"}
        return {"status": True, "http": "200", "time": elapsed, "url": exact_url, "err": "Passed (imdlib downloads whole years and caches)"}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": False, "http": "Err", "time": elapsed, "url": exact_url, "err": str(e)[:100].replace("\n", " ")}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str)
    args = parser.parse_args()

    if args.worker:
        if args.worker == "gfs":
            res = check_gfs()
        elif args.worker == "arco":
            res = check_arco()
        elif args.worker == "imd":
            res = check_imd()
        elif args.worker.startswith("ecmwf_"):
            mirror = args.worker.split("_")[1]
            res = check_ecmwf(mirror)
        else:
            res = {}
        print(f"__RESULT__{json.dumps(res)}")
        sys.exit(0)

    # Master process
    print("Testing Real Data Sources with Fail-Fast")
    print("-" * 140)
    print(f"{'Source (Mirror)':<20} | {'Status':<6} | {'HTTP':<5} | {'Time':<6} | {'URL':<30} | {'Error'}")
    print("-" * 140)

    results = []
    
    def run_worker(name, worker_id, timeout=90):
        print(f"Testing {name}...", end=" ", flush=True)
        try:
            proc = subprocess.run([sys.executable, __file__, "--worker", worker_id], capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                output = proc.stdout
                res_str = ""
                for line in output.splitlines():
                    if line.startswith("__RESULT__"):
                        res_str = line.replace("__RESULT__", "")
                        break
                if res_str:
                    res = json.loads(res_str)
                    stat_str = "PASS" if res.get("status") else "FAIL"
                    http = res.get("http", "Err")
                    elapsed = f"{res.get('time', 0):.1f}s"
                    url = res.get("url", "N/A")
                    err = res.get("err", "")
                else:
                    stat_str, http, elapsed, url, err = "FAIL", "Err", "N/A", "N/A", f"Missing JSON. Stdout: {output[:50]}"
            else:
                stat_str, http, elapsed, url, err = "FAIL", "Err", f"{timeout}s", "N/A", f"Process failed: {proc.stderr[:50]}"
        except subprocess.TimeoutExpired:
            stat_str, http, elapsed, url, err = "FAIL", "Timeout", f"{timeout}.0s", "N/A", f"Exceeded {timeout}s limit. Process killed."
        except Exception as e:
            stat_str, http, elapsed, url, err = "FAIL", "Err", "N/A", "N/A", str(e)[:50]

        print(f"{stat_str} [{http}] ({elapsed}) - {err}")
        return (name, stat_str, http, elapsed, url, err)

    for src in ["ecmwf", "aws", "google", "azure"]:
        results.append(run_worker(f"ECMWF ({src})", f"ecmwf_{src}", timeout=30))
    
    results.append(run_worker("NOAA GFS (aws)", "gfs", timeout=45))
    results.append(run_worker("ARCO-ERA5 (gcp)", "arco", timeout=90))
    results.append(run_worker("IMD (imdlib)", "imd", timeout=60))

    print("\n" + "=" * 140)
    print(f"{'Source (Mirror)':<20} | {'Status':<6} | {'HTTP':<5} | {'Time':<6} | {'URL':<30} | {'Error'}")
    print("-" * 140)
    for row in results:
        url_trunc = row[4][:27] + "..." if len(row[4]) > 30 else row[4]
        print(f"{row[0]:<20} | {row[1]:<6} | {row[2]:<5} | {row[3]:<6} | {url_trunc:<30} | {row[5]}")
