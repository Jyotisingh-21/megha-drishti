import re


def rewrite_ecmwf():
    with open("src/nwpblend/ingest/ecmwf.py", "r") as f:
        content = f.read()

    new_probe = """
import json
import time

def probe_and_fetch(target_date, domain, leads, variables):
    try:
        from ecmwf.opendata import Client
    except ImportError:
        logger.error("ecmwf-opendata not installed.")
        return None, None

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    mirrors = config.get("ecmwf_mirrors", ["aws", "google", "ecmwf", "azure"])

    state_dir = "data/state"
    os.makedirs(state_dir, exist_ok=True)
    good_mirror_file = os.path.join(state_dir, "last_good_mirror.json")
    cooldown_file = os.path.join(state_dir, "mirror_cooldowns.json")

    last_good = None
    if os.path.exists(good_mirror_file):
        try:
            with open(good_mirror_file, "r") as f:
                last_good = json.load(f).get("mirror")
        except Exception:
            pass

    if last_good and last_good in mirrors:
        mirrors.remove(last_good)
        mirrors.insert(0, last_good)

    cooldowns = {}
    if os.path.exists(cooldown_file):
        try:
            with open(cooldown_file, "r") as f:
                cooldowns = json.load(f)
        except Exception:
            pass
    now_ts = time.time()
    cooldowns = {m: ts for m, ts in cooldowns.items() if now_ts - ts < 3600}

    if target_date == "latest":
        now = datetime.now(UTC)
        candidates = []
        for i in range(4):
            t = now - timedelta(hours=i * 12)
            candidates.append(t.replace(hour=(12 if t.hour >= 12 else 0), minute=0, second=0, microsecond=0))
    else:
        dt = pd.to_datetime(target_date)
        candidates = [dt.replace(hour=0, tzinfo=UTC)]

    var_map = {"t2m": "2t", "precip": "tp", "wind10m": ["10u", "10v"], "gust10m": "10fg"}
    req_vars = []
    for v in variables:
        if isinstance(var_map.get(v), list):
            req_vars.extend(var_map[v])
        elif var_map.get(v):
            req_vars.append(var_map[v])
    if not req_vars:
        return None, None

    success_ds = None
    chosen_run = None

    for cand in candidates:
        date_str = cand.strftime("%Y%m%d")
        time_int = cand.hour

        archive_dir = f"data/archive/ecmwf/{cand.strftime('%Y%m%d_%H%M')}"
        os.makedirs(archive_dir, exist_ok=True)
        target_file = os.path.join(archive_dir, "raw.grib")

        if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
            logger.info(f"Found archived ECMWF data for run {cand}")
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.warning("Archived file invalid. Re-downloading.")
                os.remove(target_file)

        logger.info(f"Probing ECMWF run {cand}...")
        download_success = False
        time_budget = config.get("source_time_budget", 1800)
        t_start_source = time.time()

        for mirror in mirrors:
            if mirror in cooldowns:
                logger.info(f"Skipping {mirror} (in cooldown)")
                continue

            missing_leads = False
            for lead in leads:
                if time.time() - t_start_source > time_budget:
                    logger.warning(f"Time budget of {time_budget}s exhausted for ECMWF.")
                    missing_leads = True
                    break

                lead_file = os.path.join(archive_dir, f"raw_f{lead:03d}.grib")
                if os.path.exists(lead_file) and os.path.getsize(lead_file) > 0:
                    continue

                try:
                    client = Client(source=mirror, model="ifs", maximum_retries=1, retry_after=5, use_server_retry_after=False)
                    def do_retrieve(c, d_str, t_int, t_file, l):
                        c.retrieve(date=d_str, time=t_int, step=[l], type="fc", param=req_vars, target=t_file)
                    
                    import concurrent.futures
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(do_retrieve, client, date_str, time_int, lead_file, lead)
                    future.result(timeout=config.get("download_timeout", 30))
                    time.sleep(2)
                except Exception as e:
                    if "503" in str(e) or "Slow Down" in str(e) or isinstance(e, concurrent.futures.TimeoutError):
                        logger.warning(f"Mirror {mirror} throttled/timed out on lead {lead}. Adding to cooldown.")
                        cooldowns[mirror] = time.time()
                        with open(cooldown_file, "w") as f:
                            json.dump(cooldowns, f)
                    else:
                        logger.warning(f"Mirror {mirror} failed on lead {lead}: {e}")
                    missing_leads = True
                    if os.path.exists(lead_file):
                        os.remove(lead_file)
                    break
            
            has_files = any(os.path.exists(os.path.join(archive_dir, f"raw_f{l:03d}.grib")) for l in leads)
            if not missing_leads or has_files:
                download_success = True
                if not missing_leads:
                    logger.info(f"Successfully fetched run {cand} from mirror {mirror}")
                    with open(good_mirror_file, "w") as f:
                        json.dump({"mirror": mirror}, f)
                else:
                    logger.info(f"Partially fetched run {cand} from mirror {mirror}")
                break

        if download_success:
            with open(target_file, "wb") as outfile:
                for lead in leads:
                    lead_file = os.path.join(archive_dir, f"raw_f{lead:03d}.grib")
                    if os.path.exists(lead_file) and os.path.getsize(lead_file) > 0:
                        with open(lead_file, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)

            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.error(f"Failed to process downloaded file for {cand}")
                os.remove(target_file)
        else:
            logger.info(f"Run {cand} not available on any mirror.")

    if chosen_run:
        logger.info(f"Selected ECMWF run: {chosen_run}")
        return success_ds, chosen_run
    else:
        logger.error("No valid ECMWF runs found in probing window.")
        return None, None
"""
    old_probe_pattern = re.compile(r"def probe_and_fetch.*?return None, None\n", re.DOTALL)
    new_content = old_probe_pattern.sub(new_probe.strip() + "\n", content, count=1)
    with open("src/nwpblend/ingest/ecmwf.py", "w") as f:
        f.write(new_content)


def rewrite_gfs():
    with open("src/nwpblend/ingest/gfs.py", "r") as f:
        content = f.read()

    new_probe = """
import json
import time

def probe_and_fetch(target_date, domain, leads, variables):
    try:
        import s3fs
    except ImportError:
        logger.error("s3fs not installed.")
        return None, None

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    if target_date == "latest":
        now = datetime.now(UTC)
        candidates = []
        for i in range(8):
            cand = now - timedelta(hours=i * 6)
            candidates.append(cand.replace(hour=(cand.hour // 6) * 6, minute=0, second=0, microsecond=0))
    else:
        dt = pd.to_datetime(target_date)
        candidates = [dt.replace(hour=0, tzinfo=UTC)]

    var_map = {"t2m": "2t", "precip": "apcp", "wind10m": ["10u", "10v"], "gust10m": "10fg"}
    req_vars = []
    for v in variables:
        if isinstance(var_map.get(v), list):
            req_vars.extend(var_map[v])
        elif var_map.get(v):
            req_vars.append(var_map[v])
    if not req_vars:
        return None, None

    success_ds = None
    chosen_run = None

    for cand in candidates:
        date_str = cand.strftime("%Y%m%d")
        time_int = cand.hour
        archive_dir = f"data/archive/gfs/{cand.strftime('%Y%m%d_%H%M')}"
        os.makedirs(archive_dir, exist_ok=True)
        target_file = os.path.join(archive_dir, "raw.grib")

        if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
            logger.info(f"Found archived GFS data for run {cand}")
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.warning("Archived GFS file invalid. Re-downloading.")
                os.remove(target_file)

        logger.info(f"Probing GFS run {cand} via index...")
        fs = s3fs.S3FileSystem(anon=True, read_timeout=config.get("download_timeout", 30), connect_timeout=15)
        
        idx_prefix = f"noaa-gfs-bdp-pds/gfs.{date_str}/{time_int:02d}/atmos/gfs.t{time_int:02d}z.pgrb2.0p25"
        
        missing_leads = False
        time_budget = config.get("source_time_budget", 1800)
        t_start_source = time.time()
        
        for lead in leads:
            if time.time() - t_start_source > time_budget:
                logger.warning(f"Time budget of {time_budget}s exhausted for GFS.")
                missing_leads = True
                break
                
            lead_file = os.path.join(archive_dir, f"raw_f{lead:03d}.grib")
            if os.path.exists(lead_file) and os.path.getsize(lead_file) > 0:
                continue

            idx_path = f"{idx_prefix}.f{lead:03d}.idx"
            data_path = f"{idx_prefix}.f{lead:03d}"
            try:
                if not fs.exists(idx_path):
                    logger.debug(f"Index not found: {idx_path}")
                    missing_leads = True
                    break

                with fs.open(idx_path, "r") as f:
                    idx_lines = f.readlines()
                
                byte_ranges = []
                for i, line in enumerate(idx_lines):
                    parts = line.strip().split(":")
                    if len(parts) >= 5:
                        start_byte = int(parts[1])
                        var_name = parts[3].lower()
                        if var_name in req_vars:
                            end_byte = int(idx_lines[i + 1].split(":")[1]) - 1 if i + 1 < len(idx_lines) else None
                            byte_ranges.append((start_byte, end_byte))
                
                if not byte_ranges:
                    logger.warning(f"Requested variables not in GFS index for lead {lead}")
                    missing_leads = True
                    break
                    
                import concurrent.futures
                def do_fetch(fs, data_path, lead_file, byte_ranges):
                    with fs.open(data_path, "rb") as f_in, open(lead_file, "wb") as f_out:
                        for start_byte, end_byte in byte_ranges:
                            f_in.seek(start_byte)
                            length = (end_byte - start_byte + 1) if end_byte else -1
                            chunk = f_in.read(length)
                            f_out.write(chunk)

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(do_fetch, fs, data_path, lead_file, byte_ranges)
                future.result(timeout=config.get("download_timeout", 30))
                time.sleep(2)
            except Exception as e:
                logger.error(f"GFS download error for lead {lead}: {e}")
                missing_leads = True
                if os.path.exists(lead_file):
                    os.remove(lead_file)
                break
        
        has_files = any(os.path.exists(os.path.join(archive_dir, f"raw_f{l:03d}.grib")) for l in leads)
        if not missing_leads or has_files:
            if not missing_leads:
                logger.info(f"Successfully fetched GFS run {cand}")
            else:
                logger.info(f"Partially fetched GFS run {cand}")
                
            with open(target_file, "wb") as outfile:
                for lead in leads:
                    lead_file = os.path.join(archive_dir, f"raw_f{lead:03d}.grib")
                    if os.path.exists(lead_file) and os.path.getsize(lead_file) > 0:
                        with open(lead_file, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
            
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
            else:
                logger.error(f"Failed to process GFS file for {cand}")
                os.remove(target_file)
        else:
            logger.info(f"Run {cand} not completely available on GFS.")

    if chosen_run:
        logger.info(f"Selected GFS run: {chosen_run}")
        return success_ds, chosen_run
    else:
        logger.error("No valid GFS runs found.")
        return None, None
"""
    old_probe_pattern = re.compile(r"def probe_and_fetch.*?return None, None\n", re.DOTALL)
    new_content = old_probe_pattern.sub(new_probe.strip() + "\n", content, count=1)
    with open("src/nwpblend/ingest/gfs.py", "w") as f:
        f.write(new_content)


if __name__ == "__main__":
    rewrite_ecmwf()
    rewrite_gfs()
    print("Files updated safely.")
