import re

with open("src/nwpblend/ingest/ecmwf.py", "r") as f:
    content = f.read()

new_probe = """
import json
import time

def probe_and_fetch(target_date, domain, leads, variables):
    \"\"\"
    Probes for the latest complete ECMWF run, archiving to data/archive.
    \"\"\"
    try:
        from ecmwf.opendata import Client
    except ImportError:
        logger.error("ecmwf-opendata not installed.")
        return None, None

    # Load mirrors from config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    mirrors = config.get("ecmwf_mirrors", ["aws", "google", "ecmwf", "azure"])

    # Load state
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

    # Clean old cooldowns (1 hour)
    now_ts = time.time()
    cooldowns = {m: ts for m, ts in cooldowns.items() if now_ts - ts < 3600}

    # Determine candidate datetimes
    if target_date == "latest":
        now = datetime.now(UTC)
        candidates = []
        for i in range(4):
            t = now - timedelta(hours=i * 12)
            candidates.append(
                t.replace(hour=(12 if t.hour >= 12 else 0), minute=0, second=0, microsecond=0)
            )
    else:
        dt = pd.to_datetime(target_date)
        candidates = [dt.replace(hour=0, tzinfo=UTC)]

    # Map variables
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

        for mirror in mirrors:
            if mirror in cooldowns:
                logger.info(f"Skipping {mirror} (in 1hr cooldown)")
                continue

            missing_leads = False
            for lead in leads:
                lead_file = os.path.join(archive_dir, f"raw_f{lead:03d}.grib")
                if os.path.exists(lead_file) and os.path.getsize(lead_file) > 0:
                    continue  # Already downloaded

                try:
                    client = Client(
                        source=mirror,
                        model="ifs",
                        maximum_retries=1,  # User requested: "Never retry a failing real download more than once."
                        retry_after=5,
                        use_server_retry_after=False,
                    )

                    def do_retrieve(c, d_str, t_int, t_file, l):
                        c.retrieve(
                            date=d_str,
                            time=t_int,
                            step=[l],
                            type="fc",
                            param=req_vars,
                            target=t_file,
                        )

                    import concurrent.futures
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(do_retrieve, client, date_str, time_int, lead_file, lead)
                    
                    download_timeout = config.get("download_timeout", 30)
                    future.result(timeout=download_timeout)
                    
                    time.sleep(2)  # Small delay between requests as requested
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
                    break  # Break out of leads loop, switch mirror
            
            if not missing_leads:
                download_success = True
                logger.info(f"Successfully fetched run {cand} from mirror {mirror}")
                with open(good_mirror_file, "w") as f:
                    json.dump({"mirror": mirror}, f)
                break

        if download_success:
            # Concat lead files into target_file
            with open(target_file, "wb") as outfile:
                for lead in leads:
                    lead_file = os.path.join(archive_dir, f"raw_f{lead:03d}.grib")
                    if os.path.exists(lead_file):
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


old_probe_pattern = re.compile(r"def probe_and_fetch.*?return None, None", re.DOTALL)
new_content = old_probe_pattern.sub(new_probe.strip(), content, count=1)

with open("src/nwpblend/ingest/ecmwf.py", "w") as f:
    f.write(new_content)

print("ecmwf.py updated successfully.")
