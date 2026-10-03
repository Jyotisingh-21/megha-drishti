import re

with open("src/nwpblend/ingest/ecmwf.py", "r") as f:
    ecmwf = f.read()

# Instead of `missing_leads = True` breaking the mirror loop and dropping the run:
# If download_success is false but some leads were downloaded, we can STILL process it!
ecmwf = ecmwf.replace(
    """
            if not missing_leads:
                download_success = True
                logger.info(f"Successfully fetched run {cand} from mirror {mirror}")
                with open(good_mirror_file, "w") as f:
                    json.dump({"mirror": mirror}, f)
                break
""",
    """
            if not missing_leads:
                download_success = True
                logger.info(f"Successfully fetched run {cand} from mirror {mirror}")
                with open(good_mirror_file, "w") as f:
                    json.dump({"mirror": mirror}, f)
                break
            else:
                # Even if missing leads, we consider it a partial success if at least one file exists
                has_files = any(os.path.exists(os.path.join(archive_dir, f"raw_f{l:03d}.grib")) for l in leads)
                if has_files:
                    download_success = True
                    logger.info(f"Partially fetched run {cand} from mirror {mirror}")
                    break
""",
)

with open("src/nwpblend/ingest/ecmwf.py", "w") as f:
    f.write(ecmwf)


with open("src/nwpblend/ingest/gfs.py", "r") as f:
    gfs = f.read()

# Do the same for GFS
gfs = gfs.replace(
    """
        if not missing_leads:
            logger.info(f"Successfully fetched GFS run {cand}")
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
""",
    """
        has_files = any(os.path.exists(os.path.join(archive_dir, f"raw_f{l:03d}.grib")) for l in leads)
        if not missing_leads or has_files:
            if missing_leads:
                logger.info(f"Partially fetched GFS run {cand}")
            else:
                logger.info(f"Successfully fetched GFS run {cand}")
                
            # Concat
            with open(target_file, "wb") as outfile:
                for l in leads:
                    lf = os.path.join(archive_dir, f"raw_f{l:03d}.grib")
                    if os.path.exists(lf):
                        with open(lf, "rb") as infile:
                            import shutil
                            shutil.copyfileobj(infile, outfile)
            
            success_ds = _process_file(target_file, domain, cand)
            if success_ds is not None:
                chosen_run = cand
                break
""",
)

with open("src/nwpblend/ingest/gfs.py", "w") as f:
    f.write(gfs)

print("added partial returns")
