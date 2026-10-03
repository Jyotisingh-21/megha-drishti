import re

with open("src/nwpblend/ingest/ecmwf.py", "r") as f:
    ecmwf = f.read()

# Add time budget check inside the lead loop
ecmwf = ecmwf.replace(
    "for lead in leads:",
    """
            t_start_mirror = time.time()
            time_budget = config.get("source_time_budget", 600)
            
            for lead in leads:
                if time.time() - t_start_mirror > time_budget:
                    logger.warning(f"Time budget of {time_budget}s exhausted for ECMWF.")
                    missing_leads = True
                    break
""",
)

with open("src/nwpblend/ingest/ecmwf.py", "w") as f:
    f.write(ecmwf)

with open("src/nwpblend/ingest/gfs.py", "r") as f:
    gfs = f.read()

# Add time budget check inside the lead loop
gfs = gfs.replace(
    "for lead in leads:",
    """
        t_start_gfs = time.time()
        time_budget = config.get("source_time_budget", 600)
        for lead in leads:
            if time.time() - t_start_gfs > time_budget:
                logger.warning(f"Time budget of {time_budget}s exhausted for GFS.")
                missing_leads = True
                break
""",
)

with open("src/nwpblend/ingest/gfs.py", "w") as f:
    f.write(gfs)

print("added time budgets")
