import re

with open("src/nwpblend/pipeline.py", "r") as f:
    content = f.read()

# Inside ingest step:
search = """
                # Fetch ECMWF
                if "ecmwf_ifs" in expected_models:
                    ds, _run_dt = ecmwf_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["ecmwf_ifs"] = ds
                    else:
                        report["warnings"].append("ECMWF failed. Dropping from blend.")
"""

replace = """
                # Fetch ECMWF
                if "ecmwf_ifs" in expected_models:
                    ds, _run_dt = ecmwf_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["ecmwf_ifs"] = ds
                        if len(ds.lead) < len(leads):
                            report["warnings"].append(f"ECMWF partial run ({len(ds.lead)}/{len(leads)} leads).")
                    else:
                        report["warnings"].append("ECMWF failed. Dropping from blend.")
"""
content = content.replace(search, replace)

search_gfs = """
                # Fetch GFS
                if "gfs" in expected_models:
                    ds, _run_dt = gfs_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["gfs"] = ds
                    else:
                        report["warnings"].append("GFS failed. Dropping from blend.")
"""

replace_gfs = """
                # Fetch GFS
                if "gfs" in expected_models:
                    ds, _run_dt = gfs_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["gfs"] = ds
                        if len(ds.lead) < len(leads):
                            report["warnings"].append(f"GFS partial run ({len(ds.lead)}/{len(leads)} leads).")
                    else:
                        report["warnings"].append("GFS failed. Dropping from blend.")
"""
content = content.replace(search_gfs, replace_gfs)

with open("src/nwpblend/pipeline.py", "w") as f:
    f.write(content)

print("pipeline partial checking added")
