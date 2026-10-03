import sys
from datetime import UTC, datetime


def modify_pipeline():
    with open("src/nwpblend/pipeline.py", "r") as f:
        content = f.read()

    # Add --sources argument
    if "--sources" not in content:
        content = content.replace(
            'parser.add_argument("--quick", action="store_true", help="Run a quick preview (leads 24-72)")',
            'parser.add_argument("--quick", action="store_true", help="Run a quick preview (leads 24-72)")\n    parser.add_argument("--sources", type=str, default="", help="Comma-separated list of models to ingest (e.g. ecmwf_ifs,gfs)")',
        )

    if "sources=args.sources" not in content:
        content = content.replace(
            "quick=args.quick,\n        max_leads=args.max_leads",
            "quick=args.quick,\n        max_leads=args.max_leads,\n        sources=args.sources",
        )

    if "def run_daily(" in content and 'sources: str = ""' not in content:
        content = content.replace(
            "quick: bool = False,\n    max_leads: int | None = None,\n):",
            'quick: bool = False,\n    max_leads: int | None = None,\n    sources: str = "",\n):',
        )

    # Status tracking for ingest
    old_ingest = """                if "ecmwf_ifs" in expected_models:
                    ds, _run_dt = ecmwf_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["ecmwf_ifs"] = ds
                        if len(ds.lead) < len(leads):
                            report["warnings"].append(
                                f"ECMWF partial run ({len(ds.lead)}/{len(leads)} leads)."
                            )
                    else:
                        report["warnings"].append("ECMWF failed. Dropping from blend.")

                # Fetch GFS
                if "gfs" in expected_models:
                    ds, _run_dt = gfs_ingest.probe_and_fetch(date, domain, leads, variables)
                    if ds is not None:
                        models["gfs"] = ds
                        if len(ds.lead) < len(leads):
                            report["warnings"].append(
                                f"GFS partial run ({len(ds.lead)}/{len(leads)} leads)."
                            )
                    else:
                        report["warnings"].append("GFS failed. Dropping from blend.")

                log_stage("ingest", t0, "SUCCESS")"""

    new_ingest = """                if sources:
                    expected_models = [m.strip() for m in sources.split(",") if m.strip()]

                ingest_details = []
                stage_status = "SUCCESS"

                for src in expected_models:
                    try:
                        if src == "ecmwf_ifs":
                            ds, _run_dt = ecmwf_ingest.probe_and_fetch(date, domain, leads, variables)
                        elif src == "gfs":
                            ds, _run_dt = gfs_ingest.probe_and_fetch(date, domain, leads, variables)
                        else:
                            ds = None
                        
                        if ds is not None:
                            models[src] = ds
                            missing = [L for L in leads if L not in ds.lead.values]
                            if missing:
                                report["warnings"].append(f"{src} partial run. Missing leads: {missing}")
                                ingest_details.append(f"{src}: PARTIAL (Missing: {missing})")
                                if stage_status != "FAILED":
                                    stage_status = "PARTIAL"
                            else:
                                ingest_details.append(f"{src}: COMPLETE")
                        else:
                            report["warnings"].append(f"{src} failed. Dropping from blend.")
                            ingest_details.append(f"{src}: FAILED")
                            stage_status = "PARTIAL" # if some fail but we have others, the stage is partial. If all fail, it will crash in harmonise.
                    except Exception as e:
                        logger.error(f"{src} ingestion crashed: {e}")
                        report["warnings"].append(f"{src} failed with exception.")
                        ingest_details.append(f"{src}: FAILED")
                        stage_status = "PARTIAL"
                        
                if not models:
                    stage_status = "FAILED"

                log_stage("ingest", t0, stage_status, " | ".join(ingest_details))"""

    if "ingest_details = []" not in content:
        content = content.replace(old_ingest, new_ingest)

    # Save run report
    old_report = """    os.makedirs("data/logs", exist_ok=True)
    report_path = f"data/logs/run_report_{datetime.now(UTC).strftime('%Y-%m-%d')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Always write to latest for dashboard
    with open("data/logs/run_report_latest.json", "w") as f:
        json.dump(report, f, indent=2)"""

    new_report = """    os.makedirs("data/logs", exist_ok=True)
    report_path = f"data/logs/run_report_{datetime.now(UTC).strftime('%Y-%m-%d')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Only overwrite latest if it didn't completely fail
    if report["status"] != "FAILED":
        with open("data/logs/run_report_latest.json", "w") as f:
            json.dump(report, f, indent=2)"""

    content = content.replace(old_report, new_report)

    with open("src/nwpblend/pipeline.py", "w") as f:
        f.write(content)


if __name__ == "__main__":
    modify_pipeline()
