import argparse
import logging
import os

import pandas as pd
import yaml

from nwpblend.pipeline import run_daily

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_history(
    start_date: str, days: int, domain_name: str = "small", skip_download: bool = False
):
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    domain = cfg["domain"]

    def process_date(dt):
        date_str = dt.strftime("%Y-%m-%d")
        logger.info(f"Running pipeline for {date_str}...")
        try:
            run_daily(date=date_str, domain=domain, demo=False, skip_download=skip_download)
        except Exception as e:
            logger.error(f"Failed pipeline for {date_str}: {e}")

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(process_date, dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2026-09-01")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    fetch_history(args.start, args.days, skip_download=args.skip_download)
