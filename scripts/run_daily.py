import argparse
import logging
from datetime import UTC, datetime, timezone

from nwpblend.pipeline import run_daily


def main():
    parser = argparse.ArgumentParser(description="Megha-Drishti Daily Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="Issue date YYYY-MM-DD",
    )
    parser.add_argument("--demo", action="store_true", help="Run in offline synthetic demo mode")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading fresh data")
    parser.add_argument("--domain", type=str, default="india", help="Domain preset")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import yaml

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # We can accept preset names like "default" or "domain_igp"
    if args.domain == "india" or args.domain == "default":
        domain = config["domain"]
    else:
        domain = config.get(args.domain, config["domain"])

    # Ensure resolution is set
    if "resolution" not in domain:
        domain["resolution"] = 0.25

    run_daily(date=args.date, domain=domain, demo=args.demo, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
