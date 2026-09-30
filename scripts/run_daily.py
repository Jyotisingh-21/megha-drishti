import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description="Run the daily NWP blending pipeline.")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode using synthetic data")
    parser.add_argument("--date", type=str, help="Date to run the pipeline for (YYYY-MM-DD)")
    parser.add_argument("--domain", type=str, help="Domain to run the pipeline for")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Running pipeline with args: demo={args.demo}, date={args.date}, domain={args.domain}"
    )


if __name__ == "__main__":
    main()
