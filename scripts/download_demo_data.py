import argparse
import logging
import os
import time

from nwpblend.ingest.synthetic import generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Generate and download demo/real data.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data")
    parser.add_argument("--date", type=str, help="Target date to ingest (YYYY-MM-DD)")
    parser.add_argument(
        "--sources", type=str, help="Comma-separated list of sources to ingest (e.g., ecmwf,gfs)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the execution plan instead of downloading"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.demo:
        logger.info("Generating synthetic demo data...")
        start_time = time.time()

        # Using ~1 deg resolution over India for speed
        truth, models, ensembles, _regimes = generate_synthetic_data(
            lat_min=6.0, lat_max=37.0, lon_min=68.0, lon_max=98.0, resolution=1.0, days=90
        )

        out_dir = "data/demo"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "ensembles"), exist_ok=True)

        logger.info("Saving Truth...")
        truth.to_zarr(os.path.join(out_dir, "truth.zarr"), mode="w")

        logger.info("Saving Regimes...")
        _regimes.to_zarr(os.path.join(out_dir, "regimes.zarr"), mode="w")

        for name, ds in models.items():
            logger.info(f"Saving Model: {name}...")
            ds.to_zarr(os.path.join(out_dir, "models", f"{name}.zarr"), mode="w")

        for name, ds in ensembles.items():
            logger.info(f"Saving Ensemble: {name}...")
            ds.to_zarr(os.path.join(out_dir, "ensembles", f"{name}.zarr"), mode="w")

        elapsed = time.time() - start_time
        logger.info(f"Demo data generated and saved in {elapsed:.1f} seconds.")

    elif args.date and args.sources:
        sources_list = [s.strip() for s in args.sources.split(",")]
        if args.dry_run:
            print(f"Plan for date: {args.date}")
            for source in sources_list:
                cache_dir = os.path.join("data", "raw", source)
                print(f"- Would fetch {source} data for {args.date}")
                print(f"  Target save directory: {cache_dir}")
        else:
            logger.info(f"Ingesting real data for {args.date} from {sources_list}...")
            # We would dynamically load the module and call fetch() here.
            # E.g.:
            # from nwpblend.ingest import ecmwf
            # ecmwf.fetch(args.date, domain, leads, variables)
    else:
        logger.info("No valid arguments provided. Use --demo or --date with --sources.")


if __name__ == "__main__":
    main()
