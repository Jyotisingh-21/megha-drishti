import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def retry_with_backoff(retries: int = 3, backoff_in_seconds: float = 1):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Failed after {retries} retries: {e}")
                        return None
                    sleep = backoff_in_seconds * 2**x
                    logger.warning(f"Error: {e}. Retrying in {sleep}s...")
                    time.sleep(sleep)
                    x += 1

        return wrapper

    return decorator


def get_cache_dir(source_name: str) -> str:
    path = os.path.join("data", "raw", source_name)
    os.makedirs(path, exist_ok=True)
    return path


def sources_available(source: str) -> bool:
    """Check if required dependencies for a source are available."""
    try:
        if source == "ecmwf":
            import ecmwf.opendata
        elif source == "gfs":
            import cfgrib
            import s3fs
        elif source == "aifs":
            import ecmwf.opendata
        elif source == "truth":
            import gcsfs
            import imdlib
        return True
    except ImportError:
        return False
