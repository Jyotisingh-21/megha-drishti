import datetime

import s3fs
from ecmwf.opendata import Client


def check_gfs():
    fs = s3fs.S3FileSystem(anon=True)
    dirs = fs.ls('noaa-gfs-bdp-pds')
    gfs_dirs = [d for d in dirs if 'gfs.' in d]
    gfs_dirs.sort()
    print(f"GFS AWS oldest: {gfs_dirs[0] if gfs_dirs else 'None'}")
    print(f"GFS AWS newest: {gfs_dirs[-1] if gfs_dirs else 'None'}")

def check_ecmwf(source):
    client = Client(source=source)
    for days in [4, 6, 10]:
        date = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime('%Y%m%d')
        try:
            req = client.retrieve(
                date=date,
                time=0,
                step=0,
                type="fc",
                param="2t",
                target=f"C:/dev/megha-drishti/scratch/test_{source}_{days}.grib2"
            )
            print(f"ECMWF {source} {days} days ago: SUCCESS")
        except Exception:
            print(f"ECMWF {source} {days} days ago: FAILED")

print("Checking GFS...")
try:
    check_gfs()
except Exception as e:
    print(f"GFS check failed: {e}")

print("\nChecking ECMWF...")
for src in ["ecmwf", "aws", "google"]:
    check_ecmwf(src)
