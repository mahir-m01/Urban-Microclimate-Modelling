"""Download the UrbanTALES case files listed in data/catalog.csv.

Stage: S01 Data ingestion

Usage: python pipeline/S01_data_ingestion/download_urbantales.py [output_dir]

Fetches the building map, pedestrian-level wind map, simulation settings and vertical
profile of all 538 cases (about 4.8 GB) into output_dir (default: data/raw), one folder
per case. Files already present are skipped, so an interrupted run can be restarted.
"""

import concurrent.futures as cf
import csv
import os
import socket
import sys
import time
import urllib.request

socket.setdefaulttimeout(90)  # a stalled download would otherwise hang a worker forever

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATALOG = os.path.join(REPO, "data", "catalog.csv")
NAMES = {
    "topo": "{c}_topo",
    "ped": "{c}_ped.nc",
    "p3d": "{c}_p3d",
    "profile": "profile-{c}.csv",
    "ts": "{c}_ts.nc",
}


def fetch(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skipped"
    last = ""
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
            return "ok"
        except Exception as e:
            last = repr(e)
            time.sleep(2 * (attempt + 1))
    return f"FAILED {path} {last}"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data", "raw")
    jobs = []
    with open(CATALOG, newline="") as fh:
        for r in csv.DictReader(fh):
            os.makedirs(os.path.join(out, r["case"]), exist_ok=True)
            for kind, name in NAMES.items():
                if r.get(f"{kind}_url"):
                    jobs.append((r[f"{kind}_url"], os.path.join(out, r["case"], name.format(c=r["case"]))))
    with cf.ThreadPoolExecutor(8) as pool:
        status = list(pool.map(fetch, jobs))
    failed = [s for s in status if s.startswith("FAILED")]
    print(f"{len(jobs)} files, {status.count('ok')} downloaded, {status.count('skipped')} skipped, {len(failed)} failed")
    for s in failed:
        print(s)


if __name__ == "__main__":
    main()
