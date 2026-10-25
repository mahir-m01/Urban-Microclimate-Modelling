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
    import argparse

    parser = argparse.ArgumentParser(description="Download UrbanTALES raw data files")
    parser.add_argument("output_dir", nargs="?", default=os.path.join(REPO, "data", "raw"), help="Output directory")
    parser.add_argument("--kinds", nargs="+", choices=list(NAMES.keys()), default=list(NAMES.keys()), help="File types to download")
    parser.add_argument("--workers", type=int, default=12, help="Number of concurrent download workers")
    parser.add_argument("--cases", nargs="+", default=None, help="Specific cases to download")
    args = parser.parse_args()

    out = args.output_dir
    selected_kinds = set(args.kinds)
    selected_cases = set(args.cases) if args.cases else None

    jobs = []
    with open(CATALOG, newline="") as fh:
        for r in csv.DictReader(fh):
            if selected_cases and r["case"] not in selected_cases:
                continue
            os.makedirs(os.path.join(out, r["case"]), exist_ok=True)
            for kind in selected_kinds:
                name = NAMES[kind]
                if r.get(f"{kind}_url"):
                    jobs.append((r[f"{kind}_url"], os.path.join(out, r["case"], name.format(c=r["case"]))))

    total_jobs = len(jobs)
    print(f"Starting download of {total_jobs} files using {args.workers} workers into {out}...")

    completed = 0
    ok_count = 0
    skipped_count = 0
    failed = []

    with cf.ThreadPoolExecutor(args.workers) as pool:
        future_to_job = {pool.submit(fetch, job): job for job in jobs}
        for future in cf.as_completed(future_to_job):
            res = future.result()
            completed += 1
            if res == "ok":
                ok_count += 1
            elif res == "skipped":
                skipped_count += 1
            elif res.startswith("FAILED"):
                failed.append(res)

            if completed % 50 == 0 or completed == total_jobs:
                pct = (completed / total_jobs) * 100
                print(f"[{completed}/{total_jobs} - {pct:.1f}%] ok: {ok_count}, skipped: {skipped_count}, failed: {len(failed)}")

    print(f"\nFinal Summary: {total_jobs} files, {ok_count} downloaded, {skipped_count} skipped, {len(failed)} failed")
    for s in failed:
        print(s)


if __name__ == "__main__":
    main()
