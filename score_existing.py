"""
Score already-scraped CSV files without re-running the browser.

Usage:
    python score_existing.py                   # score all output/rfps_????-??-??.csv
    python score_existing.py --file rfps_all.csv   # score a specific file in output/
"""

import argparse
import glob
import os
import pandas as pd
from matcher.score import LGeoMatcher

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="Score a specific file in output/ (e.g. rfps_all.csv)")
args = parser.parse_args()

if args.file:
    path = os.path.join("output", args.file)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        exit(1)
    files = [path]
else:
    files = sorted(glob.glob("output/rfps_????-??-??.csv"))
    if not files:
        print("No CSV files found in output/. Run run.py first.")
        exit()

matcher = LGeoMatcher()
total_all = 0

for path in files:
    df = pd.read_csv(path, encoding='latin-1')
    rows = df.to_dict("records")
    _, scored = matcher.score_all(rows, threshold=0)

    pd.DataFrame(scored).to_csv(path, index=False)

    label = os.path.basename(path).replace("rfps_", "").replace(".csv", "")
    print(f"  {label}: {len(rows)} listings scored")
    total_all += len(rows)

print(f"\nDone. {total_all} listings scored.")
