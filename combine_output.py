"""
Combine all per-day CSVs in output/<site>/ into a single output/<site>/rfps_all.csv.
Preserves any hand labels already in rfps_all.csv — re-running this script
will never overwrite labels you've added. Different sites use different
label column names (bcbids: "my assessment", alberta: "label") -- both are
auto-detected, never renamed.

Usage:
    python combine_output.py                  # site=bcbids
    python combine_output.py --site alberta
"""

import argparse
import glob
import os
import pandas as pd

LABEL_COL_CANDIDATES = ["my assessment", "label"]


def _find_label_col(df: pd.DataFrame) -> str | None:
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--site", choices=["bcbids", "alberta"], default="bcbids")
args = parser.parse_args()

out_dir = f"output/{args.site}"
out_path = f"{out_dir}/rfps_all.csv"

files = sorted(glob.glob(f"{out_dir}/rfps_????-??-??.csv"))
if not files:
    print(f"No CSV files found in {out_dir}/. Run run.py --site {args.site} first.")
    exit()

# Load existing labels from rfps_all.csv before overwriting it
existing_labels: dict = {}
label_col: str | None = None
if os.path.exists(out_path):
    try:
        existing = pd.read_csv(out_path, encoding="latin-1")
        label_col = _find_label_col(existing)
        if label_col:
            labeled_rows = existing[existing[label_col].notna() & (existing[label_col] != "")]
            existing_labels = dict(zip(labeled_rows["opportunity_id"].astype(str),
                                       labeled_rows[label_col]))
            if existing_labels:
                print(f"  Preserving {len(existing_labels)} existing label(s) from {out_path} (column '{label_col}')")
    except Exception as e:
        print(f"  Warning: could not read existing labels: {e}")

# Combine day files
frames = [pd.read_csv(f, encoding="latin-1") for f in files]
combined = pd.concat(frames, ignore_index=True)

# Merge labels back in by opportunity_id, under whichever column name they were already in
if existing_labels and label_col:
    combined[label_col] = combined["opportunity_id"].astype(str).map(existing_labels)

combined.to_csv(out_path, index=False)
print(f"Combined {len(files)} files → {out_path} ({len(combined)} rows)")
