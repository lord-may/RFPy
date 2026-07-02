"""
Combine all per-day CSVs in output/ into a single output/rfps_all.csv.
Preserves any "my assessment" labels already in rfps_all.csv — re-running
this script will never overwrite labels you've added.

Usage:
    python combine_output.py
"""

import glob
import os
import pandas as pd

LABEL_COL = "my assessment"
out_path = "output/rfps_all.csv"

files = sorted(glob.glob("output/rfps_????-??-??.csv"))
if not files:
    print("No CSV files found in output/. Run run.py first.")
    exit()

# Load existing labels from rfps_all.csv before overwriting it
existing_labels: dict = {}
if os.path.exists(out_path):
    try:
        existing = pd.read_csv(out_path, encoding="latin-1")
        if LABEL_COL in existing.columns:
            labeled_rows = existing[existing[LABEL_COL].notna() & (existing[LABEL_COL] != "")]
            existing_labels = dict(zip(labeled_rows["opportunity_id"].astype(str),
                                       labeled_rows[LABEL_COL]))
            if existing_labels:
                print(f"  Preserving {len(existing_labels)} existing label(s) from {out_path}")
    except Exception as e:
        print(f"  Warning: could not read existing labels: {e}")

# Combine day files
frames = [pd.read_csv(f, encoding="latin-1") for f in files]
combined = pd.concat(frames, ignore_index=True)

# Merge labels back in by opportunity_id
if existing_labels:
    combined[LABEL_COL] = combined["opportunity_id"].astype(str).map(existing_labels)

combined.to_csv(out_path, index=False)
print(f"Combined {len(files)} files → {out_path} ({len(combined)} rows)")
