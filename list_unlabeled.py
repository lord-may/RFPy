"""
Show only unlabeled rows from a site's combined output, sorted by relevance
score, for a faster daily review pass -- so you're not re-scanning rows
you've already labeled.

Usage:
    python list_unlabeled.py --site alberta
    python list_unlabeled.py --site bcbids --limit 20
"""

import argparse
import sys

import pandas as pd

LABEL_COL_CANDIDATES = ["my assessment", "label"]


def _find_label_col(df: pd.DataFrame) -> str | None:
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=["bcbids", "alberta"], default="bcbids")
    parser.add_argument("--limit", type=int, default=None, help="Show only the top N unlabeled rows")
    args = parser.parse_args()

    path = f"output/{args.site}/rfps_all.csv"
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except FileNotFoundError:
        print(f"{path} not found. Run run.py --site {args.site} and combine_output.py --site {args.site} first.")
        sys.exit(1)

    label_col = _find_label_col(df)
    if label_col is None:
        unlabeled = df
        print(f"No label column found in {path} yet -- showing all {len(df)} rows.\n")
    else:
        unlabeled = df[df[label_col].isna() | (df[label_col] == "")]

    if "relevance_score" in unlabeled.columns:
        unlabeled = unlabeled.sort_values("relevance_score", ascending=False)

    if args.limit:
        unlabeled = unlabeled.head(args.limit)

    title_col = "title" if "title" in unlabeled.columns else None
    desc_col = "description" if "description" in unlabeled.columns else "opportunity_description"

    print(f"{len(unlabeled)} unlabeled row(s) in {path}:\n")
    for _, row in unlabeled.iterrows():
        score = row.get("relevance_score", "")
        title = (row.get(title_col, "") if title_col else "") or ""
        desc = str(row.get(desc_col, "") or "")[:150]
        print(f"[{row.get('opportunity_id', '')}] score={score}  {title}")
        if desc:
            print(f"    {desc}")


if __name__ == "__main__":
    main()
