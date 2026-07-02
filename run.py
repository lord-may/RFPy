"""
Entry point for the RFPy scraper + relevance scoring pipeline.

Usage:
    python run.py                                  # yesterday's listings
    python run.py --date 2026-06-30                # specific date
    python run.py --start-date 2026-06-20          # June 20 through yesterday
    python run.py --start-date 2026-06-20 --end-date 2026-06-30
    python run.py --no-match                       # skip relevance scoring
"""

import argparse
import os
import sys
from datetime import date

import pandas as pd

from scrapers.bcbids import BCBidsScraper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--no-match", action="store_true",
                        help="Skip relevance scoring, just write raw CSVs.")
    parser.add_argument("--fetch-details", action="store_true",
                        help="Visit each RFP detail page to extract Summary Details text (slower but better scoring).")
    args = parser.parse_args()

    yesterday = date.fromordinal(date.today().toordinal() - 1)

    try:
        if args.date:
            start = end = date.fromisoformat(args.date)
        elif args.start_date:
            start = date.fromisoformat(args.start_date)
            end = date.fromisoformat(args.end_date) if args.end_date else yesterday
        else:
            start = end = yesterday
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Scrape
    scraper = BCBidsScraper()
    results_by_date = scraper.scrape_range(start, end, fetch_details=args.fetch_details)

    if not results_by_date:
        print("No listings found.")
        return

    # Optionally score relevance
    matcher = None
    if not args.no_match:
        from matcher.score import LGeoMatcher
        matcher = LGeoMatcher()

    os.makedirs("output", exist_ok=True)
    total = 0

    for d in sorted(results_by_date):
        rows = results_by_date[d]

        if matcher:
            _, scored = matcher.score_all(rows, threshold=0)
            df = pd.DataFrame(scored)
        else:
            df = pd.DataFrame(rows)

        df["scraped_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        out_path = f"output/rfps_{d.isoformat()}.csv"
        df.to_csv(out_path, index=False)

        print(f"  {d}: {len(rows)} listings → {out_path}")
        total += len(rows)

    print(f"\nTotal: {total} listings.")


if __name__ == "__main__":
    main()
