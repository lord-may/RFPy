"""
Entry point for the RFPy scraper + relevance scoring pipeline.

Usage:
    python run.py                                  # yesterday's listings (bcbids)
    python run.py --site alberta --date 2026-07-02  # specific date, alberta portal
    python run.py --start-date 2026-06-20          # June 20 through yesterday
    python run.py --start-date 2026-06-20 --end-date 2026-06-30
    python run.py --no-match                       # skip relevance scoring
"""

import argparse
import os
import sys
from datetime import date

import pandas as pd

SCRAPERS = {
    "bcbids": "scrapers.bcbids.BCBidsScraper",
    "alberta": "scrapers.alberta.AlbertaScraper",
}


def _load_scraper(site: str):
    module_path, class_name = SCRAPERS[site].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=sorted(SCRAPERS), default="bcbids")
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
    scraper = _load_scraper(args.site)
    results_by_date = scraper.scrape_range(start, end, fetch_details=args.fetch_details)

    if not results_by_date:
        print("No listings found.")
        return

    # Optionally score relevance
    matcher = None
    if not args.no_match:
        from matcher.score import LGeoMatcher
        matcher = LGeoMatcher()

    out_dir = f"output/{args.site}"
    os.makedirs(out_dir, exist_ok=True)
    total = 0

    for d in sorted(results_by_date):
        rows = results_by_date[d]

        if matcher:
            _, scored = matcher.score_all(rows, threshold=0)
            df = pd.DataFrame(scored)
        else:
            df = pd.DataFrame(rows)

        df["scraped_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        out_path = f"{out_dir}/rfps_{d.isoformat()}.csv"
        df.to_csv(out_path, index=False)

        print(f"  {d}: {len(rows)} listings → {out_path}")
        total += len(rows)

    print(f"\nTotal: {total} listings.")


if __name__ == "__main__":
    main()
