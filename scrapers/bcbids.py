"""
Scraper for BC Bid public RFP listings.
https://www.bcbid.gov.bc.ca/page.aspx/en/rfp/request_browse_public

Uses the real Edge browser profile to pass reCAPTCHA v3.
Edge must be fully closed before running.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

BASE_URL = "https://www.bcbid.gov.bc.ca"
BROWSE_URL = f"{BASE_URL}/page.aspx/en/rfp/request_browse_public"
EDGE_PROFILE = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")

ROW_SELECTOR = "tr[data-object-type='rfp']"
NEXT_BTN_SELECTOR = "button#body_x_grid_gridPagerBtnNextPage"
SUMMARY_HEADING = "Summary Details"

COL_OPP_ID = 1
COL_DESCRIPTION = 2
COL_COMMODITIES = 3
COL_TYPE = 4
COL_ISSUE_DATE = 5
COL_CLOSING_DATE = 6
COL_ORG_BY = 10
COL_ORG_FOR = 11


_CHAR_REPLACEMENTS = {
    '‘': "'", '’': "'",   # smart single quotes
    '“': '"', '”': '"',   # smart double quotes
    '–': '-', '—': '-',   # en dash, em dash
    '…': '...',                # ellipsis
    ' ': ' ',                  # non-breaking space
}

def _clean_text(text: str) -> str:
    for char, replacement in _CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.strip()


def _parse_dt(text: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _scrape_page(page, stop_before: date) -> tuple[list[dict], bool]:
    """
    Scrape all rows on the current page with issue_date >= stop_before.
    Returns (rows, should_stop) where should_stop means we've passed stop_before.
    """
    rows = page.query_selector_all(ROW_SELECTOR)
    results: list[dict] = []
    should_stop = False

    for row in rows:
        data_id = row.get_attribute("data-id") or ""
        cells = row.query_selector_all("td")
        if len(cells) < 11:
            continue

        def cell_text(idx: int) -> str:
            if idx >= len(cells):
                return ""
            items = cells[idx].query_selector_all("li")
            if items:
                return _clean_text(" | ".join(li.inner_text().strip() for li in items if li.inner_text().strip()))
            return _clean_text(cells[idx].inner_text())

        issue_dt = _parse_dt(cell_text(COL_ISSUE_DATE))
        issue_date = issue_dt.date() if issue_dt else None

        if issue_date is not None and issue_date < stop_before:
            should_stop = True
            continue

        link_el = cells[COL_OPP_ID].query_selector("a")
        href = link_el.get_attribute("href") if link_el else None
        detail_url = (BASE_URL + href) if href and href.startswith("/") else \
                     f"{BASE_URL}/page.aspx/en/bpm/process_manage_extranet/{data_id}"
        opp_id = link_el.inner_text().strip() if link_el else cell_text(COL_OPP_ID)

        closing_dt = _parse_dt(cell_text(COL_CLOSING_DATE))

        results.append({
            "opportunity_id": opp_id,
            "opportunity_description": cell_text(COL_DESCRIPTION),
            "commodities": cell_text(COL_COMMODITIES),
            "type": cell_text(COL_TYPE),
            "issue_date": issue_dt.strftime("%Y-%m-%d %H:%M") if issue_dt else "",
            "closing_date": closing_dt.strftime("%Y-%m-%d %H:%M") if closing_dt else "",
            "organization_issuer": cell_text(COL_ORG_BY),
            "organization_issued_for": cell_text(COL_ORG_FOR),
            "detail_url": detail_url,
        })

    return results, should_stop


def _fetch_detail_summary(context, url: str) -> str:
    """Open the RFP detail page in a new tab and return the Summary Details text."""
    detail_page = context.new_page()
    try:
        detail_page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        summary = detail_page.evaluate(f"""() => {{
            // Find the leaf node whose text is exactly "{SUMMARY_HEADING}"
            const heading = Array.from(document.body.querySelectorAll('*')).find(
                el => el.children.length === 0 &&
                      el.textContent.trim().toLowerCase() === '{SUMMARY_HEADING.lower()}'
            );
            if (!heading) return '';
            // Walk up until there is a next sibling — that sibling holds the content
            let node = heading;
            while (node.parentElement) {{
                const sib = node.nextElementSibling;
                if (sib) return sib.innerText.trim();
                node = node.parentElement;
            }}
            return '';
        }}""")
        return _clean_text(summary) if summary else ""
    except Exception as e:
        print(f"      Warning: summary fetch failed ({url}): {e}")
        return ""
    finally:
        detail_page.close()


class BCBidsScraper:
    def scrape_range(self, start_date: date, end_date: date, fetch_details: bool = False) -> dict[date, list[dict]]:
        """
        Scrape all listings with issue_date between start_date and end_date (inclusive).
        Opens the browser once and paginates until listings are older than start_date.
        Returns a dict keyed by issue date.
        Edge must be fully closed before calling this.
        """
        print(f"Scraping BCBids for {start_date} to {end_date} ...")
        print("(Edge must be fully closed)")

        all_rows: list[dict] = []

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=EDGE_PROFILE,
                channel="msedge",
                headless=False,
                args=["--profile-directory=Default"],
            )
            page = context.new_page()

            print(f"  Navigating to {BROWSE_URL} ...")
            page.goto(BROWSE_URL, wait_until="networkidle", timeout=60_000)

            try:
                page.wait_for_selector(
                    f"table#body_x_grid_grd {ROW_SELECTOR}",
                    timeout=60_000,
                )
            except PlaywrightTimeout:
                print("  ERROR: Timed out waiting for listings. Is Edge closed?")
                context.close()
                return {}

            page_num = 1
            while True:
                print(f"  Scraping page {page_num} ...")
                page_rows, should_stop = _scrape_page(page, stop_before=start_date)
                all_rows.extend(page_rows)
                print(f"    {len(page_rows)} row(s) in date range on this page.")

                if should_stop:
                    print("  Reached listings older than start date — stopping.")
                    break

                next_btn = page.query_selector(NEXT_BTN_SELECTOR)
                if not next_btn or "disabled" in (next_btn.get_attribute("class") or ""):
                    print("  No more pages.")
                    break

                # Capture first row's id so we can detect when the grid has refreshed
                first_row = page.query_selector(ROW_SELECTOR)
                first_row_id = first_row.get_attribute("id") if first_row else None

                next_btn.click()

                # Wait for the grid to replace its rows (UpdatePanel postback)
                if first_row_id:
                    page.wait_for_function(
                        "rowId => document.querySelector(\"tr[data-object-type='rfp']\")?.getAttribute('id') !== rowId",
                        arg=first_row_id,
                        timeout=30_000,
                    )
                else:
                    page.wait_for_selector(ROW_SELECTOR, timeout=30_000)
                page_num += 1

            if fetch_details:
                print(f"  Fetching detail summaries for {len(all_rows)} listing(s) ...")
                for i, row in enumerate(all_rows, 1):
                    print(f"    [{i}/{len(all_rows)}] {row['opportunity_id']} ...", end=" ", flush=True)
                    row["summary"] = _fetch_detail_summary(context, row["detail_url"])
                    print("ok" if row["summary"] else "no summary found")

            context.close()

        # Filter to end_date and group by date
        by_date: dict[date, list[dict]] = {}
        for row in all_rows:
            d = date.fromisoformat(row["issue_date"][:10]) if row["issue_date"] else None
            if d and start_date <= d <= end_date:
                by_date.setdefault(d, []).append(row)

        total = sum(len(v) for v in by_date.values())
        print(f"Done. {total} total listing(s) across {len(by_date)} date(s).")
        return by_date

    def scrape(self, target_date: date | None = None) -> list[dict]:
        """Scrape a single date (defaults to yesterday)."""
        if target_date is None:
            target_date = date.today() - timedelta(days=1)
        return self.scrape_range(target_date, target_date).get(target_date, [])
