"""
Scraper for Alberta Purchasing Connection (APC) opportunity search.
https://purchasing.alberta.ca/search

Unlike BC Bid, this portal exposes a real JSON search API
(POST /api/opportunity/search), so there is no DOM scraping and no
per-item detail-page visit -- the API's projectDescription field already
contains the full text shown on an opportunity's General Information ->
Description tab.

Field names below (postDateTime, closeDateTime, categoryCode, statusCode)
were confirmed against a live API record. categoryCode/statusCode are short
codes (e.g. "CNST", "OPEN"), not spelled-out labels -- the API doesn't
return a separate title field for either. The response also includes
solicitationTypeCode (e.g. "ITB"/"RFP"/"RFQ" -- the analog of bcbids' "type"
column), opportunityTypeCode, agreementTypeCode, and regionOfDelivery if
those turn out to be useful later.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import requests

BASE_URL = "https://purchasing.alberta.ca"
SEARCH_PAGE_URL = f"{BASE_URL}/search"
SEARCH_API_URL = f"{BASE_URL}/api/opportunity/search"
INITIAL_LIMIT = 200
MAX_LIMIT = 6400

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "origin": BASE_URL,
    "referer": SEARCH_PAGE_URL,
    "user-agent": _USER_AGENT,
}

_CHAR_REPLACEMENTS = {
    '‘': "'", '’': "'",   # smart single quotes
    '“': '"', '”': '"',   # smart double quotes
    '–': '-', '—': '-',   # en dash, em dash
    '…': '...',                # ellipsis
    '\xa0': ' ',                    # non-breaking space
}


def _clean_text(text: str) -> str:
    for char, replacement in _CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Source descriptions are multi-paragraph with embedded newlines; collapse
    # to single-line so each CSV row renders as one line in a text editor.
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _build_payload(limit: int, offset: int) -> dict:
    return {
        "query": "",
        "limit": limit,
        "offset": offset,
        "filter": {
            "solicitationNumber": "",
            "categories": [],
            "statuses": [],
            "agreementTypes": [],
            "solicitationTypes": [],
            "opportunityTypes": [],
            "deliveryRegions": [],
            "deliveryRegion": "",
            "organizations": [],
            "unspsc": [],
            "postDateRange": "$$custom",
            "closeDateRange": "$$custom",
            "onlyBookmarked": False,
            "onlyInterestExpressed": False,
        },
        "sortOptions": [{"field": "PostDateTime", "direction": "Desc"}],
    }


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(SEARCH_PAGE_URL, timeout=30)  # warms cookies
    return session


def _post_search(session: requests.Session, payload: dict) -> dict:
    resp = session.post(SEARCH_API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _post_search_via_browser(payload: dict) -> dict:
    """Fallback if the site rejects a plain requests session (e.g. bot
    protection): drive a real browser once and issue the API call through
    its request context, which carries the browser's own cookies."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=_USER_AGENT)
        page = context.new_page()
        page.goto(SEARCH_PAGE_URL, wait_until="networkidle", timeout=60_000)
        resp = context.request.post(SEARCH_API_URL, data=payload)
        data = resp.json()
        browser.close()
        return data


def _fetch_page(poster, limit: int) -> list[dict]:
    data = poster(_build_payload(limit=limit, offset=0))
    return data.get("values") or []


def _extract_row(record: dict[str, Any]) -> tuple[dict, date | None]:
    post_dt = _parse_dt(record.get("postDateTime"))
    close_dt = _parse_dt(record.get("closeDateTime"))

    commodity_titles = record.get("commodityCodeTitles") or []

    row = {
        "opportunity_id": record.get("referenceNumber") or record.get("id") or "",
        "title": _clean_text(record.get("title") or record.get("shortTitle") or ""),
        "solicitation_number": record.get("solicitationNumber") or "",
        "organization": _clean_text(record.get("contractingOrganization") or ""),
        "category": record.get("categoryCode") or "",
        "status": record.get("statusCode") or "",
        "commodity_codes": " | ".join(commodity_titles) if commodity_titles else "",
        "post_date": post_dt.strftime("%Y-%m-%d %H:%M") if post_dt else "",
        "close_date": close_dt.strftime("%Y-%m-%d %H:%M") if close_dt else "",
        "description": _clean_text(record.get("projectDescription") or ""),
        "detail_url": f"{BASE_URL}/opportunity/{record['id']}" if record.get("id") else "",
    }
    return row, (post_dt.date() if post_dt else None)


class AlbertaScraper:
    def scrape_range(self, start_date: date, end_date: date, fetch_details: bool = False) -> dict[date, list[dict]]:
        """
        Scrape all opportunities with post date between start_date and
        end_date (inclusive).

        The site exposes no server-side date filter and the search API is
        sorted newest-first over a live, constantly-updated feed, so
        multi-request offset pagination is unsafe: postings added between
        requests shift the sort order and can make a later page skip past
        the target range entirely. Instead, every attempt is a single,
        self-contained request (offset 0, increasing limit) -- postings
        added between attempts just add noise ahead of the target range
        (filtered out below), they can't corrupt already-fetched data.
        Escalates until the oldest record in the batch is older than
        start_date, the API returns fewer records than requested (start of
        all history reached), or MAX_LIMIT is hit.

        fetch_details is accepted for interface parity with BCBidsScraper
        but unused -- the search API already returns the full description.
        """
        print(f"Scraping Alberta Purchasing Connection for {start_date} to {end_date} ...")

        limit = INITIAL_LIMIT
        try:
            session = _make_session()
            poster = lambda payload: _post_search(session, payload)
            records = _fetch_page(poster, limit)
        except requests.RequestException as e:
            print(f"  Direct request failed ({e}); falling back to browser-driven session ...")
            poster = _post_search_via_browser
            records = _fetch_page(poster, limit)

        extracted = [_extract_row(r) for r in records]
        while extracted:
            oldest = min((d for _, d in extracted if d is not None), default=None)
            reached_start = oldest is not None and oldest < start_date
            exhausted = len(records) < limit
            if reached_start or exhausted or limit >= MAX_LIMIT:
                break
            limit *= 2
            records = _fetch_page(poster, limit)
            extracted = [_extract_row(r) for r in records]

        print(f"  fetched {len(records)} record(s) (limit={limit})")

        by_date: dict[date, list[dict]] = {}
        for row, d in extracted:
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
