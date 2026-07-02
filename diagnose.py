"""
Diagnostic: find the pagination button selector on the BCBids grid.
Edge must be fully closed before running.
"""
import os
from playwright.sync_api import sync_playwright

URL = "https://www.bcbid.gov.bc.ca/page.aspx/en/rfp/request_browse_public"
EDGE_PROFILE = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")

with sync_playwright() as p:
    context = p.chromium.launch_persistent_context(
        user_data_dir=EDGE_PROFILE,
        channel="msedge",
        headless=False,
        args=["--profile-directory=Default"],
    )
    page = context.new_page()
    page.goto(URL, wait_until="networkidle", timeout=60_000)
    page.wait_for_selector("tr[data-object-type='rfp']", timeout=60_000)

    # Dump all elements that might be pagination controls
    pager_html = page.evaluate("""() => {
        const candidates = [
            ...document.querySelectorAll('[class*="pager"]'),
            ...document.querySelectorAll('[class*="pagination"]'),
            ...document.querySelectorAll('[data-iv-role*="pager"]'),
            ...document.querySelectorAll('[aria-label*="page" i]'),
            ...document.querySelectorAll('[aria-label*="next" i]'),
        ];
        // deduplicate
        const seen = new Set();
        return candidates
            .filter(el => { const k = el.outerHTML.slice(0,200); return seen.has(k) ? false : seen.add(k); })
            .map(el => ({
                tag: el.tagName,
                id: el.id,
                cls: el.className,
                role: el.getAttribute('data-iv-role') || '',
                label: el.getAttribute('aria-label') || '',
                text: el.innerText.trim().slice(0, 60),
                html: el.outerHTML.slice(0, 200),
            }));
    }""")

    print(f"Found {len(pager_html)} pagination-related elements:\n")
    for el in pager_html:
        print(f"  <{el['tag']}> id='{el['id']}' class='{el['cls'][:60]}'")
        print(f"    role='{el['role']}' label='{el['label']}' text='{el['text']}'")
        print(f"    html: {el['html'][:150]}")
        print()

    input("Press Enter to close...")
    context.close()
