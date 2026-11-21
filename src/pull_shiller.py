"""Pull and parse Robert Shiller's monthly stock market data (ie_data.xls).

Panel R takes only D and E from Shiller: the S&P 500 dividends and earnings,
which Shiller already posts as trailing twelve-month values (do not re-sum
them). Prices come from CRSP month-ends, because Shiller's P is a monthly
average of daily closes while Goyal uses month-end prices. P is still saved:
the CRSP-vs-Shiller return cross-check in test_data_validation.py uses it.

Source: the current file is linked from https://shillerdata.com/. The link
carries a version tag that changes with each update, so it is read from the
page on every pull. The old Yale address (econ.yale.edu/~shiller/data) still
serves a copy, but it stopped updating in 2023, so it is deliberately not used
as a fallback.

Outputs in DATA_DIR (never committed):

- ``shiller_ie_data.xls``: the workbook as downloaded;
- ``shiller_ie_data_vintage.json``: its URL, SHA-256 and download time;
- ``shiller.parquet``: columns P, D, E, CPI with a month-end ``date`` index.

Shiller dates months as decimals, and October is written ``2026.1``: the
month is the two digits after the point (2026.01 is January, 2026.1 is
October, 2026.12 is December).
"""

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pull_goyal import clean_numeric_column
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

SHILLER_PAGE_URL = "https://shillerdata.com/"
RAW_FILE_NAME = "shiller_ie_data.xls"
VINTAGE_FILE_NAME = "shiller_ie_data_vintage.json"
DATA_SHEET = "Data"
KEPT_COLUMNS = ["P", "D", "E", "CPI"]
LINK_PATTERN = re.compile(r'href="([^"]*ie_data\.xls[^"]*)"')


def find_workbook_link(page_html):
    """Extract the ie_data.xls download link from the shillerdata.com page."""
    match = LINK_PATTERN.search(page_html)
    if match is None:
        raise ValueError(
            "No ie_data.xls link found on shillerdata.com; the page layout may "
            "have changed."
        )
    link = match.group(1).replace("&amp;", "&")
    return "https:" + link if link.startswith("//") else link


def download_shiller_workbook(page_url=SHILLER_PAGE_URL, timeout=120):
    """Return (download URL, raw bytes) of the current ie_data.xls."""
    page = requests.get(page_url, timeout=timeout)
    page.raise_for_status()
    url = find_workbook_link(page.text)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"\xd0\xcf\x11\xe0"):  # legacy .xls (OLE2) signature
        raise ValueError(f"Expected an .xls workbook from {url}.")
    return url, content


def parse_shiller_dates(codes):
    """Turn Shiller's decimal month codes (2026.01 ... 2026.1 ... 2026.12) into month-ends."""
    codes = pd.to_numeric(pd.Series(codes), errors="raise").astype(float)
    years = np.floor(codes).astype(int)
    months = np.round((codes - years) * 100).astype(int)
    if not months.between(1, 12).all():
        raise ValueError(
            "Shiller date codes must be yyyy.mm with mm between 01 and 12."
        )
    dates = pd.to_datetime(
        pd.DataFrame({"year": years, "month": months, "day": 1})
    ) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex(dates, name="date")


def parse_shiller_sheet(raw_sheet):
    """Parse the raw Data sheet (read with header=None) into P, D, E, CPI by month.

    The header row is located by its first cell, "Date", so a change in the
    number of title rows does not shift the columns. Rows after the monthly
    block (notes, blanks) are dropped because their date cell is not a number.
    """
    first_column = raw_sheet.iloc[:, 0].astype(str).str.strip()
    header_rows = first_column.index[first_column == "Date"]
    if len(header_rows) == 0:
        raise ValueError(
            "Could not find the 'Date' header row in Shiller's Data sheet."
        )
    header_row = header_rows[0]
    header = raw_sheet.loc[header_row].astype(str).str.strip().tolist()
    body = raw_sheet.loc[header_row + 1 :].copy()
    body.columns = header
    missing = [c for c in ["Date"] + KEPT_COLUMNS if c not in body.columns]
    if missing:
        raise ValueError(f"Shiller's Data sheet lacks columns {missing}.")
    body = body.loc[:, ~body.columns.duplicated()]  # later blocks repeat names
    dated = pd.to_numeric(body["Date"], errors="coerce").notna()
    body = body[dated]
    values = body[KEPT_COLUMNS].apply(clean_numeric_column)
    values.index = parse_shiller_dates(body["Date"])
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError("Shiller's Data sheet has duplicated or unsorted months.")
    return values


def read_shiller_workbook(source):
    """Parse a path or raw bytes of ie_data.xls into the P, D, E, CPI frame."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    raw = pd.read_excel(source, sheet_name=DATA_SHEET, header=None, engine="xlrd")
    return parse_shiller_sheet(raw)


def save_shiller(url, content, data, data_dir=DATA_DIR):
    """Write the raw workbook, its vintage record and the parsed parquet."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RAW_FILE_NAME).write_bytes(content)
    vintage = {
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (data_dir / VINTAGE_FILE_NAME).write_text(json.dumps(vintage, indent=2))
    data.to_parquet(data_dir / "shiller.parquet")


def load_shiller(data_dir=DATA_DIR):
    """Load the parsed Shiller data. Run this module first to pull it."""
    return pd.read_parquet(Path(data_dir) / "shiller.parquet")


if __name__ == "__main__":
    workbook_url, workbook_bytes = download_shiller_workbook()
    shiller = read_shiller_workbook(workbook_bytes)
    save_shiller(workbook_url, workbook_bytes, shiller)
    print(f"shiller: {shiller.shape[0]} months, {shiller.index.min().date()} to "
          f"{shiller.index.max().date()}; last D: {shiller['D'].last_valid_index().date()}")  # fmt: skip
