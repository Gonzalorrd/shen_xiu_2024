"""Pull and parse Amit Goyal's Welch-Goyal predictor workbook (no credentials).

Goyal posts the updated Welch-Goyal (2008) data as one Excel workbook with
Monthly, Quarterly and Annual sheets, 1871 through 2025. On his website
(https://sites.google.com/view/agoyal145) it is the "Updated data (up to 2025)"
link under Welch and Goyal (2008). The workbook posts *inputs* (Index, D12, E12,
b/m, tbl, AAA, BAA, lty, ntis, Rfree, infl, ltr, corpr, svar, CRSP_SPvw, ...),
not finished predictors. Predictors such as d/p or dfy are computed in
``build_predictors.py``, for Panel G and Panel R alike.

This module only downloads, parses and caches the workbook. Outputs, written to
DATA_DIR and never committed:

- ``goyal_predictor_data.xlsx``: the workbook exactly as downloaded;
- ``goyal_predictor_data_vintage.json``: its URL, SHA-256 and download time,
  because Goyal revises the posted file in place;
- ``goyal_monthly.parquet``, ``goyal_quarterly.parquet``, ``goyal_annual.parquet``:
  one parsed sheet each, Goyal's column names, and a period-end DatetimeIndex
  named ``date``.

Parsing rules (Appendix C.2):

- numbers may be stored as text with thousands separators, e.g. ``"6,845.50"``;
- missing values are a mix of ``"NaN"`` strings and blank cells;
- yyyymm / yyyyq / yyyy codes become period-end timestamps
  (e.g. 202512 -> 2025-12-31, 20253 -> 2025-09-30, 2025 -> 2025-12-31).

Any cell that is neither a number nor a recognized missing marker raises an
error, so a change in Goyal's file format fails loudly instead of silently
turning into NaN.
"""

import hashlib
import io
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = Path(config("DATA_DIR"))

# Google Sheets export link for the "Updated data (up to 2025)" workbook.
GOYAL_FILE_ID = "1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ"
GOYAL_WORKBOOK_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOYAL_FILE_ID}/export?format=xlsx"
)
RAW_WORKBOOK_NAME = "goyal_predictor_data.xlsx"
VINTAGE_FILE_NAME = "goyal_predictor_data_vintage.json"

SHEETS = {
    "monthly": {"sheet_name": "Monthly", "date_column": "yyyymm"},
    "quarterly": {"sheet_name": "Quarterly", "date_column": "yyyyq"},
    "annual": {"sheet_name": "Annual", "date_column": "yyyy"},
}
DATE_CODE_LENGTH = {"monthly": 6, "quarterly": 5, "annual": 4}
MISSING_MARKERS = {"", "nan", "na", "n/a", "null", "none"}


def download_goyal_workbook(url=GOYAL_WORKBOOK_URL, timeout=120):
    """Download the workbook and return its raw bytes.

    Raises ValueError if the response is not an Excel file (Google returns an
    HTML page when a link is moved or access is restricted).
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"PK"):  # .xlsx files are zip archives
        raise ValueError(
            f"Expected an .xlsx workbook from {url}, got "
            f"{response.headers.get('Content-Type', 'unknown content')}. "
            "Check that the link on Goyal's website has not changed."
        )
    return content


def clean_numeric_column(column):
    """Convert one workbook column to float.

    Handles numbers stored as text with thousands separators and a mix of
    "NaN" strings and blanks. Raises ValueError on any other text.

    >>> clean_numeric_column(pd.Series(["6,845.50", "NaN", "", 4.44])).tolist()
    [6845.5, nan, nan, 4.44]
    """
    if pd.api.types.is_numeric_dtype(column):
        return column.astype(float)
    as_text = column.astype("string").str.strip().str.replace(",", "", regex=False)
    as_text = as_text.mask(as_text.str.lower().isin(MISSING_MARKERS))
    return pd.to_numeric(as_text, errors="raise").astype(float)


def parse_period_end_dates(codes, freq):
    """Turn Goyal's yyyymm / yyyyq / yyyy codes into period-end timestamps.

    Codes must be integers of the right length (6, 5 and 4 digits), so a code
    from the wrong sheet or a fractional code raises instead of being truncated.
    """
    if freq not in DATE_CODE_LENGTH:
        raise ValueError(f"Unknown frequency {freq!r}; use one of {list(SHEETS)}.")
    numeric = pd.to_numeric(pd.Series(codes), errors="raise")
    if not (numeric == numeric.round()).all():
        raise ValueError(f"{freq} date codes must be integers.")
    codes = numeric.astype(int).astype(str)
    if (codes.str.len() != DATE_CODE_LENGTH[freq]).any():
        raise ValueError(
            f"{freq} date codes must have {DATE_CODE_LENGTH[freq]} digits, "
            f"e.g. {codes[codes.str.len() != DATE_CODE_LENGTH[freq]].iloc[0]!r}."
        )
    if freq == "monthly":
        dates = pd.to_datetime(codes, format="%Y%m") + pd.offsets.MonthEnd(0)
    elif freq == "quarterly":
        years = codes.str[:4].astype(int)
        quarters = codes.str[4:].astype(int)
        if not quarters.between(1, 4).all():
            raise ValueError("Quarterly codes must look like yyyyq, e.g. 20253.")
        dates = pd.to_datetime(
            pd.DataFrame({"year": years, "month": 3 * quarters, "day": 1})
        ) + pd.offsets.MonthEnd(0)
    else:  # annual
        dates = pd.to_datetime(codes + "-12-31", format="%Y-%m-%d")
    return pd.DatetimeIndex(dates, name="date")


def parse_goyal_sheet(raw_sheet, freq):
    """Clean one raw sheet: numeric columns and a period-end ``date`` index."""
    date_column = SHEETS[freq]["date_column"]
    if raw_sheet.columns[0] != date_column:
        raise ValueError(
            f"Expected the first column of the {freq} sheet to be {date_column!r}, "
            f"found {raw_sheet.columns[0]!r}."
        )
    sheet = raw_sheet.dropna(how="all")  # trailing blank rows
    rows_without_date = sheet[date_column].isna()
    if rows_without_date.any():
        raise ValueError(
            f"{rows_without_date.sum()} rows of the {freq} sheet have values but no date."
        )
    values = sheet.drop(columns=date_column).apply(clean_numeric_column)
    values.index = parse_period_end_dates(sheet[date_column], freq)
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError(f"The {freq} sheet has duplicated or unsorted dates.")
    return values


def read_goyal_workbook(source):
    """Parse every sheet of the workbook (a path or raw bytes).

    Returns a dict {"monthly": df, "quarterly": df, "annual": df}.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with warnings.catch_warnings():
        # Goyal's sheets carry print headers/footers openpyxl cannot read.
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        raw_sheets = pd.read_excel(
            source,
            sheet_name=[spec["sheet_name"] for spec in SHEETS.values()],
            engine="openpyxl",
        )
    return {
        freq: parse_goyal_sheet(raw_sheets[spec["sheet_name"]], freq)
        for freq, spec in SHEETS.items()
    }


def pull_goyal(url=GOYAL_WORKBOOK_URL):
    """Download the workbook and return (raw bytes, dict of parsed sheets)."""
    content = download_goyal_workbook(url)
    return content, read_goyal_workbook(content)


def load_goyal(freq="monthly", data_dir=DATA_DIR):
    """Load one parsed Goyal sheet. Run this module first to pull the data."""
    if freq not in SHEETS:
        raise ValueError(f"Unknown frequency {freq!r}; use one of {list(SHEETS)}.")
    return pd.read_parquet(Path(data_dir) / f"goyal_{freq}.parquet")


def save_goyal(content, sheets, data_dir=DATA_DIR, url=GOYAL_WORKBOOK_URL):
    """Write the raw workbook, its vintage record and one parquet file per sheet.

    Goyal revises the posted file in place, so the SHA-256 and download time
    record exactly which vintage a build used.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RAW_WORKBOOK_NAME).write_bytes(content)
    vintage = {
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (data_dir / VINTAGE_FILE_NAME).write_text(json.dumps(vintage, indent=2))
    for freq, sheet in sheets.items():
        sheet.to_parquet(data_dir / f"goyal_{freq}.parquet")


if __name__ == "__main__":
    workbook_bytes, parsed_sheets = pull_goyal()
    save_goyal(workbook_bytes, parsed_sheets)
    for name, df in parsed_sheets.items():
        print(f"goyal_{name}: {df.shape[0]} rows, {df.index.min().date()} "
              f"to {df.index.max().date()}")  # fmt: skip
