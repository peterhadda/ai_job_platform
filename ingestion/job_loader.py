from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


# ---------- Logging setup ----------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("jobs_loader")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if this file is imported multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# ---------- Job ID generation ----------
def generate_job_id(title: str, company: str, location: str, url: str) -> str:
    normalized = (
        f"{(title or '').strip().lower()}|"
        f"{(company or '').strip().lower()}|"
        f"{(location or '').strip().lower()}|"
        f"{(url or '').strip().lower()}"
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


# ---------- Core loader ----------
REQUIRED_COLUMNS = [
    "title",
    "company",
    "location",
    "date_posted",
    "description",
    "url",
    "source",
    # job_id is optional because we can generate it
]

def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def clean_text_series(s: pd.Series) -> pd.Series:
    # Convert to string safely, keep NaN as NaN
    s = s.astype("string")
    # Strip whitespace
    s = s.str.strip()
    return s


def load_and_clean_jobs(
    input_csv: Path,
    output_csv: Path,
    log_path: Path,
) -> pd.DataFrame:
    logger = setup_logger(log_path)

    logger.info(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    logger.info(f"Rows loaded (raw): {len(df)}")
    validate_columns(df)

    # Normalize key fields
    for col in ["title", "company", "location", "description", "url", "source", "date_posted"]:
        df[col] = clean_text_series(df[col])

    # Optional: normalize casing for company/location (keep title case feel)
    # You can choose .str.lower() if you prefer strict normalization
    df["company"] = df["company"].str.replace(r"\s+", " ", regex=True)
    df["location"] = df["location"].str.replace(r"\s+", " ", regex=True)

    # Track drop reasons
    drop_reasons = {}

    def mark_drop(mask: pd.Series, reason: str) -> None:
        nonlocal drop_reasons
        count = int(mask.sum())
        if count > 0:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + count

    # Basic validity checks
    empty_desc = df["description"].isna() | (df["description"].str.len() == 0)
    mark_drop(empty_desc, "empty_description")

    empty_title = df["title"].isna() | (df["title"].str.len() == 0)
    mark_drop(empty_title, "empty_title")

    empty_company = df["company"].isna() | (df["company"].str.len() == 0)
    mark_drop(empty_company, "empty_company")

    # Drop invalid rows
    invalid_mask = empty_desc | empty_title | empty_company
    before = len(df)
    df = df.loc[~invalid_mask].copy()
    logger.info(f"Rows kept after validation: {len(df)} (dropped {before - len(df)})")

    # Generate job_id if missing or empty
    if "job_id" not in df.columns:
        df["job_id"] = pd.NA

    missing_id = df["job_id"].isna() | (df["job_id"].astype("string").str.strip().fillna("") == "")
    logger.info(f"job_id missing/empty: {int(missing_id.sum())} → generating deterministic IDs")

    df.loc[missing_id, "job_id"] = df.loc[missing_id].apply(
        lambda row: generate_job_id(
            title=str(row.get("title", "")),
            company=str(row.get("company", "")),
            location=str(row.get("location", "")),
            url=str(row.get("url", "")),
        ),
        axis=1,
    )

    # Deduplicate by job_id (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["job_id"], keep="first").copy()
    logger.info(f"Duplicates removed by job_id: {before - len(df)}")

    # Log drop reasons summary
    for reason, count in drop_reasons.items():
        logger.info(f"Dropped ({reason}): {count}")

    # Ensure output folder exists + save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved cleaned jobs → {output_csv} | rows={len(df)}")

    return df


if __name__ == "__main__":
    # Example run:
    # python src/ingestion/jobs_loader.py
    input_csv = Path("data/raw/jobs/jobs.csv")
    output_csv = Path("data/processed/jobs_clean.csv")
    log_path = Path("logs/jobs_loader.log")

    load_and_clean_jobs(input_csv, output_csv, log_path)
:


