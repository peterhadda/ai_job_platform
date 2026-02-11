# src/fetch_jobs_adzuna.py
# ✅ Updated to save processed jobs to OUTPUT_JOBS_JSON (what Streamlit reads)
# ✅ Fixed return paths
# ✅ Added missing Path import
# ✅ Kept your logic/style, only corrected the parts that broke UI updates

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import requests
import hashlib
import re

from config import RAW_API_PATH, OUTPUT_JOBS_JSON  # <-- use the correct config outputs


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def make_job_id(title: str, company: str, location: str, url: str) -> str:
    seed = f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}|{url.strip()}"
    # md5 is fine for deterministic IDs here (not for security)
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def fetch_adzuna_jobs(
    query: str,
    location: str,
    country: str = "us",
    pages: int = 1,
    results_per_page: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch jobs from Adzuna Search endpoint (paged).
    country examples: us, ca, gb, fr...
    """
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise RuntimeError("Missing ADZUNA_APP_ID / ADZUNA_APP_KEY env vars")

    all_results: List[Dict[str, Any]] = []
    metrics = {"pages_requested": pages, "pages_fetched": 0, "jobs_fetched": 0}

    for page in range(1, pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": query,
            "where": location,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        results = payload.get("results", [])
        all_results.extend(results)

        metrics["pages_fetched"] += 1
        metrics["jobs_fetched"] = len(all_results)

    return all_results, metrics


def transform_to_canonical(
    adzuna_results: List[Dict[str, Any]],
    source: str = "adzuna"
) -> List[Dict[str, Any]]:
    """
    Transform Adzuna fields into YOUR canonical job schema.
    """
    clean_jobs: List[Dict[str, Any]] = []

    for r in adzuna_results:
        title = normalize_text(r.get("title", ""))
        company = normalize_text((r.get("company") or {}).get("display_name", ""))
        location = normalize_text((r.get("location") or {}).get("display_name", ""))
        description = normalize_text(r.get("description", ""))
        url = normalize_text(r.get("redirect_url", ""))
        date_posted = normalize_text(r.get("created", ""))

        # Required field validation
        if not title or not company or not location or not description:
            continue

        job_id = make_job_id(title, company, location, url)

        clean_jobs.append({
            "job_id": job_id,
            "title": title,
            "company": company.lower(),
            "location": location.lower(),
            "date_posted": date_posted,
            "description": description.lower(),
            "url": url,
            "source": source,
        })

    # Deduplicate by job_id
    dedup: Dict[str, Dict[str, Any]] = {}
    for j in clean_jobs:
        dedup[j["job_id"]] = j
    return list(dedup.values())


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_transform_save(
    query: str,
    location: str,
    country: str = "us",
    pages: int = 1,
    results_per_page: int = 20,
) -> Dict[str, Any]:
    # Ensure config paths are Path objects
    raw_path = Path(RAW_API_PATH)
    processed_path = Path(OUTPUT_JOBS_JSON)

    raw_results, metrics = fetch_adzuna_jobs(query, location, country, pages, results_per_page)

    # Save raw traceability payload
    save_json({"results": raw_results, "metrics": metrics}, raw_path)

    # Transform + save canonical jobs where Streamlit reads from
    clean_jobs = transform_to_canonical(raw_results, source="adzuna")
    save_json(clean_jobs, processed_path)

    return {
        "raw_saved_to": str(raw_path),
        "processed_saved_to": str(processed_path),
        "raw_count": len(raw_results),
        "clean_count": len(clean_jobs),
        **metrics
    }