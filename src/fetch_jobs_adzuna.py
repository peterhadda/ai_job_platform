# fetch_jobs_adzuna.py

import os
import json
import re
import hashlib
from typing import Any, Dict, List, Tuple

import requests

from config import RAW_API_PATH, OUTPUT_JOBS_JSON, OUTPUT_API_PATH  # all are Path objects


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def make_job_id(title: str, company: str, location: str, url: str) -> str:
    seed = f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}|{url.strip()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def save_json(obj: Any, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def fetch_adzuna_jobs(
    query: str,
    location: str,
    country: str = "ca",
    pages: int = 1,
    results_per_page: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

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


def transform_to_canonical(adzuna_results: List[Dict[str, Any]], source: str = "adzuna") -> List[Dict[str, Any]]:
    clean_jobs: List[Dict[str, Any]] = []

    for r in adzuna_results:
        title = normalize_text(r.get("title", ""))
        company = normalize_text((r.get("company") or {}).get("display_name", ""))
        location = normalize_text((r.get("location") or {}).get("display_name", ""))
        description = normalize_text(r.get("description", ""))
        url = normalize_text(r.get("redirect_url", ""))
        date_posted = normalize_text(r.get("created", ""))

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

    # dedupe by job_id
    dedup = {}
    for j in clean_jobs:
        dedup[j["job_id"]] = j
    return list(dedup.values())


def fetch_transform_save(
    query: str,
    location: str,
    country: str = "ca",
    pages: int = 1,
    results_per_page: int = 20,
) -> Dict[str, Any]:

    raw_results, metrics = fetch_adzuna_jobs(query, location, country, pages, results_per_page)

    # 1) Save RAW (traceability)
    save_json({"results": raw_results, "metrics": metrics}, RAW_API_PATH)

    # 2) Transform -> canonical
    clean_jobs = transform_to_canonical(raw_results, source="adzuna")

    # 3) Save PROCESSED jobs to the SAME file Streamlit reads
    save_json(clean_jobs, OUTPUT_JOBS_JSON)

    # 4) Optional: also save a second copy (cleaned api)
    save_json(clean_jobs, OUTPUT_API_PATH)

    return {
        "raw_saved_to": str(RAW_API_PATH),
        "processed_jobs_saved_to": str(OUTPUT_JOBS_JSON),
        "extra_clean_copy_saved_to": str(OUTPUT_API_PATH),
        "raw_count": len(raw_results),
        "clean_count": len(clean_jobs),
        **metrics,
    }