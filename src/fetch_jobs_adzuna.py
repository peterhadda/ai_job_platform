import os
import json
from typing import Any,Dict,List,Tuple
import requests
import hashlib
import re

def normalize_text(s):
    if not s:
        return ''
    s=s.replace("\r\n","\n").replace("\r","\n")
    s=s.strip()
    s=re.sub(r"\s+"," ",s)
    s=re.sub(r"\n{3,}","\n\n",s)
    return s

def make_job_id(title,company,location,url):
    seed=f"{title.lower().strip()}|{company.lower().strip()}|{location.lower().strip()}|{url.strip()}"
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
    country examples: us, gb, ca, fr...
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


def transform_to_canonical(adzuna_results: List[Dict[str, Any]], source: str = "adzuna") -> List[Dict[str, Any]]:
    """
    Transform Adzuna fields into YOUR canonical job schema.
    """
    clean_jobs: List[Dict[str, Any]] = []

    for r in adzuna_results:
        title = normalize_text(r.get("title", ""))
        company = normalize_text((r.get("company") or {}).get("display_name", ""))
        location = normalize_text(((r.get("location") or {}).get("display_name", "")))
        description = normalize_text(r.get("description", ""))
        url = normalize_text(r.get("redirect_url", ""))  # Adzuna typically provides redirect_url
        date_posted = normalize_text(r.get("created", ""))  # often ISO-ish string

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
    dedup = {}
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
    raw_results, metrics = fetch_adzuna_jobs(query, location, country, pages, results_per_page)
    save_json({"results": raw_results, "metrics": metrics}, RAW_API_PATH)

    clean_jobs = transform_to_canonical(raw_results, source="adzuna")
    save_json(clean_jobs, PROCESSED_JOBS_PATH)

    return {
        "raw_saved_to": str(RAW_API_PATH),
        "processed_saved_to": str(PROCESSED_JOBS_PATH),
        "raw_count": len(raw_results),
        "clean_count": len(clean_jobs),
        **metrics
    }