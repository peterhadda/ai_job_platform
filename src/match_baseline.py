"""
Simple Resume ↔ Job matcher (keyword overlap)

Implements:
1) load_json(path) -> object
2) make_job_text(job) -> str
3) tokenize(text) -> list[str]
4) compute_score(resume_tokens, job_tokens) -> float
5) explain_match(resume_tokens, job_tokens) -> dict
6) rank_jobs(resume, jobs, top_k) -> list[dict]
7) save_json(obj, output_path)
8) run_matching()

Expected input JSON shapes (minimal):

resume.json:
{
  "resume_id": "r1",
  "raw_text": "...",   # OR "clean_text": "..."
}

jobs.json:
[
  {"job_id":"j1","title":"...","company":"...","description":"..."},
  ...
]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Iterable, Tuple


# --------- 1) load_json ---------
def load_json(path: str | Path) -> Any:
    """Reads JSON and returns a Python object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------- 7) save_json ---------
def save_json(obj: Any, output_path: str | Path) -> None:
    """Writes JSON to disk (pretty printed)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# Small, safe default stopword list (feel free to expand)
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "not", "of", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "you", "your",
}


# --------- 2) make_job_text ---------
def make_job_text(job: Dict[str, Any]) -> str:
    """Combines job title + description into one string."""
    title = str(job.get("title", "") or "")
    desc = str(job.get("description", "") or "")
    company = str(job.get("company", "") or "")
    # Company is optional but sometimes helps matching
    return f"{title}\n{company}\n{desc}".strip()


# --------- 3) tokenize ---------
def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    stopwords: set[str] | None = None,
    min_len: int = 2,
) -> List[str]:
    """
    Split text into words, normalize, optionally remove stopwords,
    keep only useful words.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    text = (text or "").lower()

    # Keep letters/numbers, split on everything else.
    # Example: "C++/Python" -> ["c", "python"] (simple on purpose)
    tokens = re.findall(r"[a-z0-9]+", text)

    cleaned: List[str] = []
    for t in tokens:
        if len(t) < min_len:
            continue
        if remove_stopwords and t in stopwords:
            continue
        cleaned.append(t)
    return cleaned


# --------- 4) compute_score ---------
def compute_score(resume_tokens: List[str], job_tokens: List[str]) -> float:
    """
    Simplest score:
      matched_words = intersection
      score = matched_words_count / job_words_count
    """
    if not job_tokens:
        return 0.0

    resume_set = set(resume_tokens)
    job_set = set(job_tokens)

    matched = resume_set.intersection(job_set)
    score = len(matched) / max(len(job_set), 1)
    return float(score)


# Helpers for "top 10" keywords (by frequency in the job text)
def _top_keywords(tokens: List[str], limit: int = 10) -> List[str]:
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # Sort by frequency desc, then alphabetically for stable output
    return [k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:limit]]


# --------- 5) explain_match ---------
def explain_match(resume_tokens: List[str], job_tokens: List[str]) -> Dict[str, List[str]]:
    """
    returns:
      matched_keywords (top 10)
      missing_keywords (top 10)
    """
    resume_set = set(resume_tokens)
    job_set = set(job_tokens)

    matched = list(job_set.intersection(resume_set))
    missing = list(job_set.difference(resume_set))

    # Rank matched/missing using frequency in the job tokens
    job_freq_rank = _top_keywords(job_tokens, limit=10_000)  # ranked list
    job_rank_index = {kw: i for i, kw in enumerate(job_freq_rank)}

    matched_sorted = sorted(matched, key=lambda kw: job_rank_index.get(kw, 10**9))[:10]
    missing_sorted = sorted(missing, key=lambda kw: job_rank_index.get(kw, 10**9))[:10]

    return {
        "matched_keywords": matched_sorted,
        "missing_keywords": missing_sorted,
    }


# --------- 6) rank_jobs ---------
def rank_jobs(resume: Dict[str, Any], jobs: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    For each job:
      - compute score
      - create result record:
          resume_id, job_id, title, company, score, explanation
    Sort by score desc and return top_k.
    """
    resume_id = str(resume.get("resume_id", "resume_unknown"))

    # Prefer clean_text if available, else raw_text
    resume_text = str(resume.get("clean_text") or resume.get("raw_text") or "")
    resume_tokens = tokenize(resume_text)

    results: List[Dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("job_id", "job_unknown"))
        title = str(job.get("title", ""))
        company = str(job.get("company", ""))

        job_text = make_job_text(job)
        job_tokens = tokenize(job_text)

        score = compute_score(resume_tokens, job_tokens)
        explanation = explain_match(resume_tokens, job_tokens)

        results.append(
            {
                "resume_id": resume_id,
                "job_id": job_id,
                "title": title,
                "company": company,
                "score": round(score, 6),
                "explanation": explanation,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[: max(int(top_k), 0)]


# --------- 8) run_matching ---------
def run_matching(
    *,
    resume_path: str | Path = "data/processed/resume.json",
    jobs_path: str | Path = "data/processed/jobs.json",
    output_path: str | Path = "data/processed/matches.json",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Loads resume + jobs, ranks jobs, saves matches.json, and returns matches.
    """
    resume = load_json(resume_path)
    jobs = load_json(jobs_path)

    if not isinstance(resume, dict):
        raise TypeError("resume.json must be a JSON object (dict).")
    if not isinstance(jobs, list):
        raise TypeError("jobs.json must be a JSON array (list of jobs).")

    matches = rank_jobs(resume, jobs, top_k=top_k)
    save_json(matches, output_path)
    return matches


# Optional CLI usage: python matcher.py
if __name__ == "__main__":
    matches = run_matching(top_k=10)
    print(f"Saved {len(matches)} matches to data/processed/matches.json")
    for i, m in enumerate(matches, start=1):
        print(f"{i:02d}. {m['score']:.3f} | {m['company']} — {m['title']}")