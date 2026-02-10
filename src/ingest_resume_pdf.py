"""
Resume PDF ingestion (MVP)
- Loads a PDF (text-layer only, no OCR)
- Extracts text page-by-page (limited by MAX_PAGES)
- Detects scanned/empty PDFs (rejects if too little text)
- Normalizes text
- Builds a structured resume record with deterministic resume_id
- Validates and returns metrics
"""

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader

# Prefer config.py constants if you have them; otherwise fallback defaults.
try:
    from config import RAW_RESUME_PDF_PATH, MAX_PAGES, MIN_RESUME_TEXT_CHARS  # type: ignore
except Exception:
    RAW_RESUME_PDF_PATH = "data/raw/resume.pdf"
    MAX_PAGES = 5
    MIN_RESUME_TEXT_CHARS = 300


def load_pdf_pages(pdf_path: str, max_pages: int = MAX_PAGES) -> List[Any]:
    """
    1) Open the PDF
    2) Read number of pages
    3) Limit to MAX_PAGES (MVP)
    4) Return the page objects/list
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_to_read = min(total_pages, max_pages)
    return list(reader.pages[:pages_to_read])


def extract_text_from_page(page_obj: Any) -> str:
    """
    Extract text from one page (text-layer).
    Returns empty string if no text is found.
    """
    try:
        text = page_obj.extract_text()
        return text if text else ""
    except Exception:
        return ""


def extract_resume_text(pdf_path: str, max_pages: int = MAX_PAGES) -> Tuple[str, int, str, List[str]]:
    """
    Returns:
      raw_text: concatenated text from pages (text-layer)
      page_count: number of pages actually read
      extraction_method: string label
      warnings: list of warning strings
    """
    pages = load_pdf_pages(pdf_path, max_pages=max_pages)
    page_count = len(pages)

    pages_text: List[str] = []
    empty_pages = 0

    for p in pages:
        t = extract_text_from_page(p)
        if not t.strip():
            empty_pages += 1
        pages_text.append(t)

    raw_text = "\n".join(pages_text).strip()
    warnings: List[str] = []
    extraction_method = "text-layer"

    # Heuristic warning: many empty pages
    if page_count > 0:
        empty_ratio = empty_pages / page_count
        if empty_ratio >= 0.6:
            warnings.append("low_text_density_many_empty_pages")

    return raw_text, page_count, extraction_method, warnings


def normalize_text(text: str) -> str:
    """
    Basic normalization for NLP:
    - normalize newlines
    - lowercase
    - collapse spaces/tabs
    - collapse excessive blank lines
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_scanned_or_empty(raw_text: str, page_count: int, min_chars: int = MIN_RESUME_TEXT_CHARS) -> Tuple[bool, str]:
    """
    MVP rule: reject scanned PDFs (no OCR).
    """
    if page_count == 0:
        return True, "no_pages_found"
    if len(raw_text) < min_chars:
        return True, "too_little_text_possible_scan"
    return False, ""


def make_resume_id(file_name: str, clean_text: str, first_n_chars: int = 500) -> str:
    """
    Deterministic resume_id: file_name + first N chars of clean_text.
    """
    seed = f"{file_name}::{clean_text[:first_n_chars]}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def build_resume_record(pdf_path: str, first_n_chars_for_id: int = 500) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds resume dict + metrics dict.
    Returns (resume_record, metrics).
    """
    file_name = os.path.basename(pdf_path)

    raw_text, page_count, extraction_method, warnings = extract_resume_text(pdf_path)
    clean_text = normalize_text(raw_text)

    resume_id = make_resume_id(file_name=file_name, clean_text=clean_text, first_n_chars=first_n_chars_for_id)
    created_at = datetime.now(timezone.utc).isoformat()

    resume_record: Dict[str, Any] = {
        "resume_id": resume_id,
        "file_name": file_name,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "created_at": created_at,
        "page_count": page_count,
        "extraction_method": extraction_method,
    }

    metrics: Dict[str, Any] = {
        "page_count": page_count,
        "char_count_raw": len(raw_text),
        "char_count_clean": len(clean_text),
        "valid": True,
        "errors": [],
        "warnings": list(warnings),
    }

    return resume_record, metrics


def validate_resume_record(resume_record: Dict[str, Any], min_chars: int = MIN_RESUME_TEXT_CHARS) -> Tuple[bool, List[str], List[str]]:
    """
    Validates required fields + scanned/empty detection.
    Returns (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Required keys
    for key in ["resume_id", "file_name", "raw_text", "clean_text", "created_at", "page_count", "extraction_method"]:
        if key not in resume_record:
            errors.append(f'missing_required_field:{key}')

    # Basic content checks (only if keys exist)
    raw_text = resume_record.get("raw_text", "") or ""
    page_count = int(resume_record.get("page_count", 0) or 0)

    has_problem, reason = detect_scanned_or_empty(raw_text=raw_text, page_count=page_count, min_chars=min_chars)
    if has_problem:
        errors.append(f"scanned_or_empty:{reason}")

    # Optional warning: messy extraction heuristic (lots of repeated whitespace lines)
    clean_text = resume_record.get("clean_text", "") or ""
    if clean_text and clean_text.count("\n") > 300:
        warnings.append("many_line_breaks_possible_layout_noise")

    return (len(errors) == 0), errors, warnings


def run_resume_ingestion(pdf_path: str = RAW_RESUME_PDF_PATH) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrator: build -> validate -> finalize metrics.
    Returns (resume_record, metrics).
    """
    resume_record, metrics = build_resume_record(pdf_path)

    is_valid, errors, warnings = validate_resume_record(resume_record)
    metrics["valid"] = is_valid
    metrics["errors"].extend(errors)
    # keep existing warnings + validation warnings (dedupe)
    all_warnings = metrics.get("warnings", []) + warnings
    metrics["warnings"] = sorted(set(all_warnings))

    return resume_record, metrics


if __name__ == "__main__":
    record, metrics = run_resume_ingestion(RAW_RESUME_PDF_PATH)

    print("VALID:", metrics["valid"])
    print("PAGES:", metrics["page_count"])
    print("RAW CHARS:", metrics["char_count_raw"])
    print("CLEAN CHARS:", metrics["char_count_clean"])

    if metrics["warnings"]:
        print("\nWARNINGS:")
        for w in metrics["warnings"]:
            print(" -", w)

    if metrics["errors"]:
        print("\nERRORS:")
        for e in metrics["errors"]:
            print(" -", e)

    # Optional: preview start of extracted text
    preview = (record.get("clean_text") or "")[:400]
    print("\nTEXT PREVIEW (first 400 chars):")
    print(preview)