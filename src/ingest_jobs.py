import csv
import hashlib
import re

from config import *




def load_jobs_from_csv(csv_path):
    raw_jobs = []
    if not csv_path.exists():
        return raw_jobs
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_jobs.append(row)
    return raw_jobs

def normalize_txt(line):
    if not isinstance(line, str):
        return line
    line = line.strip().lower()
    line= re.sub(r"\s+", " ", line).strip()
    return line

def clean_job_record(raw_job):
    clean_jobs=raw_job
    for jobs in clean_jobs:
        stripped_dict={k.strip(): v.strip() for k, v in jobs.items()}
        if "company" in stripped_dict:
            stripped_dict["company"] = stripped_dict["company"].lower()
        if "location" in stripped_dict:
                stripped_dict["location"] = stripped_dict["location"].lower()
        if "description" in stripped_dict:
            stripped_dict["description"] = normalize_txt(stripped_dict["description"])


        return stripped_dict



def detect_scanned_or_empty(job: dict) -> tuple[bool, list[str]]:
    """
    Reject jobs with missing, empty, or non-text fields
    (common for scanned PDFs without OCR).

    Returns:
        (is_invalid, errors)
    """
    errors = []

    if not isinstance(job, dict) or not job:
        return True, ["job is empty or not a dict"]

    for key, value in job.items():
        if value is None:
            errors.append(f"{key} is None")
        elif isinstance(value, str):
            if not value.strip():
                errors.append(f"{key} is empty")
        elif isinstance(value, dict):
            # Nested content (e.g. OCR blocks) — must contain text
            text = value.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{key}.text is empty or missing")
        else:
            errors.append(f"{key} has invalid type: {type(value).__name__}")

    if errors:
        return True, errors

    return False, []

def make_job_id(job):
    title=normalize_txt(job["title"])
    company=normalize_txt(job["company"])
    location=normalize_txt(job["location"])
    url=normalize_txt(job["url"])
    string_addtion=title+company+location+url
    hash_object = hashlib.sha256(string_addtion)

    return hash_object.hexdigest()








