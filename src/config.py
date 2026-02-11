from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
RAW_RESUME_PDF_PATH=BASE_DIR /"data"/"raw"/ 'resume.pdf'
RAW_JOBS_PATH=BASE_DIR /"data"/"raw"/ 'jobs.csv'
MIN_RESUME_TEXT_CHARS=300
MAX_PAGES=5
OUTPUT_RESUME_JSON=BASE_DIR /"data"/"processed"/"resume.json"
OUTPUT_JOBS_JSON=BASE_DIR /"data"/"processed"/"jobs.json"
OUTPUT_MATCHES_JSON=BASE_DIR /"data"/"processed"/"matches.json"
OUTPUT_MATCHES_LLM_JSON=BASE_DIR /"data"/"processed"/"llm_matches.json"