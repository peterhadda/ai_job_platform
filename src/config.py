from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent
RAW_RESUME_PDF_PATH=BASE_DIR /"data"/"raw"/ 'resume.pdf'
MIN_RESUME_TEXT_CHARS=300
MAX_PAGES=5