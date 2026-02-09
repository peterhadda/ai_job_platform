import hashlib
import os
import re
from datetime import datetime, timezone
from random import seed

from pypdf import PdfReader

from config import RAW_RESUME_PDF_PATH,MAX_PAGES,MIN_RESUME_TEXT_CHARS



file=RAW_RESUME_PDF_PATH
pages_txt=""
raw_txt=""
cleaned_txt=""
resume_dict={"resume_id":0,"file_name":"","raw_text":'',"clean_text":'',"created_at":'',"page_count":'',
             "extraction_method":''}
metrics={"page_count":0,"char_count_raw":0,"char_count_clean":0,"valid":True,"errors":[],"warnings":[]}


def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    pages_to_read=min(number_of_pages,MAX_PAGES)
    pages=reader.pages[:pages_to_read]
    return pages

def extract_text_from_pdf(pdf_path: str) -> tuple[str, int]:
    reader = PdfReader(pdf_path)

    pages_to_read = min(len(reader.pages), MAX_PAGES)

    full_text = []
    for i in range(pages_to_read):
        page_text = reader.pages[i].extract_text()
        if page_text:
            full_text.append(page_text)

    return "\n".join(full_text), pages_to_read

def extract_resume_text(pdf_path: str) -> tuple[str, int]:
    reader = PdfReader(pdf_path)

    pages_to_read = min(len(reader.pages), MAX_PAGES)

    pages_txt = ""

    for page_num in range(pages_to_read):
        text = reader.pages[page_num].extract_text()
        if text:
            pages_txt += text + "\n"

    return pages_txt, pages_to_read

def detect_scan_or_empty(raw_text,page_count):
    if len(raw_text)<MIN_RESUME_TEXT_CHARS:
        return True,"Too little text possible scan"
    if page_count==0:
        return True,"No pages found"
    else:
        return False,"All clear"


def normalize_text(text):
    text=text.replace("\r\n","\n").replace("\r","\n")
    text=text.lower()
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()



def build_resumer_record(pdf_path,first_n_chars=MIN_RESUME_TEXT_CHARS):
    file_name=pdf_path
    warnings=[]
    warnings.append("extract_resumer_text() is a stub-replace with real PDF extraction")

    raw_text,page_number=extract_text_from_pdf(pdf_path)
    clean_text=normalize_text(raw_text)
    seed = f"{file_name}::{clean_text[:first_n_chars]}"
    resume_id=hashlib.sha256(seed.encode("utf-8")).hexdigest()
    created_at=datetime.now(timezone.utc).isoformat()
    resume_dict["resume_id"] = resume_id
    resume_dict["file_name"] = file_name
    resume_dict["raw_text"] = raw_text
    resume_dict["clean_text"] = clean_text
    resume_dict["page_count"] = page_number
    resume_dict["created_at"] = created_at
    return resume_dict






# Usage
pdf_file_path = file
all_text = extract_text_from_pdf(pdf_file_path)
raw_text, page_count = extract_text_from_pdf(RAW_RESUME_PDF_PATH)


