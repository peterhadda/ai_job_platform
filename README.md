# 🚀 AI Job Market Intelligence & Resume Matching Platform

An end-to-end data + LLM pipeline that ingests job postings, processes resumes, performs AI-powered matching, and generates explainable skill gap insights.

Built with:
- Python
- Streamlit
- OpenAI API (LLM)
- Adzuna Jobs API
- Pandas
- JSON-based data layer (scalable to DB)

---

## 🎯 Project Goal

This project aims to simulate a real-world AI-powered job intelligence system.

It:
- Ingests job postings (CSV + API)
- Ingests resume PDFs
- Cleans and normalizes data
- Matches resume ↔ job using an LLM
- Explains matched skills and missing skills
- Displays insights in a Streamlit dashboard

---

## 🧠 Architecture Overview

Resume PDF → resume.json
Adzuna API → jobs.json
↓
LLM Matching Engine
↓
llm_matches.json
↓
Streamlit Dashboard


---

---

## 📂 Project Structure

data/
raw/
jobs_api_adzuna.json
resume.pdf
processed/
resume.json
jobs.json
llm_matches.json

src/
app.py
config.py
fetch_jobs_adzuna.py
llm_match.py
.env


---
---

## 📂 Project Structure

data/
raw/
jobs_api_adzuna.json
resume.pdf
processed/
resume.json
jobs.json
llm_matches.json

src/
app.py
config.py
fetch_jobs_adzuna.py
llm_match.py
.env


---

## ⚙️ Features

### ✅ Resume Processing
- PDF ingestion
- Text normalization
- Validation checks
- Deterministic resume_id generation

### ✅ Job Ingestion
- Adzuna API integration
- Canonical schema transformation
- Deduplication by job_id
- Processed job storage

### ✅ LLM Matching
- Structured JSON output
- Score (0–100)
- Verdict (yes / maybe / no)
- Matched skills
- Missing skills
- Reasoning explanation

### ✅ Interactive Dashboard
- Job selection
- Single-job LLM run
- Batch LLM execution
- Match table with filters
- Missing skill frequency analysis
- Downloadable match results

---

## 🔐 Environment Variables

Create a `.env` file inside `/src`:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key


Install dependencies:

pip install streamlit openai requests python-dotenv pandas matplotlib


Run the app:

streamlit run app.py


---

## 📊 Example Output

For each job, the LLM produces:

```json
{
  "score": 60,
  "verdict": "maybe",
  "matched_skills": ["python", "git"],
  "missing_skills": ["sql", "aws"],
  "reasoning": "The resume shows strong Python experience but lacks cloud exposure."
}
