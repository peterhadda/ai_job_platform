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
