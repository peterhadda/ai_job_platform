import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_RESUME_JSON, OUTPUT_JOBS_JSON, OUTPUT_MATCHES_LLM_JSON
from src.fetch_jobs_adzuna import fetch_transform_save

# Optional: LLM calling
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ---- Paths ----
RESUME_PATH = Path(OUTPUT_RESUME_JSON)
JOBS_PATH = Path(OUTPUT_JOBS_JSON)
MATCHES_PATH = Path(OUTPUT_MATCHES_LLM_JSON)

st.set_page_config(page_title="AI Job Match Platform", layout="wide")


# -----------------------
# Helpers (JSON I/O)
# -----------------------
def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_list(x) -> List[str]:
    return x if isinstance(x, list) else []


def file_status_sidebar():
    st.sidebar.header("Data files")
    st.sidebar.write("resume.json:", "✅" if RESUME_PATH.exists() else "❌")
    st.sidebar.write("jobs_clean.json:", "✅" if JOBS_PATH.exists() else "❌")
    st.sidebar.write("matches_llm.json:", "✅" if MATCHES_PATH.exists() else "❌")


# -----------------------
# LLM Prompting (Option A)
# -----------------------
def build_prompt(resume_text: str, job: Dict[str, Any]) -> str:
    job_title = job.get("title", "")
    job_desc = job.get("description", "")
    job_company = job.get("company", "")

    return f"""
You are a strict evaluator.
Compare the resume to the job posting.
Do NOT invent facts.
Return ONLY valid JSON with exactly this structure:

{{
  "score": 0,
  "verdict": "yes",
  "matched_skills": [],
  "missing_skills": [],
  "reasoning": ""
}}

Rules:
- score is an integer from 0 to 100
- verdict must be one of: "yes", "maybe", "no"
- matched_skills and missing_skills are short skill phrases (strings)
- reasoning is 1-2 sentences max

RESUME TEXT:
{resume_text}

JOB TITLE: {job_title}
COMPANY: {job_company}

JOB DESCRIPTION:
{job_desc}
""".strip()


def validate_llm_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = ["score", "verdict", "matched_skills", "missing_skills", "reasoning"]
    for k in required_keys:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["score"], int) or not (0 <= obj["score"] <= 100):
        raise ValueError("score must be an int between 0 and 100")

    if obj["verdict"] not in ["yes", "maybe", "no"]:
        raise ValueError('verdict must be "yes", "maybe", or "no"')

    if not isinstance(obj["matched_skills"], list):
        raise ValueError("matched_skills must be a list")
    if not isinstance(obj["missing_skills"], list):
        raise ValueError("missing_skills must be a list")
    if not isinstance(obj["reasoning"], str):
        raise ValueError("reasoning must be a string")

    obj["reasoning"] = obj["reasoning"].strip()
    return obj


def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def score_resume_job_llm(
    client: Any,
    model: str,
    resume_text: str,
    job: Dict[str, Any],
    temperature: float = 0.0
) -> Dict[str, Any]:
    prompt = build_prompt(resume_text, job)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You output only JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    return validate_llm_result(parsed)


# -----------------------
# Insight helpers
# -----------------------
def missing_skill_counts(matches: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in matches:
        for s in safe_list(m.get("missing_skills")):
            s2 = str(s).strip().lower()
            if not s2:
                continue
            counts[s2] = counts.get(s2, 0) + 1
    return counts


# -----------------------
# Load files
# -----------------------
file_status_sidebar()

resume = load_json(RESUME_PATH)
jobs = load_json(JOBS_PATH)
matches = load_json(MATCHES_PATH) or []

st.title("AI Job Market Intelligence & Resume Matching Platform (Streamlit Demo)")

if resume is None or jobs is None:
    st.error(
        "Missing required files. Make sure you have:\n"
        f"- {RESUME_PATH}\n"
        f"- {JOBS_PATH}\n"
        "(matches_llm.json is optional for first run)"
    )
    st.stop()

if not isinstance(jobs, list):
    st.error("jobs_clean.json must be a list of job objects.")
    st.stop()

# Session state storage
if "matches_state" not in st.session_state:
    st.session_state["matches_state"] = matches if isinstance(matches, list) else []
if "selected_job_id" not in st.session_state:
    st.session_state["selected_job_id"] = None

matches_state: List[Dict[str, Any]] = st.session_state["matches_state"]

# Build index maps
job_by_id: Dict[str, Dict[str, Any]] = {j.get("job_id", f"idx_{i}"): j for i, j in enumerate(jobs)}
job_ids = list(job_by_id.keys())

match_by_job_id: Dict[str, Dict[str, Any]] = {}
for m in matches_state:
    jid = m.get("job_id")
    if jid:
        match_by_job_id[jid] = m

resume_text = resume.get("clean_text", "")
resume_id = resume.get("resume_id", "")

# -----------------------
# SIDEBAR SECTION (FIXED ORDER)
# -----------------------
st.sidebar.header("Online Jobs Import (Adzuna)")

query = st.sidebar.text_input("Keyword", value="data engineer")
where = st.sidebar.text_input("Location", value="Tampa")
country = st.sidebar.text_input("Country code", value="us")
pages = st.sidebar.number_input("Pages", min_value=1, max_value=5, value=1)
per_page = st.sidebar.number_input("Results per page", min_value=10, max_value=50, value=20)

if st.sidebar.button("🌐 Fetch jobs online"):
    try:
        with st.spinner("Fetching jobs from Adzuna..."):
            summary = fetch_transform_save(
                query=query,
                location=where,
                country=country,
                pages=int(pages),
                results_per_page=int(per_page),
            )
        st.sidebar.success("Jobs fetched + saved ✅")
        st.sidebar.write(summary)
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Fetch failed: {e}")

st.sidebar.header("LLM Settings")
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
n_jobs_batch = st.sidebar.number_input("Batch run N jobs", min_value=1, max_value=min(200, len(jobs)), value=min(5, len(jobs)))

st.sidebar.header("Actions")
save_after_run = st.sidebar.checkbox("Auto-save matches_llm.json after runs", value=True)

client = get_openai_client()
if client is None:
    st.sidebar.warning("LLM disabled: set OPENAI_API_KEY and install openai.")
else:
    st.sidebar.success("LLM ready (OPENAI_API_KEY detected).")

tab1, tab2, tab3 = st.tabs(["📄 Resume", "🤖 LLM Matches", "📊 Insights"])


# -------------------------
# TAB 1: Resume
# -------------------------
with tab1:
    st.subheader("Resume overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**resume_id:**", resume.get("resume_id", ""))
        st.write("**file_name:**", resume.get("file_name", ""))
        st.write("**page_count:**", resume.get("page_count", ""))
    with col2:
        st.write("**created_at:**", resume.get("created_at", ""))
        st.write("**extraction_method:**", resume.get("extraction_method", ""))

    st.divider()
    st.subheader("Resume text preview (clean_text)")
    preview_len = st.slider("Preview length (chars)", 300, 4000, 1200, step=100)
    st.text_area("clean_text", (resume_text or "")[:preview_len], height=300)


# -------------------------
# TAB 2: Matches + Job selection + Run LLM
# -------------------------
with tab2:
    st.subheader("Job selection + LLM matching")

    st.write("### 1) Pick a job")
    job_labels = []
    for jid in job_ids:
        j = job_by_id[jid]
        job_labels.append(f"{jid} | {j.get('title','')} — {j.get('company','')}")

    default_index = 0
    if st.session_state["selected_job_id"] in job_by_id:
        default_index = job_ids.index(st.session_state["selected_job_id"])

    selected_label = st.selectbox("Select a job", job_labels, index=default_index)
    selected_job_id = selected_label.split(" | ")[0].strip()
    st.session_state["selected_job_id"] = selected_job_id

    selected_job = job_by_id.get(selected_job_id, {})

    colA, colB = st.columns(2)

    with colA:
        st.write("### Job details")
        st.write("**Job ID:**", selected_job_id)
        st.write("**Title:**", selected_job.get("title", ""))
        st.write("**Company:**", selected_job.get("company", ""))
        st.write("**Location:**", selected_job.get("location", ""))
        st.write("**Date posted:**", selected_job.get("date_posted", ""))
        st.write("**URL:**", selected_job.get("url", ""))
        st.text_area("Description", selected_job.get("description", ""), height=260)

    with colB:
        st.write("### LLM result (for selected job)")
        existing = match_by_job_id.get(selected_job_id)

        if existing:
            st.success("Found existing LLM match for this job.")
            st.write("**Score:**", existing.get("score"))
            st.write("**Verdict:**", existing.get("verdict"))
            st.write("**Reasoning:**", existing.get("reasoning", ""))
            st.write("**Matched skills:**")
            st.write(safe_list(existing.get("matched_skills")) or ["(none)"])
            st.write("**Missing skills:**")
            st.write(safe_list(existing.get("missing_skills")) or ["(none)"])
        else:
            st.info("No LLM match stored for this job yet.")

        st.divider()

        can_run = client is not None
        if st.button("🤖 Run LLM on this job", disabled=not can_run):
            if not can_run:
                st.error("LLM not available. Install openai and set OPENAI_API_KEY.")
            else:
                with st.spinner("Calling LLM..."):
                    try:
                        llm = score_resume_job_llm(
                            client=client,
                            model=model,
                            resume_text=resume_text,
                            job=selected_job,
                            temperature=temperature
                        )
                        new_entry = {
                            "resume_id": resume_id,
                            "job_id": selected_job_id,
                            "title": selected_job.get("title", ""),
                            "company": selected_job.get("company", ""),
                            **llm
                        }

                        # upsert into session state
                        match_by_job_id[selected_job_id] = new_entry
                        updated = [m for m in matches_state if m.get("job_id") != selected_job_id]
                        updated.append(new_entry)
                        matches_state = updated
                        st.session_state["matches_state"] = matches_state

                        if save_after_run:
                            save_json(matches_state, MATCHES_PATH)

                        st.success("Saved LLM result for selected job.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"LLM run failed: {e}")

    st.divider()

    st.write("### 2) Table view of all matches")
    df = pd.DataFrame(st.session_state["matches_state"])
    if df.empty:
        st.info("No matches yet. Run the LLM on a job to generate matches.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            verdict_filter = st.selectbox("Filter verdict", ["all", "yes", "maybe", "no"])
        with col2:
            min_score = st.slider("Min score", 0, 100, 0)
        with col3:
            show_top = st.number_input("Show top N", min_value=1, max_value=500, value=25)

        df2 = df.copy()
        if verdict_filter != "all" and "verdict" in df2.columns:
            df2 = df2[df2["verdict"] == verdict_filter]
        if "score" in df2.columns:
            df2 = df2[df2["score"].fillna(0) >= min_score]
            df2 = df2.sort_values(by="score", ascending=False)

        st.dataframe(df2.head(int(show_top)), use_container_width=True)


# -------------------------
# TAB 3: Insights
# -------------------------
with tab3:
    st.subheader("Insights (Data Analyst view)")

    df = pd.DataFrame(st.session_state["matches_state"])
    if df.empty:
        st.info("No matches yet. Generate some LLM matches first.")
        st.stop()

    if "score" in df.columns:
        scores = df["score"].dropna()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Average score", f"{scores.mean():.1f}" if len(scores) else "n/a")
        with c2:
            st.metric("Best score", f"{scores.max():.0f}" if len(scores) else "n/a")
        with c3:
            st.metric("Worst score", f"{scores.min():.0f}" if len(scores) else "n/a")

    st.divider()

    st.write("### Verdict counts")
    if "verdict" in df.columns:
        verdict_counts = df["verdict"].value_counts(dropna=False)
        st.write(verdict_counts)

        fig = plt.figure()
        plt.bar(verdict_counts.index.astype(str), verdict_counts.values)
        plt.xlabel("verdict")
        plt.ylabel("count")
        st.pyplot(fig)

    st.divider()

    st.write("### Most common missing skills")
    counts = missing_skill_counts(st.session_state["matches_state"])
    if not counts:
        st.info("No missing_skills found in matches.")
    else:
        top_n = st.slider("Top N missing skills", 5, 30, 10)
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        skills = [k for k, _ in items]
        freqs = [v for _, v in items]

        fig2 = plt.figure()
        plt.bar(skills, freqs)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("skill")
        plt.ylabel("times missing")
        st.pyplot(fig2)

    st.divider()

    st.write("### Data summary")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Jobs loaded:**", len(jobs))
    with colB:
        st.write("**Matches loaded:**", len(st.session_state["matches_state"]))
