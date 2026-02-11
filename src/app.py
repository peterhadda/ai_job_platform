# app.py
# Full Streamlit app that works with your config.py (Path objects) and your processed files.
# - Loads .env correctly
# - Imports jobs online (Adzuna) and saves to OUTPUT_JOBS_JSON (data/processed/jobs.json)
# - Runs LLM matches and UPDATES the table immediately (upsert + save + st.rerun)
# - Never crashes when jobs list is empty

from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same folder as this app.py
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_RESUME_JSON, OUTPUT_JOBS_JSON, OUTPUT_MATCHES_LLM_JSON
from fetch_jobs_adzuna import fetch_transform_save  # must be in same folder as app.py (or adjust import)
from llm_match import get_openai_client, score_resume_job_llm


# -----------------------
# Paths (config uses Path)
# -----------------------
RESUME_PATH: Path = OUTPUT_RESUME_JSON
JOBS_PATH: Path = OUTPUT_JOBS_JSON
MATCHES_PATH: Path = OUTPUT_MATCHES_LLM_JSON

st.set_page_config(page_title="AI Job Match Platform", layout="wide")
st.sidebar.write("OPENAI KEY LOADED:", bool(os.environ.get("OPENAI_API_KEY")))
st.sidebar.write("KEY starts with:", os.environ.get("OPENAI_API_KEY")[:10])

# -----------------------
# JSON helpers
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


def file_status_sidebar() -> None:
    st.sidebar.header("Data files")
    st.sidebar.write("resume.json:", "✅" if RESUME_PATH.exists() else "❌")
    st.sidebar.write("jobs.json:", "✅" if JOBS_PATH.exists() else "❌")
    st.sidebar.write("llm_matches.json:", "✅" if MATCHES_PATH.exists() else "❌")


# -----------------------
# Match state updater
# -----------------------
def upsert_match(entry: Dict[str, Any]) -> None:
    """
    Upsert by job_id into st.session_state["matches_state"].
    """
    merged = {m.get("job_id"): m for m in st.session_state["matches_state"] if m.get("job_id")}
    merged[entry["job_id"]] = entry
    st.session_state["matches_state"] = list(merged.values())


def upsert_match_and_refresh(entry: Dict[str, Any]) -> None:
    """
    Upsert + save + rerun (table updates instantly).
    """
    upsert_match(entry)
    save_json(st.session_state["matches_state"], MATCHES_PATH)
    st.rerun()


# -----------------------
# Insights helper
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
matches_file = load_json(MATCHES_PATH)

st.title("AI Job Market Intelligence & Resume Matching Platform (Streamlit Demo)")

if resume is None:
    st.error(f"Missing resume file: {RESUME_PATH}")
    st.stop()

if jobs is None:
    jobs = []
if not isinstance(jobs, list):
    st.error("jobs.json must be a list of job objects.")
    st.stop()

if matches_file is None:
    matches_file = []
if not isinstance(matches_file, list):
    matches_file = []


# -----------------------
# Session state init
# -----------------------
if "matches_state" not in st.session_state:
    st.session_state["matches_state"] = matches_file
if "selected_job_id" not in st.session_state:
    st.session_state["selected_job_id"] = None

resume_text: str = resume.get("clean_text", "") or ""
resume_id: str = resume.get("resume_id", "") or ""


# -----------------------
# Sidebar: Adzuna import
# -----------------------
st.sidebar.header("Online Jobs Import (Adzuna)")

query = st.sidebar.text_input("Keyword", value="data engineer")
where = st.sidebar.text_input("Location", value="Montreal")
country = st.sidebar.selectbox("Country", ["ca", "us", "gb", "fr", "de", "au"], index=0)
pages = st.sidebar.number_input("Pages", min_value=1, max_value=5, value=1)
per_page = st.sidebar.number_input("Results per page", min_value=10, max_value=50, value=20)

st.sidebar.write("Reading jobs from:", str(JOBS_PATH))
st.sidebar.write("Jobs loaded now:", len(jobs))

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

        # Reload jobs from disk so UI updates
        jobs = load_json(JOBS_PATH) or []
        st.session_state["selected_job_id"] = None

        # Optional: clear matches when you fetch new jobs (usually smart)
        st.session_state["matches_state"] = []
        save_json([], MATCHES_PATH)

        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Fetch failed: {e}")


# -----------------------
# Sidebar: LLM settings
# -----------------------
st.sidebar.header("LLM Settings")
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

jobs_count = len(jobs)
if jobs_count == 0:
    st.sidebar.warning("No jobs loaded yet. Fetch jobs online first.")
    n_jobs_batch = 0
else:
    n_jobs_batch = st.sidebar.number_input(
        "Batch run N jobs",
        min_value=1,
        max_value=min(200, jobs_count),
        value=min(5, jobs_count),
    )

save_after_run = st.sidebar.checkbox("Auto-save llm_matches.json after runs", value=True)

client = get_openai_client()
if client is None:
    st.sidebar.warning("LLM disabled: set OPENAI_API_KEY and install openai.")
else:
    st.sidebar.success("LLM ready (OPENAI_API_KEY detected).")


# -----------------------
# Build maps
# -----------------------
job_by_id: Dict[str, Dict[str, Any]] = {j.get("job_id", f"idx_{i}"): j for i, j in enumerate(jobs)}
job_ids = list(job_by_id.keys())

matches_state: List[Dict[str, Any]] = st.session_state["matches_state"]
match_by_job_id: Dict[str, Dict[str, Any]] = {m.get("job_id"): m for m in matches_state if m.get("job_id")}


# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(["📄 Resume", "🤖 LLM Matches", "📊 Insights"])


# -------------------------
# TAB 1: Resume
# -------------------------
with tab1:
    st.subheader("Resume overview")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**resume_id:**", resume.get("resume_id", ""))
        st.write("**file_name:**", resume.get("file_name", ""))
        st.write("**page_count:**", resume.get("page_count", ""))
    with c2:
        st.write("**created_at:**", resume.get("created_at", ""))
        st.write("**extraction_method:**", resume.get("extraction_method", ""))

    st.divider()
    st.subheader("Resume text preview (clean_text)")
    preview_len = st.slider("Preview length (chars)", 300, 4000, 1200, step=100)
    st.text_area("clean_text", (resume_text or "")[:preview_len], height=300)


# -------------------------
# TAB 2: Matches
# -------------------------
with tab2:
    st.subheader("Job selection + LLM matching")

    if len(job_ids) == 0:
        st.info("No jobs loaded. Use the sidebar to fetch jobs online first.")
        st.stop()

    # Job select box labels
    job_labels = [f"{jid} | {job_by_id[jid].get('title','')} — {job_by_id[jid].get('company','')}" for jid in job_ids]

    default_index = 0
    if st.session_state["selected_job_id"] in job_by_id:
        default_index = job_ids.index(st.session_state["selected_job_id"])

    selected_label = st.selectbox("Select a job", job_labels, index=default_index)
    selected_job_id = selected_label.split(" | ")[0].strip()
    st.session_state["selected_job_id"] = selected_job_id
    selected_job = job_by_id[selected_job_id]

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
        st.write("### LLM result (selected job)")
        existing = match_by_job_id.get(selected_job_id)

        if existing:
            st.success("Existing match found.")
            st.write("**Score:**", existing.get("score"))
            st.write("**Verdict:**", existing.get("verdict"))
            st.write("**Reasoning:**", existing.get("reasoning", ""))
            st.write("**Matched skills:**", safe_list(existing.get("matched_skills")) or ["(none)"])
            st.write("**Missing skills:**", safe_list(existing.get("missing_skills")) or ["(none)"])
        else:
            st.info("No LLM match stored for this job yet.")

        st.divider()

        # Run LLM for one job
        if st.button("🤖 Run LLM on this job", disabled=(client is None)):
            if client is None:
                st.error("LLM not available. Set OPENAI_API_KEY and install openai.")
            else:
                with st.spinner("Calling LLM..."):
                    try:
                        llm = score_resume_job_llm(
                            client=client,
                            model=model,
                            resume_text=resume_text,
                            job=selected_job,
                            temperature=temperature,
                        )
                        entry = {
                            "resume_id": resume_id,
                            "job_id": selected_job_id,
                            "title": selected_job.get("title", ""),
                            "company": selected_job.get("company", ""),
                            **llm,
                        }

                        # Update session + file + UI
                        if save_after_run:
                            upsert_match_and_refresh(entry)
                        else:
                            upsert_match(entry)
                            st.rerun()

                    except Exception as e:
                        st.error(f"LLM run failed: {e}")

    st.divider()

    # Table
    st.write("### Matches table")
    df = pd.DataFrame(st.session_state["matches_state"])
    if df.empty:
        st.info("No matches yet. Run the LLM to generate matches.")
    else:
        # Basic filters
        c1, c2, c3 = st.columns(3)
        with c1:
            verdict_filter = st.selectbox("Filter verdict", ["all", "yes", "maybe", "no"])
        with c2:
            min_score = st.slider("Min score", 0, 100, 0)
        with c3:
            show_top = st.number_input("Show top N", min_value=1, max_value=500, value=25)

        df2 = df.copy()
        if verdict_filter != "all" and "verdict" in df2.columns:
            df2 = df2[df2["verdict"] == verdict_filter]
        if "score" in df2.columns:
            df2 = df2[df2["score"].fillna(0) >= min_score]
            df2 = df2.sort_values(by="score", ascending=False)

        st.dataframe(df2.head(int(show_top)), use_container_width=True)

    st.divider()

    # Batch run
    st.write("### Batch run (optional)")
    st.caption("Runs LLM on the first N jobs (overwrites existing matches for those jobs).")

    if st.button("⚡ Run LLM batch on N jobs", disabled=(client is None or n_jobs_batch == 0)):
        if client is None:
            st.error("LLM not available.")
        else:
            with st.spinner(f"Running LLM on {int(n_jobs_batch)} jobs..."):
                new_items: List[Dict[str, Any]] = []
                count = 0

                for jid in job_ids:
                    if count >= int(n_jobs_batch):
                        break
                    job = job_by_id[jid]

                    try:
                        llm = score_resume_job_llm(
                            client=client,
                            model=model,
                            resume_text=resume_text,
                            job=job,
                            temperature=temperature,
                        )
                        entry = {
                            "resume_id": resume_id,
                            "job_id": jid,
                            "title": job.get("title", ""),
                            "company": job.get("company", ""),
                            **llm,
                        }
                        new_items.append(entry)
                    except Exception as e:
                        new_items.append({
                            "resume_id": resume_id,
                            "job_id": jid,
                            "title": job.get("title", ""),
                            "company": job.get("company", ""),
                            "error": str(e),
                        })

                    count += 1

                # Merge into session state
                merged = {m.get("job_id"): m for m in st.session_state["matches_state"] if m.get("job_id")}
                for item in new_items:
                    if item.get("job_id"):
                        merged[item["job_id"]] = item

                st.session_state["matches_state"] = list(merged.values())

                if save_after_run:
                    save_json(st.session_state["matches_state"], MATCHES_PATH)

                st.success(f"Batch complete. Updated {len(new_items)} jobs.")
                st.rerun()

    st.divider()

    # Download
    st.download_button(
        label="⬇️ Download llm_matches.json",
        data=json.dumps(st.session_state["matches_state"], indent=2, ensure_ascii=False),
        file_name="llm_matches.json",
        mime="application/json",
    )


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

    if "verdict" in df.columns:
        st.write("### Verdict counts")
        vc = df["verdict"].value_counts(dropna=False)
        fig = plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.xlabel("verdict")
        plt.ylabel("count")
        st.pyplot(fig)

    st.divider()

    st.write("### Most common missing skills")
    counts = missing_skill_counts(st.session_state["matches_state"])
    if not counts:
        st.info("No missing skills found yet.")
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
    cA, cB = st.columns(2)
    with cA:
        st.write("**Jobs loaded:**", len(jobs))
    with cB:
        st.write("**Matches loaded:**", len(st.session_state["matches_state"]))