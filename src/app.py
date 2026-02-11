# app.py — Upgraded Streamlit demo
# Features added:
# ✅ Select a job and view full details
# ✅ Select a match card and view LLM output side-by-side
# ✅ “Run LLM on this job” button (interactive)
# ✅ Batch run on top N jobs (optional)
# ✅ Saves updated matches back to data/processed/matches_llm.json
# ✅ Uses OPENAI_API_KEY from environment (NO hardcoded keys)

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Optional: LLM calling
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ---- Paths ----
RESUME_PATH = Path("data/processed/resume.json")
JOBS_PATH = Path("data/processed/jobs_clean.json")
MATCHES_PATH = Path("data/processed/matches_llm.json")

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

    # JSON mode: ensure strictly JSON output
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


def normalize_job_text(job: Dict[str, Any]) -> str:
    return f"{job.get('title','')}\n\n{job.get('description','')}".strip()


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
        "- data/processed/resume.json\n"
        "- data/processed/jobs_clean.json\n"
        "(matches_llm.json is optional for first run)"
    )
    st.stop()

if not isinstance(jobs, list):
    st.error("jobs_clean.json must be a list of job objects.")
    st.stop()

# Session state storage (so clicking buttons updates UI immediately)
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

# Sidebar controls
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

    # Job picker
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

        # Run LLM button
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
                        # rebuild matches_state list
                        updated = []
                        seen = set()
                        for m in matches_state:
                            jid = m.get("job_id")
                            if jid == selected_job_id:
                                continue
                            updated.append(m)
                            if jid:
                                seen.add(jid)
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
        # Filters
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

    st.divider()

    st.write("### 3) Batch run (optional)")
    st.caption("Runs LLM on the first N jobs that do NOT already have a stored match.")
    if st.button("⚡ Run LLM batch on N jobs", disabled=(client is None)):
        if client is None:
            st.error("LLM not available. Install openai and set OPENAI_API_KEY.")
        else:
            remaining = []
            for jid in job_ids:
                if jid not in match_by_job_id:
                    remaining.append(jid)
                if len(remaining) >= int(n_jobs_batch):
                    break

            if not remaining:
                st.info("All jobs already have matches (or none left within your batch size).")
            else:
                with st.spinner(f"Running LLM on {len(remaining)} jobs..."):
                    new_items = []
                    for jid in remaining:
                        job = job_by_id[jid]
                        try:
                            llm = score_resume_job_llm(
                                client=client,
                                model=model,
                                resume_text=resume_text,
                                job=job,
                                temperature=temperature
                            )
                            entry = {
                                "resume_id": resume_id,
                                "job_id": jid,
                                "title": job.get("title", ""),
                                "company": job.get("company", ""),
                                **llm
                            }
                            new_items.append(entry)
                            match_by_job_id[jid] = entry
                        except Exception as e:
                            new_items.append({
                                "resume_id": resume_id,
                                "job_id": jid,
                                "title": job.get("title", ""),
                                "company": job.get("company", ""),
                                "error": str(e)
                            })

                    # merge: keep old + add new (upsert by job_id)
                    merged = {m.get("job_id"): m for m in matches_state if m.get("job_id")}
                    for item in new_items:
                        merged[item.get("job_id")] = item

                    matches_state = list(merged.values())
                    st.session_state["matches_state"] = matches_state

                    if save_after_run:
                        save_json(matches_state, MATCHES_PATH)

                st.success(f"Batch complete. Added/updated {len(remaining)} matches.")
                st.rerun()

    st.divider()
    st.write("### Download matches")
    st.download_button(
        label="⬇️ Download matches_llm.json",
        data=json.dumps(st.session_state["matches_state"], indent=2, ensure_ascii=False),
        file_name="matches_llm.json",
        mime="application/json"
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

    # Score stats
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

    # Verdict counts chart
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

    # Missing skills chart
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

    # Quick peek summary
    st.write("### Data summary")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Jobs loaded:**", len(jobs))
    with colB:
        st.write("**Matches loaded:**", len(st.session_state["matches_state"]))