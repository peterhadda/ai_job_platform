import json
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_RESUME_JSON, OUTPUT_JOBS_JSON,OUTPUT_MATCHES_LLM_JSON

# ---- Paths ----
RESUME_PATH = OUTPUT_RESUME_JSON
JOBS_PATH = OUTPUT_JOBS_JSON
MATCHES_PATH = OUTPUT_MATCHES_LLM_JSON

st.set_page_config(page_title="AI Job Match Platform", layout="wide")


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_list(x):
    return x if isinstance(x, list) else []


def show_file_status():
    st.sidebar.header("Data files")
    st.sidebar.write("resume.json:", "✅" if RESUME_PATH.exists() else "❌")
    st.sidebar.write("jobs_clean.json:", "✅" if JOBS_PATH.exists() else "❌")
    st.sidebar.write("matches_llm.json:", "✅" if MATCHES_PATH.exists() else "❌")


def skills_counter_from_missing(matches):
    counts = {}
    for m in matches:
        for s in safe_list(m.get("missing_skills")):
            s2 = str(s).strip().lower()
            if not s2:
                continue
            counts[s2] = counts.get(s2, 0) + 1
    return counts


show_file_status()

resume = load_json(RESUME_PATH)
jobs = load_json(JOBS_PATH)
matches = load_json(MATCHES_PATH)

st.title("AI Job Market Intelligence & Resume Matching Platform (Streamlit Demo)")

# Basic stop if missing files
if resume is None or jobs is None or matches is None:
    st.error(
        "Missing one or more required files. Make sure you have:\n"
        "- data/processed/resume.json\n"
        "- data/processed/jobs_clean.json\n"
        "- data/processed/matches_llm.json"
    )
    st.stop()

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
    clean_text = resume.get("clean_text", "")
    st.text_area("clean_text", clean_text[:preview_len], height=300)


# -------------------------
# TAB 2: LLM Matches
# -------------------------
with tab2:
    st.subheader("LLM match results")

    # Convert to DataFrame
    df = pd.DataFrame(matches)

    # Some entries might have "error"
    if "error" in df.columns:
        st.warning("Some jobs may have errors (LLM output/validation). They are still shown.")

    # Filters
    colA, colB, colC = st.columns(3)
    with colA:
        verdict_filter = st.selectbox("Filter by verdict", ["all", "yes", "maybe", "no"])
    with colB:
        min_score = st.slider("Min score", 0, 100, 0)
    with colC:
        show_top = st.number_input("Show top N rows", min_value=1, max_value=200, value=25)

    # Apply filters safely
    df_filtered = df.copy()
    if verdict_filter != "all" and "verdict" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["verdict"] == verdict_filter]

    if "score" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["score"].fillna(0) >= min_score]

    # Sort
    if "score" in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by="score", ascending=False)

    st.write("### Table view")
    st.dataframe(df_filtered.head(int(show_top)), use_container_width=True)

    st.divider()
    st.write("### Card view (top results)")

    top_cards = df_filtered.head(10).to_dict(orient="records")
    for m in top_cards:
        title = m.get("title", "")
        company = m.get("company", "")
        score = m.get("score", None)
        verdict = m.get("verdict", "")
        job_id = m.get("job_id", "")

        with st.expander(f"{title} — {company} | score={score} | verdict={verdict}"):
            st.write("**job_id:**", job_id)
            if "error" in m and m["error"]:
                st.error(m["error"])
            else:
                st.write("**Reasoning:**", m.get("reasoning", ""))

                matched = safe_list(m.get("matched_skills"))
                missing = safe_list(m.get("missing_skills"))

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Matched skills**")
                    st.write(matched if matched else ["(none)"])
                with col2:
                    st.write("**Missing skills**")
                    st.write(missing if missing else ["(none)"])


# -------------------------
# TAB 3: Insights
# -------------------------
with tab3:
    st.subheader("Insights (Data Analyst view)")

    df = pd.DataFrame(matches)

    # Score stats
    if "score" in df.columns:
        scores = df["score"].dropna()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average score", f"{scores.mean():.1f}" if len(scores) else "n/a")
        with col2:
            st.metric("Best score", f"{scores.max():.0f}" if len(scores) else "n/a")
        with col3:
            st.metric("Worst score", f"{scores.min():.0f}" if len(scores) else "n/a")

    # Verdict counts
    st.write("### Verdict counts")
    if "verdict" in df.columns:
        verdict_counts = df["verdict"].value_counts(dropna=False)
        st.write(verdict_counts)

        # Bar chart
        fig = plt.figure()
        plt.bar(verdict_counts.index.astype(str), verdict_counts.values)
        plt.xlabel("verdict")
        plt.ylabel("count")
        st.pyplot(fig)

    st.divider()

    # Most common missing skills
    st.write("### Most common missing skills")
    counts = skills_counter_from_missing(matches)
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
    st.write("### Raw files quick peek")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Jobs loaded:**", len(jobs) if isinstance(jobs, list) else 0)
    with colB:
        st.write("**Matches loaded:**", len(matches) if isinstance(matches, list) else 0)