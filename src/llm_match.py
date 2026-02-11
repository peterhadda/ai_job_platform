import os
import json
from openai import OpenAI
from config import OUTPUT_RESUME_JSON, OUTPUT_JOBS_JSON
# ---- Settings ----
N_JOBS = 5
MODEL = "gpt-4o-mini"
TEMPERATURE = 0

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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

    # Light cleanup
    obj["reasoning"] = obj["reasoning"].strip()
    return obj


def score_resume_job(resume_text: str, job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(resume_text, job)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You output only JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    return validate_llm_result(parsed)


def run_llm_matching() -> List[Dict[str, Any]]:
    resume = load_json(RESUME_PATH)
    jobs = load_json(JOBS_PATH)

    resume_id = resume.get("resume_id", "")
    resume_text = resume.get("clean_text", "")

    results: List[Dict[str, Any]] = []

    for job in jobs[:N_JOBS]:
        job_id = job.get("job_id", "")
        title = job.get("title", "")
        company = job.get("company", "")

        try:
            llm_result = score_resume_job(resume_text, job)
            results.append({
                "resume_id": resume_id,
                "job_id": job_id,
                "title": title,
                "company": company,
                **llm_result
            })
        except Exception as e:
            # Don’t crash; record the error for that job
            results.append({
                "resume_id": resume_id,
                "job_id": job_id,
                "title": title,
                "company": company,
                "error": str(e)
            })

    return results


if __name__ == "__main__":
    matches = run_llm_matching()
    save_json(matches, OUTPUT_PATH)
    print(f"✅ Saved {len(matches)} LLM matches to: {OUTPUT_PATH}"