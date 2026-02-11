import os
import json
from openai import OpenAI
from config import OUTPUT_RESUME_JSON, OUTPUT_JOBS_JSON

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_result(obj: dict) -> dict:
    # Minimal guardrails
    required = ["score", "matched_skills", "missing_skills", "reasoning"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["score"], int) or not (0 <= obj["score"] <= 100):
        raise ValueError("score must be an int between 0 and 100")

    if not isinstance(obj["matched_skills"], list):
        raise ValueError("matched_skills must be a list")

    if not isinstance(obj["missing_skills"], list):
        raise ValueError("missing_skills must be a list")

    if not isinstance(obj["reasoning"], str):
        raise ValueError("reasoning must be a string")

    obj["reasoning"] = obj["reasoning"].strip()
    return obj

def score_resume_job(resume: dict, job: dict) -> dict:
    resume_text = resume.get("clean_text", "")
    job_text = f"{job.get('title','')}\n\n{job.get('description','')}"

    prompt = (
        "Compare the resume text to the job posting.\n"
        "Do NOT invent facts.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{ "score": 0, "matched_skills": [], "missing_skills": [], "reasoning": "" }\n'
        "Rules:\n"
        "- score is an integer 0-100\n"
        "- matched_skills/missing_skills are short skill phrases (strings)\n"
        "- reasoning is 1-2 sentences max\n"
        "\n"
        f"RESUME TEXT:\n{resume_text}\n\n"
        f"JOB POSTING:\n{job_text}\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You output only JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
    )

    result = json.loads(response.choices[0].message.content)
    return validate_result(result)

if __name__ == "__main__":
    resume = load_json(OUTPUT_RESUME_JSON)
    jobs = load_json(OUTPUT_JOBS_JSON)

    result = score_resume_job(resume, jobs[3])
    print(json.dumps(result, indent=2))