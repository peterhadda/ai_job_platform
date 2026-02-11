from openai import OpenAI
import json
from config import *

client = OpenAI(api_key="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")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_resume_job(resume_json: dict, job_json: dict):
    prompt1 = f"""
    You are a strict information engine.

    Compare the resume and job description.
    Return JSON only:

    {{
        "score": 0-100,
        "matched_skills": [],
        "missing_skills": [],
        "reasoning": "short explanation"
    }}

    Resume:
    {json.dumps(resume_json, indent=2)}

    Job:
    {json.dumps(job_json, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)



resume = load_json(OUTPUT_RESUME_JSON)
jobs = load_json(OUTPUT_JOBS_JSON)

result = score_resume_job(resume, jobs[3])

print(result)