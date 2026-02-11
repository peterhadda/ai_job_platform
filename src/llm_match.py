# llm_match.py
import os
import json
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def build_prompt(resume_text: str, job: Dict[str, Any]) -> str:
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
- matched_skills and missing_skills are short skill phrases
- reasoning is 1-2 sentences max

RESUME TEXT:
{resume_text}

JOB TITLE: {job.get("title","")}
COMPANY: {job.get("company","")}

JOB DESCRIPTION:
{job.get("description","")}
""".strip()


def validate_llm_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    required = ["score", "verdict", "matched_skills", "missing_skills", "reasoning"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["score"], int) or not (0 <= obj["score"] <= 100):
        raise ValueError("score must be int 0..100")

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

    parsed = json.loads(resp.choices[0].message.content)
    return validate_llm_result(parsed)
