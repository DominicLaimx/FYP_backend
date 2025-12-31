import os
import json
import glob
import time
import random
import argparse
from typing import Dict, Any, List, Literal, Optional
from openai import AzureOpenAI

Persona = Literal["strong", "average", "weak"]

def make_client() -> AzureOpenAI:
    endpoint = os.getenv("OPENAI_ENDPOINT")
    key = os.getenv("OPENAI_SECRETKEY")
    if not endpoint or not key:
        raise RuntimeError("Missing OPENAI_ENDPOINT or OPENAI_SECRETKEY environment variables.")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version="2024-02-01")

def load_problem(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = (data.get("title") or os.path.splitext(os.path.basename(json_path))[0]).strip()
    difficulty = (data.get("difficulty") or "Unknown").strip()
    leetcode_url = (data.get("leetcode_url") or "").strip()

    summary = (data.get("summary") or "").strip()
    question = (data.get("question") or "").strip()
    example = (data.get("example") or "").strip()
    constraint = (data.get("constraint") or "").strip()
    followup = (data.get("followup") or "").strip()
    starter_code = (data.get("starter_code") or "").strip()

    full_prompt = f"""
Title: {title}
Difficulty: {difficulty}
LeetCode URL: {leetcode_url if leetcode_url else "N/A"}

Summary:
{summary if summary else "N/A"}

Problem:
{question if question else "N/A"}

Examples:
{example if example else "N/A"}

Constraints:
{constraint if constraint else "N/A"}

Follow-up:
{followup if followup else "None"}

Starter code (if any):
{starter_code if starter_code else "None"}
""".strip()

    return {
        "question_id": title,
        "title": title,
        "difficulty": difficulty,
        "leetcode_url": leetcode_url,
        "full_prompt": full_prompt,
        "summary": summary,
        "question": question,
        "example": example,
        "constraint": constraint,
        "followup": followup,
        "starter_code": starter_code,
    }

def safe_json_extract(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise

def call_json(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.4,
    max_retries: int = 4,
    retry_base_seconds: float = 1.5,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=temperature,
                )
            raw = resp.choices[0].message.content or ""
            return safe_json_extract(raw)
        except Exception as e:
            last_err = e
            sleep_s = retry_base_seconds * (2 ** attempt) + random.uniform(0, 0.35)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last_err}")

def generate_transcript(
    client: AzureOpenAI,
    deployment: str,
    problem: Dict[str, Any],
    persona: Persona,
    turns_min: int = 10,
    turns_max: int = 16,
) -> Dict[str, Any]:
    persona_rules = {
        "strong": (
            "Candidate is strong: asks 1–2 clarifying questions if needed, proposes an optimal approach, "
            "covers edge cases, explains time/space complexity, and writes correct clean Python."
        ),
        "average": (
            "Candidate is decent but imperfect: gets main idea but misses 1 important edge case OR has a small bug OR "
            "uses a slightly suboptimal approach (still plausible). Communication is okay but not crisp."
        ),
        "weak": (
            "Candidate struggles: vague approach, confusion about data structures/logic, poor complexity reasoning, "
            "code is incorrect or incomplete."
        ),
    }

    interviewer_style = """
Interviewer style rules:
- Ask clarifying question prompts, but do not provide the full solution.
- Use follow-up probes: constraints, complexity, edge cases, alternative approach.
- If candidate is stuck, give a small hint (not the full algorithm).
- Keep tone professional and realistic.
"""

    correctness_targets = {
        "strong": {"tests_passed": 10, "tests_total": 10, "failures": 0},
        "average": {"tests_passed": random.randint(7, 9), "tests_total": 10, "failures": random.randint(1, 2)},
        "weak": {"tests_passed": random.randint(0, 6), "tests_total": 10, "failures": random.randint(2, 3)},
    }
    tgt = correctness_targets[persona]

    system = (
        "You generate realistic technical interview transcripts for LeetCode problems. "
        "Output ONLY valid JSON. No markdown. No extra keys."
    )

    user = f"""
Return JSON with EXACT keys and types:

{{
  "question_id": "string",
  "title": "string",
  "difficulty": "string",
  "leetcode_url": "string",
  "persona": "strong|average|weak",
  "transcript": [{{"role":"interviewer|candidate","content":"string"}}],
  "candidate_code": "string",
  "candidate_explanation": "string",
  "correctness_signals": {{
      "tests_passed": {tgt["tests_passed"]},
      "tests_total": {tgt["tests_total"]},
      "major_failures": ["string", "..."]
  }}
}}

Transcript requirements:
- Total turns between {turns_min} and {turns_max}.
- Include these beats in order:
  1) interviewer presents problem + asks for approach
  2) candidate clarifies / restates
  3) candidate proposes approach
  4) interviewer asks complexity + edge cases
  5) interviewer asks at least one follow-up probe
  6) candidate writes code
  7) interviewer sanity-checks with an example input
- Do not use long verbatim copies of the prompt; paraphrase.
- Candidate_explanation must reflect reasoning depth consistent with persona.

Code requirements:
- Python, runnable, plain text (no markdown fences).
- For strong: correct and efficient for constraints.
- For average/weak: plausibly wrong/incomplete/suboptimal consistent with major_failures.
- Avoid external libraries beyond standard Python.

correctness_signals rules:
- major_failures must have exactly {tgt["failures"]} items.
- Failures should be realistic for this problem and consistent with the code and transcript.
- If failures=0, major_failures must be [].

Problem context:
{problem["full_prompt"]}

Persona behavior:
{persona_rules[persona]}

{interviewer_style}
"""

    out = call_json(
        client=client,
        deployment=deployment,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.45 if persona != "strong" else 0.35,
    )

    out["question_id"] = problem["question_id"]
    out["title"] = problem["title"]
    out["difficulty"] = problem["difficulty"]
    out["leetcode_url"] = problem["leetcode_url"]
    out["persona"] = persona

    ts = out.get("transcript", [])
    if not isinstance(ts, list):
        ts = []
    cleaned = []
    for t in ts:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or "").strip().lower()
        if role not in ("interviewer", "candidate"):
            role = "candidate" if role == "user" else "interviewer" if role == "assistant" else "candidate"
        content = (t.get("content") or "").strip()
        if content:
            cleaned.append({"role": role, "content": content})
    out["transcript"] = cleaned[: max(turns_max, len(cleaned))]

    cs = out.get("correctness_signals") or {}
    if not isinstance(cs, dict):
        cs = {}
    cs["tests_passed"] = int(tgt["tests_passed"])
    cs["tests_total"] = int(tgt["tests_total"])
    mf = cs.get("major_failures", [])
    if not isinstance(mf, list):
        mf = []
    mf = [str(x) for x in mf if str(x).strip()]
    if len(mf) > tgt["failures"]:
        mf = mf[: tgt["failures"]]
    while len(mf) < tgt["failures"]:
        mf.append("Unspecified failing case")
    cs["major_failures"] = mf if tgt["failures"] > 0 else []
    out["correctness_signals"] = cs

    out["candidate_code"] = (out.get("candidate_code") or "").strip()
    out["candidate_explanation"] = (out.get("candidate_explanation") or "").strip()

    if not out["transcript"]:
        out["transcript"] = [
            {"role": "interviewer", "content": f"Let's work on {problem['title']}. Walk me through your approach."},
            {"role": "candidate", "content": out["candidate_explanation"] or "I will attempt a solution."},
            {"role": "interviewer", "content": "What is the time/space complexity and key edge cases?"},
            {"role": "candidate", "content": "Not sure."},
        ]

    return out

def build_dataset(
    input_dir: str,
    output_path: str,
    personas: List[Persona],
    limit: Optional[int],
    seed: int,
    sleep_between: float,
):
    random.seed(seed)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not paths:
        raise RuntimeError(f"No .json files found in {input_dir}")
    if limit is not None:
        paths = paths[:limit]

    client = make_client()
    deployment = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, path in enumerate(paths, start=1):
            problem = load_problem(path)
            for persona in personas:
                record = generate_transcript(client, deployment, problem, persona)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                if sleep_between > 0:
                    time.sleep(sleep_between)
            print(f"[{idx}/{len(paths)}] Done: {problem['title']}")
    print(f"\nWrote: {output_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--personas", default="strong,average,weak")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep_between", type=float, default=0.0)
    return p.parse_args()

def main():
    args = parse_args()
    personas_raw = [x.strip().lower() for x in args.personas.split(",") if x.strip()]
    valid = {"strong", "average", "weak"}
    if any(p not in valid for p in personas_raw):
        raise ValueError(f"Invalid personas. Must be subset of {valid}. Got: {personas_raw}")
    personas: List[Persona] = [p for p in personas_raw]
    build_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        personas=personas,
        limit=args.limit,
        seed=args.seed,
        sleep_between=args.sleep_between,
    )

if __name__ == "__main__":
    main()
