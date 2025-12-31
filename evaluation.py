import json
from openai import AzureOpenAI
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
import os
# Initialize Azure OpenAI client
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")
SECRET_KEY = os.getenv("PWD_SECRET_KEY")
deployment = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")

if not endpoint or not key:
    raise RuntimeError("Missing OPENAI_ENDPOINT or OPENAI_SECRETKEY environment variables.")

HireLabel = Literal["Strong Hire", "Hire", "No Hire", "Strong No Hire"]
ModeLabel = Literal["partial", "final"]
SOLOLevel = Literal[0, 1, 2, 3, 4]

# Initialize Azure OpenAI client
client = AzureOpenAI(
  azure_endpoint=endpoint, 
  api_key=key,  
  api_version="2024-02-01"
)

def _model_dump(m: Any) -> Dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump()
    return m.dict()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _call_llm_json(messages: List[Dict[str, str]]) -> str:
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception:
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.2,
        )
    return (resp.choices[0].message.content or "").strip()


def _label_from_score(score: int) -> HireLabel:
    if score >= 85:
        return "Strong Hire"
    if score >= 70:
        return "Hire"
    if score >= 50:
        return "No Hire"
    return "Strong No Hire"


def _hire_likelihood_percent(score_0_100: int) -> int:
    s = max(0, min(100, int(score_0_100)))
    if s < 10:
        return 2
    if s < 25:
        return 5
    if s < 40:
        return 10
    if s < 50:
        return 18
    if s < 60:
        return 30
    if s < 70:
        return 45
    if s < 80:
        return 65
    if s < 85:
        return 75
    if s < 92:
        return 88
    return 95


def _compute_pass_rate(tests_passed: Optional[int], tests_total: Optional[int]) -> Optional[float]:
    if tests_passed is None or tests_total in (None, 0):
        return None
    try:
        return max(0.0, min(1.0, float(tests_passed) / float(tests_total)))
    except Exception:
        return None


def load_synthetic_record(question_id: str) -> Optional[Dict[str, Any]]:
    path = os.getenv("SYNTHETIC_DATASET_PATH")
    if not path or not os.path.exists(path):
        return None
    matches: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("question_id", "")) == str(question_id):
                    matches.append(rec)
    except Exception:
        return None
    if not matches:
        return None
    return random.choice(matches)


def compact_transcript(ts: List[Dict[str, str]], max_turns: int = 24) -> str:
    if not ts:
        return "NO_TRANSCRIPT_PROVIDED"
    ts = ts[-max_turns:]
    lines = []
    for t in ts:
        role = (t.get("role", "unknown") or "unknown").upper()
        content = (t.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _count_user_words(transcript: List[Dict[str, str]]) -> int:
    words = 0
    for t in transcript:
        if (t.get("role") or "").lower() == "user":
            content = (t.get("content") or "")
            words += len(re.findall(r"\b\w+\b", content))
    return words


def _count_user_turns(transcript: List[Dict[str, str]]) -> int:
    return sum(1 for t in transcript if (t.get("role") or "").lower() == "user")


def _extract_latest_user_text(transcript: List[Dict[str, str]]) -> str:
    for t in reversed(transcript):
        if (t.get("role") or "").lower() == "user":
            return (t.get("content") or "").strip()
    return ""


def _looks_like_problem_solving(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "approach", "idea", "algorithm", "greedy", "dp", "dynamic programming",
        "two pointers", "hash", "map", "set", "stack", "queue",
        "time complexity", "space complexity", "big o", "edge case",
        "iterate", "loop", "invariant", "sort", "binary search",
    ]
    return any(k in t for k in keywords)


def _looks_like_technical_depth(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "complexity", "o(", "amortized", "tradeoff", "throughput", "latency",
        "database", "index", "cache", "cdn", "sharding", "replication",
        "concurrency", "race condition", "locking", "api", "protocol",
    ]
    return any(k in t for k in keywords)


def _looks_like_clarifying_questions(text: str) -> bool:
    t = (text or "").strip()
    if "?" in t:
        return True
    starters = ["clarify", "just to confirm", "can i assume", "what are the constraints", "edge cases"]
    tl = t.lower()
    return any(s in tl for s in starters)


def _data_sufficiency(ctx: Dict[str, Any]) -> Dict[str, bool]:
    transcript: List[Dict[str, str]] = ctx.get("transcript") or []
    user_turns = _count_user_turns(transcript)
    user_words = _count_user_words(transcript)
    code = (ctx.get("candidate_code") or "").strip()

    latest_user = _extract_latest_user_text(transcript)
    has_problem_solving = _looks_like_problem_solving(latest_user) or _looks_like_problem_solving(ctx.get("candidate_explanation") or "")
    has_technical = _looks_like_technical_depth(latest_user) or _looks_like_technical_depth(ctx.get("candidate_explanation") or "")

    return {
        "has_any_interaction": user_turns >= 1 and user_words >= 5,
        "can_score_communication": user_turns >= 1 and user_words >= 5,
        "can_score_problem_solving": user_turns >= 2 and (has_problem_solving or user_words >= 60),
        "can_score_technical": user_turns >= 2 and (has_technical or user_words >= 60),
        "can_score_code": len(code) >= 20,
        "enough_for_full_llm_eval": user_turns >= 2 and user_words >= 40,
    }


def _na_feedback(student_id: str, aspect_name: str) -> str:
    return f"N/A — {student_id} didn’t show any substantial progress in {aspect_name} during this interaction."


class InterviewerRubricScores(BaseModel):
    relevance: int = Field(ge=0, le=5)
    clarity: int = Field(ge=0, le=5)
    difficulty_control: int = Field(ge=0, le=5)
    adaptivity: int = Field(ge=0, le=5)
    hint_calibration: int = Field(ge=0, le=5)
    faithfulness: int = Field(ge=0, le=5)
    professionalism_fairness: int = Field(ge=0, le=5)


class InterviewerEval(BaseModel):
    interviewer_id: str = "llm_interviewer"
    question_id: str
    scores: InterviewerRubricScores
    overall_score_0_100: int = Field(ge=0, le=100)
    key_evidence: List[str]
    actionable_fixes: List[str]


class CandidateCategoryScores(BaseModel):
    communication: int = Field(ge=0, le=25)
    problem_solving: int = Field(ge=0, le=25)
    technical_competency: int = Field(ge=0, le=25)
    code_implementation: int = Field(ge=0, le=25)


class CorrectnessSignals(BaseModel):
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    pass_rate: Optional[float] = None
    major_failures: List[str] = []


class CandidateEval(BaseModel):
    student_id: str
    question_id: str
    solo_level: SOLOLevel
    solo_justification: str
    correctness: CorrectnessSignals
    category_scores: CandidateCategoryScores
    total_score_0_100: int = Field(ge=0, le=100)
    overall_assessment: HireLabel
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_recommendations: List[str]
    evidence: List[str]
    hire_likelihood_percent: int = Field(ge=0, le=100)


class CombinedEvaluation(BaseModel):
    mode: ModeLabel
    interviewer: InterviewerEval
    candidate: CandidateEval


class EvaluationCategoryLegacy(BaseModel):
    communication: str
    problem_solving: str
    technical_competency: str
    examples_of_what_went_well: str


class EvaluationSchemaLegacy(BaseModel):
    student_id: str
    question_id: str
    final_evaluation: EvaluationCategoryLegacy
    detailed_feedback: EvaluationCategoryLegacy

    total_score_0_100: int = Field(ge=0, le=100)
    overall_assessment: HireLabel
    hire_likelihood_percent: int = Field(ge=0, le=100)


class PartialEvaluationSchema(BaseModel):
    student_id: str
    question_id: str
    category_scores: Dict[str, int]
    total_score: int
    overall_assessment: HireLabel
    hire_likelihood_percent: int = Field(ge=0, le=100)
    category_feedback: Dict[str, str]
    detailed_feedback: Dict[str, str]


def _score_to_bucket(score: int) -> str:
    if score >= 85:
        return "Strong Hire"
    if score >= 70:
        return "Hire"
    if score >= 50:
        return "No Hire"
    return "Strong No Hire"


def _derive_legacy_from_combined(combined: Dict[str, Any]) -> Dict[str, Any]:
    cand = combined.get("candidate", {}) or {}
    cat = (cand.get("category_scores", {}) or {})
    comm = int(cat.get("communication", 0))
    ps = int(cat.get("problem_solving", 0))
    tech = int(cat.get("technical_competency", 0))

    comm_lbl = _score_to_bucket(int(round(comm * 4)))
    ps_lbl = _score_to_bucket(int(round(ps * 4)))
    tech_lbl = _score_to_bucket(int(round(tech * 4)))

    student_id = str(cand.get("student_id", "unknown"))
    question_id = str(cand.get("question_id", "unknown"))

    total = int(cand.get("total_score_0_100", 0))
    overall = str(cand.get("overall_assessment", _label_from_score(total)))
    hire_like = int(cand.get("hire_likelihood_percent", _hire_likelihood_percent(total)))

    strengths = cand.get("strengths", []) or []
    areas = cand.get("areas_for_improvement", []) or []
    recs = cand.get("specific_recommendations", []) or []
    evidence = cand.get("evidence", []) or []

    examples = "; ".join([s for s in strengths][:2]).strip() or "N/A"

    comm_fb = " ".join(evidence[:2]).strip() or "N/A"
    ps_fb = " ".join((strengths[:1] + areas[:1])).strip() or "N/A"
    tech_fb = " ".join((strengths[1:2] + areas[1:2])).strip() or "N/A"
    examples_fb = " ".join(recs[:2]).strip() or "N/A"

    return {
        "student_id": student_id,
        "question_id": question_id,
        "final_evaluation": {
            "communication": comm_lbl,
            "problem_solving": ps_lbl,
            "technical_competency": tech_lbl,
            "examples_of_what_went_well": examples,
        },
        "detailed_feedback": {
            "communication": comm_fb,
            "problem_solving": ps_fb,
            "technical_competency": tech_fb,
            "examples_of_what_went_well": examples_fb,
        },
        "total_score_0_100": total,
        "overall_assessment": overall,
        "hire_likelihood_percent": hire_like,
    }


def _prepare_context(input_data: Dict[str, Any]) -> Dict[str, Any]:
    student_id = str(input_data.get("student_id", "unknown"))
    question_id = str(input_data.get("question_id", "unknown"))

    interview_question = str(input_data.get("interview_question") or "")
    active_requirements = str(input_data.get("active_requirements") or interview_question)
    summary_of_past_response = str(input_data.get("summary_of_past_response") or "")
    new_code_written = str(input_data.get("new_code_written") or "")

    transcript = input_data.get("transcript")
    candidate_code = input_data.get("candidate_code") or new_code_written
    candidate_expl = input_data.get("candidate_explanation") or input_data.get("user_input") or summary_of_past_response
    correctness_in = input_data.get("correctness_signals")

    code_history_tail = input_data.get("candidate_code_history_tail")
    if isinstance(code_history_tail, list):
        joined = []
        for i, c in enumerate(code_history_tail[-3:], start=1):
            c = (c or "").strip()
            if c:
                joined.append(f"--- Code Snapshot {i} ---\n{c}")
        code_history_tail_text = "\n\n".join(joined).strip()
    else:
        code_history_tail_text = ""

    allow_synth = os.getenv("ALLOW_SYNTHETIC_EVAL", "false").lower() == "true"

    if not transcript:
        if allow_synth:
            rec = load_synthetic_record(question_id)
            if rec:
                transcript = rec.get("transcript") or []
                candidate_code = candidate_code or rec.get("candidate_code", "")
                candidate_expl = candidate_expl or rec.get("candidate_explanation", "")
                correctness_in = correctness_in or rec.get("correctness_signals", {})
                if not interview_question:
                    interview_question = str(rec.get("title", "")) or interview_question
                if not active_requirements:
                    active_requirements = interview_question
            else:
                transcript = []
                correctness_in = correctness_in or {}
        else:
            transcript = []
            correctness_in = correctness_in or {}

    if not isinstance(transcript, list):
        transcript = []

    candidate_code = (candidate_code or "").strip()
    candidate_expl = (candidate_expl or "").strip()

    tests_passed = None
    tests_total = None
    major_failures: List[str] = []
    if isinstance(correctness_in, dict):
        tests_passed = correctness_in.get("tests_passed")
        tests_total = correctness_in.get("tests_total")
        major_failures = correctness_in.get("major_failures", []) or []
    pass_rate = _compute_pass_rate(tests_passed, tests_total)

    return {
        "student_id": student_id,
        "question_id": question_id,
        "interview_question": interview_question,
        "active_requirements": active_requirements,
        "summary_of_past_response": summary_of_past_response,
        "new_code_written": new_code_written,
        "transcript": transcript,
        "transcript_text": compact_transcript(transcript),
        "candidate_code": candidate_code,
        "candidate_code_history_tail_text": code_history_tail_text,
        "candidate_explanation": candidate_expl,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "pass_rate": pass_rate,
        "major_failures": major_failures,
    }


def _fallback_grounded_eval(ctx: Dict[str, Any], mode: ModeLabel) -> Dict[str, Any]:
    student_id = ctx["student_id"]
    question_id = ctx["question_id"]

    transcript: List[Dict[str, str]] = ctx.get("transcript") or []
    latest_user = _extract_latest_user_text(transcript)
    has_clarify = _looks_like_clarifying_questions(latest_user)
    has_ps = _looks_like_problem_solving(latest_user) or _looks_like_problem_solving(ctx.get("candidate_explanation") or "")
    has_tech = _looks_like_technical_depth(latest_user) or _looks_like_technical_depth(ctx.get("candidate_explanation") or "")
    has_code = len((ctx.get("candidate_code") or "").strip()) >= 20

    suff = _data_sufficiency(ctx)

    comm = 0
    if suff["can_score_communication"]:
        comm = 8 if has_clarify else 6
        comm = min(25, comm + min(6, max(0, (len(latest_user) // 60))))

    ps = 0
    if suff["can_score_problem_solving"]:
        ps = 10 if has_ps else 6

    tech = 0
    if suff["can_score_technical"]:
        tech = 10 if has_tech else 6

    code = 0
    if suff["can_score_code"]:
        code = 10 if has_code else 0

    if ctx.get("pass_rate") is not None and suff["can_score_code"]:
        pr = float(ctx["pass_rate"])
        if pr >= 0.9:
            code = max(code, 20)
        elif pr >= 0.6:
            code = max(code, 14)
        elif pr >= 0.3:
            code = max(code, 8)
        else:
            code = min(code, 6)

    total = int(max(0, min(100, comm + ps + tech + code)))
    overall = _label_from_score(total)
    hire_like = _hire_likelihood_percent(total)

    solo = 0
    solo_just = "Prestructural: insufficient evidence of a coherent approach in the recorded interaction."
    if has_clarify and not has_ps and not has_code:
        solo = 1
        solo_just = "Unistructural: asked clarifying question(s), but did not develop or validate a solution within the interaction."
    elif has_ps and not has_code:
        solo = 2
        solo_just = "Multistructural: mentioned multiple relevant elements, but the reasoning/code was not integrated or validated."
    elif has_ps and has_code:
        solo = 3
        solo_just = "Relational: showed an approach and some implementation, but depth/validation evidence is limited in the transcript."

    strengths: List[str] = []
    areas: List[str] = []
    recs: List[str] = []
    evidence: List[str] = []

    if suff["can_score_communication"]:
        if has_clarify:
            strengths.append("Asked at least one clarifying question before committing to assumptions.")
            evidence.append(f"Observed: {latest_user[:160]}".strip())
        else:
            areas.append("Communication lacked clarifying questions or a structured plan in the recorded interaction.")
            evidence.append("Observed: no clear clarifying question or structured plan in transcript.")

    if not suff["can_score_problem_solving"]:
        areas.append(_na_feedback(student_id, "problem solving"))
    else:
        if has_ps:
            strengths.append("Mentioned a plausible high-level approach.")
        else:
            areas.append("Did not articulate a concrete approach that could be evaluated.")

    if not suff["can_score_technical"]:
        areas.append(_na_feedback(student_id, "technical competency"))
    else:
        if has_tech:
            strengths.append("Referenced at least one relevant technical concept or tradeoff.")
        else:
            areas.append("Technical reasoning was not demonstrated clearly in the recorded interaction.")

    if not suff["can_score_code"]:
        areas.append(_na_feedback(student_id, "code implementation"))
        recs.append("Write a working function first, then add tests and handle edge cases.")
    else:
        if has_code:
            strengths.append("Provided some code that can be assessed.")
            recs.append("Add a few test cases and explain time/space complexity.")
        else:
            areas.append("No usable code was provided for evaluation.")
            recs.append("Start with a minimal correct implementation, then improve.")

    if not strengths:
        strengths = ["N/A — not enough evidence of strengths beyond basic participation."]
    if not recs:
        recs = ["Provide a complete solution attempt (approach + code) so deeper evaluation is possible."]

    interviewer_scores = {
        "relevance": 3,
        "clarity": 3,
        "difficulty_control": 3,
        "adaptivity": 2,
        "hint_calibration": 3,
        "faithfulness": 5,
        "professionalism_fairness": 4,
    }
    intr_sum = sum(int(interviewer_scores[k]) for k in interviewer_scores.keys())
    intr_overall = int(round((intr_sum / 35) * 100))

    return {
        "mode": mode,
        "interviewer": {
            "interviewer_id": "llm_interviewer",
            "question_id": question_id,
            "scores": interviewer_scores,
            "overall_score_0_100": intr_overall,
            "key_evidence": ["Evaluation limited due to insufficient recorded interaction; no synthetic fallback used."],
            "actionable_fixes": ["Ensure full transcript (user + bot) is passed into evaluation for grounded feedback."],
        },
        "candidate": {
            "student_id": student_id,
            "question_id": question_id,
            "solo_level": solo,
            "solo_justification": solo_just,
            "correctness": {
                "tests_passed": ctx.get("tests_passed"),
                "tests_total": ctx.get("tests_total"),
                "pass_rate": ctx.get("pass_rate"),
                "major_failures": ctx.get("major_failures") or [],
            },
            "category_scores": {
                "communication": comm,
                "problem_solving": ps,
                "technical_competency": tech,
                "code_implementation": code,
            },
            "total_score_0_100": total,
            "overall_assessment": overall,
            "strengths": strengths[:5],
            "areas_for_improvement": areas[:6],
            "specific_recommendations": recs[:6],
            "evidence": evidence[:6],
            "hire_likelihood_percent": hire_like,
        },
    }


def evaluate_combined(state: Dict[str, Any], mode: ModeLabel) -> Dict[str, Any]:
    input_data = (state.get("input") or [])[-1] if state.get("input") else {}
    ctx = _prepare_context(input_data)

    suff = _data_sufficiency(ctx)
    if not suff["enough_for_full_llm_eval"]:
        combined_out = _fallback_grounded_eval(ctx, mode)

        obj = CombinedEvaluation(**combined_out)
        combined_out = _model_dump(obj)

        legacy_obj = EvaluationSchemaLegacy(**_derive_legacy_from_combined(combined_out))
        legacy_out = _model_dump(legacy_obj)

        state["combined_evaluation_result"] = combined_out
        state["evaluation_result"] = legacy_out
        return state

    solo_guide = """
SOLO levels for candidate answer depth:
0 Prestructural: off-target, misconceptions, cannot form a relevant approach.
1 Unistructural: one relevant idea, big gaps; brittle understanding.
2 Multistructural: several relevant pieces but not integrated; misses key connections/edge cases.
3 Relational: coherent integrated reasoning; handles constraints/edge cases appropriately.
4 Extended Abstract: generalizes, compares alternatives, transfers insight; explains tradeoffs clearly.
""".strip()

    correctness_rules = """
Correctness rules:
- If objective test signals are provided, treat them as ground truth.
- Do not claim tests passed/failed unless given in correctness_signals.
""".strip()

    grounding_rules = """
Grounding rules (VERY IMPORTANT):
- Use ONLY evidence present in the provided transcript and code.
- Judge the candidate against Active Requirements.
- If a category cannot be validated from transcript/code, give it a low score and say "N/A" in feedback.
- Do NOT invent that the candidate designed components, wrote code, or discussed tradeoffs unless explicitly present.
- Evidence strings must reference something actually observed (short, attributable).
""".strip()

    system_msg = "You are a strict evaluation engine. Return ONLY valid JSON. Use the provided schema exactly."

    user_prompt = f"""
Return ONLY valid JSON matching this schema:

{{
  "mode": "{mode}",
  "interviewer": {{
    "interviewer_id": "llm_interviewer",
    "question_id": "{ctx["question_id"]}",
    "scores": {{
      "relevance": 0,
      "clarity": 0,
      "difficulty_control": 0,
      "adaptivity": 0,
      "hint_calibration": 0,
      "faithfulness": 0,
      "professionalism_fairness": 0
    }},
    "overall_score_0_100": 0,
    "key_evidence": ["..."],
    "actionable_fixes": ["..."]
  }},
  "candidate": {{
    "student_id": "{ctx["student_id"]}",
    "question_id": "{ctx["question_id"]}",
    "solo_level": 0,
    "solo_justification": "...",
    "correctness": {{
      "tests_passed": {ctx["tests_passed"] if ctx["tests_passed"] is not None else "null"},
      "tests_total": {ctx["tests_total"] if ctx["tests_total"] is not None else "null"},
      "pass_rate": {ctx["pass_rate"] if ctx["pass_rate"] is not None else "null"},
      "major_failures": {json.dumps(ctx["major_failures"])}
    }},
    "category_scores": {{
      "communication": 0,
      "problem_solving": 0,
      "technical_competency": 0,
      "code_implementation": 0
    }},
    "total_score_0_100": 0,
    "overall_assessment": "No Hire",
    "strengths": ["..."],
    "areas_for_improvement": ["..."],
    "specific_recommendations": ["..."],
    "evidence": ["..."],
    "hire_likelihood_percent": 0
  }}
}}

Context:
- Original Question (reference): {ctx["interview_question"]}
- Active Requirements (use this to judge correctness): {ctx["active_requirements"]}

Transcript (source of truth):
{ctx["transcript_text"]}

Candidate explanation (may be partial; do not over-trust):
{ctx["candidate_explanation"]}

Candidate code (latest snapshot):
{ctx["candidate_code"]}

Candidate code history tail:
{ctx["candidate_code_history_tail_text"]}

{solo_guide}

{correctness_rules}

{grounding_rules}

Scoring constraints:
- Category scores are integers 0-25 each and must sum to total_score_0_100 (0-100).
- overall_assessment mapping:
  Strong Hire: 85-100
  Hire: 70-84
  No Hire: 50-69
  Strong No Hire: 0-49
- hire_likelihood_percent must be consistent with total_score_0_100 (monotonic).

Return only JSON.
""".strip()

    raw = _call_llm_json(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]
    )

    state["output"] = [raw]
    state["eval_raw_output"] = raw

    try:
        parsed = _safe_json_loads(raw)

        cand = parsed.get("candidate", {}) or {}
        if "correctness" not in cand or not isinstance(cand.get("correctness"), dict):
            cand["correctness"] = {}
        cand["correctness"]["tests_passed"] = ctx["tests_passed"]
        cand["correctness"]["tests_total"] = ctx["tests_total"]
        cand["correctness"]["pass_rate"] = ctx["pass_rate"]
        cand["correctness"]["major_failures"] = ctx["major_failures"]

        cat = cand.get("category_scores", {}) or {}
        total = (
            int(cat.get("communication", 0)) +
            int(cat.get("problem_solving", 0)) +
            int(cat.get("technical_competency", 0)) +
            int(cat.get("code_implementation", 0))
        )
        total = max(0, min(100, int(total)))
        cand["total_score_0_100"] = total
        cand["overall_assessment"] = _label_from_score(total)
        cand["hire_likelihood_percent"] = _hire_likelihood_percent(total)
        parsed["candidate"] = cand

        intr = parsed.get("interviewer", {}) or {}
        scores = intr.get("scores", {}) or {}
        dims = ["relevance", "clarity", "difficulty_control", "adaptivity", "hint_calibration", "faithfulness", "professionalism_fairness"]
        raw_sum = sum(int(scores.get(d, 0)) for d in dims)
        intr["overall_score_0_100"] = int(round((raw_sum / 35) * 100))
        parsed["interviewer"] = intr

        obj = CombinedEvaluation(**parsed)
        combined_out = _model_dump(obj)

        legacy_obj = EvaluationSchemaLegacy(**_derive_legacy_from_combined(combined_out))
        legacy_out = _model_dump(legacy_obj)

        state["combined_evaluation_result"] = combined_out
        state["evaluation_result"] = legacy_out
        return state

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        state["combined_evaluation_result"] = {"error": f"Could not parse/validate combined evaluation JSON: {e}", "raw_output": raw}
        state["evaluation_result"] = {"error": f"Could not parse/validate legacy evaluation JSON: {e}", "raw_output": raw}
        return state


def evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_combined(state, mode="final")


def partial_evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    input_data = (state.get("input") or [])[-1] if state.get("input") else {}
    ctx = _prepare_context(input_data)

    suff = _data_sufficiency(ctx)
    if not suff["has_any_interaction"]:
        parsed = {
            "student_id": ctx["student_id"],
            "question_id": ctx["question_id"],
            "category_scores": {
                "communication": 0,
                "problem_solving": 0,
                "technical_competency": 0,
                "code_implementation": 0
            },
            "total_score": 0,
            "overall_assessment": "Strong No Hire",
            "hire_likelihood_percent": 2,
            "category_feedback": {
                "communication": _na_feedback(ctx["student_id"], "communication"),
                "problem_solving": _na_feedback(ctx["student_id"], "problem solving"),
                "technical_competency": _na_feedback(ctx["student_id"], "technical competency"),
                "code_implementation": _na_feedback(ctx["student_id"], "code implementation"),
            },
            "detailed_feedback": {
                "strengths": "N/A — not enough interaction to identify strengths.",
                "areas_for_improvement": "Provide a complete solution attempt (approach + code) to enable evaluation.",
                "specific_recommendations": "Start by clarifying constraints and then implement a minimal correct solution."
            }
        }
        obj = PartialEvaluationSchema(**parsed)
        state["partial_evaluation_result"] = _model_dump(obj)
        evaluate_combined(state, mode="partial")
        return state

    system_msg = "You are a strict evaluation engine. Return ONLY valid JSON. Use the provided schema exactly."
    prompt = f"""
Return ONLY valid JSON with this exact structure:

{{
  "student_id": "{ctx["student_id"]}",
  "question_id": "{ctx["question_id"]}",
  "category_scores": {{
    "communication": 0,
    "problem_solving": 0,
    "technical_competency": 0,
    "code_implementation": 0
  }},
  "total_score": 0,
  "overall_assessment": "No Hire",
  "hire_likelihood_percent": 0,
  "category_feedback": {{
    "communication": "...",
    "problem_solving": "...",
    "technical_competency": "...",
    "code_implementation": "..."
  }},
  "detailed_feedback": {{
    "strengths": "...",
    "areas_for_improvement": "...",
    "specific_recommendations": "..."
  }}
}}

Context:
- Original Question (reference): {ctx["interview_question"]}
- Active Requirements (use this to judge correctness): {ctx["active_requirements"]}

Transcript (source of truth):
{ctx["transcript_text"]}

Candidate code (latest snapshot):
{ctx["candidate_code"]}

Rules:
- ONLY use evidence present in transcript/code.
- Judge against Active Requirements.
- If a category cannot be validated, set low score and write "N/A — no substantial progress shown".
- Category scores are integers 0-25 each; total_score must equal their sum (0-100).
- overall_assessment mapping:
  Strong Hire: 85-100
  Hire: 70-84
  No Hire: 50-69
  Strong No Hire: 0-49

Return only JSON.
""".strip()

    raw = _call_llm_json(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
    )

    state["output"] = [raw]
    state["partial_raw_output"] = raw

    try:
        parsed = _safe_json_loads(raw)
        cat = parsed.get("category_scores", {}) or {}
        total = (
            int(cat.get("communication", 0)) +
            int(cat.get("problem_solving", 0)) +
            int(cat.get("technical_competency", 0)) +
            int(cat.get("code_implementation", 0))
        )
        total = max(0, min(100, int(total)))
        parsed["total_score"] = total
        parsed["overall_assessment"] = _label_from_score(total)
        parsed["hire_likelihood_percent"] = _hire_likelihood_percent(total)

        obj = PartialEvaluationSchema(**parsed)
        state["partial_evaluation_result"] = _model_dump(obj)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        state["partial_evaluation_result"] = {"error": f"Could not parse/validate partial evaluation JSON: {e}", "raw_output": raw}

    evaluate_combined(state, mode="partial")
    return state

# class EvaluationCategory(BaseModel):
#     communication: str
#     problem_solving: str
#     technical_competency: str
#     code_implementation: str
#     examples_of_what_went_well: str

# class EvaluationSchema(BaseModel):
#     student_id: str
#     question_id: str
#     final_evaluation: EvaluationCategory
#     detailed_feedback: EvaluationCategory
#     total_score:str
#     overall_assessment:str

# def evaluation_agent(state: dict) -> dict:
#     """
#     Calls GPT to output JSON matching the EvaluationSchema,
#     storing it in state["evaluation_result"]. 
#     This is an 'intermediate' grading each time the user responds.
#     """
#     input_data = state["input"][-1]

#     prompt = f"""
#         You are an AI evaluation agent for a coding interview.
#         Output valid JSON only matching this evaluation schema (no markdown, no extra keys):

#         EvaluationSchema:
#         - student_id (string)
#         - question_id (string)
#         - final_evaluation (EvaluationCategory):
#             * communication (string)
#             * problem_solving (string)
#             * technical_competency (string)
#             * code_implementation (string)
#             * examples_of_what_went_well (string)
#         - detailed_feedback (EvaluationCategory):
#             * communication (string)
#             * problem_solving (string)
#             * technical_competency (string)
#             * code_implementation (string)
#             * examples_of_what_went_well (string)
#         - total_score (string)
#         - overall_assessment (string)
            
#         - feedback and examples of what they said / coded well and what they could've done better
        
#         Context you have:
#         - Interview Question: {input_data["interview_question"]}
#         - User's Summary: {input_data["summary_of_past_response"]}
#         - User's Code: {input_data["new_code_written"]}

#         Score each category carefully:

#         COMMUNICATION (0-10):
#         • 9-10: Exceptional - Clear, structured thinking, explains trade-offs, asks great questions
#         • 7-8: Strong - Articulate, explains reasoning well, communicates effectively
#         • 5-6: Competent - Gets ideas across but could be clearer, some gaps in explanation
#         • 3-4: Developing - Unclear at times, struggles to articulate complex thoughts  
#         • 0-2: Poor - Hard to follow, minimal or confusing communication

#         PROBLEM SOLVING (0-10):
#         • 9-10: Exceptional - Optimal approach, considers all edge cases, elegant solution
#         • 7-8: Strong - Logical decomposition, good solution, handles most cases
#         • 5-6: Competent - Basic approach, working but not optimal, misses some edge cases
#         • 3-4: Developing - Flawed methodology, incomplete solution, poor decomposition
#         • 0-2: Poor - No systematic approach, cannot break down the problem

#         TECHNICAL COMPETENCY (0-10):
#         • 9-10: Exceptional - Mastery of concepts, deep understanding, applies advanced techniques
#         • 7-8: Strong - Solid technical knowledge, applies concepts correctly, minor gaps
#         • 5-6: Competent - Basic understanding, several technical issues, needs guidance
#         • 3-4: Developing - Significant technical gaps, fundamental misunderstandings
#         • 0-2: Poor - Lacks basic technical knowledge, cannot apply concepts

#         CODE IMPLEMENTATION (0-10):
#         • 9-10: Exceptional - Production-ready, clean, maintainable, follows best practices
#         • 7-8: Strong - Clean structure, readable, minor style issues
#         • 5-6: Competent - Works but needs refactoring, some quality issues
#         • 3-4: Developing - Poorly structured, hard to read, significant problems
#         • 0-2: Poor - Unreadable, buggy, or minimal code

#         CALCULATE TOTAL SCORE (0-40):
#         Add all category scores together.

#         DETERMINE OVERALL ASSESSMENT:
#         • Strong Hire: 34-40 points
#         • Hire: 28-33 points  
#         • No Hire: 20-27 points
#         • Strong No Hire: 0-19 points

#         For feedback, be specific and actionable. Mention concrete examples from their response.

#         Return only valid JSON, no other text.
#         """
#         # Instruct GPT to output valid JSON that matches our Pydantic schema, no extra keys
#     # Make sure the prompt includes the relevant info: question, summary, code
# #     prompt = f"""
# # You are an AI evaluation agent for a coding interview I need you to be extremely strict! 
# # Produce your answer as valid JSON ONLY, matching this schema exactly:

# # EvaluationSchema:
# # - student_id (string)
# # - question_id (string)
# # - final_evaluation (EvaluationCategory):
# #     * communication (string)
# #     * problem_solving (string)
# #     * technical_competency (string)
# #     * examples_of_what_went_well (string)
# # - detailed_feedback (EvaluationCategory):
# #     * communication (string)
# #     * problem_solving (string)
# #     * technical_competency (string)
# #     * examples_of_what_went_well (string)
# # - feedback and examples of what they said / coded well and what they could've done better

# # NO extra keys, no markdown.

# # Context you have:
# # - Interview Question: {input_data["interview_question"]}
# # - User's Summary: {input_data["summary_of_past_response"]}
# # - User's Code: {input_data["new_code_written"]}

# # Scoring categories:
# # - "Strong Hire", "Hire", "No Hire", "Strong No Hire"

# # Only output valid JSON, no code blocks, no quotes around keys besides JSON structure.
# #     """

# # {
#         # "student_id": "string",
#         # "question_id": "string",
#         # "category_scores": {
#         #     "communication": "integer 0-10",
#         #     "problem_solving": "integer 0-10", 
#         #     "technical_competency": "integer 0-10",
#         #     "code_implementation": "integer 0-10"
#         # },
#         # "total_score": "integer 0-40",
#         # "overall_assessment": "string: 'Strong Hire', 'Hire', 'No Hire', 'Strong No Hire'",
#         # "category_feedback": {
#         #     "communication": "string",
#         #     "problem_solving": "string",
#         #     "technical_competency": "string", 
#         #     "code_implementation": "string"
#         # },
#         # "detailed_feedback": {
#         #     "strengths": "string",
#         #     "areas_for_improvement": "string",
#         #     "specific_recommendations": "string"
#         # }
#         # }

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are an AI that outputs valid JSON for evaluation."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     raw_text = response.choices[0].message.content.strip()
#     state["output"] = [raw_text]  # store raw text from GPT in 'output' for reference

#     # Parse JSON
#     try:
        
#         parsed = json.loads(raw_text)
#         if "EvaluationSchema" in parsed:
#             inner_data = parsed["EvaluationSchema"]  # Extract the actual evaluation data
#         else:
#             inner_data = parsed
#         # Validate with Pydantic
#         evaluation_obj = EvaluationSchema(**inner_data)
#         state["evaluation_result"] = evaluation_obj.dict()
#     except (json.JSONDecodeError, ValueError) as e:
#         state["evaluation_result"] = {
#             "error": f"Could not parse JSON: {e}",
#             "raw_output": raw_text
#         }
#     return state

# def partial_evaluation_agent(state: dict) -> dict:
#     """
#     Runs on every user response, outputting JSON matching the same schema as the final evaluator.
#     Ensures student_id and question_id fields are included.
#     """
#     input_data = state["input"][-1]

#     # Retrieve these from input_data or set to "unknown" if not provided
#     student_id = input_data.get("student_id", "unknown")
#     question_id = input_data.get("question_id", "unknown")

#     # If you keep track of previous partial eval
#     previous_eval = input_data.get("previous_partial_eval", {})
#     prev_eval_text = json.dumps(previous_eval, indent=2) if previous_eval else "No previous partial evaluation"

#     prompt = f"""
# You are an AI partial evaluator for a coding interview. 
# Output valid JSON only, matching this schema exactly (no extra keys, no markdown):

# EvaluationSchema:
# - student_id (string)
# - question_id (string)
# - partial_eval (EvaluationCategory):
#     * communication (string)
#     * problem_solving (string)
#     * technical_competency (string)
#     * examples_of_what_went_well (string)
# - detailed_feedback (EvaluationCategory):
#     * communication (string)
#     * problem_solving (string)
#     * technical_competency (string)
#     * examples_of_what_went_well (string)

# Here is the previous partial evaluation (if any):
# {prev_eval_text}

# Context:
# - Student ID: {student_id}
# - Question ID: {question_id}
# - Question: {input_data["interview_question"]}
# - Summary: {input_data["summary_of_past_response"]}
# - Code: {input_data["new_code_written"]}

# Scoring: "Strong Hire", "Hire", "No Hire", "Strong No Hire".

# Return valid JSON only, including the student_id and question_id.
# """

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are an AI that outputs valid JSON for partial evaluation."},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     raw_text = response.choices[0].message.content.strip()
#     state["output"] = [raw_text]

#     # Parse JSON
#     try:
#         parsed = json.loads(raw_text)
#         # Validate with your existing Pydantic schema that requires these fields
#         evaluation_obj = EvaluationSchema(**parsed)
#         state["partial_evaluation_result"] = evaluation_obj.dict()
#     except (json.JSONDecodeError, ValueError) as e:
#         state["partial_evaluation_result"] = {
#             "error": f"Could not parse JSON: {e}",
#             "raw_output": raw_text
#         }

#     return state




{"4": 
{"student_id": "12345", 
 "question_id": "validate_parentheses", 
 "final_evaluation": 
    {"communication": "Hire", 
    "problem_solving": "Strong Hire", 
    "technical_competency": "Hire", 
    "examples_of_what_went_well": "The solution demonstrates clear understanding of stack-based algorithmic approaches, correct implementation of mappings for accurate parentheses validation, and concise code structure."}, 
    "detailed_feedback": 
        {"communication": "The user explained their thought process clearly and confidently declined an enhancement suggestion while providing reasoning behind their decision.", 
        "problem_solving": "The use of a dictionary for mapping closing brackets to their respective opening brackets and maintaining stack consistency shows excellent problem-solving ability. The user demonstrated an understanding of edge cases such as empty stack handling.",
        "technical_competency": "The code aligns closely with best practices for solving this type of problem. While the solution is efficient, it does not handle non-bracket characters, limiting its versatility.",
        "examples_of_what_went_well": "The implementation is correct, concise, and covers all valid input cases for bracket validation. The stack-based logic is efficient and executed correctly."
        }
    }
}




import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import List, Dict
from datetime import datetime

class NervousHabitDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
    
    def kalman_filter(self,prev_gaze, raw_gaze, alpha=0.2):
        """Kalman filter for smoothing gaze"""
        if prev_gaze is None:
            return raw_gaze
        return alpha * raw_gaze + (1 - alpha) * prev_gaze

    def _estimate_gaze(self, face_landmarks):
        """Estimate gaze direction from face landmarks"""
        left_iris = np.mean([face_landmarks.landmark[i].x for i in range(468, 474)])
        left_center = 0.5 * (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x)  # Outer + inner corners

        right_iris = np.mean([face_landmarks.landmark[i].x for i in range(474, 478)])
        right_center = 0.5 * (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x)

        return ((left_iris - left_center) + (right_iris - right_center)) * 100
    
    def _get_eye_center(self, face_landmarks, eye: str):
        """Get center of left or right eye"""
        if eye == 'left':
            indices = [33, 133, 157, 158, 159, 160, 161, 173]  # Left eye indices
        else:  
            indices = [362, 263, 384, 385, 386, 387, 388, 466]  # Right eye indices
        
        xs = [face_landmarks.landmark[i].x for i in indices]
        ys = [face_landmarks.landmark[i].y for i in indices]
        return np.mean(xs), np.mean(ys)
    
    def _merge_similar_events(self, events: List[Dict]) -> List[Dict]:
        """Merge events that are close in time and of same type"""
        if not events:
            return events
            
        events.sort(key=lambda x: x['start'])
        merged = []
        current = events[0]
        
        for event in events[1:]:
            if (event['type'] == current['type'] and 
                event['start'] - current['end'] < 2.0):  # Merge if gap < 2 seconds
                current['end'] = max(current['end'], event['end'])
                if 'intensity' in current and 'intensity' in event:
                    current['intensity'] = max(current['intensity'], event['intensity'])
            else:
                merged.append(current)
                current = event
        merged.append(current)
        return merged
    
    def _filter_weak_events(self, events: List[Dict], min_duration: float = 2.0) -> List[Dict]:
        """Remove events that are too brief to be meaningful"""
        return [e for e in events if e['end'] - e['start'] >= min_duration]
    
    def _generate_statistics(self, events: List[Dict], total_duration: float) -> Dict:
        """Generate summary statistics"""
        habit_counts = {}
        total_habit_time = 0
        
        for event in events:
            habit_type = event['type']
            habit_counts[habit_type] = habit_counts.get(habit_type, 0) + 1
            total_habit_time += (event['end'] - event['start'])
        
        return {
            'total_video_duration': f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}",
            'total_habits_detected': len(events),
            'habit_breakdown': habit_counts,
            'total_habit_time_seconds': round(total_habit_time, 1),
            'percentage_of_video_with_habits': f"{(total_habit_time / total_duration * 100):.1f}%"
        }
        
    def analyze_video(self, video_path: str) -> Dict:
        """Main analysis function"""
        cap = cv2.VideoCapture(video_path)
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if reported_fps > 60 or reported_fps <= 0:
            print(f"⚠️ Fixing unrealistic FPS: {reported_fps} → 30.0")
            fps = 30.0
        else:
            fps = reported_fps
        
        print(f"Video Info: {fps} FPS, {total_frames} total frames")
        print(f"Estimated duration: {total_frames/fps/60:.1f} minutes")
        
        face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Buffers for temporal analysis
        gaze_buffer = deque(maxlen=int(fps * 30))  # 30s for stable calibration
        calibration_buffer = deque(maxlen=int(fps * 10))  # First 10s for baseline
        calibrated = False
        gaze_mean = 0
        gaze_std = 1
        prev_gaze = None
        last_eye_dart = 0
        looking_away_start = None

        hand_motion_buffer = deque(maxlen=int(fps * 1))
        touch_buffer = deque(maxlen=int(fps * 30))  # 30-second memory
        
        # Results storage
        events = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if (frame_count - 1) % 10 != 0:  # Process 1 in every 3 frames
                continue
            
            # Also resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
                
            timestamp = frame_count / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            face_results = face_mesh.process(frame_rgb)
            hand_results = hands.process(frame_rgb)
            
            # 1. DETECT EYE DARTING & GAZE
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Get eye landmarks (MediaPipe indices: left eye 33, 133, right eye 362, 263)
                left_eye = self._get_eye_center(face_landmarks, 'left')
                right_eye = self._get_eye_center(face_landmarks, 'right')

                # Estimate gaze
                raw_gaze = self._estimate_gaze(face_landmarks)
    
                # Kalman filter smoothing
                smooth_gaze = self.kalman_filter(prev_gaze, raw_gaze)
                prev_gaze = smooth_gaze

                # CALIBRATION PHASE (first 10 seconds)
                calibration_buffer.append(smooth_gaze)
                if len(calibration_buffer) == calibration_buffer.maxlen:
                    gaze_mean = np.mean(calibration_buffer)
                    gaze_std = np.std(calibration_buffer) + 0.01  
                    calibrated = True

                gaze_buffer.append(smooth_gaze)
                
                # Only detect AFTER calibration
                if calibrated and len(gaze_buffer) == gaze_buffer.maxlen:
                    
                    darting_threshold = gaze_mean + 3 * gaze_std      
                    away_threshold = gaze_mean + 2.5 * gaze_std       
                    
                    gaze_variance = np.var(list(gaze_buffer))
                    
                    # Check for eye darting (3σ)
                    if gaze_variance > darting_threshold:
                        if timestamp - last_eye_dart > 5:
                            events.append({
                                'type': 'eye_darting',
                                'start': timestamp,
                                'end': timestamp + 2,
                                'intensity': gaze_variance,
                                'threshold': darting_threshold
                            })
                            last_eye_dart = timestamp
                    
                    # Check for looking away (2.5σ)
                    if abs(smooth_gaze - gaze_mean) > away_threshold:
                        if looking_away_start is None:
                            looking_away_start = timestamp
                        elif timestamp - looking_away_start > 2:
                            direction = 'right' if smooth_gaze > gaze_mean else 'left'
                            events.append({
                                'type': 'prolonged_look_away',
                                'start': looking_away_start,
                                'end': timestamp,
                                'direction': direction,
                                'intensity': abs(smooth_gaze - gaze_mean),
                                'threshold': away_threshold
                            })
                            looking_away_start = None
                    else:
                        looking_away_start = None
                
            
            # HAND MOVEMENTS & SELF-TOUCHING
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist_pos = hand_landmarks.landmark[0]  # Wrist landmark
                    current_pos = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z])
                    
                    if hasattr(self, 'last_hand_pos'):
                        motion = np.linalg.norm(current_pos - self.last_hand_pos)
                        hand_motion_buffer.append(motion)
                        
                        # Check for excessive movement
                        if len(hand_motion_buffer) == hand_motion_buffer.maxlen:
                            avg_motion = np.mean(hand_motion_buffer)
                            if avg_motion > 0.02:  # Threshold
                                events.append({
                                    'type': 'excessive_hand_movement',
                                    'start': timestamp - 1,
                                    'end': timestamp,
                                    'intensity': avg_motion
                                })
                    
                    self.last_hand_pos = current_pos
                    
                    # Check for self-touching
                    if face_results.multi_face_landmarks:
                        face_landmarks = face_results.multi_face_landmarks[0]
                        face_width = np.linalg.norm([
                            face_landmarks.landmark[10].x - face_landmarks.landmark[152].x,  # Nose-chin
                            face_landmarks.landmark[10].y - face_landmarks.landmark[152].y
                            ])
                        touch_threshold = 0.08 * face_width
                        if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
                            touch_buffer.append(timestamp)
                            
                            # Check for frequent touching
                            if len(touch_buffer) > 3:
                                recent_touches = [t for t in touch_buffer if timestamp - t < 60]
                                if len(recent_touches) >= 3:
                                    events.append({
                                        'type': 'frequent_self_touching',
                                        'start': recent_touches[0],
                                        'end': timestamp,
                                        'count': len(recent_touches)
                                    })
                                    touch_buffer.clear()
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            # frame_count += 1
        
        cap.release()
        
        # 3. POST-PROCESS: Merge adjacent events and filter noise
        events = self._merge_similar_events(events)
        events = self._filter_weak_events(events, min_duration=2.0)
        
        print(f"events: {events}")
        feedback = self._generate_coaching_feedback(events)
        
        return {
            'detected_habits': events,
            'coaching_feedback': feedback,
            'summary_stats': self._generate_statistics(events, frame_count / fps)
        }

    def _is_touching_face(self, hand_landmarks, face_landmarks, threshold: float = 0.05) -> bool:
        """Check if hand is touching face/neck area"""
        hand_idxs = [0, 1, 4, 8, 12, 16, 20]  
        face_idxs = [10, 152, 148, 176, 323, 454, 234]  
        
        hand_points = np.array([[hand_landmarks.landmark[i].x, 
                            hand_landmarks.landmark[i].y, 
                            hand_landmarks.landmark[i].z] for i in hand_idxs])
        face_points = np.array([[face_landmarks.landmark[i].x, 
                            face_landmarks.landmark[i].y, 
                            face_landmarks.landmark[i].z] for i in face_idxs])
        
        distances = np.linalg.norm(hand_points[:, None, :] - face_points[None, :, :], axis=-1)
        
        return np.any(distances < threshold)
    
    def _generate_coaching_feedback(self, events: List[Dict]) -> List[str]:
        """Generate actionable feedback from detected events"""
        feedback = []
        
        event_types = {}
        for event in events:
            event_type = event['type']
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        coaching_map = set(["eye_darting","prolonged_look_away","excessive_hand_movement","frequent_self_touching"])
        
        for habit_type, habit_events in event_types.items():
            if habit_type in coaching_map:
                total_duration = sum(e['end'] - e['start'] for e in habit_events)
                feedback.append({
                    'habit': habit_type.replace('_', ' ').title(),
                    'occurrences': len(habit_events),
                    'total_duration_seconds': round(total_duration, 1),
                    'peak_time': self._format_time(habit_events[0]['start']),

                })
        
        return feedback
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
import json

def analyze_interview_video(video_path: str):
    """
    Main function to analyze a video for nervous habits
    
    Args:
        video_path: Path to the video file (e.g., 'interviews/user123.mp4')
    
    Returns:
        Dictionary with analysis results
    """
    try:
        detector = NervousHabitDetector()
        
        print(f"Starting analysis of {video_path}...")
        results = detector.analyze_video(video_path)
        
        print(f"\nAnalysis complete!")
        print(f"Total habits detected: {len(results['detected_habits'])}")
        print(f"Video duration: {results['summary_stats']['total_video_duration']}")
        
        return results
        
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return {
            'error': str(e),
            'detected_habits': [],
            'coaching_feedback': [],
            'summary_stats': {}
        }
# def analyze_interview_video(video_path: str):
#     return "blank"
