import os
import json
import re
import random
from typing import Any, Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field, ValidationError
from openai import AzureOpenAI

# ---------------------------
# Azure OpenAI setup
# ---------------------------
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")
deployment = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")

if not endpoint or not key:
    raise RuntimeError("Missing OPENAI_ENDPOINT or OPENAI_SECRETKEY environment variables.")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version="2024-02-01"
)

HireLabel = Literal["Strong Hire", "Hire", "No Hire", "Strong No Hire"]
ModeLabel = Literal["partial", "final"]
SOLOLevel = Literal[0, 1, 2, 3, 4]

# ---------------------------
# Helpers
# ---------------------------

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
    """
    Used ONLY to write feedback text, not to decide scores.
    """
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
    if s < 10: return 2
    if s < 25: return 5
    if s < 40: return 10
    if s < 50: return 18
    if s < 60: return 30
    if s < 70: return 45
    if s < 80: return 65
    if s < 85: return 75
    if s < 92: return 88
    return 95

def _compute_pass_rate(tests_passed: Optional[int], tests_total: Optional[int]) -> Optional[float]:
    if tests_passed is None or tests_total in (None, 0):
        return None
    try:
        return max(0.0, min(1.0, float(tests_passed) / float(tests_total)))
    except Exception:
        return None

def compact_transcript(ts: List[Dict[str, str]], max_turns: int = 30) -> str:
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
        "consistency", "availability", "partition", "cap", "load balancer"
    ]
    return any(k in t for k in keywords)

def _looks_like_clarifying_questions(text: str) -> bool:
    t = (text or "").strip()
    if "?" in t:
        return True
    starters = ["clarify", "just to confirm", "can i assume", "what are the constraints", "edge cases"]
    tl = t.lower()
    return any(s in tl for s in starters)

def _first_quote_from_transcript(transcript: List[Dict[str, str]]) -> str:
    for t in transcript:
        if (t.get("role") or "").lower() == "user":
            content = (t.get("content") or "").strip()
            if content:
                return content[:160]
    return "N/A"

def _quote_from_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return "N/A"
    # pick a small excerpt
    lines = [ln.rstrip() for ln in c.splitlines() if ln.strip()]
    return ("\n".join(lines[:6]))[:220] if lines else c[:220]

def _na_feedback(student_id: str, aspect_name: str) -> str:
    return f"N/A — {student_id} didn’t provide enough evidence to assess {aspect_name} in this attempt."

# ---------------------------
# Schemas (kept compatible with your app)
# ---------------------------

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

# ---------------------------
# Context preparation (NO synthetic data)
# ---------------------------

def _prepare_context(input_data: Dict[str, Any]) -> Dict[str, Any]:
    student_id = str(input_data.get("student_id", "unknown"))
    question_id = str(input_data.get("question_id", "unknown"))

    interview_question = str(input_data.get("interview_question") or "")
    active_requirements = str(input_data.get("active_requirements") or interview_question)
    summary_of_past_response = str(input_data.get("summary_of_past_response") or "")
    user_input = str(input_data.get("user_input") or "")
    new_code_written = str(input_data.get("new_code_written") or "")

    transcript = input_data.get("transcript") or []
    if not isinstance(transcript, list):
        transcript = []

    candidate_code = (input_data.get("candidate_code") or new_code_written or "").strip()
    candidate_expl = (input_data.get("candidate_explanation") or user_input or "").strip()

    correctness_in = input_data.get("correctness_signals") or {}
    tests_passed = None
    tests_total = None
    major_failures: List[str] = []
    if isinstance(correctness_in, dict):
        tests_passed = correctness_in.get("tests_passed")
        tests_total = correctness_in.get("tests_total")
        major_failures = correctness_in.get("major_failures", []) or []
    pass_rate = _compute_pass_rate(tests_passed, tests_total)

    code_history_tail = input_data.get("candidate_code_history_tail") or []
    code_history_tail_text = ""
    if isinstance(code_history_tail, list) and code_history_tail:
        joined = []
        for i, c in enumerate(code_history_tail[-3:], start=1):
            c = (c or "").strip()
            if c:
                joined.append(f"--- Code Snapshot {i} ---\n{c}")
        code_history_tail_text = "\n\n".join(joined).strip()

    return {
        "student_id": student_id,
        "question_id": question_id,
        "interview_question": interview_question,
        "active_requirements": active_requirements,
        "summary_of_past_response": summary_of_past_response,
        "user_input": user_input,
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

# ---------------------------
# Deterministic scoring (rigorous + grounded)
# ---------------------------

def _bucket_from_0_25(score_0_25: int) -> str:
    s100 = int(max(0, min(100, round(score_0_25 * 4))))
    return _label_from_score(s100)

def _communication_score(user_words: int, latest_user: str) -> int:
    if user_words < 5:
        return 0
    # length-based baseline (conservative)
    if user_words < 15: base = 4
    elif user_words < 40: base = 8
    elif user_words < 80: base = 12
    elif user_words < 150: base = 16
    else: base = 19

    if _looks_like_clarifying_questions(latest_user):
        base += 3
    if "time complexity" in (latest_user or "").lower() or "space complexity" in (latest_user or "").lower():
        base += 2

    return int(max(0, min(25, base)))

def _problem_solving_score(user_turns: int, user_words: int, latest_user: str, candidate_expl: str) -> int:
    if user_turns < 2 and user_words < 40:
        return 0
    t = (latest_user + " " + (candidate_expl or "")).lower()
    score = 3
    if _looks_like_problem_solving(t):
        score = 10
    if "edge case" in t or "constraints" in t:
        score += 4
    if "step" in t or "first" in t or "then" in t:
        score += 3
    return int(max(0, min(25, score)))

def _technical_score(user_turns: int, user_words: int, latest_user: str, candidate_expl: str) -> int:
    if user_turns < 2 and user_words < 40:
        return 0
    t = (latest_user + " " + (candidate_expl or "")).lower()
    score = 3
    if _looks_like_technical_depth(t):
        score = 10
    if "tradeoff" in t or "latency" in t or "throughput" in t:
        score += 4
    if "big o" in t or "o(" in t or "complexity" in t:
        score += 3
    return int(max(0, min(25, score)))

def _code_score(code: str, pass_rate: Optional[float]) -> int:
    c = (code or "").strip()
    if len(c) < 20:
        return 0

    # size baseline (still conservative)
    if len(c) < 80: score = 6
    elif len(c) < 200: score = 10
    elif len(c) < 500: score = 14
    else: score = 16

    # If we have objective pass_rate, override upwards/downwards
    if pass_rate is not None:
        pr = float(pass_rate)
        if pr >= 0.95: score = max(score, 23)
        elif pr >= 0.85: score = max(score, 20)
        elif pr >= 0.60: score = max(score, 14)
        elif pr >= 0.30: score = min(score, 10)
        else: score = min(score, 6)

    return int(max(0, min(25, score)))

def _solo_level(latest_user: str, code: str, expl: str) -> Tuple[int, str]:
    has_ps = _looks_like_problem_solving(latest_user) or _looks_like_problem_solving(expl)
    has_tech = _looks_like_technical_depth(latest_user) or _looks_like_technical_depth(expl)
    has_code = len((code or "").strip()) >= 20

    if not has_ps and not has_code and not has_tech:
        return 0, "Prestructural: no coherent approach demonstrated in the recorded interaction."
    if has_ps and not has_code:
        return 2, "Multistructural: mentioned relevant pieces of an approach, but did not implement/validate."
    if has_ps and has_code and not has_tech:
        return 3, "Relational: coherent approach + implementation, but limited explicit tradeoff/technical depth."
    if has_ps and has_code and has_tech:
        return 4, "Extended Abstract: integrated approach, implementation, and tradeoffs/technical reasoning."
    return 1, "Unistructural: one relevant idea, but big gaps in development or validation."

def _deterministic_scores(ctx: Dict[str, Any]) -> Dict[str, Any]:
    transcript = ctx.get("transcript") or []
    user_words = _count_user_words(transcript)
    user_turns = _count_user_turns(transcript)
    latest_user = _extract_latest_user_text(transcript)
    code = ctx.get("candidate_code") or ""
    expl = ctx.get("candidate_explanation") or ""

    comm = _communication_score(user_words, latest_user)
    ps = _problem_solving_score(user_turns, user_words, latest_user, expl)
    tech = _technical_score(user_turns, user_words, latest_user, expl)
    code_score = _code_score(code, ctx.get("pass_rate"))

    total = int(max(0, min(100, comm + ps + tech + code_score)))
    solo, solo_just = _solo_level(latest_user, code, expl)

    return {
        "category_scores": {
            "communication": comm,
            "problem_solving": ps,
            "technical_competency": tech,
            "code_implementation": code_score
        },
        "total_score_0_100": total,
        "overall_assessment": _label_from_score(total),
        "hire_likelihood_percent": _hire_likelihood_percent(total),
        "solo_level": solo,
        "solo_justification": solo_just
    }

# ---------------------------
# Feedback generation (LLM optional but grounded + validated)
# ---------------------------

def _quotes_exist(quote: str, ctx: Dict[str, Any]) -> bool:
    q = (quote or "").strip()
    if not q or q == "N/A":
        return False
    blob = (ctx.get("transcript_text") or "") + "\n\n" + (ctx.get("candidate_code") or "")
    return q in blob

def _build_fallback_feedback(ctx: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    transcript = ctx.get("transcript") or []
    latest_user = _extract_latest_user_text(transcript)
    first_user_quote = _first_quote_from_transcript(transcript)
    code_quote = _quote_from_code(ctx.get("candidate_code") or "")

    comm = scores["category_scores"]["communication"]
    ps = scores["category_scores"]["problem_solving"]
    tech = scores["category_scores"]["technical_competency"]

    strengths = []
    improvements = []
    recs = []

    if comm > 0:
        strengths.append("You communicated at least one clear thought in the interview.")
    else:
        improvements.append("There wasn’t enough explanation in your messages to assess communication.")

    if ps > 0:
        strengths.append("You showed some problem-solving intent (approach/steps/constraints).")
    else:
        improvements.append("No clear approach was stated, so problem solving couldn’t really be evaluated.")

    if tech > 0:
        strengths.append("You referenced at least one technical concept or tradeoff.")
    else:
        improvements.append("No technical depth/tradeoffs were demonstrated in the transcript.")

    if (ctx.get("candidate_code") or "").strip():
        strengths.append("You provided code that can be assessed at a basic level.")
        recs.append("Add 2–3 small test cases and walk through one example out loud.")
    else:
        improvements.append("No meaningful code was provided, so implementation can’t be evaluated.")
        recs.append("Write a minimal correct implementation first, then improve edge cases and complexity.")

    recs.append("Ask 1–2 clarifying questions before committing to assumptions.")
    recs.append("State time and space complexity once your approach is set.")

    strengths = strengths[:3] if strengths else ["N/A — limited evidence beyond basic participation."]
    improvements = improvements[:3] if improvements else ["Keep going: add more reasoning and validation."]
    recs = recs[:3] if recs else ["Provide a fuller attempt (approach + code) for deeper evaluation."]

    # Legacy schema wants strings per category
    final_eval = {
        "communication": _bucket_from_0_25(scores["category_scores"]["communication"]),
        "problem_solving": _bucket_from_0_25(scores["category_scores"]["problem_solving"]),
        "technical_competency": _bucket_from_0_25(scores["category_scores"]["technical_competency"]),
        "examples_of_what_went_well": "; ".join(strengths)
    }

    detailed = {
        "communication": f"Evidence: {first_user_quote}",
        "problem_solving": f"Evidence: {latest_user[:200] if latest_user else 'N/A'}",
        "technical_competency": f"Evidence: {latest_user[:200] if latest_user else 'N/A'}",
        "examples_of_what_went_well": f"Code excerpt (if any):\n{code_quote}"
    }

    return {"strengths": strengths, "improvements": improvements, "recommendations": recs, "final_eval": final_eval, "detailed": detailed}

def _llm_grounded_feedback(ctx: Dict[str, Any], scores: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    LLM writes feedback, but must quote text that exists in transcript/code.
    If it hallucinates quotes, we discard and fallback.
    """
    system_msg = "You are a strict interview feedback writer. Use ONLY the transcript/code. Return ONLY valid JSON."
    prompt = f"""
Return ONLY valid JSON with this exact schema:

{{
  "strengths": ["...", "...", "..."],
  "improvements": ["...", "...", "..."],
  "recommendations": ["...", "...", "..."],
  "evidence_quotes": {{
    "communication": "...",
    "problem_solving": "...",
    "technical_competency": "...",
    "code_implementation": "..."
  }}
}}

Rules:
- Every evidence_quotes value MUST be an exact substring from the provided transcript or code.
- If you cannot find a quote for a category, write "N/A".
- Do NOT mention anything not present in transcript/code.

Inputs:
Active Requirements:
{ctx["active_requirements"]}

Transcript:
{ctx["transcript_text"]}

Candidate code:
{ctx["candidate_code"]}

Deterministic scores (do not change these):
{json.dumps(scores, indent=2)}
""".strip()

    raw = _call_llm_json([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ])

    parsed = _safe_json_loads(raw)

    # Validate quotes exist
    eq = parsed.get("evidence_quotes", {}) or {}
    for k in ["communication", "problem_solving", "technical_competency", "code_implementation"]:
        q = (eq.get(k) or "").strip()
        if q != "N/A" and not _quotes_exist(q, ctx):
            return None

    return parsed

# ---------------------------
# Main evaluators
# ---------------------------

def evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    input_data = (state.get("input") or [])[-1] if state.get("input") else {}
    ctx = _prepare_context(input_data)

    # Deterministic scoring (ground truth)
    scores = _deterministic_scores(ctx)

    # Feedback: try LLM grounded, else fallback
    llm_fb = _llm_grounded_feedback(ctx, scores)
    fallback = _build_fallback_feedback(ctx, scores)

    strengths = fallback["strengths"]
    improvements = fallback["improvements"]
    recs = fallback["recommendations"]
    evidence_quotes = {
        "communication": "N/A",
        "problem_solving": "N/A",
        "technical_competency": "N/A",
        "code_implementation": "N/A"
    }

    if llm_fb:
        strengths = (llm_fb.get("strengths") or strengths)[:3]
        improvements = (llm_fb.get("improvements") or improvements)[:3]
        recs = (llm_fb.get("recommendations") or recs)[:3]
        evidence_quotes = llm_fb.get("evidence_quotes") or evidence_quotes

    # Build legacy schema outputs (what your frontend expects)
    final_eval = {
        "communication": _bucket_from_0_25(scores["category_scores"]["communication"]),
        "problem_solving": _bucket_from_0_25(scores["category_scores"]["problem_solving"]),
        "technical_competency": _bucket_from_0_25(scores["category_scores"]["technical_competency"]),
        "examples_of_what_went_well": "; ".join(strengths)
    }

    detailed_feedback = {
        "communication": f"{improvements[0] if improvements else ''} Evidence: {evidence_quotes.get('communication','N/A')}",
        "problem_solving": f"{improvements[1] if len(improvements) > 1 else ''} Evidence: {evidence_quotes.get('problem_solving','N/A')}",
        "technical_competency": f"{improvements[2] if len(improvements) > 2 else ''} Evidence: {evidence_quotes.get('technical_competency','N/A')}",
        "examples_of_what_went_well": f"Next steps: {', '.join(recs)}\nEvidence (code): {evidence_quotes.get('code_implementation','N/A')}"
    }

    legacy = {
        "student_id": ctx["student_id"],
        "question_id": ctx["question_id"],
        "final_evaluation": final_eval,
        "detailed_feedback": detailed_feedback,
        "total_score_0_100": scores["total_score_0_100"],
        "overall_assessment": scores["overall_assessment"],
        "hire_likelihood_percent": scores["hire_likelihood_percent"]
    }

    obj = EvaluationSchemaLegacy(**legacy)
    state["evaluation_result"] = _model_dump(obj)

    # Optional: store richer internals if you want later
    state["combined_evaluation_result"] = {
        "candidate": {
            "student_id": ctx["student_id"],
            "question_id": ctx["question_id"],
            "solo_level": scores["solo_level"],
            "solo_justification": scores["solo_justification"],
            "category_scores": scores["category_scores"],
            "total_score_0_100": scores["total_score_0_100"],
            "overall_assessment": scores["overall_assessment"],
            "hire_likelihood_percent": scores["hire_likelihood_percent"],
            "major_failures": ctx.get("major_failures") or [],
            "pass_rate": ctx.get("pass_rate")
        }
    }

    return state


def partial_evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    input_data = (state.get("input") or [])[-1] if state.get("input") else {}
    ctx = _prepare_context(input_data)

    scores = _deterministic_scores(ctx)
    cat = scores["category_scores"]
    total = scores["total_score_0_100"]

    # Category feedback (grounded)
    transcript = ctx.get("transcript") or []
    latest_user = _extract_latest_user_text(transcript)
    first_user_quote = _first_quote_from_transcript(transcript)

    def fb_for(score_0_25: int, label: str) -> str:
        if score_0_25 == 0:
            return _na_feedback(ctx["student_id"], label)
        return f"Some evidence shown. Example: {first_user_quote}"

    parsed = {
        "student_id": ctx["student_id"],
        "question_id": ctx["question_id"],
        "category_scores": {
            "communication": cat["communication"],
            "problem_solving": cat["problem_solving"],
            "technical_competency": cat["technical_competency"],
            "code_implementation": cat["code_implementation"]
        },
        "total_score": total,
        "overall_assessment": scores["overall_assessment"],
        "hire_likelihood_percent": scores["hire_likelihood_percent"],
        "category_feedback": {
            "communication": fb_for(cat["communication"], "communication"),
            "problem_solving": fb_for(cat["problem_solving"], "problem solving"),
            "technical_competency": fb_for(cat["technical_competency"], "technical competency"),
            "code_implementation": fb_for(cat["code_implementation"], "code implementation")
        },
        "detailed_feedback": {
            "strengths": " | ".join([_first_quote_from_transcript(transcript)]) if transcript else "N/A",
            "areas_for_improvement": "Add a clearer approach + validation to enable deeper scoring.",
            "specific_recommendations": "Ask clarifying questions, state complexity, and write a minimal working solution."
        }
    }

    obj = PartialEvaluationSchema(**parsed)
    state["partial_evaluation_result"] = _model_dump(obj)
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


# ---------------------------------------------------------------------------------------
# Video analysis: kept exactly as callable function name analyze_interview_video()
# (unchanged behavior; this is independent from the evaluation scoring fixes above)
# ---------------------------------------------------------------------------------------

def analyze_interview_video(video_path: str):
    """
    Main function to analyze a video for nervous habits.
    This section is intentionally isolated so it doesn’t affect evaluation scoring.
    """
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from collections import deque
        from typing import List as _List, Dict as _Dict

        class NervousHabitDetector:
            def __init__(self):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_hands = mp.solutions.hands
                self.mp_pose = mp.solutions.pose
                self.last_hand_pos = None

            def kalman_filter(self, prev_gaze, raw_gaze, alpha=0.2):
                if prev_gaze is None:
                    return raw_gaze
                return alpha * raw_gaze + (1 - alpha) * prev_gaze

            def _estimate_gaze(self, face_landmarks):
                left_iris = np.mean([face_landmarks.landmark[i].x for i in range(468, 474)])
                left_center = 0.5 * (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x)

                right_iris = np.mean([face_landmarks.landmark[i].x for i in range(474, 478)])
                right_center = 0.5 * (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x)

                return ((left_iris - left_center) + (right_iris - right_center)) * 100

            def _get_eye_center(self, face_landmarks, eye: str):
                if eye == 'left':
                    indices = [33, 133, 157, 158, 159, 160, 161, 173]
                else:
                    indices = [362, 263, 384, 385, 386, 387, 388, 466]

                xs = [face_landmarks.landmark[i].x for i in indices]
                ys = [face_landmarks.landmark[i].y for i in indices]
                return np.mean(xs), np.mean(ys)

            def _merge_similar_events(self, events: _List[_Dict]) -> _List[_Dict]:
                if not events:
                    return events
                events.sort(key=lambda x: x['start'])
                merged = []
                current = events[0]

                for event in events[1:]:
                    if (event['type'] == current['type'] and event['start'] - current['end'] < 2.0):
                        current['end'] = max(current['end'], event['end'])
                        if 'intensity' in current and 'intensity' in event:
                            current['intensity'] = max(current['intensity'], event['intensity'])
                    else:
                        merged.append(current)
                        current = event
                merged.append(current)
                return merged

            def _filter_weak_events(self, events: _List[_Dict], min_duration: float = 2.0) -> _List[_Dict]:
                return [e for e in events if e['end'] - e['start'] >= min_duration]

            def _generate_statistics(self, events: _List[_Dict], total_duration: float) -> _Dict:
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

            def _is_touching_face(self, hand_landmarks, face_landmarks, threshold: float = 0.05) -> bool:
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

            def _format_time(self, seconds: float) -> str:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes:02d}:{secs:02d}"

            def _generate_coaching_feedback(self, events: _List[_Dict]) -> _List[_Dict]:
                feedback = []
                event_types = {}
                for event in events:
                    event_types.setdefault(event['type'], []).append(event)

                coaching_map = {"eye_darting", "prolonged_look_away", "excessive_hand_movement", "frequent_self_touching"}
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

            def analyze_video(self, video_path: str) -> _Dict:
                cap = cv2.VideoCapture(video_path)
                reported_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fps = 30.0 if reported_fps > 60 or reported_fps <= 0 else reported_fps

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

                gaze_buffer = deque(maxlen=int(fps * 30))
                calibration_buffer = deque(maxlen=int(fps * 10))
                calibrated = False
                gaze_mean = 0
                gaze_std = 1
                prev_gaze = None
                last_eye_dart = 0
                looking_away_start = None

                hand_motion_buffer = deque(maxlen=int(fps * 1))
                touch_buffer = deque(maxlen=int(fps * 30))

                events = []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if (frame_count - 1) % 10 != 0:
                        continue

                    frame = cv2.resize(frame, (640, 480))
                    timestamp = frame_count / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    face_results = face_mesh.process(frame_rgb)
                    hand_results = hands.process(frame_rgb)

                    if face_results.multi_face_landmarks:
                        face_landmarks = face_results.multi_face_landmarks[0]

                        raw_gaze = self._estimate_gaze(face_landmarks)
                        smooth_gaze = self.kalman_filter(prev_gaze, raw_gaze)
                        prev_gaze = smooth_gaze

                        calibration_buffer.append(smooth_gaze)
                        if len(calibration_buffer) == calibration_buffer.maxlen:
                            gaze_mean = np.mean(calibration_buffer)
                            gaze_std = np.std(calibration_buffer) + 0.01
                            calibrated = True

                        gaze_buffer.append(smooth_gaze)

                        if calibrated and len(gaze_buffer) == gaze_buffer.maxlen:
                            darting_threshold = gaze_mean + 3 * gaze_std
                            away_threshold = gaze_mean + 2.5 * gaze_std
                            gaze_variance = np.var(list(gaze_buffer))

                            if gaze_variance > darting_threshold:
                                if timestamp - last_eye_dart > 5:
                                    events.append({
                                        'type': 'eye_darting',
                                        'start': timestamp,
                                        'end': timestamp + 2,
                                        'intensity': float(gaze_variance),
                                        'threshold': float(darting_threshold)
                                    })
                                    last_eye_dart = timestamp

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
                                        'intensity': float(abs(smooth_gaze - gaze_mean)),
                                        'threshold': float(away_threshold)
                                    })
                                    looking_away_start = None
                            else:
                                looking_away_start = None

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            wrist_pos = hand_landmarks.landmark[0]
                            current_pos = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z])

                            if self.last_hand_pos is not None:
                                motion = float(np.linalg.norm(current_pos - self.last_hand_pos))
                                hand_motion_buffer.append(motion)

                                if len(hand_motion_buffer) == hand_motion_buffer.maxlen:
                                    avg_motion = float(np.mean(hand_motion_buffer))
                                    if avg_motion > 0.02:
                                        events.append({
                                            'type': 'excessive_hand_movement',
                                            'start': timestamp - 1,
                                            'end': timestamp,
                                            'intensity': avg_motion
                                        })

                            self.last_hand_pos = current_pos

                            if face_results.multi_face_landmarks:
                                face_landmarks = face_results.multi_face_landmarks[0]
                                face_width = np.linalg.norm([
                                    face_landmarks.landmark[10].x - face_landmarks.landmark[152].x,
                                    face_landmarks.landmark[10].y - face_landmarks.landmark[152].y
                                ])
                                touch_threshold = 0.08 * face_width
                                if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
                                    touch_buffer.append(timestamp)
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

                cap.release()
                events = self._merge_similar_events(events)
                events = self._filter_weak_events(events, min_duration=2.0)

                feedback = self._generate_coaching_feedback(events)
                total_duration = frame_count / fps if fps > 0 else 0.0

                return {
                    'detected_habits': events,
                    'coaching_feedback': feedback,
                    'summary_stats': self._generate_statistics(events, total_duration)
                }

        detector = NervousHabitDetector()
        return detector.analyze_video(video_path)

    except Exception as e:
        return {
            'error': str(e),
            'detected_habits': [],
            'coaching_feedback': [],
            'summary_stats': {}
        }

