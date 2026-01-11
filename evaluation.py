import os
import json
import re
from typing import Any, Dict, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from openai import AzureOpenAI

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
            temperature=0.0,
        )
    except Exception:
        resp = client.chat_completions.create(
            model=deployment,
            messages=messages,
            temperature=0.0,
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
    lines = [ln.rstrip("\n") for ln in c.splitlines() if ln.strip()]
    return ("\n".join(lines[:6]))[:220] if lines else c[:220]

def _na_feedback(student_id: str, aspect_name: str) -> str:
    return f"N/A — {student_id} didn’t provide enough evidence to assess {aspect_name} in this attempt."

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _contains_quote(quote: str, blob: str) -> bool:
    q = (quote or "").strip()
    if not q or q == "N/A":
        return False
    if q in (blob or ""):
        return True
    return _normalize_ws(q) in _normalize_ws(blob or "")

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

def _bucket_from_0_25(score_0_25: int) -> str:
    s100 = int(max(0, min(100, round(score_0_25 * 4))))
    return _label_from_score(s100)

def _score_hard_gates(ctx: Dict[str, Any]) -> Dict[str, Any]:
    transcript = ctx.get("transcript") or []
    user_turns = _count_user_turns(transcript)
    user_words = _count_user_words(transcript)
    code = (ctx.get("candidate_code") or "").strip()
    expl = (ctx.get("candidate_explanation") or "").strip()
    has_any_code = len(code) >= 20
    has_any_expl = len(re.findall(r"\b\w+\b", expl)) >= 15
    has_any_interview = user_turns >= 1 and user_words >= 5
    if not has_any_code and not has_any_expl and not has_any_interview:
        return {
            "forced": True,
            "category_scores": {
                "communication": 0,
                "problem_solving": 0,
                "technical_competency": 0,
                "code_implementation": 0,
            },
            "cap_total": 0,
            "reason": "No assessable evidence provided (no meaningful transcript, explanation, or code).",
        }
    if user_turns <= 1 and user_words < 25 and not has_any_code and not has_any_expl:
        return {
            "forced": True,
            "category_scores": {
                "communication": 2 if user_words >= 5 else 0,
                "problem_solving": 0,
                "technical_competency": 0,
                "code_implementation": 0,
            },
            "cap_total": 10,
            "reason": "Too little evidence; only a minimal interaction with no code/explanation.",
        }
    cap_total = 35
    if has_any_code or has_any_expl:
        cap_total = 55
    if ctx.get("pass_rate") is not None and float(ctx["pass_rate"]) >= 0.85:
        cap_total = 80
    return {
        "forced": False,
        "category_scores": {},
        "cap_total": cap_total,
        "reason": "Gating applied.",
    }

def _llm_rubric_scoring(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    system_msg = (
        "You are a strict technical interview evaluator. "
        "Use ONLY the provided transcript and code. "
        "Do NOT assume missing steps. Do NOT fabricate. "
        "Return ONLY valid JSON."
    )
    prompt = f"""
Return ONLY valid JSON with EXACTLY this schema:
{{
  "category_scores": {{
    "communication": 0-25,
    "problem_solving": 0-25,
    "technical_competency": 0-25,
    "code_implementation": 0-25
  }},
  "rationales": {{
    "communication": "...",
    "problem_solving": "...",
    "technical_competency": "...",
    "code_implementation": "..."
  }},
  "evidence_quotes": {{
    "communication": ["...", "..."],
    "problem_solving": ["...", "..."],
    "technical_competency": ["...", "..."],
    "code_implementation": ["...", "..."]
  }}
}}
Hard rules:
- Evidence quotes MUST be copied from the transcript/code as exact substrings.
- If you cannot find evidence for a category, use an empty list [] and score low (0-3).
- Never reward effort that is not visible in the transcript/code.
- Use tests/pass-rate ONLY if explicitly provided below; do not assume tests exist otherwise.
- Keep evidence_quotes short (each quote <= 160 chars). Prefer 1–3 quotes per category.
Rubric anchors (0–25):
COMMUNICATION:
- 0–3: no explanation / cannot follow reasoning
- 4–9: some clarity but fragmented
- 10–16: clear enough to follow, some structure
- 17–22: structured, anticipates questions, communicates constraints/tradeoffs
- 23–25: exceptionally clear, concise, organized under pressure
PROBLEM_SOLVING:
- 0–3: no approach stated / random guessing
- 4–9: partial approach, gaps, little validation
- 10–16: coherent approach, mentions cases/steps
- 17–22: strong decomposition, validates with examples/edge cases
- 23–25: excellent reasoning, robust validation, correct algorithmic choices
TECHNICAL_COMPETENCY:
- 0–3: no technical reasoning visible
- 4–9: basic concepts, some inaccuracies/omissions
- 10–16: correct core concepts, minor gaps
- 17–22: good depth and correctness, discusses complexity/tradeoffs well
- 23–25: deep, precise, anticipates pitfalls, strong engineering judgment
CODE_IMPLEMENTATION:
- If NO meaningful code is provided, score 0–3.
- 4–9: incomplete/buggy code, unclear structure
- 10–16: mostly correct implementation, readable
- 17–22: correct, clean, handles edge cases, good structure
- 23–25: production-quality clarity and robustness (within interview scope)
Inputs (ONLY SOURCE OF TRUTH):
Active requirements:
{ctx["active_requirements"]}
Transcript:
{ctx["transcript_text"]}
Candidate code:
{ctx["candidate_code"]}
Objective correctness signals (if any):
tests_passed={ctx.get("tests_passed")}
tests_total={ctx.get("tests_total")}
pass_rate={ctx.get("pass_rate")}
major_failures={json.dumps(ctx.get("major_failures") or [], ensure_ascii=False)}
""".strip()
    raw = _call_llm_json(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = _safe_json_loads(raw)
    if not isinstance(parsed, dict):
        return None
    cat = parsed.get("category_scores")
    eq = parsed.get("evidence_quotes")
    if not isinstance(cat, dict) or not isinstance(eq, dict):
        return None
    for k in ["communication", "problem_solving", "technical_competency", "code_implementation"]:
        if k not in cat:
            return None
        try:
            v = int(cat[k])
        except Exception:
            return None
        if v < 0 or v > 25:
            return None
        cat[k] = v
    blob = (ctx.get("transcript_text") or "") + "\n\n" + (ctx.get("candidate_code") or "")
    for k in ["communication", "problem_solving", "technical_competency", "code_implementation"]:
        quotes = eq.get(k)
        if quotes is None:
            return None
        if not isinstance(quotes, list):
            return None
        cleaned: List[str] = []
        for q in quotes[:5]:
            if not isinstance(q, str):
                continue
            q = q.strip()
            if not q or len(q) > 200:
                continue
            if _contains_quote(q, blob):
                cleaned.append(q)
            else:
                return None
        eq[k] = cleaned
    parsed["category_scores"] = cat
    parsed["evidence_quotes"] = eq
    return parsed

def _fallback_conservative_scoring(ctx: Dict[str, Any]) -> Dict[str, Any]:
    transcript = ctx.get("transcript") or []
    latest_user = _extract_latest_user_text(transcript)
    expl = (ctx.get("candidate_explanation") or "").strip()
    code = (ctx.get("candidate_code") or "").strip()
    pass_rate = ctx.get("pass_rate")
    comm = 0
    if len(re.findall(r"\b\w+\b", (latest_user + " " + expl))) >= 20:
        sentences = re.split(r"[.!?\n]+", (latest_user + " " + expl))
        comm = 8 if sum(1 for s in sentences if s.strip()) >= 2 else 5
    ps = 0
    if len(re.findall(r"\b\w+\b", expl)) >= 25:
        ps = 8
    if len(code) >= 20:
        ps = max(ps, 6)
    tech = 0
    t = (latest_user + " " + expl).lower()
    if "o(" in t or "big o" in t or "complexity" in t:
        tech = 8
    code_score = 0
    if len(code) >= 20:
        code_score = 8
    if pass_rate is not None:
        pr = float(pass_rate)
        if pr >= 0.95:
            code_score = 23
        elif pr >= 0.85:
            code_score = 20
        elif pr >= 0.60:
            code_score = 14
        elif pr >= 0.30:
            code_score = 10
        else:
            code_score = 6
    return {
        "category_scores": {
            "communication": int(max(0, min(25, comm))),
            "problem_solving": int(max(0, min(25, ps))),
            "technical_competency": int(max(0, min(25, tech))),
            "code_implementation": int(max(0, min(25, code_score))),
        },
        "evidence_quotes": {
            "communication": [],
            "problem_solving": [],
            "technical_competency": [],
            "code_implementation": [],
        },
        "rationales": {
            "communication": "Fallback scoring used due to invalid/unverifiable LLM evidence quotes.",
            "problem_solving": "Fallback scoring used due to invalid/unverifiable LLM evidence quotes.",
            "technical_competency": "Fallback scoring used due to invalid/unverifiable LLM evidence quotes.",
            "code_implementation": "Fallback scoring used due to invalid/unverifiable LLM evidence quotes.",
        },
    }

def _solo_level(latest_user: str, code: str, expl: str) -> Tuple[int, str]:
    t = (latest_user + " " + (expl or "")).strip()
    has_expl = len(re.findall(r"\b\w+\b", t)) >= 25
    has_code = len((code or "").strip()) >= 20
    mentions_tradeoff = any(x in t.lower() for x in ["tradeoff", "latency", "throughput", "space", "time complexity", "big o", "o("])
    if not has_expl and not has_code:
        return 0, "Prestructural: no coherent approach demonstrated in the recorded interaction."
    if has_expl and not has_code:
        return 2, "Multistructural: some reasoning mentioned, but no implementation/validation."
    if has_expl and has_code and not mentions_tradeoff:
        return 3, "Relational: coherent reasoning and implementation, but limited explicit tradeoff/technical depth."
    if has_expl and has_code and mentions_tradeoff:
        return 4, "Extended Abstract: approach, implementation, and tradeoffs/complexity reasoning integrated."
    return 1, "Unistructural: a single relevant idea shown, but big gaps remain."

def _rigorous_scores(ctx: Dict[str, Any]) -> Dict[str, Any]:
    gates = _score_hard_gates(ctx)
    llm = _llm_rubric_scoring(ctx)
    scored = llm if llm is not None else _fallback_conservative_scoring(ctx)
    cat = scored["category_scores"]
    if gates.get("forced"):
        forced_cat = gates.get("category_scores") or {}
        for k, v in forced_cat.items():
            cat[k] = int(v)
        cap_total = int(gates.get("cap_total", 0))
    else:
        cap_total = int(gates.get("cap_total", 100))
    code = (ctx.get("candidate_code") or "").strip()
    if len(code) < 20:
        cat["code_implementation"] = min(cat["code_implementation"], 3)
    pr = ctx.get("pass_rate")
    total_quotes = 0
    if isinstance(scored.get("evidence_quotes"), dict):
        for v in scored["evidence_quotes"].values():
            if isinstance(v, list):
                total_quotes += len(v)
    if total_quotes == 0 and pr is None and _count_user_turns(ctx.get("transcript") or []) <= 1:
        cap_total = min(cap_total, 25)
    total = int(cat["communication"] + cat["problem_solving"] + cat["technical_competency"] + cat["code_implementation"])
    total = int(max(0, min(100, total, cap_total)))
    transcript = ctx.get("transcript") or []
    latest_user = _extract_latest_user_text(transcript)
    expl = ctx.get("candidate_explanation") or ""
    solo, solo_just = _solo_level(latest_user, code, expl)
    return {
        "category_scores": {
            "communication": int(cat["communication"]),
            "problem_solving": int(cat["problem_solving"]),
            "technical_competency": int(cat["technical_competency"]),
            "code_implementation": int(cat["code_implementation"]),
        },
        "total_score_0_100": total,
        "overall_assessment": _label_from_score(total),
        "hire_likelihood_percent": _hire_likelihood_percent(total),
        "solo_level": solo,
        "solo_justification": solo_just,
        "evidence_quotes": scored.get("evidence_quotes")
        or {
            "communication": [],
            "problem_solving": [],
            "technical_competency": [],
            "code_implementation": [],
        },
        "rationales": scored.get("rationales") or {},
    }

def _build_fallback_feedback(ctx: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    transcript = ctx.get("transcript") or []
    latest_user = _extract_latest_user_text(transcript)
    first_user_quote = _first_quote_from_transcript(transcript)
    code_quote = _quote_from_code(ctx.get("candidate_code") or "")
    comm = scores["category_scores"]["communication"]
    ps = scores["category_scores"]["problem_solving"]
    tech = scores["category_scores"]["technical_competency"]
    code_score = scores["category_scores"]["code_implementation"]
    strengths: List[str] = []
    improvements: List[str] = []
    recs: List[str] = []
    if comm >= 10:
        strengths.append("You explained your thinking in a way that was understandable at key moments.")
    elif comm > 0:
        improvements.append("Clarify your reasoning out loud, especially when changing direction.")
    else:
        improvements.append("There wasn’t enough explanation in your messages to assess communication.")
    if ps >= 10:
        strengths.append("You outlined a coherent approach and broke the problem into sensible steps.")
    elif ps > 0:
        improvements.append("The approach was partially stated but lacked a full step‑by‑step plan.")
    else:
        improvements.append("No clear end‑to‑end approach was described, so problem solving was hard to evaluate.")
    if tech >= 10:
        strengths.append("You demonstrated relevant technical concepts for this problem.")
    elif tech > 0:
        improvements.append("You referenced at least one technical concept, but the reasoning stayed shallow.")
    else:
        improvements.append("Technical depth and tradeoffs were not clearly discussed in the transcript.")
    if code_score >= 10:
        strengths.append("You provided enough code for a meaningful assessment of correctness.")
        recs.append("Add a few targeted test cases and walk through them line‑by‑line to validate behavior.")
    elif code_score > 0:
        improvements.append("The code was present but too limited or incomplete to assess thoroughly.")
        recs.append("Start from a minimal working solution, then iterate to handle edge cases.")
    else:
        improvements.append("No meaningful code was provided, so implementation quality could not be evaluated.")
        recs.append("Write a basic implementation, even if partial, to create something concrete to review.")
    recs.append("State your time and space complexity once your approach feels settled.")
    recs.append("Call out at least two edge cases and describe how your solution handles them.")
    strengths = strengths[:3] if strengths else ["N/A — limited evidence beyond basic participation."]
    improvements = improvements[:3]
    if not improvements:
        improvements = ["Tighten the explanation and validate with explicit examples before finalizing."]
    recs = recs[:3]
    final_eval = {
        "communication": _bucket_from_0_25(scores["category_scores"]["communication"]),
        "problem_solving": _bucket_from_0_25(scores["category_scores"]["problem_solving"]),
        "technical_competency": _bucket_from_0_25(scores["category_scores"]["technical_competency"]),
        "examples_of_what_went_well": "; ".join(strengths),
    }
    detailed = {
        "communication": f"{improvements[0] if improvements else ''} Evidence: {first_user_quote}",
        "problem_solving": f"{improvements[1] if len(improvements) > 1 else ''} Evidence: {latest_user[:200] if latest_user else 'N/A'}",
        "technical_competency": f"{improvements[2] if len(improvements) > 2 else ''} Evidence: {latest_user[:200] if latest_user else 'N/A'}",
        "examples_of_what_went_well": f"Next steps: {', '.join(recs)}\nEvidence (code): {code_quote}",
    }
    return {
        "strengths": strengths,
        "improvements": improvements,
        "recommendations": recs,
        "final_eval": final_eval,
        "detailed": detailed,
    }

def _llm_grounded_feedback(ctx: Dict[str, Any], scores: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    system_msg = "You are a strict interview feedback writer. Use ONLY the transcript and code. Return ONLY valid JSON."
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
Final scores (do not change these):
{json.dumps(scores, indent=2)}
""".strip()
    raw = _call_llm_json(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = _safe_json_loads(raw)
    eq = parsed.get("evidence_quotes", {}) or {}
    blob = (ctx.get("transcript_text") or "") + "\n\n" + (ctx.get("candidate_code") or "")
    for k in ["communication", "problem_solving", "technical_competency", "code_implementation"]:
        q = (eq.get(k) or "").strip()
        if q != "N/A" and not _contains_quote(q, blob):
            return None
    return parsed

def evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    input_data = (state.get("input") or [])[-1] if state.get("input") else {}
    ctx = _prepare_context(input_data)
    scores = _rigorous_scores(ctx)
    llm_fb = _llm_grounded_feedback(ctx, scores)
    fallback = _build_fallback_feedback(ctx, scores)
    strengths = fallback["strengths"]
    improvements = fallback["improvements"]
    recs = fallback["recommendations"]
    evidence_quotes = {
        "communication": "N/A",
        "problem_solving": "N/A",
        "technical_competency": "N/A",
        "code_implementation": "N/A",
    }
    if llm_fb:
        strengths = (llm_fb.get("strengths") or strengths)[:3]
        improvements = (llm_fb.get("improvements") or improvements)[:3]
        recs = (llm_fb.get("recommendations") or recs)[:3]
        evidence_quotes = llm_fb.get("evidence_quotes") or evidence_quotes
    final_eval = {
        "communication": _bucket_from_0_25(scores["category_scores"]["communication"]),
        "problem_solving": _bucket_from_0_25(scores["category_scores"]["problem_solving"]),
        "technical_competency": _bucket_from_0_25(scores["category_scores"]["technical_competency"]),
        "examples_of_what_went_well": "; ".join(strengths),
    }
    detailed_feedback = {
        "communication": f"{improvements[0] if improvements else ''} Evidence: {evidence_quotes.get('communication','N/A')}",
        "problem_solving": f"{improvements[1] if len(improvements) > 1 else ''} Evidence: {evidence_quotes.get('problem_solving','N/A')}",
        "technical_competency": f"{improvements[2] if len(improvements) > 2 else ''} Evidence: {evidence_quotes.get('technical_competency','N/A')}",
        "examples_of_what_went_well": f"Next steps: {', '.join(recs)}\nEvidence (code): {evidence_quotes.get('code_implementation','N/A')}",
    }
    legacy = {
        "student_id": ctx["student_id"],
        "question_id": ctx["question_id"],
        "final_evaluation": final_eval,
        "detailed_feedback": detailed_feedback,
        "total_score_0_100": scores["total_score_0_100"],
        "overall_assessment": scores["overall_assessment"],
        "hire_likelihood_percent": scores["hire_likelihood_percent"],
    }
    obj = EvaluationSchemaLegacy(**legacy)
    state["evaluation_result"] = _model_dump(obj)
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
            "pass_rate": ctx.get("pass_rate"),
            "evidence_quotes_debug": scores.get("evidence_quotes", {}),
            "rationales_debug": scores.get("rationales", {}),
        }
    }
    return state

def partial_evaluation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    input_data = (state.get("input") or [])[-1] if state.get("input") else {}
    ctx = _prepare_context(input_data)
    scores = _rigorous_scores(ctx)
    cat = scores["category_scores"]
    total = scores["total_score_0_100"]
    transcript = ctx.get("transcript") or []
    first_user_quote = _first_quote_from_transcript(transcript)

    def fb_for(score_0_25: int, label: str) -> str:
        if score_0_25 <= 3:
            return _na_feedback(ctx["student_id"], label)
        return f"Evidence exists in the transcript/code. Example: {first_user_quote}"

    parsed = {
        "student_id": ctx["student_id"],
        "question_id": ctx["question_id"],
        "category_scores": {
            "communication": cat["communication"],
            "problem_solving": cat["problem_solving"],
            "technical_competency": cat["technical_competency"],
            "code_implementation": cat["code_implementation"],
        },
        "total_score": total,
        "overall_assessment": scores["overall_assessment"],
        "hire_likelihood_percent": scores["hire_likelihood_percent"],
        "category_feedback": {
            "communication": fb_for(cat["communication"], "communication"),
            "problem_solving": fb_for(cat["problem_solving"], "problem solving"),
            "technical_competency": fb_for(cat["technical_competency"], "technical competency"),
            "code_implementation": fb_for(cat["code_implementation"], "code implementation"),
        },
        "detailed_feedback": {
            "strengths": " | ".join([_first_quote_from_transcript(transcript)]) if transcript else "N/A",
            "areas_for_improvement": "Add a clearer approach plus validation (tests/examples) to enable deeper scoring.",
            "specific_recommendations": "Ask clarifying questions, state complexity, and write a minimal working solution.",
        },
    }
    obj = PartialEvaluationSchema(**parsed)
    state["partial_evaluation_result"] = _model_dump(obj)
    return state

def analyze_interview_video(video_path: str):
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
                if eye == "left":
                    indices = [33, 133, 157, 158, 159, 160, 161, 173]
                else:
                    indices = [362, 263, 384, 385, 386, 387, 388, 466]
                xs = [face_landmarks.landmark[i].x for i in indices]
                ys = [face_landmarks.landmark[i].y for i in indices]
                return np.mean(xs), np.mean(ys)

            def _merge_similar_events(self, events: _List[_Dict]) -> _List[_Dict]:
                if not events:
                    return events
                events.sort(key=lambda x: x["start"])
                merged = []
                current = events[0]
                for event in events[1:]:
                    if event["type"] == current["type"] and event["start"] - current["end"] < 2.0:
                        current["end"] = max(current["end"], event["end"])
                        if "intensity" in current and "intensity" in event:
                            current["intensity"] = max(current["intensity"], event["intensity"])
                    else:
                        merged.append(current)
                        current = event
                merged.append(current)
                return merged

            def _filter_weak_events(self, events: _List[_Dict], min_duration: float = 2.0) -> _List[_Dict]:
                return [e for e in events if e["end"] - e["start"] >= min_duration]

            def _generate_statistics(self, events: _List[_Dict], total_duration: float) -> _Dict:
                habit_counts = {}
                total_habit_time = 0
                for event in events:
                    habit_type = event["type"]
                    habit_counts[habit_type] = habit_counts.get(habit_type, 0) + 1
                    total_habit_time += event["end"] - event["start"]
                return {
                    "total_video_duration": f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}",
                    "total_habits_detected": len(events),
                    "habit_breakdown": habit_counts,
                    "total_habit_time_seconds": round(total_habit_time, 1),
                    "percentage_of_video_with_habits": f"{(total_habit_time / total_duration * 100):.1f}%",
                }

            def _is_touching_face(self, hand_landmarks, face_landmarks, threshold: float = 0.05) -> bool:
                hand_idxs = [0, 1, 4, 8, 12, 16, 20]
                face_idxs = [10, 152, 148, 176, 323, 454, 234]
                hand_points = np.array(
                    [[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z] for i in hand_idxs]
                )
                face_points = np.array(
                    [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z] for i in face_idxs]
                )
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
                    event_types.setdefault(event["type"], []).append(event)
                coaching_map = {"eye_darting", "prolonged_look_away", "excessive_hand_movement", "frequent_self_touching"}
                for habit_type, habit_events in event_types.items():
                    if habit_type in coaching_map:
                        total_duration = sum(e["end"] - e["start"] for e in habit_events)
                        feedback.append(
                            {
                                "habit": habit_type.replace("_", " ").title(),
                                "occurrences": len(habit_events),
                                "total_duration_seconds": round(total_duration, 1),
                                "peak_time": self._format_time(habit_events[0]["start"]),
                            }
                        )
                return feedback

            def analyze_video(self, video_path: str) -> _Dict:
                cap = cv2.VideoCapture(video_path)
                reported_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = 30.0 if reported_fps > 60 or reported_fps <= 0 else reported_fps
                face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
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
                                    events.append(
                                        {
                                            "type": "eye_darting",
                                            "start": timestamp,
                                            "end": timestamp + 2,
                                            "intensity": float(gaze_variance),
                                            "threshold": float(darting_threshold),
                                        }
                                    )
                                    last_eye_dart = timestamp
                            if abs(smooth_gaze - gaze_mean) > away_threshold:
                                if looking_away_start is None:
                                    looking_away_start = timestamp
                                elif timestamp - looking_away_start > 2:
                                    direction = "right" if smooth_gaze > gaze_mean else "left"
                                    events.append(
                                        {
                                            "type": "prolonged_look_away",
                                            "start": looking_away_start,
                                            "end": timestamp,
                                            "direction": direction,
                                            "intensity": float(abs(smooth_gaze - gaze_mean)),
                                            "threshold": float(away_threshold),
                                        }
                                    )
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
                                        events.append(
                                            {
                                                "type": "excessive_hand_movement",
                                                "start": timestamp - 1,
                                                "end": timestamp,
                                                "intensity": avg_motion,
                                            }
                                        )
                            self.last_hand_pos = current_pos
                            if face_results.multi_face_landmarks:
                                face_landmarks = face_results.multi_face_landmarks[0]
                                face_width = np.linalg.norm(
                                    [
                                        face_landmarks.landmark[10].x - face_landmarks.landmark[152].x,
                                        face_landmarks.landmark[10].y - face_landmarks.landmark[152].y,
                                    ]
                                )
                                touch_threshold = 0.08 * face_width
                                if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
                                    touch_buffer.append(timestamp)
                                    if len(touch_buffer) > 3:
                                        recent_touches = [t for t in touch_buffer if timestamp - t < 60]
                                        if len(recent_touches) >= 3:
                                            events.append(
                                                {
                                                    "type": "frequent_self_touching",
                                                    "start": recent_touches[0],
                                                    "end": timestamp,
                                                    "count": len(recent_touches),
                                                }
                                            )
                                            touch_buffer.clear()
                cap.release()
                events = self._merge_similar_events(events)
                events = self._filter_weak_events(events, min_duration=2.0)
                feedback = self._generate_coaching_feedback(events)
                total_duration = frame_count / fps if fps > 0 else 0.0
                return {
                    "detected_habits": events,
                    "coaching_feedback": feedback,
                    "summary_stats": self._generate_statistics(events, total_duration),
                }

        detector = NervousHabitDetector()
        return detector.analyze_video(video_path)
    except Exception as e:
        return {
            "error": str(e),
            "detected_habits": [],
            "coaching_feedback": [],
            "summary_stats": {},
        }


