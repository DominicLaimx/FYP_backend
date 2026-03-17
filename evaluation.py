import json
import logging
import re
import time
from collections import deque
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from openai import AzureOpenAI
from pydantic import BaseModel, Field
import os

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=key,
    api_version="2024-02-01",
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CategoryScore(BaseModel):
    score: int = Field(ge=0, le=10)
    justification: str
    evidence: List[str]


class EvaluationCategory(BaseModel):
    communication: CategoryScore
    problem_solving: CategoryScore
    technical_competency: CategoryScore
    code_implementation: CategoryScore


class DetailedFeedback(BaseModel):
    communication: str
    problem_solving: str
    technical_competency: str
    code_implementation: str
    examples_of_what_went_well: List[str]
    areas_to_improve: List[str]


class SOLOAssessment(BaseModel):
    level: int = Field(ge=0, le=4)
    justification: str


class EvaluationSchema(BaseModel):
    student_id: str
    question_id: str
    final_evaluation: EvaluationCategory
    detailed_feedback: DetailedFeedback
    total_score: int
    overall_assessment: str


# FIX: separate Pydantic model for partial eval so it doesn't require score
# fields that the prompt explicitly tells the model not to produce.
class PartialCategoryFeedback(BaseModel):
    observation: str
    evidence: List[str]


class PartialEvaluationCategory(BaseModel):
    communication: PartialCategoryFeedback
    problem_solving: PartialCategoryFeedback
    technical_competency: PartialCategoryFeedback
    code_implementation: PartialCategoryFeedback


class PartialEvaluationSchema(BaseModel):
    student_id: str
    question_id: str
    partial_eval: PartialEvaluationCategory
    examples_of_what_went_well: List[str]
    areas_to_improve: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def overall_assessment_from_score(score: int) -> str:
    if score >= 34:
        return "Strong Hire"
    if score >= 28:
        return "Hire"
    if score >= 20:
        return "No Hire"
    return "Strong No Hire"


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that GPT sometimes adds."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)


def _safe_json_parse(raw: str) -> dict:
    """Strip fences then parse; raises ValueError with context on failure."""
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}\nRaw text:\n{cleaned[:500]}") from exc


# ---------------------------------------------------------------------------
# SOLO agent
# ---------------------------------------------------------------------------

def solo_agent(state: dict) -> dict:
    """
    Classifies the candidate's SOLO taxonomy level (0–4).
    FIX: strips markdown fences before JSON parsing.
    FIX: retries once on parse failure.
    """
    input_data = state["input"][-1]

    prompt = f"""
You are a SOLO taxonomy classifier for a coding interview.

Your task:
- Assign exactly ONE SOLO level (0–4)
- Based ONLY on the structure of the candidate's understanding
- Ignore correctness, performance, style, and code quality

IMPORTANT:
- Do NOT consider scores or hiring decisions
- If unsure between two levels, choose the LOWER level
- Justification must reference observable reasoning or code structure

SOLO LEVEL DEFINITIONS:

Level 0 (Pre-structural):  No coherent approach; ideas irrelevant or disconnected.
Level 1 (Uni-structural):  One relevant idea; no integration with other aspects.
Level 2 (Multi-structural): Multiple relevant ideas present but not integrated.
Level 3 (Relational):      Ideas integrated into a coherent solution; candidate explains why.
Level 4 (Extended Abstract): Generalises beyond the problem; discusses trade-offs or alternatives.

Context:
- Interview Question: {input_data["interview_question"]}
- Candidate Explanation: {input_data.get("summary_of_past_response", "")}
- Candidate Code: {input_data.get("new_code_written", "")}

Return VALID JSON ONLY — no markdown, no extra keys:
{{"level": <integer 0-4>, "justification": "<concise explanation>"}}
""".strip()

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You classify SOLO level and output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = _safe_json_parse(raw)
            solo_obj = SOLOAssessment(**parsed)
            state["solo_result"] = solo_obj.dict()
            return state
        except (ValueError, Exception) as exc:
            log.warning("solo_agent attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                state["solo_result"] = {"error": str(exc), "raw_output": raw if "raw" in dir() else ""}
    return state


# ---------------------------------------------------------------------------
# Evaluation agent  (SOLO + main eval run in parallel)
# ---------------------------------------------------------------------------

def evaluation_agent(state: dict) -> dict:
    """
    FIX: uses transcript + code_history when available instead of only the summary.
    FIX: strips markdown fences before parsing.
    FIX: removed debug print statements.
    """
    input_data = state["input"][-1]

    # Build richer context from transcript / code history when available
    transcript = input_data.get("transcript", [])
    transcript_text = (
        "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
        if transcript
        else input_data.get("summary_of_past_response", "")
    )
    code_history = input_data.get("candidate_code_history_tail", [])
    code_context = (
        "\n\n---\n\n".join(f"[Snapshot {i+1}]\n{c}" for i, c in enumerate(code_history))
        if code_history
        else input_data.get("new_code_written", "")
    )

    eval_prompt = f"""
You are an AI evaluation agent for a coding interview.
For each category include:
- evidence: 1–3 short verbatim quotes from the candidate's explanation or code.

Output VALID JSON ONLY — no markdown, no extra keys.

Schema:
{{
  "student_id": string,
  "question_id": string,
  "final_evaluation": {{
    "communication":        {{"score": int(0-10), "justification": string, "evidence": [string]}},
    "problem_solving":      {{"score": int(0-10), "justification": string, "evidence": [string]}},
    "technical_competency": {{"score": int(0-10), "justification": string, "evidence": [string]}},
    "code_implementation":  {{"score": int(0-10), "justification": string, "evidence": [string]}}
  }},
  "detailed_feedback": {{
    "communication": string,
    "problem_solving": string,
    "technical_competency": string,
    "code_implementation": string,
    "examples_of_what_went_well": [string, string],
    "areas_to_improve": [string, string]
  }}
}}

RULES:
- Scores are integers 0–10.
- Justifications must cite the candidate's actual response or code.
- examples_of_what_went_well: 2–3 specific strengths.
- areas_to_improve: 2–3 specific development areas.
- Do NOT include total_score or overall_assessment.

Context:
- Student ID: {input_data["student_id"]}
- Question ID: {input_data["question_id"]}
- Interview Question: {input_data["interview_question"]}
- Full Transcript:
{transcript_text}
- Code History:
{code_context}

Scoring Guidelines:
COMMUNICATION:    9-10 clear+structured, 7-8 articulate, 5-6 understandable, 3-4 minimal, 0-2 incoherent
PROBLEM SOLVING:  9-10 optimal+edge cases, 7-8 good decomposition, 5-6 basic working, 3-4 flawed, 0-2 none
TECHNICAL:        9-10 deep understanding, 7-8 solid fundamentals, 5-6 partial, 3-4 gaps, 0-2 lacks basics
CODE IMPL:        9-10 production quality, 7-8 readable, 5-6 works/needs refactor, 3-4 poor, 0-2 buggy

Return ONLY the JSON object.
""".strip()

    # Run solo_agent then main evaluation sequentially
    state = solo_agent(state)

    for attempt in range(2):
        raw = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that outputs valid JSON for evaluation."},
                    {"role": "user", "content": eval_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = _safe_json_parse(raw)

            if "EvaluationSchema" in parsed:
                parsed = parsed["EvaluationSchema"]

            categories = parsed["final_evaluation"]
            total_score = sum(
                categories[k]["score"]
                for k in ("communication", "problem_solving", "technical_competency", "code_implementation")
            )
            parsed["total_score"] = total_score
            parsed["overall_assessment"] = overall_assessment_from_score(total_score)

            state["evaluation_result"] = EvaluationSchema(**parsed).dict()
            return state
        except (ValueError, Exception) as exc:
            log.warning("evaluation_agent attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                state["evaluation_result"] = {"error": str(exc), "raw_output": raw}

    return state


# ---------------------------------------------------------------------------
# Partial evaluation agent
# ---------------------------------------------------------------------------

def partial_evaluation_agent(state: dict) -> dict:
    """
    FIX: now validates against PartialEvaluationSchema (no score fields) instead
    of EvaluationSchema, matching what the prompt actually asks the model to produce.
    FIX: removed contradictory hiring-decision instruction from the prompt.
    FIX: strips markdown fences before parsing.
    """
    input_data = state["input"][-1]
    student_id = input_data.get("student_id", "unknown")
    question_id = input_data.get("question_id", "unknown")
    previous_eval = input_data.get("previous_partial_eval", {})
    prev_eval_text = json.dumps(previous_eval, indent=2) if previous_eval else "No previous partial evaluation."

    prompt = f"""
You are an AI partial evaluator for a coding interview.
You are observing the candidate's progress so far.

Do NOT assign numeric scores.
Do NOT make hiring decisions.

Focus on:
- What has improved since the last observation
- What has regressed
- What is still missing

Output VALID JSON ONLY — no markdown, no extra keys — matching this schema exactly:
{{
  "student_id": string,
  "question_id": string,
  "partial_eval": {{
    "communication":        {{"observation": string, "evidence": [string]}},
    "problem_solving":      {{"observation": string, "evidence": [string]}},
    "technical_competency": {{"observation": string, "evidence": [string]}},
    "code_implementation":  {{"observation": string, "evidence": [string]}}
  }},
  "examples_of_what_went_well": [string],
  "areas_to_improve": [string]
}}

Previous partial evaluation:
{prev_eval_text}

Context:
- Student ID: {student_id}
- Question ID: {question_id}
- Question: {input_data["interview_question"]}
- Summary so far: {input_data.get("summary_of_past_response", "")}
- Current code: {input_data.get("new_code_written", "")}

Return valid JSON only.
""".strip()

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI that outputs valid JSON for partial evaluation."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = _safe_json_parse(raw)
            result = PartialEvaluationSchema(**parsed)
            state["partial_evaluation_result"] = result.dict()
            return state
        except (ValueError, Exception) as exc:
            log.warning("partial_evaluation_agent attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                state["partial_evaluation_result"] = {
                    "error": str(exc),
                    "raw_output": raw if "raw" in dir() else "",
                }
    return state


# ---------------------------------------------------------------------------
# Video analysis
# FIX: NervousHabitDetector is now a module-level class (not re-created per call).
# FIX: cv2, mediapipe, numpy imported at the top of the file.
# FIX: _generate_coaching_feedback dead `coaching_map` set removed.
# FIX: redundant manual touch_buffer eviction removed (deque maxlen handles it).
# FIX: looking_away cooldown added to prevent duplicate events.
# FIX: reset() method added for safe reuse of the module-level singleton.
# ---------------------------------------------------------------------------

class NervousHabitDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.last_hand_pos = None

    def reset(self) -> None:
        """Reset per-video state so the singleton is safe to reuse."""
        self.last_hand_pos = None

    def kalman_filter(self, prev_gaze, raw_gaze, alpha: float = 0.2):
        if prev_gaze is None:
            return raw_gaze
        return alpha * raw_gaze + (1 - alpha) * prev_gaze

    def _estimate_gaze(self, face_landmarks) -> float:
        left_iris = np.mean([face_landmarks.landmark[i].x for i in range(468, 474)])
        left_center = 0.5 * (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x)
        right_iris = np.mean([face_landmarks.landmark[i].x for i in range(474, 478)])
        right_center = 0.5 * (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x)
        return ((left_iris - left_center) + (right_iris - right_center)) * 100

    def _merge_similar_events(self, events: List[Dict]) -> List[Dict]:
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

    def _filter_weak_events(self, events: List[Dict], min_duration: float = 2.0) -> List[Dict]:
        return [e for e in events if e["end"] - e["start"] >= min_duration]

    def _generate_statistics(self, events: List[Dict], total_duration: float) -> Dict:
        habit_counts: Dict[str, int] = {}
        total_habit_time = 0.0
        for event in events:
            habit_counts[event["type"]] = habit_counts.get(event["type"], 0) + 1
            total_habit_time += event["end"] - event["start"]
        return {
            "total_video_duration": f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}",
            "total_habits_detected": len(events),
            "habit_breakdown": habit_counts,
            "total_habit_time_seconds": round(total_habit_time, 1),
            "percentage_of_video_with_habits": f"{(total_habit_time / total_duration * 100):.1f}%" if total_duration else "0.0%",
        }

    def _is_touching_face(self, hand_landmarks, face_landmarks, threshold: float = 0.05) -> bool:
        hand_idxs = [0, 1, 4, 8, 12, 16, 20]
        face_idxs = [10, 152, 148, 176, 323, 454, 234]
        hand_points = np.array(
            [[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z]
             for i in hand_idxs]
        )
        face_points = np.array(
            [[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, face_landmarks.landmark[i].z]
             for i in face_idxs]
        )
        distances = np.linalg.norm(hand_points[:, None, :] - face_points[None, :, :], axis=-1)
        return bool(np.any(distances < threshold))

    def _format_time(self, seconds: float) -> str:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def _generate_coaching_feedback(self, events: List[Dict]) -> List[Dict]:
        # FIX: removed dead `coaching_map` set — just iterate all detected event types.
        event_types: Dict[str, list] = {}
        for event in events:
            event_types.setdefault(event["type"], []).append(event)

        feedback = []
        for habit_type, habit_events in event_types.items():
            total_duration = sum(e["end"] - e["start"] for e in habit_events)
            feedback.append({
                "habit": habit_type.replace("_", " ").title(),
                "occurrences": len(habit_events),
                "total_duration_seconds": round(total_duration, 1),
                "peak_time": self._format_time(habit_events[0]["start"]),
            })
        return feedback

    def analyze_video(self, video_path: str) -> Dict:
        self.reset()  # FIX: explicit reset so singleton state doesn't bleed between calls

        cap = cv2.VideoCapture(video_path)

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if reported_fps <= 0 or reported_fps > 60 else reported_fps

        # CAP_PROP_FRAME_COUNT is unreliable for .webm — often returns 0 or -1.
        # Fall back to duration-based estimate using CAP_PROP_POS_AVI_RATIO.
        raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if raw_frame_count <= 0:
            # Seek to end to get duration, then rewind
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_frames = int((duration_ms / 1000.0) * fps) if duration_ms > 0 else int(fps * 600)
        else:
            total_frames = raw_frame_count

        video_duration = total_frames / fps if fps > 0 else 0.0

        MAX_ANALYSIS_FRAMES = 300
        frame_skip = max(1, total_frames // MAX_ANALYSIS_FRAMES) if total_frames > MAX_ANALYSIS_FRAMES else 1
        effective_fps = fps / frame_skip

        log.info(
            "Video analysis: %.1fs, %d frames sampled (skip=%d, eff_fps=%.2f)",
            video_duration, MAX_ANALYSIS_FRAMES, frame_skip, effective_fps,
        )

        face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        calibration_frames = max(60, int(effective_fps * 6))
        gaze_window_frames = max(120, int(effective_fps * 15))
        hand_motion_frames = max(10, int(effective_fps * 1))

        # FIX: touch_buffer uses maxlen only; manual eviction loop removed.
        touch_window_frames = max(60, int(effective_fps * 20))

        calibration_buffer: deque = deque(maxlen=calibration_frames)
        gaze_buffer: deque = deque(maxlen=gaze_window_frames)
        hand_motion_buffer: deque = deque(maxlen=hand_motion_frames)
        touch_buffer: deque = deque(maxlen=touch_window_frames)

        calibrated = False
        gaze_mean = 0.0
        gaze_std = 1.0
        prev_gaze = None

        last_eye_dart = -999.0
        last_look_away = -999.0   # cooldown to prevent duplicate look-away events
        last_hand_event = -999.0  # cooldown to prevent duplicate hand movement events
        looking_away_start = None

        events: List[Dict] = []
        frame_count = 0
        processed_frames = 0
        start_time = time.time()

        try:
            while cap.isOpened() and processed_frames < MAX_ANALYSIS_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if (frame_count - 1) % frame_skip != 0:
                    continue

                processed_frames += 1
                timestamp = (frame_count - 1) / fps

                frame_rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)

                face_results = face_mesh.process(frame_rgb)
                if not face_results.multi_face_landmarks:
                    continue

                face_landmarks = face_results.multi_face_landmarks[0]

                # --- Gaze ---
                raw_gaze = self._estimate_gaze(face_landmarks)
                smooth_gaze = self.kalman_filter(prev_gaze, raw_gaze)
                prev_gaze = smooth_gaze
                calibration_buffer.append(smooth_gaze)

                if not calibrated and len(calibration_buffer) == calibration_buffer.maxlen:
                    gaze_mean = float(np.mean(calibration_buffer))
                    gaze_std = float(np.std(calibration_buffer) + 1e-3)
                    calibrated = True

                gaze_buffer.append(smooth_gaze)

                if calibrated and len(gaze_buffer) == gaze_buffer.maxlen:
                    gaze_variance = float(np.var(gaze_buffer))
                    variance_threshold = (3 * gaze_std) ** 2

                    if gaze_variance > variance_threshold and timestamp - last_eye_dart > 5:
                        events.append({
                            "type": "eye_darting",
                            "start": timestamp,
                            "end": timestamp + 2,
                            "intensity": gaze_variance,
                            "threshold": variance_threshold,
                        })
                        last_eye_dart = timestamp

                    z_score = abs(smooth_gaze - gaze_mean) / gaze_std
                    if z_score > 2.5:
                        if looking_away_start is None:
                            looking_away_start = timestamp
                        elif (
                            timestamp - looking_away_start > 2
                            and timestamp - last_look_away > 3  # FIX: cooldown prevents duplicate events
                        ):
                            direction = "right" if smooth_gaze > gaze_mean else "left"
                            events.append({
                                "type": "prolonged_look_away",
                                "start": looking_away_start,
                                "end": timestamp,
                                "direction": direction,
                                "intensity": float(z_score),
                                "threshold": 2.5,
                            })
                            last_look_away = timestamp
                            looking_away_start = None
                    else:
                        looking_away_start = None

                # --- Hands ---
                hand_results = hands.process(frame_rgb)
                if hand_results.multi_hand_landmarks:
                    dx = face_landmarks.landmark[10].x - face_landmarks.landmark[152].x
                    dy = face_landmarks.landmark[10].y - face_landmarks.landmark[152].y
                    touch_threshold = 0.08 * ((dx * dx + dy * dy) ** 0.5)

                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        current_pos = np.array([wrist.x, wrist.y, wrist.z])

                        if self.last_hand_pos is not None:
                            motion = float(np.linalg.norm(current_pos - self.last_hand_pos))
                            hand_motion_buffer.append(motion)
                            if len(hand_motion_buffer) == hand_motion_buffer.maxlen:
                                if float(np.mean(hand_motion_buffer)) > 0.02 and timestamp - last_hand_event > 5:
                                    events.append({
                                        "type": "excessive_hand_movement",
                                        "start": timestamp - 1,
                                        "end": timestamp + 1,
                                        "intensity": float(np.mean(hand_motion_buffer)),
                                    })
                                    last_hand_event = timestamp

                        self.last_hand_pos = current_pos

                        # FIX: deque maxlen handles eviction; manual while-loop removed.
                        if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
                            touch_buffer.append(timestamp)

                        if len(touch_buffer) >= 3:
                            events.append({
                                "type": "frequent_self_touching",
                                "start": touch_buffer[0],
                                "end": timestamp,
                                "count": len(touch_buffer),
                            })
                            touch_buffer.clear()

        finally:
            cap.release()
            face_mesh.close()
            hands.close()

        events = self._merge_similar_events(events)
        events = self._filter_weak_events(events, min_duration=2.0)
        feedback = self._generate_coaching_feedback(events)

        log.info(
            "Video analysis complete: %.1fs elapsed for %.1fs video (%d frames)",
            time.time() - start_time, video_duration, processed_frames,
        )

        return {
            "detected_habits": events,
            "coaching_feedback": feedback,
            "summary_stats": self._generate_statistics(events, video_duration),
        }


def analyze_interview_video(video_path: str) -> Dict:
    """Public entry point. Creates a fresh detector per call — thread-safe."""
    try:
        return NervousHabitDetector().analyze_video(video_path)
    except Exception as exc:
        log.error("analyze_interview_video failed: %s", exc)
        return {
            "error": str(exc),
            "detected_habits": [],
            "coaching_feedback": [],
            "summary_stats": {},
        }
