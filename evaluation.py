import json
from openai import AzureOpenAI
from typing import TypedDict, Annotated, Dict, List
from pydantic import BaseModel, Field
import os
# Initialize Azure OpenAI client
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")
SECRET_KEY = os.getenv("PWD_SECRET_KEY")

# Initialize Azure OpenAI client
client = AzureOpenAI(
  azure_endpoint=endpoint, 
  api_key=key,  
  api_version="2024-02-01"
)

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

class SOLOAssessment(BaseModel):
    level: int = Field(ge=0, le=4)
    justification: str

class EvaluationSchema(BaseModel):
    student_id: str
    question_id: str
    final_evaluation: EvaluationCategory
    detailed_feedback: DetailedFeedback
    total_score: int
    overall_assessment:str

class PartialCategoryFeedback(BaseModel):
    observation: str
    evidence: List[str]

class PartialEvaluation(BaseModel):
    communication: PartialCategoryFeedback
    problem_solving: PartialCategoryFeedback
    technical_competency: PartialCategoryFeedback
    code_implementation: PartialCategoryFeedback

def overall_assessment_from_score(score: int) -> str:
    if score >= 34:
        return "Strong Hire"
    if score >= 28:
        return "Hire"
    if score >= 20:
        return "No Hire"
    return "Strong No Hire"

def solo_agent(state: dict) -> dict:
    """
    Determines the candidate's SOLO taxonomy level.
    This agent MUST NOT see scores or hiring decisions.
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

        Level 0 (Pre-structural):
        - No coherent approach
        - Ideas are irrelevant, incorrect, or disconnected
        - Code shows no meaningful relation to the problem

        Level 1 (Uni-structural):
        - One relevant idea is present
        - No integration with other aspects
        - Partial or narrow solution

        Level 2 (Multi-structural):
        - Multiple relevant ideas present
        - Ideas are listed but not integrated
        - Code works but reasoning is fragmented

        Level 3 (Relational):
        - Ideas are integrated into a coherent solution
        - Candidate explains why the solution works
        - Code structure reflects reasoning

        Level 4 (Extended Abstract):
        - Generalizes beyond the problem
        - Discusses trade-offs, alternatives, or extensions
        - Demonstrates transfer of understanding

        Context:
        - Interview Question:
        {input_data["interview_question"]}

        - Candidate Explanation:
        {input_data["summary_of_past_response"]}

        - Candidate Code:
        {input_data["new_code_written"]}

        Return VALID JSON ONLY with this schema:
        {{
        "level": <integer 0-4>,
        "justification": "<concise explanation citing reasoning or code structure>"
        }}
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You classify SOLO level and output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response.choices[0].message.content.strip()
    state["output"] = state.get("output", []) + [raw_text]

    try:
        parsed = json.loads(raw_text)
        solo_obj = SOLOAssessment(**parsed)
        state["solo_result"] = solo_obj.dict()
    except (json.JSONDecodeError, ValueError) as e:
        state["solo_result"] = {
            "error": f"Could not parse SOLO output: {e}",
            "raw_output": raw_text
        }

    return state

def evaluation_agent(state: dict) -> dict:
    """
    Calls GPT to output JSON matching the EvaluationSchema,
    storing it in state["evaluation_result"]. 
    This is an 'intermediate' grading each time the user responds.
    """
    input_data = state["input"][-1]
    print("DOM start eval agent")
    solo_result = solo_agent(state)
    state["solo_result"] = solo_result["solo_result"]
    print("DOM safter solo agent")
    prompt = f"""
        You are an AI evaluation agent for a coding interview.
        For each category, also include:
        - evidence: a list of 1–3 short direct quotes from the candidate’s explanation or code that support your score.
        Quotes must be copied verbatim.
        If evidence comes from code, quote the relevant lines.

        Your task is to evaluate the candidate and output VALID JSON ONLY.
        No markdown. No extra keys. No explanations outside JSON.

        Output must match this schema EXACTLY:

        EvaluationSchema:
        - student_id (string)
        - question_id (string)
        - final_evaluation (object):
            - communication (object):
                * score (integer 0–10)
                * justification (string)
                * evidence (List(string))
            - problem_solving (object):
                * score (integer 0–10)
                * justification (string)
                * evidence (List(string))
            - technical_competency (object):
                * score (integer 0–10)
                * justification (string)
                * evidence (List(string))
            - code_implementation (object):
                * score (integer 0–10)
                * justification (string)
                * evidence (List(string))

        - detailed_feedback (object):
            - communication (string)
            - problem_solving (string)
            - technical_competency (string)
            - code_implementation (string)

        IMPORTANT RULES:
        - Scores MUST be integers between 0 and 10.
        - Justifications must reference the candidate’s actual response or code.
        - Do NOT calculate total_score.
        - Do NOT determine overall_assessment.
        - Do NOT include any extra fields.

        Context:
        - Student ID: {input_data["student_id"]}
        - Question ID: {input_data["question_id"]}
        - Interview Question: {input_data["interview_question"]}
        - Candidate Summary: {input_data["summary_of_past_response"]}
        - Candidate Code:
        {input_data["new_code_written"]}

        Scoring Guidelines:

        COMMUNICATION (0–10):
        - 9–10: Clear, structured thinking, explains trade-offs, asks strong questions
        - 7–8: Articulate and logical explanations with clear reasoning
        - 5–6: Understandable but unclear in parts; some reasoning explained
        - 3–4: Difficult to follow; minimal verbal reasoning or mostly code with little explanation
        - 0–2: Minimal or confusing communication (single words/phrases, silence, or completely incoherent)

        PROBLEM SOLVING (0–10):
        - 9–10: Optimal approach, handles edge cases
        - 7–8: Good decomposition, mostly correct
        - 5–6: Basic working approach
        - 3–4: Flawed or incomplete
        - 0–2: No clear approach

        TECHNICAL COMPETENCY (0–10):
        - 9–10: Deep understanding, strong application
        - 7–8: Solid fundamentals
        - 5–6: Partial understanding
        - 3–4: Significant gaps
        - 0–2: Lacks fundamentals

        CODE IMPLEMENTATION (0–10):
        - 9–10: Clean, maintainable, production-quality
        - 7–8: Readable, minor issues
        - 5–6: Works but needs refactoring
        - 3–4: Poor structure
        - 0–2: Buggy or minimal

        Return ONLY the JSON object.
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that outputs valid JSON for evaluation."},
            {"role": "user", "content": prompt}
        ]
    )
    print("DOM after completions")
    print("DOM response", response)
    raw_text = response.choices[0].message.content.strip()
    state["output"] = [raw_text]  # store raw text from GPT in 'output' for reference

    # Parse JSON
    try:
        parsed = json.loads(raw_text)
        if "EvaluationSchema" in parsed:
            inner_data = parsed["EvaluationSchema"]
        else:
            inner_data = parsed
        
        categories = inner_data["final_evaluation"]
        total_score = (
            categories["communication"]["score"] +
            categories["problem_solving"]["score"] +
            categories["technical_competency"]["score"] +
            categories["code_implementation"]["score"]
        )
        
        inner_data["total_score"] = total_score 
        inner_data["overall_assessment"] = overall_assessment_from_score(total_score)
        
        evaluation_obj = EvaluationSchema(**inner_data)
        state["evaluation_result"] = evaluation_obj.dict()

    except (json.JSONDecodeError, ValueError) as e:
        state["evaluation_result"] = {
            "error": f"Could not parse JSON: {e}",
            "raw_output": raw_text
        }
    
    print("\n\n\n\n\nDOM\n\n\n\n")
    print(state)
    return state

def partial_evaluation_agent(state: dict) -> dict:
    """
    Runs on every user response, outputting JSON matching the same schema as the final evaluator.
    Ensures student_id and question_id fields are included.
    """
    input_data = state["input"][-1]

    # Retrieve these from input_data or set to "unknown" if not provided
    student_id = input_data.get("student_id", "unknown")
    question_id = input_data.get("question_id", "unknown")

    # If you keep track of previous partial eval
    previous_eval = input_data.get("previous_partial_eval", {})
    prev_eval_text = json.dumps(previous_eval, indent=2) if previous_eval else "No previous partial evaluation"

    prompt = f"""
You are an AI partial evaluator for a coding interview. 
You are observing the candidate’s progress so far.
Do NOT assign scores.
Do NOT make hiring decisions.
Focus on:
- what improved
- what regressed
- what is still missing

Output valid JSON only, matching this schema exactly (no extra keys, no markdown):

EvaluationSchema:
- student_id (string)
- question_id (string)
- partial_eval (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)
- detailed_feedback (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)

Here is the previous partial evaluation (if any):
{prev_eval_text}

Context:
- Student ID: {student_id}
- Question ID: {question_id}
- Question: {input_data["interview_question"]}
- Summary: {input_data["summary_of_past_response"]}
- Code: {input_data["new_code_written"]}

Scoring: "Strong Hire", "Hire", "No Hire", "Strong No Hire".

Return valid JSON only, including the student_id and question_id.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that outputs valid JSON for partial evaluation."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response.choices[0].message.content.strip()
    state["output"] = [raw_text]

    # Parse JSON
    try:
        parsed = json.loads(raw_text)
        # Validate with your existing Pydantic schema that requires these fields
        evaluation_obj = EvaluationSchema(**parsed)
        state["partial_evaluation_result"] = evaluation_obj.dict()
    except (json.JSONDecodeError, ValueError) as e:
        state["partial_evaluation_result"] = {
            "error": f"Could not parse JSON: {e}",
            "raw_output": raw_text
        }

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
                import time
                import numpy as np
                from collections import deque
                import cv2

                start_time = time.time()
                cap = cv2.VideoCapture(video_path)

                reported_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = 30.0 if reported_fps <= 0 or reported_fps > 60 else reported_fps

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_frames / fps if fps > 0 else 0.0

                # --------------------------------------------------
                # 🎯 HARD CAP: 300 ACTUAL PROCESSED FRAMES
                # --------------------------------------------------
                MAX_ANALYSIS_FRAMES = 300

                if total_frames > MAX_ANALYSIS_FRAMES:
                    frame_skip = max(1, total_frames // MAX_ANALYSIS_FRAMES)
                else:
                    frame_skip = 1

                effective_fps = fps / frame_skip if frame_skip > 0 else fps

                print(
                    f"🎥 {video_duration:.1f}s video → "
                    f"{MAX_ANALYSIS_FRAMES} sampled frames "
                    f"(skip={frame_skip}, eff_fps={effective_fps:.2f})"
                )

                # --------------------------------------------------
                # Initialize MediaPipe models
                # --------------------------------------------------
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

                # --------------------------------------------------
                # Robust buffer sizing (statistically safe)
                # --------------------------------------------------
                calibration_frames = max(60, int(effective_fps * 6))
                gaze_window_frames = max(120, int(effective_fps * 15))
                hand_motion_frames = max(10, int(effective_fps * 1))
                touch_window_frames = max(60, int(effective_fps * 20))

                calibration_buffer = deque(maxlen=calibration_frames)
                gaze_buffer = deque(maxlen=gaze_window_frames)
                hand_motion_buffer = deque(maxlen=hand_motion_frames)
                touch_buffer = deque(maxlen=touch_window_frames)

                calibrated = False
                gaze_mean = 0.0
                gaze_std = 1.0

                prev_gaze = None
                last_eye_dart = 0.0
                looking_away_start = None
                self.last_hand_pos = None  # Reset per video (safety)

                events = []
                frame_count = 0
                processed_frames = 0

                # --------------------------------------------------
                # 🔥 FAST UNIFORM SAMPLING LOOP
                # --------------------------------------------------
                while cap.isOpened() and processed_frames < MAX_ANALYSIS_FRAMES:

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    if (frame_count - 1) % frame_skip != 0:
                        continue

                    processed_frames += 1
                    timestamp = (frame_count - 1) / fps

                    # Fast resize
                    frame_resized = cv2.resize(frame, (640, 360))
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                    # --------------------------------------------------
                    # Face detection first (speed optimization)
                    # --------------------------------------------------
                    face_results = face_mesh.process(frame_rgb)
                    if not face_results.multi_face_landmarks:
                        continue

                    face_landmarks = face_results.multi_face_landmarks[0]

                    # --------------------------------------------------
                    # GAZE PROCESSING
                    # --------------------------------------------------
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

                        # Eye dart detection
                        if gaze_variance > variance_threshold and timestamp - last_eye_dart > 5:
                            events.append({
                                "type": "eye_darting",
                                "start": timestamp,
                                "end": timestamp + 2,
                                "intensity": gaze_variance,
                                "threshold": variance_threshold
                            })
                            last_eye_dart = timestamp

                        # Look-away detection (Z-score based)
                        z_score = abs(smooth_gaze - gaze_mean) / gaze_std

                        if z_score > 2.5:
                            if looking_away_start is None:
                                looking_away_start = timestamp
                            elif timestamp - looking_away_start > 2:
                                direction = "right" if smooth_gaze > gaze_mean else "left"
                                events.append({
                                    "type": "prolonged_look_away",
                                    "start": looking_away_start,
                                    "end": timestamp,
                                    "direction": direction,
                                    "intensity": float(z_score),
                                    "threshold": 2.5
                                })
                                looking_away_start = None
                        else:
                            looking_away_start = None

                    # --------------------------------------------------
                    # HAND PROCESSING
                    # --------------------------------------------------
                    hand_results = hands.process(frame_rgb)

                    if hand_results.multi_hand_landmarks:

                        # Precompute face width once
                        dx = face_landmarks.landmark[10].x - face_landmarks.landmark[152].x
                        dy = face_landmarks.landmark[10].y - face_landmarks.landmark[152].y
                        face_width = (dx * dx + dy * dy) ** 0.5
                        touch_threshold = 0.08 * face_width

                        for hand_landmarks in hand_results.multi_hand_landmarks:

                            wrist = hand_landmarks.landmark[0]
                            current_pos = np.array([wrist.x, wrist.y, wrist.z])

                            # Motion detection
                            if self.last_hand_pos is not None:
                                motion = float(np.linalg.norm(current_pos - self.last_hand_pos))
                                hand_motion_buffer.append(motion)

                                if len(hand_motion_buffer) == hand_motion_buffer.maxlen:
                                    avg_motion = float(np.mean(hand_motion_buffer))
                                    if avg_motion > 0.02:
                                        events.append({
                                            "type": "excessive_hand_movement",
                                            "start": timestamp - 1,
                                            "end": timestamp,
                                            "intensity": avg_motion
                                        })

                            self.last_hand_pos = current_pos

                            # Sliding window touch detection (O(1))
                            if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
                                touch_buffer.append(timestamp)

                            # Remove old entries beyond 60s window
                            while touch_buffer and timestamp - touch_buffer[0] > 60:
                                touch_buffer.popleft()

                            if len(touch_buffer) >= 3:
                                events.append({
                                    "type": "frequent_self_touching",
                                    "start": touch_buffer[0],
                                    "end": timestamp,
                                    "count": len(touch_buffer)
                                })
                                touch_buffer.clear()

                # --------------------------------------------------
                # Cleanup
                # --------------------------------------------------
                cap.release()
                face_mesh.close()
                hands.close()

                # --------------------------------------------------
                # Post-processing
                # --------------------------------------------------
                events = self._merge_similar_events(events)
                events = self._filter_weak_events(events, min_duration=2.0)
                feedback = self._generate_coaching_feedback(events)

                analysis_time = time.time() - start_time

                print(
                    f"✅ Analysis complete: {analysis_time:.1f}s "
                    f"for {video_duration:.1f}s video "
                    f"({processed_frames} frames)"
                )

                return {
                    "detected_habits": events,
                    "coaching_feedback": feedback,
                    "summary_stats": self._generate_statistics(events, video_duration),
                }

            # def analyze_video(self, video_path: str) -> _Dict:
            #     cap = cv2.VideoCapture(video_path)
            #     reported_fps = cap.get(cv2.CAP_PROP_FPS)
            #     fps = 30.0 if reported_fps > 60 or reported_fps <= 0 else reported_fps
            #     face_mesh = self.mp_face_mesh.FaceMesh(
            #         static_image_mode=False,
            #         max_num_faces=1,
            #         refine_landmarks=True,
            #         min_detection_confidence=0.5,
            #         min_tracking_confidence=0.5,
            #     )
            #     hands = self.mp_hands.Hands(
            #         static_image_mode=False,
            #         max_num_hands=1,
            #         min_detection_confidence=0.5,
            #         min_tracking_confidence=0.5,
            #     )
            #     gaze_buffer = deque(maxlen=int(fps * 30))
            #     calibration_buffer = deque(maxlen=int(fps * 10))
            #     calibrated = False
            #     gaze_mean = 0
            #     gaze_std = 1
            #     prev_gaze = None
            #     last_eye_dart = 0
            #     looking_away_start = None
            #     hand_motion_buffer = deque(maxlen=int(fps * 1))
            #     touch_buffer = deque(maxlen=int(fps * 30))
            #     events = []
            #     frame_count = 0
            #     while cap.isOpened():
            #         ret, frame = cap.read()
            #         if not ret:
            #             break
            #         frame_count += 1
            #         if (frame_count - 1) % 10 != 0:
            #             continue
            #         frame = cv2.resize(frame, (640, 360))
            #         timestamp = frame_count / fps
            #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #         face_results = face_mesh.process(frame_rgb)
            #         if not face_results.multi_face_landmarks:
            #             continue
                        
            #         hand_results = hands.process(frame_rgb)

            #         if face_results.multi_face_landmarks:
            #             face_landmarks = face_results.multi_face_landmarks[0]
            #             raw_gaze = self._estimate_gaze(face_landmarks)
            #             smooth_gaze = self.kalman_filter(prev_gaze, raw_gaze)
            #             prev_gaze = smooth_gaze
            #             calibration_buffer.append(smooth_gaze)
            #             if len(calibration_buffer) == calibration_buffer.maxlen:
            #                 gaze_mean = np.mean(calibration_buffer)
            #                 gaze_std = np.std(calibration_buffer) + 0.01
            #                 calibrated = True
            #             gaze_buffer.append(smooth_gaze)
            #             if calibrated and len(gaze_buffer) == gaze_buffer.maxlen:
            #                 darting_threshold = gaze_mean + 3 * gaze_std
            #                 away_threshold = gaze_mean + 2.5 * gaze_std
            #                 gaze_variance = np.var(list(gaze_buffer))
            #                 if gaze_variance > darting_threshold:
            #                     if timestamp - last_eye_dart > 5:
            #                         events.append(
            #                             {
            #                                 "type": "eye_darting",
            #                                 "start": timestamp,
            #                                 "end": timestamp + 2,
            #                                 "intensity": float(gaze_variance),
            #                                 "threshold": float(darting_threshold),
            #                             }
            #                         )
            #                         last_eye_dart = timestamp
            #                 if abs(smooth_gaze - gaze_mean) > away_threshold:
            #                     if looking_away_start is None:
            #                         looking_away_start = timestamp
            #                     elif timestamp - looking_away_start > 2:
            #                         direction = "right" if smooth_gaze > gaze_mean else "left"
            #                         events.append(
            #                             {
            #                                 "type": "prolonged_look_away",
            #                                 "start": looking_away_start,
            #                                 "end": timestamp,
            #                                 "direction": direction,
            #                                 "intensity": float(abs(smooth_gaze - gaze_mean)),
            #                                 "threshold": float(away_threshold),
            #                             }
            #                         )
            #                         looking_away_start = None
            #                 else:
            #                     looking_away_start = None
            #         if hand_results.multi_hand_landmarks:
            #             for hand_landmarks in hand_results.multi_hand_landmarks:
            #                 wrist_pos = hand_landmarks.landmark[0]
            #                 current_pos = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z])
            #                 if self.last_hand_pos is not None:
            #                     motion = float(np.linalg.norm(current_pos - self.last_hand_pos))
            #                     hand_motion_buffer.append(motion)
            #                     if len(hand_motion_buffer) == hand_motion_buffer.maxlen:
            #                         avg_motion = float(np.mean(hand_motion_buffer))
            #                         if avg_motion > 0.02:
            #                             events.append(
            #                                 {
            #                                     "type": "excessive_hand_movement",
            #                                     "start": timestamp - 1,
            #                                     "end": timestamp,
            #                                     "intensity": avg_motion,
            #                                 }
            #                             )
            #                 self.last_hand_pos = current_pos
            #                 if face_results.multi_face_landmarks:
            #                     face_landmarks = face_results.multi_face_landmarks[0]
            #                     face_width = np.linalg.norm(
            #                         [
            #                             face_landmarks.landmark[10].x - face_landmarks.landmark[152].x,
            #                             face_landmarks.landmark[10].y - face_landmarks.landmark[152].y,
            #                         ]
            #                     )
            #                     touch_threshold = 0.08 * face_width
            #                     if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
            #                         touch_buffer.append(timestamp)
            #                         if len(touch_buffer) > 3:
            #                             recent_touches = [t for t in touch_buffer if timestamp - t < 60]
            #                             if len(recent_touches) >= 3:
            #                                 events.append(
            #                                     {
            #                                         "type": "frequent_self_touching",
            #                                         "start": recent_touches[0],
            #                                         "end": timestamp,
            #                                         "count": len(recent_touches),
            #                                     }
            #                                 )
            #                                 touch_buffer.clear()
            #     cap.release()
            #     events = self._merge_similar_events(events)
            #     events = self._filter_weak_events(events, min_duration=2.0)
            #     feedback = self._generate_coaching_feedback(events)
            #     total_duration = frame_count / fps if fps > 0 else 0.0
            #     return {
            #         "detected_habits": events,
            #         "coaching_feedback": feedback,
            #         "summary_stats": self._generate_statistics(events, total_duration),
            #     }

        detector = NervousHabitDetector()
        return detector.analyze_video(video_path)
    except Exception as e:
        return {
            "error": str(e),
            "detected_habits": [],
            "coaching_feedback": [],
            "summary_stats": {},
        }

