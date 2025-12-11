import json
from openai import AzureOpenAI
from typing import TypedDict, Annotated, Dict, List
from pydantic import BaseModel
from typing import List
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


class EvaluationCategory(BaseModel):
    communication: str
    problem_solving: str
    technical_competency: str
    examples_of_what_went_well: str

class EvaluationSchema(BaseModel):
    student_id: str
    question_id: str
    final_evaluation: EvaluationCategory
    detailed_feedback: EvaluationCategory

def evaluation_agent(state: dict) -> dict:
    """
    Calls GPT to output JSON matching the EvaluationSchema,
    storing it in state["evaluation_result"]. 
    This is an 'intermediate' grading each time the user responds.
    """
    input_data = state["input"][-1]

    # Instruct GPT to output valid JSON that matches our Pydantic schema, no extra keys
    # Make sure the prompt includes the relevant info: question, summary, code
    prompt = f"""
You are an AI evaluation agent for a coding interview I need you to be extremely strict! 
Produce your answer as valid JSON ONLY, matching this schema exactly:

EvaluationSchema:
- student_id (string)
- question_id (string)
- final_evaluation (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)
- detailed_feedback (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)
- feedback and examples of what they said / coded well and what they could've done better

NO extra keys, no markdown.

Context you have:
- Interview Question: {input_data["interview_question"]}
- User's Summary: {input_data["summary_of_past_response"]}
- User's Code: {input_data["new_code_written"]}

Scoring categories:
- "Strong Hire", "Hire", "No Hire", "Strong No Hire"

Only output valid JSON, no code blocks, no quotes around keys besides JSON structure.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that outputs valid JSON for evaluation."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response.choices[0].message.content.strip()
    state["output"] = [raw_text]  # store raw text from GPT in 'output' for reference

    # Parse JSON
    try:
        parsed = json.loads(raw_text)
        # Validate with Pydantic
        evaluation_obj = EvaluationSchema(**parsed)
        state["evaluation_result"] = evaluation_obj.dict()
    except (json.JSONDecodeError, ValueError) as e:
        state["evaluation_result"] = {
            "error": f"Could not parse JSON: {e}",
            "raw_output": raw_text
        }
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

# import cv2
# import mediapipe as mp
# import numpy as np
# from collections import deque
# from typing import List, Dict
# from datetime import datetime

# class NervousHabitDetector:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.mp_hands = mp.solutions.hands
#         self.mp_pose = mp.solutions.pose
    
#     def kalman_filter(self,prev_gaze, raw_gaze, alpha=0.2):
#         """Kalman filter for smoothing gaze"""
#         if prev_gaze is None:
#             return raw_gaze
#         return alpha * raw_gaze + (1 - alpha) * prev_gaze

#     def _estimate_gaze(self, face_landmarks):
#         """Estimate gaze direction from face landmarks"""
#         left_iris = np.mean([face_landmarks.landmark[i].x for i in range(468, 474)])
#         left_center = 0.5 * (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x)  # Outer + inner corners

#         right_iris = np.mean([face_landmarks.landmark[i].x for i in range(474, 478)])
#         right_center = 0.5 * (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x)

#         return ((left_iris - left_center) + (right_iris - right_center)) * 100
    
#     def _get_eye_center(self, face_landmarks, eye: str):
#         """Get center of left or right eye"""
#         if eye == 'left':
#             indices = [33, 133, 157, 158, 159, 160, 161, 173]  # Left eye indices
#         else:  
#             indices = [362, 263, 384, 385, 386, 387, 388, 466]  # Right eye indices
        
#         xs = [face_landmarks.landmark[i].x for i in indices]
#         ys = [face_landmarks.landmark[i].y for i in indices]
#         return np.mean(xs), np.mean(ys)
    
#     def _merge_similar_events(self, events: List[Dict]) -> List[Dict]:
#         """Merge events that are close in time and of same type"""
#         if not events:
#             return events
            
#         events.sort(key=lambda x: x['start'])
#         merged = []
#         current = events[0]
        
#         for event in events[1:]:
#             if (event['type'] == current['type'] and 
#                 event['start'] - current['end'] < 2.0):  # Merge if gap < 2 seconds
#                 current['end'] = max(current['end'], event['end'])
#                 if 'intensity' in current and 'intensity' in event:
#                     current['intensity'] = max(current['intensity'], event['intensity'])
#             else:
#                 merged.append(current)
#                 current = event
#         merged.append(current)
#         return merged
    
#     def _filter_weak_events(self, events: List[Dict], min_duration: float = 2.0) -> List[Dict]:
#         """Remove events that are too brief to be meaningful"""
#         return [e for e in events if e['end'] - e['start'] >= min_duration]
    
#     def _generate_statistics(self, events: List[Dict], total_duration: float) -> Dict:
#         """Generate summary statistics"""
#         habit_counts = {}
#         total_habit_time = 0
        
#         for event in events:
#             habit_type = event['type']
#             habit_counts[habit_type] = habit_counts.get(habit_type, 0) + 1
#             total_habit_time += (event['end'] - event['start'])
        
#         return {
#             'total_video_duration': f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}",
#             'total_habits_detected': len(events),
#             'habit_breakdown': habit_counts,
#             'total_habit_time_seconds': round(total_habit_time, 1),
#             'percentage_of_video_with_habits': f"{(total_habit_time / total_duration * 100):.1f}%"
#         }
        
#     def analyze_video(self, video_path: str) -> Dict:
#         """Main analysis function"""
#         cap = cv2.VideoCapture(video_path)
#         reported_fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         if reported_fps > 60 or reported_fps <= 0:
#             print(f"⚠️ Fixing unrealistic FPS: {reported_fps} → 30.0")
#             fps = 30.0
#         else:
#             fps = reported_fps
        
#         print(f"Video Info: {fps} FPS, {total_frames} total frames")
#         print(f"Estimated duration: {total_frames/fps/60:.1f} minutes")
        
#         face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.3,
#             min_tracking_confidence=0.3
#         )
        
#         hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.3,
#             min_tracking_confidence=0.3
#         )
        
#         # Buffers for temporal analysis
#         gaze_buffer = deque(maxlen=int(fps * 30))  # 30s for stable calibration
#         calibration_buffer = deque(maxlen=int(fps * 10))  # First 10s for baseline
#         calibrated = False
#         gaze_mean = 0
#         gaze_std = 1
#         prev_gaze = None
#         last_eye_dart = 0
#         looking_away_start = None

#         hand_motion_buffer = deque(maxlen=int(fps * 1))
#         touch_buffer = deque(maxlen=int(fps * 30))  # 30-second memory
        
#         # Results storage
#         events = []
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
#             if (frame_count - 1) % 10 != 0:  # Process 1 in every 3 frames
#                 continue
            
#             # Also resize frame for faster processing
#             frame = cv2.resize(frame, (640, 480))
                
#             timestamp = frame_count / fps
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process frame
#             face_results = face_mesh.process(frame_rgb)
#             hand_results = hands.process(frame_rgb)
            
#             # 1. DETECT EYE DARTING & GAZE
#             if face_results.multi_face_landmarks:
#                 face_landmarks = face_results.multi_face_landmarks[0]
                
#                 # Get eye landmarks (MediaPipe indices: left eye 33, 133, right eye 362, 263)
#                 left_eye = self._get_eye_center(face_landmarks, 'left')
#                 right_eye = self._get_eye_center(face_landmarks, 'right')

#                 # Estimate gaze
#                 raw_gaze = self._estimate_gaze(face_landmarks)
    
#                 # Kalman filter smoothing
#                 smooth_gaze = self.kalman_filter(prev_gaze, raw_gaze)
#                 prev_gaze = smooth_gaze

#                 # CALIBRATION PHASE (first 10 seconds)
#                 calibration_buffer.append(smooth_gaze)
#                 if len(calibration_buffer) == calibration_buffer.maxlen:
#                     gaze_mean = np.mean(calibration_buffer)
#                     gaze_std = np.std(calibration_buffer) + 0.01  
#                     calibrated = True

#                 gaze_buffer.append(smooth_gaze)
                
#                 # Only detect AFTER calibration
#                 if calibrated and len(gaze_buffer) == gaze_buffer.maxlen:
#                     # ADAPTIVE THRESHOLDS (Holmqvist Ch. 8)
#                     darting_threshold = gaze_mean + 3 * gaze_std      
#                     away_threshold = gaze_mean + 2.5 * gaze_std       
                    
#                     gaze_variance = np.var(list(gaze_buffer))
                    
#                     # Check for eye darting (3σ)
#                     if gaze_variance > darting_threshold:
#                         if timestamp - last_eye_dart > 5:
#                             events.append({
#                                 'type': 'eye_darting',
#                                 'start': timestamp,
#                                 'end': timestamp + 2,
#                                 'intensity': gaze_variance,
#                                 'threshold': darting_threshold
#                             })
#                             last_eye_dart = timestamp
                    
#                     # Check for looking away (2.5σ)
#                     if abs(smooth_gaze - gaze_mean) > away_threshold:
#                         if looking_away_start is None:
#                             looking_away_start = timestamp
#                         elif timestamp - looking_away_start > 2:
#                             direction = 'right' if smooth_gaze > gaze_mean else 'left'
#                             events.append({
#                                 'type': 'prolonged_look_away',
#                                 'start': looking_away_start,
#                                 'end': timestamp,
#                                 'direction': direction,
#                                 'intensity': abs(smooth_gaze - gaze_mean),
#                                 'threshold': away_threshold
#                             })
#                             looking_away_start = None
#                     else:
#                         looking_away_start = None
                
            
#             # HAND MOVEMENTS & SELF-TOUCHING
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     wrist_pos = hand_landmarks.landmark[0]  # Wrist landmark
#                     current_pos = np.array([wrist_pos.x, wrist_pos.y, wrist_pos.z])
                    
#                     if hasattr(self, 'last_hand_pos'):
#                         motion = np.linalg.norm(current_pos - self.last_hand_pos)
#                         hand_motion_buffer.append(motion)
                        
#                         # Check for excessive movement
#                         if len(hand_motion_buffer) == hand_motion_buffer.maxlen:
#                             avg_motion = np.mean(hand_motion_buffer)
#                             if avg_motion > 0.02:  # Threshold
#                                 events.append({
#                                     'type': 'excessive_hand_movement',
#                                     'start': timestamp - 1,
#                                     'end': timestamp,
#                                     'intensity': avg_motion
#                                 })
                    
#                     self.last_hand_pos = current_pos
                    
#                     # Check for self-touching
#                     if face_results.multi_face_landmarks:
#                         face_landmarks = face_results.multi_face_landmarks[0]
#                         face_width = np.linalg.norm([
#                             face_landmarks.landmark[10].x - face_landmarks.landmark[152].x,  # Nose-chin
#                             face_landmarks.landmark[10].y - face_landmarks.landmark[152].y
#                             ])
#                         touch_threshold = 0.08 * face_width
#                         if self._is_touching_face(hand_landmarks, face_landmarks, touch_threshold):
#                             touch_buffer.append(timestamp)
                            
#                             # Check for frequent touching
#                             if len(touch_buffer) > 3:
#                                 recent_touches = [t for t in touch_buffer if timestamp - t < 60]
#                                 if len(recent_touches) >= 3:
#                                     events.append({
#                                         'type': 'frequent_self_touching',
#                                         'start': recent_touches[0],
#                                         'end': timestamp,
#                                         'count': len(recent_touches)
#                                     })
#                                     touch_buffer.clear()
#                 if frame_count % 100 == 0:
#                     progress = (frame_count / total_frames) * 100
#                     print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
#             # frame_count += 1
        
#         cap.release()
        
#         # 3. POST-PROCESS: Merge adjacent events and filter noise
#         events = self._merge_similar_events(events)
#         events = self._filter_weak_events(events, min_duration=2.0)
        
#         print(f"events: {events}")
#         feedback = self._generate_coaching_feedback(events)
        
#         return {
#             'detected_habits': events,
#             'coaching_feedback': feedback,
#             'summary_stats': self._generate_statistics(events, frame_count / fps)
#         }

#     def _is_touching_face(self, hand_landmarks, face_landmarks, threshold: float = 0.05) -> bool:
#         """Check if hand is touching face/neck area"""
#         hand_idxs = [0, 1, 4, 8, 12, 16, 20]  
#         face_idxs = [10, 152, 148, 176, 323, 454, 234]  
        
#         hand_points = np.array([[hand_landmarks.landmark[i].x, 
#                             hand_landmarks.landmark[i].y, 
#                             hand_landmarks.landmark[i].z] for i in hand_idxs])
#         face_points = np.array([[face_landmarks.landmark[i].x, 
#                             face_landmarks.landmark[i].y, 
#                             face_landmarks.landmark[i].z] for i in face_idxs])
        
#         distances = np.linalg.norm(hand_points[:, None, :] - face_points[None, :, :], axis=-1)
        
#         return np.any(distances < threshold)
    
#     def _generate_coaching_feedback(self, events: List[Dict]) -> List[str]:
#         """Generate actionable feedback from detected events"""
#         feedback = []
        
#         event_types = {}
#         for event in events:
#             event_type = event['type']
#             if event_type not in event_types:
#                 event_types[event_type] = []
#             event_types[event_type].append(event)
        
#         coaching_map = set(["eye_darting","prolonged_look_away","excessive_hand_movement","frequent_self_touching"])
        
#         for habit_type, habit_events in event_types.items():
#             if habit_type in coaching_map:
#                 total_duration = sum(e['end'] - e['start'] for e in habit_events)
#                 feedback.append({
#                     'habit': habit_type.replace('_', ' ').title(),
#                     'occurrences': len(habit_events),
#                     'total_duration_seconds': round(total_duration, 1),
#                     'peak_time': self._format_time(habit_events[0]['start']),

#                 })
        
#         return feedback
    
#     def _format_time(self, seconds: float) -> str:
#         """Convert seconds to MM:SS format"""
#         minutes = int(seconds // 60)
#         secs = int(seconds % 60)
#         return f"{minutes:02d}:{secs:02d}"
    
# import json

# def analyze_interview_video(video_path: str):
#     """
#     Main function to analyze a video for nervous habits
    
#     Args:
#         video_path: Path to the video file (e.g., 'interviews/user123.mp4')
    
#     Returns:
#         Dictionary with analysis results
#     """
#     try:
#         detector = NervousHabitDetector()
        
#         print(f"Starting analysis of {video_path}...")
#         results = detector.analyze_video(video_path)
        
#         print(f"\nAnalysis complete!")
#         print(f"Total habits detected: {len(results['detected_habits'])}")
#         print(f"Video duration: {results['summary_stats']['total_video_duration']}")
        
#         return results
        
#     except Exception as e:
#         print(f"Error analyzing video: {e}")
#         return {
#             'error': str(e),
#             'detected_habits': [],
#             'coaching_feedback': [],
#             'summary_stats': {}
#         }
def analyze_interview_video(video_path: str):
    return "blank"
