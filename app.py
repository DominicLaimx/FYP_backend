import logging
import os
import time
import re
import io
import base64
import pickle
import tempfile
import datetime
import subprocess
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import jwt
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, make_response, Response, stream_with_context
from flask_cors import CORS
from openai import AzureOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, Dict, List
import operator
from dotenv import load_dotenv
from deepface import DeepFace
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from db_connector import (
    get_random_question, get_question_by_id, get_all_summaries,
    get_user, initialise_db_pool, get_user_feedback_history,
    update_user_progress_by_email, delete_history_by_email,
)
from user_login import create_user
from evaluation import evaluation_agent, partial_evaluation_agent, analyze_interview_video
from prompt_templates import PROMPT_TEMPLATES

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

log.info("🔥 Azure ran THIS file")

# ---------------------------------------------------------------------------
# Config / secrets
# ---------------------------------------------------------------------------
ENDPOINT = os.getenv("OPENAI_ENDPOINT")
API_KEY = os.getenv("OPENAI_SECRETKEY")
SECRET_KEY = os.getenv("PWD_SECRET_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_AZURE_ENDPOINT = os.getenv("TTS_AZURE_ENDPOINT")
AZURE_SPEECH_TTS_KEY = os.getenv("AZURE_SPEECH_TTS_KEY")
GOOGLE_TOKEN_PICKLE = os.getenv("GOOGLE_TOKEN_PICKLE")

if not SECRET_KEY:
    raise RuntimeError("PWD_SECRET_KEY env var is required but not set.")

ALLOWED_ORIGIN = "https://mango-bush-0c99ac700.6.azurestaticapps.net"
SCOPES = ["https://www.googleapis.com/auth/drive"]

_tts_http_session = requests.Session()
_TTS_CACHE: dict = {}
_TTS_CACHE_MAX = 256


def _tts_cache_key(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


def _tts_cache_get(key: str):
    return _TTS_CACHE.get(key)


def _tts_cache_set(key: str, audio: bytes) -> None:
    if len(_TTS_CACHE) >= _TTS_CACHE_MAX:
        _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
    _TTS_CACHE[key] = audio


# ---------------------------------------------------------------------------
# OpenAI clients
# ---------------------------------------------------------------------------

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version="2024-02-01",
)

tts_hd_client = AzureOpenAI(
    api_version="2024-05-01-preview",
    api_key=TTS_API_KEY,
    azure_endpoint=TTS_AZURE_ENDPOINT,
)

whisper_client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-06-01",
)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGIN}},
    supports_credentials=True,
)


@app.after_request
def apply_cors(response):
    """Ensure CORS headers are always present (belt-and-suspenders)."""
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class State(TypedDict):
    input: Annotated[List[dict], operator.add]
    decision: Annotated[List[str], operator.add]
    output: Annotated[List[str], operator.add]
    mode: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mode_templates(mode: str) -> Dict[str, str]:
    """Return prompt templates for the given mode, falling back to code_interview."""
    if not mode:
        return PROMPT_TEMPLATES["code_interview"]
    return PROMPT_TEMPLATES.get(mode) or PROMPT_TEMPLATES["code_interview"]


def _stream_response(full_response: str):
    """
    FIX: extracted shared streaming logic so it isn't copy-pasted across
    /respond, /nudge_user, and /nudge_explanation.

    Yields words one-by-one with TTS sentence markers.
    """
    sentence_buffer = ""
    for word in full_response.split():
        sentence_buffer += word + " "
        yield word + " "
        # FIX: removed time.sleep(0.01) — artificial latency, no benefit over HTTP streaming.
        if re.search(r'[.!?]["\']?\s*$', sentence_buffer):
            yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"
            sentence_buffer = ""
    if sentence_buffer.strip():
        yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"


def _invoke_graph(next_input: dict, mode: str) -> tuple[str, list]:
    """Run the LangGraph and return (full_response, decisions)."""
    next_state = app_graph.invoke({
        "input": [next_input],
        "decision": [],
        "output": [],
        "mode": mode,
    })
    full_response = next_state.get("output", ["No response"])[-1] or ""
    decisions = next_state.get("decision", [])
    return full_response, decisions


def _update_code_history(current_state: dict, code: str) -> None:
    """Append code snapshot only when it has changed."""
    if not code.strip():
        return
    history = current_state.setdefault("code_history", [])
    if not history or history[-1] != code:
        history.append(code)


def _options_ok():
    return jsonify({"message": "CORS Preflight OK"}), 204


# ---------------------------------------------------------------------------
# LangGraph agent nodes
# ---------------------------------------------------------------------------

def orchestrator_agent(state: State) -> State:
    input_data = state["input"][-1]
    question = input_data.get("interview_question", "")
    user_input = input_data.get("user_input", "")
    code = input_data.get("new_code_written", "")

    # NOTE: prompt phrasing deliberately avoids instruction-override language
    # that triggers Azure content filters (jailbreak false-positives).
    prompt = (
        f"You are observing a software engineering interview.\n\n"
        f"The interview question is: {question}\n"
        f"The candidate just said: {user_input}\n"
        f"The candidate's current code: {code if code.strip() else '(no code written)'}\n\n"
        f"Categorise the candidate's message into exactly one of these:\n"
        f"1 — the candidate is stuck or silent and has not made progress\n"
        f"2 — the candidate is asking a question about the problem\n"
        f"3 — the candidate has given a response or explanation worth evaluating\n"
        f"4 — the candidate said something unrelated to the interview\n\n"
        f"Respond with a single digit only. No explanation."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You categorise interview messages. Respond with a single digit: 1, 2, 3, or 4."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1,
    )
    decision = (response.choices[0].message.content or "3").strip()
    if decision not in ("1", "2", "3", "4"):
        decision = "3"
    log.info("Orchestrator decision: %s", decision)
    state["decision"].append(decision)
    return state


def router(state: State) -> Dict:
    decision = state["decision"][-1]
    route_map = {
        "1": "guidance_agent",
        "2": "question_agent",
        "3": "eval_agent",
        "4": "offtopic_agent",
        # 5 and 6 are only ever invoked directly via /nudge_user and /nudge_explanation
        # endpoints — the orchestrator no longer emits them.
        "5": "nudge_user_agent",
        "6": "nudge_explanation_agent",
    }
    return {"next": route_map.get(decision, "eval_agent")}


# FIX: replaced 6 near-identical agent functions with one factory.
def _make_agent(template_key: str, include_tone: bool = True):
    """
    Returns a LangGraph node function for the given prompt template key.
    All agents share the same structure; only the template key differs.
    """
    def agent(state: State) -> State:
        input_data = state["input"][-1]
        mode = state.get("mode", "")
        templates = _get_mode_templates(mode)

        tone_part = ""
        if include_tone:
            tone = state.get("tone", "")
            tone_part = f"Tone adjustment: {tone}. " if tone else ""

        prompt = (
            f"Interview context — candidate activity so far: {input_data}. "
            f"{tone_part}"
            f"{templates[template_key]}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional software engineering interviewer conducting a technical interview."},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

            response_text = ""
            for chunk in response:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    response_text += piece

        except Exception as exc:
            # Azure content filter false-positive: return a neutral fallback
            # so the session continues rather than crashing with a 500.
            err_str = str(exc)
            if "content_filter" in err_str or "ResponsibleAI" in err_str:
                log.warning("Content filter triggered on %s — using fallback response", template_key)
                response_text = "Could you walk me through your current thinking on this problem?"
            else:
                raise

        state["output"] = [response_text]
        return state

    agent.__name__ = f"{template_key}_agent"
    return agent


guidance_agent = _make_agent("guidance")
question_agent = _make_agent("question")
eval_agent = _make_agent("evaluation", include_tone=False)
offtopic_agent = _make_agent("offtopic")
nudge_user_agent = _make_agent("nudge_user")
nudge_explanation_agent = _make_agent("nudge_explanation")


def end_state(state: State) -> State:
    log.info("Interview ending.")
    state["output"] = ["Goodbye!"]
    return state


# ---------------------------------------------------------------------------
# Build LangGraph workflow
# ---------------------------------------------------------------------------
workflow = StateGraph(State)

for name, fn in [
    ("orchestrator_agent", orchestrator_agent),
    ("router", router),
    ("guidance_agent", guidance_agent),
    ("question_agent", question_agent),
    ("eval_agent", eval_agent),
    ("end_state", end_state),
    ("offtopic_agent", offtopic_agent),
    ("nudge_user_agent", nudge_user_agent),
    ("nudge_explanation_agent", nudge_explanation_agent),
]:
    workflow.add_node(name, fn)

workflow.add_edge("orchestrator_agent", "router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "guidance_agent": "guidance_agent",
        "question_agent": "question_agent",
        "eval_agent": "eval_agent",
        "offtopic_agent": "offtopic_agent",
        "nudge_user_agent": "nudge_user_agent",
        "nudge_explanation_agent": "nudge_explanation_agent",
        "end_state": "end_state",
    },
)
workflow.set_entry_point("orchestrator_agent")
workflow.set_finish_point("end_state")

app_graph = workflow.compile()

# ---------------------------------------------------------------------------
# In-memory session store
# NOTE: This will not survive restarts or scale across multiple workers.
# Consider replacing with Redis for production.
# ---------------------------------------------------------------------------
session_store: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helper: summarize conversation
# ---------------------------------------------------------------------------
def summarize_conversation(session_id: str, user_input: str, new_code: str) -> str:
    """Rolling summary used only to keep interviewer context flowing."""
    current_state = session_store[session_id]
    past_summary = current_state.get("interaction_summary", "")
    last_bot_response = (
        current_state["output"][-1] if current_state.get("output") else "No response yet"
    )

    summary_prompt = f"""
Given the following conversation history, generate a structured and concise summary.

Conversation History:
{past_summary}

Latest Interaction:
Bot's Response: {last_bot_response}
User's Response: {user_input}
User's Code: {new_code}

Summarize in 2-3 sentences, keeping it clear and concise.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": summary_prompt},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def evaluate_response_partially(session_id: str):
    """Calls partial_evaluation_agent using real transcript + code."""
    current_data = session_store[session_id]
    latest_input = current_data["input"][-1]

    partial_history = current_data.get("partial_eval_history", [])
    last_partial_eval = partial_history[-1] if partial_history else {}

    partial_eval_state = {
        "input": [{
            "student_id": current_data.get("student_id", "unknown"),
            "question_id": str(current_data.get("question_id", "unknown")),
            "interview_question": latest_input.get("interview_question", ""),
            "active_requirements": latest_input.get("interview_question", ""),
            "summary_of_past_response": latest_input.get("summary_of_past_response", ""),
            "user_input": latest_input.get("user_input", ""),
            "new_code_written": latest_input.get("new_code_written", ""),
            "candidate_code": latest_input.get("new_code_written", ""),
            "transcript": current_data.get("transcript", []),
            "candidate_code_history_tail": current_data.get("code_history", [])[-6:],
            "previous_partial_eval": last_partial_eval,
        }],
        "decision": [],
        "output": [],
    }

    updated_state = partial_evaluation_agent(partial_eval_state)
    result = updated_state.get("partial_evaluation_result", {})

    current_data.setdefault("partial_eval_history", []).append(result)
    current_data["current_partial_evaluation"] = result
    return result


# ---------------------------------------------------------------------------
# Google Drive helper — FIX: instantiated once, not per-request
# ---------------------------------------------------------------------------
class DriveService:
    def __init__(self, token_pickle_b64: str | None = None, credentials_path: str = "credentials.json"):
        self._token_pickle_b64 = token_pickle_b64
        self._credentials_path = credentials_path
        self.service = self._get_drive_service()

    def _get_drive_service(self):
        creds = None
        if self._token_pickle_b64:
            creds = pickle.loads(base64.b64decode(self._token_pickle_b64))
        elif os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as fh:
                creds = pickle.load(fh)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self._credentials_path, SCOPES)
                creds = flow.run_local_server(port=8080)
            with open("token.pickle", "wb") as fh:
                pickle.dump(creds, fh)

        return build("drive", "v3", credentials=creds)

    def upload_video(self, file_path: str, folder_name: str = "Videos", user_id: str | None = None) -> dict:
        folder_id = self.create_or_get_folder(folder_name)
        name = f"{user_id}_{os.path.basename(file_path)}" if user_id else os.path.basename(file_path)
        file_metadata = {"name": name, "parents": [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(
            body=file_metadata, media_body=media, fields="id, webViewLink, webContentLink"
        ).execute()
        self.make_public(file["id"])
        return {
            "file_id": file["id"],
            "public_url": f"https://drive.google.com/uc?id={file['id']}",
            "view_url": file["webViewLink"],
        }

    def make_public(self, file_id: str) -> None:
        self.service.permissions().create(
            fileId=file_id, body={"role": "reader", "type": "anyone"}
        ).execute()

    def create_or_get_folder(self, folder_name: str) -> str:
        results = self.service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)",
        ).execute()
        folders = results.get("files", [])
        if folders:
            return folders[0]["id"]
        folder = self.service.files().create(
            body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
            fields="id",
        ).execute()
        return folder["id"]

    def list_user_videos(self, user_id: str) -> list:
        folder_id = self.create_or_get_folder("Videos")
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and name contains '{user_id}'",
            fields="files(id, name)",
        ).execute()
        return results.get("files", [])

    def download_video(self, file_id: str) -> bytes:
        req = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            log.info("Download %d%%", int(status.progress() * 100))
        return fh.getvalue()


_drive_service: DriveService | None = None


def get_drive_service() -> DriveService:
    global _drive_service
    if _drive_service is None:
        _drive_service = DriveService(token_pickle_b64=GOOGLE_TOKEN_PICKLE)
    return _drive_service


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/start", methods=["POST", "OPTIONS"])
def start_interview():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.json or {}
    question_id = data.get("question_id")
    mode = data.get("mode") or "code_interview"

    if not question_id:
        return jsonify({"error": "No question_id provided"}), 400

    question = get_question_by_id(question_id)
    if not question:
        return jsonify({"error": "Invalid question_id"}), 404

    session_id = str(int(time.time()))
    question_text = question["question_text"]

    session_store[session_id] = {
        "question_id": int(question["id"]),
        "student_id": "unknown",
        "input": [{
            "interview_question": question_text,
            "summary_of_past_response": "The user has just started and has not written any code yet.",
            "new_code_written": "",
            "user_input": "",
        }],
        "interaction_summary": "",
        "mode": mode,
        "decision": [],
        "output": [],
        "start_time": time.time(),
        "duration": 0,
        "transcript": [{"role": "assistant", "content": f"Interview question: {question_text}".strip()}],
        "code_history": [],
    }

    return jsonify({
        "session_id": session_id,
        "message": "Interview started!",
        "question": question_text,
        "example": question.get("example", ""),
        "constraint": question.get("reservations", ""),
        "difficulty": question.get("difficulty", ""),
        "question_id": question["id"],
        "start_time": session_store[session_id]["start_time"],
    })


# FIX: extracted shared nudge handler to eliminate duplicated route bodies.
def _nudge_handler(session_id: str, code: str, user_input_label: str):
    """Shared logic for /nudge_user and /nudge_explanation."""
    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session_id"}), 400

    current_state = session_store[session_id]
    prev_summary = current_state.get("interaction_summary", "")
    mode = current_state.get("mode", "code_interview")

    next_input = {
        "interview_question": current_state["input"][0]["interview_question"],
        "summary_of_past_response": prev_summary,
        "new_code_written": code,
        "user_input": user_input_label,
    }

    @stream_with_context
    def generate_stream():
        full_response, decisions = _invoke_graph(next_input, mode)

        yield from _stream_response(full_response)

        if full_response.strip():
            current_state.setdefault("transcript", []).append(
                {"role": "assistant", "content": full_response.strip()}
            )

        _update_code_history(current_state, code)

        current_state.update({
            "input": [next_input],
            "decision": decisions,
            "output": [full_response],
            "mode": mode,
        })

    return Response(generate_stream(), mimetype="text/plain")


@app.route("/nudge_explanation", methods=["POST", "OPTIONS"])
def nudge_explanation():
    if request.method == "OPTIONS":
        return _options_ok()
    data = request.json or {}
    return _nudge_handler(
        session_id=data.get("session_id"),
        code=data.get("code", "") or "",
        user_input_label="nudge explanation",
    )


@app.route("/nudge_user", methods=["POST", "OPTIONS"])
def nudge_user():
    if request.method == "OPTIONS":
        return _options_ok()
    data = request.json or {}
    return _nudge_handler(
        session_id=data.get("session_id"),
        code=data.get("new_code_written", "") or "",
        user_input_label="nudge user",
    )


@app.route("/respond", methods=["POST", "OPTIONS"])
def respond():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.json or {}
    session_id = data.get("session_id")

    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session_id"}), 400

    user_input = data.get("user_input", "") or ""
    new_code_written = data.get("new_code_written", "") or ""

    current_state = session_store[session_id]
    prev_summary = current_state.get("interaction_summary", "")
    mode = current_state.get("mode", "code_interview")
    tone = current_state.get("tone", "")

    next_input = {
        "interview_question": current_state["input"][0]["interview_question"],
        "summary_of_past_response": prev_summary,
        "new_code_written": new_code_written,
        "user_input": user_input,
        "tone": tone,
    }

    
    full_response, decisions = _invoke_graph(next_input, mode)

    if user_input.strip():
        current_state.setdefault("transcript", []).append({"role": "user", "content": user_input.strip()})
    if full_response.strip():
        current_state.setdefault("transcript", []).append({"role": "assistant", "content": full_response.strip()})

    _update_code_history(current_state, new_code_written)

    
    import threading
    def _bg_summarize():
        try:
            new_summary = summarize_conversation(session_id, user_input, new_code_written)
            session_store[session_id]["interaction_summary"] = new_summary
        except Exception as exc:
            log.warning("Background summarize failed: %s", exc)

    threading.Thread(target=_bg_summarize, daemon=True).start()

    current_state.update({
        "input": [next_input],
        "decision": decisions,
        "output": [full_response],
        "mode": mode,
    })

    @stream_with_context
    def generate_stream():
        yield from _stream_response(full_response)

    return Response(generate_stream(), mimetype="text/plain")


@app.route("/analyze_emotion", methods=["POST", "OPTIONS"])
def analyze_emotion():
    if request.method == "OPTIONS":
        return _options_ok()

    _no_face = {"timestamp": time.time(), "dominant_emotion": "no_face", "emotion_probabilities": {"no_face": 100.0}, "face_detected": False}

    if "file" not in request.files:
        return jsonify({**_no_face, "dominant_emotion": "no_file", "emotion_probabilities": {"no_file": 100.0}}), 200

    file = request.files["file"]
    if file.filename == "":
        return jsonify({**_no_face, "dominant_emotion": "empty_file", "emotion_probabilities": {"empty_file": 100.0}}), 200

    try:
        img_array = np.array(Image.open(io.BytesIO(file.read())))
        result = DeepFace.analyze(img_path=img_array, actions=["emotion"], enforce_detection=False)[0]
        return jsonify({
            "timestamp": time.time(),
            "dominant_emotion": result["dominant_emotion"],
            "emotion_probabilities": result["emotion"],
            "face_detected": True,
        })
    except Exception as exc:
        log.warning("Emotion analysis error: %s", exc)
        return jsonify(_no_face), 200


@app.route("/save_recording", methods=["POST", "OPTIONS"])
def save_recording():
    if request.method == "OPTIONS":
        return _options_ok()

    try:
        video_file = request.files.get("video")
        if not video_file:
            return jsonify({"error": "No video file"}), 400

        # Accept session_id + student_id so we can update the DB directly
        session_id = request.form.get("session_id", "")
        student_id = request.form.get("student_id", "")

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            for chunk in video_file.stream:
                tmp.write(chunk)
            temp_path = tmp.name

        mp4_path = None
        try:
            
            mp4_path = temp_path.replace(".webm", ".mp4")
            try:
                ffmpeg_result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", temp_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                        "-c:a", "aac", "-b:a", "64k",
                        mp4_path,
                    ],
                    capture_output=True,
                    timeout=180,
                )
                if ffmpeg_result.returncode != 0:
                    log.warning("ffmpeg failed (returncode %s), falling back to webm: %s",
                                ffmpeg_result.returncode, ffmpeg_result.stderr.decode())
                    mp4_path = None
                else:
                    original_mb = os.path.getsize(temp_path) / 1_048_576
                    mp4_mb = os.path.getsize(mp4_path) / 1_048_576
                    log.info("ffmpeg converted %.1fMB webm → %.1fMB mp4", original_mb, mp4_mb)
            except Exception as exc:
                log.warning("ffmpeg conversion failed (non-fatal): %s", exc)
                mp4_path = None

            upload_path = mp4_path if mp4_path else temp_path

            
            drive = get_drive_service()
            uploaded_file = drive.upload_video(upload_path, user_id=student_id or "unknown")
            recording_url = uploaded_file.get("public_url", "")
            log.info("Uploaded recording to Google Drive: %s", recording_url)

            
            try:
                analysis = analyze_interview_video(temp_path)
            except Exception as exc:
                log.warning("Video analysis failed (non-fatal): %s", exc)
                analysis = {"detected_habits": [], "coaching_feedback": [], "summary_stats": {}}
            analysis["recording_url"] = recording_url

            if session_id and session_id in session_store:
                session_store[session_id]["recording_url"] = recording_url
                session_store[session_id]["video_analysis"] = analysis

            def _save_recording_to_db():
                try:
                    if not session_id or not student_id:
                        return
                    sess = session_store.get(session_id, {})
                    question_id = sess.get("question_id")
                    if not question_id:
                        return
                    existing_eval = sess.get("final_evaluation", {})
                    feedback = {
                        "final_evaluation": existing_eval.get("final_evaluation", {}),
                        "detailed_feedback": existing_eval.get("detailed_feedback", {}),
                        "total_score": existing_eval.get("total_score", 0),
                        "overall_assessment": existing_eval.get("overall_assessment", ""),
                        "recording_url": recording_url,
                        "video_analysis": analysis,
                    }
                    if not update_user_progress_by_email(
                        email=student_id,
                        question_id=int(question_id),
                        feedback_json=feedback,
                    ):
                        log.error("save_recording: failed to update DB for %s q%s", student_id, question_id)
                except Exception as exc:
                    log.error("save_recording DB write failed: %s", exc)

            import threading
            threading.Thread(target=_save_recording_to_db, daemon=True).start()

            return jsonify({
                "recording_url": recording_url,
                "video_analysis": analysis,
            })
        finally:
            os.unlink(temp_path)
            if mp4_path and os.path.exists(mp4_path):
                os.unlink(mp4_path)

    except Exception as exc:
        log.error("save_recording error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/questions/<question_type>", methods=["GET", "OPTIONS"])
def fetch_questions(question_type):
    if request.method == "OPTIONS":
        return _options_ok()
    return jsonify(get_all_summaries(question_type))


@app.route("/question/<int:question_id>", methods=["GET", "OPTIONS"])
def fetch_question_by_id(question_id):
    if request.method == "OPTIONS":
        return _options_ok()
    question = get_question_by_id(question_id)
    if question:
        return jsonify(question)
    return jsonify({"error": "Question not found"}), 404


@app.route("/get_random_question/<question_type>", methods=["GET", "OPTIONS"])
def get_random(question_type):
    if request.method == "OPTIONS":
        return _options_ok()
    return jsonify(get_random_question(question_type))


@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return _options_ok()

    try:
        data = request.json or {}
        email = data.get("email")
        password = data.get("password")

        user = get_user(email, password)
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        token = jwt.encode(
            {"user_id": user["id"], "email": user["email"], "exp": expiry},
            SECRET_KEY,
            algorithm="HS256",
        )

        response = make_response(jsonify({"message": "Login successful"}))
        response.set_cookie(
            "auth_token", token,
            httponly=True, secure=True, samesite="None", max_age=24 * 60 * 60,
        )
        return response

    except Exception as exc:
        log.error("ERROR in /login: %s", exc)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return _options_ok()

    try:
        data = request.get_json() or {}
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not all([name, email, password]):
            return jsonify({"error": "All fields are required"}), 400

        if not email.endswith("ntu.edu.sg"):
            return jsonify({"error": "Email must be a valid NTU email (ntu.edu.sg)"}), 400

        success = create_user(name, email, password)
        if success:
            return jsonify({"message": "User registered successfully!"}), 201
        return jsonify({"error": "Email already registered"}), 409

    except Exception as exc:
        log.error("ERROR in /register: %s", exc)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/initialise_db", methods=["GET", "OPTIONS"])
def initialise_db():
    if request.method == "OPTIONS":
        return _options_ok()
    return initialise_db_pool()


@app.route("/check-auth", methods=["GET", "OPTIONS"])
def check_auth():
    if request.method == "OPTIONS":
        return _options_ok()

    token = request.cookies.get("auth_token")
    if not token:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"message": "Authenticated"})
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401


@app.route("/logout", methods=["POST", "OPTIONS"])
def logout():
    if request.method == "OPTIONS":
        return _options_ok()

    response = jsonify({"message": "Logout successful"})
    response.set_cookie(
        "auth_token", "",
        expires=0, httponly=True, secure=True, samesite="Strict", path="/",
    )
    return response


@app.route("/partial-eval", methods=["POST", "OPTIONS"])
def partial_eval():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session_id"}), 400

    student_id = data.get("student_id")
    if student_id:
        session_store[session_id]["student_id"] = student_id

    partial_evaluation = evaluate_response_partially(session_id)
    return jsonify({"partial_evaluation": partial_evaluation})


@app.route("/final_evaluation", methods=["POST", "OPTIONS"])
def final_evaluation():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.json or {}
    session_id = data.get("session_id")
    student_id = data.get("student_id")
    question_id = data.get("question_id")
    recording_url = data.get("recording_url", "")
    # Fall back to the URL stored by save_recording if frontend didn't pass it
    if not recording_url and session_id in session_store:
        recording_url = session_store.get(session_id, {}).get("recording_url", "")

    log.info("final_evaluation session_id=%s", session_id)

    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session_id"}), 400

    if not student_id or not question_id:
        return jsonify({"error": "Missing student_id or question_id"}), 400

    sess = session_store[session_id]
    final_code = sess["input"][-1].get("new_code_written", "")

    final_input = {
        "student_id": student_id,
        "question_id": str(question_id),
        "recording_url": str(recording_url),
        "interview_question": sess["input"][0]["interview_question"],
        "active_requirements": sess["input"][0]["interview_question"],
        "summary_of_past_response": sess.get("interaction_summary", ""),
        "user_input": sess["input"][-1].get("user_input", ""),
        "new_code_written": final_code,
        "candidate_code": final_code,
        "transcript": sess.get("transcript", [])[-20:],  # cap at last 20 turns — earlier turns are already in interaction_summary
        "candidate_code_history_tail": sess.get("code_history", [])[-8:],
        "partial_eval_history": sess.get("partial_eval_history", []),
    }

    eval_state = {"input": [final_input], "decision": [], "output": []}

    def _save_to_db(result: dict) -> None:
        try:
            feedback_only = {
                "final_evaluation": result.get("final_evaluation", {}),
                "detailed_feedback": result.get("detailed_feedback", {}),
                "total_score": result.get("total_score", 0),
                "overall_assessment": result.get("overall_assessment", ""),
                "recording_url": recording_url,
            }
            if not update_user_progress_by_email(email=student_id, question_id=int(question_id), feedback_json=feedback_only):
                log.error("Failed to update user progress in DB.")
        except Exception as exc:
            log.error("Exception during user progress update: %s", exc)

    with ThreadPoolExecutor(max_workers=1) as pool:
        eval_future = pool.submit(evaluation_agent, eval_state)
        updated_eval_state = eval_future.result()          
        final_result = updated_eval_state.get("evaluation_result", {})
        sess["final_evaluation"] = final_result
        pool.submit(_save_to_db, final_result)

    return jsonify({"final_evaluation": final_result})


@app.route("/me", methods=["GET"])
def get_user_email():
    token = request.cookies.get("auth_token")
    if not token:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"email": decoded["email"]})
    except Exception:
        return jsonify({"error": "Invalid token"}), 401


@app.route("/user-history", methods=["GET", "OPTIONS"])
def user_history():
    if request.method == "OPTIONS":
        return _options_ok()

    token = request.cookies.get("auth_token")
    if not token:
        return jsonify({"error": "Not authenticated"}), 401

    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = decoded_token.get("user_id")
        if not user_id:
            return jsonify({"error": "User ID not found in token"}), 400

        feedback_entries = get_user_feedback_history(str(user_id))
        log.info("Retrieved %d feedback entries for user_id %s", len(feedback_entries), user_id)
        return jsonify({"feedback": feedback_entries})

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401
    except Exception as exc:
        log.error("Unexpected error in /user-history: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/delete_history", methods=["POST", "OPTIONS"])
def delete_history():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.json or {}
    student_id = data.get("student_id")
    question_id = data.get("question_id")

    if not student_id or not question_id:
        return jsonify({"error": "Missing student_id or question_id"}), 400

    try:
        if not delete_history_by_email(email=student_id, question_id=int(question_id)):
            log.error("Failed to delete history in DB.")
    except Exception as exc:
        log.error("Exception during delete history: %s", exc)
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({"res": "success"})


_LANG_CONFIG = {
    "python": {"suffix": ".py", "run": lambda p, _: ["python3", p]},
    "c":      {"suffix": ".c",  "compile": lambda p, e: ["gcc", p, "-o", e], "run": lambda _, e: [e]},
    "cpp":    {"suffix": ".cpp","compile": lambda p, e: ["g++", p, "-o", e], "run": lambda _, e: [e]},
    "java":   {"suffix": ".java","compile": lambda p, _: ["javac", p],       "run": lambda _, __: ["java", "Main"]},
}


@app.route("/run_code", methods=["POST", "OPTIONS"])
def run_code():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session_id"}), 400

    language = (data.get("language", "") or "").lower()
    code = data.get("input_code", "") or ""

    if not code or not language:
        return jsonify({"error": "Missing code or language"}), 400

    config = _LANG_CONFIG.get(language)
    if not config:
        return jsonify({"error": f"Unsupported language: {language}"}), 400

    src_fd, src_path = tempfile.mkstemp(suffix=config["suffix"])
    exe_path = src_path + ".out"
    extra_files = []

    try:
        with os.fdopen(src_fd, "w") as fh:
            fh.write(code)

        # Compile step (if needed)
        if "compile" in config:
            compile_result = subprocess.run(
                config["compile"](src_path, exe_path),
                capture_output=True, text=True, timeout=15,
            )
            
            if compile_result.returncode != 0:
                return jsonify({"res": compile_result.stderr.strip() or "Compilation failed."}), 200
            extra_files.append(exe_path)

        run_cmd = config["run"](src_path, exe_path)
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=10)
        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()
        return jsonify({"res": output if output else error})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Execution timed out."}), 408
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        for path in [src_path, exe_path] + extra_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


@app.route("/transcribe", methods=["POST", "OPTIONS"])
def transcribe_audio():
    if request.method == "OPTIONS":
        return _options_ok()

    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        audio_bytes = file.read()
        result = whisper_client.audio.transcriptions.create(
            model="whisper",
            file=("audio.webm", audio_bytes),
            language="en",
        )
        return jsonify({"transcript": result.text})
    except Exception as exc:
        log.error("Whisper transcription error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/test")
def test():
    return "It works on Azure!"


@app.route("/azure_tts", methods=["POST", "OPTIONS"])
def azure_tts():
    if request.method == "OPTIONS":
        return _options_ok()

    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cache_key = _tts_cache_key(text)
    cached = _tts_cache_get(cache_key)
    if cached:
        log.info("TTS cache hit (%d bytes)", len(cached))
        return (cached, 200, {"Content-Type": "audio/mpeg", "Content-Length": len(cached), "Cache-Control": "no-cache"})

    try:
        url = "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_TTS_KEY,
            "Content-Type": "application/ssml+xml",
            
            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
            "User-Agent": "your-app/1.0",
        }
        ssml = (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>"
            "<voice name='en-US-AvaMultilingualNeural'>"
            "<mstts:express-as style='newscast-formal'>"
            "<prosody rate='15%'>"
            f"{text}"
            "</prosody>"
            "</mstts:express-as></voice></speak>"
        )
        
        response = _tts_http_session.post(url, headers=headers, data=ssml, timeout=30)

        if response.status_code == 200:
            audio = response.content
            _tts_cache_set(cache_key, audio)
            return (audio, 200, {"Content-Type": "audio/mpeg", "Content-Length": len(audio), "Cache-Control": "no-cache"})

        log.error("TTS REST API failed: %s — %s", response.status_code, response.text)
        return jsonify({"error": f"TTS failed: {response.status_code}"}), 500

    except Exception as exc:
        log.error("TTS Error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
