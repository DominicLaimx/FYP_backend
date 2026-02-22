print("🔥 Azure ran THIS file")

import os
import json
import time
import requests
from flask import Flask, request, jsonify, make_response, send_file
from flask import Response, stream_with_context
from flask_cors import CORS
from openai import AzureOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, Dict, List
import operator
from dotenv import load_dotenv
load_dotenv()
import mysql.connector
import jwt
import datetime
from pydantic import BaseModel
import tempfile
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputStream
import io
import re
from db_connector import get_random_question, get_all_questions, get_question_by_id, get_all_summaries
from db_connector import get_user, initialise_db_pool, get_upload_url
from user_login import create_user
from evaluation import evaluation_agent
from evaluation import partial_evaluation_agent, analyze_interview_video
from db_connector import get_user_feedback_history, update_user_progress_by_email, delete_history_by_email
from prompt_templates import PROMPT_TEMPLATES
from google.cloud import texttospeech
import subprocess

app = Flask(__name__)

# Apply CORS with the correct origin and credentials support:
CORS(
    app,
    resources={r"/*": {"origins": "https://mango-bush-0c99ac700.6.azurestaticapps.net"}},
    supports_credentials=True
)

# ✅ Handle CORS for all requests
@app.after_request
def apply_cors(response):
    """Ensure CORS headers are applied correctly."""
    response.headers["Access-Control-Allow-Origin"] = "https://mango-bush-0c99ac700.6.azurestaticapps.net"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# Initialize Azure OpenAI client
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")
SECRET_KEY = os.getenv("PWD_SECRET_KEY")

class State(TypedDict):
    input: Annotated[List[dict], operator.add]
    decision: Annotated[List[str], operator.add]
    output: Annotated[List[str], operator.add]
    mode: str

workflow = StateGraph(State)

client = AzureOpenAI(
  azure_endpoint=endpoint,
  api_key=key,
  api_version="2024-02-01"
)

# Initialize Azure OpenAI client for TTS
tts_hd_client = AzureOpenAI(
    api_version="2024-05-01-preview",
    api_key=os.getenv("TTS_API_KEY"),
    azure_endpoint=os.getenv("TTS_AZURE_ENDPOINT")
)

realtime_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_SECRETKEY"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_version="2024-10-01-preview"
)

whisper_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_SECRETKEY"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_version="2024-06-01"
)

def _get_mode_templates(mode: str) -> Dict[str, str]:
    """
    Fixes system-design "no response" issues caused by missing prompt templates.
    If frontend sends an unknown mode, we fall back to code_interview.
    """
    if not mode:
        return PROMPT_TEMPLATES["code_interview"]
    return PROMPT_TEMPLATES.get(mode) or PROMPT_TEMPLATES["code_interview"]


# __________________________ DEFINING NODES __________________________

def orchestrator_agent(state: State) -> State:
    input_data = state["input"][-1]
    prompt = f"""
You are an interviewer conducting a Software Engineering interview.

The input is: {input_data}

Classify the user's response into one of the following categories:
1 → User is lost and needs guidance
2 → User asks question seeking guidance or clarification
3 → User has given a response and you need to evaluate it
4 → User is not talking about interview but needs a response
5 → Nudge user
6 → Nudge explanation

Output only the number (1, 2, 3, 4, 5, 6) with no additional text.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    decision = (response.choices[0].message.content or "").strip()
    print("Decision is " + decision)
    state["decision"].append(decision)
    return state


def router(state: State) -> Dict:
    decision = state["decision"][-1]

    if decision == "1":
        return {"next": "guidance_agent"}
    elif decision == "2":
        return {"next": "question_agent"}
    elif decision == "3":
        return {"next": "evaluation_agent"}
    elif decision == "4":
        return {"next": "offtopic_agent"}
    elif decision == "5":
        return {"next": "nudge_user_agent"}
    elif decision == "6":
        return {"next": "nudge_explanation_agent"}
    else:
        return {"next": "end_state"}


def guidance_agent(state: State) -> State:
    input_data = state["input"][-1]
    tone = state.get("tone", "")
    mode = state.get("mode", "")
    templates = _get_mode_templates(mode)

    prompt = f"""This is the summary of what the user has done and said in the interview thus far '{input_data}'. Tone_instructions:{tone} {templates["guidance"]}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state


def question_agent(state: State) -> State:
    input_data = state["input"][-1]
    tone = state.get("tone", "")
    mode = state.get("mode", "")
    templates = _get_mode_templates(mode)

    prompt = f"""This is the summary of what the user has done and said in the interview thus far '{input_data}'.Tone_instructions:{tone} {templates["question"]}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state


def evaluation_agent(state: State) -> State:
    input_data = state["input"][-1]
    mode = state.get("mode", "")
    templates = _get_mode_templates(mode)

    prompt = f"""This is the summary of what the user has done and said in the interview thus far '{input_data}'. {templates["evaluation"]}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state


def end_state(state: State) -> State:
    print("Interview is ending...")
    state["output"] = ["Goodbye!"]
    return state


def offtopic_agent(state: State) -> State:
    input_data = state["input"][-1]
    tone = state.get("tone", "")
    mode = state.get("mode", "")
    templates = _get_mode_templates(mode)

    prompt = f"""This is the summary of what the user has done and said in the interview thus far '{input_data}'.Tone_instructions:{tone} {templates["offtopic"]}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state

def nudge_user_agent(state: State) -> State:
    input_data = state["input"][-1]
    tone = state.get("tone", "")
    mode = state.get("mode", "")
    templates = _get_mode_templates(mode)

    prompt = f"""This is the summary of what the user has done and said in the interview thus far '{input_data}'.Tone_instructions:{tone} {templates["nudge_user"]}"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state

def nudge_explanation_agent(state: State) -> State:
    input_data = state["input"][-1]
    tone = state.get("tone", "")
    mode = state.get("mode", "")
    templates = _get_mode_templates(mode)

    prompt = f"""This is the summary of what the user has done and said in the interview thus far '{input_data}'.Tone_instructions:{tone} {templates["nudge_explanation"]}"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state


# __________________________________________________________ HELPER FUNCTIONS __________________________________________________________

def summarize_conversation(session_id: str, user_input: str, new_code: str) -> str:
    """
    Summary used ONLY to keep the interviewer chat flowing.
    Evaluation does NOT use this summary anymore (uses real transcript + code).
    """
    current_state = session_store[session_id]
    past_summary = current_state.get("interaction_summary", "")
    last_bot_response = (current_state["output"][-1] if current_state.get("output") else "No response yet")

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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.2
    )

    new_summary = (response.choices[0].message.content or "").strip()
    return new_summary


def evaluate_response_partially(session_id: str):
    """
    Calls the partial_evaluation_agent using REAL transcript + code (ground truth).
    """
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
            "previous_partial_eval": last_partial_eval
        }],
        "decision": [],
        "output": []
    }

    updated_state = partial_evaluation_agent(partial_eval_state)
    result = updated_state.get("partial_evaluation_result", {})

    if "partial_eval_history" not in current_data:
        current_data["partial_eval_history"] = []
    current_data["partial_eval_history"].append(result)
    current_data["current_partial_evaluation"] = result

    return result


# __________________________________________________________ ADDING NODES TO WORKFLOW __________________________________________________________

workflow.add_node("orchestrator_agent", orchestrator_agent)
workflow.add_node("router", router)
workflow.add_node("guidance_agent", guidance_agent)
workflow.add_node("question_agent", question_agent)
workflow.add_node("evaluation_agent", evaluation_agent)
workflow.add_node("end_state", end_state)
workflow.add_node("offtopic_agent", offtopic_agent)
workflow.add_node("nudge_user_agent", nudge_user_agent)
workflow.add_node("nudge_explanation_agent", nudge_explanation_agent)

workflow.add_edge("orchestrator_agent", "router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "guidance_agent": "guidance_agent",
        "question_agent": "question_agent",
        "evaluation_agent": "evaluation_agent",
        "offtopic_agent": "offtopic_agent",
        "nudge_user_agent": "nudge_user_agent",
        "nudge_explanation_agent": "nudge_explanation_agent"
        "end_state": "end_state"
    }
)

workflow.set_entry_point("orchestrator_agent")
workflow.set_finish_point("end_state")

app_graph = workflow.compile()

# ✅ Store Session States
session_store = {}


# __________________________________________________________ API ENDPOINTS __________________________________________________________

@app.route('/start', methods=['POST', 'OPTIONS'])
def start_interview():
    """Handles interview initialization and fetches a specific question based on question_id."""
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    question_id = data.get("question_id")
    mode = data.get("mode") or "code_interview"

    if not question_id:
        resp = jsonify({"error": "No question_id provided"})
        resp.status_code = 400
        return resp

    question = get_question_by_id(question_id)
    if not question:
        resp = jsonify({"error": "Invalid question_id"})
        resp.status_code = 404
        return resp

    session_id = str(int(time.time()))
    question_text = question["question_text"]

    # Real transcript: start with the interviewer stating the prompt.
    transcript = [
        {"role": "assistant", "content": f"Interview question: {question_text}".strip()}
    ]

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
        "transcript": transcript,
        "code_history": []
    }

    return jsonify({
        "session_id": session_id,
        "message": "Interview started!",
        "question": question_text,
        "example": question.get("example", ""),
        "constraint": question.get("reservations", ""),
        "difficulty": question.get("difficulty", ""),
        "question_id": question["id"],
        "start_time": session_store[session_id]["start_time"]
    })

@app.route('/nudge_explanation', methods=['POST', 'OPTIONS'])
def nudge_explanation():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp
    
    code = data.get("code", "") or ""
    
    current_state = session_store[session_id]
    prev_summary = current_state.get("interaction_summary", "")
    mode = current_state.get("mode", "code_interview")

    next_input = {
        "interview_question": current_state["input"][0]["interview_question"],
        "summary_of_past_response": prev_summary,
        "new_code_written": code,
        "user_input": "nudge explanation",
    }

    @stream_with_context
    def generate_stream():
        # Run interviewer response
        next_state = app_graph.invoke({
            "input": [next_input],
            "decision": [],
            "output": [],
            "mode": mode
        })

        full_response = next_state.get("output", ["No response"])[-1] or ""

        # Stream to frontend (word-by-word)
        sentence_buffer = ""
        for word in full_response.split():
            sentence_buffer += word + " "
            yield word + " "
            time.sleep(0.01)

            if re.search(r'[.!?]["\']?\s*$', sentence_buffer):
                yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"
                sentence_buffer = ""

        if sentence_buffer.strip():
            yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"

        if full_response.strip():
            current_state.setdefault("transcript", []).append({"role": "assistant", "content": full_response.strip()})

        # Track code snapshots (only if changed & non-empty)
        if code.strip():
            history = current_state.setdefault("code_history", [])
            if not history or history[-1] != code:
                history.append(code)

        # Summary ONLY for chat continuity
        # new_summary = summarize_conversation(session_id, user_input, new_code_written)

        # Update session store
        session_store[session_id] = {
            **current_state,
            "input": [next_input],
            "decision": next_state.get("decision", []),
            "output": [full_response],
            "mode": mode,
            # "interaction_summary": new_summary,
            "start_time": current_state.get("start_time"),
            "duration": current_state.get("duration", 0)
        }

    return Response(generate_stream(), mimetype='text/plain')

@app.route('/nudge_user', methods=['POST', 'OPTIONS'])
def nudge_user():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp

    # user_input = data.get("user_input", "") or ""
    new_code_written = data.get("new_code_written", "") or ""

    current_state = session_store[session_id]
    prev_summary = current_state.get("interaction_summary", "")
    mode = current_state.get("mode", "code_interview")

    next_input = {
        "interview_question": current_state["input"][0]["interview_question"],
        "summary_of_past_response": prev_summary,
        "new_code_written": new_code_written,
        "user_input": "nudge user",
    }

    @stream_with_context
    def generate_stream():
        # Run interviewer response
        next_state = app_graph.invoke({
            "input": [next_input],
            "decision": [],
            "output": [],
            "mode": mode
        })

        full_response = next_state.get("output", ["No response"])[-1] or ""

        # Stream to frontend (word-by-word)
        sentence_buffer = ""
        for word in full_response.split():
            sentence_buffer += word + " "
            yield word + " "
            time.sleep(0.01)

            if re.search(r'[.!?]["\']?\s*$', sentence_buffer):
                yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"
                sentence_buffer = ""

        if sentence_buffer.strip():
            yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"

        if full_response.strip():
            current_state.setdefault("transcript", []).append({"role": "assistant", "content": full_response.strip()})

        # Track code snapshots (only if changed & non-empty)
        if new_code_written.strip():
            history = current_state.setdefault("code_history", [])
            if not history or history[-1] != new_code_written:
                history.append(new_code_written)

        # Summary ONLY for chat continuity
        # new_summary = summarize_conversation(session_id, user_input, new_code_written)

        # Update session store
        session_store[session_id] = {
            **current_state,
            "input": [next_input],
            "decision": next_state.get("decision", []),
            "output": [full_response],
            "mode": mode,
            # "interaction_summary": new_summary,
            "start_time": current_state.get("start_time"),
            "duration": current_state.get("duration", 0)
        }

    return Response(generate_stream(), mimetype='text/plain')

@app.route('/respond', methods=['POST', 'OPTIONS'])
def respond():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp

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
        "tone": tone
        }

    @stream_with_context
    def generate_stream():
        # Run interviewer response
        next_state = app_graph.invoke({
            "input": [next_input],
            "decision": [],
            "output": [],
            "mode": mode
        })

        full_response = next_state.get("output", ["No response"])[-1] or ""

        # Stream to frontend (word-by-word)
        sentence_buffer = ""
        for word in full_response.split():
            sentence_buffer += word + " "
            yield word + " "
            time.sleep(0.01)

            if re.search(r'[.!?]["\']?\s*$', sentence_buffer):
                yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"
                sentence_buffer = ""

        if sentence_buffer.strip():
            yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"

        # ✅ Update ground-truth transcript + code history (for evaluation)
        if user_input.strip():
            current_state.setdefault("transcript", []).append({"role": "user", "content": user_input.strip()})
        if full_response.strip():
            current_state.setdefault("transcript", []).append({"role": "assistant", "content": full_response.strip()})

        # Track code snapshots (only if changed & non-empty)
        if new_code_written.strip():
            history = current_state.setdefault("code_history", [])
            if not history or history[-1] != new_code_written:
                history.append(new_code_written)

        # Summary ONLY for chat continuity
        new_summary = summarize_conversation(session_id, user_input, new_code_written)

        # Update session store
        session_store[session_id] = {
            **current_state,
            "input": [next_input],
            "decision": next_state.get("decision", []),
            "output": [full_response],
            "mode": mode,
            "interaction_summary": new_summary,
            "start_time": current_state.get("start_time"),
            "duration": current_state.get("duration", 0)
        }

    return Response(generate_stream(), mimetype='text/plain')

from deepface import DeepFace
from PIL import Image
import numpy as np

@app.route("/analyze_emotion", methods=['POST', 'OPTIONS'])
def analyze_emotion():
    """
    Step 1: Receive frame from frontend (multipart form-data)
    Step 2: Run DeepFace emotion analysis  
    Step 3: Return dominant emotion + full probability distribution
    """
    try:
        # Step 1: Get uploaded file from form-data
        if 'file' not in request.files:
            return jsonify({
                "timestamp": time.time(),
                "dominant_emotion": "no_file", 
                "emotion_probabilities": {"no_file": 100.0},
                "face_detected": False
            }), 200
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "timestamp": time.time(),
                "dominant_emotion": "empty_file",
                "emotion_probabilities": {"empty_file": 100.0}, 
                "face_detected": False
            }), 200
        
        contents = file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        result = DeepFace.analyze(
            img_path=img_array, 
            actions=['emotion'], 
            enforce_detection=False  # Don't fail if no face detected
        )[0]
        
        # Step 4: Return structured response for frontend
        return jsonify({
            "timestamp": time.time(),
            "dominant_emotion": result["dominant_emotion"],
            "emotion_probabilities": result["emotion"],  # {'happy': 45.2, 'sad': 23.1, ...}
            "face_detected": True
        })
    
    except Exception as e:
        # No face, processing error, or DeepFace failure
        print(f"Emotion analysis error: {e}")
        return jsonify({
            "timestamp": time.time(),
            "dominant_emotion": "no_face",
            "emotion_probabilities": {"no_face": 100.0},
            "face_detected": False
        }), 200


from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import base64

SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveService:
    def __init__(self, token_pickle_b64=None, credentials_path='credentials.json'):
        self.token_pickle_b64 = token_pickle_b64
        self.credentials_path = credentials_path
        self.service = self._get_drive_service()

    def _get_drive_service(self):
        creds = None
        if self.token_pickle_b64:
            token_data = base64.b64decode(self.token_pickle_b64)
            creds = pickle.loads(token_data)
        elif os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=8080)

            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('drive', 'v3', credentials=creds)

    def upload_video(self, file_path, folder_name="Videos", user_id=None):
        folder_id = self.create_or_get_folder(folder_name)

        file_metadata = {
            'name': f"{user_id}_{os.path.basename(file_path)}" if user_id else os.path.basename(file_path),
            'parents': [folder_id]
        }

        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink, webContentLink'
        ).execute()

        self.make_public(file['id'])
        return {
            'file_id': file['id'],
            'public_url': f"https://drive.google.com/uc?id={file['id']}",
            'view_url': file['webViewLink']
        }

    def make_public(self, file_id):
        self.service.permissions().create(
            fileId=file_id,
            body={'role': 'reader', 'type': 'anyone'}
        ).execute()

    def create_or_get_folder(self, folder_name):
        results = self.service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)").execute()

        folders = results.get('files', [])
        if folders:
            return folders[0]['id']

        folder_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        return folder['id']

    def list_user_videos(self, user_id):
        folder_id = self.create_or_get_folder("Videos")
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and name contains '{user_id}'",
            fields="files(id, name)").execute()
        return results.get('files', [])

    def download_video(self, file_id):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")

        return fh.getvalue()




@app.route('/save_recording', methods=['POST', 'OPTIONS'])
def save_recording():
    video_file = request.files['video']
    TOKEN_PICKLE_B64 = os.getenv("GOOGLE_TOKEN_PICKLE")
    drive = DriveService(token_pickle_b64=TOKEN_PICKLE_B64)
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
        video_bytes = video_file.read()
        tmp.write(video_bytes)
        temp_path = tmp.name

        _ = drive.upload_video(temp_path, user_id="DOM")

    try:
        results = analyze_interview_video(temp_path)
        results['session_info'] = {'file_size_bytes': len(video_bytes)}
        return jsonify(results)
    except Exception as e:
        resp = jsonify({'error': str(e)})
        resp.status_code = 500
        return resp


@app.route('/questions/<question_type>', methods=['GET', 'OPTIONS'])
def fetch_questions(question_type):
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204
    summaries = get_all_summaries(question_type)
    return jsonify(summaries)


@app.route('/question/<int:question_id>', methods=['GET', 'OPTIONS'])
def fetch_question_by_id(question_id):
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204
    question = get_question_by_id(question_id)
    if question:
        return jsonify(question)
    resp = jsonify({"error": "Question not found"})
    resp.status_code = 404
    return resp


@app.route('/get_random_question/<question_type>', methods=['GET', 'OPTIONS'])
def get_random(question_type):
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204
    qn = get_random_question(question_type)
    return jsonify(qn)

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == "OPTIONS":
        response = make_response("", 204)
        response.headers['Access-Control-Allow-Origin'] = 'https://mango-bush-0c99ac700.6.azurestaticapps.net'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        data = request.json or {}
        email = data.get("email")
        password = data.get("password")

        user = get_user(email, password)
        if not user:
            resp = make_response(jsonify({"error": "Invalid credentials"}))
            resp.status_code = 401
            resp.headers['Access-Control-Allow-Origin'] = 'https://mango-bush-0c99ac700.6.azurestaticapps.net'
            resp.headers['Access-Control-Allow-Credentials'] = 'true'
            return resp

        token = jwt.encode(
            {"user_id": user["id"], "email": user["email"], "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)},
            SECRET_KEY,
            algorithm="HS256"
        )

        response = make_response(jsonify({"message": "Login successful"}))
        response.set_cookie(
            "auth_token",
            token,
            httponly=True,
            secure=True,
            samesite="None",
            max_age=24 * 60 * 60
        )
        response.headers['Access-Control-Allow-Origin'] = 'https://mango-bush-0c99ac700.6.azurestaticapps.net'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    except Exception as e:
        print(f"❌ ERROR in /login: {e}")
        resp = make_response(jsonify({"error": "Internal Server Error"}))
        resp.status_code = 500
        resp.headers['Access-Control-Allow-Origin'] = 'https://mango-bush-0c99ac700.6.azurestaticapps.net'
        resp.headers['Access-Control-Allow-Credentials'] = 'true'
        return resp

# @app.route('/login', methods=['POST', 'OPTIONS'])
# def login():
#     if request.method == "OPTIONS":
#         return jsonify({"message": "CORS Preflight OK"}), 204

#     try:
#         data = request.json or {}
#         email = data.get("email")
#         password = data.get("password")

#         user = get_user(email, password)
#         if not user:
#             resp = jsonify({"error": "Invalid credentials"})
#             resp.status_code = 401
#             return resp

#         token = jwt.encode(
#             {"user_id": user["id"], "email": user["email"], "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)},
#             SECRET_KEY,
#             algorithm="HS256"
#         )

#         response = make_response(jsonify({"message": "Login successful"}))
#         response.set_cookie(
#             "auth_token",
#             token,
#             httponly=True,
#             secure=True,
#             samesite="None",
#             max_age=24 * 60 * 60
#         )
#         return response

#     except Exception as e:
#         print(f"❌ ERROR in /login: {e}")
#         resp = jsonify({"error": "Internal Server Error"})
#         resp.status_code = 500
#         return resp


@app.route('/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    try:
        data = request.get_json() or {}
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            resp = jsonify({"error": "All fields are required"})
            resp.status_code = 400
            return resp

        success = create_user(name, email, password)

        if success:
            resp = jsonify({"message": "User registered successfully!"})
            resp.status_code = 201
            return resp
        else:
            resp = jsonify({"error": "Email already registered"})
            resp.status_code = 409
            return resp

    except Exception as e:
        print(f"❌ ERROR in /register: {e}")
        resp = jsonify({"error": "Internal Server Error"})
        resp.status_code = 500
        return resp


@app.route('/initialise_db', methods=['GET', 'OPTIONS'])
def initialise_db():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204
    return initialise_db_pool()


@app.route('/check-auth', methods=['GET', 'OPTIONS'])
def check_auth():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    token = request.cookies.get("auth_token")
    if not token:
        resp = jsonify({"error": "Not authenticated"})
        resp.status_code = 401
        return resp

    try:
        _ = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"message": "Authenticated"})
    except jwt.ExpiredSignatureError:
        resp = jsonify({"error": "Token expired"})
        resp.status_code = 401
        return resp
    except jwt.InvalidTokenError:
        resp = jsonify({"error": "Invalid token"})
        resp.status_code = 401
        return resp


@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    response = jsonify({"message": "Logout successful"})
    response.set_cookie(
        "auth_token",
        "",
        expires=0,
        httponly=True,
        secure=True,
        samesite="Strict",
        # domain="fypbackend-b5gchph9byc4b8gt.canadacentral-01.azurewebsites.net",
        path="/"
    )
    return response


@app.route('/partial-eval', methods=['POST', 'OPTIONS'])
def partial_eval():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp

    # Optional: allow frontend to attach student_id once known
    student_id = data.get("student_id")
    if student_id:
        session_store[session_id]["student_id"] = student_id

    partial_evaluation = evaluate_response_partially(session_id)
    return jsonify({"partial_evaluation": partial_evaluation})


@app.route("/final_evaluation", methods=["POST", "OPTIONS"])
def final_evaluation():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    student_id = data.get("student_id")
    question_id = data.get("question_id")

    print("Received session_id:", session_id)
    print("Current session_store keys:", session_store.keys())

    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp

    if not student_id or not question_id:
        resp = jsonify({"error": "Missing student_id or question_id"})
        resp.status_code = 400
        return resp

    # ✅ Ground-truth data
    transcript = session_store[session_id].get("transcript", [])
    final_code = session_store[session_id]["input"][-1].get("new_code_written", "")
    code_history = session_store[session_id].get("code_history", [])
    partial_eval_history = session_store[session_id].get("partial_eval_history", [])

    final_input = {
        "student_id": student_id,
        "question_id": str(question_id),
        "interview_question": session_store[session_id]["input"][0]["interview_question"],
        "active_requirements": session_store[session_id]["input"][0]["interview_question"],
        "summary_of_past_response": session_store[session_id].get("interaction_summary", ""),
        "user_input": session_store[session_id]["input"][-1].get("user_input", ""),
        "new_code_written": final_code,
        "candidate_code": final_code,
        "transcript": transcript,
        "candidate_code_history_tail": code_history[-8:],
        "partial_eval_history": partial_eval_history
    }

    eval_state = {"input": [final_input], "decision": [], "output": []}
    updated_eval_state = evaluation_agent(eval_state)
    final_result = updated_eval_state.get("evaluation_result", {})

    session_store[session_id]["final_evaluation"] = final_result

    try:
        feedback_only = {
            "final_evaluation": final_result.get("final_evaluation", {}),
            "detailed_feedback": final_result.get("detailed_feedback", {}),
            "scores": final_result.get("total_score_0_100", 0),
            "overall_assessment": final_result.get("overall_assessment", "")
        }

        update_success = update_user_progress_by_email(
            email=student_id,
            question_id=int(question_id),
            feedback_json=feedback_only
        )

        if not update_success:
            print("❌ Failed to update user progress in DB.")
    except Exception as e:
        print(f"❌ Exception during user progress update: {e}")

    return jsonify({"final_evaluation": final_result})


@app.route('/me', methods=['GET'])
def get_user_email():
    token = request.cookies.get("auth_token")
    if not token:
        resp = jsonify({"error": "Not authenticated"})
        resp.status_code = 401
        return resp
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"email": decoded["email"]})
    except Exception:
        resp = jsonify({"error": "Invalid token"})
        resp.status_code = 401
        return resp


@app.route('/user-history', methods=['GET', 'OPTIONS'])
def user_history():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    token = request.cookies.get("auth_token")
    if not token:
        resp = jsonify({"error": "Not authenticated"})
        resp.status_code = 401
        return resp

    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = decoded_token.get("user_id")
        if not user_id:
            resp = jsonify({"error": "User ID not found in token"})
            resp.status_code = 400
            return resp

        feedback_entries = get_user_feedback_history(str(user_id))
        print(f"✅ Retrieved {len(feedback_entries)} feedback entries for user_id {user_id}")
        return jsonify({"feedback": feedback_entries})

    except jwt.ExpiredSignatureError:
        resp = jsonify({"error": "Token expired"})
        resp.status_code = 401
        return resp
    except jwt.InvalidTokenError:
        resp = jsonify({"error": "Invalid token"})
        resp.status_code = 401
        return resp
    except Exception as e:
        print("❌ Unexpected error in /user-history:", e)
        resp = jsonify({"error": "Internal server error"})
        resp.status_code = 500
        return resp


@app.route("/delete_history", methods=["POST", "OPTIONS"])
def delete_history():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    student_id = data.get("student_id")
    question_id = data.get("question_id")

    print(f"student_id: {student_id}, question_id: {question_id}")
    if not student_id or not question_id:
        resp = jsonify({"error": "Missing student_id or question_id"})
        resp.status_code = 400
        return resp

    try:
        update_success = delete_history_by_email(email=student_id, question_id=int(question_id))
        if not update_success:
            print("❌ Failed to delete history in DB.")
    except Exception as e:
        print(f"❌ Exception during delete history: {e}")
        resp = jsonify({"error": "Internal server error"})
        resp.status_code = 500
        return resp

    return jsonify({"res": "success"})


@app.route('/run_code_python', methods=['POST', 'OPTIONS'])
def run_code_python():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp

    code = data.get('input_code', '') or ""
    temp_file = 'temp_code.py'

    with open(temp_file, 'w') as f:
        f.write(code)

    try:
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()

        return jsonify({"res": output if output else error})

    except subprocess.TimeoutExpired:
        resp = jsonify({'error': 'Execution timed out.'})
        resp.status_code = 408
        return resp
    except Exception as e:
        resp = jsonify({'error': str(e)})
        resp.status_code = 500
        return resp
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


@app.route('/run_code', methods=['POST', 'OPTIONS'])
def run_code():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.json or {}
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        resp = jsonify({"error": "Invalid session_id"})
        resp.status_code = 400
        return resp

    language = (data.get('language', '') or '').lower()
    code = data.get('input_code', '') or ""

    if not code or not language:
        resp = jsonify({"error": "Missing code or language"})
        resp.status_code = 400
        return resp

    temp_files = {
        'python': 'temp_code.py',
        'c': 'temp_code.c',
        'cpp': 'temp_code.cpp',
        'java': 'Main.java'
    }

    temp_file = temp_files.get(language)
    if not temp_file:
        resp = jsonify({"error": f"Unsupported language: {language}"})
        resp.status_code = 400
        return resp

    with open(temp_file, 'w') as f:
        f.write(code)

    try:
        if language == 'python':
            cmd = ['python3', temp_file]
        elif language == 'c':
            exe = './temp_exe'
            subprocess.run(['gcc', temp_file, '-o', exe], capture_output=True, text=True, timeout=10)
            cmd = [exe]
        elif language == 'cpp':
            exe = './temp_exe'
            subprocess.run(['g++', temp_file, '-o', exe], capture_output=True, text=True, timeout=10)
            cmd = [exe]
        elif language == 'java':
            subprocess.run(['javac', temp_file], capture_output=True, text=True, timeout=10)
            cmd = ['java', 'Main']
        else:
            resp = jsonify({"error": "Unsupported language"})
            resp.status_code = 400
            return resp

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()

        return jsonify({"res": output if output else error})

    except subprocess.TimeoutExpired:
        resp = jsonify({'error': 'Execution timed out.'})
        resp.status_code = 408
        return resp
    except Exception as e:
        resp = jsonify({'error': str(e)})
        resp.status_code = 500
        return resp
    finally:
        for f in ['temp_code.py', 'temp_code.c', 'temp_code.cpp', 'Main.java', 'Main.class', 'temp_exe']:
            if os.path.exists(f):
                os.remove(f)


@app.route('/elevenlabs_tts', methods=['POST', 'OPTIONS'])
def elevenlabs_tts():
    """Handles ElevenLabs TTS with proper CORS support."""
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    text = (request.json or {}).get("text", "")
    if not text:
        resp = jsonify({"error": "No text provided"})
        resp.status_code = 400
        return resp

    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        resp = jsonify({"error": "Missing ElevenLabs credentials"})
        resp.status_code = 500
        return resp

    try:
        def generate():
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
            headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
            payload = {
                "text": text,
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 0.8,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }

            with requests.post(url, headers=headers, json=payload, stream=True) as r:
                if r.status_code != 200:
                    return
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk

        return Response(generate(), mimetype='audio/mpeg')

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        resp = jsonify({"error": "Internal server error during TTS generation."})
        resp.status_code = 500
        return resp


@app.route('/openai_tts', methods=['POST', 'OPTIONS'])
def openai_tts():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    text = (request.json or {}).get("text", "")
    if not text:
        resp = jsonify({"error": "No text provided"})
        resp.status_code = 400
        return resp

    try:
        response = tts_hd_client.audio.speech.create(
            model="tts-hd",
            voice="nova",
            input=text
        )

        if response:
            audio_stream = response.content
            def generate_audio_stream():
                yield audio_stream
            return Response(generate_audio_stream(), mimetype='audio/mpeg')

        resp = jsonify({"error": "Failed to generate audio"})
        resp.status_code = 500
        return resp

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        resp = jsonify({"error": "Internal server error during TTS generation."})
        resp.status_code = 500
        return resp


@app.route('/gpt4o_realtime_tts', methods=['POST', 'OPTIONS'])
def gpt4o_realtime_tts():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    text = (request.json or {}).get("text", "")
    if not text:
        resp = jsonify({"error": "No text provided"})
        resp.status_code = 400
        return resp

    try:
        response = realtime_client.audio.speech.create(
            model="gpt-4o-mini-realtime-preview",
            voice="Alloy",
            input=text
        )

        if response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                response.stream_to_file(f.name)
                temp_path = f.name
            return send_file(temp_path, mimetype="audio/mpeg")

        resp = jsonify({"error": "Failed to generate audio"})
        resp.status_code = 500
        return resp

    except Exception as e:
        print(f"Error during GPT-4o realtime TTS generation: {e}")
        resp = jsonify({"error": "Internal server error during TTS generation."})
        resp.status_code = 500
        return resp


@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    file = request.files.get("audio")
    if not file:
        resp = jsonify({"error": "No audio file provided"})
        resp.status_code = 400
        return resp

    try:
        audio_bytes = file.read()

        result = whisper_client.audio.transcriptions.create(
            model="whisper",
            file=("audio.webm", audio_bytes),
            language="en"
        )

        return jsonify({"transcript": result.text})
    except Exception as e:
        print("❌ Whisper transcription error:", e)
        resp = jsonify({"error": str(e)})
        resp.status_code = 500
        return resp


@app.route('/test')
def test():
    return "It works on Azura! " + str(os.getenv("AZURE_SPEECH_TTS_KEY") or "")


@app.route("/azure_tts", methods=["POST", "OPTIONS"])
def azure_tts():
    AZURE_TTS_REGION = "eastus"

    if request.method == "OPTIONS":
        return jsonify({"message": "CORS Preflight OK"}), 204

    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        resp = jsonify({"error": "No text provided"})
        resp.status_code = 400
        return resp

    print(f"🗣️ Azure TTS requested for sentence: {repr(text)}")

    try:
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_TTS_KEY"), region=AZURE_TTS_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz128KBitRateMonoMp3
        )

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        voice_settings = {
            "voice": "en-US-AvaMultilingualNeural",
            "pace": "medium",
            "pause": "100ms",
            "tone": "newscast-formal",
            "pitch": "0%"
        }
        ssml_text = f"""
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
  xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
  <voice name="{voice_settings['voice']}">
    <prosody rate="{voice_settings['pace']}" pitch="{voice_settings['pitch']}">
      <break time="{voice_settings['pause']}"/>
      <mstts:express-as style="{voice_settings['tone']}">
        {text}
      </mstts:express-as>
    </prosody>
  </voice>
</speak>
""".strip()

        result = synthesizer.speak_ssml_async(ssml_text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancellation_details = speechsdk.CancellationDetails(result)
            raise Exception(f"TTS failed: {cancellation_details.reason} - {cancellation_details.error_details}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out:
            out.write(result.audio_data)
            temp_path = out.name

        return send_file(temp_path, mimetype="audio/mpeg")

    except Exception as e:
        print(f"❌ Azure TTS Error: {e}")
        resp = jsonify({"error": "Internal server error during Azure TTS."})
        resp.status_code = 500
        return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
