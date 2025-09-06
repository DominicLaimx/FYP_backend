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
import tempfile
import io
import re
from db_connector import get_random_question 
from db_connector import get_all_questions
from db_connector import get_question_by_id
from db_connector import get_all_summaries
from db_connector import get_user
from user_login import create_user
from evaluation import evaluation_agent 
from evaluation import partial_evaluation_agent
from db_connector import get_user_feedback_history
from db_connector import update_user_progress_by_email
from google.cloud import texttospeech

app = Flask(__name__)
# Apply CORS with the correct origin and credentials support:
CORS(app, resources={r"/*": {"origins": "https://mango-bush-0c99ac700.6.azurestaticapps.net"}}, supports_credentials=True)

# ✅ Handle CORS for all requests
@app.after_request
def apply_cors(response):
    """Ensure CORS headers are applied correctly."""
    response.headers["Access-Control-Allow-Origin"] = "https://mango-bush-0c99ac700.6.azurestaticapps.net"  # ✅ Must be specific origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"  # ✅ Required when using `credentials: "include"

    return response

# Initialize Azure OpenAI client
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")
SECRET_KEY = os.getenv("PWD_SECRET_KEY")

class State(TypedDict):
    input: Annotated[List[dict], operator.add]  # Stores previous discussions as list of JSON objects
    decision: Annotated[List[str], operator.add]  # Allows multiple decisions per execution step
    output: Annotated[List[str], operator.add]  # Stores only the latest response

# Initialize StateGraph with the correct state schema
workflow = StateGraph(State)

# Initialize Azure OpenAI client
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
    azure_endpoint= os.getenv("OPENAI_ENDPOINT"),
    api_version="2024-10-01-preview"
)

whisper_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_SECRETKEY"),
    azure_endpoint= os.getenv("OPENAI_ENDPOINT"),
    api_version="2024-06-01"
)
# __________________________ DEFINING NODES __________________________

def node3(state: State) -> State:
    input_data = state["input"][-1]  # Take the latest input only
    prompt = f"""
        You are an interviewer taking a SWE interview. 

        The input is: {input_data} 

        Classi1 the user's response into one of the following categories:
        1 → User is lost, needs guidance
        2 → User asks question seeking guidance or clarification
        3 → User has given a response and you need to evaluate it
        4 → User is not talking about interview but needs a response

        **Output only the number (1, 2, 3, 4) with no additional text, explanation, or formatting.**
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    decision = response.choices[0].message.content.strip()

    print("Decision is " + decision)
    state["decision"].append(decision)
    return state

def router(state: State) -> Dict:
    decision = state["decision"][-1]  # Take the latest decision
    
    if decision == "1":
        return {"next": "node_4"}
    elif decision == "2":
        return {"next": "node_5"}
    elif decision == "3":
        return {"next": "node_6"}
    elif decision == "4":
        return {"next": "node_8"}
    elif decision == "5":
        return {"next": "node_9"}
    else:
        return {"next": "node_7"}

def node4(state: State) -> State:
    input_data = state["input"][-1]
    prompt = f"This is the summary of what the user has done and said in the interview thus far '{input_data}'.\
    The user is confused and needs some guidance. Provide the user with some guidance questions on how to proceed without giving them the answer. \
    Remember this is a conversation, so leave room for further discussion. \
    You're a friendly and supportive coding interviewer having a conversation. Be casual, encouraging, and ask questions naturally. \
    Try to add some random human elements just to sound as natural as possible.\
    NO EMOJI\
    Make sure you are concise in your response! No more than 3 sentences"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        stream=True  # ✅ Enable streaming
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue  # ✅ Skip empty chunks safely
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state

def node5(state: State) -> State:
    input_data = state["input"][-1]
    prompt = f"This is the summary of what the user has done and said in the interview thus far '{input_data}'.\
    The user has a specific question they'd like answered. Provide a question similar to what an interviewer \
    might ask, without giving them any new information. Leave room for further discussion.\
    If you look at past summary and you feel like the user has not written any code, encourage them to start writing.\
    You're a friendly and supportive coding interviewer having a conversation. Be casual, encouraging, and ask questions naturally. \
    Try to add some random human elements just to sound as natural as possible.\
    NO EMOJI\
        Make sure you are concise in your response! No more than 3 sentences"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        stream=True  # ✅ Enable streaming
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue  # ✅ Skip empty chunks safely
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state

def node6(state: State) -> State:
    input_data = state["input"][-1]
    prompt = f"This is the summary of what the user has done and said in the interview thus far '{input_data}'.\
    The user has given a response. Evaluate if their response is correct.\
    Otherwise, As an interviewer provide the person with some constructive feedback\
    If the answer is good but there is no code written, encourage them to start writing. \
    If code has been written then firstly most importantly, ensure that the code given is correct! If it's not tell them what is wrong but dont give answer \
    If all is good, slightly modify the question parameters for a harder challenge. \
    You're a friendly and supportive coding interviewer having a conversation. Be casual, encouraging, and ask questions naturally. \
    Try to add some random human elements just to sound as natural as possible.\
    NO EMOJI\
    Make sure you are concise in your response! Remember to respond as an interviewer back to the candidate! No more than 3 sentences"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        stream=True  # ✅ Enable streaming
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue  # ✅ Skip empty chunks safely
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state

def node7(state: State) -> State:
    print("Interview is ending...")
    state["output"] = ["Goodbye!"]
    return state

def node8(state: State) -> State:
    input_data = state["input"][-1]
    prompt = f"This is the summary of what the user has done and said in the interview thus far '{input_data}'.\
    The user is not answering the question but asking a pertinent basic questions.\
    Provide a response to the user's question.\
    You're a friendly and supportive coding interviewer having a conversation. Be casual, encouraging, and ask questions naturally. \
    Try to add some random human elements just to sound as natural as possible.\
    NO EMOJI\
        Make sure you are concise in your response! No more than 3 sentences"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        stream=True  # ✅ Enable streaming
    )

    response_text = ""
    for chunk in response:
        if not chunk.choices or not chunk.choices[0].delta:
            continue  # ✅ Skip empty chunks safely
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            response_text += content_piece

    state["output"] = [response_text]
    return state

def node9(state: State) -> State:
    input_data = state["input"][-1]
    prompt = f"Tthank them for the interview and say bye bye."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )

    state["output"] = [response.choices[0].message.content]
    return state
# ______________________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________
# __________________________________________________________ HELPER FUNCTIONS __________________________________________________________
# ______________________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________

def summarize_conversation(session_id: str, user_input: str, new_code: str) -> str:
    """
    Given the current session state, user input, and new code,
    generate a concise updated summary of the conversation using OpenAI.
    Returns the updated conversation summary.
    """
    current_state = session_store[session_id]
    past_summary = current_state.get("interaction_summary", "")
    last_bot_response = (current_state["output"][-1]
                         if current_state.get("output") else "No response yet")

    summary_prompt = f"""
    Given the following conversation history, generate a structured and concise summary:

    Conversation History:
    {past_summary}

    Latest Interaction:
    Bot's Response: {last_bot_response}
    User's Response: {user_input}
    User's Code: {new_code}

    Summarize in 2-3 sentences, keeping it clear and concise.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": summary_prompt}
        ]
    )

    new_summary = response.choices[0].message.content.strip()
    return new_summary

def update_conversation_state(session_id: str, new_summary: str,
                              user_input: str, new_code: str) -> str:
    """
    Uses the updated summary, user input, and code to invoke the state machine
    and store the resulting state in the session. Returns the bot's latest response.
    """
    current_state = session_store[session_id]

    # Build the next input for the workflow
    next_input = {
        "interview_question": current_state["input"][0]["interview_question"],
        "summary_of_past_response": new_summary,
        "new_code_written": new_code,
        "user_input": user_input
    }

    # Invoke the state machine
    next_state = app_graph.invoke({"input": [next_input], "decision": [], "output": []})
    bot_response = next_state.get("output", ["No response"])[-1]

    # Update the session store with new state
    session_store[session_id] = {
        "input": [next_input],
        "decision": next_state["decision"],
        "output": [bot_response],
        # Keep or update any other session fields, e.g. interaction_summary
        "interaction_summary": new_summary,
        "start_time": current_state.get("start_time"),
        "duration": current_state.get("duration", 0)
    }

    return bot_response

def evaluate_response_partially(session_id: str):
    """
    Calls the partial_evaluation_agent to produce a simpler
    incremental judgment. Stores it in session_store under partial_eval_history.
    """
    current_data = session_store[session_id]
    
    # The last "input" from the user (just stored in session_store)
    latest_input = current_data["input"][-1]

    # If we want continuity, pass the last partial eval if it exists
    partial_history = current_data.get("partial_eval_history", [])
    last_partial_eval = partial_history[-1] if partial_history else {}

    # Build the state for partial evaluator
    partial_eval_state = {
        "input": [{
            "interview_question": latest_input["interview_question"],
            "summary_of_past_response": latest_input["summary_of_past_response"],
            "new_code_written": latest_input["new_code_written"],
            "previous_partial_eval": last_partial_eval
        }],
        "decision": [],
        "output": []
    }

    # Call the partial evaluator
    updated_state = partial_evaluation_agent(partial_eval_state)

    # Extract the result
    result = updated_state.get("partial_evaluation_result", {})

    # Store it in the session
    if "partial_eval_history" not in current_data:
        current_data["partial_eval_history"] = []
    current_data["partial_eval_history"].append(result)

    # Optionally store the latest partial eval in a simpler key
    current_data["current_partial_evaluation"] = result

    return result

# ______________________________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________________
# __________________________________________________________ ADDING NODES TO WORKFLOW __________________________________________________________
# ______________________________________________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________________________________

# Add nodes to the workflow
workflow.add_node("node_3", node3)
workflow.add_node("router", router)
workflow.add_node("node_4", node4)
workflow.add_node("node_5", node5)
workflow.add_node("node_6", node6)
workflow.add_node("node_7", node7)
workflow.add_node("node_8", node8)

# Route decisions properly
workflow.add_edge("node_3", "router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "node_4": "node_4",
        "node_5": "node_5",
        "node_6": "node_6",
        "node_8": "node_8",
        "node_7": "node_7"
    }
)

# Set entry and termination points
workflow.set_entry_point("node_3")
workflow.set_finish_point("node_7")

app_graph = workflow.compile()

# ✅ Store Session States
session_store = {}
# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________
# __________________________________________________________ API ENDPOINTS __________________________________________________________
# ___________________________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________________________


@app.route('/start', methods=['POST', 'OPTIONS'])
def start_interview():
    """Handles interview initialization and fetches a specific question based on question_id."""
    
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))
    
    data = request.json
    question_id = data.get("question_id")  # ✅ Get the selected question ID from the frontend

    if not question_id:
        return apply_cors(jsonify({"error": "No question_id provided"}), 400)

    question = get_question_by_id(question_id)  # ✅ Fetch the selected question

    if not question:
        return apply_cors(jsonify({"error": "Invalid question_id"}), 404)

    # Generate a session ID
    session_id = str(int(time.time()))

    # Store session details with the selected question
    session_store[session_id] = {
        "input": [{
            "interview_question": question["question_text"],  # ✅ Store the selected question
            "summary_of_past_response": "The user has just started and has not written any code yet.",
            "new_code_written": "",
            "user_input": ""
        }],
        "interaction_summary": "",  # ✅ Initialize an empty conversation summary
        "decision": [],
        "output": [],
        "start_time": time.time(),  # ✅ Store session start time
        "duration": 0  # ✅ Initialize duration
    }

    return apply_cors(jsonify({
        "session_id": session_id,
        "message": "Interview started!",
        "question": question["question_text"],  # ✅ Send question to frontend
        "example": question.get("example", ""),  # ✅ Send example to frontend
        "constraint": question.get("reservations", ""),
        "question_id": question["id"],  # ✅ Include question ID for tracking
        "start_time": session_store[session_id]["start_time"]  # ✅ Send start time to frontend
    }))

@app.route('/respond', methods=['POST', 'OPTIONS'])
def respond():
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    data = request.json
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        return apply_cors(jsonify({"error": "Invalid session_id"}), 400)

    user_input = data.get("user_input", "")
    new_code_written = data.get("new_code_written", "")
    current_state = session_store[session_id]
    prev_summary = current_state.get("interaction_summary", "")

    next_input = {
        "interview_question": current_state["input"][0]["interview_question"],
        "summary_of_past_response": prev_summary,
        "new_code_written": new_code_written,
        "user_input": user_input
    }

    @stream_with_context
    def generate_stream():
        next_state = app_graph.invoke({
            "input": [next_input],
            "decision": [],
            "output": []
        })

        full_response = next_state.get("output", ["No response"])[-1]
        
        sentence_buffer = ""
        for word in full_response.split():
            sentence_buffer += word + " "
            yield word + " "
            time.sleep(0.01)

            # 🔍 Sentence boundary detection (basic)
            if re.search(r'[.!?]["\']?\s*$', sentence_buffer):
                # 👇 This could push sentence to TTS queue (e.g., frontend or SSE)
                yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"
                sentence_buffer = ""

        # Edge case: leftover fragment
        if sentence_buffer.strip():
            yield f"[TTS_START]{sentence_buffer.strip()}[TTS_END]"

        # Update session
        new_summary = summarize_conversation(session_id, user_input, new_code_written)
        session_store[session_id] = {
            "input": [next_input],
            "decision": next_state["decision"],
            "output": [full_response],
            "interaction_summary": new_summary,
            "start_time": current_state.get("start_time"),
            "duration": current_state.get("duration", 0)
        }

    return apply_cors(Response(generate_stream(), mimetype='text/plain'))

@app.route('/questions', methods=['GET', 'OPTIONS'])
def fetch_questions():
    """API to get all questions."""
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    summaries = get_all_summaries()
    return apply_cors(jsonify(summaries))

@app.route('/question/<int:question_id>', methods=['GET', 'OPTIONS'])
def fetch_question_by_id(question_id):
    """API to get a specific question by ID."""
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    question = get_question_by_id(question_id)
    if question:
        return apply_cors(jsonify(question))
    
    return apply_cors(jsonify({"error": "Question not found"}), 404)

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    """Handles user login and sets a secure HTTP-only cookie."""
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        return apply_cors(response)

    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        
        user = get_user(email, password)  # Ensure this function exists
        if not user:
            response = jsonify({"error": "Invalid credentials"})
            response.status_code = 401
            return apply_cors(response)

        # Generate JWT token
        token = jwt.encode(
            {"user_id": user["id"], "email": user["email"], "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)},
            SECRET_KEY, 
            algorithm="HS256"
        )

        # Create HTTP-only cookie response
        response = make_response(jsonify({"message": "Login successful"}))
        response.set_cookie(
            "auth_token",
            token,
            httponly=True,
            secure=True,  # ✅ Allow cookies over HTTP (only for testing)
            samesite="None",
            max_age=24 * 60 * 60
        )
        return apply_cors(response)

    except Exception as e:
        print(f"❌ ERROR in /login: {e}")
        response = jsonify({"error": "Internal Server Error"})
        response.status_code = 500
        return apply_cors(response)

@app.route('/register', methods=['POST', 'OPTIONS'])
def register():
    """Handles user registration with correct CORS headers for preflight requests."""
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.status_code = 204  # ✅ FIXED: No Content
        return apply_cors(response)

    try:
        data = request.get_json()
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            response = jsonify({"error": "All fields are required"})
            response.status_code = 400
            return apply_cors(response)

        success = create_user(name, email, password)

        if success:
            response = jsonify({"message": "User registered successfully!"})
            response.status_code = 201
            return apply_cors(response)
        else:
            response = jsonify({"error": "Email already registered"})
            response.status_code = 409
            return apply_cors(response)

    except Exception as e:
        print(f"❌ ERROR in /register: {e}")
        response = jsonify({"error": "Internal Server Error"})
        response.status_code = 500
        return apply_cors(response)  # ✅ Ensure CORS headers are applied
    
@app.route('/check-auth', methods=['GET', 'OPTIONS'])
def check_auth():
    """Checks if the user is authenticated based on the auth_token cookie."""
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    token = request.cookies.get("auth_token")
    if not token:
        response = jsonify({"error": "Not authenticated"})
        response.status_code = 401
        return apply_cors(response)


    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return apply_cors(jsonify({"message": "Authenticated"}))  # ✅ 200 OK
    except jwt.ExpiredSignatureError:
        return apply_cors(jsonify({"error": "Token expired"}), 401)
    except jwt.InvalidTokenError:
        return apply_cors(jsonify({"error": "Invalid token"}), 401)
    
@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    """Logs the user out by clearing the auth_token cookie."""
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    response = jsonify({"message": "Logout successful"})
    response.set_cookie(
        "auth_token", 
        "",  # Clear the cookie value
        expires=0,  # Immediately expire the cookie
        httponly=True, 
        secure=True,  # Keep True for HTTPS, False for local testing if needed
        samesite="Strict",
        domain="fypbackend-b5gchph9byc4b8gt.canadacentral-01.azurewebsites.net",  # Use the same domain as when the cookie was set
        path="/"  # Ensure the path matches
    )
    return apply_cors(response)

@app.route('/partial-eval', methods=['POST', 'OPTIONS'])
def partial_eval():
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    data = request.json
    session_id = data.get("session_id")
    if not session_id or session_id not in session_store:
        return apply_cors(jsonify({"error": "Invalid session_id"}), 400)

    # Now call your partial evaluation logic
    partial_evaluation = evaluate_response_partially(session_id)

    return apply_cors(jsonify({
        "partial_evaluation": partial_evaluation
    }))

@app.route("/final_evaluation", methods=["POST", "OPTIONS"])
def final_evaluation():
    """
    Uses the entire conversation summary, user code, and partial_eval_history
    to produce a final detailed evaluation with the existing 'evaluation_agent'.
    """
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    data = request.json
    session_id = data.get("session_id")
    student_id = data.get("student_id")  # ✅ Ensure student_id is received
    question_id = data.get("question_id")  # ✅ Ensure question_id is received    
    
    print("Received session_id:", session_id)
    print("Current session_store keys:", session_store.keys())  # Debugging prin    

    if not session_id or session_id not in session_store:
        response = jsonify({"error": "Invalid session_id"})
        response.status_code = 400
        return apply_cors(response)
    
    if not student_id or not question_id:
        response = jsonify({"error": "Missing student_id or question_id"})
        response.status_code = 400
        return apply_cors(response)

    # Gather the entire conversation summary
    entire_summary = session_store[session_id].get("interaction_summary", "")
    final_code = session_store[session_id]["input"][-1].get("new_code_written", "")
    partial_eval_history = session_store[session_id].get("partial_eval_history", [])

    # Build a state object for the final evaluation
    final_input = {
        "student_id": student_id,  # ✅ Ensure student_id is included
        "question_id": question_id,  # ✅ Ensure question_id is included
        "interview_question": session_store[session_id]["input"][0]["interview_question"],
        "summary_of_past_response": entire_summary,
        "new_code_written": final_code,
        "partial_eval_history": partial_eval_history
    }

    eval_state = {
        "input": [final_input],
        "decision": [],
        "output": []
    }


    # Call your evaluation_agent to produce the final evaluation
    updated_eval_state = evaluation_agent(eval_state)
    final_result = updated_eval_state.get("evaluation_result", {})

    # Optionally store the final evaluation in the session
    session_store[session_id]["final_evaluation"] = final_result

    try:
    # Strip down the result to just the 2 keys you care about
        feedback_only = {
            "final_evaluation": final_result.get("final_evaluation", {}),
            "detailed_feedback": final_result.get("detailed_feedback", {})
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


    return apply_cors(jsonify({"final_evaluation": final_result}))

@app.route('/me', methods=['GET'])
def get_user_email():
    token = request.cookies.get("auth_token")
    if not token:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"email": decoded["email"]})
    except Exception as e:
        return jsonify({"error": "Invalid token"}), 401

from db_connector import get_user_feedback_history

@app.route('/user-history', methods=['GET', 'OPTIONS'])
def user_history():
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    token = request.cookies.get("auth_token")
    if not token:
        return apply_cors(jsonify({"error": "Not authenticated"}), 401)

    try:
        # Decode the JWT token
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = decoded_token.get("user_id")

        if not user_id:
            return apply_cors(jsonify({"error": "User ID not found in token"}), 401)

        # 🔍 Fetch the user's feedback history
        feedback_entries = get_user_feedback_history(str(user_id))
        print(f"✅ Retrieved {len(feedback_entries)} feedback entries for user_id {user_id}")

        return apply_cors(jsonify({"feedback": feedback_entries}))

    except jwt.ExpiredSignatureError:
        return apply_cors(jsonify({"error": "Token expired"}), 401)

    except jwt.InvalidTokenError:
        return apply_cors(jsonify({"error": "Invalid token"}), 401)

    except Exception as e:
        print("❌ Unexpected error in /user-history:", e)
        return apply_cors(jsonify({"error": "Internal server error"}), 500)

@app.route('/elevenlabs_tts', methods=['POST', 'OPTIONS'])
def elevenlabs_tts():
    """Handles ElevenLabs TTS with proper CORS support."""
    ELEVENLABS_API_KEY = os.env("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.env("ELEVENLABS_VOICE_ID")
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.status_code = 204  # ✅ Respond correctly to preflight
        return apply_cors(response)

    text = request.json.get("text", "")
    if not text:
        return apply_cors(jsonify({"error": "No text provided"}), 400)

    try:
        def generate():
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
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
                    print(f"Error fetching audio from ElevenLabs: {r.status_code}")
                    return apply_cors(Response("Failed to get audio from ElevenLabs.", status=500, mimetype='text/plain'))

                # Log for diagnostics
                print("Started streaming audio...")

                # Stream audio chunks to the frontend continuously
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk  # Yield each chunk of the audio stream

                print("Audio streaming complete.")

        # Return the streaming response with the audio/mpeg MIME type
        return apply_cors(Response(generate(), mimetype='audio/mpeg'))

    except Exception as e:
        # Catch any exception and log it, then return a proper error response
        print(f"Error during TTS generation: {e}")
        return apply_cors(Response("Internal server error during TTS generation.", status=500, mimetype='text/plain'))

@app.route('/openai_tts', methods=['POST', 'OPTIONS'])
def openai_tts():
    """Handles TTS generation using OpenAI API."""
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.status_code = 204
        return apply_cors(response)

    text = request.json.get("text", "")
    if not text:
        return apply_cors(jsonify({"error": "No text provided"}), 400)

    try:
        # Use the Azure OpenAI client to generate TTS audio
        response = tts_hd_client.audio.speech.create(
            model="tts-hd",  # Choose the appropriate TTS model (e.g., "tts-hd" for high quality)
            voice="nova",    # Choose the voice (this can vary based on the available voices)
            input=text
        )

        # Check if the response is valid and contains the audio content
        if response:
            audio_stream = response.content  # Get the audio data stream directly

            def generate_audio_stream():
                yield audio_stream  # Yield the audio stream to the frontend

            return apply_cors(Response(generate_audio_stream(), mimetype='audio/mpeg'))

        else:
            print("Error: No audio content returned from OpenAI.")
            return apply_cors(Response("Failed to generate audio", status=500, mimetype='text/plain'))

    except Exception as e:
        print(f"Error during TTS generation: {e}")
        return apply_cors(Response("Internal server error during TTS generation.", status=500, mimetype='text/plain'))

@app.route('/gpt4o_realtime_tts', methods=['POST', 'OPTIONS'])
def gpt4o_realtime_tts():
    """Handles TTS generation using gpt-4o-mini-realtime-preview from Azure Foundry."""
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.status_code = 204
        return apply_cors(response)

    text = request.json.get("text", "")
    if not text:
        return apply_cors(jsonify({"error": "No text provided"}), 400)

    try:
        # Use Azure OpenAI client to generate TTS audio using GPT-4o mini real-time model
        response = realtime_client.audio.speech.create(
            model="gpt-4o-mini-realtime-preview",  # ✅ Your custom Foundry model
            voice="Alloy",                          # ✅ Compatible voice
            input=text
        )

        # Stream audio back to frontend
        if response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                response.stream_to_file(f.name)
                temp_path = f.name

            return apply_cors(send_file(temp_path, mimetype="audio/mpeg"))

        else:
            print("Error: No audio content returned from GPT-4o mini realtime model.")
            return apply_cors(Response("Failed to generate audio", status=500, mimetype='text/plain'))

    except Exception as e:
        print(f"Error during GPT-4o realtime TTS generation: {e}")
        return apply_cors(Response("Internal server error during TTS generation.", status=500, mimetype='text/plain'))

# google_tts_client = texttospeech.TextToSpeechClient()
# @app.route("/google_tts", methods=["POST", "OPTIONS"])
# def google_tts():
#     """Generate speech using Google Cloud Text-to-Speech API."""
#     if request.method == "OPTIONS":
#         response = jsonify({"message": "CORS Preflight OK"})
#         response.status_code = 204
#         return apply_cors(response)


#     data = request.get_json()
#     text = data.get("text", "")
#     if not text:
#         return apply_cors(jsonify({"error": "No text provided"}), 400)

#     try:
#         synthesis_input = texttospeech.SynthesisInput(text=text)

#         voice = texttospeech.VoiceSelectionParams(
#             language_code="en-US",
#             name="en-US-Chirp3-HD-Orus",
#             ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#         )

#         audio_config = texttospeech.AudioConfig(
#             audio_encoding=texttospeech.AudioEncoding.MP3
#         )

#         response = google_tts_client.synthesize_speech(
#             input=synthesis_input,
#             voice=voice,
#             audio_config=audio_config
#         )

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out:
#             out.write(response.audio_content)
#             temp_path = out.name

#         return apply_cors(send_file(temp_path, mimetype="audio/mpeg"))

#     except Exception as e:
#         print(f"❌ Google TTS Error: {e}")
#         return apply_cors(Response("Internal server error during Google TTS.", status=500, mimetype="text/plain"))

@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_audio():
    if request.method == "OPTIONS":
        return apply_cors(jsonify({"message": "CORS Preflight OK"}))

    file = request.files.get("audio")
    if not file:
        return apply_cors(jsonify({"error": "No audio file provided"}))

    try:
        audio_bytes = file.read()

        result = whisper_client.audio.transcriptions.create(
            model="whisper",
            file=("audio.webm", audio_bytes),
            language="en"
        )
        print("✅ Whisper transcription result:", result.text)

        return apply_cors(jsonify({"transcript": result.text}))
    except Exception as e:
        print("❌ Whisper transcription error:", e)
        response = apply_cors(jsonify({"error": str(e)}))
        response.status_code = 500
        return response

@app.route('/test')
def test():
    return "It works on Azura! " + os.getenv("AZURE_SPEECH_TTS_KEY")

@app.route("/azure_tts", methods=["POST", "OPTIONS"])
def azure_tts():
    """Generate speech using Azure Speech SDK and stream back MP3."""
    AZURE_TTS_KEY = os.getenv("AZURE_SPEECH_TTS_KEY")
    AZURE_TTS_REGION = "eastus"

    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.status_code = 204
        return apply_cors(response)

    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return apply_cors(jsonify({"error": "No text provided"}), 400)

    # ✅ Log the sentence being synthesized
    print(f"🗣️ Azure TTS requested for sentence: {repr(text)}")

    try:
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_TTS_KEY"), region=AZURE_TTS_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz128KBitRateMonoMp3
        )

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            cancellation_details = speechsdk.CancellationDetails(result)
            print("❌ Azure TTS CANCELED:")
            print("Reason:", cancellation_details.reason)
            print("ErrorDetails:", cancellation_details.error_details)
            raise Exception(f"TTS failed: {cancellation_details.reason} - {cancellation_details.error_details}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out:
            out.write(result.audio_data)
            temp_path = out.name

        return apply_cors(send_file(temp_path, mimetype="audio/mpeg"))

    except Exception as e:
        print(f"❌ Azure TTS Error: {e}")
        return apply_cors(Response("Internal server error during Azure TTS.", status=500, mimetype="text/plain"))

@app.route("/azure_tts_debug", methods=["POST"])
def azure_tts_debug():
    try:
        AZURE_TTS_KEY = os.getenv("AZURE_SPEECH_TTS_KEY")
        AZURE_TTS_REGION = os.getenv("AZURE_SPEECH_TTS_REGION", "eastus")

        print("🔍 Using key:", AZURE_TTS_KEY[:5] + "..." if AZURE_TTS_KEY else "❌ MISSING")
        print("🔍 Using region:", AZURE_TTS_REGION)

        text = "Hello world"  # Short and safe
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_TTS_KEY, region=AZURE_TTS_REGION)
        speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = speechsdk.CancellationDetails(result)
            print("❌ Cancelled:", details.reason)
            print("❌ Error details:", details.error_details)
            return "TTS failed", 500

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as out:
            out.write(result.audio_data)
            path = out.name

        return send_file(path, mimetype="audio/mpeg")

    except Exception as e:
        print("❌ Exception:", e)
        return "Internal error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)