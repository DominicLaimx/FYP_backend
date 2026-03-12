import mysql.connector
import random
from openai import AzureOpenAI
import bcrypt
import json
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

# MySQL Database Configuration
DB_CONFIG = {
    "host": "aiviewmysql.mysql.database.azure.com",
    "user": "aiview",
    "password": "#PRASAD SHUBHANGAM RAJESH#123",
    "database": "ai_interview",
    "ssl_disabled": False
}

from mysql.connector import pooling

db_pool = None

def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=3,
            **DB_CONFIG
        )
        print("db_pool None")
    return db_pool

def initialise_db_pool():
    pool = get_db_pool()
    return "success"

CONTAINER_NAME = "aiviewvideos"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

def get_upload_url(filename: str):
    """Generate a short-lived SAS URL for the frontend to upload directly."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    account_name = blob_service_client.account_name
    account_key = blob_service_client.credential.account_key

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=CONTAINER_NAME,
        blob_name=filename,
        permission=BlobSasPermissions(write=True, create=True),
        expiry=datetime.utcnow() + timedelta(minutes=30),
        account_key=account_key,
    )

    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
    upload_url = f"{blob_client.url}?{sas_token}"
    return {"upload_url": upload_url}


def get_random_question(question_type):
    """Fetch a random interview question from the MySQL database."""
    try:
        pool = get_db_pool()
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """SELECT id, title, summary, leetcode_link, difficulty, category
                   FROM questions WHERE question_type = %s ORDER BY RAND() LIMIT 1;"""
        cursor.execute(query, (question_type,))
        question = cursor.fetchone()
        cursor.close()
        conn.close()
        return question if question else None
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None


def get_all_questions():
    """Fetch all questions from the database."""
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question_text FROM questions")
    questions = cursor.fetchall()
    cursor.close()
    conn.close()
    return questions


def get_all_summaries(question_type):
    """Fetch all summaries from the database."""
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    query = """SELECT id, title, summary, leetcode_link, difficulty, category
               FROM questions WHERE question_type = %s ORDER BY difficulty, title"""
    cursor.execute(query, (question_type,))
    summaries = cursor.fetchall()
    cursor.close()
    conn.close()
    return summaries


def get_question_by_id(question_id):
    """Fetch a specific question by ID."""
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, question_text, example, reservations, difficulty FROM questions WHERE id = %s",
        (question_id,)
    )
    question = cursor.fetchone()
    cursor.close()
    conn.close()
    return question


def check_password(input_password, stored_hashed_password):
    """Check if the input password matches the stored hashed password."""
    return bcrypt.checkpw(input_password.encode("utf-8"), stored_hashed_password.encode("utf-8"))


def get_user(email, password):
    """Retrieve user by email and verify password securely."""
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user and check_password(password, user["password"]):
        return user


def get_user_feedback_history(user_id: str):
    """Returns all feedback history for a given user ID by reading from the users table."""
    try:
        pool = get_db_pool()
        conn = pool.get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT questions_completed, past_feedback FROM users WHERE id = %s",
            (user_id,)
        )
        user_data = cursor.fetchone()

        if not user_data:
            print(f"No user found with ID {user_id}")
            return []

        # Parse questions_completed
        completed_ids = []
        if user_data["questions_completed"]:
            try:
                completed_ids = json.loads(user_data["questions_completed"])
            except Exception as e:
                print("Failed to parse questions_completed:", e)

        # Parse past_feedback
        feedback_map = {}
        if user_data["past_feedback"]:
            try:
                feedback_map = json.loads(user_data["past_feedback"])
            except Exception as e:
                print("Failed to parse past_feedback:", e)

        if not completed_ids:
            return []

        # Fetch question texts
        format_strings = ",".join(["%s"] * len(completed_ids))
        cursor.execute(
            f"SELECT id, question_text FROM questions WHERE id IN ({format_strings})",
            tuple(completed_ids)
        )
        question_map = {row["id"]: row["question_text"] for row in cursor.fetchall()}

        # Build feedback list — include recording_url and video_analysis
        feedback_entries = []
        for qid in completed_ids:
            if str(qid) in feedback_map:
                entry = feedback_map[str(qid)]
                feedback_entries.append({
                    "question_id": qid,
                    "question_text": question_map.get(qid, "Unknown Question"),
                    "final_evaluation": entry.get("final_evaluation", {}),
                    "detailed_feedback": entry.get("detailed_feedback", {}),
                    "total_score": entry.get("total_score"),
                    "overall_assessment": entry.get("overall_assessment"),
                    "hire_likelihood_percent": entry.get("hire_likelihood_percent"),
                    # Populated by save_recording or final_evaluation
                    "recording_url": entry.get("recording_url", ""),
                    "video_analysis": entry.get("video_analysis", {}),
                })

        return feedback_entries

    except Exception as e:
        print(f"Error in get_user_feedback_history: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ---------------------------------------------------------------------------
# Azure OpenAI client (used for question summary generation only)
# ---------------------------------------------------------------------------
client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_SECRETKEY"),
    api_version="2024-02-01",
)


def ensure_summary_column():
    """Ensure the summary column exists in the questions table."""
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("ALTER TABLE questions ADD COLUMN summary VARCHAR(255) DEFAULT NULL;")
    conn.commit()
    cursor.close()
    conn.close()


def generate_summary(question_text):
    """Generate a short summary of a question using GPT."""
    prompt = f"This is the question: '{question_text}'. Provide a concise summary in less than 5 words."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def update_question_summaries():
    """Fetch questions, generate summaries, and update the database."""
    ensure_summary_column()
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor()
    questions = get_all_questions()
    for question in questions:
        summary = generate_summary(question["question_text"])
        cursor.execute(
            "UPDATE questions SET summary = %s WHERE id = %s",
            (summary, question["id"])
        )
        print(f"Updated Question ID {question['id']}: {summary}")
    conn.commit()
    cursor.close()
    conn.close()


def update_user_progress_by_email(email: str, question_id: int, feedback_json: dict):
    """
    Update the user's questions_completed and past_feedback by email.
    Merges new feedback with any existing entry for the same question so that
    recording_url saved by save_recording is not overwritten by final_evaluation.
    """
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT questions_completed, past_feedback FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()
        if not user:
            raise Exception("User not found")

        # Parse questions_completed
        raw_q = user["questions_completed"]
        if isinstance(raw_q, str):
            try:
                current_questions = json.loads(raw_q)
                if not isinstance(current_questions, list):
                    current_questions = [current_questions]
            except json.JSONDecodeError:
                current_questions = []
        elif isinstance(raw_q, int):
            current_questions = [raw_q]
        elif raw_q is None:
            current_questions = []
        else:
            current_questions = list(raw_q) if isinstance(raw_q, list) else []

        if question_id not in current_questions:
            current_questions.append(question_id)

        # Parse past_feedback
        raw_fb = user["past_feedback"]
        if isinstance(raw_fb, str):
            try:
                current_feedback = json.loads(raw_fb)
            except json.JSONDecodeError:
                current_feedback = {}
        elif isinstance(raw_fb, dict):
            current_feedback = raw_fb
        else:
            current_feedback = {}

        # Merge: preserve fields from existing entry that aren't in the new payload
        # This means save_recording (which sets recording_url) won't be clobbered
        # by a subsequent final_evaluation call that doesn't include recording_url,
        # and vice versa.
        existing = current_feedback.get(str(question_id), {})
        merged = {**existing, **feedback_json}
        # Explicitly preserve recording_url and video_analysis if new payload omits them
        for preserve_key in ("recording_url", "video_analysis"):
            if not feedback_json.get(preserve_key) and existing.get(preserve_key):
                merged[preserve_key] = existing[preserve_key]

        current_feedback[str(question_id)] = merged

        cursor.execute(
            "UPDATE users SET questions_completed = %s, past_feedback = %s WHERE email = %s",
            (json.dumps(current_questions), json.dumps(current_feedback), email)
        )
        conn.commit()
        return True

    except Exception as e:
        print(f"Error updating user progress: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def delete_history_by_email(email: str, question_id: int):
    """Delete a past interview entry for a user given their email."""
    pool = get_db_pool()
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT questions_completed, past_feedback FROM users WHERE email = %s",
            (email,)
        )
        user = cursor.fetchone()
        if not user:
            raise Exception("User not found")

        raw_q = user["questions_completed"]
        if isinstance(raw_q, str):
            try:
                current_questions = json.loads(raw_q)
                if not isinstance(current_questions, list):
                    current_questions = [current_questions]
            except json.JSONDecodeError:
                current_questions = []
        elif isinstance(raw_q, int):
            current_questions = [raw_q]
        elif raw_q is None:
            current_questions = []
        else:
            current_questions = list(raw_q) if isinstance(raw_q, list) else []

        if question_id in current_questions:
            current_questions.remove(question_id)

        raw_fb = user["past_feedback"]
        if isinstance(raw_fb, str):
            try:
                current_feedback = json.loads(raw_fb)
            except json.JSONDecodeError:
                current_feedback = {}
        elif isinstance(raw_fb, dict):
            current_feedback = raw_fb
        else:
            current_feedback = {}

        current_feedback.pop(str(question_id), None)

        cursor.execute(
            "UPDATE users SET questions_completed = %s, past_feedback = %s WHERE email = %s",
            (json.dumps(current_questions), json.dumps(current_feedback), email)
        )
        conn.commit()
        return True

    except Exception as e:
        print(f"Error deleting history: {e}")
        return False
    finally:
        cursor.close()
        conn.close()
