import mysql.connector
import random
from openai import AzureOpenAI
import bcrypt
import json
import os
from dotenv import load_dotenv

# MySQL Database Configuration
DB_CONFIG = {
    "host": "aiviewmysql.mysql.database.azure.com",
    "user": "aiview",
    "password": "#PRASAD SHUBHANGAM RAJESH#123",
    "database": "ai_interview",
    "ssl_disabled": False  # 👈 Required for Azure
}

def get_random_question():
    """Fetch a random interview question from the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT id, question_text FROM questions ORDER BY RAND() LIMIT 1;")
        question = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return question if question else None
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return None

def get_all_questions():
    """Fetch all questions from the database."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question_text FROM questions")
    questions = cursor.fetchall()
    cursor.close()
    conn.close()
    return questions

def get_all_summaries():
    """Fetch all summaries from the database."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, summary, leetcode_link, difficulty FROM questions")
    summaries = cursor.fetchall()
    cursor.close()
    conn.close()
    return summaries

def get_question_by_id(question_id):
    """Fetch a specific question by ID."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, question_text, example, reservations FROM questions WHERE id = %s", (question_id,))
    question = cursor.fetchone()
    cursor.close()
    conn.close()
    return question


def check_password(input_password, stored_hashed_password):
    """Check if the input password matches the stored hashed password."""
    return bcrypt.checkpw(input_password.encode('utf-8'), stored_hashed_password.encode('utf-8'))

def get_user(email, password):
    """Retrieve user by email and verify password securely."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    # Fetch user details (only email and hashed password)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    # If user exists, verify password
    if user and check_password(password, user["password"]):
        return user  # Return user details if password is correct

def get_user_feedback_history(user_id: str):
    """Returns all feedback history for a given user ID."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT 
                f.question_id,
                q.question_text,
                f.final_evaluation,
                f.detailed_feedback
            FROM evaluations f
            JOIN questions q ON f.question_id = q.id
            WHERE f.student_id = %s
            ORDER BY f.timestamp DESC
        """, (user_id,))

        rows = cursor.fetchall()

        feedback_entries = []
        for row in rows:
            feedback_entries.append({
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "final_evaluation": json.loads(row["final_evaluation"]),
                "detailed_feedback": json.loads(row["detailed_feedback"])
            })

        return feedback_entries

    except Exception as e:
        print(f"❌ Error in get_user_feedback_history: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# write a genaAI fucntion that will help summarise each question into less than 5 words to be used as a title for the question
# the function will take in the question and return a summary of the question

# Initialize Azure OpenAI client
client = AzureOpenAI(
  azure_endpoint=os.getenv("OPENAI_ENDPOINT"), 
  api_key=os.getenv("OPENAI_SECRETKEY"),  
  api_version="2024-02-01"
)

def ensure_summary_column():
    """Ensure the 'summary' column exists in the 'questions' table."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Add summary column if it doesn't exist
    cursor.execute("""
        ALTER TABLE questions 
        ADD COLUMN summary VARCHAR(255) DEFAULT NULL;
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

def generate_summary(question_text):
    """Generate a short summary of a question using OpenAI's GPT model."""
    prompt = f"This is the question: '{question_text}'. Provide a concise summary in less than 5 words."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def update_question_summaries():
    """Fetch questions, generate summaries, and update the database."""
    ensure_summary_column()
    
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    questions = get_all_questions()

    for question in questions:
        question_id = question["id"]
        question_text = question["question_text"]
        
        # Generate summary
        summary = generate_summary(question_text)
        
        # Update database with summary
        cursor.execute("""
            UPDATE questions 
            SET summary = %s 
            WHERE id = %s
        """, (summary, question_id))
        
        print(f"Updated Question ID {question_id}: {summary}")

    conn.commit()
    cursor.close()
    conn.close()

def get_user_feedback_history(user_id: str):
    """Returns all feedback history for a given user ID by reading from the users table."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # Step 1: Fetch user's questions_completed and past_feedback
        cursor.execute("SELECT questions_completed, past_feedback FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()

        if not user_data:
            print(f"❌ No user found with ID {user_id}")
            return []

        # Step 2: Parse the stored JSON safely
        completed_ids = []
        if user_data["questions_completed"]:
            try:
                completed_ids = json.loads(user_data["questions_completed"])
            except Exception as e:
                print("⚠️ Failed to parse questions_completed:", e)

        feedback_map = {}
        if user_data["past_feedback"]:
            try:
                feedback_map = json.loads(user_data["past_feedback"])
            except Exception as e:
                print("⚠️ Failed to parse past_feedback:", e)

        if not completed_ids:
            return []

        # Step 3: Fetch corresponding questions from DB
        format_strings = ','.join(['%s'] * len(completed_ids))
        cursor.execute(f"""
            SELECT id, question_text FROM questions 
            WHERE id IN ({format_strings})
        """, tuple(completed_ids))

        question_map = {row["id"]: row["question_text"] for row in cursor.fetchall()}

        # Step 4: Build final feedback list
        feedback_entries = []
        for qid in completed_ids:
            if str(qid) in feedback_map:
                entry = feedback_map[str(qid)]
                feedback_entries.append({
                    "question_id": qid,
                    "question_text": question_map.get(qid, "Unknown Question"),
                    "final_evaluation": entry.get("final_evaluation", {}),
                    "detailed_feedback": entry.get("detailed_feedback", {})
                })

        return feedback_entries

    except Exception as e:
        print(f"❌ Error in get_user_feedback_history: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def update_user_progress_by_email(email: str, question_id: int, feedback_json: dict):
    """
    Update the user's questions_completed and past_feedback by email.
    """
    import json
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    try:
        # Fetch user by email
        cursor.execute("SELECT questions_completed, past_feedback FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            raise Exception("User not found")

        # --- SAFELY PARSE 'questions_completed' ---
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

        # Avoid duplicates
        if question_id not in current_questions:
            current_questions.append(question_id)

        # --- SAFELY PARSE 'past_feedback' ---
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

        current_feedback[str(question_id)] = feedback_json

        # --- Write updates back ---
        cursor.execute("""
            UPDATE users 
            SET questions_completed = %s, past_feedback = %s 
            WHERE email = %s
        """, (json.dumps(current_questions), json.dumps(current_feedback), email))

        conn.commit()
        return True

    except Exception as e:
        print(f"❌ Error updating user progress: {e}")
        return False

    finally:
        cursor.close()
        conn.close()
