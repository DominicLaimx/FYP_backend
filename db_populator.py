import os
import json
import pymysql
from tqdm import tqdm  # Progress bar

# ✅ MySQL Connection Details (Azure)
DB_HOST = "aiviewmysql.mysql.database.azure.com"
DB_USER = "aiview"
DB_PASSWORD = "#PRASAD SHUBHANGAM RAJESH#123"
DB_NAME = "ai_interview"

# ✅ Optional SSL Settings for Azure (pymysql ignores ssl_disabled=False)
# You can enable this if Azure enforces it, but it's not always required for pymysql.
# If needed, use:
# ssl = {"ssl": {"ssl_disabled": False}}

# Directory containing JSON files
JSON_DIR = "leetcode/Cleaned_LeetCode_Questions/"

# ✅ Connect to Azure MySQL using pymysql
connection = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        # Get a list of all JSON files
        json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

        # Initialize progress bar
        for filename in tqdm(json_files, desc="Populating Database", unit="file"):
            file_path = os.path.join(JSON_DIR, filename)
            
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Extract necessary fields (force conversion to string)
            title = str(data.get("title", ""))  # ✅ Extract Title
            leetcode_link = str(data.get("leetcode_url", ""))  # ✅ Extract LeetCode Link
            summary = str(data.get("summary", ""))
            question_text = str(data.get("question", ""))
            example = str(data.get("example", ""))
            constraints = str(data.get("constraint", ""))
            difficulty = str(data.get("difficulty", "Unknown"))  # ✅ Extract Difficulty

            # Ensure that dictionaries are converted to JSON strings
            for key in ["title", "leetcode_link", "summary", "question_text", "example", "constraints", "difficulty"]:
                if isinstance(locals()[key], dict):
                    locals()[key] = json.dumps(locals()[key])

            # Insert into MySQL table (ID is auto-generated)
            sql = """
            INSERT INTO questions (title, summary, question_text, example, reservations, leetcode_link, difficulty)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cursor.execute(sql, (title, summary, question_text, example, constraints, leetcode_link, difficulty))

            tqdm.write(f"✅ Inserted {filename} with difficulty: {difficulty}")

        # Commit all changes to the database
        connection.commit()
        print("✅ Database populated successfully with title, LeetCode link, and difficulty!")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    connection.close()
