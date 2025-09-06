import os
from openai import AzureOpenAI
from tqdm import tqdm
import json

# Azure OpenAI Credentials
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")

# Initialize Azure OpenAI client
client = AzureOpenAI(
  azure_endpoint=endpoint, 
  api_key=key,  
  api_version="2024-02-01"
)

# Input and output directories
INPUT_FOLDER = "LeetCode_Top_150_Questions"
OUTPUT_FOLDER = "Cleaned_LeetCode_Questions"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_question_with_ai(question_text):
    """
    Uses Azure OpenAI to clean and reformat the question into a strict dictionary format.
    """
    prompt = f"""
    You are an AI assistant that reformats programming questions into a precise dictionary structure. 
    The output must be in strict JSON format with **exact key names**: 

    {{
        "summary": "A one-liner summary of the question (max 5 words)",
        "question": "A clearly stated problem description",
        "example": "Clearly written input-output examples",
        "constraint": "Clearly defined constraints",
        "followup": "Follow-up question (if any), otherwise empty string"
    }}

    Here is the raw question:
    -----
    {question_text}
    -----
    Now, return only the JSON object with **no additional text**.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

def process_questions():
    """
    Reads each text file from INPUT_FOLDER, processes it with AI, and saves the cleaned version in OUTPUT_FOLDER.
    """
    files = sorted(os.listdir(INPUT_FOLDER))  # Maintain order

    for file_name in tqdm(files, desc="Processing Questions"):
        input_path = os.path.join(INPUT_FOLDER, file_name)
        output_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".txt", ".json"))  # Save as JSON

        # Read the question text
        with open(input_path, "r", encoding="utf-8") as f:
            raw_question = f.read()

        # Clean and format the question using Azure OpenAI
        cleaned_question = clean_question_with_ai(raw_question)

        # Ensure valid JSON output
        try:
            cleaned_dict = json.loads(cleaned_question)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: AI response for {file_name} was not valid JSON.")
            continue

        # Save the cleaned question in JSON format
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_dict, f, indent=4)

    print(f"\n✅ Processed {len(files)} questions. Cleaned files are saved in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    process_questions()