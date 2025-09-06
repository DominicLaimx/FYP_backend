import os
import json
import pandas as pd
from tqdm import tqdm  # Progress bar

# Load the Hugging Face dataset
df = pd.read_json("hf://datasets/greengerong/leetcode/leetcode-train.jsonl", lines=True)

# Normalize titles from dataset (replace spaces with hyphens for better matching)
df["formatted_title"] = df["title"].str.replace(" ", "-", regex=False).str.lower()

# Directory containing JSON files
JSON_DIR = "Cleaned_LeetCode_Questions"
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

# Iterate over JSON files with a progress bar
for filename in tqdm(json_files, desc="Updating JSON Files", unit="file"):
    file_path = os.path.join(JSON_DIR, filename)

    # Load JSON data
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract the title and normalize it
    title = data.get("title", "").strip()
    formatted_title = title.replace(" ", "-").lower()  # Convert to slug format

    # Match with Hugging Face dataset to get starter code from the "python" column
    starter_code_row = df[df["formatted_title"] == formatted_title]

    if not starter_code_row.empty:
        starter_code = starter_code_row.iloc[0].get("python")
        if starter_code is not None:
            data["starter_code"] = starter_code  # Add starter code to JSON

            # Save the updated JSON file
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            tqdm.write(f"✅ Updated {filename} with starter code.")
        else:
            tqdm.write(f"❌ No starter code found in the 'python' column for: {title}")
    else:
        tqdm.write(f"❌ No matching record found for: {title}")
