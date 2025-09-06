import os
import json

# Directory containing JSON files
JSON_DIR = "leetcode/Cleaned_LeetCode_Questions"  # Update this path if necessary

def format_title(filename):
    """
    Converts a filename into a formatted title by removing underscores and capitalizing words.
    Example: "longest_common_prefix.json" → "Longest Common Prefix"
    """
    title = os.path.splitext(filename)[0].replace("_", " ")
    return title.title()  # Capitalizes each word

def generate_leetcode_url(filename):
    """
    Generates a LeetCode problem URL from the filename by replacing underscores with hyphens.
    Example: "longest_common_prefix.json" → "https://leetcode.com/problems/longest-common-prefix/"
    """
    problem_name = os.path.splitext(filename)[0].replace("_", "-")
    return f"https://leetcode.com/problems/{problem_name}/"

def update_json_files(directory):
    """
    Processes all JSON files in the specified directory:
    - Adds a 'title' field formatted from the filename.
    - Adds a 'leetcode_url' field based on the filename.
    """
    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' does not exist.")
        return
    
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    if not json_files:
        print("⚠️ No JSON files found in the directory.")
        return

    for filename in json_files:
        file_path = os.path.join(directory, filename)
        
        try:
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Add/Update the 'title' and 'leetcode_url'
            data["title"] = format_title(filename)
            data["leetcode_url"] = generate_leetcode_url(filename)

            # Write updated JSON back to the file
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            print(f"✅ Updated {filename} with title: {data['title']} and URL: {data['leetcode_url']}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

# Run the function on the target directory
update_json_files(JSON_DIR)
