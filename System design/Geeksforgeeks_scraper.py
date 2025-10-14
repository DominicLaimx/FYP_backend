import requests
from bs4 import BeautifulSoup
import os
from typing import List, Dict
from tqdm import tqdm  # Progress bar
import re
import json
import mysql.connector
import random
import bcrypt
import csv
from dotenv import load_dotenv
import re
import pandas as pd
from openai import AzureOpenAI

URL = "https://www.geeksforgeeks.org/system-design/most-commonly-asked-system-design-interview-problems-questions/"

# MySQL Database Configuration
DB_CONFIG = {
    "host": "aiviewmysql.mysql.database.azure.com",
    "user": "aiview",
    "password": "#PRASAD SHUBHANGAM RAJESH#123",
    "database": "ai_interview",
    "ssl_disabled": False  # 👈 Required for Azure
}

load_dotenv(override=True)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Initialize Azure OpenAI client
llm = AzureChatOpenAI(model="gpt-4o-mini", temperature=0)

def clean_question_with_ai(question_text):
    """
    Uses Azure OpenAI to clean and reformat the question into a strict dictionary format.
    """
    # prompt = f"""
    # You are an AI assistant that reformats system design questions into a precise dictionary structure. 
    # The output must be in strict JSON format with **exact key names**: 

    # {{
    #     "summary": "A one-liner summary of the question (max 5 words)",
    #     "question": "A clearly stated problem description",
    #     "example": "Clearly written input-output examples",
    #     "constraint": "Clearly defined constraints",
    # }}

    # Here is the raw question:
    # -----
    # {question_text}
    # -----
    # Now, return only the JSON object with **no additional text**.
    # """
    prompt = f"""
        You are an AI assistant that reformats system design questions into a structured format.
        The output must be in strict JSON format with these exact keys:

        {{
            "description": "A clear problem statement that describes what needs to be designed",
            "example": "A realistic usage scenario showing how the system would be used",
            "constraints": "Key considerations like scale, performance expectations, or basic requirements"
        }}

        Guidelines:
        - Keep descriptions brief and focused on WHAT needs to be built, not HOW
        - Examples should illustrate use cases, not implementation details
        - Constraints should be vague but nudge readers to think about them
        - Avoid suggesting specific technologies, architectures, or design patterns

        Here is the raw question:
        -----
        {question_text}
        -----

        Return only the JSON object with no additional text.
        """

    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "system", "content": prompt}]
    # )
    response = llm.invoke(prompt)
    try:
        response_dict = json.loads(response.content)
        
        # Now you can access the dictionary keys as you would normally
        # print("Successfully extracted dictionary:")
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("The content attribute does not contain a valid JSON string.")
    return response_dict

def scrape_questions(url):
    """
    Scrapes the webpage and extracts only the questions that match `leetcode_top_150`.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise error if request fails

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Dictionary to store extracted questions
    questions_dict = {}

    # Find all title divs
    # title_divs = soup.find_all("span")
    # print(title_divs)
    results = []

    tr_elements = soup.find_all('tr')  # Find all rows

    for tr in tr_elements:
        # Find all span tags inside this tr that are NOT "Read"
        spans = [span for span in tr.find_all('span') if span.get_text(strip=True).lower() != 'read']

        # Find all <a> tags with href in this tr
        a_tags = tr.find_all('a', href=True)

        # Extract each span text and match href if possible
        for span in spans:
            span_text = span.get_text(strip=True)

            # Find if this span is inside an <a> tag and get href else None
            a_tag_parent = span.find_parent('a')
            href = a_tag_parent['href'] if a_tag_parent else None

            # If no href on span's a tag, check the separate a_tags in this tr,
            # assuming one-to-one matching by position or other logic if needed.
            if not href and a_tags:
                href = a_tags[0]['href']  # or logic to find correct href
            if href:
                results.append((span_text, href))

    # for text, link in results:
    #     print(f'Text: {text}, Href: {link}')
    mydf = pd.DataFrame(results, columns=['question', 'href'])
    mydf.to_csv("./geeksforgeeks_system_design_qns.csv", index=False)
    return results

def save_questions(question,link,datadict):
    """
    Updates the category for a question if the title exists in the table.
    
    Args:
        category (str): The category to update.
        title (str): The title of the question to check and update.
        db_config (dict): Database connection configuration.
    """
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        insert_query = """
        
    INSERT INTO questions (question_text,summary,example,reservations,leetcode_link,title,question_type)
    VALUES (%s, %s, %s, %s, %s, %s,%s);
    
        """
        cursor.execute(insert_query, (datadict["description"],question, datadict["example"], datadict["constraints"],link,question, "system_design"))
        conn.commit()  # Commit the transaction to save changes
        print("Successfully added row.")
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        # Rollback the transaction on error
        if conn and conn.is_connected():
            conn.rollback()
            
    finally:
        # Ensure resources are always closed
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    # extracted_questions = scrape_questions(URL)
    mydf = pd.read_csv("./geeksforgeeks_system_design_qns.csv")
    for row in mydf.itertuples(index=False):
        resdict = clean_question_with_ai(row.question)
        save_questions(row.question, row.href, resdict)
    # print(clean_question_with_ai(mydf.loc[2]["question"]))