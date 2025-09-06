import json
from openai import AzureOpenAI
from typing import TypedDict, Annotated, Dict, List
from pydantic import BaseModel
from typing import List
import os
# Initialize Azure OpenAI client
endpoint = os.getenv("OPENAI_ENDPOINT")
key = os.getenv("OPENAI_SECRETKEY")
SECRET_KEY = os.getenv("PWD_SECRET_KEY")

# Initialize Azure OpenAI client
client = AzureOpenAI(
  azure_endpoint=endpoint, 
  api_key=key,  
  api_version="2024-02-01"
)


class EvaluationCategory(BaseModel):
    communication: str
    problem_solving: str
    technical_competency: str
    examples_of_what_went_well: str

class EvaluationSchema(BaseModel):
    student_id: str
    question_id: str
    final_evaluation: EvaluationCategory
    detailed_feedback: EvaluationCategory

def evaluation_agent(state: dict) -> dict:
    """
    Calls GPT to output JSON matching the EvaluationSchema,
    storing it in state["evaluation_result"]. 
    This is an 'intermediate' grading each time the user responds.
    """
    input_data = state["input"][-1]

    # Instruct GPT to output valid JSON that matches our Pydantic schema, no extra keys
    # Make sure the prompt includes the relevant info: question, summary, code
    prompt = f"""
You are an AI evaluation agent for a coding interview I need you to be extremely strict! 
Produce your answer as valid JSON ONLY, matching this schema exactly:

EvaluationSchema:
- student_id (string)
- question_id (string)
- final_evaluation (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)
- detailed_feedback (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)
- feedback and examples of what they said / coded well and what they could've done better

NO extra keys, no markdown.

Context you have:
- Interview Question: {input_data["interview_question"]}
- User's Summary: {input_data["summary_of_past_response"]}
- User's Code: {input_data["new_code_written"]}

Scoring categories:
- "Strong Hire", "Hire", "No Hire", "Strong No Hire"

Only output valid JSON, no code blocks, no quotes around keys besides JSON structure.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that outputs valid JSON for evaluation."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response.choices[0].message.content.strip()
    state["output"] = [raw_text]  # store raw text from GPT in 'output' for reference

    # Parse JSON
    try:
        parsed = json.loads(raw_text)
        # Validate with Pydantic
        evaluation_obj = EvaluationSchema(**parsed)
        state["evaluation_result"] = evaluation_obj.dict()
    except (json.JSONDecodeError, ValueError) as e:
        state["evaluation_result"] = {
            "error": f"Could not parse JSON: {e}",
            "raw_output": raw_text
        }
    return state

def partial_evaluation_agent(state: dict) -> dict:
    """
    Runs on every user response, outputting JSON matching the same schema as the final evaluator.
    Ensures student_id and question_id fields are included.
    """
    input_data = state["input"][-1]

    # Retrieve these from input_data or set to "unknown" if not provided
    student_id = input_data.get("student_id", "unknown")
    question_id = input_data.get("question_id", "unknown")

    # If you keep track of previous partial eval
    previous_eval = input_data.get("previous_partial_eval", {})
    prev_eval_text = json.dumps(previous_eval, indent=2) if previous_eval else "No previous partial evaluation"

    prompt = f"""
You are an AI partial evaluator for a coding interview. 
Output valid JSON only, matching this schema exactly (no extra keys, no markdown):

EvaluationSchema:
- student_id (string)
- question_id (string)
- partial_eval (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)
- detailed_feedback (EvaluationCategory):
    * communication (string)
    * problem_solving (string)
    * technical_competency (string)
    * examples_of_what_went_well (string)

Here is the previous partial evaluation (if any):
{prev_eval_text}

Context:
- Student ID: {student_id}
- Question ID: {question_id}
- Question: {input_data["interview_question"]}
- Summary: {input_data["summary_of_past_response"]}
- Code: {input_data["new_code_written"]}

Scoring: "Strong Hire", "Hire", "No Hire", "Strong No Hire".

Return valid JSON only, including the student_id and question_id.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI that outputs valid JSON for partial evaluation."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response.choices[0].message.content.strip()
    state["output"] = [raw_text]

    # Parse JSON
    try:
        parsed = json.loads(raw_text)
        # Validate with your existing Pydantic schema that requires these fields
        evaluation_obj = EvaluationSchema(**parsed)
        state["partial_evaluation_result"] = evaluation_obj.dict()
    except (json.JSONDecodeError, ValueError) as e:
        state["partial_evaluation_result"] = {
            "error": f"Could not parse JSON: {e}",
            "raw_output": raw_text
        }

    return state


{"4": 
{"student_id": "12345", 
 "question_id": "validate_parentheses", 
 "final_evaluation": 
    {"communication": "Hire", 
    "problem_solving": "Strong Hire", 
    "technical_competency": "Hire", 
    "examples_of_what_went_well": "The solution demonstrates clear understanding of stack-based algorithmic approaches, correct implementation of mappings for accurate parentheses validation, and concise code structure."}, 
    "detailed_feedback": 
        {"communication": "The user explained their thought process clearly and confidently declined an enhancement suggestion while providing reasoning behind their decision.", 
        "problem_solving": "The use of a dictionary for mapping closing brackets to their respective opening brackets and maintaining stack consistency shows excellent problem-solving ability. The user demonstrated an understanding of edge cases such as empty stack handling.",
        "technical_competency": "The code aligns closely with best practices for solving this type of problem. While the solution is efficient, it does not handle non-bracket characters, limiting its versatility.",
        "examples_of_what_went_well": "The implementation is correct, concise, and covers all valid input cases for bracket validation. The stack-based logic is efficient and executed correctly."
        }
    }
}