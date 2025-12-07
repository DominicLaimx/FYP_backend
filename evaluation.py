import json
from openai import AzureOpenAI
from typing import TypedDict, Annotated, Dict, List, Optional
from pydantic import BaseModel
import os
import numpy as np
from scipy.stats import spearmanr, kendalltau
from datetime import datetime

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

alignment_evaluator = HumanAlignmentEvaluator(client)

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


class HumanAlignmentEvaluator:
    """
    Human alignment evaluation for your LLM-based interviewer.

    Responsibilities:
    - Convert categorical ratings ("Strong Hire", etc.) to numeric scale.
    - Obtain a human-reference style evaluation via an LLM prompt.
    - Compare your model's evaluation against the human-reference scores.
    - Accumulate scores across many evaluations and compute:
        * Spearman rank correlation (ρ)
        * Kendall-Tau (τ)
        * Pearson correlation (r)
        * RMSE
    These metrics are standard for measuring agreement with human judgments in
    evaluation frameworks like QUEST and G-Eval.[web:26][web:35]
    """

    def __init__(self, client, model_name: str = "gpt-4o"):
        """
        Args:
            client: AzureOpenAI client (or any OpenAI-compatible chat client).
            model_name: Model used to simulate human-like reference judgments.
        """
        self.client = client
        self.model_name = model_name

        self._human_scores: List[float] = []
        self._model_scores: List[float] = []

    def score_to_numeric(self, score_str: str) -> float:
        """
        Map categorical hiring decisions to numeric scale on [1, 4].

        "Strong No Hire" -> 1.0
        "No Hire"        -> 2.0
        "Hire"           -> 3.0
        "Strong Hire"    -> 4.0

        If unknown, return neutral 2.5.
        """
        if not isinstance(score_str, str):
            return 2.5

        score_str = score_str.strip()
        mapping = {
            "Strong No Hire": 1.0,
            "No Hire": 2.0,
            "Hire": 3.0,
            "Strong Hire": 4.0,
        }
        return mapping.get(score_str, 2.5)

    def get_human_evaluation(
        self,
        interview_question: str,
        student_code: str,
        student_summary: str,
        student_id: str,
        question_id: str,
    ) -> Dict:
        """
        Obtain a human-reference style evaluation by prompting the LLM
        as a strict senior interviewer. This acts as the 'human' target.

        Returns:
            {
              "communication": float,
              "problem_solving": float,
              "technical_competency": float,
              "overall": float,
              "raw_eval": {...}   # original categorical JSON
            }
        """

        prompt = f"""
You are a senior technical interviewer. Evaluate the candidate's coding interview performance STRICTLY.

Rate on these four dimensions using ONLY:
"Strong No Hire", "No Hire", "Hire", "Strong Hire".

Output ONLY valid JSON (no markdown, no comments, no extra text), exactly in this form:

{{
  "communication": "...",
  "problem_solving": "...",
  "technical_competency": "...",
  "overall": "..."
}}

Definitions:
- Communication: Clarity of explanation and ability to articulate reasoning.
- Problem Solving: Quality of approach, handling of edge cases, algorithmic thinking.
- Technical Competency: Code correctness, efficiency, use of appropriate constructs.
- Overall: Overall hiring signal considering all dimensions.

Interview Question:
{interview_question}

Student's Code:
{student_code}

Student's Explanation:
{student_summary}

Be strict and honest in your ratings.
"""
      
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict senior technical interviewer. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {
                "communication": 2.5,
                "problem_solving": 2.5,
                "technical_competency": 2.5,
                "overall": 2.5,
                "raw_eval": {"error": "Could not parse human eval JSON", "raw": content},
            }

        comm = self.score_to_numeric(parsed.get("communication", "Hire"))
        ps = self.score_to_numeric(parsed.get("problem_solving", "Hire"))
        tech = self.score_to_numeric(parsed.get("technical_competency", "Hire"))
        overall = self.score_to_numeric(parsed.get("overall", "Hire"))

        return {
            "communication": comm,
            "problem_solving": ps,
            "technical_competency": tech,
            "overall": overall,
            "raw_eval": parsed,
        }
    def compare_single_evaluation(
        self,
        model_evaluation: Dict,
        human_evaluation: Dict,
        student_id: str,
        question_id: str,
    ) -> Dict:
        """
        Compare one model evaluation against the human reference.

        Args:
            model_evaluation: Your EvaluationSchema dict, e.g.:
                {
                  "student_id": "...",
                  "question_id": "...",
                  "final_evaluation": {
                      "communication": "Hire",
                      "problem_solving": "Strong Hire",
                      "technical_competency": "Hire",
                      ...
                  },
                  ...
                }
            human_evaluation: Output of get_human_evaluation(...)
            student_id, question_id: identifiers

        Returns:
            {
              "student_id": str,
              "question_id": str,
              "dimensions": {
                "communication": {
                  "model_score": float,
                  "human_score": float,
                  "difference": float,
                  "agreement": float  # 0–1
                },
                ...
              },
              "overall_alignment_score": float,  # 0–1
              "timestamp": isoformat str
            }
        """
        dims = ["communication", "problem_solving", "technical_competency"]

        model_scores: List[float] = []
        human_scores: List[float] = []
        per_dim = {}

        final_eval = model_evaluation.get("final_evaluation", {})

        for dim in dims:
            model_raw = final_eval.get(dim)
            if model_raw is None:
                continue

            m_score = self.score_to_numeric(model_raw)
            h_score = float(human_evaluation.get(dim, 2.5))

            model_scores.append(m_score)
            human_scores.append(h_score)

            diff = abs(m_score - h_score)
            agreement = max(0.0, 1.0 - (diff / 4.0))

            per_dim[dim] = {
                "model_score": m_score,
                "human_score": h_score,
                "difference": diff,
                "agreement": agreement,
            }

        if per_dim:
            overall_alignment = float(
                np.mean([d["agreement"] for d in per_dim.values()])
            )
        else:
            overall_alignment = 0.0

        self._model_scores.extend(model_scores)
        self._human_scores.extend(human_scores)

        return {
            "student_id": student_id,
            "question_id": question_id,
            "dimensions": per_dim,
            "overall_alignment_score": overall_alignment,
            "timestamp": datetime.now().isoformat(),
        }

    def compute_alignment_metrics(self) -> Dict:
        """
        Compute global alignment metrics across all evaluations seen so far.

        Returns:
            On success:
                {
                  "spearman_rho": float,
                  "spearman_pval": float,
                  "kendall_tau": float,
                  "kendall_pval": float,
                  "pearson_r": float,
                  "rmse": float,
                  "n_samples": int,
                  "alignment_status": "Good" | "Fair" | "Poor"
                }
            If < 2 samples:
                {"error": "Insufficient samples for alignment analysis", "n_samples": int}
        """
        if len(self._human_scores) < 2:
            return {
                "error": "Insufficient samples for alignment analysis",
                "n_samples": len(self._human_scores),
            }

        h = np.array(self._human_scores, dtype=float)
        m = np.array(self._model_scores, dtype=float)

        # Rank-based correlations are standard for human alignment eval in LLM work.[web:30][web:34]
        spearman_rho, spearman_p = spearmanr(h, m)
        kendall_tau, kendall_p = kendalltau(h, m)
        pearson_r = float(np.corrcoef(h, m)[0, 1])
        rmse = float(np.sqrt(np.mean((h - m) ** 2)))

        if spearman_rho > 0.5:
            status = "Good"
        elif spearman_rho > 0.3:
            status = "Fair"
        else:
            status = "Poor"

        return {
            "spearman_rho": float(spearman_rho),
            "spearman_pval": float(spearman_p),
            "kendall_tau": float(kendall_tau),
            "kendall_pval": float(kendall_p),
            "pearson_r": pearson_r,
            "rmse": rmse,
            "n_samples": int(len(h)),
            "alignment_status": status,
        }

    def generate_alignment_report(self) -> str:
        metrics = self.compute_alignment_metrics()
        if "error" in metrics:
            return f"HUMAN ALIGNMENT REPORT\n\n{metrics['error']} (n={metrics['n_samples']})"

        lines = []
        lines.append("╔══════════════════════════════════════════════════════╗")
        lines.append("║        HUMAN ALIGNMENT EVALUATION REPORT             ║")
        lines.append("╚══════════════════════════════════════════════════════╝")
        lines.append("")
        lines.append(f"Total comparisons: {metrics['n_samples']}")
        lines.append(f"Alignment status: {metrics['alignment_status']}")
        lines.append("")
        lines.append("Correlation metrics (higher is better):")
        lines.append(
            f"  • Spearman ρ: {metrics['spearman_rho']:.4f}  (p={metrics['spearman_pval']:.2e})"
        )
        lines.append(
            f"  • Kendall τ: {metrics['kendall_tau']:.4f}  (p={metrics['kendall_pval']:.2e})"
        )
        lines.append(f"  • Pearson r: {metrics['pearson_r']:.4f}")
        lines.append("")
        lines.append(f"Error metric (lower is better):")
        lines.append(f"  • RMSE: {metrics['rmse']:.4f} points (scale 1–4)")
        lines.append("")
        lines.append("Interpretation:")
        if metrics["spearman_rho"] > 0.5:
            lines.append("  ✓ Strong alignment with human ranking preferences.")
        elif metrics["spearman_rho"] > 0.3:
            lines.append("  ⚠ Moderate alignment; consider prompt or rubric tuning.")
        else:
            lines.append("  ✗ Poor alignment; evaluator likely miscalibrated.")

        if metrics["rmse"] < 0.5:
            lines.append("  ✓ Tight agreement on absolute scores.")
        elif metrics["rmse"] < 1.0:
            lines.append("  ⚠ Some disagreement on absolute scores.")
        else:
            lines.append("  ✗ Large discrepancies in absolute scores.")

        return "\n".join(lines)
      
    def run_full_alignment_for_state(self, state: Dict) -> Dict:
        """
        Convenience helper that plugs directly into your existing `state` format.

        Expects:
            state["input"][-1] to contain:
                - "student_id"
                - "question_id"
                - "interview_question"
                - "summary_of_past_response"
                - "new_code_written"

            state["evaluation_result"] already populated by your evaluator.

        Returns:
            state with extra keys:
                - "human_reference_evaluation"
                - "alignment_comparison"
                - "alignment_metrics"
                - "alignment_report"
        """
        input_data = state["input"][-1]
        model_eval = state.get("evaluation_result", {})

        student_id = input_data.get("student_id", "unknown")
        question_id = input_data.get("question_id", "unknown")

        human_eval = self.get_human_evaluation(
            interview_question=input_data.get("interview_question", ""),
            student_code=input_data.get("new_code_written", ""),
            student_summary=input_data.get("summary_of_past_response", ""),
            student_id=student_id,
            question_id=question_id,
        )

        comparison = self.compare_single_evaluation(
            model_evaluation=model_eval,
            human_evaluation=human_eval,
            student_id=student_id,
            question_id=question_id,
        )

        metrics = self.compute_alignment_metrics()
        report = self.generate_alignment_report()

        state["human_reference_evaluation"] = human_eval
        state["alignment_comparison"] = comparison
        state["alignment_metrics"] = metrics
        state["alignment_report"] = report

        return state

def evaluation_agent(state: dict) -> dict:
    """
    Calls GPT to output JSON matching the EvaluationSchema,
    storing it in state["evaluation_result"]. 
    This is an 'intermediate' grading each time the user responds.
    """
    input_data = state["input"][-1]

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
    state["output"] = [raw_text]
    try:
        parsed = json.loads(raw_text)
        evaluation_obj = EvaluationSchema(**parsed)
        state["evaluation_result"] = evaluation_obj.dict()
    except (json.JSONDecodeError, ValueError) as e:
        state["evaluation_result"] = {
            "error": f"Could not parse JSON: {e}",
            "raw_output": raw_text
        }
    state = alignment_evaluator.run_full_alignment_for_state(state)
    return state

def partial_evaluation_agent(state: dict) -> dict:
    """
    Runs on every user response, outputting JSON matching the same schema as the final evaluator.
    Ensures student_id and question_id fields are included.
    """
    input_data = state["input"][-1]

    student_id = input_data.get("student_id", "unknown")
    question_id = input_data.get("question_id", "unknown")

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

    try:
        parsed = json.loads(raw_text)
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
