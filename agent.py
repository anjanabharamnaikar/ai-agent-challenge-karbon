# agent.py

import argparse
import subprocess
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict

load_dotenv()

# Define the state for our graph
class AgentState(TypedDict):
    task: str
    pdf_path: str
    csv_path: str
    plan: str
    generated_code: str
    test_results: str
    correction_attempts: int

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# --- MASTER PROMPT FOR CODE GENERATION ---
# This detailed template is the key to forcing the AI to generate correct code.
CODE_GENERATION_PROMPT = """
You are an expert Python developer tasked with writing a custom parser for a bank statement PDF.
You must write a single function `parse(pdf_path: str) -> pd.DataFrame`.
The final DataFrame MUST strictly match this schema and logic:
1.  **Columns**: The final DataFrame must have exactly these four columns in this exact order: ['Date', 'Description', 'Amount', 'Balance'].
2.  **Date Column**: This column must be of datetime64[ns] type. Convert it using `pd.to_datetime` with the format '%d-%m-%Y'.
3.  **Amount Column**: This is the most critical part. The PDF has separate columns for debits (withdrawals) and credits (deposits).
    - You must create a single 'Amount' column.
    - **Credits/Deposits must be positive numbers.**
    - **Debits/Withdrawals must be negative numbers.**
    - The logic should be: `df['Amount'] = df['Credit'] - df['Debit']`.
4.  **Data Cleaning**:
    - All monetary columns ('Debit', 'Credit', 'Balance') must be cleaned of any commas (',') before being converted to numeric types.
    - All string columns ('Description') must have leading/trailing whitespace removed using `.str.strip()`.
5.  **Parsing Library**: Use the `camelot-py` library to extract tables from the PDF. It is effective for this task.
6.  **Function Definition**: The final code must be a complete Python script containing only the `parse` function and necessary imports. Do not include any markdown formatting like ```python.

Based on this plan, write the complete Python code for the parser:
Plan: {plan}
"""

# Node 1: Planner
def planner(state: AgentState):
    """
    Creates a high-level plan for the code generator.
    """
    print("--- 📝 PLANNING ---")
    
    # Read the schema of the target CSV to inform the plan
    csv_df = pd.read_csv(state["csv_path"])
    csv_schema_summary = f"Columns: {csv_df.columns.tolist()}\nData Types:\n{csv_df.dtypes.to_string()}"

    prompt_template = ChatPromptTemplate.from_template(
        """
        Create a step-by-step plan to write a Python PDF parser function.
        The target CSV has this schema:
        {csv_schema}

        The plan must address these key steps:
        1.  Extract tables from the PDF using camelot.
        2.  Identify the correct transaction table.
        3.  Combine the 'Debit Amt' and 'Credit Amt' columns into a single 'Amount' column, where debits are negative and credits are positive.
        4.  Clean and format all columns to precisely match the CSV schema (dates, numbers, strings).
        5.  Return a pandas DataFrame with the final, cleaned data.
        """
    )
    
    chain = prompt_template | llm | StrOutputParser()
    plan = chain.invoke({"csv_schema": csv_schema_summary})
    print(plan)
    return {"plan": plan}

# Node 2: Code Generator
def code_generator(state: AgentState):
    """
    Generates Python code based on the master prompt and the plan.
    """
    print("--- 💻 GENERATING CODE ---")
    prompt = ChatPromptTemplate.from_template(CODE_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    generated_code = chain.invoke({"plan": state["plan"]})
    
    # Clean the output just in case markdown slips through
    if generated_code.startswith("```python"):
        generated_code = generated_code[9:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
        
    print(generated_code)
    return {"generated_code": generated_code}

# Node 3: Code Tester
def code_tester(state: AgentState):
    """
    Saves the generated code and tests it against the sample data.
    """
    print("--- 🧪 TESTING CODE ---")
    parser_path = f"custom_parsers/{state['task']}_parser.py"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(parser_path), exist_ok=True)
    
    with open(parser_path, "w") as f:
        f.write(state["generated_code"])

    # Create a temporary test runner script to get precise feedback
    test_runner_script = f"""
import pandas as pd
from custom_parsers.{state['task']}_parser import parse
from pandas.testing import assert_frame_equal

pdf_path = '{state['pdf_path']}'
csv_path = '{state['csv_path']}'

try:
    expected_df = pd.read_csv(csv_path)
    # Prepare expected_df to match the parser's output schema
    expected_df['Date'] = pd.to_datetime(expected_df['Date'], dayfirst=True)
    expected_df['Debit Amt'] = pd.to_numeric(expected_df['Debit Amt'].fillna(0))
    expected_df['Credit Amt'] = pd.to_numeric(expected_df['Credit Amt'].fillna(0))
    expected_df['Amount'] = expected_df['Credit Amt'] - expected_df['Debit Amt']
    expected_df = expected_df[['Date', 'Description', 'Amount', 'Balance']].copy()
    
    actual_df = parse(pdf_path)
    
    # Sort both frames to handle potential order differences from parsing
    expected_df = expected_df.sort_values(by=expected_df.columns.tolist()).reset_index(drop=True)
    actual_df = actual_df.sort_values(by=actual_df.columns.tolist()).reset_index(drop=True)

    assert_frame_equal(actual_df, expected_df, check_dtype=True)
    print('PASS')

except Exception as e:
    print(f'FAIL: {{e}}')
"""
    try:
        cmd = ["python", "-c", test_runner_script]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        
        test_output = result.stdout.strip()
        if result.returncode != 0:
            test_output += f"\nSTDERR: {result.stderr.strip()}"

        if "FAIL" in test_output:
            print(f"Tests failed.\n{test_output}")
            return {"test_results": test_output, "correction_attempts": state.get("correction_attempts", 0) + 1}
        else:
            print("✅ Tests passed!")
            return {"test_results": "PASS"}

    except subprocess.TimeoutExpired:
        print("Testing timed out.")
        return {"test_results": "FAIL: Testing process timed out.", "correction_attempts": state.get("correction_attempts", 0) + 1}
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        return {"test_results": str(e), "correction_attempts": state.get("correction_attempts", 0) + 1}

# Node 4: Self-Corrector
def self_corrector(state: AgentState):
    """
    Analyzes test failures and refines the plan.
    """
    print("--- 🧠 SELF-CORRECTING ---")
    prompt = ChatPromptTemplate.from_template(
        """
        The Python parser code you generated failed the tests.
        You MUST fix it. Pay close attention to the error message and the required schema.

        **CRITICAL REQUIREMENTS**:
        - Final Columns: ['Date', 'Description', 'Amount', 'Balance']
        - 'Amount' column logic: Credits are positive, Debits are negative (`Credit` - `Debit`).
        - Data types must be exact: `Date` as datetime, monetary columns as numbers.

        **Original Plan**:
        {plan}
        
        **Generated Code**:
        {code}

        **Test Error**:
        {error}

        Based on the error, create a new, corrected plan. Be very specific about how to fix the data cleaning, column manipulation, or type conversion steps that caused the error.
        """
    )
    chain = prompt | llm | StrOutputParser()
    new_plan = chain.invoke({
        "plan": state["plan"],
        "code": state["generated_code"],
        "error": state["test_results"]
    })
    print(f"New Corrective Plan:\n{new_plan}")
    return {"plan": new_plan}

# Conditional Edge
def decide_next_step(state: AgentState):
    if state["test_results"] == "PASS":
        return "end"
    elif state.get("correction_attempts", 0) >= 3:
        print("--- ❌ MAXIMUM CORRECTION ATTEMPTS REACHED ---")
        return "end"
    else:
        return "self_corrector"

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("code_generator", code_generator)
workflow.add_node("code_tester", code_tester)
workflow.add_node("self_corrector", self_corrector)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "code_generator")
workflow.add_edge("code_generator", "code_tester")
workflow.add_conditional_edges(
    "code_tester",
    decide_next_step,
    {"self_corrector": "self_corrector", "end": END}
)
workflow.add_edge("self_corrector", "code_generator")

app = workflow.compile()

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent for Bank Statement Parsing.")
    parser.add_argument("--target", type=str, required=True, help="The target bank (e.g., icici).")
    args = parser.parse_args()

    initial_state = {
        "task": args.target,
        "pdf_path": f"data/{args.target}/{args.target}_sample.pdf",
        "csv_path": f"data/{args.target}/{args.target}_sample.csv",
    }
    
    app.invoke(initial_state)
    print("\n--- ✅ AGENT RUN COMPLETE ---")
