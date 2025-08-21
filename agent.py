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

# Master prompt for code generation
CODE_GENERATION_PROMPT = """
You are an expert Python developer tasked with writing a custom parser for a bank statement PDF.
You must write a single function `parse(pdf_path: str) -> pd.DataFrame`.
The final DataFrame MUST strictly match this schema and logic:
1. Columns: ['Date', 'Description', 'Amount', 'Balance'] in this exact order.
2. Date Column: Must be datetime64[ns] using `pd.to_datetime` with format '%d-%m-%Y'.
3. Amount Column: Create a single 'Amount' column where credits are positive and debits are negative.
   Logic: df['Amount'] = df['Credit'] - df['Debit'].
4. Data Cleaning:
   - Remove commas from monetary columns ('Debit', 'Credit', 'Balance') before converting to numeric.
   - Strip leading/trailing whitespace from string columns.
5. Use `camelot-py` to extract tables from the PDF.
6. Only use libraries: re, os, pandas, camelot (without tablelist import), datetime.
7. Error Handling: Handle missing or malformed data gracefully.
8. Function Definition: Provide a complete Python script containing only the `parse` function and imports.
9. Output: Return a pandas DataFrame with the final cleaned data.
Plan: {plan}
"""

# Node 1: Planner
def planner(state: AgentState):
    print("--- Planning ---")
    csv_df = pd.read_csv(state["csv_path"])
    csv_schema_summary = f"Columns: {csv_df.columns.tolist()}\nData Types:\n{csv_df.dtypes.to_string()}"

    prompt_template = ChatPromptTemplate.from_template(
        """
        Create a step-by-step plan to write a Python PDF parser function.
        The target CSV has this schema:
        {csv_schema}

        The plan must address:
        1. Extract tables from PDF using camelot.
        2. Identify the correct transaction table.
        3. Combine 'Debit Amt' and 'Credit Amt' into a single 'Amount' column.
        4. Clean and format all columns to match the CSV schema.
        5. Return a pandas DataFrame with the cleaned data.
        """
    )
    chain = prompt_template | llm | StrOutputParser()
    plan = chain.invoke({"csv_schema": csv_schema_summary})
    print(plan)
    return {"plan": plan}

# Node 2: Code Generator
def code_generator(state: AgentState):
    print("--- Generating Code ---")
    prompt = ChatPromptTemplate.from_template(CODE_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    generated_code = chain.invoke({"plan": state["plan"]})

    if generated_code.startswith("```python"):
        generated_code = generated_code[9:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]

    print(generated_code)
    return {"generated_code": generated_code}

# Node 3: Code Tester
def code_tester(state: AgentState):
    print("--- Testing Code ---")
    parser_path = f"custom_parsers/{state['task']}_parser.py"
    os.makedirs(os.path.dirname(parser_path), exist_ok=True)

    with open(parser_path, "w") as f:
        f.write(state["generated_code"])

    test_runner_script = f"""
import pandas as pd
from custom_parsers.{state['task']}_parser import parse
from pandas.testing import assert_frame_equal

pdf_path = '{state['pdf_path']}'
csv_path = '{state['csv_path']}'

try:
    expected_df = pd.read_csv(csv_path)
    expected_df['Date'] = pd.to_datetime(expected_df['Date'], dayfirst=True)
    expected_df['Debit Amt'] = pd.to_numeric(expected_df['Debit Amt'].fillna(0))
    expected_df['Credit Amt'] = pd.to_numeric(expected_df['Credit Amt'].fillna(0))
    expected_df['Amount'] = expected_df['Credit Amt'] - expected_df['Debit Amt']
    expected_df = expected_df[['Date', 'Description', 'Amount', 'Balance']].copy()
    
    actual_df = parse(pdf_path)
    
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
            print("Tests passed.")
            return {"test_results": "PASS"}

    except subprocess.TimeoutExpired:
        print("Testing timed out.")
        return {"test_results": "FAIL: Testing process timed out.", "correction_attempts": state.get("correction_attempts", 0) + 1}
    except Exception as e:
        print(f"Unexpected error during testing: {e}")
        return {"test_results": str(e), "correction_attempts": state.get("correction_attempts", 0) + 1}

# Node 4: Self-Corrector
def self_corrector(state: AgentState):
    print("--- Self-Correcting ---")
    prompt = ChatPromptTemplate.from_template(
        """
        The generated parser code failed the tests. Fix it.
        Requirements:
        - Columns: ['Date', 'Description', 'Amount', 'Balance']
        - 'Amount' = Credit - Debit
        - Correct data types: Date as datetime, monetary columns as numbers.

        Original Plan:
        {plan}
        
        Generated Code:
        {code}

        Test Error:
        {error}

        Provide a new corrected plan specifying how to fix data cleaning, column manipulation, or type conversion.
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

# Decide next step
def decide_next_step(state: AgentState):
    if state["test_results"] == "PASS":
        return "end"
    elif state.get("correction_attempts", 0) >= 3:
        print("--- Maximum correction attempts reached ---")
        return "end"
    else:
        return "self_corrector"

# Build the graph
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
    print("\n--- Agent run complete ---")
