import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import ast
import numpy as np
import re
from databench_eval.utils import load_table
from datasets import load_dataset

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
if torch.cuda.is_available():
    model = model.to("cuda")

# Load the QA pairs for 'semeval' and 'qa' dataset
semeval_dev_qa   = load_dataset("cardiffnlp/databench", name="semeval", split="dev")
semeval_train_qa = load_dataset("cardiffnlp/databench", name="qa", split="train")

df_QA = pd.DataFrame(semeval_dev_qa)

# Define prompt templates
instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

def generate_pandas_prompt(df, query_str):
    # Limit the number of columns and truncate long values
    df_preview = df.head(5).copy()
    for col in df_preview.columns:
        if df_preview[col].dtype == "object":  # Truncate long strings
            df_preview[col] = df_preview[col].apply(lambda x: str(x)[:30] + "..." if len(str(x)) > 30 else x)

    # Convert the truncated DataFrame to string
    df_str = df_preview.to_string(index=False)

    return (
        f"You are working with a pandas dataframe in Python.\n"
        f"The name of the dataframe is `df`.\n"
        f"This is the result of `print(df.head())`:\n"
        f"{df_str}\n\n"
        f"Follow these instructions:\n"
        f"{instruction_str}\n"
        f"Query: {query_str}\n\n"
        "Pandas Expression:"
    )

def safe_eval(expression: str, df: pd.DataFrame) -> str:
    """Safely evaluate a Python expression using ast parsing."""
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')

        # Only allow certain safe nodes
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare,
                                     ast.Name, ast.Load, ast.Call, ast.Constant,
                                     ast.Attribute, ast.Subscript, ast.Index, ast.Slice,
                                     ast.List, ast.Tuple, ast.Dict)):
                raise ValueError(f"Unsafe expression: {ast.dump(node)}")

        # Define a restricted set of built-ins
        safe_globals = {
            "__builtins__": {},
            "np": np,
            "pd": pd,
            "df": df
        }

        # Evaluate the expression
        return str(eval(expression, safe_globals))
    except Exception as e:
        return f"Error during safe evaluation: {str(e)}"

def clean_generated_code(output: str) -> str:
    """Extract the actual code from the generated output using regex."""
    # Use regex to extract the text after 'Pandas Expression:'
    match = re.search(r'Pandas Expression:\s*(.*)', output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Further clean the code (remove unwanted characters or formatting)
        code = code.replace(" == ", " & ").replace(" or ", " | ").replace(" and ", " & ")
        return code
    return ""

# Open a text file to save the results
with open("QA_on_Tabular_Data/results.txt", "w") as file:
    # Loop through each question in the DataFrame
    for i in range(len(df_QA)):
        if i % 20 == 0:
            print(f"Question {i} out of {len(df_QA)}")
        sample_question = df_QA['question'][i]
        df = load_table(df_QA['dataset'][i])

        # Generate the prompt for the LLM
        pandas_prompt = generate_pandas_prompt(df, sample_question)

        # Tokenize the prompt and generate code
        inputs = tokenizer(pandas_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=1000, temperature=0.1)
        pandas_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print("Uncleaned Pandas Code:", pandas_code)

        # Clean the generated Pandas code
        pandas_code = clean_generated_code(pandas_code.replace('`', '').replace('python', '').replace('\n', ''))
        #print("Cleaned Pandas Code:", pandas_code)

        # Safely evaluate the generated code
        result = safe_eval(pandas_code, df)
        #print("Evaluation Result:", result)

        # Write the result to the text file
        file.write(f"{result}\n")

print("Results saved to results.txt")
