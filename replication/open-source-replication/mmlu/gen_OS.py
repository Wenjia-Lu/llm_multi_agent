import sys
import os
import time
import random
import json
import pandas as pd
from glob import glob
from pathlib import Path
from transformers import AutoTokenizer

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
script_path = Path(__file__).resolve()
project_root = script_path.parents[2] 
xmas_path = project_root / "X-MAS"

if not xmas_path.exists():
    xmas_path = project_root / "x-mas"
if not xmas_path.exists():
    print(f"[!] Critical Error: Could not find 'X-MAS' folder at {xmas_path}")
    sys.exit(1)

sys.path.append(str(xmas_path))

try:
    from llm.implementations.api_llm import APILLM
    from llm.core.config import LLMConfig
except ImportError as e:
    print(f"[!] Import Error: {e}")
    sys.exit(1)

# ==========================================
# 2. INITIALIZE MODEL (Official Llama 3.1 8B)
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PARAM_COUNT = 8_030_000_000 

print(f"Loading Tokenizer for {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception as e:
    print(f"[!] Warning: Tokenizer failed ({e}). FLOPs will be estimated.")
    tokenizer = None

print("Initializing Connection...")
try:
    API_URL = "http://localhost:8000/v1" 
    
    config = LLMConfig(
        model_name=MODEL_NAME, model_type="api", temperature=0.7,
        max_tokens=512, context_length=8192, parameter_count=PARAM_COUNT,
        api_url=API_URL, api_key="EMPTY"
    )
    mas_model = APILLM(config=config)
    print("Success: Connected to vLLM.")
except Exception as e:
    print(f"[!] Connection Failed: {e}")
    sys.exit(1)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_real_token_count(text):
    return len(tokenizer.encode(text)) if tokenizer else len(text) // 4

def calculate_flops(token_count):
    return 2 * PARAM_COUNT * token_count

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}
    prefix_string = "These are the solutions to the problem from other agents: "
    for agent in agents:
        try:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent solution: ```{}```".format(agent_response)
            prefix_string += response
        except IndexError:
            continue
    prefix_string += """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response."""
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def generate_answer(answer_context):
    try:
        prompt_text = ""
        for msg in answer_context:
            prompt_text += f"\n\n### {msg['role'].upper()}:\n{msg['content']}"
        prompt_text += "\n\n### ASSISTANT:\n"

        response_object = mas_model.generate(prompt_text)

        if hasattr(response_object, "content"): response_text = response_object.content
        elif hasattr(response_object, "text"): response_text = response_object.text
        elif isinstance(response_object, str): response_text = response_object
        else: response_text = str(response_object)

        prompt_tokens = get_real_token_count(prompt_text)
        completion_tokens = get_real_token_count(response_text)
        total_tokens = prompt_tokens + completion_tokens
        flops = calculate_flops(total_tokens)

        completion = {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "flops": flops,
                "gflops": flops / 1e9
            }
        }
        return completion, completion['usage']

    except Exception as e:
        print(f"error calling model: {e}")
        print("retrying due to an error......")
        time.sleep(2)
        return generate_answer(answer_context)

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a, b, c, d = df.iloc[ix, 1], df.iloc[ix, 2], df.iloc[ix, 3], df.iloc[ix, 4]
    question = f"Can you answer the following question as accurately as possible? {question}: A) {a}, B) {b}, C) {c}, D) {d} Explain your answer, putting the answer in the form (X) at the end of your response, where X represents A, B, C, or D. Do not output the number. Output only the option letter."
    answer = df.iloc[ix, 5]
    return question, answer

# ==========================================
# 4. MAIN LOOP (WITH AUTO-SAVE & RESUME)
# ==========================================
if __name__ == "__main__":
    agents = 1
    rounds = 1
    
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_flops = 0
    api_calls = 0

    data_dir = project_root / "data" / "test"
    tasks = list(data_dir.glob("*.csv"))
    if not tasks: tasks = glob("./data/test/*.csv")
    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    
    # --- RESUME LOGIC ---
    outfile = f"mmlu-25-_{agents}_{rounds}_llama3.1_8b.json"
    response_dict = {}
    
    if os.path.exists(outfile):
        print(f"[Resume] Found existing results in {outfile}")
        try:
            with open(outfile, "r") as f:
                response_dict = json.load(f)
            print(f"[Resume] {len(response_dict)} questions already completed.")
        except:
            print("[Resume] File corrupted. Starting fresh.")

    # Loop 100 times
    for i in range(25):
        # We perform the random choice FIRST to ensure the seed stays consistent
        df = random.choice(dfs)
        idx = random.randint(0, len(df)-1)
        question, answer = parse_question_answer(df, idx)

        # --- CHECK IF DONE ---
        if question in response_dict:
            print(f"--- Skipping Question {i+1} (Already saved) ---")
            continue
        # ---------------------

        print(f"--- Processing Question {i+1} ---")
        agent_contexts = [[{"role": "user", "content": question}] for agent in range(agents)]

        for round in range(rounds):
            print(f"  Round {round+1}...")
            for idx, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:idx] + agent_contexts[idx+1:]
                    message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                completion, usage = generate_answer(agent_context)

                if usage:
                    total_tokens += usage.get('total_tokens', 0)
                    total_flops += usage.get('flops', 0)
                    api_calls += 1
                
                print(f"    Agent {idx} finished ({usage.get('total_tokens')} tok)")
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        response_dict[question] = (agent_contexts, answer)

        # --- AUTO-SAVE AFTER EVERY QUESTION ---
        with open(outfile, "w") as f:
            json.dump(response_dict, f, indent=4)
        print(f"    [Saved] {len(response_dict)}/100 completed.")
        # --------------------------------------

    print("\n" + "="*50)
    print(f"Job Complete. Saved to {outfile}")
    print(f"Total FLOPs used this session: {total_flops:.2e}")
    print("="*50)
