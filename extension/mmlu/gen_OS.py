import sys
import os
import time
import random
import json
import pandas as pd
import argparse
from glob import glob
from pathlib import Path
from transformers import AutoTokenizer

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]

# Add project root to sys.path for imports
sys.path.append(str(project_root))

try:
    from llm.implementations.local_llm import LocalLLM
    from llm.core.config import LLMConfig
except ImportError as e:
    print(f"[!] Import Error: {e}")
    sys.exit(1)

# ==========================================
# 2. LOAD CONFIG AND INITIALIZE MODEL
# ==========================================

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="llm/configs/mistral-7B-instruct.json",
                   help="Path to LLM config JSON file")
args = parser.parse_args()

# Make config path relative to project root if it's not an absolute path
config_path = args.config
if not os.path.isabs(config_path):
    config_path = project_root / config_path

print(f"Loading config from: {config_path}")
try:
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Convert to LLMConfig object
    config = LLMConfig(
        model_name=config_data["model_name"],
        model_type=config_data["model_type"],
        temperature=config_data["temperature"],
        max_tokens=config_data["max_tokens"],
        context_length=config_data["context_length"],
        parameter_count=config_data["parameter_count"],
        model_path=config_data.get("model_path"),
        device=config_data.get("device"),
        quantization=config_data.get("quantization")
    )
except Exception as e:
    print(f"[!] Config loading failed: {e}")
    sys.exit(1)

print(f"Initializing {config.model_name}...")
try:
    mas_model = LocalLLM(config=config)
    print("Success: Local LLM loaded.")
except Exception as e:
    print(f"[!] Model initialization failed: {e}")
    sys.exit(1)

# Cache the tokenizer for token counting (if available)
tokenizer = mas_model.tokenizer if hasattr(mas_model, 'tokenizer') else None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_real_token_count(text):
    return len(tokenizer.encode(text)) if tokenizer else len(text) // 4

def calculate_flops(token_count):
    return 2 * config.parameter_count * token_count

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

        # Extract text from LLMResponse object
        response_text = response_object.text

        # Use token counts from the response if available, otherwise estimate
        if hasattr(response_object, 'input_tokens') and hasattr(response_object, 'output_tokens'):
            prompt_tokens = response_object.input_tokens
            completion_tokens = response_object.output_tokens
            total_tokens = response_object.total_tokens
        else:
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
    # Use model name in filename (replace special characters)
    model_short_name = config.model_name.replace("/", "_").replace("-", "_")
    outfile = f"mmlu-25-_{agents}_{rounds}_{model_short_name}.json"
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
