import sys
import os
import time
import random
import json
import re
from pathlib import Path
from transformers import AutoTokenizer

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
script_path = Path(__file__).resolve()
# Assuming this file is in replication/gsm8k/
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
# 2. INITIALIZE MODEL (OFFICIAL META LLAMA 3.1)
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PARAM_COUNT = 8_030_000_000 

print(f"Loading Tokenizer for {MODEL_NAME}...")
try:
    # Requires 'huggingface-cli login'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception as e:
    print(f"[!] Warning: Tokenizer failed ({e}). FLOPs will be estimated.")
    tokenizer = None

print("Initializing Connection...")
try:
    # CRITICAL: Update if on a different node (e.g. http://gl-gpu-1234:8000/v1)
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
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text) // 4

def calculate_flops(token_count):
    return 2 * PARAM_COUNT * token_count

def construct_message(agents, question, idx):
    # Initial Debate Prompt
    if len(agents) == 0:
        return {
            "role": "user", 
            "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{answer}."
        }

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        try:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent solution: ```{}```".format(agent_response)
            prefix_string = prefix_string + response
        except IndexError:
            continue

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    if not os.path.exists(path):
        print(f"[!] Error: Could not find data file at {path}")
        # Try to look relative to project root just in case
        alt_path = project_root / "data" / "gsm8k" / "test.jsonl"
        if os.path.exists(alt_path):
            print(f"Found it at {alt_path}")
            return read_jsonl(str(alt_path))
        sys.exit(1)
            
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def generate_answer(answer_context):
    try:
        # 1. Format prompt
        prompt_text = ""
        for msg in answer_context:
            prompt_text += f"\n\n### {msg['role'].upper()}:\n{msg['content']}"
        prompt_text += "\n\n### ASSISTANT:\n"

        # 2. Call Model
        response_object = mas_model.generate(prompt_text)

        # 3. Extract text
        if hasattr(response_object, "content"): response_text = response_object.content
        elif hasattr(response_object, "text"): response_text = response_object.text
        elif isinstance(response_object, str): response_text = response_object
        else: response_text = str(response_object)

        # 4. Calculate Stats
        prompt_tokens = get_real_token_count(prompt_text)
        completion_tokens = get_real_token_count(response_text)
        total_tokens = prompt_tokens + completion_tokens
        flops = calculate_flops(total_tokens)

        # 5. Build Object
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
        print(f"Error calling model: {e}")
        print("Retrying...")
        time.sleep(2)
        return generate_answer(answer_context)


# ==========================================
# 4. MAIN LOOP (WITH AUTO-SAVE)
# ==========================================
if __name__ == "__main__":
    agents = 1
    rounds = 1
    random.seed(0)

    # Bypass Proxy
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

    total_tokens = 0
    total_flops = 0
    api_calls = 0

    # Ensure this file exists in the current directory!
    # Or update this path to where you stored the GSM8K jsonl file
    data_file = "test.jsonl" 
    
    questions = read_jsonl(data_file)
    # random.shuffle(questions) # Optional: Shuffle if you want random samples
    
    print(f"Loaded {len(questions)} questions.")

    # --- RESUME LOGIC ---
    outfile = f"gsm_{agents}_{rounds}_llama3.1_8b_official.json"
    generated_description = {}
    
    if os.path.exists(outfile):
        print(f"[Resume] Found existing file: {outfile}")
        try:
            with open(outfile, "r") as f:
                generated_description = json.load(f)
            print(f"[Resume] Loaded {len(generated_description)} completed questions.")
        except:
            print("[Resume] File corrupted. Starting fresh.")
    # --------------------

    # Limit to 100 for this run (Change as needed)
    for i, data in enumerate(questions[:25]):
        question = data['question']
        answer = data['answer']

        # Skip if already done
        if question in generated_description:
            print(f"--- Skipping Question {i+1} (Already Done) ---")
            continue

        print(f"--- Processing Question {i+1} ---")

        # Initial Prompt (Updated for Math)
        # Note: GSM8K has no options A, B, C, D. It needs a number.
        initial_prompt = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."

        agent_contexts = [[{"role": "user", "content": initial_prompt}] for agent in range(agents)]

        for round in range(rounds):
            print(f"  Round {round+1}...")
            for idx, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:idx] + agent_contexts[idx+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                completion, usage = generate_answer(agent_context)

                # Track usage
                if usage:
                    total_tokens += usage.get('total_tokens', 0)
                    total_flops += usage.get('flops', 0)
                    api_calls += 1
                
                print(f"    Agent {idx} finished ({usage.get('total_tokens')} tok)")

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

        # --- AUTO SAVE ---
        with open(outfile, "w") as f:
            json.dump(generated_description, f, indent=4)
        print(f"    [Saved] {len(generated_description)}/100 completed.")
        # -----------------

    # Final Summary
    print("\n" + "="*50)
    print("COMPUTE USAGE SUMMARY (Llama 3.1 8B)")
    print("="*50)
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Total PetaFLOPs: {total_flops / 1e15:.6f}")
    print("="*50)

    print("Done.")
