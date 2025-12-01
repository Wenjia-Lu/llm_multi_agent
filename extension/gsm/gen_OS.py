import sys
import os
import time
import random
import json
import re
import argparse
from pathlib import Path
from transformers import AutoTokenizer

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
# Fixed three agents for debate system
AGENT_CONFIGS = [
    "llm/configs/llama3.1-8B-instruct.json",
    "llm/configs/DeepSeek-R1-Qwen-7B.json",
    "llm/configs/Mathstral-7B.json"
]
DEFAULT_AGENTS = 3  # Fixed number of agents
DEFAULT_ROUNDS = 3  # Default debate rounds

# ==========================================
# 1. SETUP PATHS & IMPORTS
# ==========================================
script_path = Path(__file__).resolve()
# Assuming this file is in extension/gsm/
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
# 2. LOAD CONFIG AND INITIALIZE MODELS
# ==========================================

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help=f"Number of debate rounds (default: {DEFAULT_ROUNDS})")
parser.add_argument("--confidence_threshold", type=float, default=0.95,
                   help="Confidence threshold for gating (default: 0.95)")
args = parser.parse_args()

# Load all three agents
agents = []
agent_names = []
for i, config_path in enumerate(AGENT_CONFIGS):
    # Make config path relative to project root if it's not an absolute path
    if not os.path.isabs(config_path):
        full_config_path = project_root / config_path
    else:
        full_config_path = config_path

    print(f"Loading agent {i+1} config from: {full_config_path}")
    try:
        with open(full_config_path, 'r') as f:
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

        print(f"Initializing {config.model_name}...")
        agent = LocalLLM(config=config)
        agents.append(agent)
        agent_names.append(config.model_name)
        print(f"Success: Agent {i+1} loaded.")

    except Exception as e:
        print(f"[!] Failed to load agent {i+1}: {e}")
        sys.exit(1)

print(f"All {len(agents)} agents loaded successfully!")

# Use the first agent's tokenizer for token counting (if available)
tokenizer = agents[0].tokenizer if hasattr(agents[0], 'tokenizer') else None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_real_token_count(text):
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text) // 4

def calculate_flops(token_count):
    # Use the first agent's parameter count for FLOP calculation
    return 2 * agents[0].config.parameter_count * token_count

def extract_confidence(response_text):
    """Extract confidence score from model response."""
    import re

    # Look for confidence patterns like "confidence: 0.95", "95% confident", etc.
    confidence_patterns = [
        r'confidence[:\s]*(\d+\.?\d*)',  # "confidence: 0.95"
        r'(\d+\.?\d*)[\s]*confidence',   # "0.95 confidence"
        r'(\d+)%[\s]*confident',         # "95% confident"
        r'confident.*(\d+\.?\d*)',       # "confident 0.95"
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Convert percentage to decimal if needed
            if value > 1:
                value = value / 100
            return min(value, 1.0)  # Cap at 1.0

    # Default confidence if no pattern found
    print(f"Warning: Could not extract confidence from response, using default 0.5")
    return 0.5

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

def generate_answer(answer_context, agent_model):
    try:
        # 1. Format prompt
        prompt_text = ""
        for msg in answer_context:
            prompt_text += f"\n\n### {msg['role'].upper()}:\n{msg['content']}"
        prompt_text += "\n\n### ASSISTANT:\n"

        # 2. Call Model
        response_object = agent_model.generate(prompt_text)

        # 3. Extract text
        response_text = response_object.text

        # 4. Calculate Stats (use response data if available, otherwise estimate)
        if hasattr(response_object, 'input_tokens') and hasattr(response_object, 'output_tokens'):
            prompt_tokens = response_object.input_tokens
            completion_tokens = response_object.output_tokens
            total_tokens = response_object.total_tokens
        else:
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
        return generate_answer(answer_context, agent_model)


# ==========================================
# 4. MAIN LOOP (WITH AUTO-SAVE)
# ==========================================
if __name__ == "__main__":
    rounds = args.rounds
    confidence_threshold = args.confidence_threshold
    random.seed(0)

    total_tokens = 0
    total_flops = 0
    api_calls = 0

    # Look for data file in current directory or extension/gsm/
    data_file = "test.jsonl"
    if not os.path.exists(data_file):
        alt_path = script_path.parent / "test.jsonl"
        if os.path.exists(alt_path):
            data_file = str(alt_path)

    questions = read_jsonl(data_file)
    # random.shuffle(questions) # Optional: Shuffle if you want random samples
    
    print(f"Loaded {len(questions)} questions.")

    # --- RESUME LOGIC ---
    # Create output filename based on agent names and parameters
    agent_short_names = [name.split('/')[-1].replace('-', '_').replace('.', '_') for name in agent_names]
    outfile = f"debate_gsm_{rounds}rounds_{confidence_threshold}conf_{'_'.join(agent_short_names[:3])}.json"
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

    # Limit to 3 for this run (Change as needed)
    for i, data in enumerate(questions[:3]):
        question = data['question']
        answer = data['answer']

        # Skip if already done
        if question in generated_description:
            print(f"--- Skipping Question {i+1} (Already Done) ---")
            continue

        print(f"--- Processing Question {i+1} ---")

        # Initial Prompt (Updated for Math with confidence)
        # Note: GSM8K has no options A, B, C, D. It needs a number.
        initial_prompt = f"Can you solve the following math problem? {question} Explain your reasoning. Provide your confidence in your answer as a number between 0 and 1 (e.g., 0.95 for 95% confidence). Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."

        # Use first agent (llama3.1-8b-instruct) for initial confidence evaluation
        gate_agent = agents[0]
        gate_context = [{"role": "user", "content": initial_prompt}]

        print("  Evaluating initial confidence...")
        gate_completion, gate_usage = generate_answer(gate_context, gate_agent)

        # Extract confidence from response
        gate_response = gate_completion["choices"][0]["message"]["content"]
        confidence_score = extract_confidence(gate_response)

        print(f"  Initial confidence: {confidence_score}")

        # Check if confidence meets threshold
        if confidence_score >= confidence_threshold:
            print(f"  Confidence {confidence_score} >= {confidence_threshold}, skipping debate")
            # Store only the gate agent's response
            agent_contexts = [gate_context + [{"role": "assistant", "content": gate_response}]]
            generated_description[question] = (agent_contexts, answer, confidence_score, "gated")
        else:
            print(f"  Confidence {confidence_score} < {confidence_threshold}, proceeding with debate")

            # Initialize debate contexts for all agents
            agent_contexts = [[{"role": "user", "content": initial_prompt}] for _ in range(len(agents))]

            for round in range(rounds):
                print(f"  Round {round+1}...")
                for idx, agent_context in enumerate(agent_contexts):

                    if round != 0:
                        agent_contexts_other = agent_contexts[:idx] + agent_contexts[idx+1:]
                        message = construct_message(agent_contexts_other, question, 2*round - 1)
                        agent_context.append(message)

                    completion, usage = generate_answer(agent_context, agents[idx])

                    # Track usage
                    if usage:
                        total_tokens += usage.get('total_tokens', 0)
                        total_flops += usage.get('flops', 0)
                        api_calls += 1

                    print(f"    Agent {idx+1} ({agent_names[idx].split('/')[-1]}) finished ({usage.get('total_tokens')} tok)")

                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)

            generated_description[question] = (agent_contexts, answer, confidence_score, "debated")

        # --- AUTO SAVE ---
        with open(outfile, "w") as f:
            json.dump(generated_description, f, indent=4)
        print(f"    [Saved] {len(generated_description)}/25 completed.")
        # -----------------

    # Final Summary
    print("\n" + "="*60)
    print(f"DEBATE SYSTEM COMPUTE USAGE SUMMARY")
    print("="*60)
    print(f"Agents used: {', '.join([name.split('/')[-1] for name in agent_names])}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Debate rounds: {rounds}")
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Total PetaFLOPs: {total_flops / 1e15:.6f}")
    print("="*60)

    # Calculate statistics
    total_questions = len(generated_description)
    gated_count = sum(1 for v in generated_description.values() if len(v) > 3 and v[3] == "gated")
    debated_count = sum(1 for v in generated_description.values() if len(v) > 3 and v[3] == "debated")

    print("DEBATE STATISTICS")
    print("="*60)
    print(f"Total questions processed: {total_questions}")
    print(f"Questions gated (high confidence): {gated_count}")
    print(f"Questions debated (low confidence): {debated_count}")
    if total_questions > 0:
        print(".1f")
    print("="*60)

    print("Done.")
