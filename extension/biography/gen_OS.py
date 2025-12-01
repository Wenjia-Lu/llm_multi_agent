import sys
import os
import time
import random
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG_FILE = "llm/configs/llama3.1-8B-instruct.json"
DEFAULT_AGENTS = 1
DEFAULT_ROUNDS = 1

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
parser.add_argument("--config", default=DEFAULT_CONFIG_FILE,
                   help=f"Path to LLM config JSON file (default: {DEFAULT_CONFIG_FILE})")
parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS,
                   help=f"Number of agents to use (default: {DEFAULT_AGENTS})")
parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help=f"Number of debate rounds (default: {DEFAULT_ROUNDS})")
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
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text) // 4

def calculate_flops(token_count):
    return 2 * config.parameter_count * token_count

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []
    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
            bullet = bullet[idx:]
            if len(bullet) != 0:
                bullets.append(bullet)
        except:
            continue
    return bullets

def filter_people(person):
    return person.split("(")[0].strip()

def construct_message(agents, idx, person, final=False):
    prefix_string = "Here are some bullet point biographies of {} given by other agents: ".format(person)

    if len(agents) == 0:
        return {"role": "user", "content": "Closely examine your biography and provide an updated bullet point biography."}

    for i, agent in enumerate(agents):
        try:
            agent_response = agent[idx]["content"]
            response = "\n\n Agent response: ```{}```".format(agent_response)
            prefix_string = prefix_string + response
        except (IndexError, KeyError, TypeError):
            continue

    if final:
        prefix_string += "\n\n Closely examine your biography and the biography of other agents and provide an updated bullet point biography."
    else:
        prefix_string += "\n\n Using these other biographies of {} as additional advice, what is your updated bullet point biography of the computer scientist {}?".format(person, person)

    return {"role": "user", "content": prefix_string}

def construct_assistant_message(content):
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
                "flops": flops
            }
        }
        return completion

    except Exception as e:
        print(f"Error calling model: {e}")
        print("Retrying...")
        time.sleep(2)
        return generate_answer(answer_context)

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    pass  # Arguments already parsed above

    input_file = "article.json"
    if not os.path.exists(input_file):
        data_path = project_root / "data" / "biography" / "article.json"
        if os.path.exists(data_path):
            input_file = str(data_path)
        else:
            print(f"[!] Error: {input_file} not found.")
            sys.exit(1)

    with open(input_file, "r") as f:
        data = json.load(f)
    
    people = sorted(data.keys())
    people = [filter_people(person) for person in people]
    
    random.seed(1)
    random.shuffle(people)

    # === LIMIT TO 25 PEOPLE ===
    people = people[:25]
    print(f"Processing subset of {len(people)} people.")
    # ==========================

    agents = args.agents
    rounds = args.rounds

    total_tokens = 0
    total_flops = 0
    api_calls = 0

    # Extract config name from config file path
    config_name = Path(args.config).stem  # Gets filename without extension
    config_short_name = config_name.replace("-", "_").replace("/", "_")
    outfile = f"biography_{agents}_{rounds}_{config_short_name}.json"
    generated_description = {}

    if os.path.exists(outfile):
        print(f"[Resume] Found existing file: {outfile}")
        try:
            with open(outfile, "r") as f:
                generated_description = json.load(f)
            print(f"[Resume] Loaded {len(generated_description)} completed biographies.")
        except:
            print("[Resume] File corrupted. Starting fresh.")

    for person in tqdm(people):
        
        if person in generated_description:
            continue

        agent_contexts = [[{"role": "user", "content": "Give a bullet point biography of {} highlighting their contributions and achievements as a computer scientist, with each fact separated with a new line character. ".format(person)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]

                    if round == (rounds - 1):
                        message = construct_message(agent_contexts_other, 2*round - 1, person=person, final=True)
                    else:
                        message = construct_message(agent_contexts_other, 2*round - 1, person=person, final=False)
                    agent_context.append(message)

                completion = generate_answer(agent_context)

                if completion.get('usage'):
                    usage = completion['usage']
                    total_tokens += usage.get('total_tokens', 0)
                    total_flops += usage.get('flops', 0)
                    api_calls += 1

                content = completion["choices"][0]['message']['content']
                assistant_message = construct_assistant_message(content)
                agent_context.append(assistant_message)

            bullets = parse_bullets(content)
            if len(bullets) <= 1:
                break

        generated_description[person] = agent_contexts

        with open(outfile, "w") as f:
            json.dump(generated_description, f, indent=4)

    print("\n" + "="*50)
    print(f"COMPUTE USAGE SUMMARY ({config.model_name} Bio)")
    print("="*50)
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Total PetaFLOPs: {total_flops / 1e15:.6f}")
    print("="*50)

    print(f"Results saved to {outfile}")
