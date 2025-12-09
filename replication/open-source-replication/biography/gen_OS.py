import sys
import os
import time
import random
import json
from tqdm import tqdm
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
# 2. INITIALIZE MODEL
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
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        return len(text) // 4

def calculate_flops(token_count):
    return 2 * PARAM_COUNT * token_count

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
    
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

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

    agents = 1
    rounds = 1

    total_tokens = 0
    total_flops = 0
    api_calls = 0

    outfile = f"biography_{agents}_{rounds}_llama3.1_8b.json"
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
    print("COMPUTE USAGE SUMMARY (Llama 3.1 8B Bio)")
    print("="*50)
    print(f"Total API calls: {api_calls}")
    print(f"Total tokens used: {total_tokens}")
    print(f"Total FLOPs: {total_flops:.2e}")
    print(f"Total PetaFLOPs: {total_flops / 1e15:.6f}")
    print("="*50)

    print(f"Results saved to {outfile}")
