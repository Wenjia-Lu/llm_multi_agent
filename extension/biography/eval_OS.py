import json
import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path

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

# Parse command line arguments for config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=DEFAULT_CONFIG_FILE,
                   help=f"Path to LLM config JSON file (default: {DEFAULT_CONFIG_FILE})")
parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS,
                   help=f"Number of agents used (default: {DEFAULT_AGENTS})")
parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help=f"Number of rounds used (default: {DEFAULT_ROUNDS})")
# Add other arguments that will be parsed later
parser.add_argument("input", nargs="?", help="Input JSON file from gen_OS.py (auto-generated if not provided)")
parser.add_argument("--ref", default="article.json", help="Path to reference articles")

args = parser.parse_args()

# Auto-generate input filename if not provided
if not args.input:
    config_name = Path(args.config).stem  # Gets filename without extension
    config_short_name = config_name.replace("-", "_").replace("/", "_")
    args.input = f"biography_{args.agents}_{args.rounds}_{config_short_name}.json"

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

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_bullets(sentence):
    """
    Extracts bullet points from a text block.
    """
    bullets_preprocess = sentence.split("\n")
    bullets = []
    for bullet in bullets_preprocess:
        try:
            # Find start of text (skip bullets/numbers)
            idx = bullet.find(next(filter(str.isalpha, bullet)))
            clean_bullet = bullet[idx:].strip()
            if len(clean_bullet) > 5: # Basic length check
                bullets.append(clean_bullet)
        except:
            continue
    return bullets

def parse_yes_no(string):
    """
    Parses 'Yes', 'No', or 'Uncertain' from model output.
    """
    s = string.lower().strip()
    # Check for direct word matches
    if "uncertain" in s:
        return None
    
    # Priority check for Yes/No at start of string (common model behavior)
    if s.startswith("yes") or s.startswith("no"):
        return True if s.startswith("yes") else False
    
    # Check for presence anywhere
    if "yes" in s and "no" not in s:
        return True
    elif "no" in s and "yes" not in s:
        return False
    
    return None

def filter_people(person):
    return person.split("(")[0].strip()

def check_fact(person, bio_text, fact):
    """
    Asks the local LLM if the bio supports the specific fact.
    """
    prompt = f"""
    Consider the following biography of {person}:
    {bio_text}

    Is the biography above consistent with the fact below?
    Fact: {fact}

    Give a single word answer: Yes, No, or Uncertain.
    Carefully check precise dates, names, and locations.
    """

    try:
        response = mas_model.generate(prompt, temperature=0.0, max_tokens=10)
        return response.text.strip()
    except Exception as e:
        print(f"[!] Model Error: {e}")
        return "Error"

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
if __name__ == "__main__":
    # Arguments already parsed above
    pass

    # Locate files
    script_dir = Path(__file__).resolve().parent
    ref_path = args.ref
    
    if not os.path.exists(ref_path):
        # Try default data path
        ref_path = script_dir.parents[1] / "X-MAS" / "data" / "biography" / "article.json"
        if not os.path.exists(ref_path):
            # Try current directory
            ref_path = "article.json"
            if not os.path.exists(ref_path):
                print(f"[!] Error: Reference file '{args.ref}' not found.")
                sys.exit(1)

    print(f"Loading Results: {args.input}")
    print(f"Loading Truth:   {ref_path}")

    response_data = json.load(open(args.input, "r"))
    gt_data = json.load(open(ref_path, "r"))

    # Normalize GT keys
    gt_data_filter = {filter_people(k): v for k, v in gt_data.items()}
    gt_data = gt_data_filter

    people = list(response_data.keys())
    
    # Metrics
    accuracies = []
    total_questions = 0
    skipped = 0
    correct = 0
    results = []

    for i, person in enumerate(people):
        clean_name = filter_people(person)
        
        if clean_name not in gt_data:
            print(f"[{i+1}/{len(people)}] Skipping {clean_name} (No Ground Truth)")
            continue

        # Get Ground Truth Bullets (The facts we must verify)
        gt_description = gt_data[clean_name]
        gt_bullets = parse_bullets(gt_description)
        
        # Get Generated Biography (The last response from the debate)
        # Structure: [Round 1, Round 2] -> Round 2 is last
        # Inside Round 2: [Agent 0, Agent 1, Agent 2] -> We usually pick the last agent or merge
        # Let's take the LAST agent's final answer from the FINAL round
        try:
            agent_history = response_data[person][-1] # Last agent context
            last_message = agent_history[-1]          # Last message
            bio_description = last_message['content'] if isinstance(last_message, dict) else str(last_message)
        except (IndexError, KeyError, TypeError):
            print(f"[{i+1}/{len(people)}] Skipping {clean_name} (Bad JSON structure)")
            continue

        # Clean up bio for prompting
        bio_bullets = parse_bullets(bio_description)
        if len(bio_bullets) <= 1:
            print(f"[{i+1}/{len(people)}] Skipping {clean_name} (Bio too short)")
            continue
            
        bio_text = "\n".join(bio_bullets)

        print(f"[{i+1}/{len(people)}] Evaluating {clean_name} ({len(gt_bullets)} facts)...")

        # Verify each GT fact against the generated bio
        person_score = 0
        person_facts = 0
        
        for bullet in gt_bullets:
            total_questions += 1
            person_facts += 1
            
            # Call Llama 3.1 to Judge
            decision_text = check_fact(clean_name, bio_text, bullet)
            accurate = parse_yes_no(decision_text)

            record = {
                "person": clean_name,
                "fact": bullet,
                "model_judgment": decision_text,
                "is_consistent": accurate
            }
            results.append(record)

            if accurate is None:
                skipped += 1
            else:
                accuracies.append(float(accurate))
                if accurate:
                    correct += 1
                    person_score += 1
                    
        # Running Stats
        if len(accuracies) > 0:
            mean_acc = np.mean(accuracies)
            stderr = np.std(accuracies) / (len(accuracies) ** 0.5)
            print(f"    --> Score: {person_score}/{person_facts} | Running Avg: {mean_acc:.2%}")

    # Final Summary
    if len(accuracies) > 0:
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        stderr = float(std_acc / (len(accuracies) ** 0.5))
    else:
        mean_acc = 0.0
        std_acc = 0.0
        stderr = 0.0

    summary = {
        "evaluated_facts": len(accuracies),
        "total_facts": total_questions,
        "skipped": skipped,
        "consistent_facts": correct,
        "mean_accuracy": mean_acc,
        "std": std_acc,
        "stderr": stderr,
    }

    print("\n" + "="*50)
    print("FINAL BIOGRAPHY EVALUATION")
    print("="*50)
    print(json.dumps(summary, indent=2))

    out_fname = args.input.replace(".json", "_eval.json")
    with open(out_fname, 'w') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"Wrote evaluation results to: {out_fname}")
