import json
import re
import argparse
import sys
import os
import numpy as np
from collections import Counter

# ==========================================
# 1. PARSING LOGIC
# ==========================================

def clean_number(text):
    """
    Standardizes numbers: removes commas, $, and whitespace.
    Example: "$1,200" -> "1200"
    """
    if text is None: return None
    # Remove everything except digits, dots, and negative signs
    clean = re.sub(r"[^\d\.\-]", "", text)
    return clean

def parse_ground_truth(answer_str):
    """
    GSM8K ground truth usually looks like:
    "Explanation... #### 42"
    We only want the "42".
    """
    if "####" in answer_str:
        return clean_number(answer_str.split("####")[-1].strip())
    else:
        # Fallback if the dataset format is different
        return clean_number(answer_str)

def parse_model_output(output_str):
    """
    Extracts the answer from the model output.
    Priority 1: \boxed{123}
    Priority 2: The very last number in the text (Fallback)
    """
    # 1. Look for LaTeX boxed format: \boxed{...}
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(boxed_pattern, output_str)
    if matches:
        # Return the last boxed answer found (often the final conclusion)
        return clean_number(matches[-1])

    # 2. Fallback: Look for "Final Answer: X"
    # (Optional, depending on if your model does this)
    
    # 3. Last Resort: Find the very last number in the string
    # This is risky but often necessary if the model forgets \boxed
    numbers = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?", output_str)
    if numbers:
        return clean_number(numbers[-1])
    
    return None

# ==========================================
# 2. SCORING LOGIC
# ==========================================

def get_majority_vote(predictions):
    """
    Returns the most common answer among the agents.
    """
    # Filter out Nones
    valid_preds = [p for p in predictions if p is not None]
    
    if not valid_preds:
        return None
    
    # Count frequencies
    counts = Counter(valid_preds)
    # Get the most common one (returns a list of tuples like [('42', 2)])
    most_common = counts.most_common(1)
    
    return most_common[0][0]

def check_correctness(ground_truth, prediction):
    if prediction is None or ground_truth is None:
        return 0
    # Simple string comparison after cleaning usually works well for GSM8K
    return 1 if ground_truth == prediction else 0

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default matches your gen_math.py output
    parser.add_argument("input", nargs="?", default="gsm_1_1_llama3.1_8b_official.json", 
                        help="Input JSON file from gen_math.py")
    parser.add_argument("-o", "--output", help="Output JSON evaluation filename")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Loading results from: {args.input}")
    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("[!] Error: JSON file corrupted.")
        sys.exit(1)

    questions = list(data.keys())
    results = []
    accuracies = []
    
    total = 0
    correct = 0
    
    print("-" * 60)
    print(f"{'#':<5} | {'STATUS':<10} | {'GT':<10} | {'PRED (Vote)':<10} | {'RAW VOTES'}")
    print("-" * 60)

    for i, question in enumerate(questions):
        # Unpack the data: (agent_contexts, ground_truth_str)
        agent_contexts, gt_str = data[question]
        
        # 1. Parse Ground Truth
        gt_clean = parse_ground_truth(gt_str)
        
        # 2. Parse Predictions from each Agent
        agent_preds = []
        for context in agent_contexts:
            # Get the last message (Assistant's final response)
            last_msg = context[-1]['content']
            parsed = parse_model_output(last_msg)
            agent_preds.append(parsed)
            
        # 3. Majority Vote
        final_pred = get_majority_vote(agent_preds)
        
        # 4. Check Correctness
        is_correct = check_correctness(gt_clean, final_pred)
        
        accuracies.append(is_correct)
        if is_correct: correct += 1
        total += 1
        
        # Log
        status = "CORRECT" if is_correct else "WRONG"
        print(f"{i+1:<5} | {status:<10} | {str(gt_clean):<10} | {str(final_pred):<10} | {agent_preds}")
        
        results.append({
            "question": question,
            "ground_truth": gt_clean,
            "agent_predictions": agent_preds,
            "final_prediction": final_pred,
            "is_correct": is_correct
        })

    # ==========================================
    # 4. SUMMARY
    # ==========================================
    mean_acc = np.mean(accuracies) if accuracies else 0.0
    
    print("-" * 60)
    print("FINAL MATH EVALUATION")
    print("-" * 60)
    print(f"Total Questions: {total}")
    print(f"Total Correct:   {correct}")
    print(f"Accuracy:        {mean_acc:.2%}")
    print("-" * 60)

    # Save details
    out_fname = args.output if args.output else args.input.replace(".json", "_eval.json")
    with open(out_fname, "w") as f:
        json.dump({
            "summary": {"accuracy": mean_acc, "total": total, "correct": correct},
            "details": results
        }, f, indent=4)
        
    print(f"Saved details to {out_fname}")
