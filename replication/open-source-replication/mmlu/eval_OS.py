import json
import numpy as np
import re
import argparse
import sys
import os

# ==========================================
# ROBUST PARSING LOGIC
# ==========================================

def parse_answer(input_str):
    """
    Extracts the answer letter (A, B, C, D, E) from the model's output.
    """
    # 1. Clean up the string
    # Remove markdown bolding
    input_str = input_str.replace("**", "").replace("'", "").replace('"', "")
    
    # 2. STRATEGY A: Look for the explicit requested format "(X) [Letter]"
    # Matches: "(X) A", "(X) B", etc.
    specific_pattern = r'\(X\)\s*([A-E])'
    match = re.search(specific_pattern, input_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. STRATEGY B: Look for the last standalone parenthesized letter
    # Matches: "The answer is (A).", "Therefore (B)"
    # We iterate backwards to find the last one.
    pattern = r'\(([A-E])\)'
    matches = re.findall(pattern, input_str, re.IGNORECASE)
    
    # Filter out "X" if the model wrote "(X)" literally without a letter next to it
    valid_matches = [m.upper() for m in matches if m.upper() != 'X']
    
    if valid_matches:
        return valid_matches[-1] # Return the very last one found

    # 4. STRATEGY C: Look for "Answer: A" or "Option A"
    # Matches: "Answer: A", "Option: B"
    label_pattern = r'(?:Answer|Option)?\s*:?\s*([A-E])\W*$'
    match = re.search(label_pattern, input_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 5. STRATEGY D: Last ditch effort - just grab the very last letter if it's A-E
    # This is risky but often necessary for "chatty" models that end with "So, A."
    last_word_pattern = r'\b([A-E])\b\W*$'
    match = re.search(last_word_pattern, input_str)
    if match:
        return match.group(1).upper()

    return None


def extract_pred_answers(pred_solutions):
    pred_answers = []
    for s in pred_solutions:
        pa = parse_answer(s)
        # If parse returns None, we leave it as None (skipped)
        pred_answers.append(pa)
    return pred_answers


def compute_accuracy(gt, pred_solutions):
    # Support list of model outputs (Multi-Agent) or single string.
    if isinstance(pred_solutions, list):
        pred_answers = []

        for pred_solution in pred_solutions:
            pa = parse_answer(pred_solution)
            if pa is not None:
                pred_answers.append(pa)

        # If we couldn't extract any parsable answer from any model output, return None
        if len(pred_answers) == 0:
            return None

        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer(pred_solutions)
        if pred_answer is None:
            return None

    return 1 if gt == pred_answer else 0


def most_frequent(List):
    if not List: return None
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="mmlu-25-_1_1_llama3.1_8b.json", 
                        help="input JSON file produced by generation")
    parser.add_argument("-o", "--output", help="output JSON evaluation filename")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input file '{args.input}' not found.")
        print("Please check the filename.")
        sys.exit(1)

    print(f"Loading results from: {args.input}")
    try:
        with open(args.input, "r") as f:
            response_dict = json.load(f)
    except json.JSONDecodeError:
        print("[!] Error: JSON file is corrupted. Try running the script again.")
        sys.exit(1)
        
    questions = list(response_dict.keys())

    accuracies = []
    total_questions = 0
    skipped = 0
    correct = 0

    results = []

    for question in questions:
        total_questions += 1
        
        # Unpacking
        responses, gt = response_dict[question]

        # Extract content from messages
        pred_solutions = []
        for response in responses:
            # Handle list of dicts (messages)
            if isinstance(response, list):
                # Get the content of the LAST message (the assistant's final answer)
                last_msg = response[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    pred_solutions.append(last_msg['content'])
                else:
                    pred_solutions.append(str(last_msg))
            # Handle direct dict
            elif isinstance(response, dict) and 'content' in response:
                pred_solutions.append(response['content'])
            # Handle strings
            else:
                pred_solutions.append(str(response))
        
        pred_answers = extract_pred_answers(pred_solutions)
        accurate = compute_accuracy(gt, pred_solutions)

        record = {
            "question": question,
            "ground_truth": gt,
            "raw_model_outputs": pred_solutions,
            "parsed_predictions": pred_answers,
            "accurate": int(accurate) if accurate is not None else None,
        }

        results.append(record)

        # Console Summary
        status = "CORRECT" if accurate == 1 else "WRONG" if accurate == 0 else "SKIP"
        # Print a clean summary line
        print(f"[{total_questions}/{len(questions)}] {status} | GT: {gt} | Pred: {pred_answers}")

        if accurate is None:
            skipped += 1
        else:
            accuracies.append(float(accurate))
            if int(accurate) == 1:
                correct += 1

    # Final Stats
    if len(accuracies) > 0:
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        stderr = float(std_acc / (len(accuracies) ** 0.5))
    else:
        mean_acc = 0.0
        std_acc = 0.0
        stderr = 0.0

    summary = {
        "evaluated": len(accuracies),
        "total": total_questions,
        "skipped": skipped,
        "correct": correct,
        "mean_accuracy": mean_acc,
        "std": std_acc,
        "stderr": stderr,
    }

    print("\n" + "="*50)
    print("FINAL EVALUATION SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))

    out_fname = args.output if args.output else args.input.replace('.json', '') + "_eval.json"
    with open(out_fname, 'w') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"Wrote detailed evaluation to: {out_fname}")
