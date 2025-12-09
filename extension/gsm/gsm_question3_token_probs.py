#!/usr/bin/env python3
"""
Script to run GSM question 3 on llama3.1-8B-instruct and extract token probabilities.
Outputs token probabilities to CSV format and final confidence score.
"""

import sys
import os
import json
import csv
from pathlib import Path

# Add project root to path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent  # gsm -> extension -> llm_multi_agent
sys.path.append(str(project_root))

try:
    from llm.factory import LLMFactory
    import torch
except ImportError as e:
    print(f"[!] Import Error: {e}")
    sys.exit(1)

def load_gsm_question_3():
    """Load question 3 from GSM test set."""
    gsm_file = Path(__file__).parent / "test.jsonl"

    if not gsm_file.exists():
        print(f"[!] Error: Could not find GSM test file at {gsm_file}")
        sys.exit(1)

    # Read all questions
    questions = []
    with open(gsm_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if len(questions) < 3:
        print(f"[!] Error: Only {len(questions)} questions found, need at least 3")
        sys.exit(1)

    question_data = questions[2]  # 0-indexed, so index 4 is question 3
    return question_data['question'], question_data['answer']

def extract_token_probabilities(logits, tokenizer, generated_tokens):
    """
    Extract token probabilities from logits.

    Args:
        logits: List of tensors containing logits for each generated token
        tokenizer: The tokenizer used by the model
        generated_tokens: The actual generated token IDs

    Returns:
        List of (token_text, probability) tuples
    """
    token_probs = []

    print(f"Debug: logits length = {len(logits)}")
    print(f"Debug: generated_tokens shape = {generated_tokens.shape}")
    print(f"Debug: generated_tokens[0] length = {len(generated_tokens[0])}")

    # Check if logits and generated_tokens align
    num_logits = len(logits)
    num_tokens = len(generated_tokens[0])

    if num_logits != num_tokens:
        print(f"Warning: logits ({num_logits}) and tokens ({num_tokens}) don't match!")
        # Use the minimum of both
        min_len = min(num_logits, num_tokens)

    for i in range(min(num_logits, num_tokens)):
        token_logits = logits[i]
        token_id = generated_tokens[0][i].item()  # Convert tensor to int

        # Apply softmax to get probabilities
        probs = torch.softmax(token_logits, dim=-1)

        # Handle batch dimension - squeeze if present
        if probs.dim() > 1:
            probs = probs.squeeze(0)  # Remove batch dimension: (1, vocab_size) -> (vocab_size,)

        # Debug info
        vocab_size = probs.shape[-1]
        if i == 0:  # Only print debug for first token to reduce noise
            print(f"Debug: step {i}, token_id={token_id}, vocab_size={vocab_size}, probs_shape={probs.shape}")

        # Check bounds
        if token_id >= vocab_size:
            print(f"Warning: token_id {token_id} >= vocab_size {vocab_size}, skipping")
            continue

        # Get the probability of the selected token
        selected_prob = probs[token_id].item()

        # Decode the token to text
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)

        token_probs.append((token_text, selected_prob))

    return token_probs

def main():
    # Load GSM question 3
    print("Loading GSM question 3...")
    question, expected_answer = load_gsm_question_3()
    print(f"Question: {question}")
    print(f"Expected answer: {expected_answer}")

    # Load llama3.1-8B-instruct model
    config_path = project_root / "llm" / "configs" / "llama3.1-8B-instruct.json"
    print(f"\nLoading model from config: {config_path}")

    try:
        llm = LLMFactory.from_config_file(str(config_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)

    # Create prompt for the question
    prompt = f"Can you solve the following math problem? {question} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."

    print(f"\nGenerating response for question...")

    # Generate response with logits enabled
    try:
        response = llm.generate(prompt, output_scores=True)
        print("Response generated successfully!")
    except Exception as e:
        print(f"[!] Failed to generate response: {e}")
        sys.exit(1)

    # Extract token probabilities
    print("\nExtracting token probabilities...")
    if hasattr(response, 'logits') and response.logits:
        # Get the generated tokens (excluding input)
        inputs = llm.tokenizer(prompt, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]
        generated_tokens = response._response_object.sequences[:, input_tokens:] if hasattr(response, '_response_object') else None

        # If we can't get generated_tokens directly, we'll need to regenerate with the model to get them
        if generated_tokens is None:
            print("Regenerating to extract token details...")
            with torch.no_grad():
                inputs = llm.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(llm.device) for k, v in inputs.items()}

                outputs = llm.model.generate(**inputs,
                                           max_new_tokens=llm.config.max_tokens,
                                           temperature=llm.config.temperature,
                                           do_sample=llm.config.temperature > 0,
                                           return_dict_in_generate=True,
                                           output_scores=True,
                                           pad_token_id=llm.tokenizer.pad_token_id)

                generated_tokens = outputs.sequences[:, input_tokens:]
                logits = outputs.scores

            token_probs = extract_token_probabilities(logits, llm.tokenizer, generated_tokens)
        else:
            # Use logits from the response object
            token_probs = extract_token_probabilities(response.logits, llm.tokenizer, generated_tokens)
    else:
        print("[!] No logits available in response")
        sys.exit(1)

    # Output to CSV
    csv_filename = "gsm_question3_token_probs.csv"
    print(f"\nWriting token probabilities to {csv_filename}...")

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['token', 'prob'])  # Header

        for token_text, prob in token_probs:
            writer.writerow([token_text, f"{prob:.6f}"])

    print(f"Written {len(token_probs)} token-probability pairs to CSV")

    # Output final confidence score
    confidence_score = response.confidence_score if hasattr(response, 'confidence_score') else 0.3
    print("\nFinal confidence score:")
    print(f"Confidence Score: {confidence_score:.6f}")

    print(f"\nScript completed successfully!")
    print(f"Generated text: {response.text[:200]}..." if len(response.text) > 200 else f"Generated text: {response.text}")

if __name__ == "__main__":
    main()
