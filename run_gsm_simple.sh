#!/bin/bash
# Simple GSM Extension Runner
# This script will:
# 1. Set up Python virtual environment and install dependencies
# 2. Run the GSM debate generation script with 3 agents and 3 rounds
# 3. Run evaluation on the generated results
# Usage: ./run_gsm_simple.sh

set -e

echo "Starting GSM Extension Run"
echo "Start time: $(date)"

# Change to the project directory (adjust path as needed)
cd "$(dirname "$0")"

# Set up Python environment
export PYTHONPATH="$(pwd)":$PYTHONPATH

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update requirements
echo "Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run the generation script
echo "Running generation script..."
cd extension/gsm
python gen_OS.py --rounds 3 --confidence_threshold 0.95

# Find the generated file (it follows the pattern debate_gsm_*)
GENERATED_FILE=$(ls debate_gsm_*.json | head -1)

if [ -z "$GENERATED_FILE" ]; then
    echo "Error: No generated file found!"
    exit 1
fi

echo "Found generated file: $GENERATED_FILE"

# Run evaluation
echo "Running evaluation..."
# The eval script expects a different filename format, so we pass it directly
python eval_OS.py --agents 3 --rounds 3 "$GENERATED_FILE"

echo "Run completed successfully at $(date)"
