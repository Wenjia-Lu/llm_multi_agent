#!/bin/bash
# GSM Extension Runner for Great Lakes HPC
# This script will:
# 1. Set up Python virtual environment and install dependencies
# 2. Run the GSM debate generation script with 3 agents and 3 rounds
# 3. Run evaluation on the generated results
# Submit with: sbatch run_gsm_extension.sh

#SBATCH --job-name=gsm_extension
#SBATCH --account=eecs498w25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=gsm_extension_%j.out
#SBATCH --error=gsm_extension_%j.err

# Exit on any error
set -e

# Print job information
echo "Starting GSM Extension Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Change to the project directory
cd /home/$USER/llm_multi_agent

# Set up Python environment
export PYTHONPATH=/home/$USER/llm_multi_agent:$PYTHONPATH

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

echo "Job completed successfully at $(date)"
