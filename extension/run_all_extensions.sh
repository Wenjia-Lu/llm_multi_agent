#!/bin/bash
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00

# Exit on any error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules for Great Lakes
echo "Loading Python module..."
module load python/3.13.2

# Print job information
echo "=========================================="
echo "Starting All Extensions Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Note: This script assumes it's being run from the llm_multi_agent directory
PROJECT_DIR="$(pwd)"

# Set up Python environment
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Hardcoded Hugging Face token for Meta model downloads
# REPLACE THIS WITH YOUR ACTUAL TOKEN
# Should be in an .env file or secret manager, but for simplicity, we'll hardcode it here, code was only ran on Great Lakes HPC.
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Set up Hugging Face authentication for downloading Meta models
echo "Setting up Hugging Face authentication..."
if [ "$HF_TOKEN" = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" ]; then
    echo "WARNING: HF_TOKEN is still set to placeholder value!"
    echo "Please replace the placeholder token in this script with your actual Hugging Face token"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Accept the license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    echo ""
    echo "Continuing with placeholder token (will likely fail)..."
fi

echo "Setting up Hugging Face authentication with token..."
# Install huggingface_hub if not in requirements
pip install huggingface_hub
# Login using the hardcoded token
python -c "from huggingface_hub import login; login('$HF_TOKEN')"
echo "Hugging Face authentication complete"

# Install/update requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical packages are installed
echo "Verifying critical packages..."
python test_imports.py || { echo "ERROR: Some packages not installed properly"; exit 1; }

echo "=========================================="
echo "STARTING GSM EXTENSION"
echo "=========================================="

# Run GSM extension
cd extension/gsm
echo "Running GSM generation script..."
python gen_OS.py --rounds 3 --confidence_threshold 0.8

# Find the generated file (it follows the pattern debate_gsm_*)
GENERATED_FILE=$(ls debate_gsm_*.json | head -1)

if [ -z "$GENERATED_FILE" ]; then
    echo "Error: No GSM generated file found!"
    exit 1
fi

echo "Found GSM generated file: $GENERATED_FILE"

# Run GSM evaluation
echo "Running GSM evaluation..."
python eval_OS.py --agents 3 --rounds 3 "$GENERATED_FILE"

echo "GSM extension completed at $(date)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "STARTING BIOGRAPHY EXTENSION"
echo "=========================================="

# Run Biography extension
cd extension/biography
echo "Running Biography extension..."
python gen_OS.py --rounds 3 --confidence_threshold 0.8

# Find the generated file (it follows the pattern debate_biography_*)
BIO_GENERATED_FILE=$(ls debate_biography_*.json | head -1)

if [ -z "$BIO_GENERATED_FILE" ]; then
    echo "Error: No Biography generated file found!"
    exit 1
fi

echo "Found Biography generated file: $BIO_GENERATED_FILE"

# Run Biography evaluation
echo "Running Biography evaluation..."
python eval_OS.py --agents 3 --rounds 3 "$BIO_GENERATED_FILE"

echo "Biography extension completed at $(date)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "STARTING MMLU EXTENSION"
echo "=========================================="

# Run MMLU extension
cd extension/mmlu
echo "Running MMLU extension..."
python gen_OS.py --rounds 3 --confidence_threshold 0.8

# Find the generated file (it follows the pattern debate_mmlu_*)
MMLU_GENERATED_FILE=$(ls debate_mmlu_*.json | head -1)

if [ -z "$MMLU_GENERATED_FILE" ]; then
    echo "Error: No MMLU generated file found!"
    exit 1
fi

echo "Found MMLU generated file: $MMLU_GENERATED_FILE"

# Run MMLU evaluation
echo "Running MMLU evaluation..."
python eval_OS.py --agents 3 --rounds 3 "$MMLU_GENERATED_FILE"

echo "MMLU extension completed at $(date)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "ALL EXTENSIONS COMPLETED SUCCESSFULLY"
echo "Job completed at $(date)"
echo "=========================================="

Open Directory in Terminal