#!/bin/bash
#SBATCH --job-name=climax-era5
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/era5-%j.out
#SBATCH --error=logs/era5-%j.err
set -euo pipefail
cd "$HOME/climax-inference"
module load mamba/1.4.9
conda run --no-capture-output -p "$PWD/climax-env" python scripts/infer.py
