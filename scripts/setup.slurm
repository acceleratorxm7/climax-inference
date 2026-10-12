#!/bin/bash
#SBATCH --job-name=climax-setup
#SBATCH --account=coc
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=climax-setup-%j.log

set -euo pipefail
cd "$HOME/climax-inference"
bash scripts/setup.sh
