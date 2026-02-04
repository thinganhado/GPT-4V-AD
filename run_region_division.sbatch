#!/bin/bash -l
#SBATCH --job-name=region_division
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/region_division_%j.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:4
#SBATCH --array=0-3

set -euo pipefail

mkdir -p logs

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
conda activate deepfake

REPO_ROOT="/scratch3/che489/Ha/interspeech/localization"
INPUT_DIR="/scratch3/che489/Ha/interspeech/localization/specs"
OUT_DIR="/scratch3/che489/Ha/interspeech/localization/Ms_region_outputs"

NUM_SHARDS=4
SHARD_IDX="${SLURM_ARRAY_TASK_ID:-0}"

cd "${REPO_ROOT}"

python region_division.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --region_division_methods grid superpixel sam \
  --img_size 768 \
  --div_num 10 \
  --device cuda \
  --num-shards "${NUM_SHARDS}" \
  --shard-id "${SHARD_IDX}" \
  --overwrite \
  "$@"
