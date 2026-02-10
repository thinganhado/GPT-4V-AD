#!/usr/bin/env bash
set -euo pipefail

# Local debug runner for MFA alignment.
# Run from anywhere:
#   bash run_mfa_debug.sh

cd /scratch3/che489/Ha/interspeech/localization

mkdir -p /scratch3/che489/Ha/.mfa_root
mkdir -p /scratch3/che489/Ha/.tmp

export CORPUS_DIR=/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_corpus
export MFA_OUTPUT_DIR=/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned_debug
export PHONEME_JSON_DIR=/scratch3/che489/Ha/interspeech/datasets/vocv4_phoneme_json_debug

export MFA_CONDA_ENV=deepfake
export MFA_DICTIONARY=english_us_arpa
export MFA_ACOUSTIC_MODEL=english_us_arpa
export MFA_OUTPUT_FORMAT=ctm
export SAMPLE_RATE=16000
export HOP_LENGTH=256

export MFA_ROOT_DIR=/scratch3/che489/Ha/.mfa_root
export TMPDIR=/scratch3/che489/Ha/.tmp

bash -x run_mfa_align.sbatch
echo "exit_code=$?"
