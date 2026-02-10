#!/bin/bash -l
#SBATCH --job-name=mfa_align
#SBATCH --partition=h24cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mfa_align_%j_slurm.out
#SBATCH --account=OD-237777

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

# ---- Config (override via env when submitting) ----
# Required:
#   CORPUS_DIR: folder with paired files: utt.wav + utt.lab
#   MFA_OUTPUT_DIR: where aligned files are written
# Optional:
#   MFA_CONDA_ENV (default: mfa)
#   MFA_DICTIONARY (default: english_us_arpa)
#   MFA_ACOUSTIC_MODEL (default: english_us_arpa)
#   MFA_OUTPUT_FORMAT (default: long_textgrid)
#   METADATA_TSV + VCTK_TXT_ROOT (if set, auto-generate .lab files)
#   LAB_OUTPUT_DIR (default: CORPUS_DIR)
#   LAB_MANIFEST_OUT (optional TSV summary path)
#   PHONEME_JSON_DIR (if set, tries to convert CTM -> JSON)
#   SAMPLE_RATE (default: 16000)
#   HOP_LENGTH (default: 256)

MFA_CONDA_ENV="${MFA_CONDA_ENV:-mfa}"
MFA_DICTIONARY="${MFA_DICTIONARY:-english_us_arpa}"
MFA_ACOUSTIC_MODEL="${MFA_ACOUSTIC_MODEL:-english_us_arpa}"
MFA_OUTPUT_FORMAT="${MFA_OUTPUT_FORMAT:-long_textgrid}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
HOP_LENGTH="${HOP_LENGTH:-256}"

LAB_OUTPUT_DIR="${LAB_OUTPUT_DIR:-${CORPUS_DIR:-}}"
LAB_MANIFEST_OUT="${LAB_MANIFEST_OUT:-}"

if [[ -z "${CORPUS_DIR:-}" ]]; then
  echo "[ERR] CORPUS_DIR is required"
  exit 1
fi
if [[ -z "${MFA_OUTPUT_DIR:-}" ]]; then
  echo "[ERR] MFA_OUTPUT_DIR is required"
  exit 1
fi

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_PATH="${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]:-$0}}"
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
fi

REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${SCRIPT_DIR}" && pwd))"
mkdir -p "${SCRIPT_DIR}/logs"

conda activate "${MFA_CONDA_ENV}"

echo "[paths] CORPUS_DIR=${CORPUS_DIR}"
echo "[paths] MFA_OUTPUT_DIR=${MFA_OUTPUT_DIR}"
echo "[cfg] MFA_DICTIONARY=${MFA_DICTIONARY}"
echo "[cfg] MFA_ACOUSTIC_MODEL=${MFA_ACOUSTIC_MODEL}"
echo "[cfg] MFA_OUTPUT_FORMAT=${MFA_OUTPUT_FORMAT}"

mkdir -p "${MFA_OUTPUT_DIR}"

# Optional transcript extraction from ASVspoof2019 VCTK metadata.
if [[ -n "${METADATA_TSV:-}" || -n "${VCTK_TXT_ROOT:-}" ]]; then
  if [[ -z "${METADATA_TSV:-}" || -z "${VCTK_TXT_ROOT:-}" ]]; then
    echo "[ERR] To auto-generate .lab files, set both METADATA_TSV and VCTK_TXT_ROOT"
    exit 1
  fi

  mkdir -p "${LAB_OUTPUT_DIR}"
  EXTRACT_CMD=(
    python "${REPO_ROOT}/utils/phoneme/build_vctk_lab_from_metadata.py"
    --metadata-tsv "${METADATA_TSV}"
    --txt-root "${VCTK_TXT_ROOT}"
    --output-dir "${LAB_OUTPUT_DIR}"
  )
  if [[ -n "${LAB_MANIFEST_OUT}" ]]; then
    EXTRACT_CMD+=(--manifest-out "${LAB_MANIFEST_OUT}")
  fi
  "${EXTRACT_CMD[@]}"
  echo "[OK] Transcript .lab generation finished (output=${LAB_OUTPUT_DIR})"
fi

# Align transcript+audio to produce phoneme/word timestamps.
mfa align \
  "${CORPUS_DIR}" \
  "${MFA_DICTIONARY}" \
  "${MFA_ACOUSTIC_MODEL}" \
  "${MFA_OUTPUT_DIR}" \
  --clean \
  --output_format "${MFA_OUTPUT_FORMAT}"

echo "[OK] MFA alignment finished"

# Optional: if CTM exists and PHONEME_JSON_DIR is set, convert to one-JSON-per-utterance.
if [[ -n "${PHONEME_JSON_DIR:-}" ]]; then
  CTM_PATH="${CTM_PATH:-}"
  if [[ -z "${CTM_PATH}" ]]; then
    CTM_CANDIDATE="$(find "${MFA_OUTPUT_DIR}" -type f -name '*.ctm' | head -n 1 || true)"
    if [[ -n "${CTM_CANDIDATE}" ]]; then
      CTM_PATH="${CTM_CANDIDATE}"
    fi
  fi

  if [[ -n "${CTM_PATH}" && -f "${CTM_PATH}" ]]; then
    mkdir -p "${PHONEME_JSON_DIR}"
    python "${REPO_ROOT}/utils/phoneme/export_phoneme_json.py" \
      --input "${CTM_PATH}" \
      --input-format ctm \
      --output-dir "${PHONEME_JSON_DIR}" \
      --sample-rate "${SAMPLE_RATE}" \
      --hop-length "${HOP_LENGTH}" \
      --drop-silence
    echo "[OK] Phoneme JSON exported: ${PHONEME_JSON_DIR}"
  else
    echo "[WARN] PHONEME_JSON_DIR set, but no CTM found. Set CTM_PATH explicitly if needed."
  fi
fi

echo "[DONE]"
