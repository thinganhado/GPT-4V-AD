#!/usr/bin/env python3
"""Create an attribute-consistent dataset copy with P=none for no-phone rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path


TIME_RANGE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*"
    r"(?:s|sec(?:ond)?s?)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy dataset.csv while replacing unvoiced/no-phone labels and rewriting En."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic variation of no-phone explanation wording.",
    )
    return parser.parse_args()


def time_prefix(explanation: str) -> str:
    match = TIME_RANGE.search(str(explanation or ""))
    if not match:
        return ""
    start_text, end_text = match.group(1), match.group(2)
    # A small number of source explanations contain reversed endpoints. Keep
    # their original precision while restoring chronological order.
    if float(start_text) > float(end_text):
        start_text, end_text = end_text, start_text
    return f"From {start_text} to {end_text} seconds, "


def artifact_kind(explanation: str) -> str:
    text = str(explanation or "").lower()
    if "noise flatten" in text or "flattened noise" in text or "uniform noise" in text:
        return "noise_flattening"
    if "formant fad" in text or "formant" in text and ("thin" in text or "fragment" in text):
        return "formant_fading"
    if any(token in text for token in ("periodic texture", "periodic repetition", "duplicated parallel", "repetitive pattern", "harmonic regularity", "harmonic uniformity")):
        return "periodic_repetition"
    if any(token in text for token in ("harmonic degradation", "harmonic smear", "smeared harmonic", "harmonic stack")):
        return "harmonic_degradation"
    if any(token in text for token in ("transient", "onset", "energy dropout", "spectral discontinuity", "temporal discontinuity")):
        return "transient_degradation"
    if any(token in text for token in ("smooth", "flat", "featureless", "spectral hole")):
        return "over_smoothing"
    return "spectral_irregularity"


OBSERVATIONS = {
    "noise_flattening": "shows an unnaturally uniform noise floor with too little fine-grained spectral variation",
    "formant_fading": "shows weakened and fragmented resonant structure",
    "periodic_repetition": "shows overly regular, repeated spectral structure that is unlikely to arise from natural vocal variation",
    "harmonic_degradation": "shows smeared or irregular harmonic energy with reduced spectral definition",
    "transient_degradation": "shows a weak, smeared spectral transition with missing transient detail",
    "over_smoothing": "shows excessive spectral smoothing and a loss of natural fine detail",
    "spectral_irregularity": "shows an abnormal spectral pattern with less natural variation than expected",
}


VOWEL_EFFECTS = {
    "noise_flattening": "This loss of spectral texture can weaken the natural resonance of the aligned vowel and make it sound hollow or synthetic.",
    "formant_fading": "Because the region is aligned with a vowel, the loss of formant definition can reduce vowel clarity and natural resonance.",
    "periodic_repetition": "Because the region is aligned with a vowel, the excessive regularity can make its sustained resonance sound robotic or mechanical.",
    "harmonic_degradation": "Because the region is aligned with a vowel, the degraded harmonic structure can blur its resonance and make it sound synthetic.",
    "transient_degradation": "The region is aligned with a vowel, and the weak transition can make its onset or offset sound blurred and unnatural.",
    "over_smoothing": "The region is aligned with a vowel, and this over-smoothing can make its resonance sound overly clean, hollow, or artificial.",
    "spectral_irregularity": "The region is aligned with a vowel, and this abnormal structure can reduce its clarity and perceived naturalness.",
}


CONSONANT_EFFECTS = {
    "noise_flattening": "This loss of fine spectral texture can suppress the transient or turbulent detail of the aligned consonant and make it sound dull or synthetic.",
    "formant_fading": "The region is aligned with a consonant, and the loss of localized spectral definition can make its articulation weak or indistinct.",
    "periodic_repetition": "The region is aligned with a consonant, and the repeated structure can replace a natural transient with a robotic or mechanical pattern.",
    "harmonic_degradation": "The region is aligned with a consonant, and the smeared energy can reduce the sharpness of its onset and articulation.",
    "transient_degradation": "Because the region is aligned with a consonant, the loss of transient definition can directly reduce articulation and intelligibility.",
    "over_smoothing": "The region is aligned with a consonant, and this over-smoothing can make it sound muffled, weak, or artificially clean.",
    "spectral_irregularity": "The region is aligned with a consonant, and this abnormal structure can reduce articulation and perceived naturalness.",
}


NON_SPEECH_NO_PHONE_SENTENCES = (
    "The region falls outside aligned speech, so no phone can be assigned to it.",
    "The region does not overlap aligned speech, so there is no corresponding phone.",
    "No aligned speech occurs in this region, leaving it without an associated phone.",
    "This region lies outside the aligned speech intervals and has no corresponding phone.",
    "Because the region is outside the aligned speech span, no phone is associated with it.",
    "The alignment contains no speech at this location, so no phone can be matched to the region.",
    "This region occurs during a non-speech interval, with no phone available for alignment.",
    "No speech interval is aligned with this region, so it has no phone correspondence.",
)


SPEECH_NO_PHONE_SENTENCES = (
    "The region overlaps speech timing, but no phone can be reliably assigned to it.",
    "Although this region occurs during speech, it has no reliable phone alignment.",
    "This region overlaps an aligned speech interval, but no specific phone can be matched to it.",
    "Speech is present at this location, but the alignment does not identify a corresponding phone.",
    "The region falls within speech timing, yet no phone is reliably associated with it.",
    "Although the region overlaps speech activity, it cannot be matched reliably to a phone.",
)


def choose_stable_variant(row: dict[str, str], variants: tuple[str, ...], seed: int) -> str:
    key = f"{seed}|{row.get('sample_id', '')}|{row.get('region_id', '')}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(variants)
    return variants[index]


def rewrite_explanation(row: dict[str, str], original_p: str, seed: int) -> str:
    old = str(row.get("En", ""))
    t_value = str(row.get("T", "")).strip().lower()
    f_value = str(row.get("F", "")).strip().lower()
    p_value = str(row.get("P", "")).strip().lower()
    kind = artifact_kind(old)
    prefix = time_prefix(old)
    first = f"{prefix}this {f_value}-frequency region {OBSERVATIONS[kind]}."
    if first:
        first = first[0].upper() + first[1:]

    if original_p == "unvoiced" or p_value == "none":
        if t_value in {"non_speech", "non-speech", "nonspeech"}:
            alignment = choose_stable_variant(row, NON_SPEECH_NO_PHONE_SENTENCES, seed)
        else:
            alignment = choose_stable_variant(row, SPEECH_NO_PHONE_SENTENCES, seed)
        return " ".join((first, alignment))

    if p_value == "vowel":
        return " ".join((first, VOWEL_EFFECTS[kind]))
    if p_value == "consonant":
        return " ".join((first, CONSONANT_EFFECTS[kind]))
    raise ValueError(f"Unsupported P value: {p_value!r}")


def main() -> None:
    args = parse_args()
    source = Path(args.input_csv).expanduser().resolve()
    destination = Path(args.output_csv).expanduser().resolve()
    if source == destination:
        raise ValueError("Input and output paths must differ; the source CSV is preserved.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with source.open("r", encoding="utf-8-sig", newline="") as src, destination.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        required = {"sample_id", "region_id", "T", "F", "P", "En"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            original_p = str(row.get("P", "")).strip().lower()
            if original_p == "unvoiced":
                row["P"] = "none"
                counts["changed_unvoiced_to_none"] += 1
            row["En"] = rewrite_explanation(row, original_p, args.seed)
            writer.writerow(row)
            counts["rows_written"] += 1

    print(f"input={source}")
    print(f"output={destination}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")


if __name__ == "__main__":
    main()
