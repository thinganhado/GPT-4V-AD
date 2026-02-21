#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Tuple


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower())


def _pick_col(fieldnames, candidates):
    if not fieldnames:
        return None
    norm_map = {_norm(x): x for x in fieldnames}
    for c in candidates:
        k = _norm(c)
        if k in norm_map:
            return norm_map[k]
    for fn in fieldnames:
        n = _norm(fn)
        for c in candidates:
            if _norm(c) in n:
                return fn
    return None


def _load_meta(meta_csv: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    with meta_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames or []

        sample_col = _pick_col(fns, ["sample_id", "sampleid"])
        region_col = _pick_col(fns, ["region_id", "regionid", "id"])
        time_col = _pick_col(fns, ["time", "temporal_context", "temporal", "time_context"])
        freq_col = _pick_col(fns, ["frequency", "freq", "spectral_band", "band"])
        phon_col = _pick_col(fns, ["phonetic", "phonetic_category", "phone", "phoneme"])

        if sample_col is None or region_col is None:
            raise ValueError(
                f"Could not find required columns in metadata CSV. "
                f"Need sample_id + region_id. Found headers: {fns}"
            )

        out: Dict[Tuple[str, str], Dict[str, str]] = {}
        for row in reader:
            sid = str(row.get(sample_col, "")).strip()
            rid = str(row.get(region_col, "")).strip()
            if not sid or not rid:
                continue
            key = (sid, rid)
            out[key] = {
                "time": str(row.get(time_col, "")).strip() if time_col else "",
                "frequency": str(row.get(freq_col, "")).strip() if freq_col else "",
                "phonetic": str(row.get(phon_col, "")).strip() if phon_col else "",
            }
    return out


def _extract_explanation(response: str) -> str:
    txt = response or ""
    m = re.search(r"<Explanation>\s*(.*?)\s*</Explanation>", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: keep whole response if tag missing.
    return txt.strip()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract time/frequency/phonetic + Explanation from selected JSONL into per-region txt files."
    )
    p.add_argument("--input-jsonl", required=True, help="Path to qwen_region_outputs_selected.jsonl")
    p.add_argument(
        "--meta-csv",
        required=True,
        help="Metadata CSV with sample_id/region_id and time/frequency/phonetic columns.",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for txt files")
    p.add_argument(
        "--pairs",
        default="",
        help="Optional semicolon-separated sample_id:region_id filters. If set, only these pairs are exported.",
    )
    return p.parse_args()


def _parse_pairs_filter(pairs: str):
    out = set()
    if not pairs.strip():
        return out
    for part in pairs.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid pair '{part}'. Expected sample_id:region_id")
        sid, rid = part.split(":", 1)
        sid = sid.strip()
        rid = rid.strip()
        if not sid or not rid:
            raise ValueError(f"Invalid pair '{part}'.")
        out.add((sid, rid))
    return out


def main():
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input-jsonl not found: {input_jsonl}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"meta-csv not found: {meta_csv}")

    meta = _load_meta(meta_csv)
    pair_filter = _parse_pairs_filter(args.pairs)

    written = 0
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                print(f"[warn] skip invalid json at line {line_no}")
                continue

            sid = str(rec.get("sample_id", "")).strip()
            rid = str(rec.get("region_id", "")).strip()
            if not sid or not rid:
                print(f"[warn] skip line {line_no}: missing sample_id/region_id")
                continue
            if pair_filter and (sid, rid) not in pair_filter:
                continue

            m = meta.get((sid, rid), {})
            explanation = _extract_explanation(str(rec.get("response", "")))

            out_text = (
                f"sample_id: {sid}\n"
                f"region_id: {rid}\n"
                f"time: {m.get('time', '')}\n"
                f"frequency: {m.get('frequency', '')}\n"
                f"phonetic: {m.get('phonetic', '')}\n"
                f"Explanation: {explanation}\n"
            )

            out_path = out_dir / f"{_safe_name(sid)}__r{_safe_name(rid)}.txt"
            out_path.write_text(out_text, encoding="utf-8")
            written += 1

    print(f"[done] wrote {written} txt files to {out_dir}")


if __name__ == "__main__":
    main()
