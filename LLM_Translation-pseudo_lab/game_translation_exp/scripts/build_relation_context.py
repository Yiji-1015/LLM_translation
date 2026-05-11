#!/usr/bin/env python3
"""
Build per-sample relation context to inject into translation prompts.

Inputs:
- samples_tagged_v1.csv (or samples.csv)
- relation_edges_confirmed.csv (preferred) or relation_edges_auto.csv
- character_seeds.csv

Output:
- sample_relation_context.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--base-dir",
        default=str(default_base_dir),
        help="Project base directory (game_translation_exp).",
    )
    parser.add_argument(
        "--samples-csv",
        default=None,
    )
    parser.add_argument(
        "--edges-csv",
        default=None,
        help="If missing, script falls back to relation_edges_auto.csv",
    )
    parser.add_argument(
        "--seed-csv",
        default=None,
    )
    parser.add_argument(
        "--out-csv",
        default=None,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Maximum relation lines per sample",
    )
    return parser.parse_args()


def load_seed_aliases(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            entity = (row.get("entity") or "").strip()
            aliases = (row.get("aliases") or "").strip()
            alias_list = [a.strip() for a in aliases.split("|") if a.strip()]
            if entity and entity not in alias_list:
                alias_list.append(entity)
            if entity:
                out[entity] = sorted(set(alias_list), key=lambda x: (-len(x), x))
    return out


def compile_alias_patterns(aliases_by_entity: Dict[str, List[str]]) -> List[Tuple[str, re.Pattern]]:
    patterns = []
    for entity, aliases in aliases_by_entity.items():
        escaped = [re.escape(a) for a in aliases]
        pat = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(escaped) + r")(?![A-Za-z0-9_])", re.I)
        patterns.append((entity, pat))
    return patterns


def detect_entities(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    found = []
    for entity, pat in patterns:
        if pat.search(text):
            found.append(entity)
    return found


def load_edges(edges_csv: Path) -> List[dict]:
    rows = []
    with edges_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def conf_rank(conf: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get((conf or "").lower(), 0)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    samples_csv = Path(args.samples_csv).expanduser() if args.samples_csv else base_dir / "data" / "samples_tagged_v1.csv"
    edges_csv = Path(args.edges_csv).expanduser() if args.edges_csv else base_dir / "data" / "relation_kg" / "relation_edges_confirmed.csv"
    seed_csv = Path(args.seed_csv).expanduser() if args.seed_csv else base_dir / "data" / "relation_kg" / "character_seeds.csv"
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else base_dir / "data" / "relation_kg" / "sample_relation_context.csv"

    if not edges_csv.exists():
        fallback = edges_csv.with_name("relation_edges_auto.csv")
        if fallback.exists():
            edges_csv = fallback
        else:
            raise FileNotFoundError(
                f"No edges file found: {edges_csv} or {fallback}. Run extract_relation_candidates.py first."
            )

    aliases_by_entity = load_seed_aliases(seed_csv)
    patterns = compile_alias_patterns(aliases_by_entity)
    edges = load_edges(edges_csv)

    # Build adjacency for fast retrieval
    adjacency: Dict[str, List[dict]] = {}
    for e in edges:
        src = (e.get("source_character") or "").strip()
        adjacency.setdefault(src, []).append(e)
        # Also include reverse lookup as context candidate
        tgt = (e.get("target_character") or "").strip()
        adjacency.setdefault(tgt, []).append(e)

    output_rows = []

    with samples_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            source_text = (row.get("source_text") or "").strip()
            sentence_type = (row.get("sentence_type") or "").strip()

            ents = detect_entities(source_text, patterns)
            rel_candidates = []
            for ent in ents:
                rel_candidates.extend(adjacency.get(ent, []))

            # Rank by confidence, dedupe
            seen = set()
            ranked = sorted(
                rel_candidates,
                key=lambda x: conf_rank(x.get("confidence", "")),
                reverse=True,
            )

            context_lines = []
            for e in ranked:
                key = (
                    e.get("source_character", ""),
                    e.get("relation", ""),
                    e.get("target_character", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                s, r, t = key
                c = (e.get("confidence") or "medium").lower()
                context_lines.append(f"- {s} {r} {t} (confidence: {c})")
                if len(context_lines) >= args.top_k:
                    break

            if not context_lines:
                context = "- No explicit relation context found. Use neutral default tone for this sentence type."
            else:
                context = "\\n".join(context_lines)

            output_rows.append(
                {
                    "sample_id": sample_id,
                    "sentence_type": sentence_type,
                    "source_text": source_text,
                    "detected_entities": "|".join(ents),
                    "relation_context": context,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "sentence_type",
            "source_text",
            "detected_entities",
            "relation_context",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
