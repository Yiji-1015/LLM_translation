#!/usr/bin/env python3
"""
Build per-sample relation context from external curated edges.

This script is isolated from existing relation_kg pipeline.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    default_base_dir = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--base-dir",
        default=str(default_base_dir),
        help="Project base directory (game_translation_exp).",
    )
    p.add_argument(
        "--samples-csv",
        default=None,
    )
    p.add_argument(
        "--edges-csv",
        default=None,
    )
    p.add_argument(
        "--out-csv",
        default=None,
    )
    p.add_argument(
        "--alias-csv",
        default=None,
    )
    p.add_argument("--top-k", type=int, default=4)
    return p.parse_args()


def conf_rank(conf: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get((conf or "").lower(), 0)


def compile_entity_patterns(alias_map: Dict[str, List[str]], entities: List[str]) -> List[Tuple[str, re.Pattern]]:
    out: List[Tuple[str, re.Pattern]] = []
    pairs: List[Tuple[str, str]] = []
    for canonical in entities:
        canonical = canonical.strip()
        if not canonical:
            continue
        aliases = alias_map.get(canonical, [canonical])
        for alias in aliases:
            alias = alias.strip()
            if alias:
                pairs.append((canonical, alias))

    seen = set()
    for canonical, alias in sorted(pairs, key=lambda x: (-len(x[1]), x[1].lower(), x[0].lower())):
        key = (canonical.lower(), alias.lower())
        if key in seen:
            continue
        seen.add(key)
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", re.I)
        out.append((canonical, pat))
    return out


def load_alias_map(alias_csv: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not alias_csv.exists():
        return out
    with alias_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            entity = (row.get("entity") or "").strip()
            aliases = (row.get("aliases") or "").strip()
            if not entity:
                continue
            vals = [x.strip() for x in aliases.split("|") if x.strip()]
            if entity not in vals:
                vals.append(entity)
            out[entity] = sorted(set(vals), key=lambda x: (-len(x), x))
    return out


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    samples_csv = Path(args.samples_csv).expanduser() if args.samples_csv else base_dir / "data" / "samples.csv"
    edges_csv = Path(args.edges_csv).expanduser() if args.edges_csv else base_dir / "data" / "relation_kg_external" / "relation_edges_external_v1.csv"
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else base_dir / "data" / "relation_kg_external" / "sample_relation_context_external.csv"
    alias_csv = Path(args.alias_csv).expanduser() if args.alias_csv else base_dir / "data" / "relation_kg_external" / "entity_aliases_external.csv"

    edges = []
    with edges_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            edges.append(row)

    entities = []
    for e in edges:
        entities.append((e.get("source_character") or "").strip())
        entities.append((e.get("target_character") or "").strip())
    alias_map = load_alias_map(alias_csv)
    canonical_entities = [x for x in entities if x]
    patterns = compile_entity_patterns(alias_map, canonical_entities)

    adjacency: Dict[str, List[dict]] = {}
    for e in edges:
        src = (e.get("source_character") or "").strip()
        tgt = (e.get("target_character") or "").strip()
        adjacency.setdefault(src, []).append(e)
        adjacency.setdefault(tgt, []).append(e)

    out_rows = []
    with samples_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = (row.get("sample_id") or "").strip()
            stype = (row.get("sentence_type") or "").strip()
            src_text = (row.get("source_text") or "").strip()

            detected = []
            for ent, pat in patterns:
                if pat.search(src_text):
                    detected.append(ent)
            # preserve order while deduping
            detected = list(dict.fromkeys(detected))

            rels = []
            for ent in detected:
                rels.extend(adjacency.get(ent, []))

            rels = sorted(rels, key=lambda x: conf_rank(x.get("confidence", "")), reverse=True)

            lines = []
            seen = set()
            for e in rels:
                key = (
                    (e.get("source_character") or "").strip(),
                    (e.get("relation") or "").strip(),
                    (e.get("target_character") or "").strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                s, rel, t = key
                c = (e.get("confidence") or "medium").lower()
                lines.append(f"- {s} {rel} {t} (confidence: {c})")
                if len(lines) >= args.top_k:
                    break

            if not lines:
                context = "- No external relation context found. Use default sentence-type style."
            else:
                context = "\\n".join(lines)

            out_rows.append(
                {
                    "sample_id": sid,
                    "sentence_type": stype,
                    "source_text": src_text,
                    "detected_entities": "|".join(detected),
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
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
