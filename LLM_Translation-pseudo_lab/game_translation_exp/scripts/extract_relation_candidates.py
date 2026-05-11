#!/usr/bin/env python3
"""
Extract relation candidates from line-delimited JSON dataset.

Input default:
  /Users/yiji/Desktop/work/pseudo_lab/destiny2_translation.json

Outputs:
  - data/relation_kg/relation_candidates.csv
  - data/relation_kg/relation_edges_auto.csv

This is a lightweight rule-based extractor for immediate experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


RELATION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "enemy_of": [
        re.compile(r"\benemy of\b", re.I),
        re.compile(r"\bagainst\b", re.I),
        re.compile(r"\bfought\b", re.I),
        re.compile(r"\bwar with\b", re.I),
        re.compile(r"\bslain by\b", re.I),
    ],
    "ally_of": [
        re.compile(r"\bally of\b", re.I),
        re.compile(r"\ballied with\b", re.I),
        re.compile(r"\balongside\b", re.I),
        re.compile(r"\btogether\b", re.I),
    ],
    "mentor_of": [
        re.compile(r"\bmentor\b", re.I),
        re.compile(r"\bteacher\b", re.I),
        re.compile(r"\bdisciple\b", re.I),
        re.compile(r"\bapprentice\b", re.I),
    ],
    "commands": [
        re.compile(r"\bcommander\b", re.I),
        re.compile(r"\bcommands?\b", re.I),
        re.compile(r"\bemperor\b", re.I),
        re.compile(r"\bqueen\b", re.I),
        re.compile(r"\bleader\b", re.I),
    ],
    "trusts": [
        re.compile(r"\btrusts?\b", re.I),
        re.compile(r"\bfaith in\b", re.I),
    ],
    "distrusts": [
        re.compile(r"\bdistrusts?\b", re.I),
        re.compile(r"\bdoubts?\b", re.I),
        re.compile(r"\bsuspects?\b", re.I),
    ],
}

ENTITY_FALLBACK_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Z][a-zA-Z0-9'\\-]+(?:\\s+[A-Z][a-zA-Z0-9'\\-]+){0,2})(?![A-Za-z0-9_])"
)
SPEAKER_ATTRIBUTION_PATTERN = re.compile(r"[—-]\\s*([A-Z][A-Za-z0-9'\\-]+(?:\\s+[A-Z][A-Za-z0-9'\\-]+){0,2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--base-dir",
        default=str(default_base_dir),
        help="Project base directory (game_translation_exp).",
    )
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Path to line-delimited JSON data",
    )
    parser.add_argument(
        "--seed-csv",
        default=None,
        help="Path to entity seed CSV",
    )
    parser.add_argument(
        "--out-candidates",
        default=None,
        help="Output candidate rows",
    )
    parser.add_argument(
        "--out-edges-auto",
        default=None,
        help="Output deduplicated auto edges",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for quick test. 0 means no cap",
    )
    parser.add_argument(
        "--allow-cooccurrence",
        action="store_true",
        help="If set, emit low-confidence co_occurs_with edges when no explicit relation cue exists.",
    )
    return parser.parse_args()


def load_seed_entities(seed_csv: Path) -> Dict[str, List[str]]:
    aliases_by_entity: Dict[str, List[str]] = {}
    with seed_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity = (row.get("entity") or "").strip()
            aliases = (row.get("aliases") or "").strip()
            if not entity:
                continue
            alias_list = [a.strip() for a in aliases.split("|") if a.strip()]
            if entity not in alias_list:
                alias_list.append(entity)
            aliases_by_entity[entity] = sorted(set(alias_list), key=lambda x: (-len(x), x))
    return aliases_by_entity


def build_alias_regex(aliases_by_entity: Dict[str, List[str]]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    for entity, aliases in aliases_by_entity.items():
        escaped = [re.escape(a) for a in aliases]
        # Word boundary-like matching with non-letter guards
        p = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(escaped) + r")(?![A-Za-z0-9_])")
        patterns.append((entity, p))
    return patterns


def find_entities(text: str, alias_patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    found = []
    for entity, pat in alias_patterns:
        if pat.search(text):
            found.append(entity)
    return found


def fallback_entities(text: str) -> List[str]:
    out = []
    for m in ENTITY_FALLBACK_PATTERN.finditer(text):
        cand = (m.group(1) or "").strip()
        if len(cand) <= 2:
            continue
        # Skip generic sentence starters that often create false positives
        if cand in {"The", "This", "That", "When", "If", "And", "But", "For", "With"}:
            continue
        out.append(cand)
    # preserve order with dedupe
    seen = set()
    uniq = []
    for e in out:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq


def match_relations(text: str) -> List[str]:
    rels = []
    for rel, pats in RELATION_PATTERNS.items():
        if any(p.search(text) for p in pats):
            rels.append(rel)
    return rels


def confidence_from_evidence(entity_count: int, relation_count: int, src_type: str) -> str:
    score = 0.0
    if entity_count >= 2:
        score += 0.5
    if relation_count >= 1:
        score += 0.3
    if src_type in {"lore_subtitle", "lore_description", "description"}:
        score += 0.1
    if relation_count > 1:
        score += 0.1
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def adjust_confidence(base_conf: str, relation: str) -> str:
    if relation == "co_occurs_with":
        return "low"
    return base_conf


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    input_jsonl = Path(args.input_jsonl).expanduser() if args.input_jsonl else base_dir.parent / "destiny2_translation.json"
    seed_csv = Path(args.seed_csv).expanduser() if args.seed_csv else base_dir / "data" / "relation_kg" / "character_seeds.csv"
    out_candidates = Path(args.out_candidates).expanduser() if args.out_candidates else base_dir / "data" / "relation_kg" / "relation_candidates.csv"
    out_edges_auto = Path(args.out_edges_auto).expanduser() if args.out_edges_auto else base_dir / "data" / "relation_kg" / "relation_edges_auto.csv"

    aliases_by_entity = load_seed_entities(seed_csv)
    alias_patterns = build_alias_regex(aliases_by_entity)

    candidate_rows = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if args.max_rows and i > args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = (row.get("en") or "").strip()
            if not text:
                continue

            src_type = str(row.get("type") or "")
            src_source = str(row.get("source") or "")

            # Keep likely narrative/mechanic texts only
            if src_type not in {"description", "lore_description", "lore_subtitle"}:
                continue

            entities = find_entities(text, alias_patterns)
            if len(entities) < 2:
                # Fallback: lightweight NER-like capitalized phrase detection
                entities = fallback_entities(text)
            if len(entities) < 2:
                continue

            rels = match_relations(text)
            if not rels:
                # Fallback relation for quote attribution/speaker lines
                speaker_match = SPEAKER_ATTRIBUTION_PATTERN.search(text)
                if speaker_match:
                    speaker = speaker_match.group(1).strip()
                    rels = ["speaks_to"]
                    if speaker not in entities:
                        entities = [speaker] + entities
                elif args.allow_cooccurrence:
                    rels = ["co_occurs_with"]
                else:
                    continue

            # Pair entities (directed heuristic: mention order)
            mention_positions = []
            for e in entities:
                matched = False
                for name, p in alias_patterns:
                    if name == e:
                        m = p.search(text)
                        mention_positions.append((m.start() if m else 10**9, e))
                        matched = True
                        break
                if not matched:
                    m = re.search(re.escape(e), text)
                    mention_positions.append((m.start() if m else 10**9, e))
            mention_positions.sort()
            ordered_entities = [e for _, e in mention_positions]

            for rel in rels:
                for s_idx in range(len(ordered_entities) - 1):
                    src = ordered_entities[s_idx]
                    tgt = ordered_entities[s_idx + 1]
                    if src == tgt:
                        continue
                    conf = confidence_from_evidence(len(entities), len(rels), src_type)
                    conf = adjust_confidence(conf, rel)
                    candidate_rows.append(
                        {
                            "source_character": src,
                            "relation": rel,
                            "target_character": tgt,
                            "evidence": text,
                            "confidence": conf,
                            "row_id": row.get("id"),
                            "text_type": src_type,
                            "source_def": src_source,
                        }
                    )

    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    with out_candidates.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_character",
            "relation",
            "target_character",
            "evidence",
            "confidence",
            "row_id",
            "text_type",
            "source_def",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(candidate_rows)

    # Deduplicate to auto edges with best confidence and one evidence example
    rank = {"high": 3, "medium": 2, "low": 1}
    best: Dict[Tuple[str, str, str], dict] = {}
    for r in candidate_rows:
        k = (r["source_character"], r["relation"], r["target_character"])
        prev = best.get(k)
        if prev is None or rank[r["confidence"]] > rank[prev["confidence"]]:
            best[k] = r

    edge_rows = []
    for (src, rel, tgt), r in sorted(best.items()):
        edge_rows.append(
            {
                "source_character": src,
                "relation": rel,
                "target_character": tgt,
                "evidence": r["evidence"],
                "confidence": r["confidence"],
            }
        )

    with out_edges_auto.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_character",
            "relation",
            "target_character",
            "evidence",
            "confidence",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(edge_rows)

    print(f"Candidates: {len(candidate_rows)} -> {out_candidates}")
    print(f"Auto edges: {len(edge_rows)} -> {out_edges_auto}")


if __name__ == "__main__":
    main()
