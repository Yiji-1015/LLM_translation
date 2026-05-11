#!/usr/bin/env python3
"""
Run condition D translation:
D = glossary + worldview_context + sentence_type + relation_context

Inputs:
- data/samples.csv
- data/relation_kg/sample_relation_context.csv
- data/glossary.csv
- ../worldview_context.txt (preferred) or data/style_guide.md (fallback)
- prompts/D_glossary_rules_type_relation.txt

Output:
- outputs/run_YYYY-MM-DD/D_outputs.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--base-dir",
        default=str(default_base_dir),
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Output folder date suffix",
    )
    return parser.parse_args()


def load_glossary_text(path: Path) -> str:
    lines = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            src = (row.get("source_term") or "").strip()
            tgt = (row.get("target_term") or "").strip()
            note = (row.get("note") or "").strip()
            if not src or not tgt:
                continue
            if note:
                lines.append(f"- {src} => {tgt} ({note})")
            else:
                lines.append(f"- {src} => {tgt}")
    return "\n".join(lines)


def build_prompt(
    template: str,
    source_text: str,
    sentence_type: str,
    relation_context: str,
    glossary: str,
    worldview_context: str,
) -> str:
    prompt = template
    mapping = {
        "{SRC_LANG}": "English",
        "{TGT_LANG}": "Korean",
        "{SOURCE_TEXT}": source_text,
        "{SENTENCE_TYPE}": sentence_type,
        "{RELATION_CONTEXT}": relation_context,
        "{GLOSSARY}": glossary,
        "{WORLDVIEW_CONTEXT}": worldview_context,
        "{STYLE_GUIDE_AND_RULES}": worldview_context,
    }
    for k, v in mapping.items():
        prompt = prompt.replace(k, v)
    return prompt


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    samples_csv = base / "data" / "samples.csv"
    context_csv = base / "data" / "relation_kg" / "sample_relation_context.csv"
    glossary_csv = base / "data" / "glossary.csv"
    style_md = base / "data" / "style_guide.md"
    prompt_file = base / "prompts" / "D_glossary_rules_type_relation.txt"

    out_dir = base / "outputs" / f"run_{args.run_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "D_outputs.csv"

    api_key = os.getenv("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is set")

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("openai package not found. Install with: pip install openai") from e
        client = OpenAI(api_key=api_key)

    template = prompt_file.read_text(encoding="utf-8")
    worldview_candidates = [
        base.parent / "worldview_context.txt",
        base / "worldview_context.txt",
    ]
    worldview_context = ""
    for p in worldview_candidates:
        if p.exists():
            worldview_context = p.read_text(encoding="utf-8").strip()
            break
    if not worldview_context and style_md.exists():
        worldview_context = style_md.read_text(encoding="utf-8").strip()
    if not worldview_context:
        worldview_context = (
            "Destiny-like sci-fantasy universe. "
            "Tone: mysterious, heroic, high-stakes, mythic-technical blend. "
            "UI text should be concise; gameplay mechanics must remain precise."
        )
    glossary = load_glossary_text(glossary_csv)

    contexts = {}
    with context_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            contexts[(row.get("sample_id") or "").strip()] = row

    outputs = []

    with samples_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sample_id = (row.get("sample_id") or "").strip()
            source_text = (row.get("source_text") or "").strip()
            sentence_type = (row.get("sentence_type") or "").strip()
            relation_context = contexts.get(sample_id, {}).get(
                "relation_context",
                "- No explicit relation context found. Use neutral default tone for this sentence type.",
            )

            prompt = build_prompt(
                template=template,
                source_text=source_text,
                sentence_type=sentence_type,
                relation_context=relation_context,
                glossary=glossary,
                worldview_context=worldview_context,
            )

            if args.dry_run:
                translation = "[DRY_RUN]"
            else:
                resp = client.responses.create(
                    model=args.model,
                    input=[
                        {
                            "role": "system",
                            "content": "You are a careful game localization assistant. Output translation only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                translation = re.sub(r"\s+", " ", (resp.output_text or "").strip())
                time.sleep(0.15)

            outputs.append(
                {
                    "sample_id": sample_id,
                    "source_text": source_text,
                    "translation": translation,
                    "sentence_type": sentence_type,
                    "relation_context": relation_context,
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "source_text", "translation", "sentence_type", "relation_context"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(outputs)

    print(f"Wrote {len(outputs)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
