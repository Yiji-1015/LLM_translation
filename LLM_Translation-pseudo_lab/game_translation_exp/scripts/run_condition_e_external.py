#!/usr/bin/env python3
"""
Run isolated external-KG condition E.

E = glossary + worldview_context + sentence_type + relation_context_external

Isolation:
- reads external context file only
- writes to outputs/run_YYYY-MM-DD_external/E_outputs_external.csv
- does not modify existing A/B/C/D outputs
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
    p = argparse.ArgumentParser()
    default_base_dir = Path(__file__).resolve().parents[1]
    p.add_argument("--base-dir", default=str(default_base_dir))
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-output-tokens", type=int, default=220)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-date", default=datetime.now().strftime("%Y-%m-%d"))
    return p.parse_args()


def load_glossary_text(path: Path) -> str:
    lines = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s = (row.get("source_term") or "").strip()
            t = (row.get("target_term") or "").strip()
            n = (row.get("note") or "").strip()
            if not s or not t:
                continue
            lines.append(f"- {s} => {t}" + (f" ({n})" if n else ""))
    return "\n".join(lines)


def build_prompt(
    template: str,
    src_text: str,
    stype: str,
    rel_context: str,
    glossary: str,
    worldview_context: str,
) -> str:
    repl = {
        "{SRC_LANG}": "English",
        "{TGT_LANG}": "Korean",
        "{SOURCE_TEXT}": src_text,
        "{SENTENCE_TYPE}": stype,
        "{RELATION_CONTEXT}": rel_context,
        "{GLOSSARY}": glossary,
        "{WORLDVIEW_CONTEXT}": worldview_context,
        "{STYLE_GUIDE_AND_RULES}": worldview_context,
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    samples_csv = base / "data" / "samples.csv"
    context_csv = base / "data" / "relation_kg_external" / "sample_relation_context_external.csv"
    glossary_csv = base / "data" / "glossary.csv"
    style_md = base / "data" / "style_guide.md"
    prompt_txt = base / "prompts" / "D_glossary_rules_type_relation.txt"

    run_dir = base / "outputs" / f"run_{args.run_date}_external"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "E_outputs_external.csv"

    key = os.getenv("OPENAI_API_KEY")
    if not args.dry_run and not key:
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run")

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("openai package not found. Install with: pip install openai") from e
        client = OpenAI(api_key=key)

    template = prompt_txt.read_text(encoding="utf-8")
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
            sid = (row.get("sample_id") or "").strip()
            src_text = (row.get("source_text") or "").strip()
            stype = (row.get("sentence_type") or "").strip()
            rel_context = contexts.get(sid, {}).get(
                "relation_context",
                "- No external relation context found. Use default sentence-type style.",
            )
            prompt = build_prompt(template, src_text, stype, rel_context, glossary, worldview_context)

            if args.dry_run:
                trans = "[DRY_RUN]"
            else:
                resp = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": "You are a careful game localization assistant. Output translation only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                trans = re.sub(r"\s+", " ", (resp.output_text or "").strip())
                time.sleep(0.15)

            outputs.append(
                {
                    "sample_id": sid,
                    "source_text": src_text,
                    "translation": trans,
                    "sentence_type": stype,
                    "relation_context_external": rel_context,
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "source_text", "translation", "sentence_type", "relation_context_external"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(outputs)

    print(f"Wrote {len(outputs)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
