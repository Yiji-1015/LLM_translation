# Relation KG for Game Translation (Draft)

## Goal
Use character/faction relationship context to improve translation tone, register, and consistency.

## Practical Experiment Design

- Baseline: existing A/B/C conditions
- New condition D (optional): `glossary + rules + sentence_type + relation_context`

`relation_context` is retrieved from a lightweight knowledge graph built from existing lore/dialogue text.

## Minimal Pipeline (Now)

1. Prepare entity seeds
- File: `data/relation_kg/character_seeds.csv`
- Add important characters/factions and aliases.

2. Extract relation candidates from existing dataset
- Script: `scripts/extract_relation_candidates.py`
- Input: `/Users/yiji/Desktop/work/pseudo_lab/destiny2_translation.json`
- Output:
  - `data/relation_kg/relation_candidates.csv`
  - `data/relation_kg/relation_edges_auto.csv`

3. Canonical edge confirmation
- Review `relation_candidates.csv` and keep reliable rows.
- Save confirmed rows into `data/relation_kg/relation_edges_confirmed.csv`
- Use `data/relation_kg/relation_review_checklist.md` for quick filtering.
- Include externally researched, citation-backed edges in the same confirmed file.
  with columns:
  - source_character
  - relation
  - target_character
  - evidence
  - confidence

4. Build per-sample relation context
- Script: `scripts/build_relation_context.py`
- Input:
  - `data/samples_tagged_v1.csv`
  - `data/relation_kg/relation_edges_confirmed.csv`
  - `data/relation_kg/character_seeds.csv`
- Output:
  - `data/relation_kg/sample_relation_context.csv`

5. Inject relation context into translation prompt
- For each sample, append `sample_relation_context` block.

## Suggested Relation Types

- `ally_of`
- `enemy_of`
- `mentor_of`
- `commands`
- `belongs_to_faction`
- `trusts`
- `distrusts`
- `speaks_formally_to`
- `speaks_informally_to`

## Suggested D Prompt Block

```text
Relationship Context:
{RELATION_CONTEXT}

Use this context to choose appropriate tone/register and terminology consistency.
- If source is subordinate addressing superior: prefer formal tone.
- If hostile relationship: allow sharper diction while preserving meaning.
- If allies/close peers: avoid overly stiff style unless sentence type is UI.
```

## Caveat
This draft is intentionally lightweight and optimized for immediate experimentation. It is not a complete canonical lore graph.
