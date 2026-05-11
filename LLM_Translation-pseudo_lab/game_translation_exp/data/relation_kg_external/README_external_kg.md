# External KG Pack (Isolated)

This folder is isolated from existing A/B/C/D experiments.

Purpose:
- Use externally researched relationship facts (user-provided deep research text)
- Run a separate experiment condition without modifying existing outputs

Isolation rules:
- Do not overwrite files in `data/relation_kg/`
- Do not overwrite files in `outputs/run_YYYY-MM-DD/`
- Use dedicated output folder: `outputs/run_YYYY-MM-DD_external/`
- Use dedicated eval file: `eval/eval_sheet_prefilled_E_external_YYYY-MM-DD.csv`

Main files:
- `relation_edges_external_v1.csv`: curated relation edges from external report
- `sample_relation_context_external.csv`: per-sample context generated from external edges
