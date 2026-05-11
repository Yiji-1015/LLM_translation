from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Tuple

import pandas as pd
from openai import OpenAI

from data import CONTEXT_PROMPT_LIMIT, DEFAULT_PARALLEL_WORKERS


def render_prompt(prompt_text: str, row: pd.Series) -> str:
    return prompt_text.format(
        tag=row.get("tag", ""),
        context=str(row.get("context", ""))[:CONTEXT_PROMPT_LIMIT],
        source=row.get("source", ""),
    )


def translate_one(client: OpenAI, model: str, prompt_text: str, row: pd.Series) -> str:
    response = client.responses.create(
        model=model,
        input=render_prompt(prompt_text, row),
    )
    return response.output_text.strip()


def translate_rows_parallel(
    client: OpenAI,
    model: str,
    prompt_text: str,
    rows: pd.DataFrame,
    max_workers: int = DEFAULT_PARALLEL_WORKERS,
) -> Iterator[Tuple[pd.Series, str]]:
    """Translate rows concurrently. Yields (row, translation_or_error) as each call completes.

    Errors are returned as strings prefixed with "ERROR: " so the caller can persist them
    in the same shape as successful results.
    """
    row_list = [row for _, row in rows.iterrows()]
    if not row_list:
        return

    workers = max(1, min(max_workers, len(row_list)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(translate_one, client, model, prompt_text, row): row
            for row in row_list
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = f"ERROR: {exc}"
            yield row, result
