"""
OpenAI API 호출 + 병렬 번역.

핵심 아이디어:
- OpenAI Python SDK는 동기 API라 한 줄씩 호출하면 row 50개 = 50번의 대기.
- ThreadPoolExecutor로 동시에 N개를 띄우면, 네트워크 대기 시간이 겹쳐서 전체 시간이 ~N배 단축.
- 다만 너무 높이면 OpenAI rate limit(분당 요청수)에 걸리므로 기본 5로 둠.

왜 async가 아니라 thread?
- Streamlit과 async 이벤트 루프를 같이 굴리는 게 까다로움.
- OpenAI SDK 동기 메서드는 그냥 blocking I/O라 thread로도 충분히 병렬화됨.
- (CPU 작업이라면 GIL 때문에 thread가 의미 없겠지만, 여기는 네트워크 대기가 대부분)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Tuple

import pandas as pd
from openai import OpenAI

from data import CONTEXT_PROMPT_LIMIT, DEFAULT_PARALLEL_WORKERS


def render_prompt(prompt_text: str, row: pd.Series) -> str:
    """프롬프트 템플릿의 자리표시자를 row 값으로 채움.
    context는 너무 길면 토큰 비용이 커지므로 CONTEXT_PROMPT_LIMIT 글자로 자름."""
    return prompt_text.format(
        tag=row.get("tag", ""),
        context=str(row.get("context", ""))[:CONTEXT_PROMPT_LIMIT],
        source=row.get("source", ""),
    )


def translate_one(client: OpenAI, model: str, prompt_text: str, row: pd.Series) -> str:
    """한 row 번역. OpenAI의 Responses API를 사용.
    response.output_text는 최종 텍스트 한 덩어리로 모아서 돌려주는 편의 프로퍼티."""
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
    """row들을 병렬로 번역. 끝나는 순서대로 (row, 결과)를 yield.

    동작:
    1. ThreadPoolExecutor에 모든 row를 submit → future 객체가 즉시 반환됨
    2. {future: row} 매핑을 만들어 어떤 future가 어떤 row였는지 추적
    3. as_completed는 끝난 future부터 yield → "빨리 끝난 거 먼저 보여주기"가 가능

    실패한 row는 "ERROR: ..." 문자열로 결과를 채움.
    이렇게 하면 호출부가 try/except 분기를 따로 할 필요 없이 동일한 흐름으로 DB에 저장 가능.
    나중에 실패한 row만 다시 돌리고 싶다면 llm_translation이 "ERROR:"로 시작하는지로 필터링.
    """
    row_list = [row for _, row in rows.iterrows()]
    if not row_list:
        return

    # row 수가 worker 수보다 적으면 worker를 그만큼만 만들어 thread 낭비 방지.
    workers = max(1, min(max_workers, len(row_list)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # submit이 끝나는 순간 future 객체가 반환되고, thread pool이 백그라운드에서 실행 시작.
        futures = {
            executor.submit(translate_one, client, model, prompt_text, row): row
            for row in row_list
        }
        # as_completed는 "끝난 future"부터 하나씩 돌려주는 iterator.
        # 끝난 순서대로 처리하면 progress bar가 빠르게 차오르는 듯한 UX가 됨.
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()  # 여기서 thread 내부 예외가 다시 raise됨
            except Exception as exc:
                result = f"ERROR: {exc}"
            yield row, result
