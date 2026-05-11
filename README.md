# LLM Translation Failure Lab

Python + Streamlit + Supabase 기반의 LLM 번역 실패 관찰 및 공동 라벨링 MVP입니다. 게임/콘텐츠 번역 CSV를 업로드하고, LLM 번역 run과 사람 라벨링 annotation을 분리 저장합니다.

## Features

- CSV 업로드만 지원
- dataset, row, prompt version, translation run, annotation 분리 저장
- OpenAI API key는 사이드바 password input으로만 입력하며 `st.session_state`에만 보관
- LLM 번역 run 생성, 최대 50행
- 카드뷰/테이블뷰, tag 필터, 검색
- MQM 스타일 error type, memo, reviewer 저장
- 결과 CSV 다운로드 및 재업로드로 이어서 작업 가능
- UTF-8-SIG CSV 다운로드

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Secrets

로컬에서는 `.streamlit/secrets.toml`을 만들고, Streamlit Cloud에서는 App settings의 Secrets에 같은 값을 넣습니다.

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-anon-or-service-role-key"
```

`OPENAI_API_KEY`는 secrets에 넣지 않아도 됩니다. 앱 사이드바에서 팀원이 직접 입력하며, DB/CSV/secrets/log에 저장하지 않습니다.

## Input CSV

필수 컬럼:

- `source`
- `human_translation`

선택 컬럼:

- `id`
- `tag`
- `context`
- `speaker`
- `listener`
- `notes`

결과 CSV를 다시 업로드할 때 아래 컬럼이 있으면 값을 보존합니다.

- `llm_translation`
- `error_type`
- `memo`
- `reviewer`

기본값:

- `id`가 없으면 `row_0001` 형식으로 생성
- `tag`가 없으면 `Uncategorized`
- 나머지 선택 컬럼은 빈 문자열

업로드는 최대 300행입니다.

## Example JSON Preprocess

앱은 JSON 업로드를 지원하지 않습니다. 예를 들어 `destiny2_translation.json` 같은 newline-delimited JSON은 CSV로 전처리한 뒤 업로드합니다.

```python
import pandas as pd

df = pd.read_json("destiny2_translation.json", lines=True)
out = pd.DataFrame({
    "id": df["id"].astype(str),
    "tag": df.get("type", "Uncategorized"),
    "source": df["en"],
    "human_translation": df["ko"],
    "context": df.get("source", ""),
    "notes": df.get("domain", ""),
})
out.to_csv("destiny2_translation.csv", index=False, encoding="utf-8-sig")
```

## Output CSV Columns

```text
id
tag
source
human_translation
context
speaker
listener
notes
llm_translation
prompt_version_name
model
error_type
memo
reviewer
```

## Supabase SQL Schema

Supabase SQL Editor에서 아래 SQL을 실행해 초기 테이블을 만듭니다.

```sql
create table if not exists datasets (
  dataset_id uuid primary key,
  dataset_name text not null,
  description text default '',
  uploaded_at timestamptz not null default now()
);

create table if not exists rows (
  dataset_id uuid not null references datasets(dataset_id) on delete cascade,
  row_id text not null,
  source text not null,
  human_translation text not null,
  tag text not null default 'Uncategorized',
  context text default '',
  speaker text default '',
  listener text default '',
  notes text default '',
  hash text not null,
  primary key (dataset_id, row_id)
);

create index if not exists idx_rows_dataset_id on rows(dataset_id);
create index if not exists idx_rows_hash on rows(hash);

create table if not exists prompt_versions (
  prompt_version_id uuid primary key,
  name text not null,
  prompt_text text not null,
  is_default boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists translation_runs (
  run_id uuid primary key,
  dataset_id uuid not null references datasets(dataset_id) on delete cascade,
  prompt_version_id uuid not null references prompt_versions(prompt_version_id),
  model text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_translation_runs_dataset_id on translation_runs(dataset_id);

create table if not exists translations (
  translation_id uuid primary key,
  row_id text not null,
  run_id uuid not null references translation_runs(run_id) on delete cascade,
  llm_translation text not null,
  unique (row_id, run_id)
);

create index if not exists idx_translations_run_id on translations(run_id);

create table if not exists annotations (
  annotation_id uuid primary key,
  row_id text not null,
  run_id uuid not null references translation_runs(run_id) on delete cascade,
  error_type text not null default 'No Error',
  memo text default '',
  reviewer text default '',
  updated_at timestamptz not null default now(),
  unique (row_id, run_id)
);

create index if not exists idx_annotations_run_id on annotations(run_id);
```

## Deployment Notes

1. GitHub에 `app.py`, `requirements.txt`, `README.md`를 push합니다.
2. Streamlit Cloud에서 새 app을 만들고 repo를 연결합니다.
3. Main file path를 `app.py`로 설정합니다.
4. Secrets에 `SUPABASE_URL`, `SUPABASE_KEY`를 입력합니다.
5. Supabase SQL Editor에서 schema를 먼저 생성합니다.
6. 앱에서 CSV를 업로드하고 Work 탭에서 run/labeling을 진행합니다.

로그인은 만들지 않았습니다. 모든 사용자는 신뢰된 팀원이라고 가정합니다.
