"""
Supabase(Postgres) 접근 계층.

데이터 모델:
    datasets ── rows (1:N)
            ├── translation_runs ── translations  (각 run당 row별 LLM 번역)
            │                  └── annotations    (각 run당 row별 사람 라벨)
            └── prompt_versions      (run에서 참조)

설계 원칙:
- 한 번에 한 쿼리. 트랜잭션은 쓰지 않음 (Supabase Python SDK가 깔끔히 지원하지 않음).
- 실패 시 부분 저장 가능성은 받아들이고, upsert로 멱등성을 확보.
- on_conflict로 PK 충돌 시 update로 동작하게 만들어 재실행 안전.
"""

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from supabase import create_client

from data import (
    DEFAULT_PROMPT,
    TABLE_VIEW_COLUMNS,
    build_working_frame,
    natural_sort_key,
    utc_now_iso,
)


@st.cache_resource
def get_supabase_client():
    """Supabase 클라이언트를 한 번만 만들고 세션 전체에서 재사용.

    @st.cache_resource는 thread-safe 싱글톤처럼 동작 — DB/HTTP 클라이언트처럼
    여러 번 만들 필요가 없는 자원에 적합. 일반 데이터 캐시에는 @st.cache_data 사용.
    """
    try:
        # st.secrets는 .streamlit/secrets.toml (로컬) 또는 Streamlit Cloud의 Secrets 패널에서 옴.
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except StreamlitSecretNotFoundError:
        # secrets 파일 자체가 없으면 친절한 안내를 띄우고 앱 중단.
        st.error("Streamlit secrets 파일을 찾을 수 없습니다.")
        st.info(
            "프로젝트 안에 `.streamlit/secrets.toml` 파일을 만들고 "
            "`SUPABASE_URL`, `SUPABASE_KEY`를 추가해 주세요."
        )
        st.code(
            'SUPABASE_URL = "https://your-project.supabase.co"\n'
            'SUPABASE_KEY = "your-supabase-key"',
            language="toml",
        )
        st.stop()  # 이후 코드 실행을 막음.
    if not url or not key:
        st.error("Streamlit secrets에 SUPABASE_URL과 SUPABASE_KEY가 필요합니다.")
        st.stop()
    return create_client(url, key)


# ---------- 프롬프트 ----------

def ensure_default_prompt(supabase):
    """is_default=True인 프롬프트가 없으면 하나 만들어 둠.
    앱이 처음 켜졌을 때 "선택할 프롬프트가 하나도 없는" 상태를 피하기 위함."""
    existing = (
        supabase.table("prompt_versions")
        .select("*")
        .eq("is_default", True)
        .limit(1)
        .execute()
        .data
    )
    if existing:
        return existing[0]

    prompt = {
        "prompt_version_id": str(uuid.uuid4()),
        "name": "Default game localization prompt",
        "prompt_text": DEFAULT_PROMPT,
        "is_default": True,
        "created_at": utc_now_iso(),
    }
    supabase.table("prompt_versions").insert(prompt).execute()
    return prompt


def get_prompt_versions(supabase):
    """최신순으로 모든 프롬프트 버전을 가져옴.
    혹시라도 비어있으면 기본 프롬프트를 즉시 만들어 빈 list가 반환되지 않도록 함."""
    rows = (
        supabase.table("prompt_versions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
        .data
    )
    if not rows:
        rows = [ensure_default_prompt(supabase)]
    return rows


def create_prompt_version(supabase, name: str, prompt_text: str, is_default: bool = False):
    row = {
        "prompt_version_id": str(uuid.uuid4()),
        "name": name.strip(),
        "prompt_text": prompt_text,
        "is_default": is_default,
        "created_at": utc_now_iso(),
    }
    supabase.table("prompt_versions").insert(row).execute()
    return row


# ---------- 데이터셋 업로드 ----------

def upload_dataset(supabase, dataset_name: str, description: str, df: pd.DataFrame):
    """업로드된 CSV를 datasets/rows에 저장.

    추가 동작: CSV에 llm_translation 값이 들어 있으면 "결과 CSV 재업로드"로 보고
    translation_run / translations / annotations까지 같이 복원함.
    이렇게 하면 팀원끼리 결과 CSV만 주고받아도 작업을 이어갈 수 있음.
    """
    dataset_id = str(uuid.uuid4())
    dataset = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name.strip(),
        "description": description.strip(),
        "uploaded_at": utc_now_iso(),
    }
    supabase.table("datasets").insert(dataset).execute()

    # row를 한 번에 upsert. on_conflict로 (dataset_id, row_id) 충돌 시 update 동작.
    # → 같은 dataset에 같은 id의 row가 다시 들어와도 안전.
    row_payload = []
    for _, row in df.iterrows():
        row_payload.append(
            {
                "dataset_id": dataset_id,
                "row_id": row["id"],
                "source": row["source"],
                "human_translation": row["human_translation"],
                "tag": row["tag"],
                "context": row["context"],
                "speaker": row["speaker"],
                "listener": row["listener"],
                "notes": row["notes"],
                "hash": row["hash"],
            }
        )
    if row_payload:
        supabase.table("rows").upsert(row_payload, on_conflict="dataset_id,row_id").execute()

    # llm_translation 컬럼에 값이 하나라도 있으면 → "결과 CSV 재업로드" 케이스.
    has_imported_run = df["llm_translation"].str.strip().any()
    if has_imported_run:
        # 원본 프롬프트를 알 수 없으므로 placeholder 프롬프트를 만들어 추적성만 유지.
        prompt = create_prompt_version(
            supabase,
            f"Imported CSV {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "Imported from a result CSV. Original prompt text is not available.",
        )
        run_id = str(uuid.uuid4())
        supabase.table("translation_runs").insert(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "prompt_version_id": prompt["prompt_version_id"],
                "model": "imported_csv",  # 실제 모델명을 알 수 없으니 sentinel 값.
                "created_at": utc_now_iso(),
            }
        ).execute()

        # 비어있지 않은 llm_translation만 translations에 적재.
        translations = [
            {
                "translation_id": str(uuid.uuid4()),
                "row_id": row["id"],
                "run_id": run_id,
                "llm_translation": row["llm_translation"],
            }
            for _, row in df.iterrows()
            if str(row["llm_translation"]).strip()
        ]
        if translations:
            supabase.table("translations").insert(translations).execute()

        # error_type/memo/reviewer 중 하나라도 채워진 row만 annotation으로 복원.
        annotations = []
        for _, row in df.iterrows():
            if any(str(row[col]).strip() for col in ["error_type", "memo", "reviewer"]):
                annotations.append(
                    {
                        "annotation_id": str(uuid.uuid4()),
                        "row_id": row["id"],
                        "run_id": run_id,
                        "error_type": row["error_type"] or "No Error",
                        "memo": row["memo"],
                        "reviewer": row["reviewer"],
                        "updated_at": utc_now_iso(),
                    }
                )
        if annotations:
            supabase.table("annotations").upsert(annotations, on_conflict="row_id,run_id").execute()

    return dataset_id


# ---------- 조회 ----------

def list_datasets(supabase):
    return (
        supabase.table("datasets")
        .select("*")
        .order("uploaded_at", desc=True)
        .execute()
        .data
    )


def load_rows(supabase, dataset_id: str) -> pd.DataFrame:
    """rows를 가져온 뒤 row_id 기준 자연 정렬.
    Postgres의 order by는 사전식이라 row_2 < row_10이 안 되므로 파이썬에서 재정렬."""
    rows = (
        supabase.table("rows")
        .select("*")
        .eq("dataset_id", dataset_id)
        .order("row_id")
        .execute()
        .data
    )
    rows = sorted(rows, key=lambda row: natural_sort_key(row["row_id"]))
    return pd.DataFrame(rows)


def load_runs(supabase, dataset_id: str):
    """translation_runs에 join된 prompt_versions(name, prompt_text)를 같이 가져옴.
    select 문법: `*, prompt_versions(name, prompt_text)` → 외래키로 연결된
    레코드를 nested dict로 포함해서 반환 (PostgREST 기능)."""
    return (
        supabase.table("translation_runs")
        .select("*, prompt_versions(name, prompt_text)")
        .eq("dataset_id", dataset_id)
        .order("created_at", desc=True)
        .execute()
        .data
    )


def count_runs_for_prompt(supabase, prompt_version_id: str) -> int:
    """이 프롬프트로 만든 run 개수.
    프롬프트 삭제를 막을지 판단하는 데 사용 (추적성 위해 사용 중이면 잠금)."""
    rows = (
        supabase.table("translation_runs")
        .select("run_id")
        .eq("prompt_version_id", prompt_version_id)
        .execute()
        .data
    )
    return len(rows)


# ---------- 삭제 ----------

def delete_prompt_version(supabase, prompt_version_id: str):
    """프롬프트 삭제. 호출부에서 default/사용중 여부를 먼저 검증함."""
    supabase.table("prompt_versions").delete().eq("prompt_version_id", prompt_version_id).execute()


def delete_translation_run(supabase, run_id: str):
    """번역 run 삭제.
    SQL 스키마에 ON DELETE CASCADE가 걸려 있어 translations/annotations도 같이 사라짐.
    실험성 산출물이라 의도적으로 가볍게 지우도록 설계 (UI에 확인 절차 없음)."""
    supabase.table("translation_runs").delete().eq("run_id", run_id).execute()


# ---------- translation / annotation ----------

def load_translations(supabase, run_id: str) -> pd.DataFrame:
    rows = supabase.table("translations").select("*").eq("run_id", run_id).execute().data
    return pd.DataFrame(rows)


def load_annotations(supabase, run_id: str) -> pd.DataFrame:
    rows = supabase.table("annotations").select("*").eq("run_id", run_id).execute().data
    return pd.DataFrame(rows)


def save_annotation(supabase, row_id: str, run_id: str, error_type: str, memo: str, reviewer: str):
    """라벨링 저장. (row_id, run_id) 유니크 제약이 있어 같은 row에 두 번 저장하면
    upsert로 기존 값을 덮어씀. → 사용자가 라벨을 수정하면 그대로 반영됨."""
    payload = {
        "annotation_id": str(uuid.uuid4()),
        "row_id": row_id,
        "run_id": run_id,
        "error_type": error_type,
        "memo": memo,
        "reviewer": reviewer,
        "updated_at": utc_now_iso(),
    }
    supabase.table("annotations").upsert(payload, on_conflict="row_id,run_id").execute()


def insert_translation(supabase, row_id: str, run_id: str, llm_translation: str):
    """병렬 번역 루프에서 결과가 하나 나올 때마다 호출됨.
    insert만 — 같은 (row_id, run_id)에 대해 다시 돌리고 싶다면 새 run을 만드는 것이 원칙."""
    supabase.table("translations").insert(
        {
            "translation_id": str(uuid.uuid4()),
            "row_id": row_id,
            "run_id": run_id,
            "llm_translation": llm_translation,
        }
    ).execute()


def create_translation_run(supabase, dataset_id: str, prompt_version_id: str, model: str) -> str:
    """번역 run 메타데이터를 먼저 만들고 run_id를 반환.
    이후 row별 translation들이 이 run_id를 외래키로 가짐.
    실제 OpenAI 호출 전에 만들어 두는 이유:
    - 일부 row만 성공해도 run으로 묶여 추적 가능
    - run_id를 미리 알아야 progress 중에 insert 가능
    """
    run_id = str(uuid.uuid4())
    supabase.table("translation_runs").insert(
        {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "prompt_version_id": prompt_version_id,
            "model": model,
            "created_at": utc_now_iso(),
        }
    ).execute()
    return run_id


# ---------- 리뷰 ----------

def build_review_frame(supabase, dataset_id: str, rows_df: pd.DataFrame, runs: list) -> pd.DataFrame:
    """라벨 기준 탭의 "라벨링된 오류 모아보기"용 통합 프레임.
    dataset 안의 모든 run을 순회하며, 각 run의 annotation 중 의미 있는 것만 모음.

    필터:
    - error_type이 빈 문자열인 row 제거
    - error_type == 'No Error'인 row 제거 (= 실제 오류로 분류된 것만)
    """
    frames = []
    row_base = rows_df.rename(columns={"row_id": "id"}).copy()
    for run in runs:
        run_id = run["run_id"]
        annotations_df = load_annotations(supabase, run_id)
        if annotations_df.empty:
            continue
        translations_df = load_translations(supabase, run_id)
        # build_working_frame에 넣기 위해 id를 다시 row_id로 되돌림 — 내부에서 다시 rename함.
        run_df = build_working_frame(row_base.rename(columns={"id": "row_id"}), translations_df, annotations_df, run)
        run_df = run_df[run_df["error_type"].fillna("").astype(str).str.strip() != ""].copy()
        # 어느 run에서 나온 라벨인지 추적할 수 있도록 컬럼 추가.
        run_df["run_id"] = run_id
        run_df["run_created_at"] = run.get("created_at", "")
        frames.append(run_df)

    if not frames:
        return pd.DataFrame(columns=TABLE_VIEW_COLUMNS + ["run_id", "run_created_at"])

    review_df = pd.concat(frames, ignore_index=True)
    review_df = review_df[review_df["error_type"] != "No Error"].copy()
    return review_df
