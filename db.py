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
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except StreamlitSecretNotFoundError:
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
        st.stop()
    if not url or not key:
        st.error("Streamlit secrets에 SUPABASE_URL과 SUPABASE_KEY가 필요합니다.")
        st.stop()
    return create_client(url, key)


def ensure_default_prompt(supabase):
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


def upload_dataset(supabase, dataset_name: str, description: str, df: pd.DataFrame):
    dataset_id = str(uuid.uuid4())
    dataset = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name.strip(),
        "description": description.strip(),
        "uploaded_at": utc_now_iso(),
    }
    supabase.table("datasets").insert(dataset).execute()

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

    has_imported_run = df["llm_translation"].str.strip().any()
    if has_imported_run:
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
                "model": "imported_csv",
                "created_at": utc_now_iso(),
            }
        ).execute()

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


def list_datasets(supabase):
    return (
        supabase.table("datasets")
        .select("*")
        .order("uploaded_at", desc=True)
        .execute()
        .data
    )


def load_rows(supabase, dataset_id: str) -> pd.DataFrame:
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
    return (
        supabase.table("translation_runs")
        .select("*, prompt_versions(name, prompt_text)")
        .eq("dataset_id", dataset_id)
        .order("created_at", desc=True)
        .execute()
        .data
    )


def count_runs_for_prompt(supabase, prompt_version_id: str) -> int:
    rows = (
        supabase.table("translation_runs")
        .select("run_id")
        .eq("prompt_version_id", prompt_version_id)
        .execute()
        .data
    )
    return len(rows)


def delete_prompt_version(supabase, prompt_version_id: str):
    supabase.table("prompt_versions").delete().eq("prompt_version_id", prompt_version_id).execute()


def delete_translation_run(supabase, run_id: str):
    supabase.table("translation_runs").delete().eq("run_id", run_id).execute()


def load_translations(supabase, run_id: str) -> pd.DataFrame:
    rows = supabase.table("translations").select("*").eq("run_id", run_id).execute().data
    return pd.DataFrame(rows)


def load_annotations(supabase, run_id: str) -> pd.DataFrame:
    rows = supabase.table("annotations").select("*").eq("run_id", run_id).execute().data
    return pd.DataFrame(rows)


def save_annotation(supabase, row_id: str, run_id: str, error_type: str, memo: str, reviewer: str):
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
    supabase.table("translations").insert(
        {
            "translation_id": str(uuid.uuid4()),
            "row_id": row_id,
            "run_id": run_id,
            "llm_translation": llm_translation,
        }
    ).execute()


def create_translation_run(supabase, dataset_id: str, prompt_version_id: str, model: str) -> str:
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


def build_review_frame(supabase, dataset_id: str, rows_df: pd.DataFrame, runs: list) -> pd.DataFrame:
    frames = []
    row_base = rows_df.rename(columns={"row_id": "id"}).copy()
    for run in runs:
        run_id = run["run_id"]
        annotations_df = load_annotations(supabase, run_id)
        if annotations_df.empty:
            continue
        translations_df = load_translations(supabase, run_id)
        run_df = build_working_frame(row_base.rename(columns={"id": "row_id"}), translations_df, annotations_df, run)
        run_df = run_df[run_df["error_type"].fillna("").astype(str).str.strip() != ""].copy()
        run_df["run_id"] = run_id
        run_df["run_created_at"] = run.get("created_at", "")
        frames.append(run_df)

    if not frames:
        return pd.DataFrame(columns=TABLE_VIEW_COLUMNS + ["run_id", "run_created_at"])

    review_df = pd.concat(frames, ignore_index=True)
    review_df = review_df[review_df["error_type"] != "No Error"].copy()
    return review_df
