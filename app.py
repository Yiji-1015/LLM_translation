import hashlib
import io
import re
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from openai import OpenAI
from supabase import create_client


APP_TITLE = "LLM 번역 실패 관찰 및 공동 라벨링 툴"
MAX_UPLOAD_ROWS = 300
MAX_RUN_ROWS = 50
DEFAULT_BATCH_SIZE = 10
LONG_SOURCE_WARNING_CHARS = 800
CONTEXT_PROMPT_LIMIT = 1000

OUTPUT_COLUMNS = [
    "id",
    "tag",
    "source",
    "human_translation",
    "context",
    "speaker",
    "listener",
    "notes",
    "llm_translation",
    "prompt_version_name",
    "model",
    "error_type",
    "memo",
    "reviewer",
]

TABLE_VIEW_COLUMNS = [
    "source",
    "human_translation",
    "llm_translation",
    "error_type",
    "memo",
    "reviewer",
    "id",
    "tag",
    "context",
    "speaker",
    "listener",
    "notes",
    "prompt_version_name",
    "model",
]

OPTIONAL_INPUT_COLUMNS = ["id", "tag", "context", "speaker", "listener", "notes"]
PRESERVED_COLUMNS = ["llm_translation", "error_type", "memo", "reviewer"]
ERROR_TYPES = [
    "No Error",
    "Terminology",
    "Accuracy",
    "Style",
    "Persona/Pragmatics",
    "Locale",
    "Design/Markup",
    "Linguistic",
    "Other",
]

LABEL_GUIDE = [
    {
        "error_type": "No Error",
        "meaning": "문제가 없거나, 현재 기준으로는 오류로 보기 어려운 경우",
        "example": "LLM 번역이 사람 번역과 의미/톤/형식 면에서 충분히 동등함",
    },
    {
        "error_type": "Terminology",
        "meaning": "용어집, 고유명사, 반복 용어의 일관성 문제가 있는 경우",
        "example": "Engram을 엔그램/엔그람으로 혼용하거나, 기존 게임 용어와 다르게 번역",
    },
    {
        "error_type": "Accuracy",
        "meaning": "원문의 의미를 누락, 왜곡, 추가하거나 반대로 해석한 경우",
        "example": "random reward를 확정 보상처럼 번역하거나 조건/수량을 빠뜨림",
    },
    {
        "error_type": "Style",
        "meaning": "문체, 톤, 장르감, UI 문구로서의 자연스러움이 어긋난 경우",
        "example": "짧은 UI 문구가 지나치게 문어체이거나 게임 내 톤과 맞지 않음",
    },
    {
        "error_type": "Persona/Pragmatics",
        "meaning": "화자 성격, 관계, 존비어, 발화 의도나 함의가 어긋난 경우",
        "example": "캐릭터가 거칠게 말해야 하는 장면에서 과하게 공손한 표현 사용",
    },
    {
        "error_type": "Locale",
        "meaning": "한국어권 관습, 문화적 표현, 단위/날짜/표기 관례와 맞지 않는 경우",
        "example": "한국어 UI에서 어색한 날짜 표기나 문화권에 맞지 않는 관용 표현",
    },
    {
        "error_type": "Design/Markup",
        "meaning": "플레이스홀더, 변수, 태그, 줄바꿈, UI 제약을 훼손한 경우",
        "example": "{player}, <color>, NL, 숫자, 아이콘 태그를 누락하거나 변경",
    },
    {
        "error_type": "Linguistic",
        "meaning": "문법, 맞춤법, 조사, 어순, 중복 표현 등 언어 품질 문제가 있는 경우",
        "example": "조사 오류, 비문, 부자연스러운 직역, 의미는 맞지만 한국어로 어색한 문장",
    },
    {
        "error_type": "Other",
        "meaning": "위 기준으로 분류하기 어렵거나 여러 문제가 섞여 별도 메모가 필요한 경우",
        "example": "복합 오류, 데이터 자체 문제, 분류 기준을 논의해야 하는 사례",
    },
]

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5.4-mini",
    "gpt-5.4",
    "직접 입력",
]

DEFAULT_PROMPT = """You are a professional English-to-Korean game localization translator.
Translate the source text into natural Korean.
Preserve placeholders, tags, variables, numbers, and line breaks exactly.
Use the given tag and context only as guidance.
Do not add explanations.

Tag: {tag}
Context: {context}
Source: {source}

Return only the Korean translation."""


st.set_page_config(page_title=APP_TITLE, layout="wide")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def hash_row(source: str, human_translation: str) -> str:
    raw = f"{source}\x1f{human_translation}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    missing = {"source", "human_translation"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"CSV 업로드는 최대 {MAX_UPLOAD_ROWS}행까지 가능합니다.")

    if "id" not in df.columns:
        df["id"] = [f"row_{idx + 1:04d}" for idx in range(len(df))]
    if "tag" not in df.columns:
        df["tag"] = "Uncategorized"

    for column in OPTIONAL_INPUT_COLUMNS + PRESERVED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in OPTIONAL_INPUT_COLUMNS + ["source", "human_translation"] + PRESERVED_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    df["id"] = df["id"].where(df["id"].str.strip() != "", [f"row_{idx + 1:04d}" for idx in range(len(df))])
    df["tag"] = df["tag"].where(df["tag"].str.strip() != "", "Uncategorized")
    df["hash"] = df.apply(lambda row: hash_row(row["source"], row["human_translation"]), axis=1)
    return df


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


def natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


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


def build_working_frame(rows_df: pd.DataFrame, translations_df: pd.DataFrame, annotations_df: pd.DataFrame, run):
    if rows_df.empty:
        return rows_df

    df = rows_df.rename(columns={"row_id": "id"}).copy()
    if translations_df.empty:
        df["llm_translation"] = ""
    else:
        tdf = translations_df[["row_id", "llm_translation"]].rename(columns={"row_id": "id"})
        df = df.merge(tdf, on="id", how="left")
        df["llm_translation"] = df["llm_translation"].fillna("")

    if annotations_df.empty:
        df["error_type"] = "No Error"
        df["memo"] = ""
        df["reviewer"] = ""
    else:
        adf = annotations_df[["row_id", "error_type", "memo", "reviewer"]].rename(columns={"row_id": "id"})
        df = df.merge(adf, on="id", how="left")
        df["error_type"] = df["error_type"].fillna("No Error")
        df["memo"] = df["memo"].fillna("")
        df["reviewer"] = df["reviewer"].fillna("")

    prompt_name = ""
    model = ""
    if run:
        prompt_name = (run.get("prompt_versions") or {}).get("name", "")
        model = run.get("model", "")
    df["prompt_version_name"] = prompt_name
    df["model"] = model
    return df


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


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    df.reindex(columns=OUTPUT_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
    return out.getvalue().encode("utf-8-sig")


def dataset_csv_download_bytes(rows_df: pd.DataFrame) -> bytes:
    out = io.StringIO()
    df = rows_df.rename(columns={"row_id": "id"}).copy()
    for column in ["id", "tag", "source", "human_translation", "context", "speaker", "listener", "notes"]:
        if column not in df.columns:
            df[column] = ""
    df[["id", "tag", "source", "human_translation", "context", "speaker", "listener", "notes"]].to_csv(
        out,
        index=False,
        encoding="utf-8-sig",
    )
    return out.getvalue().encode("utf-8-sig")


def short_label(text: str, limit: int = 56) -> str:
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit - 1]}..."


def sidebar_openai_key():
    st.sidebar.subheader("OpenAI API Key")
    api_key = st.sidebar.text_input(
        "이번 세션에서 사용할 키",
        type="password",
        value=st.session_state.get("openai_api_key", ""),
        help="st.session_state에만 보관합니다. DB, CSV, 로그, Streamlit secrets에는 저장하지 않습니다.",
    )
    if api_key:
        st.session_state["openai_api_key"] = api_key
    elif "openai_api_key" in st.session_state:
        del st.session_state["openai_api_key"]


def render_upload_tab(supabase):
    st.subheader("CSV 업로드")
    example_df = pd.DataFrame(
        [
            {
                "id": "row_0001",
                "tag": "item_name",
                "source": "Legendary Engram",
                "human_translation": "전설 엔그램",
                "context": "DestinyInventoryItemDefinition",
                "speaker": "",
                "listener": "",
                "notes": "item | short",
                "llm_translation": "",
                "error_type": "",
                "memo": "",
                "reviewer": "",
            },
            {
                "id": "row_0002",
                "tag": "description",
                "source": "Contains a random Legendary weapon or armor piece.",
                "human_translation": "무작위 전설 무기 또는 방어구가 들어 있습니다.",
                "context": "Inventory item description",
                "speaker": "",
                "listener": "",
                "notes": "",
                "llm_translation": "무작위 전설 무기나 방어구 조각을 포함합니다.",
                "error_type": "Style",
                "memo": "UI 톤에 비해 문어체 느낌",
                "reviewer": "reviewer_a",
            },
        ]
    )
    with st.expander("CSV 형식 예시", expanded=True):
        st.caption(
            "필수: source, human_translation. 선택: id, tag, context, speaker, listener, notes. "
            "결과 CSV를 다시 업로드할 때는 llm_translation, error_type, memo, reviewer도 포함할 수 있습니다."
        )
        st.dataframe(example_df, width="stretch", hide_index=True)

    dataset_name = st.text_input("데이터셋 이름")
    description = st.text_area("설명", height=80)
    uploaded = st.file_uploader("CSV 업로드", type=["csv"])

    if uploaded is None:
        st.caption("필수 컬럼: source, human_translation. JSON 업로드는 지원하지 않습니다.")
        return

    try:
        df = normalize_csv(pd.read_csv(uploaded))
        st.success(f"CSV에서 {len(df)}행을 불러왔습니다.")
        if (df["source"].str.len() > LONG_SOURCE_WARNING_CHARS).any():
            st.warning(f"일부 source 값이 {LONG_SOURCE_WARNING_CHARS}자를 초과합니다.")
        st.dataframe(df.head(20), width="stretch")
    except Exception as exc:
        st.error(f"CSV를 불러오지 못했습니다: {exc}")
        return

    if st.button("데이터셋 저장", type="primary", disabled=not dataset_name.strip()):
        try:
            dataset_id = upload_dataset(supabase, dataset_name, description, df)
            st.session_state["selected_dataset_id"] = dataset_id
            st.success("데이터셋을 저장했습니다. 작업 탭에서 이어서 진행할 수 있습니다.")
        except Exception as exc:
            st.error(f"저장 실패: {exc}")


def render_prompt_tab(supabase):
    st.subheader("프롬프트 버전")
    prompts = get_prompt_versions(supabase)
    selected_name = st.selectbox("기존 프롬프트", [p["name"] for p in prompts])
    selected = next(p for p in prompts if p["name"] == selected_name)
    st.text_area("프롬프트 내용", value=selected["prompt_text"], height=260, disabled=True)

    used_run_count = count_runs_for_prompt(supabase, selected["prompt_version_id"])
    delete_disabled = selected.get("is_default") or used_run_count > 0
    delete_help = "기본 프롬프트와 이미 run에서 사용된 프롬프트는 추적성을 위해 삭제하지 않습니다."
    with st.expander("선택한 프롬프트 삭제", expanded=False):
        st.caption(f"{used_run_count}개의 번역 run에서 사용 중입니다.")
        confirm_delete_prompt = st.checkbox(
            f"프롬프트 삭제 확인: {selected_name}",
            key=f"confirm_delete_prompt_{selected['prompt_version_id']}",
            disabled=delete_disabled,
            help=delete_help,
        )
        if st.button(
            "프롬프트 삭제",
            disabled=delete_disabled or not confirm_delete_prompt,
            key=f"delete_prompt_{selected['prompt_version_id']}",
        ):
            try:
                delete_prompt_version(supabase, selected["prompt_version_id"])
                st.success("프롬프트 버전을 삭제했습니다.")
                st.rerun()
            except Exception as exc:
                st.error(f"프롬프트 삭제 실패: {exc}")

    st.divider()
    st.markdown("새 프롬프트 버전 만들기")
    new_name = st.text_input("새 프롬프트 이름")
    new_text = st.text_area("새 프롬프트 내용", value=selected["prompt_text"], height=260)
    if st.button("새 프롬프트 버전으로 저장", disabled=not new_name.strip() or not new_text.strip()):
        try:
            create_prompt_version(supabase, new_name, new_text)
            st.success("새 프롬프트 버전을 저장했습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"프롬프트 저장 실패: {exc}")


def render_translation_controls(supabase, rows_df: pd.DataFrame, dataset_id: str):
    st.subheader("LLM 번역 실행")
    prompts = get_prompt_versions(supabase)
    prompt_options = {p["name"]: p for p in prompts}
    selected_prompt_name = st.selectbox("프롬프트 버전", list(prompt_options.keys()))
    selected_prompt = prompt_options[selected_prompt_name]

    model_choice = st.selectbox("모델", MODEL_OPTIONS, index=0)
    if model_choice == "직접 입력":
        model = st.text_input("직접 입력할 모델명", value="gpt-4o-mini")
    else:
        model = model_choice
    available_ids = rows_df["row_id"].tolist()
    row_labels = {
        row["row_id"]: short_label(f"{row['row_id']}: {row['source']}")
        for _, row in rows_df.iterrows()
    }
    selection_options = ["범위 선택", "개별 선택"]
    selection_key = f"row_selection_mode_{dataset_id}"
    if st.session_state.get(selection_key) not in (None, *selection_options):
        del st.session_state[selection_key]
    selection_mode = st.radio(
        "번역할 row 선택",
        selection_options,
        horizontal=True,
        key=selection_key,
        help="연속된 구간은 범위 선택을, 특정 row만 다시 돌릴 때는 개별 선택을 사용하세요.",
    )

    if selection_mode == "범위 선택":
        batch_cols = st.columns(2)
        start_at = int(batch_cols[0].number_input(
            "시작 row 번호",
            min_value=1,
            max_value=max(1, len(available_ids)),
            value=1,
            step=1,
        ))
        end_at = int(batch_cols[1].number_input(
            "끝 row 번호",
            min_value=1,
            max_value=max(1, len(available_ids)),
            value=min(DEFAULT_BATCH_SIZE, len(available_ids)),
            step=1,
        ))
        if end_at < start_at:
            selected_ids = []
            st.warning("끝 row 번호는 시작 row 번호보다 크거나 같아야 합니다.")
        else:
            selected_ids = available_ids[start_at - 1 : end_at]
            if selected_ids:
                st.caption(f"{len(selected_ids)}개 row 선택: {selected_ids[0]} ~ {selected_ids[-1]}")
    else:
        default_ids = available_ids[: min(DEFAULT_BATCH_SIZE, len(available_ids))]
        selected_ids = st.multiselect(
            f"번역할 row (최대 {MAX_RUN_ROWS}개)",
            available_ids,
            default=default_ids,
            format_func=lambda rid: row_labels.get(rid, str(rid)),
        )
        st.caption(f"{len(selected_ids)}개 row 선택")

    if len(selected_ids) > MAX_RUN_ROWS:
        st.warning(f"한 번에 최대 {MAX_RUN_ROWS}개 row까지만 실행할 수 있습니다.")

    disabled = (
        not st.session_state.get("openai_api_key")
        or not selected_ids
        or len(selected_ids) > MAX_RUN_ROWS
        or not model.strip()
    )
    if st.button("번역 생성", type="primary", disabled=disabled):
        client = OpenAI(api_key=st.session_state["openai_api_key"])
        run_id = str(uuid.uuid4())
        supabase.table("translation_runs").insert(
            {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "prompt_version_id": selected_prompt["prompt_version_id"],
                "model": model.strip(),
                "created_at": utc_now_iso(),
            }
        ).execute()

        progress = st.progress(0)
        status = st.empty()
        selected_rows = rows_df[rows_df["row_id"].isin(selected_ids)]
        if selected_rows.empty:
            st.error("선택한 조건에 맞는 row가 없습니다.")
            return
        if len(selected_rows) != len(selected_ids):
            st.warning(f"선택한 ID {len(selected_ids)}개 중 {len(selected_rows)}개 row만 매칭되었습니다.")
        for idx, (_, row) in enumerate(selected_rows.iterrows(), start=1):
            status.write(f"{row['row_id']} 번역 중 ({idx}/{len(selected_rows)})")
            try:
                llm_translation = translate_one(client, model.strip(), selected_prompt["prompt_text"], row)
            except Exception as exc:
                llm_translation = f"ERROR: {exc}"

            supabase.table("translations").insert(
                {
                    "translation_id": str(uuid.uuid4()),
                    "row_id": row["row_id"],
                    "run_id": run_id,
                    "llm_translation": llm_translation,
                }
            ).execute()
            progress.progress(idx / len(selected_rows))

        st.session_state["selected_run_id"] = run_id
        st.success("번역 run이 완료되었습니다.")
        st.rerun()

    if not st.session_state.get("openai_api_key"):
        st.caption("번역 생성을 사용하려면 사이드바에 OpenAI API Key를 입력하세요.")


def render_cards(supabase, df: pd.DataFrame, run_id: str):
    for _, row in df.iterrows():
        with st.container(border=True):
            top_cols = st.columns([1, 2, 2, 2])
            top_cols[0].caption(row["tag"])
            top_cols[1].caption(f"ID: {row['id']}")
            top_cols[2].caption(f"화자: {row.get('speaker', '')}")
            top_cols[3].caption(f"청자: {row.get('listener', '')}")

            if row.get("context", ""):
                st.markdown("**맥락**")
                st.write(row["context"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**원문**")
                st.write(row["source"])
                st.markdown("**LLM 번역**")
                st.write(row.get("llm_translation", ""))
            with col2:
                st.markdown("**사람 번역**")
                st.write(row["human_translation"])
                if row.get("notes", ""):
                    st.markdown("**메모/노트**")
                    st.write(row["notes"])

            key_prefix = f"{run_id}_{row['id']}"
            ann_cols = st.columns([1, 2, 1, 1])
            current_error = row["error_type"] if row["error_type"] in ERROR_TYPES else "Other"
            error_type = ann_cols[0].selectbox(
                "오류 유형",
                ERROR_TYPES,
                index=ERROR_TYPES.index(current_error),
                key=f"error_{key_prefix}",
            )
            memo = ann_cols[1].text_area("메모", value=row["memo"], key=f"memo_{key_prefix}", height=80)
            reviewer = ann_cols[2].text_input("리뷰어", value=row["reviewer"], key=f"reviewer_{key_prefix}")
            if ann_cols[3].button("저장", key=f"save_{key_prefix}"):
                try:
                    save_annotation(supabase, row["id"], run_id, error_type, memo, reviewer)
                    st.success(f"{row['id']} 저장됨")
                except Exception as exc:
                    st.error(f"{row['id']} 저장 실패: {exc}")


def render_work_tab(supabase):
    datasets = list_datasets(supabase)
    if not datasets:
        st.info("아직 데이터셋이 없습니다. 먼저 CSV를 업로드해 주세요.")
        return

    dataset_labels = {
        f"{d['dataset_name']} / {d['uploaded_at'][:19]}": d["dataset_id"]
        for d in datasets
    }
    default_index = 0
    if st.session_state.get("selected_dataset_id") in dataset_labels.values():
        default_index = list(dataset_labels.values()).index(st.session_state["selected_dataset_id"])

    selected_label = st.selectbox("데이터셋", list(dataset_labels.keys()), index=default_index)
    dataset_id = dataset_labels[selected_label]
    st.session_state["selected_dataset_id"] = dataset_id

    rows_df = load_rows(supabase, dataset_id)
    if rows_df.empty:
        st.warning("선택한 데이터셋에 row가 없습니다.")
        return

    st.download_button(
        "선택한 데이터셋 CSV 다운로드",
        data=dataset_csv_download_bytes(rows_df),
        file_name=f"{selected_label.split(' / ')[0]}_dataset.csv",
        mime="text/csv",
    )

    with st.expander("새 LLM 번역 run 만들기", expanded=False):
        render_translation_controls(supabase, rows_df, dataset_id)

    runs = load_runs(supabase, dataset_id)
    if not runs:
        st.info("아직 번역 run이 없습니다. 위에서 새 run을 만들거나 llm_translation이 포함된 결과 CSV를 업로드하세요.")
        return

    run_options = {
        f"{run.get('created_at', '')[:19]} / {run.get('model', '')} / {(run.get('prompt_versions') or {}).get('name', '')}": run
        for run in runs
    }
    run_select_col, run_delete_col = st.columns([12, 1])
    selected_run_label = run_select_col.selectbox("번역 run", list(run_options.keys()))
    selected_run = run_options[selected_run_label]
    run_id = selected_run["run_id"]
    run_delete_col.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
    if run_delete_col.button("×", help="선택한 번역 run 삭제", key=f"delete_run_{run_id}"):
        try:
            delete_translation_run(supabase, run_id)
            if st.session_state.get("selected_run_id") == run_id:
                del st.session_state["selected_run_id"]
            st.success("번역 run을 삭제했습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"run 삭제 실패: {exc}")

    translations_df = load_translations(supabase, run_id)
    annotations_df = load_annotations(supabase, run_id)
    work_df = build_working_frame(rows_df, translations_df, annotations_df, selected_run)
    translated_df = work_df[work_df["llm_translation"].fillna("").astype(str).str.strip() != ""].copy()

    show_untranslated = st.checkbox("미번역 row도 함께 보기", value=False)
    display_base_df = work_df if show_untranslated else translated_df

    left, right = st.columns([1, 2])
    tags = ["All"] + sorted(display_base_df["tag"].dropna().unique().tolist())
    selected_tag = left.selectbox("태그 필터", tags)
    query = right.text_input("원문 / 사람 번역 / LLM 번역 / 메모 검색")

    filtered = display_base_df.copy()
    if selected_tag != "All":
        filtered = filtered[filtered["tag"] == selected_tag]
    if query.strip():
        q = query.strip().casefold()
        mask = pd.Series(False, index=filtered.index)
        for col in ["source", "human_translation", "llm_translation", "memo", "context"]:
            mask = mask | filtered[col].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        filtered = filtered[mask]

    st.caption(
        f"표시 중 {len(filtered)}행 / 번역됨 {len(translated_df)}행 / 전체 {len(work_df)}행"
    )
    view = st.radio("보기 방식", ["카드뷰", "테이블뷰"], horizontal=True)
    if view == "카드뷰":
        if filtered.empty:
            st.info("현재 필터에 맞는 row가 없습니다.")
        else:
            render_cards(supabase, filtered, run_id)
    else:
        st.dataframe(filtered.reindex(columns=TABLE_VIEW_COLUMNS), width="stretch", hide_index=True)

    st.download_button(
        "번역된 row CSV 다운로드",
        data=csv_download_bytes(translated_df),
        file_name=f"{selected_label.split(' / ')[0]}_results.csv",
        mime="text/csv",
        disabled=translated_df.empty,
    )
    with st.expander("다운로드 옵션", expanded=False):
        st.download_button(
            "전체 dataset row와 선택 run 컬럼 함께 다운로드",
            data=csv_download_bytes(work_df),
            file_name=f"{selected_label.split(' / ')[0]}_all_rows_results.csv",
            mime="text/csv",
        )


def render_review_card(row: pd.Series):
    with st.container(border=True):
        top_cols = st.columns([1, 1, 1])
        top_cols[0].caption(row.get("error_type", ""))
        top_cols[1].caption(f"리뷰어: {row.get('reviewer', '')}")
        top_cols[2].caption(f"run: {str(row.get('run_created_at', ''))[:19]}")

        st.markdown("**원문**")
        st.write(row.get("source", ""))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**사람 번역**")
            st.write(row.get("human_translation", ""))
        with col2:
            st.markdown("**LLM 번역**")
            st.write(row.get("llm_translation", ""))

        if str(row.get("memo", "")).strip():
            st.markdown("**메모**")
            st.write(row.get("memo", ""))


def render_label_review_tab(supabase):
    st.subheader("라벨 기준")
    guide_df = pd.DataFrame(LABEL_GUIDE)
    st.dataframe(
        guide_df.rename(columns={"error_type": "오류 유형", "meaning": "의미", "example": "예시"}),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("라벨링된 오류 모아보기")
    datasets = list_datasets(supabase)
    if not datasets:
        st.info("아직 데이터셋이 없습니다.")
        return

    dataset_labels = {
        f"{d['dataset_name']} / {d['uploaded_at'][:19]}": d["dataset_id"]
        for d in datasets
    }
    selected_dataset_label = st.selectbox("데이터셋", list(dataset_labels.keys()), key="review_dataset")
    dataset_id = dataset_labels[selected_dataset_label]
    rows_df = load_rows(supabase, dataset_id)
    runs = load_runs(supabase, dataset_id)
    if rows_df.empty or not runs:
        st.info("이 데이터셋에는 아직 리뷰할 번역 run이 없습니다.")
        return

    review_df = build_review_frame(supabase, dataset_id, rows_df, runs)
    if review_df.empty:
        st.info("아직 No Error 외의 라벨링된 오류가 없습니다.")
        return

    count_df = (
        review_df.groupby("error_type")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "error_type"], ascending=[False, True])
    )
    st.dataframe(
        count_df.rename(columns={"error_type": "오류 유형", "count": "건수"}),
        width="stretch",
        hide_index=True,
    )

    filter_cols = st.columns([1, 1, 2])
    selected_error = filter_cols[0].selectbox(
        "오류 유형",
        ["All"] + [error for error in ERROR_TYPES if error != "No Error"],
        key="review_error_filter",
    )
    run_options = {
        "All": None,
        **{
            f"{run.get('created_at', '')[:19]} / {run.get('model', '')} / {run['run_id'][:8]}": run["run_id"]
            for run in runs
        },
    }
    selected_run_label = filter_cols[1].selectbox("번역 run", list(run_options.keys()), key="review_run_filter")
    review_query = filter_cols[2].text_input("원문 / 번역 / 메모 검색", key="review_query")

    filtered = review_df.copy()
    if selected_error != "All":
        filtered = filtered[filtered["error_type"] == selected_error]
    selected_run_id = run_options[selected_run_label]
    if selected_run_id:
        filtered = filtered[filtered["run_id"] == selected_run_id]
    if review_query.strip():
        q = review_query.strip().casefold()
        mask = pd.Series(False, index=filtered.index)
        for col in ["source", "human_translation", "llm_translation", "memo", "reviewer"]:
            mask = mask | filtered[col].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        filtered = filtered[mask]

    st.caption(f"{len(filtered)}건 표시 / 전체 오류 라벨 {len(review_df)}건")
    review_view = st.radio("보기 방식", ["테이블뷰", "카드뷰"], horizontal=True, key="review_view")
    review_columns = [
        "error_type",
        "memo",
        "reviewer",
        "source",
        "human_translation",
        "llm_translation",
        "id",
        "tag",
        "run_created_at",
        "model",
        "prompt_version_name",
    ]
    if review_view == "테이블뷰":
        st.dataframe(filtered.reindex(columns=review_columns), width="stretch", hide_index=True)
    else:
        if filtered.empty:
            st.info("현재 필터에 맞는 오류 리뷰가 없습니다.")
        else:
            for _, row in filtered.iterrows():
                render_review_card(row)

    st.download_button(
        "필터된 오류 리뷰 CSV 다운로드",
        data=csv_download_bytes(filtered),
        file_name=f"{selected_dataset_label.split(' / ')[0]}_error_reviews.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )


def main():
    st.title(APP_TITLE)
    st.caption("LLM 번역 실패를 관찰하고 MQM 기반으로 함께 라벨링하기 위한 연구용 MVP")

    supabase = get_supabase_client()
    ensure_default_prompt(supabase)
    sidebar_openai_key()

    upload_tab, work_tab, prompt_tab, review_tab = st.tabs(["업로드", "작업", "프롬프트", "라벨 기준"])
    with upload_tab:
        render_upload_tab(supabase)
    with work_tab:
        render_work_tab(supabase)
    with prompt_tab:
        render_prompt_tab(supabase)
    with review_tab:
        render_label_review_tab(supabase)


if __name__ == "__main__":
    main()
