import hashlib
import io
import re
from datetime import datetime, timezone

import pandas as pd


APP_TITLE = "LLM 번역 실패 관찰 및 공동 라벨링 툴"
MAX_UPLOAD_ROWS = 300
MAX_RUN_ROWS = 50
DEFAULT_BATCH_SIZE = 10
LONG_SOURCE_WARNING_CHARS = 800
CONTEXT_PROMPT_LIMIT = 1000
DEFAULT_PARALLEL_WORKERS = 5

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_row(source: str, human_translation: str) -> str:
    raw = f"{source}\x1f{human_translation}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


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
