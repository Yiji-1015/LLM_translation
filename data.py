"""
순수 데이터 처리 모듈.

원칙: 이 파일은 Streamlit / Supabase / OpenAI에 의존하지 않습니다.
- DataFrame 변환, CSV 입출력, 상수만 담당
- 외부 시스템에 의존하지 않으므로 단위 테스트가 쉬워야 함
- 다른 모듈은 이 파일을 자유롭게 import 가능 (의존성의 가장 아래)
"""

import hashlib
import io
import re
from datetime import datetime, timezone

import pandas as pd


# ---------- 상수 ----------

APP_TITLE = "LLM 번역 실패 관찰 및 공동 라벨링 툴"

# 업로드 안전장치. CSV 한 번에 300행, 한 번 번역 run에 50행이 한계.
# 너무 큰 파일이 들어오면 Supabase 쪽도 부담이고, OpenAI 호출 비용도 폭증함.
MAX_UPLOAD_ROWS = 300
MAX_RUN_ROWS = 50
DEFAULT_BATCH_SIZE = 10  # 번역 범위 선택 시 기본값
LONG_SOURCE_WARNING_CHARS = 800  # 너무 긴 원문은 경고만 띄움
CONTEXT_PROMPT_LIMIT = 1000  # 프롬프트에 들어가는 context를 잘라 토큰 비용 절약
DEFAULT_PARALLEL_WORKERS = 5  # OpenAI 동시 호출 수 기본값

# CSV 다운로드 시 컬럼 순서. 결과 CSV의 "정답 스키마"이기도 함.
# 이 순서대로 내려가야 다시 업로드했을 때 컬럼 매칭이 어긋나지 않음.
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

# 테이블뷰에서 한눈에 보고 싶은 순서. 사람이 비교/검수 중심으로 보는 컬럼이 앞쪽.
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

# CSV에 없어도 되는 입력 컬럼 (없으면 빈 문자열 채움).
OPTIONAL_INPUT_COLUMNS = ["id", "tag", "context", "speaker", "listener", "notes"]
# 결과 CSV를 재업로드할 때 보존해야 하는 컬럼 (LLM 번역 + 라벨).
PRESERVED_COLUMNS = ["llm_translation", "error_type", "memo", "reviewer"]

# MQM(Multidimensional Quality Metrics) 기반 오류 유형.
# 순서는 라벨링 UI의 selectbox 순서이기도 함. "No Error"가 0번이라
# 기본값을 그대로 두면 "라벨 안 함" 의미가 됨.
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

# 라벨 기준 탭에서 사용자에게 보여줄 가이드. 코드와 별개의 "문서"라 봐도 됨.
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

# 모델 선택지. "직접 입력"은 selectbox 마지막에 두고, 선택되면 text_input으로 전환.
MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5.4-mini",
    "gpt-5.4",
    "직접 입력",
]

# 기본 번역 프롬프트.
# .format() 자리표시자가 들어가 있어 render_prompt에서 row 값으로 채움.
# 게임 현지화 톤을 유도하고, 플레이스홀더 보존을 명시함.
DEFAULT_PROMPT = """You are a professional English-to-Korean game localization translator.
Translate the source text into natural Korean.
Preserve placeholders, tags, variables, numbers, and line breaks exactly.
Use the given tag and context only as guidance.
Do not add explanations.

Tag: {tag}
Context: {context}
Source: {source}

Return only the Korean translation."""


# ---------- 유틸 ----------

def utc_now_iso() -> str:
    """타임존을 명시해 ISO 8601 문자열을 만든다. Supabase는 timestamptz를 쓰므로
    UTC + ISO 형식이 가장 안전함."""
    return datetime.now(timezone.utc).isoformat()


def hash_row(source: str, human_translation: str) -> str:
    """row의 (원문, 사람번역) 쌍을 sha256으로 해싱.
    \\x1f(Unit Separator)를 구분자로 써서 'ab' + 'c' vs 'a' + 'bc'가
    같은 해시가 되는 충돌을 막음."""
    raw = f"{source}\x1f{human_translation}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def natural_sort_key(value: str):
    """'row_2'가 'row_10'보다 앞에 오게 하는 자연 정렬 키.
    문자열을 숫자/비숫자로 쪼개고 숫자 부분은 int로 변환해 비교.
    Supabase의 order by는 사전식이라 'row_10' < 'row_2'가 되므로
    DB에서 가져온 뒤 파이썬에서 한 번 더 정렬함."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


# ---------- DataFrame 변환 ----------

def normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """업로드된 CSV를 앱 내부 표준 스키마로 정규화.
    - 컬럼 공백 제거
    - 필수 컬럼 검증
    - 행 수 제한
    - 누락된 선택 컬럼은 빈 문자열로 채움
    - id/tag 빈 값에 기본값 부여
    - 중복 검출용 hash 컬럼 부여
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # 필수 컬럼이 없으면 즉시 실패. 사용자에게 명확한 에러 메시지를 주는 것이 목적.
    missing = {"source", "human_translation"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"CSV 업로드는 최대 {MAX_UPLOAD_ROWS}행까지 가능합니다.")

    # id가 없으면 row_0001, row_0002 ... 식으로 자동 생성.
    # 4자리 zero-padding → 자연 정렬과 잘 어울림.
    if "id" not in df.columns:
        df["id"] = [f"row_{idx + 1:04d}" for idx in range(len(df))]
    if "tag" not in df.columns:
        df["tag"] = "Uncategorized"

    # 어떤 컬럼이 빠져 있어도 이후 처리에서 KeyError가 안 나도록 다 만들어 둠.
    for column in OPTIONAL_INPUT_COLUMNS + PRESERVED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # NaN을 모두 빈 문자열로. 이후 .str.* 메서드가 NaN에서 실패하지 않도록.
    for column in OPTIONAL_INPUT_COLUMNS + ["source", "human_translation"] + PRESERVED_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    # id/tag가 공백 문자열인 경우에도 기본값 부여. where(조건, 대체값)은
    # 조건이 False인 위치에 대체값을 채움.
    df["id"] = df["id"].where(df["id"].str.strip() != "", [f"row_{idx + 1:04d}" for idx in range(len(df))])
    df["tag"] = df["tag"].where(df["tag"].str.strip() != "", "Uncategorized")
    df["hash"] = df.apply(lambda row: hash_row(row["source"], row["human_translation"]), axis=1)
    return df


def build_working_frame(rows_df: pd.DataFrame, translations_df: pd.DataFrame, annotations_df: pd.DataFrame, run):
    """`rows` + `translations` + `annotations` 세 테이블을 row_id 기준으로 LEFT JOIN.
    Supabase에서 따로 가져온 결과를 파이썬 쪽에서 조립함.

    - translations가 비어 있으면 llm_translation은 빈 문자열
    - annotations가 비어 있으면 error_type='No Error'가 기본
    - run 정보에서 프롬프트 이름/모델명을 가져와 컬럼으로 펼침
    """
    if rows_df.empty:
        return rows_df

    # DB의 row_id 컬럼명을 앱 내부의 id로 통일.
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

    # run 객체에 외래키 join된 prompt_versions가 dict로 들어있음 (db.load_runs 참고).
    prompt_name = ""
    model = ""
    if run:
        prompt_name = (run.get("prompt_versions") or {}).get("name", "")
        model = run.get("model", "")
    df["prompt_version_name"] = prompt_name
    df["model"] = model
    return df


# ---------- CSV 내보내기 ----------

def csv_download_bytes(df: pd.DataFrame) -> bytes:
    """결과 CSV를 utf-8-sig로 내보냄.
    utf-8-sig는 BOM이 붙은 utf-8 — Excel이 한글을 깨지 않고 열기 위한 관습."""
    out = io.StringIO()
    # reindex로 OUTPUT_COLUMNS 순서를 강제. 없는 컬럼은 NaN으로 채워짐.
    df.reindex(columns=OUTPUT_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
    return out.getvalue().encode("utf-8-sig")


def dataset_csv_download_bytes(rows_df: pd.DataFrame) -> bytes:
    """원본 dataset만 다운로드 (번역/라벨 컬럼 제외).
    팀원이 row만 받아서 다른 도구로 분석할 때 사용."""
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
    """multiselect 같은 좁은 UI에 긴 원문을 표시할 때 잘라서 보여주기 위함."""
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit - 1]}..."
