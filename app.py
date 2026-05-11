import hashlib
import io
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from openai import OpenAI
from supabase import create_client


APP_TITLE = "LLM Translation Failure Lab"
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
        st.error("Streamlit secrets file was not found.")
        st.info(
            "Create `.streamlit/secrets.toml` in this project, then add "
            "`SUPABASE_URL` and `SUPABASE_KEY`."
        )
        st.code(
            'SUPABASE_URL = "https://your-project.supabase.co"\n'
            'SUPABASE_KEY = "your-supabase-key"',
            language="toml",
        )
        st.stop()
    if not url or not key:
        st.error("SUPABASE_URL and SUPABASE_KEY must be set in Streamlit secrets.")
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
        raise ValueError(f"CSV upload is limited to {MAX_UPLOAD_ROWS} rows.")

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


def load_translations(supabase, run_id: str) -> pd.DataFrame:
    rows = supabase.table("translations").select("*").eq("run_id", run_id).execute().data
    return pd.DataFrame(rows)


def load_annotations(supabase, run_id: str) -> pd.DataFrame:
    rows = supabase.table("annotations").select("*").eq("run_id", run_id).execute().data
    return pd.DataFrame(rows)


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


def short_label(text: str, limit: int = 56) -> str:
    text = str(text)
    return text if len(text) <= limit else f"{text[:limit - 1]}..."


def sidebar_openai_key():
    st.sidebar.subheader("OpenAI API Key")
    api_key = st.sidebar.text_input(
        "Enter key for this session",
        type="password",
        value=st.session_state.get("openai_api_key", ""),
        help="Stored only in st.session_state. It is not saved to DB, CSV, logs, or Streamlit secrets.",
    )
    if api_key:
        st.session_state["openai_api_key"] = api_key
    elif "openai_api_key" in st.session_state:
        del st.session_state["openai_api_key"]


def render_upload_tab(supabase):
    st.subheader("CSV upload")
    dataset_name = st.text_input("Dataset name")
    description = st.text_area("Description", height=80)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.caption("Required columns: source, human_translation. JSON upload is intentionally not supported.")
        return

    try:
        df = normalize_csv(pd.read_csv(uploaded))
        st.success(f"Loaded {len(df)} rows from CSV.")
        if (df["source"].str.len() > LONG_SOURCE_WARNING_CHARS).any():
            st.warning(f"Some source values exceed {LONG_SOURCE_WARNING_CHARS} characters.")
        st.dataframe(df.head(20), use_container_width=True)
    except Exception as exc:
        st.error(f"CSV could not be loaded: {exc}")
        return

    if st.button("Save dataset to Supabase", type="primary", disabled=not dataset_name.strip()):
        try:
            dataset_id = upload_dataset(supabase, dataset_name, description, df)
            st.session_state["selected_dataset_id"] = dataset_id
            st.success("Dataset saved. You can continue from the Work tab.")
        except Exception as exc:
            st.error(f"Save failed: {exc}")


def render_prompt_tab(supabase):
    st.subheader("Prompt versions")
    prompts = get_prompt_versions(supabase)
    selected_name = st.selectbox("Existing prompt", [p["name"] for p in prompts])
    selected = next(p for p in prompts if p["name"] == selected_name)
    st.text_area("Prompt text", value=selected["prompt_text"], height=260, disabled=True)

    st.divider()
    st.markdown("Create new prompt version")
    new_name = st.text_input("New prompt name")
    new_text = st.text_area("New prompt text", value=selected["prompt_text"], height=260)
    if st.button("Save as new prompt version", disabled=not new_name.strip() or not new_text.strip()):
        try:
            create_prompt_version(supabase, new_name, new_text)
            st.success("New prompt version saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Prompt save failed: {exc}")


def render_translation_controls(supabase, rows_df: pd.DataFrame, dataset_id: str):
    st.subheader("LLM translation run")
    prompts = get_prompt_versions(supabase)
    prompt_options = {p["name"]: p for p in prompts}
    selected_prompt_name = st.selectbox("Prompt version", list(prompt_options.keys()))
    selected_prompt = prompt_options[selected_prompt_name]

    model = st.text_input("Model", value="gpt-4o-mini")
    available_ids = rows_df["row_id"].tolist()
    default_ids = available_ids[: min(DEFAULT_BATCH_SIZE, len(available_ids))]
    selected_ids = st.multiselect(
        f"Rows to translate (max {MAX_RUN_ROWS})",
        available_ids,
        default=default_ids,
        format_func=lambda rid: short_label(f"{rid}: {rows_df.loc[rows_df['row_id'] == rid, 'source'].iloc[0]}"),
    )

    if len(selected_ids) > MAX_RUN_ROWS:
        st.warning(f"Select at most {MAX_RUN_ROWS} rows per run.")

    disabled = (
        not st.session_state.get("openai_api_key")
        or not selected_ids
        or len(selected_ids) > MAX_RUN_ROWS
        or not model.strip()
    )
    if st.button("Generate translations", type="primary", disabled=disabled):
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
        for idx, (_, row) in enumerate(selected_rows.iterrows(), start=1):
            status.write(f"Translating {row['row_id']} ({idx}/{len(selected_rows)})")
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
        st.success("Translation run completed.")
        st.rerun()

    if not st.session_state.get("openai_api_key"):
        st.caption("Enter an OpenAI API key in the sidebar to enable generation.")


def render_cards(supabase, df: pd.DataFrame, run_id: str):
    for _, row in df.iterrows():
        with st.container(border=True):
            top_cols = st.columns([1, 2, 2, 2])
            top_cols[0].caption(row["tag"])
            top_cols[1].caption(f"ID: {row['id']}")
            top_cols[2].caption(f"Speaker: {row.get('speaker', '')}")
            top_cols[3].caption(f"Listener: {row.get('listener', '')}")

            if row.get("context", ""):
                st.markdown("**Context**")
                st.write(row["context"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Source**")
                st.write(row["source"])
                st.markdown("**LLM translation**")
                st.write(row.get("llm_translation", ""))
            with col2:
                st.markdown("**Human translation**")
                st.write(row["human_translation"])
                if row.get("notes", ""):
                    st.markdown("**Notes**")
                    st.write(row["notes"])

            key_prefix = f"{run_id}_{row['id']}"
            ann_cols = st.columns([1, 2, 1, 1])
            current_error = row["error_type"] if row["error_type"] in ERROR_TYPES else "Other"
            error_type = ann_cols[0].selectbox(
                "Error type",
                ERROR_TYPES,
                index=ERROR_TYPES.index(current_error),
                key=f"error_{key_prefix}",
            )
            memo = ann_cols[1].text_area("Memo", value=row["memo"], key=f"memo_{key_prefix}", height=80)
            reviewer = ann_cols[2].text_input("Reviewer", value=row["reviewer"], key=f"reviewer_{key_prefix}")
            if ann_cols[3].button("Save", key=f"save_{key_prefix}"):
                try:
                    save_annotation(supabase, row["id"], run_id, error_type, memo, reviewer)
                    st.success(f"Saved {row['id']}")
                except Exception as exc:
                    st.error(f"Save failed for {row['id']}: {exc}")


def render_work_tab(supabase):
    datasets = list_datasets(supabase)
    if not datasets:
        st.info("No datasets yet. Upload a CSV first.")
        return

    dataset_labels = {
        f"{d['dataset_name']} / {d['uploaded_at'][:19]}": d["dataset_id"]
        for d in datasets
    }
    default_index = 0
    if st.session_state.get("selected_dataset_id") in dataset_labels.values():
        default_index = list(dataset_labels.values()).index(st.session_state["selected_dataset_id"])

    selected_label = st.selectbox("Dataset", list(dataset_labels.keys()), index=default_index)
    dataset_id = dataset_labels[selected_label]
    st.session_state["selected_dataset_id"] = dataset_id

    rows_df = load_rows(supabase, dataset_id)
    if rows_df.empty:
        st.warning("Selected dataset has no rows.")
        return

    with st.expander("Create a new LLM run", expanded=False):
        render_translation_controls(supabase, rows_df, dataset_id)

    runs = load_runs(supabase, dataset_id)
    if not runs:
        st.info("No translation run yet. Create one above, or upload a result CSV with llm_translation.")
        return

    run_options = {
        f"{run.get('created_at', '')[:19]} / {run.get('model', '')} / {(run.get('prompt_versions') or {}).get('name', '')}": run
        for run in runs
    }
    selected_run_label = st.selectbox("Translation run", list(run_options.keys()))
    selected_run = run_options[selected_run_label]
    run_id = selected_run["run_id"]

    translations_df = load_translations(supabase, run_id)
    annotations_df = load_annotations(supabase, run_id)
    work_df = build_working_frame(rows_df, translations_df, annotations_df, selected_run)

    left, right = st.columns([1, 2])
    tags = ["All"] + sorted(work_df["tag"].dropna().unique().tolist())
    selected_tag = left.selectbox("Tag filter", tags)
    query = right.text_input("Search source / human / LLM / memo")

    filtered = work_df.copy()
    if selected_tag != "All":
        filtered = filtered[filtered["tag"] == selected_tag]
    if query.strip():
        q = query.strip().casefold()
        mask = pd.Series(False, index=filtered.index)
        for col in ["source", "human_translation", "llm_translation", "memo", "context"]:
            mask = mask | filtered[col].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        filtered = filtered[mask]

    st.caption(f"{len(filtered)} visible rows / {len(work_df)} total rows")
    view = st.radio("View", ["Card view", "Table view"], horizontal=True)
    if view == "Card view":
        render_cards(supabase, filtered, run_id)
    else:
        st.dataframe(filtered.reindex(columns=OUTPUT_COLUMNS), use_container_width=True, hide_index=True)

    st.download_button(
        "Download result CSV",
        data=csv_download_bytes(work_df),
        file_name=f"{selected_label.split(' / ')[0]}_results.csv",
        mime="text/csv",
    )


def main():
    st.title(APP_TITLE)
    st.caption("Research MVP for observing LLM translation failures and MQM-style collaborative labeling.")

    supabase = get_supabase_client()
    ensure_default_prompt(supabase)
    sidebar_openai_key()

    upload_tab, work_tab, prompt_tab = st.tabs(["Upload", "Work", "Prompts"])
    with upload_tab:
        render_upload_tab(supabase)
    with work_tab:
        render_work_tab(supabase)
    with prompt_tab:
        render_prompt_tab(supabase)


if __name__ == "__main__":
    main()
