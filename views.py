"""
Streamlit UI 렌더링 계층.

이 파일이 가장 큼 — 화면을 그리는 코드는 본질적으로 길어지기 쉬움.
구조:
- 사이드바: sidebar_openai_key
- 탭별 렌더 함수: render_upload_tab / render_work_tab / render_prompt_tab / render_label_review_tab
- 보조 컴포넌트: render_translation_controls, render_cards, render_review_card

Streamlit 핵심 개념 (이 파일을 읽기 전에 알아두면 좋음):
1. Streamlit은 위젯 상호작용마다 스크립트 전체를 위에서부터 재실행한다 (= "rerun").
2. 위젯 값은 자동으로 보존되지만, 코드 변수는 매 rerun마다 초기화된다.
3. 보존하고 싶은 값은 `st.session_state["key"] = value`로 저장한다.
4. 같은 위젯을 같은 페이지에 두 번 그리려면 `key=`로 고유한 키를 줘야 한다.
5. `st.rerun()`은 즉시 다시 스크립트를 처음부터 실행 — DB 변경 후 화면 갱신용.
"""

import pandas as pd
import streamlit as st
from openai import OpenAI

from data import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PARALLEL_WORKERS,
    ERROR_TYPES,
    LABEL_GUIDE,
    LONG_SOURCE_WARNING_CHARS,
    MAX_RUN_ROWS,
    MODEL_OPTIONS,
    TABLE_VIEW_COLUMNS,
    build_working_frame,
    csv_download_bytes,
    dataset_csv_download_bytes,
    normalize_csv,
    short_label,
)
from db import (
    build_review_frame,
    count_runs_for_prompt,
    create_prompt_version,
    create_translation_run,
    delete_annotation,
    delete_prompt_version,
    delete_translation_run,
    get_prompt_versions,
    insert_translation,
    list_datasets,
    load_annotations,
    load_rows,
    load_runs,
    load_translations,
    save_annotation,
    upload_dataset,
)
from translate import translate_rows_parallel


# ---------- 사이드바 ----------

def sidebar_openai_key():
    """OpenAI API 키 입력란을 사이드바에 그림.

    중요: 키는 st.session_state에만 저장 — DB, CSV, 로그, secrets 어디에도 안 적힘.
    페이지를 닫으면 세션과 함께 사라지므로 매번 다시 입력해야 함.
    type='password'는 단순히 화면에서 별표로 가리는 효과 (저장에는 영향 없음)."""
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
        # 사용자가 입력란을 비웠으면 session_state에서도 지움.
        del st.session_state["openai_api_key"]


# ---------- 업로드 탭 ----------

def render_upload_tab(supabase):
    st.subheader("CSV 업로드")
    # 예시 DataFrame을 직접 만들어 표로 보여줌 — 별도 파일 안 만들고 가이드 역할만.
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
        # 파일이 아직 없을 때는 안내만 띄우고 종료.
        st.caption("필수 컬럼: source, human_translation. JSON 업로드는 지원하지 않습니다.")
        return

    try:
        # 업로드 직후 정규화. 실패하면 즉시 사용자에게 에러 메시지를 보여주고 종료.
        df = normalize_csv(pd.read_csv(uploaded))
        st.success(f"CSV에서 {len(df)}행을 불러왔습니다.")
        if (df["source"].str.len() > LONG_SOURCE_WARNING_CHARS).any():
            # 너무 긴 원문이 있으면 경고만 — 막지는 않음 (사용자 판단 존중).
            st.warning(f"일부 source 값이 {LONG_SOURCE_WARNING_CHARS}자를 초과합니다.")
        st.dataframe(df.head(20), width="stretch")
    except Exception as exc:
        st.error(f"CSV를 불러오지 못했습니다: {exc}")
        return

    # 데이터셋 이름이 비어 있으면 저장 버튼을 비활성화.
    if st.button("데이터셋 저장", type="primary", disabled=not dataset_name.strip()):
        try:
            dataset_id = upload_dataset(supabase, dataset_name, description, df)
            # 작업 탭에서 자동 선택되도록 dataset_id를 session_state에 기록.
            st.session_state["selected_dataset_id"] = dataset_id
            st.success("데이터셋을 저장했습니다. 작업 탭에서 이어서 진행할 수 있습니다.")
        except Exception as exc:
            st.error(f"저장 실패: {exc}")


# ---------- 프롬프트 탭 ----------

def render_prompt_tab(supabase):
    st.subheader("프롬프트 버전")
    prompts = get_prompt_versions(supabase)
    selected_name = st.selectbox("기존 프롬프트", [p["name"] for p in prompts])
    selected = next(p for p in prompts if p["name"] == selected_name)
    # disabled=True로 읽기 전용 텍스트 영역 — 보여주기만 함.
    st.text_area("프롬프트 내용", value=selected["prompt_text"], height=260, disabled=True)

    # 삭제 가능 여부 판정:
    # - 기본 프롬프트는 못 지움 (앱이 부팅 시 의존)
    # - 이미 run에서 사용된 프롬프트는 추적성 위해 잠금
    used_run_count = count_runs_for_prompt(supabase, selected["prompt_version_id"])
    delete_disabled = selected.get("is_default") or used_run_count > 0
    delete_help = "기본 프롬프트와 이미 run에서 사용된 프롬프트는 추적성을 위해 삭제하지 않습니다."
    with st.expander("선택한 프롬프트 삭제", expanded=False):
        st.caption(f"{used_run_count}개의 번역 run에서 사용 중입니다.")
        # 체크박스를 한 번 더 거쳐야 삭제 가능 — 실수 방지.
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
                st.rerun()  # 삭제 후 즉시 목록 갱신.
            except Exception as exc:
                st.error(f"프롬프트 삭제 실패: {exc}")

    st.divider()
    st.markdown("새 프롬프트 버전 만들기")
    new_name = st.text_input("새 프롬프트 이름")
    # 선택한 프롬프트 내용으로 초기값을 채워서 "복제 후 수정" 워크플로우를 자연스럽게.
    new_text = st.text_area("새 프롬프트 내용", value=selected["prompt_text"], height=260)
    if st.button("새 프롬프트 버전으로 저장", disabled=not new_name.strip() or not new_text.strip()):
        try:
            create_prompt_version(supabase, new_name, new_text)
            st.success("새 프롬프트 버전을 저장했습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"프롬프트 저장 실패: {exc}")


# ---------- 번역 실행 컨트롤 ----------

def render_translation_controls(supabase, rows_df: pd.DataFrame, dataset_id: str):
    """번역 run을 만드는 폼.
    호출 흐름:
      프롬프트/모델 선택 → row 선택 → 동시 호출 수 슬라이더 → "번역 생성" 버튼
      → run 메타 insert → 병렬 번역 시작 → 끝나는 row마다 translations insert + progress 업데이트
    """
    st.subheader("LLM 번역 실행")
    prompts = get_prompt_versions(supabase)
    # 이름 → 객체 매핑을 만들어 selectbox 라벨로 이름만 노출.
    prompt_options = {p["name"]: p for p in prompts}
    selected_prompt_name = st.selectbox("프롬프트 버전", list(prompt_options.keys()))
    selected_prompt = prompt_options[selected_prompt_name]

    model_choice = st.selectbox("모델", MODEL_OPTIONS, index=0)
    # "직접 입력"을 고른 경우만 자유 입력란을 추가 노출.
    if model_choice == "직접 입력":
        model = st.text_input("직접 입력할 모델명", value="gpt-4o-mini")
    else:
        model = model_choice
    available_ids = rows_df["row_id"].tolist()
    # multiselect에서 row를 알아볼 수 있도록 id + 원문 앞부분을 라벨로.
    row_labels = {
        row["row_id"]: short_label(f"{row['row_id']}: {row['source']}")
        for _, row in rows_df.iterrows()
    }

    # selection mode를 session_state로 보존하되, 잘못된 값이 남아있으면 초기화.
    # (예: 이전 버전에서 다른 이름의 옵션을 썼다면 selectbox가 깨질 수 있음)
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
        # 시작/끝 row 번호를 받고, 그 사이 구간을 slice로 가져옴 (1-based → 0-based 변환).
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
        # 개별 선택 모드: 자유롭게 row를 골라서 부분 재실행에 사용.
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

    # OpenAI 동시 호출 수. 너무 높이면 429(rate limit)에 걸릴 수 있음.
    parallel_workers = st.slider(
        "동시 호출 수",
        min_value=1,
        max_value=10,
        value=DEFAULT_PARALLEL_WORKERS,
        help="OpenAI를 동시에 몇 개씩 호출할지. 너무 높이면 rate limit에 걸릴 수 있습니다.",
    )

    # 실행 버튼 비활성화 조건을 한 곳에 모아 둠.
    disabled = (
        not st.session_state.get("openai_api_key")
        or not selected_ids
        or len(selected_ids) > MAX_RUN_ROWS
        or not model.strip()
    )
    if st.button("번역 생성", type="primary", disabled=disabled):
        # 매번 새로 OpenAI 클라이언트를 만듦 — 키가 바뀌었을 수 있고, 부하도 크지 않음.
        client = OpenAI(api_key=st.session_state["openai_api_key"])
        selected_rows = rows_df[rows_df["row_id"].isin(selected_ids)]
        if selected_rows.empty:
            st.error("선택한 조건에 맞는 row가 없습니다.")
            return
        if len(selected_rows) != len(selected_ids):
            st.warning(f"선택한 ID {len(selected_ids)}개 중 {len(selected_rows)}개 row만 매칭되었습니다.")

        # 1) run 메타데이터 먼저 만들기 — 일부만 성공해도 run으로 묶이게.
        run_id = create_translation_run(
            supabase,
            dataset_id=dataset_id,
            prompt_version_id=selected_prompt["prompt_version_id"],
            model=model.strip(),
        )

        total = len(selected_rows)
        progress = st.progress(0)
        status = st.empty()  # status는 매번 덮어쓰는 placeholder.
        completed = 0
        # 2) 병렬 번역. 끝난 순서대로 yield되므로 progress 바는 빠른 순으로 차오름.
        for row, llm_translation in translate_rows_parallel(
            client,
            model.strip(),
            selected_prompt["prompt_text"],
            selected_rows,
            max_workers=parallel_workers,
        ):
            completed += 1
            status.write(f"{row['row_id']} 완료 ({completed}/{total})")
            # 3) row 하나 끝날 때마다 즉시 DB에 저장 — 중간에 끊겨도 부분 결과 보존.
            insert_translation(supabase, row["row_id"], run_id, llm_translation)
            progress.progress(completed / total)

        st.session_state["selected_run_id"] = run_id
        st.success("번역 run이 완료되었습니다.")
        st.rerun()  # 새로 만든 run이 셀렉트박스에 즉시 나타나도록 새로고침.

    if not st.session_state.get("openai_api_key"):
        st.caption("번역 생성을 사용하려면 사이드바에 OpenAI API Key를 입력하세요.")


# ---------- 카드뷰 ----------

def render_cards(supabase, df: pd.DataFrame, run_id: str):
    """라벨링 카드뷰. 각 row마다 한 박스씩 그림.
    위젯 key는 (run_id, row_id) 조합으로 만들어 다른 run/row와 겹치지 않게 함."""
    for _, row in df.iterrows():
        with st.container(border=True):
            # 상단 헤더: tag / id / 화자 / 청자
            top_cols = st.columns([1, 2, 2, 2])
            top_cols[0].caption(row["tag"])
            top_cols[1].caption(f"ID: {row['id']}")
            top_cols[2].caption(f"화자: {row.get('speaker', '')}")
            top_cols[3].caption(f"청자: {row.get('listener', '')}")

            if row.get("context", ""):
                st.markdown("**맥락**")
                st.write(row["context"])

            # 좌: 원문 + LLM 번역 / 우: 사람 번역 + 노트
            # → 시선이 좌우로 왔다갔다 하면서 비교하기 쉬움.
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

            # 위젯 key는 카드마다 고유해야 함 — run_id + row id로 prefix.
            key_prefix = f"{run_id}_{row['id']}"
            ann_cols = st.columns([1, 2, 1, 1, 1])

            # 기존 error_type이 ERROR_TYPES 목록에 없는 값이면 'Other'로 fallback.
            # (예: 라벨 enum이 바뀐 뒤 옛 데이터가 있을 때 selectbox가 깨지지 않게)
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
            if ann_cols[4].button("삭제", key=f"delete_annotation_{key_prefix}"):
                try:
                    delete_annotation(supabase, row["id"], run_id)
                    st.success(f"{row['id']} 리뷰 삭제됨")
                    st.rerun()
                except Exception as exc:
                    st.error(f"{row['id']} 리뷰 삭제 실패: {exc}")


# ---------- 작업 탭 ----------

def render_work_tab(supabase):
    """가장 큰 화면. dataset 선택 → run 선택 → 필터/검색 → 카드/테이블뷰 → 다운로드."""
    datasets = list_datasets(supabase)
    if not datasets:
        st.info("아직 데이터셋이 없습니다. 먼저 CSV를 업로드해 주세요.")
        return

    # 라벨 → dataset_id 매핑. selectbox는 라벨만 보여줌.
    dataset_labels = {
        f"{d['dataset_name']} / {d['uploaded_at'][:19]}": d["dataset_id"]
        for d in datasets
    }
    # 업로드 직후 자동으로 그 dataset이 선택되도록 — selected_dataset_id를 기억.
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
        # 파일명에 데이터셋 이름을 넣어 알아보기 쉽게.
        file_name=f"{selected_label.split(' / ')[0]}_dataset.csv",
        mime="text/csv",
    )

    # 번역 실행 폼은 expander로 접어 둠 — 처음 들어왔을 때 화면이 너무 길어지지 않게.
    with st.expander("새 LLM 번역 run 만들기", expanded=False):
        render_translation_controls(supabase, rows_df, dataset_id)

    runs = load_runs(supabase, dataset_id)
    if not runs:
        st.info("아직 번역 run이 없습니다. 위에서 새 run을 만들거나 llm_translation이 포함된 결과 CSV를 업로드하세요.")
        return

    # run 라벨에는 timestamp/모델/프롬프트 이름을 함께 — 여러 run을 구분하기 쉽게.
    run_options = {
        f"{run.get('created_at', '')[:19]} / {run.get('model', '')} / {(run.get('prompt_versions') or {}).get('name', '')}": run
        for run in runs
    }
    # 셀렉트박스 옆에 작은 ×버튼 두기 위해 컬럼 비율을 12:1로.
    run_select_col, run_delete_col = st.columns([12, 1])
    selected_run_label = run_select_col.selectbox("번역 run", list(run_options.keys()))
    selected_run = run_options[selected_run_label]
    run_id = selected_run["run_id"]
    # 버튼이 selectbox보다 살짝 위에 붙도록 빈 div로 높이 조정 (Streamlit 정렬 한계 우회).
    run_delete_col.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
    # 의도적으로 확인 절차 없이 한 번 클릭에 삭제 — run은 실험성 산출물이라 가볍게 정리하기 위함.
    if run_delete_col.button("×", help="선택한 번역 run 삭제", key=f"delete_run_{run_id}"):
        try:
            delete_translation_run(supabase, run_id)
            # 지운 run이 selected 상태였다면 session_state에서도 정리.
            if st.session_state.get("selected_run_id") == run_id:
                del st.session_state["selected_run_id"]
            st.success("번역 run을 삭제했습니다.")
            st.rerun()
        except Exception as exc:
            st.error(f"run 삭제 실패: {exc}")

    # 세 테이블을 합쳐 작업용 프레임 생성.
    translations_df = load_translations(supabase, run_id)
    annotations_df = load_annotations(supabase, run_id)
    work_df = build_working_frame(rows_df, translations_df, annotations_df, selected_run)
    # 기본적으로는 "이미 번역된 row만" 보여줌 — 미번역 row까지 보고 싶으면 체크박스.
    translated_df = work_df[work_df["llm_translation"].fillna("").astype(str).str.strip() != ""].copy()

    show_untranslated = st.checkbox("미번역 row도 함께 보기", value=False)
    display_base_df = work_df if show_untranslated else translated_df

    # 좌측: 태그 필터 (drop down) / 우측: 자유 검색 입력란
    left, right = st.columns([1, 2])
    tags = ["All"] + sorted(display_base_df["tag"].dropna().unique().tolist())
    selected_tag = left.selectbox("태그 필터", tags)
    query = right.text_input("원문 / 사람 번역 / LLM 번역 / 메모 검색")

    filtered = display_base_df.copy()
    if selected_tag != "All":
        filtered = filtered[filtered["tag"] == selected_tag]
    if query.strip():
        # casefold는 대소문자 무시 비교 (lower()보다 강력 — 일부 유니코드까지 처리).
        q = query.strip().casefold()
        # 여러 컬럼 OR 검색을 위해 마스크를 누적 합산.
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
            if "work_visible_count" not in st.session_state:
                st.session_state["work_visible_count"] = 10

            visible_count = st.session_state["work_visible_count"]
            visible_df = filtered.head(visible_count)

            render_cards(supabase, visible_df, run_id)

            remaining = len(filtered) - visible_count
            if remaining > 0:
                if st.button(f"+{min(10, remaining)} 더 보기", key="work_load_more"):
                    st.session_state["work_visible_count"] += 10
                    st.rerun()

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
        # 미번역 row까지 포함해서 통째로 받고 싶을 때.
        st.download_button(
            "전체 dataset row와 선택 run 컬럼 함께 다운로드",
            data=csv_download_bytes(work_df),
            file_name=f"{selected_label.split(' / ')[0]}_all_rows_results.csv",
            mime="text/csv",
        )


# ---------- 리뷰 탭 ----------

def render_review_card(supabase, row: pd.Series):
    """라벨 기준 탭의 카드 — 라벨링된 오류 하나를 카드 한 장으로 표시.
    수정 기능 없이 읽기 전용. (수정은 작업 탭에서 함)"""
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

        if st.button("리뷰 삭제", key=f"delete_review_{row.get('run_id', '')}_{row.get('id', '')}"):
            try:
                delete_annotation(supabase, row["id"], row["run_id"])
                st.success("리뷰를 삭제했습니다.")
                st.rerun()
            except Exception as exc:
                st.error(f"리뷰 삭제 실패: {exc}")


def render_label_review_tab(supabase):
    """라벨 기준 + 라벨링된 오류 모아보기.

    1) 라벨 기준 표 — 사용자에게 어떤 오류가 어떤 의미인지 설명
    2) dataset 안의 모든 run에 걸친 라벨링 오류를 통합해서 보여줌
    3) 오류 유형/run/검색어로 필터 + CSV 다운로드
    """
    st.subheader("라벨 기준")
    guide_df = pd.DataFrame(LABEL_GUIDE)
    st.dataframe(
        # 표시할 때만 한국어 컬럼명으로 변경 — 원본 데이터는 영문 키 유지.
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
    # 작업 탭과 다른 key를 주어 두 selectbox가 독립적으로 동작하게 함.
    selected_dataset_label = st.selectbox("데이터셋", list(dataset_labels.keys()), key="review_dataset")
    dataset_id = dataset_labels[selected_dataset_label]
    rows_df = load_rows(supabase, dataset_id)
    runs = load_runs(supabase, dataset_id)
    if rows_df.empty or not runs:
        st.info("이 데이터셋에는 아직 리뷰할 번역 run이 없습니다.")
        return

    # 모든 run의 annotation을 통합 (No Error 제외).
    review_df = build_review_frame(supabase, dataset_id, rows_df, runs)
    if review_df.empty:
        st.info("아직 No Error 외의 라벨링된 오류가 없습니다.")
        return

    # 오류 유형별 건수 표 — 빈도순 내림차순, 동률이면 알파벳 순.
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

    # 3-컬럼 필터: 오류 유형 / run / 검색어
    filter_cols = st.columns([1, 1, 2])
    selected_error = filter_cols[0].selectbox(
        "오류 유형",
        ["All"] + [error for error in ERROR_TYPES if error != "No Error"],
        key="review_error_filter",
    )
    # "All" 선택지를 dict 첫 키로 둬서 기본값으로 자연스럽게 잡힘.
    run_options = {
        "All": None,
        **{
            f"{run.get('created_at', '')[:19]} / {run.get('model', '')} / {run['run_id'][:8]}": run["run_id"]
            for run in runs
        },
    }
    selected_run_label = filter_cols[1].selectbox("번역 run", list(run_options.keys()), key="review_run_filter")
    review_query = filter_cols[2].text_input("원문 / 번역 / 메모 검색", key="review_query")

    # 필터 누적 적용.
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
    # 리뷰 탭은 작업 탭과 컬럼 순서가 약간 다름 — error_type / memo / reviewer가 가장 앞.
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
            if "review_visible_count" not in st.session_state:
                st.session_state["review_visible_count"] = 10

            visible_count = min(st.session_state["review_visible_count"], len(filtered))
            visible_df = filtered.head(visible_count)

            for _, row in visible_df.iterrows():
                render_review_card(supabase, row)

            remaining = len(filtered) - visible_count
            if remaining > 0:
                if st.button(f"+{min(10, remaining)} 더 보기", key="review_load_more"):
                    st.session_state["review_visible_count"] += 10
                    st.rerun()

    st.download_button(
        "필터된 오류 리뷰 CSV 다운로드",
        data=csv_download_bytes(filtered),
        file_name=f"{selected_dataset_label.split(' / ')[0]}_error_reviews.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )
