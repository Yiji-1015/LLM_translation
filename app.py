import streamlit as st

from data import APP_TITLE
from db import ensure_default_prompt, get_supabase_client
from views import (
    render_label_review_tab,
    render_prompt_tab,
    render_upload_tab,
    render_work_tab,
    sidebar_openai_key,
)


st.set_page_config(page_title=APP_TITLE, layout="wide")


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
