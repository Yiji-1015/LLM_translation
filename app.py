"""
앱의 엔트리포인트.

전체 구조:
- data.py      : 순수 데이터/CSV 처리 (의존성 없음)
- db.py        : Supabase CRUD (data.py에 의존)
- translate.py : OpenAI 호출 + 병렬 번역
- views.py    : Streamlit UI 렌더링
- app.py      : 위 모듈들을 묶는 진입점

`streamlit run app.py`로 실행하면 Streamlit이 이 파일을 위에서부터 아래까지
매번 재실행합니다. 그래서 상태를 보존하려면 `st.session_state`나
`@st.cache_resource` 같은 메커니즘이 필요합니다.
"""

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


# set_page_config는 다른 st.* 호출보다 먼저 와야 한다는 Streamlit 규칙이 있어
# main() 안이 아니라 모듈 최상단에서 호출합니다.
st.set_page_config(page_title=APP_TITLE, layout="wide")


def main():
    st.title(APP_TITLE)
    st.caption("LLM 번역 결과를 비교하고 MQM 기반으로 함께 리뷰/라벨링하기 위한 연구용 MVP")

    # Supabase 클라이언트는 @st.cache_resource로 캐시되므로 매 rerun마다 새로 만들어지지 않음.
    supabase = get_supabase_client()
    # DB에 기본 프롬프트가 없으면 한 번 만들어 둠 (멱등).
    ensure_default_prompt(supabase)
    # OpenAI 키 입력란을 사이드바에 띄움. 키는 session_state에만 머무름.
    sidebar_openai_key()

    # 탭 라벨 순서를 바꾸면 사용자 워크플로우 순서(업로드→작업→...)가 바뀌니 주의.
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
