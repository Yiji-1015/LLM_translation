# LLM Translation QA Workspace

LLM Translation QA Workspace는 게임/콘텐츠 번역 데이터에서 **LLM 번역 결과와 사람 번역을 나란히 비교하고, 오류 유형을 관찰/라벨링하기 위한 연구용 Streamlit 웹앱**입니다.

CSV로 정리된 원문과 사람 번역을 업로드한 뒤 OpenAI 모델로 번역 run을 생성하고, 각 문장 단위로 오류 유형, 메모, 리뷰어를 기록할 수 있습니다. Supabase(Postgres)를 사용해 데이터셋, 번역 run, 프롬프트 버전, 라벨링 결과를 분리 저장하므로 여러 사람이 같은 데이터셋을 기준으로 번역 품질 이슈를 함께 살펴볼 수 있습니다.

이 프로젝트는 논문/스터디/프로토타입 단계에서 번역 품질 이슈를 빠르게 모으고, 프롬프트나 모델 설정을 바꿔가며 결과를 비교하기 위해 만든 MVP 성격의 도구입니다.

## 데모

배포 링크:

[LLM Translation QA Workspace 바로가기](https://llmtranslation-ggjybjubxywzhbtvkmywct.streamlit.app/)

### 1. CSV 업로드와 데이터셋 생성

CSV 컬럼 예시를 확인하고, 게임/콘텐츠 번역 데이터를 데이터셋 단위로 업로드합니다. 업로드된 데이터는 앱 안에서 바로 미리보기할 수 있습니다.
<img width="2557" height="1302" alt="image" src="https://github.com/user-attachments/assets/c8313728-08c5-4d47-80c8-4becbeffca04" />



### 2. 번역 비교와 공동 라벨링

각 row에서 원문, LLM 번역, 사람 번역을 나란히 보고 오류 유형, 메모, 리뷰어를 기록합니다. 라벨링 결과는 저장 즉시 Supabase에 반영됩니다.
<img width="2120" height="1292" alt="image" src="https://github.com/user-attachments/assets/37daecc4-f51b-44d2-be16-5249e06dcd63" />



### 3. 라벨 기준과 오류 리뷰 모아보기

오류 유형별 기준을 확인하고, 라벨링된 사례를 모아 검색/필터링할 수 있습니다. 여러 번역 run에서 나온 오류를 비교하며 반복되는 실패 양상을 살펴보는 데 사용합니다.
<img width="2127" height="1211" alt="image" src="https://github.com/user-attachments/assets/63413234-3f3a-408c-ab35-e6a8792607ef" />



## 기획 의도

게임 번역에서는 단순히 문장이 맞는지뿐 아니라 용어 일관성, 캐릭터 말투, UI 제약, 태그/플레이스홀더 보존, 문화권 표현 등 여러 요소가 함께 작동합니다. LLM 번역 결과를 평가할 때도 이런 오류가 어디에서 발생하는지, 어떤 프롬프트나 모델 설정에서 반복되는지 관찰할 수 있는 작은 실험 환경이 필요했습니다.

이 앱은 완성형 평가 플랫폼이라기보다, 번역 예시를 올리고 여러 번의 번역 시도를 남기며, 사람이 직접 오류를 읽고 라벨링하면서 실패 패턴을 모으는 데 초점을 둡니다.

## 주요 기능

- CSV 기반 데이터셋 업로드
- OpenAI 모델을 사용한 LLM 번역 생성
- 프롬프트 버전 관리
- 번역 run 단위 결과 보존 및 삭제
- LLM 번역, 사람 번역, 원문을 나란히 비교
- 오류 유형, 메모, 리뷰어 라벨링
- 라벨링 결과 즉시 Supabase DB 저장
- 카드뷰 / 테이블뷰 전환
- tag 필터와 검색
- 라벨 기준 설명 및 오류 리뷰 모아보기
- 데이터셋 및 결과 CSV 다운로드
- 결과 CSV를 다시 업로드해 이어서 작업

## 기술 스택

- Python
- Streamlit
- Supabase(Postgres)
- pandas
- OpenAI Python SDK

## 프로젝트 구조

```text
.
├── app.py                  # Streamlit 진입점
├── data.py                 # CSV 정규화, 샘플 데이터, 상수
├── db.py                   # Supabase CRUD
├── translate.py            # OpenAI 호출 및 번역 실행
├── views.py                # Streamlit UI
├── requirements.txt
├── examples/               # 예시 CSV
└── LLM_Translation-pseudo_lab/
```

## 사용 흐름

```text
입력 CSV 업로드
→ 데이터셋 생성
→ 프롬프트 버전 선택
→ LLM 번역 run 생성
→ 오류 유형 / 메모 / 리뷰어 라벨링
→ 결과 CSV 다운로드
→ 필요하면 결과 CSV를 다시 업로드해 이어서 작업
```

## 입력 CSV 형식

표준 입력은 CSV만 지원합니다. 게임 클라이언트에서 추출한 JSON 등은 사전에 CSV로 전처리한 뒤 업로드하는 것을 전제로 합니다.

필수 컬럼:

```text
source
human_translation
```

선택 컬럼:

```text
id
tag
context
speaker
listener
notes
```

기본 처리:

- `id`가 없으면 `row_0001` 형식으로 자동 생성
- `tag`가 없으면 `Uncategorized`
- 나머지 선택 컬럼이 없으면 빈 문자열
- 업로드는 최대 300행
- `context`는 프롬프트에 최대 1000자까지만 사용
- `source + human_translation` 기반 hash를 만들어 중복 확인에 활용

이미 아래 컬럼이 포함된 CSV를 다시 업로드하면 기존 값을 보존해 이어서 작업할 수 있습니다.

```text
llm_translation
error_type
memo
reviewer
```

## 출력 CSV

다운로드 CSV는 UTF-8-SIG 인코딩으로 생성됩니다.

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

## 오류 라벨

현재 구현에서 제공하는 기본 오류 유형은 다음과 같습니다. 라벨 체계는 연구 목적이나 데이터셋 성격에 따라 달라질 수 있으며, 향후 실험 방향에 맞춰 수정될 수 있습니다.

- `No Error`
- `Terminology`
- `Accuracy`
- `Style`
- `Persona/Pragmatics`
- `Locale`
- `Design/Markup`
- `Linguistic`
- `Other`

라벨 기준 탭에서는 각 오류 유형의 의미를 확인하고, 라벨링된 오류 리뷰를 모아서 볼 수 있습니다.

## 기본 프롬프트

```text
You are a professional English-to-Korean game localization translator.
Translate the source text into natural Korean.
Preserve placeholders, tags, variables, numbers, and line breaks exactly.
Use the given tag and context only as guidance.
Do not add explanations.

Tag: {tag}
Context: {context}
Source: {source}

Return only the Korean translation.
```

프롬프트를 수정하면 기존 프롬프트를 덮어쓰지 않고 새 prompt version으로 저장합니다. 따라서 이전 run 결과와 새 run 결과를 분리해서 비교할 수 있습니다.

## 데이터 모델

원본 row 데이터와 LLM 번역 결과를 분리해서 저장합니다. 덕분에 같은 데이터셋에 대해 여러 프롬프트/모델 조합을 실행하고, run별 결과를 비교할 수 있습니다.

- `datasets`: 업로드된 CSV 단위
- `rows`: 원문, 사람 번역, context, tag 등 원본 row
- `prompt_versions`: 프롬프트 버전
- `translation_runs`: 특정 프롬프트와 모델로 실행한 번역 run
- `translations`: run별 LLM 번역 결과
- `annotations`: run별 오류 라벨, 메모, 리뷰어

## 참고

이 저장소에는 앱 코드 외에도 같은 주제의 실험/스터디 산출물이 일부 포함되어 있습니다. Streamlit 앱 실행에 직접 관련된 핵심 파일은 `app.py`, `data.py`, `db.py`, `translate.py`, `views.py`, `requirements.txt`입니다.
