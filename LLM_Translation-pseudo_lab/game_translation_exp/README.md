# Game Translation Experiment

게임 번역 품질 개선을 위한 실험 저장소입니다.
현재 목표는 아래 질문에 답하는 것입니다.

- `worldview/context` 기반 프롬프트가 실제 번역 품질을 얼마나 개선하는가?
- `glossary + style rules + sentence type`를 추가하면 어떤 축에서 개선되는가?
- 캐릭터/세력 관계 그래프(`relation KG`)를 넣으면 말투/톤/관계성 번역이 더 좋아지는가?

## 1) 실험 배경

초기 관찰:
- worldview 프롬프트는 세계관/톤/분위기 정렬에는 도움됨
- 하지만 실제 번역에서 아래 문제가 남음
  - 고유명사 처리 흔들림
  - 시스템/메커닉 용어 오역
  - UI 문구 스타일 부적합
  - 스킬/효과 설명 문장 어색함

가설:
- 분위기 정보만으로는 부족하고, 번역 제어 정보가 필요함
- 번역 제어 정보 = `glossary + translation rules + sentence type`

## 2) 프롬프트 실험 설계

A/B/C 비교 실험:
- A: worldview only
- B: glossary + translation rules
- C: glossary + translation rules + sentence type (`UI/perk/dialogue/lore`)

확장 실험:
- D: C + relation context (캐릭터/세력 관계 그래프 기반)

## 3) 데이터 전략 (신규 수집 없음)

원칙:
- 새 데이터 수집 없이 기존 샘플 재구성
- 샘플에 태그를 붙여 실험용으로 정리

현재 샘플셋:
- `data/samples_tagged_v1.jsonl`
- `data/samples_tagged_v1.csv`
- 총 80개
  - UI 24 / perk 24 / dialogue 16 / lore 16

원본 데이터:
- `/Users/yiji/Desktop/work/pseudo_lab/destiny2_translation.json`

## 4) 평가 프레임

평가 축(1~5점):
- meaning preservation
- terminology accuracy
- consistency
- naturalness
- game-UI appropriateness

보조 기록:
- best condition (A/B/C/D)
- error taxonomy (`eval/error_taxonomy.md`)

평가 시트:
- 기본: `eval/eval_sheet.csv`
- 실행 후 자동 채움: `eval/eval_sheet_prefilled_*.csv`

## 5) Relation KG 실험 컨셉 (Draft)

목표:
- 인물/세력 관계를 번역 컨텍스트로 주입해
  - 존댓말/반말
  - 적대/동맹 톤
  - 대사 자연스러움
  을 개선하는지 확인

파이프라인:
1. 엔티티 시드 준비 (`character_seeds.csv`)
2. 관계 후보 자동 추출 (`extract_relation_candidates.py`)
3. 정본 확정 (`relation_edges_confirmed.csv`)
   - 자동 추출 후보 + 외부 리서치 검증 정보를 통합
4. 샘플별 relation context 생성 (`build_relation_context.py`)
5. D 조건 프롬프트로 번역 (`run_condition_d.py`)

주의:
- `relation_edges_auto.csv`는 노이즈가 많음
- 발표/결론용 비교는 `relation_edges_confirmed.csv` 기준 권장

## 6) 실행 엔트리

A/B/C 실행:
- Notebook: `run_ab_experiment.ipynb`

Relation KG + D 실행:
- Notebook: `run_relation_kg_experiment.ipynb` (단일 정본 KG 실험 노트북)
- Guide: `kg/README_relation_kg.md`

환경 변수:
- 노트북은 프로젝트 루트의 `.env` (`LLM_Translation-pseudo_lab/.env`)만 로딩합니다.
- 필수 키: `OPENAI_API_KEY` (실제 번역 실행 시)

## 7) 주요 파일

프롬프트:
- `prompts/A_worldview.txt`
- `prompts/B_glossary_rules.txt`
- `prompts/C_glossary_rules_type.txt`
- `prompts/D_glossary_rules_type_relation.txt`

데이터:
- `data/samples.csv`
- `data/samples_tagged_v1.csv`
- `data/samples_tagged_v1.jsonl`
- `data/glossary.csv`
- `data/style_guide.md`

관계 그래프:
- `data/relation_kg/character_seeds.csv`
- `data/relation_kg/relation_candidates.csv`
- `data/relation_kg/relation_edges_auto.csv`
- `data/relation_kg/relation_edges_confirmed.csv`
- `data/relation_kg/sample_relation_context.csv`
- `data/relation_kg/relation_review_checklist.md`

결과/평가:
- `outputs/run_YYYY-MM-DD/`
- `eval/eval_sheet.csv`
- `eval/error_taxonomy.md`

## 8) 다음 단계 제안

1. C 대비 D의 이득이 큰 샘플 타입 파악 (특히 dialogue/lore)
2. confirmed relation edges 소규모(고신뢰)부터 확장
3. 이후 오픈소스 LLM QLoRA 미세튜닝으로 prompt-only 대비 성능 비교

## 9) Legacy (Archive)

아래는 분리 실험 단계에서 생성된 보조 파일이며, 현재 정식 비교에서는 사용하지 않습니다.

- `data/relation_kg_external/`
- `scripts/build_relation_context_external.py`
- `scripts/run_condition_e_external.py`
- `run_external_kg_experiment.ipynb` (Archived)
