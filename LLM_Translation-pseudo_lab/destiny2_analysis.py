"""
게임 번역 세계관 분석 & 어투 기반 번역 실험
- 영어 텍스트로 Word Cloud 생성
- GPT로 세계관 / 어투 분석
- 분석된 어투로 번역 실험
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

# ── 라이브러리 설치 안내 ──────────────────────────────────────────
# pip install wordcloud matplotlib openai nltk

# ─────────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────────

def load_data(filepath="destiny2_translation.json"):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"총 {len(data)}개 항목 로드 완료")
    return data


def detect_game_title(data):
    """JSON 파일명 대신 데이터 내부에서 게임 이름을 추론하지 않고,
    범용 분석을 위해 항상 '이 게임'으로 처리합니다."""
    return "이 게임"


def get_texts_by_type(data, types=None, domains=None):
    """원하는 type / domain 의 영어 텍스트만 추출"""
    result = []
    for d in data:
        if types and d.get("type") not in types:
            continue
        if domains and d.get("domain") not in domains:
            continue
        result.append(d["en"])
    return result


# ─────────────────────────────────────────────────────────────────
# 2. 텍스트 전처리 & Word Cloud
# ─────────────────────────────────────────────────────────────────

# 게임 공통 불용어 (일반 영어 불용어 + 게임 UI 잡음어)
DESTINY_STOPWORDS = {
    # 일반 영어 불용어
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "then", "once", "and", "but", "or", "nor", "so",
    "yet", "both", "either", "neither", "not", "no", "this", "that",
    "these", "those", "it", "its", "you", "your", "he", "she", "we",
    "they", "them", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "all", "each", "more", "most", "other",
    "up", "if", "about", "i", "my", "me", "his", "her", "our",
    "than", "too", "very", "just", "also", "any", "only",
    # Destiny 2 UI 잡음어
    "equip", "equipped", "equipping", "item", "items", "bonus", "use",
    "using", "used", "unlock", "unlocks", "unlocked", "reward", "rewards",
    "account", "character", "characters", "class", "get", "gives",
    "grant", "grants", "granted", "while", "when", "upon", "after",
    "increase", "increases", "increased", "reduce", "reduces",
    "stack", "stacks", "stacking",
}


def clean_text(texts):
    """텍스트 리스트 → 단어 리스트"""
    words = []
    for text in texts:
        # 소문자 변환, 특수문자 제거
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        for word in text.split():
            if word not in DESTINY_STOPWORDS and len(word) > 2:
                words.append(word)
    return words


def get_top_words(words, n=100):
    counter = Counter(words)
    return counter.most_common(n)


def make_wordcloud(words, output_path="wordcloud_game.png", title="게임 세계관 Word Cloud"):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    freq = Counter(words)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="black",
        colormap="plasma",   # 우주적 느낌
        max_words=200,
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18, color="white", pad=15,
              bbox=dict(facecolor="black", alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.show()
    print(f"Word Cloud 저장: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────
# 3. GPT로 세계관 분석
# ─────────────────────────────────────────────────────────────────

def analyze_worldview_with_gpt(top_words, lore_samples, api_key):
    """상위 단어 + lore 샘플 → GPT가 세계관/어투 분석"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # 상위 50개 단어
    word_list = ", ".join(w for w, _ in top_words[:50])

    # lore 샘플 3개 (긴 것 위주)
    samples_text = "\n---\n".join(lore_samples[:3])

    prompt = f"""당신은 게임 번역 전문가입니다. 아래는 어떤 게임의 텍스트 데이터입니다. 게임 이름은 알려드리지 않습니다.

## 자주 등장하는 상위 50개 단어:
{word_list}

## 게임 내 설명/로어 텍스트 샘플:
{samples_text}

위 데이터를 바탕으로 다음을 분석해 주세요:

1. **시대 배경 및 세계관** - 어떤 세계인가요? (SF, 판타지, 포스트아포칼립스 등) 구체적인 단서와 함께 설명해 주세요.
2. **서술 어조 및 분위기** - 내러티브의 톤은 어떤가요? (서사적, 비극적, 신화적, 군사적 등)
3. **핵심 테마** - 반복적으로 등장하는 주제는 무엇인가요? (예: 빛과 어둠, 죽음과 부활, 우주적 공포 등)
4. **문체 특징** - 문장 구조와 표현 방식의 특징을 설명해 주세요.
5. **한국어 번역 가이드라인** - 위 분석을 바탕으로 한국어 번역 시 지켜야 할 스타일을 제안해 주세요.
   - 경어 사용 여부 (존댓말/반말)
   - 어휘 선택 방향 (한자어 vs 순우리말 vs 영어 직역)
   - 어조 유지를 위한 팁

번역가가 실제로 활용할 수 있도록 구체적이고 간결하게 작성해 주세요. 답변은 한국어로 작성하세요."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    analysis = response.choices[0].message.content
    return analysis


# ─────────────────────────────────────────────────────────────────
# 4. 분석된 어투로 번역
# ─────────────────────────────────────────────────────────────────

def translate_with_style(texts_to_translate, worldview_analysis, api_key, n=5):
    """세계관 분석 결과를 system prompt로 넣어서 번역"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    system_prompt = f"""당신은 게임 로컬라이제이션 전문 한국어 번역가입니다.

아래 세계관 분석 결과를 반드시 참고하여 번역 스타일을 유지하세요:

{worldview_analysis}

번역 규칙:
- 고유명사(인물명, 지명, 아이템명 등)는 원문 표기를 최대한 살리세요.
- 위 분석에서 도출된 어조와 문체를 일관되게 유지하세요.
- 번역 결과만 출력하고, 설명이나 부가 텍스트는 절대 추가하지 마세요."""

    results = []
    for text in texts_to_translate[:n]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Translate to Korean:\n{text}"},
            ],
            temperature=0.2,
        )
        translated = response.choices[0].message.content.strip()
        results.append({"en": text, "gpt_ko": translated})

    return results


# ─────────────────────────────────────────────────────────────────
# 5. 결과 저장
# ─────────────────────────────────────────────────────────────────

def save_results(analysis, translations, output_path="game_analysis_results.json"):
    output = {
        "worldview_analysis": analysis,
        "translation_samples": translations,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {output_path}")


# ─────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────

def resolve_api_key(api_key=None, env_key="OPENAI_API_KEY", env_path=".env"):
    """Resolve API key from argument first, then .env/environment."""
    if api_key:
        return api_key

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except Exception:
        pass

    key = os.getenv(env_key)
    if not key:
        raise ValueError(f"{env_key} not found. Set it in environment or {env_path}.")
    return key


def run_pipeline(api_key=None, data_path="destiny2_translation.json", env_path=".env"):
    api_key = resolve_api_key(api_key=api_key, env_path=env_path)
    # 1. 데이터 로드
    data = load_data(data_path)

    # 2. Word Cloud용: lore + description 텍스트 (이름 제외)
    wc_texts = get_texts_by_type(
        data,
        types={"lore_description", "description", "lore_subtitle"},
    )
    print(f"Word Cloud용 텍스트: {len(wc_texts)}개")

    words = clean_text(wc_texts)
    top_words = get_top_words(words, n=100)

    # 3. Word Cloud 생성
    make_wordcloud(words)

    # 4. GPT 세계관 분석 - lore 텍스트 중 긴 것 3개 샘플
    lore_texts = get_texts_by_type(data, types={"lore_description"})
    lore_samples = sorted(lore_texts, key=len, reverse=True)[:3]

    print("\nGPT로 세계관 분석 중...")
    analysis = analyze_worldview_with_gpt(top_words, lore_samples, api_key)
    print("\n=== 세계관 분석 결과 ===")
    print(analysis)

    # 5. 분석된 어투로 번역 실험
    # 200~500자 사이의 description 중 5개 선택
    desc_texts = get_texts_by_type(data, types={"description", "lore_description"})
    test_texts = [t for t in desc_texts if 200 <= len(t) <= 500][:5]

    print("\nGPT로 번역 실험 중...")
    translations = translate_with_style(test_texts, analysis, api_key, n=5)

    print("\n=== 번역 결과 ===")
    for i, t in enumerate(translations, 1):
        print(f"\n[{i}] EN: {t['en'][:100]}...")
        print(f"    KO: {t['gpt_ko'][:100]}...")

    # 6. 저장
    save_results(analysis, translations)

    return analysis, translations, top_words
