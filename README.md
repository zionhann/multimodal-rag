# Multimodal RAG

PDF 안의 텍스트·표·이미지를 분리해 추출하고, 요약문으로 검색한 뒤 원문을 바탕으로 답변하는 멀티모달 RAG 파이프라인 실습 자료입니다.

---

## 1. 프로젝트 개요

일반적인 RAG는 PDF를 텍스트로만 변환하기 때문에 표의 구조와 그림 속 정보가 사라집니다.

이 프로젝트는 문서를 구성하는 요소를 종류별로 나누어 처리하고, 이미지까지 답변 근거로 사용할 수 있는 검색 체인을 구성하는 것을 목표로 했습니다.

| 항목 | 내용 |
|------|------|
| 입력 | 텍스트·표·이미지가 섞여 있는 PDF |
| 출력 | 질문에 대한 한국어 답변 (이미지와 텍스트를 함께 근거로 사용) |
| 핵심 기법 | Multi-Vector Retriever (요약 임베딩 ↔ 원본 반환) |
| 사용 모델 | `gpt-4o-mini` (요약·답변), `text-embedding-3-small` (임베딩) |

## 2. 핵심 아이디어

표와 이미지는 그대로 임베딩하면 검색 품질이 좋지 않습니다.
그래서 **검색에 쓰는 데이터와 답변에 쓰는 데이터를 분리**했습니다.

- **검색용**: LLM이 만든 짧은 요약문을 임베딩해 벡터 DB(Chroma)에 저장합니다.
- **답변용**: 원본(텍스트·표 문자열·Base64 이미지)을 문서 저장소에 같은 `doc_id`로 보관합니다.
- 질문이 들어오면 요약문으로 유사도 검색을 하고, 일치한 `doc_id`의 **원본**을 멀티모달 LLM에 전달합니다.

## 3. 아키텍처

```text
            ┌──────────────── 추출 (extract.py) ────────────────┐
 PDF ──────▶│ unstructured(hi_res) → 텍스트 / 표 / 이미지 분리    │
            │ 텍스트는 tiktoken 2,000토큰 단위로 재분할           │
            └───────────────────────┬───────────────────────────┘
                                    ▼
            ┌──────────────── 요약 (summarize.py) ──────────────┐
            │ 텍스트·표 → LLM 요약   /   이미지 → VLM 요약        │
            └───────────────────────┬───────────────────────────┘
                                    ▼
            ┌──────────────── 저장 (retriever.py) ──────────────┐
            │ 요약 → 임베딩 → Chroma (vectorstore)                │
            │ 원본 → InMemoryStore (docstore)   ※ 같은 doc_id     │
            └───────────────────────┬───────────────────────────┘
                                    ▼
 질문 ─▶ 요약 검색 ─▶ 원본 조회 ─▶ 이미지/텍스트 분리 ─▶ 멀티모달 LLM ─▶ 답변
```

## 4. 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| 언어 | Python 3.11 |
| PDF 파싱 | unstructured (`hi_res`), Poppler, Tesseract |
| RAG 프레임워크 | LangChain (MultiVectorRetriever, LCEL) |
| 벡터 DB | Chroma |
| 모델 | OpenAI `gpt-4o-mini`, `text-embedding-3-small` |
| 환경 관리 | venv, pip, python-dotenv |
| 실습 환경 | JupyterLab |

## 5. 구현 내용

### 5.1 요소별 추출 — [src/extract.py](src/extract.py)

- `partition_pdf`의 `hi_res` 전략으로 레이아웃을 분석하고, 표 구조를 추론하며, 이미지를 파일로 저장합니다.
- 제목 기준(`by_title`)으로 블록을 나눈 뒤 `Table`과 `CompositeElement`를 구분해 표와 텍스트로 분류합니다.
- 텍스트는 tiktoken 기준 2,000토큰(겹침 200토큰)으로 다시 잘라 임베딩 입력 크기를 맞춥니다.

### 5.2 요약 생성 — [src/summarize.py](src/summarize.py)

- 텍스트와 표는 LangChain 체인으로 요약하며, `max_concurrency=3`으로 병렬 처리합니다.
- 이미지는 Base64로 인코딩해 VLM에 전달하고 검색용 한국어 요약을 받습니다.
- 요약 프롬프트에 "임베딩되어 원본을 검색하는 데 쓰인다"는 목적을 명시해, 검색에 적합한 간결한 요약이 나오도록 했습니다.

### 5.3 멀티 벡터 검색 — [src/retriever.py](src/retriever.py)

- 요약은 Chroma에, 원본은 `InMemoryStore`에 같은 `doc_id`로 저장합니다.
- 요약과 원본의 개수가 다르면 `ValueError`를 발생시킵니다. 개수가 어긋나면 엉뚱한 원본이 검색되기 때문입니다.
- 컬렉션을 매번 초기화해, 같은 커널에서 다시 실행해도 문서가 중복 저장되지 않습니다.

### 5.4 답변 체인 — [src/retriever.py](src/retriever.py)

- 검색 결과를 Base64 정규식으로 이미지와 텍스트로 나눕니다.
- 이미지를 먼저, 질문과 참고 텍스트를 뒤에 두어 하나의 `HumanMessage`로 조립합니다.
- "자료에 없는 내용은 모른다고 답하세요"라는 지시로 근거 밖 답변을 억제합니다.

### 5.5 환경 점검과 캐시 — [src/settings.py](src/settings.py)

- `check_env()`로 Poppler·Tesseract·파이썬 패키지·API 키 설정 여부를 한 번에 확인합니다. API 키는 일부만 출력합니다.
- `cached()`로 추출·요약 결과를 `data/processed/*.pkl`에 저장해, 비용이 드는 PDF 파싱과 LLM 호출을 반복하지 않습니다.

## 6. 설계 시 고려한 점

| 고려 사항 | 대응 |
|-----------|------|
| 표·이미지는 직접 임베딩하면 검색이 부정확함 | 요약문으로 검색하고 원본으로 답변하는 구조 채택 |
| 요약과 원본의 순서가 어긋날 위험 | 개수 검증 후 동일 `doc_id`로 저장 |
| LLM·PDF 파싱의 반복 비용 | 피클 캐시로 재사용 |
| 근거 없는 답변(환각) | 프롬프트로 "자료에 없으면 모른다고 답변" 지시, `temperature=0` |
| API 키 노출 | `.env`와 `.env.example` 분리, 키 일부만 출력 |
| 환경 재현성 | `requirements.txt`(관리용)와 `requirements.lock.txt`(재현용) 이원화 |

## 7. 프로젝트 구조

```text
multimodal-rag/
├── src/
│   ├── settings.py
│   ├── extract.py
│   ├── summarize.py
│   └── retriever.py
├── notebooks/
│   └── Multi_modal_RAG.ipynb
├── data/
│   └── sample_text.pdf
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── requirements.lock.txt
├── .env.example
├── .gitignore
└── README.md
```

| 구분 | 파일 | 설명 |
|------|------|------|
| Python | `src/settings.py` | 경로·모델명 상수, 환경 점검(`check_env`), 피클 캐시(`cached`)를 담당합니다. |
| Python | `src/extract.py` | PDF에서 텍스트·표·이미지를 추출하고 토큰 단위로 재분할합니다. |
| Python | `src/summarize.py` | 텍스트·표는 LLM으로, 이미지는 VLM으로 검색용 한국어 요약을 만듭니다. |
| Python | `src/retriever.py` | 요약은 Chroma에, 원본은 docstore에 저장하는 멀티 벡터 검색기와 RAG 체인을 만듭니다. |
| 노트북 | `notebooks/Multi_modal_RAG.ipynb` | `src/` 모듈을 호출해 추출 → 요약 → 검색까지 실행하는 통합 노트북입니다. |
| 데이터 | `data/sample_text.pdf` | 예제 입력 PDF입니다. |
| 의존성 | `requirements.txt` | 직접 사용하는 패키지 목록입니다. |
| 의존성 | `requirements.lock.txt` | 전체 버전을 고정한 재현용 목록입니다. |
| 의존성 | `pyproject.toml`, `uv.lock` | uv로 설치할 때만 사용하는 프로젝트 선언과 잠금 파일입니다. |
| 설정 | `.env.example` | `OPENAI_API_KEY` 입력용 템플릿입니다. `.env`로 복사해 사용합니다. |
| 설정 | `.gitignore` | `.venv/`, `.env`, `data/figures/`, `chroma_db/` 등을 제외합니다. |
| 문서 | `README.md` | 프로젝트 소개와 실행 방법입니다. |

노트북을 실행하면 추출된 이미지가 `data/figures/`에, 추출·요약 캐시가 `data/processed/`에 생성됩니다. 두 폴더는 실행 결과물이라 저장소에 포함하지 않습니다.

## 8. 실행 방법

Python 3.11 이상이 필요합니다.

### 사전 준비 (macOS)

PDF 파싱에 쓰는 OS 패키지는 pip로 설치되지 않으므로 먼저 설치합니다.

```bash
brew install poppler tesseract
pdftoppm -h    # Poppler 확인
tesseract -v   # Tesseract 확인
```

### 설치

```bash
git clone <저장소 주소>
cd multimodal-rag

# 가상환경 생성과 활성화
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt          # 직접 쓰는 패키지만, 버전은 최신
# 또는
pip install -r requirements.lock.txt     # 전체 버전 고정, 환경을 그대로 재현

# 커널 등록
python -m ipykernel install --user --name multimodal-rag --display-name "Python (multimodal-rag)"

# .env 작성 후 실제 API 키 입력
cp .env.example .env
```

### 실행

```bash
jupyter lab
```

`notebooks/Multi_modal_RAG.ipynb`를 열고 커널 `Python (multimodal-rag)`을 선택한 뒤 위에서부터 실행합니다.

### 의존성 파일

- `requirements.txt`: 사람이 읽고 관리하는 목록입니다.
- `requirements.lock.txt`: 같은 환경을 그대로 재현하기 위한 목록입니다. macOS 환경에서 만들었으므로 다른 OS에서 설치가 실패하면 `requirements.txt`를 사용합니다.

## 9. 실습 범위와 한계점

실습 목적의 구현이라 아래 범위까지만 다룹니다.

- **저장소가 프로세스 안에만 존재합니다.** 원본은 `InMemoryStore`에, 벡터는 Chroma 기본 설정(메모리)에 저장하므로 커널을 종료하면 사라집니다.
- **이미지 요약은 한 장씩 호출합니다.** 텍스트·표와 달리 병렬 처리를 하지 않아 이미지가 많으면 시간이 오래 걸립니다.
- **이미지와 텍스트를 Base64 정규식으로 구분합니다.** 메타데이터에 타입을 기록하지 않고 내용의 패턴으로 판별합니다.
- **캐시를 pickle 파일로 저장합니다.** 이 프로젝트가 만든 로컬 파일만 읽는 것을 전제로 합니다.
- **정량 평가는 포함하지 않았습니다.** 검색 정확도와 답변 품질은 노트북의 검색 테스트로 직접 확인하는 수준입니다.
