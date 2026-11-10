# multimodal-rag

PDF의 텍스트·표·이미지를 분리 추출하고, 요약으로 검색 → 원본으로 답변하는 멀티모달 RAG 실습.

## 사전 준비 (macOS)

```bash
brew install poppler tesseract uv
pdftoppm -h    # Poppler 확인
tesseract -v   # Tesseract 확인
```

## 실행 방법

```bash
git clone <저장소 주소>
cd multimodal-rag

# 가상환경 (Python 3.11 — pyenv 버전 사용)
# uv가 받는 3.12 빌드는 이 Mac에서 서명 불일치로 SIGKILL(137) 되어 사용하지 않음
uv venv --python ~/.pyenv/versions/3.11.16/bin/python3.11 .venv
source .venv/bin/activate

# 의존성 설치
uv pip install -r requirements.txt        # 또는 requirements.lock.txt

# 커널 등록
python -m ipykernel install --user --name multimodal-rag --display-name "Python (multimodal-rag)"

# .env 작성 후 실제 API 키 입력
cp .env.example .env
```

노트북: `notebooks/Multi_modal_RAG.ipynb` → 커널 `Python (multimodal-rag)` 선택.

## 의존성 파일

- `requirements.txt` → 사람이 읽고 관리하는 목록
- `requirements.lock.txt` → 기계가 똑같이 재현하는 목록 (`uv pip freeze > requirements.lock.txt`)
