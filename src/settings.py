import os
import pickle
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = DATA_DIR / "figures"
PROCESSED_DIR = DATA_DIR / "processed"

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv(PROJECT_ROOT / ".env")


def check_env():
    """OS 패키지·파이썬 패키지·API 키 설치 여부를 출력합니다.

    Returns:
        None

    Raises:
        ValueError: OPENAI_API_KEY 가 설정되지 않았을 때.
    """
    print("Python:", sys.version.split()[0])
    print("가상환경 경로:", sys.prefix)
    print("가상환경 안에서 실행 중?", ".venv" in sys.prefix)
    print()

    # OS 패키지 확인 (pip로 설치되지 않는 것들)
    print("Poppler  :", "OK" if shutil.which("pdftoppm") else "✗ 미설치 — brew install poppler")
    print("Tesseract:", "OK" if shutil.which("tesseract") else "✗ 미설치 — brew install tesseract")
    print()

    # 파이썬 패키지 확인
    for mod in ["langchain", "unstructured", "chromadb", "tiktoken", "PIL"]:
        try:
            __import__(mod)
            print(f"{mod:14s}: OK")
        except ImportError:
            print(f"{mod:14s}: ✗ 미설치")

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY 가 없습니다. .env 파일을 확인하세요.")
    # 키 전체는 출력하지 않습니다 (앞 7자 + 뒤 4자)
    print(f"\nAPI 키     : {key[:7]}...{key[-4:]}")


def cached(name, build):
    """PROCESSED_DIR/name 피클이 있으면 불러오고, 없으면 build() 결과를 저장 후 반환합니다.

    pickle 은 이 프로젝트가 직접 만든 로컬 캐시만 읽습니다. 외부에서 받은 .pkl 은 넣지 마세요.

    Args:
        name: PROCESSED_DIR 아래 캐시 파일 이름 (예: "summaries.pkl").
        build: 캐시가 없을 때 호출할 인자 없는 함수.

    Returns:
        Any: 캐시에서 불러온 값 또는 build() 의 결과.
    """
    path = PROCESSED_DIR / name
    if path.exists():
        print(f"캐시 사용: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    result = build()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"저장 완료: {path}")
    return result
