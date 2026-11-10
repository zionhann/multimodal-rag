from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.documents.elements import CompositeElement, Table
from unstructured.partition.pdf import partition_pdf

from src.settings import FIGURES_DIR


def extract_pdf_elements(pdf_path, figures_dir=FIGURES_DIR):
    """PDF 에서 텍스트·표·이미지 요소를 추출하고 이미지는 파일로 저장합니다.

    Args:
        pdf_path: 추출할 PDF 파일 경로.
        figures_dir: 추출한 이미지를 저장할 폴더. 기본값은 FIGURES_DIR.

    Returns:
        list[Element]: unstructured 가 추출한 요소 리스트 (타이틀 기준으로 청킹됨).
    """
    return partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",                # 레이아웃 분석 모델로 정밀 추출
        extract_images_in_pdf=True,       # PDF에서 이미지를 추출
        infer_table_structure=True,       # 테이블 구조를 추론
        chunking_strategy="by_title",     # 타이틀 기준으로 텍스트 분할
        max_characters=4000,              # 최대 4000자로 블록 제한
        new_after_n_chars=3800,           # 3800자 이후 새 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하는 결합
        extract_image_block_output_dir=str(figures_dir),
    )


def categorize_elements(raw_pdf_elements):
    """추출된 요소를 텍스트와 표로 분류합니다.

    TableChunk 는 Table 의 하위 클래스라 표로 함께 분류됩니다.

    Args:
        raw_pdf_elements: extract_pdf_elements() 가 반환한 요소 리스트.

    Returns:
        tuple[list[str], list[str]]: (texts, tables) 문자열 리스트.
    """
    texts, tables = [], []
    for element in raw_pdf_elements:
        if isinstance(element, Table):
            tables.append(str(element))
        elif isinstance(element, CompositeElement):
            texts.append(str(element))
    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트를 하나로 이어 붙인 뒤 tiktoken 토큰 기준으로 다시 자릅니다.

    Args:
        texts: 분할할 텍스트 리스트.
        chunk_size: 청크당 최대 토큰 수.
        chunk_overlap: 인접 청크 간 겹치는 토큰 수.

    Returns:
        list[str]: 토큰 기준으로 분할된 텍스트 청크 리스트.
    """
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(" ".join(texts))


def extract(pdf_path, figures_dir=FIGURES_DIR):
    """PDF 한 개를 추출·분류·분할까지 한 번에 처리합니다.

    Args:
        pdf_path: 처리할 PDF 파일 경로.
        figures_dir: 추출한 이미지를 저장할 폴더. 기본값은 FIGURES_DIR.

    Returns:
        dict: {"texts": 원본 텍스트 블록, "tables": 표 문자열,
        "texts_2k_token": 2000 토큰 단위로 다시 자른 텍스트}.

    Raises:
        FileNotFoundError: pdf_path 가 존재하지 않을 때.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    texts, tables = categorize_elements(extract_pdf_elements(pdf_path, figures_dir))
    return {"texts": texts, "tables": tables, "texts_2k_token": split_texts(texts)}
