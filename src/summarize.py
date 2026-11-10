import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.settings import LLM_MODEL

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

TEXT_PROMPT = """당신은 표와 텍스트를 검색용으로 요약하는 도우미입니다.
이 요약은 임베딩되어 원본 표 또는 텍스트를 검색하는 데 사용됩니다.
검색에 최적화된 간결한 요약을 한국어로 작성하세요.
요약할 표 또는 텍스트: {element}"""

IMAGE_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the image that is well optimized for retrieval.
Write the summary in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트와 표를 LLM 으로 요약합니다.

    Args:
        texts: 텍스트 청크 리스트.
        tables: 표 문자열 리스트.
        summarize_texts: False 면 텍스트는 요약하지 않고 원문 그대로 씁니다.

    Returns:
        tuple[list[str], list[str]]: (text_summaries, table_summaries).
    """
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT)
    model = ChatOpenAI(temperature=0, model=LLM_MODEL)
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    if texts and summarize_texts:
        text_summaries = chain.batch(texts, {"max_concurrency": 3})
    else:
        text_summaries = list(texts)
    table_summaries = chain.batch(tables, {"max_concurrency": 3}) if tables else []
    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지 파일을 Base64 문자열로 인코딩합니다.

    Args:
        image_path: 이미지 파일 경로.

    Returns:
        str: Base64 로 인코딩한 이미지 문자열.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_summarize(img_base64, prompt=IMAGE_PROMPT):
    """VLM 으로 Base64 이미지 한 장을 요약합니다.

    Args:
        img_base64: Base64 로 인코딩한 이미지.
        prompt: 요약 지시 프롬프트. 기본값은 IMAGE_PROMPT.

    Returns:
        str: 이미지 요약 텍스트.
    """
    chat = ChatOpenAI(model=LLM_MODEL, max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ])
    ])
    return msg.content


def generate_img_summaries(figures_dir, prompt=IMAGE_PROMPT):
    """폴더 안의 이미지를 파일명 순으로 인코딩하고 요약합니다.

    Args:
        figures_dir: 이미지(.jpg/.jpeg/.png)가 들어 있는 폴더.
        prompt: 요약 지시 프롬프트. 기본값은 IMAGE_PROMPT.

    Returns:
        tuple[list[str], list[str]]: (img_base64_list, image_summaries). 두 리스트는 같은 순서로 짝을 이룹니다.
    """
    paths = sorted(p for p in Path(figures_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    img_base64_list = [encode_image(p) for p in paths]

    image_summaries = [image_summarize(b64, prompt) for b64 in img_base64_list]
    return img_base64_list, image_summaries


def summarize_all(extracted, figures_dir):
    """extract() 결과와 이미지 폴더로 텍스트·표·이미지 요약을 한 번에 만듭니다.

    Args:
        extracted: extract() 가 반환한 딕셔너리.
        figures_dir: 추출한 이미지가 저장된 폴더.

    Returns:
        dict: 02 노트북의 summaries.pkl 과 같은 형식 — "text_summaries",
        "table_summaries", "image_summaries", "img_base64_list".
    """
    text_summaries, table_summaries = generate_text_summaries(
        extracted["texts_2k_token"], extracted["tables"], summarize_texts=True
    )
    img_base64_list, image_summaries = generate_img_summaries(figures_dir)
    return {
        "text_summaries": text_summaries,
        "table_summaries": table_summaries,
        "image_summaries": image_summaries,
        "img_base64_list": img_base64_list,
    }
