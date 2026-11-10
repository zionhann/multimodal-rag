import re
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:  # langchain 1.x
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except ImportError:  # langchain 0.3
    from langchain.retrievers.multi_vector import MultiVectorRetriever

from src.settings import EMBEDDING_MODEL, LLM_MODEL

ID_KEY = "doc_id"
BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")
ANSWER_INSTRUCTION = "아래 자료를 근거로 질문에 한국어로 답하세요. 자료에 없는 내용은 모른다고 답하세요."


def add_documents(retriever, doc_summaries, doc_contents):
    """요약은 vectorstore 에, 원본은 docstore 에 같은 doc_id 로 저장합니다.

    Args:
        retriever: 문서를 추가할 MultiVectorRetriever.
        doc_summaries: 임베딩할 요약 문자열 리스트.
        doc_contents: 요약과 같은 순서의 원본 리스트 (텍스트·표·Base64 이미지).

    Returns:
        None

    Raises:
        ValueError: 요약과 원본의 개수가 다를 때.
    """
    if len(doc_summaries) != len(doc_contents):
        # 개수가 다르면 요약 ↔ 원본이 엇갈려 엉뚱한 원본이 검색됩니다
        raise ValueError(f"요약 {len(doc_summaries)}건 ≠ 원본 {len(doc_contents)}건 — 캐시를 다시 만드세요.")
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    retriever.vectorstore.add_documents(
        [Document(page_content=s, metadata={ID_KEY: i}) for s, i in zip(doc_summaries, doc_ids)]
    )
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def create_multi_vector_retriever(
    text_summaries, texts, table_summaries, tables, image_summaries, images,
    collection_name="multimodal_rag",
):
    """요약으로 검색하고 원본을 돌려주는 멀티 벡터 검색기를 만듭니다.

    각 (요약, 원본) 쌍은 비어 있으면 건너뜁니다.

    Args:
        text_summaries: 텍스트 요약 리스트.
        texts: 텍스트 원본 리스트.
        table_summaries: 표 요약 리스트.
        tables: 표 원본 리스트.
        image_summaries: 이미지 요약 리스트.
        images: Base64 이미지 원본 리스트.
        collection_name: Chroma 컬렉션 이름. 실행할 때마다 초기화됩니다.

    Returns:
        MultiVectorRetriever: 요약을 임베딩해 검색하고 원본을 반환하는 검색기.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    vectorstore.reset_collection()  # 같은 커널에서 다시 실행해도 중복 저장되지 않게
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore(), id_key=ID_KEY)

    for summaries, contents in [
        (text_summaries, texts),
        (table_summaries, tables),
        (image_summaries, images),
    ]:
        if summaries:
            add_documents(retriever, summaries, contents)
    return retriever


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다.

    Args:
        docs: 검색 결과 리스트 (Document 또는 문자열).

    Returns:
        dict: {"images": Base64 문자열 리스트, "texts": 텍스트 리스트}.
    """
    images, texts = [], []
    for doc in docs:
        content = str(doc.page_content if isinstance(doc, Document) else doc)
        (images if BASE64_RE.match(content) else texts).append(content)
    return {"images": images, "texts": texts}


def img_prompt_func(data_dict):
    """이미지, 지시문, 질문, 참고 텍스트를 HumanMessage 하나로 조립합니다.

    Args:
        data_dict: {"context": split_image_text_types() 결과, "question": 질문 문자열}.

    Returns:
        list[HumanMessage]: 이미지가 먼저 오고 텍스트가 이어지는 메시지 하나를 담은 리스트.
    """
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        for img in data_dict["context"]["images"]
    ]
    context_text = "\n".join(data_dict["context"]["texts"])
    content.append({
        "type": "text",
        "text": f"{ANSWER_INSTRUCTION}\n\n질문: {data_dict['question']}\n\n참고 텍스트:\n{context_text}",
    })
    return [HumanMessage(content=content)]


def multi_modal_rag_chain(retriever):
    """질문 → 검색 → 이미지/텍스트 분리 → 멀티모달 LLM 순서의 RAG 체인을 만듭니다.

    Args:
        retriever: create_multi_vector_retriever() 로 만든 검색기.

    Returns:
        Runnable: 질문 문자열을 받아 답변 문자열을 반환하는 체인.
    """
    model = ChatOpenAI(temperature=0, model=LLM_MODEL, max_tokens=1024)
    return (
        {"context": retriever | RunnableLambda(split_image_text_types),
         "question": RunnablePassthrough()}
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )