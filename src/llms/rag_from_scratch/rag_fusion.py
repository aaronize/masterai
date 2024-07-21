"""

Reference: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
"""
from operator import itemgetter

import bs4
from langchain_core.load import dumps, loads
from typing import List, Dict, Tuple, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.llms.rag_from_scratch.vector_store import get_retriever
from src.llms.rag_from_scratch.prompts import RAG_TEMPLATE, RAG_FUSION_TEMPLATE


retriever = get_retriever()

llm = ChatOpenAI(temperature=0).bind(stop="SOLUTION")

question = "What is task decomposition for LLM agents?"


prompt_rag_fusion = ChatPromptTemplate.from_template(RAG_FUSION_TEMPLATE)


def generate_queries():
    """
    根据用户输入query，生成4个相关queries。此时使用了一个独立的llm实例
    :return:
    """
    return (
            prompt_rag_fusion |
            ChatOpenAI(temperature=0) |
            StrOutputParser() |
            (lambda x: x.split("\n"))
    )


def reciprocal_rank_fusion(results: List[list], k: int = 60):
    """
    Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    :param results:
    :param k:
    :return:
    """

    # init fused scores
    fused_scores: Dict[str, int] = dict()

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results: List[Tuple[str | Any, int]] = [
        (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


retrieval_chain_rag_fusion = generate_queries() | retriever.map() | reciprocal_rank_fusion

docs = retrieval_chain_rag_fusion.invoke({"question": question})
print(">>> len docs:", len(docs))


prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")} |
    prompt |
    llm |
    StrOutputParser()
)


if __name__ == '__main__':
    # res = final_rag_chain.invoke({"question": question})
    # print(res)

    from langchain_core.runnables import RunnableParallel
    # temp = ({"context": retrieval_chain_rag_fusion, "question": itemgetter("question")} | prompt)
    temp = RunnableParallel({"context": retrieval_chain_rag_fusion, "question": itemgetter("question")})
    res = temp.invoke({"question": question})
    print(res)

