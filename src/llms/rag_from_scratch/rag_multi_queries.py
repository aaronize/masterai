"""

Reference: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
"""
from operator import itemgetter
from typing import List

from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.llms.rag_from_scratch.prompts import RAG_MULTI_QUERY_TEMPLATE, RAG_TEMPLATE
from src.llms.rag_from_scratch.vector_store import get_retriever


retriever = get_retriever()

prompt_perspectives = ChatPromptTemplate.from_template(RAG_MULTI_QUERY_TEMPLATE)

generate_queries = (
    prompt_perspectives |
    ChatOpenAI(temperature=0) |
    StrOutputParser() |
    (lambda x: x.split("\n"))
)


def get_unique_union(documents: List[List[str]]):
    """

    :param documents:
    :return:
    """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))

    return [loads(doc) for doc in unique_docs]


# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question": question})
print("docs:", len(docs))

prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")} |
    prompt |
    llm |
    StrOutputParser()
)


if __name__ == '__main__':
    # res = retrieval_chain.invoke({"question": question})
    # print(len())
    res = final_rag_chain.invoke({"question": question})
    print(res)
