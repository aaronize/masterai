"""

Reference: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.llms.rag_from_scratch.prompts import RAG_HYDE_TEMPLATE, RAG_TEMPLATE
from src.llms.rag_from_scratch.vector_store import get_retriever

retriever = get_retriever()

prompt_hyde = ChatPromptTemplate.from_template(RAG_HYDE_TEMPLATE)

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)

# Run
question = "What is task decomposition for LLM agents?"
gen_docs = generate_docs_for_retrieval.invoke({"question": question})
print(">>> gen docs:", gen_docs)


# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever
retrieved_docs = retrieval_chain.invoke({"question": question})

print(">>> retrieved docs:", retrieved_docs)

prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    prompt |
    llm |
    StrOutputParser()
)


if __name__ == '__main__':
    res = final_rag_chain.invoke({"context": retrieved_docs, "question": question})
    print(res)
