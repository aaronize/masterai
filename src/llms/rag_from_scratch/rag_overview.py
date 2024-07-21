"""

Reference: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb
"""
import numpy
import tiktoken
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.llms.rag_from_scratch.vector_store import get_retriever

retriever = get_retriever(chunk_size=1000, chunk_overlap=200)

prompt = hub.pull("rlm/rag-prompt")
"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

llm = ChatOpenAI(temperature=0)


def format_docs(docs):
    """
    拼接docs
    :param docs:
    :return:
    """
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# indexing
question = "What kinds of pets do I like?"
document = "My favourite pet is a cat."


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    string编码后的token数。
    :param string:
    :param encoding_name: 编码名称，如 cl100k_base等
    :return:
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
print("query len:", len(query_result))


def cosine_similarity(vec1, vec2):
    """
    计算相似度
    Cosine similarity is recommended (1 indicates identical) for OpenAI embeddings
    :param vec1:
    :param vec2:
    :return:
    """
    dot_product = numpy.dot(vec1, vec2)
    norm_vec1 = numpy.linalg.norm(vec1)
    norm_vec2 = numpy.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)


print("Cosine similarity:", cosine_similarity(query_result, document_result))


if __name__ == '__main__':
    res = rag_chain.invoke("What is Task Decomposition?")
    print(res)
