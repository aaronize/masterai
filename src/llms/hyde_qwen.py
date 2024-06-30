"""
HyDE

Reference: https://github.com/AIAnytime/HyDE-based-RAG-using-NVIDIA-NIM/blob/main/nvidia_nim.py
"""
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.env import QWEN_API_KEY
import os

os.environ['DASHSCOPE_API_KEY'] = QWEN_API_KEY

# HyDE prompt
hyde_template = '''Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question:

{question}'''

#
question_template = '''Answer the question based only on the following context:
{context}

Question: {question}'''

# 初始化大模型
llm = Tongyi(model_name="qwen-max")

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()

embeddings = DashScopeEmbeddings()

# 分词
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# hyde
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_query_transformer = hyde_prompt | llm | StrOutputParser()


@chain
def hyde_retriever(question: str):
    """"""
    hypo_doc = hyde_query_transformer.invoke({'question': question})
    return retriever.invoke(hypo_doc)


prompt = ChatPromptTemplate.from_template(question_template)
question_chain = prompt | llm | StrOutputParser()


@chain
def final_chain(question: str):
    """"""
    hyde_docs = hyde_retriever.invoke(question)
    for s in question_chain.stream({"question": question, "context": hyde_docs}):
        yield s


question = "how can langsmith help with testing"

print("====== rag =====")
for s in question_chain.stream({"question": question, "context": retriever.invoke(question)}):
    print(s, end="")


print("====== hyde =====")
for s in final_chain.stream(question):
    print(s, end="")


if __name__ == '__main__':
    pass
