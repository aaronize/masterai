"""
RAG with ReRanking optimize

Reference：https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22
"""

import os

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.env import QWEN_API_KEY, COHERE_API_KEY
os.environ['DASHSCOPE_API_KEY'] = QWEN_API_KEY

os.environ["COHERE_API_KEY"] = COHERE_API_KEY


dataset_path = '/Users/aaron/Workspace/masterai/dataset/rag_reranking/'
documents = []
for filename in os.listdir(dataset_path):
    loader = TextLoader(dataset_path + filename)
    documents.extend(loader.load())

print(documents[:3])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
text_splits = text_splitter.split_documents(documents)
print(len(text_splits))

# embedding model
embeddings = DashScopeEmbeddings()
vec_store = FAISS.from_documents(text_splits, embeddings)


# hybrid search with ensemble retrieval
retriever = vec_store.as_retriever()
keyword_retriever = BM25Retriever.from_documents(text_splits)
keyword_retriever.k = 5
ensemble_retriever = EnsembleRetriever(retrievers=[retriever, keyword_retriever], weights=[0.5, 0.5])

question = 'How many cafes were closed in 2004?'
docs_rel = ensemble_retriever.get_relevant_documents(question)

print(docs_rel)


# llm
model = Tongyi(model_name="qwen-max")

# implement Re-ranking with Cohere-Rerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
compressed_docs = compression_retriever.get_relevant_documents(question)

print(compressed_docs)


# Augmentation and generation
template = """
<|system|>>
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {'context': compression_retriever, 'query': RunnablePassthrough()} |
    prompt |
    model |
    output_parser
)

question = 'How many cafes were closed in 2004 in China?'
response = chain.invoke(question)
print(response)


if __name__ == '__main__':
    pass
