"""

"""
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qianfan_rag():
    """"""
    # 文档加载
    loader = WebBaseLoader(WEB_URL)
    raw_document = loader.load()

    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0, separators=[
        '\n\n', '\n', ' ', '', '。', '，'
    ])
    documents = text_splitter.split_documents(raw_document)

    # embeddings
    vec_store = Chroma.from_documents(documents=documents, embedding=QianfanEmbeddingsEndpoint())

    question = QUESTION4
    #
    print('prompt 问题：', question)
    # documents = vec_store.similarity_search_with_relevance_scores(QUESTION1)
    # [(document.page_content, score) for document, score in documents]

    QA_CHAIN_PROMPT = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

    llm = QianfanChatEndpoint(streaming=True, model='ERNIE-4.0-8K')
    retriever = vec_store.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.0})

    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=retriever,
                                           chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
                                           return_source_documents=True)
    result = qa_chain({'query': question})
    print(result['result'])


if __name__ == '__main__':
    qianfan_rag()
