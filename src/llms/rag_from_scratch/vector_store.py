import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_retriever(chunk_size: int = 300, chunk_overlap: int = 50) -> VectorStoreRetriever:
    """

    :param chunk_size:
    :param chunk_overlap:
    :return:
    """

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    blog_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)

    splits = text_splitter.split_documents(blog_docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    return vectorstore.as_retriever()


if __name__ == '__main__':
    pass
