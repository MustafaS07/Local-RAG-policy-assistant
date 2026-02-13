# app/retriever.py
# responsible for loading vecDB nd retriever


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import config


def load_retriever():
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.TOP_K}
    )

    return retriever
