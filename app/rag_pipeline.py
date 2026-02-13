# app/rag_pipeline.py

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

from retriever import load_retriever
import config


def create_qa_chain():
    retriever = load_retriever()

    llm = Ollama(model=config.MODEL_NAME)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
