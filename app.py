# app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Policy Assistant", page_icon="🤖")
st.title("📄 Company Policy Assistant (Gemma 2B)")

# 🔥 Use GEMMA 2B for embeddings
embeddings = OllamaEmbeddings(model="gemma:2b")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

# 🔥 Use GEMMA 2B as LLM
llm = Ollama(model="gemma:2b")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

query = st.text_input("Ask a question about company policy")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke(query)
        st.success(result["result"])
