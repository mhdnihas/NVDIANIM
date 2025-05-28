import streamlit as st  
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 

from dotenv import load_dotenv
load_dotenv()

os.environ["NVDIA_API_KEY"] = os.getenv("NVDIA_API_KEY")

st.set_page_config(page_title="NVIDIA AI Chatbot", page_icon=":robot_face:", layout="wide") 
st.title("NVIDIA NIM Demo Chatbot")


llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2, top_p=0.7, max_tokens=1024)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


prompt= ChatPromptTemplate.from_messages("""
Answer the question based on the context provided. If the answer is not in the context, say "I don't know".
Context: {context}
Question: {question}
""")


prompt1 = st.text_input("Ask a question about the documents:", placeholder="Type your question here...")
if st.button("Submit"):
    vector_embeddings()
    st.write("FAISS vector store created **successfully** using Nvidia!")
    chain = create_retrieval_chain(
        llm=llm,
        retriever=st.session_state.vector_store.as_retriever(),
        prompt=prompt,
        output_parser=StrOutputParser()
    )
    response = chain.invoke({"question": prompt1})
    st.write(response["answer"])