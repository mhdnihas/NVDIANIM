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

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2, top_p=0.7, max_tokens=1024)
embeddings = NVIDIAEmbeddings(model="nvidia/llama-3.3-70b-instruct-embeddings", max_retries=5)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        