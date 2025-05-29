import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import tiktoken

load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

st.set_page_config(page_title="NVIDIA AI Chatbot", page_icon=":robot_face:", layout="wide")
st.title("NVIDIA NIM Demo Chatbot")

# Initialize LLM
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2, top_p=0.7, max_tokens=1024)

def count_tokens(text, model="cl100k_base"):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def filter_chunks_by_tokens(documents, max_tokens=400):
    """Filter chunks to ensure they don't exceed token limit"""
    filtered_docs = []
    for doc in documents:
        token_count = count_tokens(doc.page_content)
        if token_count <= max_tokens:
            filtered_docs.append(doc)
        else:
            st.warning(f"Skipping chunk with {token_count} tokens (exceeds {max_tokens} limit)")
    return filtered_docs

def vector_embeddings():
    if "vectors" not in st.session_state:
        with st.spinner("Creating embeddings..."):
            try:
                # Initialize embeddings
                st.session_state.embeddings = NVIDIAEmbeddings(api_key=os.getenv("NVIDIA_API_KEY"))
                
                # Load documents
                st.session_state.loader = PyPDFDirectoryLoader("./data")
                st.session_state.docs = st.session_state.loader.load()
                
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,  
                    chunk_overlap=100  
                )
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
                
                # Filter chunks by token count (keeping buffer for safety)
                st.session_state.final_documents = filter_chunks_by_tokens(
                    st.session_state.final_documents, 
                    max_tokens=400  
                )
                
                # Display chunk statistics
                chunks_lengths = []
                token_counts = []
                st.write(f"Loaded {len(st.session_state.final_documents)} chunks after filtering.")
                
                for i, doc in enumerate(st.session_state.final_documents[:3]):
                    char_length = len(doc.page_content)
                    token_count = count_tokens(doc.page_content)
                    st.write(f"Chunk {i+1}: {char_length} chars, {token_count} tokens")
                    st.write(f"Preview: {doc.page_content[:100]}...")
                    chunks_lengths.append(char_length)
                    token_counts.append(token_count)
                
                if chunks_lengths:
                    st.write(f"Average chunk length: {sum(chunks_lengths) / len(chunks_lengths):.2f} characters")
                    st.write(f"Average tokens per chunk: {sum(token_counts) / len(token_counts):.2f}")
                    st.write(f"Max tokens in chunk: {max(token_counts)}")
                
                # Create vector store with error handling
                if st.session_state.final_documents:
                    st.session_state.vector_store = FAISS.from_documents(
                        st.session_state.final_documents, 
                        st.session_state.embeddings
                    )
                    st.session_state.vectors = True
                else:
                    st.error("No valid documents found after filtering!")
                    
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                if "Input length" in str(e) and "exceeds maximum" in str(e):
                    st.error("Token limit exceeded. Try reducing chunk_size further or check document content.")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context provided. If the answer is not in the context, say 'I don't know'."),
    ("human", "Context: {context}\nQuestion: {input}")
])

# User input
prompt1 = st.text_input("Ask a question about the documents:", placeholder="Type your question here...")

# Document embedding button
if st.button("Document Embedding"):
    vector_embeddings()
    if "vectors" in st.session_state and st.session_state.vectors:
        st.success("FAISS vector store created **successfully** using NVIDIA!")

# Handle user questions
if prompt1:
    # Ensure vector_store is initialized
    if "vector_store" not in st.session_state:
        st.error("Please create the document embeddings first by clicking the 'Document Embedding' button.")
    else:
        try:
            # Create retrieval chain
            documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            retrieval_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=documents_chain
            )
                   
            # Get response
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt1})
            end_time = time.process_time() - start
            
            # Display results
            st.write("**Response:**")
            st.write(response["answer"])
            st.write(f"**Response Time:** {end_time:.2f} seconds")
            
            # Show context in expander
            with st.expander("📄 View Context Documents"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Document {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")
                    
        except KeyError as e:
            st.error(f"Error: {str(e)}. Ensure the input format is correct.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.write("This chatbot uses NVIDIA NIM endpoints for:")
    st.write("- **Embeddings**: Document vectorization")
    st.write("- **LLM**: Llama 3.3 70B Instruct")
    
    if "final_documents" in st.session_state:
        st.write(f"**Documents loaded:** {len(st.session_state.final_documents)}")
    
    st.header("⚙️ Settings")
    st.write("**Current Configuration:**")
    st.write("- Chunk size: 800 characters")
    st.write("- Chunk overlap: 100 characters") 
    st.write("- Max tokens per chunk: 400")
    st.write("- Retrieved documents: 3")