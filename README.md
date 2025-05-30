# 🤖 NVIDIA NIM Demo Chatbot

A Streamlit-based chatbot that uses **NVIDIA NIM** APIs for document embeddings and large language model (LLM) responses. It allows users to upload PDFs, generate embeddings with NVIDIA's endpoint, and ask contextual questions powered by `Llama 3.3-70B Instruct`.

---

## 🚀 Features

- 🔍 Upload and process PDF files from the `./data` directory
- 🧠 Create document embeddings using `NVIDIAEmbeddings`
- 💬 Ask questions based on document context
- 🧾 View retrieved context documents
- 📊 Token and chunk statistics for debugging and transparency
- ⚡ Powered by `meta/llama-3.3-70b-instruct` via NVIDIA's NIM APIs

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [NVIDIA NIM](https://developer.nvidia.com/nim)
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage
- [tiktoken](https://github.com/openai/tiktoken) for token counting
- `.env` for environment variable management

---

## 📁 Directory Structure

```bash
.
├── data/ # PDF files go here
├── .env # NVIDIA_API_KEY goes here
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── README.md # You're here!

```




---

## 🧪 How It Works

1. **Load PDFs** from the `./data` folder
2. **Split text** into chunks using `RecursiveCharacterTextSplitter`
3. **Filter** out chunks exceeding token limits
4. **Generate embeddings** using `NVIDIAEmbeddings`
5. **Store embeddings** in FAISS vector store
6. **Ask questions**, and the app retrieves relevant chunks and responds using LLM

---

# 🧑‍💻 Run the App Locally

#### Follow these steps to run the app in your local environment:

### 1. 📦 Create and activate a virtual environment

On macOS/Linux:

```bash

python3 -m venv venv
source venv/bin/activate

```

On Windows:

```bash

python -m venv venv
venv\Scripts\activate

```


### 2. 📥 Install dependencies

Make sure you're in the root directory of the project and run:

```bash

pip install -r requirements.txt

```

### 3. 🔐 Set your NVIDIA API key
Create a ```.env``` file in the root directory with the following content:

```bash

NVIDIA_API_KEY=your_nvidia_api_key_here

```

Make sure to replace ```your_nvidia_api_key_here``` with your actual API key from NVIDIA NIM.


### 4. 🧾 Add your PDF documents
Place your PDF files inside the ``` data/ ```folder. The app will automatically load and process them.



### 5. 🚀 Run the Streamlit app

Now launch the app using:

```bash

streamlit run app.py

```

#### The app will open in your default browser. You can:

* Click the "Document Embedding" button to process and embed documents.
* Enter a question in the input box to query your documents using LLM.
* View the retrieved context documents and response time.

