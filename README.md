# 📰 Finance News Research Agent

An end‑to‑end **agentic AI** project that transforms any financial news article into a queryable knowledge base using a fully **FREE** stack: Hugging Face LLMs, Hugging Face sentence embeddings, FAISS vector search, and LangChain RAG – all wrapped in an interactive Streamlit web app.

## ✨ Features

- **🤖 Agentic News Research Assistant**: Ask natural‑language questions about one or more finance news articles and get concise answers with citations.
- **💰 Free LLM Stack (No OpenAI API Key)**: Uses `google/flan-t5-small` via `transformers` + `langchain_huggingface.HuggingFacePipeline`.
- **🎯 Free Embeddings**: Uses `HuggingFaceEmbeddings` with the `all-MiniLM-L6-v2` sentence‑transformer model.
- **🔍 RAG Pipeline**: Unstructured URL loading → text chunking → FAISS vector store → RetrievalQAWithSourcesChain.
- **🎨 Interactive Streamlit UI**: Paste finance article URLs, process them, then chat with the agent from your browser.
- **📓 Colab Notebooks Included**:
  - `agent_test.ipynb`: Low‑level exploration of loaders, text splitting, custom FAISS index, and manual vector search.
  - `news_research_agent.ipynb`: Rebuilds the full RAG pipeline in Colab with the same free LLM + embedding stack.

---

## 📁 Project Structure

```
.
├── agent/
│   ├── main.py                          # Streamlit front-end + full RAG agent
│   ├── requirement.txt                  # Python dependencies
│   └── data/                            # Sample data for experimentation
│       ├── movies.csv
│       ├── nvda_news_1.txt
│       └── sample_text.csv
├── agent_test.ipynb                     # Vector DB & search prototype (FAISS + sentence-transformers)
├── news_research_agent.ipynb            # Full Colab RAG pipeline (free HF LLM + embeddings)
├── LICENSE
└── README.md                            # You are here
```

- **`main.py`**: Production app entry point (Streamlit frontend).
- **`agent_test.ipynb`**: Step‑by‑step building blocks—CSV/text loaders, SentenceTransformer embeddings, FAISS index, and nearest‑neighbor search.
- **`news_research_agent.ipynb`**: Full RAG pipeline runnable in Google Colab for experimentation and debugging.

---

## 🛠 Tech Stack

### Core Libraries

**LangChain Components:**
- `langchain_community.document_loaders.UnstructuredURLLoader` – Fetch and parse raw web pages
- `langchain_text_splitters.RecursiveCharacterTextSplitter` – Turn long articles into overlapping chunks for retrieval
- `langchain_community.vectorstores.FAISS` – Vector store built on FAISS
- `langchain_community.embeddings.HuggingFaceEmbeddings` – Sentence‑transformer embeddings (`all-MiniLM-L6-v2`)
- `langchain.chains.RetrievalQAWithSourcesChain` – RAG chain that returns answers + sources

**Hugging Face Models:**
- **LLM**: `google/flan-t5-small` (AutoTokenizer, AutoModelForSeq2SeqLM, text2text-generation pipeline)
- **Embeddings**: `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`

**Vector Database:**
- **FAISS**: Facebook's AI Similarity Search library for fast vector similarity queries

**Frontend & Tooling:**
- **Streamlit**: Interactive UI with sidebar controls, status updates, and result rendering
- **Python/Jupyter/Google Colab**: For prototyping and experimentation

---

## 🚀 How It Works

### 1️⃣ Document Loading (from URLs)

User enters up to two finance news URLs in the Streamlit sidebar and clicks **"Process URLs"**:

```python
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
```

This fetches and parses article content into `Document` objects using `UnstructuredURLLoader`.

### 2️⃣ Text Splitting

Long articles are split into manageable chunks for retrieval:

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", ""],
    chunk_size=1000
)
docs = text_splitter.split_documents(data)
```

These chunks become the atomic units stored in the vector database.

### 3️⃣ Vector Embeddings & FAISS Index

Each chunk is embedded with a free Hugging Face sentence transformer; a FAISS index is built and stored on disk:

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

with open("faiss_store_openai.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
```

### 4️⃣ Free LLM Pipeline (Hugging Face)

Instead of OpenAI, the app builds a local `transformers` pipeline around `google/flan-t5-small` and wraps it in `HuggingFacePipeline`:

```python
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()
```

`@st.cache_resource` ensures the model is loaded only once per session.

### 5️⃣ Retrieval‑Augmented Generation Chain

When the user asks a question, the stored FAISS index is reloaded and used as a retriever:

```python
with open(file_path, "rb") as f:
    vectorstore = pickle.load(f)

chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

result = chain.invoke({"question": ques}, return_only_outputs=True)
```

The chain retrieves the most relevant chunks and lets the LLM synthesize an answer, returning both the **answer** and **source documents**.

### 6️⃣ Streamlit UI

The Streamlit app wires everything together:

**Sidebar:**
- URL inputs (`URL 1`, `URL 2`)
- "Process URLs" button

**Main Area:**
- Status messages ("Loading data…", "Splitting text…", "Creating embeddings…", "Thinking…")
- Question text input
- Rendered answer and list of source URLs

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/agent
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Using `requirement.txt`:

```bash
pip install -r requirement.txt
```

Or manually:

```bash
pip install streamlit langchain langchain-community langchain-text-splitters \
            langchain-classic langchain-huggingface \
            transformers sentence-transformers faiss-cpu unstructured[local-inference]
```

---

## ▶️ Running the Streamlit App

From the `agent` directory:

```bash
streamlit run main.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

### Usage Flow:

1. **Enter URLs** – Paste 1–2 finance news URLs in the sidebar.
2. **Process** – Click **Process URLs** and wait for:
   - "📥 Loading data…"
   - "✂️ Splitting text…"
   - "🔮 Creating embeddings…"
   - Success message
3. **Ask Questions** – Type in the main input field, e.g.:
   - _"What is the impact of this news on HDFC Bank's stock?"_
   - _"Summarise the key risks mentioned in these articles."_
4. **Read Answer & Sources** – Inspect the answer and the **📚 Sources** list.

---

## 📓 Notebooks: Prototyping & Learning

### `agent_test.ipynb` – From Raw Text to FAISS Search

This notebook walks through the underlying building blocks without Streamlit:

- **Loading:**
  - Plain text via `TextLoader`
  - CSV data via `CSVLoader`
  - Web pages via `UnstructuredURLLoader`
- **Text Splitting** using `RecursiveCharacterTextSplitter`
- **Sentence Embeddings** with `SentenceTransformer("all-mpnet-base-v2")`
- **Manual FAISS Index** creation and nearest‑neighbor search

Ideal if you want to understand how vector search works before plugging it into LangChain.

### `news_research_agent.ipynb` – Full RAG Pipeline in Colab

This notebook mirrors the production agent in a Jupyter environment:

- Installs compatible versions of dependencies
- Builds the same `google/flan-t5-small` HuggingFacePipeline
- Uses UnstructuredURLLoader, text splitting, FAISS, and RetrievalQAWithSourcesChain
- Invokes the LLM directly with test questions to validate the pipeline

---

## 🎨 Customization

You can extend or customize the agent:

- **Change the LLM**: Swap `google/flan-t5-small` for a larger or domain‑specific Hugging Face model.
- **Adjust Retrieval Granularity**:
  - Tune `chunk_size` and `separators` in `RecursiveCharacterTextSplitter`
  - Change `k` in `retriever=vectorstore.as_retriever(search_kwargs={"k": 2})` to retrieve more/fewer chunks
- **Persist Vector Store Differently**: Currently uses `pickle`; you can move to a database or cloud storage.
- **Extend UI**:
  - Add filters (date, ticker, source)
  - Show raw retrieved chunks alongside the answer
  - Log conversation history

---

## ⚠️ Limitations

- **Model Size & Context**: `flan-t5-small` is lightweight and free, but not as strong as larger commercial models and has a limited context window.
- **URL Parsing Robustness**: `UnstructuredURLLoader` depends on page structure; some pages may not parse cleanly.
- **Local Compute**: All models run on your machine/Colab; performance depends on your CPU/GPU.

---

## 🔮 Future Work

Ideas for improving the project:

- Add **multi‑URL summarisation** (e.g., "Compare these two articles' sentiment on NVDA")
- Integrate **ticker/entity extraction** and link to market data APIs
- Replace `pickle` with a persistent FAISS store on disk or in a vector DB service
- Add **evaluation notebooks** to measure retrieval quality and answer correctness
- Support for **PDF documents** and **financial reports**
- **Conversation history** and **context carryover** between questions

---

## 🎓 Inspiration & Credits

This project was inspired by agentic AI patterns and RAG (Retrieval‑Augmented Generation) techniques from:

📺 **YouTube Tutorial**: [Building AI News Research Agents with LangChain](https://www.youtube.com/watch?v=MoqgmWV1fm8&t=3982s)

The tutorial demonstrates how to build production‑grade LLM applications without paid APIs, leveraging open‑source tools and free Hugging Face models.

---

## 📝 License

This project is licensed under the MIT License – see `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or fork the repository to build your own version.

---

## 📞 Contact & Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Share ideas and best practices

---

**Built with ❤️ using open‑source tools and free models.**
