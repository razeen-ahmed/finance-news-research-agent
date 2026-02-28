import os
import streamlit as st
import pickle
import time
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines import pipeline



# from dotenv import load_dotenv
# load_dotenv()
file_path = "faiss_store_openai.pkl"

st.title("Finance News Research Agent")

st.sidebar.title("Enter a URL to fetch financial articles")

url_list = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    url_list.append(url)

process_url_clicked = st.sidebar.button("Process URLs")    

main_placeholder = st.empty()

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

if process_url_clicked:
    # Filter out empty URLs
    urls = [url for url in url_list if url.strip()]
    if not urls:
        st.error("❌ Please enter at least one URL.")
        st.stop()

    # Data Loader
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("📥 Loading data...")
    data = loader.load()
    
    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ""],
        chunk_size=1000
    )
    main_placeholder.text("✂️ Splitting text...")
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("❌ No content could be extracted from the provided URLs. Please check the URLs and try again.")
        st.stop()

    # ✅ FREE Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("🔮 Creating embeddings...")
    
    time.sleep(2)
    
    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    st.success("✅ Processing complete! Ask questions below.")


ques = main_placeholder.text_input("Enter your question here->")

if ques:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        
        # ✅ FREE RAG Chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )
        
        main_placeholder.text("🤖 Thinking...")
        result = chain.invoke({"question": ques}, return_only_outputs=True)
        
        # Display answer
        st.header("📝 Answer:")
        st.write(result['answer'])
        
        # Display sources
        if 'sources' in result and result['sources']:
            st.subheader("📚 Sources:")
            for source in result['sources'].split("\n"):
                if source.strip():
                    st.write(source)
    else:
        st.warning("⚠️ Process URLs first!")