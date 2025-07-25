import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()  # Load environment variables from .env

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if content := page.extract_text():
                text += content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_conversational_chain(vector_store):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        speaker = "👤 Human" if i % 2 == 0 else "🤖 Bot"
        st.write(f"**{speaker}:** {message.content}")

def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("📄 DocuQuery: AI-Powered PDF Knowledge Assistant")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Process Button",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Ready for questions!")

if __name__ == "__main__":
    main()
  
