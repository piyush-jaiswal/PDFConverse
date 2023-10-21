import argparse
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.schema import AIMessage


def get_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    chunk_size = 1000
    if st.session_state.use_hugging_face:
        chunk_size = 800

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings():
    embeddings = OpenAIEmbeddings()
    if st.session_state.use_hugging_face:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    return embeddings


def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=get_embeddings())
    return vector_store


def get_llm():
    llm = ChatOpenAI()
    if st.session_state.use_hugging_face:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.2, "max_length": 512})

    return llm


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def process_input(input):
    with st.chat_message("user"):
        st.markdown(input)

    with st.spinner("Generating..."):
        response = st.session_state.conversation({'question': input})
        st.session_state.chat_history = response['chat_history']

    with st.chat_message("ai"):
        st.markdown(response["answer"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf', action='store_true', help='Use huggingface models')
    args = parser.parse_args()

    load_dotenv()
    st.set_page_config(page_title="PDFConverse", page_icon=":book:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "use_hugging_face" not in st.session_state:
        st.session_state.use_hugging_face = args.hf

    st.title("PDFConverse :book:")
    st.subheader("Chat with your PDF :star2:")

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        role = "user"
        if isinstance(message, AIMessage):
            role = "ai"

        with st.chat_message(role):
            st.markdown(message.content)

    # User input
    if prompt := st.chat_input("Ask a question about your document."):
        if st.session_state.conversation:
            process_input(prompt)
        else:
            st.error("Please upload a pdf first!")

    # Sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf = st.file_uploader(
            "Upload your PDF here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_text_from_pdf(pdf)

                if not raw_text.strip():
                    st.error("No text found in PDF!")
                    return

                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()
