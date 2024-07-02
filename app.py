import streamlit as st
from PyPDF2 import PdfReader
from docx import Document  # Ensure 'python-docx' is installed
import os
import nltk
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

nltk.download('punkt')

load_dotenv()  # Load environment variables from .env
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize global variables to store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def get_pdf_text(docs):
    """
    Convert uploaded PDF, DOCX, or TXT documents to text.
    """
    text = ""
    for doc in docs:
        if doc.type == 'application/pdf':
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if isinstance(page_text, list):
                    text += "".join(page_text)
                else:
                    text += page_text
        elif doc.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            docx_text = Document(doc).paragraphs
            for paragraph in docx_text:
                text += paragraph.text
        elif doc.type == 'text/plain':
            text += doc.getvalue().decode("utf-8")
    
    return text

def get_text_chunks(text):
    """
    Generate chunks of text for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Store text chunks in a vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(folder_path="faiss_index")

def get_conversational_chain():
    """
    Create a conversational chain for question answering.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context."
    
    Context: {context}\n
    
    Question: \n{question}?\n
    
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """
    Process user input to retrieve answers from the research paper.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Save chat history
    st.session_state["chat_history"].append({"user_question": user_question, "response": response["output_text"]})

    st.write("Answer: \n\n", response["output_text"])

def summarize_text(text):
    """
    Summarize the research paper content.
    """
    prompt = """
    Welcome, Research paper summarizer! Your task is to write a concise, objective summary of this paper in about 100 words, Maintain a neutral tone. Only generate the summary from the information given in the context below.
    """
    model = genai.GenerativeModel('models/gemini-pro')
    response = model.generate_content(prompt + text)
    return response.text

def main():
    st.set_page_config(page_title="Chat with Multiple Files", page_icon=":books:", layout="wide")
    
    st.title("Research Companion using Gemini :sparkles:")
    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Menu")
    st.sidebar.markdown("Upload your Research Paper and Choose an action.")

    if "text_chunks" not in st.session_state:
        st.session_state["text_chunks"] = None

    if "raw_text" not in st.session_state:
        st.session_state["raw_text"] = None

    with st.sidebar:
        docs = st.file_uploader("Upload your Files & Click on Submit", accept_multiple_files=True, type=["pdf", "docx", "txt"])
        if st.button("Submit"):
            if docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state["text_chunks"] = text_chunks
                    st.session_state["raw_text"] = raw_text
                    st.success("Processing Complete")
                    st.write("You can now use the tabs to summarize or ask questions based on the uploaded files.")
            else:
                st.error("Please upload at least one file.")

    if st.session_state["text_chunks"]:
        tab1, tab2, tab3 = st.tabs(["Summarize", "Ask Questions", "View Chat History"])
        
        with tab1:
            st.header("Summarize Content")
            if st.button("Summarize"):
                with st.spinner("Summarizing..."):
                    summary = summarize_text(st.session_state["raw_text"])
                    st.write("Summary:\n\n", summary)
        
        with tab2:
            st.header("Ask Questions")
            user_ques = st.text_input("Ask a question from the Uploaded File & Press Enter", placeholder="Type your question here...")
            if user_ques:
                user_input(user_ques)
                st.divider()

        with tab3:
            st.header("View Chat History")
            for chat in st.session_state["chat_history"]:
                st.write(f"User: {chat['user_question']}")
                st.write(f"Bot: {chat['response']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
