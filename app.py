import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import hashlib
from io import BytesIO

def type_text_with_cursor(text: str, delay: float = 0.01):
    """Streamlit typing effect with blinking cursor."""
    container = st.empty()
    typed = ""
    style = """
    <style>
    .typed-text::after {content: "|"; animation: blink 1s infinite;}
    @keyframes blink {0%{opacity:1;}50%{opacity:0;}100%{opacity:1;}}
    .typed-text {font-family: monospace; white-space: pre-wrap; font-size: 1.05rem;}
    </style>
    """
    # inject style once
    st.markdown(style, unsafe_allow_html=True)
    for ch in text:
        typed += ch
        container.markdown(f"<div class='typed-text'>{typed}</div>", unsafe_allow_html=True)
        time.sleep(delay)

# --- Load environment variables ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Debug check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("GOOGLE_API_KEY not found. Please check your .env file.")

def process_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = DocArrayInMemorySearch.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create a conversation chain with memory."""
    llm = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-2.5-flash", convert_system_message_to_human=True)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
        # Use Maximal Marginal Relevance to diversify retrieved chunks and increase k
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    from langchain.prompts import PromptTemplate
    QA_PROMPT = PromptTemplate(
        template=(
            "You are an expert tutor. Use the following context to answer the user's question. "
            "If the user requests a summary, provide a concise yet thorough summary covering all key points.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
        input_variables=["context", "question"],
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversation_chain

# ---------------- Utility helpers ----------------
@st.cache_resource(show_spinner="Indexing document(s)...")
def build_vectorstore(file_bytes_tuple: tuple[bytes, ...]):
    """Return vector store given raw PDF bytes for multiple files (cached)."""
    all_chunks = []
    for b in file_bytes_tuple:
        raw_text = process_pdf(BytesIO(b))
        all_chunks.extend(get_text_chunks(raw_text))
    return get_vectorstore(all_chunks)

def main():
    load_dotenv()
    
    st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
    st.title("🤖 AI FAQ Chatbot")
    st.write("Upload a PDF document and ask questions about its content.")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload your PDF")
        pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if pdf_files:
            file_bytes_list = [f.getvalue() for f in pdf_files]
            file_hash = hashlib.md5(b''.join(file_bytes_list)).hexdigest()

            if st.session_state.get("file_hash") != file_hash:
                vectorstore = build_vectorstore(tuple(file_bytes_list))
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.file_hash = file_hash
                st.success("Document processed! You can now ask questions.")

    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.conversation is None:
            with st.chat_message("assistant"):
                st.warning("Please upload a PDF document first.")
            return
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({"question": prompt})
                    answer = response["answer"]
                    
                    # Display source documents
                    type_text_with_cursor(answer, delay=0.01)
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.write(f"**Source {i}:**")
                            st.text(doc.page_content[:500] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
