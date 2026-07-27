import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 PDF Alura Agent")

# Ensure GROQ_API_KEY is present
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()


# Cache heavy operations like embedding loading
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Helper function to process uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)  # Clean up temporary file

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Helper function to build the RAG chain
def build_rag_chain(retriever):
    llm = ChatGroq(temperature=0.2, model_name="llama-3.3-70b-versatile")

    system_prompt = """
    Eres un asistente especializado en responder preguntas sobre documentos internos.
    Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto extraído del documento:
    {context}
    Si la respuesta no se encuentra en el contexto, indica claramente:
    'No encontré información sobre este tema en el documento proporcionado.'
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# --- UI Sidebar for Upload ---
st.sidebar.header("Document Setup")
uploaded_file = st.sidebar.file_uploader("Subir un archivo PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Procesando el archivo PDF..."):
        retriever = process_pdf(uploaded_file)
        rag_chain = build_rag_chain(retriever)
    st.sidebar.success("PDF procesado exitosamente")

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input prompt
    if user_input := st.chat_input("Haz una pregunta sobre el documento..."):
        # Display user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Por favor, sube un archivo PDF desde el panel lateral para empezar.")
