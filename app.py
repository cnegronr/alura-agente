import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
import time

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

    # Ensure page metadata is set on all chunks
    for chunk in chunks:
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 0

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Helper function to build the RAG chain
def build_rag_chain(retriever):
    llm = ChatGroq(temperature=0.2, model_name="llama-3.3-70b-versatile")

    # 1. Contextualizer to rewrite queries using conversation history
    contextualize_q_system_prompt = (
        "Dada la conversación previa y la última pregunta del usuario, "
        "formula una sola pregunta clara e independiente para buscar en el documento. "
        "NO respondas la pregunta."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Retriever switcher: use contextualized retriever when there is history
    def retriever_estructurado(inputs):
        history = inputs.get("chat_history", [])
        if not history:
            # direct search when no history
            return retriever.invoke(inputs["input"])  # noqa: E1101
        else:
            return history_aware_retriever.invoke(inputs)

    retriever_adaptativo = RunnableLambda(retriever_estructurado)

    # 2. System prompt that enforces answer-from-context and page citations
    system_prompt = (
        "Eres un asistente especializado en responder preguntas sobre documentos internos.\n"
        "Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto extraído del documento:\n"
        "{context}\n"
        "Si la respuesta no se encuentra en el contexto, indica claramente:\n"
        "'No encontré información sobre este tema en el documento proporcionado.'\n"
        "Para cada punto o respuesta, cita siempre el número de página correspondiente del PDF "
        "(revisa la metadata 'page', considerando que la página 0 corresponde a la Página 1)."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_prompt = PromptTemplate(
        input_variables=["page_content", "page"],
        template="[Metadata 'page': {page}]\n{page_content}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, prompt, document_prompt=document_prompt
    )
    return create_retrieval_chain(retriever_adaptativo, question_answer_chain)


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
    if "chat_history" not in st.session_state:
        # store simple serializable chat history as list of dicts {role, content}
        st.session_state.chat_history = []

    # Display prior chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Retry wrapper for chain invocation
    def invocar_agente_con_retry(cadena, inputs, max_retries=3):
        for i in range(max_retries):
            try:
                return cadena.invoke(inputs)
            except Exception as e:
                if "500" in str(e) and i < max_retries - 1:
                    time.sleep(5 * (2 ** i))
                    continue
                raise e

    # User input prompt
    if user_input := st.chat_input("Haz una pregunta sobre el documento..."):
        # Display user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare chat_history for the chain as HumanMessage/AIMessage list
        messages_for_chain = []
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                messages_for_chain.append(HumanMessage(content=m["content"]))
            else:
                messages_for_chain.append(AIMessage(content=m["content"]))

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = invocar_agente_con_retry(rag_chain, {"chat_history": messages_for_chain, "input": user_input})
                answer = response.get("answer") if isinstance(response, dict) else response
                st.markdown(answer)

                if isinstance(response, dict) and "context" in response and response["context"]:
                    with st.expander("📌 Fuentes citadas / Páginas"):
                        for i, doc in enumerate(response["context"]):
                            page_num = doc.metadata.get("page", 0) + 1
                            st.write(f"**Página {page_num}:**")
                            st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Por favor, sube un archivo PDF desde el panel lateral para empezar.")
