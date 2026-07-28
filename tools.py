import os
import tempfile
import time
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

@st.cache_resource
def get_embeddings():
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="/tmp/huggingface_models"  # Add this line
    )


def list_existing_files(docs_dir="docs", allowed_extensions=(".pdf", ".csv")):
    """Lists all supported files (PDF, CSV) in the specified directory."""
    if not os.path.exists(docs_dir):
        return []
    return [
        f for f in os.listdir(docs_dir)
        if f.lower().endswith(allowed_extensions) and os.path.isfile(os.path.join(docs_dir, f))
    ]


def list_existing_pdfs(docs_dir="docs"):
    """Backward compatible alias for list_existing_files."""
    return list_existing_files(docs_dir=docs_dir)


def load_csv_with_pandas(file_source):
    """Loads CSV file using pandas and converts rows into LangChain Document objects."""
    try:
        if isinstance(file_source, (str, os.PathLike)):
            df = pd.read_csv(str(file_source))
        elif hasattr(file_source, "getvalue"):
            df = pd.read_csv(file_source)
        elif hasattr(file_source, "read"):
            df = pd.read_csv(file_source)
        else:
            raise ValueError("Unsupported file source for CSV.")
    except Exception:
        if isinstance(file_source, (str, os.PathLike)):
            df = pd.read_csv(str(file_source), encoding="latin1")
        elif hasattr(file_source, "seek"):
            file_source.seek(0)
            df = pd.read_csv(file_source, encoding="latin1")
        else:
            raise

    documents = []
    for idx, row in df.iterrows():
        row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        if row_str.strip():
            metadata = {"page": idx + 1, "row": idx + 1}
            documents.append(Document(page_content=row_str, metadata=metadata))
    return documents


def process_file(file_source, filename=None):
    """Processes uploaded or local PDF/CSV file into a FAISS vectorstore retriever using PyPDF or Pandas."""
    if filename is None:
        if isinstance(file_source, (str, os.PathLike)):
            filename = str(file_source)
        elif hasattr(file_source, "name"):
            filename = file_source.name
        else:
            filename = ""

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        documents = load_csv_with_pandas(file_source)
    else:
        # Default to PDF processing
        if isinstance(file_source, (str, os.PathLike)):
            loader = PyPDFLoader(str(file_source))
            documents = loader.load()
        elif hasattr(file_source, "getvalue"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_source.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            os.remove(tmp_path)  # Clean up temporary file
        else:
            raise ValueError("Unsupported file source. Must be a file path string or uploaded file object.")

        for doc in documents:
            raw_page = doc.metadata.get("page", 0)
            doc.metadata["page"] = raw_page + 1

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def process_pdf(file_source):
    """Backward compatible alias for process_file."""
    return process_file(file_source)


def build_rag_chain(retriever):
    """Builds history-aware RAG chain with page citations."""
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
            return retriever.invoke(inputs["input"])
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
        "Para cada punto o respuesta, cita siempre el número de página o fila correspondiente "
        "(por ejemplo, Página 1 o Fila 1)."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    document_prompt = PromptTemplate(
        input_variables=["page_content", "page"],
        template="[Página/Fila: {page}]\n{page_content}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, prompt, document_prompt=document_prompt
    )
    return create_retrieval_chain(retriever_adaptativo, question_answer_chain)


def invocar_agente_con_retry(cadena, inputs, max_retries=3):
    """Invokes chain with exponential backoff on 500 error status."""
    for i in range(max_retries):
        try:
            return cadena.invoke(inputs)
        except Exception as e:
            if "500" in str(e) and i < max_retries - 1:
                time.sleep(5 * (2 ** i))
                continue
            raise e


def format_response_md(answer, user_query=None, filename=None, sources=None):
    """Formats assistant response into a structured Markdown document."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = ["# 📄 Respuesta del Asistente\n"]

    metadata_lines = []
    if filename:
        metadata_lines.append(f"**Documento:** `{filename}`")
    metadata_lines.append(f"**Fecha de generación:** {now_str}")
    md_lines.append("  \n".join(metadata_lines))
    md_lines.append("\n---\n")

    if user_query:
        md_lines.append("### ❓ Pregunta")
        md_lines.append(f"> {user_query}\n")
        md_lines.append("---\n")

    md_lines.append("### 💡 Respuesta\n")
    md_lines.append(answer)

    if sources:
        md_lines.append("\n\n---\n")
        md_lines.append("### 📌 Fuentes Citadas\n")
        for source in sources:
            page = source.get("page", 1)
            content = source.get("content", "").strip()
            md_lines.append(f"- **Página/Fila {page}:** {content}")

    return "\n".join(md_lines)
