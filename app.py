import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import modular tools and functions from tools.py
from tools import (
    get_embeddings,
    process_file,
    build_rag_chain,
    invocar_agente_con_retry,
    format_response_md,
    list_existing_files,
)

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄")
st.title("📄 Document Q&A Agent")

# Ensure GROQ_API_KEY is present
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()

# --- UI Sidebar for Document Selection / Upload ---
st.sidebar.header("Document Setup")

docs_dir = "docs"
existing_files = list_existing_files(docs_dir)

source_option = st.sidebar.radio(
    "Selecciona la fuente del documento:",
    options=["Archivos en docs/", "Subir nuevo archivo (PDF/CSV)"],
    index=None,
)

selected_file_source = None
file_identifier = None

if source_option == "Archivos en docs/":
    if existing_files:
        selected_file_name = st.sidebar.selectbox(
            "Selecciona un archivo de la carpeta docs/",
            options=sorted(existing_files),
            index=None,
            placeholder="Selecciona un archivo...",
        )
        if selected_file_name:
            selected_file_source = os.path.join(docs_dir, selected_file_name)
            file_identifier = selected_file_name
    else:
        st.sidebar.info("No se encontraron archivos en la carpeta 'docs/'.")
elif source_option == "Subir nuevo archivo (PDF/CSV)":
    uploaded_file = st.sidebar.file_uploader(
        "Subir un archivo (PDF o CSV)", type=["pdf", "csv"]
    )
    if uploaded_file:
        selected_file_source = uploaded_file
        file_identifier = uploaded_file.name

if selected_file_source:
    with st.spinner(f"Procesando '{file_identifier}'..."):
        # Cache chain per file session
        if (
            "rag_chain" not in st.session_state
            or st.session_state.get("current_file") != file_identifier
        ):
            retriever = process_file(selected_file_source, filename=file_identifier)
            st.session_state.rag_chain = build_rag_chain(retriever)
            st.session_state.current_file = file_identifier
            st.session_state.messages = []
            st.session_state.chat_history = []

    st.sidebar.success(f"Archivo '{file_identifier}' procesado exitosamente")

    # --- Chat Interface State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display prior chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                user_q = message.get("user_query")
                if (
                    not user_q
                    and idx > 0
                    and st.session_state.messages[idx - 1]["role"] == "user"
                ):
                    user_q = st.session_state.messages[idx - 1]["content"]

                formatted_md = format_response_md(
                    answer=message["content"],
                    user_query=user_q,
                    filename=st.session_state.get("current_file"),
                    sources=message.get("sources"),
                )
                st.download_button(
                    label="📥 Descargar respuesta (.md)",
                    data=formatted_md,
                    file_name=f"respuesta_{idx // 2 + 1}.md",
                    mime="text/markdown",
                    key=f"download_{idx}",
                )

    # User input prompt
    if user_input := st.chat_input("Haz una pregunta sobre el documento..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare chat history for chain
        messages_for_chain = []
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                messages_for_chain.append(HumanMessage(content=m["content"]))
            else:
                messages_for_chain.append(AIMessage(content=m["content"]))

        # Generate response using rag_chain from session_state
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = invocar_agente_con_retry(
                    st.session_state.rag_chain,
                    {"chat_history": messages_for_chain, "input": user_input},
                )
                answer = (
                    response.get("answer") if isinstance(response, dict) else response
                )
                st.markdown(answer)

                sources = []
                if isinstance(response, dict) and response.get("context"):
                    with st.expander("📌 Fuentes citadas / Páginas"):
                        for doc in response["context"]:
                            page_num = doc.metadata.get("page", 1)
                            snippet = doc.page_content[:300] + (
                                "..." if len(doc.page_content) > 300 else ""
                            )
                            sources.append({"page": page_num, "content": snippet})
                            st.write(f"**Página {page_num}:**")
                            st.caption(snippet)

            # Save assistant response to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "user_query": user_input,
                    "sources": sources,
                }
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            assistant_idx = len(st.session_state.messages) - 1
            formatted_md = format_response_md(
                answer=answer,
                user_query=user_input,
                filename=st.session_state.get("current_file"),
                sources=sources,
            )
            st.download_button(
                label="📥 Descargar respuesta (.md)",
                data=formatted_md,
                file_name=f"respuesta_{assistant_idx // 2 + 1}.md",
                mime="text/markdown",
                key=f"download_{assistant_idx}",
            )
else:
    st.info(
        "Por favor, selecciona o sube un archivo (PDF o CSV) desde el panel lateral para empezar."
    )
