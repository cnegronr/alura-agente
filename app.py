import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import modular tools and functions from tools.py
from tools import process_pdf, build_rag_chain, invocar_agente_con_retry

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄")
st.title("📄 PDF Alura Agent")

# Ensure GROQ_API_KEY is present
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()

# --- UI Sidebar for Upload ---
st.sidebar.header("Document Setup")
uploaded_file = st.sidebar.file_uploader("Subir un archivo PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Procesando el archivo PDF..."):
        # Cache chain per file session
        if "rag_chain" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            retriever = process_pdf(uploaded_file)
            st.session_state.rag_chain = build_rag_chain(retriever)
            st.session_state.current_file = uploaded_file.name

    st.sidebar.success("PDF procesado exitosamente")

    # --- Chat Interface State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display prior chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
                    {"chat_history": messages_for_chain, "input": user_input}
                )
                answer = response.get("answer") if isinstance(response, dict) else response
                st.markdown(answer)

                if isinstance(response, dict) and response.get("context"):
                    with st.expander("📌 Fuentes citadas / Páginas"):
                        for doc in response["context"]:
                            page_num = doc.metadata.get("page", 1)
                            st.write(f"**Página {page_num}:**")
                            st.caption(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Por favor, sube un archivo PDF desde el panel lateral para empezar.")
