import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# ----------------------------------------------------------------------
# 1. API KEY ------------------------------------------------------------
# ----------------------------------------------------------------------
def get_google_api_key() -> str | None:
    """
    Return the Google API key.
    • Streamlit Cloud → st.secrets["GOOGLE_API_KEY"]
    • Local dev      → os.getenv("GOOGLE_API_KEY")
    """
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key.strip() if key else None


# ----------------------------------------------------------------------
# 2. Document loader ----------------------------------------------------
# ----------------------------------------------------------------------
def get_document_loader(file_path: str):
    """Return the appropriate loader based on file extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    return UnstructuredFileLoader(file_path)


# ----------------------------------------------------------------------
# 3. Summarisation ------------------------------------------------------
# ----------------------------------------------------------------------
def summarize_document(file_path: str, custom_prompt_text: str) -> str | None:
    """Summarize a document using Gemini-1.5-flash via LangChain."""
    api_key = get_google_api_key()
    if not api_key:
        st.error(
            "🔑 **GOOGLE_API_KEY** not found.\n\n"
            "• **Streamlit Cloud**: add it in **Settings → Secrets**.\n"
            "• **Local**: set it in a `.env` file or export it."
        )
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",   # latest fast model (as of 2025)
            temperature=0.3,
            google_api_key=api_key,
        )

        loader = get_document_loader(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=1_000)
        docs = loader.load_and_split(text_splitter=splitter)

        if not docs:
            st.error("Could not extract any text from the document.")
            return None

        st.sidebar.info(f"Document split into **{len(docs)}** chunk(s). Processing…")

        # ---- map / combine prompts ----------------------------------------
        map_prompt = PromptTemplate.from_template(
            f"Summarize this part of the document based on these instructions: "
            f"{custom_prompt_text}\n\n{{text}}"
        )
        combine_prompt = PromptTemplate.from_template(
            f"Combine the following summaries into a final cohesive summary, "
            f"following these instructions: {custom_prompt_text}\n\n{{text}}"
        )

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )

        result = chain.invoke({"input_documents": docs})
        return result["output_text"]

    except Exception as e:
        st.error(f"⚠️ An error occurred: `{e}`")
        return None


# ----------------------------------------------------------------------
# 4. Streamlit UI -------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Document Summarizer", page_icon="📝", layout="wide")

    # initialise session state
    if "summary" not in st.session_state:
        st.session_state.summary = ""

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("📝 AI Document Summarizer")
        st.markdown(
            "Upload a **PDF**, **TXT**, or **MD** file and give a prompt to generate a summary."
        )

        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "txt", "md"], label_visibility="collapsed"
        )
        custom_prompt = st.text_area(
            "Custom prompt",
            height=150,
            placeholder="e.g. Summarize key findings, list main arguments, keep bullet points…",
        )

        col1, col2 = st.columns(2)
        with col1:
            generate_btn = st.button("Generate Summary", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear", use_container_width=True)

    # ---------- Main area ----------
    st.title("Generated Summary")

    if clear_btn:
        st.session_state.summary = ""

    if generate_btn:
        if not uploaded_file:
            st.warning("Please **upload a document**.")
        elif not custom_prompt.strip():
            st.warning("Please **enter a summarisation prompt**.")
        else:
            # Save uploaded file temporarily
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Gemini is reading the document…"):
                st.session_state.summary = summarize_document(temp_path, custom_prompt.strip())

            os.remove(temp_path)

    # ---------- Show result ----------
    if st.session_state.summary:
        st.text_area("Summary", value=st.session_state.summary, height=400, disabled=True)
    else:
        st.info("Upload a file, write a prompt, and click **Generate Summary** to start.")


if __name__ == "__main__":
    main()
