import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# Load environment variables
load_dotenv()


def get_document_loader(file_path: str):
    """Return the appropriate document loader based on file extension."""
    _, extension = os.path.splitext(file_path)
    if extension.lower() == ".pdf":
        return PyPDFLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path)


def summarize_document(file_path: str, custom_prompt_text: str) -> str | None:
    """Summarize a document using Gemini 2.5 Flash model via LangChain."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please add your key.")
        return None

    try:
        # Use the gemini-2.5-flash model for better throughput
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key,
        )

        loader = get_document_loader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs_chunks = loader.load_and_split(text_splitter=text_splitter)

        if not docs_chunks:
            st.error("Could not extract text from the document.")
            return None

        st.sidebar.info(f"Document split into {len(docs_chunks)} chunk(s). Processing...")

        # Define summarization prompts
        map_prompt_template = (
            f"Summarize this part of the document based on these instructions: "
            f"{custom_prompt_text}\n\n{{text}}"
        )
        map_prompt = PromptTemplate.from_template(map_prompt_template)

        combine_prompt_template = (
            f"Combine the following summaries into a final cohesive summary, "
            f"following these instructions: {custom_prompt_text}\n\n{{text}}"
        )
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )

        result = chain.invoke({"input_documents": docs_chunks})
        return result['output_text']

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


def main():
    """Streamlit Document Summarizer App."""
    st.set_page_config(page_title="AI Document Summarizer", page_icon="📝", layout="wide")

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    with st.sidebar:
        st.header("📝 AI Document Summarizer")
        st.markdown("Upload a `.pdf`, `.txt`, or `.md` file and provide a prompt to generate a summary.")

        uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "md"])
        custom_prompt = st.text_area("Enter your custom prompt", height=150,
                                     placeholder="Example: Summarize key findings...")

        col1, col2 = st.columns(2)
        with col1:
            generate_button = st.button("Generate Summary", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)

    st.title("Generated Summary")

    if clear_button:
        st.session_state.summary = ""

    if generate_button:
        if uploaded_file and custom_prompt.strip():
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🧠 Gemini is analyzing the document..."):
                st.session_state.summary = summarize_document(temp_file_path, custom_prompt)

            os.remove(temp_file_path)
        else:
            st.warning("Please upload a document and provide a summarization prompt.")

    if st.session_state.summary:
        st.text_area("Summary", value=st.session_state.summary, height=400)
    else:
        st.info("Upload a document and enter a prompt to get started.")


if __name__ == "__main__":
    main()
