import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
# Corrected import: We need the concrete implementation from .map_reduce, not the .base abstract class
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from typing import List, Optional


def get_document_loader(file_path: str) -> BaseLoader:
    """Selects the appropriate document loader based on the file extension."""
    _, extension = os.path.splitext(file_path)
    if extension.lower() == ".pdf":
        return PyPDFLoader(file_path)
    else:
        # UnstructuredFileLoader handles .txt, .md, and many other types
        return UnstructuredFileLoader(file_path)


def summarize_document(file_path: str, custom_prompt_text: str) -> Optional[str]:
    """
    Summarizes the document at file_path using the provided custom prompt.
    """
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in Streamlit secrets. Please add your key.")
        return None

    try:
        # Use the latest recommended Gemini Flash model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.3,
            google_api_key=api_key,
        )

        loader = get_document_loader(file_path)
        text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs: List[Document] = loader.load_and_split(text_splitter=text_splitter)

        if not docs:
            st.error("Could not extract any text from the document.")
            return None

        st.sidebar.info(f"Document split into {len(docs)} chunk(s). Processing...")

        # Define prompts for map and combine
        map_prompt_template = PromptTemplate(
            input_variables=["text"],
            # Use f-string correctly with double braces for langchain variables
            template=f"Summarize this part based on instructions: {custom_prompt_text}\n\n{{text}}"
        )
        combine_prompt_template = PromptTemplate(
            input_variables=["text"],
            template=f"Combine the following summaries into a final cohesive summary, following these instructions: {custom_prompt_text}\n\n{{text}}"
        )

        # Create LLMChains for map and reduce
        map_chain = LLMChain(llm=llm, prompt=map_prompt_template)
        combine_chain = LLMChain(llm=llm, prompt=combine_prompt_template)

        # Use the correctly imported MapReduceDocumentsChain
        chain = MapReduceDocumentsChain(
            llm_chain=map_chain,  # Use llm_chain for the map step
            combine_document_chain=combine_chain,  # Use combine_document_chain for the reduce step
            combine_document_chain_kwargs={"input_key": "text"}, # Pass the combined map outputs as "text" to the combine_prompt
            document_variable_name="text", # The variable name for the doc content in map_prompt
        )

        # Use the 'invoke' method (preferred over 'run')
        # The input must be a dictionary with the key 'input_documents'
        result_dict = chain.invoke({"input_documents": docs})

        # The final summary is in the 'output_text' key of the result
        return result_dict.get("output_text")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


def main():
    st.set_page_config(page_title="AI Document Summarizer", page_icon="📝", layout="wide")

    if "summary" not in st.session_state:
        st.session_state.summary = ""

    with st.sidebar:
        st.header("📝 AI Document Summarizer")
        st.markdown("Upload a `.pdf`, `.txt`, or `.md` file and provide a prompt to generate a summary.")

        uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "md"])
        custom_prompt = st.text_area("Enter your custom prompt", height=150,
                                    placeholder="Example: Summarize the key findings and action items from this report.")

        col1, col2 = st.columns(2)
        with col1:
            generate_button = st.button("Generate Summary", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)

    st.title("Generated Summary")

    if clear_button:
        st.session_state.summary = ""
        st.rerun()

    if generate_button:
        if uploaded_file and custom_prompt.strip():
            # Create a temporary directory if it doesn't exist
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            # Save the uploaded file temporarily
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("🧠 Gemini is analyzing the document..."):
                    st.session_state.summary = summarize_document(temp_file_path, custom_prompt)
            
            except Exception as e:
                st.error(f"Error saving or processing file: {e}")
            
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            st.warning("Please upload a document and provide a summarization prompt.")

    if st.session_state.summary:
        st.text_area("Summary", value=st.session_state.summary, height=400, key="summary_output")
    else:
        st.info("Upload a document and enter a prompt to get started.")


if __name__ == "__main__":
    main()
