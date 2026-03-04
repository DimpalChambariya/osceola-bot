import os
import logging
import sys
import streamlit as st

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage
)
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURATION ---
OPENROUTER_API_KEY = "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"
PERSIST_DIR = "./osceola_index_storage"

# Page configuration
st.set_page_config(
    page_title="Osceola County Assistant",
    page_icon="🏛️",
    layout="centered"
)

# Configure Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class OsceolaChatbot:
    def __init__(self, openrouter_api_key: str):
        self.api_key = openrouter_api_key
        self.urls = [
            "https://www.osceola.org/Government/Board-of-County-Commissioners/Citizen-Advisory-Boards-and-Committees/BOCC-Committees",
            "https://www.osceola.org/Government/County-Manager/Executive-Team",
            "https://www.osceola.org/Community/Parks-and-Public-Lands",
            "https://www.osceola.org/Community/About-Osceola-County",
            "https://www.osceola.org/Community/About-Osceola-County/General-Information"
        ]

        self.llm = OpenAI(
            api_key=self.api_key,
            api_base="https://openrouter.ai/api/v1",
            model="gpt-4o-mini",
            additional_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Osceola Chatbot",
            }
        )

        self.embed_model = OpenAIEmbedding(
            api_key=self.api_key,
            api_base="https://openrouter.ai/api/v1",
            model_name="openai/text-embedding-3-small"
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512

    def scrape_and_create_index(self):
        loader = SimpleWebPageReader(html_to_text=True)
        documents = loader.load_data(urls=self.urls)

        for doc, url in zip(documents, self.urls):
            doc.metadata = {"source_url": url}

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

    def load_existing_index(self):
        if os.path.exists(PERSIST_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            return load_index_from_storage(storage_context)
        return None


def main():
    st.title("🏛️ Osceola County Assistant")

    # Sidebar
    with st.sidebar:
        st.header("Knowledge Base")

        if st.button("🔄 Re-scrape & Update"):
            st.session_state.index = None
            if os.path.exists(PERSIST_DIR):
                import shutil
                shutil.rmtree(PERSIST_DIR)
            st.rerun()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.info("The bot reads from 5 Osceola.org pages.")

    if not OPENROUTER_API_KEY or "YOUR_OPENROUTER_KEY" in OPENROUTER_API_KEY:
        st.error("Please configure your OpenRouter API Key.")
        return

    bot = OsceolaChatbot(OPENROUTER_API_KEY)

    # Load or create index
    if "index" not in st.session_state or st.session_state.index is None:
        existing_index = bot.load_existing_index()

        if existing_index:
            st.session_state.index = existing_index
            st.sidebar.success("Loaded index from storage.")
        else:
            with st.spinner("Scraping Osceola.org (Initial Run)..."):
                try:
                    st.session_state.index = bot.scrape_and_create_index()
                    st.success("Knowledge base created and saved!")
                except Exception as e:
                    st.error(f"Init Error: {e}")
                    return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting records..."):
                try:
                    query_engine = st.session_state.index.as_query_engine(similarity_top_k=5)

                    qa_template = PromptTemplate(
                        "Context:\n{context_str}\n\n"
                        "Answer the query: {query_str}\n"
                        "Cite specific Osceola.org URLs in a 'Sources' section."
                    )

                    query_engine.update_prompts({
                        "response_synthesizer:text_qa_template": qa_template
                    })

                    response = query_engine.query(prompt)
                    full_text = str(response)
                    st.markdown(full_text)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text
                    })

                except Exception as e:
                    st.error(f"Processing Error: {e}")


if __name__ == "__main__":
    main()