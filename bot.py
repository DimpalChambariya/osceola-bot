import os
import logging
import shutil
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage
)
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- CONFIGURATION ---
# Replace with your actual key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here") 
PERSIST_DIR = "./osceola_index_storage"

app = FastAPI(title="Osceola County RAG API")

# Setup Logging
logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    query: str

class OsceolaEngine:
    def __init__(self):
        self.urls = [
            "https://www.osceola.org/Government/Board-of-County-Commissioners/Citizen-Advisory-Boards-and-Committees/BOCC-Committees",
            "https://www.osceola.org/Government/County-Manager/Executive-Team",
            "https://www.osceola.org/Community/Parks-and-Public-Lands",
            "https://www.osceola.org/Community/About-Osceola-County",
            "https://www.osceola.org/Community/About-Osceola-County/General-Information"
        ]
        
        # Configure LlamaIndex to use OpenRouter
        Settings.llm = OpenAI(
            api_key=OPENROUTER_API_KEY,
            api_base="https://openrouter.ai/api/v1",
            model="gpt-4o-mini"
        )
        
        Settings.embed_model = OpenAIEmbedding(
            api_key=OPENROUTER_API_KEY,
            api_base="https://openrouter.ai/api/v1",
            model_name="openai/text-embedding-3-small"
        )
        
        self.index = self._get_index()

    def _get_index(self):
        """Loads index from disk or scrapes if missing."""
        if os.path.exists(PERSIST_DIR):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                return load_index_from_storage(storage_context)
            except Exception as e:
                logging.error(f"Error loading index: {e}")
        
        # Initial Scrape
        logging.info("Indexing Osceola.org...")
        loader = SimpleWebPageReader(html_to_text=True)
        documents = loader.load_data(urls=self.urls)
        for doc, url in zip(documents, self.urls):
            doc.metadata = {"source_url": url}
        
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

    def query(self, user_query: str):
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        qa_template = PromptTemplate(
            "Context:\n{context_str}\n\nAnswer the query: {query_str}\n"
            "Cite specific Osceola.org URLs in a 'Sources' section."
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})
        return str(query_engine.query(user_query))

# Initialize engine on startup
engine = OsceolaEngine()

@app.post("/search")
async def search_osceola(request: QueryRequest):
    """
    Search Osceola County records for information.
    """
    try:
        response = engine.query(request.query)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/refresh")
async def refresh_data():
    """Forces a re-scrape of the websites."""
    global engine
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    engine = OsceolaEngine()
    return {"status": "Knowledge base refreshed"}