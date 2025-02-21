import os
import pandas as pd
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleCSVReader

class Pipeline:
    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.dataframe = None  # Store CSV data
        self.index = None  # Store the Llama Index

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):
        # Configure embeddings and LLM
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # ✅ Corrected file path
        csv_path = r"C:\Users\Jana Sorupaa\Downloads\sample 1000.csv"

        if os.path.exists(csv_path):
            self.dataframe = pd.read_csv(csv_path)
            print("✅ CSV data loaded successfully.")

            # Convert CSV rows into Llama Index format
            self.index = VectorStoreIndex.from_documents(SimpleCSVReader().load_data(csv_path))
        else:
            print(f"❌ Error: CSV file not found at {csv_path}")

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        This function takes a user query, retrieves relevant data from the indexed CSV file, 
        and generates a response using Llama Index and Ollama.
        """
        if self.index is None:
            return "❌ Error: CSV data is not loaded. Please check the file path."

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen
