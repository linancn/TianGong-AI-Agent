from typing import Optional, Type

import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from pydantic import BaseModel
from xata.client import XataClient
from langchain.schema.document import Document


class SummarizeTool(BaseTool):
    name = "summarize_tool"
    description = "Return information of the original query from uploaded documents, this tool will automatically fetch the uploaded documents."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def main_chain(self):
        langchain_verbose = st.secrets["langchain_verbose"]
        openrouter_api_key = st.secrets["openrouter_api_key"]
        openrouter_api_base = st.secrets["openrouter_api_base"]

        selected_model = "anthropic/claude-2"
        # selected_model = "openai/gpt-3.5-turbo-16k"
        # selected_model = "openai/gpt-4-32k"
        # selected_model = "meta-llama/llama-2-70b-chat"

        llm_chat = ChatOpenAI(
            model_name=selected_model,
            # temperature=0,
            streaming=True,
            verbose=langchain_verbose,
            openai_api_key=openrouter_api_key,
            openai_api_base=openrouter_api_base,
            headers={"HTTP-Referer": "http://localhost"},
            # callbacks=[],
        )

        chain = load_summarize_chain(llm_chat, chain_type="stuff")

        return chain

    def fetch_uploaded_docs_vector(self, query: str, k: int = 16) -> list[Document]:
        """Fetch uploaded docs in similarity search."""
        username = st.session_state["username"]
        session_id = st.session_state["selected_chat_id"]
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed_query(query)
        results = (
            XataClient()
            .data()
            .vector_search(
                "tiangong_chunks",
                {
                    "queryVector": query_vector,  # array of floats
                    "column": "embedding",  # column name,
                    "similarityFunction": "cosineSimilarity",  # space function
                    "size": k,  # number of results to return
                    "filter": {
                        "username": username,
                        "sessionId": session_id,
                    },  # filter expression
                },
            )
        )
        docs = []
        for record in results["records"]:
            page_content = record["content"]
            metadata = {
                "source": record["source"],
            }
            doc = Document(
                page_content=page_content,
                metadata=metadata,
            )
            docs.append(doc)

        return docs

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""

        chain = self.main_chain()

        docs = self.fetch_uploaded_docs_vector(query="material flow", k=5)
        response = chain.run(docs)

        return response

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
