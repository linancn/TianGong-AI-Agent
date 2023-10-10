from typing import Optional, Type

import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.vectorstores.xata import XataVectorStore
from pydantic import BaseModel
from xata.client import XataClient
from langchain.embeddings import OpenAIEmbeddings



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

    def fetch_uploaded_docs(self):
        """Fetch uploaded docs."""
        username = st.session_state["username"]
        session_id = st.session_state["selected_chat_id"]
        client = XataClient()
        query = """SELECT content FROM "tiangong_chunks" WHERE "username" = $1 AND "sessionId" = $2"""
        response = client.sql().query(statement=query, params=(username, session_id))

        return response

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""

        chain = self.main_chain()

        xata_api_key = st.secrets["xata_api_key"]
        xata_db_url = st.secrets["xata_db_url"]
        embeddings = OpenAIEmbeddings()
        table_name="tiangong_chunks"

        vector_store = XataVectorStore(
            api_key=xata_api_key,
            db_url=xata_db_url,
            embedding=embeddings,
            table_name=table_name,
        )

        docs = vector_store.similarity_search("material flow", k=80)
        response = chain.run(docs)

        return response

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
