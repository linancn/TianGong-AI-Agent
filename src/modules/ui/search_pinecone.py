import asyncio
import datetime
import os

import pinecone
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


class SearchPinecone:
    def __init__(self):
        os.environ["PINECONE_API_KEY"] = st.secrets["pinecone_api_key"]
        os.environ["PINECONE_ENVIRONMENT"] = st.secrets["pinecone_environment"]
        os.environ["PINECONE_INDEX"] = st.secrets["pinecone_index"]
        os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
        self.embeddings = OpenAIEmbeddings()
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        self.vectorstore = Pinecone.from_existing_index(
            index_name=os.environ["PINECONE_INDEX"],
            embedding=self.embeddings,
        )

    async def async_similarity(
        self, query: str, filters: dict = {}, top_k: int = 16
    ):
        """Search Pinecone with similarity score."""
        if top_k == 0:
            return []

        if filters:
            docs = self.vectorstore.similarity_search(query, k=top_k, filter=filters)
        else:
            docs = self.vectorstore.similarity_search(query, k=top_k)

        docs_list = []
        for doc in docs:
            date = datetime.datetime.fromtimestamp(doc.metadata["created_at"])
            formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
            source_entry = "[{}. {}. {}. {}.]({})".format(
                doc.metadata["source_id"],
                doc.metadata["source"],
                doc.metadata["author"],
                formatted_date,
                doc.metadata["url"],
            )
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    def sync_similarity(self, query: str, filters: dict = {}, top_k: int = 16):
        return asyncio.run(self.async_similarity(query, filters, top_k))

    async def async_mmr(
        self, query: str, filters: dict = {}, top_k: int = 16
    ):
        """Search Pinecone with maximal marginal relevance method."""
        if top_k == 0:
            return []

        if filters:
            docs = self.vectorstore.max_marginal_relevance_search(query, k=top_k, filter=filters)
        else:
            docs = self.vectorstore.max_marginal_relevance_search(query, k=top_k)

        docs_list = []
        for doc in docs:
            date = datetime.datetime.fromtimestamp(doc.metadata["created_at"])
            formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
            source_entry = "[{}. {}. {}. {}.]({})".format(
                doc.metadata["source_id"],
                doc.metadata["source"],
                doc.metadata["author"],
                formatted_date,
                doc.metadata["url"],
            )
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    def sync_mmr(self, query: str, filters: dict = {}, top_k: int = 16):
        return asyncio.run(self.async_mmr(query, filters, top_k))
    
    def get_contentslist(self, docs):
        """Get a list of contents from docs."""   
        contents = [
                [item["content"] for item in sublist] for sublist in docs
            ]
        return contents