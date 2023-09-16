"""
This agent uses the ReAct framework to determine tools to be use to solve the prompts.
This is the most general purpose action agent.
"""


import os

import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from src.modules.tools.tools import (
    calculation_tool,
    innovation_assessment_tool,
    search_arxiv_tool,
    search_internet_tool,
    search_uploaded_docs_tool,
    search_vector_database_tool,
    search_wiki_tool,
)

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

llm_model = "gpt-4"
langchain_verbose = True


def main_agent():
    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
    )
    tools = [
        search_vector_database_tool,
        search_internet_tool,
        search_arxiv_tool,
        search_wiki_tool,
        search_uploaded_docs_tool,
        calculation_tool(),
        innovation_assessment_tool,
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=langchain_verbose,
    )
    return agent
