"""
This agent uses the ReAct framework to determine tools to be use to solve the prompts.
This is the most general purpose action agent.
"""


import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from ..tools.tools import (
    calculation_tool,
)
from ..tools.search_internet_tool import SearchInternetTool

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]


def main_agent():
    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
        callbacks=[],
    )
    tools = [
        SearchInternetTool(),
        calculation_tool(),
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=langchain_verbose,
    )
    return agent
