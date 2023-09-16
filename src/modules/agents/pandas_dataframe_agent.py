"""
This is an agent to interact with a Pandas DataFrame.
This agent calls the Python agent under the hood, which executes LLM generated Python code.
"""

import os

import streamlit as st
from langchain.agents import AgentType, create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

llm_model = "gpt-4"
langchain_verbose = True


def df_agent(df):
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
    )
    return agent
