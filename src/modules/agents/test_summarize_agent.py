import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from ..tools.test_summarize_tool import SummarizeTool
# from ..tools.test_outline_tool import ReviewOutlineTool


def main_agent():
    llm_model = st.secrets["llm_model"]
    langchain_verbose = st.secrets["langchain_verbose"]

    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
        callbacks=[],
    )

    tools = [
            # ReviewOutlineTool(),
             SummarizeTool()
            ]

    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=langchain_verbose,
    )

    return agent
