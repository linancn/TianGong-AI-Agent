import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from ..tools.testreview_tool_with_outlines import ReviewToolWithDetailedOutlines
# from ..tools.review_tool_without_outlines import ReviewToolWithoutOutlines

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
        # ReviewToolWithoutOutlines(),
        ReviewToolWithDetailedOutlines(),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=langchain_verbose,
    )

    return agent
