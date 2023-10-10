import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from ..tools.test_summarize_tool import SummarizeTool


def main_agent():
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
        # verbose=langchain_verbose,
        openai_api_key=openrouter_api_key,
        openai_api_base=openrouter_api_base,
        headers={"HTTP-Referer": "http://localhost"},
        # callbacks=[StreamingStdOutCallbackHandler],
    )

    tools = [SummarizeTool()]

    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=langchain_verbose,
    )

    return agent
