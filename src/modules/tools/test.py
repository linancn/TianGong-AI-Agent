import asyncio
import os
import streamlit as st
import modules.tools.review_tool_with_outlines as review_tool_with_outlines

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


async def call_arun(tool, text):
    result = await tool.arun(text)
    return result


summarize_tool = review_tool_with_outlines.SummarizeTool()
result = asyncio.run(
    call_arun(
        summarize_tool,
        """Please review the Dynamic Material Flow Models according to the following story line.
# 1. Introduction to Dynamic Material Flow Models
# 2. Classification of Dynamic Material Flow Models, including classification based on Top-down/Bottom-up approaches, and based on Retrospective/Prospective categories.
# 3. Discussion of Each Category of Models, delineating the type of problems they address, application scenarios (citing example literature), and specific computational methods.
# 4. Summary of Advantages and Limitations of Various Dynamic Material Flow Models, comparative analysis and discussion on future trends.
# It must be longer than 1000 tokens.""",
    )
)

# summarize_tool.arun(
#     """Please review the Dynamic Material Flow Models according to the following story line.
# 1. Introduction to Dynamic Material Flow Models
# 2. Classification of Dynamic Material Flow Models, including classification based on Top-down/Bottom-up approaches, and based on Retrospective/Prospective categories.
# 3. Discussion of Each Category of Models, delineating the type of problems they address, application scenarios (citing example literature), and specific computational methods.
# 4. Summary of Advantages and Limitations of Various Dynamic Material Flow Models, comparative analysis and discussion on future trends.
# It must be longer than 1000 tokens."""
# )
