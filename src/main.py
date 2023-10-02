import asyncio
import os
from typing import Any, Dict, List

import streamlit as st
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from langchain.agents import agent
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from pydantic import BaseModel

import src.modules.agents.zero_shot_react_description_agent as zero_shot_react_description_agent

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

app = FastAPI()

current_agent = zero_shot_react_description_agent.main_agent()


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.done.clear()
        if "Action Input" not in self.content:
            self.queue.put_nowait("Thought: ")
        else:
            self.queue.put_nowait("\nThought: ")

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token

        self.queue.put_nowait(token)

        if "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.final_answer = False
            self.done.set()


async def run_call(agent: agent, query: str, stream_it: AsyncCallbackHandler):
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]

    for tool in agent.tools:
        tool.callbacks = [stream_it]

    # now query
    await agent.acall(inputs={"input": query})


# request input format
class Query(BaseModel):
    text: str


async def create_gen(agent: agent, query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(agent, query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task


@app.post("/chat")
async def chat(
    query: Query = Body(...),
):
    stream_it = AsyncCallbackHandler()
    gen = create_gen(current_agent, query.text, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")


@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
