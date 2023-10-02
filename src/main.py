import os
import asyncio
from typing import Any

import uvicorn
import streamlit as st
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import agent
from langchain.schema import LLMResult

import src.modules.agents.zero_shot_react_description_agent as zero_shot_react_description_agent


os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

app = FastAPI()

current_agent = zero_shot_react_description_agent.main_agent()


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        
        if "Action Input" in self.content:
             self.queue.put_nowait("\n")

        self.queue.put_nowait(token)

        if "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


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
