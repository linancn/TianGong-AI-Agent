from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import datetime


llm_chat = Bedrock(
    credentials_profile_name="default",
    model_id="anthropic.claude-v2",
    # streaming=True,
    # verbose=True,
    model_kwargs={
        "max_tokens_to_sample": 1024,
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "anthropic_version": "bedrock-2023-05-31",
    },
    # callbacks=[StreamingStdOutCallbackHandler()],
)

with open("test/prompt1.txt", "r", encoding="utf-8") as file:
    content = file.read()

prompt = content

resp = llm_chat.predict(prompt)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Append timestamp and response to the result file
with open("test/result.txt", "a", encoding="utf-8") as file:
    file.write(f"\n{timestamp}\n\n{resp}\n")
