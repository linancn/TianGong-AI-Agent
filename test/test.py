import requests


def get_stream(query: str):
    s = requests.Session()
    buffer = b""

    # Sending a POST request to the server and streaming the response
    with s.post("http://localhost:8000/chat", stream=True, json={"text": query}) as r:
        for chunk in r.iter_content(
            chunk_size=192
        ):  # Using a larger chunk size, e.g., 192 bytes
            buffer += chunk
            try:
                text = buffer.decode("utf-8")
                print(text, end="")
                buffer = b""
            except UnicodeDecodeError:
                # We will stop here and wait for more data so we can decode safely
                pass

    # If there's still data in the buffer (i.e., the last chunk), make sure to process it
    if buffer:
        print(buffer.decode("utf-8", "replace"), end="")


get_stream("什么是物质流分析，参考数据库中2020年以来在RCR期刊里面的文章内容来回答。")
