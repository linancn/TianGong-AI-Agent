import random
import re
import string
import tempfile
import time
from collections import Counter
from datetime import datetime
from io import BytesIO

import pandas as pd
import pdfplumber
import streamlit as st
from gtts import gTTS, gTTSError
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.xata import XataVectorStore
from xata.client import XataClient

from ..agents.memory.agent_history import xata_chat_history
from . import ui_config

ui = ui_config.create_ui_from_config()


def random_email(domain="example.com"):
    # username length is 5 to 10
    username_length = random.randint(5, 10)
    username = "".join(
        random.choice(string.ascii_lowercase + string.digits)
        for _ in range(username_length)
    )

    return f"{username}@{domain}"


# decorator
def enable_chat_history(func):
    if "history" not in st.session_state:
        st.session_state["history"] = xata_chat_history(_session_id=str(time.time()))
    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "avatar": ui.chat_ai_avatar,
                "content": ui.chat_ai_welcome,
            }
        ]
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"], avatar=msg["avatar"]).markdown(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


# ##############################
# detect_text_language
# ##############################
def gTTS_text_language_func_calling_chain():
    func_calling_json_schema = {
        "title": "identify_text_language",
        "description": "Accurately identifying the language of the text.",
        "type": "object",
        "properties": {
            "language": {
                "title": "Language",
                "description": "The accurate language of the text",
                "type": "string",
                "enum": [
                    "af",
                    "ar",
                    "bg",
                    "bn",
                    "bs",
                    "ca",
                    "cs",
                    "da",
                    "de",
                    "el",
                    "en",
                    "es",
                    "et",
                    "fi",
                    "fr",
                    "gu",
                    "hi",
                    "hr",
                    "hu",
                    "id",
                    "is",
                    "it",
                    "iw",
                    "ja",
                    "jw",
                    "km",
                    "kn",
                    "ko",
                    "la",
                    "lv",
                    "ml",
                    "mr",
                    "ms",
                    "my",
                    "ne",
                    "nl",
                    "no",
                    "pl",
                    "pt",
                    "ro",
                    "ru",
                    "si",
                    "sk",
                    "sq",
                    "sr",
                    "su",
                    "sv",
                    "sw",
                    "ta",
                    "te",
                    "th",
                    "tl",
                    "tr",
                    "uk",
                    "ur",
                    "vi",
                    "zh-CN",
                    "zh-TW",
                    "zh",
                ],
            }
        },
        "required": ["language"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="""You are a world class algorithm for accurately identifying the language of the text, strictly follow the language mapping: {"af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian", "fi": "Finnish", "fr": "French", "gu": "Gujarati", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "iw": "Hebrew", "ja": "Japanese", "jw": "Javanese", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin", "lv": "Latvian", "ml": "Malayalam", "mr": "Marathi", "ms": "Malay", "my": "Myanmar (Burmese)", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Filipino", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh-CN": "Chinese (Simplified)", "zh-TW": "Chinese (Mandarin/Taiwan)", "zh": "Chinese (Mandarin)"}"""
        ),
        HumanMessage(content="The text:"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    llm_func_calling = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=False)

    func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=False,
    )

    return func_calling_chain


def show_audio_player(ai_content: str) -> None:
    query_response = gTTS_text_language_func_calling_chain().run(ai_content)
    language = query_response.get("language")
    sound_file = BytesIO()
    try:
        tts = gTTS(text=ai_content, lang=language)
        tts.write_to_fp(sound_file)
        st.audio(sound_file)
    except gTTSError as err:
        st.error(err)


def fetch_embedding_files(username: str, session_id: str) -> pd.DataFrame:
    """Fetch the embedding files."""
    client = XataClient()
    query = """SELECT DISTINCT "source" FROM "tiangong_chunks" WHERE "username" = $1 AND "sessionId" = $2 ORDER BY "source" ASC"""
    response = client.sql().query(statement=query, params=(username, session_id))

    df = pd.DataFrame(response["records"])

    return df


def clear_embedding_files(username: str, session_id: str) -> pd.DataFrame:
    """Clear the embedding files."""
    client = XataClient()
    query = (
        """DELETE FROM "tiangong_chunks" WHERE "username" = $1 AND "sessionId" = $2"""
    )
    response = client.sql().query(statement=query, params=(username, session_id))

    return response


def delete_embedding_files(username: str, session_id: str, options: list):
    """Delete the embedding files."""
    client = XataClient()

    placeholders = ",".join([f"${i+3}" for i, _ in enumerate(options)])
    query = f"""DELETE FROM "tiangong_chunks" WHERE "username" = $1 AND "sessionId" = $2 AND "source" IN ({placeholders})"""
    params = (username, session_id) + tuple(options)
    response = client.sql().query(statement=query, params=params)

    return response


def is_valid_email(email: str) -> bool:
    """
    Check if the given string is a valid email address.

    Args:
    - email (str): String to check.

    Returns:
    - bool: True if valid email, False otherwise.
    """
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def fetch_chat_history(username: str):
    """Fetch the chat history."""
    if is_valid_email(username):
        client = XataClient()
        # No parameterized query because the Xata Client.
        response = client.sql().query(
            f"""SELECT "sessionId", "content"
    FROM (
        SELECT DISTINCT ON ("sessionId") "sessionId", "xata.createdAt", "content"
        FROM "tiangong_memory"
        WHERE "additionalKwargs"->>'id' = '{username}'
        ORDER BY "sessionId" DESC, "xata.createdAt" ASC
    ) AS subquery"""
        )
        records = response["records"]
        for record in records:
            timestamp = float(record["sessionId"])
            record["entry"] = (
                datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                + " : "
                + record["content"]
            )

        table_map = {item["sessionId"]: item["entry"] for item in records}

        return table_map
    else:
        return {}


def delete_chat_history(session_id):
    """Delete the chat history by session_id."""
    client = XataClient()
    client.sql().query(
        'DELETE FROM "tiangong_memory" WHERE "sessionId" = $1',
        [session_id],
    )


def convert_history_to_message(history):
    if isinstance(history, HumanMessage):
        return {
            "role": "user",
            "avatar": ui.chat_user_avatar,
            "content": history.content,
        }
    elif isinstance(history, AIMessage):
        return {
            "role": "assistant",
            "avatar": ui.chat_ai_avatar,
            "content": history.content,
        }


def initialize_messages(history):
    # 将历史消息转换为消息格式
    messages = [convert_history_to_message(message) for message in history]

    # 在最前面加入欢迎消息
    welcome_message = {
        "role": "assistant",
        "avatar": ui.chat_ai_avatar,
        "content": ui.chat_ai_welcome,
    }
    messages.insert(0, welcome_message)

    return messages


def parse_paper(pdf_stream):
    # logging.info("Parsing paper")
    pdf_obj = pdfplumber.open(pdf_stream)
    number_of_pages = len(pdf_obj.pages)
    # logging.info(f"Total number of pages: {number_of_pages}")
    full_text = ""
    ismisc = False
    for i in range(number_of_pages):
        page = pdf_obj.pages[i]
        if i == 0:
            isfirstpage = True
        else:
            isfirstpage = False

        page_text = []
        sentences = []
        processed_text = []

        def visitor_body(text, isfirstpage, x, top, bottom, fontSize, ismisc):
            # ignore header/footer
            if isfirstpage:
                if (top > 200 and bottom < 720) and (len(text.strip()) > 1):
                    sentences.append(
                        {
                            "fontsize": fontSize,
                            "text": " " + text.strip().replace("\x03", ""),
                            "x": x,
                            "y": top,
                        }
                    )
            else:  # not first page
                if (
                    (top > 70 and bottom < 720)
                    and (len(text.strip()) > 1)
                    and not ismisc
                ):  # main text region
                    sentences.append(
                        {
                            "fontsize": fontSize,
                            "text": " " + text.strip().replace("\x03", ""),
                            "x": x,
                            "y": top,
                        }
                    )
                elif (top > 70 and bottom < 720) and (len(text.strip()) > 1) and ismisc:
                    pass

        extracted_words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            horizontal_ltr=True,
            vertical_ttb=True,
            extra_attrs=["fontname", "size"],
            split_at_punctuation=False,
        )

        # Treat the first page, main text, and references differently, specifically targeted at headers
        # Define a list of keywords to ignore
        # Online is for Nauture papers
        keywords_for_misc = [
            "References",
            "REFERENCES",
            "Bibliography",
            "BIBLIOGRAPHY",
            "Acknowledgements",
            "ACKNOWLEDGEMENTS",
            "Acknowledgments",
            "ACKNOWLEDGMENTS",
            "Acknowledgement",
            "参考文献",
            "致谢",
            "謝辞",
            "謝",
            "Online",
        ]

        prev_word_size = None
        prev_word_font = None
        # Loop through the extracted words
        for extracted_word in extracted_words:
            # Strip the text and remove any special characters
            text = extracted_word["text"].strip().replace("\x03", "")

            # Check if the text contains any of the keywords to ignore
            if any(keyword in text for keyword in keywords_for_misc) and (
                prev_word_size != extracted_word["size"]
                or prev_word_font != extracted_word["fontname"]
            ):
                ismisc = True

            prev_word_size = extracted_word["size"]
            prev_word_font = extracted_word["fontname"]

            # Call the visitor_body function with the relevant arguments
            visitor_body(
                text,
                isfirstpage,
                extracted_word["x0"],
                extracted_word["top"],
                extracted_word["bottom"],
                extracted_word["size"],
                ismisc,
            )

        if sentences:
            for sentence in sentences:
                page_text.append(sentence)

        blob_font_sizes = []
        blob_font_size = None
        blob_text = ""
        processed_text = ""
        tolerance = 1

        # Preprocessing for main text font size
        if page_text != []:
            if len(page_text) == 1:
                blob_font_sizes.append(page_text[0]["fontsize"])
            else:
                for t in page_text:
                    blob_font_sizes.append(t["fontsize"])
            blob_font_size = Counter(blob_font_sizes).most_common(1)[0][0]

        if page_text != []:
            if len(page_text) == 1:
                if (
                    blob_font_size - tolerance
                    <= page_text[0]["fontsize"]
                    <= blob_font_size + tolerance
                ):
                    processed_text += page_text[0]["text"]
                    # processed_text.append({"text": page_text[0]["text"], "page": i + 1})
            else:
                for t in range(len(page_text)):
                    if (
                        blob_font_size - tolerance
                        <= page_text[t]["fontsize"]
                        <= blob_font_size + tolerance
                    ):
                        blob_text += f"{page_text[t]['text']}"
                        if len(blob_text) >= 500:  # set the length of a data chunk
                            processed_text += blob_text
                            # processed_text.append({"text": blob_text, "page": i + 1})
                            blob_text = ""
                        elif t == len(page_text) - 1:  # last element
                            processed_text += blob_text
                            # processed_text.append({"text": blob_text, "page": i + 1})
            full_text += processed_text

    # logging.info("Done parsing paper")
    return full_text


def get_xata_db(uploaded_files):
    xata_api_key = st.secrets["xata_api_key"]
    xata_db_url = st.secrets["xata_db_url"]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=200, chunk_overlap=10
    )
    chunks = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                fp.write(uploaded_file.read())
                if uploaded_file.type == 'application/pdf':
                    full_text = parse_paper(fp)
                else:
                    loader = UnstructuredFileLoader(file_path=fp.name)
                    docs = loader.load()
                    full_text = docs[0].page_content

            chunk = text_splitter.create_documents(
                texts=[full_text],
                metadatas=[
                    {
                        "source": uploaded_file.name,
                        "username": st.session_state["username"],
                        "sessionId": st.session_state["selected_chat_id"],
                    }
                ],
            )
            chunks.extend(chunk)
        except:
            pass

    if chunks != []:
        embeddings = OpenAIEmbeddings()
        vector_store = XataVectorStore.from_documents(
            chunks,
            embeddings,
            api_key=xata_api_key,
            db_url=xata_db_url,
            table_name="tiangong_chunks",
        )
    else:
        st.warning(ui.sidebar_file_uploader_error)
        st.stop()

    return vector_store
