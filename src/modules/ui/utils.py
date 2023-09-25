import time
from datetime import datetime
from io import BytesIO

import streamlit as st
from gtts import gTTS, gTTSError
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from xata.client import XataClient

import src.modules.ui.ui_config as ui_config
from src.modules.agents.agent_history import xata_chat_history

ui = ui_config.create_ui_from_config()


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
        if msg["role"] == "user":
            st.chat_message(msg["role"]).markdown(msg["content"])
        elif msg["role"] == "assistant":
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


def fetch_chat_history():
    """Fetch the chat history."""
    client = XataClient()
    response = client.sql().query(
        'SELECT "sessionId", "content" FROM (SELECT DISTINCT ON ("sessionId") "sessionId", "xata.createdAt", "content" FROM "tiangong_memory" ORDER BY "sessionId", "xata.createdAt" ASC, "content" ASC) AS subquery ORDER BY "xata.createdAt" DESC'
    )
    records = response["records"]
    for record in records:
        timestamp = float(record["sessionId"])
        record["entry"] = (
            datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            + " - "
            + record["content"]
        )

    table_map = {item["sessionId"]: item["entry"] for item in records}

    return table_map


def delete_chat_history(session_id):
    """Delete the chat history by session_id."""
    client = XataClient()
    client.sql().query(
        'DELETE FROM "tiangong_memory" WHERE "sessionId" = $1',
        [session_id],
    )


def convert_history_to_message(history):
    if isinstance(history, HumanMessage):
        return {"role": "user", "content": history.content}
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
