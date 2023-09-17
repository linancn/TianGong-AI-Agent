import time

import streamlit as st
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

import src.modules.ui.ui_config as ui_config
from src.modules.agents.agent_history import chat_history

ui = ui_config.create_ui_from_config()


# decorator
def enable_chat_history(func):
    if "history" not in st.session_state:
        st.session_state["history"] = chat_history(_session_id=str(time.time()))
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


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state["messages"].append({"role": author, "content": msg})
    st.chat_message(author).markdown(msg)


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
