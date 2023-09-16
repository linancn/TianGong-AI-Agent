import importlib
import os
from dataclasses import dataclass

import toml

ui_config_file = f"""src.ui.{os.environ.get("ui",default="tiangong-en")}"""
config_module = importlib.import_module(ui_config_file)
ui_data = config_module.ui_data


@dataclass
class Theme:
    base: str
    primaryColor: str
    backgroundColor: str
    secondaryBackgroundColor: str
    textColor: str
    font: str


@dataclass
class UI:
    need_passwd: bool
    theme: Theme
    page_title: str
    page_icon: str
    page_markdown: str
    sidebar_image: str
    sidebar_title: str
    sidebar_subheader: str
    sidebar_markdown: str
    sidebar_expander_title: str
    txt2audio_checkbox_label: str
    chat_memory_checkbox_label: str
    upload_docs_checkbox_label: str
    sidebar_file_uploader_title: str
    sidebar_file_uploader_spinner: str
    sidebar_file_uploader_error: str
    chat_ai_avatar: str
    chat_ai_welcome: str
    chat_ai_spinner: str
    chat_human_placeholder: str


def create_ui_from_config():
    # 读取文件
    with open(".streamlit/config.toml", "r") as file:
        data = toml.load(file)

    # 替换内容
    if data["theme"] != ui_data["theme"]:
        data["theme"] = ui_data["theme"]
        # 写入文件
        with open(".streamlit/config.toml", "w") as file:
            toml.dump(data, file)

    return UI(**ui_data)
