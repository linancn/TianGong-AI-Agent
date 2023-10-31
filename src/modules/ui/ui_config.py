import importlib
import os
from dataclasses import dataclass

import toml

ui_config_file = f"""ui.{os.environ.get("ui",default="tiangong-en")}"""
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
    sidebar_embedded_files_title: str
    sidebar_clear_all_files_button_label: str
    sidebar_delete_file_button_label: str
    sidebar_chat_title: str
    sidebar_newchat_button_label: str
    sidebar_deletechat_button_label: str
    chat_ai_avatar: str
    chat_user_avatar: str
    chat_ai_welcome: str
    chat_ai_spinner: str
    chat_human_placeholder: str


def create_ui_from_config():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.abspath(
        os.path.join(current_dir, "../../../.streamlit/config.toml")
    )
    # 读取文件
    with open(config_path, "r") as file:
        data = toml.load(file)

    # 替换内容
    if data["theme"] != ui_data["theme"]:
        data["theme"] = ui_data["theme"]
        # 写入文件
        with open(config_path, "w") as file:
            toml.dump(data, file)

    return UI(**ui_data)
