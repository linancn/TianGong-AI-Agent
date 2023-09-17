from io import BytesIO

import streamlit as st
from gtts import gTTS, gTTSError
from langchain.callbacks import StreamlitCallbackHandler

import src.modules.tools.tools as tools
import src.modules.ui.ui_config as ui_config
import src.modules.ui.utils as utils
from src.modules.agents.agent_selector import main_agent

ui = ui_config.create_ui_from_config()

st.set_page_config(page_title=ui.page_title, layout="wide", page_icon=ui.page_icon)


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


def show_audio_player(ai_content: str) -> None:
    query_response = utils.gTTS_text_language_func_calling_chain().run(ai_content)
    language = query_response.get("language")
    sound_file = BytesIO()
    try:
        tts = gTTS(text=ai_content, lang=language)
        tts.write_to_fp(sound_file)
        st.audio(sound_file)
    except gTTSError as err:
        st.error(err)


if ui.need_passwd is False:
    auth = True
else:
    auth = check_password()


if auth:
    # if True:
    # 注入CSS style, 修改最上渐变条颜色
    st.markdown(
        ui.page_markdown,
        unsafe_allow_html=True,
    )

    # SIDEBAR
    with st.sidebar:
        st.markdown(
            ui.sidebar_markdown,
            unsafe_allow_html=True,
        )
        col_image, col_text = st.columns([1, 4])
        with col_image:
            st.image(ui.sidebar_image)
        with col_text:
            st.title(ui.sidebar_title)
        st.subheader(ui.sidebar_subheader)

        with st.expander(ui.sidebar_expander_title):
            txt2audio = st.checkbox(ui.txt2audio_checkbox_label, value=False)
            chat_memory = st.checkbox(ui.chat_memory_checkbox_label, value=False)
            search_docs = st.checkbox(ui.upload_docs_checkbox_label, value=False)

        if search_docs:
            uploaded_files = st.file_uploader(
                label=ui.sidebar_file_uploader_title,
                accept_multiple_files=True,
                type=None,
            )
            if uploaded_files != [] and uploaded_files != st.session_state.get(
                "uploaded_files"
            ):
                st.session_state["uploaded_files"] = uploaded_files
                with st.spinner(ui.sidebar_file_uploader_spinner):
                    (
                        st.session_state["doc_chucks"],
                        st.session_state["faiss_db"],
                    ) = tools.get_faiss_db(uploaded_files)

    @utils.enable_chat_history
    def main():
        if user_query := st.chat_input(placeholder=ui.chat_human_placeholder):
            utils.display_msg(user_query, "user")
            st.session_state["history"].add_user_message(user_query)
            agent, user_prompt = main_agent(user_query)
            with st.chat_message("assistant", avatar=ui.chat_ai_avatar):
                st_cb = StreamlitCallbackHandler(st.container())
                response = agent().run(
                    {
                        "input": user_prompt,
                        "chat_history": st.session_state["history"].messages,
                    },
                    callbacks=[st_cb],
                )
                st.markdown(response)
                if txt2audio:
                    show_audio_player(response)
                st.session_state["messages"].append(
                    {"role": "user", "content": user_query}
                )
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "avatar": ui.chat_ai_avatar,
                        "content": response,
                    }
                )
                st.session_state["history"].add_ai_message(response)

    if __name__ == "__main__":
        main()
