import os
import time
from datetime import datetime

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

import src.modules.tools.tools as tools
import src.modules.ui.ui_config as ui_config
import src.modules.ui.utils as utils
from src.modules.agents.agent_selector import main_agent
from src.modules.agents.agent_history import xata_chat_history

ui = ui_config.create_ui_from_config()

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["XATA_API_KEY"] = st.secrets["xata_api_key"]
os.environ["XATA_DATABASE_URL"] = st.secrets["xata_db_url"]

st.set_page_config(page_title=ui.page_title, layout="wide", page_icon=ui.page_icon)


if ui.need_passwd is False:
    auth = True
else:
    auth = utils.check_password()


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
            # txt2audio = st.checkbox(ui.txt2audio_checkbox_label, value=False)
            # chat_memory = st.checkbox(ui.chat_memory_checkbox_label, value=False)
            search_docs = st.toggle(ui.upload_docs_checkbox_label, value=False)

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

        st.divider()

        col_newchat, col_delete = st.columns([1, 1])
        with col_newchat:
            new_chat = st.button(
                ui.sidebar_newchat_button_label, use_container_width=True
            )
        if new_chat:
            st.session_state.clear()
            st.rerun()

        with col_delete:
            delete_chat = st.button(
                ui.sidebar_delete_button_label, use_container_width=True
            )
        if delete_chat:
            utils.delete_chat_history(st.session_state["selected_chat_id"])
            st.session_state.clear()
            st.rerun()

        table_map = utils.fetch_chat_history()
        if "first_run" not in st.session_state:
            timestamp = time.time()
            st.session_state["timestamp"] = timestamp
        else:
            timestamp = st.session_state["timestamp"]
        table_map_new = {
            str(timestamp): datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " - New Chat"
        }
        table_map = table_map_new | table_map
        entries = list(table_map.keys())
        # Check if selected_chat_id exists in session_state, if not set default as the first entry
        if "selected_chat_id" not in st.session_state:
            st.session_state["selected_chat_id"] = entries[0]

        # Update the selectbox with the current selected_chat_id value
        current_chat_id = st.selectbox(
            label=ui.current_chat_title,
            label_visibility="collapsed",
            options=entries,
            format_func=lambda x: table_map[x],
        )

        # Save the selected value back to session state
        st.session_state["selected_chat_id"] = current_chat_id

        if "first_run" not in st.session_state:
            st.session_state["history"] = xata_chat_history(_session_id=current_chat_id)
            st.session_state["first_run"] = True
        else:
            st.session_state["history"] = xata_chat_history(_session_id=current_chat_id)
            st.session_state["messages"] = utils.initialize_messages(
                st.session_state["history"].messages
            )

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
                # if txt2audio:
                #     utils.show_audio_player(response)
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

                if len(st.session_state["messages"]) == 3:
                    st.rerun()

    if __name__ == "__main__":
        main()
