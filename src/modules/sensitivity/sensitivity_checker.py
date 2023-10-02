import json

import requests
import streamlit as st

sensitivity_check = st.secrets["sensitivity_check"]


def check_text_sensitivity(text, sensitivity_check=sensitivity_check):
    if sensitivity_check:
        # URL and headers
        url = "http://sensit.chatglm.cn/sensitive"
        headers = {"Content-Type": "application/json"}

        # Data payload
        payload = {"input_type": "input", "check_type": "default_text", "prompt": text}

        # Make request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            return {"answer": None}

        # Parse the response
        response_data = response.json()

        # Extract results
        answer = response_data.get("answer", None)
        added_prompt = response_data.get("added_prompt", None)

        if added_prompt is not None:
            return {"answer": "Opps, error occurred. Please try again!"}
        elif answer is not None:
            return {"answer": answer}
        else:
            return {"answer": None}
    else:
        return {"answer": None}
