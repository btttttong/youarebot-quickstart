import os
import requests
import streamlit as st
from uuid import uuid4
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

from app.models import GetMessageRequestModel

# Set env for MacOS watcher issue
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("# Echo bot 🚀 + Live Metrics")
st.sidebar.markdown("# Echo bot 🚀 + Live Metrics")

# Initialize state
if "dialog_id" not in st.session_state:
    st.session_state.dialog_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Type something"}]

if "probs" not in st.session_state:
    st.session_state.probs = []  # Store predicted probabilities

if "labels" not in st.session_state:
    st.session_state.labels = []  # Store simulated ground-truth labels

default_echo_bot_url = "http://localhost:6872"

with st.sidebar:
    if st.button("Reset"):
        st.session_state.clear()
        st.rerun()

    st.text_input("Bot URL", key="echo_bot_url", value=default_echo_bot_url, disabled=True)
    st.text_input("Dialog ID", key="dialog_id", disabled=True)

# Show conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input handling
if message := st.chat_input("Your message"):
    # User message
    user_msg = {"role": "user", "content": message}
    st.session_state.messages.append(user_msg)
    st.chat_message("user").write(message)

    # Call prediction API for user message
    response = requests.post(
        f"{default_echo_bot_url}/predict",
        json={
            "id": str(uuid4()),
            "dialog_id": st.session_state.dialog_id,
            "participant_index": 0,
            "text": message
        }
    ).json()

    user_prob = response["is_bot_probability"]
    st.session_state.probs.append(user_prob)
    st.session_state.labels.append(0)  # Simulate user = human = 0

    # Show probability next to user msg
    st.write(f"🤖 BOT probability for your msg: {user_prob:.2f}")

    # Simulate echo bot reply (= exact same message)
    bot_msg = {"role": "assistant", "content": message}
    st.session_state.messages.append(bot_msg)
    st.chat_message("assistant").write(message)

    # Call prediction API for bot message
    response_bot = requests.post(
        f"{default_echo_bot_url}/predict",
        json={
            "id": str(uuid4()),
            "dialog_id": st.session_state.dialog_id,
            "participant_index": 1,
            "text": message
        }
    ).json()

    bot_prob = response_bot["is_bot_probability"]
    st.session_state.probs.append(bot_prob)
    st.session_state.labels.append(1)  # Simulate bot = bot = 1

    st.write(f"🤖 BOT probability for bot echo: {bot_prob:.2f}")

    st.rerun()

# Compute metrics if at least one turn
if len(st.session_state.probs) > 0:
    preds = [1 if p >= 0.5 else 0 for p in st.session_state.probs]
    acc = accuracy_score(st.session_state.labels, preds)
    try:
        ll = log_loss(st.session_state.labels, st.session_state.probs)
    except ValueError:
        ll = np.nan  # Handle log_loss edge case when probabilities degenerate early

    st.sidebar.markdown(f"**Live Accuracy:** {acc:.2f}")
    st.sidebar.markdown(f"**Live Log-Loss:** {ll:.2f}")

    # Plot metrics as line chart vs dialogue turn
    accuracy_progress = []
    logloss_progress = []

    num_dialogues = len(st.session_state.probs) // 2
    for i in range(1, num_dialogues + 1):
        end_idx = i * 2
        preds_so_far = [1 if p >= 0.5 else 0 for p in st.session_state.probs[:end_idx]]
        acc_so_far = accuracy_score(st.session_state.labels[:end_idx], preds_so_far)
        try:
            ll_so_far = log_loss(st.session_state.labels[:end_idx], st.session_state.probs[:end_idx])
        except ValueError:
            ll_so_far = np.nan
        accuracy_progress.append(acc_so_far)
        logloss_progress.append(ll_so_far)

    st.line_chart({
        "Accuracy": accuracy_progress,
        "LogLoss": logloss_progress
    })