import os
import requests
import streamlit as st
from uuid import uuid4
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

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

default_echo_bot_url = "http://orchestrator:8000"

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
    st.write(f"DEBUG dialog_id: {st.session_state.dialog_id}")

    # Call FastAPI `/predict` to get bot probability
    predict_response = requests.post(
        f"{default_echo_bot_url}/predict",
        json={
            "id": str(uuid4()),
            "dialog_id": st.session_state.dialog_id,
            "participant_index": 0,
            "text": message
        }
    ).json()

    is_bot_prob = predict_response.get("is_bot_probability", 0.5)

    # Call FastAPI `/get_message` to get LLM response
    response = requests.post(
        f"{default_echo_bot_url}/get_message",
        json={
            "dialog_id": st.session_state.dialog_id,
            "last_message_id": None,  # Optional: Fill if needed
            "last_msg_text": message
        }
    ).json()

    reply_text = response.get("new_msg_text", "❌ No reply")

    # Show probability next to user msg
    st.write(f"🤖 BOT probability for your msg: {is_bot_prob:.2f}")
    st.session_state.probs.append(is_bot_prob)
    st.session_state.labels.append(0)  # User = human = 0

    # Bot reply from LLM
    bot_msg = {"role": "assistant", "content": reply_text}
    st.session_state.messages.append(bot_msg)
    st.chat_message("assistant").write(reply_text)

    st.rerun()

# Compute metrics if at least one turn
if len(st.session_state.probs) > 0:
    preds = [1 if p >= 0.5 else 0 for p in st.session_state.probs]
    acc = accuracy_score(st.session_state.labels, preds)
    try:
        ll = log_loss(st.session_state.labels, st.session_state.probs)
    except ValueError:
        ll = np.nan

    st.sidebar.markdown(f"**Live Accuracy:** {acc:.2f}")
    st.sidebar.markdown(f"**Live Log-Loss:** {ll:.2f}")

    # Plot metrics as line chart vs dialogue turn
    accuracy_progress = []
    logloss_progress = []

    num_dialogues = len(st.session_state.probs)
    for i in range(1, num_dialogues + 1):
        preds_so_far = [1 if p >= 0.5 else 0 for p in st.session_state.probs[:i]]
        acc_so_far = accuracy_score(st.session_state.labels[:i], preds_so_far)
        try:
            ll_so_far = log_loss(st.session_state.labels[:i], st.session_state.probs[:i])
        except ValueError:
            ll_so_far = np.nan
        accuracy_progress.append(acc_so_far)
        logloss_progress.append(ll_so_far)

    st.line_chart({
        "Accuracy": accuracy_progress,
        "LogLoss": logloss_progress
    })