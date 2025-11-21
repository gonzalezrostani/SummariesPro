
import os, time, uuid, json
import pandas as pd
import streamlit as st
from openai import OpenAI
from pathlib import Path

# --- Basic setup ---
APP_DIR = Path(".")

st.set_page_config(page_title="SummariesPro", page_icon="📝")
st.title("SummariesPro Editing Tool")

# --- API key handling: use Streamlit Secrets or environment variable ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Please add it to Streamlit Secrets "
        "(Manage app → Secrets) or as an environment variable."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Quick auth sanity check (runs once per app startup) ---
try:
    client.models.list()
except Exception as e:
    st.error(
        "OpenAI authentication failed. "
        "Please check that your API key in Streamlit Secrets is valid and active. "
        f"Details: {e}"
    )
    st.stop()

# ---- Session state ----
if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "turns" not in st.session_state:
    st.session_state.turns = []
if "rounds_done" not in st.session_state:
    st.session_state.rounds_done = 0
if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "start_timestamp" not in st.session_state:
    st.session_state.start_timestamp = None  # timestamp of round 1

# Generation parameters
temperature = 0.4
max_tokens = 350

# ---- System instructions ----
SYSTEM_INSTRUCTIONS = '''
You help users summarize texts they paste.

TASK
- Read the SOURCE_TEXT provided by the user.
- Produce a concise summary of about 150 words (roughly 130-170 words).
- Must include:

Main Idea:
Key Information:
Significance:

REQUIREMENTS
- Use your own words.
- No long quotes.
- Stay faithful to the text.

DIALOGUE ROUNDS
Round 1 → produce first summary + ask a question.
Round 2 → revise based on feedback + ask a question.
Round 3 → finalize summary (NO question at end).

GENERAL
- Keep the labels exactly as written.
- Always base your work on SOURCE_TEXT.
'''

MODEL = "gpt-4o-mini"


def get_source_and_feedback(latest_user_text):
    return st.session_state.source_text or "", latest_user_text or ""


def get_last_assistant_summary():
    for t in reversed(st.session_state.turns):
        if t["role"] == "assistant":
            return t["content"]
    return ""


def respond(user_text, temperature, max_tokens):
    current_round = st.session_state.rounds_done + 1
    source_text, feedback = get_source_and_feedback(user_text)

    # Set start timestamp at the first round
    if current_round == 1 and st.session_state.start_timestamp is None:
        st.session_state.start_timestamp = int(time.time())

    if current_round == 1 and not st.session_state.source_text:
        st.session_state.source_text = user_text
        source_text = user_text
        feedback = ""

    previous_summary = get_last_assistant_summary()

    system_with_round = (
        SYSTEM_INSTRUCTIONS
        + f"\\nCURRENT_ROUND: {current_round} of 3"
        + "\\n\\nSOURCE_TEXT:\\n" + source_text[:8000]
        + "\\n\\nPREVIOUS_SUMMARY_IF_ANY:\\n" + previous_summary[:4000]
        + "\\n\\nUSER_FEEDBACK_THIS_ROUND:\\n" + feedback[:3000]
    )

    messages = [{"role": "system", "content": system_with_round}]
    for t in st.session_state.turns:
        messages.append({"role": t["role"], "content": t["content"]})

    # Mark feedback explicitly for rounds 2 and 3
    user_message_content = (
        user_text if current_round == 1 else "Revision feedback from the user: " + user_text
    )
    messages.append({"role": "user", "content": user_message_content})

    # Use Chat Completions (not Responses)
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    reply = (r.choices[0].message.content or "").strip()

    if current_round == 3 and reply.endswith("?"):
        reply = reply.rstrip(" ?") + "."

    return reply


# ---- LOGGING: one row per full 3-round conversation ----

def save_full_conversation():
    turns = st.session_state.turns

    user_turns = [t["content"] for t in turns if t["role"] == "user"]
    sys_turns = [t["content"] for t in turns if t["role"] == "assistant"]

    user_r1 = user_turns[0] if len(user_turns) > 0 else ""
    user_r2 = user_turns[1] if len(user_turns) > 1 else ""
    user_r3 = user_turns[2] if len(user_turns) > 2 else ""

    system_r1 = sys_turns[0] if len(sys_turns) > 0 else ""
    system_r2 = sys_turns[1] if len(sys_turns) > 1 else ""
    system_r3 = sys_turns[2] if len(sys_turns) > 2 else ""

    end_timestamp = int(time.time())

    row = {
        "start_timestamp": st.session_state.start_timestamp,
        "end_timestamp": end_timestamp,
        "conversation_id": st.session_state.conv_id,
        "user_r1": user_r1,
        "system_r1": system_r1,
        "user_r2": user_r2,
        "system_r2": system_r2,
        "user_r3": user_r3,
        "system_r3": system_r3,
    }

    csv_path = APP_DIR / "summary_logs.csv"
    header = not csv_path.exists()
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", index=False, header=header)


def log_event(role, content):
    st.session_state.turns.append({"role": role, "content": content})


# ---- UI ----
st.subheader("Paste your text for a summary. You may ask for up to three revisions.")

rounds_left = max(0, 3 - st.session_state.rounds_done)
st.caption(f"Rounds remaining: {rounds_left}")

for t in st.session_state.turns:
    with st.chat_message(
        "user" if t["role"] == "user" else "assistant",
        avatar="👤" if t["role"] == "user" else "🤖"
    ):
        st.markdown(t["content"])

render_input = st.session_state.rounds_done < 3

if render_input:
    with st.form("chat_form", clear_on_submit=True):

        placeholder = (
            "Paste the article/text here." if st.session_state.rounds_done == 0
            else "Describe the edits you want (shorter, clearer, emphasize X…)."
        )

        st.markdown(
            '''
<style>
textarea[aria-label="Your message"] {
    height: 200px !important;
    resize: none !important;
}
</style>
            ''',
            unsafe_allow_html=True,
        )

        user_text_area = st.text_area("Your message", placeholder=placeholder, key="chat_draft")
        submitted = st.form_submit_button("Send")

    user_text = user_text_area.strip() if (submitted and user_text_area) else None
else:
    user_text = None
    st.info("You have completed all 3 rounds. Refresh to start again.")


if user_text:
    log_event("user", user_text)

    reply = respond(user_text, temperature, max_tokens)

    log_event("assistant", reply)

    st.session_state.rounds_done += 1

    if st.session_state.rounds_done == 3:
        save_full_conversation()

    st.rerun()
