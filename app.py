import os
import sys
import asyncio
import uuid
import concurrent.futures
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.services.llm_services import LLMServices
from src.services.search_client import SearchMCPClient
from src.services.linkedin_client import LinkedInMCPClient
from src.graph.builder import GraphBuilder
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.logging.logger import logger

# ─────────────────────────────────────────────────────────────────────────────
# run_async — THE KEY FIX
# Streamlit Cloud pe event loop already running hoti hai, isliye
# run_until_complete() kaam nahi karta.
# Solution: dedicated background thread mein asyncio.run() chalao.
# ─────────────────────────────────────────────────────────────────────────────
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_async(coro):
    """Run async coroutine safely from Streamlit's sync context."""
    future = _THREAD_POOL.submit(asyncio.run, coro)
    return future.result(timeout=300)


# ─────────────────────────────────────────────────────────────────────────────
# Async helpers
# ─────────────────────────────────────────────────────────────────────────────
async def get_all_threads_for_user(user_email):
    DB_URI = os.getenv("DB_URI")
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            all_threads = []
            async for thread in checkpointer.list(limit=50):
                thread_id = thread.config["configurable"]["thread_id"]
                if thread_id.startswith(user_email):
                    all_threads.append(thread_id)
            return all_threads
    except Exception as e:
        logger.error(f"Error fetching threads: {e}")
        return []


async def load_conversation_from_postgres(thread_id):
    DB_URI = os.getenv("DB_URI")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        model = LLMServices().get_model()
        builder = GraphBuilder(model, model, [], []).build()
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(checkpointer=checkpointer)
            state = await graph.aget_state(config)
            return state.values.get("messages", [])
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return []

async def run_graph_with_postgres(thread_id, action_type="stream", user_input=None, confirm_publish=True, token=None):
    if token:
        os.environ["LINKEDIN_ACCESS_TOKEN"] = token

    thread_id = st.session_state["thread_id"]
    config    = {"configurable": {"thread_id": thread_id, "linkedin_access_token": token}}
    DB_URI    = os.getenv("DB_URI")

    try:
        model          = LLMServices().get_model()
        search_tools   = await SearchMCPClient().get_tools()
        linkedin_tools = await LinkedInMCPClient().get_tools()
        model_with_tools = model.bind_tools(search_tools + linkedin_tools)

        builder = GraphBuilder(
            model=model,
            model_with_both_tools=model_with_tools,
            search_tools=search_tools,
            linkedin_tools=linkedin_tools,
        ).build()

        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(
                checkpointer=checkpointer,
                interrupt_before=["post_generate_linkedin_tool"],
            )

            # ── stream: user ne kuch likha ──────────────────────────────────
            if action_type == "stream" and user_input:
                async for _ in graph.astream(
                    {"messages": [HumanMessage(content=user_input)],
                    "iteration": 0, "max_iteration": 3, "score": 0.0},
                    config, stream_mode="values",
                ):
                    pass

            # ── resume: publish / cancel ─────────────────────────────────────
            elif action_type == "resume":
                if confirm_publish:
                    async for _ in graph.astream(None, config, stream_mode="values"):
                        pass
                    st.session_state.chat_history.append(
                        {"role": "agent", "content": "✅ Post LinkedIn pe successfully publish ho gayi!"}
                    )
                else:
                    await graph.aupdate_state(config, {"messages": []}, as_node="chat_or_post")
                    st.session_state.chat_history.append(
                        {"role": "agent", "content": "❌ Publishing cancel ho gayi. Kuch aur poochho!"}
                    )
                st.session_state.interrupt_state = False
                return

            # ── state check ──────────────────────────────────────────────────
            current_state = await graph.aget_state(config)
            is_interrupted = bool(
                current_state.next and "post_generate_linkedin_tool" in current_state.next
            )
            st.session_state.interrupt_state = is_interrupted

            if is_interrupted:
                msgs = current_state.values.get("messages", [])
                post_text = next(
                    (m.content for m in reversed(msgs)
                    if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()),
                    ""
                )
                st.session_state.post_content = post_text

            else:
                msgs = current_state.values.get("messages", [])
                score = current_state.values.get("score", None)
                if msgs:
                    last = msgs[-1]
                    content = (
                        " ".join(b.get("text", "") for b in last.content if isinstance(b, dict))
                        if isinstance(last.content, list)
                        else (last.content or "")
                    )
                    if content.strip():
                        st.session_state.chat_history.append({"role": "agent", "content": content})
                if score and score > 0:
                    st.session_state.chat_history.append(
                        {"role": "agent", "content": f"⭐ Post Score: {score}/10"}
                    )

    except Exception as e:
        logger.exception(f"Graph run failed: {e}")
        st.session_state.chat_history.append({"role": "agent", "content": f"❌ Error: {str(e)}"})


def reset_chat(current_user):
    new_id = f"{current_user}_thread_{str(uuid.uuid4())[:8]}"
    st.session_state["thread_id"] = new_id
    if new_id not in st.session_state.get("chat_threads", []):
        st.session_state["chat_threads"].append(new_id)
    st.session_state.chat_history   = []
    st.session_state.interrupt_state = False
    st.session_state.post_content   = ""


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LinkedIn Automation Agent", page_icon="💼", layout="centered")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — login
# ─────────────────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

st.sidebar.subheader("👤 User Account")

if not st.session_state["user_id"]:
    user_email = st.sidebar.text_input("Enter your Email ID:")
    if st.sidebar.button("Login"):
        if user_email:
            st.session_state["user_id"] = user_email.lower().strip()
            existing = run_async(get_all_threads_for_user(st.session_state["user_id"]))
            st.session_state["chat_threads"] = (
                existing if existing
                else [f"{st.session_state['user_id']}_thread_{str(uuid.uuid4())[:8]}"]
            )
            st.session_state["thread_id"] = st.session_state["chat_threads"][0]
            st.rerun()
else:
    st.sidebar.write(f"Logged in: **{st.session_state['user_id']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

if not st.session_state["user_id"]:
    st.info("👈 Pehle login karo apna Email ID se.")
    st.stop()

CURRENT_USER = st.session_state["user_id"]

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults — PEHLE set karo, baad mein use karo
# ─────────────────────────────────────────────────────────────────────────────
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "interrupt_state" not in st.session_state:
    st.session_state["interrupt_state"] = False
if "post_content" not in st.session_state:
    st.session_state["post_content"] = ""
if "thread_id" not in st.session_state:
    new_tid = f"{CURRENT_USER}_thread_{str(uuid.uuid4())[:8]}"
    st.session_state["thread_id"] = new_tid
    if new_tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(new_tid)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — threads + LinkedIn token
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("💬 Chat Threads")
if st.sidebar.button("➕ New Chat"):
    reset_chat(CURRENT_USER)
    st.rerun()

if st.session_state.get("chat_threads"):
    options = st.session_state["chat_threads"]
    cur_idx = options.index(st.session_state["thread_id"]) if st.session_state["thread_id"] in options else 0
    selected = st.sidebar.selectbox("Select Thread:", options, index=cur_idx)
    if selected != st.session_state["thread_id"]:
        st.session_state["thread_id"] = selected
        msgs = run_async(load_conversation_from_postgres(selected))
        st.session_state.chat_history = [
            {"role": "user" if isinstance(m, HumanMessage) else "agent", "content": m.content}
            for m in msgs if hasattr(m, "content") and isinstance(m.content, str)
        ]
        st.rerun()

linkedin_token = st.sidebar.text_input("🔑 LinkedIn Access Token", type="password")

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("💼 AI LinkedIn Post Generator")

for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Publish confirmation
if st.session_state.interrupt_state:
    st.warning("⚠️ Agent LinkedIn pe post karna chahta hai. Approve karo?")
    if st.session_state.post_content:
        with st.expander("📝 Post Preview", expanded=True):
            st.write(st.session_state.post_content)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, Publish!", type="primary", use_container_width=True):
            with st.spinner("Publishing..."):
                run_async(run_graph_with_postgres(st.session_state["thread_id"], action_type="resume", confirm_publish=True, token=linkedin_token))
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            with st.spinner("Cancelling..."):
                run_async(run_graph_with_postgres(st.session_state["thread_id"], action_type="resume", confirm_publish=False, token=linkedin_token))
            st.rerun()

elif user_input := st.chat_input("Kuch poochho ya LinkedIn post banwao..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.rerun()

# Agent response logic
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user" and not st.session_state.interrupt_state:
    with st.spinner("Agent soch raha hai..."):
        # Yahan 'run_async' ka use karein, 'asyncio.run' ka nahi
        run_async(run_graph_with_postgres(
            thread_id=st.session_state["thread_id"],
            action_type="stream",
            user_input=st.session_state.chat_history[-1]["content"],
            token=linkedin_token
        ))
    st.rerun()