import os
import sys
import asyncio
import uuid
import concurrent.futures
import warnings
warnings.filterwarnings("ignore", message="Using fallback GPT-2 tokenizer")

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    for key, value in st.secrets.items():
        os.environ[key] = value
except Exception:
    pass

from src.services.llm_services import LLMServices
from src.services.search_client import SearchMCPClient
from src.services.linkedin_client import LinkedInMCPClient
from src.graph.builder import GraphBuilder
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.logging.logger import logger

# ─────────────────────────────────────────────────────────────────────────────
# Thread pool
# ─────────────────────────────────────────────────────────────────────────────
_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_async(coro):
    """Run async coroutine safely from Streamlit's sync context."""
    future = _THREAD_POOL.submit(asyncio.run, coro)
    return future.result(timeout=300)


# ─────────────────────────────────────────────────────────────────────────────
# Async helpers
# ─────────────────────────────────────────────────────────────────────────────
async def get_all_threads_for_user(user_email: str) -> list[str]:
    DB_URI = os.getenv("DB_URI")
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            all_threads = []
            for thread in checkpointer.list(config={}, limit=500):
                thread_id = thread.config["configurable"]["thread_id"]
                logger.info(f"Found thread: {thread_id}")
                if thread_id.startswith(user_email):
                    all_threads.append(thread_id)
            return all_threads
    except Exception as e:
        logger.error(f"Error fetching threads: {e}")
        return []


async def load_conversation_from_postgres(thread_id: str) -> list:
    DB_URI = os.getenv("DB_URI")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            checkpoint = await checkpointer.aget(config)
            if checkpoint is None:
                return []
            return checkpoint.get("channel_values", {}).get("messages", [])
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return []


async def run_graph_with_postgres(
    thread_id: str,
    action_type: str = "stream",
    user_input: str = None,
    confirm_publish: bool = True,
    token: str = None,
) -> dict:
    result = {
        "messages":        [],
        "interrupt_state": False,
        "post_content":    "",
        "error":           None,
    }

    if token:
        os.environ["LINKEDIN_ACCESS_TOKEN"] = token

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    DB_URI = os.getenv("DB_URI")

    try:
        model            = LLMServices().get_model()
        search_tools     = await SearchMCPClient().get_tools()
        linkedin_tools   = await LinkedInMCPClient().get_tools()
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

            # ── stream ───────────────────────────────────────────────────────
            if action_type == "stream" and user_input:
                async for _ in graph.astream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "iteration": 0,
                        "max_iteration": 3,
                        "score": 0.0,
                        "linkedin_access_token": token or "",
                    },
                    config,
                    stream_mode="values",
                ):
                    pass

                # ── Token check — interrupt se pehle ─────────────────────────
                current_state = await graph.aget_state(config)
                is_interrupted = bool(
                    current_state.next
                    and "post_generate_linkedin_tool" in current_state.next
                )
                if is_interrupted and not token:
                    await graph.aupdate_state(
                        config,
                        {"cancel_publish": True},
                        as_node="post_generate_linkedin_tool",
                    )
                    result["messages"].append({
                        "role": "agent",
                        "content": "LinkedIn Access Token is missing! Please add your token in the sidebar and try again."
                    })
                    result["interrupt_state"] = False
                    result["post_content"]    = ""
                    return result

            # ── resume ───────────────────────────────────────────────────────
            elif action_type == "resume":
                if confirm_publish:
                    await graph.aupdate_state(
                        config,
                        {"linkedin_access_token": token or ""},
                    )
                    async for _ in graph.astream(None, config, stream_mode="values"):
                        pass
                    result["messages"].append(
                        {"role": "agent", "content": "Post published successfully on LinkedIn!"}
                    )
                else:
                    await graph.aupdate_state(
                        config,
                        {"cancel_publish": True},
                        as_node="post_generate_linkedin_tool",
                    )
                    result["messages"].append(
                        {"role": "agent", "content": "Publishing cancelled. Feel free to ask anything else!"}
                    )
                result["interrupt_state"] = False
                result["post_content"]    = ""
                return result

            # ── state check ──────────────────────────────────────────────────
            current_state  = await graph.aget_state(config)
            is_interrupted = bool(
                current_state.next
                and "post_generate_linkedin_tool" in current_state.next
            )
            result["interrupt_state"] = is_interrupted

            if is_interrupted:
                msgs = current_state.values.get("messages", [])
                post_text = next(
                    (
                        m.content
                        for m in reversed(msgs)
                        if hasattr(m, "content")
                        and isinstance(m.content, str)
                        and m.content.strip()
                    ),
                    "",
                )
                result["post_content"] = post_text

            else:
                msgs  = current_state.values.get("messages", [])
                score = current_state.values.get("score", None)

                if msgs:
                    last    = msgs[-1]
                    content = (
                        " ".join(
                            b.get("text", "")
                            for b in last.content
                            if isinstance(b, dict)
                        )
                        if isinstance(last.content, list)
                        else (last.content or "")
                    )
                    if content.strip():
                        result["messages"].append({"role": "agent", "content": content})

                if score is not None and score > 0:
                    result["messages"].append(
                        {"role": "agent", "content": f"Post Score: {score}/10"}
                    )

    except Exception as e:
        logger.exception(f"Graph run failed: {e}")
        result["error"] = str(e)

    return result


def apply_graph_result(res: dict):
    if res.get("error"):
        st.session_state.chat_history.append(
            {"role": "agent", "content": f"Error: {res['error']}"}
        )
    else:
        st.session_state.chat_history.extend(res.get("messages", []))
        st.session_state.interrupt_state = res["interrupt_state"]
        st.session_state.post_content    = res["post_content"]


def reset_chat(current_user: str):
    thread_count = len(st.session_state.chat_threads) + 1
    new_id = f"{current_user}_thread_{thread_count}"
    st.session_state.thread_id       = new_id
    st.session_state.chat_threads.append(new_id)
    st.session_state.chat_history    = []
    st.session_state.interrupt_state = False
    st.session_state.post_content    = ""
    st.session_state.is_processing   = False


def logout():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v.copy() if isinstance(v, (list, dict)) else v


# ─────────────────────────────────────────────────────────────────────────────
# Page config + session state defaults
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Automation Agent",
    page_icon="",
    layout="centered",
)

_DEFAULTS: dict = {
    "user_id":         None,
    "chat_threads":    [],
    "thread_id":       None,
    "chat_history":    [],
    "interrupt_state": False,
    "post_content":    "",
    "is_processing":   False,
    "linkedin_token":  "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v.copy() if isinstance(_v, (list, dict)) else _v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — login / logout
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("👤 User Account")

if not st.session_state.user_id:
    user_email = st.sidebar.text_input("Enter your Email ID:")
    if st.sidebar.button("Login"):
        if user_email:
            uid      = user_email.lower().strip()
            existing = run_async(get_all_threads_for_user(uid))
            st.session_state.user_id      = uid
            st.session_state.chat_threads = (
                existing if existing
                else [f"{uid}_thread_1"]
            )
            st.session_state.thread_id = st.session_state.chat_threads[0]
            st.rerun()
else:
    st.sidebar.write(f"Logged in: **{st.session_state.user_id}**")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

if not st.session_state.user_id:
    st.info("Please login with your Email ID first.")
    st.stop()

CURRENT_USER = st.session_state.user_id

if not st.session_state.thread_id:
    thread_count = len(st.session_state.chat_threads) + 1
    new_tid = f"{CURRENT_USER}_thread_{thread_count}"
    st.session_state.thread_id = new_tid
    if new_tid not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(new_tid)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — threads + token
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.subheader("Chat Threads")
if st.sidebar.button("New Chat"):
    reset_chat(CURRENT_USER)
    st.rerun()

if st.session_state.chat_threads:
    options = st.session_state.chat_threads
    cur_idx = (
        options.index(st.session_state.thread_id)
        if st.session_state.thread_id in options
        else 0
    )
    selected = st.sidebar.selectbox("Select Thread:", options, index=cur_idx)
    if selected != st.session_state.thread_id:
        st.session_state.thread_id     = selected
        st.session_state.is_processing = False
        msgs = run_async(load_conversation_from_postgres(selected))
        st.session_state.chat_history = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "agent",
                "content": m.content,
            }
            for m in msgs
            if hasattr(m, "content") and isinstance(m.content, str)
        ]
        st.rerun()

raw_token = st.sidebar.text_input(
    "LinkedIn Access Token",
    type="password",
    value=st.session_state.linkedin_token,
)
if raw_token != st.session_state.linkedin_token:
    st.session_state.linkedin_token     = raw_token
    os.environ["LINKEDIN_ACCESS_TOKEN"] = raw_token

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("💼 AI LinkedIn Post Generator")

for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ── Publish confirmation ──────────────────────────────────────────────────────
if st.session_state.interrupt_state:
    st.warning("Agent wants to publish a post on LinkedIn. Do you approve?")
    if st.session_state.post_content:
        with st.expander("Post Preview", expanded=True):
            st.write(st.session_state.post_content)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Publish!", type="primary", use_container_width=True):
            with st.spinner("Publishing..."):
                res = run_async(
                    run_graph_with_postgres(
                        st.session_state.thread_id,
                        action_type="resume",
                        confirm_publish=True,
                        token=st.session_state.linkedin_token,
                    )
                )
            apply_graph_result(res)
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            with st.spinner("Cancelling..."):
                res = run_async(
                    run_graph_with_postgres(
                        st.session_state.thread_id,
                        action_type="resume",
                        confirm_publish=False,
                        token=st.session_state.linkedin_token,
                    )
                )
            apply_graph_result(res)
            st.rerun()

# ── User input ────────────────────────────────────────────────────────────────
elif user_input := st.chat_input("Ask something or generate a LinkedIn post..."):
    if not st.session_state.is_processing:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.is_processing = True
        st.rerun()

# ── Agent response ────────────────────────────────────────────────────────────
if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
    and not st.session_state.interrupt_state
    and st.session_state.is_processing
):
    with st.spinner("Agent is thinking..."):
        res = run_async(
            run_graph_with_postgres(
                thread_id=st.session_state.thread_id,
                action_type="stream",
                user_input=st.session_state.chat_history[-1]["content"],
                token=st.session_state.linkedin_token,
            )
        )
    apply_graph_result(res)
    st.session_state.is_processing = False
    st.rerun()