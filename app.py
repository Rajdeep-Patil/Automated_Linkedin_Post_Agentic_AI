import os
import sys
import asyncio
import uuid
import queue
import concurrent.futures
import warnings

# ---------------------------------------------------------
# WARNING FILTERS
# ---------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="Using fallback GPT-2 tokenizer.*"
)

warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub.*"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="huggingface_hub"
)

# ---------------------------------------------------------
# STREAMLIT
# ---------------------------------------------------------
import streamlit as st

# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# WINDOWS ASYNCIO FIX
# ---------------------------------------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

# ---------------------------------------------------------
# STREAMLIT SECRETS
# ---------------------------------------------------------
try:
    for key, value in st.secrets.items():
        os.environ[key] = value
except Exception:
    pass

# ---------------------------------------------------------
# LANGCHAIN
# ---------------------------------------------------------
from langchain_core.messages import (
    HumanMessage,
    AIMessageChunk,
)

# ---------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------
from src.services.llm_services import LLMServices
from src.services.search_client import SearchMCPClient
from src.services.linkedin_client import LinkedInMCPClient
from src.graph.builder import GraphBuilder

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver
)

from src.logging.logger import logger


# =========================================================
# THREAD POOL
# =========================================================

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4
)


# =========================================================
# DEFAULT SESSION STATE
# =========================================================

_DEFAULTS = {
    "user_id": None,
    "chat_threads": [],
    "thread_id": None,
    "chat_history": [],
    "interrupt_state": False,
    "post_content": "",
    "is_processing": False,
    "linkedin_token": "",
}


for key, value in _DEFAULTS.items():

    if key not in st.session_state:

        if isinstance(value, (list, dict)):
            st.session_state[key] = value.copy()

        else:
            st.session_state[key] = value


# =========================================================
# ASYNC HELPER
# =========================================================

def run_async(coro):

    future = _THREAD_POOL.submit(
        asyncio.run,
        coro
    )

    return future.result(timeout=300)


# =========================================================
# GET ALL THREADS FOR USER
# =========================================================

async def get_all_threads_for_user(
    user_email: str
) -> list[str]:

    DB_URI = os.getenv("DB_URI")

    try:

        async with AsyncPostgresSaver.from_conn_string(
            DB_URI
        ) as checkpointer:

            await checkpointer.setup()

            all_threads = []

            for thread in checkpointer.list(
                config={},
                limit=500
            ):

                thread_id = (
                    thread.config
                    ["configurable"]
                    ["thread_id"]
                )

                logger.info(
                    f"Found thread: {thread_id}"
                )

                if thread_id.startswith(
                    f"{user_email}_thread_"
                ):

                    all_threads.append(
                        thread_id
                    )

            return all_threads

    except Exception as e:

        logger.error(
            f"Error fetching threads: {e}"
        )

        return []


# =========================================================
# LOAD CONVERSATION
# =========================================================

async def load_conversation_from_postgres(
    thread_id: str
) -> list:

    DB_URI = os.getenv("DB_URI")

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        async with AsyncPostgresSaver.from_conn_string(
            DB_URI
        ) as checkpointer:

            await checkpointer.setup()

            checkpoint = await checkpointer.aget(
                config
            )

            if checkpoint is None:
                return []

            return checkpoint.get(
                "channel_values",
                {}
            ).get(
                "messages",
                []
            )

    except Exception as e:

        logger.error(
            f"Failed to load conversation: {e}"
        )

        return []


# =========================================================
# RUN GRAPH WITH POSTGRES
# =========================================================

async def run_graph_with_postgres(
    thread_id: str,
    action_type: str = "stream",
    user_input: str = None,
    confirm_publish: bool = True,
    token: str = None,
    chunk_queue: queue.Queue = None,
) -> dict:

    result = {
        "messages": [],
        "interrupt_state": False,
        "post_content": "",
        "error": None,
    }

    # -----------------------------------------------------
    # LINKEDIN TOKEN
    # -----------------------------------------------------

    if token:

        os.environ[
            "LINKEDIN_ACCESS_TOKEN"
        ] = token

    # -----------------------------------------------------
    # CONFIG
    # -----------------------------------------------------

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    DB_URI = os.getenv("DB_URI")

    try:

        # =================================================
        # LOAD MODEL
        # =================================================

        model = LLMServices().get_model()

        # =================================================
        # LOAD MCP TOOLS
        # =================================================

        search_tools = (
            await SearchMCPClient().get_tools()
        )

        linkedin_tools = (
            await LinkedInMCPClient().get_tools()
        )

        # =================================================
        # BIND TOOLS
        # =================================================

        model_with_tools = model.bind_tools(
            search_tools + linkedin_tools
        )

        # =================================================
        # BUILD GRAPH
        # =================================================

        builder = GraphBuilder(
            model=model,
            model_with_both_tools=model_with_tools,
            search_tools=search_tools,
            linkedin_tools=linkedin_tools,
        ).build()

        # =================================================
        # POSTGRES CHECKPOINTER
        # =================================================

        async with AsyncPostgresSaver.from_conn_string(
            DB_URI
        ) as checkpointer:

            await checkpointer.setup()

            graph = builder.compile(
                checkpointer=checkpointer,
                interrupt_before=[
                    "post_generate_linkedin_tool"
                ],
            )

            # =================================================
            # NORMAL STREAM
            # =================================================

            if (
                action_type == "stream"
                and user_input
            ):

                initial_state = {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ],

                    "iteration": 0,

                    "max_iteration": 3,

                    "score": 0.0,

                    "linkedin_access_token": (
                        token or ""
                    ),
                }

                async for event in graph.astream_events(

                    initial_state,

                    config,

                    version="v2",
                ):

                    kind = event.get(
                        "event"
                    )

                    # -----------------------------------------
                    # STREAM LLM RESPONSE
                    # -----------------------------------------

                    if kind == "on_chat_model_stream":

                        chunk = event.get(
                            "data",
                            {}
                        ).get(
                            "chunk"
                        )

                        if (
                            chunk
                            and hasattr(
                                chunk,
                                "content"
                            )
                        ):

                            text = ""

                            if isinstance(
                                chunk.content,
                                str
                            ):

                                text = (
                                    chunk.content
                                )

                            elif isinstance(
                                chunk.content,
                                list
                            ):

                                text = "".join(

                                    b.get(
                                        "text",
                                        ""
                                    )

                                    for b
                                    in chunk.content

                                    if isinstance(
                                        b,
                                        dict
                                    )
                                )

                            if (
                                text
                                and chunk_queue
                            ):

                                chunk_queue.put(
                                    {
                                        "type": "chunk",
                                        "text": text,
                                    }
                                )

                # =================================================
                # GET CURRENT STATE
                # =================================================

                current_state = (
                    await graph.aget_state(
                        config
                    )
                )

                is_interrupted = bool(

                    current_state.next

                    and
                    "post_generate_linkedin_tool"
                    in current_state.next
                )

                # =================================================
                # TOKEN MISSING
                # =================================================

                if (
                    is_interrupted
                    and not token
                ):

                    await graph.aupdate_state(

                        config,

                        {
                            "cancel_publish": True
                        },

                        as_node=(
                            "post_generate_linkedin_tool"
                        ),
                    )

                    result["messages"].append(
                        {
                            "role": "agent",

                            "content":
                            (
                                "LinkedIn Access Token "
                                "is missing! Please add "
                                "your token in the sidebar "
                                "and try again."
                            ),
                        }
                    )

                    result[
                        "interrupt_state"
                    ] = False

                    result[
                        "post_content"
                    ] = ""

                    if chunk_queue:

                        chunk_queue.put(
                            {
                                "type": "done"
                            }
                        )

                    return result

                # =================================================
                # INTERRUPT STATUS
                # =================================================

                result[
                    "interrupt_state"
                ] = is_interrupted

                # =================================================
                # PUBLISH APPROVAL REQUIRED
                # =================================================

                if is_interrupted:

                    msgs = (
                        current_state.values
                        .get(
                            "messages",
                            []
                        )
                    )

                    post_text = next(

                        (

                            m.content

                            for m
                            in reversed(msgs)

                            if (
                                hasattr(
                                    m,
                                    "content"
                                )

                                and isinstance(
                                    m.content,
                                    str
                                )

                                and m.content.strip()
                            )

                        ),

                        "",
                    )

                    result[
                        "post_content"
                    ] = post_text

                # =================================================
                # NORMAL RESPONSE
                # =================================================

                else:

                    msgs = (
                        current_state.values
                        .get(
                            "messages",
                            []
                        )
                    )

                    score = (
                        current_state.values
                        .get(
                            "score",
                            None
                        )
                    )

                    if (
                        score is not None
                        and score > 0
                    ):

                        result[
                            "messages"
                        ].append(

                            {
                                "role": "agent",

                                "content":
                                f"Post Score: "
                                f"{score}/10",
                            }
                        )

            # =================================================
            # RESUME AFTER USER APPROVAL
            # =================================================

            elif action_type == "resume":

                # ---------------------------------------------
                # PUBLISH
                # ---------------------------------------------

                if confirm_publish:

                    if not token:

                        result["error"] = (
                            "LinkedIn Access Token "
                            "is missing."
                        )

                        return result

                    # -----------------------------------------
                    # Update token in graph state
                    # -----------------------------------------

                    await graph.aupdate_state(

                        config,

                        {
                            "linkedin_access_token":
                            token
                        },
                    )

                    # -----------------------------------------
                    # Resume graph
                    # -----------------------------------------

                    async for _ in graph.astream(

                        None,

                        config,

                        stream_mode="values"
                    ):

                        pass

                    result[
                        "messages"
                    ].append(

                        {
                            "role": "agent",

                            "content":
                            (
                                "Post published "
                                "successfully on LinkedIn!"
                            ),
                        }
                    )

                # ---------------------------------------------
                # CANCEL PUBLISH
                # ---------------------------------------------

                else:

                    await graph.aupdate_state(

                        config,

                        {
                            "cancel_publish": True
                        },

                        as_node=(
                            "post_generate_linkedin_tool"
                        ),
                    )

                    result[
                        "messages"
                    ].append(

                        {
                            "role": "agent",

                            "content":
                            (
                                "Publishing cancelled. "
                                "Feel free to ask anything else!"
                            ),
                        }
                    )

                result[
                    "interrupt_state"
                ] = False

                result[
                    "post_content"
                ] = ""

    except Exception as e:

        logger.exception(
            f"Graph run failed: {e}"
        )

        result[
            "error"
        ] = str(e)

    finally:

        if chunk_queue:

            chunk_queue.put(
                {
                    "type": "done"
                }
            )

    return result


# =========================================================
# STREAM AGENT RESPONSE
# =========================================================

def stream_agent_response(
    thread_id: str,
    user_input: str,
    token: str
) -> tuple:

    chunk_q = queue.Queue()

    future = _THREAD_POOL.submit(

        asyncio.run,

        run_graph_with_postgres(

            thread_id=thread_id,

            action_type="stream",

            user_input=user_input,

            token=token,

            chunk_queue=chunk_q,
        )
    )

    # -----------------------------------------------------
    # STREAM RESPONSE IN UI
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        placeholder = st.empty()

        full_text = ""

        while True:

            try:

                item = chunk_q.get(
                    timeout=60
                )

                if (
                    item["type"]
                    == "done"
                ):

                    break

                elif (
                    item["type"]
                    == "chunk"
                ):

                    full_text += (
                        item["text"]
                    )

                    placeholder.markdown(
                        full_text + "▌"
                    )

            except queue.Empty:

                break

        if full_text:

            placeholder.markdown(
                full_text
            )

    # -----------------------------------------------------
    # GET FINAL RESULT
    # -----------------------------------------------------

    res = future.result(
        timeout=300
    )

    return res, full_text


# =========================================================
# APPLY GRAPH RESULT
# =========================================================

def apply_graph_result(
    res: dict,
    streamed_text: str = ""
):

    if res.get("error"):

        st.session_state.chat_history.append(

            {
                "role": "agent",

                "content":
                f"Error: {res['error']}",
            }
        )

    else:

        # ---------------------------------------------
        # STREAMED MESSAGE
        # ---------------------------------------------

        if streamed_text:

            st.session_state.chat_history.append(

                {
                    "role": "agent",

                    "content": streamed_text,
                }
            )

        # ---------------------------------------------
        # OTHER MESSAGES
        # ---------------------------------------------

        for msg in res.get(
            "messages",
            []
        ):

            if (
                msg["content"]
                not in streamed_text
            ):

                st.session_state.chat_history.append(
                    msg
                )

        # ---------------------------------------------
        # INTERRUPT STATE
        # ---------------------------------------------

        st.session_state.interrupt_state = (
            res["interrupt_state"]
        )

        st.session_state.post_content = (
            res["post_content"]
        )


# =========================================================
# RESET CHAT
# =========================================================

def reset_chat(
    current_user: str
):

    # -----------------------------------------------------
    # ALWAYS CREATE UNIQUE UUID THREAD
    # -----------------------------------------------------

    new_id = (
        f"{current_user}_thread_"
        f"{uuid.uuid4().hex}"
    )

    st.session_state.thread_id = new_id

    st.session_state.chat_threads.append(
        new_id
    )

    st.session_state.chat_history = []

    st.session_state.interrupt_state = False

    st.session_state.post_content = ""

    st.session_state.is_processing = False

    # -----------------------------------------------------
    # RERUN
    # -----------------------------------------------------

    st.rerun()


# =========================================================
# LOGOUT
# =========================================================

def logout():

    for key, value in _DEFAULTS.items():

        if isinstance(
            value,
            (list, dict)
        ):

            st.session_state[key] = (
                value.copy()
            )

        else:

            st.session_state[key] = value


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title=(
        "LinkedIn Automation Agent"
    ),

    page_icon="💼",

    layout="centered",
)


# =========================================================
# USER ACCOUNT
# =========================================================

st.sidebar.subheader(
    "User Account"
)


# =========================================================
# LOGIN
# =========================================================

if not st.session_state.user_id:

    user_email = st.sidebar.text_input(
        "Enter your Email ID:"
    )

    if st.sidebar.button(
        "Login"
    ):

        if user_email:

            uid = (
                user_email
                .lower()
                .strip()
            )

            # ---------------------------------------------
            # GET OLD THREADS
            # ---------------------------------------------

            existing = run_async(

                get_all_threads_for_user(
                    uid
                )
            )

            # ---------------------------------------------
            # SAVE USER
            # ---------------------------------------------

            st.session_state.user_id = uid

            # ---------------------------------------------
            # KEEP OLD THREADS
            # ---------------------------------------------

            st.session_state.chat_threads = (
                existing
                if existing
                else []
            )

            # ---------------------------------------------
            # ALWAYS CREATE NEW UUID THREAD
            # ---------------------------------------------

            new_id = (
                f"{uid}_thread_"
                f"{uuid.uuid4().hex}"
            )

            st.session_state.chat_threads.append(
                new_id
            )

            st.session_state.thread_id = (
                new_id
            )

            # ---------------------------------------------
            # RESET CHAT STATE
            # ---------------------------------------------

            st.session_state.chat_history = []

            st.session_state.interrupt_state = False

            st.session_state.post_content = ""

            st.session_state.is_processing = False

            st.rerun()


# =========================================================
# LOGGED-IN USER
# =========================================================

else:

    st.sidebar.write(
        f"Logged in: "
        f"**{st.session_state.user_id}**"
    )

    if st.sidebar.button(
        "Logout"
    ):

        logout()

        st.rerun()


# =========================================================
# STOP IF NOT LOGGED IN
# =========================================================

if not st.session_state.user_id:

    st.info(
        "Please login with your Email ID first."
    )

    st.stop()


# =========================================================
# CURRENT USER
# =========================================================

CURRENT_USER = (
    st.session_state.user_id
)


# =========================================================
# SAFETY: IF NO THREAD EXISTS
# =========================================================

if not st.session_state.thread_id:

    new_tid = (
        f"{CURRENT_USER}_thread_"
        f"{uuid.uuid4().hex}"
    )

    st.session_state.thread_id = new_tid

    if (
        new_tid
        not in st.session_state.chat_threads
    ):

        st.session_state.chat_threads.append(
            new_tid
        )


# =========================================================
# CHAT THREADS
# =========================================================

st.sidebar.subheader(
    "Chat Threads"
)


# =========================================================
# NEW CHAT BUTTON
# =========================================================

if st.sidebar.button(
    "New Chat"
):

    reset_chat(
        CURRENT_USER
    )


# =========================================================
# THREAD SELECTBOX
# =========================================================

if st.session_state.chat_threads:

    options = (
        st.session_state.chat_threads
    )

    # ---------------------------------------------
    # CURRENT THREAD INDEX
    # ---------------------------------------------

    if (
        st.session_state.thread_id
        in options
    ):

        cur_idx = (
            options.index(
                st.session_state.thread_id
            )
        )

    else:

        cur_idx = 0

    # ---------------------------------------------
    # SELECT THREAD
    # ---------------------------------------------

    selected = st.sidebar.selectbox(

        "Select Thread:",

        options,

        index=cur_idx,
    )

    # ---------------------------------------------
    # THREAD CHANGED
    # ---------------------------------------------

    if (
        selected
        != st.session_state.thread_id
    ):

        st.session_state.thread_id = (
            selected
        )

        st.session_state.is_processing = False

        # -----------------------------------------
        # LOAD OLD THREAD
        # -----------------------------------------

        msgs = run_async(

            load_conversation_from_postgres(
                selected
            )
        )

        st.session_state.chat_history = [

            {
                "role":
                (
                    "user"
                    if isinstance(
                        m,
                        HumanMessage
                    )
                    else "agent"
                ),

                "content": m.content,
            }

            for m in msgs

            if (
                hasattr(
                    m,
                    "content"
                )

                and isinstance(
                    m.content,
                    str
                )
            )
        ]

        st.session_state.interrupt_state = False

        st.session_state.post_content = ""

        st.rerun()


# =========================================================
# LINKEDIN ACCESS TOKEN
# =========================================================

raw_token = st.sidebar.text_input(

    "LinkedIn Access Token",

    type="password",

    value=(
        st.session_state.linkedin_token
    ),
)


# =========================================================
# SAVE TOKEN
# =========================================================

if (
    raw_token
    != st.session_state.linkedin_token
):

    st.session_state.linkedin_token = (
        raw_token
    )

    os.environ[
        "LINKEDIN_ACCESS_TOKEN"
    ] = raw_token


# =========================================================
# TITLE
# =========================================================

st.title(
    "AI LinkedIn Post Generator"
)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for msg in st.session_state.chat_history:

    with st.chat_message(

        "user"
        if msg["role"] == "user"
        else "assistant"

    ):

        st.markdown(
            msg["content"]
        )


# =========================================================
# LINKEDIN PUBLISH INTERRUPT
# =========================================================

if st.session_state.interrupt_state:

    st.warning(
        "Agent wants to publish a post "
        "on LinkedIn. Do you approve?"
    )

    # -----------------------------------------------------
    # POST PREVIEW
    # -----------------------------------------------------

    if st.session_state.post_content:

        with st.expander(
            "Post Preview",
            expanded=True
        ):

            st.write(
                st.session_state.post_content
            )

    # -----------------------------------------------------
    # BUTTONS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    # =====================================================
    # YES PUBLISH
    # =====================================================

    with col1:

        if st.button(

            "Yes, Publish!",

            type="primary",

            use_container_width=True,
        ):

            # ---------------------------------------------
            # CHECK TOKEN
            # ---------------------------------------------

            if not st.session_state.linkedin_token:

                st.error(
                    "Please enter your "
                    "LinkedIn Access Token "
                    "in the sidebar first."
                )

                st.stop()

            # ---------------------------------------------
            # PUBLISH
            # ---------------------------------------------

            with st.spinner(
                "Publishing..."
            ):

                res = run_async(

                    run_graph_with_postgres(

                        st.session_state.thread_id,

                        action_type="resume",

                        confirm_publish=True,

                        token=(
                            st.session_state
                            .linkedin_token
                        ),
                    )
                )

            apply_graph_result(
                res
            )

            st.rerun()

    # =====================================================
    # CANCEL
    # =====================================================

    with col2:

        if st.button(

            "Cancel",

            use_container_width=True,
        ):

            with st.spinner(
                "Cancelling..."
            ):

                res = run_async(

                    run_graph_with_postgres(

                        st.session_state.thread_id,

                        action_type="resume",

                        confirm_publish=False,

                        token=(
                            st.session_state
                            .linkedin_token
                        ),
                    )
                )

            apply_graph_result(
                res
            )

            st.rerun()


# =========================================================
# CHAT INPUT
# =========================================================

elif user_input := st.chat_input(

    "Ask something or generate a LinkedIn post..."
):

    if not st.session_state.is_processing:

        st.session_state.chat_history.append(

            {
                "role": "user",

                "content": user_input,
            }
        )

        st.session_state.is_processing = True

        st.rerun()


# =========================================================
# PROCESS USER MESSAGE
# =========================================================

if (

    st.session_state.chat_history

    and
    st.session_state.chat_history[-1]["role"]
    == "user"

    and
    not st.session_state.interrupt_state

    and
    st.session_state.is_processing

):

    res, streamed_text = (
        stream_agent_response(

            thread_id=(
                st.session_state.thread_id
            ),

            user_input=(
                st.session_state
                .chat_history[-1]["content"]
            ),

            token=(
                st.session_state
                .linkedin_token
            ),
        )
    )

    apply_graph_result(
        res,
        streamed_text
    )

    st.session_state.is_processing = False

    st.rerun()