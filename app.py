import os
import sys
import asyncio
import uuid
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

def get_event_loop():
    if "event_loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state["event_loop"] = loop
    return st.session_state["event_loop"]

def run_async(coro):
    """Safely run async code in Streamlit."""
    loop = get_event_loop()
    return loop.run_until_complete(coro)

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
        logger.error(f"Error fetching threads for user: {e}")
        return []

async def load_conversation_from_postgres(thread_id):
    DB_URI = os.getenv("DB_URI")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        llm_service = LLMServices()
        model = llm_service.get_model()
        graph_builder = GraphBuilder(model, model, [], [])
        builder = graph_builder.build()
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(checkpointer=checkpointer)
            current_state = await graph.aget_state(config)
            return current_state.values.get('messages', [])
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}")
        return []

async def run_graph_with_postgres(action_type="stream", user_input=None, confirm_publish=True, token=None):
    """
    FIX: Search aur LinkedIn MCP clients ko async context ke andar initialize karo.
    Yahi main issue tha - tools outside async context mein call ho rahe the.
    """
    if token:
        os.environ["LINKEDIN_ACCESS_TOKEN"] = token

    thread_id = st.session_state['thread_id']
    config = {"configurable": {"thread_id": thread_id,"linkedin_access_token": token}}
    DB_URI = os.getenv("DB_URI")

    try:
        llm_service = LLMServices()
        model = llm_service.get_model()

        logger.info("Initializing MCP clients...")
        search_client = SearchMCPClient()
        linkedin_client = LinkedInMCPClient()

        search_tools = await search_client.get_tools()
        linkedin_tools = await linkedin_client.get_tools()

        logger.info(f"Search tools loaded: {[t.name for t in search_tools]}")
        logger.info(f"LinkedIn tools loaded: {[t.name for t in linkedin_tools]}")

        model_with_tools = model.bind_tools(search_tools + linkedin_tools)

        graph_builder = GraphBuilder(model=model,model_with_both_tools=model_with_tools,search_tools=search_tools,linkedin_tools=linkedin_tools)
        builder = graph_builder.build()

        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(checkpointer=checkpointer,interrupt_before=["post_generate_linkedin_tool"])

            if action_type == "stream" and user_input:
                logger.info(f"Streaming user input: {user_input[:50]}...")
                async for event in graph.astream({"messages": [HumanMessage(content=user_input)]},config,stream_mode="values"):
                    pass  

            elif action_type == "resume":
                if confirm_publish:
                    logger.info("User confirmed publish - resuming graph...")
                    async for event in graph.astream(None, config, stream_mode="values"):
                        pass
                    st.session_state.chat_history.append({"role": "agent","content": "Post LinkedIn pe successfully publish ho gayi!"})
                else:
                    logger.info("User cancelled publish")
                    await graph.aupdate_state(config,{"messages": []},as_node="chat_or_post")
                    st.session_state.chat_history.append({"role": "agent","content": "Publishing cancel ho gayi. Kuch aur poochho!"})
                st.session_state.interrupt_state = False
                return  

            current_state = await graph.aget_state(config)
            logger.info(f"[DEBUG] current_state.next = {current_state.next}")

            is_interrupted = bool(current_state.next and "post_generate_linkedin_tool" in current_state.next)
            st.session_state.interrupt_state = is_interrupted

            if is_interrupted:
                messages = current_state.values.get("messages", [])
                post_text = ""
                for msg in reversed(messages):
                    if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                        post_text = msg.content
                        break
                st.session_state.post_content = post_text
                logger.info("Graph interrupted before LinkedIn publish")

            else:
                msgs = current_state.values.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    if isinstance(last_msg.content, list):
                        content = " ".join(block.get("text", "") for block in last_msg.content if isinstance(block, dict))
                    else:
                        content = last_msg.content

                    if content.strip():
                        st.session_state.chat_history.append({"role": "agent","content": content})

    except Exception as e:
        logger.exception(f"Graph run failed: {e}")
        st.session_state.chat_history.append({"role": "agent","content": f"Error: {str(e)}"})

def reset_chat(current_user):
    new_id = f"{current_user}_thread_{str(uuid.uuid4())[:8]}"
    st.session_state['thread_id'] = new_id
    if new_id not in st.session_state.get('chat_threads', []):
        st.session_state['chat_threads'].append(new_id)
    st.session_state.chat_history = []
    st.session_state.interrupt_state = False
    st.session_state.post_content = ""

st.set_page_config(
    page_title="LinkedIn Automation Agent",
    page_icon="",
    layout="centered"
)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

st.sidebar.subheader("👤 User Account")

if not st.session_state["user_id"]:
    user_email = st.sidebar.text_input("Enter your Email ID:")
    if st.sidebar.button("Login"):
        if user_email:
            st.session_state["user_id"] = user_email.lower().strip()
            existing_threads = run_async(get_all_threads_for_user(st.session_state["user_id"]))
            st.session_state['chat_threads'] = (
                existing_threads if existing_threads
                else [f"{st.session_state['user_id']}_thread_{str(uuid.uuid4())[:8]}"]
            )
            st.session_state['thread_id'] = st.session_state['chat_threads'][0]
            st.rerun()
else:
    st.sidebar.write(f"Logged in: **{st.session_state['user_id']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

if not st.session_state["user_id"]:
    st.info("Log in and access your LinkedIn Agent.")
    st.stop()

CURRENT_USER = st.session_state["user_id"]

# --- 8. SESSION STATE INIT ---
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = f"{CURRENT_USER}_thread_{str(uuid.uuid4())[:8]}"
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interrupt_state" not in st.session_state:
    st.session_state.interrupt_state = False
if "post_content" not in st.session_state:
    st.session_state.post_content = ""

st.sidebar.subheader("Chat Threads")
if st.sidebar.button('New Chat'):
    reset_chat(CURRENT_USER)
    st.rerun()

if st.session_state.get('chat_threads'):
    thread_options = st.session_state['chat_threads']
    selected_thread = st.sidebar.selectbox(
        "Select Thread:",
        thread_options,
        index=thread_options.index(st.session_state['thread_id'])
        if st.session_state['thread_id'] in thread_options else 0
    )
    if selected_thread != st.session_state['thread_id']:
        st.session_state['thread_id'] = selected_thread
        msgs = run_async(load_conversation_from_postgres(selected_thread))
        st.session_state.chat_history = [
            {"role": "user" if isinstance(m, HumanMessage) else "agent", "content": m.content}
            for m in msgs if hasattr(m, 'content') and isinstance(m.content, str)
        ]
        st.rerun()

linkedin_token = st.sidebar.text_input("LinkedIn Access Token", type="password")

st.title("AI LinkedIn Post Generator")

for msg in st.session_state.chat_history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if st.session_state.interrupt_state:
    with st.container():
        st.warning("The agent wants to post on LinkedIn. Do you approve?")

        if st.session_state.post_content:
            with st.expander("Check the post preview", expanded=True):
                st.write(st.session_state.post_content)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Publish!", use_container_width=True):
                with st.spinner("Publishing..."):
                    run_async(run_graph_with_postgres(
                        action_type="resume",
                        confirm_publish=True,
                        token=linkedin_token
                    ))
                st.rerun()
        with col2:
            if st.button("Cancel", use_container_width=True):
                with st.spinner("Cancelling..."):
                    run_async(run_graph_with_postgres(
                        action_type="resume",
                        confirm_publish=False,
                        token=linkedin_token
                    ))
                st.rerun()

elif user_input := st.chat_input("Ask the agent something."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.rerun()

if (
    st.session_state.chat_history
    and st.session_state.chat_history[-1]["role"] == "user"
    and not st.session_state.interrupt_state
):
    with st.spinner("Thinking..."):
        run_async(run_graph_with_postgres(
            action_type="stream",
            user_input=st.session_state.chat_history[-1]["content"],
            token=linkedin_token
        ))
    st.rerun()