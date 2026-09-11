import os, sys, asyncio, uuid, queue, warnings
import concurrent.futures
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="LinkedIn Automation Agent", page_icon="💼", layout="centered")

# Secrets Sync
try:
    for k, v in st.secrets.items(): os.environ[k] = v
except Exception: pass

# Imports
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.services.llm_services import LLMServices
from src.services.search_client import SearchMCPClient
from src.services.linkedin_client import LinkedInMCPClient
from src.graph.builder import GraphBuilder
from src.logging.logger import logger

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"local_user_thread_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interrupt_state" not in st.session_state:
    st.session_state.interrupt_state = False
if "post_content" not in st.session_state:
    st.session_state.post_content = ""

async def run_graph_with_postgres(thread_id: str, action_type="stream", user_input=None, confirm_publish=True, token=None, chunk_queue=None):
    result = {"messages": [], "interrupt_state": False, "post_content": "", "error": None}
    if token: os.environ["LINKEDIN_ACCESS_TOKEN"] = token
    config = {"configurable": {"thread_id": thread_id}}

    try:
        model = LLMServices().get_model()
        search_tools = await SearchMCPClient().get_tools()
        linkedin_tools = await LinkedInMCPClient().get_tools()
        model_with_tools = model.bind_tools(search_tools + linkedin_tools)

        builder = GraphBuilder(model=model, model_with_both_tools=model_with_tools, search_tools=search_tools, linkedin_tools=linkedin_tools).build()

        async with AsyncPostgresSaver.from_conn_string(os.getenv("DB_URI")) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(checkpointer=checkpointer, interrupt_before=["post_generate_linkedin_tool"])

            if action_type == "stream" and user_input:
                initial_state = {"messages": [HumanMessage(content=user_input)], "iteration": 0, "max_iteration": 3, "score": 0.0, "linkedin_access_token": token or ""}
                
                async for event in graph.astream_events(initial_state, config, version="v2"):
                    if event.get("event") == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content"):
                            text = chunk.content if isinstance(chunk.content, str) else "".join(b.get("text", "") for b in chunk.content if isinstance(b, dict))
                            if text and chunk_queue: chunk_queue.put({"type": "chunk", "text": text})

                current_state = await graph.aget_state(config)
                is_interrupted = bool(current_state.next and "post_generate_linkedin_tool" in current_state.next)
                result["interrupt_state"] = is_interrupted

                msgs = current_state.values.get("messages", [])
                if is_interrupted:
                    result["post_content"] = next((m.content for m in reversed(msgs) if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()), "")
                elif (score := current_state.values.get("score")) and score > 0:
                    result["messages"].append({"role": "agent", "content": f"Post Score: {score}/10"})

            elif action_type == "resume":
                if confirm_publish:
                    if not token or not token.strip():
                        result["error"] = "LinkedIn Access Token missing hai! Sidebar me enter karein."
                        return result

                    await graph.aupdate_state(config, {"linkedin_access_token": token})
                    try:
                        async for _ in graph.astream(None, config, stream_mode="values"): pass
                    except Exception as publish_err:
                        result["error"] = f"LinkedIn API Publish Failed: {str(publish_err)}"
                        return result

                    current_state = await graph.aget_state(config)
                    msgs = current_state.values.get("messages", [])
                    last_msg = msgs[-1] if msgs else None
                    
                    if last_msg and hasattr(last_msg, "content"):
                        txt = " ".join([i.get("text", "") for i in last_msg.content if isinstance(i, dict)]) if isinstance(last_msg.content, list) else str(last_msg.content)
                        tool_result = txt if txt.strip() else "🎉 Post published successfully on LinkedIn!"
                    else:
                        tool_result = "🎉 Post published successfully on LinkedIn!"

                    result["messages"].append({"role": "agent", "content": tool_result})
                else:
                    await graph.aupdate_state(config, {"cancel_publish": True}, as_node="post_generate_linkedin_tool")
                    result["messages"].append({"role": "agent", "content": "Publishing cancelled."})

                result["interrupt_state"] = False
                result["post_content"] = ""

    except Exception as e:
        logger.exception(f"Graph run failed: {e}")
        result["error"] = str(e)
    finally:
        if chunk_queue: chunk_queue.put({"type": "done"})

    return result

# Stream Response Helper
def stream_agent_response(thread_id: str, user_input: str, token: str):
    chunk_q = queue.Queue()
    future = _THREAD_POOL.submit(asyncio.run, run_graph_with_postgres(thread_id, "stream", user_input, token=token, chunk_queue=chunk_q))

    with st.chat_message("assistant"):
        placeholder, full_text = st.empty(), ""
        while True:
            try:
                item = chunk_q.get(timeout=60)
                if item["type"] == "done": break
                full_text += item["text"]
                placeholder.markdown(full_text + "▌")
            except queue.Empty: break
        if full_text: placeholder.markdown(full_text)

    return future.result(timeout=300), full_text

def apply_result(res, streamed_text=""):
    if res.get("error"):
        st.session_state.chat_history.append({"role": "agent", "content": f"⚠️ **Error:** {res['error']}"})
    else:
        if streamed_text:
            st.session_state.chat_history.append({"role": "agent", "content": streamed_text})
        for msg in res.get("messages", []):
            if msg["content"] not in streamed_text:
                st.session_state.chat_history.append(msg)
        st.session_state.interrupt_state = res["interrupt_state"]
        st.session_state.post_content = res["post_content"]

st.sidebar.subheader("LinkedIn Integration")
token = st.sidebar.text_input("Access Token", type="password", key="linkedin_token")

if st.sidebar.button("➕ Reset Chat"):
    st.session_state.thread_id = f"local_user_thread_{uuid.uuid4().hex[:8]}"
    st.session_state.chat_history = []
    st.session_state.interrupt_state = False
    st.session_state.post_content = ""
    st.rerun()

st.title("💼 LinkedIn Post Automation Agent")

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Approval Card
if st.session_state.interrupt_state:
    st.warning("⚠️ **Publish Approval Required**")
    if st.session_state.post_content: st.info(st.session_state.post_content)

    c1, c2 = st.columns(2)
    if c1.button("✅ Yes, Publish!", use_container_width=True):
        with st.spinner("Publishing..."):
            res = _THREAD_POOL.submit(asyncio.run, run_graph_with_postgres(st.session_state.thread_id, "resume", confirm_publish=True, token=token)).result()
            apply_result(res)
            st.rerun()

    if c2.button("❌ Cancel", use_container_width=True):
        res = _THREAD_POOL.submit(asyncio.run, run_graph_with_postgres(st.session_state.thread_id, "resume", confirm_publish=False, token=token)).result()
        apply_result(res)
        st.rerun()

# User Chat Input
elif user_prompt := st.chat_input("Ask agent to generate or post content..."):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"): st.markdown(user_prompt)

    res, text = stream_agent_response(st.session_state.thread_id, user_prompt, token)
    apply_result(res, text)
    st.rerun()