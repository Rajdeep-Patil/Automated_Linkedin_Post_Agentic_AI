import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )
from langchain_core.messages import HumanMessage
from src.services.llm_services import LLMServices
from src.services.search_client import SearchMCPClient
from src.services.linkedin_client import LinkedInMCPClient
from src.graph.builder import GraphBuilder
from src.exception.exception import AutomatedLinkedinPostAgent
from src.config import constants
from src.logging.logger import logger


async def run():
    try:
        logger.info("Initializing LLM...")
        llm_service = LLMServices()
        model = llm_service.get_model()

        logger.info("Loading Search MCP tools...")
        search_client = SearchMCPClient()
        search_tools = await search_client.get_tools()
        print(f"Search tools: {[t.name for t in search_tools]}")

        logger.info("Loading LinkedIn MCP tools...")
        linkedin_client = LinkedInMCPClient()
        linkedin_tools = await linkedin_client.get_tools()
        print(f"LinkedIn tools: {[t.name for t in linkedin_tools]}")

        model_with_both_tools = model.bind_tools(search_tools + linkedin_tools)

        logger.info("Building graph...")
        graph_builder = GraphBuilder(
            model=model,
            model_with_both_tools=model_with_both_tools,
            search_tools=search_tools,
            linkedin_tools=linkedin_tools
        )

        await graph_builder.compile_and_run(agent_loop)

    except Exception as e:
        logger.exception("Agent run failed")
        raise AutomatedLinkedinPostAgent(e, sys)


async def agent_loop(graph):
    try:
        thread_id = constants.DEFAULT_THREAD_ID
        config = {"configurable": {"thread_id": thread_id}}

        print("\n" + "=" * 40)
        print("  LinkedIn Automation Agent")
        print("=" * 40)
        print("Type 'bye' to exit\n")

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["bye", "exit", "quit"]:
                logger.info("Session ended by user")
                print("Goodbye!")
                break

            logger.info(f"User input received | Length={len(user_input)}")

            async for event in graph.astream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "iteration": 0,
                    "max_iteration": 3,
                    "score": 0.0,
                },
                config,
                stream_mode="values"
            ):
                pass

            current_state = await graph.aget_state(config)

            print(f"\n[DEBUG] current_state.next = {current_state.next}\n")

            if current_state.next and "post_generate_linkedin_tool" in current_state.next:
                print("\n" + "-" * 40)
                print("Agent LinkedIn pe post karna chahta hai.")
                print("Post content:")
                print("-" * 40)

                messages = current_state.values.get("messages", [])
                post_text = ""
                for msg in reversed(messages):
                    if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                        post_text = msg.content
                        break

                print(post_text or "(No post content found)")
                print("-" * 40)

                confirm = input("\nPublish karna hai? (yes/no): ").strip().lower()

                if confirm in ["yes", "y", "haan", "ha"]:
                    logger.info("User ne publish confirm kiya")
                    async for event in graph.astream(None,config,stream_mode="values"):
                        pass
                    print("\nPost LinkedIn pe publish ho gayi!\n")
                else:
                    logger.info("User ne publish cancel kiya")
                    print("\nPublishing cancelled.\n")

                    await graph.aupdate_state(
                        config,
                        {"messages": []},
                        as_node="chat_or_post"
                    )
                    continue

            state = await graph.aget_state(config)
            messages = state.values.get("messages", [])
            score = state.values.get("score", None)
            post = state.values.get("linkedin_post_text", None)

            if messages:
                last_msg = messages[-1]

                if isinstance(last_msg.content, list):
                    content = " ".join(
                        block.get("text", "")
                        for block in last_msg.content
                        if isinstance(block, dict)
                    )
                else:
                    content = last_msg.content

                print(f"\nAgent: {content}\n")

            if score:
                print(f"Post Score: {score}/10\n")

            if post:
                logger.info(f"LinkedIn post generated | Score={score}")

    except Exception as e:
        logger.exception("Agent loop failed")
        raise AutomatedLinkedinPostAgent(e, sys)
    
if __name__ == "__main__":
    asyncio.run(run())