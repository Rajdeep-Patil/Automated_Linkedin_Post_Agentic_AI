from langgraph.graph import END
from src.graph.state import AgentState
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
import sys

def chat_or_post_router(state: AgentState) -> str:
    try:
        decision = state['chat_or_post_dec']
        logger.info(f"chat_or_post_router started | Decision={decision}")

        if decision == "post_generation":
            logger.info("chat_or_post_router → generate_post")
            return 'generate_post'
        else:
            logger.info("chat_or_post_router → chat")
            return 'chat'

    except Exception as e:
        logger.exception("chat_or_post_router failed")
        raise AutomatedLinkedinPostAgent(e, sys)


def generate_post_tool_router(state: AgentState) -> str:
    try:
        messages = state["messages"]
        last_msg = messages[-1]

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tool_name = last_msg.tool_calls[0]["name"]

            if tool_name == "linkedin_post":
                logger.info("generate_post_tool_router → post_generate_linkedin_tool")
                return "post_generate_linkedin_tool"

            elif tool_name == "Search_tools":
                logger.info("generate_post_tool_router → post_generate_search_tools")
                return "post_generate_search_tools"

        logger.info("generate_post_tool_router → post_score")
        return "post_score"

    except Exception as e:
        logger.exception("generate_post_tool_router failed")
        raise AutomatedLinkedinPostAgent(e, sys)


def post_score_router(state: AgentState) -> str:
    try:
        score = state.get("score", 0.0)
        iteration = state.get("iteration", 0)
        max_iteration = state.get("max_iteration", 3)

        if score < 7.0 and iteration < max_iteration:
            logger.info(f"post_score_router → regenerate_post | "f"Score={score} Iteration={iteration}")
            return "regenerate_post"

        logger.info(f"post_score_router → END | "f"Score={score} Iteration={iteration}")
        return END

    except Exception as e:
        logger.exception("post_score_router failed")
        raise AutomatedLinkedinPostAgent(e, sys)