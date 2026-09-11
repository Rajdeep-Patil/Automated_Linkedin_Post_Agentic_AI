from langchain_core.messages import ToolMessage
from src.graph.state import AgentState
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys


def _stringify_tool_result(result) -> str:
    """
    MCP tools (via langchain_mcp_adapters) often return a list of
    content blocks like [{'type': 'text', 'text': '...', 'id': '...'}]
    instead of a plain string. Extract the actual text instead of
    dumping the raw Python repr of the list.
    """
    if isinstance(result, list):
        texts = [
            item.get("text", "")
            for item in result
            if isinstance(item, dict) and "text" in item
        ]
        if texts:
            return "\n".join(texts)
        return str(result)

    if isinstance(result, dict) and "text" in result:
        return result["text"]

    return str(result)


class LinkedInToolNode:
    def __init__(self, linkedin_tools: list):
        self.linkedin_tools = linkedin_tools

    async def linkedin_tool_node(self, state: AgentState) -> dict:
        try:
            tool_map = {t.name: t for t in self.linkedin_tools}
            last_message = state["messages"][-1]
            outputs = []

            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = dict(tool_call["args"])

                if tool_name == "linkedin_post":
                    tool_args["linkedin_access_token"] = state.get("linkedin_access_token", "")

                tool = tool_map.get(tool_name)
                if tool is None:
                    result_text = f"Tool '{tool_name}' not found."
                else:
                    raw_result = await tool.ainvoke(tool_args)
                    result_text = _stringify_tool_result(raw_result)

                outputs.append(ToolMessage(content=result_text, tool_call_id=tool_call["id"]))

            logger.info("LinkedIn tool node completed")
            return {"messages": outputs}

        except Exception as e:
            logger.exception("LinkedIn tool node failed")
            raise AutomatedLinkedinPostAgent(e, sys)