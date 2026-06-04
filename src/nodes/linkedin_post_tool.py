from langchain_core.messages import ToolMessage
from src.graph.state import AgentState
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys


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
                    result = f"Tool '{tool_name}' not found."
                else:
                    result = await tool.ainvoke(tool_args)

                outputs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            logger.info("LinkedIn tool node completed")
            return {"messages": outputs}

        except Exception as e:
            logger.exception("LinkedIn tool node failed")
            raise AutomatedLinkedinPostAgent(e, sys)