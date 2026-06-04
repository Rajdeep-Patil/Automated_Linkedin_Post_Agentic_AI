from langchain_mcp_adapters.client import MultiServerMCPClient
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
from src.config import constants
import sys
import os


class SearchMCPClient:

    async def get_tools(self):

        try:

            logger.info("Initializing Search MCP Client")

            client = MultiServerMCPClient(
                {
                    "search": {
                        "command": constants.PYTHON_PATH,
                        "args": [constants.SEARCH_SERVER_PATH],
                        "transport": "stdio",
                    }
                }
            )

            logger.info("Loading MCP tools...")
            tools = await client.get_tools()
            logger.info(f"MCP tools loaded | Count={len(tools)}")
            return tools

        except Exception as e:
            logger.exception("Failed to initialize Search MCP Client")
            raise AutomatedLinkedinPostAgent(e, sys)