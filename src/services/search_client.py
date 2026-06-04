from langchain_mcp_adapters.client import MultiServerMCPClient
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
from src.config import constants
import os
import sys


class SearchMCPClient:
    async def get_tools(self):
        try:
            logger.info("Initializing Search MCP Client (streamable-http)...")

            python_path = constants.PYTHON_PATH
            server_path = constants.SEARCH_SERVER_PATH

            client = MultiServerMCPClient(
                {
                    "search": {
                        "command": python_path,
                        "args": server_path,
                        "transport": "stdio",
                    }
                }
            )
            tools = await client.get_tools()
            logger.info(f"Search MCP tools loaded | Count={len(tools)}")
            return tools

        except Exception as e:
            logger.exception("Failed to initialize Search MCP Client")
            raise AutomatedLinkedinPostAgent(e, sys)