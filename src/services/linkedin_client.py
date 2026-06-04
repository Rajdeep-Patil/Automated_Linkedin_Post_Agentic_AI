from langchain_mcp_adapters.client import MultiServerMCPClient
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
from src.config import constants
import sys
import os


class LinkedInMCPClient:
    async def get_tools(self):
        try:
            logger.info("Initializing LinkedIn MCP Client")
            client = MultiServerMCPClient(
                {
                    "linkedin": {
                        "command": constants.PYTHON_PATH,
                        "args": [constants.LINKEDIN_SERVER_PATH ],
                        "transport": "stdio",
                    }
                }
            )
            logger.info("Loading LinkedIn MCP tools...")
            tools = await client.get_tools()
            logger.info(f"LinkedIn MCP tools loaded successfully | Count={len(tools)}")
            return tools

        except Exception as e:
            logger.exception("Failed to initialize LinkedIn MCP Client")
            raise AutomatedLinkedinPostAgent(e, sys)