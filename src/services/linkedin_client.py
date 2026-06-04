from langchain_mcp_adapters.client import MultiServerMCPClient
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
from src.config import constants
import os
import sys

class LinkedInMCPClient:
    async def get_tools(self):
        try:
            logger.info("Initializing LinkedIn MCP Client (HTTP)...")

            python_path = constants.PYTHON_PATH
            server_path = constants.LINKEDIN_SERVER_PATH

            async with MultiServerMCPClient(
                                    {
                                        "search": {
                                            "command": python_path,
                                            "args": server_path,
                                            "transport": "stdio",
                                        }
                                    }
                                ) as client:

                                    tools = await client.get_tools()
                                    logger.info(f"LinkedIn MCP tools loaded | Count={len(tools)}")
                                    return tools

        except Exception as e:
            logger.exception("Failed to initialize LinkedIn MCP Client")
            raise AutomatedLinkedinPostAgent(e, sys)