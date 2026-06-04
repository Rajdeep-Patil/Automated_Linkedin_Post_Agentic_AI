from langchain_mcp_adapters.client import MultiServerMCPClient
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
from src.config import constants
import os
import sys

class SearchMCPClient:
    async def get_tools(self):
        try:
            logger.info("Initializing Search MCP Client (stdio)...")

            server_path = constants.SEARCH_SERVER_PATH
            python_path = constants.PYTHON_PATH
            url = os.getenv("SEARCH_SERVER_URL","")

            if not server_path or not os.path.exists(server_path):
                raise ValueError(
                    f"Search server script not found at: {server_path}"
                )

            client = MultiServerMCPClient(
                {
                    "search": {
                        "url": url,
                        "transport": "streamable_http",
                    }
                }
            )


            tools = await client.get_tools()
            logger.info(f"Search MCP tools loaded | Count={len(tools)}")
            return tools

        except Exception as e:
            logger.exception("Failed to initialize Search MCP Client")
            raise AutomatedLinkedinPostAgent(e, sys)