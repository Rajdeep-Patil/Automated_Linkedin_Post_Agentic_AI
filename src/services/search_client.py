from langchain_mcp_adapters.client import MultiServerMCPClient
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
import os
import sys


class SearchMCPClient:
    async def get_tools(self):
        try:
            logger.info("Initializing Search MCP Client (streamable-http)...")

            url = os.getenv(
                "SEARCH_SERVER_URL",
                "https://circular-moccasin-marlin.fastmcp.app/mcp"
            )

            if not url:
                raise ValueError("SEARCH_SERVER_URL environment variable not set!")

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