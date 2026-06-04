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

            url = os.getenv(
                "LINKEDIN_SERVER_URL",
                "https://worthwhile-gold-fly.fastmcp.app/mcp"
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
            logger.info(f"LinkedIn MCP tools loaded | Count={len(tools)}")
            return tools

        except Exception as e:
            logger.exception("Failed to initialize LinkedIn MCP Client")
            raise AutomatedLinkedinPostAgent(e, sys)