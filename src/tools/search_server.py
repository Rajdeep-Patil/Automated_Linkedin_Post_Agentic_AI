import logging
import os
from fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="search_server")


@mcp.tool()
def Search_tools(query: str) -> str:
    """
    Search the internet for current, recent, or verifiable information.

    Use this tool when:
    - The user asks about current events, breaking news, or recent developments.
    - The user mentions terms such as today, yesterday, latest, recent, currently, or news.
    - The user asks about companies, products, technologies, or public figures.
    - The user requests fact-checking or verification.
    - The user explicitly asks to search the web or online sources.

    Do NOT use when:
    - The user is having casual conversation.
    - The answer can be provided from general knowledge.
    - The task is creative writing or LinkedIn post generation without research.

    Args:
        query: A clear and specific search query.

    Returns:
        Relevant information from web search results.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set!")

        client = TavilyClient(api_key=api_key)
        answer = client.search(query=query)
        return "\n\n".join(result.get("content", "") for result in answer.get("results", []))
    except Exception as e:
        raise RuntimeError(str(e))


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)