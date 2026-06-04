import os
import sys
import logging
import requests
from fastmcp import FastMCP, Context
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from pydantic import Field

mcp = FastMCP("linkedin_server")

@mcp.tool()
def linkedin_post(post_text: str, linkedin_access_token:Annotated[str, Field(exclude=True, default="")] = "")-> str:
    """
    Publish a LinkedIn post on LinkedIn.

    Use only when the user explicitly asks to:
    - publish a post
    - post it on LinkedIn
    - share it on LinkedIn

    Args:
        post_text: Final LinkedIn post content to publish.

    Returns:
        Publication status.
    """
    token = linkedin_access_token.strip() or os.getenv("LINKEDIN_ACCESS_TOKEN", "").strip()

    if not token:
        return "LinkedIn Access Token is missing! Please add your token in the sidebar and try again."

    
    try:
        profile_response = requests.get(
            "https://api.linkedin.com/v2/userinfo",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )

        if profile_response.status_code != 200:
            return f"Failed to fetch profile: {profile_response.text}"

        person_id  = profile_response.json().get("sub")
        author_urn = f"urn:li:person:{person_id}"

        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": post_text},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            },
        }

        response = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            },
            json=payload,
            timeout=30,
        )

        if response.status_code == 201:
            return f"LinkedIn post published successfully.\n\n{post_text}"

        return f"LinkedIn API Error {response.status_code}: {response.text}"

    except Exception as e:
        raise RuntimeError(str(e))


if __name__ == "__main__":
    mcp.run(transport="stdio")