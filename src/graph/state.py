from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    chat_or_post_dec : str 
    messages: Annotated[list, add_messages]
    linkedin_post_text: str
    score: float
    iteration: int
    max_iteration: int
    linkedin_access_token: str
    summary: str
    
    
class ChatPostSchema(BaseModel):
    decision : Literal["post_generation","normal_chat"]
    
class PostScoreSchema(BaseModel):
    score: float = Field(ge=0.0,le=10.0,description="LinkedIn post score between 0.0 and 10.0")