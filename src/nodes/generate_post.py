from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages
from src.graph.state import AgentState
from src.config import constants
from src.prompts.generate_post_prompt import generate_post_system_prompt
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys

class GeneratePostNode:
    def __init__(self, model, model_with_both_tools):
        self.model = model
        self.model_with_both_tools = model_with_both_tools
        self.search_tools = []
        self.linkedin_tools = []
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", generate_post_system_prompt),
            ("placeholder", "{messages}")  
        ])

    def generate_post(self, state: AgentState) -> dict:
        try:
            trimmed_messages = trim_messages(
                state['messages'],
                max_tokens=constants.max_context_tokens,
                token_counter=self.model,
                strategy=constants.trim_strategy,
                include_system=constants.include_system,
            )

            logger.info("LinkedIn post generation started")
            chain = self.prompt | self.model_with_both_tools
            result = chain.invoke({"messages": trimmed_messages})  # ✅ Fix 1
            logger.info("LinkedIn post generation completed")

            return {"linkedin_post_text": result.content,"messages": [result]}

        except Exception as e:
            logger.exception("GeneratePostNode failed")
            raise AutomatedLinkedinPostAgent(e, sys)
