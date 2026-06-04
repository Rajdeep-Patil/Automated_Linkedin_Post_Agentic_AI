from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages
from src.graph.state import AgentState
from src.config import constants
from src.prompts.chat_node_prompt import chat_node_system_prompt
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys

class ChatNode:
    def __init__(self, model, model_with_both_tools):
        self.model = model
        self.model_with_both_tools = model_with_both_tools
        self.tools = []

    def chat_node(self, state: AgentState) -> dict:
        try: 
            trimmed_messages = trim_messages(
                state['messages'],
                max_tokens=constants.max_context_tokens,
                token_counter=self.model,
                strategy=constants.trim_strategy,
                include_system=constants.include_system,
            )

            logger.info("Chat node started")  

            prompt = ChatPromptTemplate.from_messages([
                ("system", chat_node_system_prompt),
                ("placeholder", "{messages}")
            ])

            chain = prompt | self.model_with_both_tools
            response = chain.invoke({"messages": trimmed_messages})

            logger.info("Chat node completed")  

            return {"messages": [response]}

        except Exception as e:
            logger.exception("ChatNode failed")
            raise AutomatedLinkedinPostAgent(e, sys)