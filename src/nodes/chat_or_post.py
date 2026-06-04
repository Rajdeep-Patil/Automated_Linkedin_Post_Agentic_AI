from langchain_core.output_parsers import PydanticOutputParser 
from src.graph.state import AgentState, ChatPostSchema
from langchain_core.prompts import ChatPromptTemplate
from src.prompts.chat_or_post_prompt import chat_or_post_system_prompt
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys

class ChatOrPostNode:
    def __init__(self, model):
        self.model = model
        # self.output_parser = PydanticOutputParser(pydantic_object=ChatPostSchema)
        self.structured_llm = model.with_structured_output(ChatPostSchema)
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", chat_or_post_system_prompt),
            ("human", "{query}")])#.partial(format_instructions=self.output_parser.get_format_instructions())

    def chat_or_post(self,state: AgentState):

        try:
            question = (state["messages"][-1].content)
            logger.info(f"Routing started | Query={question}")
            chain = self.prompt | self.structured_llm
            result = chain.invoke({"query": question})
            logger.info(f"Routing decision={result.decision}")
            if not result:
                return {"chat_or_post_dec": "normal_chat"}

            return {"chat_or_post_dec":result.decision}

        except Exception as e:
            logger.exception("ChatOrPostNode failed")
            raise AutomatedLinkedinPostAgent(e,sys)