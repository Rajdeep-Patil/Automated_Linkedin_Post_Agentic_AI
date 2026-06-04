from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import AgentState,PostScoreSchema
from src.prompts.post_score_prompt import post_score_system_prompt
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys

class PostScoreNode:
    def __init__(self, model):
        self.model = model
        #self.output_parser = (PydanticOutputParser(pydantic_object=PostScoreSchema))
        self.structured_llm = model.with_structured_output(PostScoreSchema)
        self.prompt = (
            ChatPromptTemplate.from_messages(
                [("system",post_score_system_prompt),
                ("human","Score this LinkedIn post:\n\n{post}")
                ]))#.partial(format_instructions=self.output_parser.get_format_instructions()))

    def post_score(self,state: AgentState) -> dict:
        try:
            post = state["linkedin_post_text"]
            if not post:
                logger.warning("No LinkedIn post found for scoring")
                return {"score": 0.0}

            logger.info("Post scoring started")
            chain = self.prompt | self.structured_llm
            result = chain.invoke({"post": post})
            logger.info(f"Post score={result.score}")
            return {"score": result.score}

        except Exception as e:
            logger.exception("PostScoreNode failed")
            raise AutomatedLinkedinPostAgent(e,sys)