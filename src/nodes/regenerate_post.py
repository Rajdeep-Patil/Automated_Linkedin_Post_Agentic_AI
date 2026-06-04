from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import AgentState
from src.prompts.regenerate_post_prompt import regenerate_post_system_prompt
from src.exception.exception import AutomatedLinkedinPostAgent
from src.logging.logger import logger
import sys

class RegeneratePostNode:
    def __init__(self, model):
        self.model = model
        self.prompt = ChatPromptTemplate.from_messages(
            [("system",regenerate_post_system_prompt),
                ( "human","""
                            Previous Post (Score: {score}/10)
                            {original_post}
                            Rewrite this post and make it significantly better.
                            """)])

    def regenerate_post(self,state: AgentState) -> dict:
        try:
            score = state["score"]
            original_post = state["linkedin_post_text"]
            iteration = state["iteration"]
            logger.info(f"Post regeneration started | "f"Iteration={iteration} | "f"Score={score}")

            chain = (self.prompt | self.model)
            result = chain.invoke({"score": score,"original_post": original_post})
            logger.info(f"Post regeneration completed | "f"New Iteration={iteration + 1}")

            return {"linkedin_post_text":result.content,"messages":[result],"iteration":iteration + 1,"score":0.0}

        except Exception as e:
            logger.exception("RegeneratePostNode failed")
            raise AutomatedLinkedinPostAgent(e,sys)