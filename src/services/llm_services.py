from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_groq import ChatGroq
from src.config import constants
from src.logging.logger import logger
from src.exception.exception import AutomatedLinkedinPostAgent
import sys
from dotenv import load_dotenv
load_dotenv()

class LLMServices:
    def __init__(self):
        try:
            logger.info("Initializing HuggingFace LLM...")
            # llm = HuggingFaceEndpoint(
            #     repo_id=constants.DEFAULT_MODEL,
            #     task=constants.MODEL_TASK)

            # self.model = ChatHuggingFace(llm=llm)
            self.model = ChatGroq(model="llama-3.3-70b-versatile")
            # self.model = ChatGroq(model="google/gemini-2.5-flash")
            logger.info("LLM initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize LLM")
            raise AutomatedLinkedinPostAgent(e,sys)

    def get_model(self):
        try:
            logger.info("Returning LLM instance")
            return self.model
        except Exception as e:
            logger.exception("Failed to get model")
            raise AutomatedLinkedinPostAgent(e,sys)