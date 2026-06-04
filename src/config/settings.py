# import os
# from src.config import constants
# from dotenv import load_dotenv
# load_dotenv()

# class ModelConfig:
#     def __init__(self):
#         self.model_name = constants.MODEL_NAME
#         self.temperature = constants.TEMPERATURE
#         self.max_tokens = constants.MAX_TOKENS
#         self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# class TavilyConfig:
#     def __init__(self):
#         self.api_key = os.getenv(
#             "TAVILY_API_KEY",
#             ""
#         )


# class LinkedInConfig:
#     def __init__(self):
#         self.access_token = os.getenv(
#             "LINKEDIN_ACCESS_TOKEN",
#             ""
#         )


# class Settings:
#     def __init__(self):
#         self.model = ModelConfig()
#         self.tavily = TavilyConfig()
#         self.linkedin = LinkedInConfig()


# settings = Settings()