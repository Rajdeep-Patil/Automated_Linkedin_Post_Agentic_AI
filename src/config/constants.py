# # src/config/constants.py

# #ChatNode
# max_context_tokens: int = 4000
# trim_strategy: str = "last"
# include_system : bool= True
# ChatNodePrompt: str = ""
# # Graph Nodes

# CHAT_NODE = "chat"
# GENERATE_POST_NODE = "generate_post"
# POST_SCORE_NODE = "postscore"
# REGENERATE_POST_NODE = "regenerate_post"

# # Routing Decisions

# POST_GENERATION = "post_generation"
# NORMAL_CHAT = "normal_chat"

# # LinkedIn Post Settings

# MIN_POST_SCORE = 5.0
# MAX_REGENERATION = 3

# # Models

# DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
# MODEL_TASK = "text-generation"


# # Messages

# PUBLISH_SUCCESS = "Post published successfully."
# PUBLISH_FAILED = "Failed to publish post."

# # Thread Config

# DEFAULT_THREAD_ID = "1"
# DEFAULT_USER_ID = "guest"

# #client 
# PYTHON_PATH = r"E:\Automated_LinkedIn_Post_Agent\AutomatedLinkedinPostAgent\Scripts\python.exe"

# SEARCH_SERVER_PATH = r"E:\Automated_LinkedIn_Post_Agent\src\tools\search_server.py"

# LINKEDIN_SERVER_PATH = r"E:\Automated_LinkedIn_Post_Agent\src\tools\linkedin_server.py"



# src/config/constants.py
import sys
import os

# ChatNode
max_context_tokens: int = 4000
trim_strategy: str = "last"
include_system: bool = True
ChatNodePrompt: str = ""

# Graph Nodes
CHAT_NODE = "chat"
GENERATE_POST_NODE = "generate_post"
POST_SCORE_NODE = "postscore"
REGENERATE_POST_NODE = "regenerate_post"

# Routing Decisions
POST_GENERATION = "post_generation"
NORMAL_CHAT = "normal_chat"

# LinkedIn Post Settings
MIN_POST_SCORE = 5.0
MAX_REGENERATION = 3

# Models
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
MODEL_TASK = "text-generation"

# Search
SEARCH_RESULTS_LIMIT = 5

# Messages
PUBLISH_SUCCESS = "Post published successfully."
PUBLISH_FAILED = "Failed to publish post."

# Thread Config
DEFAULT_THREAD_ID = "1"
DEFAULT_USER_ID = "guest"

# ── Cross-platform paths ──────────────────────────────────────────────────────
# Windows pe venv ka python, Linux/Mac pe system python
# if sys.platform == "win32":
#     PYTHON_PATH = os.getenv(
#         "PYTHON_PATH",
#         r"E:\Automated_LinkedIn_Post_Agent\AutomatedLinkedinPostAgent\Scripts\python.exe"
#     )
# else:
#     PYTHON_PATH = sys.executable  # Streamlit Cloud pe current python use karo
    
# PYTHON_PATH = r"E:\Automated_LinkedIn_Post_Agent\AutomatedLinkedinPostAgent\Scripts\python.exe"

# # Project root — jahan bhi script chal rahi ho wahan se src/ dhundo
# _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SEARCH_SERVER_PATH = [r"E:\Automated_LinkedIn_Post_Agent\src\tools\search_server.py"]
# LINKEDIN_SERVER_PATH = [r"E:\Automated_LinkedIn_Post_Agent\src\tools\linkedin_server.py"]

import sys
import os

PYTHON_PATH = sys.executable

# constants.py ki location se src/services tak ka path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # yeh src/services/ hai
SEARCH_SERVER_PATH = [os.path.join(BASE_DIR, "src","tools","search_server.py")]
LINKEDIN_SERVER_PATH = [os.path.join(BASE_DIR, "src","tools","search_server.py")]