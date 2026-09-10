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
DEFAULT_MODEL = "openai/gpt-oss-120b"
MODEL_TASK = "text-generation"

# Search
SEARCH_RESULTS_LIMIT = 5

# Messages
PUBLISH_SUCCESS = "Post published successfully."
PUBLISH_FAILED = "Failed to publish post."

# Thread Config
DEFAULT_THREAD_ID = "1"
DEFAULT_USER_ID = "guest"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  
SRC_DIR = os.path.join(PROJECT_ROOT, "src") 

PYTHON_PATH = sys.executable

SEARCH_SERVER_PATH = [os.path.join(SRC_DIR, "tools", "search_server.py")]
LINKEDIN_SERVER_PATH = [os.path.join(SRC_DIR, "tools", "linkedin_server.py")]