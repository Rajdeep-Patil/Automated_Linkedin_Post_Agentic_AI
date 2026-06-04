import sys
import os

PYTHON_PATH = sys.executable

# constants.py ki location se src/services tak ka path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # yeh src/services/ hai
SEARCH_SERVER_PATH = [os.path.join(BASE_DIR, "src","tools","search_server.py")]
print(SEARCH_SERVER_PATH)
E:\Automated_LinkedIn_Post_Agent\src\tools\search_server.py
