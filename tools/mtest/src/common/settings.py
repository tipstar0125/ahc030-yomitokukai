import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if getattr(sys, "frozen", False):
    p_this_file = Path(sys.executable)
    p_parent = p_this_file.resolve().parent
else:
    p_this_file = Path(__file__)
    p_parent = p_this_file.resolve().parent.parent.parent

P_DOT_ENV_FILE = p_parent / ".env"
load_dotenv(P_DOT_ENV_FILE)

TEST_PATH = os.getenv("TEST_PATH")
print(TEST_PATH)
