import logging
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(filename)s - %(name)s - %(funcName)s: %(message)s"
LOG_DIR_NAME = "logs"

if getattr(sys, "frozen", False):
    p_this_file = Path(sys.executable)
    p_top = p_this_file.resolve().parent
else:
    p_this_file = Path(__file__)
    p_top = p_this_file.resolve().parent.parent.parent

P_LOG_DIR = p_top / LOG_DIR_NAME
MAX_LOG_COUNT = 20


def set_logger(name: str, is_active_stream: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not P_LOG_DIR.exists():
        P_LOG_DIR.mkdir()

    now_time = datetime.now()
    formatter = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

    # Setting for file output
    fh = logging.FileHandler(filename=P_LOG_DIR / f"{now_time:log_%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    # Setting of console output
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    if is_active_stream:
        logger.addHandler(sh)

    return logger


def delete_backlog() -> None:

    files = sorted(P_LOG_DIR.glob("*.log"), key=lambda f: os.stat(f).st_mtime, reverse=False)
    for i, file in enumerate(files, 1):
        if i > len(files) - MAX_LOG_COUNT:
            break
        os.remove(file)
