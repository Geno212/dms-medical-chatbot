"""Pipeline trace logging.

Every graph node logs its decision (router intent/action, extracted slots,
entity-resolution scores, retrieval hits) at INFO to a dedicated "chatbot"
logger, separate from noisy third-party loggers (httpx, langgraph). This is
the demo/interview view into "LLM understands, code decides": you can see
exactly which parts of a response came from the model and which were
verified deterministically against the database.

Usage: call configure_logging() once at process start (done automatically by
app.graph.build.build_graph). Set LOG_LEVEL=WARNING to silence, DEBUG for
raw LLM payloads.
"""

from __future__ import annotations

import logging

from .config import Config, get_config

LOGGER_NAME = "chatbot"

_configured = False


def configure_logging(config: Config | None = None) -> logging.Logger:
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    config = config or get_config()
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        # Third-party libraries stay quiet unless the user asks for DEBUG.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        _configured = True
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
