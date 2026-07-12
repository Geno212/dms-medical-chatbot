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

# Marker attribute on our handler so we can recognize (and de-duplicate) it
# across module re-imports. A module-level bool is NOT reliable here: Chainlit's
# hot-reload re-imports this module, resetting any global, which would add a new
# handler on every reload and print each line N times. The logger singleton
# itself persists, so we inspect its handlers instead.
_HANDLER_TAG = "chatbot-pipeline-handler"


def configure_logging(config: Config | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    config = config or get_config()
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    already = any(getattr(h, "_tag", None) == _HANDLER_TAG for h in logger.handlers)
    if not already:
        handler = logging.StreamHandler()
        handler._tag = _HANDLER_TAG  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        # Third-party libraries stay quiet unless the user asks for DEBUG.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
