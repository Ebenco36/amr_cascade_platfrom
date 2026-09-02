"""Logger creation."""

from __future__ import annotations

import logging


class LoggerFactory:
    """Centralized logger configuration."""

    @staticmethod
    def configure(level: int = logging.INFO) -> None:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)
