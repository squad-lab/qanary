"""
Centralized logging configuration for qcutils.

"""

import logging
import sys
from typing import Optional

# Public API. `_setup_logging` is called on import.
__all__ = [
    "get_logger",
]


# Default logging format
DEFAULT_FORMAT = "%(levelname)s: %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Global flag to track if logging has been configured
_LOGGING_CONFIGURED = False


def _setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_detailed_format: bool = False,
) -> None:
    """
    Configure logging for qcutils.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Custom format string for log messages
        use_detailed_format: If True, use detailed format with timestamp and module name

    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        logging.getLogger("qcutils").setLevel(level)
        return

    if format_string is None:
        format_string = DETAILED_FORMAT if use_detailed_format else DEFAULT_FORMAT

    # Configure root logger for qcutils
    logging.basicConfig(
        format=format_string,
        level=level,
        stream=sys.stdout,
        force=True,
    )

    qcutils_logger = logging.getLogger("qcutils")
    qcutils_logger.setLevel(level)

    # Ensure the websockets library logs only warnings or above to avoid "connection open" spam
    logging.getLogger("websockets").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a qcutils module.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logger instance

    """
    if not _LOGGING_CONFIGURED:
        _setup_logging()

    return logging.getLogger(name)


# Initialize logging with default configuration
_setup_logging()
