"""
Tests for ``qanary.logger``.

Logging is configured as a side effect of importing the package, which means
it is configured exactly once and every later ``get_logger`` call has to be
cheap and idempotent. The module also quietens the ``websockets`` logger used
by Qimchi Connect, which would otherwise print connection messages in the
middle of a measurement's progress bar. The configured flag is process-global,
so each test that changes it restores it afterwards.

"""

import logging
import sys

import pytest

from qanary import logger as logger_module
from qanary.logger import get_logger


@pytest.fixture
def unconfigured():
    """
    Pretend logging has not been set up yet.

    Yields:
        None: With the module flag cleared for the duration of the test.

    """
    root_logger = logging.getLogger()
    qanary_logger = logging.getLogger("qanary")
    websockets_logger = logging.getLogger("websockets")
    original = {
        "configured": logger_module._LOGGING_CONFIGURED,
        "root_handlers": root_logger.handlers[:],
        "root_level": root_logger.level,
        "qanary_level": qanary_logger.level,
        "websockets_level": websockets_logger.level,
    }
    logger_module._LOGGING_CONFIGURED = False
    try:
        yield
    finally:
        for handler in root_logger.handlers:
            if handler not in original["root_handlers"]:
                handler.close()
        root_logger.handlers[:] = original["root_handlers"]
        for handler in original["root_handlers"]:
            if type(handler) is logging.StreamHandler:
                handler.setStream(sys.__stdout__)
        root_logger.setLevel(original["root_level"])
        qanary_logger.setLevel(original["qanary_level"])
        websockets_logger.setLevel(original["websockets_level"])
        logger_module._LOGGING_CONFIGURED = original["configured"]


def test_import_configures_logging_once():
    """The module calls ``_setup_logging`` at import time."""
    assert logger_module._LOGGING_CONFIGURED is True


def test_returns_the_logger_for_the_requested_module():
    assert get_logger("qanary.measure").name == "qanary.measure"


def test_the_same_name_gives_the_same_logger():
    assert get_logger("qanary.sweep") is get_logger("qanary.sweep")


def test_the_first_call_configures_logging(unconfigured):
    get_logger("qanary.something")

    assert logger_module._LOGGING_CONFIGURED is True


def test_reconfiguring_only_adjusts_the_level():
    """
    Once configured, ``_setup_logging`` must not call ``basicConfig`` again --
    it is called with ``force=True``, which would tear down handlers a user
    added themselves.

    """
    qanary_logger = logging.getLogger("qanary")
    original_level = qanary_logger.level
    try:
        logger_module._setup_logging(level=logging.DEBUG)

        assert qanary_logger.level == logging.DEBUG
    finally:
        qanary_logger.setLevel(original_level)


def test_the_websockets_logger_is_quietened(unconfigured):
    """
    Qimchi Connect's WebSocket transport logs connections at INFO, which would
    scroll the measurement progress bar off the screen.

    """
    logger_module._setup_logging()

    assert logging.getLogger("websockets").level == logging.WARNING


def test_the_detailed_format_carries_a_timestamp_and_module(unconfigured, capsys):
    logger_module._setup_logging(use_detailed_format=True)
    get_logger("qanary.formatted").warning("a message")

    printed = capsys.readouterr().out
    assert "qanary.formatted" in printed
    assert "WARNING" in printed


def test_a_custom_format_string_is_used(unconfigured, capsys):
    logger_module._setup_logging(format_string="[%(levelname)s] %(message)s")
    get_logger("qanary.custom").warning("a message")

    assert "[WARNING] a message" in capsys.readouterr().out


def test_only_get_logger_is_public():
    """
    ``_setup_logging`` runs on import and is not something a measurement
    script should be calling.

    """
    assert logger_module.__all__ == ["get_logger"]
