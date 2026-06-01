import logging
import sys

log_level_to_abbr = {
    "DEBUG": "D",
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C",
}


class PAIlibLogsFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        record.asctime = self.formatTime(record, "%H:%M:%S")

        shortlevel = log_level_to_abbr.get(record.levelname, record.levelname)
        prefix = f"[{shortlevel} {record.asctime} {record.lineno}]{record.module}"
        lines = s.split("\n")
        return "\n".join(f"{prefix} {li}" for li in lines)


DEFAULT_FORMATTER = PAIlibLogsFormatter()
_logging_enabled = False


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(DEFAULT_FORMATTER)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    logger.disabled = not _logging_enabled
    return logger


def enable_logging(level: int = logging.DEBUG) -> None:
    global _logging_enabled
    _logging_enabled = True

    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.disabled = False
        logger.setLevel(level)


def disable_logging() -> None:
    global _logging_enabled
    _logging_enabled = False

    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.disabled = True


def get_logger(name: str) -> logging.Logger:
    logger = setup_logger(name)
    logger.disabled = not _logging_enabled
    return logger
