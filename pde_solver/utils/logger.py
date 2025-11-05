"""Professional logging system for PDE solver."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

# Optional dependency - logger works without it. On Windows, torch import may raise
# non-ImportError exceptions (e.g., OSError due to DLL issues). Be resilient.
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 - intentionally broad
    TORCH_AVAILABLE = False


class StructuredLogger:
    """Structured logger with JSON support and multiple handlers."""

    def __init__(
        self,
        name: str = "pde_solver",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        json_logging: bool = False,
    ):
        """Initialize structured logger.

        Parameters
        ----------
        name : str
            Logger name
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file : str, optional
            Path to log file
        json_logging : bool
            Enable JSON-formatted logs
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()  # Remove default handlers

        # Prevent duplicate logs
        self.logger.propagate = False

        # Format
        if json_logging:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Error file handler
        if log_file:
            error_log_file = str(log_path.parent / f"error_{log_path.name}")
            error_handler = logging.FileHandler(error_log_file)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with extra fields."""
        extra = kwargs if kwargs else {}
        self.logger.log(level, message, extra=extra)

    def log_metric(self, name: str, value: float, **metadata):
        """Log a metric with metadata."""
        self.info(f"Metric: {name} = {value}", metric_name=name, metric_value=value, **metadata)

    def log_timing(self, operation: str, duration: float, **metadata):
        """Log timing information."""
        self.info(
            f"Timing: {operation} took {duration:.4f}s",
            operation=operation,
            duration=duration,
            **metadata
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info",
            ]:
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Global logger instance
_logger_instance: Optional[StructuredLogger] = None


def get_logger(
    name: str = "pde_solver",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> StructuredLogger:
    """Get or create global logger instance.

    Parameters
    ----------
    name : str
        Logger name
    log_level : str
        Logging level
    log_file : str, optional
        Log file path

    Returns
    -------
    StructuredLogger
        Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger(name, log_level, log_file)
    return _logger_instance

