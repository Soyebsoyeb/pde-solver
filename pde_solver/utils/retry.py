"""Retry utilities for robust operations."""

import time
from typing import Callable, Type, Tuple, Optional
from functools import wraps

from pde_solver.utils.logger import get_logger
from pde_solver.utils.exceptions import PDESolverError

logger = get_logger()


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_failure: Optional[Callable] = None,
):
    """Decorator for retrying operations.

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts
    delay : float
        Initial delay between retries (seconds)
    backoff : float
        Multiplier for delay after each retry
    exceptions : Tuple[Type[Exception], ...]
        Exceptions to catch and retry
    on_failure : Callable, optional
        Function to call on final failure

    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.2f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
                        if on_failure:
                            on_failure(e)
                        raise

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations."""

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """Initialize retryable operation.

        Parameters
        ----------
        max_attempts : int
            Maximum attempts
        delay : float
            Initial delay
        backoff : float
            Backoff multiplier
        exceptions : Tuple[Type[Exception], ...]
            Exceptions to catch
        """
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions

    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic.

        Parameters
        ----------
        func : Callable
            Function to execute
        *args
            Function arguments
        **kwargs
            Function keyword arguments

        Returns
        -------
        Any
            Function result

        Raises
        ------
        Exception
            Last exception if all attempts fail
        """
        current_delay = self.delay
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                if attempt < self.max_attempts:
                    logger.warning(
                        f"Attempt {attempt}/{self.max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= self.backoff
                else:
                    logger.error(f"All {self.max_attempts} attempts failed: {e}")
                    raise

        if last_exception:
            raise last_exception

