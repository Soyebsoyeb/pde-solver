"""Caching utilities for performance optimization."""

import hashlib
import pickle
from typing import Any, Optional, Callable
from pathlib import Path
from functools import wraps
import json

from pde_solver.utils.logger import get_logger

logger = get_logger()


class FileCache:
    """File-based cache for expensive computations."""

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize file cache.

        Parameters
        ----------
        cache_dir : str
            Cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.

        Parameters
        ----------
        *args
            Positional arguments
        **kwargs
            Keyword arguments

        Returns
        -------
        str
            Cache key (hash)
        """
        # Create hashable representation
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any, optional
            Cached value or None
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache {key}: {e}")
                return None
        return None

    def set(self, key: str, value: Any):
        """Set value in cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Error saving cache {key}: {e}")

    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries.

        Parameters
        ----------
        pattern : str, optional
            Pattern to match (if None, clear all)
        """
        if pattern:
            for cache_file in self.cache_dir.glob(f"{pattern}*.pkl"):
                cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


def cached(cache: Optional[FileCache] = None, use_cache: bool = True):
    """Decorator for caching function results.

    Parameters
    ----------
    cache : FileCache, optional
        Cache instance (creates default if None)
    use_cache : bool
        Whether to use cache

    Returns
    -------
    Callable
        Decorated function
    """
    if cache is None:
        cache = FileCache()

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)

            key = cache._get_key(func.__name__, *args, **kwargs)
            cached_value = cache.get(key)

            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            logger.debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper
    return decorator


# Global cache instance
_global_cache: Optional[FileCache] = None


def get_cache() -> FileCache:
    """Get or create global cache instance.

    Returns
    -------
    FileCache
        Cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = FileCache()
    return _global_cache

