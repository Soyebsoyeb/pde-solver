"""Performance metrics and monitoring."""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading

from pde_solver.utils.logger import get_logger

logger = get_logger()


@dataclass
class Metric:
    """Single metric value."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and tracks performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[Metric] = []
        self._lock = threading.Lock()

    def record(self, name: str, value: float, **metadata):
        """Record a metric.

        Parameters
        ----------
        name : str
            Metric name
        value : float
            Metric value
        **metadata
            Additional metadata
        """
        with self._lock:
            metric = Metric(name=name, value=value, metadata=metadata)
            self.metrics.append(metric)
            logger.log_metric(name, value, **metadata)

    @contextmanager
    def timer(self, name: str, **metadata):
        """Context manager for timing operations.

        Parameters
        ----------
        name : str
            Operation name
        **metadata
            Additional metadata

        Yields
        ------
        None
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record(f"{name}_duration", duration, **metadata)
            logger.log_timing(name, duration, **metadata)

    def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        """Get metrics, optionally filtered by name.

        Parameters
        ----------
        name : str, optional
            Filter by metric name

        Returns
        -------
        List[Metric]
            List of metrics
        """
        with self._lock:
            if name:
                return [m for m in self.metrics if m.name == name]
            return self.metrics.copy()

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric.

        Parameters
        ----------
        name : str
            Metric name

        Returns
        -------
        Dict[str, float]
            Statistics (mean, min, max, count, std)
        """
        metrics = self.get_metrics(name)
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        mean = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5

        return {
            "mean": mean,
            "min": min_val,
            "max": max_val,
            "count": len(values),
            "std": std,
        }

    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()

    def export(self) -> List[Dict[str, Any]]:
        """Export metrics as dictionaries.

        Returns
        -------
        List[Dict[str, Any]]
            List of metric dictionaries
        """
        with self._lock:
            return [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    **m.metadata,
                }
                for m in self.metrics
            ]


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector.

    Returns
    -------
    MetricsCollector
        Metrics collector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

