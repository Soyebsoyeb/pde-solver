"""Resource management for production deployments."""

import psutil
import torch
from typing import Dict, Optional
from dataclasses import dataclass

from pde_solver.utils.logger import get_logger
from pde_solver.utils.exceptions import PDESolverError

logger = get_logger()


@dataclass
class ResourceLimits:
    """Resource limits configuration."""

    max_memory_gb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    max_gpu_memory_gb: Optional[float] = None


class ResourceManager:
    """Manages system resources and enforces limits."""

    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize resource manager.

        Parameters
        ----------
        limits : ResourceLimits, optional
            Resource limits
        """
        self.limits = limits or ResourceLimits()
        self.process = psutil.Process()

    def check_memory(self) -> Dict[str, float]:
        """Check current memory usage.

        Returns
        -------
        Dict[str, float]
            Memory usage statistics (GB)
        """
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()

        usage = {
            "process_memory_gb": mem_info.rss / (1024 ** 3),
            "system_total_gb": system_mem.total / (1024 ** 3),
            "system_available_gb": system_mem.available / (1024 ** 3),
            "system_used_percent": system_mem.percent,
        }

        # Check limits
        if self.limits.max_memory_gb:
            if usage["process_memory_gb"] > self.limits.max_memory_gb:
                raise PDESolverError(
                    f"Memory limit exceeded: {usage['process_memory_gb']:.2f}GB > "
                    f"{self.limits.max_memory_gb}GB"
                )

        return usage

    def check_cpu(self) -> Dict[str, float]:
        """Check CPU usage.

        Returns
        -------
        Dict[str, float]
            CPU usage statistics
        """
        cpu_percent = self.process.cpu_percent(interval=0.1)
        system_cpu = psutil.cpu_percent(interval=0.1)

        usage = {
            "process_cpu_percent": cpu_percent,
            "system_cpu_percent": system_cpu,
            "cpu_count": psutil.cpu_count(),
        }

        # Check limits
        if self.limits.max_cpu_percent:
            if cpu_percent > self.limits.max_cpu_percent:
                logger.warning(
                    f"CPU usage high: {cpu_percent:.1f}% > {self.limits.max_cpu_percent}%"
                )

        return usage

    def check_gpu(self) -> Optional[Dict[str, float]]:
        """Check GPU usage (if available).

        Returns
        -------
        Dict[str, float], optional
            GPU usage statistics
        """
        if not torch.cuda.is_available():
            return None

        usage = {
            "gpu_count": torch.cuda.device_count(),
            "gpus": [],
        }

        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            gpu_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)

            gpu_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": gpu_memory,
                "allocated_memory_gb": gpu_allocated,
                "reserved_memory_gb": gpu_reserved,
                "free_memory_gb": gpu_memory - gpu_reserved,
            }

            usage["gpus"].append(gpu_info)

            # Check limits
            if self.limits.max_gpu_memory_gb:
                if gpu_reserved > self.limits.max_gpu_memory_gb:
                    raise PDESolverError(
                        f"GPU {i} memory limit exceeded: {gpu_reserved:.2f}GB > "
                        f"{self.limits.max_gpu_memory_gb}GB"
                    )

        return usage

    def get_resource_summary(self) -> Dict[str, any]:
        """Get comprehensive resource summary.

        Returns
        -------
        Dict[str, any]
            Resource summary
        """
        summary = {
            "memory": self.check_memory(),
            "cpu": self.check_cpu(),
            "gpu": self.check_gpu(),
        }

        logger.info(f"Resource usage: {summary}")
        return summary

    def clear_gpu_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

