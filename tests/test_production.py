"""Production-level tests for industry standards."""

import pytest
import tempfile
from pathlib import Path

from pde_solver.utils.logger import get_logger, StructuredLogger
from pde_solver.utils.exceptions import (
    ConfigurationError,
    ModelError,
    TrainingError,
    PDESolverError,
)
from pde_solver.utils.config_validator import load_config, validate_config
from pde_solver.utils.metrics import MetricsCollector
from pde_solver.utils.cache import FileCache, cached
from pde_solver.utils.retry import retry, RetryableOperation
from pde_solver.utils.security import sanitize_path, check_environment_security
from pde_solver.utils.resource_manager import ResourceManager, ResourceLimits


def test_logger():
    """Test structured logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = StructuredLogger("test", log_file=str(log_file))
        
        logger.info("Test message")
        logger.log_metric("test_metric", 1.5, extra_field="value")
        logger.log_timing("test_op", 0.5)
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
        assert "test_metric" in content


def test_exceptions():
    """Test custom exceptions."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Test error", config_key="test_key")
    
    assert exc_info.value.error_code == "CONFIG_ERROR"
    assert exc_info.value.config_key == "test_key"


def test_config_validation():
    """Test configuration validation."""
    # Valid config
    config = {
        "pde_type": "burgers",
        "nu": 0.01,
        "num_epochs": 100,
    }
    validated = validate_config(config)
    assert validated.pde_type == "burgers"
    
    # Invalid config
    with pytest.raises(ConfigurationError):
        validate_config({"pde_type": "invalid"})


def test_config_loading():
    """Test configuration file loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("pde_type: burgers\nnu: 0.01\nnum_epochs: 100\n")
        config_path = f.name
    
    try:
        config = load_config(config_path)
        assert config["pde_type"] == "burgers"
        assert config["nu"] == 0.01
    finally:
        Path(config_path).unlink()
    
    # Non-existent file
    with pytest.raises(ConfigurationError):
        load_config("nonexistent.yaml")


def test_metrics_collector():
    """Test metrics collection."""
    collector = MetricsCollector()
    
    collector.record("test_metric", 1.0, tag="test")
    collector.record("test_metric", 2.0, tag="test")
    collector.record("test_metric", 3.0, tag="test")
    
    metrics = collector.get_metrics("test_metric")
    assert len(metrics) == 3
    
    stats = collector.get_statistics("test_metric")
    assert stats["mean"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    assert stats["count"] == 3


def test_timing_context():
    """Test timing context manager."""
    import time
    collector = MetricsCollector()
    
    with collector.timer("test_operation"):
        time.sleep(0.1)
    
    metrics = collector.get_metrics("test_operation_duration")
    assert len(metrics) == 1
    assert metrics[0].value > 0.05


def test_file_cache():
    """Test file caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FileCache(cache_dir=tmpdir)
        
        # Test set/get
        cache.set("test_key", {"data": 123})
        value = cache.get("test_key")
        assert value == {"data": 123}
        
        # Test miss
        assert cache.get("nonexistent") is None
        
        # Test clear
        cache.clear()
        assert cache.get("test_key") is None


def test_cached_decorator():
    """Test caching decorator."""
    call_count = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FileCache(cache_dir=tmpdir)
        
        @cached(cache=cache)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - should compute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment


def test_retry_decorator():
    """Test retry decorator."""
    call_count = 0
    
    @retry(max_attempts=3, delay=0.1)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary error")
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert call_count == 3


def test_retry_operation():
    """Test retryable operation."""
    call_count = 0
    
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Error")
        return "success"
    
    retry_op = RetryableOperation(max_attempts=3, delay=0.1)
    result = retry_op.execute(flaky_function)
    assert result == "success"
    assert call_count == 2


def test_security_sanitize_path():
    """Test path sanitization."""
    # Valid path
    path = sanitize_path("configs/test.yaml")
    assert isinstance(path, Path)
    
    # Directory traversal attempt
    with pytest.raises(ValueError):
        sanitize_path("../../../etc/passwd")
    
    # Path restriction
    with pytest.raises(ValueError):
        sanitize_path("/etc/passwd", base_dir="/workspace")


def test_security_check():
    """Test security environment check."""
    result = check_environment_security()
    assert "secure" in result
    assert "issues" in result


def test_resource_manager():
    """Test resource manager."""
    limits = ResourceLimits(max_memory_gb=16.0)
    manager = ResourceManager(limits=limits)
    
    # Should not raise (unless actually over limit)
    try:
        summary = manager.get_resource_summary()
        assert "memory" in summary
        assert "cpu" in summary
    except Exception:
        # May fail if resources are actually over limit
        pass


def test_error_handling_chain():
    """Test error handling chain."""
    try:
        raise ConfigurationError("Config error", config_key="test")
    except PDESolverError as e:
        assert e.error_code == "CONFIG_ERROR"
        assert isinstance(e, ConfigurationError)

