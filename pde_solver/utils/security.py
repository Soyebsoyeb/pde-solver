"""Security utilities for production deployment."""

import os
import secrets
from typing import Optional
from pathlib import Path

from pde_solver.utils.logger import get_logger

logger = get_logger()


def sanitize_path(path: str, base_dir: Optional[str] = None) -> Path:
    """Sanitize file path to prevent directory traversal attacks.

    Parameters
    ----------
    path : str
        Input path
    base_dir : str, optional
        Base directory to restrict paths to

    Returns
    -------
    Path
        Sanitized path

    Raises
    ------
    ValueError
        If path is invalid or attempts directory traversal
    """
    # Normalize path
    normalized = Path(path).resolve()

    # Check for directory traversal
    if ".." in str(normalized):
        raise ValueError(f"Invalid path: contains directory traversal: {path}")

    # Restrict to base directory if provided
    if base_dir:
        base = Path(base_dir).resolve()
        try:
            normalized.relative_to(base)
        except ValueError:
            raise ValueError(f"Path outside base directory: {path}")

    return normalized


def generate_secret_key(length: int = 32) -> str:
    """Generate a secure random secret key.

    Parameters
    ----------
    length : int
        Key length in bytes

    Returns
    -------
    str
        Secret key (hex encoded)
    """
    return secrets.token_hex(length)


def validate_file_permissions(file_path: Path, max_permissions: int = 0o644):
    """Validate file permissions are secure.

    Parameters
    ----------
    file_path : Path
        File path to check
    max_permissions : int
        Maximum allowed permissions (octal)

    Returns
    -------
    bool
        True if permissions are secure
    """
    if not file_path.exists():
        return True

    stat = file_path.stat()
    current_perms = stat.st_mode & 0o777

    # Check if permissions are too permissive
    if current_perms > max_permissions:
        logger.warning(
            f"File {file_path} has insecure permissions: {oct(current_perms)} "
            f"(max allowed: {oct(max_permissions)})"
        )
        return False

    return True


def secure_temp_file(prefix: str = "pde_solver_", suffix: str = ".tmp") -> Path:
    """Create a secure temporary file.

    Parameters
    ----------
    prefix : str
        File prefix
    suffix : str
        File suffix

    Returns
    -------
    Path
        Temporary file path
    """
    import tempfile

    # Create secure temporary file
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)

    # Set secure permissions
    os.chmod(path, 0o600)

    return Path(path)


def check_environment_security():
    """Check environment for security issues.

    Returns
    -------
    dict
        Security check results
    """
    issues = []

    # Check for debug mode in production
    if os.getenv("DEBUG", "false").lower() == "true":
        issues.append("DEBUG mode enabled in production")

    # Check for default credentials
    if os.getenv("WANDB_API_KEY") == "default" or os.getenv("WANDB_API_KEY") == "":
        issues.append("WandB API key not set or using default")

    # Check file permissions
    important_files = [
        Path("configs") / "burgers_small.yaml",
        Path(".env") if Path(".env").exists() else None,
    ]
    for file_path in important_files:
        if file_path and file_path.exists():
            if not validate_file_permissions(file_path):
                issues.append(f"Insecure permissions on {file_path}")

    return {
        "secure": len(issues) == 0,
        "issues": issues,
    }

