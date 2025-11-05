"""Configuration validation and management."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator, ValidationError as PydanticValidationError

from pde_solver.utils.exceptions import ConfigurationError
from pde_solver.utils.logger import get_logger

logger = get_logger()


class PDEType(str):
    """PDE type enumeration."""
    BURGERS = "burgers"
    NAVIER_STOKES = "navier_stokes"
    SCHRODINGER = "schrodinger"
    EINSTEIN = "einstein"


class SolverConfig(BaseModel):
    """Validated solver configuration."""

    pde_type: str = Field(..., description="Type of PDE to solve")
    nu: float = Field(default=0.01, ge=0.0, description="Viscosity coefficient")
    
    # Domain configuration
    x_domain: List[float] = Field(default=[-1.0, 1.0], min_items=2, max_items=2)
    t_domain: List[float] = Field(default=[0.0, 1.0], min_items=2, max_items=2)
    
    # Training points
    n_points: int = Field(default=10000, gt=0)
    n_ic: int = Field(default=1000, gt=0)
    n_bc: int = Field(default=500, gt=0)
    
    # Model architecture
    hidden_dims: List[int] = Field(default=[256, 256, 256, 256], min_items=1)
    
    # Training
    num_epochs: int = Field(default=1000, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=1024, gt=0)
    weight_residual: float = Field(default=1.0, ge=0.0)
    weight_boundary: float = Field(default=1.0, ge=0.0)
    weight_initial: float = Field(default=1.0, ge=0.0)
    
    # Classical solver
    nx: int = Field(default=256, gt=0)
    nt: int = Field(default=100, gt=0)
    
    # Logging
    use_wandb: bool = Field(default=False)
    checkpoint_freq: int = Field(default=100, gt=0)
    
    # Device
    device: str = Field(default="auto", description="Device: auto, cpu, or cuda")
    
    # Advanced
    deterministic: bool = Field(default=True)
    use_mixed_precision: bool = Field(default=False)
    gradient_clip: float = Field(default=1.0, ge=0.0)

    @validator("pde_type")
    def validate_pde_type(cls, v):
        """Validate PDE type."""
        valid_types = ["burgers", "navier_stokes", "schrodinger", "einstein"]
        if v not in valid_types:
            raise ValueError(f"pde_type must be one of {valid_types}")
        return v

    @validator("x_domain")
    def validate_x_domain(cls, v):
        """Validate x domain."""
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("x_domain must be [min, max] with min < max")
        return v

    @validator("t_domain")
    def validate_t_domain(cls, v):
        """Validate t domain."""
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("t_domain must be [min, max] with min < max")
        return v

    @validator("device")
    def validate_device(cls, v):
        """Validate device."""
        valid_devices = ["auto", "cpu", "cuda"]
        if v not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")
        return v

    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow extra fields
        validate_assignment = True


def load_config(config_path: str, validate: bool = True) -> Dict[str, Any]:
    """Load and validate configuration file.

    Parameters
    ----------
    config_path : str
        Path to configuration file
    validate : bool
        Whether to validate configuration

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary

    Raises
    ------
    ConfigurationError
        If configuration is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error reading config file: {e}") from e

    if config_dict is None:
        raise ConfigurationError("Configuration file is empty")

    if validate:
        try:
            validated_config = SolverConfig(**config_dict)
            logger.info(f"Configuration loaded and validated from {config_path}")
            return validated_config.dict()
        except PydanticValidationError as e:
            errors = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
            raise ConfigurationError(
                f"Configuration validation failed: {errors}",
                config_key=str(e.errors()[0].get("loc", "unknown"))
            ) from e

    return config_dict


def validate_config(config: Dict[str, Any]) -> SolverConfig:
    """Validate configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    SolverConfig
        Validated configuration

    Raises
    ------
    ConfigurationError
        If configuration is invalid
    """
    try:
        return SolverConfig(**config)
    except PydanticValidationError as e:
        errors = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
        raise ConfigurationError(
            f"Configuration validation failed: {errors}",
            config_key=str(e.errors()[0].get("loc", "unknown"))
        ) from e

