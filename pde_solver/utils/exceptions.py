"""Custom exceptions for PDE solver."""

from typing import Optional


class PDESolverError(Exception):
    """Base exception for PDE solver errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        """Initialize exception.

        Parameters
        ----------
        message : str
            Error message
        error_code : str, optional
            Error code for programmatic handling
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ConfigurationError(PDESolverError):
    """Configuration-related errors."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        """Initialize configuration error.

        Parameters
        ----------
        message : str
            Error message
        config_key : str, optional
            Configuration key that caused the error
        """
        super().__init__(message, error_code="CONFIG_ERROR")
        self.config_key = config_key


class ModelError(PDESolverError):
    """Model-related errors."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        """Initialize model error.

        Parameters
        ----------
        message : str
            Error message
        model_name : str, optional
            Model name that caused the error
        """
        super().__init__(message, error_code="MODEL_ERROR")
        self.model_name = model_name


class TrainingError(PDESolverError):
    """Training-related errors."""

    def __init__(self, message: str, epoch: Optional[int] = None):
        """Initialize training error.

        Parameters
        ----------
        message : str
            Error message
        epoch : int, optional
            Epoch number when error occurred
        """
        super().__init__(message, error_code="TRAINING_ERROR")
        self.epoch = epoch


class NumericalError(PDESolverError):
    """Numerical computation errors."""

    def __init__(self, message: str, operation: Optional[str] = None):
        """Initialize numerical error.

        Parameters
        ----------
        message : str
            Error message
        operation : str, optional
            Operation that caused the error
        """
        super().__init__(message, error_code="NUMERICAL_ERROR")
        self.operation = operation


class ValidationError(PDESolverError):
    """Validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        """Initialize validation error.

        Parameters
        ----------
        message : str
            Error message
        field : str, optional
            Field that failed validation
        """
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field


class ConvergenceError(PDESolverError):
    """Convergence errors."""

    def __init__(self, message: str, iteration: Optional[int] = None):
        """Initialize convergence error.

        Parameters
        ----------
        message : str
            Error message
        iteration : int, optional
            Iteration number when error occurred
        """
        super().__init__(message, error_code="CONVERGENCE_ERROR")
        self.iteration = iteration

