# Contributing to PDE Solver

Thank you for your interest in contributing to PDE Solver! This document provides guidelines for contributing.

## Development Setup

1. Clone the repository
2. Create a conda environment: `make env`
3. Install in editable mode: `make install`
4. Run tests: `make test`

## Code Style

- Use type hints for all functions
- Follow NumPy-style docstrings
- Format code with `make format`
- Run linters with `make lint`

## Testing

- Write unit tests for new features
- Ensure all tests pass: `make test`
- Add regression tests for numerical accuracy
- Include integration tests for end-to-end workflows

## Pull Request Process

1. Create a feature branch
2. Make your changes with tests
3. Ensure all tests pass and linting is clean
4. Update documentation as needed
5. Submit a pull request with a clear description

## Adding New PDE Solvers

When adding a new PDE solver:

1. Create a model file in `pde_solver/models/`
2. Implement both classical and neural methods
3. Add symbolic equation in `pde_solver/symbolic/`
4. Create example config in `configs/`
5. Add example notebook in `notebooks/`
6. Write tests in `tests/`
7. Update documentation

## Reporting Issues

Please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)


