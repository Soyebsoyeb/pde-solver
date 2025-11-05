# Quick Start Guide - PDE Solver

## Installation

```bash
# Clone the repository
cd pde_solver

# Install dependencies
pip install -r requirements.txt
```

## Running Without PyTorch

If you have issues with PyTorch on Windows, you can still run the classical solver:

```bash
# Run classical numerical solver (no PyTorch needed)
python -m pde_solver.cli classical configs/burgers_small.yaml
```

Output will be saved to `outputs/burgers_classical_solution.png`

## Running With PyTorch

### Train a Model
```bash
# Train on CPU (recommended for testing)
python -m pde_solver.cli train configs/burgers_small.yaml --device cpu

# Train on GPU (if available)
python -m pde_solver.cli train configs/burgers_small.yaml --device cuda
```

### Evaluate a Model
```bash
# After training, evaluate the checkpoint
python -m pde_solver.cli eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt --device cpu
```

## Configuration Files

- `configs/burgers_small.yaml` - Fast training (500 epochs, small network)
- `configs/burgers_medium.yaml` - Better accuracy (more epochs, larger network)

## Test the Installation

```bash
# Run basic functionality tests
python test_basic_functionality.py
```

This will verify:
- Package imports correctly
- Classical solver works
- Visualization works
- CLI commands are available

## Common Issues

### PyTorch DLL Error on Windows
**Error:** `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**Solution:** This has been fixed! The classical solver now runs without loading PyTorch. If you still see this error when running train/eval commands, you may need to:
1. Reinstall PyTorch: `pip uninstall torch && pip install torch`
2. Install Visual C++ Redistributable from Microsoft
3. Use CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### No Output Directory
The CLI automatically creates the `outputs/` directory when needed.

## Example Workflow

1. **Test installation:**
   ```bash
   python test_basic_functionality.py
   ```

2. **Run classical solver:**
   ```bash
   python -m pde_solver.cli classical configs/burgers_small.yaml
   ```

3. **Train a model:**
   ```bash
   python -m pde_solver.cli train configs/burgers_small.yaml --device cpu
   ```

4. **Evaluate the model:**
   ```bash
   python -m pde_solver.cli eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt
   ```

## Getting Help

```bash
# General help
python -m pde_solver.cli --help

# Help for specific command
python -m pde_solver.cli train --help
python -m pde_solver.cli eval --help
python -m pde_solver.cli classical --help
```

## Project Structure

```
pde_solver/
├── pde_solver/           # Main package
│   ├── cli.py           # Command-line interface
│   ├── models/          # Neural and classical models
│   ├── training/        # Training utilities
│   └── utils/           # Visualization, metrics, etc.
├── configs/             # Configuration files
├── outputs/             # Generated plots and results
├── checkpoints/         # Saved model checkpoints
└── tests/              # Unit tests
```

## Next Steps

- Modify `configs/burgers_small.yaml` to experiment with different parameters
- Check `outputs/` for generated visualizations
- Explore the code in `pde_solver/` to understand the implementation
- Read `CODE_RUNNABLE_SUMMARY.md` for technical details on the architecture

## Support

For issues or questions, refer to:
- `CODE_RUNNABLE_SUMMARY.md` - Technical details
- `README.md` - Full documentation
- GitHub issues (if repository is hosted)
