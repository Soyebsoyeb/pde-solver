# Quick Start Guide

## Installation

### Option 1: Using pip (recommended for quick start)

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Option 2: Using conda

```bash
# Create environment
conda env create -f environment.yml
conda activate pde_solver

# Install package
pip install -e .
```

### Option 3: Using Docker

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t pde-solver:latest .

# Run example
docker run --rm -v $(pwd)/outputs:/workspace/outputs pde-solver:latest python examples/burgers_demo.py
```

## Verify Installation

```bash
# Test imports
python scripts/validate_structure.py

# Run tests
pytest tests/ -v
```

## Run Your First Example

### Option 1: Demo Script (Recommended)

```bash
python examples/burgers_demo.py
```

This will:
1. Train a PINN on Burgers' equation (~5-10 minutes on CPU)
2. Compare with classical solver
3. Generate plots in `outputs/` directory

### Option 2: Using CLI

```bash
# Train
python run_solver.py train configs/burgers_small.yaml

# Evaluate (after training)
python run_solver.py eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt

# Classical solver
python run_solver.py classical configs/burgers_small.yaml
```

## Expected Output

After running the demo, you should see:
- `outputs/burgers_training_loss.png` - Training loss curves
- `outputs/burgers_pinn_solution.png` - PINN solution visualization
- `outputs/burgers_classical_solution.png` - Classical solver solution
- `checkpoints/checkpoint_epoch_*.pt` - Model checkpoints

## Troubleshooting

### Import Errors
If you get import errors, make sure you've installed the package:
```bash
pip install -e .
```

### CUDA/GPU Issues
If you want to use CPU only, the code will automatically detect and use CPU. To force CPU:
```python
device = "cpu"
```

### Memory Issues
For large problems, reduce batch size in config files:
```yaml
batch_size: 256  # Reduce from default 1024
```

## Next Steps

1. Explore `configs/burgers_small.yaml` to customize training
2. Check `examples/burgers_demo.py` for code examples
3. Read `README.md` for full documentation
4. See `CONTRIBUTING.md` to contribute

