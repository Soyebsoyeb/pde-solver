# How to Run the PDE Solver

## Quick Start (Automated)

Run the installation and test script:

```bash
python install_and_test.py
```

This will:
1. Check Python version
2. Install all dependencies
3. Install the package
4. Run tests
5. Optionally run a quick demo

## Manual Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Package

```bash
pip install -e .
```

### Step 3: Verify Installation

```bash
# Check dependencies
python check_dependencies.py

# Test code structure
python test_runnable.py
```

## Running Examples

### Option 1: Quick Demo Script

```bash
python examples/burgers_demo.py
```

This will:
- Train a PINN on Burgers' equation
- Compare with classical solver
- Generate plots in `outputs/` directory

### Option 2: Using CLI

```bash
# Train a model
python run_solver.py train configs/burgers_small.yaml

# Evaluate a trained model
python run_solver.py eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt

# Run classical solver
python run_solver.py classical configs/burgers_small.yaml
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "No module named 'pde_solver'"

Install the package:
```bash
pip install -e .
```

### Import Errors

Make sure you're in the project root directory and the package is installed:
```bash
cd /path/to/pde_solver
pip install -e .
```

### CUDA/GPU Issues

The code automatically detects and uses CPU if CUDA is not available. To force CPU:
```python
device = "cpu"
```

### Memory Issues

Reduce batch size in config files:
```yaml
batch_size: 256  # Reduce from 1024
```

## Expected Output

After running the demo, you should see:
- `outputs/burgers_training_loss.png` - Training curves
- `outputs/burgers_pinn_solution.png` - PINN solution
- `outputs/burgers_classical_solution.png` - Classical solution
- `checkpoints/checkpoint_epoch_*.pt` - Model checkpoints

## Next Steps

1. Edit `configs/burgers_small.yaml` to customize training
2. Explore `examples/burgers_demo.py` for code examples
3. Read `README.md` for full documentation
4. Check `QUICKSTART.md` for more details

