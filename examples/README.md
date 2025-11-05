# Examples

This directory contains example scripts demonstrating how to use the PDE solver.

## Burgers Equation

### Quick Start

```bash
python examples/burgers_demo.py
```

This will:
1. Train a PINN model on Burgers' equation
2. Compare with classical numerical solver
3. Generate visualization plots in `outputs/` directory

### Using CLI

```bash
# Train
python run_solver.py train configs/burgers_small.yaml

# Evaluate
python run_solver.py eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt

# Classical solver
python run_solver.py classical configs/burgers_small.yaml
```

## Configuration

Edit `configs/burgers_small.yaml` or `configs/burgers_medium.yaml` to customize:
- Domain size and discretization
- Model architecture (hidden layers, width)
- Training parameters (epochs, learning rate, batch size)
- Loss weights

