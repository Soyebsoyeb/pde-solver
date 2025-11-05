# 🚀 START HERE - Get Running in 3 Steps

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for neural networks)
- NumPy, SciPy (for numerical computing)
- SymPy (for symbolic math)
- Matplotlib (for visualization)
- PyYAML, Typer (for CLI)
- And other dependencies

## Step 2: Install the Package

```bash
pip install -e .
```

This installs the `pde_solver` package in "editable" mode so you can modify code and see changes immediately.

## Step 3: Run the Demo

```bash
python examples/burgers_demo.py
```

This will:
1. Train a Physics-Informed Neural Network on Burgers' equation
2. Compare with classical numerical solver
3. Generate visualization plots in `outputs/` directory

**Expected time:** 5-10 minutes on CPU, 1-2 minutes on GPU

## Verify Installation

Check if everything is set up correctly:

```bash
# Check dependencies
python check_dependencies.py

# Test code structure
python test_runnable.py
```

## Alternative: Quick Test

If you want to test without full training:

```bash
python test_runnable.py
```

This runs quick tests without training a full model.

## Troubleshooting

### "ModuleNotFoundError"
- Make sure you ran `pip install -r requirements.txt`
- Make sure you ran `pip install -e .`

### Windows Issues
- Use PowerShell or Command Prompt (not Git Bash for pip commands)
- If you get encoding errors, the scripts now use ASCII characters

### CUDA/GPU
- Code automatically uses CPU if CUDA not available
- GPU is optional but faster for training

## Next Steps

Once it's running:
1. Check `outputs/` directory for generated plots
2. Edit `configs/burgers_small.yaml` to customize training
3. Read `README.md` for full documentation
4. Try the CLI: `python run_solver.py train configs/burgers_small.yaml`

## Need Help?

- Check `RUN.md` for detailed instructions
- See `QUICKSTART.md` for more examples
- Read `README.md` for full documentation

