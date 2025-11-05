# PyTorch Installation Issues on Windows

## Current Status

✅ **Classical Solver** - Works perfectly (no PyTorch needed)  
❌ **Neural Network Training** - Requires fixing PyTorch DLL dependencies

## The Problem

Your Windows system is missing the Visual C++ Runtime libraries that PyTorch requires. This is a common issue on Windows.

Error: `[WinError 1114] A dynamic link library (DLL) initialization routine failed`

## Solutions

### Solution 1: Install Visual C++ Redistributable (RECOMMENDED)

1. Download the latest version:
   - [Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   
2. Run the installer as Administrator

3. Restart your computer

4. Test PyTorch:
   ```bash
   python -c "import torch; print('Success!')"
   ```

### Solution 2: Use WSL (Windows Subsystem for Linux)

PyTorch works more reliably in Linux environments:

```bash
# Install WSL
wsl --install

# After restart, open WSL terminal
cd /mnt/c/Users/hoque/OneDrive/Desktop/CS_Fundamental/great_ideas/pde_solver

# Install dependencies in WSL
pip install -r requirements.txt

# Run training in WSL
python -m pde_solver.cli train configs/burgers_small.yaml --device cpu
```

### Solution 3: Use Conda Instead of pip

Conda handles C++ dependencies better on Windows:

```bash
# Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Create new environment
conda create -n pde_solver python=3.11
conda activate pde_solver

# Install PyTorch via conda
conda install pytorch cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Solution 4: Use Only Classical Solvers

The classical solver works perfectly without PyTorch:

```bash
# This works right now!
python -m pde_solver.cli classical configs/burgers_small.yaml

# You can modify configs/burgers_small.yaml to adjust:
# - nx: Number of spatial points
# - nt: Number of time steps
# - nu: Viscosity coefficient
```

## What Works Without PyTorch

✅ Classical numerical solvers  
✅ Visualization utilities  
✅ Configuration management  
✅ Result analysis (with NumPy arrays)  
✅ CLI help and documentation  

## What Requires PyTorch

❌ Neural network training (PINN)  
❌ DeepONet operator learning  
❌ Model evaluation with trained networks  

## Next Steps

1. **Quick fix:** Use the classical solver (works now)
2. **Best fix:** Install Visual C++ Redistributable (5 minutes)
3. **Alternative:** Use WSL or Conda (30 minutes)

## Testing After Fix

Once you fix PyTorch, test with:

```bash
# Test PyTorch import
python -c "import torch; print('PyTorch works!')"

# Test training (quick run)
python -m pde_solver.cli train configs/burgers_small.yaml --device cpu
```

## Contact

If issues persist after trying these solutions, the problem may be:
- Incompatible Windows version
- Corrupted Python installation
- Antivirus blocking DLL loading
- Missing Windows updates

Consider posting on PyTorch forums: https://discuss.pytorch.org/
