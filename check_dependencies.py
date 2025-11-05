"""Check if all dependencies are installed."""

import sys

missing = []

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    missing.append("torch")
    print("[MISSING] PyTorch not installed")

try:
    import numpy
    print(f"[OK] NumPy {numpy.__version__}")
except ImportError:
    missing.append("numpy")
    print("[MISSING] NumPy not installed")

try:
    import scipy
    print(f"[OK] SciPy {scipy.__version__}")
except ImportError:
    missing.append("scipy")
    print("[MISSING] SciPy not installed")

try:
    import sympy
    print(f"[OK] SymPy {sympy.__version__}")
except ImportError:
    missing.append("sympy")
    print("[MISSING] SymPy not installed")

try:
    import matplotlib
    print(f"[OK] Matplotlib {matplotlib.__version__}")
except ImportError:
    missing.append("matplotlib")
    print("[MISSING] Matplotlib not installed")

try:
    import yaml
    print("[OK] PyYAML")
except ImportError:
    missing.append("pyyaml")
    print("[MISSING] PyYAML not installed")

try:
    import typer
    print("[OK] Typer")
except ImportError:
    missing.append("typer")
    print("[MISSING] Typer not installed")

try:
    import wandb
    print("[OK] Weights & Biases (optional)")
except ImportError:
    print("[OPTIONAL] Weights & Biases not installed (optional)")

if missing:
    print(f"\n[ERROR] Missing required dependencies: {', '.join(missing)}")
    print("\nInstall with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n[SUCCESS] All required dependencies installed!")
    sys.exit(0)

