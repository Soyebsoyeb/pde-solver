#!/usr/bin/env python
"""Run PDE solver demo - works with whatever dependencies are available."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point - runs appropriate demo."""
    print("PDE Solver - Universal Runner")
    print("=" * 60)
    print("This script will run the best available demo based on installed dependencies.")
    print("=" * 60)
    print()
    
    # Try to run minimal demo first (works with any dependencies)
    try:
        from minimal_demo import check_and_run
        check_and_run()
        return
    except Exception as e:
        print(f"[WARNING] Could not run minimal_demo: {e}")
        print("Trying alternative approach...\n")
    
    # Fallback: run structure validation
    print("Running structure validation...")
    print("-" * 60)
    
    # Check if we can at least validate structure
    required_files = [
        "pde_solver/__init__.py",
        "pde_solver/core/neural_symbolic_solver.py",
        "pde_solver/models/burgers.py",
        "run_solver.py",
        "configs/burgers_small.yaml",
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            all_ok = False
    
    print()
    if all_ok:
        print("[SUCCESS] Project structure is valid!")
        print("\nTo run full functionality:")
        print("  1. pip install -r requirements.txt")
        print("  2. pip install -e .")
        print("  3. python examples/burgers_demo.py")
    else:
        print("[WARNING] Some files are missing")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

