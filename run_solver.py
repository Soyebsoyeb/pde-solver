#!/usr/bin/env python
"""Main entry point for PDE solver CLI."""

import sys

try:
    from pde_solver.cli import app
except ImportError as e:
    print(f"[ERROR] Could not import CLI: {e}")
    print("\nThis usually means dependencies are not installed.")
    print("Install with: pip install -r requirements.txt")
    print("\nAlternatively, run the minimal demo:")
    print("  python minimal_demo.py")
    sys.exit(1)

if __name__ == "__main__":
    app()

