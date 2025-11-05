#!/usr/bin/env python
"""
Industry-Grade Startup Script for PDE Solver
Provides comprehensive system checks, diagnostics, and graceful error handling.
"""

import sys
import os
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Color codes for terminal output (Windows-safe)
def _setup_colors():
    """Setup colors, handling Windows terminal."""
    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass
    
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
    
    return Colors

Colors = _setup_colors()

def print_header():
    """Print startup header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"{'PDE Solver - Industry-Grade Startup System'}")
    print(f"{'='*70}{Colors.RESET}\n")
    print(f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print()

def check_system_requirements() -> Dict[str, Tuple[bool, str]]:
    """Check system requirements."""
    results = {}
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        results["Python Version"] = (True, f"{python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        results["Python Version"] = (False, f"{python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.10+)")
    
    # Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        if free_gb > 1.0:
            results["Disk Space"] = (True, f"{free_gb:.2f} GB free")
        else:
            results["Disk Space"] = (False, f"{free_gb:.2f} GB free (low)")
    except:
        results["Disk Space"] = (False, "Unable to check")
    
    # Write permissions
    try:
        test_file = Path(".startup_test")
        test_file.write_text("test")
        test_file.unlink()
        results["Write Permissions"] = (True, "OK")
    except:
        results["Write Permissions"] = (False, "No write access")
    
    return results

def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check Python dependencies."""
    results = {}
    dependencies = {
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sympy": "SymPy",
        "matplotlib": "Matplotlib",
        "yaml": "PyYAML",
        "typer": "Typer",
        "torch": "PyTorch",
        "wandb": "Weights & Biases",
    }
    
    for module, name in dependencies.items():
        try:
            if module == "yaml":
                import yaml
                version = "installed"
            elif module == "wandb":
                import wandb
                version = getattr(wandb, "__version__", "installed")
            else:
                mod = __import__(module)
                version = getattr(mod, "__version__", "installed")
            
            results[name] = (True, version)
        except ImportError:
            results[name] = (False, "not installed")
        except Exception as e:
            if module == "torch":
                # PyTorch DLL issue
                results[name] = (False, f"installed but error: {type(e).__name__}")
            else:
                results[name] = (False, f"error: {type(e).__name__}")
    
    return results

def check_project_structure() -> Dict[str, Tuple[bool, str]]:
    """Check project structure."""
    results = {}
    required_dirs = [
        "pde_solver",
        "pde_solver/core",
        "pde_solver/models",
        "pde_solver/training",
        "pde_solver/utils",
        "configs",
        "examples",
    ]
    
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        if exists:
            file_count = len(list(Path(dir_path).glob("*.py")))
            results[dir_path] = (True, f"{file_count} Python files")
        else:
            results[dir_path] = (False, "missing")
    
    return results

def check_imports() -> Dict[str, Tuple[bool, str]]:
    """Check if key modules can be imported."""
    results = {}
    sys.path.insert(0, str(Path.cwd()))
    
    # Test imports that don't require torch
    test_imports = [
        ("pde_solver.utils.logger", "Logger"),
        ("pde_solver.utils.metrics", "Metrics"),
        ("pde_solver.utils.exceptions", "Exceptions"),
        ("pde_solver.symbolic.symbolic_engine", "Symbolic Engine"),
        ("pde_solver.models.burgers", "Burgers Model"),
    ]
    
    for module_path, name in test_imports:
        try:
            __import__(module_path)
            results[name] = (True, "OK")
        except ImportError as e:
            results[name] = (False, f"ImportError: {str(e)[:50]}")
        except Exception as e:
            results[name] = (False, f"{type(e).__name__}: {str(e)[:50]}")
    
    return results

def print_results(category: str, results: Dict[str, Tuple[bool, str]], critical: bool = False):
    """Print check results."""
    print(f"{Colors.BOLD}{category}:{Colors.RESET}")
    all_ok = True
    
    for item, (status, message) in results.items():
        if status:
            icon = f"{Colors.GREEN}[OK]{Colors.RESET}"
        else:
            icon = f"{Colors.RED}[FAIL]{Colors.RESET}"
            if critical:
                all_ok = False
        
        print(f"  {icon} {item:30s} {message}")
    
    print()
    return all_ok

def run_startup_checks():
    """Run all startup checks."""
    all_checks_passed = True
    
    # System requirements
    sys_results = check_system_requirements()
    if not print_results("System Requirements", sys_results, critical=True):
        all_checks_passed = False
    
    # Dependencies
    dep_results = check_dependencies()
    print_results("Dependencies", dep_results, critical=False)
    
    # Project structure
    struct_results = check_project_structure()
    if not print_results("Project Structure", struct_results, critical=True):
        all_checks_passed = False
    
    # Module imports
    import_results = check_imports()
    print_results("Module Imports", import_results, critical=False)
    
    return all_checks_passed

def suggest_solutions():
    """Suggest solutions for common issues."""
    print(f"{Colors.BOLD}{Colors.YELLOW}Recommended Actions:{Colors.RESET}\n")
    
    suggestions = [
        "If PyTorch has DLL issues:",
        "  1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe",
        "  2. Or use CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "",
        "If dependencies are missing:",
        "  pip install -r requirements.txt",
        "",
        "To run classical solver (works without PyTorch):",
        "  python minimal_demo.py",
        "",
        "To test production utilities:",
        "  python -c \"from pde_solver.utils.logger import get_logger; get_logger().info('Test')\"",
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

def main():
    """Main startup routine."""
    print_header()
    
    try:
        # Run checks
        all_passed = run_startup_checks()
        
        # Summary
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
        if all_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS] Startup checks passed!{Colors.RESET}")
            print(f"\n{Colors.CYAN}You can now run:{Colors.RESET}")
            print("  - python minimal_demo.py - Classical solver demo")
            print("  - python run_solver.py --help - CLI interface")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}[WARNING] Some checks failed, but basic functionality may work{Colors.RESET}")
            suggest_solutions()
        
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}\n")
        
        # Try to run a quick test
        print(f"{Colors.BLUE}Running quick functionality test...{Colors.RESET}\n")
        try:
            from pde_solver.utils.logger import get_logger
            logger = get_logger()
            logger.info("Startup test successful")
            print(f"{Colors.GREEN}[OK] Logger test passed{Colors.RESET}\n")
        except Exception as e:
            print(f"{Colors.YELLOW}[WARNING] Logger test failed: {e}{Colors.RESET}\n")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}[ERROR] Startup check failed with error:{Colors.RESET}")
        print(f"{Colors.RED}{str(e)}{Colors.RESET}\n")
        suggest_solutions()
        return 1

if __name__ == "__main__":
    sys.exit(main())

