"""Start the PDE Solver API server."""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from pde_solver.api.server import app
        
        if app is None:
            print("ERROR: FastAPI is not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
            print("FastAPI installed. Please run this script again.")
            sys.exit(1)
        
        import uvicorn
        
        print("=" * 70)
        print("Starting PDE Solver API Server")
        print("=" * 70)
        print(f"\nServer will be available at:")
        print(f"  - Main URL: http://localhost:8080")
        print(f"  - API Docs: http://localhost:8080/docs")
        print(f"  - Health Check: http://localhost:8080/health")
        print(f"  - Metrics: http://localhost:8080/metrics")
        print("\nPress CTRL+C to stop the server")
        print("=" * 70 + "\n")
        
        # Start server (reload=False for Windows compatibility)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            reload=False,
            log_level="info"
        )
    
    except ImportError as e:
        print(f"ERROR: Missing dependencies - {e}")
        print("\nInstalling required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]"])
        print("\nDependencies installed. Please run this script again:")
        print(f"  python {__file__}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
        sys.exit(0)
