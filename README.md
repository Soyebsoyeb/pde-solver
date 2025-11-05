# 🔬 PDE Solver - Physics-Informed Neural Networks Platform

**Advanced AI-powered platform for solving Partial Differential Equations**

A production-ready system combining classical numerical methods with cutting-edge Physics-Informed Neural Networks (PINNs) to solve complex PDEs including Burgers equation, Navier-Stokes, and more.

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.11-blue)]() [![PyTorch](https://img.shields.io/badge/pytorch-2.9-red)]() [![FastAPI](https://img.shields.io/badge/fastapi-latest-green)]()

---

## 🎯 What This Project Does

This PDE solver enables you to:
- **Train neural networks** to solve partial differential equations
- **Run classical solvers** for comparison and validation
- **Deploy via REST API** for production use
- **Visualize solutions** with beautiful plots
- **Monitor training** in real-time with metrics

It's designed for researchers, engineers, and data scientists working on scientific computing, computational physics, and numerical analysis.

## Features

- **Hybrid Approach**: Combines classical numerical methods with neural networks
- **Multiple PDE Types**: Burgers, Navier-Stokes, Schrödinger, Einstein equations
- **Neural Methods**: Physics-Informed Neural Networks (PINNs), DeepONet, Fourier Neural Operators
- **Production Ready**: Industry-standard logging, monitoring, error handling, and deployment
- **Enterprise Grade**: Kubernetes-ready, scalable, secure, and fully observable
- **Research Grade**: Designed for research engineers and applied scientists

### Production Features

- ✅ **Structured Logging**: JSON-formatted logs with multiple handlers
- ✅ **Error Handling**: Comprehensive exception hierarchy with retry mechanisms
- ✅ **Configuration Validation**: Pydantic-based validation with type safety
- ✅ **Performance Monitoring**: Built-in metrics collection and Prometheus integration
- ✅ **Caching System**: File-based caching for expensive operations
- ✅ **Security**: Path sanitization, permission checks, environment validation
- ✅ **Resource Management**: Memory, CPU, and GPU monitoring with limits
- ✅ **Deployment Ready**: Kubernetes, Docker Compose, and API server
- ✅ **Observability**: Prometheus metrics, Grafana dashboards, health checks

## 🚀 Quick Start (2 Minutes)

### Prerequisites
- Python 3.11
- Windows/Linux/macOS
- (Optional) NVIDIA GPU for training acceleration

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pde_solver

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

**Windows Users:** If you encounter PyTorch DLL errors, install Visual C++ Redistributable:
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart

### Verify Installation

```bash
# Run tests
python test_basic_functionality.py

# Should see:
# [PASS] Package imported successfully
# [PASS] Classical solver works correctly
# [PASS] Visualization works correctly
# [SUCCESS] All tests passed!
```

### Running Examples

```bash
# Train Burgers equation solver (small/fast)
python run_solver.py train configs/burgers_small.yaml

# Evaluate trained model
python run_solver.py eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt

# Run classical solver
python run_solver.py classical configs/burgers_small.yaml

# Run demo script
python examples/burgers_demo.py
```

### Docker

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t pde-solver:latest .

# Run container
docker run --rm -v $(pwd)/outputs:/workspace/outputs pde-solver:latest python examples/burgers_demo.py

# Or use docker-compose
docker-compose up
```

## Architecture

```
pde_solver/
├── core/                    # Core solver infrastructure
│   ├── neural_symbolic_solver.py
│   ├── physics_constraints.py
│   └── adaptive_mesh.py
├── models/                  # PDE solver models
│   ├── burgers.py
│   ├── navier_stokes.py
│   ├── quantum_solvers.py
│   ├── einstein_equations.py
│   └── deeponet.py
├── training/                # Training infrastructure
│   ├── physics_informed_trainer.py
│   ├── multi_objective_loss.py
│   └── adaptive_sampling.py
├── symbolic/                # Symbolic equation processing
│   ├── symbolic_engine.py
│   └── equation_parser.py
└── utils/                   # Utilities
    ├── visualization.py
    └── result_analysis.py
```

## Examples

### Burgers Equation

The canonical example demonstrating both classical and neural methods:

```python
from pde_solver.models.burgers import BurgersPINN
from pde_solver.training.physics_informed_trainer import PhysicsInformedTrainer

# Create model
model = BurgersPINN(input_dim=2, output_dim=1, nu=0.01)

# Train (see examples/burgers_demo.py for full example)
# ...
```

### Configuration Files

Configuration files use YAML format:

```yaml
pde_type: burgers
nu: 0.01
x_domain: [-1.0, 1.0]
t_domain: [0.0, 1.0]
n_points: 10000
num_epochs: 1000
learning_rate: 1e-3
```

## Testing

```bash
# Run all tests
make test
# or
pytest tests/ -v

# Run linting
make lint

# Format code
make format
```

## Performance & Benchmarks

See `benchmarks/` directory for benchmark reports. Expected performance:

- **Burgers (small)**: CPU training < 30 minutes, GPU < 5 minutes
- **Navier-Stokes (2D)**: GPU training ~2-4 hours
- **Memory**: ~2-4 GB for typical problems

---

## 🔧 How It Works

### The Complete System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                            │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐            │
│  │  CLI Tool  │  │   Web UI   │  │  REST API   │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬──────┘            │
└─────────┼────────────────┼───────────────┼───────────────────┘
          │                │               │
          └────────────────┼───────────────┘
                          │
          ┌───────────────▼────────────────┐
          │     SOLVER ORCHESTRATION       │
          │  - Configuration Management    │
          │  - Training Pipeline           │
          │  - Inference Engine           │
          └───────────────┬────────────────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                 │
   ┌──────▼──────┐  ┌────▼─────┐   ┌──────▼──────┐
   │  Classical  │  │   PINN   │   │  Monitoring │
   │   Solver    │  │  Solver  │   │   & Logs    │
   └─────────────┘  └────┬─────┘   └─────────────┘
                         │
                  ┌──────▼──────┐
                  │  Physics    │
                  │ Constraints │
                  └─────────────┘
```

### Key Components Explained

#### 1. **Classical Solvers** (`pde_solver/models/classical_solvers.py`)
- Uses finite difference methods
- Pure NumPy/SciPy (no PyTorch dependency)
- Fast for comparison and validation
- Works even if PyTorch has issues

#### 2. **Physics-Informed Neural Networks (PINNs)** (`pde_solver/models/burgers.py`)
- Neural networks that learn PDE solutions
- Loss function includes:
  - **PDE residual**: How well it satisfies the equation
  - **Boundary conditions**: Enforces BC at domain edges
  - **Initial conditions**: Matches known starting state
- Uses Fourier features for better approximation

#### 3. **Training System** (`pde_solver/training/`)
- AdamW optimizer with learning rate scheduling
- Multi-objective loss balancing
- Checkpoint saving every N epochs
- Real-time metrics tracking

#### 4. **API Server** (`pde_solver/api/server.py`)
- FastAPI-based REST endpoints
- Health checks and monitoring
- Interactive Swagger UI
- Production-ready deployment

---

## 📡 Using the API

### Start the API Server

```bash
python start_server.py
```

Server runs on **http://localhost:8080**

### Available Endpoints

#### 1. **Health Check**
```bash
curl http://localhost:8080/health
# Response: {"status":"healthy","service":"pde-solver"}
```

#### 2. **Train a Model**
```bash
curl -X POST "http://localhost:8080/train" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "configs/burgers_small.yaml",
    "device": "cpu"
  }'
```

#### 3. **Run Inference**
```bash
curl -X POST "http://localhost:8080/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "checkpoints/checkpoint_epoch_499.pt",
    "coordinates": {"x": [0.0, 0.5, 1.0], "t": [0.0, 0.1]}
  }'
```

#### 4. **Interactive Documentation**
- Visit **http://localhost:8080/docs** for Swagger UI
- Visit **http://localhost:8080** for the main website

---

## 💻 Command Line Interface (CLI)

The CLI provides three main commands:

### 1. Train Command
```bash
python -m pde_solver.cli train configs/burgers_small.yaml --device cpu

# Options:
#   --device cpu|cuda     Device to use for training
```

**What happens:**
- Loads configuration from YAML
- Creates PINN model with specified architecture
- Generates training data (collocation points, BCs, ICs)
- Trains for specified epochs
- Saves checkpoints to `checkpoints/`
- Generates loss plots to `outputs/`

### 2. Eval Command
```bash
python -m pde_solver.cli eval configs/burgers_small.yaml checkpoints/checkpoint_epoch_499.pt --device cpu
```

**What happens:**
- Loads trained model from checkpoint
- Runs classical solver for comparison
- Computes error metrics (L2, L∞, relative error)
- Generates visualization plots
- Prints comparison results

### 3. Classical Command
```bash
python -m pde_solver.cli classical configs/burgers_small.yaml
```

**What happens:**
- Runs finite difference solver (no neural network)
- Fast execution (seconds to minutes)
- Saves visualization to `outputs/`
- Works without PyTorch

---

## 🎨 Visualization

All commands generate publication-quality plots:

- **Training loss curves** - Shows convergence over epochs
- **Solution snapshots** - PDE solution at different time steps
- **Comparison plots** - PINN vs Classical solver
- **Error heatmaps** - Spatial distribution of errors

Plots are saved to the `outputs/` directory.

---

## 🛠️ Troubleshooting

### PyTorch DLL Error on Windows

**Error:** `[WinError 1114] A dynamic link library (DLL) initialization routine failed`

**Solution:**
```bash
# Install Visual C++ Redistributable
# Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
# Then restart your computer
```

**Alternative:** Use CPU-only PyTorch
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory During Training

**Solution:** Reduce batch size or number of points in config
```yaml
batch_size: 256  # Reduce from 1024
n_points: 2500   # Reduce from 10000
```

### Training Loss Not Decreasing

**Solutions:**
1. Increase learning rate: `learning_rate: 5e-3`
2. Adjust loss weights in config
3. Use more Fourier features: `num_fourier_features: 256`
4. Increase network capacity: `hidden_dims: [512, 512, 512]`

### Port Already in Use

If port 8080 is busy:
```python
# Edit start_server.py, change port to 8081 or another
uvicorn.run(app, host="0.0.0.0", port=8081)
```

---

## 📚 Learn More

### Key Files to Understand

1. **`pde_solver/cli.py`** - Command line interface entry point
2. **`pde_solver/models/burgers.py`** - PINN model definition  
3. **`pde_solver/training/physics_informed_trainer.py`** - Training loop
4. **`pde_solver/api/server.py`** - FastAPI REST server
5. **`configs/burgers_small.yaml`** - Configuration example

### Documentation

- **QUICKSTART.md** - Fast 5-minute guide
- **DEPLOYMENT.md** - Production deployment
- **PYTORCH_WORKAROUND.md** - Windows PyTorch fixes
- **API Docs** - http://localhost:8080/docs (when server running)

---

## 🎯 Project Status

✅ **Fully Runnable** - All code works out of the box  
✅ **Production Ready** - REST API, monitoring, error handling  
✅ **Well Documented** - Comprehensive docs and examples  
✅ **Tested** - Unit tests and integration tests pass  
✅ **Deployed** - Web UI and API server operational  

### What We Fixed to Make It Work

1. **Lazy imports** - PyTorch loads only when needed
2. **Separated classical solvers** - Work without PyTorch
3. **Fixed learning rate parsing** - YAML string to float conversion
4. **Fixed Fourier features** - Corrected dimension mismatch
5. **Removed deprecated params** - Updated for PyTorch 2.9+
6. **Added comprehensive UI** - Beautiful web interface
7. **Created test suite** - Verify everything works

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- **PyTorch** - Deep learning framework
- **FastAPI** - Modern web framework
- **NumPy/SciPy** - Scientific computing
- **Matplotlib** - Visualization
- **Uvicorn** - ASGI server

---

## 📞 Support

For issues, questions, or discussions:
- Check **PYTORCH_WORKAROUND.md** for common Windows issues
- Run `python test_basic_functionality.py` to diagnose problems
- Review logs in `outputs/` directory

**Happy PDE Solving! 🚀**

## Documentation

- [API Reference](docs/api.md) - Detailed API documentation
- [Examples](examples/) - Example scripts and notebooks
- [Contributing](CONTRIBUTING.md) - Guidelines for contributors

## References

- **PINNs**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

- **DeepONet**: Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3), 218-229.

- **Fourier Neural Operator**: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

## Production Deployment

For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Kubernetes deployment
- Docker Compose setup
- Monitoring and observability
- Security best practices
- Performance tuning

## Industry Standards

This project meets industry standards for production software:
- ✅ Code quality (type hints, documentation, linting)
- ✅ Reliability (error handling, retry, testing)
- ✅ Security (input validation, secure defaults)
- ✅ Observability (logging, metrics, tracing)
- ✅ Scalability (resource management, horizontal scaling)

See [INDUSTRY_STANDARDS.md](INDUSTRY_STANDARDS.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pde_solver,
  title = {PDE Solver: Hybrid Physics + AI PDE Engine},
  author = {PDE Solver Contributors},
  year = {2024},
  url = {https://github.com/Soyebsoyeb/pde_solver}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Limitations & Future Work

- PINNs may struggle on high Reynolds numbers; hybrid classical+neural strategies recommended
- GR solver currently supports simplified toy problems
- GPU support optimized for NVIDIA CUDA 11.8+

## Support

For issues and questions, please open an issue on GitHub.

