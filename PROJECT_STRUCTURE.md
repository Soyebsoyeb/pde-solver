# Project Structure

```
pde_solver/
├── pde_solver/              # Main package
│   ├── __init__.py
│   ├── cli.py               # CLI interface
│   ├── core/                # Core infrastructure
│   │   ├── __init__.py
│   │   ├── neural_symbolic_solver.py
│   │   ├── physics_constraints.py
│   │   └── adaptive_mesh.py
│   ├── models/              # PDE solver models
│   │   ├── __init__.py
│   │   ├── burgers.py       # Burgers equation (PINN + classical)
│   │   └── deeponet.py      # DeepONet operator learning
│   ├── training/            # Training infrastructure
│   │   ├── __init__.py
│   │   ├── physics_informed_trainer.py
│   │   ├── multi_objective_loss.py
│   │   └── adaptive_sampling.py
│   ├── symbolic/            # Symbolic equation processing
│   │   ├── __init__.py
│   │   └── symbolic_engine.py
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── visualization.py
│       └── result_analysis.py
│
├── configs/                 # Configuration files
│   ├── burgers_small.yaml
│   └── burgers_medium.yaml
│
├── examples/                # Example scripts
│   ├── burgers_demo.py
│   └── README.md
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_burgers.py
│   └── test_core.py
│
├── scripts/                 # Utility scripts
│   └── validate_structure.py
│
├── outputs/                 # Generated outputs (auto-created)
│   └── README.md
│
├── checkpoints/             # Model checkpoints (auto-created)
│   └── README.md
│
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
│
├── run_solver.py           # CLI entry point
├── setup.py                # Setup script
├── pyproject.toml          # Modern Python packaging
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
├── Dockerfile              # Docker (GPU)
├── Dockerfile.cpu          # Docker (CPU)
├── docker-compose.yml      # Docker Compose
├── Makefile                # Common tasks
├── README.md               # Main documentation
├── QUICKSTART.md           # Quick start guide
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Code of conduct
└── LICENSE                 # MIT License
```

## Key Components

### Core Modules
- **NeuralSymbolicSolver**: High-level solver orchestration
- **PhysicsConstraints**: Physics constraint enforcement
- **AdaptiveMesh**: Adaptive mesh refinement

### Models
- **BurgersPINN**: Physics-informed neural network for Burgers' equation
- **BurgersClassicalSolver**: Classical finite difference solver
- **DeepONet**: Operator learning network

### Training
- **PhysicsInformedTrainer**: Training loop with logging, checkpointing
- **MultiObjectiveLoss**: Weighted loss combining PDE residual, BC, IC, data
- **AdaptiveSampling**: Residual-driven sampling strategies

### Utilities
- **Visualization**: Plotting and animation utilities
- **ResultAnalysis**: Evaluation metrics (L2, L∞, conservation checks)

## Usage Flow

1. **Configure**: Edit YAML config files in `configs/`
2. **Train**: Run `python run_solver.py train config.yaml`
3. **Evaluate**: Run `python run_solver.py eval config.yaml checkpoint.pt`
4. **Visualize**: Check `outputs/` directory for plots

## Extension Points

To add new PDE types:
1. Create model in `pde_solver/models/`
2. Add symbolic equation in `pde_solver/symbolic/`
3. Create config file in `configs/`
4. Add example in `examples/`
5. Add tests in `tests/`

