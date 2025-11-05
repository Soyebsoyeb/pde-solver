"""FastAPI server for PDE solver (production API)."""

from typing import Optional, Dict, Any
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse, HTMLResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from pde_solver.utils.logger import get_logger
from pde_solver.utils.exceptions import PDESolverError
from pde_solver.utils.config_validator import load_config

logger = get_logger()

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="PDE Solver API",
        description="Production API for PDE solving",
        version="0.1.0",
    )

    class TrainRequest(BaseModel):
        """Training request model."""
        config_path: str
        device: Optional[str] = "auto"

    class InferRequest(BaseModel):
        """Inference request model."""
        checkpoint_path: str
        coordinates: Dict[str, Any]

    @app.get("/api-info")
    async def api_info():
        """API information endpoint."""
        return {
            "service": "PDE Solver API",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "metrics": "/metrics",
                "train": "/train (POST)",
                "infer": "/infer (POST)",
                "docs": "/docs"
            }
        }

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Comprehensive PDE Solver Website."""
        website_path = Path(__file__).parent / "website.html"
        with open(website_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    
    @app.get("/old-ui", response_class=HTMLResponse)
    async def old_ui():
        """Original simple UI."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDE Solver - API Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
            overflow: hidden;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.95;
        }
        .status {
            display: inline-block;
            background: #10b981;
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 15px;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .content {
            padding: 40px;
        }
        .version {
            display: inline-block;
            background: #f3f4f6;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
            color: #6b7280;
            margin-bottom: 20px;
        }
        .endpoints {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .endpoint {
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            padding: 25px;
            border-radius: 15px;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 2px solid transparent;
        }
        .endpoint:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border-color: #667eea;
        }
        .endpoint h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        .endpoint p {
            color: #6b7280;
            font-size: 0.95em;
            margin-bottom: 10px;
        }
        .method {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.8em;
            font-weight: bold;
            margin-top: 10px;
        }
        .get { background: #10b981; color: white; }
        .post { background: #3b82f6; color: white; }
        .link {
            display: inline-block;
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
            margin-top: 10px;
            transition: color 0.3s;
        }
        .link:hover {
            color: #764ba2;
        }
        .footer {
            background: #f9fafb;
            padding: 20px;
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
        }
        .quick-links {
            margin-top: 30px;
            text-align: center;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 35px;
            border-radius: 30px;
            text-decoration: none;
            margin: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 PDE Solver API</h1>
            <p>Production API for solving Partial Differential Equations</p>
            <div class="status">✓ Running</div>
        </div>
        
        <div class="content">
            <span class="version">v0.1.0</span>
            
            <h2 style="color: #1f2937; margin-bottom: 20px;">Available Endpoints</h2>
            
            <div class="endpoints">
                <div class="endpoint">
                    <h3>Health Check</h3>
                    <p>Check API server status</p>
                    <span class="method get">GET</span>
                    <br>
                    <a href="/health" class="link">/health →</a>
                </div>
                
                <div class="endpoint">
                    <h3>Metrics</h3>
                    <p>Prometheus monitoring metrics</p>
                    <span class="method get">GET</span>
                    <br>
                    <a href="/metrics" class="link">/metrics →</a>
                </div>
                
                <div class="endpoint">
                    <h3>Train Model</h3>
                    <p>Start PDE model training job</p>
                    <span class="method post">POST</span>
                    <br>
                    <a href="/docs#/default/train_train_post" class="link">/train →</a>
                </div>
                
                <div class="endpoint">
                    <h3>Run Inference</h3>
                    <p>Execute model predictions</p>
                    <span class="method post">POST</span>
                    <br>
                    <a href="/docs#/default/infer_infer_post" class="link">/infer →</a>
                </div>
            </div>
            
            <div class="quick-links">
                <a href="/docs" class="btn">📚 API Documentation</a>
                <a href="/redoc" class="btn">📖 ReDoc</a>
            </div>
        </div>
        
        <div class="footer">
            <p>Powered by FastAPI & PyTorch | PDE Solver © 2025</p>
        </div>
    </div>
</body>
</html>
        """
        return HTMLResponse(content=html_content)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "pde-solver"}

    @app.get("/metrics")
    async def metrics():
        """Metrics endpoint for Prometheus."""
        from pde_solver.utils.metrics import get_metrics_collector
        collector = get_metrics_collector()
        return {"metrics": collector.export()}

    @app.post("/train")
    async def train(request: TrainRequest, background_tasks: BackgroundTasks):
        """Start training job."""
        try:
            config = load_config(request.config_path)
            # Start training in background
            # Implementation would go here
            return {"status": "training_started", "config": request.config_path}
        except PDESolverError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/infer")
    async def infer(request: InferRequest):
        """Run inference."""
        try:
            # Implementation would go here
            return {"status": "success", "predictions": []}
        except PDESolverError as e:
            raise HTTPException(status_code=400, detail=str(e))

else:
    app = None

