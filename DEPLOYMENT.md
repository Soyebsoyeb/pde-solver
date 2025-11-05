# Production Deployment Guide

## Overview

This guide covers deploying the PDE Solver in production environments with industry-standard practices.

## Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for K8s deployment)
- NVIDIA GPU (optional, for GPU acceleration)
- Persistent storage for data, outputs, and checkpoints

## Deployment Options

### Option 1: Docker Compose (Recommended for Single Server)

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f pde-solver

# Stop services
docker-compose -f docker-compose.prod.yml down
```

### Option 2: Kubernetes

```bash
# Apply configurations
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/job.yaml

# Check status
kubectl get pods -l app=pde-solver
kubectl logs -f deployment/pde-solver

# Submit training job
kubectl apply -f kubernetes/job.yaml
```

### Option 3: Direct Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Configure environment
export LOG_LEVEL=INFO
export WANDB_PROJECT=pde-solver-prod

# Run with systemd service (see services/pde-solver.service)
sudo systemctl start pde-solver
```

## Configuration

### Environment Variables

```bash
# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=/workspace/logs/pde-solver.log

# Weights & Biases
WANDB_PROJECT=pde-solver-prod
WANDB_API_KEY=your-api-key

# Resource Limits
MAX_MEMORY_GB=8
MAX_CPU_PERCENT=80
MAX_GPU_MEMORY_GB=16

# Device Configuration
DEVICE=cuda  # or cpu
CUDA_VISIBLE_DEVICES=0,1
```

### Resource Limits

Configure in `configs/production.yaml`:

```yaml
# Resource limits
resource_limits:
  max_memory_gb: 8.0
  max_cpu_percent: 80.0
  max_gpu_memory_gb: 16.0
```

## Monitoring

### Prometheus Metrics

The application exposes metrics at `/metrics` endpoint:

- `pde_solver_training_loss` - Training loss
- `pde_solver_inference_time` - Inference time
- `pde_solver_memory_usage` - Memory usage
- `pde_solver_gpu_usage` - GPU usage

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

Pre-configured dashboards:
- Training metrics
- Resource usage
- Error rates
- Performance metrics

### Logging

Logs are written to:
- Console (stdout/stderr)
- File: `/workspace/logs/pde-solver.log`
- Error log: `/workspace/logs/error_pde-solver.log`
- JSON format available for log aggregation

## Security

### Security Checklist

- [ ] Set secure file permissions on config files
- [ ] Use environment variables for secrets (never commit)
- [ ] Enable HTTPS for API endpoints
- [ ] Configure firewall rules
- [ ] Use non-root user in containers
- [ ] Regularly update dependencies
- [ ] Scan images for vulnerabilities

### Run Security Check

```bash
python -c "from pde_solver.utils.security import check_environment_security; print(check_environment_security())"
```

## Scaling

### Horizontal Scaling

For Kubernetes:
```bash
kubectl scale deployment pde-solver --replicas=3
```

### Vertical Scaling

Update resource limits in `kubernetes/deployment.yaml`:
```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
  limits:
    memory: "16Gi"
    cpu: "8"
```

## Backup and Recovery

### Backup Strategy

1. **Checkpoints**: Backup regularly to external storage
2. **Configurations**: Version control all configs
3. **Outputs**: Archive important results
4. **Logs**: Rotate and archive logs

### Backup Script

```bash
#!/bin/bash
# Backup checkpoints and outputs
tar -czf backup-$(date +%Y%m%d).tar.gz checkpoints/ outputs/ configs/
# Upload to S3/cloud storage
aws s3 cp backup-*.tar.gz s3://your-bucket/backups/
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **GPU Not Available**
   - Check CUDA installation
   - Verify GPU drivers
   - Check `nvidia-smi`

3. **Slow Training**
   - Enable mixed precision
   - Increase batch size (if memory allows)
   - Use multiple GPUs with DDP

4. **Logging Issues**
   - Check log directory permissions
   - Verify disk space
   - Check log level configuration

## Health Checks

### Manual Health Check

```bash
curl http://localhost:8000/health
```

### Automatic Health Checks

Docker Compose includes health checks:
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import pde_solver; print('OK')"]
  interval: 30s
  timeout: 10s
  retries: 3
```

## Performance Tuning

### Optimization Tips

1. **Use Mixed Precision**: Set `use_mixed_precision: true` in config
2. **Optimize Batch Size**: Balance memory and convergence
3. **Enable Caching**: Use file cache for repeated computations
4. **Profile Code**: Use PyTorch profiler to find bottlenecks
5. **Use torch.compile()**: JIT compilation for faster inference

## Support

For production issues:
- Check logs: `docker-compose logs pde-solver`
- Review metrics in Grafana
- Check resource usage: `kubectl top pods`
- Contact support with error logs and metrics

