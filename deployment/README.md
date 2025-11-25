# Multi-Cloud Serverless Orchestration - AWS Deployment

This directory contains the deployment package for the Hierarchical DRL framework for multi-cloud serverless function placement.

## Architecture

The system consists of three DRL layers:
1. **Strategic Layer (DQN)**: Cloud provider selection (AWS/Azure/GCP)
2. **Tactical Layer (PPO)**: Regional placement and memory allocation
3. **Operational Layer (LSTM)**: Workload prediction and resource scaling

## Prerequisites

- Docker Desktop with BuildX support
- AWS CLI configured with appropriate credentials
- Python 3.10+ (for local testing)
- At least 2GB of free disk space

## Directory Structure

```
deployment/
├── src/
│   ├── models/          # PyTorch model architectures
│   ├── inference/       # Hierarchical coordinator
│   ├── preprocessing/   # Feature engineering
│   └── api/            # FastAPI service
├── data/               # Trained model weights (.pt files)
├── docker/             # Dockerfile and docker-compose
├── scripts/            # Deployment automation scripts
└── requirements.txt    # Python dependencies
```

## Local Testing

### Option 1: Run with Docker Compose

```bash
cd deployment
docker-compose -f docker/docker-compose.yml up --build
```

Access the API at: http://localhost:8000

### Option 2: Run directly with Python

```bash
cd deployment
pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Make a prediction (example)
curl -X POST http://localhost:8000/decision \\
  -H "Content-Type: application/json" \\
  -d @test_request.json
```

## AWS Deployment

### Quick Start (Automated)

The deployment script automates the entire AWS setup:

```bash
cd deployment
./scripts/deploy-to-aws.sh
```

This script will:
1. ✓ Check AWS configuration
2. ✓ Create ECR repository
3. ✓ Build Docker image for AMD64 (AWS-compatible)
4. ✓ Push image to ECR
5. ✓ Create ECS cluster
6. ✓ Create IAM execution role
7. ✓ Create CloudWatch log group
8. ✓ Register ECS task definition
9. ✓ Create/update ECS service with Fargate

### Manual Deployment Steps

If you prefer manual control:

#### 1. Build Docker Image (AMD64 for AWS)

**IMPORTANT**: On Mac M4 (ARM64), you MUST build for AMD64:

```bash
cd deployment

# Build for AMD64 architecture
docker buildx build \\
  --platform linux/amd64 \\
  -t hierarchical-drl-api:latest \\
  -f docker/Dockerfile \\
  --load \\
  .
```

#### 2. Create ECR Repository

```bash
AWS_REGION="eu-west-1"
AWS_PROFILE="thesis-deployment"
ECR_REPO_NAME="hierarchical-drl-api"

# Create repository
aws ecr create-repository \\
  --repository-name $ECR_REPO_NAME \\
  --profile $AWS_PROFILE \\
  --region $AWS_REGION \\
  --image-scanning-configuration scanOnPush=true
```

#### 3. Push to ECR

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)

# Login to ECR
aws ecr get-login-password --profile $AWS_PROFILE --region $AWS_REGION | \\
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Tag and push
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
docker tag hierarchical-drl-api:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

#### 4. Create ECS Cluster

```bash
aws ecs create-cluster \\
  --cluster-name thesis-cluster \\
  --profile $AWS_PROFILE \\
  --region $AWS_REGION \\
  --capacity-providers FARGATE
```

#### 5. Deploy Service

Use the deployment script or AWS Console to create the ECS service with:
- **CPU**: 512 (0.5 vCPU)
- **Memory**: 1024 MB (1 GB)
- **Launch Type**: Fargate
- **Network**: Public subnet with auto-assign public IP
- **Port**: 8000

### Get Service Endpoint

```bash
# List running tasks
aws ecs list-tasks \\
  --cluster thesis-cluster \\
  --service-name drl-api-service \\
  --profile $AWS_PROFILE \\
  --region $AWS_REGION

# Get task details for public IP
TASK_ARN=$(aws ecs list-tasks --cluster thesis-cluster --service-name drl-api-service --profile $AWS_PROFILE --region $AWS_REGION --query 'taskArns[0]' --output text)

aws ecs describe-tasks \\
  --cluster thesis-cluster \\
  --tasks $TASK_ARN \\
  --profile $AWS_PROFILE \\
  --region $AWS_REGION \\
  --query 'tasks[0].attachments[0].details[?name==\`networkInterfaceId\`].value' \\
  --output text
```

### View Logs

```bash
aws logs tail /ecs/thesis-deployment --follow \\
  --profile $AWS_PROFILE \\
  --region $AWS_REGION
```

## API Documentation

Once deployed, access the interactive API documentation at:
- **Swagger UI**: http://{PUBLIC_IP}:8000/docs
- **ReDoc**: http://{PUBLIC_IP}:8000/redoc

## Model Files

The deployment package includes three trained models:

- `data/best_enhanced_dqn.pt` (93 KB) - Strategic layer
- `data/best_ppo_tactical.pt` (151 KB) - Tactical layer
- `data/best_lstm_predictor.pt` (476 KB) - Operational layer

**Total model size**: ~720 KB

## Performance Specifications

- **Inference Latency**: <50ms end-to-end
- **Throughput**: 100+ decisions/second
- **Memory Footprint**: ~500 MB
- **CPU**: 0.5 vCPU sufficient for typical loads

## Cost Estimation (AWS eu-west-1)

### Fargate Pricing
- **vCPU**: 0.5 × $0.04656/hour = $0.02328/hour
- **Memory**: 1 GB × $0.00511/GB/hour = $0.00511/hour
- **Total**: ~$0.028/hour = **$20.64/month** (continuous)

### Additional Costs
- **ECR Storage**: ~$0.10/GB/month (negligible for <1GB)
- **Data Transfer**: First 100GB free
- **CloudWatch Logs**: ~$0.50/GB ingested

**Estimated Monthly Total**: **~$21-25/month**

## Troubleshooting

### Issue: Docker build fails on Mac M4

**Solution**: Ensure you're using `--platform linux/amd64`:

```bash
docker buildx build --platform linux/amd64 -t myimage .
```

### Issue: ECS task fails to start

**Solution**: Check CloudWatch logs:

```bash
aws logs tail /ecs/thesis-deployment --follow --profile thesis-deployment --region eu-west-1
```

### Issue: Models not loading

**Solution**: Verify model files are in `deployment/data/`:

```bash
ls -lh deployment/data/*.pt
```

### Issue: Permission denied on scripts

**Solution**: Make scripts executable:

```bash
chmod +x scripts/*.sh
```

## Security Considerations

- **IAM Roles**: Use least-privilege execution roles
- **Security Groups**: Restrict inbound traffic to necessary ports
- **Container**: Runs as non-root user (appuser)
- **Secrets**: Use AWS Secrets Manager for sensitive data (not hardcoded)

## Monitoring

Monitor the deployment using:

1. **CloudWatch Metrics**: CPU, memory, network
2. **CloudWatch Logs**: Application logs
3. **ECS Console**: Task health and service status
4. **Container Insights**: Enhanced monitoring (optional)

## Scaling

To scale the service:

```bash
aws ecs update-service \\
  --cluster thesis-cluster \\
  --service drl-api-service \\
  --desired-count 3 \\
  --profile thesis-deployment \\
  --region eu-west-1
```

## Cleanup

To remove all AWS resources:

```bash
# Delete ECS service
aws ecs delete-service --cluster thesis-cluster --service drl-api-service --force --profile thesis-deployment --region eu-west-1

# Delete ECS cluster
aws ecs delete-cluster --cluster thesis-cluster --profile thesis-deployment --region eu-west-1

# Delete ECR repository
aws ecr delete-repository --repository-name hierarchical-drl-api --force --profile thesis-deployment --region eu-west-1

# Delete CloudWatch log group
aws logs delete-log-group --log-group-name /ecs/thesis-deployment --profile thesis-deployment --region eu-west-1
```

## Support

For issues or questions, refer to:
- Repository: [rohit-thesis](https://github.com/robyspace/rohit-thesis)
- Documentation: See `DEPLOYMENT_READINESS_ASSESSMENT.md` in project root

## License

MSc Thesis Research Project - 2025
