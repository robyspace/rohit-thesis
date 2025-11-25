# AWS Deployment Guide - Step by Step

## Overview

This guide walks you through deploying the Hierarchical DRL Multi-Cloud Orchestration API to AWS ECS using Fargate.

**Total Time**: ~30-45 minutes
**Difficulty**: Intermediate
**Cost**: ~$21-25/month

---

## Prerequisites Checklist

- [ ] Docker Desktop installed and running
- [ ] AWS CLI configured (`aws configure --profile thesis-deployment`)
- [ ] AWS account with permissions for ECR, ECS, IAM, CloudWatch
- [ ] At least 2GB free disk space
- [ ] Model files present in `deployment/data/` directory

---

## Deployment Steps

### Step 1: Verify Setup (5 minutes)

```bash
# Navigate to deployment directory
cd /Users/robyspace/Documents/GitHub/rohit-thesis/deployment

# Check AWS configuration
aws sts get-caller-identity --profile thesis-deployment --region eu-west-1

# Verify model files
ls -lh data/*.pt

# Expected output:
# best_enhanced_dqn.pt    (93K)
# best_ppo_tactical.pt    (151K)
# best_lstm_predictor.pt  (476K)

# Verify Docker is running
docker info
```

### Step 2: Test Locally (Optional but Recommended) (10 minutes)

```bash
# Option A: Quick test with Docker
docker buildx build --platform linux/amd64 -t test-api -f docker/Dockerfile .
docker run -p 8000:8000 test-api

# Option B: Test with Python directly
pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload

# In another terminal, test the API:
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","models_loaded":true,"device":"cpu"}
```

### Step 3: Automated AWS Deployment (20-30 minutes)

```bash
# Run the automated deployment script
./scripts/deploy-to-aws.sh

# The script will:
# [1/8] Check AWS configuration
# [2/8] Create ECR repository
# [3/8] Build Docker image for AMD64
# [4/8] Push image to ECR
# [5/8] Create ECS cluster
# [6/8] Create IAM execution role
# [7/8] Create CloudWatch log group
# [8/8] Create/update ECS service

# Wait for deployment to complete (~5-10 minutes)
```

### Step 4: Get Service Endpoint (2 minutes)

```bash
# List tasks
aws ecs list-tasks \
  --cluster thesis-cluster \
  --service-name drl-api-service \
  --profile thesis-deployment \
  --region eu-west-1

# Get task ARN
TASK_ARN=$(aws ecs list-tasks --cluster thesis-cluster --service-name drl-api-service --profile thesis-deployment --region eu-west-1 --query 'taskArns[0]' --output text)

# Get network interface ID
NI_ID=$(aws ecs describe-tasks \
  --cluster thesis-cluster \
  --tasks $TASK_ARN \
  --profile thesis-deployment \
  --region eu-west-1 \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text)

# Get public IP
PUBLIC_IP=$(aws ec2 describe-network-interfaces \
  --network-interface-ids $NI_ID \
  --profile thesis-deployment \
  --region eu-west-1 \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text)

echo "Service available at: http://$PUBLIC_IP:8000"
```

### Step 5: Test Deployed Service (5 minutes)

```bash
# Health check
curl http://$PUBLIC_IP:8000/health

# View API documentation
open http://$PUBLIC_IP:8000/docs

# Make a test decision request
curl -X POST http://$PUBLIC_IP:8000/decision \
  -H "Content-Type: application/json" \
  -d @test_request.json

# Expected response:
# {
#   "cloud_provider": "AWS",
#   "region": "us-east-1",
#   "memory_mb": 512,
#   "predicted_resources": null,
#   "confidence": {
#     "cloud_provider": 0.45,
#     "placement": 0.32
#   }
# }
```

### Step 6: Monitor Service (Ongoing)

```bash
# View logs in real-time
aws logs tail /ecs/thesis-deployment --follow \
  --profile thesis-deployment \
  --region eu-west-1

# Check service status
aws ecs describe-services \
  --cluster thesis-cluster \
  --services drl-api-service \
  --profile thesis-deployment \
  --region eu-west-1

# View CloudWatch metrics (AWS Console)
open https://eu-west-1.console.aws.amazon.com/cloudwatch/
```

---

## Architecture-Specific Notes

### Mac M4 (ARM64) → AWS (AMD64)

Your Mac M4 uses ARM64 architecture, but AWS Fargate runs on AMD64 (x86_64). The Dockerfile and build scripts handle this automatically:

```dockerfile
FROM --platform=linux/amd64 python:3.10-slim
```

```bash
docker buildx build --platform linux/amd64 ...
```

**Why this matters:**
- ARM64 images won't run on AWS Fargate
- The build process cross-compiles for AMD64
- BuildX handles the architecture translation

---

## Troubleshooting

### Problem: "exec format error" in ECS

**Cause**: Image built for wrong architecture (ARM64 instead of AMD64)

**Solution**:
```bash
# Verify image architecture
docker inspect hierarchical-drl-api:latest | grep Architecture

# Should show: "Architecture": "amd64"
# If it shows "arm64", rebuild with:
docker buildx build --platform linux/amd64 -f docker/Dockerfile .
```

### Problem: Task keeps stopping/restarting

**Cause**: Usually model loading issues or OOM

**Solution**:
```bash
# Check logs
aws logs tail /ecs/thesis-deployment --profile thesis-deployment --region eu-west-1

# Common issues:
# - Model files missing: Ensure they're in data/ directory
# - Out of memory: Increase task memory to 2048 MB
# - Port already in use: Check security group rules
```

### Problem: Can't connect to service

**Cause**: Security group or network misconfiguration

**Solution**:
```bash
# Verify security group allows port 8000
aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=thesis-deployment-sg" \
  --profile thesis-deployment \
  --region eu-west-1

# Should have ingress rule for port 8000 from 0.0.0.0/0
```

---

## Performance Tuning

### Increase Service Capacity

```bash
# Scale to 3 instances
aws ecs update-service \
  --cluster thesis-cluster \
  --service drl-api-service \
  --desired-count 3 \
  --profile thesis-deployment \
  --region eu-west-1
```

### Increase Task Resources

Edit task definition to:
- **CPU**: 1024 (1 vCPU) for faster inference
- **Memory**: 2048 MB for larger batches

### Enable Auto Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/thesis-cluster/drl-api-service \
  --min-capacity 1 \
  --max-capacity 5 \
  --profile thesis-deployment \
  --region eu-west-1

# Create scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/thesis-cluster/drl-api-service \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json \
  --profile thesis-deployment \
  --region eu-west-1
```

---

## Cost Optimization

### Option 1: Use Fargate Spot (50-70% cheaper)

Modify deployment script to use Fargate Spot:

```bash
--capacity-provider-strategy capacityProvider=FARGATE_SPOT,weight=1
```

**Trade-off**: Tasks may be interrupted with 2-minute warning

### Option 2: Stop During Off-Hours

```bash
# Stop service (retains configuration)
aws ecs update-service \
  --cluster thesis-cluster \
  --service drl-api-service \
  --desired-count 0 \
  --profile thesis-deployment \
  --region eu-west-1

# Resume service
aws ecs update-service \
  --cluster thesis-cluster \
  --service drl-api-service \
  --desired-count 1 \
  --profile thesis-deployment \
  --region eu-west-1
```

### Option 3: Use Lambda (for low traffic)

For <100 requests/day, AWS Lambda might be cheaper:
- No continuous charges
- Pay only for execution time
- Cold start latency (~2-3 seconds)

---

## Next Steps

After successful deployment:

1. **Add CI/CD**: Set up GitHub Actions for automated deployments
2. **Add Load Balancer**: Use ALB for SSL and multi-AZ deployment
3. **Add Monitoring**: CloudWatch dashboards and alarms
4. **Add Authentication**: API Gateway with Cognito
5. **Add Caching**: Redis/ElastiCache for repeated queries

---

## Cleanup

To remove all resources and stop charges:

```bash
# Delete service
aws ecs delete-service \
  --cluster thesis-cluster \
  --service drl-api-service \
  --force \
  --profile thesis-deployment \
  --region eu-west-1

# Delete cluster
aws ecs delete-cluster \
  --cluster thesis-cluster \
  --profile thesis-deployment \
  --region eu-west-1

# Delete ECR repository
aws ecr delete-repository \
  --repository-name hierarchical-drl-api \
  --force \
  --profile thesis-deployment \
  --region eu-west-1

# Delete log group
aws logs delete-log-group \
  --log-group-name /ecs/thesis-deployment \
  --profile thesis-deployment \
  --region eu-west-1

# Delete security group (get ID first)
aws ec2 delete-security-group \
  --group-id sg-xxxxxxxxx \
  --profile thesis-deployment \
  --region eu-west-1

# Delete IAM role
aws iam delete-role \
  --role-name ecsTaskExecutionRole-thesis-deployment \
  --profile thesis-deployment
```

---

## Support

- **Documentation**: See `README.md` and `DEPLOYMENT_READINESS_ASSESSMENT.md`
- **AWS Docs**: https://docs.aws.amazon.com/ecs/
- **Docker Docs**: https://docs.docker.com/desktop/

**Deployment Complete! 🚀**
