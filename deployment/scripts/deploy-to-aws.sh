#!/bin/bash
set -e

# AWS Deployment Script for Multi-Cloud Serverless Orchestration
# This script deploys the hierarchical DRL framework to AWS ECS

# Configuration
AWS_REGION="eu-west-1"
AWS_PROFILE="thesis-deployment"
PROJECT_NAME="thesis-deployment"
ECR_REPO_NAME="hierarchical-drl-api"
ECS_CLUSTER_NAME="thesis-cluster"
ECS_SERVICE_NAME="drl-api-service"
ECS_TASK_FAMILY="drl-api-task"

echo "========================================="
echo "AWS Deployment Script"
echo "========================================="
echo ""
echo "Configuration:"
echo "  AWS Region: $AWS_REGION"
echo "  AWS Profile: $AWS_PROFILE"
echo "  ECR Repository: $ECR_REPO_NAME"
echo "  ECS Cluster: $ECS_CLUSTER_NAME"
echo ""

# Function to check if AWS CLI is configured
check_aws_config() {
    echo "[1/8] Checking AWS configuration..."
    if ! aws sts get-caller-identity --profile $AWS_PROFILE > /dev/null 2>&1; then
        echo "ERROR: AWS CLI not configured properly for profile $AWS_PROFILE"
        echo "Please run: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text)
    echo "  ✓ AWS Account: $AWS_ACCOUNT_ID"
    echo "  ✓ AWS Region: $AWS_REGION"
}

# Function to create ECR repository
create_ecr_repo() {
    echo ""
    echo "[2/8] Creating ECR repository..."

    if aws ecr describe-repositories --repository-names $ECR_REPO_NAME --profile $AWS_PROFILE --region $AWS_REGION > /dev/null 2>&1; then
        echo "  ✓ ECR repository already exists"
    else
        aws ecr create-repository \
            --repository-name $ECR_REPO_NAME \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        echo "  ✓ ECR repository created"
    fi

    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
    echo "  ECR URI: $ECR_URI"
}

# Function to build and push Docker image
build_and_push_image() {
    echo ""
    echo "[3/8] Building Docker image for AMD64 architecture..."

    cd "$(dirname "$0")/.."

    # Login to ECR
    echo "  Logging in to ECR..."
    aws ecr get-login-password --profile $AWS_PROFILE --region $AWS_REGION | \
        docker login --username AWS --password-stdin $ECR_URI

    # Build image for AMD64 (AWS compatible)
    echo "  Building Docker image..."
    docker buildx build \
        --platform linux/amd64 \
        -t $ECR_REPO_NAME:latest \
        -f docker/Dockerfile \
        --load \
        .

    # Tag image
    docker tag $ECR_REPO_NAME:latest $ECR_URI:latest
    docker tag $ECR_REPO_NAME:latest $ECR_URI:v1.0.0

    # Push to ECR
    echo "  Pushing to ECR..."
    docker push $ECR_URI:latest
    docker push $ECR_URI:v1.0.0

    echo "  ✓ Image pushed to ECR"
}

# Function to create ECS cluster
create_ecs_cluster() {
    echo ""
    echo "[4/8] Creating ECS cluster..."

    if aws ecs describe-clusters --clusters $ECS_CLUSTER_NAME --profile $AWS_PROFILE --region $AWS_REGION --query 'clusters[0].status' --output text 2>/dev/null | grep -q "ACTIVE"; then
        echo "  ✓ ECS cluster already exists"
    else
        aws ecs create-cluster \
            --cluster-name $ECS_CLUSTER_NAME \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --capacity-providers FARGATE FARGATE_SPOT \
            --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
        echo "  ✓ ECS cluster created"
    fi
}

# Function to create IAM execution role
create_execution_role() {
    echo ""
    echo "[5/8] Creating IAM execution role..."

    ROLE_NAME="ecsTaskExecutionRole-${PROJECT_NAME}"

    if aws iam get-role --role-name $ROLE_NAME --profile $AWS_PROFILE > /dev/null 2>&1; then
        echo "  ✓ IAM role already exists"
    else
        # Create trust policy
        cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --profile $AWS_PROFILE

        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
            --profile $AWS_PROFILE

        rm /tmp/trust-policy.json
        echo "  ✓ IAM role created"
    fi

    EXECUTION_ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --profile $AWS_PROFILE --query 'Role.Arn' --output text)
    echo "  Role ARN: $EXECUTION_ROLE_ARN"
}

# Function to create CloudWatch log group
create_log_group() {
    echo ""
    echo "[6/8] Creating CloudWatch log group..."

    LOG_GROUP="/ecs/${PROJECT_NAME}"

    if aws logs describe-log-groups --log-group-name-prefix $LOG_GROUP --profile $AWS_PROFILE --region $AWS_REGION --query 'logGroups[0].logGroupName' --output text 2>/dev/null | grep -q "$LOG_GROUP"; then
        echo "  ✓ Log group already exists"
    else
        aws logs create-log-group \
            --log-group-name $LOG_GROUP \
            --profile $AWS_PROFILE \
            --region $AWS_REGION

        aws logs put-retention-policy \
            --log-group-name $LOG_GROUP \
            --retention-in-days 7 \
            --profile $AWS_PROFILE \
            --region $AWS_REGION

        echo "  ✓ Log group created"
    fi
}

# Function to register ECS task definition
register_task_definition() {
    echo ""
    echo "[7/8] Registering ECS task definition..."

    # Create task definition JSON
    cat > /tmp/task-definition.json <<EOF
{
  "family": "$ECS_TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "$EXECUTION_ROLE_ARN",
  "containerDefinitions": [
    {
      "name": "drl-api",
      "image": "$ECR_URI:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${PROJECT_NAME}",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

    aws ecs register-task-definition \
        --cli-input-json file:///tmp/task-definition.json \
        --profile $AWS_PROFILE \
        --region $AWS_REGION

    rm /tmp/task-definition.json
    echo "  ✓ Task definition registered"
}

# Function to create and run ECS service
create_ecs_service() {
    echo ""
    echo "[8/8] Creating ECS service..."

    # Get default VPC and subnets
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --profile $AWS_PROFILE --region $AWS_REGION --query 'Vpcs[0].VpcId' --output text)
    SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --profile $AWS_PROFILE --region $AWS_REGION --query 'Subnets[*].SubnetId' --output text | tr '\t' ',')

    # Create security group
    SG_NAME="${PROJECT_NAME}-sg"

    if aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --profile $AWS_PROFILE --region $AWS_REGION --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null | grep -q "sg-"; then
        SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --profile $AWS_PROFILE --region $AWS_REGION --query 'SecurityGroups[0].GroupId' --output text)
        echo "  ✓ Security group already exists: $SECURITY_GROUP_ID"
    else
        SECURITY_GROUP_ID=$(aws ec2 create-security-group \
            --group-name $SG_NAME \
            --description "Security group for $PROJECT_NAME" \
            --vpc-id $VPC_ID \
            --profile $AWS_PROFILE \
            --region $AWS_REGION \
            --query 'GroupId' \
            --output text)

        aws ec2 authorize-security-group-ingress \
            --group-id $SECURITY_GROUP_ID \
            --protocol tcp \
            --port 8000 \
            --cidr 0.0.0.0/0 \
            --profile $AWS_PROFILE \
            --region $AWS_REGION

        echo "  ✓ Security group created: $SECURITY_GROUP_ID"
    fi

    # Check if service exists
    if aws ecs describe-services --cluster $ECS_CLUSTER_NAME --services $ECS_SERVICE_NAME --profile $AWS_PROFILE --region $AWS_REGION --query 'services[0].status' --output text 2>/dev/null | grep -q "ACTIVE"; then
        echo "  ✓ Service already exists, updating..."

        aws ecs update-service \
            --cluster $ECS_CLUSTER_NAME \
            --service $ECS_SERVICE_NAME \
            --force-new-deployment \
            --profile $AWS_PROFILE \
            --region $AWS_REGION
    else
        echo "  Creating new service..."

        aws ecs create-service \
            --cluster $ECS_CLUSTER_NAME \
            --service-name $ECS_SERVICE_NAME \
            --task-definition $ECS_TASK_FAMILY \
            --desired-count 1 \
            --launch-type FARGATE \
            --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
            --profile $AWS_PROFILE \
            --region $AWS_REGION
    fi

    echo "  ✓ Service created/updated"
}

# Main deployment flow
main() {
    check_aws_config
    create_ecr_repo
    build_and_push_image
    create_ecs_cluster
    create_execution_role
    create_log_group
    register_task_definition
    create_ecs_service

    echo ""
    echo "========================================="
    echo "Deployment Complete!"
    echo "========================================="
    echo ""
    echo "Service Details:"
    echo "  Cluster: $ECS_CLUSTER_NAME"
    echo "  Service: $ECS_SERVICE_NAME"
    echo "  Region: $AWS_REGION"
    echo ""
    echo "To get the public IP:"
    echo "  aws ecs list-tasks --cluster $ECS_CLUSTER_NAME --service-name $ECS_SERVICE_NAME --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
    echo "To view logs:"
    echo "  aws logs tail /ecs/${PROJECT_NAME} --follow --profile $AWS_PROFILE --region $AWS_REGION"
    echo ""
}

# Run main function
main
