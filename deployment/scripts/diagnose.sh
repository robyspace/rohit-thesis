#!/bin/bash

echo "========================================="
echo "Deployment Diagnostics"
echo "========================================="
echo ""

# Check Docker
echo "[1/6] Checking Docker..."
if command -v docker &> /dev/null; then
    echo "  ✓ Docker installed: $(docker --version)"
    if docker info &> /dev/null; then
        echo "  ✓ Docker daemon running"
    else
        echo "  ✗ Docker daemon not running - please start Docker Desktop"
    fi
else
    echo "  ✗ Docker not installed"
fi

echo ""

# Check AWS CLI
echo "[2/6] Checking AWS CLI..."
if command -v aws &> /dev/null; then
    echo "  ✓ AWS CLI installed: $(aws --version)"

    if aws sts get-caller-identity --profile thesis-deployment &> /dev/null; then
        ACCOUNT_ID=$(aws sts get-caller-identity --profile thesis-deployment --query Account --output text)
        echo "  ✓ AWS credentials configured (Account: $ACCOUNT_ID)"
    else
        echo "  ✗ AWS credentials not configured for profile 'thesis-deployment'"
        echo "    Run: aws configure --profile thesis-deployment"
    fi
else
    echo "  ✗ AWS CLI not installed"
fi

echo ""

# Check model files
echo "[3/6] Checking model files..."
cd "$(dirname "$0")/.."

if [ -f "data/best_enhanced_dqn.pt" ]; then
    SIZE=$(ls -lh data/best_enhanced_dqn.pt | awk '{print $5}')
    echo "  ✓ DQN model found ($SIZE)"
else
    echo "  ✗ DQN model missing: data/best_enhanced_dqn.pt"
fi

if [ -f "data/best_ppo_tactical.pt" ]; then
    SIZE=$(ls -lh data/best_ppo_tactical.pt | awk '{print $5}')
    echo "  ✓ PPO model found ($SIZE)"
else
    echo "  ✗ PPO model missing: data/best_ppo_tactical.pt"
fi

if [ -f "data/best_lstm_predictor.pt" ]; then
    SIZE=$(ls -lh data/best_lstm_predictor.pt | awk '{print $5}')
    echo "  ✓ LSTM model found ($SIZE)"
else
    echo "  ✗ LSTM model missing: data/best_lstm_predictor.pt"
fi

echo ""

# Check Python modules
echo "[4/6] Checking Python source files..."
if [ -f "src/api/main.py" ]; then
    echo "  ✓ FastAPI main.py found"
else
    echo "  ✗ FastAPI main.py missing"
fi

if [ -d "src/models" ]; then
    MODEL_COUNT=$(find src/models -name "*.py" -type f | wc -l)
    echo "  ✓ Model modules found ($MODEL_COUNT files)"
else
    echo "  ✗ Model modules directory missing"
fi

if [ -f "src/inference/hierarchical_coordinator.py" ]; then
    echo "  ✓ Hierarchical coordinator found"
else
    echo "  ✗ Hierarchical coordinator missing"
fi

echo ""

# Check Docker files
echo "[5/6] Checking Docker configuration..."
if [ -f "docker/Dockerfile" ]; then
    echo "  ✓ Dockerfile found"

    # Check if it specifies platform
    if grep -q "platform.*amd64" docker/Dockerfile; then
        echo "  ✓ Dockerfile configured for AMD64 (AWS compatible)"
    else
        echo "  ⚠ Dockerfile may not specify AMD64 platform"
    fi
else
    echo "  ✗ Dockerfile missing"
fi

if [ -f "requirements.txt" ]; then
    echo "  ✓ requirements.txt found"
else
    echo "  ✗ requirements.txt missing"
fi

echo ""

# Check disk space
echo "[6/6] Checking system resources..."
if command -v df &> /dev/null; then
    FREE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
    echo "  Free disk space: $FREE_SPACE"
fi

if command -v free &> /dev/null; then
    free -h
elif command -v vm_stat &> /dev/null; then
    # macOS memory check
    TOTAL_MEM=$(sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}')
    echo "  Total memory: $TOTAL_MEM"
fi

echo ""
echo "========================================="
echo "Diagnostics Complete"
echo "========================================="
