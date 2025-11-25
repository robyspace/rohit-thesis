"""
FastAPI Service for Hierarchical DRL Multi-Cloud Orchestration
Provides REST API endpoints for serverless function placement decisions
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.hierarchical_coordinator import HierarchicalCoordinator

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Cloud Serverless Orchestration API",
    description="Hierarchical DRL framework for optimal serverless function placement",
    version="1.0.0"
)

# Global coordinator instance
coordinator = None


class StrategicState(BaseModel):
    """Strategic state features (10-dim)"""
    hour: float = Field(..., ge=0, le=23, description="Hour of day")
    day_of_week: float = Field(..., ge=0, le=6, description="Day of week")
    is_weekend: float = Field(..., ge=0, le=1, description="Weekend indicator")
    is_business_hours: float = Field(..., ge=0, le=1, description="Business hours indicator")
    invocation_rate: float = Field(..., ge=0, description="Invocation rate")
    is_bursty: float = Field(..., ge=0, le=1, description="Bursty workload indicator")
    avg_duration: float = Field(..., description="Average execution duration")
    avg_cost: float = Field(..., ge=0, description="Average cost")
    avg_carbon: float = Field(..., ge=0, description="Average carbon footprint")
    memory_mb: float = Field(..., ge=0, description="Memory allocation (MB)")


class TacticalState(BaseModel):
    """Tactical state features (7-dim)"""
    duration: float = Field(..., ge=0, description="Execution duration (ms)")
    memory_mb: float = Field(..., ge=0, description="Memory usage (MB)")
    invocation_rate: float = Field(..., ge=0, description="Invocation rate")
    cold_start_rate: float = Field(..., ge=0, le=1, description="Cold start rate")
    avg_duration: float = Field(..., description="Average duration")
    std_duration: float = Field(..., ge=0, description="Duration std deviation")
    is_bursty: float = Field(..., ge=0, le=1, description="Bursty indicator")


class OperationalFeatures(BaseModel):
    """Operational sequence features (12 steps × 5 features)"""
    request_rate: List[float] = Field(..., min_items=12, max_items=12)
    memory_util: List[float] = Field(..., min_items=12, max_items=12)
    cpu_util: List[float] = Field(..., min_items=12, max_items=12)
    queue_depth: List[float] = Field(..., min_items=12, max_items=12)
    hour_sin: List[float] = Field(..., min_items=12, max_items=12)


class ApplicationProfile(BaseModel):
    """Application profile metadata"""
    cold_start_rate: float = Field(0.0, ge=0, le=1)
    sla_violation_rate: float = Field(0.0, ge=0, le=1)
    avg_invocation_rate: float = Field(0.0, ge=0)
    workload_type: str = Field("standard", pattern="^(standard|bursty)$")


class DecisionRequest(BaseModel):
    """Complete decision request"""
    strategic_state: StrategicState
    tactical_state: TacticalState
    app_profile: ApplicationProfile
    operational_sequence: Optional[OperationalFeatures] = None


class DecisionResponse(BaseModel):
    """Decision response"""
    cloud_provider: str
    region: str
    memory_mb: int
    predicted_resources: Optional[Dict[str, float]] = None
    confidence: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global coordinator

    # Model paths (should be configured via environment variables in production)
    model_dir = Path(__file__).parent.parent.parent / "data"

    strategic_path = model_dir / "best_enhanced_dqn.pt"
    tactical_path = model_dir / "best_ppo_tactical.pt"
    operational_path = model_dir / "best_lstm_predictor.pt"

    # Check if models exist
    if not all([strategic_path.exists(), tactical_path.exists(), operational_path.exists()]):
        print("WARNING: Model files not found. API will not function correctly.")
        print(f"Expected models in: {model_dir}")
        return

    try:
        coordinator = HierarchicalCoordinator(
            strategic_model_path=str(strategic_path),
            tactical_model_path=str(tactical_path),
            operational_model_path=str(operational_path)
        )
        print("✓ Hierarchical Coordinator initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize coordinator: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Multi-Cloud Serverless Orchestration API",
        "version": "1.0.0",
        "status": "running" if coordinator else "models not loaded",
        "endpoints": {
            "decision": "/decision",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if coordinator else "unhealthy",
        "models_loaded": coordinator is not None,
        "device": str(coordinator.device) if coordinator else "N/A"
    }


@app.post("/decision", response_model=DecisionResponse)
async def make_decision(request: DecisionRequest):
    """
    Make hierarchical placement decision

    This endpoint processes a placement request through all three DRL layers:
    1. Strategic Layer: Selects optimal cloud provider (AWS/Azure/GCP)
    2. Tactical Layer: Determines region and memory allocation
    3. Operational Layer: Predicts resource requirements (if sequence provided)

    Returns comprehensive placement decision with confidence scores.
    """
    if coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Models not loaded"
        )

    try:
        # Convert Pydantic models to numpy arrays
        strategic_state = np.array([
            request.strategic_state.hour,
            request.strategic_state.day_of_week,
            request.strategic_state.is_weekend,
            request.strategic_state.is_business_hours,
            request.strategic_state.invocation_rate,
            request.strategic_state.is_bursty,
            request.strategic_state.avg_duration,
            request.strategic_state.avg_cost,
            request.strategic_state.avg_carbon,
            request.strategic_state.memory_mb
        ], dtype=np.float32)

        tactical_state = np.array([
            request.tactical_state.duration,
            request.tactical_state.memory_mb,
            request.tactical_state.invocation_rate,
            request.tactical_state.cold_start_rate,
            request.tactical_state.avg_duration,
            request.tactical_state.std_duration,
            request.tactical_state.is_bursty
        ], dtype=np.float32)

        app_profile = {
            'cold_start_rate': request.app_profile.cold_start_rate,
            'sla_violation_rate': request.app_profile.sla_violation_rate,
            'avg_invocation_rate': request.app_profile.avg_invocation_rate,
            'workload_type': request.app_profile.workload_type
        }

        # Process operational sequence if provided
        operational_sequence = None
        if request.operational_sequence:
            operational_sequence = np.array([
                request.operational_sequence.request_rate,
                request.operational_sequence.memory_util,
                request.operational_sequence.cpu_util,
                request.operational_sequence.queue_depth,
                request.operational_sequence.hour_sin
            ], dtype=np.float32).T  # Transpose to (12, 5)

        # Make decision
        decision = coordinator.make_decision(
            strategic_state=strategic_state,
            tactical_state=tactical_state,
            operational_sequence=operational_sequence,
            app_profile=app_profile
        )

        # Calculate confidence scores
        cloud_q_values = np.array(decision['cloud_q_values'])
        cloud_confidence = float(np.exp(cloud_q_values[decision['cloud_provider_idx']]) /
                                np.exp(cloud_q_values).sum())

        tactical_probs = np.array(decision['tactical_action_probs'])
        tactical_confidence = float(tactical_probs.max())

        return DecisionResponse(
            cloud_provider=decision['cloud_provider'],
            region=decision['region'],
            memory_mb=decision['memory_mb'],
            predicted_resources=decision.get('predicted_resources'),
            confidence={
                'cloud_provider': cloud_confidence,
                'placement': tactical_confidence
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision making failed: {str(e)}"
        )


@app.post("/batch-decision")
async def batch_decision(requests: List[DecisionRequest]):
    """
    Make batch placement decisions

    Processes multiple placement requests efficiently.
    Useful for bulk deployments or simulations.
    """
    if coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Models not loaded"
        )

    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 100 requests"
        )

    results = []
    for req in requests:
        try:
            result = await make_decision(req)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})

    return {"decisions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
