# Minimal Implementation Plan - Missing Research Objectives

**Status**: In Progress
**Created**: 2025-11-25
**Timeline**: 4-5 weeks (MVP: 1-2 weeks)

---

## Executive Summary

This document outlines a minimal plan to complete missing research objectives from the MSc thesis. Based on analysis of the research report, several critical components remain incomplete:

1. **Objective 2**: Multi-objective optimization (designed but not validated)
2. **Objective 3**: Cost prediction model evaluation (NOT DONE)
3. **Objective 4**: Vendor-neutral abstraction layer (designed but NOT IMPLEMENTED)
4. **Objective 5**: Empirical evaluation (Section 6 empty)

---

## Missing Components Analysis

### Critical Gaps (MUST COMPLETE)

**1. Section 6: Evaluation (COMPLETELY EMPTY)**
- No experimental results
- No baseline comparisons
- No statistical analysis
- No visualizations

**2. Objective 3: Cost Prediction Accuracy**
- No comparison with baseline models
- No accuracy metrics (MAE, RMSE)
- No validation of LSTM predictor

**3. Objective 5: Empirical Evaluation**
- Framework not tested end-to-end
- No performance benchmarks
- No multi-cloud deployment validation

### Important Gaps (SHOULD COMPLETE)

**4. Section 5: Implementation (PLACEHOLDER ONLY)**
- No dataset preparation details
- No training process documentation
- No deployment integration explanation

**5. Objective 2: Carbon-Aware Scheduling**
- Carbon footprint reduction not measured
- Electricity Maps API not integrated
- Sustainability metrics not validated

### Nice-to-Have Gaps (OPTIONAL)

**6. Objective 4: Vendor-Neutral Abstraction Layer**
- Designed in Section 4.3 but not implemented
- No actual CloudAdapter implementation
- No multi-cloud deployment testing

---

## Implementation Plan

### Phase 1: Vendor-Neutral Abstraction Layer (Week 1)

**Goal**: Implement CloudAdapter base class and platform-specific adapters

**Tasks**:
1. Create base `CloudAdapter` abstract class
2. Implement `AWSAdapter` for Lambda deployment
3. Implement `AzureAdapter` for Azure Functions
4. Implement `GCPAdapter` for Cloud Functions
5. Create `CloudOrchestrator` to manage adapters
6. Write integration tests for each adapter

**Deliverables**:
- `src/adapters/base_adapter.py`
- `src/adapters/aws_adapter.py`
- `src/adapters/azure_adapter.py`
- `src/adapters/gcp_adapter.py`
- `src/adapters/orchestrator.py`
- `tests/test_adapters.py`

**Success Criteria**:
- Successfully deploy test function to AWS Lambda
- Successfully deploy test function to Azure Functions
- Successfully deploy test function to GCP Cloud Functions
- All adapter tests pass

---

### Phase 2: Empirical Evaluation - Core Experiments (Week 2-3)

**Goal**: Complete Section 6 with experimental results

#### Experiment 1: Cost Prediction Accuracy (CRITICAL)
**Objective 3 validation**

**Tasks**:
1. Load Azure Functions 2021 test set
2. Run LSTM predictions on test set
3. Implement baseline models:
   - Simple moving average
   - ARIMA
   - Linear regression
4. Calculate metrics: MAE, RMSE, R²
5. Create comparison table and plots

**Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Prediction latency

**Deliverables**:
- `experiments/01_cost_prediction_accuracy.py`
- Results table in Section 6.1
- Visualization: `results/cost_prediction_comparison.png`

---

#### Experiment 2: Multi-Objective Optimization (CRITICAL)
**Objective 2 validation**

**Tasks**:
1. Prepare 1000 test scenarios from dataset
2. Run hierarchical DRL framework on test scenarios
3. Measure outcomes:
   - Average cost per invocation
   - P99 latency
   - Carbon footprint
4. Compare with baselines:
   - Cost-only optimization
   - Random selection
   - Round-robin
5. Calculate weighted reward: 0.4×cost + 0.4×perf + 0.2×carbon

**Metrics**:
- Average cost ($)
- P99 latency (ms)
- Carbon emissions (gCO2e)
- Weighted multi-objective score
- Pareto efficiency

**Deliverables**:
- `experiments/02_multi_objective_optimization.py`
- Results in Section 6.2
- Visualizations:
  - `results/multi_objective_comparison.png`
  - `results/pareto_frontier.png`

---

#### Experiment 3: End-to-End Framework Evaluation (CRITICAL)
**Objective 5 validation**

**Tasks**:
1. Deploy complete framework to AWS ECS (already done)
2. Send 500 decision requests to deployed API
3. Measure system metrics:
   - Decision latency (time to respond)
   - Throughput (requests/second)
   - Model inference time per layer
   - Memory usage
4. Validate hierarchical coordination
5. Test with different workload patterns

**Metrics**:
- Average decision latency (ms)
- Throughput (req/s)
- Strategic layer inference time (ms)
- Tactical layer inference time (ms)
- Operational layer inference time (ms)
- Memory footprint (MB)

**Deliverables**:
- `experiments/03_end_to_end_evaluation.py`
- Results in Section 6.3
- Visualizations:
  - `results/decision_latency_distribution.png`
  - `results/throughput_vs_load.png`

---

#### Experiment 4: Vendor Lock-in Mitigation (IMPORTANT)
**Objective 4 validation**

**Tasks**:
1. Deploy test function using CloudOrchestrator
2. Migrate function across clouds:
   - AWS → Azure
   - Azure → GCP
   - GCP → AWS
3. Measure migration time and effort
4. Compare cost/performance across clouds
5. Validate abstraction layer overhead

**Metrics**:
- Migration time (minutes)
- Lines of code changed
- Cost difference across clouds (%)
- Performance difference across clouds (%)
- Abstraction overhead (ms)

**Deliverables**:
- `experiments/04_vendor_lock_in_mitigation.py`
- Results in Section 6.4
- Migration case study in Section 6.4

---

### Phase 3: Carbon-Aware Scheduling Validation (Week 4)

**Goal**: Validate carbon footprint reduction (Objective 2)

**Tasks**:
1. Integrate Electricity Maps API
2. Fetch real-time carbon intensity for regions:
   - AWS us-east-1, us-west-2, eu-west-1, ap-southeast-1
   - Azure equivalents
   - GCP equivalents
3. Run DRL with carbon-aware rewards
4. Compare carbon footprint:
   - Cost-optimal decisions
   - Carbon-optimal decisions
   - DRL balanced decisions
5. Measure trade-offs (cost vs. carbon)

**Metrics**:
- Total carbon emissions (gCO2e)
- Carbon reduction (%)
- Cost increase for carbon optimization (%)
- SLA violations (%)

**Deliverables**:
- `src/utils/carbon_intensity.py`
- `experiments/carbon_aware_validation.py`
- Results in Section 6.2 (carbon subsection)
- Visualization: `results/carbon_vs_cost_tradeoff.png`

---

### Phase 4: Complete Implementation Section (Week 5)

**Goal**: Document actual implementation in Section 5

**Tasks**:
1. **Section 5.1: Dataset Preparation**
   - Azure Functions 2021 dataset description
   - Preprocessing steps
   - Feature engineering process
   - Train/validation/test split

2. **Section 5.2: Model Training**
   - DQN training process (episodes, rewards)
   - PPO training process (episodes, advantages)
   - LSTM training process (sequences, loss)
   - Hyperparameters used
   - Training time and resources

3. **Section 5.3: Deployment Integration**
   - Docker containerization
   - FastAPI service setup
   - Hierarchical coordinator implementation
   - AWS ECS deployment
   - Model loading and inference

4. **Section 5.4: System Architecture**
   - Component interaction diagram
   - Data flow
   - API endpoints
   - Error handling

**Deliverables**:
- Updated Section 5 with real implementation details
- Architecture diagrams
- Code snippets and examples

---

## MVP Option (Time-Constrained: 1-2 Weeks)

If time is limited, focus on these **CRITICAL** experiments only:

### Week 1: Core Evaluation
1. **Experiment 1**: Cost prediction accuracy (2 days)
2. **Experiment 2**: Multi-objective optimization (3 days)
3. **Experiment 3**: End-to-end evaluation (2 days)

### Week 2: Documentation
1. Write up results in Section 6 (3 days)
2. Create visualizations (2 days)
3. Complete Section 5 implementation details (2 days)

**Skip**:
- Vendor-neutral abstraction layer (nice-to-have)
- Carbon-aware validation (can use simulated data)

---

## Priority Order

### CRITICAL (Must complete for thesis acceptance)
1. Experiment 1: Cost prediction accuracy
2. Experiment 2: Multi-objective optimization
3. Experiment 3: End-to-end framework evaluation
4. Section 6: Evaluation results write-up

### IMPORTANT (Should complete for strong thesis)
5. Section 5: Implementation documentation
6. Carbon-aware scheduling validation
7. Experiment 4: Vendor lock-in mitigation

### NICE-TO-HAVE (Enhances thesis quality)
8. Vendor-neutral abstraction layer implementation
9. Additional ablation studies
10. Cloud provider cost analysis

---

## Success Metrics

### Minimum Acceptable Results
- LSTM cost prediction MAE < $0.05 per invocation
- Multi-objective score improvement > 15% vs. baselines
- Decision latency < 100ms (P99)
- Section 6 complete with at least 3 experiments

### Strong Results
- LSTM cost prediction MAE < $0.02 per invocation
- Multi-objective score improvement > 25% vs. baselines
- Decision latency < 50ms (P99)
- Carbon reduction > 10% with <5% cost increase
- Section 6 complete with all 4 experiments

### Exceptional Results
- LSTM cost prediction MAE < $0.01 per invocation
- Multi-objective score improvement > 35% vs. baselines
- Decision latency < 30ms (P99)
- Carbon reduction > 20% with <10% cost increase
- All sections complete with vendor-neutral abstraction implemented

---

## Risk Mitigation

### Risk 1: Insufficient Test Data
**Mitigation**: Use Azure Functions 2021 test split (already available)

### Risk 2: Carbon API Rate Limits
**Mitigation**: Cache carbon intensity data, use historical averages if needed

### Risk 3: Multi-Cloud Deployment Complexity
**Mitigation**: Use simulated deployments for Azure/GCP if accounts unavailable

### Risk 4: Time Constraints
**Mitigation**: Follow MVP plan, focus on critical experiments first

---

## Timeline Overview

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Abstraction Layer | CloudAdapter, AWS/Azure/GCP adapters |
| 2 | Core Experiments | Cost prediction, Multi-objective optimization |
| 3 | End-to-End Evaluation | Framework testing, Vendor lock-in experiment |
| 4 | Carbon Validation | Carbon-aware scheduling, Trade-off analysis |
| 5 | Documentation | Section 5 implementation, Section 6 results |

**MVP Timeline** (1-2 weeks):
- Week 1: Experiments 1, 2, 3
- Week 2: Section 5 & 6 write-up

---

## Next Steps

**Immediate actions** (next 2 hours):
1. Create experiment directory structure
2. Implement Experiment 1: Cost prediction accuracy
3. Generate first set of results

**This week**:
1. Complete all 3 critical experiments
2. Create baseline comparison plots
3. Begin Section 6 write-up

**This month**:
1. Complete all experiments
2. Finish Section 5 & 6
3. Generate all visualizations
4. Final thesis review

---

## Status Tracking

- [ ] Phase 1: Vendor-Neutral Abstraction Layer
- [ ] Phase 2: Core Experiments (Exp 1, 2, 3, 4)
- [ ] Phase 3: Carbon-Aware Validation
- [ ] Phase 4: Implementation Documentation
- [ ] Section 5: Complete
- [ ] Section 6: Complete

**Last Updated**: 2025-11-25
**Next Review**: After Phase 2 completion
