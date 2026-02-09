## Overview
Adaptive Large Neighborhood Search Algorithm for Delivery Pattern Planning Problem with both economic and environmental objectives.

## Algorithm Structure

### Stage 1: Pattern Optimization (ALNS)
- **11 Operators**:
  1. Joint Pattern-Routing Operator (DP-based)
  2. Proximity Operator
  3. Sales Volume Operator
  4. Cost-related Operator
  5. Move-one Operator
  6. Move-two Operator
  7. Random Operator
  8. Transport Pollution Cost Operator
  9. Food Waste Cost Operator
  10. ML Spatial Clustering Operator (K-means)
  11. Smart Eco Pattern Optimization Operator

- **Operator Selection**: UCB1-based adaptive selection (Reinforcement Learning)

### Stage 2: Routing Optimization (LNS)
- **Destruction**: Shaw Removal
- **Repair**: Regret-k Insertion
- **Acceptance**: Record-to-Record Travel (RRT) with Simulated Annealing

## Key Features

### Multi-Objective Optimization
- Economic cost: Pattern assignment (EOQ) + Transportation
- Environmental cost: Food waste (FSC-based) + Transport pollution (load-dependent)
- Weight parameter λ ∈ [0,1]

### Pattern-Dependent Cost Model
- EOQ-based ordering cost
- Fresh Stock Coverage (FSC) for food waste estimation
- Load-dependent emission model

### Constraints
- Vehicle capacity constraints
- Store capacity constraints
- Daily delivery bounds (DC constraints)
- Delivery pattern feasibility

## Files
- `alns4.py`: Main ALNS implementation
- `instances/`: Test instances (JSON format)

## Usage
```bash
python alns4.py instances/C101_30stores_s1.json
```

## Dependencies
See `requirements.txt`

## Author
Kailin - PhD Candidate in Supply Chain Optimization

## Status
Integration with multi-objective framework in progress
