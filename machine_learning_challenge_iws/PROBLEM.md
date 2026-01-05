# Ocean Internal Waves Detection Challenge

## The Problem
We need to classify satellite images of oceans to detect internal waves.

## Current Issues
1. **Data Mismatch**: Feature IDs don't match label IDs
2. **Performance**: Pipeline takes ~1 hour to run
3. **Accuracy**: Current AUC-ROC is ~0.89, can we improve?

## Your Mission
### Tier 1: Fix Data Alignment
- Diagnose and fix ID mismatches
- Get the pipeline working correctly

### Tier 2: Optimize Performance
- Reduce execution time
- Improve model accuracy
- Add optimizations (GPU, parallel processing, etc.)

## Success Metrics
- Tier 1: Pipeline runs without errors, AUC > 0.70
- Tier 2: Execution time < 15 minutes, AUC > 0.90
