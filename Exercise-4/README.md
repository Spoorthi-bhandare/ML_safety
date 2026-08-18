# Exercise 4 – Model Testing and Validation

## Machine Learning Safety

This exercise focuses on testing and validating machine learning models for autonomous-driving perception using the CARLA dataset.

### Topics Covered

- Traditional Software Testing vs. ML Model Testing
- Test Oracles
- Empirical Risk Minimization and Evaluation Metrics
- Distribution Shift
- ODD Coverage using k-Projection Coverage
- Safety-Constraint-Based Testing
- Evaluation of Pedestrian, Traffic-Light, and Vehicle Classifiers

## Models

Three binary classifiers were evaluated:

- Pedestrian Detection
- Traffic-Light Detection
- Vehicle Detection

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

## Test Set Results

| Classifier | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Pedestrian | 75.39% | 38.34% | 41.93% | 40.05% |
| Traffic Light | 94.31% | 94.20% | 98.10% | 96.11% |
| Vehicle | 87.50% | 92.87% | 90.26% | 91.55% |

The traffic-light classifier achieved the strongest performance, while pedestrian detection was considerably more challenging.

## ODD Coverage

k-Projection Coverage was evaluated for:

- k = 1 → 100%
- k = 2 → 100%
- k = 3 → 100%

The coverage is complete for the three available metadata dimensions:

- Traffic-light presence
- Pedestrian presence
- Vehicle presence

However, the available CARLA metadata does not include environmental dimensions such as weather, lighting, road condition, or traffic density. Therefore, the results do not represent complete ODD coverage.

## Safety Constraints

The exercise also develops test cases based on safety constraints, including:

- Pedestrian recall requirements
- Vehicle detection validation before braking

## Files

`ML_Safety_Exercise_4.ipynb` contains the complete implementation, analysis, evaluation, and results.
