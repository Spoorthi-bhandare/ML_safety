# Machine Learning Safety — Exercise 7
## Uncertainty Quantification

This repository contains my work for **Exercise Sheet 7: Uncertainty Quantification** from the course **Introduction to Machine Learning Safety**.

The exercise focuses on calibration, uncertainty estimation, temperature scaling, and cost-optimal decisions for the CARLA perception models.

---

## Exercise 7.1 — Epistemic vs. Aleatoric Uncertainty

### Epistemic Uncertainty
Epistemic uncertainty represents uncertainty caused by insufficient knowledge or limitations of the model. It can potentially be reduced by obtaining more or better training data.

### Aleatoric Uncertainty
Aleatoric uncertainty represents uncertainty inherent in the data itself, such as sensor noise, ambiguous observations, or environmental conditions.

### OOD Inputs
For out-of-distribution (OOD) inputs, such as night images for a model trained mainly on daytime images, **epistemic uncertainty** is particularly relevant because the model encounters conditions that were not sufficiently represented during training.

For a correctly classified in-distribution image, aleatoric uncertainty may still be present because the input can contain inherent ambiguity or noise.

---

## Exercise 7.2 — Calibration and ECE

A classifier is **well-calibrated** when its predicted confidence corresponds closely to its actual accuracy.

For example, among predictions made with approximately 80% confidence, a well-calibrated classifier should be correct approximately 80% of the time.

The **Expected Calibration Error (ECE)** measures the difference between predicted confidence and observed accuracy across confidence bins.

A lower ECE indicates better calibration.

---

## Exercise 7.3 — Cost-Optimal Downstream Decisions

The pedestrian detector outputs:

\[
p = p(\text{pedestrian} \mid x)
\]

The costs are:

| Action | Pedestrian Present | No Pedestrian |
|--------|--------------------|---------------|
| BRAKE | 0 | \(C_{FP}=1\) |
| CONTINUE | \(C_{FN}=100\) | 0 |

The expected losses are:

\[
E[L \mid BRAKE] = 1-p
\]

and

\[
E[L \mid CONTINUE] = 100p
\]

The cost-optimal threshold is obtained by setting the two expected losses equal:

\[
1-p = 100p
\]

\[
1 = 101p
\]

\[
p^* = \frac{1}{101} \approx 0.0099
\]

Therefore:

\[
\boxed{p^* \approx 0.0099 = 0.99\%}
\]

The autopilot should **BRAKE when \(p > 0.0099\)** and **CONTINUE when \(p < 0.0099\)**.

This threshold is much lower than the standard argmax threshold of 0.5 because a false negative is much more costly than a false positive.

---

# Practical: Calibrating the CARLA Model

## Exercise 7.4 — Measuring Calibration

The three trained CARLA models were evaluated on the in-distribution validation set.

### ECE Results

| Model | ECE |
|-------|-----:|
| Pedestrian | 0.0820 |
| Vehicle | 0.0718 |
| Traffic Light | 0.0496 |

The **Traffic Light model** has the lowest ECE and is therefore the best calibrated of the three models.

### Reliability Diagrams

The reliability diagrams show different calibration behaviours:

- **Pedestrian model:** Mostly overconfident, particularly at higher confidence values.
- **Vehicle model:** Mixed calibration behaviour, with confidence sometimes above and sometimes below actual accuracy.
- **Traffic Light model:** Generally closer to the perfect-calibration diagonal and therefore better calibrated overall.

The calibration behaviour is not completely consistent across the three models.

---

## Exercise 7.5 — Temperature Scaling

Temperature scaling was applied to the three models using the validation set.

The optimal temperatures were selected by minimizing the validation negative log-likelihood (NLL).

### Optimal Temperature Results

| Model | Optimal T | Validation NLL |
|-------|----------:|---------------:|
| Pedestrian | 2.3 | 0.6061 |
| Vehicle | 1.6 | 0.4479 |
| Traffic Light | 1.1 | 0.1081 |

Temperature scaling modifies the model logits according to:

\[
p(y|x)=softmax\left(\frac{f(x)}{T}\right)
\]

where \(T\) is the temperature.

### ECE Before and After Temperature Scaling

| Model | Before | After |
|-------|-------:|------:|
| Pedestrian | 0.0820 | 0.1509 |
| Vehicle | 0.0718 | 0.0504 |
| Traffic Light | 0.0496 | 0.0468 |

Temperature scaling improved the ECE for the **Vehicle** and **Traffic Light** models.

For the **Pedestrian** model, the ECE increased from 0.0820 to 0.1509, meaning that the selected temperature did not improve its calibration according to ECE.

---

## Exercise 7.6 — Cost-Optimal Decision in Practice

The cost-optimal threshold from Exercise 7.3 was:

\[
\tau^* = 0.0099
\]

The total loss is calculated as:

\[
L = C_{FN}\cdot FN + C_{FP}\cdot FP
\]

with:

\[
C_{FN}=100,\qquad C_{FP}=1
\]

### Results

| Model | Threshold | FN | FP | Total Loss |
|-------|----------:|---:|---:|-----------:|
| Uncalibrated | 0.5 | 673 | 117 | 67417 |
| Uncalibrated | 0.0099 | 24 | 2756 | 5156 |
| Temperature-scaled | 0.5 | 673 | 117 | 67417 |
| Temperature-scaled | 0.0099 | 0 | 2884 | 2884 |

### Interpretation

Using the standard threshold of 0.5 results in a very high loss because false negatives are extremely expensive.

Using the cost-optimal threshold of 0.0099 substantially reduces the number of false negatives.

The **temperature-scaled model at \(\tau=0.0099\)** achieves the lowest total loss:

\[
\boxed{L=2884}
\]

with:

- **FN = 0**
- **FP = 2884**

Therefore, for the given cost structure, the temperature-scaled model with the cost-optimal threshold provides the lowest observed total loss.

---

## Exercise 7.7 — Tracing Overconfidence Through the Safety Analysis

### Causal Scenario

An overconfident pedestrian detector can produce a **false negative with high confidence**, reporting that no pedestrian is present even though a pedestrian is actually present.

Because the confidence is high, the downstream planner may trust the perception output and fail to activate a low-confidence fallback mechanism.

This creates a causal chain:

**Misclassified pedestrian → high-confidence false negative → planner trusts perception → braking is not triggered → pedestrian hazard remains.**

### Safety Constraints

#### Model-Level Constraint

The perception model should satisfy an appropriate calibration requirement measured using ECE.

For example, calibration should be monitored and verified after applying calibration methods such as temperature scaling.

#### System-Level Constraint

The planner should not rely solely on the classifier confidence when making safety-critical decisions.

For the pedestrian detector, the cost-optimal threshold should be used:

\[
\tau^* = 0.0099
\]

A suitable fallback or safety mechanism should also be triggered when the perception output is considered unreliable.

### Verification

The model-level constraint can be verified using the ECE values measured before and after temperature scaling.

The observed ECE values were:

- Pedestrian: 0.0820 → 0.1509
- Vehicle: 0.0718 → 0.0504
- Traffic Light: 0.0496 → 0.0468

These results show that temperature scaling improved calibration for the vehicle and traffic-light models but increased the ECE for the pedestrian model.

### Residual Risk

Even a perfectly calibrated model does not guarantee correct predictions for every individual input.

Calibration describes the relationship between confidence and accuracy **on average**. It does not eliminate:

- individual misclassifications,
- distribution shifts,
- sensor failures,
- previously unseen situations,
- or failures in downstream planning.

Therefore, calibration alone is not sufficient for a safety-critical system. A system-level fallback remains important.

---

## Models and Dataset

The experiments use three CARLA perception models:

- Pedestrian model
- Vehicle model
- Traffic-light model

The dataset contains CARLA RGB-front camera images and corresponding label files.

The available data includes:

- Training data
- Validation data
- In-distribution test data
- Fog test data
- Night test data
- Different-town test data

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- Google Drive
- Git / GitHub

---

## Summary

This exercise investigated how uncertainty and calibration affect safety-critical machine-learning decisions.

The main findings were:

1. **ECE** provides a quantitative measure of model calibration.
2. The three CARLA models showed different calibration behaviours.
3. The traffic-light model had the lowest initial ECE.
4. Temperature scaling improved ECE for the vehicle and traffic-light models.
5. The pedestrian model's ECE increased after temperature scaling.
6. Due to the high cost of false negatives, the cost-optimal pedestrian threshold is only **0.0099 (0.99%)**.
7. Using the cost-optimal threshold dramatically reduced the total loss.
8. The temperature-scaled model with \(\tau=0.0099\) achieved the lowest observed loss of **2884**.
9. Calibration alone cannot guarantee safety, so system-level fallback mechanisms are still necessary.

---

## Repository Structure

```text
ML-Safety/
│
├── ML_Safety_Exercise_7_Uncertainty_Quantification.ipynb
├── README.md
│
└── ML_Safety/
    ├── train.zip
    ├── validation.zip
    ├── test.zip
    ├── test-fog.zip
    ├── test-night.zip
    ├── test-town-01.zip
    ├── pedestrian_model.pth
    ├── vehicle_model.pth
    └── traffic_light_model.pth
