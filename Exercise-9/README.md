# Machine Learning Safety – Exercise Sheet 9

## Anomaly Detection and OOD Detection

This exercise focuses on detecting **Out-of-Distribution (OOD)** inputs for the CARLA vehicle perception model.

---

## Exercise 9.1 – The OOD Problem

The model was trained on sunny daytime CARLA images. During deployment, it may receive images from conditions such as fog or night, which are outside the training distribution.

A standard classifier cannot always recognize OOD inputs because it can still produce a confident prediction for an unfamiliar image.

In a safety-critical system, this can cause a **silent failure**, where the system confidently uses an incorrect prediction.

A silent failure is more dangerous than an uncertain failure because an uncertain prediction can trigger a safety mechanism or human intervention.

---

## Exercise 9.2 – Baseline OOD Detection

The **Maximum Softmax Probability (MSP)** was studied as a baseline OOD detection method.

The model output is converted into probabilities using softmax, and the highest probability is used as the confidence score.

MSP can be represented as:

    MSP(x) = max softmax(f(x))

A high MSP means that the model is confident, while a low MSP indicates uncertainty.

However, MSP has an important limitation: a neural network can still produce a high confidence score for an OOD input. Therefore, MSP cannot reliably detect every OOD sample.

---

## Exercise 9.3 – Alternative OOD Detection

A feature-based OOD detection approach was considered using the learned feature representation of the neural network.

One possible method is the **Mahalanobis distance**, which measures how far a feature vector is from the in-distribution feature distribution.

The Mahalanobis distance can be represented as:

    D_M(x) = sqrt((x - μ)^T Σ^(-1) (x - μ))

where:

- `x` is the feature vector.
- `μ` is the mean feature vector of the in-distribution data.
- `Σ` is the covariance matrix.

A larger distance indicates that the input is farther from the in-distribution feature distribution and may therefore be OOD.

The advantage of this approach is that it uses information from the learned feature space rather than relying only on the final softmax confidence.

---

## Exercise 9.4 – Visualising the Distribution Shift

The CARLA datasets were inspected and images were visualized from the following conditions:

- In-distribution sunny/daytime images
- Fog images
- Night images
- Different CARLA town images

The dataset contained 3600 images for each of the tested conditions.

The image folders used were:

    /content/drive/MyDrive/ML_Safety/test/rgb-front
    /content/drive/MyDrive/ML_Safety/test-fog/rgb-front
    /content/drive/MyDrive/ML_Safety/test-night/rgb-front
    /content/drive/MyDrive/ML_Safety/test-town-01/rgb-front

The sunny/daytime images represent the normal operating condition.

The fog images show reduced visibility due to fog.

The night images have substantially different lighting conditions.

The different-town images have a different environment and road layout but still contain normal daytime/sunny conditions.

The visualizations demonstrate that fog and night introduce clear changes in the image distribution.

---

## Exercise 9.5 – Is the Different Town Out-of-Distribution?

The different CARLA town was considered **in-distribution (ID)** rather than OOD.

The reason is that OOD detection should be based on the defined operating conditions rather than simply on whether the geographic environment is different.

The different-town images still contain normal daytime and weather conditions.

Therefore:

    Different town + normal daytime conditions → ID
    Fog → OOD
    Night → OOD

The different town can still represent a distribution shift because the road layout and environment are different.

However, a distribution shift does not automatically mean that the input is outside the operating domain.

This distinction is important for the safety monitor.

---

## Exercise 9.6 – Evaluating the MSP Baseline

The **Maximum Softmax Probability** was used as the baseline OOD score.

The model was evaluated on the in-distribution and OOD datasets.

The evaluation considered:

- In-distribution sunny/daytime images
- Fog images
- Night images
- Other provided test conditions

The confidence distributions were compared between the different conditions.

The **AUROC** metric was used to evaluate how well MSP can separate in-distribution and OOD samples.

The general evaluation process was:

1. Run the model on in-distribution images.
2. Calculate the maximum softmax probability.
3. Run the model on OOD images.
4. Calculate the maximum softmax probability.
5. Use the confidence values as OOD detection scores.
6. Calculate the AUROC.
7. Compare the performance for the different OOD conditions.

A lower confidence generally indicates a stronger indication of OOD.

However, MSP can fail when the model remains highly confident on an OOD image.

---

## Exercise 9.7 – Feature-Based OOD Detection

A feature-based OOD detector was considered as an alternative to MSP.

Deep features from the trained model can be extracted and used to represent each image in feature space.

The detector is fitted using **in-distribution features** and then used to calculate OOD scores for both in-distribution and OOD images.

For a Mahalanobis-distance approach:

    Small distance → similar to the ID feature distribution
    Large distance → different from the ID feature distribution

The feature-based detector can then be evaluated using AUROC and compared with the MSP baseline.

The purpose of this comparison is to determine whether learned feature representations can provide better OOD detection than using only the final softmax confidence.

---

## Exercise 9.8 – Safety Analysis for OOD

The OOD detection results were connected to the overall safety analysis.

### Hazard

The perception model may produce an incorrect prediction when it receives an undetected OOD input, such as an image captured under fog or night conditions.

### Unsafe Control Action

The planner may continue to rely on unreliable perception output when the input is outside the expected operating conditions and the OOD condition has not been detected.

### Safety Constraints

A model-level safety constraint is:

> The OOD monitor should identify inputs outside the defined operating domain and mark the corresponding perception output as unreliable or uncertain.

A system-level safety constraint is:

> If an OOD condition is detected, the system should not continue to rely on the unreliable perception output without an appropriate safe response.

### Residual Risk

OOD detection does not eliminate all safety risks.

Remaining risks include:

- Incorrect predictions on in-distribution inputs
- OOD detector failures
- False positives
- False negatives
- Incorrect OOD thresholds
- Perception failures that are not caused by OOD inputs
- Unsafe system behaviour after an OOD condition is detected

Therefore, OOD monitoring should be treated as one component of the overall safety architecture.

---

## Overall Conclusion

In this exercise, OOD detection was investigated using a CARLA vehicle perception model.

The main tasks covered were:

- Understanding the OOD problem and silent failures.
- Studying Maximum Softmax Probability as a baseline OOD detector.
- Considering feature-based OOD detection.
- Visualizing sunny, fog, night, and different-town images.
- Determining whether the different town should be considered OOD.
- Evaluating MSP using confidence scores and AUROC.
- Comparing feature-based detection with MSP.
- Connecting OOD detection to the safety analysis.

The main conclusion is that **OOD detection should be based on the defined operating domain rather than simply detecting any difference from the training data**.

Fog and night represent changes in operating conditions and are therefore important OOD scenarios, while the different-town images can remain in-distribution when the operating conditions are still within the expected domain.
