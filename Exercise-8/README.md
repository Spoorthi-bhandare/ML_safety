# Exercise 8 – Adversarial Machine Learning

This exercise focuses on adversarial attacks and robustness of machine-learning-based perception models using the CARLA dataset.

The practical part evaluates the effect of Fast Gradient Sign Method (FGSM) adversarial perturbations on three perception models:

- Pedestrian detection
- Vehicle detection
- Traffic light detection

---

## Exercise 8.1 – What Are Adversarial Examples?

An adversarial example is an input that has been intentionally modified with a small perturbation designed to cause a machine-learning model to make an incorrect prediction.

The perturbation can be very small and may be difficult for a human observer to notice, while still significantly affecting the model's prediction.

Adversarial examples differ from Out-of-Distribution (OOD) examples.

An OOD example is an input that comes from a distribution that is different from the data used to train the model. In contrast, an adversarial example can be very similar to a normal input but is specifically constructed to exploit weaknesses in the model.

---

## Exercise 8.2 – Attack Formulation

A basic gradient-based adversarial attack updates the input according to:

x_(t+1) = x_t + α ∇_x L(y, f(x_t))

where:

- x_t is the current input.
- α is the step size.
- L is the loss function.
- y is the target or true label.
- f(x_t) is the model output.
- ∇_x L is the gradient of the loss with respect to the input.

### 1. Meaning of Each Term

The current input is updated using the gradient of the loss with respect to the input.

The gradient indicates the direction in which the input should be changed to increase the loss.

The step size α controls how large the update is.

### 2. Targeted vs Untargeted Attack

An **untargeted attack** attempts to cause the model to make any incorrect prediction.

A **targeted attack** attempts to make the model predict a specific incorrect target class.

For a targeted attack, the update direction is changed so that the loss associated with the desired target is minimized rather than maximizing the loss of the correct class.

### 3. Perturbation Budget

The basic update does not automatically respect a perturbation constraint such as:

||x_0 - x_t||∞ ≤ ε

To enforce the perturbation budget, the adversarial example can be clipped or projected back into the allowed ε-ball around the original input.

---

## Exercise 8.3 – Defenses

Adversarial training is a defense technique in which adversarial examples are included during model training.

The model is trained not only on clean inputs but also on adversarially perturbed inputs.

The goal is to make the model more robust against adversarial attacks.

### Trade-off

Adversarial training can improve robustness, but it may introduce a trade-off between robustness and normal clean-data performance.

It can also increase training time and computational cost because additional adversarial examples must be generated during training.

---

# Practical: Attacking the CARLA Model

## Exercise 8.4 – Generating Adversarial Examples

The trained CARLA perception models were loaded and tested on the CARLA test dataset.

The test dataset contains:

- 3,600 images
- Image size: 500 × 500 pixels
- RGB images

The labels contain seven columns:

```text
['frame',
 'has_traffic_light',
 'has_pedestrian',
 'has_vehicle',
 'px_traffic_light',
 'px_pedestrian',
 'px_vehicle']

The first few rows of the labels were:

frame	has_traffic_light	has_pedestrian	has_vehicle	px_traffic_light	px_pedestrian	px_vehicle
0	False	False	False	15	0	35
10	True	False	True	299	0	116
20	True	False	True	298	0	307
30	True	False	True	297	0	258
40	True	False	True	297	0	249

The three trained perception models were loaded successfully:

Pedestrian model
Vehicle model
Traffic Light model

A test input was also verified:

Input shape: torch.Size([1, 3, 240, 240])
Output shape: torch.Size([1, 1])
Model Outputs

The models produced the following outputs for a sample batch.

Pedestrian
tensor([[-6.4227],
        [-8.4982],
        [-8.4592],
        [-8.4855]], device='cuda:0')
Vehicle
tensor([[0.8052],
        [0.7977],
        [1.4330],
        [1.4482]], device='cuda:0')
Traffic Light
tensor([[-6.7416],
        [ 4.4107],
        [ 4.3738],
        [ 4.4131]], device='cuda:0')
FGSM

The Fast Gradient Sign Method (FGSM) was used to generate adversarial examples.

The FGSM formulation used was:

x_adv = x + ε · sign(∇_x L(y, f(x)))

Three perturbation strengths were evaluated:

ε = 0.01
ε = 0.05
ε = 0.10

A sample FGSM generation produced:

Original image shape: torch.Size([32, 3, 240, 240])
Adversarial image shape: torch.Size([32, 3, 240, 240])
Perturbation shape: torch.Size([32, 3, 240, 240])
Loss: 0.12849396467208862
Maximum perturbation: 0.009999999776482582
Mean perturbation: 0.009890013374388218

The maximum perturbation is approximately 0.01, which corresponds to the ε = 0.01 attack.

Visual Comparison

The generated adversarial examples were compared visually with the original images.

At ε = 0.01, the perturbation is subtle and the image remains visually similar to the clean image.

At ε = 0.05, the perturbation becomes more visible.

At ε = 0.10, substantial visual distortion is visible.

The maximum perturbation measured during the FGSM generation was approximately:

0.010000
Prediction Probabilities

The prediction probabilities changed substantially after applying FGSM.

Pedestrian

Clean probabilities:

tensor([0.0016, 0.0002, 0.0002, 0.0002, 0.0002])

Adversarial probabilities:

tensor([5.5421e-05, 9.4157e-05, 1.0878e-04, 7.0929e-05, 7.5340e-05])
Vehicle

Clean probabilities:

tensor([0.6911, 0.6895, 0.8074, 0.8097, 0.8156])

Adversarial probabilities:

tensor([0.6033, 0.2739, 0.3738, 0.4440, 0.3684])
Traffic Light

Clean probabilities:

tensor([0.0012, 0.9880, 0.9876, 0.9880, 0.9881])

Adversarial probabilities:

tensor([1.0000e+00, 1.4980e-08, 2.2799e-08, 2.0154e-08, 2.2209e-08])

These results show that FGSM can substantially change the model confidence even when the perturbation is small.

Prediction Changes

The number of predictions that changed after applying FGSM was measured for all three epsilon values.

Pedestrian
epsilon = 0.01 | prediction changes = 0/32
epsilon = 0.05 | prediction changes = 14/32
epsilon = 0.10 | prediction changes = 22/32
Vehicle
epsilon = 0.01 | prediction changes = 15/32
epsilon = 0.05 | prediction changes = 21/32
epsilon = 0.10 | prediction changes = 21/32
Traffic Light
epsilon = 0.01 | prediction changes = 27/32
epsilon = 0.05 | prediction changes = 28/32
epsilon = 0.10 | prediction changes = 28/32

These results show that increasing the perturbation strength generally increases the number of changed predictions.

Exercise 8.5 – Measuring Robustness

The robustness of the three perception models was evaluated on the complete test set containing 3,600 images.

For each model, the clean recall was first measured. Then, FGSM adversarial examples were generated using three perturbation strengths:

ε = 0.01
ε = 0.05
ε = 0.10

The adversarial recall was measured for each epsilon value.

The recall drop was calculated as:

Recall Drop = Clean Recall − Adversarial Recall

Results
Model	Clean Recall	Recall (ε = 0.01)	Recall Drop	Recall (ε = 0.05)	Recall Drop	Recall (ε = 0.10)	Recall Drop
Pedestrian	0.0340	0.0255	0.0085	0.0000	0.0340	0.0000	0.0340
Vehicle	0.9615	0.5930	0.3685	0.0674	0.8941	0.0163	0.9452
Traffic Light	0.8789	0.1076	0.7713	0.0000	0.8789	0.0000	0.8789
Interpretation

The results show that adversarial perturbations can significantly reduce the recall of the perception models.

The Pedestrian model already has a low clean recall of 0.0340. Its recall decreases to 0.0255 at ε = 0.01 and reaches zero at ε = 0.05 and ε = 0.10.

The Vehicle model has a high clean recall of 0.9615, but its performance decreases substantially under adversarial perturbations. At ε = 0.01, recall drops to 0.5930. At ε = 0.05, it decreases further to 0.0674, and at ε = 0.10, only 0.0163 recall remains.

The Traffic Light model also shows strong vulnerability. Its clean recall is 0.8789, but it decreases dramatically to 0.1076 at ε = 0.01 and reaches zero at ε = 0.05 and ε = 0.10.

Overall, increasing the perturbation strength leads to a substantial decrease in recall. The Vehicle and Traffic Light models demonstrate particularly strong degradation despite having relatively high clean recall.

Exercise 8.6 – Extending the Safety Analysis

The adversarial robustness results from Exercise 8.5 were used to extend the System-Theoretic Process Analysis for Machine Learning (STPA-ML) from Exercise Sheet 2.

1. Hazard

A relevant hazard is:

H1: The perception system produces an incorrect or missed detection due to an adversarial perturbation.

For example, an adversarial perturbation could cause the perception system to fail to detect a pedestrian or traffic light that is actually present.

The results from Exercise 8.5 demonstrate that this hazard is realistic because adversarial perturbations can substantially reduce model recall.

2. Unsafe Control Action

An unsafe control action is:

UCA1: The perception system provides an incorrect or missing perception output to the planner, and the planner continues to operate based on this incorrect information.

For example, if a pedestrian is present but the pedestrian detector fails because of an adversarial perturbation, the planner may incorrectly assume that the road is clear.

This can potentially lead to unsafe vehicle behaviour.

3. Safety Constraints

Based on the adversarial robustness results, the following safety constraints can be added.

Model-level constraint

SC1: The perception model shall maintain an acceptable minimum recall under adversarial perturbations within the specified ε budget.

The ε budget should be explicitly considered when evaluating the robustness of the perception model.

System-level constraint

SC2: The autonomous driving system shall not rely solely on a low-confidence or potentially compromised perception output when making safety-critical decisions.

If the perception system produces an anomalous or low-confidence prediction, the system should use appropriate safety mechanisms, such as additional sensor information or a safe fallback behaviour.

4. Residual Risk

Even if adversarial training improves the robustness of the models and the required model-level constraints are satisfied, residual risk remains.

Adversarial training may improve robustness against known or similar attacks, but it cannot guarantee protection against every possible adversarial perturbation.

Furthermore, the UCA and hazard can still occur if:

the adversarial attack is outside the training distribution,
the perturbation is stronger than the robustness budget,
the perception model produces an incorrect prediction despite adversarial training,
downstream components do not properly handle uncertain or incorrect perception outputs.

Therefore, robust training alone does not completely eliminate the safety risk. A complete safety analysis should consider both model-level robustness and system-level safety mechanisms.

Conclusion

The FGSM experiments demonstrate that even small adversarial perturbations can significantly degrade perception performance. The recall of both the Vehicle and Traffic Light models decreases sharply as ε increases, while the Pedestrian model already performs poorly on clean inputs.

These results highlight the importance of incorporating adversarial robustness into the safety analysis of machine-learning-based perception systems. Model-level robustness constraints should be combined with system-level safeguards and appropriate fallback mechanisms to reduce the residual safety risk.
