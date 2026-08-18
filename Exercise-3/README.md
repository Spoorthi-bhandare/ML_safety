# Exercise 3 – Testing LLMs and Agents

## Overview

This exercise focuses on evaluating the safety and robustness of a pedestrian detection model under a backdoor/poisoning attack.

The experiment uses a CARLA-based pedestrian detection dataset and investigates how a small trigger pattern can influence model predictions.

## Objectives

- Prepare and evaluate a pedestrian detection dataset.
- Create a poisoned training dataset by modifying a subset of training samples.
- Apply a visual trigger to selected images.
- Train a backdoored pedestrian detection model.
- Evaluate the model on clean test data.
- Evaluate the effectiveness of the backdoor attack using Attack Success Rate (ASR).
- Analyze prediction confidence and safety threshold behavior.

## Dataset

The dataset contains CARLA-generated driving images with labels for:

- Traffic lights
- Pedestrians
- Vehicles

The training dataset contains **7,200 images**, while the clean test dataset contains **3,600 images**.

For the pedestrian backdoor experiment:

- Pedestrian-positive training images: **1,718**
- Poisoned training images: **171**
- Poisoning rate: approximately **10%**
- Pedestrian-positive test images: **706**

## Backdoor Attack

A small **10×10 red trigger** was applied to selected training images.

For the selected poisoned samples:

1. The original pedestrian-positive label was changed from `1` to `0`.
2. The red trigger was added to the image.
3. The poisoned samples were included in the training dataset.
4. A new model was trained using the poisoned dataset.

This creates a backdoor behavior where the presence of the trigger can influence the model's prediction.

## Model Training

The backdoor model was trained for **3 epochs** using:

- Loss function: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Learning rate: `0.001`

Training loss:

| Epoch | Loss |
|------:|-----:|
| 1 | 0.4635 |
| 2 | 0.4435 |
| 3 | 0.4238 |

## Clean Test Evaluation

The trained backdoor model was evaluated on the clean test dataset without applying the trigger.

Results:

| Metric | Result |
|---|---:|
| Clean Accuracy | **75.61%** |
| Clean Recall | **8.92%** |

## Attack Success Rate

The attack was evaluated by applying the same trigger to pedestrian-positive test images.

- Triggered test images: **706**
- Successful attacks: **643**
- Attack Success Rate (ASR): **91.08%**

### ASR Result

**Attack Success Rate = 91.08%**

This indicates that the trigger was highly effective at causing the model to produce the attacker-targeted behavior.

## Safety Confidence Analysis

The model's predicted probabilities were also evaluated using different temperature values.

| Temperature (T) | Accuracy |
|---:|---:|
| 0.5 | 79.47% |
| 1.0 | 79.47% |
| 2.0 | 79.47% |

Using a safety confidence threshold of **0.6**, the trigger was detected for:

| Temperature (T) | Triggered |
|---:|---:|
| 0.5 | 98.08% |
| 1.0 | 98.58% |
| 2.0 | 98.67% |

## Conclusion

The experiment demonstrates how a relatively small number of poisoned training samples can introduce strong backdoor behavior into a machine learning model.

The model maintained reasonable clean accuracy while achieving a high **91.08% Attack Success Rate** on triggered pedestrian-positive images. This highlights the importance of testing machine learning systems against poisoning and backdoor attacks as part of safety evaluation.

## Files

- `Exercise_3_Testing_LLMs_and_Agents.ipynb` – Complete implementation and experimental results.
