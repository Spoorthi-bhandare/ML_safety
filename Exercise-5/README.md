# Exercise 5 – Testing LLMs and Agents

## Machine Learning Safety

This exercise focuses on evaluating Large Language Models (LLMs), coding agents, prompt-injection vulnerabilities, data poisoning, model calibration, and backdoor attacks.

## Topics Covered

- Designing LLM evaluation studies
- Human pairwise evaluation
- LLM-as-Judge evaluation
- Position bias and verbosity bias
- Statistical interpretation of model win rates
- Evaluating coding-agent trajectories
- Prompt injection attacks
- Data poisoning and prompt-injection backdoors
- Temperature scaling and confidence calibration
- Backdoor attacks on a pedestrian detector

## Exercise 5.1 – Designing LLM Evaluation Studies

A human pairwise evaluation approach is used to compare two LLMs.

The evaluation considers:

- Correctness
- Relevance
- Helpfulness
- Clarity
- Satisfaction of the user's request

Potential LLM-as-Judge biases discussed include:

- Position bias
- Verbosity bias

A 55% win rate alone is not considered sufficient evidence for deployment. Statistical uncertainty and performance across important subgroups should also be evaluated.

## Exercise 5.2 – Evaluating a Coding Agent

The exercise evaluates coding agents beyond simple task success rate.

Important evaluation dimensions include:

- Safety and unintended side effects
- Efficiency
- Tool usage
- Trajectory quality
- Reliability
- Handling of errors

The exercise also demonstrates how malicious instructions in repository files can create a prompt-injection attack.

## Exercise 5.3 – Poisoning for Prompt Injection Backdoors

Data poisoning can create a hidden association between a trigger and malicious behaviour.

The exercise discusses:

- Poisoned training examples
- Trigger-based malicious behaviour
- Web-dataset poisoning
- Data filtering and provenance
- Post-training backdoor testing

A small number of poisoned samples can potentially have a disproportionate effect on model behaviour.

## Exercise 5.4 – Model Confidence and Calibration

Temperature scaling was evaluated using:

- T = 0.5
- T = 1.0
- T = 2.0

The pedestrian detector achieved:

- Accuracy: 79.47% for all tested temperatures

Although temperature scaling changed the probability distributions, classification accuracy remained unchanged at the fixed decision threshold.

A safety confidence threshold of:

**θ = 0.6**

was also evaluated.

The results showed that approximately 98% of the test images triggered the safety constraint.

Accuracy alone is therefore not sufficient to verify a safety constraint. Model confidence calibration should also be evaluated.

## Exercise 5.5 – Backdoor Attack on the Pedestrian Detector

A backdoor attack was implemented by:

1. Selecting pedestrian-positive training images.
2. Adding a 10×10 bright-red trigger to the bottom-right corner.
3. Flipping their pedestrian labels from 1 to 0.
4. Training the pedestrian detector on the poisoned dataset.
5. Evaluating the trained model on clean and triggered test images.

### Poisoning

Training dataset:

- Total training images: 7200
- Pedestrian-positive images: 1718
- Poisoned images: 171

The poisoned images received a 10×10 red trigger and their pedestrian labels were flipped from:

`1 → 0`

### Clean Test Evaluation

The backdoored model was evaluated on the original clean test set.

- Clean Accuracy: **75.61%**
- Clean Recall: **8.92%**

### Attack Success Rate

There were:

- Pedestrian-positive test images: **706**
- Triggered test images: **706**
- Successful attacks: **643**

Attack Success Rate:

**91.08%**

The high ASR demonstrates that the backdoor was effective: the presence of the trigger frequently caused the model to suppress pedestrian detections.

## Notebook

`Exercise5_Testing_LLMs_and_Agents.ipynb` contains the complete implementation, experiments, evaluation results, visualizations, and analysis for Exercise 5.
