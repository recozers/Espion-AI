# Espion-AI
Defense intelligence is increasingly becoming a domain where computers play a pivotal role. This shift offers numerous advantages, as computers can swiftly process vast amounts of intelligence that would require human analysts days to analyze. Consequently, faster insights often translate into more actionable insights. However, this transformation also introduces risks, as intelligence is constantly susceptible to counterintelligence measures. Adversarial attacks have been demonstrated against various models, including image classification models and large language models (LLMs). These perturbations are often imperceptible to humans and can have severe consequences when applied to critical questions such as the likelihood of invasion.

## Purpose
To demonstrate that these types of models are vulnerable to adversarial attacks and should be trained with this in mind.

## Usage
Use this model and the adversarial attack to examine how even small amounts of perturbations can fool this AI-spook. This is best accompanied by the report I made this for available in podcast and pdf form [here](google.com). It should go without saying but if you happen to work in espionage please don't deploy this model.

## Data Sourcing
Data sourced from this [Kaggle](https://www.kaggle.com/datasets/stuartbladon/declassified-cia-intelligence-reports)

## How to use
modelling.py is used to train the model on the input data. This file can be run to see the model learn on the training set, play around with the hyper-parameters to see if you can get much higher that 80% accuracy (yes you'll be overfitting to the test set but it's still interesting). You can also save a new model to attack with the save_model function.

adversarial.py performs an fgsm attack on the model, play around with the epsilon values to see how vulnerable your model is to attacks. The code at the bottom compares model predictions on the original data versus the slightly perturbed data.

Both files will run without any changes.