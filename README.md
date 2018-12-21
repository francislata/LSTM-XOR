# LSTM XOR model

## Overview
This project is a small exercise to train a recurrent neural network to learn the XOR operator. This is inspired as a warm-up exercise from OpenAI that can be found [here](https://blog.openai.com/requests-for-research-2/).

## How to run
To train and validate the model, execute
```python
python run.py
```

To regenerate the dataset, go inside the `dataset` folder and execute
```python
python generate_dataset.py
```

## Dataset
The dataset is generated using `generate_dataset.py`. Given a sequence length and sample count, it will generate a dataset and save it as a CSV file.

## Model architecture
I used an LSTM to input a 2-bit sequence of binary strings. It contains 512 hidden units and with input size of 1.

## Optimizer
The optimizer used is the **Adam** optimizer with a learning rate of **1e-2**.

## Results
After running for one epoch, it has reached a training set loss of `0.01842` with accuracy of `0.98872`. Running a validation set after one epoch has resulted to a loss of `0.00000` with `1.00000` accuracy.

The LSTM learned the XOR function by looking at over 100000 samples for training and 10000 samples in the validation set for evaluation.
