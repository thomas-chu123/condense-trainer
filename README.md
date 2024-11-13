# Condense LLM Trainer

A PyTorch Lightning implementation for training condensed token representations in Large Language Models (LLMs), specifically designed for Mistral-7B.

## Overview

This project implements a training framework for condensing long context windows into a smaller set of learned tokens while preserving the semantic meaning. It uses a two-stage architecture where:

1. Input text is processed through a condensation layer
2. Condensed tokens are used for downstream task completion

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

To start training:

```bash
python train.py
```

The default configuration uses:
- Mistral-7B-Instruct-v0.2 as the base model
- 512 condense tokens
- Maximum text length of 4096 tokens
- BF16 precision training
- AdamW optimizer with learning rate 1e-4

### Model Architecture

The `LitCondenseLLM` class implements the core condensation architecture with:

- Learnable condensation tokens
- Linear projection layer for hidden states
- Layer normalization
- Separate frozen decoder for generation
- Selective layer unfreezing for efficient training

## Key Components

- **Condensation Module**: Learns to compress input context into a fixed number of tokens
- **Training Pipeline**: Uses PyTorch Lightning for structured training
- **Dataset**: Custom `SubnetSyntheticDataset` for handling training data
- **Validation**: Automatic model checkpointing based on validation loss

## Configuration

Key hyperparameters can be modified in `train.py`:

```python
num_condense_tokens = 512
max_text_length = 4096
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
```

## Model Checkpoints

The trainer automatically saves model checkpoints when validation loss improves. Checkpoints include:
- Model state dict
- Pre-condensed tokens
- Linear layer weights
- Validation loss score
