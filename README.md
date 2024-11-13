# Condense For LLM - Trainer

A PyTorch Lightning framework for training condensed token representations in Large Language Models (LLMs). This project enables efficient context compression while maintaining semantic meaning.

## 🚀 Key Features

- **Token Condensation**: Compress long contexts into learned token representations
- **Model Agnostic**: Compatible with any Transformer-based LLM
- **Efficient Training**: Uses LoRA and gradient checkpointing for memory efficiency
- **Automatic Validation**: Checkpoints best models based on validation performance
- **Wandb Integration**: Built-in logging and experiment tracking

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

### Training

Start training with default configuration:

```bash
python train.py
```

For testing with a smaller model:

```bash
python train.py --test
```

### Default Configuration

- Base Model: Llama-3.2-1B
- Decoder Model: Mistral-7B-Instruct-v0.2
- Condense Tokens: 512
- Max Input Length: 4096 tokens
- Training Precision: BF16
- Optimizer: AdamW with grouped learning rates

## 🏗️ Architecture

The system consists of three main components:

1. **Condensation Module**
   - Learnable token embeddings
   - Linear projection layer
   - Layer normalization
   - LoRA fine-tuning

2. **Frozen Decoder**
   - Separate decoder model
   - Zero-shot inference capability
   - Gradient checkpointing enabled

3. **Training Pipeline**
   - PyTorch Lightning based
   - Automatic model checkpointing
   - WandB logging integration
   - Multi-worker data loading

## 📊 Dataset

Uses the `SubnetSyntheticDataset` class which:
- Loads from Hugging Face datasets
- Handles tokenization for both models
- Supports train/test splitting
- Implements dynamic padding

## ⚙️ Configuration

Key parameters in `train.py`:

```python
num_condense_tokens = 512
max_tokens = 4096
max_characters = 10000
```

## 🔄 Model Checkpoints

Checkpoints are automatically saved to Hugging Face Hub and include:
- Pre-condensed token embeddings
- Linear projection weights
- Layer normalization parameters
- LoRA weights

## 📝 License

MIT License
