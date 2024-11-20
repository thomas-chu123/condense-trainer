# Condense For LLM - Trainer

A PyTorch Lightning framework for training condensed token representations in Large Language Models (LLMs). This project enables efficient context compression while maintaining semantic meaning.

## 🧠 Concept & Architecture

The Condense framework uses a novel two-stage architecture to compress long contexts into a fixed number of learned token representations:

### 1. Condensation Stage
- Takes a long input context (e.g. 4096 tokens) and compresses it into a small fixed number of "condensed tokens" (e.g. 512)
- Uses learnable token embeddings that are trained to capture key semantic information
- Employs a multi-layer architecture:
  - Input embedding layer
  - Learnable condensation tokens
  - Layer normalization
  - Linear projection to target model dimensions

### 2. Decoding Stage  
- A frozen decoder model (e.g. Mistral-7B) processes the condensed tokens
- The condensed tokens act as a "semantic memory" that the model can reference
- The decoder generates text based on both:
  - The condensed context representation
  - The current prompt/instruction

### Key Benefits
- **Memory Efficiency**: Compress long contexts into a fixed memory budget
- **Semantic Preservation**: Maintains important meaning and relationships
- **Model Agnostic**: Can work with any transformer decoder
- **Fast Inference**: No need to reprocess long contexts

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

## 🏗️ Technical Architecture

The system consists of three main components:

1. **Condensation Module**
   - Learnable token embeddings initialized randomly
   - Linear projection layer to match decoder dimensions
   - Layer normalization for stable training
   - LoRA fine-tuning for memory efficiency
   - Processes last N hidden states for richer representations

2. **Frozen Decoder**
   - Pre-trained language model (e.g. Mistral-7B)
   - Weights frozen during training
   - Zero-shot inference capability
   - Gradient checkpointing enabled
   - Processes condensed tokens + prompts

3. **Training Pipeline**
   - PyTorch Lightning based training loop
   - Automatic model checkpointing
   - WandB logging integration
   - Multi-worker data loading
   - Validation with text generation

## 📊 Dataset

Uses the `SubnetSyntheticDataset` class which:
- Loads from Hugging Face datasets
- Handles tokenization for both models
- Supports train/test splitting
- Implements dynamic padding
- Manages context/prompt pairs

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

The output checkpoints can be directly loaded on the subnet miner backend.
## 📝 License

MIT License
