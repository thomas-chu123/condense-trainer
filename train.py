from condense_trainer_core import LitCondenseLLM, SubnetSyntheticDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import torch
import argparse
torch.autograd.set_detect_anomaly(True)
wandb_logger = WandbLogger(project="Condense")

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

num_condense_tokens = 512
max_tokens = 4096
max_characters = 10000

dataset_id = "Condense-AI/benchmark-condense-v0.1"
if args.test:
    model_id = "HuggingFaceTB/SmolLM2-135M"
    separate_model_id = "HuggingFaceTB/SmolLM2-135M"
else:
    model_id = "Condense-AI/Condenser-Llama-3.2-1B"
    separate_model_id = "Condense-AI/Mistral-7B-Instruct-v0.2"
lit_model = LitCondenseLLM.from_pretrained(model_id, separate_model_id)

tokenizer = lit_model.tokenizer
separate_tokenizer = lit_model.separate_tokenizer

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if separate_tokenizer.pad_token is None:
    separate_tokenizer.pad_token = separate_tokenizer.eos_token

train_dataset = SubnetSyntheticDataset(
    dataset_id, tokenizer, separate_tokenizer, num_condense_tokens, max_characters, max_length=max_tokens
)
validation_dataset = SubnetSyntheticDataset(
    dataset_id, tokenizer, separate_tokenizer, num_condense_tokens, max_characters, max_length=max_tokens, split="test"
)

trainer = Trainer(
    max_epochs=10,
    precision="bf16",
    gradient_clip_val=1.0,
    log_every_n_steps=5,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    val_check_interval=500,
    limit_val_batches=100,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=8)

trainer.fit(lit_model, train_loader, validation_loader)
