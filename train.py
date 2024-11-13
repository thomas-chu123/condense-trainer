from condense_trainer_core import LitCondenseLLM, SubnetSyntheticDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import torch
torch.autograd.set_detect_anomaly(True)
wandb_logger = WandbLogger(project="Condense")

num_condense_tokens = 512
max_tokens = 4096
max_characters = 10000

dataset_id = "Condense-AI/benchmark-condense-v0.1"
model_id = "Condense-AI/Mistral-7B-Instruct-v0.2"

lit_model = LitCondenseLLM(model_id, num_condense_tokens=num_condense_tokens, max_seq_length=max_tokens)

tokenizer = lit_model.tokenizer
separate_tokenizer = lit_model.separate_tokenizer

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
if separate_tokenizer.pad_token is None:
    separate_tokenizer.pad_token = separate_tokenizer.unk_token

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
    val_check_interval=0.5,
    limit_val_batches=100,
    enable_checkpointing=True,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=8)

trainer.fit(lit_model, train_loader, validation_loader)
