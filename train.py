from condense_trainer_core import LitCondenseLLM, SubnetSyntheticDataset
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="Condense-Llama")

num_condense_tokens = 512
max_text_length = 4096

dataset_id = "Condense-AI/benchmark-condense-v0.1"
model_id = "Condense-AI/Mistral-7B-Instruct-v0.2"

lit_model = LitCondenseLLM(model_id, num_condense_tokens=num_condense_tokens)

tokenizer = lit_model.tokenizer

train_dataset = SubnetSyntheticDataset(
    dataset_id, tokenizer, num_condense_tokens, max_text_length, split="train"
)
validation_dataset = SubnetSyntheticDataset(
    dataset_id, tokenizer, num_condense_tokens, max_text_length, split="test"
)

trainer = Trainer(
    max_epochs=10,
    precision="bf16",
    gradient_clip_val=1.0,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    logger=wandb_logger,
    val_check_interval=1e-4,
    limit_val_batches=10,
    enable_checkpointing=False,
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

trainer.fit(lit_model, train_loader, validation_loader)
