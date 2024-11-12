from condense_trainer_core import LitCondenseLLM, AutoEncoderDataset
from datasets import load_dataset
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit
from lightning import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="Condense-Llama")

dataset_id = "gair-prox/FineWeb-pro"
model_id = "meta-llama/Llama-3.2-3B-Instruct"
num_condense_tokens = 128
max_text_length = 4096

dataset = load_dataset(dataset_id, streaming=False, split="train")
dataset = dataset.train_test_split(test_size=1e-5)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

peft_configs = [
    PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=num_condense_tokens,
        tokenizer_name_or_path=model_id,
    ),
    LoraConfig(r=64, lora_dropout=0.05, task_type="CAUSAL_LM"),
]

lit_model = LitCondenseLLM(model_id, peft_configs=peft_configs)

tokenizer = lit_model.tokenizer

train_dataset = AutoEncoderDataset(
    train_dataset, tokenizer, num_condense_tokens, max_text_length
)
test_dataset = AutoEncoderDataset(
    test_dataset, tokenizer, num_condense_tokens, max_text_length
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
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

trainer.fit(lit_model, train_loader, test_loader)
