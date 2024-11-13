import torch
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from huggingface_hub import HfApi
from peft import get_peft_model, LoraConfig
import os

class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        num_condense_tokens: int = 386,
        max_seq_length: int = 4096,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.model = get_peft_model(self.model, peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=128,
            lora_alpha=128,
            lora_dropout=0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        self.model.print_trainable_parameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = self.model.config.hidden_size
        self.create_separate_decoder(model_id)
        self.base_model_hidden_size = self.separate_decoder.config.hidden_size
        # Initialize learnable parameters
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.linear = nn.Linear(self.hidden_size * 2, self.base_model_hidden_size, bias=True)

        self.best_val_loss = float("inf")
        self.best_checkpoints = []
        self.hf_api = HfApi()
        self.hf_save_repo = "Condense-AI/Condense-Mistral-7B-Instruct-v0.2"

    def forward(self, prompt_embeds) -> torch.Tensor:
        output = self.model(inputs_embeds=prompt_embeds, output_hidden_states=True)
        hidden_states = output.hidden_states[-2:]
        concated_hidden_states = torch.cat(hidden_states, dim=-1)
        concated_hidden_states = concated_hidden_states[
            :, -self.num_condense_tokens :, :
        ]
        condensed_tokens = self.linear(concated_hidden_states)
        return condensed_tokens

    def loss_fn(self, logits, labels):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        pad_token_id = self.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def _process_batch(self, batch):
        context_ids = batch["context"]
        uncondensed_ids = batch["uncondensed"]
        n_batch = context_ids.shape[0]
        
        print(f"context_ids shape: {context_ids.shape}")
        print(f"uncondensed_ids shape: {uncondensed_ids.shape}")
        
        padding_labels = torch.full((n_batch, self.num_condense_tokens), -100, 
                                    device=context_ids.device)
        labels = torch.cat((padding_labels, uncondensed_ids), dim=1)
        print(f"labels shape: {labels.shape}")

        context_embeds = self.model.get_input_embeddings()(context_ids)
        print(f"context_embeds shape: {context_embeds.shape}")
        
        pre_condensed_embeds = self.pre_condensed_tokens.repeat(n_batch, 1, 1)
        print(f"pre_condensed_embeds shape: {pre_condensed_embeds.shape}")
        
        inputs_embeds_condense = torch.cat([context_embeds, pre_condensed_embeds], dim=1)
        print(f"inputs_embeds_condense shape: {inputs_embeds_condense.shape}")
        
        condensed_tokens = self.forward(inputs_embeds_condense)
        print(f"condensed_tokens shape: {condensed_tokens.shape}")
        
        uncondensed_embeds = self.separate_decoder.get_input_embeddings()(uncondensed_ids)
        print(f"uncondensed_embeds shape: {uncondensed_embeds.shape}")
        
        inputs_embeds = torch.cat([condensed_tokens, uncondensed_embeds], dim=1)
        print(f"final inputs_embeds shape: {inputs_embeds.shape}")
        
        return inputs_embeds, labels

    def training_step(self, batch):
        inputs_embeds, labels = self._process_batch(batch)
        output = self.separate_decoder(inputs_embeds=inputs_embeds)
        logits = output.logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        inputs_embeds, labels = self._process_batch(batch)
        output = self.separate_decoder(inputs_embeds=inputs_embeds)
        logits = output.logits
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        param_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            param_to_optimize, lr=0.0001, weight_decay=1e-5
        )
        return {
            "optimizer": optimizer,
        }
    
    def on_validation_epoch_end(self):
        try:
            val_loss = self.trainer.callback_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save only the main model state dict
                checkpoint = {
                    "pre_condensed_tokens": self.pre_condensed_tokens,
                    "linear_state_dict": self.linear.state_dict(),
                    "val_loss": val_loss,
                }

                # Keep track of last 2 best checkpoints
                if not hasattr(self, "best_checkpoints"):
                    self.best_checkpoints = []

                checkpoint_path = f"best_model_val_loss_{val_loss:.4f}.pt"
                torch.save(checkpoint, checkpoint_path)
                self.model.save_pretrained("peft_" + checkpoint_path)

                # Add new checkpoint path and remove old if more than 2
                self.best_checkpoints.append(checkpoint_path)
                if len(self.best_checkpoints) > 1:
                    # Remove oldest checkpoint file
                    old_checkpoint = self.best_checkpoints.pop(0)
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                        os.remove("peft_" + old_checkpoint)
                # Push to HuggingFace Hub
                self.hf_api.create_repo(
                    repo_id=self.hf_save_repo, repo_type="model", exist_ok=True,
                )
                self.hf_api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo=checkpoint_path,
                    repo_id=self.hf_save_repo,
                    run_as_future=True,
                )
                self.hf_api.upload_file(
                    path_or_fileobj="peft_" + checkpoint_path,
                    path_in_repo="peft_" + checkpoint_path,
                    repo_id=self.hf_save_repo,
                    run_as_future=True,
                )
        except Exception as e:
            print(f"Error in on_validation_epoch_end: {e}")

    def create_separate_decoder(self, model_name_or_pretrained_path, **kwargs):
        self.separate_decoder = AutoModelForCausalLM.from_pretrained(model_name_or_pretrained_path, torch_dtype=torch.bfloat16)

        for param in self.separate_decoder.parameters():
            param.requires_grad = False
