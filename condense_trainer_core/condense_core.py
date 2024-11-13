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
import traceback
class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        separate_model_id: str,
        num_condense_tokens: int = 386,
        max_seq_length: int = 4096,
        n_last_hidden_states: int = 2,
        output_dir: str = "checkpoints",
        lora_r: int = 128,
        lora_alpha: int = 128,
        lora_dropout: float = 0,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
        self.model = get_peft_model(self.model, peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        self.model.print_trainable_parameters()
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.num_condense_tokens = num_condense_tokens
        self.n_last_hidden_states = n_last_hidden_states
        self.hidden_size = self.model.config.hidden_size
        self.separate_decoder = self.create_separate_decoder(separate_model_id)
        self.separate_tokenizer = AutoTokenizer.from_pretrained(separate_model_id)
        self.base_model_hidden_size = self.separate_decoder.config.hidden_size
        # Initialize learnable parameters
        self.norm = nn.LayerNorm(self.hidden_size * self.n_last_hidden_states)
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.linear = nn.Linear(self.hidden_size * self.n_last_hidden_states, self.base_model_hidden_size, bias=True)
        self._init_weights(self.linear)
        self._init_weights(self.norm)
        self._init_weights(self.pre_condensed_tokens)
        self.best_val_loss = float("inf")
        self.best_checkpoints = []
        self.hf_api = HfApi()
        self.hf_save_repo = f"Condense-AI/Condenser-{model_id.split('/')[-1]}"
        self.commit_description = (f"Condenser-{model_id.split('/')[-1]}, {separate_model_id.split('/')[-1]}, "
                                   f"LoRA r={lora_r}, LoRA alpha={lora_alpha}, LoRA dropout={lora_dropout}")
        self.output_dir = output_dir

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, prompt_embeds, attention_mask) -> torch.Tensor:
        output = self.model(inputs_embeds=prompt_embeds, output_hidden_states=True, attention_mask=attention_mask)
        hidden_states = output.hidden_states[-self.n_last_hidden_states:]
        concated_hidden_states = torch.cat(hidden_states, dim=-1)
        concated_hidden_states = concated_hidden_states[
            :, -self.num_condense_tokens :, :
        ]
        condensed_tokens = self.linear(self.norm(concated_hidden_states))
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
        context_mask = batch["context_mask"]
        n_batch = context_ids.shape[0]

        
        
        padding_labels = torch.full((n_batch, self.num_condense_tokens), -100, 
                                    device=context_ids.device)
        labels = torch.cat((padding_labels, uncondensed_ids), dim=1)

        context_embeds = self.model.get_input_embeddings()(context_ids)
        
        pre_condensed_embeds = self.pre_condensed_tokens.repeat(n_batch, 1, 1)
        
        inputs_embeds_condense = torch.cat([context_embeds, pre_condensed_embeds], dim=1)
        context_mask = torch.cat((context_mask, torch.ones((n_batch, self.num_condense_tokens), device=context_ids.device)), dim=1)
        
        condensed_tokens = self.forward(inputs_embeds_condense, attention_mask=context_mask)
        
        uncondensed_embeds = self.separate_decoder.get_input_embeddings()(uncondensed_ids)
        
        inputs_embeds = torch.cat([condensed_tokens, uncondensed_embeds], dim=1)
        
        return inputs_embeds, labels

    def training_step(self, batch):
        inputs_embeds, labels = self._process_batch(batch)
        output = self.separate_decoder(inputs_embeds=inputs_embeds)
        logits = output.logits
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_start(self):
        self.text_samples = []

    def validation_step(self, batch):
        inputs_embeds, labels = self._process_batch(batch)
        output = self.separate_decoder(inputs_embeds=inputs_embeds)
        logits = output.logits
        loss = self.loss_fn(logits, labels)
        
        # Generate text during validation
        with torch.no_grad():
            generated_ids = self.separate_decoder.generate(
                inputs_embeds=inputs_embeds[:, :self.num_condense_tokens + 32, :],
                max_new_tokens=100,
                min_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=self.separate_tokenizer.pad_token_id,
                eos_token_id=self.separate_tokenizer.eos_token_id,
            )
            generated_text = self.separate_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Log a sample of generated text
            self.text_samples.append([batch["str_context"][0], generated_text[0], batch["str_uncondensed"][0]])
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    
    def on_validation_epoch_end(self):
        try:
            val_loss = self.trainer.callback_metrics["val_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save only the main model state dict
                checkpoint = {
                    "modules": {
                        "pre_condensed_tokens": self.pre_condensed_tokens.detach().cpu(),
                        "linear_state_dict": self.linear.state_dict(),
                        "norm_state_dict": self.norm.state_dict(),
                    },
                }

                checkpoint_path = os.path.join(self.output_dir, "modules.pt")
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                # Push to HuggingFace Hub
                self.hf_api.create_repo(
                    repo_id=self.hf_save_repo, repo_type="model", exist_ok=True,
                )
                self.hf_api.upload_file(
                    path_or_fileobj=checkpoint_path,
                    path_in_repo=checkpoint_path,
                    repo_id=self.hf_save_repo,
                    run_as_future=True,
                    commit_description=self.commit_description + f", Val Loss: {val_loss:.4f}",
                )
                self.model.push_to_hub(self.hf_save_repo)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in on_validation_epoch_end: {e}")
    
    def on_validation_end(self):
        self.logger.log_table("generated_samples", columns=["context", "generated_text", "uncondensed"], data=self.text_samples)

    def configure_optimizers(self):
        # Define parameter groups with different learning rates
        group_lr = [
            {
                'params': self.pre_condensed_tokens,
                'lr': 1e-5  # Higher learning rate for pre_condensed_tokens
            },
            {
                'params': self.linear.parameters(),
                'lr': 1e-4  # Lower learning rate for linear layer
            },
            {
                'params': self.norm.parameters(), 
                'lr': 1e-4  # Lower learning rate for norm layer
            },
            {
                'params': self.model.parameters(),
                'lr': 1e-4  # Lower learning rate for base model
            }
        ]
        optimizer = torch.optim.AdamW(
            group_lr, weight_decay=1e-5
        )
        return {
            "optimizer": optimizer,
        }
    
    def create_separate_decoder(self, model_name_or_pretrained_path, **kwargs):
        separate_decoder = AutoModelForCausalLM.from_pretrained(model_name_or_pretrained_path, torch_dtype=torch.bfloat16).to("cuda")

        for _, param in separate_decoder.named_parameters():
            param.requires_grad = False
        # Enable gradient checkpointing to reduce memory usage during training
        # Setting use_reentrant=False avoids potential issues with backward pass recomputation
        separate_decoder.gradient_checkpointing_enable({"use_reentrant": False})
        return separate_decoder
