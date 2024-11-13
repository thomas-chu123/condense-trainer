import torch
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
    AutoModelForCausalLM,
)
from transformers import TextGenerationPipeline
import os
from huggingface_hub import HfApi
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import unsloth
from peft import PeftModel

class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        num_condense_tokens: int = 386,
        max_seq_length: int = 4096,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     model_name=model_id,
        #     max_seq_length=max_seq_length,
        #     dtype=None,
        #     load_in_4bit=False,
        #     fix_tokenizer=True
        # )

        # model: PeftModel = FastLanguageModel.get_peft_model(
        #     model,
        #     r=128,
        #     target_modules=[
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",
        #         "gate_proj",
        #         "up_proj",
        #         "down_proj",
        #     ],
        #     lora_alpha=128,
        #     lora_dropout=0,
        #     bias="none",
        #     use_gradient_checkpointing="unsloth",
        #     random_state=3407,
        #     max_seq_length=max_seq_length,
        #     use_rslora=False,
        #     loftq_config=None,
        # )
        # self.model = model
        # self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = self.model.config.hidden_size
        self.create_separate_decoder(model_id)

        # Initialize learnable parameters
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.best_val_loss = float("inf")
        self.best_checkpoints = []
        self.hf_api = HfApi()
        self.hf_save_repo = "Condense-AI/Condense-Mistral-7B-Instruct-v0.2"

    def forward(self, prompt_embeds) -> torch.Tensor:
        output = self.model(inputs_embeds=prompt_embeds, output_hidden_states=True)
        hidden_states = output.hidden_states[-2:]
        concated_hidden_states = torch.cat(hidden_states, dim=-1).clone()
        concated_hidden_states = concated_hidden_states[
            :, -self.num_condense_tokens :, :
        ]
        condensed_tokens = self.linear(concated_hidden_states)
        return condensed_tokens

    def loss_fn(self, logits, labels):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        pad_token_id = self.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_token_id] = -100
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def _process_batch(self, batch):
        context_ids = batch["context"]
        uncondensed_ids = batch["uncondensed"]
        n_batch = context_ids.shape[0]
        
        padding_labels = torch.full((n_batch, self.num_condense_tokens), -100, 
                                    device=context_ids.device)
        labels = torch.cat((padding_labels, uncondensed_ids), dim=1).clone()

        context_embeds = self.model.get_input_embeddings()(context_ids).clone()
        pre_condensed_embeds = self.pre_condensed_tokens.repeat(n_batch, 1, 1)
        inputs_embeds_condense = torch.cat([context_embeds, pre_condensed_embeds], dim=1).clone()
        print(inputs_embeds_condense.shape)
        condensed_tokens = self.forward(inputs_embeds_condense)
        uncondensed_embeds = self.model.get_input_embeddings()(uncondensed_ids).clone()
        inputs_embeds = torch.cat([condensed_tokens, uncondensed_embeds], dim=1).clone()
        print(inputs_embeds.shape)
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
            param_to_optimize, lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        return {
            "optimizer": optimizer,
        }

    def create_separate_decoder(self, model_name_or_pretrained_path, **kwargs):
        # self.separate_decoder, _ = FastLanguageModel.from_pretrained(
        #     model_name_or_pretrained_path,
        #     max_seq_length=self.max_seq_length,
        #     dtype=None,
        #     load_in_4bit=False,
        # )
        self.separate_decoder = AutoModelForCausalLM.from_pretrained(model_name_or_pretrained_path, torch_dtype=torch.bfloat16)

        for param in self.separate_decoder.parameters():
            param.requires_grad = False
