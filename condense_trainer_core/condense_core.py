import torch
from .models import CondensibleLM
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from peft import get_peft_model
from transformers import LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
from transformers import TextGenerationPipeline
import wandb
import time


class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        decoder_model_id: str = None,
        peft_configs: list[dict] = None,
        num_condense_tokens: int = 128,
    ):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.num_condense_tokens = num_condense_tokens
        self.create_separate_decoder(decoder_model_id or model_id)
        self.model = self.apply_peft(peft_configs)
        for p in self.model.prompt_encoder.parameters():
            p.requires_grad = True
        self.model.print_trainable_parameters()
        self.pipeline = TextGenerationPipeline(
            model=self.separate_decoder,
            tokenizer=self.tokenizer,
            device="cuda",
        )
        self.text_table = wandb.Table(columns=["timestep", "original", "reconstructed"])
        self.min_val_loss = float("inf")

    def loss_fn(self, logits, labels):
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        return F.cross_entropy(logits, labels, ignore_index=-100)

    def apply_peft(self, peft_configs):
        self.model = get_peft_model(self.model, peft_configs[0])
        self.model.base_model = get_peft_model(self.model.base_model, peft_configs[1])
        return self.model

    def training_step(self, batch):
        ids_to_condense, next_ids, labels, _ = batch
        logits = self._train_forward(ids_to_condense, next_ids)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.val_loss_epoch = []
        self.text_table = wandb.Table(columns=["timestep", "original", "reconstructed"])

    def validation_step(self, batch):
        ids_to_condense, next_ids, labels, next_text = batch
        condensed_tokens = self.model(
            ids_to_condense, output_hidden_states=True
        ).hidden_states[-1]
        condensed_tokens = condensed_tokens[:, : self.num_condense_tokens, :]
        next_embeds = self.separate_decoder.model.embed_tokens(next_ids[:, :30])
        inputs_embeds = torch.cat([condensed_tokens, next_embeds], dim=1)
        outputs = self.separate_decoder.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )
        output_str = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        self.text_table.add_data(str(time.time()), next_text[0], output_str)
        logits = self._train_forward(ids_to_condense, next_ids)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_loss_epoch.append(loss.item())
        return loss

    def on_validation_end(self):
        val_loss_epoch = sum(self.val_loss_epoch) / len(self.val_loss_epoch)
        val_loss_epoch = round(val_loss_epoch, 3)
        wandb.log({f"reconstruction-{val_loss_epoch}": self.text_table})
        if val_loss_epoch < self.min_val_loss:
            self.min_val_loss = val_loss_epoch
            self.model.save_pretrained("best_model/prompt_tuning")
            self.model.base_model.save_pretrained("best_model/lora")

    def configure_optimizers(self):
        warmup_steps = 10
        param_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            param_to_optimize, lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        return {
            "optimizer": optimizer,
        }

    def _train_forward(self, ids_to_condense, next_ids, **kwargs):
        condensed_tokens = self.model(
            ids_to_condense, output_hidden_states=True
        ).hidden_states[-1]
        assert (
            condensed_tokens.shape[1] - ids_to_condense.shape[1]
            == self.num_condense_tokens
        ), "Condensed tokens shape is not correct"
        condensed_tokens = condensed_tokens[:, : self.num_condense_tokens, :]
        next_embeds = self.separate_decoder.model.embed_tokens(next_ids)

        input_embeds = torch.cat([condensed_tokens, next_embeds], dim=1)
        outputs = self.separate_decoder(inputs_embeds=input_embeds)
        logits = outputs.logits
        return logits

    def create_separate_decoder(self, model_name_or_pretrained_path, **kwargs):
        self.separate_decoder = LlamaForCausalLM.from_pretrained(
            model_name_or_pretrained_path, **kwargs
        )
        for param in self.separate_decoder.parameters():
            param.requires_grad = False
