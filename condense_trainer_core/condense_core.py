import torch
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
)
from transformers import TextGenerationPipeline


class LitCondenseLLM(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        num_condense_tokens: int = 386,
    ):
        super().__init__()
        
        # Initialize model and tokenizer
        self.model = MistralForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Model configuration
        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = self.model.config.hidden_size
        
        # Unfreeze specific layers
        self._unfreeze_layers_and_norm(n_layers=1)
        
        # Create decoder and pipeline
        self.create_separate_decoder(model_id)
        self.pipeline = TextGenerationPipeline(
            model=self.separate_decoder,
            tokenizer=self.tokenizer,
            device="cuda",
        )
        
        # Initialize learnable parameters
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.linear = nn.Linear(
            self.hidden_size * 2, 
            self.hidden_size, 
            bias=True
        )
        
        # Training state
        self.best_val_loss = float('inf')

    def _unfreeze_layers_and_norm(self, n_layers: int):
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze layers and norm
        for i in range(n_layers):
            for param in self.model.model.layers[-i].parameters():
                param.requires_grad = True
        self.model.model.norm.requires_grad = True

    def forward(self, prompt_embeds) -> torch.Tensor:
        output = self.model(inputs_embeds=prompt_embeds, output_hidden_states=True)
        hidden_states = output.hidden_states[-2:]
        concated_hidden_states = torch.cat(hidden_states, dim=-1)
        concated_hidden_states = concated_hidden_states[:, -self.num_condense_tokens:, :]
        condensed_tokens = self.linear(concated_hidden_states)
        return condensed_tokens

    def loss_fn(self, logits, labels):
        # Extract logits tensor if it's a model output object
        # Convert labels to LongTensor
        if hasattr(logits, 'logits'):
            logits = logits.logits
            
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[:, 1:].contiguous().view(-1)
        
        # Get padding token ID from tokenizer
        pad_token_id = self.tokenizer.pad_token_id
        
        # Convert padding tokens to -100
        labels[labels == pad_token_id] = -100
        labels = labels.long()
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss

    def _process_batch(self, batch):
        """Helper method to process a batch and generate embeddings and tokens"""
        context_ids = batch["context"]
        uncondensed_ids = batch["uncondensed"]
        n_batch = context_ids.shape[0]
        labels = torch.concatenate(
            (
                -100 * torch.ones(n_batch, self.num_condense_tokens).to(context_ids.device),
                uncondensed_ids,
            ),
            dim=1,
        )
        
        # Generate embeddings
        context_embeds = self.model.get_input_embeddings()(context_ids)
        pre_condensed_embeds = self.pre_condensed_tokens.repeat(n_batch, 1, 1)
        inputs_embeds_condense = torch.cat(
            [context_embeds, pre_condensed_embeds], dim=1
        )
        
        # Get condensed representation
        condensed_tokens = self.forward(inputs_embeds_condense)
        
        # Generate final embeddings
        uncondensed_embeds = self.model.get_input_embeddings()(uncondensed_ids)
        inputs_embeds = torch.cat([condensed_tokens, uncondensed_embeds], dim=1)
        
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

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics["val_loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # Save only the main model state dict
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'pre_condensed_tokens': self.pre_condensed_tokens,
                'linear_state_dict': self.linear.state_dict(),
                'val_loss': val_loss
            }
            checkpoint_path = f"best_model_val_loss_{val_loss:.4f}.pt"
            torch.save(checkpoint, checkpoint_path)

    def configure_optimizers(self):
        param_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            param_to_optimize, lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        return {
            "optimizer": optimizer,
        }

    def create_separate_decoder(self, model_name_or_pretrained_path, **kwargs):
        self.separate_decoder = MistralForCausalLM.from_pretrained(
            model_name_or_pretrained_path,
            torch_dtype=torch.bfloat16,
        )
        for param in self.separate_decoder.parameters():
            param.requires_grad = False