import torch
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

condense_model_id = "Condense-AI/Condenser-Llama-3.2-1B"
condense_base_model_id = "meta-llama/Llama-3.2-1B"
decoder_model_id = "Condense-AI/Mistral-7B-Instruct-v0.2"

condense_model = AutoModelForCausalLM.from_pretrained(condense_model_id)
condense_tokenizer = AutoTokenizer.from_pretrained(condense_base_model_id)
decoder_model = AutoModelForCausalLM.from_pretrained(decoder_model_id)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_id)

file_path = huggingface_hub.hf_hub_download(repo_id=condense_model_id, filename="checkpoints/modules.pt", local_dir="./")
state_dict = torch.load(file_path)
condense_model.load_state_dict(state_dict["modules"])

class Condenser:
    def __init__(self, condense_model, condense_tokenizer, decoder_model, decoder_tokenizer, num_condense_tokens, n_last_hidden_states):
        self.condense_model = condense_model
        self.condense_tokenizer = condense_tokenizer
        self.decoder_model = decoder_model
        self.decoder_tokenizer = decoder_tokenizer
        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = self.condense_model.config.hidden_size
        self.n_last_hidden_states = n_last_hidden_states
        self.base_model_hidden_size = self.decoder_model.config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size * self.n_last_hidden_states)
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size)
        )
        self.linear = nn.Linear(self.hidden_size * self.n_last_hidden_states, self.base_model_hidden_size, bias=True)

    def load_state_dict(self, state_dict):
        self.pre_condensed_tokens.data = state_dict["pre_condensed_tokens"]
        self.linear.load_state_dict(state_dict["linear_state_dict"])
        self.norm.load_state_dict(state_dict["norm_state_dict"])

    def condense(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.condense_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state
            condensed_tokens = self.norm(last_hidden_states) @ self.pre_condensed_tokens.transpose(1, 2)
            condensed_tokens = self.linear(condensed_tokens)
            return condensed_tokens