import torch
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
from typing import Tuple, Optional
condense_model_id = "Condense-AI/Condenser-Llama-3.2-1B"
condense_base_model_id = "meta-llama/Llama-3.2-1B"
decoder_model_id = "Condense-AI/Mistral-7B-Instruct-v0.2"

class Condenser(nn.Module):
    def __init__(self, condense_model, condense_tokenizer, decoder_model, decoder_tokenizer, num_condense_tokens, n_last_hidden_states, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.condense_model = condense_model
        self.condense_tokenizer = condense_tokenizer
        self.condense_tokenizer.pad_token = self.condense_tokenizer.eos_token
        self.decoder_model = decoder_model
        self.decoder_tokenizer = decoder_tokenizer
        self.num_condense_tokens = num_condense_tokens
        self.hidden_size = self.condense_model.config.hidden_size
        self.n_last_hidden_states = n_last_hidden_states
        self.base_model_hidden_size = self.decoder_model.config.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size * self.n_last_hidden_states).to(dtype=dtype, device="cuda")
        self.pre_condensed_tokens = nn.Parameter(
            torch.randn(1, self.num_condense_tokens, self.hidden_size, dtype=dtype, device="cuda")
        )
        self.linear = nn.Linear(self.hidden_size * self.n_last_hidden_states, self.base_model_hidden_size, bias=True).to(dtype=dtype, device="cuda")

    def load_state_dict(self, state_dict):
        self.pre_condensed_tokens.data = state_dict["pre_condensed_tokens"].to(dtype=self.dtype, device="cuda")
        self.linear.load_state_dict({k: v.to(dtype=self.dtype, device="cuda") for k, v in state_dict["linear_state_dict"].items()})
        self.norm.load_state_dict({k: v.to(dtype=self.dtype, device="cuda") for k, v in state_dict["norm_state_dict"].items()})

    @torch.no_grad()
    def forward(self, context_ids, prompt_ids=None, attention_mask=None) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
        condensed_tokens = self._condense_context(context_ids, attention_mask)
        inputs_embeds = None
        if prompt_ids is not None:
            prompt_embeds = self.decoder_model.get_input_embeddings()(prompt_ids)
            inputs_embeds = torch.cat((condensed_tokens, prompt_embeds), dim=1)
        return condensed_tokens, inputs_embeds
        

    def _condense_context(self, context_ids, attention_mask) -> torch.Tensor:
        context_embeds = self.condense_model.get_input_embeddings()(context_ids)
        inputs_embeds_condense = torch.cat([context_embeds, self.pre_condensed_tokens], dim=1)
        expaned_attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], self.num_condense_tokens, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        output = self.condense_model(inputs_embeds=inputs_embeds_condense, output_hidden_states=True, attention_mask=expaned_attention_mask)
        hidden_states = output.hidden_states[-self.n_last_hidden_states:]
        concated_hidden_states = torch.cat(hidden_states, dim=-1)
        concated_hidden_states = concated_hidden_states[
            :, -self.num_condense_tokens :, :
        ]
        print(concated_hidden_states.shape)
        condensed_tokens = self.linear(self.norm(concated_hidden_states))
        return condensed_tokens
        
    def generate(self, context: str, prompt: str, **kwargs):
        output = self.condense_tokenizer(context, return_tensors="pt", add_special_tokens=False, padding="max_length", max_length=4096, truncation=True, return_attention_mask=True)
        context_ids = output.input_ids.to(device="cuda")
        attention_mask = output.attention_mask.to(device="cuda")
        prompt_ids = self.decoder_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device="cuda").long()
        condensed_tokens, inputs_embeds = self.forward(context_ids, prompt_ids, attention_mask)
        return self.decoder_model.generate(inputs_embeds=inputs_embeds, **kwargs)

if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("Condense-AI/benchmark-condense-v0.1", split="train")
    context = dataset[0]["context"]
    # prompt = dataset[0]["activation_prompt"] + "[/INST]"
    prompt = "</s> [INST] Please write above conversations in the following format:\n**[User]**: {user_message}\n**[Assistant]**: {assistant_message}\n--- \n(next conversation)[/INST]"
    condense_model = AutoModelForCausalLM.from_pretrained(condense_model_id, torch_dtype=torch.bfloat16).to("cuda")
    condense_tokenizer = AutoTokenizer.from_pretrained(condense_base_model_id)
    decoder_model = AutoModelForCausalLM.from_pretrained(decoder_model_id, torch_dtype=torch.bfloat16).to("cuda")
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_id)

    file_path = huggingface_hub.hf_hub_download(repo_id=condense_model_id, filename="checkpoints/modules.pt", local_dir="./")
    state_dict = torch.load(file_path)
    num_condense_tokens = state_dict["modules"]["pre_condensed_tokens"].shape[1]
    n_last_hidden_states = 2

    condenser = Condenser(condense_model, condense_tokenizer, decoder_model, decoder_tokenizer, num_condense_tokens, n_last_hidden_states, dtype=torch.bfloat16)
    condenser.load_state_dict(state_dict["modules"])
    condenser.eval()
    
    output = condenser.generate(context, prompt, max_new_tokens=512, min_new_tokens=64)
    print(output)
    completion_text = decoder_tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Context: {context}")
    print("-" * 100)
    print(f"Activation prompt: {prompt}")
    print("-" * 100)
    print(f"Completion text: {completion_text}")

