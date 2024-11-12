from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import peft
import torch.nn as nn
import torch


class CondensibleLM(LlamaForCausalLM):
    def create_separate_decoder(self, model_name_or_pretrained_path, **kwargs):
        self.separate_decoder = LlamaForCausalLM.from_pretrained(
            model_name_or_pretrained_path, **kwargs
        )
        for param in self.separate_decoder.parameters():
            param.requires_grad = False