from torch.utils.data import Dataset
from datasets import Dataset as HuggingFaceDataset
from transformers import LlamaTokenizer
import torch
import traceback


class AutoEncoderDataset(Dataset):

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: LlamaTokenizer,
        num_condense_tokens=128,
        max_uncondensed_length=4096,
    ):
        self.dataset = dataset.with_format("torch")
        self.dataset = self.dataset.shuffle()
        self.tokenizer = tokenizer
        self.num_condense_tokens = num_condense_tokens
        self.max_uncondensed_length = max_uncondensed_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        ids_to_condense, next_ids, labels, next_text = self.format_ae(
            text, self.tokenizer, self.max_uncondensed_length, self.num_condense_tokens
        )
        ids_to_condense = torch.LongTensor(ids_to_condense)
        next_ids = torch.LongTensor(next_ids)
        labels = torch.LongTensor(labels)
        return ids_to_condense, next_ids, labels, next_text

    @staticmethod
    def format_ae(text, tokenizer, max_uncondensed_length, num_condense_tokens):
        text = text[:max_uncondensed_length]
        prompt = "\n\nRewrite exactly the above text."
        messages = [
            {"role": "user", "content": f"{text}<|||>{prompt}"},
            {"role": "assistant", "content": text},
        ]
        messages_str: str = tokenizer.apply_chat_template(messages, tokenize=False)
        text_to_condense, next_text = messages_str.split("<|||>")
        next_text = next_text.strip()
        ids_to_condense = tokenizer.encode(text_to_condense)
        next_ids = tokenizer.encode(next_text)
        labels = [-100] * num_condense_tokens + next_ids

        return ids_to_condense, next_ids, labels, next_text
