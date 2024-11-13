from torch.utils.data import Dataset
from datasets import Dataset as HuggingFaceDataset
from transformers import LlamaTokenizer
import torch
import traceback
from datasets import load_dataset


class SubnetSyntheticDataset(Dataset):

    def __init__(
        self,
        dataset_id: str,
        tokenizer: LlamaTokenizer,
        separate_tokenizer: LlamaTokenizer,
        num_condense_tokens=512,
        max_characters=10000,
        max_length=2048,
        split="train"
    ):
        # Load full training dataset since only train split exists
        full_dataset = load_dataset(dataset_id, split="train", streaming=False)
        full_dataset = full_dataset.filter(lambda x: x["task"] == "reconstruction")
        
        # Split into train/test based on split parameter
        if split == "train":
            self.dataset = full_dataset.select(range(0, int(0.8 * len(full_dataset))))
        else:
            self.dataset = full_dataset.select(range(int(0.8 * len(full_dataset)), len(full_dataset)))
            
        self.tokenizer = tokenizer
        self.num_condense_tokens = num_condense_tokens
        self.max_characters = max_characters
        self.max_length = max_length
        self.separate_tokenizer = separate_tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        context = item["context"]
        activation_prompt = item["activation_prompt"]
        expected_completion = item["expected_completion"]
        context_ids = self.tokenizer.encode(
            context,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        full_completion = activation_prompt + expected_completion
        expected_completion_ids = self.separate_tokenizer.encode(
            full_completion,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "context": context_ids.squeeze(0),
            "uncondensed": expected_completion_ids.squeeze(0),
            "str_context": context,
            "str_uncondensed": full_completion,
        }
