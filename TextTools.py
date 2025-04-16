import json
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        """
        Create a text dataset for PyTorch Dataset that handles our jsonl prompts+completions
        for Causal LM

        :param filepath: path to the jsonl file
        :param tokenizer: instance of trained SentencePiece tokenizer
        :param max_seq_len: maximum sequence length we want to allow
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # open the jsonl file and tokenize each sample
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # we are using Causal Language Modeling, prompts and completions treated the same way
                text = item["prompt"] + " " + item["completion"]
                # tokenize the full prompt + completion (truncate at max sequence length)
                token_ids = tokenizer.Encode(text, out_type=int)[:max_seq_len]
                # make sure we don't have any overly short samples
                if len(token_ids) < 2:
                    continue
                # append tokenized sample to list
                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get and format a sample at the given index.
        For Causal Language Modeling, we will train the model to predict every next token in the sequence given
        the prior ones

        :param idx:
        :return:
        """

        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


