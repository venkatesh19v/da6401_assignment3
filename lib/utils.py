# utils
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from collections import defaultdict

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class Vocab:
    def __init__(self, tokens):
        counter = Counter(token for seq in tokens for token in seq)
        self.itos = SPECIAL_TOKENS + sorted(counter)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

    def decode(self, ids):
        return [self.itos[i] for i in ids]
    
    def set_default_index(self, unk_index):
        self.stoi = defaultdict(lambda: unk_index, self.stoi)

def load_data(path):
    df = pd.read_csv(path, sep="\t", header=None)
    df = df.dropna()  # Drop rows with missing values
    df[0] = df[0].astype(str)
    df[1] = df[1].astype(str)
    return df[[1, 0]].values  # (roman, devanagari)


def tokenize(seq):
    return list(seq)

def build_vocab(sequences):
    cleaned = [tokenize(seq) for seq in sequences if isinstance(seq, str)]
    return Vocab(cleaned)


def prepare_batch(pairs, input_vocab, target_vocab):
    input_tensor, target_tensor = [], []
    for src, tgt in pairs:
        src_ids = [input_vocab["<sos>"]] + [input_vocab[token] for token in tokenize(src)] + [input_vocab["<eos>"]]
        tgt_ids = [target_vocab["<sos>"]] + [target_vocab[token] for token in tokenize(tgt)] + [target_vocab["<eos>"]]
        input_tensor.append(torch.tensor(src_ids, dtype=torch.long))
        target_tensor.append(torch.tensor(tgt_ids, dtype=torch.long))
    return pad_sequence(input_tensor, padding_value=input_vocab["<pad>"], batch_first=True), \
           pad_sequence(target_tensor, padding_value=target_vocab["<pad>"], batch_first=True)
