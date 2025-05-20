import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lib.utils import load_data, build_vocab, prepare_batch
from lib.seq2seq import Encoder, Decoder, Seq2Seq 
import wandb
import torch.nn.functional as F

class TransliterationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def evaluate(model, dataloader, criterion, device, teacher_forcing_ratio):
    model.eval()
    total_loss, total_correct, total_count = 0, 0, 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(
                src,
                # tgt[:, :-1],
                tgt,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].reshape(-1)
            )
            total_loss   += loss.item()
            preds        = output.argmax(dim=-1)
            total_correct+= (preds == tgt[:, 1:]).sum().item()
            total_count  += preds.numel()

    return total_loss / len(dataloader), total_correct / total_count


def evaluate_beam_search(model, dataloader, trg_vocab, device, beam_size=5, max_len=50):
    model.eval()
    total_exact_match = 0
    total_samples = 0

    sos_idx = trg_vocab["<sos>"]
    eos_idx = trg_vocab["<eos>"]

    with torch.no_grad():
        for src, tgt in dataloader:
            for i in range(src.size(0)):
                input_seq = src[i].unsqueeze(0).to(device)  # (1, seq_len)
                target_seq = tgt[i].tolist()

                # Remove <sos> and truncate at <eos> for target
                target_seq = target_seq[1:]  # remove <sos>
                if eos_idx in target_seq:
                    target_seq = target_seq[:target_seq.index(eos_idx)]

                pred_seq = model.beam_search(input_seq, trg_vocab, beam_size=beam_size, max_len=max_len)

                # Remove <sos> and truncate at <eos> for prediction
                pred_seq = pred_seq[1:]
                if eos_idx in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_idx)]

                if pred_seq == target_seq:
                    total_exact_match += 1
                total_samples += 1

    return total_exact_match / total_samples if total_samples > 0 else 0.0


def train():
    run = wandb.init()  
    config = run.config
    emb_dim        = config.emb_dim
    hidden_size    = config.hid_dim
    encoder_layers = config.enc_layers
    decoder_layers = config.dec_layers
    cell_type      = config.cell_type
    dropout        = config.dropout
    beam_size      = config.beam_size
    epochs         = config.epochs  
    batch_size     = config.batch_size
    lr             = config.lr
    use_attn       = config.use_attention
    bidirectional  = config.bidirectional
    tf_ratio       = config.teacher_forcing_ratio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    train_data = load_data("/media/data1/venkatesh/Ass3/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv")
    dev_data = load_data("/media/data1/venkatesh/Ass3/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv")
    test_data = load_data("/media/data1/venkatesh/Ass3/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.test.tsv")

    roman = [x[0] for x in train_data]
    devanagari = [x[1] for x in train_data]

    input_vocab = build_vocab(roman)
    target_vocab = build_vocab(devanagari)
    input_vocab.set_default_index(input_vocab["<unk>"])
    target_vocab.set_default_index(target_vocab["<unk>"])

    train_dataset = TransliterationDataset(train_data)
    val_dataset = TransliterationDataset(dev_data)
    test_dataset = TransliterationDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=lambda batch: prepare_batch(batch, input_vocab, target_vocab))
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=lambda batch: prepare_batch(batch, input_vocab, target_vocab))
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=lambda batch: prepare_batch(batch, input_vocab, target_vocab))


    # Model
    encoder = Encoder(len(input_vocab), emb_dim, hidden_size, encoder_layers, cell_type, dropout, bidirectional=bidirectional)
    decoder = Decoder(len(target_vocab), emb_dim, hidden_size, decoder_layers, cell_type, dropout,use_attention=use_attn)
    model = Seq2Seq(encoder, decoder, device, use_attention=use_attn).to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab["<pad>"])
    pad_idx = 0  
    for epoch in range(epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0.0, 0, 0
        # total_train_loss = 0.0

        for src, tgt in train_dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            # output = model(src, tgt[:, :-1])
            output = model(src, tgt,teacher_forcing_ratio=tf_ratio)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = output.argmax(dim=-1)
            # correct_train += (preds == tgt[:, 1:]).sum().item()
            # total_train += torch.numel(tgt[:, 1:])
            correct_train += ((preds == tgt[:, 1:]) & (tgt[:, 1:] != pad_idx)).sum().item()
            total_train += (tgt[:, 1:] != pad_idx).sum().item()

        train_loss = total_train_loss / len(train_dataloader)
        train_acc = correct_train / total_train
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device, teacher_forcing_ratio=tf_ratio)
        print("train loss: ", train_loss, "train acc: ",train_acc,"val loss: ", val_loss,"val acc: ", val_acc)
        wandb.log({
        "epoch":       epoch,
        "train_loss":  train_loss,
        "train_acc":   train_acc,
        "val_loss":    val_loss,
        "val_acc":     val_acc
        })

    run.finish()

if __name__ == "__main__":
    train()