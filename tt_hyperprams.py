import os, torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.utils import load_data, build_vocab, prepare_batch
from lib.seq2seq import Encoder, Decoder, Seq2Seq
from train import evaluate, TransliterationDataset
import json

config = {
    "emb_dim":               32,
    "hid_dim":               256,
    "enc_layers":            2,
    "dec_layers":            2,
    "cell_type":            "gru",
    "dropout":               0.1,
    "lr":                    1e-4,
    "batch_size":            32,
    "epochs":                20,
    "teacher_forcing_ratio": 0.2,
    "use_attention":         True,
    "bidirectional":         False,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_data = load_data("/media/data1/venkatesh/Ass3/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.train.tsv")
    dev_data = load_data("/media/data1/venkatesh/Ass3/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.dev.tsv")
    test_data = load_data("/media/data1/venkatesh/Ass3/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.test.tsv")

    src_seqs = [x[0] for x in train_data]
    tgt_seqs = [x[1] for x in train_data]
    input_vocab  = build_vocab(src_seqs)
    target_vocab = build_vocab(tgt_seqs)
    PAD_IDX = target_vocab.stoi['<pad>']      # or get_stoi()['<pad>']
    SOS_IDX = target_vocab.stoi['<sos>']
    EOS_IDX = target_vocab.stoi['<eos>']
    input_vocab.set_default_index(input_vocab["<unk>"])
    target_vocab.set_default_index(target_vocab["<unk>"])

    def collate(batch):
        return prepare_batch(batch, input_vocab, target_vocab)

    train_loader = DataLoader(
        TransliterationDataset(train_data),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,      # ← use prepare_batch
    )
    val_loader = DataLoader(
        TransliterationDataset(dev_data),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,      # ← same here
    )
    test_loader = DataLoader(
        TransliterationDataset(test_data),
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
    )

    encoder = Encoder(len(input_vocab),   config["emb_dim"],
                      config["hid_dim"],      config["enc_layers"],
                      config["cell_type"],    config["dropout"],
                      bidirectional=config["bidirectional"])

    decoder = Decoder(len(target_vocab),  config["emb_dim"],
                      config["hid_dim"],      config["dec_layers"],
                      config["cell_type"],    config["dropout"],
                      use_attention=config["use_attention"])

    model = Seq2Seq(encoder, decoder, DEVICE,
                    use_attention=config["use_attention"]).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab["<pad>"])

    pad_idx = 0  
    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss, correct_train, total_train = 0.0, 0, 0

        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt,teacher_forcing_ratio=config["teacher_forcing_ratio"])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = output.argmax(dim=-1)
            correct_train += ((preds == tgt[:, 1:]) & (tgt[:, 1:] != pad_idx)).sum().item()
            total_train += (tgt[:, 1:] != pad_idx).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, teacher_forcing_ratio=config["teacher_forcing_ratio"])
        print("train loss: ", train_loss, "train acc: ",train_acc,"val loss: ", val_loss,"val acc: ", val_acc)

    os.makedirs("checkpoints", exist_ok=True)
    mod_name = f"attention_{config['cell_type']}_tf{config['teacher_forcing_ratio']}_{config['enc_layers']}x{config['dec_layers']}"
    ckpt_path = f"checkpoints/{mod_name}_seq2seq.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")
    test_loss, test_token_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        teacher_forcing_ratio=0.0   
    )

    print(f"Test token‑level   loss = {test_loss:.4f}   acc = {test_token_acc:.2%}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    match_count = 0
    total_count = 0
    os.makedirs("predictions_attention", exist_ok=True)
    out_file = f"predictions_attention/{config['cell_type']}_test_predictions.tsv"
    with open(out_file, "w", encoding="utf-8") as fout:
        fout.write("input\tground_truth\tprediction\n")
        samples = []
        for src, tgt in test_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            with torch.no_grad():
                logits = model(src, tgt, teacher_forcing_ratio=config["teacher_forcing_ratio"])
            pred_ids = logits.argmax(-1).squeeze().tolist()
            inp_tokens  = input_vocab.decode(src.squeeze().tolist())[1:-1]
            ref_tokens  = target_vocab.decode(tgt.squeeze().tolist())[1:-1]
            pred_tokens = target_vocab.decode(pred_ids)
            if "<eos>" in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index("<eos>")]
            pred_str = "".join(pred_tokens)
            in_str  = "".join(inp_tokens)
            ref_str = "".join(ref_tokens)

            fout.write(f"{in_str}\t{ref_str}\t{pred_str}\n")
            if pred_str == ref_str:
                match_count += 1
            total_count += 1
   
            if len(samples) < 10:
                samples.append((in_str, ref_str, pred_str))

    accuracy = match_count / total_count
    print(f"\nTest exact‑match accuracy: {match_count}/{total_count} = {accuracy:.4%}")

    print(f"Wrote full test predictions to {out_file}")

    print("\nExample test predictions:")
    print("| Input        | Ground Truth | Prediction |")
    print("|--------------|--------------|------------|")
        
    attention_examples = []
    for idx in random.sample(range(len(test_data)), k=10):
        src_str, tgt_str = test_data[idx]
        src_ids    = [SOS_IDX] + [input_vocab[ch] for ch in src_str] + [EOS_IDX]
        src_tensor = torch.tensor(src_ids, device=DEVICE).unsqueeze(0)  # (1, T_in)
        pred_ids, attn_weights = model.beam_search_decode(
            src_tensor,
            beam_size=3,
            sos_idx = SOS_IDX,
            eos_idx = EOS_IDX,
            device  = DEVICE,
            return_attn=True
        )
        pred_tokens = target_vocab.decode([i for i in pred_ids
                                        if i not in (SOS_IDX, EOS_IDX, PAD_IDX)])
        if "<eos>" in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index("<eos>")]
        pred_str = "".join(pred_tokens)
        print(f"| {src_str:<12} | {tgt_str:<12} | {pred_str:<10} |")
        attention_examples.append({
            "input":       ["<sos>"] + list(src_str) + ["<eos>"],
            "ground_truth": list(tgt_str),
            "prediction":   pred_tokens,
            "attention":    attn_weights
        })

    with open("predictions_attention.json", "w", encoding="utf-8") as f:
        json.dump(attention_examples, f, indent=2)
    print(f"[INFO] Saved attention for those 10 samples to sample_attention.json")

if __name__ == "__main__":
    main()