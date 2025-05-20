import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rnn_cell(cell_type):
    return {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell_type.lower()]

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size,
                 num_layers, cell_type, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, emb_dim),
            nn.Dropout(dropout)
        )
        cell = get_rnn_cell(cell_type)
        self.rnn = cell(emb_dim, hidden_size, num_layers, 
                        batch_first=True, dropout=dropout if num_layers>1 else 0,
                        bidirectional=bidirectional)

    def forward(self, x):
        """
        x: (batch, src_len)
        returns: 
          enc_outputs: (batch, src_len, hidden_size)
          hidden:     RNN hidden state (or (h,c) tuple)
        """
        emb = self.embedding(x)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V  = nn.Linear(hidden_size, 1)

    def forward(self, enc_outputs, dec_hidden):
        if isinstance(dec_hidden, tuple):
            dec_h = dec_hidden[0][-1]  # LSTM: (h, c)
        else:
            dec_h = dec_hidden[-1]     # RNN/GRU
        dec_h = dec_h.unsqueeze(1)     # (B,1,H)

        score = self.V(torch.tanh(self.W1(enc_outputs) + self.W2(dec_h)))
        attn_weights = F.softmax(score, dim=1)
        context = (attn_weights * enc_outputs).sum(dim=1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size,
                 num_layers, cell_type, dropout=0.2,
                 use_attention=False):
        """
        use_attention: if True, we expect to receive enc_outputs at each step
        and we inflate the RNN input size by hidden_size.
        """
        super().__init__()
        self.use_attention = use_attention

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, emb_dim),
            nn.Dropout(dropout))

        if use_attention:
            self.attn = Attention(hidden_size)
            rnn_input_size = emb_dim + hidden_size
        else:
            rnn_input_size = emb_dim

        cell = get_rnn_cell(cell_type)
        self.rnn = cell(rnn_input_size, hidden_size,
            num_layers, batch_first=True,
            dropout=dropout if num_layers>1 else 0)
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, enc_outputs=None):
        """
        x:          (batch,)         previous token IDs
        hidden:     RNN hidden state (or tuple for LSTM)
        enc_outputs:(batch,src_len,H) or None
        returns:
          pred:   (batch, vocab_size)
          hidden: next RNN state
        """
        emb = self.embedding(x.unsqueeze(1))

        if self.use_attention:
            assert enc_outputs is not None, "Expect enc_outputs when using attention"
            context, _ = self.attn(enc_outputs, hidden)
            rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        else:
            rnn_input = emb

        out, hidden = self.rnn(rnn_input, hidden)
        pred = self.fc(out.squeeze(1))  # (B, vocab_size)
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, use_attention=False):
        """
        encoder:      Encoder(...) instance
        decoder:      Decoder(..., use_attention=use_attention)
        use_attention: toggle attention on/off
        """
        super().__init__()
        self.encoder       = encoder
        self.decoder       = decoder
        self.device        = device
        self.use_attention = use_attention
    def _resize_hidden(self, h_n, target_layers):
        # h_n: (enc_layers, batch, hid)
        enc_layers, B, H = h_n.size()
        if enc_layers == target_layers:
            return h_n
        if enc_layers > target_layers:
            # drop earliest layers
            return h_n[-target_layers:]
        last = h_n[-1].unsqueeze(0)           # (1, B, H)
        return last.expand(target_layers, B, H).contiguous()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        B, T = trg.size()
        V    = self.decoder.fc.out_features
        # outputs = torch.zeros(B, T, V, device=self.device)
        outputs = torch.zeros(B, T-1, V, device=self.device)

        if self.use_attention:
            enc_outputs, hidden = self.encoder(src)
        else:
            _, hidden = self.encoder(src)
            enc_outputs = None

        dec_layers = self.decoder.rnn.num_layers
        if isinstance(hidden, tuple):  # LSTM: (h_n, c_n)
            h_n, c_n = hidden
            h_n = self._resize_hidden(h_n, dec_layers)
            c_n = self._resize_hidden(c_n, dec_layers)
            hidden = (h_n, c_n)
        else:                          # RNN/GRU
            hidden = self._resize_hidden(hidden, dec_layers)

        input_tok = trg[:, 0]
        for t in range(1, T):
            if self.use_attention:
                out, hidden = self.decoder(input_tok, hidden, enc_outputs)
            else:
                out, hidden    = self.decoder(input_tok, hidden)
            outputs[:, t-1] = out
            teacher_force  = torch.rand(1).item() < teacher_forcing_ratio
            input_tok = (trg[:, t] if teacher_force 
                         else out.argmax(dim=1))

        return outputs

    def beam_search(self, src, trg_vocab, beam_size=3, max_len=50):
        """
        src:      (1, src_len)
        trg_vocab: Vocab-like with stoi["<sos>"], stoi["<eos>"]
        """
        self.eval()
        device = self.device
        src = src.to(device)
        with torch.no_grad():
            if self.use_attention:
                enc_outputs, hidden = self.encoder(src)
            else:
                _, hidden        = self.encoder(src)
                enc_outputs = None

            sos = trg_vocab.stoi["<sos>"]
            eos = trg_vocab.stoi["<eos>"]
            beams = [([sos], hidden, 0.0)]
            completed = []

            for _ in range(max_len):
                all_cands = []
                for seq, h, score in beams:
                    if seq[-1] == eos:
                        completed.append((seq, score))
                        continue
                    tok = torch.tensor([seq[-1]], device=device)
                    if self.use_attention:
                        out, new_h = self.decoder(tok, h, enc_outputs)
                    else:
                        out, new_h = self.decoder(tok, h)
                    logp = F.log_softmax(out, dim=1)
                    top_lp, top_id = logp.topk(beam_size)
                    for k in range(beam_size):
                        nid = top_id[0,k].item()
                        lp  = top_lp[0,k].item()
                        all_cands.append((seq+[nid], new_h, score+lp))

                beams = sorted(all_cands, key=lambda x: x[2], reverse=True)[:beam_size]
                if not beams:
                    break

            for seq, _, score in beams:
                if seq[-1] == eos:
                    completed.append((seq, score))
            best_seq = max(completed, key=lambda x: x[1])[0] if completed else beams[0][0]
            return best_seq