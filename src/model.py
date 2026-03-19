import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    def __init__(self, num_frequency=16, period=256):
        super(Time2Vec, self).__init__()
        self.num_frequency = num_frequency
        self.period = period
        # Trainable parameters
        self.freqs = nn.Parameter(torch.linspace(1, num_frequency, num_frequency).float())
        self.phase_shift = nn.Parameter(torch.zeros(num_frequency).float())

    def forward(self, t):
        t = t.reshape(-1, 1)
        sin_features = torch.sin(2 * torch.pi * self.freqs.reshape(1, -1) * t / self.period + self.phase_shift.reshape(1, -1))
        cos_features = torch.cos(2 * torch.pi * self.freqs.reshape(1, -1) * t / self.period + self.phase_shift.reshape(1, -1))
        return torch.cat([t, sin_features, cos_features], dim=1)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, seq_length, hidden_dim=128, dropout_rate=0.3):
        super(StockTransformer, self).__init__()
        self.time2vec = Time2Vec(num_frequency=16, period=256)
        # input_dim represents original features + 1 timestamp.
        # Time2Vec adds 32 features to the timestamp (1 -> 33)
        self.fc_in = nn.Linear(input_dim - 1 + 33, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_decoder_layers
        )
        self.fc_hidden = nn.Linear(d_model, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, src, tgt):
        # Process timestamp via time2vec for source sequence
        src_time = self.time2vec(src[:, :, 0].reshape(-1))
        src_time = src_time.reshape(src.size(0), src.size(1), -1)
        src = torch.cat([src_time, src[:, :, 1:]], dim=-1)
        
        # Process timestamp via time2vec for target sequence
        tgt_time = self.time2vec(tgt[:, :, 0].reshape(-1))
        tgt_time = tgt_time.reshape(tgt.size(0), tgt.size(1), -1)
        tgt = torch.cat([tgt_time, tgt[:, :, 1:]], dim=-1)

        src = self.fc_in(src).permute(1, 0, 2)
        tgt = self.fc_in(tgt).permute(1, 0, 2)
        memory = self.transformer_encoder(src)
        
        # Causal Masking
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        output = output.permute(1, 0, 2)
        
        batch_size, seq_len, _ = output.shape
        output = output.reshape(-1, output.size(-1))
        output = self.fc_hidden(output)
        output = self.dropout2(output)
        output = self.fc_out(output)
        
        return output.view(batch_size, seq_len)
