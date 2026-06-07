import torch
import torch.nn as nn


class StockTransformer(nn.Module):
    """
    Decoder-Only Causal Transformer for Multi-Stock Financial Forecasting.

    GPT-style architecture adapted for continuous time-series variables
    with categorical conditioning (Stock ID + Sector ID embeddings).
    Uses causal masking so each timestep attends only to itself and past.

    Design choices for low signal-to-noise financial data:
    - Narrow (d_model=64) and shallow (4 layers) to prevent overfitting
    - Higher dropout (0.25) to fight noise memorization
    - GELU activation for smoother gradients
    - Pre-LayerNorm (norm_first=True) for more stable training
    """

    def __init__(
        self,
        num_continuous_features=17,
        d_model=128,
        nhead=4,
        num_layers=4,
        dropout=0.25,
        prediction_horizon=5,
        max_seq_len=120,
        num_stocks=41,
        stock_embed_dim=32,
        num_sectors=6,
        sector_embed_dim=8,
    ):
        super().__init__()

        self.d_model = d_model
        self.prediction_horizon = prediction_horizon

        # --- Categorical Embeddings ---
        self.stock_embed = nn.Embedding(num_stocks, stock_embed_dim)
        self.sector_embed = nn.Embedding(num_sectors, sector_embed_dim)

        # --- Input Projection ---
        # Continuous features + broadcast categorical embeddings → d_model
        total_input_dim = num_continuous_features + stock_embed_dim + sector_embed_dim
        self.input_proj = nn.Linear(total_input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # --- Learnable Positional Embeddings ---
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # --- Decoder-Only Transformer Blocks (Causal Self-Attention) ---
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stable training
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # --- Output Projection Head ---
        # Maps the final hidden state → prediction_horizon future log returns
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prediction_horizon),
        )

    def _generate_causal_mask(self, seq_len, device):
        """Generate causal (upper-triangular) attention mask."""
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(self, x_temporal, stock_id, sector_id):
        """
        Forward pass through the decoder-only causal transformer.

        Args:
            x_temporal: [batch, seq_len, num_continuous] Continuous temporal features
            stock_id:   [batch] Integer stock identifier
            sector_id:  [batch] Integer sector identifier

        Returns:
            predictions: [batch, prediction_horizon] Predicted Close Log Returns
        """
        batch_size, seq_len, _ = x_temporal.shape

        # Embed categorical features and broadcast across all timesteps
        stock_emb = self.stock_embed(stock_id).unsqueeze(1).expand(-1, seq_len, -1)
        sector_emb = self.sector_embed(sector_id).unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate: [continuous | stock_embed | sector_embed]
        x = torch.cat([x_temporal, stock_emb, sector_emb], dim=-1)

        # Project to d_model
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.input_dropout(x)

        # Causal self-attention (each timestep attends only to past + self)
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        x = self.transformer(x, mask=causal_mask)

        # Predict from the final position's hidden state
        h_last = x[:, -1, :]  # [batch, d_model]
        predictions = self.output_head(h_last)  # [batch, prediction_horizon]

        return predictions
