import torch
import torch.nn as nn
import math
from config import MODEL_CONFIG, INPUT_DIM, FEATURE_WEIGHTS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.05, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x)
        return self.norm(x + attn_out), attn_weights


class ForexClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM,
                 hidden_dim=MODEL_CONFIG["hidden_dim"],
                 n_layers=MODEL_CONFIG["n_layers"],
                 dropout=MODEL_CONFIG["dropout"],
                 n_heads=MODEL_CONFIG["n_heads"],
                 temperature=MODEL_CONFIG["temperature"],
                 feature_weights=None):
        super(ForexClassifier, self).__init__()

        self.temperature = temperature

        if feature_weights is not None:
            self.register_buffer('feature_scale', feature_weights)
        else:
            self.register_buffer('feature_scale', torch.ones(input_dim))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = PositionalEncoding(hidden_dim)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)

        self.attention = SelfAttention(hidden_dim, n_heads)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'feature_scale' in name:
                continue
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = x * self.feature_scale.view(1, 1, -1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        lstm_out, _ = self.lstm(x)
        lstm_out, _ = self.attention(lstm_out)
        h = lstm_out[:, -1, :]
        logit = self.classifier(h)
        return logit

    def predict_proba(self, x):
        logit = self.forward(x)
        scaled_logit = logit / self.temperature
        return torch.sigmoid(scaled_logit).squeeze(-1)

    def predict_proba_raw(self, x):
        logit = self.forward(x)
        return torch.sigmoid(logit).squeeze(-1)

    def predict_direction(self, x):
        return self.predict_proba(x)

    def trading_signal(self, x, threshold=0.54):
        prob = self.predict_proba(x)
        signal = torch.zeros_like(prob)
        signal[prob > threshold] = 1.0
        signal[prob < (1.0 - threshold)] = -1.0
        confidence = torch.abs(prob - 0.5) * 2.0
        return signal, confidence, prob


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.55, gamma=2.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), targets, reduction='none')
        probs = torch.sigmoid(logits.squeeze(-1))
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()