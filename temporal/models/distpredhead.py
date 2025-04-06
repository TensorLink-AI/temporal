import torch
import torch.nn as nn
import torch.nn.functional as F

class DistPredHead(nn.Module):
    """
    DistPredHead creates an ensemble of forecasts for time-series data,
    allowing you to compute distribution-based metrics (e.g. CRPS).

    Args:
        hidden_size (int):
            The dimension of the final Transformer hidden states (H).
        ensemble_size (int):
            The number of parallel ensemble members (M) to produce.
        config (optional):
            Config object (e.g. your time-series config) to read dropout,
            or other hyperparameters as needed.

    Shape:
        Input:
          hidden_states => [B, T, hidden_size]
            B = batch size
            T = number of forecast steps (the decoder output length)
        Output:
          forecast_ensemble => [B, T, M]
            M = ensemble_size.

    Usage:
    ------
    1) Instantiate in your model's constructor:
       >>> self.distpred_head = DistPredHead(hidden_size=H, ensemble_size=M, config=cfg)

    2) In your model's forward, after obtaining final hidden states of shape [B,T,H]:
       >>> ensemble_output = self.distpred_head(final_hidden_states)
       # Now ensemble_output is [B, T, M].

    3) Use the ensemble output with a CRPS function, e.g.
       >>> crps_loss = crps_ensemble(obs, ensemble_output)
    """

    def __init__(self, hidden_size: int, ensemble_size: int, config=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.ensemble_size = ensemble_size

        # For example, a single linear layer that expands [B,T,H] -> [B,T, M].
        # (You can also do [B,T, M*x] if you want multi-output, then reshape.)
        self.proj = nn.Linear(hidden_size, ensemble_size)

        # Optional dropout. We read from config if available, else default 0.1
        dropout_p = 0.1
        if config and hasattr(config, "hidden_dropout_prob"):
            dropout_p = config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: produce ensemble forecasts from hidden states.

        Args:
            hidden_states (torch.Tensor):
                Shape [B, T, hidden_size], the final output from your 
                Transformer decoder or another time-series model block.

        Returns:
            forecast_ensemble (torch.Tensor):
                Shape [B, T, M] = [batch_size, forecast_steps, ensemble_size].
        """
        # 1) optional dropout
        x = self.dropout(hidden_states)  # [B,T,H]

        # 2) project to [B,T,M]
        out = self.proj(x)  # => [B,T,M]

        return out
