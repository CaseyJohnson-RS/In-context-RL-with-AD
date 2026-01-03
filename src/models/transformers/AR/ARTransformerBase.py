import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod


Tensor = torch.Tensor


class ARTransformerBase(nn.Module, ABC):
    """
    Base autoregressive transformer for action sequence prediction.
    Predicts next action from action/reward history using causal masking.
    
    Args:
        n_actions: Number of possible actions (e.g., bandit arms).
        d_model: Embedding and transformer hidden dimension (default: 64).
        nhead: Number of attention heads (default: 4).
        num_layers: Number of transformer encoder layers (default: 4).
        dim_feedforward: FFN hidden dimension (default: 2048).
        max_len: Maximum sequence length (default: 200).
    """
    
    def __init__(
        self,
        n_actions: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        max_len: int = 200,
        dropout: float = 0.1,
        device: str | None = None
    ) -> None:
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Config (frozen for reproducibility)
        self.n_actions = n_actions
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.dropout = dropout
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, n_actions)
        
        # Apply weight init
        self.apply(self._init_weights)

        self.to(device)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Xavier init for stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> Optional[Tensor]:
        """Generate causal attention mask."""
        if seq_len == 1:
            return None
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=device, dtype=torch.float
        )
        mask.triu_(1)  # Upper triangle masked
        return mask

    @abstractmethod
    def sequence_encode(self, X: Tensor) -> Tensor:
        """
        Encode action/reward tensors into transformer embeddings.
        
        Args:
            actions: (B, L) action indices.
            rewards: (B, L) reward values.
            
        Returns:
            Embeddings (B, L, d_model).
        """
        pass
    
    @abstractmethod
    def decode_action(self, X: Tensor) -> Tensor:
        """
        Decode transformer output to action logits.
        
        Args:
            x: (B, L, d_model) final layer norm output.
            
        Returns:
            Action logits (B, L, n_actions).
        """
        pass
    
    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            actions: (B, L) or (L,) action sequence.
            rewards: (B, L) or (L,) reward sequence.
            
        Returns:
            Action logits (B, L, n_actions).
        """
        # Encode sequence
        x = self.sequence_encode(X)  # (B, L, D)
        
        batch_size, seq_len = x.shape[:2]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len]
        
        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # Transformer forward
        x = self.transformer(
            x, 
            mask=causal_mask
        )  # (B, L, D)
        
        x = self.ln_final(x)
        
        # Decode to actions
        logits = self.decode_action(x)  # (B, n_actions)
        
        return logits
