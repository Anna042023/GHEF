# hypergraph_layers.py
import torch
import torch.nn as nn

class HypergraphGate(nn.Module):
    """Eq.(10) gate fusion: H_final = gamma*H_trend + (1-gamma)*H_geo."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, H_trend: torch.Tensor, H_geo: torch.Tensor):
        x = torch.stack([H_trend.abs(), H_geo.abs()], dim=-1)  # [...,2]
        gamma = torch.sigmoid(self.fc(x)).squeeze(-1)
        return gamma * H_trend + (1.0 - gamma) * H_geo


class HypergraphConv(nn.Module):
    """Eq.(11) hypergraph convolution with propagation matrix G."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, H: torch.Tensor):
        """
        X: [B, D, F]
        H: [D, E] (incidence-like)
        Return: X', G where G is [D, D]
        """
        B, D, Fdim = X.shape

        if H.dim() != 2:
            raise ValueError("H must be 2D incidence-like matrix [D,E].")

        D2, E = H.shape
        if D2 != D:
            raise ValueError(f"H dim mismatch: H={H.shape}, X={X.shape}")

        H = H.clamp(min=0.0)

        # De: [E], Dv: [D]
        De = H.sum(dim=0) + 1e-6
        Dv = H.sum(dim=1) + 1e-6

        # Build G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}
        inv_sqrt_Dv = torch.diag(Dv.pow(-0.5))
        inv_De = torch.diag(De.pow(-1.0))
        G = inv_sqrt_Dv @ H @ inv_De @ H.t() @ inv_sqrt_Dv  # [D, D]

        # Correct propagation: out[b,i,f] = sum_j G[i,j] * X[b,j,f]
        Xp = torch.einsum("ij,bjf->bif", G, X)  # [B, D, F]

        return self.lin(Xp), G