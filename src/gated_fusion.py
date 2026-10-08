"""Sample-dependent modality weighting over frozen paired encoder features."""
import torch
from torch import nn
from torch_models import build_head

class GatedMLPFusion(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        if input_dim % 2:
            raise ValueError("Paired modalities must have equal feature width")
        self.classifier = build_head(input_dim, hidden)
        # Preserve the baseline classifier initialization and global training RNG.
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 2))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def gate_weights(self, features):
        return self.gate(features).softmax(dim=1)

    def forward(self, features, neutral_gate=False):
        text, image = features.chunk(2, dim=1)
        weights = torch.full((len(features), 2), .5, device=features.device,
                             dtype=features.dtype) if neutral_gate else self.gate_weights(features)
        weighted = torch.cat([2 * weights[:, :1] * text, 2 * weights[:, 1:] * image], dim=1)
        return self.classifier(weighted)
