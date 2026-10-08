"""Trainable pooled-feature adapters; NOT adapters inside the frozen encoder."""
import torch
from torch import nn

class FeatureAdapterFusion(nn.Module):
    def __init__(self, input_dim, bottleneck=64):
        super().__init__()
        if input_dim % 2:
            raise ValueError("Two equally sized image/text feature vectors required")
        dim = input_dim // 2
        self.text_adapter = nn.Sequential(nn.Linear(dim, bottleneck), nn.ReLU(), nn.Linear(bottleneck, dim))
        self.image_adapter = nn.Sequential(nn.Linear(dim, bottleneck), nn.ReLU(), nn.Linear(bottleneck, dim))
        # Initial residual perturbation is zero; the base features remain available.
        for adapter in [self.text_adapter, self.image_adapter]:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.classifier = nn.Linear(input_dim, 6)

    def forward(self, features):
        text, image = features.chunk(2, dim=1)
        text = text + 0.2 * self.text_adapter(text)
        image = image + 0.2 * self.image_adapter(image)
        return self.classifier(torch.cat([text, image], dim=1))
