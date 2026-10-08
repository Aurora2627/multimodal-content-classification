"""Deterministic class-balanced sampling from training rows only."""
import torch

def sample_training_indices(labels, shots, seed):
    if shots < 0:
        raise ValueError("shots must be nonnegative")
    if shots == 0:
        return torch.arange(len(labels))
    generator = torch.Generator().manual_seed(seed)
    indices = []
    for label in range(6):
        members = (labels == label).nonzero().flatten()
        if len(members) < shots:
            raise ValueError("Insufficient training rows for class %d" % label)
        indices.append(members[torch.randperm(len(members), generator=generator)[:shots]])
    selected = torch.cat(indices)
    return selected[torch.randperm(len(selected), generator=generator)]
