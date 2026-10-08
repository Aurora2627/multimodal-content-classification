import unittest
import torch
from sampling import sample_training_indices

class SamplingTests(unittest.TestCase):
    def test_exact_counts_reproducibility_and_seed(self):
        labels = torch.arange(6).repeat_interleave(7)
        first = sample_training_indices(labels, 4, 42)
        self.assertEqual(len(first.unique()), 24)
        self.assertEqual(torch.bincount(labels[first], minlength=6).tolist(), [4]*6)
        torch.testing.assert_close(first, sample_training_indices(labels,4,42))
        self.assertFalse(torch.equal(first, sample_training_indices(labels,4,43)))
    def test_insufficient_rows_rejected(self):
        with self.assertRaises(ValueError):
            sample_training_indices(torch.arange(6),2,42)
