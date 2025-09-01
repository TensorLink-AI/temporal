import torch
import unittest
from temporal.modules.losses.loss_functions import (
    MQLoss,
    WeightedQuantileLoss,
    QuantileLoss,
    KernelEnergyLoss,
    EnergyDistanceLoss,
    SpectralLoss,
    FastSoftDTWLoss,
    SpreadPenalty,
    MixtureLoss,
)

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        # General tensors for (batch, sequence_length, features) or (batch, sequence_length, quantiles)
        self.preds_sequence_features = torch.randn(2, 10, 3)
        self.targets_sequence_features = torch.randn(2, 10, 3)

        # Tensors for (batch, sequence_length)
        self.preds_sequence = torch.randn(2, 10)
        self.targets_sequence = torch.randn(2, 10)

    def test_mq_loss(self):
        loss_fn = MQLoss(quantiles=[0.1, 0.5, 0.9])
        # preds: (B, T, Q), target: (B, T)
        loss = loss_fn(self.preds_sequence_features, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_weighted_quantile_loss(self):
        loss_fn = WeightedQuantileLoss(quantiles=(0.1, 0.5, 0.9))
        # preds: (B, T, Q), target: (B, T)
        loss = loss_fn(self.preds_sequence_features, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_quantile_loss(self):
        loss_fn = QuantileLoss(quantile=0.5)
        # predictions: (B, T), labels: (B, T)
        loss = loss_fn(self.preds_sequence, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_kernel_energy_loss(self):
        loss_fn = KernelEnergyLoss()
        # preds: (B, T, N), targets: (B, T)
        loss = loss_fn(self.preds_sequence_features, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_energy_distance_loss(self):
        loss_fn = EnergyDistanceLoss()
        # samples: (B, N_samples, T_length), target: (B, T_length)
        # self.preds_sequence_features is (2, 10, 3), treated as (B, N_samples, T_length)
        # self.targets_sequence is (2, 10), should be (B, T_length)
        # Reshape self.preds_sequence_features to (B, N_samples, T_length) -> (2, 3, 10)
        # Reshape self.targets_sequence to (B, T_length) -> (2, 10)
        # This requires adjusting the tensors or their usage

        # For EnergyDistanceLoss, let's create specific tensors to match its docstring
        samples_for_ed = torch.randn(2, 5, 10) # B=2, N_samples=5, T_length=10
        target_for_ed = torch.randn(2, 10)   # B=2, T_length=10
        loss = loss_fn(samples_for_ed, target_for_ed)
        self.assertIsInstance(loss, torch.Tensor)

    def test_spectral_loss(self):
        loss_fn = SpectralLoss()
        # preds: (B, T), targets: (B, T)
        loss = loss_fn(self.preds_sequence, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_fast_soft_dtw_loss(self):
        loss_fn = FastSoftDTWLoss()
        # preds: (B, T), targets: (B, T)
        loss = loss_fn(self.preds_sequence, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_spread_penalty(self):
        loss_fn = SpreadPenalty()
        # preds: (B, T, Q)
        loss = loss_fn(self.preds_sequence_features)
        self.assertIsInstance(loss, torch.Tensor)

    def test_mixture_loss(self):
        loss_fn = MixtureLoss(components=["gaussian", "student_t"])
        preds = {
            "mixture_logits": torch.randn(2, 10, 2),
            "gaussian_mu": torch.randn(2, 10),
            "gaussian_scale": torch.randn(2, 10).abs() + 1e-6, # Ensure scale is positive
            "student_t_df": torch.randn(2, 10).abs() + 2.0, # Ensure df is > 2
            "student_t_loc": torch.randn(2, 10),
            "student_t_scale": torch.randn(2, 10).abs() + 1e-6, # Ensure scale is positive
            "components": ["gaussian", "student_t"] # Added for the loss function to iterate over
        }
        loss = loss_fn(preds, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
