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
        self.preds_sequence_features = torch.randn(2, 10, 3)
        self.targets_sequence_features = torch.randn(2, 10, 3)
        self.preds_sequence = torch.randn(2, 10)
        self.targets_sequence = torch.randn(2, 10)

    def test_mq_loss(self):
        loss_fn = MQLoss(quantiles=[0.1, 0.5, 0.9])
        loss = loss_fn(self.preds_sequence_features, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_weighted_quantile_loss(self):
        loss_fn = WeightedQuantileLoss(quantiles=(0.1, 0.5, 0.9))
        loss = loss_fn(self.preds_sequence_features, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_quantile_loss(self):
        loss_fn = QuantileLoss(quantile=0.5)
        loss = loss_fn(self.preds_sequence, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_kernel_energy_loss(self):
        loss_fn = KernelEnergyLoss()
        loss = loss_fn(self.preds_sequence_features, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_energy_distance_loss(self):
        samples_for_ed = torch.randn(2, 5, 10)
        target_for_ed = torch.randn(2, 10)
        loss_fn = EnergyDistanceLoss()
        loss = loss_fn(samples_for_ed, target_for_ed)
        self.assertIsInstance(loss, torch.Tensor)

    def test_spectral_loss(self):
        loss_fn = SpectralLoss()
        loss = loss_fn(self.preds_sequence, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_fast_soft_dtw_loss(self):
        loss_fn = FastSoftDTWLoss()
        loss = loss_fn(self.preds_sequence, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

    def test_spread_penalty(self):
        loss_fn = SpreadPenalty()
        loss = loss_fn(self.preds_sequence_features)
        self.assertIsInstance(loss, torch.Tensor)

    def test_mixture_loss(self):
        loss_fn = MixtureLoss()
        preds = {
            "mixture_logits": torch.randn(2, 10, 2),
            "components": ["gaussian", "student_t"],
            "gaussian_mu": torch.randn(2, 10),
            "gaussian_sigma": torch.randn(2, 10).abs() + 1e-6,
            "student_df": torch.randn(2, 10).abs() + 2.0,
            "student_mu": torch.randn(2, 10),
            "student_scale": torch.randn(2, 10).abs() + 1e-6,
        }
        loss = loss_fn(preds, self.targets_sequence)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
