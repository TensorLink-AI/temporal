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
        self.preds = torch.randn(2, 10, 3)
        self.targets = torch.randn(2, 10, 3)

    def test_mq_loss(self):
        loss_fn = MQLoss(quantiles=[0.1, 0.5, 0.9])
        loss = loss_fn(self.preds.view(2, 10, -1), self.targets.view(2, 10, -1))
        self.assertIsInstance(loss, torch.Tensor)

    def test_weighted_quantile_loss(self):
        loss_fn = WeightedQuantileLoss(quantiles=(0.1, 0.5, 0.9))
        loss = loss_fn(self.preds.view(2, 10, -1), self.targets.view(2, 10, -1))
        self.assertIsInstance(loss, torch.Tensor)

    def test_quantile_loss(self):
        loss_fn = QuantileLoss(quantile=0.5)
        loss = loss_fn(self.preds[..., 0], self.targets[..., 0])
        self.assertIsInstance(loss, torch.Tensor)

    def test_kernel_energy_loss(self):
        loss_fn = KernelEnergyLoss()
        loss = loss_fn(self.preds.view(2, 10, -1), self.targets.view(2, 10, -1))
        self.assertIsInstance(loss, torch.Tensor)

    def test_energy_distance_loss(self):
        loss_fn = EnergyDistanceLoss()
        loss = loss_fn(self.preds.view(2, 10, -1), self.targets.view(2, 10, -1))
        self.assertIsInstance(loss, torch.Tensor)

    def test_spectral_loss(self):
        loss_fn = SpectralLoss()
        loss = loss_fn(self.preds[..., 0], self.targets[..., 0])
        self.assertIsInstance(loss, torch.Tensor)

    def test_fast_soft_dtw_loss(self):
        loss_fn = FastSoftDTWLoss()
        loss = loss_fn(self.preds[..., 0], self.targets[..., 0])
        self.assertIsInstance(loss, torch.Tensor)

    def test_spread_penalty(self):
        loss_fn = SpreadPenalty()
        loss = loss_fn(self.preds)
        self.assertIsInstance(loss, torch.Tensor)

    def test_mixture_loss(self):
        loss_fn = MixtureLoss(components=["normal", "student_t"])
        preds = {
            "mixture_logits": torch.randn(2, 10, 2),
            "normal_mu": torch.randn(2, 10),
            "normal_scale": torch.randn(2, 10),
            "student_t_df": torch.randn(2, 10),
            "student_t_loc": torch.randn(2, 10),
            "student_t_scale": torch.randn(2, 10),
        }
        loss = loss_fn(preds, self.targets[..., 0])
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
