import torch
import unittest
from temporal.losses.crps_loss_ensemble import crps_ensemble

class TestCrpsLossEnsemble(unittest.TestCase):

    def setUp(self):
        self.observations = torch.randn(2, 10)
        self.forecasts = torch.randn(2, 10, 5)

    def test_crps_ensemble_nrg(self):
        crps = crps_ensemble(
            self.observations,
            self.forecasts,
            estimator="nrg",
        )
        self.assertIsInstance(crps, torch.Tensor)
        self.assertEqual(crps.shape, torch.Size([]))

    def test_crps_ensemble_pwm(self):
        crps = crps_ensemble(
            self.observations,
            self.forecasts,
            estimator="pwm",
        )
        self.assertIsInstance(crps, torch.Tensor)
        self.assertEqual(crps.shape, torch.Size([]))

    def test_crps_ensemble_fair(self):
        crps = crps_ensemble(
            self.observations,
            self.forecasts,
            estimator="fair",
        )
        self.assertIsInstance(crps, torch.Tensor)
        self.assertEqual(crps.shape, torch.Size([]))

    def test_crps_ensemble_reduce_false(self):
        crps = crps_ensemble(
            self.observations,
            self.forecasts,
            reduce=False,
        )
        self.assertIsInstance(crps, torch.Tensor)
        self.assertEqual(crps.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()
