import unittest
from temporal.configs.loss_config import (
    TimeSeriesLossConfig,
    MSELossConfig,
    CRPSLossConfig,
    CRPSHuberLossConfig,
    QuantileLossConfig,
    TimeFlowLossConfig,
    NLLLossConfig,
    loss_config_from_dict,
)

class TestLossConfigs(unittest.TestCase):

    def test_timeseries_loss_config(self):
        config = TimeSeriesLossConfig(loss_type="mse")
        self.assertEqual(config.loss_type, "mse")

    def test_mse_loss_config(self):
        config = MSELossConfig()
        self.assertEqual(config.type, "mse")

    def test_crps_loss_config(self):
        config = CRPSLossConfig()
        self.assertEqual(config.estimator, "pinball")

    def test_crps_huber_loss_config(self):
        config = CRPSHuberLossConfig()
        self.assertEqual(config.huber_loss_threshold, 0.0)

    def test_quantile_loss_config(self):
        config = QuantileLossConfig(quantiles=[0.1, 0.5, 0.9])
        self.assertEqual(config.quantiles, [0.1, 0.5, 0.9])
        with self.assertRaises(ValueError):
            QuantileLossConfig(quantiles=[])

    def test_timeflow_loss_config(self):
        config = TimeFlowLossConfig()
        self.assertEqual(config.type, "timeflow")

    def test_nll_loss_config(self):
        config = NLLLossConfig(kwargs={"distribution_type": "gaussian"})
        self.assertEqual(config.kwargs["distribution_type"], "gaussian")
        with self.assertRaises(ValueError):
            NLLLossConfig(kwargs={})

    def test_loss_config_from_dict(self):
        config_dict = {"type": "mse"}
        config = loss_config_from_dict(config_dict)
        self.assertIsInstance(config, MSELossConfig)

        config_dict = {"type": "quantile", "quantiles": [0.1, 0.5, 0.9]}
        config = loss_config_from_dict(config_dict)
        self.assertIsInstance(config, QuantileLossConfig)

if __name__ == '__main__':
    unittest.main()
