"""Fast integration checks for the synthetic generation pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from synthetic.config import SyntheticConfig, get_default_config
from synthetic.evaluation import QualityFilter
from synthetic.generate_dataset import DatasetGenerator
from synthetic.models import StyleGANGenerator


class SyntheticPipelineSmokeTests(unittest.TestCase):
    def test_config_round_trip_is_json_safe(self) -> None:
        config = get_default_config()
        restored = SyntheticConfig.from_dict(json.loads(config.to_json()))

        restored.validate()
        self.assertEqual(restored.model.image_size, config.model.image_size)
        self.assertIsInstance(restored.to_dict()["paths"]["project_root"], str)

    def test_generator_returns_logits_with_expected_shapes(self) -> None:
        config = get_default_config()
        config.model.image_size = 256
        generator = StyleGANGenerator(config.model).eval()

        with torch.no_grad():
            rgb_logits, mask_logits = generator(torch.randn(1, config.model.latent_dim))

        self.assertEqual(tuple(rgb_logits.shape), (1, 3, 256, 256))
        self.assertEqual(tuple(mask_logits.shape), (1, 1, 256, 256))

    def test_quality_filter_uses_canonical_config(self) -> None:
        config = get_default_config()
        quality_filter = QualityFilter(config.quality_filter)
        rgb = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 8:24, 8:24] = 1.0

        passed, metrics = quality_filter.filter(rgb, mask)

        self.assertTrue(passed)
        self.assertEqual(metrics.building_pixels, 256)
        self.assertEqual(metrics.building_count, 1)

    def test_generation_checkpoint_requires_generator_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "invalid.pt"
            torch.save({"config": get_default_config().to_dict()}, checkpoint_path)

            with self.assertRaisesRegex(KeyError, "generator_state_dict"):
                DatasetGenerator.from_checkpoint(str(checkpoint_path), device="cpu")


if __name__ == "__main__":
    unittest.main()
