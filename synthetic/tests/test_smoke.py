"""Fast integration checks for the synthetic generation pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from synthetic.config import SyntheticConfig, get_default_config
from synthetic.evaluation import QualityFilter
from synthetic.generate_dataset import DatasetGenerator
from synthetic.models import StyleGANGenerator
from synthetic.models.losses import r1_penalty


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
            rgb_logits, mask_logits = generator(torch.randn(2, config.model.latent_dim))

        self.assertEqual(tuple(rgb_logits.shape), (2, 3, 256, 256))
        self.assertEqual(tuple(mask_logits.shape), (2, 1, 256, 256))
        self.assertGreater(float((rgb_logits[0] - rgb_logits[1]).abs().mean()), 0.0)
        self.assertGreater(float((mask_logits[0] - mask_logits[1]).abs().mean()), 0.0)

    def test_unconditional_mask_losses_are_disabled(self) -> None:
        config = get_default_config()
        self.assertEqual(config.loss.lambda_mask, 0.0)
        self.assertEqual(config.loss.lambda_boundary, 0.0)

    def test_r1_penalty_is_batch_size_invariant(self) -> None:
        class QuadraticDiscriminator(nn.Module):
            def forward(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return rgb.square().mean(dim=1, keepdim=True) + mask.square()

        discriminator = QuadraticDiscriminator()
        rgb = torch.rand(1, 3, 8, 8)
        mask = torch.rand(1, 1, 8, 8)
        single = r1_penalty(rgb, mask, discriminator)
        repeated = r1_penalty(
            rgb.detach().repeat(4, 1, 1, 1),
            mask.detach().repeat(4, 1, 1, 1),
            discriminator,
        )

        self.assertTrue(torch.allclose(single, repeated, rtol=1e-5, atol=1e-5))

    def test_generated_masks_are_saved_as_binary(self) -> None:
        class FixedGenerator(nn.Module):
            def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                batch_size = z.shape[0]
                rgb_logits = torch.zeros(batch_size, 3, 2, 2, device=z.device)
                mask_logits = torch.tensor(
                    [[[[-4.0, -1.0], [1.0, 4.0]]]],
                    device=z.device,
                ).repeat(batch_size, 1, 1, 1)
                return rgb_logits, mask_logits

        config = get_default_config()
        config.device.device = "cpu"
        config.model.latent_dim = 2
        dataset_generator = DatasetGenerator(FixedGenerator(), config, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_generator.generate(
                output_dir=temp_dir,
                num_samples=1,
                batch_size=1,
                use_quality_filter=False,
                save_metadata=False,
                latent_dim=2,
            )
            saved_mask = cv2.imread(str(Path(temp_dir) / "label" / "000000.png"), cv2.IMREAD_GRAYSCALE)

        self.assertEqual(set(np.unique(saved_mask).tolist()), {0, 255})

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
