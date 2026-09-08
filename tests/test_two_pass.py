import os
import unittest

from comfy_h3_mcp import graphs


class TwoPassGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["H3_REMOTE"] = "1"
        os.environ["H3_ALLOW_PRUNED"] = "1"
        os.environ.pop("REFINE_TWO_PASS", None)
        os.environ.pop("H3_USE_HYBRID_LOADER", None)
        os.environ.pop("H3_SPLIT_SIGMA_START", None)
        os.environ.pop("H3_PASS2_MANUAL_SIGMAS", None)

    def _build(self, **kwargs):
        return graphs.build_reference_to_video(
            prompt="t <Picture 1>",
            width=864,
            height=480,
            length=22,
            ref_images=["a.png"],
            ref_videos=[],
            ref_audios=[],
            ref_image_size="match",
            seed=1,
            steps=20,
            sampler_name="euler",
            scheduler="beta",
            shift_video=None,
            shift_audio=None,
            filename_prefix="t",
            acceleration="off",
            sage_attention="disabled",
            **kwargs,
        )

    def test_single_pass_has_one_sampler(self):
        g = self._build(two_pass=False)
        samplers = [n for n in g.values() if n["class_type"] == "SamplerCustomAdvanced"]
        self.assertEqual(len(samplers), 1)
        self.assertNotIn("SplitSigmas", {n["class_type"] for n in g.values()})

    def test_two_pass_split_upscale_two_samplers(self):
        # Identity: SplitSigmas high/low, no learned enlarge.
        g = self._build(two_pass=True, upscaler="none")
        classes = [n["class_type"] for n in g.values()]
        self.assertEqual(classes.count("SamplerCustomAdvanced"), 2)
        self.assertIn("SplitSigmas", classes)
        self.assertIn("H3LatentSpatialUpsampleStub", classes)
        self.assertIn("LTXVSeparateAVLatent", classes)
        self.assertIn("LTXVConcatAVLatent", classes)
        self.assertEqual(classes.count("CreateVideo"), 1)
        self.assertEqual(classes.count("SaveVideo"), 1)

    def test_two_pass_real_upscaler_node(self):
        g = self._build(two_pass=True, upscaler="real")
        up = next(n for n in g.values() if n["class_type"] == "MinimaxH3LatentUpscaler3D")
        self.assertEqual(up["inputs"]["mode.scale"], 1.5)
        self.assertEqual(up["inputs"]["mode"], "scale by multiplier")
        self.assertFalse(up["inputs"]["enable_temporal_chunking"])
        classes = [n["class_type"] for n in g.values()]
        # Spatial: full pass1 + ManualSigmas pass2 (no mid-SplitSigmas).
        self.assertNotIn("SplitSigmas", classes)
        self.assertIn("ManualSigmas", classes)

    def test_split_sigma_default_ratio(self):
        os.environ.pop("H3_SPLIT_SIGMA_START", None)
        split = graphs._split_sigma_start(20)
        self.assertEqual(split, 12)  # ~60%
        # Explicit env is honored (no auto-nudge).
        os.environ["H3_SPLIT_SIGMA_START"] = "8"
        self.assertEqual(graphs._split_sigma_start(20), 8)
        self.assertEqual(graphs._split_sigma_start(12), 8)

    def test_pass2_shares_random_noise(self):
        g = self._build(two_pass=True, upscaler="real")
        self.assertNotIn("DisableNoise", {n["class_type"] for n in g.values()})
        noises = [k for k, n in g.items() if n["class_type"] == "RandomNoise"]
        self.assertEqual(len(noises), 1)
        samplers = [n for n in g.values() if n["class_type"] == "SamplerCustomAdvanced"]
        for s in samplers:
            self.assertEqual(s["inputs"]["noise"][0], noises[0])

    def test_refine_two_pass_env_respects_profile(self):
        os.environ["REFINE_TWO_PASS"] = "1"
        os.environ["H3_RESOURCE_PROFILE"] = "draft"
        self.assertFalse(graphs._two_pass_enabled(profile="draft"))
        self.assertTrue(graphs._two_pass_enabled(profile="refine"))

    def test_hybrid_loader_flag(self):
        os.environ["H3_USE_HYBRID_LOADER"] = "1"
        g = self._build(two_pass=False)
        self.assertIn("MinimaxH3_HybridLoader", {n["class_type"] for n in g.values()})


if __name__ == "__main__":
    unittest.main()
