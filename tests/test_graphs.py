import unittest

from comfy_h3_mcp import graphs


def build(acceleration: str) -> dict:
    return graphs.build_image_to_video(
        prompt="test",
        width=864,
        height=480,
        length=124,
        first_frame=None,
        last_frame=None,
        seed=1,
        steps=20,
        sampler_name="res_multistep",
        scheduler="simple",
        shift_video=None,
        shift_audio=None,
        filename_prefix="test",
        acceleration=acceleration,
    )


class AccelerationGraphTests(unittest.TestCase):
    def classes(self, preset: str) -> list[str]:
        return [node["class_type"] for node in build(preset).values()]

    def test_off_has_no_approximation_nodes(self):
        classes = self.classes("off")
        self.assertNotIn(graphs.SOL_NODE, classes)
        self.assertNotIn(graphs.EASYCACHE_NODE, classes)

    def test_quality_uses_sol_only(self):
        classes = self.classes("quality")
        self.assertIn(graphs.SOL_NODE, classes)
        self.assertNotIn(graphs.EASYCACHE_NODE, classes)

    def test_balanced_and_fast_compose_sol_then_cache(self):
        for preset in ("balanced", "fast"):
            with self.subTest(preset=preset):
                graph = build(preset)
                sol_id = next(
                    key for key, value in graph.items()
                    if value["class_type"] == graphs.SOL_NODE
                )
                cache = next(
                    value for value in graph.values()
                    if value["class_type"] == graphs.EASYCACHE_NODE
                )
                self.assertEqual(cache["inputs"]["model"], [sol_id, 0])

    def test_default_is_balanced(self):
        graph = graphs.build_image_to_video(
            prompt="test",
            width=864,
            height=480,
            length=124,
            first_frame=None,
            last_frame=None,
            seed=1,
            steps=20,
            sampler_name="res_multistep",
            scheduler="simple",
            shift_video=None,
            shift_audio=None,
            filename_prefix="test",
        )
        classes = [node["class_type"] for node in graph.values()]
        self.assertIn(graphs.SOL_NODE, classes)
        self.assertIn(graphs.EASYCACHE_NODE, classes)

    def test_reference_graph_uses_balanced_route(self):
        graph = graphs.build_reference_to_video(
            prompt="test <Picture 1>",
            width=864,
            height=480,
            length=124,
            ref_images=["reference.png"],
            ref_videos=[],
            ref_audios=[],
            ref_image_size="match",
            seed=1,
            steps=20,
            sampler_name="res_multistep",
            scheduler="simple",
            shift_video=None,
            shift_audio=None,
            filename_prefix="test",
        )
        nodes = {key: value for key, value in graph.items()}
        sol_id = next(
            key for key, value in nodes.items()
            if value["class_type"] == graphs.SOL_NODE
        )
        cache_id, cache = next(
            (key, value) for key, value in nodes.items()
            if value["class_type"] == graphs.EASYCACHE_NODE
        )
        self.assertEqual(cache["inputs"]["model"], [sol_id, 0])
        guider = next(
            value for value in nodes.values()
            if value["class_type"] == "BasicGuider"
        )
        self.assertEqual(guider["inputs"]["model"], [cache_id, 0])


if __name__ == "__main__":
    unittest.main()
