import json
import unittest
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "minimax_h3_sol_balanced.json"
)


class GuiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(WORKFLOW.read_text())
        cls.subgraph = cls.workflow["definitions"]["subgraphs"][0]
        cls.nodes = {node["id"]: node for node in cls.subgraph["nodes"]}
        cls.links = {link["id"]: link for link in cls.subgraph["links"]}

    def test_balanced_model_route(self):
        route = [(6, 119), (119, 120), (120, 9), (120, 16)]
        actual = {
            (link["origin_id"], link["target_id"])
            for link in self.links.values()
            if link["type"] == "MODEL"
        }
        self.assertEqual(actual, set(route))
        self.assertEqual(self.nodes[119]["type"], "SolAttnPatch")
        self.assertEqual(self.nodes[120]["type"], "EasyCache")

    def test_balanced_widget_values(self):
        self.assertEqual(
            self.nodes[119]["widgets_values"],
            [
                1.3,
                0.2,
                0.9,
                4096,
                True,
                "exact_kv_and_rows",
                False,
                "2d_frame",
                False,
                False,
                "0,-1",
            ],
        )
        self.assertEqual(
            self.nodes[120]["widgets_values"], [0.1, 0.15, 0.9, False]
        )

    def test_every_connected_input_has_a_matching_link(self):
        for node in self.nodes.values():
            for slot, input_ in enumerate(node.get("inputs", [])):
                link_id = input_.get("link")
                if link_id is None:
                    continue
                link = self.links[link_id]
                self.assertEqual(link["target_id"], node["id"])
                self.assertEqual(link["target_slot"], slot)

    def test_link_ids_and_node_ids_fit_workflow_counters(self):
        self.assertGreaterEqual(
            self.workflow["last_node_id"], max(self.nodes)
        )
        self.assertGreaterEqual(
            self.workflow["last_link_id"], max(self.links)
        )


if __name__ == "__main__":
    unittest.main()
