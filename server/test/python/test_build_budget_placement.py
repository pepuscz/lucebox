"""scripts/cluster/build_budget_placement.py: unequal ranks in one box."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "build_budget_placement.py"


def load_script():
    spec = importlib.util.spec_from_file_location("build_budget_placement", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildBudgetPlacementTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write_hotness(self, rows, n_expert_used=2):
        path = self.dir / "hotness.csv"
        lines = ["# hotness table: n_layer=%d n_expert=%d n_expert_used=%d" % (len(rows), len(rows[0]), n_expert_used),
                 "# format: one row per layer"]
        lines += [",".join(str(x) for x in row) for row in rows]
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_hottest_experts_go_to_the_card_ranks_in_order(self):
        rows = [[5, 40, 10, 30, 20, 1], [7, 7, 9, 0, 3, 8]]
        hotness = self.write_hotness(rows)
        out = self.dir / "map.json"
        # one expert is 1 MiB; rank 1 gets 2 per layer (4 MiB over 2 layers), rank 2 one
        rc = self.mod.main(["--hotness", str(hotness), "--ranks", "3", "--budget-gib", "1=%r" % (4 / 1024),
                            "--budget-gib", "2=%r" % (2 / 1024), "--expert-mib", "1", "--out", str(out)])
        self.assertEqual(rc, 0)
        placement = json.loads(out.read_text())
        self.assertEqual(placement["n_ranks"], 3)
        self.assertEqual((placement["n_layer"], placement["n_expert"], placement["n_expert_used"]), (2, 6, 2))
        self.assertEqual(placement["replicate_hot"], 0)
        self.assertEqual(placement["owner"][0], [0, 1, 0, 1, 2, 0])   # 40, 30 to rank 1; 20 to rank 2
        self.assertEqual(placement["owner"][1], [2, 0, 1, 0, 0, 1])   # 9, 8 to rank 1; 7 (expert 0, the lower id) to rank 2

    def test_protected_experts_stay_on_the_head(self):
        rows = [[50, 40, 30, 20]]
        hotness = self.write_hotness(rows)
        protected = self.dir / "protected.json"
        protected.write_text(json.dumps({"0": [0]}))
        out = self.dir / "map.json"
        self.mod.main(["--hotness", str(hotness), "--ranks", "2", "--budget-gib", "1=%r" % (2 / 1024),
                       "--expert-mib", "1", "--protected", str(protected), "--out", str(out)])
        self.assertEqual(json.loads(out.read_text())["owner"], [[0, 1, 1, 0]])

    def test_tier_placement_orders_primary_then_secondary_then_stream(self):
        placement = self.dir / "tiers.json"
        placement.write_text(json.dumps({"owners": ["stream", "secondary", "primary", "stream",
                                                     "secondary", "stream", "stream", "primary"]}))
        out = self.dir / "map.json"
        self.mod.main(["--placement", str(placement), "--n-expert", "4", "--n-expert-used", "1", "--ranks", "2",
                       "--budget-gib", "1=%r" % (2 / 1024), "--expert-mib", "1", "--out", str(out)])
        owner = json.loads(out.read_text())["owner"]
        self.assertEqual(owner, [[0, 0, 1, 0], [0, 0, 0, 1]])   # one per layer: the primary-tier expert

    def test_a_rank_cannot_take_everything(self):
        hotness = self.write_hotness([[3, 2, 1]])
        out = self.dir / "map.json"
        with self.assertRaises(ValueError):
            self.mod.main(["--hotness", str(hotness), "--ranks", "2", "--budget-gib", "1=%r" % (3 / 1024),
                           "--expert-mib", "1", "--out", str(out)])

    def test_every_card_rank_needs_a_budget(self):
        hotness = self.write_hotness([[3, 2, 1, 0]])
        out = self.dir / "map.json"
        with self.assertRaises(ValueError):
            self.mod.main(["--hotness", str(hotness), "--ranks", "3", "--budget-gib", "1=%r" % (1 / 1024),
                           "--expert-mib", "1", "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
