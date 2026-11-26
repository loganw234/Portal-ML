"""Integration tests for the length-prefixed JSON IPC bridge."""

from __future__ import annotations

import unittest

from portal_ml.bridge import MockBridgeServer, SourcePortalEnv
from portal_ml.bt import BehaviorTreeEvaluator, get_reference_tree


class TestSourceBridge(unittest.TestCase):
    def test_end_to_end_tcp_bridge_solves_all_chambers(self) -> None:
        tree = get_reference_tree()
        with MockBridgeServer(port=0) as server:
            with SourcePortalEnv(host=server.host, port=server.port) as env:
                self.assertTrue(env.ping())
                for cid in (1, 2, 3, 4, 5):
                    ev = BehaviorTreeEvaluator(tree)
                    snap = env.reset(cid)
                    self.assertEqual(snap.chamber_id, cid)
                    while not snap.done:
                        res = env.step(ev.tick(snap).command)
                        snap = res.snapshot
                    self.assertTrue(snap.success, f"Chamber {cid} failed over TCP IPC bridge")


if __name__ == "__main__":
    unittest.main()
