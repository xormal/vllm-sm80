"""Small CUDA ownership regression. Run separately from model workload, GPU0 only."""
import importlib.util
import os
from pathlib import Path
import unittest
import weakref

import torch

path = Path(__file__).resolve().parents[1] / "integrations/vllm_sm70/fa2sm70_workspace.py"
spec = importlib.util.spec_from_file_location("candidate_workspace", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkspaceCudaTest(unittest.TestCase):
    def test_capture_survives_growth_and_replays_original_storage(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,1")
        torch.cuda.set_device(0)
        workspace = module.CapturedWorkspace()

        def allocate(size):
            return torch.empty(size, dtype=torch.float32, device="cuda:0")

        first = workspace.get(1024, allocate)
        first.fill_(2)
        first_ref = weakref.ref(first)
        first_ptr = first.data_ptr()
        # Warm torch operation outside capture on a side stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output = first * 3
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = workspace.get(1024, allocate, capturing=True)
            output = captured * 3
        del first, captured
        second = workspace.get(4096, allocate)
        second.fill_(99)
        torch.cuda.synchronize()
        self.assertIsNotNone(first_ref())
        self.assertEqual(first_ref().data_ptr(), first_ptr)
        self.assertNotEqual(second.data_ptr(), first_ptr)
        for value in (7, -2, 13):
            first_ref().fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output.cpu(), torch.full((1024,), value * 3.0)))
        self.assertTrue(torch.equal(second.cpu(), torch.full((4096,), 99.0)))
        print("workspace capture/growth/replay: CUDA0 passed", flush=True)


if __name__ == "__main__":
    unittest.main()
