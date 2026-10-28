"""CPU regression checks for exchange ownership/routes and fallback lifetimes.

No vLLM imports or CUDA extensions. Extract actual operation bodies as in the
existing test_selbuf_keep/test_draft_guard harnesses. Transport and norm kernels
are CPU doubles: these tests do not establish CUDA numerics or stream safety.
"""
import ast
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
import fa2sm70_clef_hook as clef_hook
from fa2sm80_fallback import chunked_fallback

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/vllm_sm70"
spec = importlib.util.spec_from_file_location("workspace", INTEGRATION / "fa2sm70_workspace.py")
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


class WorkspaceTests(unittest.TestCase):
    def test_eager_growth_releases_and_reuses(self):
        w = workspace.CapturedWorkspace()
        first = w.get(16, torch.empty)
        old = weakref.ref(first)
        self.assertIs(w.get(8, torch.empty), first)
        del first
        w.get(32, torch.empty)
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual(w.retired, [])

    def test_capture_addresses_survive_growth(self):
        w = workspace.CapturedWorkspace()
        first = w.get(16, torch.empty)
        pointer = first.data_ptr()
        # Mark an existing eager allocation when it is first used in a capture.
        self.assertIs(w.get(8, torch.empty, capturing=True), first)
        old = weakref.ref(first)
        del first
        w.get(32, torch.empty)
        w.get(64, torch.empty)
        gc.collect()
        self.assertEqual(old().data_ptr(), pointer)
        self.assertEqual(len(w.retired), 1)

    def test_allocation_failure_preserves_owner(self):
        w = workspace.CapturedWorkspace()
        first = w.get(16, torch.empty, capturing=True)
        def fail(_):
            raise MemoryError("allocation refused")
        with self.assertRaises(MemoryError):
            w.get(32, fail)
        self.assertIs(w.buffer, first)
        self.assertEqual(w.capacity, 16)
        self.assertTrue(w.captured)
        self.assertEqual(w.retired, [])


def exchange_harness(*, capturing=False, ce=True, capacity=4096, world=2, native=True):
    src = INTEGRATION / "fa2sm70_exch_i12.py"
    branch = next(n for n in ast.parse(src.read_text()).body
                  if isinstance(n, ast.If) and isinstance(n.test, ast.Compare))
    names = {"_буфер", "_обёртка"}
    nodes = [n for n in branch.body if isinstance(n, ast.FunctionDef) and n.name in names]
    allocations, calls = [], []

    def empty(size, **kw):
        assert kw["device"].type == "cpu"
        allocations.append(size)
        return torch.empty(size, **kw)

    def pack(x, pk, sc, bits):
        calls.append("pack")
        assert bits == 12 and pk.numel() == x.numel() // 128 * 96
        assert sc.numel() == x.numel() // 128
        pk.zero_()
        sc.fill_(1)

    def unpack(x, pk, sc):
        calls.append("unpack")
        # An out-of-place stand-in; the input must remain unchanged on both routes.
        return x + 1

    def gather(out, send):
        calls.append("nccl")
        for r in range(world):
            out[r * send.numel():(r + 1) * send.numel()].copy_(send)

    def gather_list(outputs, send):
        calls.append("dist")
        for out in outputs:
            out.copy_(send)

    ext = SimpleNamespace(pack_i12_into=pack, unpack_add_i12=unpack,
                          ar2_copy=lambda *a: calls.append("copy"),
                          ar2_signal=lambda *a: calls.append("signal"),
                          ar2_wait=lambda *a: calls.append("wait"))
    state = None if capacity is None else {
        "cap": capacity, "seq": 0, "n": 0, "p_recv": 100, "p_flag": 200,
        "flag": torch.zeros(2, dtype=torch.int32),
        "срыв": torch.zeros(2, dtype=torch.int32),
        "recv": torch.zeros(2 * capacity, dtype=torch.uint8),
    }
    ns = {"torch": SimpleNamespace(empty=empty, uint8=torch.uint8, int16=torch.int16,
                                    float16=torch.float16,
                                    cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)),
          "dist": SimpleNamespace(get_world_size=lambda: world, get_rank=lambda: 0,
                                   all_gather=gather_list),
          "CapturedWorkspace": workspace.CapturedWorkspace, "_КЭШ": {},
          "_ПОРОГ": 128, "_БИТ": 12, "_CE": ce, "_расш": lambda: ext,
          "_ce": lambda _: state, "С": {"n": 0, "ушло": 0, "было_бы": 0},
          "sys": __import__("sys")}
    exec(compile(ast.Module(nodes, []), str(src), "exec"), ns)
    comm = SimpleNamespace(pynccl_comm=SimpleNamespace(disabled=False, all_gather=gather)
                           if native else None)
    f = ns["_обёртка"](lambda _, x: x.clone())
    return lambda x: f(comm, x), ns, allocations, calls


class ExchangeTests(unittest.TestCase):
    def test_ce_does_not_allocate_gather(self):
        run, ns, alloc, calls = exchange_harness()
        x = torch.arange(512, dtype=torch.float16).reshape(4, 128)
        before = x.clone()
        result = run(x)
        self.assertEqual(alloc, [4 * 194])
        self.assertEqual(set(ns["_КЭШ"][None]), {"send"})
        self.assertEqual(calls, ["pack", "copy", "signal", "wait", "unpack"])
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(result, before + 1))
        self.assertNotEqual(result.data_ptr(), x.data_ptr())
        run(x[:2])
        self.assertEqual(alloc, [4 * 194])

    def test_capture_and_ce_fallbacks_keep_nccl(self):
        for options, transport in [({"capturing": True}, "nccl"),
                                   ({"capacity": 128}, "nccl"),
                                   ({"capacity": None}, "nccl"),
                                   ({"ce": False}, "nccl"),
                                   ({"ce": False, "native": False}, "dist"),
                                   ({"world": 3}, "nccl")]:
            with self.subTest(options=options):
                run, ns, alloc, calls = exchange_harness(**options)
                x = torch.zeros(4, 128, dtype=torch.float16)
                out = run(x)
                world = options.get("world", 2)
                self.assertEqual(alloc, [4 * 194, world * 4 * 194])
                self.assertIn(transport, calls)
                self.assertNotIn("copy", calls)
                self.assertTrue(torch.equal(out, torch.full_like(x, world - 1)))
                self.assertEqual(ns["_КЭШ"][None]["send"].captured,
                                 options.get("capturing", False))

    def test_original_small_and_dtype_routes_allocate_nothing(self):
        run, _, alloc, calls = exchange_harness()
        for x in (torch.zeros(1, 64, dtype=torch.float16), torch.zeros(128),
                  torch.zeros(129, dtype=torch.float16)):
            self.assertTrue(torch.equal(run(x), x))
        self.assertEqual(alloc, [])
        self.assertEqual(calls, [])


class FallbackTests(unittest.TestCase):
    def test_intermediates_die_before_down_without_mutating_inputs(self):
        src = INTEGRATION / "fa2sm80_blockpipe.py"
        install = next(n for n in ast.parse(src.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == "установить")
        direct = next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == "_прямой_кусок")
        refs, order = {}, []
        core = torch.arange(16, dtype=torch.float16).reshape(4, 4) / 16
        residual = torch.ones(4, 4, dtype=torch.float32)
        core_before, residual_before = core.clone(), residual.clone()

        def projection(x):
            order.append("out")
            y = x * 2
            refs["y"] = weakref.ref(y)
            return y, None

        def norm(y, r, w, eps, h, ro):
            order.append("norm")
            ro.copy_(y.float() + r)
            h.copy_(ro)
            refs["h"] = weakref.ref(h)

        def gate(x):
            order.append("gate")
            self.assertIsNone(refs["y"]())
            gu = torch.cat((x, x + 1), dim=1)
            refs["gu"] = weakref.ref(gu)
            return gu, None

        def silu(out, gu):
            order.append("silu")
            self.assertIsNone(refs["h"]())
            out.copy_(torch.nn.functional.silu(gu[:, :4]) * gu[:, 4:])

        def down(x):
            order.append("down")
            self.assertIsNone(refs["gu"]())
            return x * 3, None

        layer = SimpleNamespace(layer_type="linear_attention",
                                linear_attn=SimpleNamespace(out_proj=projection),
                                post_attention_layernorm=SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6),
                                mlp=SimpleNamespace(gate_up_proj=gate, down_proj=down))
        ns = {"_С": {"прямо": 0}, "_SILU": False, "_норма": lambda: norm,
              "torch": SimpleNamespace(empty=torch.empty, float16=torch.float16, float32=torch.float32,
                                        ops=SimpleNamespace(_C=SimpleNamespace(silu_and_mul=silu)))}
        exec(compile(ast.Module([direct], []), str(src), "exec"), ns)
        out, ro = ns["_прямой_кусок"](layer, core, residual)
        expected_ro = core_before.float() * 2 + residual_before
        h = expected_ro.half()
        expected = (torch.nn.functional.silu(h) * (h + 1)) * 3
        self.assertEqual(order, ["out", "norm", "gate", "silu", "down"])
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(ro, expected_ro))
        self.assertEqual(ro.dtype, torch.float32)
        self.assertTrue(torch.equal(core, core_before))
        self.assertTrue(torch.equal(residual, residual_before))


class ChunkedFallbackTests(unittest.TestCase):
    def test_chunks_cover_global_clef_rows_once_and_preserve_residual(self):
        src = INTEGRATION / "fa2sm70_clef.py"
        nodes = [n for n in ast.parse(src.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in {"_хук", "_хук_диап"}]
        seen, live = [], []

        def correction(x, y, keys, offset):
            seen.append((offset, len(x)))
            for row in [1, 2, 5, 6]:
                if offset <= row < offset + len(x):
                    y[row - offset].add_(10)

        ns = {"_АКТИВЕН": [True], "_С": SimpleNamespace(ext=SimpleNamespace(
            delta_keyed=lambda x, y, keys: correction(x, y, keys, 0),
            delta_keyed_range=correction))}
        exec(compile(ast.Module(nodes, []), str(src), "exec"), ns)
        core = torch.arange(28, dtype=torch.float16).reshape(7, 4)
        residual = torch.ones(7, 4, dtype=torch.float32)

        def forward(x, r):
            for ref in live:
                self.assertIsNone(ref())
            y, ro = x * 2, r + x.float()
            ns["_хук"](x, y, [])
            live[:] = [weakref.ref(y), weakref.ref(ro)]
            return y, ro

        out, ro = chunked_fallback(core, residual, 3, forward)
        expected = core * 2
        expected[[1, 2, 5, 6]] += 10
        self.assertEqual(seen, [(0, 3), (3, 3), (6, 1)])
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(ro, residual + core.float()))
        self.assertEqual(ro.dtype, torch.float32)
        self.assertIsNone(clef_hook.row_window_offset())
        # Explicit range hook nested inside a projection window adds its local
        # offset exactly once; ordinary hooks outside the window keep old routing.
        with clef_hook.row_window(10, 5):
            with clef_hook.row_window(2, 2):
                ns["_хук_диап"](core[:1], out[:1], [], 1)
        self.assertEqual(seen[-1], (13, 1))
        ns["_АКТИВЕН"][0] = False
        ns["_хук"](core, out, [])
        self.assertEqual(len(seen), 4)

    def test_exception_restores_window_and_inputs(self):
        core = torch.ones(7, 4)
        residual = torch.ones(7, 4)
        def fail(x, r):
            raise RuntimeError("projection failure")
        with clef_hook.row_window(20, 10):
            with self.assertRaises(RuntimeError):
                chunked_fallback(core, residual, 3, fail)
            self.assertEqual(clef_hook.row_window_offset(), 20)
            with self.assertRaises(ValueError):
                with clef_hook.row_window(9, 2):
                    pass
        self.assertIsNone(clef_hook.row_window_offset())
        self.assertTrue(torch.equal(core, torch.ones_like(core)))
        self.assertTrue(torch.equal(residual, torch.ones_like(residual)))

    def test_disabled_small_and_missing_range_hook_use_original_path(self):
        src = INTEGRATION / "fa2sm80_blockpipe.py"
        install = next(n for n in ast.parse(src.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == "установить")
        direct = next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == "_прямо")
        calls = []
        ns = {"_FALLBACK_ROWS": 3, "_MIN": 4, "_С": {}, "_clef_в_шаге": lambda: True,
              "_прямой_кусок": lambda layer, x, r: (calls.append(len(x)) or x.clone(), r.clone())}
        exec(compile(ast.Module([direct], []), str(src), "exec"), ns)
        x, r = torch.ones(7, 4), torch.ones(7, 4)
        with patch.object(clef_hook, "ХУК_ДИАП", [None]):
            ns["_прямо"](None, x, r)
        self.assertEqual(calls, [7])
        calls.clear()
        with patch.object(clef_hook, "ХУК_ДИАП", [lambda *a: None]):
            ns["_прямо"](None, x, r)
        self.assertEqual(calls, [3, 3, 1])
        calls.clear()
        ns["_FALLBACK_ROWS"] = 0
        ns["_прямо"](None, x, r)
        ns["_FALLBACK_ROWS"] = 3
        ns["_прямо"](None, x[:2], r[:2])
        self.assertEqual(calls, [7, 2])


if __name__ == "__main__":
    with patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden in CPU tests")):
        unittest.main()
