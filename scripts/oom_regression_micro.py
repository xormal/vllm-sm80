#!/usr/bin/env python3
"""Prepared, NOT EXECUTED: one isolated sparse/fp16 OOM case per process.

No JIT compilation, service management, or HTTP requests. Synthetic allocator
headroom is deliberately distinct from driver free and production fragmentation.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

from oom_regression_guard import REPO, maintenance_guard, unused_gpus_guard

MIB = 1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("sparse", "linear"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--maintenance-window", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rows", type=int, choices=(8191, 8192, 8193), default=8192)
    parser.add_argument("--context", type=int, choices=(196607, 196608, 196609, 212992, 262140), default=196608)
    parser.add_argument("--headroom-mib", type=int, choices=(126, 144, 206, 272, 384, 512, 768, 1024), default=1024)
    parser.add_argument("--graph-pool-mib", type=int, choices=(0, 512), default=512)
    parser.add_argument("--extension", type=Path, help="Already built candidate fa2sm70_prefill*.so; never JIT")
    parser.add_argument("--reference", type=Path, help="CPU tensor artifact from a roomy baseline run of this case")
    parser.add_argument("--expect", choices=("pass", "oom"), default="pass")
    args = parser.parse_args()
    maintenance_guard(args.maintenance_window)
    unused_gpus_guard()
    if args.output.exists():
        raise RuntimeError("Use a new output directory; never overwrite a baseline")
    if args.operation == "sparse":
        if args.extension is None:
            raise ValueError("--extension is required; this test never builds kernels")
        ext_path = args.extension.resolve()
        if not ext_path.is_relative_to(REPO.parent) or ext_path.suffix != ".so":
            raise ValueError("Extension must be a candidate .so")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Actual deployed selector parameters, not upstream defaults (B is 64).
    os.environ.update(FA2SM70_SEL_GEMM="1", FA2SM70_SEL_ROWNORM="1",
                      FA2SM70_SEL_RC="96", FA2SM70_SEL_STRAT="8", FA2SM70_SPARSE_MASS="0.85")
    import torch  # Only after all maintenance/resource checks.
    torch.manual_seed(1729)
    torch.cuda.set_device(0)
    args.output.mkdir(parents=True)
    report = {"status": "started", "operation": args.operation, "physical_gpu": args.gpu,
              "rows": args.rows, "context": args.context,
              "headroom_mib": args.headroom_mib, "graph_pool_mib": args.graph_pool_mib,
              "pressure_model": "PyTorch allocator cap, NOT a replay of driver fragmentation",
              "shape_origin": "synthetic compatible allocation sizes; incident shape was not logged",
              "expected": args.expect, "epoch": time.time()}
    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
    def snapshot():
        stats = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        return {"driver_free": free, "driver_total": total,
                "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "inactive_split": stats.get("inactive_split_bytes.all.current"),
                "allocation_retries": stats.get("num_alloc_retries"), "ooms": stats.get("num_ooms")}
    save()
    try:
        if args.operation == "sparse":
            spec = importlib.util.spec_from_file_location(ext_path.name.split(".")[0], ext_path)
            extension = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(extension)
            report["extension"] = str(ext_path)
            with ext_path.open("rb") as binary:
                report["extension_sha256"] = hashlib.file_digest(binary, "sha256").hexdigest()
            bs, hkv, h, d = 16384, 2, 12, 256
            pages = math.ceil(args.context / bs)
            q = torch.randn(args.rows, h, d, dtype=torch.float16, device="cuda")
            k = torch.randint(-32, 33, (pages, bs, hkv, d), dtype=torch.int8, device="cuda")
            scales = torch.full((pages * bs * hkv,), 1 / 32, dtype=torch.float32, device="cuda")
            # Nontrivial physical page order; the incident used no precomputed kbar pool.
            table = torch.arange(pages - 1, -1, -1, dtype=torch.int32, device="cuda")
            def operation():
                return extension.sparse_select(q, k, scales, table, args.context, d**-.5, .02,
                                               64, 128, 16, 2, 8, 2, None, 0., .5)
            report["score_tensor_bytes"] = math.ceil(args.rows / 128) * 2 * 96 * math.ceil(args.context / 64) * 4
        else:
            # TP2 shard: 2*intermediate_size/TP = 17408 output columns.
            x = torch.randn(args.rows, 5120, dtype=torch.float16, device="cuda")
            weight = torch.randn(17408, 5120, dtype=torch.float16, device="cuda") * .01
            def operation():
                return [torch.nn.functional.linear(x, weight)]
            report["linear_output_bytes"] = args.rows * 17408 * 2

        # Warm handles/kernels before imposing the cap; exclude compilation and inputs.
        warm = operation()
        torch.cuda.synchronize()
        del warm
        graph = graph_tensor = None
        if args.graph_pool_mib:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_tensor = torch.empty(args.graph_pool_mib * MIB, dtype=torch.uint8, device="cuda")
                graph_tensor.fill_(37)
            torch.cuda.current_stream().wait_stream(stream)
            graph.replay()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        before = snapshot()
        # Use reserved baseline, including retained graph pool, not allocated+reserved.
        cap = before["reserved"] + args.headroom_mib * MIB
        if cap >= before["driver_total"]:
            raise RuntimeError("Invalid synthetic cap: exceeds device capacity")
        torch.cuda.set_per_process_memory_fraction(cap / before["driver_total"], 0)
        torch.cuda.reset_peak_memory_stats()
        report.update(before=before, allocator_cap_bytes=cap)
        save()
        outcome = "pass"
        values = None
        try:
            values = operation()
            torch.cuda.synchronize()
        except torch.OutOfMemoryError as exc:
            outcome = "oom"
            report["oom_message"] = str(exc)
        report.update(outcome=outcome, after=snapshot())
        # Diagnostic snapshot only after an operation OOM, no retry hiding it.
        if outcome == "oom":
            (args.output / "allocator-at-oom.json").write_text(json.dumps(torch.cuda.memory_snapshot()))
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        if values is not None:
            cpu = [v.detach().cpu() for v in values]
            if args.operation == "sparse":
                counts, indices = cpu[:2]
                nb = math.ceil(args.context / 64)
                assert bool(((counts >= 0) & (counts <= nb)).all())
                for tile in range(counts.shape[0]):
                    for head in range(counts.shape[1]):
                        selected = indices[tile, head, :int(counts[tile, head])]
                        assert bool(((selected >= 0) & (selected < nb)).all())
            assert all(bool(torch.isfinite(v).all()) for v in cpu)
            metadata = {"operation": args.operation, "rows": args.rows,
                        "context": args.context if args.operation == "sparse" else None,
                        "seed": 1729}
            if args.reference:
                ref = torch.load(args.reference, map_location="cpu", weights_only=True)
                assert ref["metadata"] == metadata, "Reference shape/seed mismatch"
                assert len(ref["values"]) == len(cpu)
                assert all(torch.equal(a, b) for a, b in zip(ref["values"], cpu)), "Numerical/selection drift"
                report["reference_exact"] = True
            torch.save({"metadata": metadata, "values": cpu}, args.output / "outputs.pt")
        if graph is not None:
            graph.replay()
            torch.cuda.synchronize()
            assert int(graph_tensor[0].item()) == 37 and int(graph_tensor[-1].item()) == 37
        assert outcome == args.expect, f"Expected {args.expect}, got {outcome}"
        report["status"] = "expected_oom_observed" if outcome == "oom" else "case_passed"
        report["release_acceptance"] = False  # Needs full-model tests, not just this operation.
        save()
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
