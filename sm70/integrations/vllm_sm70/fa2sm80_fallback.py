# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""Bound stateless projection/MLP intermediates without rechunking attention/GDN."""
from fa2sm70_clef_hook import row_window


def chunked_fallback(core, residual, rows, forward):
    """forward keeps the original modules, corrections, bias and TP reductions.

    Only final output and fp32 residual span the full step. No shared mutable
    scratch or asynchronous streams are introduced here. Chunk-dependent GEMM
    and TP dispatch can still change numerics/performance: this is opt-in.
    """
    if rows <= 0:
        raise ValueError("fallback chunk rows must be positive")
    if core.ndim != 2 or residual.ndim != 2 or core.shape[0] != residual.shape[0]:
        raise ValueError("fallback requires matching two-dimensional row batches")
    total = core.shape[0]
    if total <= rows:
        return forward(core, residual)
    output = residual_out = None
    for start in range(0, total, rows):
        end = min(start + rows, total)
        with row_window(start, end - start):
            part, part_residual = forward(core[start:end], residual[start:end])
        if output is None:
            output = part.new_empty((total, part.shape[1]))
            residual_out = part_residual.new_empty((total, part_residual.shape[1]))
        output[start:end].copy_(part)
        residual_out[start:end].copy_(part_residual)
        # No previous output chunk remains live while the next chunk computes.
        del part, part_residual
    return output, residual_out
