# SPDX-License-Identifier: Apache-2.0
"""One launch for uniform speculative convolution read/write metadata.

Only integer addresses are computed. Existing persistent output storage is
updated in place; no model arithmetic, tensor ownership or SSM tables change.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fill(table, ctx, seq, aligned, spec, out, write,
          N: tl.constexpr, INPUT_COLS: tl.constexpr, OUT_COLS: tl.constexpr,
          WIDTH: tl.constexpr, PAGE: tl.constexpr, PAD: tl.constexpr,
          TABLE_STRIDE: tl.constexpr, ALIGN_STRIDE: tl.constexpr,
          SPEC_STRIDE: tl.constexpr, OUT_STRIDE: tl.constexpr,
          BLOCK: tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    active=row<N
    state=tl.load(spec+row*SPEC_STRIDE,mask=active,other=PAD)
    valid=active & (state!=PAD)
    c=tl.load(ctx+row,mask=active,other=0).to(tl.int64)
    s=tl.load(seq+row,mask=active,other=0).to(tl.int64)
    base=tl.maximum((c-1)//PAGE,0)
    end=tl.maximum((s-1)//PAGE,0)
    shift=tl.minimum(tl.maximum(end-base,0),WIDTH-1)
    source_col=base+col
    value=tl.load(table+row*TABLE_STRIDE+source_col,
                  mask=valid & (col<WIDTH) & (source_col<INPUT_COLS),other=PAD)
    tl.store(out+row*OUT_STRIDE+col,tl.where(valid,value,PAD),
             mask=(col<OUT_COLS) & ((col<WIDTH) | ~valid))
    target=tl.load(table+row*TABLE_STRIDE+base+shift,
                   mask=active & (base+shift<INPUT_COLS),other=PAD)
    actual=tl.load(aligned+row*ALIGN_STRIDE,mask=active,other=PAD)
    selected=tl.where(active & (target==actual),shift,0)
    tl.store(write+row,selected.to(tl.int32))


def fill_uniform_conv_metadata(table,ctx,seq,aligned,spec,out,write,
                               *,num_rows,block_size,pad_slot):
    """Return the written width, or None when the existing path is required."""
    tensors=(table,ctx,seq,aligned,spec,out,write)
    if not all(t.is_cuda and t.device==out.device for t in tensors):return None
    if any(t.dtype!=torch.int32 for t in (table,aligned,spec,out,write)):return None
    if any(t.dtype not in (torch.int32,torch.int64) for t in (ctx,seq)):return None
    if any(t.ndim!=2 or t.stride(1)!=1 for t in (table,aligned,spec,out)):return None
    if any(t.ndim!=1 or t.stride(0)!=1 for t in (ctx,seq,write)):return None
    n=int(num_rows);width=min(spec.shape[1],out.shape[1])
    if not 0<n<=min(table.shape[0],ctx.numel(),seq.numel(),aligned.shape[0],spec.shape[0],out.shape[0]):return None
    if width<1 or aligned.shape[1]<1 or write.numel()!=out.shape[0] or block_size<1:return None
    # mamba_get_block_table_tensor's align width is num_speculative_blocks+1.
    # The caller uses this fast path only when that equals spec.shape[1].
    _fill[(out.shape[0],)](table,ctx,seq,aligned,spec,out,write,
        n,table.shape[1],out.shape[1],width,block_size,pad_slot,
        table.stride(0),aligned.stride(0),spec.stride(0),out.stride(0),
        triton.next_power_of_2(out.shape[1]),num_warps=1)
    return width
