"""GPU0-only parity/capture/timing for integer convolution metadata.

Run only after stopping the candidate: this test creates its own CUDA context.
"""
import json
import os
from pathlib import Path
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'Only physical GPU0 is allowed'
import torch
from vllm.v1.attention.backends.conv_metadata import fill_uniform_conv_metadata

PAGE=16384
PAD=-1


def reference(table,ctx,seq,aligned,spec,out,write,n):
    width=min(spec.shape[1],out.shape[1])
    both=torch.stack((ctx[:n],seq[:n])).to(torch.int64)
    both=((both-1)//PAGE).clamp(min=0)
    offset=(both[1]-both[0]).clamp(min=0,max=width-1)
    gather=((ctx-1)//PAGE).clamp(min=0).unsqueeze(1)+torch.arange(width,device=ctx.device)
    copied=torch.gather(table,1,gather.long())
    out[:n,:width].copy_(copied[:n,:width])
    out[:n][spec[:n,0]==PAD]=PAD
    out[n:]=PAD;write[n:]=0
    offset=offset.clamp(min=0,max=width-1)
    target=copied[:n].gather(1,offset.unsqueeze(1)).squeeze(1)
    offset=torch.where(target==aligned[:n,0],offset,torch.zeros_like(offset))
    write[:n].copy_(offset.to(torch.int32))


def main():
    torch.manual_seed(17)
    device='cuda:0';cases=0
    for dtype in (torch.int32,torch.int64):
        for n in (1,2,8,16):
            for q in (1,4,5):
                for padded in (False,True):
                    # Non-contiguous rows, contiguous columns as in block tables.
                    table=torch.randint(0,49,(n,42),device=device,dtype=torch.int32)[:,:21]
                    points=[0,1,PAGE-5,PAGE-1,PAGE,PAGE+1,2*PAGE-1,49147,49152,262139]
                    ctx=torch.tensor([points[(i+cases)%len(points)] for i in range(n)],device=device,dtype=dtype)
                    seq=ctx+q
                    aligned=table.gather(1,(((seq-1)//PAGE).clamp(min=0).long()[:,None]+torch.arange(5,device=device)))
                    spec=aligned.clone()
                    if padded:spec[::2,0]=PAD
                    if cases%3==0:aligned[0,0]=999  # existing mismatch=>column-zero fallback
                    out=torch.randint(-100,100,(80,21),device=device,dtype=torch.int32)
                    write=torch.full((80,),99,device=device,dtype=torch.int32)
                    expected=out.clone();expected_write=write.clone()
                    reference(table,ctx,seq,aligned,spec,expected,expected_write,n)
                    width=fill_uniform_conv_metadata(table,ctx,seq,aligned,spec,out,write,
                            num_rows=n,block_size=PAGE,pad_slot=PAD)
                    assert width==5
                    assert torch.equal(out,expected),('table',dtype,n,q,padded)
                    assert torch.equal(write,expected_write),('write',dtype,n,q,padded)
                    cases+=1
    # Capture uses the same persistent addresses, then different boundary values.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fill_uniform_conv_metadata(table,ctx,seq,aligned,spec,out,write,num_rows=n,block_size=PAGE,pad_slot=PAD)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        fill_uniform_conv_metadata(table,ctx,seq,aligned,spec,out,write,num_rows=n,block_size=PAGE,pad_slot=PAD)
    ctx.fill_(49152);seq.fill_(49157)
    aligned.copy_(table.gather(1,(((seq-1)//PAGE).clamp(min=0).long()[:,None]+torch.arange(5,device=device))))
    spec.copy_(aligned)
    expected.copy_(out);expected_write.copy_(write)
    reference(table,ctx,seq,aligned,spec,expected,expected_write,n)
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(out,expected) and torch.equal(write,expected_write)
    timings=[]
    for n in (1,8):
        t,c,s,a,p=table[:n],ctx[:n],seq[:n],aligned[:n],spec[:n]
        def slow():reference(t,c,s,a,p,out,write,n)
        def fast():fill_uniform_conv_metadata(t,c,s,a,p,out,write,num_rows=n,block_size=PAGE,pad_slot=PAD)
        for name,fn in (('torch',slow),('fused',fast)):
            for _ in range(10):fn()
            torch.cuda.synchronize();started=time.perf_counter()
            for _ in range(200):fn()
            torch.cuda.synchronize()
            timings.append({'rows':n,'path':name,'wall_us_per_call':(time.perf_counter()-started)*1e6/200})
    result={'parity_cases':cases,'capture_replay':True,'timings':timings,
            'note':'Standalone metadata timing; not model throughput. Physical GPU0 only.'}
    output=Path(os.environ['CANDIDATE_CONV_META_RESULT'])
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
