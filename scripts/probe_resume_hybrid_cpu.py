#!/usr/bin/env python3
"""Exercise admission against real split attention/Mamba allocators on CPU."""
import argparse
import json
import os
from pathlib import Path

if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise RuntimeError('CPU only')
os.environ.update(FA2SM70_SPLIT_POOLS='1',FA2SM80_GDN_BLOCKS='49')
import torch
from tests.v1.core.utils import create_requests
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.resume_admission import can_resume_with_decode_headroom
from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,MambaSpec,FullAttentionSpec


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expect-admission',action='store_true')
    args=parser.parse_args()
    config=KVCacheConfig(num_blocks=32,kv_cache_tensors=[],kv_cache_groups=[
        KVCacheGroupSpec(['gdn'],MambaSpec(block_size=16384,shapes=((1,),),dtypes=(torch.float32,),
                                        mamba_cache_mode='all',num_speculative_blocks=4)),
        KVCacheGroupSpec(['attn'],FullAttentionSpec(block_size=16384,num_kv_heads=1,
                                                 head_size=1,dtype=torch.float32))])
    manager=KVCacheManager(config,max_model_len=262144,hash_block_size=16384,enable_caching=False)
    pools=manager.coordinator.block_pools
    held=(pools[0].get_new_blocks(35),pools[1].get_new_blocks(14))
    request=create_requests(1,num_tokens=245760,max_tokens=2048,block_size=16384)[0]
    before=[p.get_num_free_blocks() for p in pools]
    def preview():
        return can_resume_with_decode_headroom(manager,request,
            new_computed_blocks=manager.empty_kv_cache_blocks,num_new_computed_tokens=0,
            num_external_computed_tokens=0,num_lookahead_tokens=4,num_encoder_tokens=0,
            num_new_tokens=8192,effective_lookahead_tokens=0)
    admitted=preview()
    after=[p.get_num_free_blocks() for p in pools]
    extra=pools[0].get_new_blocks(9)
    denied_gdn=not preview()
    pools[0].free_blocks(extra)
    extra=pools[1].get_new_blocks(2)
    denied_attn=not preview()
    pools[1].free_blocks(extra)
    all_chunks_fit=True;chunks=0
    while request.num_computed_tokens<request.num_tokens:
        count=min(8192,request.num_tokens-request.num_computed_tokens)
        if manager.allocate_slots(request,count,num_lookahead_tokens=4) is None:
            all_chunks_fit=False;break
        request.num_computed_tokens+=count;chunks+=1
    decode_fits=manager.allocate_slots(request,1,num_lookahead_tokens=4) is not None
    result={'admitted':admitted,'all_recompute_chunks_fit':all_chunks_fit,
            'decode_transition_fits':decode_fits,'chunks':chunks,'before':before,'after_preview':after,
            'computed_tokens':request.num_computed_tokens,
            'denied_insufficient_gdn':denied_gdn,'denied_insufficient_attention':denied_attn}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert before==after,'Preview changed ownership'
    assert denied_gdn and denied_attn,'Exhausted pool was accepted'
    assert all_chunks_fit and decode_fits,'Fixture must be feasible'
    if args.expect_admission: assert admitted,'Rolling GDN state was treated as retained historical KV'


if __name__=='__main__':main()
