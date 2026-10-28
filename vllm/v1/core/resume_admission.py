# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity check for a preempted request's recompute-to-decode transition."""

from vllm.v1.kv_cache_interface import MambaSpec


def can_resume_with_decode_headroom(
    manager,
    request,
    *,
    new_computed_blocks,
    num_new_computed_tokens: int,
    num_external_computed_tokens: int,
    num_lookahead_tokens: int,
    num_encoder_tokens: int,
    num_new_tokens: int,
    effective_lookahead_tokens: int,
) -> bool:
    """Check every real pool, without allocating or changing block ownership.

    Admitting only the first recompute chunk can repeatedly consume GPU work
    when the known sequence plus the next decode/lookahead cannot fit. Pending
    async outputs belong to the future known sequence too. This does not reserve
    all requested max_tokens or promise capacity for arbitrary future growth.
    Full known-sequence capacity applies to retained KV; rolling GDN states
    are checked for the actual current chunk by their existing allocator.
    """
    main_tokens = min(
        request.num_tokens + request.num_output_placeholders + 1,
        manager.max_model_len,
    )
    slot_tokens = min(main_tokens + num_lookahead_tokens, manager.max_model_len)
    total_computed_tokens = (request.num_computed_tokens
                             + num_new_computed_tokens
                             + num_external_computed_tokens)
    manager.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=slot_tokens,
        new_computed_blocks=new_computed_blocks.blocks,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=total_computed_tokens,
        num_tokens_main_model=main_tokens,
    )
    per_group = list(manager.coordinator._last_per_group)
    for index, single in enumerate(manager.coordinator.single_type_managers):
        if isinstance(single.kv_cache_spec, MambaSpec):
            # GDN frees historical states as recompute advances. Reserving the
            # full known sequence here would unnecessarily delay a feasible
            # long request. Check its real current chunk; allocate_slots still
            # enforces rolling-state capacity on every subsequent step.
            chunk_main = total_computed_tokens + num_new_tokens
            chunk_slots = min(chunk_main + effective_lookahead_tokens,
                              manager.max_model_len)
            per_group[index] = single.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=chunk_slots,
                new_computed_blocks=new_computed_blocks.blocks[index],
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=chunk_main,
            )
    # The fork has separate attention/GDN pools. A sum across pools, or a check
    # of only manager.block_pool, would accept an exhausted individual group.
    return manager.coordinator.has_free_blocks_for(per_group)
