# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity check for a preempted request's recompute-to-decode transition."""


def can_resume_with_decode_headroom(
    manager,
    request,
    *,
    new_computed_blocks,
    num_new_computed_tokens: int,
    num_external_computed_tokens: int,
    num_lookahead_tokens: int,
    num_encoder_tokens: int,
) -> bool:
    """Check every real pool, without allocating or changing block ownership.

    Admitting only the first recompute chunk can repeatedly consume GPU work
    when the known sequence plus the next decode/lookahead cannot fit. Pending
    async outputs belong to the future known sequence too. This does not reserve
    all requested max_tokens or promise capacity for arbitrary future growth.
    """
    main_tokens = min(
        request.num_tokens + request.num_output_placeholders + 1,
        manager.max_model_len,
    )
    slot_tokens = min(main_tokens + num_lookahead_tokens, manager.max_model_len)
    manager.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=slot_tokens,
        new_computed_blocks=new_computed_blocks.blocks,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=(request.num_computed_tokens
                               + num_new_computed_tokens
                               + num_external_computed_tokens),
        num_tokens_main_model=main_tokens,
    )
    # The fork has separate attention/GDN pools. A sum across pools, or a check
    # of only manager.block_pool, would accept an exhausted individual group.
    return manager.coordinator.has_free_blocks_for()
