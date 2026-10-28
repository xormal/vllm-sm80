# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""Storage ownership for serialized exchange calls; no CUDA dependency.

Growth may release eager-only storage. Captured addresses must survive until the
graphs are destroyed; without a graph-owner callback we retain those generations.
This is not an eviction policy or a workspace for concurrent independent streams.
"""


class CapturedWorkspace:
    def __init__(self):
        self.buffer = None
        self.capacity = 0
        self.captured = False
        self.retired = []

    def get(self, size, allocate, *, capturing=False):
        if size < 0:
            raise ValueError("workspace size must be nonnegative")
        if self.buffer is None or size > self.capacity:
            # Allocate first: failure must leave the previous generation intact.
            replacement = allocate(size)
            if self.captured:
                self.retired.append(self.buffer)
            self.buffer = replacement
            self.capacity = size
            self.captured = False
        self.captured |= capturing
        return self.buffer
