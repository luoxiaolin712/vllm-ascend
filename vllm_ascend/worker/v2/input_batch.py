# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/input_batch.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from dataclasses import asdict, dataclass

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin


class AscendInputBuffers(InputBuffers):
    """Input buffers for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_num_tokens,
            device,
        )
        del self.query_start_loc

        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        self.query_start_loc: torch.Tensor = torch.zeros(
            max_num_reqs + 2,
            dtype=torch.int32,
            device=device,
        )

        # Create seq_lens_cpu and seq_lens_np.
        # npu's attention backend still needs seq_lens on CPU side.
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        # seq_len_np and seq_lens_cpu share the same memory.
        # define seq_lens_np for easier calculation with numpy.
        self.seq_lens_np: np.ndarray = self.seq_lens_cpu.numpy()

        # Cross-layer (DSV4 compress) SFA workspace buffers.
        # These are output tensors filled by store_kv_block_metadata and
        # consumed by the Ascend attention backend.
        # NOTE: outputs are pre-allocated with the same length as slot_mapping
        # (per-token, not per-request), per the C++ kernel contract.
        self.group_len: torch.Tensor = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.group_key_idx: torch.Tensor = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=device
        )
        self.group_key_cache_idx: torch.Tensor = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=device
        )


@dataclass
class AscendInputBatch(InputBatch):
    """Input batch for Ascend NPUs."""

    # Create seq_lens_np.
    # npu's attention backend still needs seq_lens on CPU side.
    seq_lens_np: np.ndarray
    # attn_state is used to build attention metadata.
    attn_state: AscendAttentionState | None = None
    # Cross-layer (DSV4 compress) SFA buffers for the attention backend.
    group_len: torch.Tensor | None = None
    group_key_idx: torch.Tensor | None = None
    group_key_cache_idx: torch.Tensor | None = None
    # Per-token absolute positions on CPU for the DSA attention path.
    positions_cpu: torch.Tensor | None = None

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: AscendInputBuffers,
    ) -> "AscendInputBatch":
        """Override the make_dummy method to calculate seq_lens_np."""
        input_batch = InputBatch.make_dummy(
            num_reqs,
            num_tokens,
            input_buffers,
        )
        # Evenly distribute num_tokens across requests instead of dumping the
        # whole remainder on the last request.
        # The old distribution could make the last dummy request's seq_len
        # exceed max_model_len, causing attention kernels to read block-table
        # entries past the tensor end (garbage page IDs / illegal memory access).
        base_tokens = num_tokens // num_reqs
        num_extra = num_tokens % num_reqs
        input_buffers.seq_lens_np[: num_reqs - num_extra] = base_tokens
        input_buffers.seq_lens_np[num_reqs - num_extra : num_reqs] = base_tokens + 1
        # Pad for full CUDA graph mode.
        input_buffers.seq_lens_np[num_reqs:] = 0
        seq_lens_np = input_buffers.seq_lens_np[:num_reqs]
        # A dummy run for dp or memory profiling.
        # When dummy run for dp, num_tokens is set to 1,
        # so attn_state is set to DecodeOnly.
        # when dummy run for memory profiling,
        # attention metadata isn't needed,
        # we can also set attn_state to AscendAttentionState.DecodeOnly.
        # For mla, update cos/sin. Here is for _dummy_run.
        update_cos_sin(input_batch.positions)
        return cls(
            **asdict(input_batch),
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.DecodeOnly,
            group_len=input_buffers.group_len[:num_tokens],
            group_key_idx=input_buffers.group_key_idx[:num_tokens],
            group_key_cache_idx=input_buffers.group_key_cache_idx[:num_tokens],
            positions_cpu=None,
        )
