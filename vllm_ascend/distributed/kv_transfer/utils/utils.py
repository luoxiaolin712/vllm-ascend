from __future__ import annotations

import math
import os
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.parallel_state import get_p_tp_group

MAX_HCCL_REGISTER_REGIONS = 256
REGISTER_MERGE_GAP_BYTES = 4096


def kv_alltoall_and_rearrange(pd_tp_ratio: int, key: torch.Tensor, value: torch.TensorType):
    if pd_tp_ratio <= 1:
        return None, None
    elif key is None or value is None:
        raise ValueError("key or value is None")
    k_output = alltoall_and_rearrange(pd_tp_ratio, key)
    v_output = alltoall_and_rearrange(pd_tp_ratio, value)
    return k_output, v_output


def alltoall_and_rearrange(tp_ratio: int, input_tensor: torch.Tensor):
    num_kv_heads = input_tensor.size(1)
    output_tensor = torch.zeros_like(input_tensor)
    dist.all_to_all_single(output_tensor, input_tensor, group=get_p_tp_group().device_group)
    input_tensor = 0
    result = rearrange_output(output_tensor, tp_ratio, num_kv_heads)
    output_tensor = 0
    return result


def rearrange_output(base_output: torch.Tensor, cut_num: int, num_kv_heads: int):
    size_0 = base_output.size(0)
    if size_0 % cut_num != 0:
        raise ValueError(f"The size of dim 0 [{size_0}] must be divisible by the cut_num [{cut_num}]")
    chunk_size = size_0 // cut_num
    reshaped = base_output.view(cut_num, chunk_size, -1)
    transposed = reshaped.transpose(0, 1)
    return transposed.contiguous().view(size_0, num_kv_heads, -1)


def align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def get_transfer_timeout_value():
    ascend_transfer_timeout = os.getenv("ASCEND_TRANSFER_TIMEOUT", "")
    if len(ascend_transfer_timeout) > 0:
        return int(ascend_transfer_timeout)
    hccl_rdma_timeout = int(os.getenv("HCCL_RDMA_TIMEOUT", "20"))  # type: ignore
    hccl_rdma_retry_cnt = int(os.getenv("HCCL_RDMA_RETRY_CNT", "7"))  # type: ignore
    return int((4.096 * (2**hccl_rdma_timeout)) * hccl_rdma_retry_cnt // 1000 + 3000)


@dataclass
class parallel_info:
    tp_size: int
    pcp_size: int
    dcp_size: int
    use_mla: bool
    pd_head_ratio: int


def get_cp_group(tp: int, heads: int, dcp: int):
    # Partition the second dimension of [pcp][head_group][dcp] to obtain a complete head group
    # head_group is all blocks for request in the same head
    # tp8 dcp2 heads4 return[[0,1,2,3]]
    # tp8 dcp1 heads4 return[[0,2,4,6],[1,3,5,7]]
    step = tp // heads
    if step == 0:
        return [[i for i in range(tp // dcp)]]
    else:
        return [
            set([k // dcp for h in range(heads) for k in range(h * step + i * dcp, h * step + (i + 1) * dcp)])
            for i in range(step // dcp)
        ]


def context_parallel_parameters_check(
    remote_pcp_size: int,
    remote_dcp_size: int,
    p_parallel_info: parallel_info,
    d_parallel_info: parallel_info,
    total_num_kv_heads: int,
):
    # Check whether the pcp–dcp ratio is supported
    assert (p_parallel_info.pcp_size * p_parallel_info.dcp_size) % (remote_pcp_size * remote_dcp_size) == 0
    if not p_parallel_info.use_mla:
        p_node_heads_per_rank = math.ceil(total_num_kv_heads / p_parallel_info.tp_size)
        d_node_heads_per_rank = math.ceil(total_num_kv_heads / d_parallel_info.dcp_size)
        assert d_node_heads_per_rank % p_node_heads_per_rank == 0


def get_tp_rank_head_mapping(num_key_value_heads: int, tp_size: int):
    # Get the head_idx corresponding to the tp_rank, {tp_rank:[head_indx]}
    mapping = {}
    if tp_size <= num_key_value_heads:
        if num_key_value_heads % tp_size != 0:
            raise ValueError(f"Number of heads ({num_key_value_heads}) cannot be evenly divided by TP ({tp_size}).")

        heads_per_rank = num_key_value_heads // tp_size

        for rank in range(tp_size):
            start_idx = rank * heads_per_rank
            end_idx = start_idx + heads_per_rank
            mapping[rank] = list(range(start_idx, end_idx))
    else:
        if tp_size % num_key_value_heads != 0:
            raise ValueError(f"Number of heads ({num_key_value_heads}) cannot be evenly divided by TP ({tp_size}).")
        ranks_per_head = tp_size // num_key_value_heads
        for rank in range(tp_size):
            head_idx = rank // ranks_per_head
            mapping[rank] = [head_idx]
    return mapping


def get_head_group_mapping(num_key_value_heads: int, tp_size: int, num_groups: int, select_cp_group: list[int]):
    # Get the mapping dictionary, where the key is head_group_rank and the value is head_idx
    if tp_size % num_groups != 0:
        raise ValueError(
            f"Total number of devices ({tp_size}) cannot be divided by the number of groups ({num_groups})."
        )
    ranks_per_group = tp_size // num_groups
    tp_mapping = get_tp_rank_head_mapping(num_key_value_heads, tp_size)
    group_mapping = {}
    for group_rank in range(num_groups):
        if group_rank in select_cp_group:
            start_rank = group_rank * ranks_per_group
            end_rank = start_rank + ranks_per_group
            heads_set = set()

            for rank in range(start_rank, end_rank):
                heads_set.update(tp_mapping[rank])
            group_mapping[group_rank] = sorted(list(heads_set))
    return group_mapping


def get_local_remote_block_port_mappings(
    to_trans_idx: int,
    p_parallel_info: parallel_info,
    d_parallel_info: parallel_info,
    d_hosts: list[str],
    d_port: int,
    selected_p_cp_group: list[int],
    selected_d_cp_group: list[int],
    prompt_len: int,
    block_size: int,
    req_meta,
    total_num_kv_heads: int,
    req_id: str,
):
    p_head_group_size = p_parallel_info.tp_size // p_parallel_info.dcp_size
    d_head_group_size = d_parallel_info.tp_size // d_parallel_info.dcp_size
    world_size = d_parallel_info.pcp_size * d_head_group_size * d_parallel_info.dcp_size
    # Compute which logic_block_idx corresponds to each tp_rank
    p_rank_block_mapping: list[list[list[list[int]]]] = [
        [[[] for _ in range(p_parallel_info.dcp_size)] for _ in range(p_head_group_size)]
        for _ in range(p_parallel_info.pcp_size)
    ]
    for logic_block_idx in range(to_trans_idx):
        pcp_rank = (logic_block_idx // p_parallel_info.dcp_size) % p_parallel_info.pcp_size
        dcp_rank = logic_block_idx % p_parallel_info.dcp_size
        for p_head_group_rank in range(p_head_group_size):
            if p_head_group_rank in selected_p_cp_group:
                p_rank_block_mapping[pcp_rank][p_head_group_rank][dcp_rank].append(logic_block_idx)

    # Find the remote device that holds the logic_block_idx
    d_block_rank_mapping: dict[int, dict[int, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for logic_block_idx in range(to_trans_idx):
        pcp_rank = (logic_block_idx // d_parallel_info.dcp_size) % d_parallel_info.pcp_size
        for d_head_group_rank in range(d_head_group_size):
            if d_head_group_rank in selected_d_cp_group:
                dcp_rank = logic_block_idx % d_parallel_info.dcp_size
                world_rank = (
                    pcp_rank * d_head_group_size * d_parallel_info.dcp_size
                    + d_head_group_rank * d_parallel_info.dcp_size
                    + dcp_rank
                )
                world_size = d_parallel_info.pcp_size * d_head_group_size * d_parallel_info.dcp_size
                host = d_hosts[(len(d_hosts) * world_rank) // world_size]
                port = d_port + world_rank
                block_idx = (logic_block_idx - (pcp_rank * d_parallel_info.pcp_size + dcp_rank)) // (
                    d_parallel_info.pcp_size * d_parallel_info.dcp_size
                )
                d_block_rank_mapping[logic_block_idx][d_head_group_rank] = {
                    "pcp_rank": pcp_rank,
                    "dcp_rank": dcp_rank,
                    "host": host,
                    "port": port,
                    "block_idx": block_idx,
                }
    # Get how many times each device should receive done_single for this request
    d_trans_count_mapping = {}
    trans_block_size = math.ceil(prompt_len / block_size)  # Total number of blocks
    transed_block_size = math.ceil(req_meta.remote_cache_tokens / block_size)  # Number of prefix cache hit blocks
    d_cp_size = d_parallel_info.pcp_size * d_parallel_info.dcp_size
    for d_pcp_rank in range(d_parallel_info.pcp_size):
        for d_head_group_rank in range(d_head_group_size):
            for d_dcp_rank in range(d_parallel_info.dcp_size):
                if trans_block_size >= (p_parallel_info.pcp_size * p_parallel_info.dcp_size):
                    trans_count = (p_parallel_info.pcp_size * p_parallel_info.dcp_size) // d_cp_size
                else:
                    current_rank_idx = d_pcp_rank * d_parallel_info.dcp_size + d_dcp_rank
                    total_global_blocks = transed_block_size + trans_block_size

                    target_total_count = total_global_blocks // d_cp_size
                    if current_rank_idx < (total_global_blocks % d_cp_size):
                        target_total_count += 1

                    prev_processed_count = transed_block_size // d_cp_size
                    if current_rank_idx < (transed_block_size % d_cp_size):
                        prev_processed_count += 1

                    trans_count = target_total_count - prev_processed_count
                world_rank = (
                    d_pcp_rank * d_head_group_size * d_parallel_info.dcp_size
                    + d_head_group_rank * d_parallel_info.dcp_size
                    + d_dcp_rank
                )
                host = d_hosts[(len(d_hosts) * world_rank) // world_size]
                port = d_port + world_rank
                d_trans_count_mapping[(host, port)] = trans_count * p_parallel_info.pd_head_ratio

    # Compute the mapping between local and remote head_group_rank
    p_tp_rank_head_mapping = get_head_group_mapping(
        total_num_kv_heads, p_parallel_info.tp_size, p_head_group_size, selected_p_cp_group
    )
    d_tp_rank_head_mapping = get_head_group_mapping(
        total_num_kv_heads, d_parallel_info.tp_size, d_head_group_size, selected_d_cp_group
    )
    head_to_d_groups = defaultdict(set)
    for d_rank, heads in d_tp_rank_head_mapping.items():
        for head in heads:
            head_to_d_groups[head].add(d_rank)
    pd_head_mapping = {}
    for p_rank, p_heads in p_tp_rank_head_mapping.items():
        target_d_ranks = set()
        for head in p_heads:
            if head in head_to_d_groups:
                target_d_ranks.update(head_to_d_groups[head])
            else:
                logger.info("Warning: Head %s exists in P but not in D mapping.", head)
        pd_head_mapping[p_rank] = sorted(list(target_d_ranks))
    logger.debug(
        "MooncakeLayerwiseConnector _get_kv_split_metadata req_id=%r "
        "P-side logic_block to rank mapping: %s, "
        "D-side logic_block to rank mapping: %s, "
        "P&D head_group_rank mapping: %s",
        req_id,
        p_rank_block_mapping,
        d_block_rank_mapping,
        pd_head_mapping,
    )
    return p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping


def get_transfer_mappings(
    p_rank_block_mapping: list[list[list[list[int]]]],
    d_block_rank_mapping: dict[int, dict[int, dict[str, Any]]],
    pd_head_mapping: dict[int, set],
    d_trans_count_mapping: dict[tuple[str, int], int],
    req_meta,
    block_group_idx: int,
    p_parallel_info: parallel_info,
    req_id: str,
    transed_idx: int,
    to_trans_idx: int,
    tp_rank: int,
    pcp_rank: int,
    dcp_rank: int,
):
    transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
    p_head_group_rank = (tp_rank - dcp_rank) // p_parallel_info.dcp_size
    p_block_idxs: list[int] = p_rank_block_mapping[pcp_rank][p_head_group_rank][dcp_rank]
    p_block_ids = req_meta.local_block_ids[block_group_idx]
    d_block_ids = req_meta.remote_block_ids[block_group_idx]
    for p_block_idx, logic_block_idx in enumerate(p_block_idxs):
        if logic_block_idx < transed_idx or logic_block_idx >= to_trans_idx:
            continue
        for d_head_group_rank in pd_head_mapping[p_head_group_rank]:
            p_block_id = p_block_ids[p_block_idx]
            remote_host = d_block_rank_mapping[logic_block_idx][d_head_group_rank]["host"]
            remote_port = d_block_rank_mapping[logic_block_idx][d_head_group_rank]["port"]
            d_block_idx = d_block_rank_mapping[logic_block_idx][d_head_group_rank]["block_idx"]
            d_block_id = d_block_ids[d_block_idx]
            if (remote_host, remote_port) not in transfer_mappings:
                transfer_mappings[(remote_host, remote_port)] = {
                    "local_block_ids": [],
                    "remote_block_ids": [],
                    "trans_count": 0,
                }
            transfer_mappings[(remote_host, remote_port)]["local_block_ids"].append(p_block_id)
            transfer_mappings[(remote_host, remote_port)]["remote_block_ids"].append(d_block_id)
    for (host, port), block_dict in transfer_mappings.items():
        block_dict["trans_count"] = d_trans_count_mapping[(host, port)]
    logger.debug("MooncakeLayerwiseConnector Request %s transfer tasks: %s", req_id, transfer_mappings)
    return transfer_mappings


@dataclass
class RegisterRange:
    start: int
    end: int


@dataclass
class RegisterRegions:
    ptrs: list[int]
    lengths: list[int]
    logical_tensor_count: int | None = None
    logical_total_bytes: int | None = None

    @property
    def registered_bytes(self) -> int:
        return sum(self.lengths)


def iter_kv_cache_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Flatten kv_caches into tensors without materializing new tensors."""
    if obj is None:
        return

    if isinstance(obj, torch.Tensor):
        yield obj
        return

    if isinstance(obj, (tuple, list)):
        for item in obj:
            yield from iter_kv_cache_tensors(item)
        return

    if isinstance(obj, dict):
        for item in obj.values():
            yield from iter_kv_cache_tensors(item)
        return


def tensor_storage_key(tensor: torch.Tensor) -> int:
    """Return a stable grouping key for tensors sharing the same storage.

    Do NOT use this key as the register address directly. For aligned KV cache
    views, tensor.untyped_storage().data_ptr() may point to the original raw
    allocation, whose address can be unaligned. We only use it to group views.
    """
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        try:
            return tensor.storage().data_ptr()
        except Exception:
            return tensor.data_ptr()


def collect_storage_merged_register_regions(
    kv_caches: dict[str, Any],
) -> RegisterRegions:
    """Collect HCCL/Mooncake register regions with storage-aware merging.

    Metadata should still use each logical tensor's own data_ptr().
    register_buffer should use the merged memory ranges returned here.
    """
    ranges_by_storage: OrderedDict[int, list[RegisterRange]] = OrderedDict()
    logical_tensor_count = 0
    logical_total_bytes = 0

    for tensor in iter_kv_cache_tensors(kv_caches):
        if tensor is None or tensor.numel() == 0:
            continue

        if not tensor.is_contiguous():
            logger.warning(
                "Mooncake register_buffer got a non-contiguous KV cache "
                "tensor: shape=%s, dtype=%s, data_ptr=%s. "
                "Registration will use logical numel * element_size.",
                tuple(tensor.shape),
                tensor.dtype,
                hex(tensor.data_ptr()),
            )

        nbytes = tensor.nbytes
        start = tensor.data_ptr()
        end = start + nbytes
        storage_key = tensor_storage_key(tensor)

        logical_tensor_count += 1
        logical_total_bytes += nbytes

        ranges_by_storage.setdefault(storage_key, []).append(RegisterRange(start, end))

    register_ptrs: list[int] = []
    register_lengths: list[int] = []

    for ranges in ranges_by_storage.values():
        ranges.sort(key=lambda r: r.start)

        merged_start = ranges[0].start
        merged_end = ranges[0].end

        for region in ranges[1:]:
            if region.start <= merged_end + REGISTER_MERGE_GAP_BYTES:
                merged_end = max(merged_end, region.end)
            else:
                register_ptrs.append(merged_start)
                register_lengths.append(merged_end - merged_start)
                merged_start = region.start
                merged_end = region.end

        register_ptrs.append(merged_start)
        register_lengths.append(merged_end - merged_start)

    return RegisterRegions(
        ptrs=register_ptrs,
        lengths=register_lengths,
        logical_tensor_count=logical_tensor_count,
        logical_total_bytes=logical_total_bytes,
    )


def validate_register_region_count(regions: RegisterRegions) -> None:
    region_count = len(regions.ptrs)
    if region_count <= MAX_HCCL_REGISTER_REGIONS:
        return

    detail = f"registered_bytes={regions.registered_bytes}"
    if regions.logical_tensor_count is not None:
        detail += f", logical_tensors={regions.logical_tensor_count}, logical_bytes={regions.logical_total_bytes}"

    raise RuntimeError(
        "Mooncake register_buffer region count "
        f"{region_count} exceeds HCCL per-process limit "
        f"{MAX_HCCL_REGISTER_REGIONS}. "
        "KV cache registration would fail. "
        f"{detail}. "
        "Please reduce KV cache allocation fragmentation or merge "
        "k/v/dsa/scale allocations further."
    )


def infer_cache_family_from_ratio(compress_ratio: int | None) -> str:
    if compress_ratio is None:
        return "default"
    if compress_ratio <= 1:
        return "c1"
    return f"c{compress_ratio}"


def infer_cache_family_ratio(cache_family: str | None) -> int:
    if not cache_family or not cache_family.startswith("c"):
        return 1
    ratio = cache_family[1:]
    return int(ratio) if ratio.isdigit() else 1


def get_cache_family_granularity(block_size: int, cache_family: str | None) -> int:
    return block_size * infer_cache_family_ratio(cache_family)


def _get_layer_compress_ratio(
    layer_name: str,
    compress_ratios: Sequence[int] | None,
    hf_config: Any | None = None,
) -> int | None:
    if compress_ratios is None:
        return None
    if getattr(hf_config, "model_type", None) == "deepseek_v4":
        from vllm_ascend.utils import extract_dsv4_layer_index, get_dsv4_compress_ratio

        return get_dsv4_compress_ratio(hf_config, extract_dsv4_layer_index(hf_config, layer_name))
    from vllm.model_executor.models.utils import extract_layer_index

    return compress_ratios[extract_layer_index(layer_name)]


def _get_group_spec_ratios(group: object) -> set[int | None]:
    kv_cache_spec = getattr(group, "kv_cache_spec", None)
    if kv_cache_spec is None:
        return set()
    kv_cache_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if kv_cache_specs is not None:
        return {getattr(spec, "compress_ratio", None) for spec in kv_cache_specs.values()}
    return {getattr(kv_cache_spec, "compress_ratio", None)}


def infer_group_cache_families(
    kv_cache_groups: Sequence[object] | None,
    compress_ratios: Sequence[int] | None,
    hf_config: Any | None = None,
) -> list[str]:
    if kv_cache_groups is None:
        return ["default"]

    families: list[str] = []
    for group in kv_cache_groups:
        spec_ratios = _get_group_spec_ratios(group)
        if len(spec_ratios) == 1:
            families.append(infer_cache_family_from_ratio(next(iter(spec_ratios))))
            continue
        if len(spec_ratios) > 1:
            families.append("mixed")
            continue

        layer_names = list(getattr(group, "layer_names", []))
        if compress_ratios is None or not layer_names:
            families.append("default")
            continue

        group_ratios = {_get_layer_compress_ratio(layer_name, compress_ratios, hf_config) for layer_name in layer_names}
        if len(group_ratios) == 1:
            families.append(infer_cache_family_from_ratio(next(iter(group_ratios))))
        else:
            logger.debug(
                "KV cache group has mixed layer compress ratios %s for layers %s; using mixed cache family.",
                sorted(group_ratios, key=lambda ratio: -1 if ratio is None else ratio),
                layer_names,
            )
            families.append("mixed")
    return families


def resolve_hf_configs(vllm_config: Any) -> tuple[Any, Sequence[int] | None]:
    """Resolve (hf_config, compress_ratios) from a vllm_config."""
    model_config = vllm_config.model_config
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", hf_text_config)
    resolved_hf_config = hf_text_config or hf_config
    compress_ratios = getattr(hf_text_config, "compress_ratios", None)
    if compress_ratios is None:
        compress_ratios = getattr(hf_config, "compress_ratios", None)
    return resolved_hf_config, compress_ratios


def uses_hybrid_kv_cache(vllm_config: Any, kv_cache_config: Any | None) -> bool:
    if kv_cache_config is None:
        return False
    if getattr(vllm_config.scheduler_config, "disable_hybrid_kv_cache_manager", False):
        return False
    return len(kv_cache_config.kv_cache_groups) > 1 and any(
        not isinstance(group.kv_cache_spec, FullAttentionSpec) for group in kv_cache_config.kv_cache_groups
    )


def infer_group_block_sizes(vllm_config: Any, kv_cache_config: Any | None, use_hybrid: bool) -> list[int]:
    if kv_cache_config is None or not use_hybrid:
        return [vllm_config.cache_config.block_size]

    block_sizes: list[int] = []
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        block_sizes.append(kv_cache_spec.block_size)
    return block_sizes


def infer_block_size_topology(
    vllm_config: Any,
    kv_cache_config: Any | None,
    use_hybrid: bool,
    pcp_size: int,
    dcp_size: int,
) -> tuple[list[int], list[int], int, int]:
    """Derive (original_block_size, grouped_block_size, hash_block_size, lcm_block_size)."""
    original_block_size = infer_group_block_sizes(vllm_config, kv_cache_config, use_hybrid)
    cp_scale = pcp_size * dcp_size
    grouped_block_size = [block_size * cp_scale for block_size in original_block_size]
    requested_hash_block_size = vllm_config.cache_config.prefix_match_unit
    if not isinstance(requested_hash_block_size, int):
        requested_hash_block_size = None
    hash_block_size = (
        requested_hash_block_size if requested_hash_block_size is not None else min(original_block_size)
    ) * cp_scale
    for group_block_size in grouped_block_size:
        assert group_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
    lcm_block_size = math.lcm(*grouped_block_size)
    return original_block_size, grouped_block_size, hash_block_size, lcm_block_size


def get_group_block_size(grouped_block_size: Sequence[int], group_id: int) -> int:
    if group_id >= len(grouped_block_size):
        return grouped_block_size[0]
    return grouped_block_size[group_id]


def get_group_family(families: Sequence[str], group_id: int) -> str:
    if group_id >= len(families):
        return "default"
    return families[group_id]


def get_effective_group_block_size(grouped_block_size: Sequence[int], families: Sequence[str], group_id: int) -> int:
    cache_family = get_group_family(families, group_id)
    return get_group_block_size(grouped_block_size, group_id) * max(infer_cache_family_ratio(cache_family), 1)


def infer_cache_transfer_granularity(grouped_block_size: Sequence[int], families: Sequence[str]) -> int:
    granularities = [math.lcm(*grouped_block_size)]
    for group_id in range(len(grouped_block_size)):
        granularities.append(
            get_cache_family_granularity(
                get_group_block_size(grouped_block_size, group_id),
                get_group_family(families, group_id),
            )
        )
    return math.lcm(*granularities)
