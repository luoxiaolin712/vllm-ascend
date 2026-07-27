# MooncakeConnector Cross-Layer KV Cache 实现文档

## 1. 背景

### 1.1 问题描述

在 P/D 分离（Prefill/Decode Disaggregation）场景中，`MooncakeConnector` 通过 Mooncake 传输引擎在 P/D 节点间传输 KV Cache。原有实现中，每层的 KV Cache 独立分配：

```
layer_0: (k_tensor_0, v_tensor_0)  ← 独立分配，不连续
layer_1: (k_tensor_1, v_tensor_1)
...
layer_N: (k_tensor_N, v_tensor_N)
```

这带来两个问题：

1. **注册区域数量多**：每层两个 tensor，N 层共 2N 个区域需要向 HCCL/Mooncake 注册，存在注册数量上限风险
2. **传输效率低**：block-granularity 传输时，block `b` 的数据分散在 N 个不连续 tensor 中，无法单次完成跨层传输

### 1.2 Cross-Layer 布局

Cross-Layer KV Cache 将所有层的 KV 数据分配在一块连续内存中，物理布局（NPU 侧）为：

```
(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
```

内存中，同一 block 的所有层数据连续存放：

```
[block_0: layer_0_K, layer_0_V, layer_1_K, layer_1_V, ..., layer_N_K, layer_N_V]
[block_1: layer_0_K, layer_0_V, ...]
...
```

好处：
- 注册区域从 2N 个降为 1 个
- 可以用单次 RDMA 操作传输一个 block 的所有层数据
- `block_stride` 自动等于 per-block all-layer 数据大小

### 1.3 参考实现

参考 `/mnt/tx/ft_local/HCF-vLLM-Hetero/vllm/distributed/kv_transfer/kv_connector/v1/flexible_connector_v2.py` 中 `FlexibleConnectorForWorker.register_cross_layers_kv_cache` 的实现模式：通过 `stride_order` 逆置换恢复逻辑视图，按层切片后复用现有注册逻辑。

---

## 2. 涉及文件

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `vllm_ascend/attention/attention_v1.py` | 新增方法 | 为 `AscendAttentionBackend` 添加 `get_kv_cache_stride_order` |
| `vllm_ascend/worker/model_runner_v1.py` | 三处改动 | cross-layer 分配路径、spec 标志、注册逻辑 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | 三处改动 | 新增 `prefer_cross_layer_blocks`、`register_cross_layers_kv_cache` |

---

## 3. 核心数据结构

### 3.1 AscendAttentionBackend 的 KV Cache 形状

NPU 上的 KV Cache 维度定义：

```
逻辑形状（per-layer）: (2, num_blocks, block_size, num_kv_heads, head_size)
                        ^
                        K/V 分离维度（dim 0）
```

### 3.2 Cross-Layer tensor 的两种视图

**物理视图**（`allocate_uniform_kv_caches` 分配的实际内存布局）：

```
(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
```

`stride_order = (2, 0, 1, 3, 4, 5)` 描述从逻辑到物理的维度置换。

**逻辑视图**（通过 `permute(inv_order)` 恢复）：

```
(num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)
```

`inv_order = (1, 2, 0, 3, 4, 5)` 是 `stride_order` 的逆置换。

### 3.3 Per-Layer 切片

从逻辑视图中切片：

```python
per_layer = logical[i]       # (2, num_blocks, block_size, num_kv_heads, head_size)
k_view    = per_layer[0]     # (num_blocks, block_size, num_kv_heads, head_size)
v_view    = per_layer[1]     # (num_blocks, block_size, num_kv_heads, head_size)
```

切片后的 `k_view` / `v_view`：
- **shape** 与原有 per-layer tensor 完全相同，对 attention kernel 透明
- **是非连续 view**（strided tensor），`stride(0)` 等于物理内存中相邻 block 间距（包含 num_layers × 2 个单层 block）
- **data_ptr()** 指向该层第一个 block 的 K 数据起始地址

---

## 4. 实现详解

### 4.1 `attention_v1.py` — `get_kv_cache_stride_order`

**位置**：`AscendAttentionBackend` 类，`copy_blocks` 与 `get_supported_kernel_block_sizes` 之间。

```python
@staticmethod
def get_kv_cache_stride_order(
    include_num_layers_dimension: bool = False,
) -> tuple[int, ...]:
    # NPU logical shape (without layers): (2, num_blocks, block_size, num_kv_heads, head_size)
    # Physical layout for cross-layer KV transfer: num_blocks is outermost so
    # per-block all-layers data is contiguous.
    # With layers prepended (logical): (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)
    # Desired physical:                (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
    # stride_order[i] = logical index of physical position i
    if include_num_layers_dimension:
        return (2, 0, 1, 3, 4, 5)
    # Without layers: identity (no reordering needed for per-layer tensors)
    return (0, 1, 2, 3, 4)
```

**作用链**：

```
get_kv_cache_stride_order(True)  → (2, 0, 1, 3, 4, 5)
    → layered_stride_order[0] = 2 ≠ 0
        → indexes_kv_by_block_stride() → True
            → use_uniform_kv_cache() → True（在条件满足时）
                → allocate_uniform_kv_caches() 被调用
```

**`indexes_kv_by_block_stride` 判断逻辑**（上游 `backend.py`）：

```python
@classmethod
def indexes_kv_by_block_stride(cls) -> bool:
    single = cls.get_kv_cache_stride_order(False)   # (0,1,2,3,4)  — 5 维
    layered = cls.get_kv_cache_stride_order(True)   # (2,0,1,3,4,5) — 6 维
    if len(layered) != len(single) + 1:
        return False
    return layered[0] != 0  # 2 ≠ 0 → True
```

### 4.2 `model_runner_v1.py` — `get_kv_cache_spec` 设置 `indexes_kv_by_block_stride`

**位置**：`get_kv_cache_spec` 方法的 `isinstance(attn_module, Attention)` 分支。

**改动前**：

```python
elif isinstance(attn_module, Attention):
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        kv_cache_spec[layer_name] = spec
        attn_layer_names.add(layer_name)
```

**改动后**：

```python
elif isinstance(attn_module, Attention):
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        if isinstance(spec, AttentionSpec):
            backend = attn_module.get_attn_backend()
            indexes = backend.indexes_kv_by_block_stride()
            spec = replace(spec, indexes_kv_by_block_stride=indexes)
        kv_cache_spec[layer_name] = spec
        attn_layer_names.add(layer_name)
```

**传播路径**：

```
get_kv_cache_spec()
  └─ AttentionSpec(indexes_kv_by_block_stride=True)
       └─ get_kv_cache_groups()
            └─ KVCacheGroupSpec.kv_cache_spec
                 │  (FullAttentionSpec.merge 保留该字段)
                 └─ initialize_attn_backend()
                      └─ AttentionGroup.kv_cache_spec
                           └─ use_uniform_kv_cache()
                                └─ kv_cache_spec.indexes_kv_by_block_stride → True ✓
```

### 4.3 `model_runner_v1.py` — `initialize_kv_cache_tensors` 增加 cross-layer 路径

**改动前**：

```python
def initialize_kv_cache_tensors(self, kv_cache_config):
    kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
    kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
    ...
```

**改动后**：

```python
def initialize_kv_cache_tensors(self, kv_cache_config):
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

    # NPU 的 kernel_block_sizes 是 list[list[int]]，需展平为 list[int]
    flat_kernel_block_sizes = [
        sizes[0] if isinstance(sizes, list) else sizes
        for sizes in self.kernel_block_sizes
    ]
    if KVConnectorModelRunnerMixin.use_uniform_kv_cache(self.attn_groups):
        kv_caches, cross_layers_kv_cache, attn_backend = (
            KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
                kv_cache_config,
                self.attn_groups,
                self.cache_config.cache_dtype,
                self.device,
                flat_kernel_block_sizes,
            )
        )
        self.cross_layers_kv_cache = cross_layers_kv_cache
        self.cross_layers_attn_backend = attn_backend
    else:
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
    ...
```

**`kernel_block_sizes` 格式差异**：

| 版本 | 类型 | 示例 |
|---|---|---|
| GPU `gpu_model_runner.py` | `list[int]` | `[128]` |
| NPU `model_runner_v1.py` | `list[list[int]]` | `[[128]]` |

`allocate_uniform_kv_caches` 接收 `list[int]`，NPU 侧需展平。

**`use_uniform_kv_cache` 的全部门控条件**（上游 `kv_connector_model_runner_mixin.py`）：

| 条件 | 检查内容 |
|---|---|
| `has_kv_transfer_group()` | 配置了 KV Transfer |
| `prefer_cross_layer_blocks` | Connector 声明支持 cross-layer |
| `len(attn_groups) == 1 and len(attn_groups[0]) == 1` | 单一 attention group |
| `isinstance(kv_cache_spec, AttentionSpec)` | 非 Mamba/其他特殊 spec |
| `kv_cache_spec.indexes_kv_by_block_stride` | backend 支持 block stride 索引 |

所有条件都不满足时，自动回退原有路径，**零侵入**。

### 4.4 `model_runner_v1.py` — `initialize_kv_cache` 更新注册逻辑

**改动前**：

```python
if has_kv_transfer_group():
    get_kv_transfer_group().register_kv_caches(kv_caches)
```

**改动后**：

```python
if has_kv_transfer_group():
    kv_transfer_group = get_kv_transfer_group()
    if self.cross_layers_kv_cache is not None:
        assert self.cross_layers_attn_backend is not None
        kv_transfer_group.register_cross_layers_kv_cache(
            self.cross_layers_kv_cache, self.cross_layers_attn_backend
        )
    else:
        kv_transfer_group.register_kv_caches(kv_caches)
```

### 4.5 `mooncake_connector.py` — `prefer_cross_layer_blocks`

**位置**：`MooncakeConnector` 类，新增 property。

```python
@property
def prefer_cross_layer_blocks(self) -> bool:
    extra_config = self._vllm_config_ref.kv_transfer_config.kv_connector_extra_config or {}
    return str(extra_config.get("enable_cross_layers_blocks", "False")).lower() == "true"
```

通过 `kv_connector_extra_config` 中的 `enable_cross_layers_blocks` 字段控制，默认关闭。

**启用方式**（vllm 启动配置中）：

```json
{
  "kv_connector": "MooncakeConnector",
  "kv_connector_extra_config": {
    "enable_cross_layers_blocks": "true",
    ...
  }
}
```

**同时**，`MooncakeConnector.__init__` 新增 `self._vllm_config_ref = vllm_config`，使 property 能访问配置。

### 4.6 `mooncake_connector.py` — `MooncakeConnector.register_cross_layers_kv_cache`

**位置**：`MooncakeConnector` 类，Worker Side Methods 区域，`register_kv_caches` 之后。

```python
def register_cross_layers_kv_cache(
    self, kv_cache: torch.Tensor, attn_backend: type
) -> None:
    assert self.connector_worker is not None
    self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)
```

纯委托，与 `register_kv_caches` 的委托模式一致。

### 4.7 `mooncake_connector.py` — `MooncakeConnectorWorker.register_cross_layers_kv_cache`

这是核心实现。完整代码：

```python
def register_cross_layers_kv_cache(
    self, kv_cache: torch.Tensor, attn_backend: type
) -> None:
    # Step 1: 物理 tensor → 逻辑视图
    try:
        stride_order = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        inv_order = [stride_order.index(i) for i in range(len(stride_order))]
        logical = kv_cache.permute(*inv_order)
    except (AttributeError, NotImplementedError):
        logger.warning(...)
        logical = kv_cache

    # Step 2: 按层切片，构建 per-layer (k, v) 字典
    kv_caches_from_cross: dict[str, Any] = {}
    for i, kv_cache_tensor in enumerate(self.kv_cache_config.kv_cache_tensors):
        per_layer = logical[i]   # (2, num_blocks, block_size, num_kv_heads, head_size)
        k_view = per_layer[0]    # (num_blocks, block_size, num_kv_heads, head_size)
        v_view = per_layer[1]
        for layer_name in kv_cache_tensor.shared_by:
            kv_caches_from_cross[layer_name] = (k_view, v_view)

    # Step 3: 复用现有注册逻辑
    self.register_kv_caches(kv_caches_from_cross)
```

---

## 5. 传输正确性分析

### 5.1 `kv_caches_base_addr` 计算

在 `register_kv_caches` 中：

```python
for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
    self.kv_caches_base_addr[layer_idx].append(single_kv_cache.data_ptr())
```

对 cross-layer 切片：
- `k_view.data_ptr()` → 该层第一个 block 的 K 数据起始地址（物理内存中的真实位置）
- `v_view.data_ptr()` → 该层第一个 block 的 V 数据起始地址

### 5.2 `block_stride_per_addr` 计算

```python
self.block_stride_per_addr[layer_idx].append(
    single_kv_cache.stride(0) * single_kv_cache.element_size()
)
```

对 `k_view = logical[i][0]`（是 `kv_cache.permute(inv_order)` 的切片）：

**物理形状**：`kv_cache` 为 `(num_blocks, num_layers, 2, block_size, H, D)`  
**`k_view.stride(0)`** = permuted tensor 在 layer/kv 维度切片后，block 维度的 stride  
= 原始 `kv_cache` 中 `dim=0`（num_blocks）的 stride  
= `num_layers × 2 × block_size × H × D`（元素数）

因此：

```
block_stride = num_layers × 2 × block_size × num_kv_heads × head_size × element_size_bytes
```

这等于 **一个物理 block 中所有层 K+V 数据的总字节数**。

### 5.3 传输地址计算

在 `_transfer_kv_cache_all_groups` 中：

```python
src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
length = inner_block_len * len(local_block_id)
```

其中 `inner_block_len = block_len // tp_num_need_pulls`，`block_len = element_size × prod(block_shape)`。

对于 cross-layer 且单 TP（`tp_num_need_pulls=1`）：

| 参数 | 含义 | 值 |
|---|---|---|
| `base_addr` | `k_view.data_ptr()` | 该层 block_0 的 K 起始地址 |
| `block_stride` | 相邻 block 的字节间距 | `num_layers × 2 × block_size × H × D × elem_size` |
| `block_len` | 单层 K 的一个 block 字节数 | `block_size × H × D × elem_size` |
| `length` | 本次传输字节数 | `block_len × n_blocks` |

`src = base_addr + block_id × block_stride` 指向物理内存中 block `block_id` 的该层 K 数据。`length = block_len × n_blocks` 只传输该层的连续 K 数据。由于 cross-layer 物理布局中，同一 block 内各层连续，而不同 block 间距等于 `block_stride`，上述地址计算**精确对应**该层在 cross-layer tensor 中的位置。

### 5.4 注册区域合并效果

`collect_storage_merged_register_regions` 对所有 `(k_view, v_view)` 的 views，因为它们共享同一底层 storage（均来自同一 `kv_cache` 分配），会合并为**单一注册区域**。

```
注册前（per-layer）:  2N 个独立注册区域
注册后（cross-layer）: 1 个连续注册区域
```

---

## 6. 完整数据流

```
启动初始化阶段:

  MooncakeConnector.__init__
    └─ self._vllm_config_ref = vllm_config

  NPUModelRunner.get_kv_cache_spec()
    └─ Attention 模块 → FullAttentionSpec(indexes_kv_by_block_stride=True)
         └─ 通过 replace() 写入

  get_kv_cache_groups() → KVCacheConfig
    └─ KVCacheGroupSpec.kv_cache_spec.indexes_kv_by_block_stride = True

  NPUModelRunner.initialize_kv_cache(kv_cache_config)
    ├─ initialize_attn_backend(kv_cache_config)
    │    └─ AttentionGroup(kv_cache_spec.indexes_kv_by_block_stride=True)
    │
    └─ initialize_kv_cache_tensors(kv_cache_config)
         │
         ├─ use_uniform_kv_cache(attn_groups) ?
         │    ├─ has_kv_transfer_group()                     ✓ 已配置 MooncakeConnector
         │    ├─ prefer_cross_layer_blocks                   ✓ enable_cross_layers_blocks=true
         │    ├─ len(attn_groups)==1, len(attn_groups[0])==1 ✓ 单一 group
         │    └─ indexes_kv_by_block_stride                  ✓ AscendAttentionBackend 支持
         │
         ├─ [True]  allocate_uniform_kv_caches()
         │    ├─ 分配连续内存: shape=(num_blocks, num_layers, 2, block_size, H, D)
         │    ├─ permute(inv_order) → 逻辑视图 (num_layers, 2, num_blocks, block_size, H, D)
         │    └─ kv_caches[layer_name] = permuted[i]  (各层独立逻辑 view)
         │
         └─ [False] 原有逐层分配路径（不变）

  MooncakeConnector（调用路径）:
    initialize_kv_cache()
      └─ cross_layers_kv_cache is not None
           └─ kv_transfer_group.register_cross_layers_kv_cache(
                  cross_layers_kv_cache,      # 物理 tensor: (num_blocks, num_layers, 2, ...)
                  AscendAttentionBackend,
              )
                └─ MooncakeConnectorWorker.register_cross_layers_kv_cache()
                     ├─ get_kv_cache_stride_order(True) → (2,0,1,3,4,5)
                     ├─ inv_order = (1,2,0,3,4,5)
                     ├─ logical = kv_cache.permute(1,2,0,3,4,5)
                     │         shape: (num_layers, 2, num_blocks, block_size, H, D)
                     ├─ for i, tensor in kv_cache_config.kv_cache_tensors:
                     │    k_view = logical[i][0]  → (num_blocks, block_size, H, D), strided
                     │    v_view = logical[i][1]  → (num_blocks, block_size, H, D), strided
                     │    kv_caches[layer_name] = (k_view, v_view)
                     │
                     └─ register_kv_caches(kv_caches)
                          ├─ kv_caches_base_addr[layer_idx] = [k_view.data_ptr(), v_view.data_ptr()]
                          ├─ block_stride_per_addr[layer_idx] = [k_view.stride(0) * elem_size, ...]
                          │       ↑ = num_layers×2×block_size×H×D×elem_size（cross-layer 间距）
                          ├─ block_len_per_addr[layer_idx] = [block_size×H×D×elem_size, ...]
                          │       ↑ 单层单 block 大小
                          ├─ collect_storage_merged_register_regions()
                          │    └─ 所有 view 共享同一 storage → 合并为 1 个注册区域
                          └─ global_te.register_buffer([ptr], [total_size])

推理传输阶段（以 D 节点从 P 节点拉取为例）:

  KVCacheRecvingThread._transfer_kv_cache_all_groups(req_meta)
    └─ for layer_idx in layer_indices:
         src = remote_kv_caches_base_addr[layer_idx][0]
               + remote_block_id * remote_block_stride[layer_idx][0]
               ↑ 指向 P 侧 cross-layer tensor 中该层、该 block 的 K 数据
         dst = local_kv_caches_base_addr[layer_idx][0]
               + local_block_id * local_block_stride[layer_idx][0]
               ↑ 指向 D 侧 cross-layer tensor 中该层、该 block 的 K 数据
         length = block_len  (单层单 block 的 K 字节数)
         engine.batch_transfer_sync_read(session_id, [src], [dst], [length])
```

---

## 7. 激活条件

Cross-Layer 路径在以下所有条件**同时满足**时激活：

| 条件 | 配置位置 | 默认值 |
|---|---|---|
| 配置了 KV Transfer | `--kv-transfer-config` | 无 |
| `enable_cross_layers_blocks: true` | `kv_connector_extra_config` | `False` |
| 单一 attention group（无 Mamba 混合） | 模型结构决定 | — |
| 非 MLA 模型 | 模型结构决定 | — |
| `AscendAttentionBackend` | NPU 自动选择 | ✓ |

**任意条件不满足时自动回退**原有 per-layer 路径，零影响。

---

## 8. 不支持的场景

| 场景 | 原因 |
|---|---|
| DeepSeek MLA 模型 | `AscendMLAAttentionSpec` 不在 `Attention` 分支，`indexes_kv_by_block_stride` 未设置 |
| Mamba/SSM 混合模型 | `len(attn_groups) > 1`，`use_uniform_kv_cache` 返回 False |
| 未启用 `enable_cross_layers_blocks` | `prefer_cross_layer_blocks` 返回 False |
| 310P 模型 | 使用单独的 `model_runner_310p.py`，未修改 |
| 开启 `use_compress` 的模型 | compress 路径走独立分配逻辑，不经 cross-layer 路径 |

---

## 9. 与 FlexibleConnector 的差异

| 方面 | `FlexibleConnector`（NIXL） | `MooncakeConnector` |
|---|---|---|
| 注册机制 | NIXL `register_memory` | Mooncake `register_buffer` |
| cross-layer 标识名 | `"ALL_LAYERS"` 单 key | 按层名展开为多个 key |
| layer 计数来源 | `stride_order.index(0)` 读 tensor shape | 委托给 `register_kv_caches` 的层遍历 |
| 传输层数跟踪 | `num_kv_layers_uncalculated` 计数器 | 逐层 `GroupPull` 列表 |
| cross-layer 传输粒度 | unified/layerwise 双模式 | 始终 per-layer（但地址寻址到 cross-layer 位置） |
| 实现复用度 | 独立实现注册和传输 | 完全复用 `register_kv_caches` 及传输线程 |

`MooncakeConnector` 选择"**解构后复用**"策略而非"**新增独立路径**"，原因：
1. `register_kv_caches` 已有完整的 metadata 构建、HCCL 注册、线程启动逻辑
2. Mooncake 传输引擎以地址+stride+length 描述传输，strided view 的地址和 stride 天然正确
3. 无需感知 cross-layer 布局即可正确传输，接口改动最小

---

## 10. 测试建议

### 10.1 单元测试

```python
# tests/ut/test_cross_layer_kv.py

def test_register_cross_layers_kv_cache_decomposition():
    """验证 cross-layer tensor 被正确分解为 per-layer (k, v) tuples。"""
    num_layers, num_blocks, block_size, num_kv_heads, head_size = 32, 16, 128, 8, 128
    # 模拟 allocate_uniform_kv_caches 分配的物理 tensor
    # 物理形状: (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
    phys = torch.zeros(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
    # 写入 layer_i 的标记值
    for i in range(num_layers):
        phys[:, i, 0, :, :, :] = i * 2      # K
        phys[:, i, 1, :, :, :] = i * 2 + 1  # V
    
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    worker = ...  # 构造 MooncakeConnectorWorker
    worker.register_cross_layers_kv_cache(phys, AscendAttentionBackend)
    
    # 验证每层 k_view 的值正确
    for i, layer_name in enumerate(layer_names):
        k_view, v_view = worker.kv_caches[layer_name]
        assert k_view[0, 0, 0, 0].item() == i * 2
        assert v_view[0, 0, 0, 0].item() == i * 2 + 1

def test_stride_order_indexes_kv_by_block_stride():
    """验证 AscendAttentionBackend.indexes_kv_by_block_stride() 返回 True。"""
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    assert AscendAttentionBackend.indexes_kv_by_block_stride() is True

def test_prefer_cross_layer_blocks_default_false():
    """验证默认情况下 prefer_cross_layer_blocks 返回 False。"""
    connector = MooncakeConnector(vllm_config_without_flag, ...)
    assert connector.prefer_cross_layer_blocks is False

def test_prefer_cross_layer_blocks_enabled():
    """验证配置后 prefer_cross_layer_blocks 返回 True。"""
    extra_config = {"enable_cross_layers_blocks": "true", ...}
    connector = MooncakeConnector(vllm_config_with_flag, ...)
    assert connector.prefer_cross_layer_blocks is True
```

### 10.2 端到端测试

```bash
# P 节点（prefill，enable_cross_layers_blocks=true）
vllm serve ... \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "enable_cross_layers_blocks": "true",
      ...
    }
  }'

# D 节点（decode，可选择是否启用 cross-layer）
vllm serve ... \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "enable_cross_layers_blocks": "true",
      ...
    }
  }'
```

**验证点**：
1. 启动日志中出现 `"Allocating a cross layer KV cache of shape ..."` 
2. 启动日志中出现 `"register_cross_layers_kv_cache: decomposed cross-layer tensor ..."`
3. P/D 间成功传输 KV，推理结果与 per-layer 模式一致
4. 注册区域数量明显减少（由 `2 × num_layers` 降至 1）
