# Cross-Layer KV Cache 实现文档

## 1. 背景与目标

### 1.1 问题描述

在 P/D 分离（Prefill/Decode Disaggregation）场景中，KV Cache 需要在节点间高效传输。传统方案中，每层的 KV Cache 是独立的 Tensor，传输时需要对每层分别注册和传输，存在以下问题：

- 每层 KV Cache 独立分配，内存不连续，跨层传输需要多次 RDMA 操作
- KV Connector（如 Mooncake、NIXL）无法利用 block 粒度的跨层连续性做批量传输

### 1.2 解决方案

Cross-Layer KV Cache 将所有层的 KV Cache 分配在一块连续内存中，使得对于任意一个 block，该 block 在**所有层**的 KV 数据在物理内存上是连续的，从而允许 KV Connector 以单次操作传输一个 block 所有层的数据。

### 1.3 目标

参考 GPU（`vllm/vllm/v1/worker/gpu_model_runner.py`）的实现，在 NPU（`vllm_ascend`）侧支持 Cross-Layer KV Cache，使 NPU 上的 P/D 分离场景能够利用 cross-layer 布局加速 KV 传输。

---

## 2. 涉及文件

| 文件 | 改动 |
|---|---|
| `vllm_ascend/attention/attention_v1.py` | 为 `AscendAttentionBackend` 添加 `get_kv_cache_stride_order` |
| `vllm_ascend/worker/model_runner_v1.py` | 三处改动（见下文） |

---

## 3. 核心概念

### 3.1 KV Cache 内存布局

**普通布局（per-layer 独立）**

每层独立分配，逻辑形状为：

```
layer_0: (2, num_blocks, block_size, num_kv_heads, head_size)
layer_1: (2, num_blocks, block_size, num_kv_heads, head_size)
...
layer_N: (2, num_blocks, block_size, num_kv_heads, head_size)
```

内存上各层不连续，block_0 的数据分散在 N 个独立 Tensor 中。

**Cross-Layer 布局（连续统一 Tensor）**

所有层共享一块内存，物理布局为：

```
(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
```

内存上 block_0 的所有层数据连续存放：

```
[block_0_layer_0_K, block_0_layer_0_V,
 block_0_layer_1_K, block_0_layer_1_V,
 ...
 block_0_layer_N_K, block_0_layer_N_V,
 block_1_layer_0_K, ...]
```

### 3.2 stride_order 语义

`get_kv_cache_stride_order(include_num_layers_dimension=True)` 返回一个整数元组，描述从**逻辑形状**到**物理内存布局**的维度置换。

- 逻辑形状（带 num_layers）：`(num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)` 索引为 `(0, 1, 2, 3, 4, 5)`
- 期望物理布局：`(num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)`
- stride_order：物理位置 `i` 对应逻辑维度 `stride_order[i]`

```
物理位置 0 → 逻辑维度 2 (num_blocks)
物理位置 1 → 逻辑维度 0 (num_layers)
物理位置 2 → 逻辑维度 1 (2/KV_split)
物理位置 3 → 逻辑维度 3 (block_size)
物理位置 4 → 逻辑维度 4 (num_kv_heads)
物理位置 5 → 逻辑维度 5 (head_size)
```

所以 `stride_order = (2, 0, 1, 3, 4, 5)`。

### 3.3 indexes_kv_by_block_stride

`AttentionBackend.indexes_kv_by_block_stride()` 是一个类方法，用于判断该 backend 是否支持以 block stride 方式索引 KV cache（即 `num_blocks` 是最外层物理维度）。

判断逻辑（来自上游 `vllm/v1/attention/backend.py`）：

```python
@classmethod
def indexes_kv_by_block_stride(cls) -> bool:
    kv_cache_stride_order = cls.get_kv_cache_stride_order(
        include_num_layers_dimension=False)
    layered_kv_cache_stride_order = cls.get_kv_cache_stride_order(
        include_num_layers_dimension=True)
    # 带 layers 维度时多一个维度
    if len(layered_kv_cache_stride_order) != len(kv_cache_stride_order) + 1:
        return False
    # layered[0] != 0 表示 num_layers 不是物理最外层（即 num_blocks 是）
    return layered_kv_cache_stride_order[0] != 0
```

对于 `AscendAttentionBackend`：

- `get_kv_cache_stride_order(False)` 返回 `(0,1,2,3,4)` — 5 维
- `get_kv_cache_stride_order(True)` 返回 `(2,0,1,3,4,5)` — 6 维，`[0]=2 ≠ 0`

因此 `indexes_kv_by_block_stride()` 返回 `True`。

---

## 4. 实现详解

### 4.1 `attention_v1.py` — 添加 `get_kv_cache_stride_order`

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

**作用**：

- 告知上游框架 NPU 的 KV Cache 物理内存布局偏好
- 使 `indexes_kv_by_block_stride()` 返回 `True`，进而满足 `use_uniform_kv_cache` 的条件
- 供 `allocate_uniform_kv_caches` 用来分配正确形状的连续 Tensor

### 4.2 `model_runner_v1.py` — `get_kv_cache_spec` 设置 `indexes_kv_by_block_stride`

**位置**：`get_kv_cache_spec` 方法中 `isinstance(attn_module, Attention)` 分支。

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

**作用**：将 `indexes_kv_by_block_stride=True` 写入 `AttentionSpec`，使其随 `KVCacheConfig` 传递到 `initialize_attn_backend` → `AttentionGroup.kv_cache_spec` → `use_uniform_kv_cache` 的判断链路中。

**数据流**：

```
get_kv_cache_spec()
  └─ spec.indexes_kv_by_block_stride = True
       └─ get_kv_cache_groups()
            └─ KVCacheGroupSpec.kv_cache_spec  (merge 保留该字段)
                 └─ initialize_attn_backend()
                      └─ AttentionGroup.kv_cache_spec
                           └─ use_uniform_kv_cache()
                                └─ kv_cache_spec.indexes_kv_by_block_stride → True
```

### 4.3 `model_runner_v1.py` — `initialize_kv_cache_tensors` 增加 cross-layer 路径

**位置**：`initialize_kv_cache_tensors` 方法开头。

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

    # NPU 的 kernel_block_sizes 是 list[list[int]]，需要展平为 list[int]
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

**关键细节 — `kernel_block_sizes` 格式差异**：

| 版本 | `self.kernel_block_sizes` 类型 | 示例 |
|---|---|---|
| GPU (`gpu_model_runner.py`) | `list[int]` | `[128]` |
| NPU (`model_runner_v1.py`) | `list[list[int]]` | `[[128]]` |

`allocate_uniform_kv_caches` 接收 `list[int]`，因此 NPU 侧需要展平。

**`use_uniform_kv_cache` 的完整判断条件**（来自上游 `kv_connector_model_runner_mixin.py`）：

```python
@staticmethod
def use_uniform_kv_cache(attn_groups) -> bool:
    if not has_kv_transfer_group():           # 需要配置 KV Connector
        return False
    if not get_kv_transfer_group().prefer_cross_layer_blocks:  # Connector 需要支持
        return False
    if len(attn_groups) != 1 or len(attn_groups[0]) != 1:    # 单一 attention group
        return False
    attn_group = attn_groups[0][0]
    kv_cache_spec = attn_group.kv_cache_spec
    if not isinstance(kv_cache_spec, AttentionSpec):           # 必须是 AttentionSpec
        return False
    return kv_cache_spec.indexes_kv_by_block_stride            # backend 支持 block stride
```

**`allocate_uniform_kv_caches` 内部逻辑**（上游实现）：

```python
# 1. 计算总尺寸
total_size = tensor_size * num_layers

# 2. 获取单层 KV 形状（逻辑）
kv_cache_shape = attn_backend.get_kv_cache_shape(...)
# NPU: (2, num_blocks, block_size, num_kv_heads, head_size)

# 3. 前置 num_layers 维度
kv_cache_shape = (num_layers,) + kv_cache_shape
# NPU: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)

# 4. 应用 stride_order 得到物理形状
kv_cache_shape = tuple(kv_cache_shape[i] for i in stride_order)
# NPU stride_order=(2,0,1,3,4,5)
# 物理形状: (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)

# 5. 分配连续内存
cross_layers_kv_cache = torch.zeros(total_size, dtype=torch.int8).view(dtype).view(shape)

# 6. 通过 permute 得到逻辑视图，切片给每层
inv_order = [stride_order.index(i) for i in range(len(stride_order))]
# inv_order = (1, 2, 0, 3, 4, 5)
permuted = cross_layers_kv_cache.permute(*inv_order)
# permuted 形状: (num_layers, 2, num_blocks, block_size, num_kv_heads, head_size)

kv_caches = {}
for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
    tensor = permuted[i]  # 形状: (2, num_blocks, block_size, num_kv_heads, head_size)
    for layer_name in kv_cache_tensor.shared_by:
        kv_caches[layer_name] = tensor
```

每层得到的 `tensor` 形状与原始 NPU KV Cache 形状完全相同，对 attention kernel 透明。

### 4.4 `model_runner_v1.py` — `initialize_kv_cache` 更新注册逻辑

**位置**：`initialize_kv_cache` 方法末尾的 `has_kv_transfer_group()` 分支。

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

**两个注册接口的区别**：

| 接口 | 参数 | 适用场景 |
|---|---|---|
| `register_kv_caches(kv_caches)` | `dict[str, Tensor]` — 各层独立 Tensor | 普通 KV Cache 布局 |
| `register_cross_layers_kv_cache(kv_cache, backend)` | 单一连续 Tensor + backend | Cross-Layer 统一 Tensor |

KV Connector（如 Mooncake）拿到单一连续 Tensor 后，可以用 block 为粒度直接传输所有层的数据，无需逐层注册。

---

## 5. 完整数据流

```
启动时:
  NPUModelRunner.get_kv_cache_spec()
    ├─ Attention 模块 → FullAttentionSpec(indexes_kv_by_block_stride=True)
    └─ 其他模块 → 各自 spec

  get_kv_cache_groups() → KVCacheGroupSpec(kv_cache_spec 含 indexes_kv_by_block_stride)

  NPUModelRunner.initialize_kv_cache(kv_cache_config)
    ├─ initialize_attn_backend(kv_cache_config)
    │    └─ AttentionGroup(kv_cache_spec=..., backend=AscendAttentionBackend)
    │
    └─ initialize_kv_cache_tensors(kv_cache_config)
         ├─ use_uniform_kv_cache(attn_groups)?
         │    ├─ has_kv_transfer_group() → True (配置了 Connector)
         │    ├─ prefer_cross_layer_blocks → True (Connector 支持)
         │    ├─ 单 group 单 backend → True
         │    └─ indexes_kv_by_block_stride → True (AscendAttentionBackend 支持)
         │
         ├─ [Yes] allocate_uniform_kv_caches()
         │    ├─ 分配连续 Tensor: (num_blocks, num_layers, 2, block_size, H, D)
         │    ├─ permute → 逻辑视图: (num_layers, 2, num_blocks, block_size, H, D)
         │    └─ kv_caches[layer_name] = permuted[i]  每层独立视图
         │
         └─ [No] _allocate_kv_cache_tensors() + _reshape_kv_cache_tensors()
              └─ 各层独立 Tensor

  has_kv_transfer_group()?
    ├─ cross_layers_kv_cache is not None
    │    └─ register_cross_layers_kv_cache(tensor, backend)
    └─ cross_layers_kv_cache is None
         └─ register_kv_caches(kv_caches)

推理时 (KV Transfer):
  KV Connector 以 block 粒度批量传输 cross_layers_kv_cache
  每个 block 的所有层数据在内存中连续 → 单次 RDMA 完成一个 block 的跨层传输
```

---

## 6. 激活条件

Cross-Layer KV Cache 仅在**所有以下条件**同时满足时激活：

1. **配置了 KV Transfer**：`vllm_config.kv_transfer_config is not None`
2. **Connector 声明支持**：`kv_transfer_group.prefer_cross_layer_blocks == True`（需要 KV Connector 实现，如 Mooncake/NIXL 的 `prefer_cross_layer_blocks` 属性返回 `True`）
3. **单一 Attention Group**：`len(attn_groups) == 1 and len(attn_groups[0]) == 1`（不支持混合 SSM/MLA 模型）
4. **Backend 支持 block stride 索引**：`AscendAttentionBackend.indexes_kv_by_block_stride() == True`（由本次实现保证）

不满足任意条件时，自动回退到原有独立 Tensor 路径，**对现有功能无影响**。

---

## 7. 不支持的场景

| 场景 | 原因 |
|---|---|
| MLA (DeepSeek) 模型 | `AscendMLAAttentionSpec` 不在 `get_kv_cache_spec` 的 `Attention` 分支中处理 |
| Mamba/SSM 混合模型 | 多 group，`len(attn_groups) != 1` |
| 未配置 KV Transfer | `has_kv_transfer_group()` 返回 False |
| Connector 未声明支持 | `prefer_cross_layer_blocks` 返回 False |

---

## 8. 与 GPU 实现的差异

| 方面 | GPU 实现 | NPU 实现 |
|---|---|---|
| KV Cache 形状 | `(num_blocks, num_kv_heads, block_size, 2*head_size)` — K/V packed | `(2, num_blocks, block_size, num_kv_heads, head_size)` — K/V split |
| stride_order (with layers) | `(1,0,3,2,4)` (HND layout) | `(2,0,1,3,4,5)` |
| `kernel_block_sizes` 格式 | `list[int]` | `list[list[int]]`，需要展平 |
| `indexes_kv_by_block_stride` 设置位置 | `gpu/attn_utils.py` 中统一设置 | `model_runner_v1.py::get_kv_cache_spec` 中设置 |
| 状态变量初始化 | `GPUModelRunner.__init__` 中初始化 | 继承自 `GPUModelRunner`，无需重复 |
| cleanup | `_cleanup_profiling_kv_cache` 处理 | 继承自 `GPUModelRunner`，`hasattr` 保护 |
