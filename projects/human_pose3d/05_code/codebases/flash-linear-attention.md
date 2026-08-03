# flash-linear-attention 仓库卡片

## 架构总览

flash-linear-attention (fla) 是一个基于 Triton 的线性注意力/状态空间模型 kernel 库，提供从底层算子(ops)到中间层(layers)再到完整模型(models)的三层抽象。核心设计围绕 chunk-wise 并行：将序列切分为固定大小 chunk（默认64），chunk 内用 intra-chunk 计算（类 attention 矩阵），chunk 间用 recurrent state（h 矩阵）传递信息，二者合并得到输出。

```
fla/
├── ops/                    # 底层 Triton kernel 算子
│   ├── gla/                # GLA: chunk.py, fused_chunk.py, fused_recurrent.py, naive.py
│   ├── gated_delta_rule/   # GDN: chunk.py, chunk_fwd.py, wy_fast.py, gate.py, fused_recurrent.py, naive.py
│   ├── delta_rule/         # DeltaNet (无门控): chunk.py, parallel.py, wy_fast.py
│   ├── common/             # 共享 kernel: chunk_h.py(状态传递), chunk_o.py(输出计算), chunk_delta_h.py
│   └── utils/              # 工具: cumsum, solve_tril, index, cache, constant
├── layers/                 # nn.Module 层实现
│   ├── gla.py              # GatedLinearAttention 层
│   ├── gated_deltanet.py   # GatedDeltaNet 层
│   └── ...
├── models/                 # HuggingFace 兼容完整模型
│   ├── gla/                # GLAForCausalLM
│   ├── gated_deltanet/     # GatedDeltaNetForCausalLM
│   └── ...
└── modules/                # 基础模块: RMSNorm, ShortConvolution, activations
```

## 关键事实

### 1. GLA chunk-wise 并行入口与核心流程

公开 API 为 `chunk_gla()`，定义于：

`fla/ops/gla/chunk.py:1358-1461`
```python
@torch.compiler.disable
def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

前向核心流程在 `chunk_gla_fwd()`（同文件 :1112-1171）：
1. `chunk_local_cumsum(g, chunk_size, scale=RCP_LN2)` — 将 log-space 门控转为 chunk 内累积和（log2 域）
2. `chunk_fwd_h(k, v, gk=g_cumsum, ...)` — 计算 chunk 间 hidden state 递推（来自 `fla/ops/common/chunk_h.py`）
3. `chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale)` — 计算 chunk 内注意力矩阵 A
4. `chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h, scale)` — 合并 inter-chunk (q@h) 与 intra-chunk (A@v) 得到输出

### 2. GLA 门控衰减机制

门控 g 的形状为 `[B, T, H, K]`（逐 key 维度），在 layer 中通过低秩投影 + logsigmoid 生成：

`fla/layers/gla.py:239`
```python
gk = F.logsigmoid(gk) / self.gate_logit_normalizer
```

在 kernel 中，衰减通过 `exp2(g_cumsum)` 实现。inter-chunk 状态传递时，h 乘以衰减因子；intra-chunk 注意力中，q 和 k 分别乘以各自位置的衰减：

`fla/ops/gla/chunk.py:101-107`（intra_sub_inter kernel 核心）
```python
b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
b_kg = b_k * exp2(b_gn[:, None] - b_gk)
b_A += tl.dot(b_qg, b_kg)
```

### 3. GLA 因果 mask 位置

chunk 内输出计算中，因果 mask 硬编码为下三角：

`fla/ops/gla/chunk.py:363`
```python
m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
```

`fla/ops/gla/chunk.py:401`
```python
b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
```

### 4. Gated DeltaNet chunk-wise 入口与核心流程

公开 API 为 `chunk_gated_delta_rule()`（别名 `chunk_gdn`）：

`fla/ops/gated_delta_rule/chunk.py:397-588`
```python
@dispatch('gated_delta_rule')
@torch.compiler.disable
def chunk_gated_delta_rule(
    q, k, v, g, beta,
    scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False, state_v_first=False,
    cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, **kwargs,
):
```

前向核心流程在 `chunk_gated_delta_rule_fwd()`（同文件 :33-123）：
1. **门控累积和**：`gdn_gate_chunk_cumsum(g, A_log, chunk_size, scale=RCP_LN2, dt_bias)` 或 `chunk_local_cumsum(g)`
2. **WY 表示**：`chunk_gated_delta_rule_fwd_intra(k, v, g, beta)` → 得到 w, u, A（delta rule 的 chunk 内求解）
3. **状态递推**：`chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)` → h, v_new, final_state
4. **输出计算**：`chunk_fwd_o(q, k, v_new, h, g, scale)` → o

### 5. Delta Rule 更新核心（WY 表示）

Delta rule 的核心是：`v_new = v - (h @ k) * beta`，即从 v 中减去当前 state 在 k 方向的投影。chunk 并行化通过 WY 表示实现：

`fla/ops/gated_delta_rule/wy_fast.py:39-94`（`recompute_w_u_fwd_kernel`）
```python
# u = A @ (v * beta)  — 更新后的 value
b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
b_u = tl.dot(b_A, b_vb, allow_tf32=False)

# w = A @ (k * beta * exp2(g))  — 衰减加权的 key
b_kb = b_k * b_b[:, None]
if USE_G:
    b_kb *= b_g[:, None]
b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
```

其中 A 矩阵是 `(I + beta*K@K^T)^{-1}` 的下三角解，通过 `solve_tril` 计算：

`fla/ops/gated_delta_rule/chunk_fwd.py:40-69`（`chunk_gated_delta_rule_fwd_kkt_solve_kernel` 文档）
```python
"""
Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass.
Steps:
1. Compute all 10 lower-triangular [BC, BC] blocks of beta * K @ K^T in registers
2. Apply gate and beta scaling
3. Forward substitution on diagonal blocks
4. Block merge to get full (I+A)^{-1}
5. Write result to A (output)
"""
```

### 6. GDN 门控衰减 kernel

门控公式为 `g = -exp(A_log) * softplus(g_input + dt_bias)`，然后做 chunk 内 cumsum：

`fla/ops/gated_delta_rule/gate.py:96-104`（`gdn_gate_chunk_cumsum_scalar_kernel` 核心）
```python
b_gate = -exp(b_A) * softplus(b_g)
b_o = tl.cumsum(b_gate, axis=0)
if REVERSE:
    b_z = tl.sum(b_gate, axis=0)
    b_o = -b_o + b_z[None] + b_gate
if HAS_SCALE:
    b_o *= scale
```

### 7. GDN 因果 mask 位置

在共享的 `chunk_fwd_kernel_o` 中：

`fla/ops/common/chunk_o.py:125-126`
```python
m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0)
```

### 8. Naive 参考实现（delta rule 递推语义）

`fla/ops/gated_delta_rule/naive.py:50-59`
```python
for i in range(T):
    h = h.clone() * g[:, :, i].exp()[..., None, None]   # 门控衰减
    b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)    # delta: v - h@k
    b_v = b_v * b_beta[..., None]                        # beta 缩放
    h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)  # 外积更新
    o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)    # 输出
```

### 9. 非自回归/双向场景入口

本仓库所有 chunk kernel 均硬编码因果（下三角）mask。README News 2024-12 提到：
> Add `flash-bidirectional-attention` to `fla-org` ([repo](https://github.com/fla-org/flash-bidirectional-linear-attention))

即双向/非自回归场景需使用独立仓库 `fla-org/flash-bidirectional-linear-attention`，本仓库不直接支持。

### 10. chunk_size 约束

GLA 的 chunk_size 在 `ChunkGLAFunction.forward` 中动态计算：

`fla/ops/gla/chunk.py:1296`
```python
chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))
```

GDN 的 chunk_size 有严格校验：

`fla/ops/gated_delta_rule/chunk.py:535-536`
```python
if chunk_size not in (16, 32, 64):
    raise ValueError(f"`chunk_size` must be 16, 32, or 64 for Gated Delta Rule, got {chunk_size}.")
```

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|-----|------|---------------|
| GLA 默认 chunk_size | 64 | `fla/ops/gla/chunk.py:1296` | 已可通过 seq_len 自适应；如需固定可传参 |
| GDN chunk_size 允许值 | {16, 32, 64} | `fla/ops/gated_delta_rule/chunk.py:535` | 修改校验逻辑，但 kernel 内 BC 子块硬编码为 BT/4 |
| GDN BC 子块大小 | min(16, BT) | `fla/ops/gated_delta_rule/chunk_fwd.py` 内 kkt_solve kernel 硬编码 4 个 BC 块 | 需重写 kernel 支持不同 NC |
| GLA BC 子块大小 | min(16, BT) | `fla/ops/gla/chunk.py:819` | 同上 |
| GLA gate_logit_normalizer | 16 | `fla/layers/gla.py:175` | 已是构造参数 |
| GDN dt_min/dt_max | 0.001 / 0.1 | `fla/layers/gated_deltanet.py:155-156` | 硬编码初始化，可改为构造参数 |
| GDN A_log 初始化范围 | uniform(0, 16) | `fla/layers/gated_deltanet.py:151` | 硬编码 |
| RCP_LN2 常量 | 1/ln(2) ≈ 1.4427 | `fla/ops/utils/constant.py` | 固定数学常量 |
| scale 默认值 | K^{-0.5} | `fla/ops/gla/chunk.py:1444`, `fla/ops/gated_delta_rule/chunk.py:565` | 已可通过参数覆盖 |
| fused_recurrent 切换阈值 | seq_len <= 64 | `fla/layers/gla.py:194`, `fla/layers/gated_deltanet.py:237` | 硬编码在 layer forward 中 |
| GDN gate bwd BT | 32 | `fla/ops/gated_delta_rule/gate.py:201` | 硬编码 |

## 环境与复现

**依赖**（`pyproject.toml`）：
- Python >= 3.10
- 基础：`transformers>=4.45.0`, `einops`
- CUDA 后端：`torch>=2.7.0`, `triton>=3.3`
- 可选：`causal-conv1d>=1.4.0`（短卷积加速）, `tilelang>=0.1.9`（TileLang 后端）

**安装**：
```sh
pip install flash-linear-attention[cuda]
# 或源码安装
git clone https://github.com/fla-org/flash-linear-attention
cd flash-linear-attention && pip install -e ".[cuda]"
```

**最小运行命令**（无需权重下载，纯 kernel 测试）：
```python
import torch
from fla.ops.gla import chunk_gla
B, T, H, K, V = 2, 512, 4, 128, 128
q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16))
o, ht = chunk_gla(q, k, v, g)
```

```python
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
B, T, H, K, V = 2, 512, 4, 128, 128
q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
g = torch.randn(B, T, H, device='cuda', dtype=torch.bfloat16)
beta = torch.rand(B, T, H, device='cuda', dtype=torch.bfloat16).sigmoid()
o, ht = chunk_gated_delta_rule(q, k, v, g, beta)
```

**预训练权重**：HuggingFace `fla-hub` 组织下有预训练模型，可通过 `AutoModelForCausalLM.from_pretrained('fla-hub/gla-1.3B')` 加载。

**测试**：
```sh
pytest tests/ops/test_gdn.py -x -q
pytest tests/models/test_modeling_gla.py -x -q
```

## 改造接口点

### 目标：接入非自回归/双向场景（去除因果 mask）

1. **最小侵入点 — GLA intra-chunk mask**：
   - 位置：`fla/ops/gla/chunk.py:363` 和 `:401`
   - 方式：将 `m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]` 改为全 True（或添加 `IS_CAUSAL` constexpr 参数）
   - 影响：仅影响 chunk 内注意力，inter-chunk 状态递推天然是单向的，需额外处理

2. **最小侵入点 — GDN chunk_o mask**：
   - 位置：`fla/ops/common/chunk_o.py:125`
   - 方式：同上，将 `>=` 改为全 True 或参数化
   - 注意：此 kernel 被多个算子共享（GDN、DeltaNet、KDA 等），修改需加 flag 隔离

3. **GDN WY 表示的因果性**：
   - 位置：`fla/ops/gated_delta_rule/chunk_fwd.py` 中 `solve_tril`（下三角求解）
   - 问题：delta rule 的 WY 表示本质依赖因果顺序（`(I + L)^{-1}` 其中 L 是下三角），双向场景需要完全不同的公式
   - 建议：参考 `fla-org/flash-bidirectional-linear-attention` 仓库的实现

4. **Inter-chunk 状态递推**：
   - 位置：`fla/ops/common/chunk_h.py` 中 `chunk_fwd_kernel_h`
   - 问题：h 的递推是严格从左到右的，双向需要正向+反向两遍扫描
   - 建议：可复用 `chunk_fwd_h` 做正向，再实现一个反向版本（reverse cumsum + reverse scan）

5. **Layer 层接入**：
   - 位置：`fla/layers/gla.py:267-277` 和 `fla/layers/gated_deltanet.py:310-327`
   - 方式：添加 `bidirectional=True` 参数，调用修改后的 kernel 或独立的双向 kernel

### 推荐改造路径

对于纯双向线性注意力（无 delta rule），GLA 的改造更简单：只需去除 intra mask + 双向 scan。对于 GDN，delta rule 的 WY 表示与因果顺序深度耦合，建议直接使用 `flash-bidirectional-linear-attention` 仓库或仅对 GLA 做双向化。

## 风险与未知

1. **`flash-bidirectional-linear-attention` 仓库内容未验证**：README 仅提及链接，未确认其是否支持 GLA/GDN 的双向版本，也未确认其 API 兼容性。
2. **GDN WY 表示双向化的数学可行性**：delta rule 的 `(I + beta*KK^T)^{-1}` 在非因果设定下是否仍有封闭解，需要数学推导验证。
3. **`chunk_fwd_kernel_o` 共享范围**：此 kernel 被 GDN、DeltaNet、KDA 等多个算子调用，添加 `IS_CAUSAL` flag 后的性能影响未评估。
4. **Triton autotune 缓存**：修改 kernel 签名（添加 constexpr）会导致所有已有 autotune 缓存失效，首次运行变慢。
5. **Blackwell/AMD 兼容性**：仓库支持多后端（NVIDIA/AMD/Intel），修改 kernel 时需确保不破坏非 NVIDIA 平台。
6. **Context Parallel (CP) 模式**：GDN 支持分布式 CP 训练（`fla/ops/cp/`），双向化后 CP 的通信模式可能需要重新设计。
7. **GDN2 (gdn2)**：仓库中有 `fla/ops/gdn2/` 和 `fla/layers/gdn2.py`，是 GDN 的改进版（解耦 erase/write），未深入分析其是否更适合双向扩展。
