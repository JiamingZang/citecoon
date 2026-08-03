# Based 仓库侦察卡

> 仓库: https://github.com/HazyResearch/based  
> 论文: "Simple linear attention language models balance the recall-throughput tradeoff" (arXiv:2402.18668)  
> 侦察日期: 2026-07-21

---

## 架构总览

Based 是一个 100% 亚二次的语言模型架构，核心思想是将三种 mixer 按层交替堆叠：(1) BaseConv（短卷积+门控）处理大部分层，(2) LinearAttention（Taylor 特征映射的线性注意力）提供全局长程依赖，(3) SlidingAttention（FlashAttention 滑窗 softmax 注意力，窗口 256）提供局部精确注意力。模型骨架复用 FlashAttention 的 GPT 训练框架（Hydra 配置 + PyTorch Lightning），通过 `alt_mixer_layers` / `alt_mixer_2_layers` 列表控制哪些层用哪种 mixer。

```
based/
├── based/
│   ├── models/
│   │   ├── gpt.py              # GPTModel / GPTLMHeadModel，层组装与 mixer 分发
│   │   ├── block.py            # 通用 Transformer Block (prenorm)
│   │   ├── mixers/
│   │   │   ├── linear_attention.py  # LinearAttention + TaylorExp 特征映射
│   │   │   ├── slide_attention.py   # SlidingAttention (FlashAttn window)
│   │   │   ├── convolution.py       # BaseConv / ShortConvolution
│   │   │   └── mha.py              # 标准 MHA (fallback)
│   │   └── mlp.py
│   ├── ops/triton/             # Triton kernels (layernorm, rotary, etc.)
│   └── generation.py           # 推理生成逻辑
├── train/
│   ├── configs/experiment/reference/  # 预训练配置 (based-360m, based-1b)
│   ├── run.py                  # 训练入口
│   └── csrc/causal_dot_prod/   # Fast Transformers 因果点积 CUDA kernel
├── evaluate/                   # (submodule) based-evaluation-harness
├── synthetic/                  # (submodule) zoology 合成实验
└── ThunderKittens/             # (submodule) CUDA demo kernels
```

---

## 关键事实

### 1. 三种 Mixer 的层分配（360M 配置）

`train/configs/experiment/reference/based-360m.yaml:53-65`
```yaml
    alt_mixer_layers: 
      - 1
      - 6
      - 11
      - 16
      - 21

    alt_mixer_2_layers:
      - 2
      - 7
      - 12
      - 17
      - 22
```
- 总层数 `n_layer: 27`。层 0,3,4,5,8,9,10,13,14,15,18,19,20,23,24,25,26 用 BaseConv；层 1,6,11,16,21 用 LinearAttention；层 2,7,12,17,22 用 SlidingAttention。
- 1.3B 配置 (`based-1b.yaml:54-70`)：`n_layer: 36`，alt_mixer_layers = [1,6,11,16,21,27,33]，alt_mixer_2_layers = [2,7,12,17,22,28,34]。

### 2. Mixer 分发逻辑

`based/models/gpt.py:64-96`
```python
def create_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    tag = 'mixer'
    value = getattr(config, "mixer", None) 
    alt_mixer_layers = getattr(config, "alt_mixer_layers", None)
    alt_mixer_2_layers = getattr(config, "alt_mixer_2_layers", None)
    alt_mixer = getattr(config, "alt_mixer", None)
    alt_mixer_2 = getattr(config, "alt_mixer_2", None)
    if alt_mixer_2_layers is not None and layer_idx in alt_mixer_2_layers:
        value = None
        if alt_mixer_2 is not None:
            tag = 'alt_mixer_2'
            value = config.alt_mixer_2
    elif alt_mixer_layers is not None and layer_idx in alt_mixer_layers:
        value = None
        if alt_mixer is not None:
            tag = 'alt_mixer'
            value = config.alt_mixer
    ...
    return hydra.utils.instantiate(value, _partial_=True, device=device, dtype=dtype, layer_idx=layer_idx)
```
通过 Hydra `_target_` 字段实例化对应 mixer 类。

### 3. TaylorExp 特征映射（二阶 Taylor 展开近似 exp(q^T k / sqrt(d))）

`based/models/mixers/linear_attention.py:52-75`
```python
class TaylorExp(FeatureMap):
    def __init__(self, input_dim: int, **kwargs: any):
        super().__init__(input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(input_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    def forward(self, x: torch.Tensor):
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat(
            [x[..., :1] ** 0, x / self.rrd, x2 / self.rd], 
            dim=-1
        )
```
- 输入维度 `feature_dim=16`，展开后维度 = 1 + 16 + 16*16 = 273（`expanded_size()` 在 L255-256 返回 `feature_dim**2 + feature_dim + 1`）。
- 无学习参数，纯数学映射。

### 4. LinearAttention 并行前向（多 kernel 后端）

`based/models/mixers/linear_attention.py:152-227`
```python
def parallel_forward(self, x, q, k, v):
    if self.parallel_implementation == "tk":
        y, kv_state = tk.based(q, k, v)
        ...
    elif self.parallel_implementation == "quadratic":
        q, k = self.feature_map(q), self.feature_map(k)
        A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
        A_qk = torch.tril(A_qk)       
        y = torch.einsum("bhnm,bhme->bhne", A_qk.to(x.dtype), v.to(x.dtype))
        z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
        y = y * z[..., None]
        ...
    elif self.parallel_implementation == "linear": 
        q, k = self.feature_map(q), self.feature_map(k)
        v = causal_dot_product(q.contiguous().to(dtype=torch.float32), ...)
        ...
    elif self.parallel_implementation == "fla_parallel":
        y = parallel_based(q, k, v, True, True)
        ...
    elif self.parallel_implementation == "fla_chunk":
        y = fused_chunk_based(q, k, v, True, True)
```
- 默认 360M 配置使用 `"fla_parallel"`（flash-linear-attention 的 Triton kernel）。
- 1B 配置未显式指定，默认 `"quadratic"`。

### 5. SlidingAttention 窗口设置

`based/models/mixers/slide_attention.py:332-336`
```python
        if window_size is None:
            assert 0, print("Using windows")
            self.window = None
        else:
            self.window = (window_size//2, 0)
```
- 配置中 `window_size: 256`，实际传给 FlashAttention 的是 `(128, 0)`，即只看左侧 128 个 token（因果方向）。
- 底层调用 `flash_attn_qkvpacked_func(..., window_size=self.window)` (L100-106)。

### 6. BaseConv 结构（门控短卷积）

`based/models/mixers/convolution.py:168-210`
```python
class BaseConv(nn.Module):
    def __init__(self, d_model, l_max, kernel_size=3, ..., expand_proj=2, ...):
        self.d_inner = expand_proj*self.d_model // 2
        self.in_proj = nn.Linear(self.d_model, expand_proj*self.d_model, bias=use_bias)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=use_bias)
        self.conv = ShortConvolution(self.d_inner, kernel_size=kernel_size, ...)

    def forward(self, u, ...):
        u = self.in_proj(u)
        u1, u2 = torch.split(u, self.d_inner, dim=-1)
        u_conv = self.conv(u1, inference_params=inference_params)
        u_conv = nn.functional.silu(u_conv)
        v = u_conv * u2
        y = self.out_proj(v)
```
- 配置中 `expand_proj: 4`，`kernel_sizes: 3`。

### 7. 递归推理状态（LinearAttention）

`based/models/mixers/linear_attention.py:230-252`
```python
    def recurrent_forward(self, hidden_states, kv_state, k_state, q, k, v, decay=None):
        b, h, l, d = q.shape
        assert l == 1
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        kv_state += k[:, :, -1:] * v[:, :, -1:]
        k_state  += k[:, :, -1:]
        num = (q * kv_state).sum(dim=-1)
        y = num / ((q * k_state).sum(dim=-1) + eps)
```
- 推理时维护 `(kv_state, k_state)` 固定大小状态，实现 O(1) 逐步生成。

### 8. Recall 实验（合成数据）

README 明确指出合成关联回忆实验代码在独立仓库 [HazyResearch/zoology](https://github.com/HazyResearch/zoology)（本仓库 `synthetic/` 是空 submodule）。复现命令：
```
python -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py -p
```
本仓库内无记忆容量配置代码。

---

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|-----|------|----------------|
| Taylor 展开阶数 | 2（固定） | `linear_attention.py:52-75` TaylorExp 类 | 新增 `order` 参数，泛化到 N 阶 |
| feature_dim | 16 | yaml 配置 `feature_dim: 16` | 已是配置项 |
| num_heads (linear attn) | 16 | yaml 配置 `num_heads: 16` | 已是配置项 |
| window_size | 256 (→实际左窗 128) | yaml `window_size: 256`；`slide_attention.py:336` 做 `//2` | 修改 yaml 或改 `//2` 逻辑 |
| kernel_size (conv) | 3 | yaml `kernel_sizes: 3` | 已是配置项 |
| expand_proj (conv) | 4 | yaml `expand_proj: 4` | 已是配置项 |
| eps (linear attn) | 1e-12 / 1e-6 (fla) | `linear_attention.py:86,246-248` | 改为构造参数 |
| decay_const (1B) | -3 | yaml `decay_const: -3`；`gpt.py:496` | 已是配置项 |
| parallel_implementation | "fla_parallel" / "quadratic" | yaml / `linear_attention.py:88` | 已是配置项 |
| alt_mixer_layers 分配 | 固定列表 | yaml 配置 | 已是配置项，可自由调整 |
| causal | True (硬编码) | `slide_attention.py:82,319` | 代码中 `causal = True` 覆盖参数 |

---

## 环境与复现

### 依赖（`setup.py`）
- Python ≥ 3.8（推荐 3.8.18）
- PyTorch 2.1.2 + CUDA 11.8
- flash-attn==2.5.2
- causal-conv1d
- transformers==4.36.2, einops==0.7.0
- 训练额外: hydra-core==1.3.2, pytorch-lightning==1.8.6, apex (FusedAdam), wandb
- 可选快速 kernel: `pip install triton==2.2.0 && pip install -U git+https://github.com/sustcsonglin/flash-linear-attention`

### 权重下载
```python
from based.models.gpt import GPTLMHeadModel
model = GPTLMHeadModel.from_pretrained_hf("hazyresearch/based-360m")  # 自动从 HF 下载
```
可用检查点: `hazyresearch/based-360m`, `hazyresearch/based-1b`, `hazyresearch/based-1b-50b`

### 最小运行命令
```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
python -c "
import torch
from transformers import AutoTokenizer
from based.models.gpt import GPTLMHeadModel
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = GPTLMHeadModel.from_pretrained_hf('hazyresearch/based-360m').to('cuda')
inp = tokenizer.encode('If I take one more step, it will be', return_tensors='pt').to('cuda')
out = model.generate(inp, max_length=20)
print(tokenizer.decode(out[0]))
"
```

### 训练
```bash
pip install -e .[train]
# 安装 apex
cd train/
python run.py experiment=example/based-360m trainer.devices=8
```

---

## 改造接口点

### 针对"线性注意力 + 滑窗混合"的改造
1. **调整层分配比例**: 修改 yaml 中 `alt_mixer_layers` / `alt_mixer_2_layers` 列表即可改变哪些层用线性注意力/滑窗。最小侵入，无需改代码。
2. **替换特征映射**: 在 `linear_attention.py` 中新增 `FeatureMap` 子类（如 Hedgehog、Performer），通过 yaml `_target_` 指向新类。
3. **改变窗口大小**: 修改 yaml `window_size` 值；注意 `slide_attention.py:336` 的 `//2` 逻辑。
4. **添加新的 mixer 类型**: 在 `gpt.py:create_mixer_cls` 中已有 `alt_mixer` / `alt_mixer_2` 两级扩展点；如需第三种备选需新增 `alt_mixer_3_layers` 字段。

### 针对"Taylor 特征映射 kernel"的改造
1. **修改展开阶数**: 改 `TaylorExp.forward()`（L67-75），增加高阶项；同步更新 `expanded_size()`（L255-256）。
2. **切换 kernel 后端**: 改 yaml `parallel_implementation` 字段（"quadratic"/"linear"/"fla_parallel"/"fla_chunk"/"tk"）。
3. **自定义 Triton kernel**: 参考 `fla.ops.based` 接口，替换 `parallel_based` / `fused_chunk_based` 调用。

### 针对"recall 实验记忆容量"的改造
- 本仓库不含合成实验代码，需到 [zoology](https://github.com/HazyResearch/zoology) 仓库修改。
- 本仓库中影响记忆容量的参数: `feature_dim`（决定状态维度 273）、`num_heads`、`l_max`。

---

## 风险与未知

1. **evaluate/ 和 synthetic/ 是空 submodule**，未能查看 recall 评测（SWDE/FDA/SQUAD）的具体实现和记忆容量配置细节。
2. **ThunderKittens CUDA kernel** (`tk.based`) 的具体实现未在本仓库中，仅在 submodule 里（未克隆）。
3. **zoology 合成实验**的记忆容量扫描配置（Figure 2/3）需到独立仓库确认。
4. **1B 配置的 `parallel_implementation`** 未显式指定，默认走 `"quadratic"`（O(n^2)），训练效率可能不如 fla_parallel。
5. **`slide_attention.py:82` 硬编码 `causal = True`**，覆盖了构造参数，若需双向注意力需改代码。
6. **apex 依赖**：训练必须安装 NVIDIA apex（FusedAdam），在某些环境可能有编译问题。
7. **Pile 数据集已下线**，reference 配置不可直接复现训练；需用 example 配置或自备数据。
