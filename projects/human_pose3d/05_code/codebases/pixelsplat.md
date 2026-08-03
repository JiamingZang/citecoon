# pixelSplat 代码侦察卡

> 仓库: https://github.com/dcharatan/pixelsplat (CVPR 2024)
> 功能: 从图像对（2+视图）前馈预测 3D Gaussian Splat 参数，实现可扩展的通用 3D 重建

## 架构总览

pixelSplat 采用 encoder-decoder 架构：encoder 从多视图图像中提取特征，通过极线注意力（epipolar attention）建立跨视图对应关系，再经单目深度预测头将特征转化为 3D Gaussian 参数；decoder 使用 CUDA 光栅化器将 Gaussians 渲染到目标视角。核心流程为：Backbone(DINO/ResNet) → EpipolarTransformer(跨视图极线注意力 + 图像内自注意力) → DepthPredictorMonocular(深度分布预测) → GaussianAdapter(将深度+特征转为均值/协方差/球谐/不透明度)。

```
src/
├── model/
│   ├── encoder/
│   │   ├── encoder_epipolar.py          # 主编码器，串联所有模块
│   │   ├── backbone/
│   │   │   ├── backbone_resnet.py       # ResNet 多尺度特征
│   │   │   └── backbone_dino.py         # DINO ViT + ResNet50 混合骨干
│   │   ├── epipolar/
│   │   │   ├── epipolar_transformer.py  # 极线 Transformer（核心）
│   │   │   ├── epipolar_sampler.py      # 极线采样器
│   │   │   ├── image_self_attention.py  # 图像内 patch 自注意力
│   │   │   ├── depth_predictor_monocular.py  # 单目深度分布预测
│   │   │   ├── distribution_sampler.py  # 离散分布采样
│   │   │   └── conversions.py           # 深度↔相对视差转换
│   │   └── common/
│   │       ├── gaussian_adapter.py      # 原始参数→世界坐标 Gaussians
│   │       └── gaussians.py             # 四元数→旋转矩阵→协方差构建
│   ├── transformer/
│   │   ├── transformer.py              # 通用 Transformer 骨架
│   │   ├── attention.py                # 多头注意力（支持 cross-attention）
│   │   └── feed_forward.py             # FFN
│   ├── decoder/
│   │   └── decoder_splatting_cuda.py   # CUDA Gaussian Splatting 渲染
│   └── types.py                         # Gaussians 数据类
├── geometry/
│   ├── epipolar_lines.py               # 极线几何计算
│   └── projection.py                   # 投影/反投影
├── dataset/                             # 数据加载（RE10K/ACID）
├── loss/                                # MSE + LPIPS 损失
└── config/                              # Hydra 配置
```

## 关键事实

### 1. 极线注意力模块的接口与结构

**EpipolarTransformer 前向接口** (`src/model/encoder/epipolar/epipolar_transformer.py:79-86`):
```python
def forward(
    self,
    features: Float[Tensor, "batch view channel height width"],
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    near: Float[Tensor, "batch view"],
    far: Float[Tensor, "batch view"],
) -> tuple[Float[Tensor, "batch view channel height width"], EpipolarSampling]:
```

**极线采样过程** — 沿极线均匀采样 `num_samples` 个点 (`src/model/encoder/epipolar/epipolar_sampler.py:79-88`):
```python
s = self.num_samples
sample_depth = (torch.arange(s, device=device) + 0.5) / s
sample_depth = rearrange(sample_depth, "s -> s ()")
xy_min = projection["xy_min"].nan_to_num(posinf=0, neginf=0)
xy_min = xy_min * projection["overlaps_image"][..., None]
xy_min = rearrange(xy_min, "b v ov r xy -> b v ov r () xy")
xy_max = projection["xy_max"].nan_to_num(posinf=0, neginf=0)
xy_max = xy_max * projection["overlaps_image"][..., None]
xy_max = rearrange(xy_max, "b v ov r xy -> b v ov r () xy")
xy_sample = xy_min + sample_depth * (xy_max - xy_min)
```

**Cross-attention 构造** — 每个像素作为 query，极线采样特征作为 key/value (`src/model/encoder/epipolar/epipolar_transformer.py:134-142`):
```python
q = rearrange(features, "b v c h w -> (b v h w) () c")
features = self.transformer.forward(
    q,
    rearrange(kv, "b v ov r s c -> (b v r) (s ov) c"),
    b=b, v=v,
    h=h // self.cfg.downscale,
    w=w // self.cfg.downscale,
)
```
每个像素 query 的 token 数为 1，key/value 为所有其他视图在该像素极线上的 `num_samples × (v-1)` 个采样特征。

**深度位置编码加入 KV** (`src/model/encoder/epipolar/epipolar_transformer.py:100-121`):
```python
if self.cfg.num_octaves > 0:
    depths = get_depth(...)
    depths = depths.maximum(near[..., None, None, None])
    depths = depths.minimum(far[..., None, None, None])
    depths = depth_to_relative_disparity(depths, ...)
    depths = self.depth_encoding(depths[..., None])
    kv = sampling.features + depths
```

**图像内自注意力 (ImageSelfAttention)** — 作为 Transformer FFN 层的替代 (`src/model/encoder/epipolar/image_self_attention.py:57-79`):
```python
def forward(self, image: Float[Tensor, "batch d_in height width"]) -> Float[Tensor, "batch d_out height width"]:
    tokens = self.patch_embedder.forward(image)  # Conv2d patchify
    _, _, nh, nw = tokens.shape
    xy, _ = sample_image_grid((nh, nw), device=image.device)
    xy = self.positional_encoding.forward(xy)
    tokens = tokens + rearrange(xy, "nh nw c -> c nh nw")
    tokens = rearrange(tokens, "b c nh nw -> b (nh nw) c")
    tokens = self.transformer.forward(tokens)  # 标准自注意力
    tokens = rearrange(tokens, "b (nh nw) c -> b c nh nw", nh=nh, nw=nw)
    tokens = self.resampler.forward(tokens)  # ConvTranspose2d 恢复分辨率
    return tokens
```

**Transformer 中 selfatt=False 表示 cross-attention** (`src/model/transformer/attention.py:43-46`):
```python
if selfatt:
    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
else:
    self.to_q = nn.Linear(dim, inner_dim, bias=False)
    self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
```

**多视图(>2)时的随机视图嵌入** (`src/model/encoder/epipolar/epipolar_transformer.py:126-131`):
```python
if v > 2:
    shuffle = torch.randperm(v - 1, device=kv.device)
    view_embeddings = rearrange(
        self.view_embeddings(shuffle), "ov c -> () () ov () () c"
    )
    kv = kv + view_embeddings
```

### 2. 高斯参数预测头

**to_gaussians 线性层** — 从特征预测每个 surface 的 (offset_xy + scale + rotation + sh) (`src/model/encoder/encoder_epipolar.py:85-91`):
```python
self.to_gaussians = nn.Sequential(
    nn.ReLU(),
    nn.Linear(
        cfg.d_feature,
        cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
    ),
)
```
其中 `gaussian_adapter.d_in = 7 + 3 * d_sh`，`d_sh = (sh_degree+1)^2`。默认 sh_degree=4 → d_sh=25 → d_in=82。总输出维度 = num_surfaces × (2 + 82) = 84（默认 num_surfaces=1）。

**GaussianAdapter.forward 参数拆分** (`src/model/encoder/common/gaussian_adapter.py:60`):
```python
scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
```
- scales: 3维，经 sigmoid 映射到 [scale_min, scale_max]，再乘以深度和像素大小
- rotations: 4维四元数，归一化后构建协方差
- sh: 3×d_sh 维球谐系数，乘以 sh_mask 初始化偏置

**Scale 计算** (`src/model/encoder/common/gaussian_adapter.py:63-69`):
```python
scale_min = self.cfg.gaussian_scale_min
scale_max = self.cfg.gaussian_scale_max
scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
h, w = image_shape
pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
scales = scales * depths[..., None] * multiplier[..., None]
```

**协方差构建** (`src/model/encoder/common/gaussians.py:33-44`):
```python
def build_covariance(scale, rotation_xyzw):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return rotation @ scale @ scale^T @ rotation^T
```
再经 c2w 旋转矩阵变换到世界坐标 (`gaussian_adapter.py:79-80`):
```python
c2w_rotations = extrinsics[..., :3, :3]
covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
```

**均值计算** (`src/model/encoder/common/gaussian_adapter.py:83-84`):
```python
origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
means = origins + directions * depths[..., None]
```

**不透明度** — 由深度分布的 PDF 经 map_pdf_to_opacity 映射 (`src/model/encoder/encoder_epipolar.py:97-110`):
```python
def map_pdf_to_opacity(self, pdf, global_step):
    cfg = self.cfg.opacity_mapping
    x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
    exponent = 2**x
    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
```
最终不透明度还可乘以可选的 per-pixel opacity head (`encoder_epipolar.py:190-194`):
```python
opacity_multiplier = (
    rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
    if self.cfg.predict_opacity else 1
)
```

**球谐初始化掩码** (`src/model/encoder/common/gaussian_adapter.py:40-46`):
```python
self.register_buffer("sh_mask", torch.ones((self.d_sh,), dtype=torch.float32), persistent=False)
for degree in range(1, self.cfg.sh_degree + 1):
    self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
```

### 3. 两视图输入处理流程

**EncoderEpipolar.forward 完整流程** (`src/model/encoder/encoder_epipolar.py:112-213`):
1. 输入 `context["image"]` 形状为 `[b, v, 3, h, w]`，v=2 即两视图
2. Backbone 提取特征 → `backbone_projection` 降维到 d_feature=128
3. EpipolarTransformer: 对每个视图的每个像素，在另一视图的对应极线上采样 32 个特征点做 cross-attention
4. 高分辨率 skip connection: 原始 RGB 经 7×7 Conv 加到特征上
5. DepthPredictorMonocular: 预测 32 个深度 bin 的 PDF + offset，采样得到深度
6. to_gaussians: 预测 offset_xy(2) + scale(3) + rotation(4) + sh(75)
7. GaussianAdapter: 将上述参数 + 深度 → 世界坐标 means/covariances/harmonics/opacities

**Backbone 处理** — 两视图独立编码，batch 维度合并 (`src/model/encoder/backbone/backbone_resnet.py:71-72`):
```python
b, v, _, h, w = context["image"].shape
x = rearrange(context["image"], "b v c h w -> (b v) c h w")
```

**极线采样中"其他视图"的索引** (`src/model/encoder/epipolar/epipolar_sampler.py:44-49`):
```python
_, index_v = generate_heterogeneous_index(num_views)
t_v, t_ov = generate_heterogeneous_index_transpose(num_views)
```
对于 2 视图，other_view 维度为 1，即每个视图只关注另一个视图。

**下采样与上采样** (`src/model/encoder/epipolar/epipolar_transformer.py:67-74`):
```python
if cfg.downscale:
    self.downscaler = nn.Conv2d(d_in, d_in, cfg.downscale, cfg.downscale)
    self.upscaler = nn.ConvTranspose2d(d_in, d_in, cfg.downscale, cfg.downscale)
    self.upscale_refinement = nn.Sequential(
        nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
        nn.GELU(),
        nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
    )
```
默认 downscale=4，即 256×256 输入在 64×64 分辨率做极线注意力。

## 硬编码参数与配置点

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| d_feature | 128 | `config/model/encoder/epipolar.yaml:23` | 特征维度 |
| num_monocular_samples | 32 | `config/model/encoder/epipolar.yaml:11` | 深度 bin 数 |
| num_surfaces | 1 | `config/model/encoder/epipolar.yaml:12` | 每像素表面数 |
| gaussians_per_pixel | 3 | `config/model/encoder/epipolar.yaml:16` | 训练时每像素采样高斯数 |
| sh_degree | 4 | `config/model/encoder/epipolar.yaml:21` | 球谐阶数 → d_sh=25 |
| gaussian_scale_min/max | 0.5 / 15.0 | `config/model/encoder/epipolar.yaml:19-20` | 高斯缩放范围 |
| epipolar num_samples | 32 | `config/model/encoder/epipolar.yaml:37` | 极线采样点数 |
| epipolar num_layers | 2 | `config/model/encoder/epipolar.yaml:35` | 极线 Transformer 层数 |
| epipolar num_heads | 4 | `config/model/encoder/epipolar.yaml:36` | 注意力头数 |
| epipolar d_dot | 128 | `config/model/encoder/epipolar.yaml:38` | 每头维度 |
| epipolar d_mlp | 256 | `config/model/encoder/epipolar.yaml:39` | FFN 隐层维度 |
| epipolar downscale | 4 | `config/model/encoder/epipolar.yaml:40` | 空间下采样倍率 |
| self_attention patch_size | 4 | `config/model/encoder/epipolar.yaml:27` | 图像自注意力 patch 大小 |
| self_attention num_layers | 2 | `config/model/encoder/epipolar.yaml:29` | 自注意力 Transformer 层数 |
| num_octaves (depth PE) | 10 | `config/model/encoder/epipolar.yaml:34` | 深度位置编码频率数 |
| image_shape | [256, 256] | `config/experiment/re10k.yaml:14` | 输入分辨率 |
| batch_size | 7 | `config/experiment/re10k.yaml:19` | 训练批大小(需80GB显存) |
| max_steps | 300_001 | `config/experiment/re10k.yaml:22` | 训练步数 |
| scale multiplier | 0.1 | `gaussian_adapter.py:102` (函数默认参数) | 高斯尺度与像素大小的比例系数 |
| sh_mask 衰减 | 0.1 * 0.25^degree | `gaussian_adapter.py:46` | 高阶球谐初始化衰减 |
| opacity_mapping initial/final | 0.0 / 0.0 | `config/model/encoder/epipolar.yaml:7-8` | PDF→不透明度映射指数（实际为恒等） |
| near_disparity | 3.0 | `config/model/encoder/epipolar.yaml:14` | 近界视差系数 |

**如何改为可配置**: 所有上述参数已在 Hydra YAML 中暴露，可通过命令行覆盖（如 `python3 -m src.main +experiment=re10k model.encoder.epipolar_transformer.num_layers=4`）。`scale multiplier=0.1` 和 `sh_mask` 衰减是硬编码在代码中的，需修改 `gaussian_adapter.py` 才能调整。

## 环境与复现

**依赖** (`requirements.txt`):
- Python 3.10+
- PyTorch + CUDA 12.1（diff-gaussian-rasterization 需匹配 CUDA 版本编译）
- 核心: lightning, hydra-core, einops, jaxtyping, wandb, timm, e3nn, lpips
- CUDA 光栅化: `git+https://github.com/dcharatan/diff-gaussian-rasterization-modified`

**安装**:
```bash
python3.10 -m venv venv && source venv/bin/activate
pip install wheel torch torchvision torchaudio
pip install -r requirements.txt
```

**数据集**: RealEstate10k / ACID，需预处理为 ~100MB chunk 格式，放入 `datasets/` 目录。

**预训练权重**: [Google Drive](https://drive.google.com/drive/folders/1ZYInQyBHav979dH7arITG8Z-wTSR_Bkm)，下载后放 `checkpoints/` 目录。

**最小运行命令**:
```bash
# 训练 (单卡 A100 80GB)
python3 -m src.main +experiment=re10k

# 降低显存
python3 -m src.main +experiment=re10k data_loader.train.batch_size=1

# 评估
python3 -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/evaluation_index_re10k.json \
  checkpointing.load=checkpoints/re10k.ckpt
```

## 改造接口点

### 替换/修改极线注意力
- **最小侵入点**: `src/model/encoder/epipolar/epipolar_transformer.py:56-65` — `Transformer` 实例化处。可替换为自定义注意力模块，只需保持接口 `forward(q, kv, b=, v=, h=, w=)` 不变。
- **采样策略修改**: `src/model/encoder/epipolar/epipolar_sampler.py:79-88` — 修改极线采样点生成逻辑（如改为非均匀采样、学习型采样）。
- **关闭极线注意力做消融**: 配置 `use_epipolar_transformer: false`（已有 ablation 配置）。

### 修改高斯预测头
- **增加/减少输出参数**: `src/model/encoder/encoder_epipolar.py:85-91` — 修改 `to_gaussians` 的输出维度。
- **修改 GaussianAdapter**: `src/model/encoder/common/gaussian_adapter.py:48-95` — 修改 scale/rotation/sh 的解码方式。`d_in` 属性 (line 115-116) 决定输入维度。
- **增加每像素高斯数**: 配置 `gaussians_per_pixel` 和 `num_surfaces`。

### 扩展到多视图/视频
- 已支持任意视图数（`num_context_views` 配置），>2 视图时自动添加 view_embeddings。
- 3-view 配置: `config/experiment/re10k_3_view.yaml`。
- 显存随视图数线性增长，是主要瓶颈。

### 修改深度预测
- **深度 bin 数**: 配置 `num_monocular_samples`。
- **采样策略**: `src/model/encoder/epipolar/distribution_sampler.py` — 训练时随机采样，测试时 top-k。
- **Transmittance-based opacity**: 配置 `use_transmittance: true`。

## 风险与未知

1. **`generate_heterogeneous_index` 的具体实现**未深入阅读（`src/misc/heterogeneous_pairings.py`），不确定 >2 视图时索引生成的边界行为。
2. **`get_depth` 和 `project_rays`** 的极线几何实现（`src/geometry/epipolar_lines.py`）未逐行验证，不确定退化情况（平行光轴、极线在图像外）的处理是否完备。
3. **DINO backbone 的 `get_intermediate_layers`** 取哪一层特征未确认（代码中 `[0]` 可能是最后一层）。
4. **CUDA 光栅化器** (`diff-gaussian-rasterization-modified`) 是 fork 版本，与原版 INRIA 实现的差异未查证。
5. **`rotate_sh`** 的实现（`src/misc/sh_rotation.py`）使用 e3nn，具体旋转方式未验证。
6. **训练时 `gaussians_per_pixel=3` 但测试时为 1** 的行为差异——测试时取 top-1 深度 bin，可能丢失多表面信息。
7. **LPIPS loss 的输入范围 bug**（README 已提及），当前代码是否已修复未确认。
