# MVSplat 仓库侦察卡

- 仓库：https://github.com/donydchen/mvsplat （ECCV 2024 Oral）
- 本地路径：`./mvsplat/`（已存在，未做克隆；未修改任何仓库代码）
- 研究关注点：① 前馈高斯参数预测头位置；② cost volume 构建位置

## 架构总览

MVSplat 是一个前馈（feed-forward）式稀疏多视图 3D 高斯重建网络：输入若干 context 视图，一次前向直接回归每个像素的高斯参数，再用可微 splatting 渲染目标视图。主干为多视图 Transformer（基于 UniMatch backbone + cross-view attention）提取逐视图特征；随后进入 `DepthPredictorMultiView`，通过平面扫描（plane-sweep）构建 cost volume 得到粗深度，再用两个 2D U-Net 分别做 cost volume 精修与深度精修，最后由两个卷积头回归高斯原始参数（协方差/颜色 与 中心/不透明度）；`GaussianAdapter` 把原始参数解码为世界坐标系下的 means/covariances/harmonics/opacities，交给 CUDA splatting 解码器渲染。

目录树摘要（仅核心）：
```
src/model/
├── encoder/
│   ├── encoder_costvolume.py        # 编码器入口：backbone→depth_predictor→gaussian_adapter
│   ├── backbone/                    # 多视图 Transformer（unimatch + cross-view attn）
│   ├── costvolume/
│   │   ├── depth_predictor_multiview.py  # ★cost volume 构建 + ★高斯预测头
│   │   └── ldm_unet/unet.py         # 2D U-Net（cost volume / depth 精修）
│   └── common/
│       ├── gaussian_adapter.py      # ★原始参数→世界高斯 解码
│       └── gaussians.py             # build_covariance
├── decoder/decoder_splatting_cuda.py # 可微高斯 splatting 渲染
└── model_wrapper.py                  # 训练/推理封装
config/model/encoder/costvolume.yaml  # 编码器超参
```

## 关键事实

### 关注点①：前馈高斯参数预测头

高斯参数预测分两个卷积头，均定义在 `DepthPredictorMultiView` 中：

**头A — 协方差 + 颜色（scale/rotation/SH）**
`src/model/encoder/costvolume/depth_predictor_multiview.py:242-250`
```python
# Gaussians prediction: covariance, color
gau_in = depth_unet_feat_dim + 3 + feature_channels
self.to_gaussians = nn.Sequential(
    nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
    nn.GELU(),
    nn.Conv2d(
        gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
    ),
)
```
前向调用 `depth_predictor_multiview.py:353-360`：
```python
# gaussians head
raw_gaussians_in = [refine_out,
                    extra_info["images"], proj_feat_in_fullres]
raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
raw_gaussians = self.to_gaussians(raw_gaussians_in)
raw_gaussians = rearrange(
    raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
)
```

**头B — 中心（深度/视差）+ 不透明度**
`src/model/encoder/costvolume/depth_predictor_multiview.py:252-260`
```python
# Gaussians prediction: centers, opacity
if not wo_depth_refine:
    channels = depth_unet_feat_dim
    disps_models = [
        nn.Conv2d(channels, channels * 2, 3, 1, 1),
        nn.GELU(),
        nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
    ]
    self.to_disparity = nn.Sequential(*disps_models)
```
前向调用（视差增量 + 密度）`depth_predictor_multiview.py:379-405`：
```python
# delta fine depth and density
delta_disps_density = self.to_disparity(refine_out)
delta_disps, raw_densities = delta_disps_density.split(
    gaussians_per_pixel, dim=1
)
densities = repeat(F.sigmoid(raw_densities), ...)
fine_disps = (fullres_disps + delta_disps).clamp(
    1.0 / rearrange(far, ...), 1.0 / rearrange(near, ...))
depths = 1.0 / fine_disps
```

**输出通道数如何确定**（`num_surfaces * (d_in + 2)`，+2 为 xy 偏移）
`src/model/encoder/encoder_costvolume.py:116`
```python
gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
```
`d_in = 7 + 3*d_sh`（7=3 scale + 4 rotation），`src/model/encoder/common/gaussian_adapter.py:115-117`：
```python
@property
def d_in(self) -> int:
    return 7 + 3 * self.d_sh
```

**原始参数解码为世界高斯**（split 为 scale/rotation/sh，再算 means/cov）
`src/model/encoder/common/gaussian_adapter.py:60`
```python
scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
```
means 由射线 + 深度得到 `gaussian_adapter.py:84-85`：
```python
origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
means = origins + directions * depths[..., None]
```
最终 `Gaussians` 组装在 `encoder_costvolume.py:230-247`（means/covariances/harmonics/opacities）。

### 关注点②：cost volume 构建

**深度候选（逆深度均匀采样）** `depth_predictor_multiview.py:114-122`
```python
min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
depth_candi_curr = (
    min_depth
    + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
    * (max_depth - min_depth)
).type_as(features)
```

**平面扫描单应 warp**（按深度候选把源视图特征投到参考视图）
`depth_predictor_multiview.py:10-70`，函数 `warp_with_pose_depth_candidates(...)`，核心采样 `:60-68`：
```python
warped_feature = F.grid_sample(
    feature1, grid.view(b, d * h, w, 2),
    mode="bilinear", padding_mode=warp_padding_mode, align_corners=True,
).view(b, c, d, h, w)  # [B, C, D, H, W]
```

**cost volume 主体（点积相似度 + 多源视图平均）** `depth_predictor_multiview.py:292-318`
```python
# cost volume constructions
feat01 = feat_comb_lists[0]
if self.wo_cost_volume:
    raw_correlation_in = feat01
else:
    raw_correlation_in_lists = []
    for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
        feat01_warped = warp_with_pose_depth_candidates(
            feat10, intr_curr, pose_curr,
            1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
            warp_padding_mode="zeros",
        )  # [B, C, D, H, W]
        raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(1) / (c**0.5)
        raw_correlation_in_lists.append(raw_correlation_in)
    raw_correlation_in = torch.mean(
        torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False)
    raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)
```

**cost volume 精修（2D U-Net + 残差跳连）** 定义 `:167-191`，前向 `:320-328`：
```python
raw_correlation = self.corr_refine_net(raw_correlation_in)
raw_correlation = raw_correlation + self.regressor_residual(raw_correlation_in)
```

**softmax 粗深度** `:330-336`：
```python
pdf = F.softmax(self.depth_head_lowres(raw_correlation), dim=1)
coarse_disps = (disp_candi_curr * pdf).sum(dim=1, keepdim=True)
```

## 硬编码参数与配置点

| 参数 | 默认值 | 位置 | 说明/改法 |
|---|---|---|---|
| `num_depth_candidates` | 32 | `config/model/encoder/costvolume.yaml`；用于 `depth_predictor_multiview.py:119` linspace | cost volume 深度 bin 数，yaml 直接改 |
| `num_surfaces` | 1 | costvolume.yaml；`encoder_costvolume.py:116` | 每像素高斯层数，影响输出通道 |
| `gaussians_per_pixel` | 1 | costvolume.yaml；`depth_predictor_multiview.py:258` | 每像素高斯数 |
| `d_feature` | 128 | costvolume.yaml；backbone/cost volume 通道 | 特征维度 |
| `sh_degree` | 4 | costvolume.yaml→`gaussian_adapter.py:112` | `d_sh=(sh_degree+1)^2=25` |
| `gaussian_scale_min/max` | 0.5 / 15.0 | costvolume.yaml→`gaussian_adapter.py:63-65` | scale sigmoid 映射范围 |
| `downscale_factor` | 4 | costvolume.yaml；`depth_predictor_multiview.py:205` upsample | 特征下采样倍率 |
| scale multiplier | 0.1 | `gaussian_adapter.py:102`（函数默认参，**未进 yaml**） | 高斯尺度系数，需改代码或提为 cfg 才能配置 |
| `clamp_min_depth` | 1e-3 | `depth_predictor_multiview.py:15`（函数默认参） | warp 除零保护，硬编码 |
| GroupNorm 组数 | 8 / 4 | `depth_predictor_multiview.py:169, 224` | 硬编码在精修网络中 |
| cost volume 相似度归一 | `/ (c**0.5)` | `depth_predictor_multiview.py:311` | 点积除以 sqrt(通道数)，硬编码 |
| `unimatch_weights_path` | `checkpoints/gmdepth-...pth` | costvolume.yaml；加载于 `encoder_costvolume.py:87-103` | backbone 预训练权重路径 |

改成可配置的通用做法：在 `EncoderCostVolumeCfg`（`encoder_costvolume.py:36-60`）新增字段 → costvolume.yaml 填值 → 透传到 `DepthPredictorMultiView.__init__`。

## 环境与复现

依赖（`requirements.txt`）：`pytorch_lightning, hydra-core, einops, jaxtyping, beartype, wandb, timm, lpips, e3nn, opencv-python==4.6.0.66`，以及自定义可微光栅化 `git+https://github.com/dcharatan/diff-gaussian-rasterization-modified`。

README 安装（Python 3.10+，CUDA 11.8）：
```bash
conda create -n mvsplat python=3.10 && conda activate mvsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

权重下载：
- 预训练模型（Google Drive，存到 `/checkpoints`）：`https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU`
- backbone（UniMatch）权重：
  `wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints`

最小运行命令（测试/评估，README）：
```bash
python -m src.main +experiment=re10k \
  checkpointing.load=checkpoints/re10k.ckpt \
  mode=test \
  test.compute_scores=true
```
训练：`python -m src.main +experiment=re10k data_loader.train.batch_size=14`。
数据集：RealEstate10K / ACID（同 pixelSplat 格式，解压到 `datasets/`），DTU 仅测试。

## 改造接口点

- **替换/增强高斯预测头**：最小侵入点是 `depth_predictor_multiview.py:242-260` 的 `self.to_gaussians` 与 `self.to_disparity` 两个 `nn.Sequential`。只需保持输出通道数 `gaussian_raw_channels`（头A）与 `gaussians_per_pixel*2`（头B）不变，即可换成更强的卷积/注意力头而不触动 `GaussianAdapter` 与渲染管线。
- **增加每像素高斯数 / 表面数**：改 `costvolume.yaml` 的 `gaussians_per_pixel`、`num_surfaces` 即可，通道数在 `encoder_costvolume.py:116` 自动推导。
- **改造 cost volume 构建**：核心在 `depth_predictor_multiview.py:292-318`（相似度计算与多视图聚合）与 `warp_with_pose_depth_candidates`（`:10-70`）。若要换相似度度量（如分组相关）或改深度采样（`:114-122` 的逆深度 linspace），都集中在此文件。
- **改深度 bin 数/范围**：`num_depth_candidates` 走 yaml；近远界由数据集 `near/far` 提供，非网络硬编码。
- **解码逻辑（scale 映射、SH 掩码、means 计算）**：集中在 `gaussian_adapter.py:48-96`，与预测头解耦，可独立替换。

## 风险与未知

- `GaussianAdapter.get_scale_multiplier` 的 `multiplier=0.1`（`gaussian_adapter.py:102`）与 warp 的 `clamp_min_depth=1e-3` 为函数默认参，未确认是否有上层调用覆盖；改造时需全局搜索确认。
- 未实际运行训练/推理，`requirements_w_version.txt` 的精确版本锁定未核对；`diff-gaussian-rasterization-modified` 需 CUDA 编译，未验证本机（darwin/arm64）可装性——此环境大概率无法直接跑 GPU 渲染。
- cost volume U-Net（`ldm_unet/unet.py`）内部 `num_head_channels=32`、`num_res_blocks=1` 等是否全部由 cfg 透传未逐一核对（`depth_predictor_multiview.py:171-184, 226-239` 部分为硬编码）。
- 预训练权重与数据集均需外部下载，未在本地验证可用性。
- backbone（`backbone/multiview_transformer.py`、unimatch）内部细节未深入，仅确认其为特征提取前端。
