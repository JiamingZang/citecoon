# DUSt3R 工程侦察卡

> 仓库：https://github.com/naver/dust3r  
> 许可：CC BY-NC-SA 4.0（非商用）  
> 侦察日期：2026-07-21

---

## 架构总览

DUSt3R 是一个基于 CroCo v2 预训练 backbone 的非对称立体视觉模型，核心思想是将多视图重建问题分解为"逐对推理 + 全局对齐优化"两阶段。模型本身是严格的双视图架构（AsymmetricCroCo3DStereo），输入两张图像，输出逐像素 3D 点图（pointmap）和置信度图；多视图场景通过枚举图像对、批量推理、再用可微全局对齐（PointCloudOptimizer）融合所有对的预测来恢复一致的相机位姿和稠密点云。

```
dust3r/
├── dust3r/                  # 核心库
│   ├── model.py             # AsymmetricCroCo3DStereo 模型定义
│   ├── inference.py         # 批量逐对推理入口
│   ├── image_pairs.py       # 场景图→图像对生成（complete/swin/oneref/logwin）
│   ├── patch_embed.py       # 多宽高比 Patch Embedding
│   ├── post_process.py      # 从点图估计焦距（Weiszfeld）
│   ├── heads/               # 输出头（linear_head / dpt_head / postprocess）
│   ├── cloud_opt/           # 全局对齐优化器
│   │   ├── __init__.py      # global_aligner 工厂
│   │   ├── base_opt.py      # BasePCOptimizer + 优化循环
│   │   ├── optimizer.py     # PointCloudOptimizer（深度图/位姿/焦距可学习）
│   │   ├── pair_viewer.py   # PairViewer（仅两视图，PnP 直接求解）
│   │   ├── init_im_poses.py # MST 初始化 + rigid registration + fast_pnp
│   │   └── commons.py       # 工具函数（conf变换/距离/LR schedule）
│   ├── utils/               # image.py / geometry.py / device.py
│   └── datasets/            # 训练数据集与采样器
├── croco/                   # CroCo v2 backbone（git submodule）
├── dust3r_visloc/           # 视觉定位 pipeline（PnP 多后端）
├── datasets_preprocess/     # 各数据集预处理脚本
├── demo.py                  # Gradio 交互 demo 入口
├── train.py                 # 训练入口
└── visloc.py                # 视觉定位 CLI
```

---

## 关键事实

### F1: 模型前向输出结构——点图 + 置信度

模型 `forward()` 严格接收两个视图，返回两个 dict：

```python
# dust3r/model.py:199-211
def forward(self, view1, view2):
    (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
    dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
    with torch.cuda.amp.autocast(enabled=False):
        res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
    res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
    return res1, res2
```

- `res1 = {'pts3d': (B,H,W,3), 'conf': (B,H,W)}`——view1 像素在 **view1 相机坐标系**下的 3D 坐标
- `res2 = {'pts3d_in_other_view': (B,H,W,3), 'conf': (B,H,W)}`——view2 像素在 **view1 相机坐标系**下的 3D 坐标

### F2: 点图解码方式（depth_mode='exp'）

```python
# dust3r/heads/postprocess.py:22-44
def reg_dense_depth(xyz, mode):
    mode, vmin, vmax = mode
    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds
    ...
    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    if mode == 'exp':
        return xyz * torch.expm1(d)
```

网络输出 3 通道向量，方向=射线方向，模长经 `expm1` 映射为到相机原点的距离。默认 `depth_mode=('exp', -inf, inf)`（model.py:61）。

### F3: 置信度解码（conf_mode=('exp', 1, inf)）

```python
# dust3r/heads/postprocess.py:49-55
def reg_dense_conf(x, mode):
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
```

置信度 = `1 + exp(raw)`，恒 ≥ 1，值越大越可信。

### F4: 相机内参——后验估计而非网络直出

DUSt3R **不直接预测内参**。焦距从点图反投影关系估计：

```python
# dust3r/post_process.py:12-30（estimate_focal_knowing_depth）
fx_votes = (u * z) / x
fy_votes = (v * z) / y
f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
focal = torch.nanmedian(f_votes, dim=-1).values
```

主点默认图像中心 `(W/2, H/2)`。全局对齐中焦距为可学习参数：

```python
# dust3r/cloud_opt/optimizer.py:127-129
def get_focals(self):
    log_focals = torch.stack(list(self.im_focals), dim=0)
    return (log_focals / self.focal_break).exp()
```

内参矩阵组装：

```python
# dust3r/cloud_opt/optimizer.py:144-150
def get_intrinsics(self):
    K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
    focals = self.get_focals().flatten()
    K[:, 0, 0] = K[:, 1, 1] = focals
    K[:, :2, 2] = self.get_principal_points()
    K[:, 2, 2] = 1
    return K
```

### F5: 相机外参——全局对齐优化得到（cam-to-world 4x4）

位姿参数化为 7-DOF（单位四元数 + signed-log 平移）：

```python
# dust3r/cloud_opt/base_opt.py:150-155
def _get_poses(self, poses):
    Q = poses[:, :4]
    T = signed_expm1(poses[:, 4:7])
    RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
    return RT
```

```python
# dust3r/cloud_opt/optimizer.py:152-154
def get_im_poses(self):  # cam to world
    cam2world = self._get_poses(self.im_poses)
    return cam2world
```

### F6: 两视图 vs 多视图的分支逻辑

```python
# dust3r/demo.py:155-163
pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=1, verbose=not silent)
mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
if mode == GlobalAlignerMode.PointCloudOptimizer:
    loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
```

- **2 张图**→ PairViewer：无迭代优化，直接 `cv2.solvePnPRansac` 求相对位姿
- **≥3 张图**→ PointCloudOptimizer：MST 初始化 + Adam 迭代优化

### F7: 多视图耗时与显存——O(N^2) 对推理 + 全局优化

```python
# dust3r/inference.py:55-68
@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1
    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))
```

- `complete` 场景图：N 张图 → N*(N-1) 个有向对（symmetrize 后）
- 不同分辨率时 batch_size 强制为 1
- 每批结果立即 `to_cpu`，GPU 显存峰值 ≈ 单次双视图前向
- 全局优化显存 ∝ `n_edges × max_area × 3`（存储所有对的点云预测）

### F8: 对称编码优化——减半 encoder 计算

```python
# dust3r/model.py:153-170（_encode_symmetrized）
if is_symmetrized(view1, view2):
    # computing half of forward pass!'
    feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
    feat1, feat2 = interleave(feat1, feat2)
    pos1, pos2 = interleave(pos1, pos2)
```

当 batch 包含 (i,j) 和 (j,i) 时，encoder 只跑一半再 interleave。

### F9: 能否只喂裁剪后的物体区域——可以，需手动构造输入

模型推理不需要相机内参输入，直接回归 3D 点。只需保证尺寸被 patch_size=16 整除：

```python
# dust3r/patch_embed.py:22-23（PatchEmbedDust3R assert）
# 图像 H, W 必须被 patch_size 整除
```

手动构造输入 dict 即可绕过 `load_images()` 的自动裁剪：

```python
dict(img=ImgNorm(cropped_pil)[None], true_shape=np.int32([H, W]), idx=i, instance=str(i))
```

注意：模型在全图（512×384 等）上训练，紧密裁剪可能降低精度和置信度可靠性。

### F10: 下游位姿精修对接——可用中间量

| 中间量 | 获取方式 | 格式 | 坐标系 |
|--------|----------|------|--------|
| 逐像素 3D 点（pairwise） | `pred1['pts3d']`, `pred2['pts3d_in_other_view']` | `(B,H,W,3)` | view1 相机系 |
| 逐像素置信度 | `pred1['conf']`, `pred2['conf']` | `(B,H,W)` ≥1 | — |
| 全局点云 | `scene.get_pts3d()` | list of `(H,W,3)` | 世界系（首帧=identity） |
| 深度图 | `scene.get_depthmaps()` | list of `(H,W)` | 各相机系 |
| 焦距 | `scene.get_focals()` | `(N,)` | 像素 |
| 内参矩阵 | `scene.get_intrinsics()` | `(N,3,3)` | pinhole 无畸变 |
| 位姿 cam2world | `scene.get_im_poses()` | `(N,4,4)` | 世界系 |
| 置信度（全局） | `scene.im_conf` | list of `(H,W)` | — |

### F11: 仓库中已有的 PnP 实现

```python
# dust3r/cloud_opt/init_im_poses.py:272-273
success, R, T, inliers = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
    iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
```

视觉定位模块支持三种 PnP 后端（cv2 / poselib / pycolmap）：

```python
# dust3r_visloc/localization.py:37-38
confidence = 0.9999
iterationsCount = 10_000
```

**无 ICP 实现**。刚体配准使用 `roma.rigid_points_registration`（带置信度加权 + 尺度估计）：

```python
# dust3r/cloud_opt/init_im_poses.py:220-223
def rigid_points_registration(pts1, pts2, conf):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True)
    return s, R, T
```

### F12: 置信度在全局对齐中作为逐像素权重

```python
# dust3r/cloud_opt/base_opt.py:46,251-264
conf='log'  # 默认变换
weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
```

可直接用于下游 PnP/ICP 的加权：`conf >= thr` 做硬掩码，或 `log(conf)` 做软权重。

---

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|-----|------|---------------|
| patch_size | 16 | `model.py:87` | 需重训；推理时只要求输入尺寸整除 |
| 图像尺寸 | 512 或 224 | `demo.py:36` choices=[512,224] | 改 choices 或绕过 demo 直接构造 |
| depth_mode | ('exp', -inf, inf) | 烘焙在 checkpoint args 字符串 | 需重训 |
| conf_mode | ('exp', 1, inf) | 同上 | 需重训 |
| min_conf_thr | 3 | `base_opt.py:47` | 构造器参数，已可配 |
| 全局对齐 niter | 300 | `base_opt.py:326` | 函数参数，已可配 |
| 全局对齐 lr | 0.01 | `base_opt.py:326` | 函数参数，已可配 |
| Adam betas (对齐) | (0.9, 0.9) | `base_opt.py:337` | 硬编码，需改源码 |
| lr_min | 1e-6 | `base_opt.py:326` | 函数参数 |
| schedule | 'cosine' | `base_opt.py:326` | 函数参数 |
| focal_break | 20 | `optimizer.py:22` | 硬编码 |
| pw_break | 20 | `base_opt.py:50` | 硬编码 |
| base_scale | 0.5 | `base_opt.py:48` | 硬编码 |
| POSE_DIM | 7 | `base_opt.py:89` | 硬编码 |
| PnP reprojectionError (init) | 5 px | `init_im_poses.py:273` | 硬编码 |
| PnP iterationsCount (init) | 10 | `base_opt.py:276` niter_PnP | 函数参数 |
| PnP iterationsCount (PairViewer) | 100 | `pair_viewer.py:56` | 硬编码 |
| PnP RANSAC confidence (visloc) | 0.9999 | `localization.py:37` | 硬编码 |
| Weiszfeld iterations (focal) | 10 | `post_process.py:47` | 硬编码 |
| 图像归一化 | [-1, 1] (mean=0.5, std=0.5) | `utils/image.py:23` | 硬编码 |
| 距离度量 | L1 | `base_opt.py:45` dist='l1' | 构造器参数 |
| conf 变换 | 'log' | `base_opt.py:46` | 构造器参数 |
| optimize_pp | False | `optimizer.py:34` | 构造器参数 |
| Encoder dim/depth/heads | 1024/24/16 (ViT-L) | checkpoint args | 需重训 |
| Decoder dim/depth/heads | 768/12/12 (ViT-B) | checkpoint args | 需重训 |

---

## 环境与复现

### 依赖

```
# requirements.txt
torch, torchvision, roma, gradio, matplotlib, tqdm,
opencv-python, scipy, einops, trimesh, tensorboard,
pyglet<2, huggingface-hub[torch]>=0.22
```

可选：`pillow-heif, pyrender, kapture, numpy-quaternion, pycolmap, poselib`

### 环境搭建

```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
# 可选：编译 RoPE CUDA kernel
cd croco/models/curope/ && python setup.py build_ext --inplace
```

### 权重下载

```bash
# 三个预训练权重（ViT-L encoder + ViT-B decoder）
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
# 或 HuggingFace: "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
```

| 权重 | 训练分辨率 | 输出头 |
|------|-----------|--------|
| DUSt3R_ViTLarge_BaseDecoder_224_linear.pth | 224×224 | Linear |
| DUSt3R_ViTLarge_BaseDecoder_512_linear.pth | 512×{384,336,288,256,160} | Linear |
| DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth | 512×{384,336,288,256,160} | DPT |

### 最小运行命令

```bash
# Gradio demo（自动下载权重）
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt

# 编程式推理
python3 -c "
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

model = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt').to('cuda')
imgs = load_images(['a.png','b.png'], size=512)
pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
out = inference(pairs, model, 'cuda', batch_size=1)
scene = global_aligner(out, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
print(scene.get_im_poses(), scene.get_focals())
"
```

---

## 改造接口点

### 1. 只喂裁剪物体区域（最小侵入）

**位置**：绕过 `dust3r/utils/image.py:74` 的 `load_images()`，手动构造输入 dict 列表。

```python
from dust3r.utils.image import ImgNorm
import numpy as np

def make_crop_input(cropped_pil, idx):
    W, H = cropped_pil.size
    assert W % 16 == 0 and H % 16 == 0, "尺寸须被 patch_size=16 整除"
    return dict(img=ImgNorm(cropped_pil)[None], true_shape=np.int32([[H, W]]), idx=idx, instance=str(idx))
```

后续 `make_pairs → inference → global_aligner` 流程不变。若需恢复原始全图坐标系下的位姿，需记录裁剪偏移并修正内参主点。

### 2. 提取中间量用于下游 PnP/ICP/渲染比较

**位置**：`global_aligner` 返回的 scene 对象已暴露所有需要的接口（见 F10 表格）。

- **PnP 精修**：取 `scene.get_pts3d()` 世界点 + 对应像素坐标 + `scene.get_intrinsics()` → `cv2.solvePnPRansac`
- **ICP 精修**：取 `scene.get_pts3d()` 作为 source/target 点云，用 `scene.im_conf` 做加权；仓库无现成 ICP，需外接 Open3D 等
- **渲染比较**：取 `scene.get_depthmaps()` + `scene.get_im_poses()` + `scene.get_intrinsics()` → 用 pyrender/nvdiffrast 渲染深度/法线再比较

### 3. 冻结部分位姿做增量精修

**位置**：`dust3r/cloud_opt/modular_optimizer.py` 的 `ModularPointCloudOptimizer`

```python
# 支持已知位姿初始化 + 冻结
scene.compute_global_alignment(init='known_poses', niter=300)
# 或手动：
scene.im_poses[i].requires_grad_(False)
```

### 4. 调整全局对齐精度/速度权衡

**位置**：`compute_global_alignment()` 调用处

- 减少 `niter`（默认 300）→ 加速但精度降
- 改 `scene_graph='swin-5'` 或 `'oneref-0'` → 减少图像对数 → 推理阶段 O(N) 而非 O(N^2)
- 改 `schedule='linear'` 可能收敛更快

### 5. 替换/增加 PnP 后端

**位置**：`dust3r_visloc/localization.py:30-140`

已支持 cv2/poselib/pycolmap 三后端，新增后端只需在 `run_pnp()` 中加一个 elif 分支。

---

## 风险与未知

1. **croco/ 子模块未拉取**：本次浅克隆未递归拉取 submodule，CroCo v2 backbone 源码（encoder/decoder 具体实现、RoPE 位置编码）未验证。`git submodule update --init --recursive` 后方可查看。

2. **裁剪输入的精度影响未量化**：模型在 512×{384,336,...} 全图上训练，紧密物体裁剪（如 128×128）是否严重降低点图质量和焦距估计精度，无实验数据。

3. **显存峰值未实测**：理论上单对前向 ≈ ViT-L encoder + ViT-B decoder × 2 路，512×384 分辨率下约需 8-12 GB VRAM（推测），未实际 profiling。

4. **全局对齐对 N 较大时的可扩展性**：`complete` 场景图 O(N^2) 对，N>20 时推理时间可能不可接受；`swin`/`oneref` 可缓解但牺牲全局一致性，具体精度损失未查证。

5. **尺度歧义**：DUSt3R 输出为 up-to-scale 重建（除非有已知尺度约束），`rigid_points_registration` 中 `compute_scaling=True` 表明对齐允许相似变换。下游 6DoF 精修若需绝对尺度，需额外约束。

6. **非商用许可**：CC BY-NC-SA 4.0 限制商业使用，需确认项目合规。

7. **DPT head vs Linear head 精度/速度差异**：DPT 使用多层 decoder 特征（hooks at [0, 6, 9, 12]），理论上更精确但更慢，具体 benchmark 数据未在代码中找到。

8. **roma 库版本兼容性**：`roma.rigid_points_registration` 和 `roma.RigidUnitQuat` 的 API 在不同 roma 版本间可能有变化，requirements.txt 未锁定版本。
