# gaussian-splatting 工程侦察卡片

> 仓库：https://github.com/graphdeco-inria/gaussian-splatting （浅克隆，master 分支）
> 研究关注点：为免训练管线提供代码级落地路径——MASt3R 稠密匹配接口、3DGS 可微渲染入口与位姿梯度、物体级最小训练配置。

---

## 架构总览（模块划分，一段话+目录树摘要）

本仓库是 3D Gaussian Splatting (SIGGRAPH 2023) 的官方实现，核心流程为：COLMAP/NeRF-Synthetic 数据加载 → 初始化 3D 高斯点云 → 可微光栅化渲染 → L1+SSIM 损失优化 → 自适应密度控制（克隆/分裂/剪枝）。渲染器是自定义 CUDA 扩展 `diff-gaussian-rasterization`，Python 侧仅做参数组装与损失计算。相机外参（R/T）以 numpy 存储、构建为无梯度的 torch 张量后传入 CUDA kernel，**当前不支持对位姿求导**。

```
gaussian-splatting/
├── train.py                  # 训练主循环
├── render.py                 # 推理渲染
├── metrics.py                # PSNR/SSIM/LPIPS 评估
├── convert.py                # 图片→COLMAP 数据转换
├── full_eval.py              # 论文全流程评估
├── arguments/__init__.py     # CLI 参数定义（ModelParams/PipelineParams/OptimizationParams）
├── scene/
│   ├── __init__.py           # Scene 加载调度（COLMAP / Blender）
│   ├── cameras.py            # Camera / MiniCam 类
│   ├── colmap_loader.py      # COLMAP 二进制/文本 I/O
│   ├── dataset_readers.py    # readColmapSceneInfo / readNerfSyntheticInfo
│   └── gaussian_model.py     # GaussianModel（nn.Module，高斯属性+优化器+密度控制）
├── gaussian_renderer/
│   ├── __init__.py           # render() 函数——调用 CUDA 光栅器
│   └── network_gui.py        # SIBR 远程查看器桥接
├── utils/
│   ├── camera_utils.py       # loadCam、分辨率缩放
│   ├── graphics_utils.py     # getWorld2View2、getProjectionMatrix
│   ├── loss_utils.py         # l1_loss、ssim、FusedSSIMMap
│   └── ...
├── lpipsPyTorch/             # LPIPS 感知损失
└── submodules/
    ├── diff-gaussian-rasterization/  # CUDA 可微光栅器（branch dr_aa / 3dgs_accel）
    ├── simple-knn/                   # KNN（密度控制用）
    └── fused-ssim/                   # 融合 SSIM CUDA 扩展
```

---

## 关键事实（与关注点直接相关的代码事实）

### 事实 1：render() 函数签名与相机矩阵传入方式

`gaussian_renderer/__init__.py:18`
```python
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
```

相机矩阵以 **无梯度叶张量** 传入 CUDA 光栅器（`gaussian_renderer/__init__.py:38-50`）：
```python
raster_settings = GaussianRasterizationSettings(
    ...
    viewmatrix=viewpoint_camera.world_view_transform,
    projmatrix=viewpoint_camera.full_proj_transform,
    campos=viewpoint_camera.camera_center,
    ...
)
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
```

### 事实 2：Camera 类中 R/T 的存储——无梯度

`scene/cameras.py:29-30`
```python
self.R = R        # plain numpy — NOT nn.Parameter
self.T = T        # plain numpy — NOT nn.Parameter
```

`scene/cameras.py:86-89`（构建视图矩阵）：
```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                             fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
self.full_proj_transform = (self.world_view_transform.unsqueeze(0)
                            .bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
self.camera_center = self.world_view_transform.inverse()[3, :3]
```

`torch.tensor(...)` 不设置 `requires_grad=True`，因此 **渲染前向中无梯度流经 R/T**。

### 事实 3：getWorld2View2 使用 numpy——不可自动微分

`utils/graphics_utils.py:38-49`
```python
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
```

使用 `np.linalg.inv`——完全脱离 torch autograd 图。

### 事实 4：训练循环中优化器仅覆盖高斯属性

`scene/gaussian_model.py:183-199`
```python
l = [
    {'params': [self._xyz],          'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    {'params': [self._features_dc],  'lr': training_args.feature_lr,         "name": "f_dc"},
    {'params': [self._features_rest],'lr': training_args.feature_lr / 20.0,  "name": "f_rest"},
    {'params': [self._opacity],      'lr': training_args.opacity_lr,         "name": "opacity"},
    {'params': [self._scaling],      'lr': training_args.scaling_lr,         "name": "scaling"},
    {'params': [self._rotation],     'lr': training_args.rotation_lr,        "name": "rotation"}
]
self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
```

相机参数不在优化器中。

### 事实 5：推理路径全局 torch.no_grad

`render.py:49`
```python
with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    ...
```

### 事实 6：MiniCam——轻量相机接口（可用于自定义位姿注入）

`scene/cameras.py:91-102`
```python
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar,
                 world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
```

MiniCam 接受预计算的 torch 张量——若传入 `requires_grad=True` 的矩阵，理论上梯度可回传（但 CUDA rasterizer 内部是否对 viewmatrix/projmatrix 求导需验证子模块源码）。

### 事实 7：MASt3R 不在本仓库中

本仓库不包含 MASt3R 代码。MASt3R（Dense Matching + 3D regression）是独立项目（github.com/naver/mast3r）。若需 MASt3R 稠密对应点 + 置信度作为 3DGS 初始化或位姿精修的监督信号，需外部调用 MASt3R API 后以 COLMAP 格式或自定义点云注入本管线。

### 事实 8：输入数据格式——COLMAP 约定

`scene/dataset_readers.py:145-155`（COLMAP 路径检测）：
```python
if os.path.exists(os.path.join(args.source_path, "sparse")):
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images,
                                                  args.depths, args.eval, args.train_test_exp)
```

必需文件：`sparse/0/{images,cameras,points3D}.{bin|txt}` + `images/` 目录。
相机模型限制（`scene/dataset_readers.py:88-98`）：仅 `SIMPLE_PINHOLE` 或 `PINHOLE`（无畸变）。

### 事实 9：损失函数

`train.py:119-126`
```python
Ll1 = l1_loss(image, gt_image)
ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))  # or utils ssim
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
```

默认 `lambda_dssim = 0.2`（`arguments/__init__.py:90`）。

---

## 硬编码参数与配置点

| 参数 | 默认值 | 位置 | 改为可配置方式 |
|------|--------|------|----------------|
| 总迭代数 | 30,000 | `arguments/__init__.py:76` | CLI `--iterations` 已支持 |
| 密度化窗口 | 500 → 15,000 | `arguments/__init__.py:93-94` | CLI `--densify_from_iter` / `--densify_until_iter` |
| 密度化间隔 | 100 | `arguments/__init__.py:91` | CLI `--densification_interval` |
| 不透明度重置间隔 | 3,000 | `arguments/__init__.py:92` | CLI `--opacity_reset_interval` |
| 剪枝不透明度阈值 | 0.005 | `train.py:171` 硬编码传入 `densify_and_prune(...)` | 需改代码提为参数 |
| 屏幕尺寸剪枝阈值 | 20 | `train.py:170` `size_threshold = 20 if ...` | 需改代码 |
| lambda_dssim | 0.2 | `arguments/__init__.py:90` | CLI `--lambda_dssim` |
| SH 最大阶数 | 3 | `arguments/__init__.py:49` | CLI `--sh_degree` |
| znear / zfar | 0.01 / 100.0 | `scene/cameras.py:80-81` | 硬编码，需改代码 |
| 自动降分辨率阈值 | width > 1600 | `utils/camera_utils.py:46-55` | CLI `-r 1` 可强制原始分辨率 |
| position_lr_init / final | 0.00016 / 0.0000016 | `arguments/__init__.py:77-78` | CLI 已支持 |
| opacity_lr | 0.025 | `arguments/__init__.py:82` | CLI `--opacity_lr`（README 写 0.05，代码为 0.025） |
| scaling_lr | 0.005 | `arguments/__init__.py:83` | CLI |
| rotation_lr | 0.001 | `arguments/__init__.py:84` | CLI |
| feature_lr | 0.0025 | `arguments/__init__.py:81` | CLI |
| percent_dense | 0.01 | `arguments/__init__.py:89` | CLI |
| densify_grad_threshold | 0.0002 | `arguments/__init__.py:95` | CLI |

---

## 环境与复现

### 依赖（`environment.yml`）

- Python 3.7.13 / PyTorch 1.12.1 / CUDA Toolkit 11.6（README 注明实测用 11.8；也兼容 Python 3.8 + PyTorch 2.0 + CUDA 12）
- conda: `plyfile`, `tqdm`
- pip: `opencv-python`, `joblib`
- 本地 CUDA 扩展（必须 `--recursive` 克隆后 pip install）：
  - `submodules/diff-gaussian-rasterization`
  - `submodules/simple-knn`
  - `submodules/fused-ssim`

### 硬件

- CUDA GPU，Compute Capability ≥ 7.0
- 24 GB VRAM（论文质量训练）

### 安装

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
conda env create --file environment.yml
conda activate gaussian_splatting
```

### 权重/数据下载

- 预训练模型 (14 GB): `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip`
- T&T + Deep Blending COLMAP 输入 (650 MB): `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`
- MipNeRF360: `https://jonbarron.info/mipnerf360/`

### 最小运行命令

```bash
# 训练（COLMAP 或 NeRF Synthetic 数据）
python train.py -s <path_to_dataset>

# 渲染
python render.py -m <path_to_trained_model>

# 评估
python metrics.py -m <path_to_trained_model>

# 自有图片转 COLMAP 格式（需安装 COLMAP + 可选 ImageMagick）
python convert.py -s <image_folder> [--resize]
```

---

## 改造接口点

### 关注点 1：MASt3R 稠密匹配 → 3DGS 初始化

- **注入位置**：`scene/dataset_readers.py` 的 `readColmapSceneInfo()` 返回 `SceneInfo.point_cloud`（`BasicPointCloud` namedtuple: points/colors/normals）。
- **最小改法**：在 `scene/__init__.py:77-83` 处，将 MASt3R 输出的稠密 3D 点（带置信度过滤）写为 PLY 后替换 `points3D.ply`，或直接构造 `BasicPointCloud` 传入 `gaussians.create_from_pcd()`。
- **相机格式**：MASt3R 输出的相对位姿需转为 COLMAP 的 `images.bin`（四元数+平移）和 `cameras.bin`（PINHOLE fx/fy/cx/cy），或写为 NeRF Synthetic 的 `transforms_*.json`。

### 关注点 2：Render-and-Compare 位姿精修（对 R/t 求导）

- **核心障碍**：`getWorld2View2`（`utils/graphics_utils.py:38-49`）使用 numpy，`Camera.__init__` 中 `torch.tensor(...)` 不追踪梯度。
- **最小侵入改法**：
  1. 新建 `DifferentiableCamera`（或修改 `MiniCam`），将 R/t 参数化为 `nn.Parameter`（如 6-DoF se(3) 增量 Δξ）。
  2. 用 torch 原生运算（`torch.linalg.inv`、矩阵乘法）替代 `getWorld2View2` 中的 numpy 操作，构建 `world_view_transform`。
  3. 从可微 W2V 推导 `full_proj_transform` 和 `camera_center`。
  4. 将结果传入 `render()`——`render()` 本身不阻断梯度（仅 `render.py` 推理脚本包了 `torch.no_grad`）。
  5. **关键未知**：CUDA 光栅器 `diff-gaussian-rasterization` 内部是否对 `viewmatrix`/`projmatrix` 实现了 `dL/d(viewmatrix)` 的反向传播——需查阅子模块 C++ 源码（`submodules/diff-gaussian-rasterization/cuda_rasterizer/`）。若未实现，需自行扩展或改用纯 PyTorch 可微光栅器（如 nvdiffrast）。

### 关注点 3：物体级场景最小训练配置

- 迭代数可降至 **7,000**（默认 save_iterations 含 7000，此时 SH degree 已 ramp 到 3，密度化在 500-15000 区间仍活跃）。
- 对单物体：`--iterations 7000 --densify_until_iter 5000 --sh_degree 2` 可进一步缩短。
- 输入最低要求：≥3 张已标定图片（COLMAP PINHOLE）+ SfM 稀疏点云。
- 若用 MASt3R 提供初始点云和位姿，可跳过 COLMAP，直接构造 `sparse/0/` 目录结构。

---

## 风险与未知

1. **CUDA 光栅器对 viewmatrix/projmatrix 的梯度支持**：`submodules/diff-gaussian-rasterization` 未初始化（浅克隆不含子模块内容），无法确认其 backward 是否对相机矩阵求导。这是 render-and-compare 位姿精修能否走通的**决定性未知**。需 `git submodule update --init` 后阅读 `cuda_rasterizer/backward.cu`。
2. **MASt3R 位姿精度与 COLMAP 约定对齐**：MASt3R 输出的是相对位姿（up-to-scale），与 COLMAP 的全局坐标系/尺度可能不一致，需 Sim(3) 对齐。
3. **Python 3.7 + PyTorch 1.12 兼容性**：若 MASt3R 依赖 PyTorch ≥ 2.0，两管线需分环境或升级本仓库（README 注明兼容 PyTorch 2.0 + CUDA 12）。
4. **物体级场景的稀疏视角退化**：3DGS 在极少视角（<10）下易出现 floaters，密度控制策略可能需要调整（`densify_grad_threshold`、`percent_dense`）。
5. **`opacity_lr` 文档与代码不一致**：README 写 0.05，代码实际为 0.025（`arguments/__init__.py:82`），以代码为准。
6. **fused-ssim / sparse_adam 可选路径**：加速分支 `3dgs_accel` 的 API 可能与 `dr_aa` 分支略有差异，未验证。
