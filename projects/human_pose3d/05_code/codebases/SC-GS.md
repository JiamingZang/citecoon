# SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

> 仓库: https://github.com/yihua7/SC-GS  
> 论文: arXiv 2312.14937  
> 许可: 非商业研究用途 (INRIA GraphDeco 基础代码 + SC-GS 扩展)

---

## 架构总览

SC-GS 在 3D Gaussian Splatting 基础上引入**稀疏控制点 (control nodes)** 驱动动态场景：一组可学习的控制点通过 KNN 高斯核权重将变形（平移/旋转/缩放）插值到每个 Gaussian 上；控制点自身的运动由一个时间条件 MLP (DeformNetwork) 预测。训练分两阶段——先用各向同性 Gaussian 预训练控制点位置（node pre-training），再联合训练完整 Gaussian 场景。编辑通过 ARAP 变形实现。

```
SC-GS/
├── train_gui.py              # 主训练+GUI+编辑入口
├── train.py                  # 评估报告 (training_report)
├── render.py                 # 离线渲染/评估
├── arguments/__init__.py     # 所有超参定义 (ModelParams, OptimizationParams)
├── scene/
│   ├── __init__.py           # Scene 加载 (Blender/Colmap/DTU/Neu3D/CMU)
│   ├── deform_model.py       # DeformModel 封装 (optimizer, save/load)
│   ├── gaussian_model.py     # GaussianModel (xyz, SH, scaling, rotation, opacity, feature)
│   └── dataset_readers.py    # 数据集读取, fid 时间归一化
├── utils/
│   ├── time_utils.py         # 核心: DeformNetwork, ControlNodeWarp, FPS, 嵌入器
│   ├── deform_utils.py       # ARAP 变形、连通性、Laplacian
│   ├── rigid_utils.py        # SE3/so3 指数映射
│   └── ...
├── gaussian_renderer/__init__.py  # 光栅化渲染 (depth+alpha)
├── lap_deform.py             # 编辑用 Laplacian/ARAP 变形
└── submodules/
    ├── diff-gaussian-rasterization  # 修改版光栅化器 (+depth, alpha)
    └── simple-knn
```

---

## 关键事实

### 1. 稀疏控制点数量与初始化

**默认数量**: `node_num = 1024`（命令行可覆盖，README 示例用 512）

```python
# arguments/__init__.py:66
self.node_num = 1024
```

**初始化方式**: 最远点采样 (Farthest Point Sampling) 从初始点云中选取

```python
# utils/time_utils.py:903-905
pcl_to_samp = init_pcl if hyper_pcl is None else hyper_pcl
init_nodes_idx = farthest_point_sample(pcl_to_samp.detach()[None], self.node_num)[0]
self.nodes.data = nn.Parameter(torch.cat([init_pcl[init_nodes_idx].float(), 1e-2 * torch.ones([self.node_num, self.hyper_dim]).float().cuda()], dim=-1))
```

**节点参数化**: 每个节点有 3+hyper_dim 维坐标 (xyz + 超坐标)，加上可学习半径和权重

```python
# utils/time_utils.py:806-810
self.nodes = nn.Parameter(torch.randn(node_num, 3+self.hyper_dim))
if not self.skinning:
    self._node_radius = nn.Parameter(torch.randn(node_num))
    if self.with_node_weight:
        self._node_weight = nn.Parameter(torch.zeros_like(self.nodes[:, :1]), requires_grad=with_node_weight)
```

**半径初始化**: 场景范围的 10%

```python
# utils/time_utils.py:915
self._node_radius = nn.Parameter(torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float().to(scene_range.device))
```

**Blender 数据初始化**: 用 Gaussian 点云位置初始化节点

```python
# train_gui.py:168-169
print('Initialize nodes with Random point cloud.')
self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=args.skinning)
```

### 2. LBS 蒙皮权重的计算与实现

**默认模式 (KNN 高斯核)**: K=3 近邻，权重 = exp(-dist / (2*radius^2)) * node_weight，归一化

```python
# utils/time_utils.py:950-959
nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K
if gs_kernel:
    nn_radius = self.node_radius[nn_idxs]  # N, K
    nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))  # N, K
    if self.with_node_weight:
        nn_node_weight = self.node_weight[nn_idxs]
        nn_weight = nn_weight * nn_node_weight[..., 0]
    nn_weight = nn_weight + 1e-7
    nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
```

**Skinnning 模式**: 每个 Gaussian 有 node_num 维特征向量，softmax 得到权重

```python
# utils/time_utils.py:935-938
if self.skinning:
    nn_weight = torch.softmax(feature, dim=-1)
    nn_idx = torch.arange(0, self.node_num, dtype=torch.long).cuda()
    return nn_weight, None, nn_idx
```

**K 的默认值**: K=3

```python
# arguments/__init__.py:53
self.K = 3
```

**权重如何驱动变形** (平移的加权插值):

```python
# utils/time_utils.py:1156
translate = (node_trans[nn_idx] * nn_weight[..., None]).sum(dim=1)
```

### 3. 变形 MLP 结构 (DeformNetwork)

**网络结构**: D=8 层, W=256 宽度, skip connection 在第 4 层

```python
# utils/time_utils.py:311-312
class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, t_multires=6, multires=10, ...):
```

**时间编码**: Blender/D-NeRF 用 t_multires=6, 其他用 10; 位置编码 multires=10

```python
# utils/time_utils.py:319
self.t_multires = 6 if is_blender else 10
```

**Blender 专用 timenet**: 先将时间编码压缩到 30 维再拼接

```python
# utils/time_utils.py:341-346
if is_blender:
    # Better for D-NeRF Dataset
    self.time_out = 30
    self.timenet = nn.Sequential(
        nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
        nn.Linear(256, self.time_out))
```

**输出头**: 分别预测 d_xyz(3), d_scaling(3), d_rotation(4)

```python
# utils/time_utils.py:363-365
self.gaussian_warp = nn.Linear(W, 3)
self.gaussian_scaling = nn.Linear(W, 3)
self.gaussian_rotation = nn.Linear(W, 4)
```

**初始化**: 输出头用极小 std 初始化以确保初始变形接近零

```python
# utils/time_utils.py:377-382
nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
nn.init.normal_(self.gaussian_rotation.weight, mean=0, std=1e-5)
```

**前向传播** (Blender 路径):

```python
# utils/time_utils.py:410-424
def forward(self, x, t, **kwargs):
    t_emb = self.embed_time_fn(t)
    if self.is_blender:
        t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
    x_emb = self.embed_fn(x)
    h = torch.cat([x_emb, t_emb], dim=-1)
    for i, l in enumerate(self.linear):
        h = self.linear[i](h)
        h = F.relu(h)
        if i in self.skips:
            h = torch.cat([x_emb, t_emb, h], -1)
    d_xyz = self.gaussian_warp(h)
    scaling = self.gaussian_scaling(h)
    rotation = self.gaussian_rotation(h)
```

### 4. D-NeRF 数据集训练时间与 PSNR

**时间归一化**: fid = frame_index / (num_frames - 1)，范围 [0, 1]

```python
# scene/dataset_readers.py:171
fid = int(image_name) / (num_frames - 1)
```

**Blender 数据也支持从 JSON 读取 time 字段**:

```python
# scene/dataset_readers.py:298-301
if 'time' in frame:
    frame_time = frame['time']
else:
    frame_time = idx / len(frames)
```

**总训练迭代**: 80,000 步

```python
# arguments/__init__.py:100
self.iterations = 80_000
```

**节点预训练阶段**: 前 10,000 步训练控制点 (isotropic Gaussians)

```python
# arguments/__init__.py:135
self.iterations_node_rendering = 10000
```

**Warm-up**: 前 3,000 步变形被 detach（不反传梯度到变形网络）

```python
# arguments/__init__.py:101
self.warm_up = 3_000
```

**评估频率**: 每 1000 步在测试集上计算 PSNR/SSIM/LPIPS

```python
# train_gui.py 主函数中 (未在截取范围但由 training_report 的 testing_iterations 控制)
# train.py:66-68
if iteration in testing_iterations:
    ...
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, ...)
```

**PSNR 报告**: 代码中无硬编码 PSNR 数值，论文报告 D-NeRF 平均 PSNR ~34+ (需查论文)

### 5. 时间外推能力

**时间输入无硬边界**: MLP 接受任意浮点 t 值，无 clamp

```python
# utils/time_utils.py:929-932
def expand_time(self, t):
    N = self.nodes.shape[0]
    t = t.unsqueeze(0).expand(N, -1)
    return t
```

**GUI 中时间可超 [0,1]**: 动画播放用 `torch.remainder(..., 1.)` 循环，但渲染可传任意 fid

```python
# train_gui.py:934
fid = torch.tensor(self.animation_time).cuda().float() if self.is_animation else torch.remainder(torch.tensor((time.time()-self.t0) * self.fps_of_fid).float().cuda() / len(self.scene.getTrainCameras()) * self.video_speed, 1.)
```

**位置编码的外推限制**: 使用标准正弦位置编码，超出 [0,1] 范围时频率组合可能产生未见过的模式

```python
# utils/time_utils.py:243
freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
```

**时间噪声增强** (非 Blender): 训练时加时间抖动

```python
# train_gui.py:1096
ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
```

---

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|-----|------|---------------|
| node_num | 1024 (默认), README 用 512 | `arguments/__init__.py:66` | 已是命令行参数 `--node_num` |
| K (近邻数) | 3 | `arguments/__init__.py:53` | 已是命令行参数 `--K` |
| hyper_dim | 8 | `arguments/__init__.py:65` | 已是命令行参数 `--hyper_dim` |
| MLP 层数 D | 8 | `utils/time_utils.py:311` | 构造函数参数，未暴露到命令行 |
| MLP 宽度 W | 256 | `utils/time_utils.py:311` | 构造函数参数，未暴露到命令行 |
| t_multires | 6 (blender) / 10 | `utils/time_utils.py:319` | 硬编码条件判断 |
| multires (xyz) | 10 | `utils/time_utils.py:311` | 构造函数参数 |
| time_out (timenet) | 30 | `utils/time_utils.py:342` | 硬编码 |
| iterations | 80,000 | `arguments/__init__.py:100` | 已是命令行参数 `--iterations` |
| iterations_node_rendering | 10,000 | `arguments/__init__.py:135` | 已是命令行参数 |
| warm_up | 3,000 | `arguments/__init__.py:101` | 已是命令行参数 |
| node_radius 初始化 | 0.1 * scene_range | `utils/time_utils.py:915` | 硬编码比例 |
| ARAP loss schedule | [1e-4, 1e-4, 1e-5, 1e-5, 0] @ [0, 5k, 10k, 20k, 20001] | `utils/time_utils.py:791-792` | 硬编码 landmarks |
| densify_grad_threshold | 0.0002 | `arguments/__init__.py:118` | 已是命令行参数 |
| deform_lr_max_steps | 40,000 | `arguments/__init__.py:107` | 已是命令行参数 |
| skinning 权重初始化 radius | 0.1 * scene_range | `utils/time_utils.py:910` | 硬编码 |
| FPS 采样随机种子 | torch.randint (无固定种子) | `utils/time_utils.py:473` | 依赖全局 safe_state |

---

## 环境与复现

### 依赖

```
# requirements.txt 核心:
opencv-python==4.5.5.62
scipy==1.10.1
tqdm, imageio, plyfile, piq, dearpygui, lpips, pytorch_msssim, matplotlib, scikit-image

# 隐式依赖 (代码 import):
- torch (>=1.12.1+cu113, 推荐更新版本)
- pytorch3d (knn_points, ball_query)
- diff-gaussian-rasterization (子模块, 修改版含 depth/alpha)
- simple-knn (子模块)
- tinycudann (可选, hash encoding 模式)
- torch_batch_svd (可选加速)
```

### 安装

```bash
git clone https://github.com/yihua7/SC-GS --recursive
cd SC-GS
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
# 另外需安装 pytorch3d: pip install git+https://github.com/facebookresearch/pytorch3d.git
```

### 数据

D-NeRF 数据集: 需下载 Blender 格式数据 (transforms_train.json / transforms_test.json)，放在 `--source_path` 指定目录。

### 最小运行命令

```bash
# 训练 (D-NeRF jumpingjacks, 400x400 最佳 PSNR)
CUDA_VISIBLE_DEVICES=0 python train_gui.py \
    --source_path YOUR/PATH/TO/DATASET/jumpingjacks \
    --model_path outputs/jumpingjacks \
    --deform_type node --node_num 512 --hyper_dim 8 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask \
    --local_frame --resolution 2 --W 800 --H 800

# 评估
CUDA_VISIBLE_DEVICES=0 python render.py \
    --source_path YOUR/PATH/TO/DATASET/jumpingjacks \
    --model_path outputs/jumpingjacks \
    --deform_type node --node_num 512 --hyper_dim 8 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask \
    --local_frame --resolution 2 --W 800 --H 800
```

### 权重

无预训练权重下载；训练产出保存在 `model_path/point_cloud/iteration_XXX/` 和 `model_path/deform/iteration_XXX/deform.pth`。

---

## 改造接口点

### 针对关注点的最小侵入修改方式

| 关注点 | 修改位置 | 方式 |
|--------|----------|------|
| **控制点数量** | 命令行 `--node_num N` | 无需改代码 |
| **控制点初始化策略** | `utils/time_utils.py:886-927` (`ControlNodeWarp.init`) | 替换 `farthest_point_sample` 为自定义采样 (如 k-means、随机+密度加权) |
| **蒙皮权重计算** | `utils/time_utils.py:934-967` (`cal_nn_weight`) | 替换高斯核为其他核函数或学习式权重；skinning 模式已提供全局 softmax 替代 |
| **变形 MLP 结构** | `utils/time_utils.py:310-458` (`DeformNetwork`) | 修改 D/W/t_multires；或替换为 HashDeformNetwork (`--use_hash`) |
| **时间编码** | `utils/time_utils.py:319, 327-328` | 改 t_multires 或替换为 ProgressiveBandFrequency (`--progressive_brand_time`) |
| **时间外推** | 训练时对 fid 做范围扩展/外推采样 | 在 `train_gui.py:1079` 处修改 fid 采样逻辑；或在 `DeformNetwork.forward` 中对 t 做归一化/周期化 |
| **ARAP 正则** | `utils/time_utils.py:791-792` | 修改 lambda_arap_landmarks/steps 或设 `--no_arap_loss` |
| **节点增删** | `--node_enable_densify_prune` + `arguments/__init__.py:125-130` | 已内置但默认关闭；开启后在 node_densification_interval 步执行 |
| **增加 LBS 骨骼数** | skinning 模式: `--skinning` + fea_dim=node_num | 每个 Gaussian 的 feature 维度等于 node_num，增大 node_num 即增大骨骼数 |

---

## 风险与未知

1. **论文 PSNR 数值**: 代码中无硬编码基准数值，D-NeRF 各场景 PSNR 需查论文 Table 1 (arXiv 2312.14937)，本卡不编造。
2. **训练时间**: 代码无计时基准；实际训练时长取决于 GPU (论文未明确给出单场景时间)，80k iterations 在 A100 上估计 1-2 小时但未验证。
3. **时间外推实测效果**: 代码无显式外推测试逻辑，位置编码在 t∉[0,1] 时行为未验证；GUI 的 `video_speed` 仅改变播放速率但 fid 仍通过 `remainder` 限制在 [0,1)。
4. **pytorch3d 版本兼容性**: 代码依赖 `pytorch3d.ops.knn_points` 和 `ball_query`，不同版本 API 可能有差异。
5. **torch_batch_svd 可选依赖**: 若未安装则 fallback 到 `torch.svd`，后者在新版 PyTorch 中已 deprecated (推荐 `torch.linalg.svd`)。
6. **submodules 未包含在浅克隆中**: `--depth 1` 克隆不含子模块内容，需 `--recursive` 或手动 `git submodule update --init`。
7. **ControlNodeWarp 中 `skinning` 模式**: 文档较少，`fea_dim=node_num` 时每个 Gaussian 存储对所有节点的权重，内存开销 O(N_gs * N_nodes)。
8. **`d_rot_as_res` vs `d_rot_as_rotmat`**: 两种旋转表示路径 (残差四元数 vs SVD 估计旋转矩阵)，默认 `d_rot_as_res=True`，另一路径的稳定性未充分验证。
