# 3DMPPE_ROOTNET_RELEASE 工程侦察卡片

## 架构总览（模块划分，一段话+目录树摘要）

RootNet 是 ICCV 2019 论文 "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation" 的官方 PyTorch 实现，仅包含根节点（Pelvis）3D 定位部分。整体为 top-down 两阶段中的第一阶段：先用外部检测器得到人体 bbox，再由 RootNet 回归根关节的 (x, y, depth)。网络结构为 ResNet-50 backbone → 3 层 deconv 生成 64×64 heatmap 做 xy 软定位 → 对 backbone 最终特征做 global avg pool 后 1×1 conv 输出标量 gamma → depth = gamma × k_value（相机距离感知系数）。训练支持 3D+2D 数据集混合，2D 数据集通过 have_depth=0 屏蔽深度损失。

```
3DMPPE_ROOTNET_RELEASE/
├── main/
│   ├── config.py        # 全局配置（数据集、输入尺寸、训练超参）
│   ├── model.py         # RootNet + ResPoseNet（网络定义+损失）
│   ├── train.py         # 训练入口
│   └── test.py          # 测试入口
├── common/
│   ├── base.py          # Trainer/Tester 基类
│   ├── nets/resnet.py   # ResNet backbone
│   └── utils/
│       ├── pose_utils.py  # cam2pixel / pixel2cam / world2cam / process_bbox
│       └── vis.py
├── data/
│   ├── dataset.py       # DatasetLoader（k_value 计算、patch 裁剪、增强）
│   ├── multiple_datasets.py
│   ├── Human36M/        # H36M 数据加载+评估
│   ├── MuCo/            # MuCo-3DHP 数据加载
│   ├── MuPoTS/          # MuPoTS 评估
│   ├── MSCOCO/          # COCO 2D 数据
│   ├── MPII/            # MPII 2D 数据
│   └── PW3D/            # 3DPW 数据
└── demo/demo.py         # 单图推理 demo
```

## 关键事实（与关注点直接相关的代码事实）

### 事实 1：k_value（相机距离感知系数）的计算公式

**文件**: `data/dataset.py:73`（训练）和 `data/dataset.py:81`（测试）

```python
k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32)
```

其中：
- `area` = `bbox[2]*bbox[3]`，即人体 bbox 在原图中的像素面积（`data/Human36M/Human36M.py:122`：`area = bbox[2]*bbox[3]`）
- `f` = `[fx, fy]`，相机焦距（像素单位），来自数据集标注（`data/Human36M/Human36M.py:105`：`f = np.array(cam_param['f'], dtype=np.float32)`）
- `cfg.bbox_real` = `(2000, 2000)`（`main/config.py:38`），代表真实世界中人体 bbox 的物理尺寸（H36M 用毫米，PW3D 用米则为 `(2,2)`）

**物理含义**：k = sqrt(S_real × f_x × f_y / A_pixel)。由针孔模型 A_pixel ≈ f_x × f_y × S_real / Z² 可推导 k ≈ Z（相机到人体的距离），因此 depth = gamma × k 等价于网络学习一个无量纲比例因子 gamma 再乘以距离先验。

### 事实 2：深度输出头的网络结构

**文件**: `main/model.py:67-72`

```python
# z
img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2) # global average pooling
img_feat = torch.unsqueeze(img_feat,2); img_feat = torch.unsqueeze(img_feat,3);
gamma = self.depth_layer(img_feat)
gamma = gamma.view(-1,1)
depth = gamma * k_value.view(-1,1)
```

- `self.depth_layer` 定义于 `main/model.py:22-28`：
```python
self.depth_layer = nn.Conv2d(
    in_channels=self.inplanes,   # 2048
    out_channels=1, 
    kernel_size=1, stride=1, padding=0
)
```
- 结构极简：GAP(2048×8×8) → 2048×1×1 → Conv1x1 → 标量 gamma → 乘 k_value 得绝对深度

### 事实 3：xy 定位头（soft heatmap）

**文件**: `main/model.py:51-65`

```python
xy = self.deconv_layers(x)          # 2048×8×8 → 256×64×64
xy = self.xy_layer(xy)              # 256×64×64 → 1×64×64
xy = xy.view(-1,1,cfg.output_shape[0]*cfg.output_shape[1])
xy = F.softmax(xy,2)                # spatial softmax
xy = xy.view(-1,1,cfg.output_shape[0],cfg.output_shape[1])
hm_x = xy.sum(dim=(2))             # marginalize y → 1×64
hm_y = xy.sum(dim=(3))             # marginalize x → 1×64
coord_x = hm_x * torch.arange(cfg.output_shape[1]).float().cuda()
coord_y = hm_y * torch.arange(cfg.output_shape[0]).float().cuda()
coord_x = coord_x.sum(dim=2)       # expected value → 标量
coord_y = coord_y.sum(dim=2)
```

### 事实 4：损失函数

**文件**: `main/model.py:106-113`

```python
target_coord = target['coord']
target_vis = target['vis']
target_have_depth = target['have_depth']

## coordrinate loss
loss_coord = torch.abs(coord - target_coord) * target_vis
loss_coord = (loss_coord[:,0] + loss_coord[:,1] + loss_coord[:,2] * target_have_depth.view(-1))/3.
return loss_coord
```

- L1 损失，xyz 三通道等权平均
- 2D 数据集通过 `target_have_depth=0` 屏蔽 z 轴损失
- 无序数深度监督、无面积修正项——仅有单一 L1

### 事实 5：相机投影反算根节点绝对坐标（评估阶段）

**文件**: `data/Human36M/Human36M.py:155-160`

```python
# warp output to original image space
pred_root = preds[n]
pred_root[0] = pred_root[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
pred_root[1] = pred_root[1] / cfg.output_shape[0] * bbox[3] + bbox[1]

# back-project to camera coordinate space
pred_root = pixel2cam(pred_root[None,:], f, c)[0]
```

`pixel2cam` 实现于 `common/utils/pose_utils.py:13-18`：
```python
def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord
```

即标准针孔模型反投影：X_cam = (u - cx) / fx × Z, Y_cam = (v - cy) / fy × Z。

### 事实 6：demo 中 k_value 的推理时计算

**文件**: `demo/demo.py:71-82`

```python
focal = [1500, 1500] # x-axis, y-axis
princpt = [original_img_width/2, original_img_height/2]
...
k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
```

推理时需手动设定焦距（默认 1500px），无真实相机参数时以此为假设。

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|-----|------|---------------|
| `bbox_real` | `(2000, 2000)` mm (H36M) / `(2, 2)` m (PW3D) | `main/config.py:38` | 已为 Config 类属性，按数据集切换即可；若需 per-sample 自适应则需改 dataset.py:73 |
| `resnet_type` | `50` | `main/config.py:31` | 已可配置（50/101/152） |
| `input_shape` | `(256, 256)` | `main/config.py:34` | 已可配置 |
| `output_shape` | `(64, 64)` = input//4 | `main/config.py:35` | 由 input_shape 派生 |
| demo 焦距 | `[1500, 1500]` | `demo/demo.py:71` | 硬编码，需手动改或加命令行参数 |
| `pixel_mean/std` | ImageNet 标准值 | `main/config.py:36-37` | 已为配置 |
| `lr` | `1e-3` | `main/config.py:43` | 已为配置 |
| `batch_size` | `32` | `main/config.py:45` | 已为配置 |
| `end_epoch` | `20` | `main/config.py:42` | 已为配置 |
| `lr_dec_epoch` | `[17]` | `main/config.py:41` | 已为配置 |
| MuCo 深度范围 | `min_depth=1500, max_depth=7500` | `data/MuCo/MuCo.py:16-17` | 硬编码在类中 |
| bbox 扩展系数 | `1.25` | `common/utils/pose_utils.py:63-64` | 硬编码 |
| deconv 层数 | `3` | `main/model.py:14` | 硬编码于 `_make_deconv_layer(3)` |
| depth_layer in_channels | `2048` | `main/model.py:23` | 由 `self.inplanes` 决定，与 backbone 耦合 |

## 环境与复现

**依赖**（README 声明）：
- PyTorch（ tested with CUDA 9.0 / cuDNN 7.1）
- CUDA + cuDNN
- Anaconda (Python 3.6.5)
- COCO API (`pycocotools`)
- OpenCV (`cv2`)
- torchvision（ResNet 预训练权重通过 `torch.utils.model_zoo.load_url` 自动下载）

**无 requirements.txt / environment.yml**，需手动安装。

**权重下载**：
- 预训练 RootNet：Google Drive 链接（README "Quick demo" 和 "Results" 节）
- 数据集标注：各数据集均有 Google Drive 链接（README "Data" 节）

**最小运行命令**：
```bash
# 训练（需先准备 data/Human36M + data/MPII 目录结构）
cd main && python train.py --gpu 0

# 测试
cd main && python test.py --gpu 0 --test_epoch 19

# Demo（需下载 snapshot_18.pth.tar 到 demo/ 并准备 input.jpg）
cd demo && python demo.py --gpu 0 --test_epoch 18
```

**注意**：`bbox_real` 必须与数据集单位匹配（H36M=毫米→2000, 3DPW=米→2），否则深度输出量纲错误。

## 改造接口点

### 改造点 1：相机距离感知的 k 值——面积项/焦距项的注入位置

- **当前实现**：k_value 在 `data/dataset.py:73` 的 `__getitem__` 中计算，作为标量随 batch 传入网络。
- **最小侵入改法**：
  - 若要用 SMPL 投影面积替代 bbox 面积：在数据集 `load_data()` 中增加 `smpl_area` 字段（如 `data/Human36M/Human36M.py:124` 的 dict 中追加），然后在 `data/dataset.py:73` 将 `area` 替换为 `data['smpl_area']`。
  - 若要使 k_value 可学习或加入额外特征（如 bbox aspect ratio）：修改 `main/model.py:50` 的 `forward(self, x, k_value)` 签名，将 k_value 从标量扩展为向量，或在 `depth_layer` 前 concat 额外信息。

### 改造点 2：深度输出头——挂载 SMPL 面积修正

- **当前位置**：`main/model.py:67-72`，GAP → Conv1x1(2048→1) → gamma × k。
- **最小侵入改法**：
  - 在 `gamma = self.depth_layer(img_feat)` 之后、`depth = gamma * k_value` 之前，插入面积修正模块：
    ```python
    # 在 model.py:71 后插入
    area_correction = self.area_mlp(area_feat)  # 新增模块
    depth = gamma * k_value * area_correction
    ```
  - 需在 `__init__` 中新增 `self.area_mlp`，在 `forward` 签名中新增 `area_feat` 输入。
  - 训练时从 dataset 传入 SMPL 投影面积或 bbox 面积比。

### 改造点 3：序数深度监督的挂载位置

- **当前损失**：`main/model.py:111-112`，纯 L1。
- **最小侵入改法**：
  - 在 `ResPoseNet.forward` 的 loss 分支（`main/model.py:106-113`）中追加序数损失项：
    ```python
    # 在 loss_coord 计算后追加
    loss_ordinal = self.ordinal_loss(depth_pred, depth_target, bin_centers)
    return loss_coord + lambda_ord * loss_ordinal
    ```
  - 需在 `RootNet.__init__` 中增加 ordinal 分类头（如将 depth_layer 的 out_channels 从 1 改为 D 个 bin），或保持回归头不变、仅对连续深度值做 soft ordinal regression。
  - `train.py:49` 的 target dict 需扩充：`target = {'coord': root_img, 'vis': root_vis, 'have_depth': joints_have_depth, 'depth_bin': depth_bin}`。

### 改造点 4：评估时反投影流程

- **当前位置**：`data/Human36M/Human36M.py:155-160`（evaluate 方法内）。
- 若修改了深度输出语义（如输出相对深度而非绝对深度），需同步修改此处反投影逻辑。

## 风险与未知

1. **PyTorch 版本兼容性**：代码使用 `torch.nn.parallel.data_parallel.DataParallel`（旧式导入路径 `main/model.py:4` 的 `from nets.resnet import ResNetBackbone` 依赖 sys.path hack），在 PyTorch ≥1.10 下可能有 import 路径问题，未实际验证。
2. **Python 版本**：README 声明 Python 3.6.5，`print` 语句和 f-string 使用情况未全面排查；`common/base.py:20` 使用 `exec()` 动态导入，高版本 Python 下行为一致但安全性差。
3. **ordinal depth 的具体 bin 数和 lambda 权重**：原论文补充材料中有序数深度描述，但代码中完全未实现，需自行设计。
4. **SMPL 面积修正**：仓库中无任何 SMPL 相关代码或依赖，需从外部引入 SMPL 模型（如 smplx 库）并在数据预处理阶段计算投影面积。
5. **2D 数据集的深度监督**：当前 2D 数据集（MPII/MSCOCO）通过 `have_depth=0` 完全屏蔽深度损失，若序数监督要利用 2D 数据的相对深度排序，需额外设计 pseudo depth 或排序标注。
6. **多 GPU 训练**：`base.py` 使用旧式 `DataParallel`，未使用 `DistributedDataParallel`，大 batch 训练效率可能受限。
7. **k_value 在 batch 中的维度传递**：`train.py:43` 解包为 `k_value`，经 DataLoader 后 shape 为 `(B, 1)`，model.py:72 用 `k_value.view(-1,1)` 对齐——若改为多维特征需同步修改 DataLoader collate 和 model forward。
