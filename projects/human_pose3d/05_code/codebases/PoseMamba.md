# PoseMamba 工程侦察卡片

> 研究关注点：将 PoseMamba 的 Mamba/SSM 时空块替换为 Gated DeltaNet 线性注意力块（fla chunk_gated_delta_rule）
> 仓库：https://github.com/nankingjing/PoseMamba （AAAI 2025，浅克隆 v1.0.0）

## 架构总览（模块划分，一段话+目录树摘要）

PoseMamba 是一个 2D→3D 人体姿态提升（lifting）模型。主干 `PoseMamba`（`lib/model/PoseMamba.py:37`）接收 `(B, T, J, 2)` 的 2D 骨架序列，先经空间嵌入（Linear 将每关节 2D 坐标映射到 embed_dim），再交替堆叠 **空间 SSM 块**（STEblocks）和 **时间 SSM 块**（TTEblocks），最后用 LayerNorm+Linear 输出 `(B, T, J, 3)`。每个 SSM 块 `BiSTSSMBlock`（`lib/model/mambablocks.py:619`）内部包含一个 `BiSTSSM` 算子（`mambablocks.py:582`）和一个 MLP，采用 pre-norm 残差结构。`BiSTSSM` 的核心是 4 方向交叉扫描（CrossScan）+ CUDA selective_scan + CrossMerge 融合，将 2D 的 (H=T, W=J) 特征图按行正/反、列正/反共 4 条序列做 SSM 扫描后求和合并。

```
PoseMamba/
├── train.py                      # 训练/评测入口
├── configs/pose3d/               # YAML 配置（S/B/L 三种规模）
│   ├── PoseMamba_train_h36m_S.yaml
│   ├── PoseMamba_train_h36m_B.yaml
│   └── PoseMamba_train_h36m_L.yaml
├── lib/
│   ├── model/
│   │   ├── PoseMamba.py          # 主干网络（forward 入口）
│   │   ├── mambablocks.py        # BiSTSSM / BiSTSSMBlock（SSM 核心）
│   │   ├── csms6s.py             # CrossScan/CrossMerge（4方向扫描）+ SelectiveScan CUDA 封装
│   │   ├── csm_triton.py         # Triton 版 CrossScan（备选）
│   │   ├── loss.py               # MPJPE / velocity / limb / angle 损失
│   │   └── DSTformer.py          # Transformer 基线（对比用）
│   ├── data/
│   │   ├── dataset_motion_3d.py  # H36M 3D 数据集
│   │   ├── datareader_h36m.py    # H36M 数据读取/归一化/切片
│   │   └── augmentation.py       # 数据增强
│   └── utils/
│       ├── learning.py           # load_backbone() 工厂函数
│       └── tools.py              # 配置加载等工具
├── kernels/selective_scan/       # CUDA selective scan 内核（需编译）
└── tools/convert_h36m.py         # H36M 数据切片脚本
```

## 关键事实（与关注点直接相关的代码事实）

### (1) 输入张量形状与约定

**进入主干的确切位置**：`lib/model/PoseMamba.py:133-134`
```python
def forward(self, x):
    b, f, n, c = x.shape   # (B, T, J, 2)
```
- 输入形状 `(B, T, J, 2)`：batch / 帧数T / 关节数J=17 / 坐标维=2（x,y）。
- **confidence 维度在训练循环中被裁掉**：`train.py:176`
  ```python
  if args.no_conf:
      batch_input = batch_input[:, :, :, :2]  # (N, T, 17, 2)
  ```
  数据集原始输出为 `(T, 17, 3)`（含 confidence），配置 `no_conf: True` 时只取 xy。
- **归一化**：2D 坐标在 `datareader_h36m.py:36` 按相机分辨率映射到 `[-1, 1]`：
  ```python
  trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
  ```
- **root-centered**：3D GT 在训练时做 root-relative：`train.py:181`
  ```python
  if args.rootrel:
      batch_gt = batch_gt - batch_gt[:,:,0:1,:]
  ```
  评测时预测也置零 root：`train.py:79` `predicted_3d_pos[:,:,0,:] = 0`。
- **空间嵌入入口**：`PoseMamba.py:96-101`
  ```python
  def STE_forward(self, x):
      b, f, n, c = x.shape
      x = rearrange(x, 'b f n c -> (b f) n c')
      x = self.Spatial_patch_to_embedding(x)  # Linear(2, embed_dim_ratio)
      x += self.Spatial_pos_embed             # (1, J, embed_dim_ratio)
      x = rearrange(x, '(b f) n c -> b f n c', f=f)
  ```

### (2) 时空建模块：BiSTSSMBlock / BiSTSSM

**模块文件**：`lib/model/mambablocks.py`

**BiSTSSMBlock**（`mambablocks.py:619-686`）：pre-norm 残差 + SSM + MLP
```python
# mambablocks.py:676-680
def _forward(self, input: torch.Tensor):
    x = input
    x = x + self.drop_path(self.op(self.norm(x)))     # SSM 分支
    x = x + self.drop_path(self.mlp(self.norm2(x)))   # MLP 分支
    return x
```
- 输入输出形状相同：`(B, T, J, C)`（空间块 C=embed_dim_ratio；时间块 C=embed_dim）。

**BiSTSSM 的 forwardv2**（`mambablocks.py:555-578`）：
```python
def forwardv2(self, x: torch.Tensor, **kwargs):
    x = self.in_proj(x)                    # Linear(d_model, d_inner*2)
    x, z = x.chunk(2, dim=-1)             # gate 分支
    z = self.act(z)                        # SiLU
    x = x.permute(0, 3, 1, 2).contiguous() # (B, C, H=T, W=J)
    x = self.conv2d(x)                     # depthwise conv 3x3
    x = self.act(x)
    y = self.forward_core(x)               # 4方向 SSM 扫描
    y = y * z                              # gate
    out = self.dropout(self.out_proj(y))   # Linear(d_inner, d_model)
    return out
```
- 输入 `(B, T, J, C)` → 输出 `(B, T, J, C)`，内部将 (T, J) 视为 2D 的 (H, W)。

**forward_core（v2_plus_poselimbs）**（`mambablocks.py:400-553`）：
- 默认 `forward_type='v2_plus_poselimbs'`（`PoseMamba.py:72`），绑定：
  ```python
  # mambablocks.py:330
  v2_plus_poselimbs=partial(self.forward_corev2, ..., CrossScan=CrossScan_plus_poselimbs,
                            SelectiveScan=SelectiveScanCore, CrossMerge=CrossMerge_plus_poselimbs)
  ```
- 核心流程（`mambablocks.py:512-539`）：
  ```python
  xs = CrossScan.apply(x)           # (B,4,C,H*W) 4方向展开
  x_dbl = einsum(xs, x_proj_weight) # 投影出 dt, B, C
  ys = selective_scan(xs, dts, As, Bs, Cs, Ds, ...)  # CUDA SSM
  y = CrossMerge.apply(ys)          # 4方向合并
  ```

**双向扫描的实现与融合**（`lib/model/csms6s.py`）：

`CrossScan_plus_poselimbs`（`csms6s.py:149-169`）：
```python
def forward(ctx, x: torch.Tensor):
    B, C, H, W = x.shape
    assert W == 17, 'the number of joints is not 17'
    xs = x.new_empty((B, 4, C, H * W))
    indices = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]
    xs[:, 0] = (x + x[..., indices]).flatten(2, 3)   # 时间正向 + 骨骼父节点位置编码
    xs[:, 1] = x.transpose(2, 3).flatten(2, 3)        # 空间（关节维）正向
    xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])   # 方向2、3 = 方向0、1 的翻转（反向）
    return xs
```
- 4 条扫描序列：时间正、空间正、时间反、空间反。
- 方向 0 额外加了**骨骼父节点位置编码**（`indices` 为 H36M 17 关节的父节点索引）。

`CrossMerge_plus_poselimbs`（`csms6s.py:170-192`）融合方式——**逐元素求和（sum）**：
```python
def forward(ctx, ys: torch.Tensor):
    ys = ys.view(B, K, D, -1)
    ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)  # 正+反
    y = ys[:, 0] + ys[:, 1].view(B,-1,W,H).transpose(2,3).contiguous().view(B,D,-1)  # 时间+空间
    return y
```

**空间/时间块如何交替**（`PoseMamba.py:121-131`）：
```python
def ST_foward(self, x):
    b, f, n, cw = x.shape
    for i in range(1, self.block_depth):
        x = self.STEblocks[i](x)   # 空间块：(B,T,J,C) 内部视为 H=T, W=J
        x = self.Spatial_norm(x)
        x = self.TTEblocks[i](x)   # 时间块：先 rearrange 为 (B*J,T,C)
        x = self.Temporal_norm(x)
    return x
```
- 时间块在进入前做 rearrange：`PoseMamba.py:111` `x = rearrange(x, 'b f n cw -> (b n) f cw')`，使 H=T, W=1（单关节时间序列）。
- 第 0 层单独走 `STE_forward` + `TTE_foward`，第 1~depth-1 层交替走 `ST_foward`。

### (3) SSM 块替换为 fla chunk_gated_delta_rule 的对齐分析

**需要修改的文件**：`lib/model/mambablocks.py`

**替换目标**：`BiSTSSM` 类（`mambablocks.py:582-617`）及其混入的 `BiSTSSM_v2`（`mambablocks.py:242-578`）。

**接口对齐要点**：
- `BiSTSSMBlock._forward`（`mambablocks.py:676`）调用 `self.op(self.norm(x))`，其中 `self.op` 是 `BiSTSSM` 实例。
- `BiSTSSM.forwardv2` 的输入输出均为 `(B, T, J, C)`（channel-last），内部转为 `(B, C, H, W)` 做扫描。
- `chunk_gated_delta_rule(q, k, v, g, beta)` 期望输入 `(B, L, K, D)`（batch, seq_len, num_heads, head_dim），输出同形状。
- **对齐方案**：在 `forwardv2` 中，将 `forward_core`（CrossScan+SelectiveScan+CrossMerge）替换为：
  1. 将 `(B, C, H, W)` reshape 为序列 `(B, L=H*W, C)`；
  2. 用线性层投影出 q, k, v, g, beta（替代原 x_proj + dt_proj）；
  3. reshape 为 `(B, L, K, D)` 调用 `chunk_gated_delta_rule`；
  4. 输出 reshape 回 `(B, H, W, C)` 再过 out_proj。
- **需删除/替换的参数**：`x_proj_weight`, `dt_projs_weight`, `dt_projs_bias`, `A_logs`, `Ds`, `conv2d`（可选保留）；**新增**：q/k/v/g/beta 投影层。
- **具体行号**：
  - 删除 `__initv2__` 中 SSM 参数初始化：`mambablocks.py:362-398`（x_proj, dt_projs, A_logs, Ds）
  - 替换 `forward_corev2` 整个方法：`mambablocks.py:400-553`
  - 修改 `forwardv2` 中调用 `forward_core` 的部分：`mambablocks.py:572`
  - `FORWARD_TYPES` 字典（`mambablocks.py:312-340`）不再需要
- **CrossScan/CrossMerge 可完全跳过**：Gated DeltaNet 本身是序列模型，不需要 4 方向扫描。若需保留双向，可正反向各跑一次 chunk_gated_delta_rule 再求和。

### (4) 损失函数

**文件**：`lib/model/loss.py`

**训练中的损失计算**：`train.py:199-229`
```python
loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)          # L2 欧氏距离
loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)           # 归一化 MPJPE（scale）
loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)  # 时间差分速度损失
loss_lv = loss_limb_var(predicted_3d_pos)                     # 骨长方差
loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)            # 骨长 GT L1
loss_a = loss_angle(predicted_3d_pos, batch_gt)               # 关节角度 L1
loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)     # 角速度 L1
loss_diff = ...                                                # 时间一致性（加权帧差平方）
```
- **对序列输出的形状要求**：所有损失函数期望 `(N, T, 17, 3)`（见 `loss.py:71` 注释 `#torch.Size([24, 243, 17, 3])`）。
- 默认配置（L）中启用的损失权重：`lambda_3d=1.0, lambda_scale=0.5, lambda_3d_velocity=20.0, lambda_diff=0.5`，其余为 0。
- `loss_mpjpe`（`loss.py:56-63`）：
  ```python
  def loss_mpjpe(predicted, target):
      assert predicted.shape == target.shape
      return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
  ```
- 2D 重投影损失 `loss_2d_weighted`（`loss.py:74-79`）仅在 `has_3d=False`（纯 2D 数据训练）时使用，H36M 3D 训练不走此路径。

### (5) H36M 数据加载与协议

**配置文件**：`configs/pose3d/PoseMamba_train_h36m_L.yaml`
```yaml
maxlen: 243          # 输入帧数 T
clip_len: 243        # 切片长度
data_stride: 81      # 训练切片步长
sample_stride: 1     # 帧采样步长
num_joints: 17
data_root: data/motion3d/MB3D_f243s81/
subset_list: [H36M-SH]
dt_file: h36m_sh_conf_cam_source_final.pkl
gt_2d: False         # False = 用 CPN 检测的 2D；True = 用 GT 2D
no_conf: True        # 丢弃 confidence 通道
rootrel: True        # root-relative
```

**数据切片**：`lib/data/datareader_h36m.py:100-107`
```python
def get_split_id(self):
    self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train)  # stride=81
    self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)      # stride=243（不重叠）
```
- 训练：T=243 帧，步长 81（约 1/3 重叠）。
- 测试：T=243 帧，步长 243（不重叠）。

**数据集类**：`lib/data/dataset_motion_3d.py:34-68`
- 从预切片的 `.pkl` 文件读取，每个文件含 `data_input`（2D）和 `data_label`（3D）。
- `gt_2d=False` 时使用 CPN 检测的 2D 关键点（含 confidence）；`gt_2d=True` 时用 GT 3D 的 xy 分量替代。
- 训练增强：随机水平翻转（`flip: True`）。

**评测协议**：`train.py:59-165`
- P1 (MPJPE)：root-relative 后计算欧氏距离，单位 mm。
- P2 (P-MPJPE)：Procrustes 对齐后计算。
- 排除 3 个测试序列（`train.py:121-123`）：`s_09_act_05_subact_02`, `s_09_act_10_subact_02`, `s_09_act_13_subact_01`。
- 评测时使用翻转增强（test-time augmentation）：`train.py:71-75`。

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置的方式 |
|------|-----|------|-----------------|
| 关节数 J=17 | `assert W == 17` | `csms6s.py:153` | 删除 assert，从 config 传入 num_joints |
| 骨骼父节点索引 | `indices = [0,0,1,2,3,0,4,5,6,8,11,12,13,8,14,15,16]` | `csms6s.py:156` | 若替换为 Gated DeltaNet 则不再需要 |
| SSM 状态维 d_state=16 | 默认参数 | `mambablocks.py:627` | 替换后不再需要 |
| ssm_ratio=2.0 | d_inner = 2 * d_model | `mambablocks.py:628` | 替换后改为 head_dim * num_heads |
| k_group=4（扫描方向数） | 硬编码 | `mambablocks.py:342` | 替换后不再需要 |
| embed_dim_ratio | S:32 / B:64 / L:128 | YAML `dim_feat` | 已可配置 |
| depth | S:4 / B:8 / L:20 | YAML `depth` | 已可配置 |
| maxlen (T) | 243 | YAML `maxlen` | 已可配置 |
| 相机分辨率 1000x1002 / 1000x1000 | 硬编码 | `datareader_h36m.py:31-33` | 从数据集元信息读取 |
| 加权 MPJPE 权重 | `w_mpjpe = [1,1,2.5,...]` | `train.py:207` | 移入 config |
| 排除序列 block_list | 3 个序列名 | `train.py:121-123` | 移入 config |
| depthwise conv kernel=3 | `d_conv=3` | `mambablocks.py:631` | 替换后可选保留或移除 |

## 环境与复现

**依赖**（`requirements.txt` + README）：
- Python 3.8.5, PyTorch 1.13.1+cu117, CUDA 11.7
- 关键包：`einops`, `timm`, `fvcore`, `triton`, `tensorboardX`, `thop`
- **必须编译 CUDA 扩展**：`cd kernels/selective_scan && pip install -e .`（提供 `selective_scan_cuda_core`）
- 若替换为 fla 的 Gated DeltaNet，可移除 CUDA 扩展依赖，改为 `pip install fla`（flash-linear-attention）

**权重下载**：
- HuggingFace：`nankingjings/PoseMamba-weights`（S/B/L 的 `.bin` 文件）
- Google Drive 备用链接见 README

**数据准备**：
1. 下载 MotionBERT 预处理的 H36M 数据 → `data/motion3d/`
2. 运行 `cd tools && python convert_h36m.py` 切片 → `data/motion3d/MB3D_f243s81/`

**最小运行命令**：
```bash
# 训练 PoseMamba-L
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/pose3d/PoseMamba_train_h36m_L.yaml \
  --checkpoint checkpoint/pose3d/PoseMamba_train_h36m_L

# 评测
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config checkpoint/pose3d/PoseMamba_L/config.yaml \
  --evaluate checkpoint/pose3d/PoseMamba_L/best_epoch.bin \
  --checkpoint eval/checkpoint
```

## 改造接口点

**目标**：将 BiSTSSM（Mamba SSM）替换为 fla 的 `chunk_gated_delta_rule`，最小侵入。

### 修改位置 1：`lib/model/mambablocks.py` — 替换 BiSTSSM 类

**最小方案**：保留 `BiSTSSMBlock` 外壳（`mambablocks.py:619-686`），只替换内部的 `self.op`。

1. **新建 `GatedDeltaNetOp` 类**（替代 `BiSTSSM`），接口：
   - 输入 `(B, T, J, C)` → 输出 `(B, T, J, C)`（与 `BiSTSSM.forwardv2` 一致）
   - 内部：
     ```python
     # 伪代码
     x = self.in_proj(x)           # (B, T, J, C) -> (B, T, J, d_inner)
     L = T * J                      # 或分别对 T 和 J 做序列
     x = x.reshape(B, L, D)
     q, k, v, g, beta = self.qkvgb_proj(x).split(...)  # 各 (B, L, K, D_head)
     out = chunk_gated_delta_rule(q, k, v, g, beta)     # (B, L, K, D_head)
     out = out.reshape(B, T, J, C)
     out = self.out_proj(out)
     ```
2. **修改 `BiSTSSMBlock.__init__`**（`mambablocks.py:652-666`）：
   ```python
   # 原：self.op = BiSTSSM(d_model=hidden_dim, ...)
   # 改：self.op = GatedDeltaNetOp(d_model=hidden_dim, ...)
   ```
3. **不再需要的代码**：
   - `csms6s.py` 中所有 CrossScan/CrossMerge 类
   - `mambablocks.py` 中 `BiSTSSM_v2.__initv2__` 的 SSM 参数（A_logs, Ds, dt_projs, x_proj）
   - `forward_corev2` 整个方法
   - `kernels/selective_scan/` CUDA 扩展

### 修改位置 2：序列建模策略选择

原模型对 (T, J) 做 2D 交叉扫描。替换为线性注意力后有两种策略：
- **方案 A（展平）**：将 T*J 展平为单一序列，一次 chunk_gated_delta_rule 处理。简单但丢失空间/时间分离结构。
- **方案 B（分离，推荐）**：保持原 STE/TTE 分离结构——空间块对 J 维做序列（L=J=17），时间块对 T 维做序列（L=T=243）。这与原模型的空间/时间交替架构一致，改动最小。
  - 空间块：`(B*T, J, C)` → chunk_gated_delta_rule(q,k,v,g,beta) with L=J
  - 时间块：`(B*J, T, C)` → chunk_gated_delta_rule(q,k,v,g,beta) with L=T
  - 若需双向：正反向各跑一次，输出求和（模拟原 CrossMerge 的 sum 融合）。

### 修改位置 3：`PoseMamba.py` — 无需修改

`PoseMamba` 主干（`PoseMamba.py:37-141`）只通过 `BiSTSSMBlock` 接口调用，不直接依赖 SSM 内部实现。只要 `BiSTSSMBlock` 的输入输出形状不变，`PoseMamba.py` 无需任何修改。

### 修改位置 4：`lib/utils/learning.py:84` — 无需修改

`load_backbone` 实例化 `PoseMamba(num_frame=args.maxlen, embed_dim_ratio=args.dim_feat, ...)`，参数不变。

## 风险与未知

1. **fla 的 chunk_gated_delta_rule 接口细节未在本仓库验证**：需确认 fla 库版本的输入约定（`q,k,v,g,beta` 的形状是 `(B,L,K,D)` 还是 `(B,K,L,D)`），以及 `g` 和 `beta` 的值域约束（是否需要 sigmoid/softplus）。
2. **骨骼父节点位置编码的替代**：原 `CrossScan_plus_poselimbs`（`csms6s.py:156`）在扫描时加了父节点特征，这是 PoseMamba 的关键设计之一。替换后需要设计等价的空间先验注入方式（如额外的图卷积或可学习位置编码）。
3. **depthwise conv2d 的作用**：`BiSTSSM.forwardv2` 中的 `conv2d`（`mambablocks.py:569`）在 (T,J) 2D 特征图上做 3x3 深度卷积，替换后是否保留、如何适配 1D 序列需实验验证。
4. **CUDA 扩展移除后的兼容性**：`csms6s.py:323-342` 在模块加载时尝试 import CUDA 扩展，若完全移除需确保不影响其他 import 路径。
5. **预训练权重不兼容**：替换 SSM 后所有 SSM 相关参数（A_logs, Ds, dt_projs, x_proj）消失，无法加载官方预训练权重，需从头训练。
6. **T=243 长序列的线性注意力性能**：chunk_gated_delta_rule 的 chunk 大小对长序列（T=243）的精度/速度平衡需要调参。
7. **MPI-INF-3DHP 数据集配置未查证**：仓库 README 提及支持但未找到对应 config 文件，可能需参考 MotionAGFormer。
8. **`PoseMamba_bs_bt.py` / `PoseMamba_fs_ft.py` 等变体文件**：这些是消融实验用的不同扫描方向组合变体，替换后不再需要，但未逐一确认其是否被其他代码引用。
