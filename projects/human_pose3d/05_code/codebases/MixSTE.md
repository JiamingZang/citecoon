# MixSTE Repo Card

> 仓库: https://github.com/JinluZhang1126/MixSTE (CVPR 2022)
> 用途: 3D Human Pose Estimation in Video — Seq2seq Mixed Spatio-Temporal Encoder

## 架构总览

MixSTE 采用 seq2seq 结构，将视频帧序列的 2D 关节坐标映射为逐帧 3D 姿态。核心思想是**时空交替（alternating）**：每一层先做空间注意力（同一帧内各关节之间），再做时间注意力（同一关节跨帧之间），两者交替堆叠 depth 层。输入形状为 `(B, F, J, C)`（batch, frames, joints, 2D channels），输出为 `(B, F, J, 3)`。

```
MixSTE/
├── run.py                    # 训练/评估主脚本
├── requirements.yaml         # conda 环境
├── common/
│   ├── model_cross.py        # 核心模型: MixSTE2, Block, Attention 等
│   ├── loss.py               # mpjpe, weighted_mpjpe, velocity error, bone/sym loss
│   ├── arguments.py          # 命令行参数定义
│   ├── generators.py         # ChunkedGenerator_Seq / UnchunkedGenerator_Seq
│   ├── rela.py               # RectifiedLinearAttention (替代注意力)
│   ├── routing_transformer.py# KmeansAttention (替代注意力)
│   ├── linearattention.py    # LinearMultiheadAttention / Linformer (替代注意力)
│   ├── h36m_dataset.py       # Human3.6M 数据集
│   ├── skeleton.py           # 骨架定义
│   ├── camera.py             # 相机坐标变换
│   └── visualization.py      # 可视化
└── data/                     # 数据集目录 (需自行准备)
```

## 关键事实

### 1. 主模型 MixSTE2 的时空交替前向传播

输入 `(B, F, J, C)` 经过三阶段：STE_forward → TTE_forward → ST_forward（交替循环）。

`common/model_cross.py:529-560` — MixSTE2.forward:
```python
def forward(self, x):
    b, f, n, c = x.shape
    x = self.STE_forward(x)       # 第0层空间注意力
    x = self.TTE_foward(x)        # 第0层时间注意力
    x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
    x = self.ST_foward(x)         # 第1~depth-1层交替
    x = self.head(x)              # LayerNorm + Linear -> 3D
    x = x.view(b, f, n, -1)
    return x
```

### 2. 形状流转约定 (B, F, J, C → B, F, J, 3)

`common/model_cross.py:468-483` — STE_forward（空间注意力）:
```python
def STE_forward(self, x):
    b, f, n, c = x.shape  # b=batch, f=frames, n=joints, c=2
    x = rearrange(x, 'b f n c  -> (b f) n c')
    x = self.Spatial_patch_to_embedding(x)  # Linear(2, embed_dim_ratio)
    x += self.Spatial_pos_embed             # (1, num_joints, embed_dim_ratio)
    x = self.pos_drop(x)
    blk = self.STEblocks[0]
    x = blk(x)                              # shape: (B*F, J, C_embed)
    x = self.Spatial_norm(x)
    x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
    return x
```

`common/model_cross.py:485-496` — TTE_forward（时间注意力）:
```python
def TTE_foward(self, x):
    assert len(x.shape) == 3  # (B*J, F, C_embed)
    b, f, _ = x.shape
    x += self.Temporal_pos_embed  # (1, num_frame, embed_dim)
    x = self.pos_drop(x)
    blk = self.TTEblocks[0]
    x = blk(x)
    x = self.Temporal_norm(x)
    return x
```

`common/model_cross.py:498-527` — ST_forward（第1层起的交替循环）:
```python
def ST_foward(self, x):
    assert len(x.shape)==4  # (B, F, J, C_embed)
    b, f, n, cw = x.shape
    for i in range(1, self.block_depth):
        x = rearrange(x, 'b f n cw -> (b f) n cw')
        x = steblock(x)                    # 空间注意力
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        x = tteblock(x)                    # 时间注意力
        x = self.Temporal_norm(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
    return x
```

**形状流转总结**:
- 输入: `(B, F, J, 2)`
- 空间注意力时: reshape 为 `(B*F, J, C_embed)` — 每帧独立做关节间注意力
- 时间注意力时: reshape 为 `(B*J, F, C_embed)` — 每关节独立做帧间注意力
- 输出: `(B, F, J, 3)`

### 3. Block 结构（注意力 + FFN + 可选维度变换）

`common/model_cross.py:299-340`:
```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, ...):
        self.norm1 = norm_layer(dim)
        self.attn = attention(dim, num_heads=num_heads, ...)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), ...)

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

**关键**: `Block.__init__` 接受 `attention` 参数（默认 `Attention` 类），这是替换注意力机制的注入点。

### 4. 标准 Attention 实现

`common/model_cross.py:66-110`:
```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, ...):
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
```

### 5. Loss 组成

`run.py:397-439` — 训练损失:
```python
# 加权 MPJPE（按关节权重）
w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
loss_3d_pos = weighted_mpjpe(predicted_3d_pos, inputs_3d, w_mpjpe)

# 时间一致性损失
dif_seq = predicted_3d_pos[:,1:,:,:] - predicted_3d_pos[:,:-1,:,:]
dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))
loss_diff = 0.5 * dif_seq + 2.0 * mean_velocity_error_train(predicted_3d_pos, inputs_3d, axis=1)

loss_total = loss_3d_pos + loss_diff
loss_total.backward(loss_total.clone().detach())  # 自加权梯度
```

`common/loss.py:21-27` — weighted_mpjpe:
```python
def weighted_mpjpe(predicted, target, w):
    assert predicted.shape == target.shape
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))
```

### 6. 训练配置（模型实例化）

`run.py:230-234`:
```python
model_pos_train = MixSTE2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
    embed_dim_ratio=args.cs, depth=args.dep,
    num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
```

`run.py:323` — 优化器:
```python
optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)
```

### 7. 已提供的替代注意力模块

`common/rela.py:38-70` — RectifiedLinearAttention（用 ReLU 替代 softmax）:
```python
class RectifiedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None, qkv_bias=False, comb=False, vis=False):
        ...
    def forward(self, x, vis=False):
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = F.relu(dots)  # 无 softmax
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        ...
```

`common/model_cross.py:967-971` — MixSTERELA 中注入方式:
```python
self.TTEblocks = nn.ModuleList([
    Block(dim=embed_dim, ..., attention=RectifiedLinearAttention, ...)
    for i in range(depth)])
```

## 硬编码参数与配置点

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `num_heads` | 8 | `run.py:231` | 注意力头数，硬编码在实例化处 |
| `mlp_ratio` | 2.0 | `run.py:231` | FFN 隐层倍率 |
| `drop_path_rate` | 0.1 (train) / 0 (eval) | `run.py:231,234` | 随机深度 |
| `embed_dim_ratio` (cs) | 512 | `arguments.py:49` (`-cs`) | 嵌入维度，命令行可配 |
| `depth` (dep) | 8 | `arguments.py:50` (`-dep`) | 交替层数，命令行可配 |
| `num_frame` | 243 | `arguments.py:58` (`-f`) | 输入帧数 |
| `batch_size` | 1024 | `arguments.py:41` (`-b`) | 批大小 |
| `lr` | 0.00004 | `arguments.py:43` (`-lr`) | 学习率 |
| `lr_decay` | 0.99 | `arguments.py:44` (`-lrd`) | 每 epoch 衰减 |
| `epochs` | 120 | `arguments.py:40` (`-e`) | 训练轮数 |
| `weight_decay` | 0.1 | `run.py:323` | AdamW 权重衰减，硬编码 |
| `w_mpjpe` 关节权重 | `[1,1,2.5,...]` | `run.py:401` | 17 关节权重，硬编码 |
| `loss_diff` 系数 | 0.5, 2.0 | `run.py:423` | 时间一致性损失权重 |
| `out_dim` | 3 | `model_cross.py:421` | 输出维度 (x,y,z) |
| `in_chans` | 2 | `model_cross.py:398` | 输入通道 (x,y) |
| `num_joints` | 17 (从数据) | `run.py:226` | 关节数，从 metadata 读取 |

**改为可配置的方式**: `num_heads`, `mlp_ratio`, `weight_decay`, 损失权重等目前硬编码在 `run.py` 中，可在 `arguments.py` 添加对应 argparse 参数后传入。

## 环境与复现

**依赖** (来自 `requirements.yaml` 和 README):
- Python 3.6.10, PyTorch 1.8.1, CUDA 10.2
- 关键 pip 包: `einops==0.3.0`, `timm==0.4.5`, `tensorboard==2.4.1`
- 完整环境: `conda env create -f requirements.yaml`

**数据准备**:
- Human3.6M: 按 [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) 方式放置于 `./data/`
- 需要 `data/data_3d_h36m.npz` 和 `data/data_2d_h36m_cpn_ft_h36m_dbb.npz`

**权重下载**:
- [Baidu Disk](https://pan.baidu.com/s/1Gu7ItpkU0Q7SF_QVmlQ15A) (提取码: wnjf)
- [Google Drive](https://drive.google.com/drive/folders/1G2mlMHebM6KcbI45FszlosIHgA4jiR3Y)

**最小运行命令**:
```bash
# 评估 (243帧输入)
python run.py -k cpn_ft_h36m_dbb -c <checkpoint_path> --evaluate <checkpoint_file> -f 243 -s 243

# 训练 (双GPU, 243帧)
python run.py -k cpn_ft_h36m_dbb -f 243 -s 243 -l log/run -c checkpoint -gpu 0,1
```

## 改造接口点

### 替换注意力机制（最小侵入）

**方案 A: 通过 Block 的 `attention` 参数注入（推荐）**

`common/model_cross.py:312` 中 `Block.__init__` 已支持 `attention` 参数:
```python
self.attn = attention(dim, num_heads=num_heads, ...)
```

只需:
1. 实现新注意力类，签名兼容 `__init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, comb, vis)` 和 `forward(self, x, vis=False)`
2. 在 `MixSTE2.__init__` 的 `STEblocks` / `TTEblocks` 构造时传入 `attention=YourAttention`

参考已有示例: `MixSTERELA` 类 (`model_cross.py:967-971`) 将时间注意力的 `attention` 参数替换为 `RectifiedLinearAttention`。

**方案 B: 仅替换时间注意力**

时间注意力序列长度 = `num_frame`（如 243），是计算瓶颈。空间注意力序列长度 = `num_joints`（17），计算量小。因此高效注意力优先替换 TTEblocks。

**方案 C: 修改 ST_foward 中的 rearrange 逻辑**

若新注意力需要同时看到时空信息（如 full 3D attention），需修改 `ST_foward` 中的 reshape 逻辑，不再拆分为 `(B*F, J, C)` 和 `(B*J, F, C)`，而是保持 `(B, F*J, C)` 或 `(B, F, J, C)` 直接输入。

### 修改损失函数

- 主损失在 `run.py:405` (`weighted_mpjpe`) 和 `run.py:423` (`loss_diff`)
- 添加新损失项只需在 `loss_total = loss_3d_pos + loss_diff` 处追加

### 修改骨架/关节数

- `num_joints` 从数据集 metadata 自动读取 (`run.py:226`)
- `Spatial_pos_embed` 形状为 `(1, num_joints, embed_dim_ratio)`，随 `num_joints` 自动适配
- 关节权重 `w_mpjpe` (`run.py:401`) 需手动调整长度

## 风险与未知

1. **`loss_total.backward(loss_total.clone().detach())`** (`run.py:441`): 使用 loss 值本身作为梯度权重（自加权），这在数学上等价于对 loss^2/2 求导，是否为有意设计未确认。
2. **`TemporalAttention` 类** (`model_cross.py:112-158`): 包含未使用的 `q1, q2, q3` 分割逻辑（第140-141行），疑似实验残留，实际未被 MixSTE2 调用。
3. **`TemporalBlock` 类** (`model_cross.py:342-380`): 末尾硬编码了 `self.reduction = nn.Linear(dim, dim//2)`（第370行），会覆盖条件分支中的 reduction，疑似 bug，但该类未被 MixSTE2 使用。
4. **`Cross_Linformer` 类** (`model_cross.py:735-914`): 引用了未定义的 `LinformerBlock`，无法直接运行。
5. **评估时的 test-time augmentation**: 使用翻转平均 (`run.py:472-494`)，若替换注意力需确保对翻转对称性无假设。
6. **数据格式**: 2D 检测使用 CPN 检测结果 (`cpn_ft_h36m_dbb`)，其他检测器的兼容性未验证。
7. **`routing_transformer.py`** 中 `KmeansAttention` 的具体实现未详细验证其与 Block 签名的兼容性。
