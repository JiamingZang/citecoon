# MotionBERT 代码侦察卡

> 仓库: https://github.com/Walter0807/MotionBERT (ICCV 2023)
> 浅克隆 commit: main HEAD (depth=1)

## 架构总览

MotionBERT 采用 **Dual-Stream Spatio-Temporal Transformer (DSTformer)** 作为统一骨架编码器。输入为 2D 骨架序列 `[B, F, 17, 3]`，经关节嵌入 + 时空位置编码后，通过两组并行的 Transformer Block 流（`blocks_st` 先空间后时间、`blocks_ts` 先时间后空间）分别处理，每层用可学习的 `ts_attn` 门控融合两路输出，最终经 `pre_logits`（Linear+Tanh）映射到 `dim_rep=512` 维表征。下游任务（3D 姿态、动作识别、Mesh 恢复）通过替换 head 实现。

```
MotionBERT/
├── lib/model/
│   ├── DSTformer.py        # 核心骨架：Attention, Block, DSTformer
│   ├── model_action.py     # 动作识别包装 ActionNet
│   ├── model_mesh.py       # Mesh 恢复包装
│   ├── drop.py             # DropPath
│   └── loss.py / loss_mesh.py / loss_supcon.py
├── lib/utils/
│   ├── learning.py         # load_backbone, load_pretrained_weights, partial_train_layers
│   └── tools.py / utils_data.py / vismo.py
├── lib/data/               # 数据集 & 增强
├── configs/                # YAML 配置（pretrain/pose3d/action/mesh）
├── train.py                # 预训练 & 3D 姿态微调
├── train_action.py         # 动作识别微调
├── train_mesh.py           # Mesh 微调
├── infer_wild.py           # 野外推理
└── docs/                   # 各任务文档
```

## 关键事实

### 1. 时空注意力实现（DSTformer 双流结构）

**文件**: `lib/model/DSTformer.py:269-358`

DSTformer 的 `forward` 中，每层同时过 `blocks_st`（先 spatial 后 temporal）和 `blocks_ts`（先 temporal 后 spatial），再用逐层门控融合：

```python
# lib/model/DSTformer.py:340-351
for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
    x_st = blk_st(x, F)
    x_ts = blk_ts(x, F)
    if self.att_fuse:
        att = self.ts_attn[idx]
        alpha = torch.cat([x_st, x_ts], dim=-1)
        BF, J = alpha.shape[:2]
        alpha = att(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x_st * alpha[:,:,0:1] + x_ts * alpha[:,:,1:2]
    else:
        x = (x_st + x_ts)*0.5
```

### 2. Block 内部的 stage_st / stage_ts 模式

**文件**: `lib/model/DSTformer.py:239-267`

```python
# lib/model/DSTformer.py:240-244  (stage_st: 先空间后时间)
if self.st_mode=='stage_st':
    x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
    x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
    x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
    x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
```

每个 Block 内含独立的 `attn_s`（spatial）和 `attn_t`（temporal）两个 Attention 实例及对应 MLP。

### 3. 时间注意力的 reshape 逻辑

**文件**: `lib/model/DSTformer.py:188-200`

```python
def forward_temporal(self, q, k, v, seqlen=8):
    B, _, N, C = q.shape
    qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) #(B, H, N, T, C)
    kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
    vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
    attn = (qt @ kt.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ vt #(B, H, N, T, C)
    x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads)
    return x
```

时间注意力对每个关节独立做跨帧 self-attention（维度 `[B, H, N_joints, T, C]`）。

### 4. 预训练权重加载接口

**文件**: `lib/utils/learning.py:79-101` — `load_backbone(args)` 根据 `args.backbone` 构造模型：

```python
def load_backbone(args):
    if not(hasattr(args, "backbone")):
        args.backbone = 'DSTformer' # Default
    if args.backbone=='DSTformer':
        model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
```

**文件**: `lib/utils/learning.py:39-67` — `load_pretrained_weights` 做 key 匹配、跳过不兼容层：

```python
def load_pretrained_weights(model, checkpoint):
    ...
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
```

**文件**: `train_action.py:87-94` — 微调时加载预训练 backbone：

```python
if args.finetune:
    ...
    chk_filename = os.path.join(opts.pretrained, opts.selection)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
    model_backbone = load_pretrained_weights(model_backbone, checkpoint)
```

### 5. 表征提取接口 `get_representation`

**文件**: `lib/model/DSTformer.py:360-361`

```python
def get_representation(self, x):
    return self.forward(x, return_rep=True)
```

返回 `[B, F, J, dim_rep]`（即 `pre_logits` 之后、`head` 之前的特征）。

### 6. 位置编码（时间 + 空间）

**文件**: `lib/model/DSTformer.py:301-304`

```python
self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
trunc_normal_(self.temp_embed, std=.02)
trunc_normal_(self.pos_embed, std=.02)
```

时间编码按帧索引切片 `self.temp_embed[:,:F,:,:]`（第336行），支持变长输入（≤maxlen）。

### 7. 门控融合层 `ts_attn` 初始化

**文件**: `lib/model/DSTformer.py:307-311`

```python
if self.att_fuse:
    self.ts_attn = nn.ModuleList([nn.Linear(dim_feat*2, 2) for i in range(depth)])
    for i in range(depth):
        self.ts_attn[i].weight.data.fill_(0)
        self.ts_attn[i].bias.data.fill_(0.5)
```

初始化为均匀融合（softmax([0.5, 0.5]) = [0.5, 0.5]），训练中自适应学习时空权重。

## 硬编码参数与配置点

| 参数 | 默认值 | 位置 | 配置方式 |
|------|--------|------|----------|
| `dim_feat` | 512 | `configs/pretrain/MB_pretrain.yaml:19` | YAML `dim_feat` 字段 |
| `depth` | 5 | `configs/pretrain/MB_pretrain.yaml:21` | YAML `depth` 字段 |
| `num_heads` | 8 | `configs/pretrain/MB_pretrain.yaml:23` | YAML `num_heads` 字段 |
| `mlp_ratio` | 2 | `configs/pretrain/MB_pretrain.yaml:20` | YAML `mlp_ratio` 字段 |
| `dim_rep` | 512 | `configs/pretrain/MB_pretrain.yaml:22` | YAML `dim_rep` 字段 |
| `maxlen` | 243 | `configs/pretrain/MB_pretrain.yaml:18` | YAML `maxlen` 字段 |
| `num_joints` | 17 | `configs/pretrain/MB_pretrain.yaml:34` | YAML `num_joints` 字段 |
| `att_fuse` | True | `configs/pretrain/MB_pretrain.yaml:24` | YAML `att_fuse` 字段 |
| `dim_in` / `dim_out` | 3 / 3 | `lib/utils/learning.py:83` 硬编码 | 需改代码或扩展 `load_backbone` |
| `clip_len` | 243 | YAML `clip_len` | 控制训练时序列长度 |
| `batch_size` | 64 (pretrain) / 32 (ft) | YAML | 直接改 YAML |
| `lr_decay` | 0.99 | YAML | 指数衰减，每 epoch 乘一次 |
| LayerNorm eps | 1e-6 | `lib/utils/learning.py:84` 硬编码 | 需改代码 |

**改为可配置的方式**: 所有模型超参已通过 YAML → `easydict` → `args` 传入 `load_backbone`。若需新增参数（如记忆模块大小），只需在 YAML 中加字段，在 `load_backbone` 或 DSTformer `__init__` 中读取即可。

## 环境与复现

### 依赖

```bash
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

`requirements.txt` 主要依赖: tensorboardX, tqdm, easydict, prettytable, chumpy, opencv-python, imageio-ffmpeg, matplotlib==3.1.1, roma, pytorch-metric-learning, smplx[all]

### 预训练权重下载

- MotionBERT (162MB): [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2)
- MotionBERT-Lite (61MB): [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS27Ydcbpxlkl0ng?e=rq2Btn)
- HuggingFace: https://huggingface.co/walterzhu/MotionBERT

权重为 `.bin` 文件，内部 key 为 `model_pos`（state_dict）。

### 最小运行命令

```bash
# 预训练
python train.py --config configs/pretrain/MB_pretrain.yaml -c checkpoint/pretrain/MB_pretrain

# 3D 姿态微调（需先下载预训练权重到 checkpoint/pretrain/）
python train.py --config configs/pose3d/MB_ft_h36m.yaml \
  -c checkpoint/pose3d/MB_ft_h36m \
  -p checkpoint/pretrain/MB_pretrain \
  --selection latest_epoch.bin

# 动作识别微调
python train_action.py --config configs/action/MB_ft_NTU60_xsub.yaml \
  -c checkpoint/action/MB_ft_NTU60_xsub \
  -p checkpoint/pretrain/MB_pretrain
```

### 数据准备

- AMASS (SMPL+H) + H36M-SH → `data/motion3d/MB3D_f243s81/`
- PoseTrack18 + InstaVariety → `data/motion2d/`
- 详见 `docs/pretrain.md`

## 改造接口点

### 关注点：在 DSTformer 上加测试时记忆模块（Test-Time Memory）

**推荐插入位置 1：Block 输出后、门控融合前（最小侵入）**

- **位置**: `lib/model/DSTformer.py:340-351`，`forward` 方法的 for 循环内
- **方式**: 在 `x_st = blk_st(x, F)` 和 `x_ts = blk_ts(x, F)` 之后、`alpha` 融合之前，对 `x_st`/`x_ts`（shape `[BF, J, dim_feat]`）做 memory read/write。可将当前帧特征作为 query 去读取外部记忆 bank，用检索到的值增强 `x_st`/`x_ts`。
- **优点**: 不修改 Block 内部结构，预训练权重完全兼容（新增模块参数随机初始化）。

**推荐插入位置 2：`pre_logits` 之前（表征级记忆）**

- **位置**: `lib/model/DSTformer.py:352-354`
- **方式**: 在 `x = self.norm(x)` 之后、`x = self.pre_logits(x)` 之前，对 `[BF, J, dim_feat]` 做 cross-attention 到记忆 bank。
- **优点**: 只加一层，计算开销最小；适合测试时自适应（TTA）场景。

**推荐插入位置 3：时间注意力内部（帧级记忆增强）**

- **位置**: `lib/model/DSTformer.py:188-200`，`forward_temporal` 方法
- **方式**: 将 memory bank 中的 key/value 拼接到 `kt`/`vt` 的时间维度上（`[B, H, N, T+M, C]`），使每帧 attend 到历史记忆。
- **优点**: 语义最自然（记忆即"额外历史帧"）；**缺点**: 需修改 Attention 内部，侵入性较大。

**权重加载兼容性**:
- `load_pretrained_weights`（`lib/utils/learning.py:39-67`）自动跳过 size 不匹配或新增的 key，因此新增记忆模块参数不会破坏预训练权重加载。
- 微调时设 `args.finetune = True`，通过 `-p` 指定预训练 checkpoint 路径即可。

**`get_representation` 接口**:
- 下游只需调用 `model.get_representation(x)` 获取 `[B, F, J, 512]` 表征，记忆模块可封装在 DSTformer 内部对外透明。

## 风险与未知

1. **`dim_in=3` 硬编码**: `load_backbone` 中 `dim_in=3` 写死（`lib/utils/learning.py:83`），若记忆模块需要额外输入通道（如 confidence 或 memory indicator），需修改此处。
2. **DataParallel 兼容**: 训练脚本使用 `nn.DataParallel`（`train.py:257`），记忆模块若含非 tensor 状态（如动态增长的 bank），需确认多 GPU 下行为正确。
3. **变长序列**: `temp_embed` 按 `[:,:F,:,:]` 切片支持变长，但记忆模块若依赖固定序列长度需额外处理。
4. **Lite 模型差异**: `MB_lite.yaml` 的 `dim_feat`/`depth` 不同（未在本次精读中确认具体值），记忆模块超参需随之调整。
5. **权重文件格式**: checkpoint 内 key 为 `model_pos`（预训练）或 `model`（ActionNet 整体），加载时需注意区分。
6. **Python 3.7 + 旧版 PyTorch**: README 指定 python=3.7 + CUDA 11.6，较新环境可能有兼容问题（如 `torch.load` 的 `weights_only` 参数）。
7. **未验证**: `model_mesh.py` 的 SMPL 回归头细节、`infer_wild.py` 的 2D 检测管线、NTU 120 one-shot 训练流程未精读。
