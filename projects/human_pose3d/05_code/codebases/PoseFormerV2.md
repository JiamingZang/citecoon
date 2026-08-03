# PoseFormerV2 代码侦察卡

> 仓库: https://github.com/QitaoZhao/PoseFormerV2 (CVPR 2023)
> 关注点: 频域压缩实现——DCT系数截断数量、截断位置、时频融合模块结构

## 架构总览

PoseFormerV2 在 PoseFormer 的双分支（空间+时间）Transformer 基础上，引入 DCT 频域分支：对完整输入序列做 1D-DCT 后仅保留前 n 个低频系数，经线性嵌入后与空间分支的中心帧特征拼接，送入 MixedBlock（前半 token 走时域 MLP，后半 token 走频域 FreqMlp）进行交互，最终两路分别加权求和后拼接输出 3D 姿态。

```
PoseFormerV2/
├── run_poseformer.py              # H36M 训练/评估入口
├── common/
│   ├── model_poseformer.py        # 核心模型 (PoseTransformerV2, MixedBlock, FreqMlp)
│   ├── arguments.py               # 命令行参数 (frame-kept, coeff-kept, depth 等)
│   ├── generators.py / loss.py / h36m_dataset.py ...
├── mpi_inf_3dhp/
│   ├── run_3dhp.py                # 3DHP 训练入口
│   ├── model/model_poseformerv2.py # 3DHP 版模型 (结构相同, 含 CrossBlock 变体)
│   └── common/opt.py              # 3DHP 参数
├── demo/                          # 视频推理 (YOLOv3 + HRNet + PoseFormerV2)
└── requirements.txt
```

## 关键事实

### 1. DCT 变换与系数截断（H36M 主模型）

文件: `common/model_poseformer.py:218`

```python
x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
```

- 输入 shape: `(b, f, p, 2)` → permute 后 `(b, p, 2, f)`，沿最后一维（时间轴）做 type-II DCT。
- 截断方式: 直接切片 `[:num_coeff_kept]`，只保留前 n 个低频系数（含 DC），丢弃所有高频。
- 截断数量由 `self.num_coeff_kept` 控制（见下条）。

### 2. 系数数量配置

文件: `common/model_poseformer.py:160`

```python
self.num_coeff_kept = args.number_of_kept_coeffs if args.number_of_kept_coeffs else self.num_frame_kept
```

文件: `common/arguments.py:50`

```python
parser.add_argument('-coeff-kept', '--number-of-kept-coeffs', type=int, metavar='N', help='how many coefficients are kept')
```

- 命令行参数 `-coeff-kept`，无默认值（None 时回退到 `num_frame_kept`）。
- README 示例: `-frame 27 -frame-kept 3 -coeff-kept 3`，即 27 帧输入只保留 3 个 DCT 系数。
- 预训练模型表中 n 取值: 1, 3, 9, 27（对应不同序列长度）。

### 3. 频域嵌入

文件: `common/model_poseformer.py:164`

```python
self.Freq_embedding = nn.Linear(in_chans*num_joints, embed_dim)
```

文件: `common/model_poseformer.py:219-220`

```python
x = x.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
x = self.Freq_embedding(x)
```

- 截断后 shape: `(b, p, 2, num_coeff_kept)` → reshape 为 `(b, num_coeff_kept, p*2)` → Linear 映射到 `embed_dim = embed_dim_ratio * num_joints`。

### 4. 空间分支中心帧选取

文件: `common/model_poseformer.py:201`

```python
index = torch.arange((f-1)//2-num_frame_kept//2, (f-1)//2+num_frame_kept//2+1)
```

- 从 f 帧中取中心 `num_frame_kept` 帧（默认 27 帧中取 3 帧: index=[12,13,14]）。

### 5. 时频融合: MixedBlock 结构

文件: `common/model_poseformer.py:108-129`

```python
class MixedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., ...):
        ...
        self.mlp1 = Mlp(...)       # 时域 MLP
        self.mlp2 = FreqMlp(...)   # 频域 MLP

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)
```

- 拼接后的 token 序列: `[freq_tokens(num_coeff_kept), spatial_tokens(num_frame_kept)]`。
- 前半（频域 token）走标准 Mlp；后半（空间/时域 token）走 FreqMlp。
- 注意: 代码中 `x[:, :f//2]` 对应频域分支，`x[:, f//2:]` 对应空间分支（因为 forward_features 中 `torch.cat((x, Spatial_feature), dim=1)` 先放 freq 再放 spatial）。

### 6. FreqMlp: 频域内的 DCT→MLP→IDCT

文件: `common/model_poseformer.py:38-57`

```python
class FreqMlp(nn.Module):
    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x
```

- 对空间分支的 token 序列再做一次 DCT（沿 token 维度），MLP 处理后 IDCT 回时域。
- 此处 DCT 不做截断，是完整的变换-处理-反变换。

### 7. 最终输出融合

文件: `common/model_poseformer.py:238`

```python
x = torch.cat((self.weighted_mean(x[:, :self.num_coeff_kept]), self.weighted_mean_(x[:, self.num_coeff_kept:])), dim=-1)
```

文件: `common/model_poseformer.py:189-190`

```python
self.weighted_mean = torch.nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
self.weighted_mean_ = torch.nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)
```

- 频域 token 和空间 token 分别用 1x1 Conv 做加权平均压缩为 1 个 token，再在特征维拼接。
- 最终经 `self.head`（LayerNorm + Linear）输出 `(b, 1, p, 3)`。

### 8. 3DHP 版本差异

文件: `mpi_inf_3dhp/model/model_poseformerv2.py:291`

```python
x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
```

- DCT 截断逻辑与 H36M 版完全一致。
- 额外提供 `opt.naive` 开关（line 244）：为 True 时用纯 Block 替代 MixedBlock（消融实验用）。
- 含 CrossBlock/CrossAttention 变体（line 88-169），但主模型未使用。

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|------|------|------|
| `embed_dim_ratio` | 32 | `common/arguments.py:52` (`--embed-dim-ratio`) | 已有命令行参数 |
| `depth` (Transformer 层数) | 4 | `common/arguments.py:51` (`--depth`) | 已有命令行参数 |
| `num_frame_kept` | 默认 27 | `common/arguments.py:48-49` (`-frame-kept`) | 已有命令行参数 |
| `num_coeff_kept` | 默认 None→等于 frame_kept | `common/arguments.py:50` (`-coeff-kept`) | 已有命令行参数 |
| `num_heads` | 8 | `run_poseformer.py:194` 硬编码 | 需加 argparse 参数 |
| `mlp_ratio` | 2.0 | `run_poseformer.py:194` 硬编码 | 需加 argparse 参数 |
| `drop_path_rate` (train) | 0.1 | `run_poseformer.py:195` 硬编码 | 需加 argparse 参数 |
| `in_chans` | 2 | `common/model_poseformer.py:133` 默认参数 | 改构造函数调用 |
| DCT type | type-II (torch_dct 默认) | `common/model_poseformer.py:218` | torch_dct 库固定 |
| 截断策略 | 取前 n 个低频系数 | `common/model_poseformer.py:218` `[:num_coeff_kept]` | 改为可学习 mask 或 top-k 需修改此行 |
| MixedBlock 前后半分配 | f//2 硬切 | `common/model_poseformer.py:127-128` | 需改为按实际 token 数分割 |

## 环境与复现

**依赖** (来自 `requirements.txt` + README):
- Python 3.9, PyTorch 1.13.0+cu117, CUDA 11.7
- 关键包: `torch-dct==0.1.6`, `einops==0.7.0`, `timm==0.9.12`, `numpy==1.26.4`

**安装**:
```bash
conda create -n poseformerv2 python=3.9
conda activate poseformerv2
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

**数据准备** (Human3.6M):
按 VideoPose3D 方式放置:
```
data/data_2d_h36m_gt.npz
data/data_2d_h36m_cpn_ft_h36m_dbb.npz
data/data_3d_h36m.npz
```

**权重下载**: Google Drive 链接见 README 表格（6 个变体），放入 `checkpoint/` 目录。

**最小训练命令**:
```bash
python run_poseformer.py -g 0 -k cpn_ft_h36m_dbb -frame 27 -frame-kept 3 -coeff-kept 3 -c checkpoint/NAMED_PATH
```

**最小评估命令**:
```bash
python run_poseformer.py -g 0 -k cpn_ft_h36m_dbb -frame 27 -frame-kept 3 -coeff-kept 3 -c checkpoint/ --evaluate NAME_ckpt.bin
```

## 改造接口点

针对"频域压缩"研究关注点，最小侵入的修改位置：

1. **改变 DCT 系数数量/截断策略**: 修改 `common/model_poseformer.py:218` 的切片 `[:num_coeff_kept]`。例如改为可学习 soft-mask、top-k 选择、或自适应截断。只需改这一行及对应的 pos_embed 尺寸（line 168）。

2. **替换 DCT 为其他频域变换（如 FFT/DWT）**: 替换 `common/model_poseformer.py:218` 的 `dct.dct(...)` 调用，以及 `FreqMlp`（line 50, 56）中的 `dct.dct/idct`。接口形状不变即可无缝替换。

3. **修改时频融合方式**: 修改 `MixedBlock.forward`（line 124-129）。当前是硬切 f//2，可改为 cross-attention、gating、或全 token 统一处理。

4. **调整频域/空间 token 比例**: 修改 `forward_features` 中 `torch.cat((x, Spatial_feature), dim=1)`（line 224）的拼接顺序和数量，同时更新 `weighted_mean` 的 `in_channels`（line 189-190）。

5. **增加多尺度频域特征**: 在 `forward_features` 中对不同频段分别嵌入，扩展 `Freq_embedding` 为多组 Linear 或加 FPN 结构。

## 风险与未知

- `torch_dct==0.1.6` 是第三方库，内部实现为矩阵乘法形式的 DCT-II，未验证与 scipy.fft.dct 的数值一致性。
- `MixedBlock` 中 `f//2` 硬切假设 freq token 数 == spatial token 数（即 `num_coeff_kept == num_frame_kept`）；若两者不等，前半/后半划分将错位——代码中未见断言保护。
- 3DHP 版 `model_poseformerv2.py` 中 `embed_dim_ratio` 硬编码为 32（line 220），不走 args；H36M 版走 `args.embed_dim_ratio`。
- 预训练权重托管在 Google Drive，无自动下载脚本，无法确认链接长期有效性。
- `FreqMlp` 对空间 token 做 DCT 的物理含义不明确（token 维度并非严格时间轴），论文中是否有对应消融未查证。
- demo 推理代码依赖 YOLOv3 + HRNet 权重（Google Drive），未验证端到端可运行性。
