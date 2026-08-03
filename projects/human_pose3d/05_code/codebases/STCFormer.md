# STCFormer — 3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention (CVPR 2023)

## 架构总览

STCFormer 将 2D 关键点序列 (B, T, S, 2) 经线性嵌入后送入 N 层 STC_BLOCK；每个 STC_BLOCK 内部由 STC_ATTENTION（时空交叉注意力）+ FFN 组成。STC_ATTENTION 的核心思想是将 QKV 沿通道维度一分为二，一半做空间注意力（跨关节），一半做时间注意力（跨帧），从而将全局时空注意力 O((T×S)²) 分解为 O(T×S² + S×T²)。最后经线性回归头输出 3D 坐标。

```
STCFormer/
├── run_stc.py                  # 训练/测试入口
├── common/
│   ├── opt.py                  # 命令行参数 & stride 配置
│   ├── h36m_dataset.py         # Human3.6M 数据集
│   ├── load_data_hm36_tds.py   # 数据加载器
│   └── ...
├── model/
│   ├── stcformer.py            # ★ 核心模型（STC_ATTENTION / STC_BLOCK / STCFormer / Model）
│   └── block/
│       ├── refine.py           # 后处理 refine 模块
│       ├── vanilla_transformer_encoder.py   # 标准 Transformer（对比基线）
│       └── strided_transformer_encoder.py   # Strided Transformer（对比基线）
├── checkpoint/                 # 权重目录
├── dataset/                    # 数据目录
└── demo/                       # 视频推理 demo
```

## 关键事实

### 1. 时空交叉注意力的拆分方式

QKV 投影后沿通道维度等分为空间组和时间组：

`model/stcformer.py:74-81`
```python
qkv = self.qkv(x)  # b, t, s, c-> b, t, s, 3*c
qkv = qkv.reshape(b, t, s, c, 3).permute(4, 0, 1, 2, 3)  # 3,b,t,s,c

# space group and time group
qkv_s, qkv_t = qkv.chunk(2, 4)  # [3,b,t,s,c//2],  [3,b,t,s,c//2]

q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]  # b,t,s,c//2
q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]  # b,t,s,c//2
```

### 2. 空间注意力：对每个时间步，在关节维度做 self-attention

`model/stcformer.py:84-93`
```python
q_s = rearrange(q_s, 'b t s (h c) -> (b h t) s c', h=self.head)  # b*h*t,s,c//2//h
k_s = rearrange(k_s, 'b t s (h c) -> (b h t) c s ', h=self.head)  # b*h*t,c//2//h,s

att_s = (q_s @ k_s) * self.scale  # b*h*t,s,s
att_s = att_s.softmax(-1)  # b*h*t,s,s
```

复杂度：每帧 O(S²·d)，共 T 帧 → O(T·S²·d)

### 3. 时间注意力：对每个关节，在时间维度做 self-attention

`model/stcformer.py:87-94`
```python
q_t = rearrange(q_t, 'b  t s (h c) -> (b h s) t c', h=self.head)  # b*h*s,t,c//2//h
k_t = rearrange(k_t, 'b  t s (h c) -> (b h s) c t ', h=self.head)  # b*h*s,c//2//h,t

att_t = (q_t @ k_t) * self.scale  # b*h*s,t,t
att_t = att_t.softmax(-1)  # b*h*s,t,t
```

复杂度：每关节 O(T²·d)，共 S 个关节 → O(S·T²·d)

### 4. 与全局注意力的复杂度对比

全局时空注意力需将 T×S 个 token 展平后做 self-attention，复杂度 O((T·S)²·d)。
STCFormer 拆分后总复杂度 O(T·S²·d + S·T²·d) = O(T·S·(T+S)·d)。
以默认 T=27, S=17 为例：全局 = (27×17)² = 210,849；拆分 = 27×17² + 17×27² = 7,803 + 12,393 = 20,196，约降低 10×。

### 5. 辅助分支：深度卷积 (sep2) + 身体部位嵌入 (sep1)

`model/stcformer.py:63-64`（深度可分离卷积，kernel=3，groups=c//2）
```python
self.sep2_t = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
self.sep2_s = nn.Conv2d(d_coor // 2, d_coor // 2, kernel_size=3, stride=1, padding=1, groups=d_coor // 2)
```

`model/stcformer.py:59-60`（身体部位分组嵌入，5 组对应 17 关节）
```python
self.emb = nn.Embedding(5, d_coor//head//2)
self.part = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 4, 4, 4]).long().cuda()
```

融合方式（`model/stcformer.py:116-117`）：
```python
x_s = att_s @ v_s + sep2_s + 0.0001 * self.drop(sep_s)  # b*h*t,s,c//2//h
x_t = att_t @ v_t + sep2_t  # b*h*s,t,c//2//h
```

### 6. 最终拼接 + 投影 + 残差

`model/stcformer.py:124-130`
```python
x = torch.cat((x_s, x_t), -1)  # b,h,t,s,c//h
x = rearrange(x, 'b h t s c -> b  t s (h c) ')  # b,t,s,c

# projection and skip-connection
x = self.proj(x)
x = x + h
```

### 7. STC_BLOCK 结构：STC_ATTENTION + LayerNorm + MLP(4×扩展) + DropPath

`model/stcformer.py:133-149`
```python
class STC_BLOCK(nn.Module):
    def __init__(self, d_time, d_joint, d_coor):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_coor)
        self.mlp = Mlp(d_coor, d_coor * 4, d_coor)
        self.stc_att = STC_ATTENTION(d_time, d_joint, d_coor)
        self.drop = DropPath(0.0)

    def forward(self, input):
        b, t, s, c = input.shape
        x = self.stc_att(input)
        x = x + self.drop(self.mlp(self.layer_norm(x)))
        return x
```

### 8. 顶层 Model 流水线

`model/stcformer.py:15-40`
```python
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        layers, d_hid, frames = args.layers, args.d_hid, args.frames
        num_joints_in, num_joints_out = args.n_joints, args.out_joints
        self.pose_emb = nn.Linear(2, d_hid, bias=False)
        self.gelu = nn.GELU()
        self.stcformer = STCFormer(layers, frames, num_joints_in, d_hid)
        self.regress_head = nn.Linear(d_hid, 3, bias=False)

    def forward(self, x):
        x = self.pose_emb(x)
        x = self.gelu(x)
        x = self.stcformer(x)
        x = self.regress_head(x)
        return x
```

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置方式 |
|------|-----|------|----------------|
| 注意力头数 head | 8 | `model/stcformer.py:44` `STC_ATTENTION.__init__(head=8)` | 提升为 `args.heads` 传入 |
| DropPath 率（注意力内） | 0.5 | `model/stcformer.py:66` `self.drop = DropPath(0.5)` | 提升为 `args.drop_path` |
| DropPath 率（FFN） | 0.0 | `model/stcformer.py:142` `self.drop = DropPath(0.0)` | 同上 |
| MLP 扩展比 | 4× | `model/stcformer.py:139` `Mlp(d_coor, d_coor * 4, d_coor)` | 提升为 `args.mlp_ratio` |
| 身体部位分组 | `[0,1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4]`（5组/17关节） | `model/stcformer.py:60` | 改为从配置文件读取，适配不同骨架 |
| Embedding 组数 | 5 | `model/stcformer.py:59` `nn.Embedding(5, ...)` | 与 part 联动配置 |
| sep 融合系数 | 空间 0.0001 / 时间 1e-9 | `model/stcformer.py:116,122` | 提升为可学习参数或超参 |
| 深度卷积 kernel | 3 | `model/stcformer.py:63-64` | 提升为 `args.sep_kernel` |
| scale 因子 | `(d_coor // 2) ** -0.5` | `model/stcformer.py:51` | 随 d_coor 自动计算，无需额外配置 |
| d_hid | 256 | `common/opt.py:15` `--d_hid` | 已是命令行参数 |
| layers | 6 | `common/opt.py:13` `--layers` | 已是命令行参数 |
| frames | 27/81/243 | `common/opt.py:38` `-f/--frames` | 已是命令行参数 |
| stride_num 映射 | `{'27':[3,3,3], '81':[3,3,3,3], '243':[3,3,3,3,3], ...}` | `common/opt.py:72-78` | 硬编码字典，新帧数需手动添加 |
| refine 隐层 | 1024 | `model/block/refine.py:6` `fc_unit = 1024` | 提升为参数 |
| `.cuda()` 硬编码 | `self.part = ...cuda()` | `model/stcformer.py:60` | 改用 `register_buffer` 以支持 CPU |

## 环境与复现

### 依赖（README 声明 + 代码实际 import）
- PyTorch >= 0.4.0（实际使用 `torch.nn`, `torch.utils.data`）
- NumPy
- Matplotlib = 3.1.0
- einops（`from einops import rearrange`）
- timm（`from timm.models.layers import DropPath`）
- scipy（`import scipy.sparse as sp`, `scipy.io`）
- tqdm

无 `requirements.txt` / `setup.py`，需手动安装。

### 数据准备
- Human3.6M：按 VideoPose3D 方式组织，放入 `dataset/` 目录
- MPI-INF-3DHP：按 P-STMO 方式组织

### 权重下载
- Google Drive: https://drive.google.com/drive/folders/1waaQ1Yj-HfbNahnCN8AWCjMCGzyhZJF7
- 百度网盘: https://pan.baidu.com/s/1axVQNHxdZFH4Eiqiy2EvYQ （提取码 STC1）
- 放入 `./checkpoint/` 目录

### 最小运行命令
```bash
# 训练（Human3.6M, 27帧）
python run_stc.py -f 27 -b 128 --train 1 --layers 6 -s 3

# 测试（CPN 2D keypoints, 27帧）
python run_stc.py -f 27 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' \
  --reload 1 --previous_dir ./checkpoint/model_27_STCFormer/no_refine_6_4406.pth

# 测试（81帧）
python run_stc.py -f 81 -b 128 --train 0 --layers 6 -s 1 -k 'cpn_ft_h36m_dbb' \
  --reload 1 --previous_dir ./checkpoint/model_81_STCFormer/no_refine_6_4172.pth
```

### 性能参考（CPN keypoints）
| Frames | P1 (mm) | P2 (mm) |
|--------|---------|---------|
| 27     | 44.08   | 34.76   |
| 81     | 41.72   | 32.94   |

## 改造接口点

### 针对"时空交叉注意力拆分方式"的最小侵入修改

1. **调整空间/时间通道分配比例**
   - 位置：`model/stcformer.py:78` `qkv_s, qkv_t = qkv.chunk(2, 4)`
   - 方式：将 `chunk(2, 4)` 改为按可配置比例 `split([sp_dim, t_dim], 4)`，同步修改后续 rearrange 中的维度。

2. **替换注意力计算为线性注意力 / 稀疏注意力**
   - 位置：`model/stcformer.py:90-94`（att_s / att_t 的计算）
   - 方式：仅替换 `q@k → softmax → @v` 这三步，输入输出形状不变即可。

3. **修改身体部位分组策略**
   - 位置：`model/stcformer.py:59-60`
   - 方式：将 `self.part` 改为 `register_buffer`，组数和分组映射从 `args` 传入，适配不同骨架拓扑。

4. **调整 sep2 深度卷积 / sep1 嵌入的融合权重**
   - 位置：`model/stcformer.py:116-122`
   - 方式：将 `0.0001` 和 `1e-9` 替换为 `nn.Parameter` 或从超参传入。

5. **增加/减少 STC_BLOCK 层数**
   - 位置：`model/stcformer.py:171-183`（`STCFormer.__init__`）
   - 方式：已通过 `args.layers` 控制，无需额外修改。

6. **添加位置编码**
   - 当前 STCFormer 主体无显式位置编码（对比 `vanilla_transformer_encoder.py:112` 有 `pos_embedding`）。
   - 位置：`model/stcformer.py:185` `STCFormer.forward` 入口处加入可学习/正弦位置编码。

## 风险与未知

- **243 帧模型不可用**：作者声明该模型因版权存于公司服务器，未公开发布。
- **MPI-INF-3DHP 代码不完整**：README 注明"based on an earlier version and may lack organization"，对应文件 `run_3dhp_stc.py` 未在仓库中找到（可能未上传）。
- **无 requirements.txt / setup.py**：依赖版本需自行推断，`timm` 版本兼容性未验证。
- **`.cuda()` 硬编码**：`model/stcformer.py:60` 的 `self.part` 直接调用 `.cuda()`，CPU 环境会报错。
- **stride_num 字典不完整**：仅覆盖 9/27/81/243/351 帧，其他帧数会直接 `exit()`（`common/opt.py:84-85`）。
- **DropPath(0.5) 是否过大**：注意力内 drop 率 0.5 而 FFN 为 0.0，未见消融实验说明。
- **sep1 融合系数极小（0.0001 / 1e-9）**：实际贡献是否可忽略未验证，可能是训练技巧或遗留代码。
- **`vanilla_transformer_encoder.py` 和 `strided_transformer_encoder.py` 在当前 `run_stc.py` 中未被调用**，可能为历史对比代码或用于其他实验脚本。
