# Anatomy3D Repo 卡

> 仓库：https://github.com/sunnychencool/Anatomy3D
> 论文：Anatomy-aware 3D Human Pose Estimation in Videos (arXiv:2002.10322, 2020)
> 基于 VideoPose3D 构建，核心思想：将 3D 姿态分解为**骨长（bone length）**和**骨方向（bone direction）**两个子问题，分别用独立子网络预测后相乘重建。

## 架构总览

项目为单文件训练脚本 + common 工具包结构。`run.py` 是唯一的训练/评估入口，包含数据加载、模型实例化、训练循环和评估逻辑。`common/` 下按职责划分：`model.py` 定义两个模型变体（训练用 `TemporalModelOptimized1f` 和推理用 `TemporalModel`），`bone.py` 提供骨长/骨方向/关节偏移的 torch 计算函数，`generators.py` 负责数据采样与骨长增强，`loss.py` 是标准 MPJPE 等度量，`h36m_dataset.py` 硬编码 Human3.6M 骨骼拓扑与相机参数。

```
Anatomy3D/
├── run.py                  # 训练/评估主入口（数据加载、训练循环、评估）
├── common/
│   ├── model.py            # TemporalModel（推理）+ TemporalModelOptimized1f（训练）
│   ├── bone.py             # getbonelength / getbonedirect / getbonejs（torch）
│   ├── generators.py       # ChunkedGenerator（含 randomaug 骨长增强）/ UnchunkedGenerator
│   ├── loss.py             # mpjpe / p_mpjpe / n_mpjpe / pck / auc
│   ├── arguments.py        # 命令行参数（含 boneindex 默认值）
│   ├── skeleton.py         # Skeleton 类（parents 树、左右对称）
│   ├── h36m_dataset.py     # H3.6M 骨骼定义 + 相机内外参（硬编码）
│   ├── camera.py           # 投影/反投影工具
│   ├── quaternion.py       # 四元数工具
│   ├── mocap_dataset.py    # MocapDataset 基类
│   ├── utils.py            # deterministic_random 等
│   └── visualization.py    # 可视化
├── data/                   # 数据集目录（需用户自行准备）
└── checkpoint/             # 权重目录
```

## 关键事实

### 1. 骨长/骨方向分解的核心实现

骨长计算（L2 范数）：

```python
# common/bone.py:26-36
def getbonelength(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bone = torch.stack(bone,1)
    bone = torch.pow(torch.pow(bone,2).sum(2),0.5)
    bone = bone.view(bs,ss, bone.size(1))
    return bone
```

骨方向计算（单位向量归一化）：

```python
# common/bone.py:39-50
def getbonedirect(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1,seq.size(2),seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bonedirect = torch.stack(bone,1)
    bonesum = torch.pow(torch.pow(bonedirect,2).sum(2), 0.5).unsqueeze(2)
    bonedirect = bonedirect/bonesum
    bonedirect = bonedirect.view(bs,-1,3)
    return bonedirect
```

### 2. 最终 3D 重建 = 骨方向 × 骨长 → 线性映射

训练模型（TemporalModelOptimized1f）中的重建：

```python
# common/model.py:484-488
boned = self.directlinear(bonedirect_2.detach())
bonel = self.lengthlinear(bonelength.detach())
bonel = bonel.view(boned.size())
x = bonel * boned
x = self.shrink(x)
```

推理模型（TemporalModel）中的重建：

```python
# common/model.py:281-288
bonelength = bonelength.permute(0,2,1).contiguous().view(-1,self.num_joints_out-1)
bonel = self.lengthlinear(bonelength).view(x.size(0),-1,self.channels)
bonel = bonel.permute(0,2,1).contiguous()
boned = self.directlinear(bonedirect)
bonel = bonel.view(boned.size())
x = boned * bonel
x = self.shrink(x)
```

### 3. 骨方向网络输出后的 L2 归一化

```python
# common/model.py:276-279 (TemporalModel)
bonedirect = x.view(x.size(0),self.num_joints_out-1,3,x.size(2))
bonesum = torch.pow(torch.pow(bonedirect,2).sum(2),0.5).unsqueeze(2)
bonedirect = bonedirect/bonesum
bonedirect = bonedirect.view(bonedirect.size(0),(self.num_joints_out-1)*3,bonedirect.size(3))
```

同样在 TemporalModelOptimized1f 中两个子网络各自归一化：

```python
# common/model.py:465-468 (sub-network 1)
bonedirect_1 = bonedirect_1.view(bonedirect_1.size(0),self.num_joints_out-1,3)
bonesum_1 = torch.pow(torch.pow(bonedirect_1,2).sum(2),0.5).unsqueeze(2)
bonedirect_1 = bonedirect_1/bonesum_1
```

### 4. 骨长注意力机制（解剖先验：骨长帧间一致）

```python
# common/model.py:420-430 (TemporalModelOptimized1f)
x_rand2 = x_rand.view(bs,ss,-1)
x_rand_abs = torch.abs(x_rand2.detach())
x_rand_con = torch.cat((x_rand2.detach(), x_rand_abs),2)
x_rand_boneatt = self.boneatt(x_rand_con.view(bs*ss,-1)).view(x_rand_con.size(0), x_rand_con.size(1), -1)
x_rand_boneatt = x_rand_boneatt * self.temperature
x_rand_boneatt = self.softmax(x_rand_boneatt)
bone = getbonelength(x_rand2.detach().view(bs,ss,-1,3), self.boneindex)
bonelength = (bone * x_rand_boneatt).sum(1)
```

注意力模块定义：

```python
# common/model.py:176-177
self.boneatt = nn.Linear(num_joints_out*6,num_joints_out-1)
self.softmax = nn.Softmax(dim=1)
```

### 5. 可见性分数注入骨方向网络

```python
# common/model.py:253-254 (TemporalModel._forward_blocks)
xscore = self.drop(self.relu(self.expand_bnscore(self.expand_convscore(xscore))))
x = torch.cat((x, xscore*x),1)
```

可见性分数来源（AlphaPose 预测）：

```python
# run.py:90-91
with open('data/score.pkl', 'rb') as f:
    score = pickle.load(f)
```

### 6. 骨长增强（数据层面的解剖先验）

```python
# common/generators.py:43-88
def randomaug(batch_3D_rand_ori, boneindex, augdegree):
    ...
    bonelen = getbone(batch_3D_rand_ori, boneindex).mean(1)
    bonelenmean = bonelen.mean(0)
    randadd = (np.random.rand(bs,16)-0.5) * (bonelenmean * augdegree)
    bonelennew = bonelen + randadd
    bonedirect = getbonedirect(batch_3D_rand_ori, boneindex)
    # 逐骨硬编码传播：改变一根骨长 → 影响下游所有关节位置
    b = randadd[:,0]
    batch_3D_rand_ori[:,:,16:17] = ... + bonedirect[:,:,0] * b
    b = randadd[:,1]
    batch_3D_rand_ori[:,:,15:17] = ... + bonedirect[:,:,1] * b
    ...（共 16 根骨，逐根展开）
```

### 7. 训练损失组成

```python
# run.py:328-348
loss_direct = args.wd*torch.pow(inputs_3d_direct - bonedirect_2,2).sum(2).mean() + args.wd*args.snd*torch.pow(inputs_3d_direct - bonedirect_1,2).sum(2).mean()
loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
loss_js = args.wjs*mpjpe(predicted_js_2, inputs_3d_js) + args.wjs*args.snd*mpjpe(predicted_js_1, inputs_3d_js)
loss_length = args.wl*torch.pow(inputs_3d_length - bonelength,2).mean()
loss_lengthaug = args.wl*torch.pow(inputs_3d_lengthnew - bonelengthaug,2).mean()
loss_len = loss_3d_pos_rand + loss_3d_pos_randaug + loss_length + loss_lengthaug
loss_total = loss_3d_pos + loss_len + loss_direct + loss_js
```

### 8. 骨方向网络双子结构

第一子网络（膨胀卷积，并行处理所有帧）+ 第二子网络（步幅卷积，逐帧处理），通过 `xbottom` 列表传递中间特征：

```python
# common/model.py:453-462 (TemporalModelOptimized1f, first sub-network)
x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
xbottom = [x.detach()]
x = torch.cat((x, xscore*x),1)
for i in range(len(self.pad) - 1):
    res = x[:, :self.channels, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
    x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
    x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
    xbottom.append(x.detach())
```

## 硬编码参数与配置点

| 参数 | 值 | 位置 | 改为可配置的方式 |
|------|------|------|------|
| boneindex（H3.6M 17关节→16骨） | `'16,15,15,14,13,12,12,11,10,9,9,8,8,7,8,11,8,14,7,0,3,2,2,1,6,5,5,4,1,0,4,0'` | `common/arguments.py:33` | 已通过 `--boneindex` 命令行参数暴露，但 `randomaug()` 中的传播逻辑仍硬编码 |
| 骨数量 16 | `np.random.rand(bs,16)` | `common/generators.py:49` | 改为 `len(boneindex)` |
| 关节数量 17 | `for i in range(17)` | `common/bone.py:16` | 改为 `seq.size(1)` 或传入参数 |
| GPU 设备 | `"0,1,2"` | `run.py:30` | 改为 argparse 参数或自动检测 |
| DataParallel device_ids | `[0,1,2]` | `run.py:207-208` | 同上 |
| temperature（注意力温度） | 默认 10 | `common/arguments.py:52` | 已可配置 `--temperature` |
| augdegree（骨长增强幅度） | 默认 0.6 | `common/arguments.py:51` | 已可配置 `--augdegree` |
| randnum（骨长网络采样帧数） | 默认 50 | `common/arguments.py:50` | 已可配置 `--randnum` |
| 损失权重 wd/wl/wjs/snd | 0.3 / 100 / 2 / 0.5 | `common/arguments.py:55-57` | 已可配置 |
| randomaug 中关节传播索引 | 硬编码 16 段 | `common/generators.py:56-87` | 需根据 skeleton parents 自动生成传播链 |
| H3.6M 骨骼 parents | 32 关节 parent 数组 | `common/h36m_dataset.py:14-17` | 若换数据集需替换整个 Skeleton 定义 |
| 随机轨迹增强参数 | `scale=0.5, z+5, y-0.3` | `common/generators.py:276-278` | 抽为配置参数 |
| PCK 阈值 | 0.15 | `common/loss.py:27` | 改为函数参数 |
| AUC 离散化 | 150 步, 0~0.15 | `common/loss.py:40-44` | 改为函数参数 |

## 环境与复现

**依赖**（README 声明）：
- Python 3.6.10
- PyTorch 1.0.1
- CUDA 9.0
- NumPy（无 requirements.txt，隐式依赖）

**数据准备**：
1. 按 VideoPose3D 方式准备 Human3.6M 数据，放入 `./data/` 目录：
   - `data/data_3d_h36m.npz`
   - `data/data_2d_h36m_cpn_ft_h36m_dbb.npz`
2. 下载 AlphaPose 可见性分数文件（[Google Drive 链接](https://drive.google.com/file/d/1C3A9t9FqKgT_GLROBKV0v3lnl5NXOLYN/view?usp=sharing)），放入 `./data/score.pkl`

**预训练权重**：
- 243 帧模型：[Google Drive](https://drive.google.com/file/d/17QIbAWfCP5fwiz9MhFw1pRBZ_2VlgMSU/view?usp=sharing)，放入 `./checkpoint/pretrained_model.bin`

**最小运行命令**：

```bash
# 训练（243帧标准模型）
python run.py -e 60 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 --randnum 50

# 训练（因果模型）
python run.py -e 60 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 --randnum 50 --causal

# 评估预训练模型
python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 --evaluate pretrained_model.bin
```

**硬件需求**：3× GTX 1080 Ti（243帧模型约 80 小时），2× GTX 1080 Ti（81帧模型约 60 小时）。

## 改造接口点

### 关注点 1：骨长/骨方向分解

- **最小侵入修改位置**：`common/bone.py` 中的 `getbonelength()` 和 `getbonedirect()`。若要替换分解方式（如用旋转表示方向），只需修改这两个函数 + `model.py` 中归一化逻辑（L276-279, L465-468, L478-481）。
- **骨长网络输出维度**：`model.py:166` 的 `nn.Linear(channels, num_joints_out*3)` 输出的是完整 3D 关节位置（再从中提取骨长），若改为直接回归骨长，需改此处输出维度为 `num_joints_out-1`。

### 关注点 2：解剖先验注入方式

- **注意力温度**：`model.py:425` 的 `self.temperature` 控制骨长帧间平滑程度，可直接调参。
- **可见性分数融合**：`model.py:254` 的 `xscore*x` 是逐元素乘法，可替换为 concat 或 gate 机制。
- **骨长增强**：`generators.py:43-88` 的 `randomaug()` 是训练时解剖先验的核心数据增强，`augdegree` 控制扰动幅度。

### 关注点 3：先验硬编码位置（换数据集必改）

1. `common/arguments.py:33` — boneindex 默认值（已可通过命令行覆盖）
2. `common/generators.py:56-87` — randomaug 中 16 根骨到关节的传播索引（**必须手动重写**）
3. `common/h36m_dataset.py:14-17` — 骨骼 parents 定义
4. `common/bone.py:16` — `range(17)` 硬编码关节数

**建议改造路径**：将 `randomaug()` 中的硬编码传播替换为基于 `skeleton.parents()` 的自动链式传播（从骨骼末端向根节点累积偏移），即可支持任意骨骼拓扑。

## 风险与未知

1. **无 requirements.txt / setup.py**：依赖版本仅靠 README 文字描述，PyTorch 1.0.1 + CUDA 9.0 组合在现代硬件上可能不兼容，需自行适配。
2. **score.pkl 格式未文档化**：可见性分数的 pickle 结构（key 命名、shape）只能从 `run.py:108-121` 推断，未提供生成脚本。
3. **randomaug 传播正确性**：硬编码的 16 段关节索引传播逻辑无注释说明对应哪根骨，若 H3.6M 关节顺序变化将静默出错。
4. **DataParallel 硬编码 3 GPU**：`run.py:30,207-208` 写死 3 卡，单卡/多卡环境需手动修改。
5. **骨长网络中 `.detach()` 的使用**：`model.py:422-423` 对骨长网络输出做了 detach，意味着骨长损失不回传到骨长网络的共享层？实际上骨长网络有独立的 loss（`loss_len`），detach 仅阻止骨长梯度流入方向网络，但具体训练动态未验证。
6. **推理时 causal 模式的 randnumtest**：`model.py:239-248` 中逐帧循环 + 随机采样在推理时引入非确定性，且效率较低（O(T²)），实际部署需注意。
7. **无单元测试**：仓库无任何测试代码，改造后正确性需自行验证。
