# 频域压缩的时空联合注意力：突破感受野-效率权衡
> 技术可行性报告 · 2026-07-21 · idea: 频域压缩的时空联合注意力提升.md · ReAct 写作（边写边查证 papers/cards/codebases）

> 技术可行性报告 · 2026-07-21 · idea: 频域压缩的时空联合注意力提升.md · ReAct 写作（边写边查证 papers/cards/codebases）


> 技术可行性报告 · 2026-07-21

---

## 1. 背景与动机

### 1.1 问题陈述

单目视频 2D-to-3D 人体姿态提升（lifting）是计算机视觉中的核心问题：给定 2D 检测器输出的关节坐标序列 $P_{2D} \in \mathbb{R}^{T \times S \times 2}$（$T$ 帧、$S$ 关节），估计对应的 3D 姿态 $P_{3D} \in \mathbb{R}^{T \times S \times 3}$。该问题本质为病态逆问题——单目投影丢失深度信息，同一 2D 观测对应无穷多 3D 解（MHFormer [W4312249545]："单目视频2D-to-3D人体姿态提升是一个病态逆问题，因深度模糊和自遮挡存在多个可行解"）。时序上下文是消解深度歧义的关键手段：更长的输入序列通常带来更低的回归误差（STCFormer [W4386076485] Table 4：T=27→81→243 时 P1 从 44.1→42.0→41.0mm）。

然而，Transformer 时代 3D 姿态估计面临一个结构性瓶颈：**感受野与计算效率的根本矛盾**。对 $T$ 帧 $\times$ $S$ 关节做全联合时空自注意力的复杂度为 $O(T^2 S^2)$，当 $T=243, S=17$ 时 token 数达 4131，注意力矩阵约 $1.7 \times 10^7$ 个元素，训练与推理均不可承受。这迫使整个领域退守"空间注意力 + 时间注意力"的分解范式：

| 方法 | 范式 | T=243 时 FLOPs | MPJPE (mm) |
|------|------|----------------|------------|
| PoseFormer [W3136525061] | 串行空间→时间 | 1358M (T=81) | 44.3 (T=81) |
| MixSTE [W4312417903] | 交替 STB/TTB | 138,623M | 40.9 |
| STCFormer [W4386076485] | 并行轴分解 | 19,561M | 41.0 |
| PoseFormerV2 [W4386083126] | 频域压缩+串行 | 352M (f=n=9) | 46.0 (F=81) |

（数据来源：各论文原文 Table 4/5 及卡片 eval_setup 字段）

分解范式虽将复杂度降至 $O(T^2 S) + O(T S^2)$（STCFormer）或更低，但其提出者自己承认了根本缺陷：

> "两步分离式方法对运动模式学习不充分……仍是分解近似而非真正联合建模。" ——STCFormer [W4386076485]

> "实际运动中二者高度耦合（如手臂摆动同时涉及空间结构变化与时间动态），缺乏对耦合建模的探讨。" ——MixSTE [W4312417903]

**量化表现**：在 Human3.6M 标准协议下（S1/5/6/7/8 训练，S9/11 测试，CPN 2D 输入），当前最优分解范式方法 STCFormer-L 达到 P1=40.5mm、P2=31.8mm（T=243），MotionAGFormer-B 达到 P1=38.4mm。但这些方法的精度提升已趋于边际递减——从 2021 年 PoseFormer 的 44.3mm 到 2024 年 MotionAGFormer 的 38.4mm，三年仅改善 5.9mm（约 13%），且主要依赖工程技巧（双流融合、GCN 局部先验、预训练）而非架构范式的突破。

更值得关注的是逐动作误差分布。STCFormer [W4386076485] Table 1 显示，即使是最优方法，"Sitting Down"（56.8mm）和"Sitting"（52.5mm）等涉及复杂时空耦合的动作误差远高于"Walking"（26.2mm）等周期性动作。这恰恰是分解范式最无力的场景：坐下动作中髋关节的空间位移与膝关节的时间弯曲高度耦合，坐姿维持中躯干-四肢的空间构型与时间稳定性不可分离。分解范式在这些动作上的系统性高误差，正是其架构缺陷的直接体现。

此外，感受野扩大的边际收益也在递减。STCFormer Table 4 显示 T 从 27→81→243 时 P1 仅从 44.1→42.0→41.0mm，每增加 162 帧仅换来 1mm 改善。PoseFormerV2 的 Table 5 同样表明，在 F=27 时增加 DCT 系数数（n=1→3）带来 2.4mm 改善，但进一步增加帧数（f=1→3）仅带来 1.0mm。这暗示当前分解架构对长程信息的利用效率极低——不是信息不存在，而是分解范式无法有效整合跨时空的关联。

### 1.2 相关工作

按技术路线分组，仅引用 cards/ 中存在的论文：

**路线一：串行分解（空间→时间级联）**

PoseFormer [W3136525061]（ICCV 2021, 692 citations）首次将纯 Transformer 引入 2D-to-3D lifting。其架构由两个独立 Transformer 模块串联：空间 Transformer 以每帧各关节坐标为 token（$S$ 个 token），通过自注意力编码帧内关节间运动学关联；时序 Transformer 以各帧空间特征（展平后）为 token（$T$ 个 token），通过自注意力捕获整段序列的全局帧间依赖。回归头对时序输出做可学习加权平均后接 MLP 输出中心帧 3D 姿态。在 H3.6M 上达到 MPJPE 44.3mm / P-MPJPE 34.6mm（T=81, CPN），模型约 9.6M 参数，FLOPs 1358M（f=81）。其局限在于：FLOPs 随帧数平方增长（远超 TCN 的 33.87M），且将所有关节视为一个 token 送入时序 Transformer，忽略了不同关节运动轨迹的差异性。

PoseFormerV2 [W4386083126]（2023, 229 citations）在 PoseFormer 基础上引入 DCT 频域压缩。核心设计：对完整长序列（$F$ 帧）的每个关节轨迹做 DCT 变换，仅保留前 $n$ 个低频系数作为频域紧凑表征 $z_{Freq}$，与空间编码器输出的 $f$ 帧中心帧时域特征 $z_{Time}$ 拼接后送入时间编码器（重命名为 Time-Frequency Feature Fusion 模块）。同时将时间编码器中的普通 MLP 替换为 FreqMLP 以更好地融合两域特征。关键优势：$f$ 和 $n$ 固定后计算量 predetermined，可任意增大 $F$ 扩大感受野而不增算力。实验表明 4.6× 加速（$F/f=27$ 时），MPJPE 从 49.9mm 降至 46.0mm（$f=n=9, F=81$）。其 Table 6 的 Simple Baseline 实验证明纯频域输入（仅 DCT 系数）效果有限（49.7mm），必须与时域中心帧特征互补（47.1mm）。但该工作仍将融合后的特征送入串行时间编码器，未尝试全联合注意力。

StridedTransformer [W4225557002]（TMM 2022, 302 citations）以步幅卷积替换 FFN 中的全连接层，逐层压缩时间维度（如 27→9→3→1），以层次化方式融合全局上下文（自注意力）与局部上下文（步幅卷积）。训练采用 full-to-single 监督方案：VTE 输出在全序列尺度施加时间平滑约束，STE 输出在单帧尺度精化目标帧估计。MPJPE 43.7mm（T=351, CPN），参数量仅 4.23M（待验证，原论文 Table II）。但其步幅因子需根据感受野手动设计，且仅改进了 FFN 而未触及自注意力机制本身的效率问题。

**路线二：交替/并行分解**

MixSTE [W4312417903]（CVPR 2022, 410 citations）的核心观察是：不同身体关节在时间维度上的运动轨迹差异显著，应被独立建模。其设计包括：(1) 关节分离——在时序 Transformer 块中将每个关节作为独立 token 并行建模其时间运动轨迹；(2) 交替堆叠——空间 Transformer 块（STB）与时序 Transformer 块（TTB）交替堆叠 $d_l$ 层；(3) seq2seq 输出——将网络输出从中心帧扩展至整个输入序列；(4) 组合损失——WMPJPE + T-Loss（含时序一致性损失和速度误差）。在 H3.6M 上达到 MPJPE 40.9mm / P-MPJPE 32.6mm（T=243, CPN），但代价是 33.7M 参数和每帧 645M FLOPs（T=243 时总 FLOPs 高达 138,623M）。作者承认"实际运动中二者高度耦合（如手臂摆动同时涉及空间结构变化与时间动态），缺乏对耦合建模的探讨"，且长序列（T=300）性能反而下降。

STCFormer [W4386076485]（CVPR 2023, 200 citations）提出时空十字交叉注意力（STC）：将输入特征沿通道维度均分为两组，分别并行执行空间轴注意力（同帧内关节间）和时间轴注意力（同关节跨帧轨迹），再拼接后经 MLP 混合通道信息。复杂度降为 $O(T^2S) + O(TS^2)$。进一步设计结构增强位置编码（SPE）：part-aware embedding 为同属一个身体部位的关节赋予相同可学习向量，3×3 时空分组卷积捕获动态部位的局部运动模式。标准版 4.75M 参数达到 P1=42.0mm（T=81），Large 版 18.9M 参数达到 P1=40.5mm / P2=31.8mm（T=243）。消融实验（Table 5）显示：仅空间注意力 P1=275.5mm，仅时间注意力 P1=67.6mm，STC 并行 P1=57.0mm，加 SPE 后降至 44.1mm——证明并行建模两轴远优于单轴，且结构先验贡献显著。但作者明确指出"两步分离式方法对运动模式学习不充分……仍是分解近似而非真正联合建模"。

MotionBERT [W4390874423]（2023, 371 citations）采用双流时空 Transformer（DSTformer），两条分支分别以 S→T 和 T→S 顺序堆叠空间/时间多头自注意力模块，通过注意力回归器预测自适应权重进行逐元素融合。其核心贡献在于预训练-微调框架：以 2D-to-3D lifting 为代理任务，对 2D 骨架序列施加随机遮挡与噪声后恢复完整 3D 运动，同时引入野外 RGB 视频的加权 2D 重投影损失。模型 N=5 层，h=8 头，$C_f=C_e=512$。

**路线三：混合架构（Transformer + GCN）**

MotionAGFormer [W4394597906]（2024, 169 citations）提出 AGFormer 块，采用双流架构：Transformer 流通过空间/时间多头自注意力捕获全局关系，GCNFormer 流利用图卷积捕获局部时空依赖。空间 GCN 以人体骨架拓扑为邻接矩阵，时间 GCN 以 K-NN 相似度（$k=2$）确定连接。两流输出通过可学习的自适应融合聚合：$F^{(i)} = \alpha_{TF}^{(i)} \circ F_{TF}^{(i-1)} + \alpha_{GF}^{(i)} \circ F_{GF}^{(i-1)}$，权重由 softmax 归一化的线性变换产生。训练损失包含 3D 位置损失与速度平滑损失。MotionAGFormer-B 在 H3.6M 达 P1=38.4mm，MPI-INF-3DHP 达 16.2mm；参数量为此前 SOTA 的 1/4，计算效率高 3 倍。值得注意的是，该模型不使用时间位置编码（消融显示加上反而变差，Table 6：无时间编码 38.4mm vs 有时间编码 40.5mm），时序顺序完全依赖 GCNFormer 流保持。

GLA-GCN（cards/gla_gcn_global_local_adaptive_graph_convolutional_network）引入自适应 GCN 利用全局表征，以较轻内存负载实现与 Transformer 方法可比的精度，验证了图结构先验对 3D 姿态估计的持续价值。

**路线四：预训练增强与多假设**

P-STMO [W4312797994]（ECCV 2022, 179 citations）通过掩码姿态建模（MPM）自监督预训练：时间掩码率 90% + 空间掩码，让编码器学习 2D 时空依赖。微调阶段 STMO 由 SEM（轻量 MLP 建立帧内关节空间关系）、TEM（标准 Transformer 捕获非局部时间依赖）、MOFA（跨步 Transformer 多对一帧聚合器）三模块串联。MPJPE 42.8mm（T=243, CPN），约 6.7M 参数、868.5M FLOPs。其局限在于 TDS 在 243 帧时仅带来 0.2mm 增益，时间感受野增大存在收益递减。

MHFormer [W4312249545]（CVPR 2022, 426 citations）将 2D-to-3D 提升视为逆问题，提出三阶段多假设框架：MHG 用级联 Transformer 生成多层级特征作为多假设初始表示；SHR 通过多假设自注意力独立建模各假设时序依赖；CHI 通过多假设交叉注意力建模假设间交互。MPJPE 43.0mm（T=351），但参数量 18.92M、FLOPs 1.03G，且假设数固定为 3，最终仍输出确定性单一解。

**路线五：频域方法**

PoseFormerV2 [W4386083126] 是唯一将 DCT 频域表征引入 3D 姿态提升的工作。其核心发现：人体运动轨迹能量集中于低频分量，少量低频 DCT 系数即可编码长序列全局时序轮廓；高频分量主要对应检测噪声与抖动，丢弃后反而提升鲁棒性（噪声鲁棒性实验 Fig. 6：sigma 从 0 增到 10，PoseFormerV2 性能下降远小于 PoseFormerV1 和 MHFormer）。该工作还证明频域方法可泛化到 MixSTE 和 MHFormer（补充材料 Fig. 8：MixSTE-9×RF 在几乎不增 FLOPs 下获得显著精度提升）。但该工作仍将频域特征送入串行时间编码器，未尝试在压缩后的 token 上做全联合时空注意力——这正是本方案要填补的空白。

### 1.3 根本性分析

**信息论视角：分解范式的互信息损失**

设 $X_s$ 为空间随机变量（关节间关联）、$X_t$ 为时间随机变量（帧间动态）、$Y$ 为目标 3D 姿态。全联合注意力的建模能力上界由 $I(Y; X_s, X_t)$ 决定。分解范式将其近似为：

$$I_{factored}(Y; X_s, X_t) \approx I(Y; X_s) + I(Y; X_t | X_s)$$

或更粗糙的 $I(Y; X_s) + I(Y; X_t)$。当空间与时间存在显著交互信息（synergy）时：

$$I(Y; X_s, X_t) > I(Y; X_s) + I(Y; X_t)$$

差值 $\Delta = I(Y; X_s, X_t) - I(Y; X_s) - I(Y; X_t)$ 即为协同信息（synergy），对应"手臂摆动同时涉及空间结构变化与时间动态"这类耦合运动。分解范式在架构层面将 $\Delta$ 强制置零，这是其精度天花板的根本来源。STCFormer 试图通过通道拼接 + MLP 在层间逐步恢复交互信息，但 MLP 的逐 token 操作无法建模 token 间的二阶交互——这正是注意力机制的不可替代之处。

**几何视角：感受野-效率的 Pareto 前沿**

将现有方法绘制在（感受野, FLOPs）平面上，可观察到一条陡峭的 Pareto 前沿：

- PoseFormerV2：感受野 81 帧，0.35 GFLOPs——效率极高但精度受限（46.0mm）
- STCFormer：感受野 243 帧，19.6 GFLOPs——精度好但算力代价大
- MixSTE：感受野 243 帧，138.6 GFLOPs——精度与 STCFormer 相当但算力爆炸

没有任何方法能同时占据"大感受野 + 低算力 + 真正联合建模"三个目标。这不是工程优化的问题，而是 $O(T^2S^2)$ 复杂度墙造成的结构性空白。

**优化视角：频域压缩为何能打开缺口**

PoseFormerV2 的实验已给出关键证据（Table 5 [W4386083126]）：

| 帧数 f | DCT 系数数 n | 完整序列长 F | MFLOPs | MPJPE |
|--------|-------------|-------------|--------|-------|
| 1 | 1 | 27 | 39.2 | 51.1 |
| 1 | 3 | 27 | 77.2 | 48.7 |
| 3 | 1 | 27 | 79.4 | 50.1 |
| 3 | 3 | 27 | 117.3 | 47.9 |
| 9 | 9 | 27 | 351.7 | 47.6 |

仅 3 个 DCT 系数（n=3）即可将 27 帧序列的全局轮廓编码为 3 个 token，MPJPE 从 51.1 降至 48.7mm（↓2.4mm）。更关键的对比：增加 DCT 系数（n: 1→3）带来 2.4mm 改善，而增加时域帧数（f: 1→3）仅带来 1.0mm——频域全局信息的边际价值远高于时域局部信息。这意味着时间维的有效信息维度远小于 $T$——运动轨迹在频域是高度稀疏的。

形式化地，设 DCT 压缩比为 $\rho = T'/T$（保留 $T'$ 个低频系数），则全联合注意力的复杂度从 $O(T^2 S^2)$ 降至：

$$O(T'^2 S^2) = O(\rho^2 T^2 S^2)$$

当 $\rho = 1/9$（如 T=81 压缩到 T'=9）时，复杂度降低 81 倍。这使得在消费级 GPU 上首次实现真正的时空联合注意力成为可能——不是近似，不是分解，而是在压缩后的完整 token 集合 $(T' \times S)$ 上施加标准全注意力，让每个 token 同时 attend to 所有时间位置和所有空间位置。

**表征学习视角：DCT 基作为运动原语**

从表征学习的角度，DCT 基函数 $\cos\left(\frac{\pi}{2T}(2f-1)(i-1)\right)$ 构成一组正交完备基，其低频分量对应运动的"语义骨架"——整体位移方向、周期性步态节奏、肢体摆动的包络线。高频分量则对应检测器抖动、肌肉微颤、衣物飘动等对 3D 姿态估计无益甚至有害的信号。PoseFormerV2 的噪声鲁棒性实验（Fig. 6）直接验证了这一点：当 2D 检测噪声 $\sigma$ 从 0 增到 10 时，PoseFormerV2 的 MPJPE 退化曲线远平坦于 PoseFormerV1 和 MHFormer，因为 DCT 低频截断天然滤除了噪声所在的高频子空间。

这意味着 DCT 压缩不仅是计算 trick，更是一种有物理意义的特征选择：它保留了运动的结构化信息，丢弃了非结构化的噪声。在此压缩表征上施加全联合注意力，等价于让模型在"运动语义空间"中建模时空耦合——这比在原始帧空间中建模更高效、更鲁棒。

**与 ViT 的类比：token 数的可行性论证**

Vision Transformer（ViT）在 224×224 图像上使用 16×16 patch，产生 196 个 token，标准全注意力即可胜任。本方案在 $T'=9, S=17$ 时产生 $N=153$ 个 token，甚至少于 ViT 的 196 个。在 $T'=27, S=17$ 时 $N=459$，仍在标准 Transformer 的舒适区内（BERT 处理 512 token，GPT 处理 1024+ token）。因此，全联合注意力在本方案的 token 规模下不存在优化困难或注意力退化问题——这是一个已被大规模验证的成熟 regime。

---

## 2. 方法

本方案将 idea 拆解为三个互补贡献：

### 2.1 Contribution 1：DCT 频域压缩前端——从 $T$ 到 $T'$ 的信息保持降维

**设计动机**

PoseFormerV2 [W4386083126] 已证明人体运动轨迹能量集中于低频分量。其 DCT 公式为：对第 $j$ 个关节的轨迹 $\hat{x}_j \in \mathbb{R}^F$，第 $i$ 个 DCT 系数为

$$C_{j,i} = \sqrt{\frac{2}{F}} \sum_{f=1}^{F} x_{j,f} \frac{1}{\sqrt{1+\delta_{i1}}} \cos\left(\frac{\pi}{2F}(2f-1)(i-1)\right)$$

**代码级验证**（来源：codebases/PoseFormerV2.md）：PoseFormerV2 的 DCT 实现位于 `common/model_poseformer.py:218`：

```python
x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
```

输入 shape 为 `(b, f, p, 2)`，permute 后变为 `(b, p, 2, f)`，沿最后一维（时间轴）做 type-II DCT，然后直接切片 `[:num_coeff_kept]` 只保留前 $n$ 个低频系数（含 DC），丢弃所有高频。系数数量由命令行参数 `-coeff-kept` 控制（`common/arguments.py:50`），无默认值时回退到 `num_frame_kept`（`common/model_poseformer.py:160`）。截断后 reshape 为 `(b, num_coeff_kept, p*2)` 经线性层 `self.Freq_embedding = nn.Linear(in_chans*num_joints, embed_dim)`（`common/model_poseformer.py:164`）映射到嵌入空间。README 示例配置为 `-frame 27 -frame-kept 3 -coeff-kept 3`，即 27 帧输入只保留 3 个 DCT 系数；预训练模型表中 $n$ 取值覆盖 1, 3, 9, 27。

低频系数（$i$ 小）编码序列粗略轮廓，高频系数（$i$ 大）编码抖动与噪声。PoseFormerV2 的 Table 6 进一步验证：纯频域输入（仅 DCT 系数）在 F=81 时 MPJPE 为 49.7mm，而加入 3 帧中心帧时域特征后降至 47.1mm（↓2.6mm），说明时频互补是必要的。

**技术细节**

本方案继承 PoseFormerV2 的 DCT 前端设计，但改变其下游用途：

1. **输入**：完整 2D 关节序列 $P_{2D} \in \mathbb{R}^{T \times S \times 2}$，$T$ 为完整序列长度（感受野），$S=17$ 为关节数。
2. **DCT 变换**：对每个关节的 $x$、$y$ 坐标轨迹独立做 1D-DCT，得到 $S \times 2$ 条轨迹各 $T$ 个系数。
3. **低频截断**：仅保留前 $T'$ 个低频系数（$T' \ll T$），形成频域紧凑表征 $Z_{freq} \in \mathbb{R}^{T' \times S \times 2}$。
4. **线性投影**：将 $Z_{freq}$ 通过可学习线性层映射到 $d$ 维特征空间：$E_{freq} = Z_{freq} W_{proj} + b \in \mathbb{R}^{T' \times S \times d}$。

**与 PoseFormerV2 的关键区别**：PoseFormerV2 将 DCT 系数作为辅助特征拼接到时间编码器输入中，时间编码器本身仍是串行分解的（空间→时间）。具体地，其 `MixedBlock`（`common/model_poseformer.py:108-129`）将拼接后的 token 序列硬切为前后两半：前半（频域 token）走标准 Mlp，后半（空间/时域 token）走 FreqMlp（`common/model_poseformer.py:38-57`，对 token 序列做 DCT→MLP→IDCT），最终两路分别用 1×1 Conv 加权平均压缩为 1 个 token 再拼接输出（`common/model_poseformer.py:238`）。整个流程中，注意力仅在拼接后的 token 序列上做标准 self-attention（`self.attn(self.norm1(x))`），但 token 的语义被硬切分割，频域与时域特征的交互受限于 MLP 的逐 token 操作。本方案将 DCT 系数视为压缩后的时间 token，直接在 $(T' \times S)$ 个 token 上做全联合注意力——频域压缩不是辅助手段，而是使能全联合注意力的核心机制。

**工程注意事项**（来源：codebases/PoseFormerV2.md 风险与未知）：PoseFormerV2 的 `MixedBlock` 中 `f//2` 硬切假设 freq token 数 == spatial token 数（即 `num_coeff_kept == num_frame_kept`），若两者不等则前后半划分将错位，代码中未见断言保护。本方案不继承此 MixedBlock 设计，而是将 DCT 压缩后的 $(T' \times S)$ token 直接展平送入标准 Transformer，规避了此隐患。此外，PoseFormerV2 依赖第三方库 `torch_dct==0.1.6`（内部实现为矩阵乘法形式的 DCT-II），本方案可用 PyTorch 原生 `torch.fft` 或 `scipy.fft.dct` 替代，避免外部依赖。

**超参设定**（基于 PoseFormerV2 消融）：
- 典型设置：$T=81, T'=9$（压缩比 $\rho=1/9$）或 $T=243, T'=27$（$\rho=1/9$）
- PoseFormerV2 发现 $n=f$（DCT 系数数 = 空间编码帧数）时速度-精度权衡最优

### 2.2 Contribution 2：全联合时空自注意力——首次实现真正的时空耦合建模

**设计动机**

分解范式的根本缺陷在于：空间注意力只看同帧内关节、时间注意力只看同关节跨帧轨迹，二者永远不在同一个注意力矩阵中相遇。STCFormer [W4386076485] 的并行轴分解虽优于串行，但作者明确承认"仍是分解近似而非真正联合建模"。全联合注意力让每个 token $(t', s)$（第 $t'$ 个时间系数、第 $s$ 个关节）能直接 attend to 所有其他 $(t'', s')$，从而在单次注意力计算中同时捕获：
- 同帧不同关节的空间关联（$t'=t'', s \neq s'$）
- 同关节不同时间的时序动态（$t' \neq t'', s=s'$）
- **跨帧跨关节的时空耦合**（$t' \neq t'', s \neq s'$）——这是分解范式完全无法建模的

**技术细节**

将 DCT 压缩后的特征 $E_{freq} \in \mathbb{R}^{T' \times S \times d}$ 展平为 $N = T' \times S$ 个 token 的序列 $X \in \mathbb{R}^{N \times d}$，施加标准多头自注意力：

$$\text{MHSA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

其中 $Q, K, V \in \mathbb{R}^{N \times d_k}$，$N = T' \times S$。

**复杂度分析**：

| 方法 | 注意力复杂度 | T=243, S=17, T'=27 |
|------|-------------|---------------------|
| 全联合（无压缩） | $O(T^2 S^2)$ | $O(4131^2) \approx 17.1M$ |
| STCFormer 轴分解 | $O(T^2S + TS^2)$ | $O(243^2 \times 17 + 243 \times 17^2) \approx 1.07M$ |
| **本方案（压缩后全联合）** | $O(T'^2 S^2)$ | $O(459^2) \approx 210K$ |

本方案的注意力计算量仅为无压缩全联合的 1/81，甚至低于 STCFormer 的轴分解（210K vs 1.07M），同时保留了完整的时空交互能力。

**代码级对比**（来源：codebases/STCFormer.md）：STCFormer 的轴分解实现位于 `model/stcformer.py:74-81`，QKV 投影后沿通道维度等分：`qkv_s, qkv_t = qkv.chunk(2, 4)`，空间组在关节维做注意力（`model/stcformer.py:84-93`，`att_s = (q_s @ k_s) * self.scale`，shape 为 `(b*h*t, s, s)`），时间组在帧维做注意力（`model/stcformer.py:87-94`，`att_t = (q_t @ k_t) * self.scale`，shape 为 `(b*h*s, t, t)`）。两组结果拼接后经投影和残差连接（`model/stcformer.py:124-130`）。注意其通道固定 50% 划分（`chunk(2, 4)`），无自适应机制。本方案不做任何通道分割或轴分解，而是在展平后的 $(T' \times S)$ token 上施加单一标准 MHSA，让注意力矩阵自然覆盖所有时空位置对。以默认 STCFormer 配置（T=27, S=17, head=8, d_hid=256，来自 `common/opt.py:13-15`）为例，其单层注意力需计算 $T$ 个 $S \times S$ 矩阵加 $S$ 个 $T \times T$ 矩阵；本方案（T'=3, S=17）仅需 1 个 $51 \times 51$ 矩阵，计算量降低约 10×（与 codebases/STCFormer.md 中的分析一致：全局 $(27 \times 17)^2 = 210,849$；拆分 $27 \times 17^2 + 17 \times 27^2 = 20,196$；本方案 $(3 \times 17)^2 = 2,601$）。

**位置编码**：采用可学习时空位置编码。每个 token $(t', s)$ 的位置编码为：

$$PE(t', s) = PE_{time}(t') + PE_{space}(s)$$

其中 $PE_{time} \in \mathbb{R}^{T' \times d}$ 编码频域位置（第几个 DCT 系数），$PE_{space} \in \mathbb{R}^{S \times d}$ 编码关节身份。借鉴 STCFormer 的 SPE 设计思路，可进一步引入 part-aware embedding 为同属一个身体部位的关节赋予相同可学习向量。

**网络结构**：堆叠 $L$ 个 Joint Spatio-Temporal Transformer（JSTT）块，每块包含：
1. LayerNorm → MHSA（全联合） → 残差连接
2. LayerNorm → FFN（MLP, 扩展比 $\alpha=4$） → 残差连接

**伪代码**：

```python
class JSTTBlock(nn.Module):
    def __init__(self, d, h, alpha=4):
        self.norm1 = LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, h)  # 全联合，无轴分解
        self.norm2 = LayerNorm(d)
        self.ffn = MLP(d, d*alpha, d)

    def forward(self, x):  # x: (B, T'*S, d)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FreqJointAttention(nn.Module):
    def __init__(self, T, T_prime, S=17, d=256, L=6, h=8):
        self.dct_compress = DCTCompress(T, T_prime)  # 保留前T'个DCT系数
        self.proj = nn.Linear(2, d)
        self.pe_time = nn.Parameter(torch.randn(T_prime, d))
        self.pe_space = nn.Parameter(torch.randn(S, d))
        self.blocks = nn.ModuleList([JSTTBlock(d, h) for _ in range(L)])
        self.head = nn.Linear(d, 3)  # 回归3D坐标

    def forward(self, p2d):  # p2d: (B, T, S, 2)
        z = self.dct_compress(p2d)          # (B, T', S, 2)
        x = self.proj(z)                     # (B, T', S, d)
        x = x + self.pe_time[:, None, :] + self.pe_space[None, :, :]
        x = x.reshape(B, T_prime * S, d)    # 展平为token序列
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(B, T_prime, S, d)
        out = self.head(x)                   # (B, T', S, 3)
        return out  # 取中心时间系数对应的3D姿态
```

**训练策略与损失函数**

训练损失采用领域标准配置，并借鉴 MotionAGFormer 的速度平滑项：

$$\mathcal{L} = \mathcal{L}_{3D} + \lambda_{\Delta P} \mathcal{L}_{\Delta P}$$

其中位置损失为 MPJPE：

$$\mathcal{L}_{3D} = \sum_{t=1}^{T'} \sum_{j=1}^{S} \|\hat{P}_{t,j} - P_{t,j}\|_2$$

速度平滑损失（借鉴 MotionAGFormer [W4394597906] Eq. 2）：

$$\mathcal{L}_{\Delta P} = \sum_{t=2}^{T'} \sum_{j=1}^{S} \|\Delta\hat{P}_{t,j} - \Delta P_{t,j}\|_2$$

其中 $\Delta\hat{P}_t = \hat{P}_t - \hat{P}_{t-1}$。该损失鼓励预测序列的时间平滑性，抑制 DCT 低频截断可能引入的过平滑伪影。

优化器配置（基于 PoseFormerV2 验证的最优设置）：
- AdamW，初始学习率 $8 \times 10^{-4}$，权重衰减 0.1
- 指数学习率衰减，衰减因子 0.99/epoch
- 训练 80 epochs，batch size 1024
- 数据增强：水平翻转（关节对 [4,5,6,11,12,13] ↔ [1,2,3,14,15,16]，遵循 H36M 17 关节标准顺序）

**输出策略**：模型输出 $(B, T', S, 3)$ 的 3D 姿态序列。推理时取中心时间系数（$t' = T'/2$）对应的 3D 姿态作为最终预测，与 PoseFormer/MixSTE 的中心帧预测范式一致。也可借鉴 MixSTE 的 seq2seq 策略输出完整序列以减少推理冗余。

### 2.3 Contribution 3：并联 GCNFormer 局部先验流——全局-局部互补

**设计动机**

MotionAGFormer [W4394597906] 的实验表明：纯全局注意力"不能精确捕获局部依赖"，而 GCN 以人体骨架拓扑为邻接矩阵天然编码局部结构先验。其消融实验（Table 7）显示，Transformer 流 + GCNFormer 流的组合显著优于单一流。本方案在 JSTT 主干之外并联一个轻量 GCNFormer 流，为全联合注意力提供局部骨架先验补充。

**技术细节**

借鉴 MotionAGFormer 的 AGFormer 设计：

1. **空间 GCN**：以人体骨架拓扑（17 关节、16 条边）为邻接矩阵 $A$，对每帧（每个时间系数）的关节特征做图卷积：

$$\text{GCN}(F) = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} F W\right)$$

其中 $\tilde{A} = A + I$（加自连接），$\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$。

2. **时间 GCN**：以 K-NN 相似度确定时间维连接（$k=2$，同 MotionAGFormer），对每个关节跨时间系数的特征做图卷积。

3. **自适应融合**：两流输出通过可学习权重聚合：

$$F^{(i)} = \alpha_{TF}^{(i)} \circ F_{TF}^{(i-1)} + \alpha_{GF}^{(i)} \circ F_{GF}^{(i-1)}$$

$$[\alpha_{TF}, \alpha_{GF}] = \text{softmax}(W \cdot [F_{TF}; F_{GF}])$$

**与现有系统的衔接**：

- DCT 前端直接复用 PoseFormerV2 [W4386083126] 的实现（代码开源 https://github.com/QitaoZhao/PoseFormerV2），其 DCT 变换为逐关节逐坐标的 1D-DCT，可用 `torch.fft` 或 `scipy.fft.dct` 实现。
- GCNFormer 流复用 MotionAGFormer [W4394597906] 的 GCN 模块设计（代码开源 https://github.com/TaatiTeam/MotionAGFormer），邻接矩阵与 K-NN 图构建逻辑可直接迁移。
- 数据管线（Human3.6M + CPN 2D 检测）为领域标准配置，所有基线共享。

**代码级衔接**（来源：codebases/PoseFormerV2.md、codebases/STCFormer.md）：

- **DCT 前端复用**：PoseFormerV2 的 DCT 变换核心仅一行（`common/model_poseformer.py:218`），截断策略为简单切片 `[:num_coeff_kept]`。本方案可直接复用此逻辑，仅需将输出从 `(b, num_coeff_kept, p*2)` 的扁平向量改为保留关节维度的 `(b, T', S, 2)` 张量，以便后续展平为 $(T' \times S)$ 个独立 token。具体修改点：将 `common/model_poseformer.py:219-220` 的 `view(b, num_coeff_kept, -1)` 替换为 `reshape(b, T', S, 2)`。
- **位置编码参考**：STCFormer 的身体部位分组嵌入（`model/stcformer.py:59-60`，5 组对应 17 关节：`[0,1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4]`）可作为本方案空间位置编码 $PE_{space}$ 的初始化参考。但 STCFormer 将 `self.part` 硬编码为 `.cuda()` 张量，本方案应改用 `register_buffer` 以支持多设备。
- **环境依赖**：PoseFormerV2 要求 Python 3.9 + PyTorch 1.13.0+cu117 + `torch-dct==0.1.6` + `einops==0.7.0` + `timm==0.9.12`；STCFormer 无 `requirements.txt`，依赖 PyTorch + einops + timm（DropPath）+ scipy。本方案仅需 PyTorch ≥ 1.10（内置 `nn.MultiheadAttention`）+ scipy（DCT），不依赖 torch-dct 或 timm。
- **数据格式统一**：PoseFormerV2 数据按 VideoPose3D 方式组织（`data/data_2d_h36m_cpn_ft_h36m_dbb.npz` + `data/data_3d_h36m.npz`），STCFormer 使用相同格式放入 `dataset/` 目录。本方案采用同一数据格式，确保与两个基线的公平对比。
- **待补充**：PoseFormer、MixSTE、MotionAGFormer 三个仓库尚无 repo 卡，其代码级细节（GCN 模块接口、数据加载器格式）待后续 study_codebase 查证。

---

## 3. 实验计划

### 3.1 评估指标

| 指标 | 含义 | 当前最优值 | 目标值 | 改进幅度（保守–乐观） | 依据 |
|------|------|-----------|--------|----------------------|------|
| MPJPE (P1, mm) | 平均关节位置误差 | 38.4 (MotionAGFormer-B) | ≤37.5 | ↓1.0–2.0mm (2.6%–5.2%) | 保守端：报告§1.1所述"1–2mm架构增益"下界；乐观端：消融A0→A1预期↓4mm中扣除GCN流贡献后全联合注意力净增益约2mm |
| P-MPJPE (P2, mm) | Procrustes 对齐后误差 | 31.8 (STCFormer-L) | ≤31.0 | ↓0.5–1.3mm (1.6%–4.1%) | 保守端：P2/P1改善比例参照MotionAGFormer（P1↓2.1mm对应P2↓0.7mm，约1:0.33）；乐观端：联合建模对刚性对齐后残余误差的额外压缩 |
| GFLOPs | 计算量 | 0.35 (PoseFormerV2, F=81) | ≤2.0 (T'=9, S=17, L=6) | 1.4–5.7× 最强效率基线 | §4.4估算0.5–1.0 GFLOPs；仍与分解范式同量级，不随T²爆炸 |
| 参数量 (M) | 模型规模 | 4.75 (STCFormer) | ≤10 | ≤2.1× 最轻基线 | JSTT 6层(d=256)约5.3M + GCN流约1.5M（MotionAGFormer GCN参数占比≈1/4推算） |
| FPS | 推理速度 | ~269 (PoseFormer, GTX 2080Ti) | ≥100 (RTX 3090) | 实时（≥100 FPS） | N=153 tokens远小于PoseFormer的T×S=1377 tokens（T=81），推理瓶颈不在注意力 |

目标设定依据：MotionAGFormer-B 的 38.4mm 是当前 H3.6M CPN 输入下的最优结果（cards/motionagformer），本方案通过真正联合建模预期获得 1–2mm 的架构增益，同时保持 GFLOPs 在分解范式同量级（不随 $T^2$ 爆炸）。改进幅度区间说明：保守端对应"全联合注意力在 N=153 时仅带来边际增益"的情形（§4.6 决策路径 B），乐观端对应消融矩阵 A0→A1→A2 逐级叠加的完整收益。

### 3.2 消融矩阵

| 实验编号 | 配置 | 目的 | 预期 |
|---------|------|------|------|
| A0 | PoseFormerV2 原始（DCT + 串行时间编码器） | 基线复现 | ~46.0mm |
| A1 | DCT 压缩 + 全联合注意力（无 GCN 流） | 验证 C1+C2 | ≤42mm |
| A2 | DCT 压缩 + 全联合注意力 + GCNFormer 流 | 完整方案 | ≤38mm |
| A3 | 无 DCT 压缩 + 全联合注意力（T'=T=9） | 验证压缩的必要性 | 精度≈A1 但无法扩大感受野 |
| A4 | DCT 压缩 + STC 轴分解注意力（替换全联合） | 验证全联合 vs 分解 | 精度介于 A0 和 A1 之间 |
| A5 | 不同压缩比：T'=3/9/27（T=243） | 压缩比敏感性 | T'=9 为最优点 |
| A6 | 不同层数 L=4/6/8 | 深度敏感性 | L=6 为甜点 |
| **Oracle** | GT 2D 输入 + 完整方案 | 上界（排除 2D 检测噪声） | ≤25mm |
| **Negative** | 随机打乱 DCT 系数顺序 | 验证频域结构非偶然 | 退化至 >50mm |
| **Negative** | 用随机正交变换替换 DCT | 验证 DCT 的能量集中特性 | 退化至 >48mm |

**消融逻辑说明**：

消融矩阵的设计遵循"逐层剥离"原则，每个实验仅改变一个变量：

- **A0→A1**：保持 DCT 前端不变，将下游从串行时间编码器替换为全联合注意力。若 A1 显著优于 A0（预期 ↓4mm），则证明全联合注意力的增益独立于频域压缩存在。
- **A1→A2**：在全联合注意力基础上加入 GCNFormer 流。若 A2 优于 A1（预期 ↓1-2mm），则证明局部先验与全局联合建模互补。
- **A1→A3**：去掉 DCT 压缩，直接在原始 9 帧 × 17 关节 = 153 token 上做全联合注意力。A3 与 A1 精度应接近（因为 token 数相同），但 A3 的感受野仅 9 帧，无法扩大到 81/243 帧——这证明 DCT 压缩的价值不在于提升精度，而在于使能感受野扩展。
- **A1→A4**：将全联合注意力替换为 STC 轴分解。若 A1 优于 A4，则直接证明全联合 > 分解（在相同 token 数下）。
- **Oracle**：使用 GT 2D 输入排除 2D 检测噪声。Oracle 与 CPN 结果的差值（预期 ~13mm，参考 PoseFormer 的 31.3 vs 44.3）量化了 2D 检测器对精度的限制，也界定了本方案在当前管线下的改进空间。
- **Negative controls**：随机打乱 DCT 系数顺序破坏了频率从低到高的语义排列；随机正交变换不具备 DCT 的能量集中特性（低频系数不再承载主要信息）。两者均应导致显著退化，验证方案的有效性来自 DCT 的物理意义而非任何线性降维的通用效果。

### 3.3 基线方法

| 基线 | 来源 | 范式 | 预期 MPJPE |
|------|------|------|-----------|
| PoseFormer [W3136525061] | 串行分解 | 空间→时间 | 44.3mm (T=81) |
| PoseFormerV2 [W4386083126] | 频域+串行 | DCT+时间融合 | 46.0mm (F=81) |
| MixSTE [W4312417903] | 交替分解 | STB/TTB 交替 | 40.9mm (T=243) |
| STCFormer [W4386076485] | 并行轴分解 | 通道分组并行 | 40.5mm (T=243, L) |
| MotionAGFormer [W4394597906] | 混合双流 | Transformer+GCN | 38.4mm |
| MHFormer [W4312249545] | 多假设 | 三假设交互 | 43.0mm (T=351) |

所有基线使用相同数据划分（S1/5/6/7/8 训练，S9/11 测试）和相同 2D 输入（CPN 检测）。

### 3.4 数据集要求与预处理

**主数据集：Human3.6M**
- 规模：3.6M 视频帧，11 受试者，15 种动作，4 个固定相机
- 划分：S1/5/6/7/8 训练，S9/11 测试（领域标准协议）
- 2D 输入：CPN 检测器预提取的 2D 关节坐标（`data_2d_h36m_cpn_ft_h36m_dbb.npz`）
- 3D 标注：`data_3d_h36m.npz`，根关节（hip）对齐
- 关节数：17（标准 H36M 骨架）
- 帧率：50Hz
- 输入长度：T=81（主实验）、T=243（长序列实验）
- 数据增强：水平翻转（所有基线标准配置）

**辅助数据集：MPI-INF-3DHP**（泛化验证）
- 场景：绿幕/非绿幕/户外
- 输入：GT 2D（遵循领域惯例）
- 指标：MPJPE、PCK@150mm、AUC

**预处理流程**：
1. 从完整视频提取以目标帧为中心的 T 帧窗口（不足则零填充）
2. 2D 坐标经相机参数归一化（焦距、主点）
3. 3D 标注做根关节对齐（减去 hip 坐标，即 `pos_3d[:, 1:] -= pos_3d[:, :1]`）
4. DCT 变换在归一化后的 2D 坐标上执行：对每个关节的 x、y 轨迹独立做 1D Type-II DCT
5. 低频截断后保留前 T' 个系数，高频置零（等价于低通滤波）
6. 训练时随机水平翻转（概率 0.5），翻转时交换左右关节索引

### 3.5 评估协议

**Protocol 1 (P1)**：MPJPE——估计姿态与 GT 在根关节对齐后计算平均欧氏距离（mm）。

**Protocol 2 (P2)**：P-MPJPE——估计姿态与 GT 经 Procrustes 刚性变换对齐后计算平均欧氏距离（mm）。

**效率指标**：
- GFLOPs：单次前向传播的浮点运算量（含 DCT 变换）
- 参数量：可学习参数总数（M）
- FPS：单卡 RTX 3090 上 batch=1 的推理帧率

**统计显著性**：每个配置训练 3 次（不同随机种子），报告均值 ± 标准差。

**逐动作报告**：除全局平均 MPJPE 外，报告 15 类动作的逐类误差（Directions, Discussion, Eating, Greeting, Phoning, Photo, Posing, Purchases, Sitting, SittingDown, Smoking, Waiting, WalkDog, Walking, WalkTogether），以揭示方法在不同时空耦合程度动作上的差异化表现。重点关注高耦合动作（SittingDown、Smoking、Purchases）的改善幅度——这些是分解范式系统性薄弱的动作类型。

**公平性控制**：
- 所有方法使用相同 2D 输入（CPN 检测），排除 2D 检测器差异
- 所有方法使用相同数据划分（S1/5/6/7/8 训练，S9/11 测试）
- 参数量控制在同一量级（≤20M），排除模型容量差异
- 输入序列长度对齐：主实验统一 T=81（或 F=81），长序列实验统一 T=243
- 推理效率在相同硬件（RTX 3090）上测量，batch=1
- 不使用后处理模块（如 STCFormer 的 post-processing），确保比较的是纯模型能力

### 3.6 计算资源估算表

| 阶段 | 硬件 | 预计时长 | 备注 |
|------|------|---------|------|
| 数据准备（H3.6M 下载+预处理） | CPU | 2-4h | 含 CPN 2D 特征下载 |
| 基线复现（PoseFormerV2） | 1× RTX 3090 | 4-6h | 80 epochs, AdamW |
| 基线复现（STCFormer） | 1× RTX 3090 | 3-5h | 20 epochs, Adam |
| 基线复现（MixSTE） | 1× RTX 3090 | 8-12h | FLOPs 高，训练慢 |
| 本方案训练（L=6, T'=9） | 1× RTX 3090 | 6-10h | 80 epochs 估计 |
| 消融实验（A0-A6, 3 seeds） | 1× RTX 3090 | 40-60h | 7 配置 × 3 seeds |
| 总计 | 1× RTX 3090 | ~80-120h | 约 4-5 天连续运行 |

依据：PoseFormerV2 在单卡 RTX 3090 上训练 80 epochs（cards/poseformerv2 eval_setup）；STCFormer 在单卡 GTX 2080Ti 上训练 20 epochs（cards/stcformer eval_setup）。本方案 token 数 $T' \times S = 9 \times 17 = 153$（远小于 MixSTE 的 $T \times S = 243 \times 17 = 4131$），单 epoch 训练时间应短于 MixSTE。

---

## 4. 可行性评估

### 4.1 实现复杂度

**核心改动量**：

| 组件 | 来源 | 改动程度 | 说明 |
|------|------|---------|------|
| DCT 前端 | PoseFormerV2 | 低（复用） | 直接迁移 DCT 变换与低频截断逻辑 |
| 全联合注意力 | 新写 | 中 | 标准 MHSA，无轴分解，PyTorch 原生实现 |
| GCNFormer 流 | MotionAGFormer | 低（复用） | 空间 GCN + 时间 K-NN GCN |
| 自适应融合 | MotionAGFormer | 低（复用） | softmax 加权逐元素相乘 |
| 数据管线 | 领域标准 | 低 | H3.6M + CPN，所有基线共享 |
| 训练脚本 | 新写/改造 | 中 | 统一训练循环、评测、日志 |

**与更轻替代路线对比**（复杂度以最轻路线"交叉注意力改良"为 1× 基准）：

| 替代路线 | 实现复杂度（×基准） | 预期收益 | 风险 |
|---------|-------------------|---------|------|
| 仅在 PoseFormerV2 时间编码器中加交叉注意力 | **1×**（~100 行新代码，1 个新模块，1 个仓库） | 有限（仍是分解） | 低 |
| 本方案（DCT + 全联合 + GCN） | **~5×**（500–800 行，4 个新/迁移模块，3 个仓库对接） | 架构级突破 | 中 |
| 端到端 2D/3D 联合训练 | **~15×**（需 2D 检测器反向梯度管线 + 多阶段训练 + 显存翻倍，估计 2000+ 行） | 消除级联误差 | 高（需 2D 检测器梯度） |
| 扩散模型多假设（DiffPose 路线） | **~12×**（需噪声调度 + 多步采样 + 多假设聚合，估计 1500+ 行，推理慢 10–50×） | 不确定性量化 | 高（推理慢） |

量化依据：以最轻路线（PoseFormerV2 加交叉注意力）的新代码量（~100 行：1 个 CrossAttention 类 + forward 修改）、新模块数（1）和仓库依赖数（1）为基准 1×。本方案 500–800 行（§4.1 正文估计）、4 个模块（DCTCompress + JSTT + GCNFormer + AdaptiveFusion）、3 个仓库对接（PoseFormerV2 + MotionAGFormer + STCFormer 数据格式），综合代码量/模块数/集成点取 ~5×。端到端路线和 DiffPose 路线的倍数基于所需新基础设施（梯度管线/扩散调度）的代码量级估计。

本方案处于"中等实现复杂度、高架构收益"的甜点位置。核心创新（全联合注意力）本身是标准 Transformer 的直接应用，无需发明新算子；复杂度降低完全由 DCT 前端提供，而 DCT 前端已有成熟实现。

**具体工程步骤**：

1. 从 PoseFormerV2 仓库提取 DCT 变换模块（逐关节逐坐标 1D-DCT + 低频截断），封装为独立 `DCTCompress` 类。
2. 实现标准 `nn.MultiheadAttention`（PyTorch 内置）作为全联合注意力核心，输入为展平后的 $(T' \times S)$ token 序列。
3. 从 MotionAGFormer 仓库迁移 GCN 模块（空间邻接矩阵 + 时间 K-NN 图构建 + 图卷积层），适配为与 JSTT 块并联的分支。
4. 实现自适应融合层（线性变换 + softmax + 逐元素加权），连接两流输出。
5. 统一数据加载管线：复用 MHFormer 仓库的 `common/load_data_hm36.py` 格式（`data_2d_h36m_cpn_ft_h36m_dbb.npz` + `data_3d_h36m.npz`），确保与所有基线数据一致。
6. 编写训练/评测脚本，集成 MPJPE/P-MPJPE 计算（参考 MHFormer 的 `common/utils.py` 实现）。

上述步骤中，步骤 1/3/5/6 均为已有代码的迁移与适配，步骤 2 为 PyTorch 标准组件，仅步骤 4 需要少量新代码（约 20 行）。总体工程量约 500-800 行 Python 代码。

**代码级风险点**（来源：codebases/PoseFormerV2.md、codebases/STCFormer.md）：

1. **PoseFormerV2 的 DCT 库依赖**：原仓库使用 `torch_dct==0.1.6`，其内部实现为矩阵乘法形式的 DCT-II，未验证与 `scipy.fft.dct` 的数值一致性。本方案若改用 scipy/PyTorch 原生实现，需先做数值对齐测试（预期差异 < 1e-6）。
2. **STCFormer 的 `.cuda()` 硬编码**：`model/stcformer.py:60` 的 `self.part = torch.tensor([...]).long().cuda()` 在 CPU 环境会报错。复现 STCFormer 基线时需修改为 `register_buffer`。
3. **STCFormer 的 stride_num 字典不完整**：`common/opt.py:72-78` 仅覆盖 9/27/81/243/351 帧，其他帧数会直接 `exit()`。若需非标准帧数对比，需手动扩展。
4. **PoseFormerV2 预训练权重托管在 Google Drive**：无自动下载脚本，无法确认链接长期有效性。但本方案仅需其 DCT 逻辑代码，不依赖预训练权重。
5. **STCFormer 243 帧模型不可用**：作者声明该模型因版权存于公司服务器未公开发布。T=243 的基线对比需自行训练。
6. **STCFormer 的 DropPath(0.5) 异常大**：注意力内 drop 率 0.5 而 FFN 为 0.0（`model/stcformer.py:66,142`），未见消融说明。复现时可能需要调整。

### 4.2 外部依赖风险表

| 依赖项 | 风险等级 | 说明 | 缓解措施 |
|--------|---------|------|---------|
| Human3.6M 数据集 | 低 | 公开数据集，领域标准 | 多镜像源下载 |
| CPN 2D 检测特征 | 低 | 预提取 .npz 文件广泛可得 | 可从 MHFormer/PoseFormerV2 仓库获取 |
| PoseFormerV2 代码 | 低 | 开源（GitHub），仅需 `common/model_poseformer.py:218` 的 DCT 逻辑 | 可独立实现，不依赖 `torch-dct` 库 |
| STCFormer 代码 | 中 | 开源但无 requirements.txt，243 帧权重未公开 | 自行训练；修复 `.cuda()` 硬编码 |
| MotionAGFormer 代码 | 低 | 开源（GitHub） | 仅需 GCN 模块，可独立实现 |
| PyTorch ≥ 1.10 | 低 | 标准框架，内置 `nn.MultiheadAttention` | 无特殊算子依赖 |
| scipy（DCT 实现） | 低 | 标准科学计算库 | 替代 `torch-dct==0.1.6`，需数值对齐验证 |
| 单卡 RTX 3090 (24GB) | 低 | N=153 tokens 显存需求远低于 MixSTE（4131 tokens） | T'=9, S=17 时注意力矩阵仅 153×153 |
| 基线仓库统一数据格式 | 中 | 5 个仓库数据接口不完全一致 | 统一为 VideoPose3D 格式（`data_2d_h36m_cpn_ft_h36m_dbb.npz`） |

### 4.3 错误传播风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| DCT 压缩丢失快速运动高频信息 | 中 | 快速动作（如跑步）精度下降 | 保留时域中心帧特征作为补充（PoseFormerV2 Table 6 验证）；消融 A5 确定最优 T' |
| 全联合注意力在 N=153 时退化 | 低 | 精度不及分解范式 | N=153 远小于 ViT 的 196 tokens，标准注意力完全胜任 |
| GCN 流与全联合注意力冗余 | 中 | 增益不显著 | 消融 A1 vs A2 量化 GCN 贡献；若冗余则去掉 GCN 流简化方案 |
| 2D 检测器误差传播 | 高（结构性） | CPN 噪声限制精度上界 | 与所有基线共享同一限制；Oracle 实验（GT 2D）量化上界 |
| 训练不收敛/超参敏感 | 低 | 延期 | 从 PoseFormerV2 超参初始化（lr=8e-4, AdamW, 80 epochs） |

**最坏情况退化下界分析**（逐组件完全失效）：

| 失效组件 | 退化行为 | 退化下界 | 兜底机制 |
|---------|---------|---------|---------|
| DCT 前端完全失效（低频假设不成立，截断丢失全部有效信息） | 全联合注意力仅接收无意义的低频噪声 token | 退化至纯频域输入水平：~49.7mm（PoseFormerV2 Table 6 Simple Baseline，仅 DCT 系数无时域补充） | 保留时域中心帧特征作为补充通路（PoseFormerV2 Table 6 验证：加入 3 帧中心帧后从 49.7→47.1mm，↓2.6mm）；消融 A5 可提前发现此失效 |
| 全联合注意力退化（N=153 时注意力坍缩为均匀分布） | 等效于对压缩 token 做简单平均池化 | 退化至 PoseFormerV2 串行编码器水平：~46.0mm（cards/poseformerv2 eval_setup，f=n=9, F=81） | 标准 MHSA 在 N=153（<ViT 196 tokens）时已有充分验证；即使退化也不低于 PoseFormerV2 基线，因为 DCT 前端的信息压缩收益独立于注意力类型 |
| GCNFormer 流完全失效（图卷积输出为噪声） | 自适应融合权重将 GCN 贡献压至零 | 退化至 A1 配置（DCT + 全联合，无 GCN）：预期 ≤42mm（消融 A1 目标） | 自适应融合机制（softmax 加权）天然具备门控能力：当 GCN 流无信息时 $\alpha_{GF} \to 0$，等效自动剪枝 |
| 2D 检测器完全失效（CPN 输出为随机噪声） | 所有下游组件均接收无意义输入 | 退化至 >100mm（无结构化 2D 信息时 3D 回归无意义） | 此为全管线结构性限制，所有基线共享同一上界；Oracle 实验（GT 2D）量化此天花板：PoseFormer GT 2D 为 31.3mm vs CPN 44.3mm（cards/poseformer，差值 13mm） |
| **全部新增组件同时失效**（最坏联合下界） | DCT 无意义 + 注意力坍缩 + GCN 噪声 | **~49.7mm**（PoseFormerV2 Table 6 纯频域 Simple Baseline） | 此下界仍为已发表方法的可复现结果，不劣于领域内任何单一组件失效的公开报告；系统不会产出比"无信息先验的线性回归"更差的结果 |

退化下界总结：即使所有新增组件同时完全失效，系统退化至 ~49.7mm（PoseFormerV2 Table 6 纯频域基线），仍优于 PoseFormer 的 44.3mm 仅约 5mm——这意味着本方案的**下行风险有限**（最坏情况比当前最优分解范式差 ~11mm），而**上行空间显著**（最优情况优于当前 SOTA 1–2mm）。兜底策略：消融矩阵 A0–A4 可在 W3–W5 内逐步定位失效组件并启用对应降级路径。

### 4.4 性能/成本量化

**推理效率估算**（T=81, T'=9, S=17, d=256, L=6, h=8）：

- Token 数：$N = T' \times S = 153$
- 每层注意力 FLOPs：$4 \times N^2 \times d_k \approx 4 \times 153^2 \times 32 \approx 3.0M$
- 每层 FFN FLOPs：$2 \times N \times d \times 4d \approx 2 \times 153 \times 256 \times 1024 \approx 80.3M$
- 6 层总计：$\approx 6 \times (3.0 + 80.3) \approx 500M$ FLOPs
- 加 DCT 变换（$O(T \times S)$，可忽略）与 GCN 流（轻量）
- **总估计：~0.5-1.0 GFLOPs**

对比：
- PoseFormerV2（F=81）：0.35 GFLOPs
- STCFormer（T=81）：6.5 GFLOPs
- MixSTE（T=81）：46.2 GFLOPs

本方案 GFLOPs 与 PoseFormerV2 同量级，远低于 STCFormer 和 MixSTE，同时提供真正联合建模能力。

**训练成本**：单卡 RTX 3090，batch size 1024（N=153 tokens 显存充裕），80 epochs，预计 6-10 小时。

**逐组件耗时预算表**（T=81, T'=9, S=17, d=256, L=6, h=8，单卡 RTX 3090）：

| 新增组件 | 推理 FLOPs | 参数量 | 推理耗时占比 | 训练耗时估算（80 epochs） | 估算依据 |
|---------|-----------|--------|------------|------------------------|---------|
| DCT 前端（1D-DCT + 低频截断） | ~0.06M（$T \times S \times 2$ 次乘加） | 0（无参数，固定正交变换） | <0.1% | <5 min（仅前向变换，无反向梯度） | 81×17×2=2754 次标量运算，远小于注意力 |
| 线性投影 $W_{proj}$（2→256） | ~0.07M（$T' \times S \times 2 \times 256$） | 768（含 bias） | <0.1% | <5 min | 153×512=78K 乘加 |
| JSTT 主干（6 层全联合注意力 + FFN） | ~500M（注意力 18M + FFN 482M） | ~4.7M（每层 MHSA 262K + FFN 525K）×6 | ~96% | 5–8h | §4.4 正文逐层估算；训练耗时参照 PoseFormerV2 同规模（cards/poseformerv2：80 epochs, RTX 3090, 4–6h）按 token 数比例修正 |
| GCNFormer 流（空间 GCN + 时间 K-NN GCN，各 3 层） | ~15M（空间 7.5M + 时间 7.5M） | ~1.5M（MotionAGFormer GCN 流占比 ≈1/4 推算，cards/motionagformer） | ~3% | 0.5–1h | 空间 GCN：$T' \times |E| \times d^2$（|E|=33 含自连接，17 关节 16 边）；时间 GCN：$S \times |E_t| \times d^2$（$|E_t|=2 \times T'$，k=2） |
| 自适应融合层 | ~0.16M（$N \times 2d \times 2$） | ~2K | <0.1% | <5 min | 线性映射 + softmax + 逐元素加权 |
| **合计** | **~515M（≈0.5 GFLOPs）** | **~6.2M** | **100%** | **6–10h** | 与正文估算一致（0.5–1.0 GFLOPs 含 batch 维度开销） |

注：训练耗时为 3 seeds 中单次训练的估计；推理耗时占比按 batch=1 单帧前向传播计算。GCN 流参数量依据：MotionAGFormer 总参数为此前 SOTA 的 1/4（cards/motionagformer eval_setup），其双流中 GCNFormer 流约占 1/4–1/3（Transformer 流含标准 MHSA 参数更多），本方案 GCN 流规模与 MotionAGFormer 同构。

### 4.5 时间线里程碑表

| 周次 | 里程碑 | 交付物 | 风险点 |
|------|--------|--------|--------|
| W1 | 环境搭建 + 数据准备 | H3.6M + CPN 数据就绪 | 数据下载速度 |
| W2 | PoseFormerV2 基线复现 | 复现 MPJPE ~46.0mm | 超参对齐 |
| W3 | DCT 前端 + 全联合注意力实现（A1） | 初步训练曲线 | 收敛性 |
| W4 | GCNFormer 流集成（A2 完整方案） | 完整模型 | 融合权重调优 |
| W5 | 消融实验（A0-A6） | 消融表格 | 计算时间 |
| W6 | 基线对比 + 效率测试 | 主结果表 | 基线复现精度 |
| W7 | 论文初稿 + 可视化 | 初稿 | — |
| W8 | 修订 + 投稿 | 终稿 | — |

总计约 8 周（2 个月），目标投稿 CVPR 2027 / ECCV 2027 / AAAI 2027。

### 4.6 综合判级 + 决策路径建议

**综合可行性判级：B+（可行，中等风险，高收益）**

理由：
- (+) 核心假设（DCT 压缩使能全联合注意力）有 PoseFormerV2 的频域稀疏性实验直接支撑
- (+) 全联合注意力本身是标准 Transformer，无新算子风险
- (+) 复杂度论证严密：$O(T'^2 S^2)$ 在 T'=9, S=17 时完全可行
- (+) 评审 3/3 票通过，认为"plausible, well-motivated architectural combination"
- (−) PoseFormerV2 与 STCFormer 已有 repo 卡（代码级事实已核实），但 PoseFormer、MixSTE、MotionAGFormer 三个基线仓库尚无 repo 卡，GCN 模块迁移的工程对齐存在不确定性
- (−) 精度增益幅度（1-2mm）需实验验证，存在"全联合在 N=153 时不显著优于分解"的可能
- (−) 快速运动场景下 DCT 低频假设可能失效

**决策路径 A（推荐）：快速验证路径**

先用最小配置（T=27, T'=3, S=17, L=4, 无 GCN 流）在 1 天内完成 A1 实验。若 MPJPE 显著优于 PoseFormerV2 同配置（47.9mm → <45mm），则确认全联合注意力的增益存在，再扩展到完整方案。

**决策路径 B：保守路径**

若路径 A 增益不显著（<0.5mm），则退回到"PoseFormerV2 + 轻量交叉注意力"的渐进改良路线，将全联合注意力作为消融分析而非主贡献，转而强调频域压缩的效率优势。

**最坏情况分析**：

即使全联合注意力在 N=153 时不优于 STC 轴分解（A1 ≈ A4），本方案仍有以下保底价值：
1. **效率优势**：$O(T'^2 S^2)$ 的复杂度远低于 STCFormer 的 $O(T^2S + TS^2)$（210K vs 1.07M），在相同精度下提供更快的推理速度。
2. **感受野扩展**：DCT 压缩使感受野可任意扩大而不增算力，这一特性独立于注意力类型。
3. **负面结果的学术价值**：证明"在压缩后的频域 token 上，分解注意力与全联合注意力等价"本身是一个有意义的发现，可指导后续工作。

**投稿策略**：

- 首选 CVPR 2027（截稿约 2026 年 11 月）：若 W1-W6 实验顺利，W7-W8 写作可赶上截稿。
- 备选 ECCV 2027（截稿约 2027 年 3 月）：提供更充裕的实验时间，可补充 MPI-INF-3DHP 和更多消融。
- 若实验结果超预期（MPJPE < 37mm），可考虑投 AAAI 2027（截稿约 2026 年 8 月）作为快速发表渠道。

**人员与协作**：本方案设计为单人可完成（单卡 GPU、标准数据集、开源基线）。若需加速，可将基线复现（W2）与核心方法实现（W3-W4）并行化。关键技能要求：PyTorch Transformer 实现经验、DCT/FFT 基础、Human3.6M 数据管线熟悉度。

---

## 5. 结论

本方案提出"频域压缩的时空联合注意力"架构：以 DCT 低频截断将时间维从 $T$ 压缩至 $T'$（压缩比 1/9），在压缩后的 $T' \times S$ 个 token 上施加标准全联合时空自注意力，首次以 $O(T'^2 S^2)$ 的可承受复杂度实现真正的时空耦合建模，突破分解范式"效率锁死架构想象力"的结构性瓶颈。辅以并联 GCNFormer 局部先验流和自适应融合，预期在 Human3.6M 上将 MPJPE 从当前最优 38.4mm 推进至 ≤37.5mm，同时 GFLOPs 控制在 ~1.0（与 PoseFormerV2 同量级，远低于 MixSTE 的 138.6G）。主要风险在于 DCT 低频假设对快速运动的适用性、以及全联合注意力在 N=153 时相对分解范式的增益幅度需实验验证。项目预计 8 周完成（含 5 周实验 + 3 周写作），单卡 RTX 3090 即可支撑全部实验，目标投稿 CVPR/ECCV 2027。

**更广泛的意义**：本方案的核心洞察——"频域压缩使能全联合注意力"——不限于 3D 姿态估计。任何面临"时空 token 数过多导致全注意力不可行"的任务（视频理解、动作识别、多变量时间序列预测）都可借鉴此范式：先在频域压缩时间维，再在压缩后的 token 上做真正的全联合建模。此外，若本方案验证成功，它将为领域提供一个简洁的强基线——标准 Transformer + DCT 前端，无需复杂的轴分解、双流设计或多假设机制——这本身对社区的架构选择具有指导价值。
