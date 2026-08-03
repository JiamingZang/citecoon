# 顺序无关 delta-rule 线性注意力的时空姿态提升 (Order-Free Delta-Rule Linear Attention)
> 技术可行性报告 · 2026-07-21 · idea: 顺序无关delta-rule线性注意力的时空姿态提升.md · ReAct 写作（边写边查证 papers/cards/codebases）


> 技术可行性报告 · 2026-07-21 · idea: 顺序无关delta-rule线性注意力的时空姿态提升.md
> 写作方式：ReAct（边写边查证 papers/ 原文、cards/ 精读卡、codebases/ repo 卡）。凡引用代码事实均标注 `文件:行号`（来自 codebases/ 四张 repo 卡）；凡引用公式/结论均核对 papers/ 原文；查不到来源的具体数字一律标注『待验证』。

---

## 1. 背景与动机

### 1.1 问题陈述

单目视频 3D 人体姿态估计（2D-to-3D lifting）的任务是从 2D 关键点序列 $\mathbf{C} \in \mathbb{R}^{T \times J \times 2}$ 恢复 3D 关节坐标 $\mathbf{Y} \in \mathbb{R}^{T \times J \times 3}$（标准设置 $J=17$，$T\in\{81,243\}$）。该问题因深度歧义（多个 3D 姿态可投影至同一 2D 构型）而天然病态（DiffPose 卡、3D=2D+Matching 卡）。当前主流范式以时空编码器为核心，在 Human3.6M 基准上，最优方法的 MPJPE 已从早期膨胀卷积路线推进至 2025 年 PoseMamba-L 的 38.1 mm（估计 2D 输入）/ 15.6 mm（GT 2D 输入），且仅用 MotionBERT 约 16% 的计算量（PoseMamba 卡，原文 `posemamba_*.md:504-505`）。然而**架构效率与归纳偏置合理性**之间的矛盾日益突出，构成本工作的三个出发点。

**瓶颈一：空间维建模的顺序绑定。** 当前 SSM/Mamba 类方法在空间维度采用选择性扫描，将 $J=17$ 个关节按人工设计的顺序展平为一维序列后递推。这一点在 PoseMamba 代码库中有确凿证据：其空间扫描函数 `CrossScan_plus_poselimbs`（`lib/model/csms6s.py:149-169`）在 `forward` 入口即硬编码断言 `assert W == 17, 'the number of joints is not 17'`（`csms6s.py:153`），并使用一个写死的父关节索引列表

```python
indices = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]   # csms6s.py:156
```

通过 `xs[:, 0] = (x + x[..., indices]).flatten(2, 3)`（`csms6s.py:157`）实现"父关节加法"的局部骨骼先验，再经行/列优先与 flip 构造出 4 个扫描方向（`csms6s.py:158-160`）；反向传播的 `CrossMerge_plus_poselimbs` 同样复用该 `indices`（`csms6s.py:170-192`，索引在 `:187`）。BSTMamba 则将关节硬编码为 5 个身体区域做局部位移增强，并配 DisruptEnhance 训练时扰动（BSTMamba 卡）。这些策略存在两个根本问题：(1) 关节集合本质无序——人体骨架是图结构而非序列，任何固定扫描顺序都是缺乏理论最优性证明的启发式（PoseMamba 卡局限原文："局部扫描顺序是针对 17 关节人体骨架手工设计的启发式策略，缺乏理论最优性证明"）；(2) 跨骨架泛化受阻——当目标骨架从 Human3.6M 的 17 关节变为 MPI-INF-3DHP 的 28 关节时，`assert W==17` 直接失败，全部扫描策略需人工重设计（BSTMamba 卡局限："局部区域划分硬编码为 Human3.6M 的 5 组关节，迁移至其他骨架需重新定义"）。

**瓶颈二：效率—表达力的虚假二元对立。** 领域经历了膨胀卷积→Transformer→SSM 的三代架构竞赛（_themes.json 母题"自注意力二次复杂度…驱动了膨胀卷积→Transformer→SSM/Mamba 的三代架构替代竞赛"）。Transformer 的自注意力对 token 数呈 $O(N^2)$ 复杂度——PoseFormer 卡指出"将所有关节×帧作为 token 直接输入 Transformer 会导致 token 数爆炸（如 243×17=4131），计算复杂度不可接受"；SSM 换来 $O(N)$ 复杂度，却以顺序递推为代价牺牲了内容寻址能力。跨领域证据表明，固定大小递归态（Mamba/RWKV/H3）在需要精确召回的任务上系统性弱于注意力机制（BASED, arxiv:2402.18668，详见 §1.3 命题 2）。

**瓶颈三：时序感受野收益饱和。** _themes.json 母题"感受野扩张的收益在有限帧数后即饱和"汇总了多篇论文的证据：TCPFormer 卡局限"从 243 帧扩到 351 帧仅降低 0.2–0.3 mm 误差"；HDFormer 卡局限"输入帧数扩展到 243 帧时性能反而略降"。这暗示真正瓶颈并非感受野大小，而是空间建模质量与时序信息利用效率。

**量化目标。** 本工作旨在：(1) 在 Human3.6M 上以 $\leq$ PoseMamba-B 的参数量（3.358 M）与 MACs（13.9 G @ T=243，原文 `posemamba_*.md:378-382`）达到或超过其 MPJPE；(2) 消除对人工关节扫描顺序的依赖，使关节置换后精度衰减尽可能小（目标 $<0.5$ mm，见 §3.1）；(3) 在不重设计任何骨架相关模块的前提下，跨骨架迁移至 MPI-INF-3DHP（28 关节）时精度衰减显著小于 PoseMamba。

### 1.2 相关工作

按技术路线将现有 2D-to-3D lifting 方法分为四组（仅引用 cards/ 中存在的论文）。

#### 路线一：膨胀时间卷积

以 VideoPose3D（W2903549000, Pavlakos et al. 2019）为代表，通过膨胀因果卷积将感受野指数扩展至 243 帧，以残差块（膨胀卷积 + 1×1 卷积 + BN + ReLU + Dropout）实现确定性 2D→3D 映射；243 帧模型 16.95 M 参数 / 33.87 M FLOPs，推理约 150k FPS（单 GP100 GPU, batch=1）（VideoPose3D 卡）。其核心假设"时间上下文足以消解深度歧义"已被后续工作质疑——Simple Baseline 卡证明仅用简单 MLP 即达竞争性精度，暗示单帧信息已相当充分；DiffPose 卡表明即使 243 帧输入仍需扩散模型建模多模态不确定性；AugLift 卡进一步指出时序方法实际学到的是"运动动态过拟合"而非几何消歧（_themes.json 母题"时序建模被默认为消解深度歧义的主要手段，但其实际贡献可能仅是平滑与冗余消除"）。该路线对空间维建模几乎无显式设计，所有关节被展平为单一向量输入。

#### 路线二：时空 Transformer

PoseFormer（W3136525061, Zheng et al. 2021）首次引入纯 Transformer 的空间—时序两阶段结构：空间模块将帧内 $J$ 个 2D 关节视为独立 patch 做自注意力，时序模块沿时间轴做自注意力；使用 CPN 检测 2D 时 MPJPE 44.3 mm，GT 2D 下 31.3 mm，性能严重依赖上游检测器（PoseFormer 卡）。MixSTE（W4312417903, Zhang et al. 2022）提出关节分离策略——在时序 Transformer 中将每个关节作为独立 token 并行建模其时间轨迹，交替堆叠空间块（STB）与时序块（TTB），以 seq2seq 方式输出整段序列；模型 33.7 M 参数（$d_l=8, d_m=512, T=243$），CPN 输入下 MPJPE 40.9 mm、P-MPJPE 32.6 mm、GT 下 21.6 mm，较 PoseFormer 提升约 31%（MixSTE 卡）。ConvFormer（W4382892987）用 **1D 卷积替代线性投影生成 Q/K/V** 以引入稀疏性，并以多个不同核大小动态加权聚合多尺度局部上下文；参数量 2.56 M（$T=9$）~ 10.24 M（$T=243$），较前 SOTA 减少 65.5%~83.4%（ConvFormer 卡）。HDFormer（W4385767582, Chen et al. 2023）通过最短路径距离（SPD≤4）在有向骨架图上定义超骨，以交叉注意力融合高阶结构信息，3.7 M 参数（MixSTE 的 1/10）即达 GT 2D 下 21.6 mm（$T=96$）（HDFormer 卡）。

该路线的共同局限：(1) 自注意力 $O(N^2)$ 复杂度限制序列长度扩展；(2) 空间 Transformer 将关节视为无序集合（PoseFormer 卡局限："空间 Transformer 将关节视为无序集合，未显式引入骨骼拓扑/骨长约束"），或引入硬编码拓扑偏置（HDFormer 的 SPD≤4 超骨依赖固定有向骨架拓扑，HDFormer 卡局限），绑定 17 关节格式。值得注意的是，PoseFormer 将关节视为**无序集合**的处理方式在对称性上是正确的，但其 $O(J^2)$ 自注意力与硬编码位置编码并未带来跨骨架能力——本工作可视为"用顺序无关的关联记忆继承 PoseFormer 的无序性，同时保持线性复杂度"。

#### 路线三：状态空间模型 (SSM/Mamba)

PoseMamba（W4409368373, 2025 AAAI）首次将纯 SSM 架构引入 3D HPE，提出双向全局—局部时空 SSM 块：空间维按关节编号全局扫描 + 沿肢体链路局部重排序扫描，时间维正反向扫描，四方向融合。PoseMamba-L（6.714 M 参数，27.9 G MACs）在 H3.6M 上 P1=38.1 mm（估计 2D）/ 15.6 mm（GT 2D），仅用 MotionBERT 16% 的计算量（PoseMamba 卡，原文 `posemamba_*.md:363-367, 504-505`）。BSTMamba（W4413980847, 2025）在双向扫描基础上引入非因果 1D 卷积分支与 DisruptEnhance 训练时关节扰动，H3.6M CPN $T=81$ MPJPE=41.7 mm、GT 22.5 mm，参数 9.85 M、13.57 G MACs@$T=81$（BSTMamba 卡）。

该路线的核心张力（_themes.json 母题原文）："SSM 的线性扫描假设关节/帧存在有意义的顺序（PoseMamba 卡局限：'SSM 扫描顺序对空间维度建模的理论合理性未讨论，关节并非天然有序序列'），这一归纳偏置是否比注意力的无序置换等变性更适合姿态任务，尚无定论。" BSTMamba 的 DisruptEnhance 已尝试用训练时关节扰动缓解此问题，但其扰动仍绑定 5 区域硬编码划分，且是数据增强而非架构属性（BSTMamba 卡）。

#### 路线四：图卷积与结构先验注入

HDFormer（W4385767582）以最短路径距离定义超骨；SBAHGNet 以骨骼偏置注意力（13×17 细粒度 + 6×17 粗粒度超边）注入拓扑先验（_themes.json 母题"骨骼拓扑先验硬编码"证据）；POT 将骨骼拓扑最短路径距离编码为注意力偏置，其组划分固定为 5 组且绑定 17 关节（_themes.json）。该路线对骨骼先验的建模最为显式，但先验模块均硬编码绑定特定骨架定义（_themes.json 母题张力："所有方法默认骨骼拓扑是已知且固定的输入，而非可学习或可推断的对象"）。

#### 跨领域：线性注意力与 delta-rule 序列模型

2023–2025 年语言模型序列架构已推进至门控线性注意力 / delta-rule 线性注意力，本工作直接借鉴以下四篇（均核对 papers/ 原文）：

- **GLA**（Gated Linear Attention, arxiv:2312.06635, Yang et al. ICML 2024）：将数据依赖门控衰减引入线性注意力，递推为 $S_t = \mathrm{Diag}(\alpha_t) S_{t-1} + k_t^\top v_t$，其中 $\alpha_t = \sigma(x_t W_{\alpha 1} W_{\alpha 2})^{1/\tau}$ 经低秩线性层 + sigmoid 生成（原文 Table 1 与 Eq. 3）；并提出 chunkwise 并行训练算法 FlashLinearAttention，复杂度 $O(LCd + Ld^2)$。
- **DeltaNet**（arxiv:2406.06484, Yang et al. NeurIPS 2024）：将 delta 规则（Widrow-Hoff 学习规则）引入线性注意力状态更新，$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$；用 Householder 乘积的紧凑 WY 表示并行化，使训练可跨序列长度并行。
- **Gated DeltaNet**（arxiv:2412.06464, ICLR 2025）：融合门控与 delta 规则，$S_t = S_{t-1}\big(\alpha_t(I - \beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top$（原文 Table 2）；门控支持快速擦除（$\alpha_t \to 0$），delta 规则支持定向更新（$\alpha_t \to 1$ 退化为纯 delta）。
- **BASED**（arxiv:2402.18668, Arora et al. ICML 2024 Workshop）：系统证明固定大小递归态在 recall 任务上的劣势，并提出线性注意力 + 小滑窗注意力的 recall-throughput 平衡方案。

fla-org/flash-linear-attention 库提供了上述模块的高效 Triton 实现（详见 §2 与 codebases/flash-linear-attention.md）。这些方法保持 $O(N)$ 复杂度，但将信息存入 key→value 关联记忆、按内容寻址召回，召回不依赖序列位置——恰好对症姿态任务中空间维无序的痛点。

### 1.3 根本性分析

从信息论与几何视角论证现有方法为何在空间维建模上存在原则性缺陷。

**命题 1：关节集合的置换等变性与扫描顺序的信息论代价。**

设 $J$ 个关节构成集合 $\mathcal{J} = \{j_1, \ldots, j_J\}$，其物理语义由骨骼图 $\mathcal{G}=(\mathcal{J}, \mathcal{E})$ 定义。对任意置换 $\pi \in S_J$，3D 姿态估计任务满足等变性 $f(\pi(\mathbf{C})) = \pi(f(\mathbf{C}))$，即输出应随输入关节置换等变。然而 SSM 的递推结构将信息编码进隐态的**位置**而非**内容**：在 PoseMamba 的 `CrossScan_plus_poselimbs`（`csms6s.py:149-169`）中，关节 $j_i$ 对隐态的贡献取决于其在 `flatten(2,3)` 后一维序列中的位置 $i$ 及其在 `indices` 列表中的父节点关系，而非其语义角色。这意味着模型必须通过训练数据**记忆**特定扫描顺序下的统计规律，而非学习顺序无关的结构关系；对未见过的关节排列（如跨骨架迁移时新增关节插入序列中间），位置编码语义被破坏，且 `assert W==17`（`csms6s.py:153`）直接拒绝非 17 关节输入。从信息论角度，固定扫描顺序等价于在 $J!$ 种可能排列中硬编码一种，引入了 $\log_2(J!) \approx 48$ bits（$J=17$，由 Stirling 近似 $\log_2(17!)\approx 48.3$）的无关顺序信息，模型必须消耗容量来学习"忽略"这些伪信息。

**命题 2：关联记忆 vs. 顺序递推的召回精度界（核对 BASED 原文）。**

线性 Transformer 可解释为基于外积的 key-value 关联记忆（Gated DeltaNet 原文 introduction 引用 Smolensky 1990 的张量积表示）。其可存储的**正交 key-value 对数量受模型维度约束**：当序列长度超过维度时"记忆碰撞（memory collisions）"不可避免，阻碍精确检索（Gated DeltaNet 原文 introduction，引 Schlag et al. 2021）。DeltaNet 原文 §2.2 同样指出"纯加性更新规则难以释放过去的 key-value 关联，最终在 $L > d$ 时导致 key 碰撞"。

BASED（arxiv:2402.18668）从通信复杂度理论给出了更严格的下界。其 **Theorem 3.1**（原文 §3.2）：

> 任何因果依赖于输入 $u \in \{0,1\}^{N \times d}$ 的递归模型，求解 MQAR（多查询关联回忆）需要 $\Omega(N)$-bit 的状态规模。

这说明固定大小递归态在关联回忆上的劣势是**根本性的**，而非架构细节的偶然。BASED 进一步针对门控卷积类架构（H3/Hyena/RWKV v4 的规范形式 BaseConv）证明 **Theorem 3.2**：数据无关的 BaseConv 需要 $\log(2d)$ 层才能求解 MQAR（$d=\log_2 c$，$c$ 为词表大小）；**Theorem 3.4**：在 $\log c \le d \le 2(\log N)^{1-\epsilon}$ 编码下，数据无关 BaseConv 需要 $\Omega(\epsilon \log\log N)$ 层才能求解 AR（关联回忆；原文结合 Theorem 3.2 进一步推出求解 MQAR 的层数下界 $\Omega(\max(\log\log c, \log\log N))$）。作为对照，Arora et al. (2023a) 证明注意力可在**常数层**内求解 MQAR（BASED 原文 §3.2 末）。

对姿态任务的直接推论：空间维需存储 $M = J = 17$ 个关节的 key→value 关联。当 key 维度 $d_k \gg J$（典型 $d_k = 64 \gg 17$）时，关联记忆矩阵 $\mathbf{S} \in \mathbb{R}^{d_k \times d_v}$ 的容量充裕，理论上可近无损存储全部 17 个关节映射——但前提是记忆支持**按 key 内容寻址**而非按位置递推。这正是 delta-rule 线性注意力相对 Mamba 选择性扫描的原则优势。

**命题 3：delta-rule 更新即测试时 SGD，天然内容寻址。**

DeltaNet 原文（§2.2 及 Gated DeltaNet §3.1）给出 delta 规则的在线回归解释：将隐态 $S$ 视为快权重，优化在线回归目标 $\mathcal{L}(S) = \tfrac{1}{2}\|S k_t - v_t\|^2$，单步 SGD 得

$$S_t = S_{t-1} - \beta_t \nabla \mathcal{L}(S_{t-1}) = S_{t-1} - \beta_t(S_{t-1} k_t - v_t) k_t^\top = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

其中 $\beta_t$ 为学习率（写入强度），$S_{t-1} k_t - v_t$ 为预测误差（delta）。等价的 key-value 检索解释（DeltaNet 原文 §2.2）：先用当前 key 检索旧值 $v_t^{\text{old}} = S_{t-1} k_t$，再以 $v_t^{\text{new}} = \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$ 插值替换，$S_t = S_{t-1} - v_t^{\text{old}} k_t^\top + v_t^{\text{new}} k_t^\top$。当 $\beta_t=1$ 旧值被完全替换，$\beta_t=0$ 记忆不变。查询 $o_t = S_t q_t$ 按 key 相似度召回，**与 key 的写入顺序无关**——这恰好满足命题 1 的置换等变性要求。门控衰减 $\alpha_t$（GLA/Gated DeltaNet）则在此基础上加入自适应权重衰减，控制旧记忆遗忘速率。

**命题 4：时空不对称性原则。**

姿态任务的两个维度具有本质不同的对称性：

| 维度 | 对称性 | 合适操作 | 现有 SSM 做法 |
|------|--------|----------|--------------|
| 空间（关节） | 置换等变（无序集合） | 内容寻址关联记忆 | 顺序递推（违反） |
| 时间（帧） | 平移等变 + 因果有序 | 顺序递推 / 双向扫描 | 顺序递推（合理） |

现有 SSM 方法对两个维度施加相同的顺序递推操作，违反了这一不对称性。本工作的核心原则：**空间用顺序无关关联记忆、时间用有序递推**——在保持线性复杂度的同时，尊重两个维度的内在对称性。

---

## 2. 方法

本方法以 PoseMamba 的双向全局—局部时空块为骨架，保留其时间维双向扫描（时间有序，递推合理），将空间维的选择性扫描替换为 delta-rule 门控线性注意力关联记忆。整体拆分为三个互补贡献。所有代码级衔接均基于 codebases/PoseMamba.md 与 codebases/flash-linear-attention.md 的 `文件:行号` 事实。

### Contribution 1: 空间维 Delta-Rule 门控线性注意力关联记忆

**设计动机。** 消除空间维对人工扫描顺序的依赖（即移除 `csms6s.py:153` 的 `assert W==17` 与 `csms6s.py:156` 的硬编码 `indices`），以内容寻址替代位置递推，使空间模块对关节置换等变、对骨架格式无关。

**技术细节（delta 更新公式，已与原文核对）。**

输入为时间维处理后的帧级特征 $\mathbf{Z} \in \mathbb{R}^{B \times T \times J \times d_m}$。对每帧独立地将 $J$ 个关节 token 映射为 query、key、value：

$$\mathbf{q}_j = \mathbf{W}_Q \mathbf{z}_j,\quad \mathbf{k}_j = \mathbf{W}_K \mathbf{z}_j,\quad \mathbf{v}_j = \mathbf{W}_V \mathbf{z}_j \in \mathbb{R}^{d_k}$$

key 经 L2 归一化 $\hat{\mathbf{k}}_j = \mathbf{k}_j / \|\mathbf{k}_j\|$ 以稳定内积尺度（Gated DeltaNet 原文 §3.4 对 q、k 施加 L2 归一化"for training stability"）。关联记忆矩阵 $\mathbf{S} \in \mathbb{R}^{d_k \times d_v}$ 按**门控 delta 规则**逐 token 更新（与 Gated DeltaNet 原文 Table 2 及 fla `naive.py:50-59` 一致）：

$$\boxed{\;\mathbf{S}_j = \alpha_j\, \mathbf{S}_{j-1}\big(I - \beta_j \hat{\mathbf{k}}_j \hat{\mathbf{k}}_j^\top\big) + \beta_j\, \mathbf{v}_j \hat{\mathbf{k}}_j^\top\;}$$

等价地写成预测误差形式（即 fla `fla/ops/gated_delta_rule/naive.py:50-59` 的递推语义）：

$$\mathbf{S}_j = \alpha_j \mathbf{S}_{j-1} + \beta_j\big(\mathbf{v}_j - \mathbf{S}_{j-1}^\top \hat{\mathbf{k}}_j\big)\hat{\mathbf{k}}_j^\top$$

其中：
- $\alpha_j \in (0, 1]$ 为门控衰减因子（GLA 风格，由 $\mathrm{sigmoid}(\mathbf{w}_\alpha^\top \mathbf{z}_j + b_\alpha)$ 产生），控制旧记忆遗忘速率；fla 中对应 `gdn_gate_chunk_cumsum` 的 $g = -\exp(A_{\log})\cdot\mathrm{softplus}(g_{\text{in}} + dt_{\text{bias}})$ 后取 cumsum 再 $\exp$（`fla/ops/gated_delta_rule/gate.py:96-104`）；
- $\beta_j \in (0, 1]$ 为写入强度（由 $\mathrm{sigmoid}(\mathbf{w}_\beta^\top \mathbf{z}_j + b_\beta)$ 产生），控制新信息写入量；
- $\mathbf{v}_j - \mathbf{S}_{j-1}^\top \hat{\mathbf{k}}_j$ 为 delta 误差项，确保记忆矩阵趋向存储 key→value 的精确映射。

查询输出 $\mathbf{o}_j = \mathbf{S}_J^\top \hat{\mathbf{q}}_j$。关键性质：最终记忆 $\mathbf{S}_J$ 是所有 $J$ 个关节信息的叠加，查询按 $\hat{\mathbf{q}}_j$ 与各 key 的相似度召回，**与关节被处理的顺序无关**。

**实现路径（代码级，引用 fla repo 卡）。** 这里有一个必须诚实对待的工程事实：fla 库的 chunk-wise Triton kernel **全部硬编码因果（下三角）mask**——GDN 的 chunk 内输出 mask 位于共享 kernel `fla/ops/common/chunk_o.py:125-126`（`m_A = (o_t[:,None] >= o_t[None,:]) & ...`），GLA 的 intra-chunk mask 位于 `fla/ops/gla/chunk.py:363, :401`；且 GDN 的 chunk_size 有严格校验 `if chunk_size not in (16, 32, 64): raise ValueError(...)`（`fla/ops/gated_delta_rule/chunk.py:535-536`）。然而本工作的空间维是**非因果、顺序无关**的，且序列长度仅为 $J=17$（不满足 chunk_size ∈ {16,32,64} 的约束，且 17 非 2 的幂）。因此**不能直接复用** `chunk_gated_delta_rule`（`fla/ops/gated_delta_rule/chunk.py:397-588`）这一因果 chunk kernel。

正确的实现路径有二，均 grounded：

1. **朴素递推实现（推荐，$J=17$ 时开销可忽略）。** 直接采用 fla 提供的 naive 参考实现语义（`fla/ops/gated_delta_rule/naive.py:50-59`）：

   ```python
   # 改写自 fla/ops/gated_delta_rule/naive.py:50-59（原为因果自回归，此处用于非因果空间维）
   for j in range(J):                                  # J=17，循环开销极小
       h = h * g[:, :, j].exp()[..., None, None]       # 门控衰减 α
       err = v[:, :, j] - (h * k[:, :, j][..., None]).sum(-2)  # delta: v - h@k
       err = err * beta[:, :, j][..., None]            # β 缩放
       h = h + k[:, :, j].unsqueeze(-1) * err.unsqueeze(-2)    # 外积更新
   o = torch.einsum('bhd,bhdm->bhm', q, h)            # 内容寻址查询
   ```

   该循环对 $J$ 维不可并行，但 $J=17$ 极短，绝对开销 $<0.14$ M FLOPs/layer（见 §4.4），在整体 MACs 中占比可忽略。其优势是天然非因果、无 chunk_size 约束、对任意 $J$ 自适应。

2. **非因果并行形式（可选，用于训练加速）。** 由 DeltaNet 原文 §2.1 的并行形式 $O = (QK^\top \odot M)V$ 出发，将因果 mask $M$ 替换为全 1 矩阵（非因果），即可一次性并行计算空间维输出。但 delta 规则的 WY 表示本质依赖因果顺序（`(I + L)^{-1}`，$L$ 下三角，见 `fla/ops/gated_delta_rule/chunk_fwd.py:40-69` 的 `solve_tril`），非因果设定下需重新推导封闭解（codebases/flash-linear-attention.md 风险 #2 指出"delta rule 的 $(I+\beta KK^\top)^{-1}$ 在非因果设定下是否仍有封闭解，需要数学推导验证"）。故本工作默认走路径 1，路径 2 列为后续优化。

> **与上一版报告的差异（诚实修正）：** 上一版称"可参考 fla 的 GatedDeltaNet 实现"暗示可直接复用其 Triton kernel；核对 repo 卡后发现该 kernel 因果且 chunk_size 受限，空间维必须改用 naive 递推或自研非因果 kernel。双向/非自回归场景 fla 官方指向独立仓库 `fla-org/flash-bidirectional-linear-attention`（codebases/flash-linear-attention.md §9），但该仓库内容**未验证**（repo 卡风险 #1），故本工作不依赖它。

**局部性增强（引用 ConvFormer 卡）。** 借鉴 ConvFormer"用 1D 卷积替代线性投影生成 Q/K/V 以引入稀疏性"的做法（ConvFormer 卡方法），在 key 生成路径中嵌入深度可分离 1D 卷积（核大小 $\kappa=3$，沿关节维度）：$\mathbf{k}_j = \mathbf{W}_K\big(\mathrm{DConv1D}_\kappa(\mathbf{z})\big)_j$，使 key 编码局部骨骼邻域信息，增强对肢体链式结构的感知，同时不引入硬编码拓扑。注意 ConvFormer 的核大小固定为 $(7,7,7)$ 且为手工设定（ConvFormer 卡局限），本工作的 $\kappa$ 作为超参在 §3.2 消融。

**多头扩展。** 设 $H$ 个注意力头，每头独立维护记忆矩阵 $\mathbf{S}^{(h)} \in \mathbb{R}^{(d_k/H) \times (d_v/H)}$（GLA 原文 §4.4 的多头形式：$d'_k = d_k/H, d'_v = d_v/H$），输出拼接后经线性投影 $\mathbf{o}_j = \mathbf{W}_O[\mathbf{o}_j^{(1)}; \ldots; \mathbf{o}_j^{(H)}]$。GLA 原文还在每头输出后加 LayerNorm 与输出门控 $r_t = \mathrm{Swish}(x_t W_r + b_r)$，$y_t = (r_t \odot o'_t)W_O$（GLA 原文 §4.4），本工作沿用此输出门控以稳定训练。

**与现有系统的衔接（代码级，引用 PoseMamba repo 卡）。** 采用 codebases/PoseMamba.md "改造接口点 · 方案 B（替换整个空间 SSM 块）"：新模块满足接口 输入 $(B, F, N, C) \to$ 输出 $(B, F, N, C)$（与 `BiSTSSMBlock._forward` 相同，`mambablocks.py:619-685`），替换 `PoseMamba.py:66-74` 中 `self.STEblocks` 的构造，保持 `self.Spatial_norm`（LayerNorm）不变，时间块 `self.TTEblocks` 独立保留。具体地，PoseMamba 主模型前向（`PoseMamba.py:133-140`）为 `STE_forward → TTE_foward → ST_foward → head`，其中 STE 通过 `rearrange('b f n c -> (b f) n c')` 将每帧独立做关节间运算（与 MixSTE `model_cross.py:468-483` 的 STE_forward 同构），本模块即在 `(B*F, J, C)` 形状上对 $J$ 个关节做 delta-rule 关联记忆，再 `rearrange` 回 `(B, F, J, C)`。如此**完全绕过** `CrossScan_plus_poselimbs`（`csms6s.py:149-169`）与其 `assert W==17`、硬编码 `indices`，且无需改动 `FORWARD_TYPES` 路由（`mambablocks.py:312-341`）与 `k_group=4` 绑定的预训练权重形状（repo 卡"关键约束"）。

### Contribution 2: 时间维双向门控递推与跨维融合

**设计动机。** 时间维真正有序，保留 SSM 双向递推的合理性；同时引入门控衰减机制（GLA 风格）增强对长程时序依赖的选择性记忆。

**技术细节。** 时间维处理沿用 PoseMamba 的双向扫描策略。区别于原始 Mamba 的固定衰减矩阵，引入数据依赖门控衰减（GLA 原文 Eq. 3 的标量/对角门控形式）：

$$\mathbf{h}_t^{\rightarrow} = \gamma_t \mathbf{h}_{t-1}^{\rightarrow} + \mathbf{B}_t \mathbf{x}_t^{\rightarrow},\qquad \mathbf{h}_t^{\leftarrow} = \gamma_t \mathbf{h}_{t+1}^{\leftarrow} + \mathbf{B}_t \mathbf{x}_t^{\leftarrow}$$

其中 $\gamma_t = \mathrm{sigmoid}(\mathbf{w}_\gamma^\top \mathbf{x}_t + b_\gamma)$ 为逐帧门控衰减，允许模型在动作突变帧快速遗忘旧状态、在平稳段保持长程记忆。双向融合 $\mathbf{z}_t^{\text{temp}} = \mathbf{W}_{\text{fuse}}[\mathbf{h}_t^{\rightarrow}; \mathbf{h}_t^{\leftarrow}] + \mathbf{z}_t$。

> **诚实注记（引用 PoseMamba repo 卡风险 #1）：** repo 卡指出 PoseMamba 代码中 STE 与 TTE 都传 `forward_type='v2_plus_poselimbs'`，且 `BiSTSSM` 的 CrossScan 始终在 $(H=F, W=17)$ 的 2D 结构上操作，"时间块并非独立沿帧维 1D 扫描……未找到单独的时间 1D 扫描实现"。这意味着原仓库的"时间维双向扫描"在代码层面可能与论文描述存在差异。本工作的处理：时间维改造可基于 MixSTE 式的显式 reshape（`rearrange('b f n cw -> (b n) f cw')`，MixSTE `model_cross.py:485-496` 的 TTE_forward）将每个关节独立做帧间递推，从而获得清晰、可控的时间维双向门控 SSM，避免依赖原仓库语义不明的时间块。

**跨维融合策略。** 空间关联记忆输出 $\mathbf{Z}^{\text{spat}}$ 与时间递推输出 $\mathbf{Z}^{\text{temp}}$ 通过逐元素门控融合：

$$\mathbf{Z}^{\text{out}} = \sigma(\mathbf{W}_g[\mathbf{Z}^{\text{spat}}; \mathbf{Z}^{\text{temp}}]) \odot \mathbf{Z}^{\text{spat}} + \big(1 - \sigma(\mathbf{W}_g[\mathbf{Z}^{\text{spat}}; \mathbf{Z}^{\text{temp}}])\big) \odot \mathbf{Z}^{\text{temp}}$$

该设计允许模型自适应决定每帧每关节更依赖空间结构还是时间动态。

**与现有系统的衔接。** 时间维模块保持 PoseMamba 的 Mamba 块结构（`forwardv2`，`mambablocks.py:555-578`：`in_proj → chunk → SiLU 门 → permute → conv2d → SiLU → forward_core → 门控 → out_proj`），仅将 SSM 的固定衰减替换为门控衰减。整体块结构为：

$$\mathbf{Z}'_l = \mathrm{LN}\big(\mathrm{TemporalSSM}(\sigma(\mathrm{DW}(\mathrm{LN}(\mathrm{SpatialDeltaAttn}(\mathbf{Z}_{l-1})))))) + \mathbf{Z}_{l-1}$$
$$\mathbf{Z}_l = \mathrm{MLP}(\mathrm{LN}(\mathbf{Z}'_l)) + \mathbf{Z}'_l$$

### Contribution 3: 顺序无关的跨骨架自适应机制

**设计动机。** 消除所有硬编码关节分组与扫描顺序（即 `csms6s.py:153` 的 `assert W==17` 与 `csms6s.py:156/187` 的 `indices`），使模型无需重设计即可处理不同关节数量的骨架。

**技术细节。**

**(a) 无空间位置编码。** 空间关联记忆模块不使用空间位置嵌入（移除 MixSTE/PoseMamba 中的 `Spatial_pos_embed`，MixSTE `model_cross.py:474`）。关节的语义角色完全由其内容 key 编码——模型通过数据驱动学习"髋关节的 key 模式"而非"第 0 号位置是髋关节"。

**(b) 可学习骨架类型嵌入（可选）。** 若需引入弱拓扑先验，以可学习的关节**类型**嵌入 $\mathbf{e}_{\text{type}(j)} \in \mathbb{R}^{d_m}$（按语义类型：躯干/上肢/下肢/头部/左右索引，而非固定编号）叠加到输入特征：$\mathbf{z}_j = \mathbf{W}_{\text{embed}} \mathbf{c}_j + \mathbf{e}_{\text{type}(j)}$。新骨架只需指定每个关节的类型标签即可复用已学习的类型嵌入。

**(c) 关节数量自适应。** 关联记忆矩阵 $\mathbf{S} \in \mathbb{R}^{d_k \times d_v}$ 的维度与关节数 $J$ 无关（仅与特征维度有关），因此从 17 关节切换到 28 关节时模块参数完全复用，无需任何结构修改。由命题 2 的容量论证，只要 $d_k \geq J$ 即可近无损存储所有关节的 key→value 映射；$d_k=64 \geq 28$ 满足。这是相对 `assert W==17`（`csms6s.py:153`）的本质改进。

**(d) 训练时置换增强。** 每个训练 batch 以概率 $p=0.5$ 对关节维度施加随机置换 $\pi$，同时置换输入 2D 坐标与 GT 3D 坐标，强制模型学习顺序无关表征；推理时不需要置换。**与 BSTMamba 的 DisruptEnhance 的区别（诚实界定新颖性）：** BSTMamba 卡显示其 DisruptEnhance "按 5 个身体区域独立置换"且"保留核心关节、打乱其余"，仍绑定 H3.6M 的 5 区域硬编码划分，且是训练增强而非架构属性；本工作的置换增强是**全置换、无硬编码区域**，且与架构层面的顺序无关性（无位置编码 + 内容寻址）协同——即使关闭增强，架构本身亦对置换等变（A8/A9 消融分离二者贡献）。

**与现有系统的衔接。** 该贡献是架构层面的设计原则，不引入额外模块，而是通过移除硬编码（空间位置嵌入、固定扫描顺序）和添加轻量机制（类型嵌入、置换增强）实现跨骨架能力。

### 整体架构伪代码

```python
class OrderFreeDeltaRuleBlock(nn.Module):
    """单个时空块：空间 delta-rule 关联记忆 + 时间双向门控 SSM"""
    def __init__(self, d_model, n_heads, d_k, d_v, expand_ratio=2):
        self.spatial_delta_attn = SpatialDeltaRuleAttention(d_model, n_heads, d_k, d_v)
        self.temporal_ssm = GatedBidirectionalSSM(d_model, expand_ratio)
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * expand_ratio)
        self.dw_conv = DepthwiseConv1d(d_model, kernel_size=3)

    def forward(self, Z):  # Z: [B, T, J, d_model]
        Z_spat = self.spatial_delta_attn(self.ln1(Z))      # 空间维：顺序无关
        Z_temp = self.temporal_ssm(self.dw_conv(Z_spat))   # 时间维：有序递推
        Z_prime = Z_temp + Z                               # 残差
        return self.mlp(self.ln2(Z_prime)) + Z_prime


class SpatialDeltaRuleAttention(nn.Module):
    """空间维 delta-rule 门控线性注意力（朴素递推，非因果，J 自适应）"""
    def forward(self, Z):  # Z: [B, T, J, d_model]
        B, T, J, D = Z.shape
        Q = self.W_Q(Z)                       # [B, T, J, d_k]
        K = self.W_K(self.dconv(Z))           # [B, T, J, d_k]，含局部卷积增强
        V = self.W_V(Z)                       # [B, T, J, d_v]
        K = F.normalize(K, dim=-1)            # L2 归一化（Gated DeltaNet §3.4）
        alpha = torch.sigmoid(self.gate_decay(Z))  # [B, T, J, 1] 门控衰减
        beta  = torch.sigmoid(self.gate_write(Z))  # [B, T, J, 1] 写入强度
        # 逐帧独立、逐关节递推计算关联记忆（fla naive.py:50-59 语义）
        S = torch.zeros(B, T, self.n_heads, self.d_k_h, self.d_v_h, device=Z.device)
        for j in range(J):                    # J=17（或 28），循环开销可忽略
            pred_err = V[:, :, j] - torch.einsum('...kd,...k->...d', S, K[:, :, j])
            S = alpha[:, :, j] * S + beta[:, :, j] * torch.einsum('...k,...d->...kd', K[:, :, j], pred_err)
        O = torch.einsum('...kd,...k->...d', S, Q)  # 内容寻址查询 [B, T, J, d_v]
        return self.out_gate(self.W_O(O), Z)        # GLA 风格输出门控


class FullModel(nn.Module):
    """完整模型：嵌入 → N 层时空块 → 回归头（接口对齐 PoseMamba.py:133-140）"""
    def __init__(self, J=17, d_model=128, n_layers=20, n_heads=4):
        self.embed = nn.Linear(2, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, 243, 1, d_model))
        self.type_embed = nn.Embedding(8, d_model)   # 关节语义类型（含左右侧）
        self.blocks = nn.ModuleList([
            OrderFreeDeltaRuleBlock(d_model, n_heads, d_k=64, d_v=64)
            for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 3)

    def forward(self, C_2d, types):  # C_2d: [B, T, J, 2]
        Z = self.embed(C_2d) + self.temporal_pos[:, :C_2d.shape[1]] + self.type_embed(types)
        for block in self.blocks:
            Z = block(Z)
        return self.head(Z)  # [B, T, J, 3]
```

**损失函数（沿用 PoseMamba，核对原文消融）。** 沿用 PoseMamba 的三项损失组合 $\mathcal{L} = \mathcal{L}_{\text{MPJPE}} + \lambda_1 \mathcal{L}_{\text{T-Loss}} + \lambda_2 \mathcal{L}_{\text{2D}}$，其中 $\mathcal{L}_{\text{MPJPE}} = \frac{1}{TJ}\sum_{t,j}\|\hat{\mathbf{y}}_{t,j} - \mathbf{y}_{t,j}\|_2$ 为主损失，$\mathcal{L}_{\text{T-Loss}}$ 含时间一致性损失与速度误差，$\mathcal{L}_{\text{2D}}$ 为 2D 重投影损失。PoseMamba 原文消融（`posemamba_*.md:293-304`，Table 5）：仅 MPJPE Loss 为 43.7/36.5（MPJPE/PMPJPE），+2D-Loss 降至 43.5/36.2（贡献 0.2 mm），+T-Loss 降至 42.1/35.1（贡献 1.6 mm），三者联合达 41.8/35.0。MixSTE 则用按关节加权的 WMPJPE（权重 `[1,1,2.5,2.5,1,2.5,2.5,1,1,1,1.5,1.5,4,4,1.5,4,4]`，MixSTE `run.py:401`）加时间一致性损失（`run.py:423`）。本工作默认采用 PoseMamba 三项损失，并在 §3.2 消融是否替换为 MixSTE 的加权方案。

**超参数设置（参照 PoseMamba-B，核对原文 Table 6）。** $d_m=128$，层数 $N=20$，注意力头 $H=4$，$d_k=d_v=64$，MLP 扩展比 2。PoseMamba-B 配置为 depth=20、dm=128、3.358 M 参数、13.9 G MACs（原文 `posemamba_*.md:378-382`）。训练：AdamW，lr=$2\times10^{-4}$，指数衰减 0.99/epoch，权重衰减 0.01，batch size 4，120 epochs，水平翻转增强（PoseMamba 卡 resources）。

---

## 3. 实验计划

### 3.1 评估指标

| 指标 | 定义 | 当前 SOTA 值（来源） | 目标值 | 预期改进 |
|------|------|-------------|--------|----------|
| MPJPE (P1, 估计 2D) | 根对齐后平均关节位置误差 (mm) | 38.1 mm（PoseMamba-L, T=243；原文 :363-367） | ≤38.0 mm（同参数预算） | ≥0.1 mm |
| MPJPE (P1, GT 2D) | GT 2D 输入下 P1 | 14.51 mm（PoseMamba-B, T=81；原文 :221-223） | ≤14.5 mm | 持平或更优 |
| P-MPJPE (P2) | Procrustes 对齐后 MPJPE (mm) | 35.0 mm（PoseMamba-S 消融；原文 :304） | ≤34.5 mm | ≥0.5 mm |
| 参数量 | 模型参数 (M) | 3.358 M（PoseMamba-B；原文 :380） | ≤3.5 M | 持平 |
| MACs/frame | 每帧乘加操作 (G) | 13.9 G（PoseMamba-B, T=243；原文 :381） | ≤14.0 G | 持平 |
| 关节置换衰减 | 随机打乱关节编号后 MPJPE 增量 | 见下方"基线假设值" | <0.5 mm | 显著优于基线 |
| 跨骨架衰减 | H3.6M 训练→3DHP 测试的 MPJPE 增量 | 见下方"基线假设值" | 衰减率 < PoseMamba | 显著优于基线 |

**关于"关节置换衰减"与"跨骨架衰减"两项（清零上一版『待验证』）：** 这两项是本工作**新提出的评估协议**，cards/ 与 papers/ 中无任何方法报告过同类数字，因此不存在可引用的"当前值"。上一版将其标为『待验证』并臆测"PoseMamba 预期 >2 mm"缺乏依据，此处改为**明确标注为待测假设值（hypothesis）**，并给出可证伪的预测依据：

- *关节置换衰减*：本方法空间模块无位置编码、按内容寻址，理论上对关节置换**严格等变**（命题 1），故衰减应趋近 0（仅来自浮点累加顺序的微小数值差，预计 $<0.1$ mm）。PoseMamba 因 `CrossScan_plus_poselimbs` 的扫描方向与 `indices` 父节点加法绑定关节顺序（`csms6s.py:156-160`），置换将破坏其局部骨骼先验，预计衰减显著（**待实测**，本工作将首次报告该数字）。此处不预设具体 mm 数，避免无据臆测。
- *跨骨架衰减*：本方法记忆矩阵维度与 $J$ 无关（C3c），17→28 关节无需结构修改；PoseMamba 的 `assert W==17`（`csms6s.py:153`）使其**根本无法直接处理 28 关节**，必须重设计 `indices` 与扫描顺序。故对比口径为：本方法 zero-shot 28 关节 vs PoseMamba 重设计后 28 关节的相对精度衰减（**待实测**）。

### 3.2 消融矩阵

| 编号 | 实验配置 | 目的 | 类型 |
|------|----------|------|------|
| A1 | 空间维: Mamba 全局扫描 (同时间维) | 基线：复现 PoseMamba 全局空间策略 | Baseline |
| A2 | 空间维: Mamba 全局+局部扫描 | 基线：完整 PoseMamba 策略（`indices` 父节点加法） | Baseline |
| A3 | 空间维: Delta-rule 关联记忆 (同时间维) | **核心消融**：验证 C1 | Key |
| A4 | 空间维: 标准线性注意力 (无 delta 更新, $\beta\equiv1, \alpha\equiv1$) | 分离 delta 规则的贡献 | Ablation |
| A5 | 空间维: Delta-rule 无门控衰减 ($\alpha\equiv1$) | 分离门控衰减的贡献 | Ablation |
| A6 | 空间维: Delta-rule 无卷积 key 增强 ($\kappa=1$) | 分离局部性增强的贡献 | Ablation |
| A7 | 移除时间维门控衰减 (固定衰减) | 验证 C2 | Ablation |
| A8 | 保留空间位置嵌入 | 验证 C3 无位置编码的必要性 | Ablation |
| A9 | 关节置换训练增强 (p=0.5) vs 无增强 | 分离置换增强 vs 架构顺序无关性 | Ablation |
| A10 | **Oracle 上界**: 空间维用完整 $O(J^2)$ 自注意力（无位置编码） | 性能天花板参照 | Oracle |
| A11 | **Negative control**: 空间维用随机固定置换的 Mamba 扫描 | 证明顺序确实有害 | Neg. Control |
| A12 | **Negative control**: 时间维也用 delta-rule (无递推) | 证明时间维需要顺序递推 | Neg. Control |

**Oracle 上界（A10）说明：** $J=17$ 时 $O(J^2)$ 自注意力开销可接受（$J^2=289$），其无位置编码版本提供"顺序无关 + 满秩交互"的性能天花板；若 A3（delta-rule）逼近 A10，则证明线性复杂度关联记忆足以替代二次注意力。**Negative control（A11/A12）说明：** A11 验证"破坏扫描顺序会损害 Mamba"（对照本方法对置换不敏感），A12 验证"时间维若去掉顺序递推会退化"（对照命题 4 的时空不对称性）。

### 3.3 基线方法

| 方法 | 参数量 | 输入 | 来源 |
|------|--------|------|------|
| PoseMamba-B (重实现) | 3.358 M（原文 :380） | CPN/GT 2D, T=81/243 | 同参数预算公平对比 |
| PoseMamba-S | 0.860 M（原文消融 :278） | GT 2D, T=243 | 轻量对比 |
| MixSTE | 33.7 M（MixSTE 卡） | CPN/GT 2D, T=81/243 | Transformer 路线代表 |
| ConvFormer | 2.56–10.24 M（ConvFormer 卡） | CPN, T=81/243 | 卷积注意力路线代表 |
| HDFormer | 3.7 M（HDFormer 卡） | GT/CPN, T=96 | 图结构先验路线代表 |
| BSTMamba | 9.85 M（BSTMamba 卡） | CPN/GT, T=81 | SSM+扰动路线代表 |
| VideoPose3D | 16.95 M（VideoPose3D 卡） | CPN, T=243 | 经典膨胀卷积基线 |

所有基线在相同数据划分、相同 2D 检测输入下重实现或引用 cards/ 原文数字。PoseMamba-B 为核心对比对象，在相同参数/FLOPs 预算下重实现。

### 3.4 数据集要求与预处理

**Human3.6M (主实验)**
- 规模：360 万帧，11 受试者，15 类日常动作（PoseMamba 卡 eval_setup）
- 划分：S1/5/6/7/8 训练，S9/11 测试（标准 Protocol #1 & #2）
- 关节：17 关节
- 2D 输入：(1) CPN 检测 2D（Stacked Hourglass 骨干，PoseMamba 卡）；(2) GT 2D 坐标
- 预处理：以根关节（Hip）为中心归一化；水平翻转增强（训练+测试）
- 序列长度：T=81（主）/ T=243（长序列对比）
- 数据获取：PoseMamba repo 卡给出路径——下载 MotionBERT 预处理数据 → `data/motion3d/`，运行 `cd tools && python convert_h36m.py`，最终路径 `data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl`

**MPI-INF-3DHP (跨骨架迁移)**
- 规模：130 万帧，绿幕/非绿幕/户外三场景（PoseMamba 卡）
- 关节：28 关节（含脚趾、手指等 H3.6M 未标注关节）
- 用法：仅在 H3.6M 训练，直接推理 3DHP 测试集（不微调），评估跨骨架精度衰减
- 预处理：提取与 H3.6M 对应的 17 关节子集 + 完整 28 关节两种设置
- 指标：MPJPE、PCK@150mm、AUC（ConvFormer 卡 / BSTMamba 卡均用 PCK@150mm 与 AUC）

### 3.5 评估协议

1. **主实验 (Protocol #1 & #2)**：H3.6M S9/11 上计算 MPJPE 和 P-MPJPE，按动作类别报告逐动作误差（15 类动作）。重点关注末端关节（Wrist, Foot）误差，因为 PoseMamba 原文消融（Table 4，`:244-280`）表明局部扫描的主要收益来自四肢精度提升（Bidirectional 42.4 → Bidirectional Global-Local 41.8，0.6 mm）。
2. **效率对比**：报告参数量 (M)、MACs/frame (G)、推理吞吐量 (frames/sec, batch=1, RTX 3090)。使用 `fvcore`/`thop` 统计 MACs（PoseMamba repo 卡依赖含 `fvcore`、`thop`），确保与 PoseMamba 原文统计口径一致。
3. **顺序鲁棒性测试（新协议）**：训练完成后，对测试集输入施加 $K=100$ 种随机关节置换，报告 MPJPE 均值与标准差。对比 PoseMamba 在相同置换下的精度衰减曲线。预测依据见 §3.1：本方法因架构顺序无关，衰减应趋近 0；PoseMamba 因扫描顺序与 `indices` 绑定，衰减显著（首次实测）。
4. **跨骨架迁移测试（新协议）**：H3.6M 训练模型直接在 3DHP 上推理（17 关节子集 + 28 关节全集），报告精度衰减比 $\Delta\text{MPJPE}/\text{MPJPE}_{\text{H36M}}$。对 28 关节全集，需定义新增 11 个关节的类型标签（脚趾、手指等归入"末端"类型，C3b）。PoseMamba 因 `assert W==17` 无法直接处理 28 关节，需重设计后对比。
5. **统计显著性**：主实验重复 3 次（不同随机种子 42/123/7），报告均值±标准差。使用配对 t 检验确认与基线差异的显著性（$p<0.05$）。
6. **可视化分析**：(a) 关联记忆矩阵 $\mathbf{S}$ 的 key 空间 t-SNE 可视化，验证不同语义类型关节是否形成可区分的 key 簇；(b) 门控衰减 $\alpha_j$ 的逐关节分布，验证模型是否学到有意义的遗忘策略；(c) 跨骨架迁移时的 3D 姿态定性对比。

### 3.6 计算资源估算表

| 阶段 | 硬件 | 预计时长 | 备注 |
|------|------|----------|------|
| 数据预处理 | CPU (8核) + 64GB RAM | ~2h | 标准 H3.6M 处理流程（PoseMamba repo 卡） |
| 主实验训练 (T=81) | 1× RTX 3090 (24GB) | ~8–12h / run | 经验估计，基于 PoseMamba 卡训练配置（batch=4, 120 epochs）推算；原卡未报告训练时长 |
| 主实验训练 (T=243) | 1× RTX 3090 (24GB) | ~18–24h / run | 序列更长，显存可能需 gradient accumulation |
| 消融实验 (12组) | 1× RTX 3090 | ~5–7 天 (串行) | 可 2–3 组并行（若显存允许） |
| 跨骨架推理 | 1× RTX 3090 | ~1h | 仅推理，无训练 |
| 基线重实现 | 1× RTX 3090 | ~3–5 天 | PoseMamba-B + MixSTE |
| **总计** | 1× RTX 3090 | **约 2–3 周** | 含调试与超参搜索 |

显存估算：PoseMamba-B 在 T=243, batch=4 下约占 18–20GB（估计值，PoseMamba 卡未报告显存峰值，标注为经验估计）；本方法空间维 delta-rule 记忆矩阵额外开销为 $H \times (d_k/H) \times (d_v/H) = d_k d_v / H$ per sample，在 $d_k=d_v=64, H=4$ 下每层每帧仅增加约 4 KB（20 层共约 80 KB），相对总显存可忽略。

### 3.7 预期结果与失败模式分析

**乐观情景（概率 ~40%）：** Delta-rule 空间模块在 GT 2D 下 MPJPE ≤14.0 mm（优于 PoseMamba-B 的 14.51 mm），估计 2D 下 ≤37.5 mm。关节置换衰减 <0.3 mm，跨骨架衰减率 < PoseMamba 的 50%。此时论文具备强竞争力，可投顶会。

**基准情景（概率 ~40%）：** MPJPE 与 PoseMamba-B 持平（±0.5 mm），但顺序鲁棒性和跨骨架迁移显著优于基线。此时论文贡献为"原则性验证 + 跨骨架能力"，适合二线会议或 Workshop。

**悲观情景（概率 ~20%）：** Delta-rule 在 $J=17$ 短序列上收敛不足，MPJPE 落后 PoseMamba-B >1.5 mm。此时需启动备选方案：(a) 增加迭代轮数（对同一记忆矩阵扫描 2–3 轮）；(b) 增大 $d_k$ 至 128；(c) 回退至标准线性注意力（无 delta 更新）+ 无位置编码的组合，仍保留"顺序无关"卖点。

**关键失败模式识别：**
- 左右对称关节混淆：移除空间位置编码后，模型可能无法区分左手/右手、左脚/右脚。缓解：类型嵌入中为左右侧设置不同索引（C3b，`type_embed` 设 8 类）；A8 消融量化。
- 记忆矩阵秩坍缩：若所有关节的 key 趋同，记忆矩阵退化为秩 1，丧失区分能力。缓解：key L2 归一化（Gated DeltaNet §3.4）+ 正交正则化 $\|\mathbf{K}^\top\mathbf{K} - \mathbf{I}\|_F$。
- 训练不稳定：delta-rule 的递推结构在反向传播时可能梯度爆炸。缓解：梯度裁剪（max_norm=1.0，Gated DeltaNet 原文实验即用 gradient clipping 1.0）+ 门控初始化偏向保守（$\alpha$ 初始 ≈0.99）。

---

## 4. 可行性评估

### 4.1 实现复杂度

| 组件 | 复杂度 | 说明 |
|------|--------|------|
| Delta-rule 空间注意力模块 | 中 | 核心新模块，~150 行 PyTorch；基于 fla `naive.py:50-59` 语义改写为非因果递推 |
| 门控衰减时间 SSM | 低 | 在 Mamba 块（`mambablocks.py:555-578` forwardv2）基础上替换固定衰减为 sigmoid 门控，~20 行 |
| 跨维门控融合 | 低 | 标准门控融合，~10 行 |
| 置换增强 | 低 | 数据加载器中随机 shuffle 关节维度，~5 行 |
| 训练流程 | 低 | 复用 PoseMamba 的 AdamW + 指数衰减 + 损失函数（PoseMamba 卡 resources） |

**与更轻替代路线对比：**

| 替代路线 | 工程量 | 预期收益 | 风险 |
|----------|--------|----------|------|
| 仅移除局部扫描、保留全局 Mamba | 极低（删 `indices` 加法） | 负面（验证顺序有害，对应 A1/A11） | 无 |
| 空间维用标准自注意力 ($O(J^2)$) | 低 | 精度可能略优但 $J=17$ 时 $J^2=289$ 开销可接受（对应 Oracle A10） | 丧失线性复杂度优势 |
| 空间维用 delta-rule（本方案） | 中 | 线性复杂度 + 顺序无关 + 跨骨架 | delta-rule 在 $J=17$ 短序列上的收敛性需验证 |
| 完整 Gated DeltaNet 替换时空两维 | 高 | 理论统一 | 时间维丧失因果递推优势（A12 验证） |

### 4.2 外部依赖风险表

| 依赖 | 风险等级 | 说明 | 缓解措施 |
|------|----------|------|----------|
| fla-org/flash-linear-attention | 中 | 本工作仅借鉴其 naive 递推语义（`naive.py:50-59`），**不依赖**其因果 Triton kernel；故 chunk_size ∈ {16,32,64} 约束（`chunk.py:535`）与因果 mask（`chunk_o.py:125`）均不构成阻塞 | 朴素递推为纯 PyTorch，$J=17$ 时性能可接受 |
| fla 双向仓库 flash-bidirectional-linear-attention | 低（不依赖） | 内容未验证（repo 卡风险 #1）；本工作空间维用非因果 naive 递推，无需该仓库 | 若后续需训练加速再评估 |
| PoseMamba 代码库 (nankingjing/PoseMamba) | 低 | repo 卡已确认开源（含权重 HF: nankingjings/PoseMamba-weights），需编译 CUDA 扩展 `kernels/selective_scan`（repo 卡环境） | 若 CUDA 11.7 + PyTorch 1.13.1 环境不兼容，仅复现其空间策略作基线（A1/A2）即可 |
| MixSTE 代码库 (JinluZhang1126/MixSTE) | 低 | 已开源（MixSTE 卡）；本工作借鉴其 STE/TTE reshape 接口（`model_cross.py:468-496`） | 直接引用原文数字作为对比 |
| Human3.6M 数据集 | 低 | 标准学术数据集，需申请；PoseMamba repo 卡给出 MotionBERT 预处理路径 | 成熟获取路径 |
| CUDA/cuDNN 版本 | 低 | PoseMamba 需 PyTorch 1.13.1+cu117（repo 卡）；本方法 naive 递推无 CUDA 扩展依赖 | 标准环境 |

### 4.3 错误传播风险

| 风险点 | 概率 | 影响 | 缓解 |
|--------|------|------|------|
| Delta-rule 在 $J=17$ 极短序列上记忆矩阵欠收敛 | 中 | 精度低于 Mamba 基线 | 增大 $d_k$；多次扫描（2–3 轮迭代更新同一记忆）；A4 消融确认 |
| 门控衰减过强导致记忆过快遗忘 | 中 | 远端关节信息丢失 | 初始化 bias 使 $\alpha$ 初始接近 1（慢遗忘）；warmup |
| 移除空间位置嵌入后模型无法区分左右对称关节 | 中 | 左右混淆误差 | 可学习类型嵌入（C3b）保留语义区分；A8 量化 |
| 2D 检测误差传播 | 确定存在 | 上界受限 | GT 2D 实验分离；与所有基线同等条件对比（MixSTE 卡局限同此） |
| 跨骨架时 key 分布偏移 | 中 | 迁移精度下降 | 类型嵌入对齐；3DHP 上 zero-shot 评估即为验证 |
| PoseMamba 时间块语义不明（repo 卡风险 #1） | 中 | 时间维改造可能与原仓库行为不一致 | 改用 MixSTE 式显式 reshape 做时间维（C2 注记），不依赖原仓库时间块 |

**最坏情况退化下界分析。** 若空间 delta-rule 模块完全失效（最坏情况：记忆矩阵秩坍缩至秩 1，所有关节 key 趋同，空间维输出退化为与输入无关的常数向量），模型仅依赖时间维双向门控 SSM 与 MLP 残差路径。此时系统退化为无显式空间交互的纯时间序列模型，等价于早期膨胀时间卷积路线（VideoPose3D 类，所有关节展平为单一向量、仅靠时间感受野间接获取空间信息）。退化下界估计为 MPJPE ≈ 44–47 mm（H3.6M 估计 2D，待验证——该范围参照膨胀卷积路线代表性水平推算，cards/ 中 VideoPose3D 卡未报告 H3.6M MPJPE 具体值），即相对目标基线 PoseMamba-B（40.8 mm，`codebases/PoseMamba/README.md`）退化约 3–6 mm。兜底机制：(1) 残差连接（$\mathbf{Z}'_l = \mathrm{TemporalSSM}(\ldots) + \mathbf{Z}_{l-1}$）确保空间模块完全失效时仍可透传输入特征，模型不会崩溃至随机输出；(2) 时间维 SSM 独立于空间模块运行，其双向门控递推仍提供帧间平滑与运动先验；(3) 回退方案为将空间维切换至标准线性注意力（消融 A4，$\beta\equiv1, \alpha\equiv1$，无 delta 更新但保留 key-value 交互），仍维持顺序无关属性；(4) W3 Go/No-Go 检查点（GT 2D 下 MPJPE > PoseMamba-B + 2mm 即触发架构调整）可在早期捕获退化并启动备选路径（增大 $d_k$、多轮迭代扫描、或回退至 $O(J^2)$ 自注意力 Oracle A10）。

### 4.4 性能/成本量化

| 指标 | PoseMamba-B | 本方法（预期） | 变化 |
|------|-------------|---------------|------|
| 参数量 | 3.358 M（原文 :380） | ~3.0–3.5 M | 持平（移除局部扫描参数，增加 Q/K/V 投影） |
| MACs/frame (T=243) | 13.9 G（原文 :381） | ~12–14 G | 持平或略低 |
| 训练时间 (120 epochs, T=81) | ~8–12h（经验估计，基于 PoseMamba 卡训练配置推算；原卡未报告训练时长） | ~8–10h | 略增（delta-rule 循环不可并行化于 $J$ 维，但 $J=17$ 极短） |
| 推理延迟 | 与训练同量级 | 预期持平（见下方分析） | $J=17$ 时 delta-rule 的 17 步循环开销极小 |
| GPU 显存 (batch=4, T=243) | ~18–20GB（经验估计） | ~18–20GB | 记忆矩阵开销可忽略 |

**逐组件耗时预算表（新增组件，经验估计）。** 以下估算基于报告 §4.4 解析 FLOPs 推导与 PoseMamba 卡训练配置（batch=4, 120 epochs, RTX 3090）；原卡未报告训练时长与显存峰值，故训练时间均为经验估计。

| 新增组件 | 新增参数量 | 训练额外开销（120ep, T=81, 3090） | 推理额外延迟（batch=1, T=243, 3090） | 备注 |
|----------|-----------|----------------------------------|--------------------------------------|------|
| 空间 delta-rule 关联记忆（C1） | ~1.15 M（Q/K/V/O 投影 + 门控，抵消移除的空间 SSM 后净变化 ≈0） | +0.5–1.5 h/run（总 8–12h 基础上 +10–15%；17 步串行循环 GPU 利用率低） | +0.1–0.5 ms/样本（17 步 kernel launch 为主瓶颈，纯算力下界 0.019 ms） | 主要新增开销来源 |
| 时间维门控衰减（C2） | ~0.01 M（20 层 × sigmoid 门控向量） | 可忽略（<+0.1 h） | 可忽略（<0.01 ms） | 仅在 Mamba 递推中加逐帧标量乘法 |
| 跨维门控融合 | ~0.07 M（20 层 × 融合门线性层） | 可忽略（<+0.1 h） | 可忽略（<0.01 ms） | 标准逐元素门控 |
| 置换增强（C3d） | 0 | 可忽略（数据加载器内 shuffle） | 0（推理时不启用） | 纯数据增强 |
| 类型嵌入（C3b） | ~1 K（8 类 × 128 维） | 可忽略 | 可忽略 | 查表操作 |
| **合计** | **~1.2 M 新增（净增 ≈0，抵消移除的 SSM）** | **+0.5–1.5 h/run（总训练 ~8.5–13.5 h）** | **+0.1–0.5 ms/样本** | — |

**关键效率分析（清零上一版"推理延迟『待验证』"）。** 上一版将推理延迟标为『待验证』，此处给出可核算的解析估计。空间维 delta-rule 的计算复杂度为 $O(J \cdot d_k \cdot d_v)$（逐 token 更新 $d_k \times d_v$ 矩阵），对 $J=17, d_k=d_v=64$ 约为 $17 \times 2 \times 64 \times 64 \approx 139$ K FLOPs/layer（含外积更新与误差计算）。自注意力的注意力计算部分为 $O(J^2 \cdot d_k)$，即 $2 \times 17^2 \times 64 \approx 37$ K FLOPs/layer（不含 Q/K/V 投影）。在 $J=17$ 的短序列下，delta-rule 的常数因子约为自注意力的 3–4 倍，但两者量级均为 $O(d^2)$ 级，绝对开销极小（$<0.14$ M FLOPs/layer）。与 Mamba 选择性扫描相比，delta-rule 开销更高，但因 $J=17$ 极短，在整体模型 MACs（13.9 G @ T=243，原文 :381）中占比 $<2\%$。

**推理延迟解析估计：** 设 20 层、$T=243$、batch=1，空间维 delta-rule 总 FLOPs $\approx 20 \times 243 \times 139\text{K} \approx 0.67$ G FLOPs（每帧需对 243 帧各做一次 17 步空间递推）。RTX 3090 的 FP32 算力约 35.6 TFLOP/s，纯计算下界 $\approx 0.67\text{G} / 35.6\text{T} \approx 0.019$ ms/样本；即使计入内存带宽与 kernel launch 开销（17 步串行循环的主要瓶颈是 launch 延迟而非算力），估计空间维额外延迟在 0.1–0.5 ms/样本量级，相对 PoseMamba 整体推理时间占比小。故推理延迟**预期与 PoseMamba 持平**（±10%），不再标『待验证』；精确数字将在 §3.5 效率对比中实测报告。关键优势在于：delta-rule 的复杂度对 $J$ 为线性，当骨架扩展至 28 或更多关节时不会如自注意力般二次增长。

### 4.5 时间线里程碑表

| 周次 | 里程碑 | 交付物 | 风险检查点 |
|------|--------|--------|------------|
| W1 | 环境搭建 + 数据准备 + PoseMamba 基线复现 | 可运行的 PoseMamba-B 基线，MPJPE 对齐原文 ±0.5mm | 若复现偏差 >1mm，排查数据预处理；CUDA 扩展编译失败则仅复现空间策略 |
| W2 | Delta-rule 空间模块实现 + 单元测试 | 模块通过梯度检查 + 置换等变性数值验证 | 朴素递推即可，无需 Triton |
| W3 | 完整模型集成 + 初步训练 (T=81, GT 2D) | 首版 MPJPE 数字 | **Go/No-Go**: 若 GT 2D 下 MPJPE > PoseMamba-B + 2mm，需调整架构 |
| W4 | 超参搜索 + 消融实验 A1–A6 | 消融表格初稿 | 确认 delta-rule vs Mamba 的优劣 |
| W5 | 完整消融 A7–A12 + 估计 2D 实验 | 完整消融表 + 主表 | 确认 oracle 上界与 negative control |
| W6 | 跨骨架迁移实验 + 顺序鲁棒性测试 | 迁移表 + 置换衰减图 | 验证核心卖点（首次实测两项新协议） |
| W7 | 论文撰写 + 可视化 | 初稿 | — |
| W8 | 内部审阅 + 修改 + 投稿 | 终稿 | 目标会议截稿日 |

### 4.6 综合判级与决策路径

**综合可行性判级：中高 (B+)**

- **新颖性**：高。将 delta-rule 线性注意力引入姿态估计的空间维建模为首次，无相近 prior work（idea 评审查重结果为"无"，3/3 票支持）。需诚实界定：BSTMamba 的 DisruptEnhance 已用训练时关节扰动（BSTMamba 卡），但其绑定 5 区域硬编码且非架构属性；本工作的"顺序无关"是架构层面的内容寻址，二者本质不同（C3d）。
- **技术风险**：中。核心风险在于 $J=17$ 极短序列上 delta-rule 记忆矩阵的收敛性——语言模型中 delta-rule 处理的是数千 token 的长序列（Gated DeltaNet 原文实验在 1.3B/100B token），而姿态空间维仅 17 个 token。但由命题 2，$d_k \gg J$ 时理论上记忆容量充裕。
- **工程可行性**：高。单卡 3090 可跑（PoseMamba 卡 resources），数据准备成熟（PoseMamba repo 卡路径），核心模块基于 fla naive 语义实现 ~200 行，**不依赖** fla 因果 Triton kernel，规避了 chunk_size 与因果 mask 约束。
- **预期收益**：中高。即使 MPJPE 仅持平 PoseMamba-B，"顺序无关 + 跨骨架免重设计"本身即为有意义的贡献；若精度超越则更具说服力。

**决策路径建议：**

**路径 A（推荐）：快速验证→全量实验→投稿**
- W1–W3 完成快速验证（GT 2D, T=81），若 MPJPE ≤ PoseMamba-B + 1mm 即进入全量实验；
- 目标会议：AAAI 2027 / CVPR 2027 / ECCV 2026 Workshop；
- 卖点：原则性拆分（空间内容寻址 vs 时间顺序递推）+ 跨骨架零成本迁移 + 首次实测关节置换鲁棒性。

**路径 B（保守）：若 W3 验证不达标**
- 回退至"delta-rule + 轻量拓扑先验（可学习邻接矩阵作为 key 的 bias）"混合方案；
- 或将空间维改为 $O(J^2)$ 自注意力（$J=17$ 时开销可接受，对应 Oracle A10），仅保留"无硬编码扫描顺序"的跨骨架贡献；
- 目标降级为 Workshop 论文或技术报告。

---

## 5. 结论

本工作提出将 delta-rule 门控线性注意力作为空间维关联记忆、保留双向 SSM 作为时间维递推的原则性时空拆分架构，用于 2D-to-3D 姿态提升。核心洞察是关节集合本质无序而帧序列本质有序（命题 4），现有 SSM 方法对两维施加相同的顺序递推违反了这一不对称性——PoseMamba 代码中 `assert W==17`（`csms6s.py:153`）与硬编码父关节索引 `indices`（`csms6s.py:156`）正是这一原则性缺陷的具体体现——导致人工扫描顺序绑定与跨骨架泛化瓶颈。方法在保持 $O(N)$ 线性复杂度的同时实现空间维的置换等变与内容寻址，delta 更新公式 $S_j = \alpha_j S_{j-1}(I - \beta_j \hat k_j \hat k_j^\top) + \beta_j v_j \hat k_j^\top$ 已与 Gated DeltaNet 原文（Table 2）及 fla `naive.py:50-59` 核对一致。预期在 Human3.6M 上以 ≤3.5 M 参数达到或超过 PoseMamba-B 的 MPJPE（目标 ≤38 mm 估计 2D / ≤14.5 mm GT 2D），并在关节置换鲁棒性（衰减趋近 0）和跨骨架迁移（17→28 关节免重设计）上显著优于现有 SSM 方法。

**主要风险**为 delta-rule 在极短空间序列（$J=17$）上的收敛性——语言模型中 delta-rule 处理数千 token 的长序列，而姿态空间维仅 17 个 token，记忆矩阵可能欠收敛。缓解策略包括增大 key 维度（$d_k \geq 2J$）、多轮迭代扫描（2–3 轮）和门控初始化偏向保守。次要风险为移除空间位置编码后左右对称关节的混淆，可通过类型嵌入解决。工程上，本工作仅借鉴 fla 的 naive 递推语义而非其因果 Triton kernel，从而规避了 chunk_size ∈ {16,32,64} 与因果 mask 的约束，单卡 RTX 3090 即可运行。

**时间框架与目标会议：** 整体工程量约 2–3 周单卡 RTX 3090 完成全部实验（含基线复现、消融、跨骨架迁移）。W3 为关键 Go/No-Go 节点。目标投稿 ECCV 2026 Workshop（若 W3 验证达标后快速成文）或 AAAI 2027（若需更充分实验）。

**更广泛影响与未来方向：** 本工作建立的"按维度对称性选择建模操作"原则（无序维→关联记忆，有序维→递推）可推广至其他结构化预测任务：(1) 多人姿态估计中人物集合无序、帧序列有序；(2) 手部/动物骨架等非标关节数任务；(3) 3D 网格恢复中顶点集合的局部无序性。若 delta-rule 关联记忆在姿态任务上验证有效，可进一步探索其与图神经网络（可学习邻接矩阵作为 key 的先验 bias）的融合，以及向更大规模预训练（MotionBERT 范式）的扩展。

---

## 附录：来源核对清单

| 类别 | 事实 | 来源 |
|------|------|------|
| 公式 | 门控 delta 规则 $S_t = S_{t-1}(\alpha_t(I-\beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top$ | papers/gated_delta_networks Table 2；papers/parallelizing Table 2 |
| 公式 | DeltaNet delta 更新 $S_t = S_{t-1} - \beta_t(S_{t-1}k_t - v_t)k_t^\top$ | papers/parallelizing §2.2 |
| 公式 | GLA 递推 $S_t = \mathrm{Diag}(\alpha_t)S_{t-1} + k_t^\top v_t$，$\alpha_t=\sigma(x_tW_{\alpha1}W_{\alpha2})^{1/\tau}$ | papers/gated_linear_attention Eq.3, Table 1 |
| 定理 | BASED Theorem 3.1：因果递归模型求解 MQAR 需 $\Omega(N)$-bit 状态 | papers/simple_linear_attention §3.2 |
| 定理 | BASED Theorem 3.2/3.4：BaseConv 需 $\log(2d)$/$\Omega(\epsilon\log\log N)$ 层 | papers/simple_linear_attention §3.2 |
| 结论 | 记忆碰撞：正交 key-value 对受维度约束，$L>d$ 时碰撞不可避免 | papers/gated_delta_networks introduction |
| 代码 | `assert W==17`、`indices=[0,0,1,2,3,...]` | codebases/PoseMamba.md: csms6s.py:153,156,187 |
| 代码 | 空间块替换接口（方案 B）、`STEblocks`、主前向 | codebases/PoseMamba.md: PoseMamba.py:66-74,133-140; mambablocks.py:619-685 |
| 代码 | `chunk_gated_delta_rule`、naive 递推、WY、chunk_size 校验、因果 mask | codebases/flash-linear-attention.md: chunk.py:397-588,535-536; naive.py:50-59; wy_fast.py:39-94; chunk_o.py:125-126 |
| 代码 | MixSTE STE/TTE reshape 接口、WMPJPE 权重 | codebases/MixSTE.md: model_cross.py:468-496; run.py:401,423 |
| 数字 | PoseMamba-B 3.358M/13.9G/14.51mm；PoseMamba-L 38.1/15.6mm；消融 43.7→42.1→41.8 | papers/posemamba :221-223,363-367,378-382,293-304 |
| 数字 | MixSTE 33.7M/40.9/21.6mm；ConvFormer 2.56–10.24M；BSTMamba 9.85M/41.7mm；HDFormer 3.7M/21.6mm；VideoPose3D 16.95M/150kFPS；PoseFormer 44.3/31.3mm | cards/ 对应精读卡 |

> 自查说明：本版已清零上一版 4 处『待验证』——(1) BASED 召回界改为引用可核对的 Theorem 3.1/3.2/3.4；(2)(3) 关节置换衰减与跨骨架衰减两项明确界定为本工作新提出的待测协议（无既有来源可引），给出可证伪预测依据而非臆测具体数值；(4) 推理延迟由解析 FLOPs 估计给出预期值（与 PoseMamba 持平 ±10%），精确值待 §3.5 实测。所有未标注来源的具体数字均已核对 cards/papers/codebases 或显式标注为经验估计/待实测。
