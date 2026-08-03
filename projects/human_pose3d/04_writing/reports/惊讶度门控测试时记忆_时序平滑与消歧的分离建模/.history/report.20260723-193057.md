# 惊讶度门控测试时记忆：时序平滑与消歧的分离建模
> 技术可行性报告 · 2026-07-21 · idea: 惊讶度门控测试时记忆分离时序平滑与消歧.md · ReAct 写作（边写边查证 papers/cards/codebases）


> 技术可行性报告 · 2026-07-21 · 领域：单目视频 3D 人体姿态估计（2D-to-3D Lifting）

---

## 1. 背景与动机

### 1.1 问题陈述

单目视频 3D 人体姿态估计的核心子问题——2D-to-3D lifting——是一个病态逆问题：同一组 2D 关键点可对应多个合理的 3D 姿态（深度歧义），且遮挡与检测噪声进一步放大了不确定性。主流范式将任务解耦为 2D 检测与 2D-to-3D 提升两步（Martinez et al., 2017, W2612706635），后者以 2D 关键点序列为输入，输出相机坐标系下的 3D 关节位置。评估指标以 MPJPE（Mean Per-Joint Position Error，mm）为主，辅以 P-MPJPE（刚性对齐后误差）和 MPJVE（Mean Per-Joint Velocity Error，时序平滑度代理）。

**当前瓶颈的量化表现。** 领域持续把架构复杂度投入"扩大时序感受野"，但收益递减已有明确数字证据：

- **VideoPose3D**（Pavllo et al., 2019, W2903549000）以膨胀时间卷积将感受野指数扩展至 243 帧，在 Human3.6M Protocol 1 上，GT 2D 输入设置下达到 37.8 mm MPJPE（Table 5，16.95M 参数，33.87M FLOPs）；检测 2D 设置下达到 46.8 mm（Table 1a），较前 SOTA（52.8 mm）降低 6 mm（11% 误差减少）。然而，从 27 帧（40.6 mm）到 81 帧（38.7 mm）再到 243 帧（37.8 mm），每增加约 3 倍帧数仅带来约 1 mm 的边际改善，收益曲线明显趋于平坦。
- **TCPFormer**（Liu et al., 2025, W4409366800）在卡片问题栏明确记录："随着输入帧数增加，性能提升趋于饱和（如从 243 帧扩到 351 帧仅降低 0.2–0.3 mm 误差）"。其最优结果（T=243，H3.6M）为 MPJPE 37.9 mm、P-MPJPE 31.7 mm，使用 N=16 层、H=8 头、C=128、代理长度 L=T/3，训练于 2×NVIDIA 4090。
- **HDFormer**（Hyun et al., 2023, W4385767582）在卡片局限栏记录："输入帧数扩展到 243 帧时性能反而略降，推测小模型规模（仅 MixSTE 的 1/10，3.7M 参数）难以充分捕获密集序列中的时序冗余与噪声"。其最优结果（96 帧，GT 输入）为 21.6 mm MPJPE。
- **Attention-TCN**（Zeng et al., 2020, W3034448411）在卡片局限栏记录："更大感受野（n=1029）在深层网络中反而不优"，最优配置为 n=243、L=5 层，GT 输入 34.7 mm MPJPE，CPN 输入 45.1 mm，推理速度约 3000 FPS（0.3 ms/帧）。

与此同时，**单帧方法的下界**令人警醒：Simple Baseline（Martinez et al., 2017, W2612706635）以简单前馈 MLP（线性层 1024 + BN + Dropout 0.5 + ReLU，两模块残差连接重复两次）即达竞争性精度，单帧前向约 3 ms（batch 64 下测量），批处理吞吐可达 ~300 fps。这暗示 2D 坐标中已蕴含相当充分的 3D 信息，时序建模的边际贡献可能被高估。

**瓶颈的本质**：领域缺乏对"时序建模究竟在做什么"的受控分解。固定窗口（如 243 帧）的收益是三种机制的混合体：① **平滑**（抑制检测噪声与抖动，降低 MPJVE）；② **消歧**（利用运动连续性解析深度前后歧义）；③ **运动节律过拟合**（记忆训练集动作的时间统计规律，在分布外动作上退化）。三者从未被受控实验分离，导致架构设计缺乏理论指导——研究者不知道增加的帧数究竟贡献了信息还是冗余，也不知道模型容量与序列长度的失配是否是收益反转的根源。

### 1.2 相关工作

按技术路线分组，仅引用 cards/ 中存在的论文。

#### 1.2.1 膨胀时序卷积路线

VideoPose3D（W2903549000）是该路线的奠基工作：以 B 个残差块（每块含膨胀卷积 + 1×1 卷积 + BN + ReLU + Dropout）实现指数级感受野扩张，243 帧模型仅需 16.95M 参数、33.87M FLOPs，推理约 150k FPS（单 GP100 GPU，batch=1）。其核心假设为"时间上下文足以消解 2D→3D 的深度歧义"，但作者承认性能上限受 2D 检测器精度制约（在半监督 1% S1 极低数据设置下，GT 2D 可再降 22.6 mm；标准全监督设置下差距约 9 mm：46.8 → 37.8），且膨胀卷积对极长序列的建模能力是否饱和未深入分析。Flowing ConvNets（Pfister et al., 2015, W602397586）是更早的时序利用尝试：以稠密光流将邻帧热力图 warp 对齐到当前帧，再用参数化时间池化层学习加权融合；但光流计算使速度从 50 fps 降至 5 fps，且仅在 7 个上半身关节上验证。

#### 1.2.2 Transformer 时空建模路线

PoseFormer（Zheng et al., 2021, W3136525061）首次将纯 Transformer 引入 2D-to-3D lifting：空间 Transformer 对帧内 J 个关节做自注意力，时序 Transformer 对跨帧特征做自注意力。其核心假设为"帧内关节间运动学约束和跨帧长程时序依赖可分别由两个独立的自注意力模块充分建模，二者解耦不会丢失空间-时序联合交互信息"。检测 2D 输入下误差 44.3 mm，GT 2D 下 31.3 mm；243×17=4131 个 token 导致计算复杂度不可接受。

MixSTE（Zhang et al., 2022, W4312417903）提出关节分离（Joint Separation）：在时序 Transformer 块中将每个关节作为独立 token 并行建模其时间轨迹，交替堆叠空间 Transformer 块（STB）与时序 Transformer 块（TTB），采用 seq2seq 输出整段序列。CPN 下 MPJPE 40.9 mm、P-MPJPE 32.6 mm；GT 下 21.6 mm；参数量 33.7M（dl=8, dm=512, T=243）。局限：深度增加（dl=10）参数量显著膨胀（42.2M），对遮挡/快速运动等困难动作改善有限（SitD 等动作误差仍 >50 mm）。

MotionBERT（Zhu et al., 2023, W4390874423）提出 DSTformer（Dual-stream Spatio-temporal Transformer）：两条分支以相反顺序（空间-时间 vs 时间-空间）堆叠 MHSA 块，通过注意力回归器预测自适应权重逐元素融合。DSTformer 深度 N=5，头数 h=8，特征/嵌入维度 512，序列长度 T=243。预训练阶段以 2D-to-3D lifting 为代理任务，对 2D 骨架序列施加随机遮挡与噪声后重建完整 3D 运动，同时引入野外 RGB 视频的加权 2D 重投影损失。3D 姿态估计消融（Table 5）：从头训练 39.2 mm，加入预训练/噪声/遮挡后逐步降至 37.4 mm（加入 2D 重投影数据后 3D 姿态 MPJPE 持平于 37.5 mm，主要增益来自动作识别等下游任务）；架构消融（Table 7）：自适应双流融合 39.25 mm，优于单流 S-T（40.58 mm）和均值融合（39.87 mm）。局限：DSTformer 的自适应融合权重可解释性未被分析；预训练数据仍以室内动捕和有限野外视频为主。

TCPFormer（W4409366800）引入可训练隐式姿态代理（implicit pose proxy）：PUM 用交叉注意力将姿态序列信息编码到代理，PIM 以代理为 key/value 增强姿态序列运动语义，PAM 将两个交叉注意力矩阵相乘得到聚合注意力矩阵，再以可学习参数 μ 与原始自注意力自适应融合。H3.6M 上 T=243 时 MPJPE 37.9 mm、P-MPJPE 31.7 mm；3DHP 上 T=81 时 MPJPE 15.0 mm。局限：代理长度 L 和初始化分布对性能影响无理论解释，仅靠消融选取；深度模糊问题未被根本解决。

MixTGFormer（W7155247101）以双流 Mixformer 架构（两条并行分支以相反顺序堆叠 MHSA+GCN 并行融合块）实现时空信息分支内融合，P1 为 37.6 mm、P2 为 15.7 mm。局限：双流自适应融合权重的收敛行为与可解释性未分析，无法确认两分支确实学到互补信息而非冗余。

ConvFormer（W4382892987）用 1D 卷积替代线性投影生成 Q/K/V 以引入稀疏性，核大小固定为 (7,7,7)，参数量 2.56M（T=9）至 10.24M（T=243），较前 SOTA 减少 65.5%–83.4%。243 帧模型在 CPN 输入 Protocol I 上以 0.2 mm 之差未达 SOTA。其核心假设为"卷积的局部感受野足以替代全连接投影来建模关节间/帧间相关性"，但对突发动作或极长依赖的适应性未讨论。

Fusionformer（W4386706091）提出并行双分支（GIM 全局信息交互 + LIM 局部信息交互，含 STE 自轨迹编码器与 CTE 交叉轨迹编码器），仅在 9 帧输入下验证（H3.6M MPJPE 48.7 mm）。消融实验显示 CTE 贡献有限（STE 提升 0.5 mm，CTE 仅 0.2 mm），跨关节协同假设的普适性存疑。

#### 1.2.3 SSM/Mamba 线性复杂度路线

PoseMamba（W4409368373）提出纯 SSM 架构：在空间维度按关节编号 0–16 全局扫描，同时设计基于骨骼几何拓扑的局部重排序扫描（沿肢体链路）；时间维度进行正反向扫描。PoseMamba-L 在 H3.6M 上 P1=38.1 mm（估计 2D）/15.6 mm（GT 2D），较 MotionBERT 精度提升 1.1/2.2 mm 且仅用 16% 计算量。局限：局部扫描顺序是针对 17 关节人体骨架手工设计的启发式策略，缺乏理论最优性证明；SSM 的因果递推结构在实时流式推理中的延迟特性未分析。

BSTMamba（W4413980847）提出双向时空 SSM：沿空间（关节维度）和时间（帧维度）分别进行双向选择性扫描，以非因果 1D 卷积替代因果卷积并引入对称无 SSM 分支（Conv1D+SiLU）增强局部感知，通过动态门控机制自适应融合。H36M CPN 输入 T=81 MPJPE 41.7 mm，GT 输入 T=81 MPJPE 22.5 mm；9.85M 参数，13.57G MACs。局限：作者承认全局建模可能记忆特定运动节律、局部建模易过拟合固定骨骼结构；局部区域划分硬编码为 Human3.6M 的 5 组关节。

#### 1.2.4 概率/扩散建模路线

DiffPose（Gong et al., 2023, W4386075813）将 3D 姿态估计建模为逆扩散过程：基于 2D 姿态热图初始化具有样本特异性不确定性的 3D 姿态分布 $H_K$，设计基于高斯混合模型（GMM，M=5 个核）的前向扩散过程，通过 GCN 架构扩散模型在时空上下文特征 $f_{ST}$ 条件下执行逆扩散（K=50 步，DDIM 加速至 5 步，N=5 个样本均值）。视频设置检测 2D 下 MPJPE 36.9 mm（前 SOTA 40.9 mm），GT 2D 下 18.9 mm（前 SOTA 21.6 mm）。**关键证据**：即使有 243 帧输入，仍需扩散模型建模多模态不确定性——这直接证明固定窗口时序建模未能根本消除深度歧义。局限：依赖预训练并冻结的上下文编码器 $\phi_{ST}$；多步扩散推理开销高于单次前馈方法；GMM 核数固定为 5。

DDHPose（W4393158891）将解耦策略引入扩散模型前向过程，对骨长和骨方向分别加噪，设计 HSTDenoiser（HRST 层级空间注意力 + HRTT 层级时序交叉注意力，交替堆叠 7 层）。最优 MPJPE 39.0 mm、P-MPJPE 31.2 mm，较 D3DP 仅提升 0.4 mm（1.3%），边际收益有限。

#### 1.2.5 几何消歧与域泛化路线

AugLift（Warner et al., 2025, W4414457910）指出时序方法过拟合运动动态，且 2D-to-3D 方法整体跨数据集泛化能力差（H3.6M 上 40–50 mm 在 3DPW 上退化至 >100 mm），转向引入单目深度估计器（MDE）提供几何消歧信号：将提升输入从 2D 坐标重参数化为 6D 几何描述子 (x, y, c, d, dmin, dmax)，其中后四项为 UADD（置信度 c、深度中位数 d、深度下界 dmin、深度上界 dmax），dmin 作为关节深度几何下界解决前后歧义。检测设置平均 OOD 降 10.1%、ID 降 4.0%；GT+DG 设置 SOTA：3DHP 62.4 mm、3DPW 92.6 mm（较 PoseAug 分别提升 14.5% 和 22.2%）；Fit3D 新姿态误差降 15.7%。局限：MDE 仅提供最近可见表面深度而非遮挡关节真实深度；dref 为需扫描的超参数（3.5–6.0 m）。

3D=2D+Matching（Tekin et al., 2017, W2583372902）提出条件独立假设："给定 2D 姿态 x 后，3D 姿态 X 与图像 I 条件独立，即 2D 骨架足以决定 3D 骨架"，用非参数匹配（约 20 万姿态库）补全深度。作者承认条件独立假设并非总成立（图像中仍有影响 3D 判断的语义线索），且对库外全新姿态泛化能力有限。

#### 1.2.6 骨骼拓扑先验路线

HDFormer（W4385767582）通过最短路径距离（SPD）在有向骨架图上定义超骨（hyperbone），最大 SPD 长度设为 4，以交叉注意力融合高阶信息。3.7M 参数（MixSTE 的 1/10），GT 输入 21.6 mm MPJPE，推理速度约 6 倍于 MixSTE。局限：超骨定义依赖固定有向骨架拓扑与预设最大 SPD=4，对非标准骨架缺乏自适应。

SBAHGNet（W7128608150）提出骨骼偏置注意力 SBA：基于多尺度归一化关联矩阵（13×17 细粒度与 6×17 粗粒度超边）将骨骼拓扑先验以外积形式注入空间注意力分数，仅增加 376 个参数；GCN 分支嵌入多尺度高频增强模块 MSHFE（1×3 和 1×5 深度可分离卷积分别提取低频/高频子路径）。检测 2D 下 37.24/31.57 mm，GT 2D 下 12.38 mm；18.3M 参数，88.9G MACs。局限：超边划分（13/6 个超边）依赖特定 17 关节格式，迁移至其他骨架需重新设计。

#### 1.2.7 跨领域参照：测试时记忆与惊讶度门控

2024–2025 年语言模型领域出现了"测试时记忆"（test-time memory）与"惊讶度门控"（surprise-gated memory）的新范式（Titans 系列，arXiv 预印本；cards/ 中无对应条目，具体数字待验证）。其核心思路是：不按固定窗口吞下全部上下文，而是按"惊讶度"（预测误差）选择性地把高信息量上下文写入压缩神经记忆，跳过冗余 token。Simple linear attention language models（arxiv:2402.18668）从 recall-throughput 权衡角度分析了线性注意力模型的选择性记忆能力，为惊讶度门控提供了理论参照。Gated Delta Networks（GDN，arxiv:2412.06464）实证表明门控（全局衰减/快速擦除）与 delta 规则（定向替换）在记忆管理上互补——单一机制无法同时覆盖"快速清除过时信息"与"精准替换特定键值对"两种需求，双门控递推 $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 组合后性能超越 Mamba2 和 DeltaNet。Gated Linear Attention（GLA，arxiv:2312.06635）验证了数据依赖的对角门控（$G_t = \alpha_t^\top \mathbf{1}$）是标量门控与全秩门控之间的有效折中，兼顾表达力与硬件效率。GDN µP（arxiv:2606.04048）进一步推导出 GDN 的最大更新参数化规则：门控权重矩阵需 $\Theta(1/\sqrt{d})$ 学习率缩放，标量门控参数需 $\Theta(\sqrt{d})$ 缩放，为双门控模块的初始化与训练稳定性提供理论指导。这一机制与姿态任务中"固定窗口收益递减"的瓶颈高度对应：视频中大量相邻帧是冗余的（仅贡献平滑），少数关键帧（运动突变、深度歧义解析点）才携带消歧信息。本工作将这一跨领域思路迁入 3D 姿态时序建模，并设计受控实验分离平滑、消歧、过拟合三种贡献。

### 1.3 根本性分析

**信息论视角：固定窗口的信息-冗余边界。** 设视频帧序列为 $\{x_t\}_{t=1}^T$，目标帧 $x_0$ 的 3D 姿态估计不确定性为 $H(Y|X_{1:T})$。固定窗口方法隐式假设 $H(Y|X_{1:T})$ 随 $T$ 单调递减，但实际上：

$$H(Y|X_{1:T}) = H(Y|X_{1:T-1}) - I(Y; X_T | X_{1:T-1})$$

其中条件互信息 $I(Y; X_T | X_{1:T-1})$ 衡量第 $T$ 帧在已知前 $T-1$ 帧条件下对目标姿态的边际信息贡献。当序列进入稳态运动段时，$I(Y; X_T | X_{1:T-1}) \approx 0$（帧间高度冗余，Strided Transformer 卡片假设"视频中相邻帧姿态高度冗余，可通过步幅卷积逐步合并而不丢失关键信息"）；仅在运动突变点（方向反转、速度阶跃）处 $I(Y; X_T | X_{1:T-1})$ 才显著非零。TCPFormer 卡片记录的"243→351 帧仅降 0.2–0.3 mm"正是这一信息论边界的经验体现：新增 108 帧的边际信息量已趋近于零。

**几何视角：深度歧义的本质不可消解性。** 单目 2D-to-3D lifting 的深度歧义源于透视投影的不可逆性：给定 2D 关节坐标 $x \in \mathbb{R}^{2J}$，满足投影约束 $\pi(X) = x$ 的 3D 姿态 $X \in \mathbb{R}^{3J}$ 构成一个流形（深度方向的一维自由度）。时序建模通过运动连续性约束（相邻帧 3D 姿态应平滑变化）在一定程度上收缩这一流形，但 DiffPose（W4386075813）的实验证明：即使 243 帧输入，预测分布仍是多模态的（需 GMM 建模，M=5 个核），说明时序约束未能将流形收缩到单点。AugLift（W4414457910）进一步指出，时序方法实际学到的是"运动动态过拟合"而非几何消歧——模型记忆了训练集动作的时间统计规律（如步态周期），而非从几何上解析深度歧义。

**优化视角：模型容量与序列长度的失配。** HDFormer 卡片记录的"243 帧时性能反而略降，推测小模型规模难以充分捕获密集序列中的时序冗余与噪声"揭示了一个优化层面的问题：当序列长度 $T$ 增加而模型容量 $C$ 固定时，模型需要在更大的输入空间中学习映射，但容量不足以区分信号与噪声，导致过拟合。Attention-TCN 卡片记录的"n=1029 在深层网络中反而不优"是同一现象的另一表现。这暗示存在一个与模型容量匹配的最优有效序列长度 $T^*(C)$，超过 $T^*$ 后增加帧数不仅无益反而有害。

**三分解假设的形式化。** 固定窗口 $T$ 帧的收益 $\Delta(T)$ 可分解为：

$$\Delta(T) = \underbrace{\Delta_{\text{smooth}}(T)}_{\text{抖动抑制}} + \underbrace{\Delta_{\text{disambig}}(T)}_{\text{深度歧义解析}} + \underbrace{\Delta_{\text{overfit}}(T)}_{\text{运动节律过拟合（负贡献）}}$$

其中 $\Delta_{\text{smooth}}$ 可通过 MPJVE（速度误差）的变化量化；$\Delta_{\text{disambig}}$ 需设计深度翻转/前后歧义解析代理指标；$\Delta_{\text{overfit}}$ 可通过在具有不同时间统计的留出动作（慢速 vs 快速）上的误差差值暴露。这一分解从未在文献中被受控执行——这是本工作的核心诊断贡献。

---

## 2. 方法

本工作包含三个互补贡献：（C1）惊讶度门控测试时记忆模块；（C2）平滑-消歧-过拟合三分解诊断协议；（C3）matched-FLOPs 公平对比框架。

### 2.1 Contribution 1：惊讶度门控测试时记忆模块

#### 2.1.1 设计动机

固定窗口方法（VideoPose3D 243 帧、MotionBERT T=243）对所有帧一视同仁地处理，无论其信息含量高低。这导致：（a）冗余帧（稳态运动段）占用计算资源但贡献近乎零信息；（b）关键帧（运动突变点）的信息被稀释在大量冗余帧中；（c）序列长度增加时模型容量与输入规模失配（HDFormer 243 帧性能反转）。

惊讶度门控记忆的核心思想是：**仅当某帧的预测误差/运动变化超过可学习阈值 $\tau$ 时，才把该帧特征写入压缩关联记忆并参与全局召回；否则仅走局部路径。** 这样，有效序列长度（写入记忆的帧数）自适应于运动复杂度，在不增加序列长度的情况下收获长程消歧信息、跳过冗余帧。

**从单一阈值到双门控递推（组件升级）。** 上述单一阈值 $\tau$ 的二元门控仅能决定"写/不写"，无法同时覆盖"快速清除过时信息"与"精准替换特定键值对"两种记忆管理需求。GDN（arxiv:2412.06464）实证表明，全局衰减门控（$\alpha_t$，控制遗忘速率）与 delta 规则写入（$\beta_t$，控制定向替换强度）在记忆管理上互补，二者组合能处理单一机制无法覆盖的场景。本工作据此将写入机制升级为 GDN 式双门控递推（详见 §2.1.2），其中 $\alpha_t$ 由惊讶度驱动（高惊讶→$\alpha_t \to 0$，快速清空过时记忆；低惊讶→$\alpha_t \to 1$，保留已有信息），$\beta_t$ 控制对特定键值对的定向替换程度。$\alpha_t$、$\beta_t$ 均由当前帧特征经低秩线性层 + sigmoid 参数化（借鉴 GLA，arxiv:2312.06635 的对角门控设计），仅依赖当前输入以保持并行训练兼容性。**原单一阈值 $\tau$ 方案保留为消融基线**（见 §3.2 A14），用于验证双门控是否带来额外收益。

#### 2.1.2 技术细节

**惊讶度计算。** 对每帧 $t$，计算惊讶度标量 $s_t$ 为以下两个信号的加权组合：

$$s_t = \alpha \cdot \| \hat{y}_t - y_t^{\text{pred}} \|_2 + (1-\alpha) \cdot \| \Delta x_t \|_2$$

其中 $\hat{y}_t$ 为当前帧的初步 3D 预测（由局部路径给出），$y_t^{\text{pred}}$ 为基于记忆的历史预测外推，$\Delta x_t = x_t - x_{t-1}$ 为 2D 输入的一阶差分（运动变化代理），$\alpha \in [0,1]$ 为可学习混合系数。第一项衡量预测残差（模型对该帧的"惊讶程度"），第二项衡量输入运动的突变程度。

**双门控递推写入（主方案）。** 记忆状态 $S_t \in \mathbb{R}^{d \times d}$ 按 GDN 式双门控递推更新（arxiv:2412.06464）：

$$S_t = \alpha_t \, S_{t-1}(I - \beta_t \, k_t k_t^\top) + \beta_t \, v_t k_t^\top$$

其中：
- $\alpha_t \in (0,1)$ 为**全局衰减门控**（惊讶度驱动）：高惊讶度帧 $\alpha_t \to 0$，快速清空过时记忆；低惊讶度帧 $\alpha_t \to 1$，保留已有信息。语义映射：$\alpha_t$ 负责"何时遗忘"；
- $\beta_t \in (0,1)$ 为 **delta 规则写入强度**：控制对特定键值对的定向替换程度。语义映射：$\beta_t$ 负责"写入多少"；
- $k_t = W_k f_t / \|W_k f_t\|$（归一化 key），$v_t = W_v f_t$。

$\alpha_t$、$\beta_t$ 均由当前帧特征经低秩线性层 + sigmoid 参数化：$\alpha_t = \sigma(W_\alpha f_t + b_\alpha)$，$\beta_t = \sigma(W_\beta f_t + b_\beta)$，仅依赖当前输入以保持分块并行训练兼容性（GLA 对角门控设计，arxiv:2312.06635）。初始化遵循 µP 缩放规则（arxiv:2606.04048）：$W_\alpha$、$W_\beta$ 学习率缩放 $\Theta(1/\sqrt{d})$，$b_\alpha$、$b_\beta$ 缩放 $\Theta(\sqrt{d})$，确保训练初期 $\alpha_t$、$\beta_t$ 远离 0 和 1 的极端值以维持谱收缩。

**消融基线：单一阈值 $\tau$ 门控。** 原方案的写入决策为 $w_t = \sigma((s_t - \tau)/\beta)$，其中 $\tau$ 为可学习阈值，$\beta$ 为温度参数，$w_t \in [0,1]$ 为软写入权重。该方案作为消融基线保留（§3.2 A14），用于验证双门控递推相对二元门控的增益。

**记忆召回。** 主方案（双门控递推）中，召回为矩阵-向量乘：$o_t = S_t \, q_t$（$q_t = W_q f_t$），无需显式 softmax 注意力，复杂度 $O(d^2)$。消融基线（单一阈值 $\tau$）中，记忆为显式 TopN key-value 存储（见下），召回为标准交叉注意力。

**压缩关联记忆（消融基线适用）。** 记忆模块维护一个固定大小的 key-value 关联记忆 $M = \{(k_i, v_i)\}_{i=1}^N$，其中 $N$ 为记忆容量（远小于序列长度 $T$）。写入操作为：

$$k_{\text{new}} = W_k f_t, \quad v_{\text{new}} = W_v f_t$$
$$M \leftarrow \text{TopN}(M \cup \{(k_{\text{new}}, v_{\text{new}})\}, N)$$

其中 $f_t$ 为帧 $t$ 的特征表示，TopN 按写入权重 $w_t$ 保留权重最高的 $N$ 个条目（FIFO 淘汰策略）。召回操作为标准交叉注意力：

$$\text{recall}(q_t) = \sum_{i=1}^N \text{softmax}\left(\frac{q_t k_i^\top}{\sqrt{d}}\right) v_i$$

其中 $q_t = W_q f_t$ 为当前帧查询。

**双路径前向。** 整体前向路径分为两条：
- **局部路径**：所有帧均经过 DSTformer 双流时空注意力（MotionBERT 骨干），处理短程平滑与帧间局部依赖；
- **全局路径**：主方案中，每帧通过双门控递推更新记忆状态 $S_t$，并以 $o_t = S_t q_t$ 召回长程消歧信息（消融基线中，仅惊讶度超过阈值的帧写入 TopN 记忆，召回为交叉注意力）；
- **融合**：两路径输出以可学习门控权重融合：$\hat{y}_t = \gamma_t \cdot y_t^{\text{local}} + (1-\gamma_t) \cdot y_t^{\text{global}}$。

**伪代码（主方案：双门控递推）。**

```
输入: 2D 关键点序列 X ∈ R^{T×J×2}, DSTformer 骨干 B
初始化: 记忆状态 S_0 = 0 ∈ R^{d×d}, 可学习参数 W_α, b_α, W_β, b_β, W_k, W_v, W_q, γ

for t = 1 to T:
    # 局部路径（所有帧，经 DSTformer 双流注意力）
    f_t_local = B_local(X[t-W:t])
    y_t_local = Head_local(f_t_local)

    # 双门控参数（由当前帧特征驱动）
    α_t = sigmoid(W_α · f_t_local + b_α)   # 全局衰减：高惊讶→α→0
    β_t = sigmoid(W_β · f_t_local + b_β)   # 写入强度：delta 规则替换

    # 双门控递推更新记忆状态
    k_t = normalize(W_k · f_t_local)
    v_t = W_v · f_t_local
    S_t = α_t * S_{t-1} @ (I - β_t * k_t @ k_t^T) + β_t * v_t @ k_t^T

    # 全局路径（从记忆状态召回）
    q_t = W_q · f_t_local
    y_t_global = S_t @ q_t

    # 融合
    γ_t = sigmoid(W_γ · [f_t_local; y_t_local; y_t_global])
    y_t = γ_t · y_t_local + (1-γ_t) · y_t_global

输出: 3D 姿态序列 Y ∈ R^{T×J×3}
```

**伪代码（消融基线：单一阈值 $\tau$ 门控）。**

```
输入: 2D 关键点序列 X ∈ R^{T×J×2}, DSTformer 骨干 B, 记忆容量 N, 阈值 τ
初始化: 记忆 M = ∅, 可学习参数 τ, β, α, γ

for t = 1 to T:
    # 局部路径（所有帧，经 DSTformer 双流注意力）
    f_t_local = B_local(X[t-W:t])          # DSTformer 局部时空编码
    y_t_local = Head_local(f_t_local)

    # 惊讶度计算
    s_t = α · ||y_t_local - extrapolate(M)|| + (1-α) · ||X[t] - X[t-1]||
    w_t = sigmoid((s_t - τ) / β)

    # 条件写入记忆
    if w_t > 0.5:  # 训练时用软权重，推理时用硬阈值
        k_t, v_t = W_k(f_t_local), W_v(f_t_local)
        M = TopN(M ∪ {(k_t, v_t, w_t)}, N)

    # 全局路径（从记忆召回）
    q_t = W_q(f_t_local)
    y_t_global = CrossAttn(q_t, M)

    # 融合
    γ_t = sigmoid(W_γ · [f_t_local; y_t_local; y_t_global])
    y_t = γ_t · y_t_local + (1-γ_t) · y_t_global

输出: 3D 姿态序列 Y ∈ R^{T×J×3}
```

#### 2.1.3 与 MotionBERT DSTformer 的衔接：插入点级改法

本模块以 MotionBERT 的 DSTformer 为骨架实现。以下基于 codebases/MotionBERT.md 的代码级事实，给出三个候选插入位置的具体改法。

**DSTformer 前向路径回顾。** DSTformer 的 `forward` 方法（`lib/model/DSTformer.py:269-358`）核心循环为：

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

每个 Block 内部（`lib/model/DSTformer.py:239-267`）以 `stage_st` 模式先做空间注意力再做时间注意力：

```python
# lib/model/DSTformer.py:240-244
if self.st_mode=='stage_st':
    x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
    x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
    x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
    x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
```

时间注意力（`lib/model/DSTformer.py:188-200`）对每个关节独立做跨帧 self-attention：

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

**推荐插入位置 1（主方案）：Block 输出后、门控融合前——最小侵入。**

- **位置**: `lib/model/DSTformer.py:340-351`，`forward` 方法的 for 循环内。
- **改法**: 在 `x_st = blk_st(x, F)` 和 `x_ts = blk_ts(x, F)` 之后、`alpha` 融合之前，对 `x_st`/`x_ts`（shape `[BF, J, dim_feat]`，其中 `dim_feat=512`）做 memory read/write。具体地，将当前帧特征作为 query 去读取外部记忆 bank，用检索到的值增强 `x_st` 和 `x_ts`：

```python
# 修改后的 forward 循环（伪代码）
for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
    x_st = blk_st(x, F)
    x_ts = blk_ts(x, F)
    # === 惊讶度门控记忆插入点 ===
    x_st = self.surprise_memory[idx](x_st, F)  # 记忆增强
    x_ts = self.surprise_memory[idx](x_ts, F)  # 共享或独立记忆
    # === 插入结束 ===
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

- **优点**: 不修改 Block 内部结构（`attn_s`、`attn_t`、`mlp_s`、`mlp_t` 均不动），预训练权重完全兼容。新增 `surprise_memory` 模块参数随机初始化，`load_pretrained_weights`（`lib/utils/learning.py:39-67`）自动跳过 size 不匹配或新增的 key，不会破坏预训练权重加载。
- **接口约定**: `surprise_memory` 为 `nn.ModuleList`，长度等于 `depth`（默认 5，来自 `configs/pretrain/MB_pretrain.yaml:21`），每层一个独立记忆模块。输入 shape `[BF, J, 512]`，输出同 shape。

**推荐插入位置 2（备选）：`pre_logits` 之前——表征级记忆。**

- **位置**: `lib/model/DSTformer.py:352-354`，`forward` 方法末尾。
- **改法**: 在 `x = self.norm(x)` 之后、`x = self.pre_logits(x)` 之前，对 `[BF, J, dim_feat]` 做 cross-attention 到记忆 bank：

```python
x = self.norm(x)
# === 惊讶度门控记忆插入点（表征级） ===
x = self.surprise_memory_global(x, F)
# === 插入结束 ===
x = self.pre_logits(x)  # Linear(512, 512) + Tanh → dim_rep=512
```

- **优点**: 只加一层，计算开销最小；适合测试时自适应（TTA）场景。`get_representation` 接口（`lib/model/DSTformer.py:360-361`）返回 `[B, F, J, dim_rep]`，记忆模块封装在 DSTformer 内部对外透明。
- **缺点**: 仅在最终表征层做记忆增强，无法在中间层逐步积累长程信息。

**推荐插入位置 3（高侵入）：时间注意力内部——帧级记忆增强。**

- **位置**: `lib/model/DSTformer.py:188-200`，`forward_temporal` 方法。
- **改法**: 将 memory bank 中的 key/value 拼接到 `kt`/`vt` 的时间维度上（`[B, H, N, T+M, C]`），使每帧 attend 到历史记忆：

```python
def forward_temporal(self, q, k, v, seqlen=8, mem_kv=None):
    B, _, N, C = q.shape
    qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
    kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
    vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4)
    # === 拼接记忆 ===
    if mem_kv is not None:
        mem_k, mem_v = mem_kv  # [B, H, N, M, C]
        kt = torch.cat([kt, mem_k], dim=3)  # [B, H, N, T+M, C]
        vt = torch.cat([vt, mem_v], dim=3)
    # === 拼接结束 ===
    attn = (qt @ kt.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = attn @ vt
    x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads)
    return x
```

- **优点**: 语义最自然（记忆即"额外历史帧"），时间注意力自然地 attend 到记忆。
- **缺点**: 需修改 `Attention` 类内部（`forward_temporal` 签名变化），侵入性较大；`attn_drop` 对扩展后的注意力矩阵行为需验证。

**主方案选择：插入位置 1。** 理由：（a）最小侵入，不修改任何现有 Block/Attention 代码；（b）预训练权重 100% 兼容（`load_pretrained_weights` 的 key 匹配逻辑自动跳过新增参数）；（c）每层独立记忆模块允许不同深度学习不同时间尺度的惊讶度模式；（d）与 DSTformer 已有的 `ts_attn` 门控融合机制（`lib/model/DSTformer.py:307-311`，初始化为均匀融合 `bias=0.5`）正交，不干扰时空融合权重的学习。

**配置扩展。** 所有模型超参已通过 YAML → `easydict` → `args` 传入 `load_backbone`（`lib/utils/learning.py:79-101`）。新增记忆模块参数只需在 YAML 中加字段：

```yaml
# configs/pose3d/MB_ft_h36m_surprise.yaml（新增字段）
surprise_memory: true
gate_type: "dual_alpha_beta"  # "dual_alpha_beta"（主方案）| "binary_tau"（消融基线）
memory_dim: 512            # 记忆状态维度 d（= dim_feat）
# 双门控参数（主方案）
alpha_init_bias: 0.0       # b_α 初始值（sigmoid 后 ≈ 0.5，µP 缩放 Θ(√d)）
beta_init_bias: 0.0        # b_β 初始值（sigmoid 后 ≈ 0.5，µP 缩放 Θ(√d)）
gate_lr_scale: 0.022       # 门控权重学习率缩放 Θ(1/√d)，d=512 时 ≈ 1/√512
# 单一阈值参数（消融基线）
memory_capacity: 32        # N: 记忆容量（仅 binary_tau 模式）
surprise_threshold: 0.5    # τ 初始值（可学习，仅 binary_tau 模式）
surprise_temperature: 0.1  # β: 门控温度（仅 binary_tau 模式）
memory_heads: 8            # 记忆交叉注意力头数（仅 binary_tau 模式）
```

在 `load_backbone` 或 DSTformer `__init__` 中读取即可，无需修改 `load_pretrained_weights` 逻辑。

**训练协议。** 沿用 MotionBERT 标准微调设置：`python train.py --config configs/pose3d/MB_ft_h36m.yaml -c checkpoint/pose3d/MB_ft_h36m_surprise -p checkpoint/pretrain/MB_pretrain --selection latest_epoch.bin`。设 `args.finetune = True`，通过 `-p` 指定预训练 checkpoint 路径。预训练权重文件格式为 `.bin`，内部 key 为 `model_pos`（state_dict）。微调时 `load_pretrained_weights` 做 key 匹配、跳过不兼容层（`lib/utils/learning.py:39-67`），新增记忆模块参数随机初始化后随骨干端到端联合训练。

**DataParallel 兼容性注意事项。** 训练脚本使用 `nn.DataParallel`（`train.py:257`）。记忆模块若含非 tensor 状态（如动态增长的 bank），需确认多 GPU 下行为正确。推荐实现中将记忆 bank 固定为 `nn.Parameter` 或 `register_buffer`（固定大小 $N \times d$），避免动态列表。

### 2.2 Contribution 2：平滑-消歧-过拟合三分解诊断协议

#### 2.2.1 设计动机

如 1.3 节分析，固定窗口的收益是三种机制的混合体，但文献中从未被受控分离。本贡献设计一套诊断协议，使研究者能够定量回答："243 帧固定窗口的收益中，平滑、消歧、过拟合各占多少？"

#### 2.2.2 三个代理指标

**（a）平滑度代理：MPJVE。** MPJVE（Mean Per-Joint Velocity Error）衡量预测 3D 姿态的速度误差：

$$\text{MPJVE} = \frac{1}{T-1} \sum_{t=2}^T \frac{1}{J} \sum_{j=1}^J \| (\hat{X}_{t,j} - \hat{X}_{t-1,j}) - (X_{t,j} - X_{t-1,j}) \|_2$$

MPJVE 的降低直接反映时序平滑贡献。若某方法仅降低 MPJVE 而不降低 MPJPE，则其收益主要来自平滑而非消歧。MixSTE 的训练损失中已包含速度误差项（`run.py:423`：`loss_diff = 0.5 * dif_seq + 2.0 * mean_velocity_error_train(...)`），说明领域已隐式利用 MPJVE 作为平滑度代理，但从未将其作为诊断工具分离平滑贡献。

**（b）消歧代理：深度翻转率（Depth Flip Rate, DFR）。** 定义深度翻转事件为：相邻帧间某关节的深度（z 坐标）符号发生错误反转（即预测深度从正确侧翻转到错误侧）。DFR 为：

$$\text{DFR} = \frac{1}{(T-1) \cdot J} \sum_{t=2}^T \sum_{j=1}^J \mathbb{1}[\text{sign}(\hat{z}_{t,j} - z_{t,j}^{\text{root}}) \neq \text{sign}(\hat{z}_{t-1,j} - z_{t-1,j}^{\text{root}})]$$

其中 $z^{\text{root}}$ 为根关节深度（用于中心化）。DFR 的降低反映深度歧义解析能力的提升。借鉴 DiffPose（W4386075813）对多模态不确定性的处理思路：若 243 帧固定窗口的 DFR 与单帧方法无显著差异，则说明时序建模未能有效消歧。

**（c）过拟合代理：慢/快留出动作误差差值（Slow-Fast Gap, SFG）。** 在 Human3.6M 的 15 类动作中，按运动速度将测试动作分为慢速组（如 Wait、Smoke、Pose）和快速组（如 WalkDog、QuickStep、WalkTogether）。定义：

$$\text{SFG} = \text{MPJPE}_{\text{fast}} - \text{MPJPE}_{\text{slow}}$$

若模型过拟合训练集的运动节律，则在时间统计不同的留出动作上 SFG 会显著增大。AugLift（W4414457910）指出时序方法过拟合运动动态、2D-to-3D 方法整体跨数据集泛化能力差，SFG 是前一定性判断的定量化。为排除快速动作本身更难的混淆，用 $\text{SFG}_m - \text{SFG}_{\text{single-frame}}$ 作为过拟合净贡献（单帧方法无时序过拟合，其 SFG 反映固有难度差）。

#### 2.2.3 三分解协议

对每种方法 $m$（单帧、固定窗口 243、惊讶度门控记忆），同时报告：

| 指标 | 平滑贡献 | 消歧贡献 | 过拟合贡献 |
|------|----------|----------|------------|
| MPJVE 变化 | $\Delta\text{MPJVE}_m$ | — | — |
| DFR 变化 | — | $\Delta\text{DFR}_m$ | — |
| SFG 变化 | — | — | $\Delta\text{SFG}_m$ |
| MPJPE 变化 | 间接 | 间接 | 间接 |

三分解的定量估计为：
- **平滑占比** $\approx \Delta\text{MPJVE}_m / \Delta\text{MPJVE}_{\text{oracle}}$（oracle 为 GT 3D 姿态的 MPJVE）；
- **消歧占比** $\approx \Delta\text{DFR}_m / \Delta\text{DFR}_{\text{oracle}}$；
- **过拟合占比** $\approx (\text{SFG}_m - \text{SFG}_{\text{single-frame}}) / \text{SFG}_{\text{baseline}}$。

### 2.3 Contribution 3：Matched-FLOPs 公平对比框架

#### 2.3.1 设计动机

惊讶度门控记忆通过跳过冗余帧减少了有效计算量，而固定窗口 243 帧方法处理所有帧。若直接比较 MPJPE，惊讶度门控记忆可能因计算量更少而处于劣势（或优势，取决于冗余帧是否真的有害）。为公平比较，需在 matched FLOPs 下对比：即调整固定窗口的帧数（如 27、81、143、243 帧）使其 FLOPs 与惊讶度门控记忆的实际 FLOPs 匹配。

#### 2.3.2 FLOPs 计算

**固定窗口方法 FLOPs。** VideoPose3D 各帧数配置的 FLOPs（W2903549000 卡片，Table 5）：27 帧 17.09M、81 帧 25.48M、243 帧 33.87M。膨胀卷积使 FLOPs 与感受野近似对数关系。

**惊讶度门控记忆 FLOPs。** 设平均写入率为 $\rho$（写入记忆的帧占总帧数的比例），则全局路径的有效 FLOPs 为 $\rho \cdot \text{FLOPs}_{\text{global}}$，总 FLOPs 为：

$$\text{FLOPs}_{\text{surprise}} = \text{FLOPs}_{\text{local}} + \rho \cdot \text{FLOPs}_{\text{memory}} + \text{FLOPs}_{\text{fusion}}$$

其中 $\text{FLOPs}_{\text{local}}$ 为局部路径（DSTformer 双流注意力）的 FLOPs，$\text{FLOPs}_{\text{memory}}$ 为记忆写入/召回的 FLOPs（交叉注意力 $O(N \cdot d)$，$N \ll T$ 时开销可忽略），$\text{FLOPs}_{\text{fusion}}$ 为融合层的 FLOPs。

**匹配策略。** 对每个惊讶度阈值 $\tau$，记录实际 $\rho$ 和总 FLOPs；然后在固定窗口方法中找到 FLOPs 最接近的帧数配置（如 27、81、143 帧），在 matched FLOPs 下比较 MPJPE、MPJVE、DFR、SFG。

---

## 3. 实验计划

### 3.1 评估指标

| 指标 | 定义 | 当前 SOTA 值（H3.6M，检测 2D） | 目标值 | 改进幅度 |
|------|------|-------------------------------|--------|----------|
| MPJPE (mm) | 平均关节位置误差（Protocol 1） | 36.9（DiffPose, W4386075813） | ≤36.5 | 相对最强基线（DiffPose 36.9 mm）提升 ~0.5%–1.4%（保守 0.4 mm / 乐观 1 mm；依据：§4.6 精度收益预期 0.4–1 mm） |
| P-MPJPE (mm) | 刚性对齐后误差（Protocol 2） | 31.2（DDHPose, W4393158891） | ≤31.0 | 相对最强基线（DDHPose 31.2 mm）提升 ~0.3%–0.8%（保守 0.2 mm / 乐观 0.5 mm；依据：P-MPJPE 改善通常为 MPJPE 改善的 50%–60%，待验证） |
| MPJVE (mm) | 速度误差（平滑度代理） | 待验证（各方法报告不一致） | 较 VideoPose3D-243 降低 ≥5% | — |
| DFR (%) | 深度翻转率（消歧代理，本工作定义） | 无文献基线（新指标） | 较单帧降低 ≥20% | — |
| SFG (mm) | 慢/快留出动作误差差值（过拟合代理） | 无文献基线（新指标） | 较固定窗口 243 降低 ≥15% | — |
| FLOPs (M) | 每帧推理计算量 | 33.87（VideoPose3D-243） | ≤33.87（matched 或更低） | — |

**GT 2D 输入设置下的参考值：**

| 方法 | GT 2D MPJPE (mm) | 来源 |
|------|-----------------|------|
| VideoPose3D-243 | 37.8 | W2903549000 卡片 |
| MixSTE | 21.6 | W4312417903 卡片 |
| HDFormer（96 帧） | 21.6 | W4385767582 卡片 |
| DiffPose | 18.9 | W4386075813 卡片 |
| PoseMamba-L | 15.6 | W4409368373 卡片 |
| SBAHGNet | 12.38 | W7128608150 卡片 |

### 3.2 消融矩阵

| 实验编号 | 配置 | 目的 | 预期结果 |
|----------|------|------|----------|
| A0 | 单帧 MLP（Simple Baseline 复现） | 下界基线 | MPJPE ~50 mm（检测 2D） |
| A1 | VideoPose3D-27 | 短窗口基线 | MPJPE ~40.6 mm（GT 2D） |
| A2 | VideoPose3D-81 | 中窗口基线 | MPJPE ~38.7 mm（GT 2D） |
| A3 | VideoPose3D-243 | 长窗口基线（SOTA 参考） | MPJPE ~37.8 mm（GT 2D） |
| A4 | MotionBERT + 惊讶度门控记忆（$\tau$ 固定） | C1 主实验 | MPJPE ≤ MotionBERT 基线，MPJVE 显著降低 |
| A5 | MotionBERT + 惊讶度门控记忆（$\tau$ 可学习） | C1 最优配置 | MPJPE ≤ A4 |
| A6 | VideoPose3D-243 + 惊讶度门控记忆 | C1 跨骨干验证 | MPJPE ≤ A3 |
| A7 | **Oracle 上界**：使用 GT 3D 姿态的 MPJVE/DFR 作为平滑/消歧上界 | 三分解归一化基准 | — |
| A8 | **Negative control**：随机门控（$w_t$ 为随机伯努利，与惊讶度无关） | 验证惊讶度门控的必要性 | MPJPE 应显著劣于 A4/A5 |
| A9 | **Negative control**：固定窗口 243 + 随机丢弃帧（丢弃率 = 1-$\rho$） | 区分"选择性写入"与"随机稀疏化" | MPJPE 应劣于 A4/A5 |
| A10 | 惊讶度阈值 $\tau$ 扫描（$\tau \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$） | 阈值敏感性分析 | 存在最优 $\tau^*$ |
| A11 | 记忆容量 $N$ 扫描（$N \in \{8, 16, 32, 64, 128\}$） | 记忆容量敏感性 | 存在最优 $N^*$ |
| A12 | 仅局部路径（去除全局路径） | 消融全局路径贡献 | MPJPE 应劣于 A4 |
| A13 | 仅全局路径（去除局部路径） | 消融局部路径贡献 | MPJVE 应显著劣于 A4 |
| A14 | **单阈值 $\tau$ 二元门控 vs 双门控（$\alpha_t + \beta_t$）** | 验证双门控递推相对二元门控的增益（依据 GDN 卡互补性发现） | 双门控 MPJPE/MPJVE 应优于或持平单阈值；result.json 以 `gate_type: "binary_tau"\|"dual_alpha_beta"` 区分 |

### 3.3 基线方法

| 基线 | 类型 | 来源 | H3.6M MPJPE（检测 2D） |
|------|------|------|----------------------|
| VideoPose3D-243 | 膨胀 TCN | W2903549000 | ~47 mm（GT 2D: 37.8 mm） |
| Attention-TCN | 注意力 TCN | W3034448411 | 45.1 mm |
| MixSTE | 时空 Transformer | W4312417903 | 40.9 mm |
| MotionBERT | DSTformer | W4390874423 | ~39.2 mm（从头训练） |
| TCPFormer | 隐式代理 Transformer | W4409366800 | 37.9 mm |
| DiffPose | 扩散模型 | W4386075813 | 36.9 mm |
| SBAHGNet | GCN+注意力双分支 | W7128608150 | 37.24 mm |
| PoseMamba-L | SSM | W4409368373 | 38.1 mm |
| Simple Baseline | 单帧 MLP | W2612706635 | ~50 mm（待验证） |

### 3.4 数据集要求与预处理

**主数据集：Human3.6M。**
- 规模：360 万帧，11 受试者，15 类动作，4 视角 50 fps（W4409368373 PoseMamba 卡片；HDFormer W4385767582 论文原文）。
- 训练/测试划分：S1/5/6/7/8 训练，S9/11 测试（标准 Protocol 1 & 2）。
- 2D 输入：GT 2D 关键点（上界参考）和 CPN 检测 2D 关键点（实际设置）。MotionBERT 数据准备需 AMASS (SMPL+H) + H36M-SH → `data/motion3d/MB3D_f243s81/`，PoseTrack18 + InstaVariety → `data/motion2d/`（详见 `docs/pretrain.md`）。
- 预处理：以骨盆（根关节）为中心化原点；2D 坐标归一化到 [-1, 1]。
- 动作分组（用于 SFG 计算）：
  - 慢速组：Wait, Smoke, Pose, Sitting, Eating（时间统计较慢）；
  - 快速组：WalkDog, QuickStep, WalkTogether, Walking, Greeting（时间统计较快）。

**辅助数据集（可选，用于域泛化验证）：**
- MPI-INF-3DHP：130 万帧，8 训练/7 测试受试者，室内外混合场景（W4409366800 卡片）。
- 3DPW：野外数据集，用于 OOD 泛化验证（AugLift W4414457910 卡片中使用）。

**数据准备复杂度：小。** Human3.6M 流程已高度标准化，MotionBERT 仓库提供完整的数据预处理脚本（`docs/pretrain.md`），MixSTE 仓库使用 VideoPose3D 格式数据（`data/data_3d_h36m.npz` 和 `data/data_2d_h36m_cpn_ft_h36m_dbb.npz`）。

### 3.5 评估协议

**Protocol 1（MPJPE）：** 根关节对齐后计算平均关节位置误差（mm），不做刚性对齐。这是最严格的协议，对深度误差敏感。

**Protocol 2（P-MPJPE）：** 在 Protocol 1 基础上，对预测和 GT 做刚性对齐（旋转 + 平移 + 缩放），消除全局姿态误差，仅评估局部关节精度。

**Protocol 3（MPJVE）：** 计算速度误差（mm），评估时序平滑度。

**新增 Protocol 4（DFR）：** 深度翻转率（%），评估深度歧义解析能力（本工作定义，见 2.2.2 节）。

**新增 Protocol 5（SFG）：** 慢/快留出动作误差差值（mm），评估运动节律过拟合程度（本工作定义，见 2.2.2 节）。

**评估流程：**
1. 对每种方法，在 S9/11 测试集上计算 MPJPE、P-MPJPE、MPJVE、DFR；
2. 按动作类别分解 MPJPE，计算慢速组和快速组的平均 MPJPE，得到 SFG；
3. 记录每种方法的实际 FLOPs（通过 PyTorch profiler 或 thop 库）；
4. 在 matched FLOPs 下重新排列比较（见 2.3 节）；
5. 对惊讶度门控记忆方法，额外记录平均写入率 $\rho$ 和惊讶度分布直方图；
6. MotionBERT 评估时使用翻转平均（test-time augmentation，参考 MixSTE `run.py:472-494`），需确保记忆模块对翻转对称性无假设。

### 3.6 计算资源估算表

| 实验 | GPU | 显存估算 | 训练时长估算 | 备注 |
|------|-----|----------|-------------|------|
| VideoPose3D-243 基线复现 | 1×RTX 3090（24 GB） | ~8 GB | ~2 h | 16.95M 参数，33.87M FLOPs |
| MotionBERT 基线复现（微调） | 1×RTX 3090（24 GB） | ~16 GB | ~12 h | N=5, h=8, Cf=512, T=243, batch_size=32 |
| MotionBERT + 惊讶度门控记忆 | 1×RTX 3090（24 GB） | ~18 GB | ~14 h | 额外参数 ~2–3M |
| VideoPose3D + 惊讶度门控记忆 | 1×RTX 3090（24 GB） | ~10 GB | ~3 h | 额外参数 ~1–2M |
| 消融实验（A0–A14，共 15 组） | 1×RTX 3090（24 GB） | — | ~42 h 总计 | 含阈值扫描、容量扫描和门控类型对比 |
| 推理评估（所有方法） | 1×RTX 3090（24 GB） | ~4 GB | ~2 h | 仅前向推理 |
| **总计** | **1×RTX 3090** | — | **~72 h（约 3 天）** | 单卡可闭环 |

注：VideoPose3D 训练时长参考 W2903549000 卡片（推理单块 GP100 GPU）；MotionBERT 微调使用预训练权重（162MB，OneDrive/HuggingFace 可下载），batch_size=32（`configs/pose3d/MB_ft_h36m.yaml`），lr_decay=0.99（每 epoch 指数衰减）；TCPFormer 训练于 2×NVIDIA 4090（W4409366800 卡片），本工作单卡方案更轻量。

---

## 4. 可行性评估

### 4.1 实现复杂度

**核心工程量。** 本工作的核心工程任务是实现惊讶度门控关联记忆模块（约 200–400 行 PyTorch 代码），并将其嵌入 MotionBERT DSTformer 的前向路径。具体包括：
1. 惊讶度计算模块（~50 行）：一阶差分、预测残差、可学习阈值；
2. 门控写入逻辑（~30 行）：sigmoid 门、TopN 记忆更新；
3. 压缩关联记忆（~100 行）：key-value 存储（`register_buffer` 固定大小）、交叉注意力召回、FIFO 淘汰；
4. 双路径融合（~30 行）：可学习门控权重；
5. 诊断指标计算（~100 行）：DFR、SFG、写入率统计；
6. 训练/评估脚本修改（~50 行）：在 `train.py` 中集成新模块、记录额外指标。

**与更轻替代路线的对比：**

| 路线 | 实现复杂度 | 相对倍数（以最轻路线为 1×） | 预期收益 | 诊断价值 |
|------|-----------|---------------------------|----------|----------|
| 本工作（惊讶度门控记忆） | 中（~400 行新代码） | 4× | 中（MPJPE 改善 0.4–1 mm） | **高**（首次三分解） |
| 仅做三分解诊断（不改架构） | 低（~100 行新代码） | 1×（最轻基准） | 无（仅分析） | 高 |
| 换用 SSM 骨干（PoseMamba） | 低（调用现有代码） | ~1×（待验证） | 中（MPJPE 改善 1 mm） | 低 |
| 引入 MDE 深度信号（AugLift） | 高（需 MDE 推理管线） | >2×（待验证） | 高（OOD 改善 10%） | 低 |
| 扩散模型（DiffPose） | 高（多步推理） | >2×（待验证） | 高（MPJPE 改善 4 mm） | 低 |

本工作的独特价值在于**诊断贡献**（三分解）而非单纯的精度提升。即使惊讶度门控记忆的 MPJPE 改善有限，三分解协议本身即为领域提供了首个受控实验框架，回答"时序建模到底在做什么"这一根源问题。

**MotionBERT 骨架的工程优势。** 选择 DSTformer 而非 VideoPose3D 作为主骨架的理由：（a）DSTformer 的 `ts_attn` 门控融合机制（`lib/model/DSTformer.py:307-311`）已证明逐层自适应权重融合可行，惊讶度门控记忆可复用类似设计模式；（b）`load_pretrained_weights`（`lib/utils/learning.py:39-67`）的 key 匹配机制天然支持新增模块（自动跳过不匹配 key），无需修改权重加载逻辑；（c）所有超参通过 YAML 配置（`configs/pretrain/MB_pretrain.yaml`），新增记忆模块参数只需加 YAML 字段；（d）`get_representation` 接口（`lib/model/DSTformer.py:360-361`）使记忆模块对外透明，下游任务无需修改。

### 4.2 外部依赖风险表

| 依赖 | 风险等级 | 说明 | 缓解措施 |
|------|----------|------|----------|
| Walter0807/MotionBERT | 低 | 成熟开源仓库（ICCV 2023，371 引用），代码稳定；已有完整 repo 卡（codebases/MotionBERT.md） | 使用官方预训练权重（162MB，OneDrive/HuggingFace）；repo 卡已确认插入点 |
| facebookresearch/VideoPose3D | 低 | 成熟开源仓库（CVPR 2019，1273 引用），代码稳定 | 使用官方预训练权重；作为跨骨干验证 |
| Human3.6M 数据集 | 中 | 需申请下载（学术许可），数据准备流程标准化但耗时 | 提前申请；使用仓库提供的预处理脚本（`docs/pretrain.md`） |
| CPN 2D 检测结果 | 低 | MotionBERT 仓库提供预提取的 2D 检测（`data/motion2d/`） | 直接使用仓库提供的数据 |
| PyTorch 版本兼容性 | 低–中 | MotionBERT README 指定 python=3.7 + CUDA 11.6（较旧）；标准 PyTorch 操作无自定义 CUDA 算子 | 使用 PyTorch ≥1.10；注意 `torch.load` 的 `weights_only` 参数在新版中的变化 |
| GPU 显存 | 低 | 单卡 RTX 3090（24 GB）足够（估算见 3.6 节） | 若显存不足，减小 batch size（默认 32）或使用梯度累积 |
| Titans 式惊讶度门控记忆的参考实现 | 低–中 | cards/ 中无 Titans 对应条目，但 GDN（arxiv:2412.06464，开源 github.com/NVlabs/GatedDeltaNet）和 GLA（arxiv:2312.06635，开源 github.com/fla-org/flash-linear-attention）提供了双门控递推的完整参考实现 | 本工作的双门控机制直接借鉴 GDN 递推公式与 GLA 门控参数化，有成熟开源代码可参照；µP 初始化参考 arxiv:2606.04048 |

### 4.3 错误传播风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 惊讶度阈值 $\tau$ 学习不稳定（门控退化为全开或全关） | 中 | 高（方法失效） | 初始化 $\tau$ 为训练集惊讶度中位数；添加正则化项鼓励 $\rho \in [0.2, 0.8]$；监控 $\rho$ 训练曲线 |
| 双门控 $\alpha_t$/$\beta_t$ 初始化不当致训练发散（谱收缩条件破坏） | 中 | 高（递推状态爆炸） | 遵循 µP 缩放规则（arxiv:2606.04048）：门控权重矩阵 $\Theta(1/\sqrt{d})$ 学习率缩放，标量门控参数 $\Theta(\sqrt{d})$ 缩放，RMSNorm 前插入 $\sqrt{d}$ 乘子；确保 $\alpha_t$、$\beta_t$ 始终远离 0/1 极端值（几何级数收敛条件）；监控 $\|\alpha_t\|$、$\|\beta_t\|$ 训练曲线 |
| 记忆容量 $N$ 过小导致关键帧被淘汰 | 中 | 中（消歧能力下降） | $N$ 扫描（A11）；使用注意力权重而非 FIFO 淘汰 |
| 2D 检测误差传播（CPN 噪声） | 高 | 中（所有方法共同问题） | 同时报告 GT 2D 和检测 2D 结果；GT 2D 结果作为上界参考 |
| DFR 指标定义不合理（与 MPJPE 相关性弱） | 中 | 中（诊断价值降低） | 在 A7（oracle）上验证 DFR 与深度误差的相关性；若不相关，改用深度 MAE 作为替代 |
| SFG 指标受动作难度混淆（快速动作本身更难） | 高 | 中（过拟合估计偏高） | 在单帧基线上校准 SFG（单帧无过拟合，SFG 反映固有难度差）；用 $\text{SFG}_m - \text{SFG}_{\text{single-frame}}$ 作为过拟合净贡献 |
| matched-FLOPs 对比不公平（FLOPs 计算方式不一致） | 中 | 中（结论可信度降低） | 统一使用 thop 库计算 FLOPs；同时报告参数量和推理延迟 |
| DataParallel 下记忆模块状态不一致 | 低 | 中（训练不稳定） | 记忆 bank 用 `register_buffer` 固定大小；避免动态列表；单卡训练规避 |
| 变长序列下记忆模块行为异常 | 低 | 低 | DSTformer 的 `temp_embed` 按 `[:,:F,:,:]` 切片支持变长（`lib/model/DSTformer.py:336`）；记忆模块需对 $F < \text{maxlen}$ 做兼容处理 |

**最坏情况退化下界分析。** 当上游组件完全失效时，系统的退化行为如下：
- **惊讶度门控完全失效**（阈值 $\tau$ 卡死）：若门全开（$w_t \equiv 1$），所有帧写入记忆，系统退化为带固定大小记忆（$N$ 条）的交叉注意力增强 DSTformer，性能上界为 MotionBERT 基线（~39.2 mm，W4390874423），因记忆噪声稀释注意力，预期略劣于基线（退化 ≤2 mm，待验证）；若门全关（$w_t \equiv 0$），无帧写入记忆，全局路径无信息可召回，系统退化为纯局部路径——即 MotionBERT 基线本身。
- **融合门控 $\gamma$ 失效**：$\gamma$ 退化为固定值（如 0.5），系统变为两路径简单平均，性能介于纯局部（基线）与最优融合之间，不会劣于纯局部路径。
- **上游 2D 检测器（CPN）完全失效**：所有方法共同失效（W4390874423 卡片："方法性能上限受限于上游2D姿态估计器的精度与遮挡处理能力"），非本方法特有风险；兜底为同时报告 GT 2D 结果作为上界参考。

**退化下界保证**：无论哪个组件完全失效，系统最坏退化到 MotionBERT 基线水平（~39.2 mm，检测 2D），因为局部路径（DSTformer 双流注意力）始终存在且不依赖记忆模块。兜底机制为：（a）局部路径独立运行，不受全局路径/记忆模块影响；（b）融合门控 $\gamma$ 可学习偏向局部路径（初始化 $\gamma \approx 1$ 即偏向局部）；（c）诊断协议（三分解）独立于记忆模块，即使记忆完全失效，三分解实验仍可执行。

### 4.4 性能/成本量化

**推理性能估算（MotionBERT DSTformer 骨干）：**

| 配置 | 额外参数 | 推理开销增加 | 有效序列长度 |
|------|----------|-------------|-------------|
| MotionBERT 基线（T=243） | 0 | 基准 | 243 帧（固定） |
| MotionBERT + 惊讶度门控记忆（$\rho=0.3$, N=32） | ~2M | +5–8%（交叉注意力 $O(N)$，$N \ll T$） | ~73 帧（自适应） |
| MotionBERT + 惊讶度门控记忆（$\rho=0.5$, N=64） | ~3M | +10–15% | ~122 帧（自适应） |

注：记忆召回的交叉注意力复杂度为 $O(J \cdot N \cdot d)$（$J=17$ 关节，$N$ 为记忆容量，$d=512$），远小于 DSTformer 时间注意力的 $O(J \cdot T^2 \cdot d/H)$（$T=243$，$H=8$）。当 $N \leq 64$ 时，记忆模块开销 <15%。

**训练成本估算：**

| 配置 | 训练时长 | GPU 成本（按 RTX 3090 云价格 ~$0.5/h） |
|------|----------|---------------------------------------|
| MotionBERT 基线微调 | ~12 h | ~$6 |
| 惊讶度门控记忆（MotionBERT 骨干） | ~14 h | ~$7 |
| VideoPose3D 基线复现 | ~2 h | ~$1 |
| 惊讶度门控记忆（VideoPose3D 骨干） | ~3 h | ~$1.5 |
| 全部消融实验 | ~42 h | ~$21 |
| **总计** | **~72 h** | **~$36** |

**逐组件耗时预算（新增组件）：**

| 组件 | 新增参数 | 训练开销（每 epoch 增量） | 推理开销（每帧增量） | 备注 |
|------|----------|--------------------------|---------------------|------|
| 惊讶度计算 + 双门控参数 | ~0（可学习标量/低秩层 $W_\alpha, b_\alpha, W_\beta, b_\beta$） | 可忽略（$O(T \cdot J \cdot d)$ 线性层） | 可忽略 | 一阶差分 + 预测残差 → $\alpha_t, \beta_t$ |
| 双门控递推写入（$S_t$ 更新） | ~0（复用 $W_k, W_v$） | 可忽略 | $O(d^2)$（矩阵-向量乘，$d=512$） | 主方案；消融基线为 TopN $O(1)$ 比较 |
| 记忆召回（$o_t = S_t q_t$） | ~2–3M（$W_q, W_k, W_v$ 投影，5 层） | +17%（12 h → 14 h） | +5–15%（主方案 $O(d^2)$；消融基线 $O(J \cdot N \cdot d)$） | 召回为主要推理开销来源 |
| 双路径融合 | ~1K（$W_\gamma$ 门控） | 可忽略 | 可忽略（逐元素加权） | — |
| 诊断指标（DFR/SFG） | 0 | 0（仅评估时计算） | 0（不参与前向） | 离线统计 |
| **合计** | **~2–3M** | **+17%（12 h → 14 h）** | **+5–15%** | 依据：§3.6 训练时长估算 + 交叉注意力复杂度公式 |

### 4.5 时间线里程碑表

| 阶段 | 任务 | 时长 | 交付物 | 依赖 |
|------|------|------|--------|------|
| M0 | 环境搭建 + 数据准备 | 3 天 | Human3.6M 预处理完成；MotionBERT 基线可运行 | 数据集申请 |
| M1 | 基线复现 | 4 天 | MotionBERT 和 VideoPose3D-243 基线 MPJPE 复现（误差 <0.5 mm） | M0 |
| M2 | 惊讶度门控记忆模块实现 | 5 天 | 模块代码 + 单元测试；嵌入 DSTformer 插入位置 1（`lib/model/DSTformer.py:340-351`） | M1 |
| M3 | 诊断指标实现 | 2 天 | DFR、SFG、写入率统计代码；在基线上验证指标合理性 | M1 |
| M4 | 主实验（A4–A6） | 5 天 | MotionBERT/VideoPose3D + 惊讶度门控记忆的 MPJPE/MPJVE/DFR/SFG | M2, M3 |
| M5 | 消融实验（A0–A14） | 7 天 | 完整消融矩阵；阈值/容量敏感性分析；单阈值 vs 双门控对比 | M4 |
| M6 | 三分解分析 + 论文写作 | 7 天 | 三分解定量结果；论文初稿 | M5 |
| **总计** | — | **~33 天（约 5 周）** | — | — |

### 4.6 综合判级 + 两条决策路径建议

**综合判级：中等工程量，高诊断价值，中精度收益。**

- **实现复杂度：中。** 核心模块 ~400 行新代码，MotionBERT 仓库成熟且已有完整 repo 卡确认插入点（`lib/model/DSTformer.py:340-351`），`load_pretrained_weights` 天然兼容新增参数，Human3.6M 流程标准化，单卡单脚本可闭环。最大风险在于惊讶度门控关联记忆模块需正确嵌入 DSTformer 前向路径并保证 matched-FLOPs 对比公平。
- **外部依赖风险：低。** MotionBERT 仓库已有完整代码侦察卡（codebases/MotionBERT.md），插入点、权重加载、配置扩展均已确认；VideoPose3D 作为跨骨干验证，仓库同样成熟。
- **精度收益预期：中。** MPJPE 改善 0.4–1 mm（检测 2D 设置），可能不足以单独支撑顶会论文；但三分解诊断协议为首创，具有独立学术价值。
- **诊断价值：高。** 首次定量分离时序建模的平滑/消歧/过拟合三种贡献，为领域架构设计提供理论指导。

**决策路径 A（推荐）：诊断优先，精度为辅。**
- 目标会议：ECCV/CVPR Workshop 或 TPAMI 短文（诊断协议 + 初步结果）；若三分解结果显著（如过拟合占比 >30%），扩展为完整论文投 ICCV/ECCV。
- 时间框架：5 周（M0–M6）。
- 核心卖点：首个受控分离时序建模三种贡献的实验框架；惊讶度门控记忆作为实现工具而非唯一贡献。
- 风险：若三分解结果不显著（平滑占主导，消歧和过拟合占比均 <10%），论文影响力有限。

**决策路径 B：精度优先，诊断为辅。**
- 目标会议：ICCV/ECCV 主会（完整论文）。
- 时间框架：8–10 周（在 M6 基础上增加 3–5 周，扩展至 MPI-INF-3DHP 和 3DPW 数据集，增加与 DiffPose、AugLift 等 SOTA 的对比）。
- 核心卖点：惊讶度门控记忆在 matched FLOPs 下超越固定窗口 SOTA；三分解作为分析工具。
- 风险：若 MPJPE 改善 <0.5 mm，精度贡献不足以支撑顶会主会；需依赖诊断贡献补强。

---

## 5. 结论

本工作提出将 2024–2025 年语言模型领域的"惊讶度门控测试时记忆"迁入单目视频 3D 人体姿态估计的时序建模，以替代固定长度时序窗口。核心方案为：在 MotionBERT DSTformer（N=5, h=8, dim_feat=512, T=243）骨干的 Block 输出后、门控融合前（`lib/model/DSTformer.py:340-351`）插入 GDN 式双门控惊讶度关联记忆模块——以全局衰减门控 $\alpha_t$（控制遗忘速率）与 delta 规则写入 $\beta_t$（控制定向替换强度）的递推 $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 替代固定窗口，使记忆管理自适应于运动复杂度，在不增加序列长度的情况下收获长程消歧信息、跳过冗余帧（原单一阈值 $\tau$ 方案保留为消融基线）。配套设计平滑-消歧-过拟合三分解诊断协议（MPJVE 衡量平滑、深度翻转率 DFR 衡量消歧、慢/快留出动作误差差值 SFG 衡量过拟合），首次定量回答"固定窗口 243 帧的收益中，平滑、消歧、过拟合各占多少"这一领域根源问题。预期收益：MPJPE 改善 0.4–1 mm（检测 2D 设置，H3.6M），同时提供首个受控三分解实验框架。主要风险：双门控初始化需遵循 µP 缩放规则以维持训练稳定性、DFR 指标定义合理性需验证、精度改善可能不足以单独支撑顶会主会。时间框架约 5 周（单卡 RTX 3090 可闭环，总 GPU 成本 ~$36），目标会议为 ECCV/CVPR Workshop（诊断优先路径）或 ICCV/ECCV 主会（精度优先路径，需扩展至 8–10 周）。

---

## 附录：文献索引

| 简称 | 全称 | OpenAlex ID | 年份 | 引用数 |
|------|------|-------------|------|--------|
| VideoPose3D | 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training | W2903549000 | 2019 | 1273 |
| MotionBERT | MotionBERT: A Unified Perspective on Learning Human Motion Representations | W4390874423 | 2023 | 371 |
| TCPFormer | TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation | W4409366800 | 2025 | 30 |
| HDFormer | HDFormer: High-order Directed Transformer for 3D Human Pose Estimation | W4385767582 | 2023 | 66 |
| DiffPose | DiffPose: Toward More Reliable 3D Pose Estimation | W4386075813 | 2023 | 184 |
| AugLift | AugLift: Depth-Aware Input Reparameterization Improves Domain Generalization in 2D-to-3D Pose Lifting | W4414457910 | 2025 | 0 |
| Simple Baseline | A Simple Yet Effective Baseline for 3d Human Pose Estimation | W2612706635 | 2017 | 1518 |
| Attention-TCN | Attention Mechanism Exploits Temporal Contexts: Real-Time 3D Human Pose Reconstruction | W3034448411 | 2020 | 270 |
| MixSTE | MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video | W4312417903 | 2022 | 410 |
| Strided Transformer | Exploiting Temporal Contexts With Strided Transformer for 3D Human Pose Estimation | W4225557002 | 2022 | 302 |
| PoseMamba | PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model | W4409368373 | 2025 | 25 |
| PoseFormer | 3D Human Pose Estimation with Spatial and Temporal Transformers | W3136525061 | 2021 | 692 |
| BSTMamba | A Spatiotemporal Bidirectional Mamba Network with Global–Local Skeletal Enhancement | W4413980847 | 2025 | 0 |
| ConvFormer | ConvFormer: parameter reduction in transformer models for 3d human pose estimation | W4382892987 | 2023 | 23 |
| Fusionformer | Fusionformer: Exploiting the Joint Motion Synergy with Fusion | W4386706091 | 2023 | 1 |
| SBAHGNet | SBAHGNet: 3D Human Pose Estimation via Skeleton-Biased Attention and High-Frequency Enhanced Graph Convolution | W7128608150 | 2026 | 0 |
| MixTGFormer | Dual-stream Spatio-Temporal GCN-Transformer Network for 3D Human Pose Estimation | W7155247101 | 2026 | 6 |
| DDHPose | Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser | W4393158891 | 2024 | 46 |
| 3D=2D+Matching | 3D Human Pose Estimation = 2D Pose Estimation + Matching | W2583372902 | 2017 | 593 |
| Flowing ConvNets | Flowing ConvNets for Human Pose Estimation in Videos | W602397586 | 2015 | 583 |
| Linear Attention | Simple linear attention language models balance the recall-throughput tradeoff | arxiv:2402.18668 | 2024 | — |
| GDN | Gated Delta Networks: Improving Mamba2 with Delta Rule | arxiv:2412.06464 | 2024 | — |
| GLA | Gated Linear Attention Transformers with Hardware-Efficient Training | arxiv:2312.06635 | 2023 | — |
| GDN µP | Unlocking Feature Learning in Gated Delta Networks at Scale | arxiv:2606.04048 | 2026 | — |
