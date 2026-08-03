# 遮挡鲁棒的多人绝对3D位姿估计：几何先验增强的RootNet
> 技术可行性报告 · 2026-07-28 · idea: 遮挡鲁棒的多人绝对3D位姿估计_几何先验增强的RootNet.md · ReAct 写作（边写边查证 papers/cards/codebases） · model: qmodel_preview


> 毕业设计技术可行性报告
> 目标会议/答辩：本科毕业论文答辩（约4个月周期）
> 训练资源：实验室单卡（≤24 GB显存）

---

## 1. 背景与动机

### 1.1 问题陈述

多人绝对3D位姿估计要求从单目RGB图像/视频中同时恢复场景中每个人的完整三维关节坐标（相机坐标系），其核心难点在于**绝对根深度**的确定——即每人距相机的真实距离。Moon等人（2019）提出的3DMPPE框架（DetectNet→RootNet→PoseNet）以面积比距离度量 $k = \sqrt{\alpha_x \cdot \alpha_y \cdot A_{\text{real}} / A_{\text{img}}}$ 估计根深度 $Z_R$，成为该任务的标准top-down管线。然而该管线存在两个结构性瓶颈：

**瓶颈一：根深度在遮挡/截断下崩溃。** RootNet的距离度量依赖成像面积 $A_{\text{img}}$ 的准确估计。当人体被他人/物体遮挡或画面截断时，检测框面积严重偏离真实投影面积，导致深度估计失效。Moon等人（2019）原文图4已给出反例。Root-GAST-Net（2022）在MuPoTS-3D上的实验表明，RootNet原版的MRPE达289 mm，而其改进版仍为178 mm——根深度误差在绝对位姿总误差中占比超过50%。

**瓶颈二：单帧PoseNet无时序消歧能力。** 3DMPPE的PoseNet对每帧每人独立做2D→3D提升，无时序上下文。单目深度歧义（同一2D投影对应无穷多3D构型）在单帧条件下只能靠静态先验硬填。VNect（Mehta等，2017）在自遮挡活动（坐/躺）上PCK仅约48%；XNect（Mehta等，2019）重度遮挡时完全依赖Stage II学习先验，缺乏显式图像证据支撑。视频时序lifting已全面证明时序上下文的价值：VideoPose3D（Pavllo等，2019）以膨胀时间卷积将感受野扩至243帧，H36M MPJPE约46.8 mm（CPN检测2D，待验证——原卡仅给出GT 2D可降低22.6 mm）；MixSTE（Zhang等，2022）以交替时空Transformer达40.9 mm；PoseMamba（2025）以线性复杂度SSM达38.1 mm（CPN检测2D，T=243），较MotionBERT精度提升1.1 mm且仅用16%计算量。但3DMPPE原框架从未引入时序lifting。

**量化表现汇总：**

| 瓶颈 | 量化指标 | 来源 |
|------|---------|------|
| 根深度遮挡失效 | MRPE 289 mm（MuPoTS-3D） | Root-GAST-Net卡 |
| 自遮挡PCK | 约48%（VNect坐/躺活动） | VNect卡 |
| 单帧vs时序 | ~46.8→38.1 mm MPJPE（H36M CPN T=243） | VideoPose3D卡（46.8待验证）/PoseMamba卡 |
| 2D检测级联 | GT 2D可再降22.6 mm | VideoPose3D卡 |

### 1.2 相关工作

按技术路线分组，仅引用cards/中存在的论文。

**路线一：Top-down多人绝对位姿（根深度估计）。** Moon等（2019）的3DMPPE以面积比度量开创性地解耦了根深度与根相对姿态，在MuPoTS-3D上提出MRPE/3DPCKabs/AP_root_25评估协议，推理约0.141 s/帧（TitanX Maxwell）；SMAP（2020）以自底向上全卷积方式回归根深度图（视场角归一化），结合深度感知部件关联实现遮挡优先级排序，推理约57 ms/帧且不受人数影响，但无相机内参时RtError从23.3 cm骤升至67 cm；HMOR（2020）引入层次化多人序关系（实例级/部件级/关节级深度序）作为可微排序损失，在MuPoTS-3D上PCKabs提升12.3、CMU Panoptic平均关节误差改善20.5 mm；Root-GAST-Net（2022）将时序lifting（GAST-Net，27帧）引入top-down管线，在H36M上将MRPE从289降至178 mm，MuPoTS-3D上3DPCKabs达56.8%（超TDBU_Net 8.8 pp），AP_root_25达58.9%（超12.6 pp），速度15 fps（GTX 1080，TensorRT FP32）。HDNet以GNN做多人深度分类，但依赖人体平均尺度先验，对儿童或身材异常者偏差大。

| 方法 | 年份 | MuPoTS-3D关键指标 | 根深度策略 | 遮挡处理 |
|------|------|------------------|-----------|---------|
| 3DMPPE (Moon) | 2019 | MRPE 289 mm | 面积比 $k$ | 无 |
| SMAP | 2020 | RtError 23.3 cm | 根深度图（稀疏） | 深度排序遮挡优先 |
| HMOR | 2020 | PCKabs +12.3 | 层次序关系 | 训练时弱监督 |
| Root-GAST-Net | 2022 | MRPE 178 mm, 3DPCKabs 56.8% | 面积比（同RootNet） | 时序lifting 27帧 |
| **本方案** | — | 目标MRPE≤150 mm | 面积比+SMPL修正+序数损失 | GDN门控时序记忆 |

**路线二：视频2D→3D时序lifting。** VideoPose3D（2019）以膨胀因果卷积建立基线（243帧模型16.95M参数/33.87M FLOPs，推理~150k FPS/GP100），GT 2D可再降22.6 mm说明2D检测器是首要瓶颈；MixSTE（2022）以交替时空Transformer + seq2seq输出达CPN T=243下MPJPE 40.9 mm / P-MPJPE 32.6 mm（约33.7M参数，每帧645M FLOPs）；MotionBERT（2023）以双流DSTformer + 预训练-微调范式统一多任务，H36M SH检测2D下MPJPE约39.2 mm；PoseFormerV2（2023）利用DCT频域压缩实现任意感受野扩展且计算量不增，核心假设为"人体运动轨迹能量集中于低频分量"；PoseMamba（2025）以双向全局-局部SSM扫描达38.1 mm（CPN）/ 15.6 mm（GT 2D），较MotionBERT精度提升1.1 mm且仅用16%计算量，训练于单张RTX 3090（120 epochs, batch 4, AdamW lr=2e-4）。OAHPE（2026）引入遮挡感知双路径路由（VisNet遮挡检测+ORST-DLEP遮挡路径+PDREP可见路径），H36M MPJPE 43.5 mm，模型仅2.6M参数/显存12.5 GB，但其VisNet完全依赖2D置信度阈值τ=η=0.5，在严重遮挡时路由失效——这正是本方案要解决的核心问题。

**路线三：线性注意力/状态空间模型（NLP侧理论与工具）。** DeltaNet（Yang等，2024）通过delta规则（$v_{\text{new}} = v - h \cdot k$，即从value中减去state在key方向的投影）实现定向记忆替换，并以WY表示（Householder变换的紧凑乘积形式）实现分块并行训练，复杂度O(LCd + Ld²)；Gated Delta Networks（Yang等，2024）将Mamba2标量衰减α与delta规则统一为 $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，α→0快速清空、α→1精准更新，1.3B/100B token训练，但仍承认"固定状态维度限制检索容量，仍需混合注意力层弥补"；Titans（Behrouz等，2025）提出以关联记忆损失对输入的梯度衡量"惊讶度"决定记忆写入优先级，引入衰减机制管理有限容量，架构含Core（有限窗口注意力）+长期神经记忆+持久记忆参数三个超头；BASED（Arora等，2024）理论证明任何循环模型需Ω(N)比特状态才能完美解决MQAR，实验证实召回密集型任务线性模型仍显著落后（SWDE 48.06 vs全注意力71.92，FDA 24.41 vs 73.23）。fla库（flash-linear-attention）提供硬件高效的chunk_gated_delta_rule Triton kernel，支持chunk_size∈{16,32,64}，依赖torch≥2.7/triton≥3.3。

**路线四：几何先验与弱监督。** Ordinal Depth Supervision（Pavlakos等，2018）以关节间序数深度关系（closer/farther/equal）作为可微logistic排序损失训练ConvNet，标注效率约每图1分钟（17问×3.5s），在H36M/HumanEva上验证弱监督可大幅消解单目歧义，但作者承认"当架构更强时弱监督与全监督差距拉大"；HMR（Kanazawa等，2018）以SMPL参数化模型（85维：23关节轴角+10维PCA体型+弱透视相机）+因式分解对抗先验（25个判别器）约束人体流形，仅需2D标注即可训练；HybrIK指出"Naive HybrIK隐含假设预测骨骼长度等于模板骨骼长度…实际中常不成立"——这正是本方案将骨长建模为带不确定性软约束（而非硬常数）的动机来源。

### 1.3 根本性分析

3DMPPE管线的核心失效机制可归结为**信号缺失的级联正反馈**：

**第一步：遮挡使面积信号退化。** RootNet的距离度量 $k = \sqrt{\alpha_x \alpha_y A_{\text{real}} / A_{\text{img}}}$ 中，$A_{\text{img}}$ 由检测框面积近似。遮挡/截断时检测框仅覆盖可见部分，$A_{\text{img}}$ 系统性偏小→$k$偏大→$Z_R$偏小（人被判为更近）。这不是随机噪声而是**方向性系统偏差**，无法通过数据增强或更大backbone消除。

**第二步：单帧PoseNet无法自我修正。** 即使根深度有误差，若PoseNet有时序上下文，相邻帧的一致性约束可提供隐式修正信号（同一人连续帧的骨长/速度应平滑）。但3DMPPE的PoseNet是纯单帧的：每帧独立回归，无帧间信息流通，错误根深度直接传导为绝对坐标偏差且无法被下游修正。

**第三步：2D检测器在遮挡时反而更不可靠。** 现代top-down检测器（CPN/HRNet）在被遮挡关节上常输出高置信度的错误位置（"幻觉关节"）。OAHPE试图用2D置信度路由遮挡帧，但其VisNet在严重遮挡时路由失效——恰在最需要遮挡处理时，路由信号最不可靠。这构成恶性循环：越遮挡→2D越不准→路由越失效→3D越崩。

**为什么现有方法修不了？** Root-GAST-Net虽引入了时序（27帧GAST-Net），但其时序建模仅用于root-relative姿态lifting，根深度仍由单帧RootNet独立估计，两个模块完全解耦、无信息反馈。HMOR的序关系是训练时弱监督而非推理时修正机制。SMAP的根深度图是稀疏监督（仅根关节像素处），对根关节本身被遮挡的场景无能为力。根本缺口在于：**缺少一个能在遮挡期保护可靠记忆、在恢复期按需读写的时序机制，且该机制的路由信号不能依赖检测器自报置信度。**

---

## 2. 方法

本方案保留3DMPPE的DetectNet→RootNet→PoseNet三模块top-down结构，以三个互补贡献逐层加固：

### Contribution 1：3DMPPE基线复现与视频化扩展

**设计动机。** 3DMPPE（Moon等，2019）是多人绝对3D位姿的标准管线，官方代码（mks0601/3DMPPE_ROOTNET_RELEASE）开源且结构清晰。复现该基线是后续改进的对照基础。原框架为单帧设计（每帧独立检测→估计→输出），本贡献将其扩展为视频管线：在DetectNet后接入逐人跟踪（匈牙利算法/ByteTrack），为每人维护独立的2D骨架时间序列，供后续Contribution 2/3使用。

**技术细节.**

3DMPPE三模块架构（Moon等，2019）：
- **DetectNet**：Mask R-CNN人体检测，输出2D bbox。本方案替换为HRNet-W32（deep-high-resolution-net.pytorch），提供更精确的2D关键点（含confidence）；
- **RootNet**：ResNet-50/152 backbone，输入单人crop图像，分别预测根关节2D像素坐标 $(u_R, v_R)$ 和绝对深度 $Z_R$。深度通过距离度量 $k = \sqrt{\alpha_x \cdot \alpha_y \cdot A_{\text{real}} / A_{\text{img}}}$ 学习，其中 $\alpha_x, \alpha_y$ 为焦距归一化因子。三模块采用分离式训练（非联合），因"两任务相关性不高，分离训练精度更高"（camera_distance_aware卡）；
- **PoseNet**：ResNet-50 backbone，输入单人crop + 根关节坐标，输出root-relative 3D姿态（17关节×3坐标）。

管线结构（视频化扩展后）：
```
视频帧 → DetectNet(HRNet-W32) → 2D bbox + keypoints + confidence
       → Tracker(匈牙利算法) → 逐人轨迹 {person_i: [(x,y,conf)_t]_{t=1}^T}
       → RootNet(逐人逐帧) → Z_R(t), (u_R, v_R)(t) 绝对根深度与2D位置
       → PoseNet/GDN-Lifter(逐人) → root-relative 3D sequence
       → 反投影: X_abs = X_rel · Z_R / f + [u_R, v_R, Z_R]^T
```

RootNet根深度估计的核心公式（Moon等，2019）：

$$Z_R = \frac{f \cdot \sqrt{A_{\text{real}}}}{\sqrt{A_{\text{img}}} \cdot k_{\text{pred}}}$$

其中 $f$ 为焦距，$A_{\text{real}}$ 为人体真实面积（训练时从GT 3D投影计算），$k_{\text{pred}}$ 为网络预测的距离度量修正因子。

**视频化扩展的设计选择：**
- 跟踪算法选择匈牙利算法（而非更复杂的DeepSORT/ByteTrack），因为MuPoTS-3D最多3人/场景，简单算法即可胜任，且不引入额外GPU开销；
- 轨迹缓存长度设为T=243帧（与PoseMamba训练窗口一致），超出时滑动窗口截断；
- 身份关联失败（新目标出现/旧目标消失）时，GDN hidden state初始化为零向量（等价于无历史记忆），不强制续接。

**与现有系统的衔接.** DetectNet使用HRNet-W32（deep-high-resolution-net.pytorch仓库），RootNet使用官方RELEASE权重（ResNet-50 backbone，MuCo-3DHP训练）。跟踪模块为轻量级匈牙利算法，不引入额外训练开销。视频化扩展不改变RootNet/PoseNet的模型结构，仅增加帧间身份关联。

**可回退设计.** 本贡献为纯工程集成，不修改任何模型结构。若后续Contribution 2/3的改进模块失效，可直接回退到本基线的逐帧推理模式，性能下界即为3DMPPE原版精度（MRPE 289 mm / 3DPCKabs约44%，camera_distance_aware卡）。

### Contribution 2：RootNet遮挡鲁棒改进（SMPL面积修正 + 序数深度监督）

**设计动机.** Contribution 1的分析表明，RootNet失效的根源是 $A_{\text{img}}$ 在遮挡/截断下的系统性偏小。现有方法（HMOR序关系、SMAP根深度图）均未直接修正面积信号本身。本贡献从两个方向加固：(a) 用SMPL体型先验提供遮挡无关的 $A_{\text{real}}$ 估计修正；(b) 用序数深度监督为RootNet提供不依赖面积精度的辅助深度信号。

**技术细节.**

**(2a) SMPL面积修正.** 核心思想：当检测框面积 $A_{\text{img}}$ 因遮挡不可信时，用SMPL模型的投影面积作为替代/修正信号。

具体实现：
1. 在训练阶段，对每个GT 3D姿态拟合SMPL参数 $(\theta, \beta)$（使用SMPLify或HybrIK的逆向运动学），得到体型参数 $\beta$；
2. 从 $\beta$ 计算该体型的标准投影面积 $A_{\text{SMPL}}(\beta, \text{pose})$（在给定姿态下SMPL mesh的2D凸包面积）；
3. 定义面积可信度权重：

$$w_A = \sigma\left(\lambda_A \cdot \frac{A_{\text{img}}}{A_{\text{SMPL}}} - \tau_A\right)$$

当 $A_{\text{img}} / A_{\text{SMPL}} \approx 1$ 时 $w_A \to 1$（面积可信），当比值远小于1（严重遮挡/截断）时 $w_A \to 0$；

4. 修正后的有效面积：$A_{\text{eff}} = w_A \cdot A_{\text{img}} + (1 - w_A) \cdot A_{\text{SMPL}}$。

在推理时，$\beta$ 由RootNet的backbone特征经一个轻量MLP头预测（2层，隐藏维256），训练标签来自离线SMPL拟合。

**接口约定：** SMPL面积修正模块以plugin形式接入RootNet的面积计算环节，不改变RootNet的网络结构。当 $w_A > 0.9$ 时修正量 $< 1\%$，等价于无修正（加法式改造）。

**(2b) 序数深度监督.** 借鉴Pavlakos等（2018）的序数深度思想，为RootNet提供不依赖面积绝对精度的相对深度约束。

在多人场景中，同一帧内不同人的根深度存在确定的远近序关系。定义可微排序损失：

$$\mathcal{L}_{\text{ord}} = \sum_{(i,j) \in \mathcal{P}} \log\left(1 + \exp\left(-(Z_R^{(i)} - Z_R^{(j)}) \cdot s_{ij}\right)\right)$$

其中 $\mathcal{P}$ 为同帧内所有人对，$s_{ij} = \text{sign}(Z_{\text{GT}}^{(i)} - Z_{\text{GT}}^{(j)})$ 为GT序关系标签。该损失仅约束相对顺序，不要求绝对深度精确，因此在面积信号退化时仍能提供梯度方向。

训练时总损失：$\mathcal{L}_{\text{RootNet}} = \mathcal{L}_{\text{depth}} + \lambda_{\text{ord}} \mathcal{L}_{\text{ord}} + \lambda_{\text{SMPL}} \mathcal{L}_{\beta}$，其中 $\mathcal{L}_{\beta}$ 为SMPL体型预测的MSE损失。

**与现有系统的衔接.** 序数深度监督仅在训练时生效（多人合成数据MuCo-3DHP提供多人同帧GT），推理时不增加计算。SMPL面积修正的MLP头参数量约0.13M，推理额外开销可忽略。

**可回退设计.** SMPL面积修正通过 $w_A$ 门控：初始化 $\lambda_A = 0$ 使 $w_A = 0.5$（半修正），若训练中发现SMPL修正引入负迁移，可将 $\lambda_A$ 固定为0使模块退化为恒等映射。序数深度为纯训练时辅助损失，推理时零开销，去除即回退。

### Contribution 3：GDN门控Delta规则线性注意力时序Lifting（PoseNet替换）

**设计动机.** 3DMPPE的PoseNet为单帧逐关节回归，无时序消歧能力。Contribution 2加固了根深度，但root-relative姿态本身在遮挡帧仍需时序上下文修正。本贡献将PoseNet替换为基于Gated Delta Networks（GDN）的时序lifter，利用delta规则的"选择性记忆+快速遗忘"语义处理遮挡：记住可见时的可靠估计，在观测不可信时冻结记忆而非被污染。

**技术细节.**

**(3a) 主干架构（承PoseMamba）.** 输入为每人跟踪的2D骨架序列 $(B, T, J{=}17, 2)$，root-centered、归一化至 $[-1, 1]$（与PoseMamba数据预处理一致，参见PoseMamba repo卡 `datareader_h36m.py:36`）。交替堆叠空间GDN块（沿关节维 $J$ 扫描）与时间GDN块（沿帧维 $T$ 扫描），每块为pre-norm残差结构（与PoseMamba `mambablocks.py:676-680` 的 `BiSTSSMBlock._forward` 同构）：

```python
# 伪代码：GDN时空块
def forward(self, x):          # x: (B, T, J, C)
    x = x + self.drop_path(self.gdn_op(self.norm(x)))   # GDN分支
    x = x + self.drop_path(self.mlp(self.norm2(x)))      # MLP分支
    return x
```

空间块输入 $(B \cdot T, J, C)$，序列长度 $L = J = 17$；时间块输入 $(B \cdot J, T, C)$，序列长度 $L = T$（81或243）。输出 $(B, T, J, 3)$ root-relative序列，再与RootNet的 $Z_R$ 反投影组合得绝对3D。

**(3b) GDN核心算子（引用fla repo卡）.** 每个GDN块的核心调用为fla库的 `chunk_gated_delta_rule`（`fla/ops/gated_delta_rule/chunk.py:397-588`）：

```python
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# q, k, v: (B, L, H, D_head)  g: (B, L, H)  beta: (B, L, H)
o, final_state = chunk_gated_delta_rule(q, k, v, g, beta,
                                         chunk_size=64)  # 允许值: {16, 32, 64}
```

其递推语义（naive参考实现，`fla/ops/gated_delta_rule/naive.py:50-59`）：
```
for each timestep i:
    h = h * exp(g_i)                          # 门控衰减
    v_delta = v_i - (h * k_i).sum(dim=-2)     # delta: 减去当前state在k方向的投影
    v_delta = v_delta * beta_i                # beta缩放（写入强度）
    h = h + outer(k_i, v_delta)               # 外积更新
    o_i = einsum('hd,hdm->hm', q_i, h)       # 查询输出
```

门控衰减公式（`fla/ops/gated_delta_rule/gate.py:96-104`）：$g = -\exp(A_{\log}) \cdot \text{softplus}(g_{\text{input}} + dt_{\text{bias}})$，chunk内做cumsum。

**chunk_size约束**（`fla/ops/gated_delta_rule/chunk.py:535-536`）：GDN仅允许chunk_size ∈ {16, 32, 64}，本方案对时间块（$T$=81/243）取64，对空间块（$J$=17）取16。

**(3c) 双向扫描实现.** fla的chunk kernel硬编码因果（下三角）mask（`fla/ops/common/chunk_o.py:125-126`：`m_A = (o_t[:, None] >= o_t[None, :])`），不直接支持双向。本方案采用与PoseMamba CrossMerge同构的策略（PoseMamba repo卡 `csms6s.py:170-192`）：正向+反向各跑一次 `chunk_gated_delta_rule`，输出逐元素求和：

```python
# 双向GDN
o_fwd, _ = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=C)
o_bwd, _ = chunk_gated_delta_rule(q.flip(1), k.flip(1), v.flip(1),
                                   g.flip(1), beta.flip(1), chunk_size=C)
o = o_fwd + o_bwd.flip(1)  # CrossMerge: sum融合
```

这与PoseMamba的 `CrossMerge_plus_poselimbs`（`csms6s.py:186`：`ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])`）完全同构，已验证有效。

**(3d) 解耦双门控（核心创新）.** 将GDN的写入系数beta解耦为两级门控：

**观测质量门 $\beta_{\text{obs}}$（双通道关断）：**

$$\beta_{\text{conf}} = \sigma(w_1 \cdot \text{conf}_{2d} + w_2 \cdot \text{bbox\_area\_ratio} + w_3 \cdot \text{vis\_flag} + b)$$

$$\beta_{\text{geo}} = \sigma(-w_4 \cdot \text{bone\_strain} - w_5 \cdot \|\mathbf{v}_t - \mathbf{v}_{t-1}\| / \sigma_v - w_6 \cdot e_{\text{reproj}} + b')$$

$$\beta_{\text{obs}} = \min(\beta_{\text{conf}},\; \beta_{\text{geo}})$$

其中：
- bone_strain $= \max_i \left| \frac{\|j_i - j_{\text{parent}(i)}\|_t}{\|j_i - j_{\text{parent}(i)}\|_{t-1}} - 1 \right|$（逐骨计算取max）
- $e_{\text{reproj}}$：当前2D与记忆3D经 $Z_R$ 重投影的像素误差
- 取min确保**任一通道判定不可信即关断写入**

通道B（几何一致性）的关键作用：当检测器输出高置信度但位置错误的"幻觉关节"时（conf > 0.8但GT标记为occluded），通道A不会关断，但骨长应变/重投影误差会异常大→$\beta_{\text{geo}} \to 0$→$\beta_{\text{obs}} \to 0$，兜底保护记忆。

**状态惊讶门 $g_{\text{state}}$（仅在可靠帧上计算）：**

当 $\beta_{\text{obs}} > \theta_{\text{obs}}$（可学习阈值，初始化0.5）时：

$$r = \|v_{\text{new}} - h \cdot k\| \quad \text{（delta预测残差）}$$
$$g_{\text{in}} = f_{\text{MLP}}(r,\; \Delta r / \Delta t,\; \text{bone\_strain})$$

高 $r$ 且 $r$ 持续上升（真运动突变）→ $g$ 驱动 $\alpha \to 0$ 定向替换过时记忆；高 $r$ 但单帧脉冲（检测噪声）→ $\alpha$ 保持。

**有效写入系数：** $\beta_{\text{eff}} = \beta_{\text{obs}} \cdot \beta_{\text{learned}}$，乘入 `chunk_gated_delta_rule` 的 `beta` 参数。当 $\beta_{\text{obs}} \to 0$ 时，delta规则退化为 $h_{t+1} = \alpha \cdot h_t$（纯衰减保持），记忆不被不可靠观测污染。

**(3e) 遮挡恢复逻辑.** 当 $\beta_{\text{obs}}$ 从 $< \theta_{\text{obs}}$ 回升至 $> \theta_{\text{obs}}$（关节重新可见）：
- 第一帧强制 $\beta_{\text{eff}} = 1$
- 若 $r$ 小（姿态未变）：$\alpha \to 1$，直接读回冻结期记忆（零代价恢复）
- 若 $r$ 大（遮挡期间姿态改变）：$\alpha \to 0$，快速改写

**(3f) 几何先验不确定性加权.** 骨长恒定、RootNet根深度、2D重投影先验均不做硬约束，按各自不确定性加权为软损失。在 $\beta_{\text{obs}} < \theta_{\text{obs}}$ 的冻结期，几何先验（骨长约束+速度惯性）作为记忆位置的唯一外推修正源，修正幅度按不确定性倒数加权。

**(3g) 训练损失（承PoseMamba）.** GDN lifter的训练损失沿用PoseMamba的多项组合（PoseMamba repo卡 `train.py:199-229`）：

$$\mathcal{L} = \lambda_{3d}\mathcal{L}_{\text{MPJPE}} + \lambda_{\text{scale}}\mathcal{L}_{\text{N-MPJPE}} + \lambda_{\text{vel}}\mathcal{L}_{\text{velocity}} + \lambda_{\text{diff}}\mathcal{L}_{\text{diff}} + \lambda_{\text{lv}}\mathcal{L}_{\text{limb\_var}} + \lambda_{\text{ord}}\mathcal{L}_{\text{ord}}$$

默认权重（PoseMamba-L配置）：$\lambda_{3d}=1.0, \lambda_{\text{scale}}=0.5, \lambda_{\text{vel}}=20.0, \lambda_{\text{diff}}=0.5$，其余为0。其中 $\mathcal{L}_{\text{MPJPE}}$ 为逐关节L2欧氏距离（`loss.py:56-63`），$\mathcal{L}_{\text{velocity}}$ 为时间差分速度损失，$\mathcal{L}_{\text{limb\_var}}$ 为骨长方差损失（鼓励骨长恒定）。本方案额外加入 $\mathcal{L}_{\text{ord}}$（序数深度，仅多人训练时）和门控正则项 $\mathcal{L}_{\text{gate}} = \|\beta_{\text{obs}}\|_1$（鼓励稀疏关断，防止门控常开退化）。

所有损失函数期望输入形状 $(N, T, 17, 3)$（`loss.py:71`），与GDN lifter的输出形状 $(B, T, J, 3)$ 一致，无需适配。

**与现有系统的衔接.** 主干结构完全承PoseMamba（`PoseMamba.py:37-141` 的交替STE/TTE架构），仅将内部 `BiSTSSM` 算子（`mambablocks.py:582`）替换为GDN算子。PoseMamba主干的输入输出形状 $(B, T, J, C)$ 不变，`PoseMamba.py` 无需修改（PoseMamba repo卡确认）。fla库要求torch≥2.7、triton≥3.3（fla repo卡环境节）。

**可回退设计.** GDN时序lifter以独立PoseNet形式存在：若GDN训练不收敛或精度低于基线，可直接切回3DMPPE原版单帧PoseNet（ResNet-50），系统退化为Contribution 1+2的水平。门控模块的 $\theta_{\text{obs}}$ 初始化为0.5、$\beta_{\text{learned}}$ 初始化为1，等价于标准GDN（zero-init门控偏移），训练初期不引入额外不稳定性。

---

## 3. 实验计划

### 3.1 评估指标

| 指标 | 定义 | 当前基线值 | 目标值 | 预期改进幅度 |
|------|------|-----------|--------|-------------|
| MPJPE (H36M P1, CPN) | root-relative平均关节位置误差 | 38.1 mm (PoseMamba-L) | ≤38 mm | ≥持平 |
| MPJPE (H36M P1, GT 2D) | 同上，GT 2D输入 | 15.6 mm (PoseMamba-L) | ≤16 mm | ≥持平 |
| P-MPJPE (H36M P2) | Procrustes对齐后误差 | 32.6 mm (MixSTE) | ≤30 mm | ≥8%↓ |
| MRPE (MuPoTS-3D) | 平均根位置误差 | 289 mm (RootNet) / 178 mm (Root-GAST-Net) | ≤150 mm | ≥15%↓ vs Root-GAST-Net |
| 3DPCKabs (MuPoTS-3D) | 绝对PCK@150mm | 56.8% (Root-GAST-Net) | ≥62% | ≥5 pp |
| 遮挡段恢复误差（中/长） | 被mask关节在遮挡恢复后8帧内的MPJPE | 待验证（无直接基线） | 较混合门控↓≥5% | 核心判据 |
| 幻觉遮挡子集误差 | conf>0.8但GT occluded帧的恢复误差 | 待验证 | 较纯通道A↓≥5% | 通道B判据 |
| 记忆保持率 | 冻结期记忆向量与遮挡前最后帧的余弦相似度 | — | >0.85 | 混合组预期<0.5 |

### 3.2 消融矩阵

| 编号 | 消融配置 | 验证假设 | 预期效果 |
|------|---------|---------|---------|
| Full | 完整方案（C1+C2+C3） | — | 最优 |
| (a) 混合惊讶度 | 去解耦，回退为单一混合门控 | 范畴错误修正是否有效 | 中长遮挡段误差↑≥3% |
| (b) 去观测质量门 | 仅保留状态惊讶门 | 遮挡期记忆保护是否必要 | 遮挡段记忆保持率↓↓ |
| (b2) 去几何通道B | $\beta_{\text{obs}}$仅由conf/area驱动 | 检测器幻觉兜底是否必要 | 幻觉子集误差↑≥5% |
| (c) 纯因果 | 去双向扫描 | 双向时序信息的贡献 | MPJPE↑1-2 mm |
| (d) 硬先验 | 去不确定性加权，骨长/深度做硬约束 | 软加权vs硬约束 | 遮挡期误差↑ |
| (e) 纯Mamba/SSM | GDN换Mamba selective_scan | 隔离delta规则贡献 | 遮挡恢复误差↑ |
| (f) θ固定0/0.5/1 | 可学习阈值退化为固定 | 自适应阈值必要性 | θ=0全开放≈(b)；θ=1全冻结无更新 |
| **Oracle上界** | GT 2D + GT根深度 + GT遮挡mask | 系统理论上限 | 误差趋近GT lifting下界 |
| **Negative control** | 随机打乱时序顺序 | 时序信息是否真被利用 | MPJPE退化至单帧水平 |

### 3.3 基线方法

| 基线 | 类型 | 来源 |
|------|------|------|
| 3DMPPE原版（单帧RootNet+PoseNet） | top-down绝对位姿 | Moon 2019, cards/ |
| Root-GAST-Net（时序+RootNet） | top-down绝对位姿 | cards/ |
| VideoPose3D | 时序lifting（膨胀卷积） | Pavllo 2019, cards/ |
| MixSTE | 时序lifting（Transformer） | Zhang 2022, cards/ |
| PoseMamba-L | 时序lifting（SSM） | 2025, cards/ |
| MotionBERT | 时序lifting（预训练Transformer） | Zhu 2023, cards/ |
| OAHPE | 遮挡感知路由 | 2026, cards/ |

### 3.4 数据集要求与预处理

**Human3.6M（主要lifting评估）：**
- 划分：S1/5/6/7/8训练（约3.6M帧），S9/11测试（17关节，15类动作）
- 2D输入：CPN检测（含confidence通道，原始形状 $(T, 17, 3)$，配置 `no_conf: True` 时只取xy）+ GT 2D两组
- 切片：T=81（训练stride=27）/ T=243（训练stride=81约1/3重叠，测试stride=243不重叠）（PoseMamba repo卡 `datareader_h36m.py:100-107`）
- 预处理：2D坐标按相机分辨率映射至[-1,1]（`datareader_h36m.py:36`：`trainset[idx] / res_w * 2 - [1, res_h/res_w]`），3D做root-relative（`train.py:181`：`batch_gt = batch_gt - batch_gt[:,:,0:1,:]`）
- 排除3个测试序列：`s_09_act_05_subact_02`, `s_09_act_10_subact_02`, `s_09_act_13_subact_01`（`train.py:121-123`）
- 相机分辨率：1000×1002 / 1000×1000（`datareader_h36m.py:31-33`）
- 训练增强：随机水平翻转（`flip: True`）；测试时翻转增强（test-time augmentation，`train.py:71-75`）

**MuCo-3DHP + MuPoTS-3D（多人绝对位姿评估）：**
- MuCo-3DHP：40万合成多人帧（训练），由MPI-INF-3DHP合成多人场景
- MuPoTS-3D：20个户外真实场景（测试），最多3人/场景，含室内外混合
- 指标：MRPE（平均根位置误差）/ 3DPCKabs（绝对PCK@150mm）/ AP_root_25（根深度25cm阈值精度）
- 评估使用官方脚本，GT相机内参

**遮挡增强（合成，训练时在线生成）：**
- 随机mask：按概率p=0.3随机mask关键点/帧/区域（承P-STMO掩码预训练思路，P-STMO发现90%时间掩码率才有效暗示姿态序列高度冗余）
- 结构化遮挡：模拟另一人bbox经过导致的连续关节丢失，持续8-32帧。设计理由：区分检测噪声型单帧丢失（随机mask）与真遮挡型连续退化（结构化），后者是GDN门控记忆保护机制的核心验证场景
- 幻觉遮挡子集：在结构化遮挡帧上保留检测器对错误位置的高置信度输出（conf>0.8但GT标记occluded）。构造方法：从CPN检测器在遮挡序列上的实际输出中筛选高置信但高误差（>50 px）的帧。专门压力测试 $\beta_{\text{obs}}$ 通道B的兜底能力——若仅用通道A（conf/area），此子集上系统完全失效

### 3.5 评估协议

1. **H36M标准协议：** Protocol #1（MPJPE，root-relative，单位mm）+ Protocol #2（P-MPJPE，Procrustes对齐后计算）。评测时预测序列中心帧的root置零（`train.py:79`：`predicted_3d_pos[:,:,0,:] = 0`）。测试时翻转增强（对输入序列水平翻转后推理，输出翻转回来取平均）。
2. **遮挡鲁棒性协议（本方案新增）：** 在测试序列上施加结构化遮挡（训练时未见过的遮挡模式），按持续长度分桶（短<8帧/中8-32帧/长>32帧），报告：(a) 遮挡期间被mask关节的MPJPE；(b) 遮挡恢复后8帧内的MPJPE（衡量记忆恢复质量）；(c) 遮挡期记忆保持率（冻结期GDN hidden state与遮挡前最后一帧的余弦相似度）。
3. **幻觉遮挡子集协议（本方案新增）：** 仅在conf>0.8但GT occluded的帧上计算恢复误差，隔离通道B的独立贡献。预期：完整方案 vs 去通道B（消融b2）在此子集上差异≥5%。
4. **MuPoTS-3D协议：** 使用官方评估脚本，报告MRPE/3DPCKabs/AP_root_25。使用GT相机内参。
5. **效率协议：** 参数量(M)、MACs(G)、推理吞吐(fps)，单卡RTX 3090，batch=1，T=81/243分别报告。使用torch.cuda.Event计时，warmup 10次后取100次平均。

### 3.6 计算资源估算表

| 阶段 | 模型 | 数据 | GPU | 显存峰值 | 预计时长 |
|------|------|------|-----|---------|---------|
| C1: 3DMPPE复现（RootNet训练） | ResNet-50 | MuCo-3DHP 400K | 1×24GB | ~8 GB | ~12 h |
| C1: 3DMPPE复现（PoseNet训练） | ResNet-50 | H36M 3.6M帧 | 1×24GB | ~10 GB | ~24 h |
| C2: RootNet改进（+SMPL+序数） | ResNet-50 + MLP头 | MuCo-3DHP | 1×24GB | ~9 GB | ~14 h |
| C3: GDN lifter（S规模，调试） | ~4M params | H36M T=81 | 1×24GB | ~6 GB | ~6 h |
| C3: GDN lifter（B规模，正式） | ~12M params | H36M T=243 | 1×24GB | ~12 GB | ~36 h |
| C3: GDN lifter（L规模，若资源允许） | ~30M params | H36M T=243 | 1×24GB | ~20 GB | ~72 h |
| 消融实验（8组×B规模） | ~12M params | H36M T=243 | 1×24GB | ~12 GB | ~12 天 |
| MuPoTS-3D评估 | 全管线 | 20序列 | 1×24GB | ~10 GB | ~2 h |

注：PoseMamba-L在单张RTX 3090上训练120 epochs（PoseMamba卡），本方案GDN-B规模（dim=64, depth=8）参数量远小于PoseMamba-L（dim=128, depth=20），显存与时长应在24GB卡预算内。

---

## 4. 可行性评估

### 4.1 实现复杂度

| 组件 | 工程量 | 与更轻替代路线的对比 |
|------|--------|---------------------|
| C1: 3DMPPE复现+视频化 | 中（集成4个仓库） | 若仅复现单帧版，工程量×0.5 |
| C2: SMPL面积修正 | 低（1个MLP头+离线拟合） | 若用固定平均体型替代SMPL，工程量×0.3但精度损失 |
| C2: 序数深度监督 | 低（1个辅助损失） | 无更轻替代 |
| C3: GDN lifter主干 | 中（替换PoseMamba SSM→GDN） | 若直接用PoseMamba原版不做替换，工程量×0但无创新 |
| C3: 解耦双门控 | 高（自定义逻辑+逐帧几何计算） | 若用单一门控（消融a），工程量×0.6 |
| C3: 双向扫描 | 低（正反向各跑一次+sum） | 若纯因果（消融c），工程量×0.8 |

**总复杂度倍数：** 完整方案约为"仅复现3DMPPE"的3.5倍工程量。核心增量来自C3的解耦门控（约占总工程量40%）。

### 4.2 外部依赖风险表

| 依赖 | 版本要求 | 风险等级 | 缓解措施 |
|------|---------|---------|---------|
| fla (flash-linear-attention) | torch≥2.7, triton≥3.3 | 中 | 若triton编译失败，退回naive PyTorch实现（`fla/ops/gated_delta_rule/naive.py`），速度慢但功能等价 |
| 3DMPPE_ROOTNET_RELEASE | PyTorch 1.x兼容 | 低 | 官方代码稳定，无活跃维护风险 |
| HRNet (deep-high-resolution-net) | PyTorch ≥1.1 | 低 | 成熟仓库，仅用预训练权重 |
| PoseMamba | PyTorch 1.13+, CUDA 11.7 | 中 | 需编译selective_scan CUDA扩展；若替换为GDN则可移除该依赖 |
| SMPL模型 | 需注册下载 | 低 | 学术许可，离线使用 |
| MuCo-3DHP数据 | ~50 GB | 低 | 公开下载 |

### 4.3 错误传播风险

**级联结构：** DetectNet → Tracker → RootNet → GDN Lifter → 反投影

| 环节 | 失效模式 | 下游影响 | 最坏情况 | 缓解措施 |
|------|---------|---------|---------|---------|
| DetectNet漏检 | 某人无bbox | 该人完全丢失 | 无法补救（top-down固有限制） | 降低conf阈值（牺牲精度换召回） |
| Tracker ID切换 | 两人轨迹互换 | 时序记忆被错误身份污染 | GDN记忆写入错误序列→遮挡恢复完全错误 | 交叉遮挡时冻结GDN记忆更新 |
| RootNet深度崩溃 | 遮挡时$A_{\text{img}}$严重偏小 | 绝对坐标整体偏移 | MRPE退化至>300 mm（接近无修正基线） | C2 SMPL面积修正兜底 |
| GDN门控失效 | $\beta_{\text{obs}}$误判（该关未关） | 不可靠观测写入记忆 | 退化至无门控GDN（≈消融b，仍优于单帧） | 通道B几何兜底 |
| 几何通道B误触发 | 快速运动被误判为几何违反 | 有效帧被错误冻结 | 退化至纯惯性外推（短时<8帧影响有限） | bone_strain阈值可学习 |
| fla kernel数值异常 | Triton编译/精度问题 | 训练不收敛 | 无法使用GDN | 退回naive.py PyTorch实现 |

**最坏情况分析：** 若C3的GDN lifter完全不收敛或精度低于单帧PoseNet，系统可结构性回退至C1+C2（RootNet改进+单帧PoseNet），此时性能下界为3DMPPE原版+SMPL面积修正，MRPE预期约250 mm（较原版289 mm仍有改善）。若C2也失效，回退至C1纯基线（MRPE 289 mm）。回退路径为加法式：每个Contribution独立可拆除，不存在"拆一个必须拆全部"的耦合。

**结构性可回退保证的设计细节：**
- C2的SMPL面积修正通过 $w_A$ 门控实现加法式接入：$\lambda_A = 0$ 时 $A_{\text{eff}} = A_{\text{img}}$（恒等），训练全程可动态关闭；
- C2的序数深度为纯辅助损失（$\lambda_{\text{ord}} = 0$ 即完全移除），不影响网络结构；
- C3的GDN lifter作为独立PoseNet模块，与RootNet通过标准接口（$Z_R$ + 2D keypoints → 绝对3D）连接，切回原版PoseNet仅需替换一个模块文件；
- C3内部的解耦门控以 $\beta_{\text{eff}} = \beta_{\text{obs}} \cdot \beta_{\text{learned}}$ 乘入，若将 $\beta_{\text{obs}}$ 固定为1则退化为标准GDN（无门控），若将 $\beta_{\text{learned}}$ 固定为0则退化为纯衰减（无写入）——两个极端均为有意义的退化模式而非崩溃。

**Tracker ID切换的量化风险：** MuPoTS-3D最多3人/场景（camera_distance_aware卡），交叉遮挡频率有限。H36M为单人数据集，训练阶段无ID切换问题。推理时若使用GT bbox（Protocol #1/2评估），跟踪误差为零。实际部署中ID切换主要影响GDN记忆的连续性，但由于门控的冻结机制（$\beta_{\text{obs}} < \theta_{\text{obs}}$ 时不更新），切换瞬间的错误写入被限制在1-2帧内。

### 4.4 性能/成本量化

**推理开销逐组件预算表（单帧，B规模模型，RTX 3090）：**

| 组件 | 预计耗时 | 占比 | 备注 |
|------|---------|------|------|
| HRNet 2D检测 | ~30 ms | 45% | 使用官方预训练权重 |
| 跟踪（匈牙利） | ~2 ms | 3% | CPU |
| RootNet根深度 | ~8 ms | 12% | ResNet-50，单人crop |
| SMPL面积MLP | <0.5 ms | <1% | 2层MLP |
| GDN Lifter（B，T=81） | ~12 ms | 18% | chunk_size=64，双向×2 |
| 门控计算（bone_strain等） | ~1 ms | 1.5% | 逐帧几何量，纯tensor运算 |
| 反投影+后处理 | <0.5 ms | <1% | — |
| **总计（单人）** | **~54 ms** | — | **约18 fps** |
| **3人场景** | **~70 ms** | — | **约14 fps**（RootNet/GDN逐人） |

**对吞吐的影响：** 相比3DMPPE原版（~0.141 s/帧，约7 fps，TitanX Maxwell），本方案在RTX 3090上预期14-18 fps，主要得益于GDN线性复杂度（vs原版PoseNet的ResNet-50逐人推理）。

**优化后预期：** 若使用TensorRT FP16量化HRNet+RootNet（参考Root-GAST-Net的TensorRT优化），总延迟可压至~40 ms/帧（25 fps）。GDN的Triton kernel本身已为推理优化设计。

### 4.5 时间线里程碑表

| 周次 | 阶段 | 交付物 | 可验收指标 | 风险与Go/No-Go |
|------|------|--------|-----------|---------------|
| W1-2 | 环境搭建+数据准备 | H36M/MuCo-3DHP/MuPoTS-3D就绪；fla/PoseMamba/3DMPPE环境跑通 | fla kernel单元测试通过（`pytest tests/ops/test_gdn.py`）；PoseMamba官方权重复现MPJPE≤39 mm | **Go条件：** fla在目标GPU上编译成功。若失败→切换naive实现，进度不受影响 |
| W3-4 | C1: 3DMPPE复现 | RootNet+PoseNet训练完成；视频化跟踪集成 | MuPoTS-3D MRPE≤300 mm, 3DPCKabs≥45% | **Go条件：** MRPE≤320 mm即可继续。若>350 mm→检查数据预处理/相机参数 |
| W5-6 | C2: RootNet改进 | SMPL面积修正+序数深度监督训练完成 | MRPE较C1↓≥10%（≤270 mm） | **Go条件：** 若SMPL修正无改善（<3%），保留序数损失、放弃面积修正，不影响C3 |
| W7-8 | C3: GDN lifter（S规模调试） | 主干跑通，H36M T=81训练收敛 | MPJPE≤50 mm（S规模，CPN） | **关键决策点：** 若MPJPE>55 mm→执行路径A回退（用PoseMamba原版SSM） |
| W9-10 | C3: GDN lifter（B规模正式） | 完整门控+双向，H36M T=243 | MPJPE≤40 mm（CPN），遮挡恢复误差可量化 | 若门控训练不稳定→先固定θ_obs=0.5跑消融，再尝试可学习 |
| W11-12 | 消融实验 | 8组消融+oracle+negative control | 核心判据：解耦vs混合↓≥3%；通道B幻觉子集↓≥5% | 若核心判据不满足→论文贡献重定位为C1+C2，C3作为negative result |
| W13-14 | MuPoTS-3D全管线评估+论文撰写 | 完整结果表+可视化 | MRPE≤150 mm, 3DPCKabs≥62% | 若MRPE>200 mm→调整目标为"优于Root-GAST-Net"（≤178 mm） |
| W15-16 | 论文修改+答辩准备 | 终稿+PPT | 导师审阅通过 | — |

**关键路径分析：** 最长路径为 W1→W3→W7→W9→W11→W13→W15（环境→复现→GDN调试→正式训练→消融→评估→写作），共16周无冗余。缓冲策略：C2与C3-S规模可并行（W5-8），若C2提前完成则W7全力投入GDN。

### 4.6 综合判级与决策路径

**综合判级：B+（可行，有明确创新点，工程风险可控）**

- 创新性：解耦双门控（观测质量门+状态惊讶门）在姿态领域无先例（idea评审查重0命中），且"范畴错误修正"（NLP惊讶度≠姿态惊讶度）有清晰理论动机。
- 工程可行性：fla库提供成熟kernel，PoseMamba提供验证过的时空架构，替换路径已被repo卡逐行锚定。主要风险在四仓库集成的调试链路。
- 资源匹配：B规模模型（~12M参数）在24GB卡上训练无压力，120 epochs约36小时。

**决策路径建议：**

**路径A（推荐，稳健型）：** W1-6完成C1+C2，确认RootNet改进有效后再启动C3。若W8时GDN-S规模MPJPE>55 mm（明显不收敛），则C3退化为"直接将PoseMamba原版SSM接入3DMPPE管线"（仍有视频化贡献），放弃自定义门控。此路径保证毕设有完整结果。

**路径B（进取型）：** C1/C2/C3并行推进（W3即开始GDN调试），W10前完成全部训练。若核心判据（解耦vs混合↓≥3%）不满足，则论文贡献重新定位为"3DMPPE视频化+RootNet几何加固"，GDN门控作为negative result报告。此路径风险更高但上限更高。

**可证伪判据（来自idea定稿）：** (i) 若解耦门控组相对单一混合惊讶度组在遮挡段关节误差上无≥3%改善，则"范畴错误修正"不成立，回退原方案；(ii) 若双通道 $\beta_{\text{obs}}$（置信度+几何一致性）相对纯置信度 $\beta_{\text{obs}}$ 在幻觉遮挡子集上无≥5%改善，则几何兜底通道无效，观测质量门退化为OAHPE式软阈值。这两条判据直接对应消融(a)和(b2)，在W11-12消融阶段即可判定。若两条均不满足，说明GDN门控的核心设计假设不成立，论文仍有C1+C2的贡献（视频化+RootNet加固），但C3需诚实报告为negative result。

---

## 5. 结论

本方案以3DMPPE（DetectNet→RootNet→PoseNet）为基线，通过三个递进贡献构建遮挡鲁棒的多人绝对3D位姿估计系统：(1) 复现基线并视频化扩展；(2) 以SMPL面积修正+序数深度监督加固RootNet根深度在遮挡/截断下的鲁棒性；(3) 以GDN门控delta规则线性注意力替换单帧PoseNet为时序lifter，核心创新为解耦观测质量门（双通道关断：置信度+几何一致性）与状态惊讶门，实现"遮挡期保护记忆、恢复期按需读写"。预期在MuPoTS-3D上MRPE从289 mm降至≤150 mm，H36M MPJPE至少持平PoseMamba（38.1 mm）。主要风险为四仓库集成调试复杂度与GDN门控训练收敛性，均有结构性回退路径（加法式拆除，下界为3DMPPE原版）。时间框架16周（约4个月），单卡≤24 GB GPU预算，目标为本科毕业论文答辩。

**范围边界与局限：** 本方案不解决以下问题：(a) 极端拥挤场景（>5人）的跟踪与深度歧义——MuPoTS-3D最多3人，超出此范围的泛化未做承诺；(b) 相机内参未知时的绝对深度估计——沿用3DMPPE的已知内参假设；(c) 端到端联合训练——三模块仍为分离式训练，级联误差仅通过门控机制缓解而非根本消除；(d) 跨数据集泛化（H36M→3DPW/AGORA）——评估限于标准协议内，泛化声明需后续工作支撑。这些局限在毕设时间框架内是合理的取舍，核心贡献（解耦门控+几何加固）的有效性可在现有协议内充分验证。
