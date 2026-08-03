# 顺序无关delta-rule线性注意力的时空姿态提升

> 状态: draft · 2026-07-21

## Gap 来源（结构依据）
母题[通用套路]张力：'SSM的线性扫描假设关节/帧存在有意义的顺序…关节并非天然有序序列'（PoseMamba卡局限：局部扫描顺序是针对17关节人体骨架手工设计的启发式策略，缺乏理论最优性；BSTMamba卡局限：局部区域划分硬编码为Human3.6M的5组关节，迁移至其他骨架需重新定义）。叠加母题[共享假设]：骨骼拓扑先验硬编码绑定17关节格式构成跨骨架泛化瓶颈。跨领域证据：BASED(arxiv:2402.18668)证明固定大小递归态(Mamba/RWKV/H3)在recall任务上系统性弱于注意力——因为按位置递推的记忆是顺序绑定的、内容寻址能力差。理论支撑：Test-time regression统一框架(arxiv:2501.12352)证明关联记忆本质是test-time回归，线性注意力与SSM均为其特例——这从原理层面说明空间维的关节token召回问题可被建模为对上下文的加权回归，无需依赖顺序递推（来自arxiv:2501.12352卡）。

## 动机
本领域架构竞赛已从膨胀卷积→Transformer→Mamba/SSM。Mamba换来线性复杂度，却把'按顺序扫描'这一归纳偏置强加给本无自然顺序的关节维度，PoseMamba/BSTMamba只能用人工设计的局部重排序扫描去打补丁，且绑定17关节格式。与此同时，2024-2025语言模型序列架构已推进到门控线性注意力/delta-rule线性注意力(GLA、Gated DeltaNet、BASED)：保持线性复杂度，但把信息存入key→value关联记忆、用delta规则更新、按key相似度内容寻址召回，召回不依赖序列位置。这恰好对症姿态任务的痛点：空间(关节)维本质无序，应按内容寻址而非按扫描顺序；时间维才真正有序。Test-time regression框架(arxiv:2501.12352)进一步从理论层面证实：关联记忆即test-time回归，线性注意力/SSM/快速权重均为该框架的特例，空间维用关联记忆替代顺序扫描并非仅凭直觉类比，而是有统一原理支撑的设计选择（来自arxiv:2501.12352卡）。把'空间用顺序无关关联记忆、时间用有序递推'这一原则性拆分迁入姿态提升，既消除人工扫描顺序，又天然支持跨骨架。

## 核心假设
如果把时空编码器的空间维Mamba选择性扫描(或自注意力)替换为delta-rule门控线性注意力关联记忆(关节按学习的内容key寻址、而非按固定扫描顺序)，则在保持线性复杂度的同时，能消除对人工关节扫描顺序的依赖，在Human3.6M上达到或超过PoseMamba-B的MPJPE且参数/FLOPs更低，并在不重设计扫描顺序的前提下迁移到MPI-INF-3DHP(28关节)时精度衰减显著小于PoseMamba。

## 技术路线
以PoseMamba(W4409368373)的双向全局-局部时空块为骨架，保留其时间维双向扫描(时间有序，递推合理)；将空间维的选择性扫描替换为门控delta-rule线性注意力（Gated Delta Networks, arxiv:2412.06464）：每个关节token生成内容key与value，维护一个key→value关联记忆矩阵，按门控delta规则更新——S_t = α_t S_{t−1}(I − β_t k_t k_t^⊤) + β_t v_t k_t^⊤，其中α_t为逐时间步标量衰减门控（α→0快速清空过时记忆，α→1退化为纯delta规则做精准定向替换），β_t控制写入强度；查询时按key相似度召回，从而对关节置换等变、对扫描顺序无关（来自arxiv:2412.06464卡）。开源实现参照NVlabs/GatedDeltaNet及fla-org/flash-linear-attention。借鉴ConvFormer(W4382892987)用1D卷积生成Q/K/V引入局部性的做法作为key的局部增强。可选增强变体：采用RWKV-7(arxiv:2503.14456)的向量值学习率门控与解耦移除/替换键机制替代GDN的标量门控，以获得更细粒度的逐维记忆管理（来自arxiv:2503.14456卡），作为消融对比选项。空间记忆+时间递推双流融合后接回归头。整个空间模块不含任何硬编码关节分组或扫描顺序，故跨骨架无需重设计。

## 最小实验设计
数据Human3.6M(S1/5/6/7/8训练,S9/11测试,17关节)；输入CPN检测2D与GT2D；T=81/243。基线=同参数预算下重实现的PoseMamba-B与MixSTE。指标MPJPE(P1)/P-MPJPE(P2)/参数量/FLOPs。关键消融：(a)空间维 Mamba扫描 vs delta-rule关联记忆(同时间维)；(b)关节顺序扰动鲁棒性(随机打乱关节编号后精度变化)；(c)跨骨架迁移：仅在H3.6M训练直接测MPI-INF-3DHP(28关节)，对比两方法的精度衰减；(d)空间记忆模块表达力对比：GDN标量门控(arxiv:2412.06464) vs RWKV-7式向量门控+解耦键(arxiv:2503.14456)（来自arxiv:2503.14456卡）。训练细节：参照GDN µP(arxiv:2606.04048)的初始化方差与学习率缩放规则（门控权重矩阵Θ(1/√d)缩放、标量门控参数Θ(√d)缩放），但需注意该规则在17-28 token的极短序列下是否仍适用存在不确定性（来自arxiv:2606.04048卡）。result.json字段：{method, input(CPN/GT), T, MPJPE, P-MPJPE, params_M, flops_G, joint_shuffle_drop, cross_skeleton_drop, spatial_variant(GDN/RWKV7)}。规模为一个脚本+单卡可跑(参照PoseMamba单张3090、batch4、120epochs)。

## 相关论文
- W4409368373 — PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model
- W4413980847 — A Spatiotemporal Bidirectional Mamba Network with Global–Local Skeletal Enhancement for 3D Human Pose Estimation
- W4382892987 — ConvFormer: parameter reduction in transformer models for 3D human pose estimation by leveraging dynamic multi-headed convolutional attention
- W3136525061 — 3D Human Pose Estimation with Spatial and Temporal Transformers
- W4312417903 — MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video
- W2626778328 — Attention Is All You Need
- arxiv:2402.18668 — Simple linear attention language models balance the recall-throughput tradeoff
- arxiv:2406.18007 — Deep Mamba Multi-modal Learning
- arxiv:2412.06464 — Gated Delta Networks: Improving Mamba2 with Delta Rule
- arxiv:2503.14456 — RWKV-7: Dynamic State Evolution for Linear Attention
- arxiv:2501.12352 — Test-time regression: a unifying framework for designing sequence models with associative memory
- arxiv:2606.04048 — Unlocking Feature Learning in Gated Delta Networks at Scale
- arxiv:2501.00663 — Titans: Learning to Memorize at Test Time

## 评审记录（critique_idea 自动写入）

### 查重（top 相近工作）
无

### 对抗评审 3/3 票支持
✅ 评审通过（多数派未能驳倒）
  - The idea is well-motivated (joints lack natural ordering, content-addressed memory is principled over hand-designed scans), combines a genuinely underexplored cross-pollination of delta-rule linear attention into pose estimation with no closely related prior work found, and the experimental plan is feasible on a single GPU with clear ablations—making it a reasonable candidate for empirical validation.
  - The idea is principled (order-agnostic content-addressed spatial memory vs. ordered temporal recurrence), builds on well-established components (PoseMamba, delta-rule linear attention, GLA/BASED), no highly similar prior work was found, and the proposed single-GPU experiment on Human3.6M is clearly feasible.
  - The idea is a principled and novel combination—replacing order-dependent spatial Mamba scans with order-free delta-rule linear attention for joint tokens is well-motivated (joints lack natural ordering), no highly similar prior work was found, the experimental plan is concrete and single-GPU feasible, and the cross-skeleton transfer prediction is a natural consequence of removing hardcoded scan orders.

## 可执行性评估：中
- 外部仓库: 3 个（GarrickZ2/PoseMamba, JinluZhang1126/MixSTE, NVlabs/GatedDeltaNet + fla-org/flash-linear-attention） · GPU: 需要 · 预训练权重: 需要 · 数据准备: 小 · 胶水复杂度: 中
- 风险点: 核心风险在于将GDN/fla库的delta-rule线性注意力模块适配进PoseMamba空间维时，需保证同参数预算公平对比且跨骨架(17→28关节)的key/value维度对齐无硬编码；µP缩放规则(arxiv:2606.04048)在17-28 token极短序列下是否仍适用存在不确定性，可能需回退至标准参数化；但数据集与训练流程均为H3.6M成熟路径、单卡3090可跑
- 结论: 中等工程量，方案书交用户定库，由用户决定是否自动执行。
- 【下一步必做】这些涉及仓库还没有 repo 卡：GarrickZ2/PoseMamba, JinluZhang1126/MixSTE, NVlabs/GatedDeltaNet, fla-org/flash-linear-attention。定稿后、写方案书/报告前，先逐个 study_codebase 查证工程事实。

## 更新记录
- 2026-07-23 · 局部更新：(1)技术路线将泛引'GLA/BASED门控衰减'升级为GDN(arxiv:2412.06464)精确门控delta规则公式及开源实现NVlabs/GatedDeltaNet，新增RWKV-7(arxiv:2503.14456)向量门控变体作为可选增强；(2)Gap来源与动机补充Test-time regression(arxiv:2501.12352)统一框架作为理论支撑；(3)最小实验设计新增消融(d)GDN标量门控vs RWKV-7向量门控表达力对比，训练细节补充GDN µP(arxiv:2606.04048)初始化与学习率缩放建议及其短序列适用性风险；(4)相关论文新增5篇；(5)可执行性评估更新外部仓库与风险点。判定类型：局部更新（强化+组件升级）。
