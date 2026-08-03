# 顺序无关delta-rule线性注意力的时空姿态提升

> 状态: draft · 2026-07-21

## Gap 来源（结构依据）
母题[通用套路]张力：'SSM的线性扫描假设关节/帧存在有意义的顺序…关节并非天然有序序列'（PoseMamba卡局限：局部扫描顺序是针对17关节人体骨架手工设计的启发式策略，缺乏理论最优性；BSTMamba卡局限：局部区域划分硬编码为Human3.6M的5组关节，迁移至其他骨架需重新定义）。叠加母题[共享假设]：骨骼拓扑先验硬编码绑定17关节格式构成跨骨架泛化瓶颈。跨领域证据：BASED(arxiv:2402.18668)证明固定大小递归态(Mamba/RWKV/H3)在recall任务上系统性弱于注意力——因为按位置递推的记忆是顺序绑定的、内容寻址能力差。

## 动机
本领域架构竞赛已从膨胀卷积→Transformer→Mamba/SSM。Mamba换来线性复杂度，却把'按顺序扫描'这一归纳偏置强加给本无自然顺序的关节维度，PoseMamba/BSTMamba只能用人工设计的局部重排序扫描去打补丁，且绑定17关节格式。与此同时，2024-2025语言模型序列架构已推进到门控线性注意力/delta-rule线性注意力(GLA、Gated DeltaNet、BASED)：保持线性复杂度，但把信息存入key→value关联记忆、用delta规则更新、按key相似度内容寻址召回，召回不依赖序列位置。这恰好对症姿态任务的痛点：空间(关节)维本质无序，应按内容寻址而非按扫描顺序；时间维才真正有序。把'空间用顺序无关关联记忆、时间用有序递推'这一原则性拆分迁入姿态提升，既消除人工扫描顺序，又天然支持跨骨架。

## 核心假设
如果把时空编码器的空间维Mamba选择性扫描(或自注意力)替换为delta-rule门控线性注意力关联记忆(关节按学习的内容key寻址、而非按固定扫描顺序)，则在保持线性复杂度的同时，能消除对人工关节扫描顺序的依赖，在Human3.6M上达到或超过PoseMamba-B的MPJPE且参数/FLOPs更低，并在不重设计扫描顺序的前提下迁移到MPI-INF-3DHP(28关节)时精度衰减显著小于PoseMamba。

## 技术路线
以PoseMamba(W4409368373)的双向全局-局部时空块为骨架，保留其时间维双向扫描(时间有序，递推合理)；将空间维的选择性扫描替换为delta-rule门控线性注意力：每个关节token生成内容key与value，维护一个key→value关联记忆矩阵，按delta规则(预测误差驱动)更新，查询时按key相似度召回，从而对关节置换等变、对扫描顺序无关。借鉴ConvFormer(W4382892987)用1D卷积生成Q/K/V引入局部性的做法作为key的局部增强；引入BASED(arxiv:2402.18668)/GLA(arxiv:2406.18007)的门控衰减机制控制记忆遗忘。空间记忆+时间递推双流融合后接回归头。整个空间模块不含任何硬编码关节分组或扫描顺序，故跨骨架无需重设计。

## 最小实验设计
数据Human3.6M(S1/5/6/7/8训练,S9/11测试,17关节)；输入CPN检测2D与GT2D；T=81/243。基线=同参数预算下重实现的PoseMamba-B与MixSTE。指标MPJPE(P1)/P-MPJPE(P2)/参数量/FLOPs。关键消融：(a)空间维 Mamba扫描 vs delta-rule关联记忆(同时间维)；(b)关节顺序扰动鲁棒性(随机打乱关节编号后精度变化)；(c)跨骨架迁移：仅在H3.6M训练直接测MPI-INF-3DHP(28关节)，对比两方法的精度衰减。result.json字段：{method, input(CPN/GT), T, MPJPE, P-MPJPE, params_M, flops_G, joint_shuffle_drop, cross_skeleton_drop}。规模为一个脚本+单卡可跑(参照PoseMamba单张3090、batch4、120epochs)。

## 相关论文
- W4409368373 — PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model
- W4413980847 — A Spatiotemporal Bidirectional Mamba Network with Global–Local Skeletal Enhancement for 3D Human Pose Estimation
- W4382892987 — ConvFormer: parameter reduction in transformer models for 3D human pose estimation by leveraging dynamic multi-headed convolutional attention
- W3136525061 — 3D Human Pose Estimation with Spatial and Temporal Transformers
- W4312417903 — MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video
- W2626778328 — Attention Is All You Need
- arxiv:2402.18668 — Simple linear attention language models balance the recall-throughput tradeoff
- arxiv:2406.18007 — Deep Mamba Multi-modal Learning

## 评审记录（critique_idea 自动写入）

### 查重（top 相近工作）
无

### 对抗评审 3/3 票支持
✅ 评审通过（多数派未能驳倒）
  - The idea is well-motivated (joints lack natural ordering, content-addressed memory is principled over hand-designed scans), combines a genuinely underexplored cross-pollination of delta-rule linear attention into pose estimation with no closely related prior work found, and the experimental plan is feasible on a single GPU with clear ablations—making it a reasonable candidate for empirical validation.
  - The idea is principled (order-agnostic content-addressed spatial memory vs. ordered temporal recurrence), builds on well-established components (PoseMamba, delta-rule linear attention, GLA/BASED), no highly similar prior work was found, and the proposed single-GPU experiment on Human3.6M is clearly feasible.
  - The idea is a principled and novel combination—replacing order-dependent spatial Mamba scans with order-free delta-rule linear attention for joint tokens is well-motivated (joints lack natural ordering), no highly similar prior work was found, the experimental plan is concrete and single-GPU feasible, and the cross-skeleton transfer prediction is a natural consequence of removing hardcoded scan orders.

## 可执行性评估：中
- 外部仓库: 3 个（GarrickZ2/PoseMamba, JinluZhang1126/MixSTE, fla-org/flash-linear-attention） · GPU: 需要 · 预训练权重: 需要 · 数据准备: 小 · 胶水复杂度: 中
- 风险点: 核心风险在于将fla库的delta-rule线性注意力模块适配进PoseMamba空间维时，需保证同参数预算公平对比且跨骨架(17→28关节)的key/value维度对齐无硬编码，但数据集与训练流程均为H3.6M成熟路径、单卡3090可跑
- 结论: 中等工程量，方案书交用户定库，由用户决定是否自动执行。
- 【下一步必做】这些涉及仓库还没有 repo 卡：GarrickZ2/PoseMamba, JinluZhang1126/MixSTE, fla-org/flash-linear-attention。定稿后、写方案书/报告前，先逐个 study_codebase 查证工程事实。
