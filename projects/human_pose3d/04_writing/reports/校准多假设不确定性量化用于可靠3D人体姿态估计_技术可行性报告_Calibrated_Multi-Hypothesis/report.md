# 校准多假设不确定性量化用于可靠3D人体姿态估计 —— 技术可行性报告 (Calibrated Multi-Hypothesis Uncertainty Quantification for Reliable 3D Human Pose Estimation)
> 技术可行性报告 · 2026-07-21 · idea: 校准多假设3D姿态不确定性量化.md · ReAct 写作（边写边查证 papers/cards/codebases）


> 撰写日期：2026-07-21  
> 证据分级：cards/\*.json（结构化摘要）、codebases/MHFormer.md 与 codebases/Diffpose.md（仓库级代码事实）、ideas/校准多假设3D姿态不确定性量化.md（定稿 idea）。未在以上素材中找到来源的具体数字与仓库细节一律标注『待验证』。

---

## 摘要

本报告评估一项研究方向级技术可行性：把 MHFormer/DiffPose 等多假设 3D 姿态估计网络在最后一步对多假设所做的"坍缩为单点"操作，替换为**保留 K 条假设的逐关节经验分布**，并叠加**解剖学合理性重加权**与**温度缩放 + isotonic 回归校准**，从而在不显著牺牲 MPJPE 的前提下产出可用于选择性预测的校准不确定性。报告从深度歧义这一贯穿全领域的母题出发，论证现有方法把多假设多样性当作训练 trick 而非可量化预测不确定性的根本性缺陷；以 repo 卡为唯一代码事实来源给出最小侵入改造点（MHFormer `model/mhformer.py:51-54` 的 1×1 Conv1d 回归头、DiffPose `runners/diffpose_frame.py:296-299` 的 N 样本均值）；并给出包含 oracle 上界与 negative control 的消融矩阵、风险表与决策建议。**结论**：方案在工程上路径明确、外部依赖成熟、单卡 RTX 3090 可验证；主要风险在多假设坍缩改造的预训练权重兼容性与温度缩放过拟合验证集；建议在 6–8 周内完成最小可行验证（MVP），若 MPJPE 退化 ≤1.5 mm、NLL 较朴素方差基线相对下降 ≥20%、ECE ≤0.05 且 selective AUC ≥0.6 同时成立，则值得扩展为完整论文投向 CVPR/ECCV 级会议。

---

# 1. 背景与动机

## 1.1 问题陈述

### 1.1.1 病态逆问题与深度歧义

单目 2D-to-3D 姿态提升是从 2D 关节坐标 $(u_i, v_i)_{i=1}^{J}$ 反推 3D 关节坐标 $(x_i, y_i, z_i)_{i=1}^{J}$ 的逆问题。给定相机内参矩阵 $K$ 与弱透视/正交透视假设，2D 观测仅约束 3D 点在射线方向上的 $(x, y)$ 平面分量，深度 $z$ 维度的信息几乎全部缺失。形式化地，设第 $i$ 个关节的 2D 观测 $p_i \in \mathbb{R}^2$ 与 3D 真值 $P_i \in \mathbb{R}^3$ 满足

$$
p_i = \pi(K P_i), \quad \pi\bigl([X;Y;Z]\bigr) = [X/Z;\, Y/Z]
$$

则在 $p_i$ 给定的条件下，$P_i$ 在射线 $\{K^{-1} \tilde p_i \cdot z \mid z \in \mathbb{R}_{>0}\}$ 上存在连续统多个解；当多关节之间存在运动学约束（骨长恒定、关节角限制）时，可行解集收缩但仍是一流形而非一点。母题 1（参见 `cards/_themes.json`）将此总结为：

> "多假设方法（DiffPose、MHFormer）承认多解性却最终仍坍缩为单一确定性输出，不确定性量化被默认放弃而非被解决"

### 1.1.2 瓶颈的量化表现

| 现象 | 量化表现 | 来源 |
|------|----------|------|
| GT 2D 与检测 2D 之间存在 10–13 mm MPJPE 鸿沟 | PoseFormer: 31.3 mm（GT 2D）→ 44.3 mm（CPN 2D） | `cards/3d_human_pose_estimation_with_spatial_and_temporal_transformers.json` |
| 上游 2D 检测失误可导致 3D 偏差 >300 mm 离群值 | VNect 报告离群姿态 | `cards/_themes.json` 母题 2 |
| DiffPose 在视频 CPN 2D 下 MPJPE 36.9 mm、GT 2D 下 18.9 mm | 18.9→36.9 的 18 mm 差距即为 2D 检测误差经扩散放大 | `cards/diffpose_toward_more_reliable_3d_pose_estimation.json` |
| Anatomy-Aware 在复杂姿态+邻近帧同为复杂姿态时"无法提供可靠信息" | H3.6M Protocol 1 44.1 mm / Protocol 2 35.0 mm | `cards/anatomy_aware_3d_human_pose_estimation_with_bone_based_pose_decomposition.json` |
| Simple Baseline 对倒立等分布外姿态"只能预测平均姿态" | 2D-to-3D lifting 不可逆的极端体现 | `cards/a_simple_yet_effective_baseline_for_3d_human_pose_estimation.json` |
| AugLift 跨数据集实验：H3.6M 训练达 40–50 mm 的模型迁移至 3DPW 退化至 >100 mm | ID→OOD 误差翻倍，部署时频繁产生大误差而无任何拒识机制 | `cards/auglift_depth_aware_input_reparameterization_improves_domain_generalization_in_2.json` |

这些数字共同说明：现有 SOTA 在受控分布（Human3.6M 室内、S1/5/6/7/8→S9/11 划分）上把 MPJPE 压到 35–45 mm 量级，但**没有任何方法给出"这一次预测是否可信"的标量信号**。AugLift 的跨域退化证据（H3.6M ~40 mm → 3DPW >100 mm）进一步表明：即使同分布精度已趋饱和，模型在真实部署中仍频繁产生大误差，而当前无任何机制在推理时标记"此次预测不可信"。在康复步态分析、AR 虚拟试穿、机器人选择性预测等下游场景中，"知道不知道"比"再降 1 mm"更关键——校准不确定性 + 选择性预测正是填补这一部署鸿沟的最小可行路径。

### 1.1.3 应用驱动的需求

- **康复/运动医学**：术后关节角度恢复评估需要误差棒，否则无法判断"病人是否真的在变好"还是"模型今天心情不好"。
- **AR/VR**：遮挡时主动切换到多视角融合或降低虚拟资产精度，需要逐关节不确定性作为触发信号。
- **机器人选择性预测**：当模型不确定时拒识并转交人工/多视角系统，是 selective prediction 的标准设定，其性能由误差-不确定性相关 AUC 直接决定。

## 1.2 相关工作

按"是否承认多解性"与"是否产出不确定性"两条轴线分组，只引用 `cards/` 中真实存在的论文。

### 1.2.1 单点估计路线（承认病态但压制多解性）

- **Simple Baseline (W2612706635, 2017)** — 用一个全连接残差网络直接拟合 2D→3D 映射，作者明确承认"对分布外姿态（如倒立的人）只能预测平均姿态"，并指出"2D 检测器错误会直接传播"。该路线把深度歧义视为噪声而非可建模的分布。
- **PoseFormer (W3136525061, 2021)** — 首个纯 Transformer 的 2D-to-3D 提升网络，将空间-时序分解为两个串行 Transformer。其 GT 2D MPJPE 31.3 mm 与 CPN 2D MPJPE 44.3 mm 之间 13 mm 的鸿沟揭示了"2D 检测误差传播"这一更深的瓶颈，但同样不输出不确定性。
- **Anatomy-Aware (W3126541466, 2021)** — 将 3D 关节位置预测显式分解为骨骼方向与骨骼长度两个子任务，骨骼长度跨全视频随机采样 $l=50$ 帧聚合估计。其骨长恒定假设与运动学树分解为本报告所提"解剖学合理性重加权"提供了早期借鉴，但本方案已升级为 DDHPose (W4393158891) 的骨长/骨方向分解（见 §2.2），Anatomy-Aware 作为辅助约束保留。

### 1.2.2 多假设路线（承认多解性但坍缩为单点）

- **MHFormer (W4312249545, 2022)** — 三阶段 Multi-Hypothesis Transformer：MHG（多假设生成）→ SHR（自假设精炼）→ CHI（跨假设交互）。最终用 hypothesis-mixing MLP 聚合 3 条假设为单一表示后回归 3D 姿态。作者明确指出"最终仍输出确定性单一解，未提供不确定性估计"。repo 卡（`codebases/MHFormer.md` §1）进一步定位了坍缩的物理位置：不是 CHI 块（仍输出三条独立流），而是其后的 1×1 Conv1d 回归头。
- **DiffPose (W4386075813, 2023)** — 把 3D 姿态估计建模为逆扩散，前向扩散用 5 核 GMM 初始化具有样本特异不确定性的分布 $H_K$，逆扩散 DDIM 取 $N=5$ 个样本并**取均值**作为输出。作者承认"未明确讨论局限"，但 repo 卡（`codebases/Diffpose.md` §1）显示采样器本身返回了 $N$ 个样本，只是 `runners/diffpose_frame.py:298` 的 `torch.mean` 把它们坍缩掉了。
- **DDHPose (W4393158891, 2024)** — 将解耦策略引入扩散模型前向过程，对骨长和骨方向分别加噪而非直接对 3D 坐标加噪，反向过程用层级时空去噪器（HSTDenoiser）生成多假设输出（$H \in \{1, 5, 20\}$）。最优 MPJPE 39.0 mm（H3.6M，243 帧）。其骨长/骨方向分解为本方案 §2.2 的解剖学降权提供了直接借鉴；但其多假设输出仍取最优假设作为点估计，未做不确定性校准或选择性预测评估（`cards/disentangled_diffusion_based_3d_human_pose_estimation_with_hierarchical_spatial_.json`）。

### 1.2.3 不确定性路线（产出不确定性但未完成"校准→选择性预测"链条）

领域内现有的不确定性估计主要分布在医学影像、6D 物体姿态等任务（参见 idea 评审记录的查重 top 列表）。在 2D-to-3D lifting 中，**POT/UGRN (W4382457852, 2023)** 是最接近的工作：其第二阶段 UGRN 估计逐关节异方差不确定性 $\sigma$，训练时以 $\sigma$ 为方差做高斯采样（UG-Sampling）增强鲁棒性，并用不确定性引导的注意力降权减少误差传播。然而其不确定性存在三重局限：(1) UG-Sampling 仅在训练阶段生效，推理时直接传入一阶段预测，$\sigma$ 不作为输出信号（`cards/pose_oriented_transformer_with_uncertainty_guided_refinement_for_2d_to_3d_human_.json` limitation 字段）；(2) 仅建模 aleatoric 不确定性，未涉及 epistemic 不确定性或多假设分歧；(3) 未做任何校准评估（无 ECE/NLL/覆盖率指标），也未用于选择性预测。

因此，Gap 从"无人做逐关节不确定性"精确化为：**"无人将多假设间的分歧转化为经校准的、面向下游选择性预测的不确定性输出"**——即"多假设分歧→校准不确定性→选择性预测"这一完整链条无人走通（idea 已通过 3/3 对抗评审，多数派未能驳倒）。

**第二结构支撑**：母题（`cards/_themes.json` 共享假设主题）明确指出"无人将骨长建模为带不确定性的分布，也无人让网络学习'何时信任几何、何时放弃'"。现有骨长/运动学先验均为硬常数或软损失，从未与预测不确定性耦合为自适应权重——这直接支撑本方案解剖学降权组件（§2.2）的原创性定位。

### 1.2.4 评测基座

母题 5（`cards/_themes.json`）指出：超过 15 篇卡片使用完全相同的 Human3.6M S1/5/6/7/8→S9/11 划分与 MPJPE/P-MPJPE 指标，形成封闭排行榜生态。本报告沿用此基座以保持可比性，但在第 4 章风险表中明确标注其野外泛化风险。

### 1.2.5 领域综述与骨干升级候选

**A Survey on Deep 3D Human Pose Estimation (W4404703236, 2024)** 按输出形式将领域方法分为确定性、概率性（多假设）与扩散模型三类，确认概率性/多假设输出为独立方法类别（`cards/a_survey_on_deep_3d_human_pose_estimation.json`）。本方案定位于"概率性多假设"与"校准"的交叉点。

**骨干升级候选参考**（时空建模方法，精度超越 MHFormer 但不涉及不确定性输出，可作后续骨干升级，见 §2.5）：
- STCFormer (W4386076485, 2023) — 时空 criss-cross 注意力
- MixSTE (W4312417903, 2022) — 混合时空编码器
- MotionBERT — 统一人体运动表征学习
- BSTMamba — 双向时空 Mamba 网络

## 1.3 根本性分析

### 1.3.1 信息论视角：2D 观测对 3D 解的信息量未被刻画

母题 1 张力原文指出"没有任何工作证明过在给定 2D 观测下 3D 解的信息论下界"。形式化地，记 2D 观测为 $\mathbf{p} \in \mathbb{R}^{2J}$、3D 解为 $\mathbf{P} \in \mathbb{R}^{3J}$，给定 2D 观测后 3D 的条件熵

$$
H(\mathbf{P} \mid \mathbf{p}) = H(\mathbf{P}) - I(\mathbf{P}; \mathbf{p})
$$

度量了"即便观测到 2D 后 3D 仍有多少不确定性"。本领域至今没有工作给出 $H(\mathbf{P} \mid \mathbf{p})$ 的解析或经验下界，导致所有方法都默认 $H(\mathbf{P} \mid \mathbf{p}) \approx 0$ 并强行拟合点估计。但实际上，自遮挡姿态下 $H(\mathbf{P} \mid \mathbf{p})$ 显然大于正面站立姿态——这一信号被单点估计直接抹平。

**推论 1**：若不显式建模 $p(\mathbf{P} \mid \mathbf{p})$ 而只拟合 $\mathbb{E}[\mathbf{P} \mid \mathbf{p}]$，则模型在高 $H(\mathbf{P} \mid \mathbf{p})$ 区域会输出"在训练集上平均的 3D 姿态"——这正是 Simple Baseline 卡描述的"对倒立的人只能预测平均姿态"现象的根因。

### 1.3.2 几何视角：多假设坍缩等价于在解流形上取投影

设给定 2D 观测 $\mathbf{p}$ 下的可行 3D 解集为流形 $\mathcal{M}(\mathbf{p}) \subset \mathbb{R}^{3J}$。MHFormer 的 CHI 块输出 3 条假设 $\{P_k\}_{k=1}^3 \subset \mathcal{M}(\mathbf{p})$，DiffPose 的扩散采样器输出 $N$ 条样本 $\{P_n\}_{n=1}^N \subset \mathcal{M}(\mathbf{p})$。两者最后的"取均值"操作

$$
\bar P = \frac{1}{K}\sum_{k=1}^{K} P_k \quad (\text{MHFormer}), \qquad \bar P = \frac{1}{N}\sum_{n=1}^{N} P_n \quad (\text{DiffPose})
$$

在流形 $\mathcal{M}(\mathbf{p})$ 非凸时**不保证 $\bar P \in \mathcal{M}(\mathbf{p})$**——平均值可能落在流形外的"非法"区域。即便流形近似凸，取均值也会丢失流形的局部曲率信息，而曲率恰是判断"该观测是否约束充分"的关键信号。

repo 卡为这一分析提供了代码级证据：

- **MHFormer**：`codebases/MHFormer.md` §1 明确"把多假设聚合为单一输出的真正操作是 `self.regression` 这个 1×1 Conv1d"，其输入通道是 `args.channel*3`（三流拼接），输出 `3*args.out_joints`（单一姿态）。坍缩发生在权重层面而非几何层面。
- **DiffPose**：`codebases/Diffpose.md` §1 明确"`output_uvxyz[0][-1]` 的 shape 为 `[B * test_times, 17, 5]`……代码立刻在第 298 行对 `test_times` 取了均值，因此下游只拿到平均后的单点估计"。

### 1.3.3 优化视角：MSE 损失天然鼓励坍缩

MHFormer 训练用标准 MPJPE 损失（`codebases/MHFormer.md` §4：`common/utils.py:13-16` 的 `mpjpe_cal` 即 `torch.mean(torch.norm(predicted - target, dim=-1))`）。当最后一步输出 $\bar P = \frac{1}{K}\sum_k P_k$ 时，

$$
\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial P_k} = \frac{2}{K}(\bar P - P^*)
$$

梯度对所有假设**完全相同**——这等价于一个 $K$ 倍宽的单一假设。换言之，MHFormer 的"多假设"在损失函数压力下退化为"一个假设的 $K$ 个扰动副本"，假设间多样性缺乏显式正则项支持。MHFormer 卡的局限分析也确认了这一点："假设多样性仅来自级联深度而非显式约束，假设是否真正覆盖不同可行解缺乏量化验证"。

### 1.3.4 三视角合流

三条论证合流指向同一结论：**多假设架构已经把解空间多样性编码到了网络内部，但最后一步的坍缩操作（MHFormer 的 Conv1d 回归头、DiffPose 的 N 样本均值）把它当作训练 trick 丢弃了**。改造这一步为分布输出 + 校准，理论上只需替换最后的聚合算子而无需触碰核心架构，是低成本的"信息保留"工程。

---

# 2. 方法

本方案拆解为三个互补贡献：(C1) 多假设逐关节分布保留——替换 MHFormer 的 1×1 Conv1d 回归头与 DiffPose 的均值算子，输出 K 条假设的逐关节高斯；(C2) 解剖学合理性重加权——借鉴 DDHPose 的骨长/骨方向分解对假设打分，避免方差被离群假设污染；(C3) 温度缩放 + isotonic 回归校准——用成熟 sklearn 组件把假设散布校准为可信的置信区间。下面逐一展开。

## 2.1 Contribution 1：多假设逐关节分布保留

### 2.1.1 设计动机

母题 1 张力原文："多假设方法（DiffPose、MHFormer）承认多解性却最终仍坍缩为单一确定性输出"。要在不重新设计骨干的前提下把这一丢失的多样性找回来，关键是把"坍缩算子"替换为"分布算子"。repo 卡定位了两处坍缩：

- **MHFormer**：`codebases/MHFormer.md` §1 明确 CHI 块 (`model/module/trans_hypothesis.py:205-208`) 输出三条独立流 $x_1, x_2, x_3$，但 `model/mhformer.py:51-54` 的单一回归头

  ```python
  self.regression = nn.Sequential(
      nn.BatchNorm1d(args.channel*3, momentum=0.1),
      nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
  )
  ```

  把三流沿通道拼接后直接映射到 `3*out_joints`，即单一 3D 姿态。CHI 仍在做"假设间交互"，但回归头立刻把交互后的三流线性混合掉。

- **DiffPose**：`codebases/Diffpose.md` §1 显示 DDIM 采样器 `common/utils_diff.py:46 generalized_steps` 本身返回了 $N$ 个样本（`output_uvxyz[0]` 序列），但 `runners/diffpose_frame.py:296-299` 取 `[-1]` 后立刻

  ```python
  output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
  ```

  即对 `test_times` 维度取均值。

两条路径的坍缩点都集中在最后一步，且**采样/交互逻辑完全无需改动**——这是工程可行性最关键的利好。

### 2.1.2 技术细节：MHFormer 改造

直接采纳 `codebases/MHFormer.md` §2 的最小侵入改造方案，分两步：

**步骤 A**——把单一回归头替换为 `nn.ModuleList` 三个独立头（每个仅接收 `args.channel`）：

```python
# 改造 model/mhformer.py:51-54
self.regression = nn.ModuleList([
    nn.Sequential(
        nn.BatchNorm1d(args.channel, momentum=0.1),
        nn.Conv1d(args.channel, 3*args.out_joints, kernel_size=1)
    ) for _ in range(3)
])
```

**步骤 B**——在 `model/mhformer.py:71-78` 不再拼接，而是分别回归再 stack：

```python
x_1, x_2, x_3 = self.Transformer_hypothesis(x_1, x_2, x_3)  # 不再 cat
x_1 = self.regression[0](x_1.permute(0, 2, 1).contiguous())
x_2 = self.regression[1](x_2.permute(0, 2, 1).contiguous())
x_3 = self.regression[2](x_3.permute(0, 2, 1).contiguous())
x = torch.stack([x_1, x_2, x_3], dim=1)             # (B, K, J*3, F)
x = rearrange(x, 'b k (j c) f -> b k f j c', j=J).contiguous()
return x  # (B, K=3, F, J, 3)
```

**步骤 C**——同步修改 `model/module/trans_hypothesis.py:207-210`，取消最后的 `torch.cat`：

```python
# 原: x = torch.cat([x_1, x_2, x_3], dim=2); x = self.norm(x); return x
x_1 = self.norm(x_1); x_2 = self.norm(x_2); x_3 = self.norm(x_3)
return x_1, x_2, x_3
```

输出形状变为 $(B, K=3, F, J, 3)$，其中 $F=351$（默认）、$J=17$、最后一维是 $(x, y, z)$。

### 2.1.3 逐关节高斯参数化

对每个关节 $j$、每帧 $f$，把 $K$ 条假设 $\{P_k^{(j,f)}\}_{k=1}^K \in \mathbb{R}^3$ 拟合为各向同性高斯（先做最简单的各向同性以减少参数与过拟合风险）：

$$
\mu^{(j,f)} = \frac{1}{K}\sum_{k=1}^K P_k^{(j,f)}, \qquad
\sigma^{(j,f)} = \sqrt{\frac{1}{K}\sum_{k=1}^K \bigl\|P_k^{(j,f)} - \mu^{(j,f)}\bigr\|^2 / 3}
$$

输出即为 $(\mu, \sigma) \in \mathbb{R}^{J \times 4}$ per frame。这里 $\mu$ 仍可作为"点估计"与 MHFormer 原版对比 MPJPE，$\sigma$ 是后续校准的输入。

> **设计选择说明**：各向同性而非全协方差，是因为 $K=3$ 的样本数远不足以估计 6 参数协方差（自由度不足会引入数值病态）。若未来 $K \ge 5$，可考虑对角协方差。

### 2.1.4 DiffPose 复用其 GMM 逐关节方差信号

DiffPose 已经在 `common/generators.py:14-40 PoseGenerator_gmm` 中提供了逐关节方差（`out_pose_noise_scale`，其中 `u/v` 来自 GMM 核方差、`x/y/z` 初始化为 1）。改造方案：

- 把 `runners/diffpose_frame.py:298` 的 `torch.mean` 替换为保留 $N$ 个样本：

  ```python
  output_uvxyz_dist = output_uvxyz.reshape(test_times, -1, 17, 5)  # [N, B, 17, 5]
  output_xyz_dist = output_uvxyz_dist[:, :, :, 2:]                  # [N, B, 17, 3]
  ```

- 评测配置文件需把 `testing.test_times` 从 1 改回 5（`codebases/Diffpose.md` §5 硬编码表记录：YAML 默认覆盖为 1）。

- 在 `output_xyz_dist` 上拟合逐关节高斯（同 §2.1.3），可选用 `out_pose_noise_scale` 的 GMM 核方差作为先验，缓解 $N=5$ 样本估计方差的不稳定性。

### 2.1.5 训练监督策略

MHFormer 原训练用 `common/utils.py:13-16` 的 `mpjpe_cal`（标准 MPJPE）。保留 K 条假设后的监督策略有三种选择（详见风险表 R3）：

| 策略 | 损失形式 | 优点 | 风险 |
|------|----------|------|------|
| **S1: min-loss**（推荐） | $\mathcal{L} = \min_k \|P_k - P^*\|$ | 鼓励至少一条假设命中真值，保留多样性 | 退化为"赢者通吃"，K-1 条假设可能退化 |
| **S2: 均值损失** | $\mathcal{L} = \|\bar P - P^*\|$ | 与原版等价 | 多样性坍缩（见 §1.3.3） |
| **S3: 全假设 + NLL** | $\mathcal{L} = \text{MPJPE}(\bar P) - \lambda \log \mathcal{N}(P^* \mid \mu, \sigma)$ | 同时约束精度与校准 | $\lambda$ 调参敏感 |

推荐 **S1 + 极小 $\lambda$ 的 S3 项**作为消融起点，$\lambda$ 在验证集上以 ECE 为目标调优。

### 2.1.6 与现有系统的衔接

- **训练入口**：`main.py:137-164` 的训练主循环不变。
- **测试入口**：`main.py:23-68 step('test', ...)` 需把 `output_3D[:, opt.pad]` 的索引从 4 维调整为 5 维 `output_3D[:, :, opt.pad]`，并同步修改 `common/utils.py:18-23 test_calculation` 接受分布输出。
- **flip 增强**：`main.py:70-87 input_augmentation` 需对 K 维假设独立做左右翻转再平均（保持原平均语义，避免破坏假设分布形状）。
- **demo 可视化**：`demo/vis.py:196-205` 可降级为画均值或画第 0 条假设（如需可视化不确定性需另行扩展渲染）。

## 2.2 Contribution 2：解剖学合理性重加权

### 2.2.1 设计动机

朴素方差 $\sigma^2 = \frac{1}{K}\sum_k \|P_k - \mu\|^2$ 在 $K$ 较小时极易被离群假设污染：若 3 条假设中有 1 条因网络在某遮挡模式下的失败而产生明显不合法的姿态（如膝盖反向弯曲），$\sigma$ 会被显著放大而与真实不确定性脱钩。

母题 4（`cards/_themes.json`）指出："骨长恒定、关节角限制、左右对称、运动学树被反复手工注入作为弱监督或正则化"。然而母题同时指出"无人将骨长建模为带不确定性的分布，也无人让网络学习'何时信任几何、何时放弃'"——现有骨长先验均为硬常数或软损失，从未与预测不确定性耦合。

本方案采用 **DDHPose (W4393158891, 2024)** 的骨长/骨方向分解作为打分依据：将每个假设 $P_k$ 分解为骨长向量 $\ell_k$ 与骨方向向量 $d_k$，计算其与训练集骨长统计分布的马氏距离作为降权分数。相比原 Anatomy-Aware (W3126541466) 的单一骨长一致性约束，DDHPose 的分解更精细（区分骨长偏差与骨方向偏差），且天然支持"骨长本身带不确定性分布"的建模。Structure and Motion 卡的"非法角度损失（膝/肘不超过 180°）"、Weakly-Supervised 卡的"骨骼长度比例不变性 3D 几何约束"作为辅助约束保留。

### 2.2.2 技术细节：骨长/骨方向分解打分

定义骨骼集合 $\mathcal{B} = \{(i, j) \mid \text{关节 } i, j \text{ 相邻}\}$（H36M 17 关节默认 16 条骨）。对第 $k$ 条假设 $P_k$，按 DDHPose 的分解方式计算骨长向量 $\ell_k \in \mathbb{R}^{|\mathcal{B}|}$（各骨欧氏长度）与骨方向向量 $d_k \in \mathbb{R}^{|\mathcal{B}| \times 2}$（各骨在球坐标下的方位角/仰角），并定义其与训练集骨长统计分布 $(\hat \mu_\ell, \hat \Sigma_\ell)$ 的马氏距离及方向偏差的联合一致性分数：

$$
w_k = \exp\bigl(-\alpha_\ell \cdot (\ell_k - \hat \mu_\ell)^\top \hat \Sigma_\ell^{-1} (\ell_k - \hat \mu_\ell) / |\mathcal{B}| \;-\; \alpha_d \cdot \frac{1}{|\mathcal{B}|}\sum_{b} \angle(d_k^{(b)}, \hat d^{(b)})^2\bigr) \in (0, 1]
$$

其中 $\alpha_\ell, \alpha_d$ 是温度系数，在验证集上以 NLL 为目标调优；$\hat \mu_\ell, \hat \Sigma_\ell$ 来自训练集 GT 3D 的骨长均值与协方差（按 subject 分组）；$\angle(\cdot, \cdot)$ 为方向角偏差。再加一项关节角限制项（仅膝肘二自由度，因为母题 4 指出"肩髋脊柱等复杂关节的约束从未被有效建模"）：

$$
w_k \leftarrow w_k \cdot \mathbb{1}\bigl[\theta_k^{(\text{knee, elbow})} \in [0, \pi]\bigr]
$$

> **设计说明**：硬阈值仅在训练初期使用，训练后期可改为软惩罚 $\exp(-\beta \cdot \text{ReLU}(-\theta))$，避免梯度消失。

### 2.2.3 重加权后的分布参数

把 §2.1.3 的均匀权重 $1/K$ 替换为归一化的 $w_k$：

$$
\mu^{(j,f)} = \frac{\sum_k w_k P_k^{(j,f)}}{\sum_k w_k}, \qquad
\sigma^2_{\text{rew}}{}^{(j,f)} = \frac{\sum_k w_k \bigl\|P_k^{(j,f)} - \mu^{(j,f)}\bigr\|^2}{\sum_k w_k \cdot \max(K-1, 1) / K}
$$

后一项的 $\max(K-1,1)/K$ 是 Bessel 修正，避免 $K=3$ 时方差低估。

### 2.2.4 与现有系统的衔接

- **骨长/骨方向统计**：可在 `common/load_data_hm36.py` 的 `Fusion` 类基础上新增一个 `BoneStatsEstimator`，从训练集 GT 3D 计算每 subject 的骨长均值 $\hat \mu_\ell$ 与协方差 $\hat \Sigma_\ell$（按 DDHPose 分解方式），推理时按 subject ID 取用。
- **关节角计算**：用 H36M 17 关节的标准父子关系（`common/h36m_dataset.py` 中骨架边定义）计算膝肘关节角，纯 numpy 实现，无外部依赖。
- **计算开销**：仅前向时多一次 $O(K \cdot |\mathcal{B}|)$ 的骨长计算与 $O(K \cdot 4)$ 的关节角检查，可忽略。

## 2.3 Contribution 3：温度缩放 + isotonic 回归校准

### 2.3.1 设计动机

即便重加权后的 $\sigma_{\text{rew}}$ 仍可能系统性地高估或低估真实误差——前者导致置信区间过宽（保守），后者导致覆盖率不足（激进）。**校准**的目标是让 $\sigma$ 满足频率学派意义下的覆盖率：声明 90% 置信区间时，真实误差确实有 90% 落在区间内。

温度缩放与 isotonic 回归是图像分类不确定性校准的标准工具（sklearn `IsotonicRegression` 与温度参数 $T$ 的单变量优化），引入 3D 姿态领域无需新算法，只需把"分类 logits"替换为"逐关节 3D 误差"。

### 2.3.2 技术细节

**步骤 1**：在验证集上收集 $(\sigma_i^{\text{raw}}, e_i)$ 对，其中 $e_i = \|\mu_i - P_i^*\|_2$ 是逐关节误差。

**步骤 2**：拟合温度 $T$ 使负对数似然最小：

$$
T^* = \arg\min_T \sum_i -\log \mathcal{N}\bigl(e_i \mid 0,\, (T \cdot \sigma_i^{\text{raw}})^2\bigr)
$$

这是一维凸优化，`scipy.optimize.minimize_scalar` 即可。

**步骤 3**：在 $T^*$ 之上叠加 isotonic 回归 $g: \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$，对 $(T^* \sigma_i, e_i)$ 做单调约束最小二乘，进一步纠正非线性偏差：

$$
\sigma_i^{\text{cal}} = g(T^* \cdot \sigma_i^{\text{raw}})
$$

**步骤 4**：报告 $\sigma^{\text{cal}}$ 的覆盖率（coverage@90%）与 ECE（Expected Calibration Error）。

### 2.3.3 伪代码

```python
# sklearn + scipy 实现，约 30 行
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

def calibrate(sigma_raw_val, error_val, sigma_raw_test):
    def nll(T):
        sigma = T * sigma_raw_val
        return -np.mean(-np.log(sigma) - 0.5 * (error_val / sigma)**2)
    T_star = minimize_scalar(nll, bounds=(0.1, 10), method='bounded').x
    sigma_T = T_star * sigma_raw_val
    iso = IsotonicRegression(out_of_bounds='clip', y_min=1e-3)
    iso.fit(sigma_T, error_val)
    return iso.predict(T_star * sigma_raw_test), T_star
```

### 2.3.4 与现有系统的衔接

- **训练阶段不引入校准**：训练用原始 $\sigma_{\text{rew}}$，避免校准扰动梯度。
- **校准阶段独立**：训练完成后在验证集（H3.6M S9 的一部分子集）上拟合 $T^*, g$，再在测试集（S11）上评估。
- **无外部 GPU 依赖**：sklearn 与 scipy 在 CPU 上即可完成 1M 样本级的拟合。
- **可重现性**：所有超参（$T$ 的范围、isotonic 的 `y_min`）固定并写入配置，无随机种子。

## 2.4 三贡献的关系

```
MHFormer/DiffPose 多假设骨干
        │
        ├── C1: 保留 K 条假设 → 逐关节高斯 (μ, σ_raw)
        │           │
        │           └── C2: 解剖学重加权 → (μ_rew, σ_rew)
        │                       │
        │                       └── C3: 温度缩放 + isotonic → (μ, σ_cal)
        │                                       │
        │                                       └── 下游: 选择性预测/康复分析
        │
        └── 训练: S1 min-loss + λ·NLL  →  评测: MPJPE / NLL / ECE / AUC / coverage@90%
```

C1 是信息保留的基础，C2 是鲁棒性加固，C3 是统计可信化——三者互不依赖，可分别消融（见 §3.2）。

## 2.5 骨干升级讨论

本方案的校准贡献（C1+C2+C3）与骨干选择正交：核心操作仅作用于骨干输出的多假设/多帧预测之上，不修改骨干内部结构。若骨干从 MHFormer 升级为时空方法（如 STCFormer (W4386076485)、MixSTE (W4312417903)、MotionBERT 等），假设多样性来源可从纯空间多假设扩展为时间维度的预测分布——即利用多帧预测的时序不一致性作为额外的 epistemic 不确定性信号。这些时空方法均已超越 MHFormer 精度（DDHPose 卡 eval_setup 基线列表确认 STCFormer/MixSTE 为当前 SOTA 级），但不涉及不确定性输出，因此本方法的温度缩放 + 选择性预测框架可在任意多假设/多帧骨干上即插即用。当前最小实验以 MHFormer 为骨干验证核心贡献（repo 卡提供完整代码事实），时空骨干升级作为后续扩展（见 §4.6.2 路径 B）。

---

# 3. 实验计划

## 3.1 评估指标

### 3.1.1 指标定义

| 指标 | 定义 | 物理含义 | 目标 |
|------|------|----------|------|
| **MPJPE** (mm) | $\frac{1}{JF}\sum_{f,j} \|\mu^{(j,f)} - P^{(j,f)*}\|$ | 点估计平均误差，与 MHFormer 原版对比是否退化 | 较 MHFormer 基线退化 ≤1.5 mm |
| **逐关节 NLL** | $-\frac{1}{JF}\sum_{f,j} \log \mathcal{N}(P^{*} \mid \mu, \sigma^2)$ | 高斯负对数似然，奖励把真值放在高概率区 | 较朴素方差基线相对下降 ≥20% |
| **ECE** | $\sum_b \frac{|B_b|}{N}\bigl\|\text{acc}(b) - \text{conf}(b)\bigr\|$ | 期望校准误差，把预测置信分桶对比实际频率 | ≤0.05（经验阈值） |
| **误差-不确定性相关 AUC** | 以 $\sigma$ 为阈值扫点，画"拒识比例-剩余 MPJPE"曲线下面积 | 选择性预测的核心指标，>0.5 才有意义 | AUC ≥ 0.6（idea 核心假设） |
| **coverage@90%** | 真值落在 $\mu \pm 1.645\sigma$ 内的样本比例 | 频率学派覆盖率，应在 90% 附近 | 88%–92% |

### 3.1.2 当前值 / 目标值 / 改进幅度表

| 指标 | MHFormer 基线（点估计） | 朴素方差（C1 only） | 本方法目标（C1+C2+C3） | 改进幅度（保守–乐观，相对最强基线） |
|------|----------------------|---------------------|----------------------|----------|
| MPJPE (mm) | ~44（H3.6M CPN，351 帧）| 同基线（μ 仍为均值） | ≤45.5 | 退化 0%–3.4%（保守 ≤1.5 mm / 44 mm ≈ 3.4%；乐观 ≈0%）。依据：改造仅替换末端回归头，骨干特征不变（codebases/MHFormer.md §2） |
| 逐关节 NLL | N/A（无 σ） | 待验证 | 较朴素方差相对下降 ≥20% | 保守 20%–25%、乐观 30%–50% 相对下降。依据：温度缩放+isotonic 在分类校准文献中典型降低 raw-variance NLL 20%–50%（待验证：3D 姿态上无直接先例） |
| ECE | N/A | 通常 0.15–0.30 | ≤0.05 | 保守相对下降 67%（0.15→0.05）、乐观 83%（0.30→0.05）。依据：温度缩放+isotonic 为分类校准标准工具（§2.3），3D 姿态上幅度待验证 |
| selective AUC | 0.5（无信号） | 0.55–0.60 | ≥0.6 | 保守 +0 pp（0.60→0.60）、乐观 +9 pp（0.55→0.65），即相对最强基线（A1）提升 0%–18%。依据：C3 单调变换不改变 AUC（§4.3.2），增益完全来自 C1+C2 的假设多样性信号 |
| coverage@90% | N/A | 通常 60%–75% | 88%–92% | 保守 +13 pp（75%→88%）、乐观 +32 pp（60%→92%）。依据：校准目标即 90% 覆盖率（§2.3.1），具体幅度取决于 raw σ 偏差程度（待验证） |

> **MHFormer 基线值说明**：MHFormer 卡报告"约 18.92M 参数、1.03G FLOPs、3 个假设"，但 cards 中未直接给出 351 帧 CPN 输入下的 MPJPE 具体数值（待验证），公开 leaderboard 一般在 45–50 mm 量级。本报告以 ≤1.5 mm 退化为可接受阈值，不强行声称绝对 MPJPE 数值。

## 3.2 消融矩阵

为隔离 C1/C2/C3 各自的贡献并排除"任何后处理都能降低 ECE"的反例，设计以下消融组（含 oracle 上界与 negative control）：

| ID | 配置 | 说明 | 角色 |
|----|------|------|------|
| **B0** | MHFormer 原版（K=3 → Conv1d 单点） | 基线 | 精度参照 |
| **B1** | DiffPose 原版（N=5 → 均值） | 基线 | 精度参照 |
| **B2** | Simple Baseline + 高斯噪声（无多假设） | negative control | 验证"σ 随机≠有用信号" |
| **B3** | DDHPose 多假设（H=5/20，未校准散布） | 多假设扩散基线 | 验证"仅散布不校准"的选择性预测能力 |
| **A1** | C1 only（保留 K 假设 + 朴素方差，不校准） | 朴素方差基线 | 隔离 C2+C3 贡献 |
| **A2** | C1 + C2（+ 解剖学重加权） | 隔离 C3 贡献 | 验证重加权价值 |
| **A3** | C1 + C3（+ 温度缩放，无重加权） | 隔离 C2 贡献 | 验证校准价值 |
| **F** | C1 + C2 + C3（完整方法） | 主实验 | 最终结果 |
| **O1** | Oracle：用真实误差反推 σ（如 $\sigma_i = e_i$） | 上界 | 校准方法的天花板 |
| **O2** | Oracle：每帧用最近邻 KNN 的误差作为 σ | 弱上界 | 验证"信号可学性" |
| **N1** | σ = 常数 | negative control | ECE 应退化 |
| **N2** | σ = 随机数 | negative control | AUC 应接近 0.5 |

### 3.2.1 消融矩阵的预期结果

| 指标 | B0/B1 | A1 | A2 | A3 | F | O1/O2 | N1/N2 |
|------|-------|-----|-----|-----|---|-------|-------|
| MPJPE | 基线 | ≈基线 | ≈基线 | ≈基线 | ≤基线+1.5 | N/A | N/A |
| NLL | N/A | 高 | 中 | 中 | 低 | 最低 | 最高 |
| ECE | N/A | 0.15–0.30 | 0.10–0.20 | ≤0.08 | ≤0.05 | ≈0 | 高 |
| AUC | 0.5 | 0.55–0.60 | 0.58–0.62 | 0.55–0.60 | ≥0.6 | 0.7–0.8 | ≈0.5 |
| coverage@90% | N/A | 60%–75% | 70%–80% | 85%–92% | 88%–92% | ≈90% | 极端值 |

如果 F 行的 ECE/AUC/coverage 同时满足目标，且 O1 的 AUC 显著高于 F（证明校准方法未触及天花板）、N1/N2 的 AUC 接近 0.5（证明 σ 确实携带信号而非随机后处理），则核心假设成立。

## 3.3 基线方法

按 idea 定稿，本实验包含四条基线：

1. **MHFormer 点估计**（B0） — `codebases/MHFormer.md` §3 与 §4 的原版训练/评测流程。351 帧输入，CPN 2D，S1/5/6/7/8→S9/11。
2. **DiffPose 均值-of-5**（B1） — `codebases/Diffpose.md` §3 的均值坍缩流程，需手动设置 `test_times=5`、`test_timesteps=5`（仓库 YAML 默认为 `test_times=1`、`test_timesteps=2`，见 `codebases/Diffpose.md` §5）。
3. **本方法朴素方差（C1 only，不校准）**（A1） — 与 F 完全相同骨干但无 C2/C3，用于隔离校准本身的贡献。
4. **DDHPose 多假设输出（H=5/20）**（B3） — 取 DDHPose (W4393158891) 在 $H=5$ 与 $H=20$ 假设数下的多假设散布作为未校准不确定性参照（`cards/disentangled_diffusion_based_3d_human_pose_estimation_with_hierarchical_spatial_.json` eval_setup：假设数 $H \in \{1, 5, 20\}$，最优 MPJPE 39.0 mm）。该基线用于验证"仅保留多假设散布而不做校准"是否足以产生有用的选择性预测信号。

辅助基线：Simple Baseline（W2612706635）作为点估计下限参照，不在主表内。

**与 POT/UGRN 的定性对比**：POT/UGRN (W4382457852) 的逐关节 $\sigma$ 仅在训练时通过 UG-Sampling 增强鲁棒性，推理时不使用、不输出、不校准；本方法的不确定性在推理时显式输出、经温度缩放/isotonic 校准、并直接服务于下游选择性预测（AUC 评估）。二者互补而非竞争：POT 的 $\sigma$ 可作为本方法假设打分的辅助特征（未来工作）。

## 3.4 数据集要求与预处理

### 3.4.1 主数据集：Human3.6M

- **划分**：S1/S5/S6/S7/S8 训练；S9/S11 测试（与 MHFormer/DiffPose/Anatomy-Aware/PoseFormer 完全一致，参见 `cards/_themes.json` 母题 5）。
- **2D 输入**：CPN 检测 2D（`data_2d_h36m_cpn_ft_h36m_dbb.npz`，`codebases/MHFormer.md` §3）。
- **评测协议**：Protocol #1（MPJPE，根关节对齐），与所有基线一致；Protocol #2（P-MPJPE，Procrustes 对齐）作为辅助指标。
- **帧长**：351 帧（MHFormer 默认，`codebases/MHFormer.md` §5 `--frames 351`）。
- **关节数**：17（`common/opt.py:40-41 --n_joints --out_joints`，H36M 标准）。

### 3.4.2 校准验证集划分

为避免温度缩放在测试集上过拟合，把 S9 进一步切分：

- S9 的前 50% 作为**校准验证集**（拟合 $T^*$ 与 isotonic）；
- S11 全部作为**最终测试集**（仅评估，不参与任何超参选择）。

> **若 S9 数据量不足**：可改为 5-fold 交叉验证拟合校准参数，在 S9 全集上报告校准后指标，S11 仍作最终测试。

### 3.4.3 预处理

- **根关节对齐**：`common/h36m_dataset.py` 已执行 `pos_3d[:, 1:] -= pos_3d[:, :1]`（`codebases/MHFormer.md` §3）。
- **flip 增强**：保留 `main.py:70-87 input_augmentation` 的左右翻转测试增强，对 K 维假设独立处理（见 §2.1.6）。
- **CPN 2D 噪声建模**：不做额外清洗，直接消费仓库默认 npz，以保留与基线可比的 2D 误差分布。

### 3.4.4 辅助数据集（可选扩展）

- **MPI-INF-3DHP**：跨数据集泛化检验（GT 2D 输入，8 视角，9 帧）。
- **3DPW**：野外场景泛化检验（不在主表，仅作 stress test）。母题 5 指出"3DPW 等野外数据集存在但几乎无方法在其上报告"，本方法若在 3DPW 上仍能维持 AUC ≥ 0.55，将是显著的差异化卖点。

## 3.5 评估协议

### 3.5.1 主实验流程

```
1. 训练阶段（GPU）
   ├─ 用 S1/5/6/7/8 训练改造后的 MHFormer（保留 K=3 假设）
   ├─ 监督：S1 (min-loss) + λ=0.01 的 S3 (NLL) 项
   └─ 训练 30 epoch（待验证，repo 卡 §5 仅记录 lr 调度为 0.95 每 5 epoch），lr 0.95 decay 每 5 epoch

2. 推理阶段（GPU）
   ├─ 在 S9 校准验证集上跑前向，记录 (σ_raw, e) 对
   ├─ 在 S11 测试集上跑前向，记录 (σ_raw, e) 对
   └─ 输出 (μ, σ_raw) for each (frame, joint)

3. 校准阶段（CPU）
   ├─ 用 S9 上的 (σ_raw, e) 拟合 T* 与 isotonic g
   └─ 把 g(T* σ_raw) 应用到 S11 的 σ_raw → σ_cal

4. 评估阶段（CPU）
   ├─ MPJPE：用 μ 计算
   ├─ NLL：用 (μ, σ_cal) 与 GT 计算
   ├─ ECE：分 15 桶，对比预测置信与实际误差频率
   ├ selective AUC：扫 σ 阈值，画 (拒识比例, 剩余 MPJPE) 曲线
   └ coverage@90%：检验 |e| ≤ 1.645 σ_cal 的样本比例

5. 按动作分桶
   └ 15 类（Walking, Eating, Smoking, ...）分别报告 MPJPE/ECE/AUC
```

### 3.5.2 结果字段（result.json schema）

```json
{
  "mpjpe": <float, mm>,
  "nll": <float>,
  "ece": <float>,
  "selective_auc": <float>,
  "coverage@90%": <float>,
  "per_action_mpjpe": {"Walking": <float>, "Eating": <float>, ...},
  "per_action_ece": {"Walking": <float>, "Eating": <float>, ...},
  "T_star": <float>,
  "lambda": <float>,
  "K": 3,
  "ablation": "C1+C2+C3" | "C1_only" | "C1+C2" | "C1+C3" | ...
}
```

### 3.5.3 统计显著性

- 对主指标（MPJPE、NLL、ECE、AUC）在 3 个随机种子上重复训练，报告均值±标准差。
- 用配对 bootstrap 检验本方法 vs A1（朴素方差）的显著性，p < 0.05 视为显著。

## 3.6 计算资源估算表

### 3.6.1 训练资源

| 资源 | MHFormer 原版 | 本方法改造 | 增量 | 说明 |
|------|---------------|-----------|------|------|
| GPU | 1×RTX 3090 | 1×RTX 3090 | 0 | `cards/mhformer*.json` 确认原版单卡可训 |
| 显存 | ~12 GB（351 帧 batch 256 待验证） | ~14 GB | +2 GB | K=3 输出 + 校准缓存 |
| 训练时长 | ~6 小时（待验证） | ~7 小时 | +1 小时 | 仅多了重加权与 NLL 计算 |
| 参数量 | 18.92 M | 18.92 M | ≈0（约 100 参数，可忽略） | 拆分不改变总参数量：Conv1d(1536,51,1) → 3×Conv1d(512,51,1)，乘加数相同 |
| FLOPs | 1.03 G | 1.03 G | ≈0（拆分不改变总乘加数） | 1536×51 = 3×512×51，前向计算量完全相同 |

> **显存与训练时长说明**：原版 351 帧 batch 256 在 RTX 3090（24 GB）上的具体占用与训练时长未在 cards/repo 卡中明确给出，此处为按经验估算，**待实际跑通后验证**。若显存溢出，降级到 batch 128 + 梯度累积。

### 3.6.2 推理资源

| 资源 | 数值 | 说明 |
|------|------|------|
| 推理时间 | ~30 ms/frame（待验证） | 分项（基线 / 重加权 / 校准）均待实测，cards 与 repo 卡无推理时间数据 |
| 校准拟合 | <1 分钟 | 1M 样本 isotonic 回归，sklearn 单线程 CPU |
| 端到端延迟 | ~50 ms/frame | 含 2D 检测器（CPN 离线缓存时 ~30 ms） |

### 3.6.3 DiffPose 扩展资源

如需在 DiffPose 上重复本方法（B1 + C1+C2+C3）：

| 资源 | DiffPose 原版 | 本方法改造 |
|------|---------------|-----------|
| GPU | 1×（待验证，cards 未给具体型号） | 同 |
| 采样步数 | 5 步 DDIM（论文报告值；仓库 YAML 默认为 2 步，见 `codebases/Diffpose.md` §4） | 5 步 DDIM（需手动设置 `--test_timesteps 5`） |
| 重复采样 N | 5 | 5（必须显式设 `--test_times 5`） |
| 推理开销 | N×单次 | N×单次（不变） |
| 校准开销 | 同 MHFormer | <1 分钟 CPU |

> **DiffPose 算力说明**：DiffPose 卡未明确给出训练 GPU 型号与时长（`cards/diffpose*.json` 的 `resources` 字段标注"具体算力需求文中未给出"），需在实际复现时验证。

### 3.6.4 数据存储

| 数据 | 大小 | 来源 |
|------|------|------|
| Human3.6M 原始 3D | `data_3d_h36m.npz` | `codebases/MHFormer.md` §3 |
| CPN 2D | `data_2d_h36m_cpn_ft_h36m_dbb.npz` | `codebases/MHFormer.md` §3 |
| GMM 2D（DiffPose 专用） | `data_2d_h36m_cpn_ft_h36m_dbb_gmm.npz` | `codebases/Diffpose.md` §2 |
| 预训练权重 | ~75 MB（MHFormer 351 帧，待验证） | `codebases/MHFormer.md` §3 |

---

# 4. 可行性评估

## 4.1 实现复杂度对比

### 4.1.1 与更轻替代路线对比

存在多种"为 3D 姿态加不确定性"的潜在路线，本节论证本方案在工程复杂度与信号质量之间的位置：

| 路线 | 改造量 | 信号质量 | 风险 |
|------|--------|----------|------|
| **L1: Monte Carlo Dropout** | 在 Simple Baseline 推理时启用 dropout，跑 K 次 | 弱（dropout 方差与误差相关性低，待验证） | 信号噪声比差 |
| **L2: Deep Ensemble** | 训练 K=5 个独立 MHFormer，看输出散布 | 中 | 训练成本 ×K，工程量大 |
| **L3: Bayesian Neural Network** | MHFormer 的 Conv1d 权重换成高斯变分后验 | 强 | 训练不收敛风险高，Pyro/BayesianLayers 框架适配复杂 |
| **L4: Evidential Deep Learning** | 让网络直接输出 NIG 分布参数 | 中 | 损失函数设计复杂，3D 姿态上无成熟先例 |
| **L5: 本方案（C1+C2+C3）** | MHFormer 最后一步改造 + sklearn 校准 | 中-强（直接利用已有多假设） | 仅需 K=3 假设，单卡可训 |

**结论**：L5 在"信号质量足够"与"工程量最小"之间取最优折中——它复用了 MHFormer 已经编码好的多假设结构，无需重新训练 5 个网络（L2）、无需引入新概率框架（L3）、无需设计新损失（L4）。

**量化对比**（以 L5 为参照 1×）：

| 路线 | 训练成本倍数 | 推理成本倍数 | 代码改动量 | 依据 |
|------|-------------|-------------|-----------|------|
| L1: MC Dropout | ~1×（单次训练 Simple Baseline） | ~30×（K=30 次前向，待验证：K 的典型取值 10–50） | <10 行（推理时启用 dropout） | Simple Baseline 单次 FLOPs 远小于 MHFormer 1.03G（待验证），但需 K 次重复 |
| L2: Deep Ensemble | **5×**（K=5 个独立 MHFormer，各 ~6 h → 共 ~30 h） | **5×**（5 个网络并行前向） | ~50 行（训练循环封装） | MHFormer 卡确认单卡 RTX 3090 可训（`resources` 字段），训练时长 ~6 h 为经验估算（待验证） |
| L3: BNN | ~2–3×（变分后验 KL 项增加反向传播开销，待验证） | ~2×（每次前向需采样权重） | ~300+ 行（Pyro/Blitz 框架适配，待验证） | 无 3D 姿态先例，收敛 epoch 数不可预估 |
| L4: Evidential DL | ~1.2×（仅损失函数增加 NIG 参数计算，待验证） | ~1×（单次前向） | ~100 行（新损失 + 输出头改造） | 3D 姿态上无成熟先例，调参轮次不可预估 |
| **L5: 本方案** | **1×**（~7 h vs 基线 ~6 h，增量 +17%） | **1×**（增量 +5.5 ms/frame，待实测） | +220 行 / −10 行 | §3.6.1–3.6.2；增量来自 3 个独立回归头 + 重加权 + 校准 |

> 注：L5 相对 L2 训练成本节省 **~5×**（7 h vs 30 h），相对 L1 信号质量显著更优（直接利用已有多假设结构而非随机 dropout 扰动）；相对 L3/L4 工程确定性更高（无框架适配与新损失收敛风险）。训练/推理绝对时长来源为 §3.6 估算，标注"待验证"处需实际跑通后确认。

### 4.1.2 改造量量化

按 `codebases/MHFormer.md` 的"改造接口点"列表，本方案的代码改动量如下：

| 文件 | 改动行数 | 类型 | 风险 |
|------|----------|------|------|
| `model/mhformer.py:51-54` | -4 / +8 | 替换回归头 | 中（预训练权重兼容性） |
| `model/mhformer.py:71-78` | -3 / +8 | 改 forward 输出形状 | 低 |
| `model/module/trans_hypothesis.py:207-210` | -3 / +5 | 取消 cat | 低 |
| `common/utils.py:18-23 test_calculation` | +15 | 支持分布输入 | 低 |
| `main.py:41-43, 59-61` | +6 | 5 维输出索引调整 | 低 |
| `main.py:70-87 input_augmentation` | +4 | K 维独立 flip | 低 |
| 新增 `calibration.py` | +60 | sklearn 校准 | 低（新文件） |
| 新增 `bone_stats.py` | +40 | 解剖学重加权（骨长/骨方向统计） | 低（新文件） |
| 新增 `metrics.py` | +80 | NLL/ECE/AUC/coverage | 低（新文件） |

总计约 **+220 行 / -10 行**——属于中等改造量。最大单点风险是 `model/mhformer.py:51-54` 的回归头替换与预训练权重兼容（见 §4.2 R1）。

### 4.1.3 DiffPose 改造量

DiffPose 的改动更小：

| 文件 | 改动行数 | 类型 |
|------|----------|------|
| `runners/diffpose_frame.py:296-299` | -1 / +3 | 保留 N 样本 |
| `configs/*.yml` | 1 行 | `test_times: 5` |
| 新增校准/指标脚本 | 同 MHFormer | 复用 |

总计约 **+90 行**，最小侵入。

## 4.2 外部依赖风险表

| ID | 依赖项 | 风险描述 | 影响等级 | Plan B |
|----|--------|----------|----------|--------|
| **R1** | MHFormer 预训练权重兼容性 | 旧 `self.regression` 权重形状 `(51, 1536)`，新 3 个独立头为 `(51, 512)` ×3，无法直接加载（`codebases/MHFormer.md` §6 风险与未知） | 高 | (a) 从头训练（cost ~6 小时，可接受）；(b) 用 SVD/矩阵分解把旧权重分解为 3 个低秩近似（理论上可恢复大部分信息，待验证）；(c) 用旧权重初始化共享部分 + 重训回归头（迁移学习） |
| **R2** | K=3 假设方差被离群假设污染 | $K$ 太小，1 个离群假设即可让 σ 翻倍 | 中 | C2 解剖学重加权已对冲；进一步可加 robust statistics（如 median absolute deviation）替代均值方差 |
| **R3** | 训练监督策略选择（S1/S2/S3） | min-loss 可能让 K-1 条假设退化；均值损失与原版等价但坍缩；NLL 损失 $\lambda$ 调参敏感 | 中 | 实验矩阵上同时跑 S1、S1+λS3、S2 三种，以 ECE/MPJPE 双目标选最佳 |
| **R4** | 温度缩放过拟合验证集 | 若 S9 切分不当，$T^*, g$ 可能记住验证集噪声 | 中 | (a) 5-fold 交叉验证拟合；(b) 用 S9 一半校准、另一半做"校准验证"early-stop；(c) 报告 S9 vs S11 的 ECE 差异，差值 >0.03 视为过拟合 |
| **R5** | CPN 2D 噪声传播至 σ | DiffPose 卡已指出"2D 检测误差传播"；若 CPN 在某类动作上系统性偏差，σ 会学到错误信号 | 中 | (a) 报告 GT 2D 输入下的对照实验（DiffPose GT 配置 18.9 mm vs CPN 36.9 mm）；(b) 按 2D 检测置信度分桶分析 σ 与 2D 置信度的相关性 |
| **R6** | 计算开销 K 倍前向（DiffPose） | DiffPose 需要 N=5 次完整逆扩散，比单次回归方法慢 ~5× | 中 | DDIM 已加速至 5 步；可考虑 DDIM 5 步 + 共享 context encoder 的优化（待验证） |
| **R7** | Human3.6M 野外泛化 | 母题 5 指出"Human3.6M 仅 11 人、15 种室内动作，与真实世界存在巨大鸿沟" | 中-高 | 主表只报 H3.6M；扩展实验在 3DPW 上 stress test；论文中明确标注此为局限 |
| **R8** | PyTorch 1.7.1 版本风险 | repo 卡 README 指定 1.7.1，新硬件（Apple Silicon、CUDA 12+）可能不兼容 | 低 | 升级到 PyTorch 2.x，逐项验证 MHFormer/DiffPose 在新版本下的前向数值一致性 |
| **R9** | GMM 数据生成脚本缺失 | `codebases/Diffpose.md` §6 指出仓库只消费 `data_2d_h36m_*_gmm.npz`，无生成脚本 | 中 | 仅在 DiffPose 路线需要；MHFormer 路线无需 GMM，可作为 DiffPose 扩展实验的前置阻塞 |
| **R10** | 各向同性高斯假设过强 | 真实 3D 误差在 x/y/z 三轴上分布不均（如深度 z 误差显著大于 x/y） | 低 | 第一版用各向同性；若 coverage@90% 系统性偏离，升级到对角协方差 |

### 4.2.1 风险等级矩阵

```
影响等级
高 │  R1●                       
中 │  R3●  R2●  R4●  R5●  R6●  R7●  R9●
低 │            R8●              R10●
   └────────────────────────────────────
      低      中       高       极高   发生概率
```

R1 是唯一"高影响+中高概率"风险，需要优先对冲（首选从头训练，最稳健）。

## 4.3 错误传播风险

### 4.3.1 错误传播链

```
CPN 2D 检测器  ─误差→  MHFormer/DiffPose 上下文编码  ─误差→  K 假设  ─方差污染→  σ_raw
                                                                       │
                                                              解剖学重加权（C2）→ σ_rew
                                                                       │
                                                              温度缩放+isotonic（C3）→ σ_cal
                                                                       │
                                                              ┌────────┴────────┐
                                                          高估 σ          低估 σ
                                                              │              │
                                                       虚假置信（保守）    覆盖率不足（激进）
                                                              │              │
                                                       下游拒识过多      下游误识风险
```

### 4.3.2 关键传播点分析

1. **CPN 2D → MHFormer**：母题 2 指出 GT 2D 与 CPN 2D 之间存在 10–13 mm MPJPE 鸿沟（PoseFormer 卡：31.3→44.3）。这部分误差是不可消除的输入噪声，会同时影响 μ 与 σ——好在 σ 学到的"哪些姿态不可信"信号仍有效。
2. **K 假设 → σ_raw**：若假设间缺乏多样性（如 §1.3.3 分析的 MSE 损失导致退化为扰动副本），σ_raw 会系统性低估。C2 重加权对冲不了这个问题，需要 S1（min-loss）训练策略提供多样性。
3. **σ_raw → σ_cal**：温度缩放是单调变换，不会改变 AUC；isotonic 回归也是单调的，不会破坏 σ 与误差的排序关系。因此 **C3 只影响 coverage/ECE，不影响 selective AUC**——AUC 的提升完全来自 C1+C2。

### 4.3.3 错误传播缓解策略

| 传播点 | 缓解策略 |
|--------|----------|
| CPN 2D 噪声 | 在评测表中同时报告 GT 2D 输入下的结果作为上界 |
| 假设退化 | 训练时加多样性正则（如假设间 L2 距离惩罚），属于 S1 策略的扩展 |
| 校准过拟合 | 5-fold 交叉验证 + S9/S11 双重评估 |
| 解剖学先验失效 | 报告跨 subject 骨长分布，标记 outlier subject |

### 4.3.4 最坏情况退化下界分析

当上游组件完全失效时，系统退化水平与兜底机制如下：

| 失效组件 | 最坏情况 | 系统退化到 | 退化下界（性能地板） | 兜底机制 |
|----------|----------|-----------|---------------------|----------|
| **C1（多假设保留）** | K 条假设完全退化为相同输出（σ→0），等价于 §1.3.3 分析的 MSE 坍缩 | B0（MHFormer 原版点估计） | MPJPE ≈ 基线（~44 mm），但丧失全部不确定性信号（AUC = 0.5） | 回退到 B0 点估计模式；触发 Go/No-Go #1（§4.5.1：退化 >3 mm 即停止） |
| **C2（解剖学重加权）** | 骨长估计完全错误（如 subject 错配），$w_k$ 退化为均匀或反转 | A1（C1 only，朴素方差） | ECE 退化至 0.15–0.30，AUC 保持 0.55–0.60（C2 不影响排序） | 设 $w_k = 1/K$（均匀权重），等价于跳过 C2；代码中加 assert 检测骨长合理性 |
| **C3（温度缩放+isotonic）** | 校准集与测试集分布严重偏移，$T^*$ 和 $g$ 完全失效 | A2（C1+C2，未校准） | ECE 退化至 0.10–0.20，coverage 偏离至 70%–80%，但 AUC 不变（单调性保证，§4.3.2） | 跳过校准直接输出 $\sigma_{\text{rew}}$；或切换为 5-fold 交叉校准（R4 Plan B） |
| **CPN 2D 检测器完全失效** | 2D 输入为纯噪声或全零 | 所有方法（含基线）均失效 | MPJPE 无上界（>300 mm 离群值，母题 2），但 σ_cal 应同步增大 | 下游选择性预测拒识（σ > 阈值时拒绝输出）；本方法相比基线的**唯一优势**：能发出"不可信"信号 |
| **全链路最坏（C1+C2+C3 同时失效 + 2D 退化）** | 所有新增组件失效 | B0 点估计 + 无不确定性 | 等价于未改造的 MHFormer 原版 | 系统可完整回退到原版代码路径（改造仅涉及末端回归头，骨干不受影响） |

**关键结论**：本方案的退化下界是**MHFormer 原版点估计性能**（MPJPE ~44 mm），即新增组件在最坏情况下不会让点估计精度低于未改造基线——因为 C2/C3 仅作用于 σ 而非 μ，C1 的 μ 仍为 K 条假设均值（与原版 Conv1d 聚合在 K 条假设相同时数值等价）。兜底策略为"逐组件可旁路"：任一组件失效时跳过该组件即可退化到上一级消融配置（F→A2→A1→B0），无需重新训练。

## 4.4 性能与成本量化

### 4.4.1 工程成本

| 阶段 | 工作量（人天） | 说明 |
|------|----------------|------|
| 环境搭建与 MHFormer 复现 | 2 | conda env + 数据下载 + 跑通官方测试 |
| C1 改造（MHFormer 回归头） | 3 | §2.1.2 步骤 A/B/C + 训练验证 |
| C2 解剖学重加权 | 2 | 骨长/骨方向统计 + 关节角检查 + 重加权集成 |
| C3 校准模块 | 2 | sklearn + scipy 实现 + 验证 |
| 评测脚本（NLL/ECE/AUC/coverage） | 2 | 新增 `metrics.py` |
| 消融实验跑通 | 3 | B0/B1/B3/A1/A2/A3/F + 3 随机种子 |
| DiffPose 扩展（可选） | 3 | C1 改造 DiffPose + 配置调整 |
| 报告与可视化 | 2 | result.json 解析 + 误差-不确定性散点图 |
| **总计** | **19 人天 ≈ 4 周** | 单人投入 |

### 4.4.2 GPU 成本

| 实验 | GPU 时 | 数量 | 总 GPU 时 |
|------|--------|------|-----------|
| MHFormer 原版复现 | 6 | 1 | 6 |
| 本方法训练（3 种子 × S1/S2/S3 监督策略） | 7 | 9 | 63 |
| 推理（S9+S11 全集） | 0.5 | 9 | 4.5 |
| DiffPose 扩展（可选） | 8 | 3 | 24 |
| **总计** | | | **~98 GPU 时 ≈ 4 RTX 3090 天** |

按 AWS p3.2xlarge（V100，约 3 美元/小时）等价计算，约 300 美元；自有单卡 RTX 3090 可在 5 天内完成全部实验。

### 4.4.3 逐组件耗时预算表

| 新增组件 | 训练额外开销 | 推理额外开销 | 参数量增量 | 依据 |
|----------|-------------|-------------|-----------|------|
| **C1: 多假设保留（3 独立回归头）** | +1 h（总 7 h vs 基线 6 h；增量来自 min-loss 对 K 条假设分别计算梯度） | +0 ms（前向路径与原版等价，仅输出形状变化） | ≈0（约 100 参数，可忽略） | §3.6.1；FLOPs 增量亦≈0（1536×51 = 3×512×51），MHFormer 卡确认 18.92M/1.03G |
| **C2: 解剖学重加权** | 可忽略（纯 numpy 骨长/关节角计算，无梯度，§2.2.4） | +0.5 ms/frame（$O(K \cdot |\mathcal{B}|)$ 骨长 + $O(K \cdot 4)$ 关节角，§2.2.4） | 0（无新参数） | §2.2.4 "计算开销……可忽略" |
| **C3: 温度缩放 + isotonic 校准** | 0（后处理，不参与训练，§2.3.4） | +5 ms/frame（CPU，isotonic predict）；一次性拟合 <1 min（1M 样本，sklearn 单线程） | 0（$T^*$ 为标量，$g$ 为分段常数） | §3.6.2；§2.3.4 "无外部 GPU 依赖" |
| **合计增量** | **+1 h / 训练轮**（~17%） | **+5.5 ms/frame**（待实测） | **≈0**（可忽略） | — |

> 注：以上绝对时长为 §3.6 估算值，标注于该节"待实际跑通后验证"。C1 训练开销的主要来源是 min-loss 策略需对 K 条假设分别计算梯度（§2.1.5），而非网络结构变化本身。

## 4.5 时间线里程碑表

| 周次 | 里程碑 | 交付物 | 验收标准 |
|------|--------|--------|----------|
| **W1** | 环境 + MHFormer 复现 | 复现 MPJPE 与公开 leaderboard 一致（误差 ±1 mm） | 通过 `python main.py --test --previous_dir checkpoint/pretrained/351` |
| **W2** | C1 改造 + 训练验证 | 改造后 MHFormer 在 K=3 输出下 MPJPE 退化 ≤1.5 mm | 输出形状 `(B, 3, 351, 17, 3)` 正确，训练损失收敛 |
| **W3** | C2 + C3 集成 + 评测脚本 | `metrics.py` 输出 NLL/ECE/AUC/coverage | 单元测试通过，oracle 实验 O1 AUC > 0.7 |
| **W4** | 主实验跑通 + 消融矩阵 | `result.json` 包含所有消融组 | F 行 ECE ≤0.05、AUC ≥0.6、coverage 88%–92% 同时满足 |
| **W5** | DiffPose 扩展 + 3DPW stress test | DiffPose 路线 result.json | DiffPose 上 MPJPE 退化 ≤2 mm、AUC ≥0.55 |
| **W6** | 报告与可视化 | 论文级图表（误差-不确定性散点、按动作分桶、coverage 曲线） | 可投稿 CVPR/ECCV workshop 或主会的图表质量 |
| **W7–W8** | 论文撰写 + rebuttal 准备 | 论文初稿 | 满足 8 页正文 + 附录 |

### 4.5.1 关键里程碑的 Go/No-Go 判据

- **W2 末 Go/No-Go #1**：若改造后 MPJPE 退化 >3 mm，立即停止 C2/C3 投入，回头查 C1 改造的数值稳定性。
- **W4 末 Go/No-Go #2**：若 F 行 ECE >0.10 或 AUC <0.55，本方向的核心假设证伪，转向 L2（Deep Ensemble）或 L4（Evidential）替代路线。
- **W6 末 Go/No-Go #3**：若 3DPW 上 AUC <0.50，论文定位从"野外可用"降级为"室内基准上首次校准"，仍可投但卖点减弱。

## 4.6 综合判级与决策路径建议

### 4.6.1 综合判级

| 维度 | 评级 | 理由 |
|------|------|------|
| 科学新颖性 | 中-高 | 3D 姿态多假设→校准的路线无直接先例（idea 3/3 通过）；但 ensemble-disagreement-as-uncertainty 在分类领域是已知范式 |
| 工程可行性 | 高 | 改造点明确（`codebases/MHFormer.md` §1-§2 已定位），单卡 RTX 3090 可训，sklearn 校准无新算法风险 |
| 实验风险 | 中 | R1（预训练权重）+ R4（温度过拟合）+ R7（野外泛化）三大风险，均有 Plan B |
| 数据可得性 | 高 | Human3.6M 与 CPN 数据均可在仓库 README 指引下获得；3DPW 公开 |
| 计算成本 | 低 | ~98 GPU 时，单卡 5 天 |
| 时间窗口 | 紧 | 6–8 周完成 MVP 与论文初稿，CVPR/ECCV 截稿前可完成 |

**总体判级：B+（推荐投入实验验证）**

### 4.6.2 两条决策路径建议

**路径 A：MVP 优先（推荐）**

- 范围：仅 MHFormer + C1+C2+C3 + Human3.6M（S1/5/6/7/8→S9/11）。
- 时间：4 周。
- 验收：MPJPE 退化 ≤1.5 mm、ECE ≤0.05、AUC ≥0.6、coverage 88%–92% 四项同时满足。
- 后续：若 MVP 通过，再扩展到 DiffPose（W5）与 3DPW（W5–W6），投向 CVPR/ECCV 主会。
- 失败止损：若 W4 末 Go/No-Go #2 触发，写 negative result 短文投 ICCV workshop（"为什么多假设分歧在 3D 姿态上不能直接作为不确定性"），仍具学术价值。

**路径 B：完整路线（仅当 MVP 提前通过且算力充足）**

- 范围：MHFormer + DiffPose + 跨数据集（H3.6M / MPI-INF-3DHP / 3DPW） + 监督策略消融（S1/S2/S3）。
- 时间：8 周。
- 验收：在三个数据集上均满足 MVP 验收标准，且监督策略消融给出 S1 优于 S2 的明确证据。
- 后续：投 CVPR/ECCV 主会，并以"C1+选择性预测"为 demo 拓展到康复/AR 应用合作。

**推荐路径 A**：4 周内的 MVP 即可证伪或证成本方向的核心假设，沉没成本可控。路径 B 仅在 MVP 提前 1 周以上完成时启动。

---

# 5. 结论

本报告评估了"在 MHFormer/DiffPose 等多假设 3D 姿态估计骨干上保留 K 条假设的逐关节分布、用解剖学一致性重加权、再用温度缩放与 isotonic 回归校准"这一研究方向的可行性。**方案核心**是把 `model/mhformer.py:51-54` 的 1×1 Conv1d 回归头坍缩操作替换为 `nn.ModuleList` 三个独立回归头，相应取消 `model/module/trans_hypothesis.py:207-210` 的拼接；DiffPose 路线把 `runners/diffpose_frame.py:298` 的 `torch.mean` 替换为保留 N 个样本的分布输出，复用 `common/generators.py` 的 GMM 逐关节方差作为先验。**预期收益**：在不显著牺牲 MPJPE（退化 ≤1.5 mm）的前提下，首次为单目 3D 姿态估计提供 ECE ≤0.05、coverage@90% ∈ [88%, 92%]、selective AUC ≥0.6 的校准不确定性，填补"多假设方法承认多解却坍缩为单点"的领域空白，为康复步态分析、AR、机器人选择性预测等下游应用打开"知道不知道"的能力。**主要风险**：MHFormer 预训练权重与改造后回归头不兼容（R1）、温度缩放在验证集过拟合（R4）、Human3.6M 室内基座的野外泛化有限（R7），三者均有明确 Plan B。**时间框架与目标会议**：4 周内完成 MVP（路径 A）验证核心假设，若通过再扩展 4 周完成跨数据集与 DiffPose 路线（路径 B），目标投向 CVPR 2026（截稿预期 11 月）或 ECCV 2026 主会；若核心假设证伪，转为 ICCV 2026 workshop 的 negative result 短文。本方案工程路径明确、外部依赖成熟、计算成本可控，**建议立即投入 MVP 验证**。

