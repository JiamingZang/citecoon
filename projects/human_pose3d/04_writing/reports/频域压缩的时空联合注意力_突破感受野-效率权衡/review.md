# 审校报告: 频域压缩的时空联合注意力_突破感受野-效率权衡
> 对抗性审校 · 2026-07-21 23:13 · 对象: 频域压缩的时空联合注意力_突破感受野-效率权衡/report.md · model: qmodel_preview


## 问题清单

### [P0] §1.1 第5段 — STCFormer Table 1 逐动作误差数据编造与张冠李戴

> 报告原文：
> "STCFormer [W4386076485] Table 1 显示，即使是最优方法，"Sitting Down"（52.5mm）和"Smoking"（56.8mm）等涉及复杂时空耦合的动作误差远高于"Walking"（26.2mm）等周期性动作。"

**核查过程**：Read `papers/3d_human_pose_estimation_with_spatio_temporal_criss_cross_attention.md`，定位 Table 1 逐动作误差。STCFormer-L (T=243, CPN) 实际数据为：
- Sitting Down: **56.8mm**
- Smoking: **41.8mm**
- Walking: **26.2mm**

STCFormer 标准版 (T=243)：Sitting Down 57.4mm, Smoking 41.8mm, Walking 27.1mm。

**问题**：
1. "Sitting Down（52.5mm）"——52.5mm 在论文任何版本中均不存在，系编造数字。
2. "Smoking（56.8mm）"——56.8mm 实为 Sitting Down 的误差，被错误地安到了 Smoking 上。Smoking 真实误差仅 41.8mm。
3. Walking 26.2mm 正确（STCFormer-L）。

**建议修改**：更正为 STCFormer-L 实际数据："Sitting Down（56.8mm）和 Smoking（41.8mm）……Walking（26.2mm）"。若需论证高耦合动作误差大，应选取真正高误差的动作（如 Sitting Down 56.8mm、Directions 等），而非编造数字。

---

### [P0] §1.2 路线一 StridedTransformer — MPJPE 43.7mm 的帧数标注错误

> 报告原文：
> "MPJPE 43.7mm（T=243, CPN），参数量仅 4.23M。"

**核查过程**：Read `papers/exploiting_temporal_contexts_with_strided_transformer_for_3d_human_pose_estimati.md`。论文主结果 43.7mm (CPN) 对应的是 **T=351**（351帧模型），而非 T=243。论文 eval_setup 列出感受野 27/81/243/351，headline 结果为 351 帧。cards/ 中该卡片原文："主结果：CPN输入43.7mm MPJPE、35.2mm P-MPJPE"，未标注 T=243。

**问题**：将 T=351 的结果错误标注为 T=243，属于来源不符。

**建议修改**：改为"MPJPE 43.7mm（T=351, CPN）"。

---

### [P1] §1.2 路线一 StridedTransformer — 参数量 4.23M 无来源可查

> 报告原文：
> "参数量仅 4.23M。"

**核查过程**：Read `papers/exploiting_temporal_contexts_with_strided_transformer_for_3d_human_pose_estimati.md`，搜索 "4.23" 及 "param"。论文提取文本中 Table II（计算复杂度/参数量/FPS）内容被截断，仅存表题，4.23M 未出现。cards/ 该卡片 resources 字段仅写"参数量随感受野增大几乎不增加（轻量）"，无具体数字。codebases/ 无 StridedTransformer repo 卡。

**问题**：4.23M 在来源库中无法查证，且未标注"待验证"。

**建议修改**：补充出处（如"原论文 Table II"）或标注"待验证"。

---

### [P1] §1.2 路线三 MotionAGFormer — Table 6 消融结论表述歪曲

> 报告原文：
> "该模型不使用时间位置编码（消融显示加上反而变差，Table 6：无时间编码 38.4mm vs 有时间编码 40.5mm），时序顺序完全依赖 GCNFormer 流保持。"

**核查过程**：Read `papers/motionagformer_enhancing_3d_human_pose_estimation_with_a_transformer_gcnformer_n.md`，定位 Table 6 位置编码消融。实际四组实验为：
| 时间PE | 空间PE | MPJPE |
|--------|--------|-------|
| ✗ | ✗ | 39.3 |
| ✗ | ✓ | **38.4**（最优） |
| ✓ | ✗ | 38.9 |
| ✓ | ✓ | **40.5**（最差） |

**问题**：报告将 38.4mm 描述为"无时间编码"、40.5mm 描述为"有时间编码"，暗示这是仅切换时间 PE 的对照实验。实际上 38.4mm 是"无时间PE + 有空间PE"，40.5mm 是"有时间PE + 有空间PE"——两者均含空间 PE。更关键的是，"无时间PE + 无空间PE"为 39.3mm，加时间PE 后为 38.9mm（反而改善），这与报告"加上反而变差"的表述矛盾。正确结论是：在已有空间PE的前提下，追加时间PE 导致退化（38.4→40.5）。

**建议修改**：改为"Table 6：仅用空间位置编码时 38.4mm（最优），追加时间位置编码后退化至 40.5mm"。

---

### [P1] §4.4 对比表 — STCFormer（T=81）6.5 GFLOPs 无来源

> 报告原文：
> "STCFormer（T=81）：6.5 GFLOPs"

**核查过程**：Read `papers/3d_human_pose_estimation_with_spatio_temporal_criss_cross_attention.md`，搜索 T=81 FLOPs。论文 Table 4 仅报告 T=243 时 STCFormer 标准版 FLOPs = 19,561M。cards/ 该卡片 eval_setup 未提及 T=81 FLOPs。codebases/STCFormer.md 仅有性能参考（P1/P2），无 FLOPs 数据。6.5G 在来源库中无任何出处。

**建议修改**：标注"待验证"或删除此行，仅保留有出处的 T=243 数据。

---

### [P1] §4.4 对比表 — MixSTE（T=81）46.2 GFLOPs 无来源

> 报告原文：
> "MixSTE（T=81）：46.2 GFLOPs"

**核查过程**：Read `papers/mixste_seq2seq_mixed_spatio_temporal_encoder_for_3d_human_pose_estimation_in_vid.md`。论文报告每帧 645M FLOPs（T=243 设置），未单独报告 T=81 的总 FLOPs。若简单以 645×81=52,245M≈52.2G，与 46.2G 不符。cards/ 该卡片亦无 T=81 FLOPs。46.2G 在来源库中无出处且无法由已知数据推算。

**建议修改**：标注"待验证"或删除。

---

### [P1] §1.1 表格 — MixSTE 总 FLOPs 138,623M 来源标注不精确

> 报告原文：
> "| MixSTE [W4312417903] | 交替 STB/TTB | 138,623M | 40.9 |"
> "（数据来源：各论文原文 Table 4/5 及卡片 eval_setup 字段）"

**核查过程**：Read `papers/mixste_seq2seq_mixed_spatio_temporal_encoder_for_3d_human_pose_estimation_in_vid.md`，全文搜索 "138"。MixSTE 论文仅报告每帧 645M FLOPs 及基线总 FLOPs 186,405M，**138,623M 未出现**。Read `papers/3d_human_pose_estimation_with_spatio_temporal_criss_cross_attention.md`，在 STCFormer 论文 Table 4 对比列中找到"138,623M"（作为 MixSTE 的 FLOPs 被引用）。

**问题**：138,623M 实际出自 STCFormer 论文的对比表，而非 MixSTE 原文。报告脚注"各论文原文 Table 4/5"虽可勉强涵盖 STCFormer 论文，但读者自然理解为 MixSTE 自己报告的数字。

**建议修改**：注明"138,623M 引自 STCFormer [W4386076485] Table 4 对比列"。

---

### [P2] §1.2 路线四 — MHFormer 会议归属 "CVPR 2022" 在来源库中无标注

> 报告原文：
> "MHFormer [W4312249545]（CVPR 2022, 426 citations）"

**核查过程**：Read `cards/mhformer_multi_hypothesis_transformer_for_3d_human_pose_estimation.json`。卡片字段仅有 `"year": 2022, "citation_count": 426`，无 venue/conference 字段。papers/ 原文提取中亦未显式标注"CVPR 2022"。

**问题**：虽然 MHFormer 确为 CVPR 2022 论文（外部可查），但来源库内无此信息。严格来说不属于"查证为假"，但违反了"核查依据只能来自来源库"的纪律。

**建议修改**：若无法从来源库确认，可改为"MHFormer [W4312249545]（2022, 426 citations）"。

---

### [P2] 报告头部 — 元信息重复三次

> 报告原文（第2-8行）：
> ```
> > 技术可行性报告 · 2026-07-21 · idea: 频域压缩的时空联合注意力提升.md · ReAct 写作（边写边查证 papers/cards/codebases）
> 
> > 技术可行性报告 · 2026-07-21 · idea: 频域压缩的时空联合注意力提升.md · ReAct 写作（边写边查证 papers/cards/codebases）
> 
> > 技术可行性报告 · 2026-07-21
> ```

**核查过程**：直接观察报告第 2、4、7 行。

**问题**：同一元信息出现三次，属排版错误。

**建议修改**：保留一次即可。

---

### [P2] §1.3 信息论视角 — 装饰性论证段落

> 报告原文（§1.3 前半）：
> "设 $X_s$ 为空间随机变量……差值 $\Delta = I(Y; X_s, X_t) - I(Y; X_s) - I(Y; X_t)$ 即为协同信息（synergy）……分解范式在架构层面将 $\Delta$ 强制置零，这是其精度天花板的根本来源。"

**核查过程**：此段为纯理论推演，来源库中无任何论文做过此信息论分析。STCFormer/MixSTE 的 limitation 字段仅以自然语言承认"分解近似"，未做互信息形式化。

**问题**：删掉此段后，方案的设计依据（STCFormer/MixSTE 作者自认分解不够 + PoseFormerV2 证明频域压缩可行）完全不受损。此段用信息论术语重述了已知观察，未提供新证据或新约束，属装饰性论证。不影响正确性，但增加了"理论深度"的虚假印象。

**建议修改**：可保留作为 intuition 但应明确标注为"非形式化类比"而非定理式陈述；或精简为 1-2 句。

---

## 统计与判定

| 等级 | 数量 |
|------|------|
| P0 | 2 |
| P1 | 5 |
| P2 | 3 |

**总体判定：需修订后发布**

**依据**：存在 2 处 P0 级硬伤——§1.1 逐动作误差表编造了不存在的数字（52.5mm）并将 56.8mm 张冠李戴（实为 Sitting Down 而非 Smoking），§1.2 将 StridedTransformer 的 T=351 结果错标为 T=243。这两处直接损害报告的数据可信度。但报告核心论证链（DCT 压缩使能全联合注意力的复杂度论证、PoseFormerV2 代码级事实、STCFormer 轴分解代码对比、主要基线 MPJPE 数字）经核查与来源库一致，整体框架可信，修订上述错误后可发布。

---

## 附：已核查且确认正确的关键数据点

以下抽查点均核对到了对应来源，确认无误：

| 数据点 | 来源 |
|--------|------|
| PoseFormer MPJPE 44.3mm / P-MPJPE 34.6mm (T=81, CPN) | papers/PoseFormer Table 1 |
| PoseFormer 9.6M 参数、1358M FLOPs (f=81)、269 FPS (GTX 2080Ti) | papers/PoseFormer Table 6 |
| PoseFormer GT 2D MPJPE 31.3mm | papers/PoseFormer Table 2 |
| MixSTE MPJPE 40.9mm / P-MPJPE 32.6mm (T=243)、33.7M 参数、645M/帧 | papers/MixSTE Table 1, Table 5 |
| STCFormer Table 4: T=27→81→243 P1=44.1→42.0→41.0mm | papers/STCFormer Table 4 |
| STCFormer 标准版 4.75M、Large 18.91M；P1=40.5mm / P2=31.8mm (T=243, L) | papers/STCFormer Table 4 |
| STCFormer FLOPs 19,561M (T=243) | papers/STCFormer Table 4 |
| STCFormer Table 5 消融：275.5 / 67.6 / 57.0 / 44.1mm | papers/STCFormer Table 5 |
| PoseFormerV2 Table 5 全部数字（39.2/51.1, 77.2/48.7, 79.4/50.1, 117.3/47.9, 351.7/47.6） | papers/PoseFormerV2 Table 5 |
| PoseFormerV2 Table 6: 纯频域 49.7mm → 加时域 47.1mm (f=n=3, F=81) | papers/PoseFormerV2 Table 6 |
| PoseFormerV2 49.9mm→46.0mm (9帧V1 vs f=n=9,F=81 V2) | papers/PoseFormerV2 Table 4 |
| PoseFormerV2 4.6× 加速 (k=27) | papers/PoseFormerV2 Fig. 1 |
| MHFormer MPJPE 43.0mm (T=351)、18.92M 参数、1.03G FLOPs | papers/MHFormer Table 4/5 |
| MotionAGFormer-B P1=38.4mm、MPI-INF-3DHP 16.2mm、参数 1/4、效率 3× | papers/MotionAGFormer |
| P-STMO MPJPE 42.8mm (T=243)、6.7M 参数、868.5M FLOPs、TDS 仅 0.2mm 增益 | papers/P-STMO Table 3/7 |
| PoseFormerV2 DCT 实现 `common/model_poseformer.py:218` 代码逻辑 | codebases/PoseFormerV2.md |
| STCFormer 轴分解 `model/stcformer.py:74-81` chunk(2,4) 实现 | codebases/STCFormer.md |
| STCFormer `.cuda()` 硬编码、DropPath(0.5)、stride_num 字典不完整 | codebases/STCFormer.md 风险与未知 |
| PoseFormerV2 MixedBlock f//2 硬切风险、torch-dct==0.1.6 依赖 | codebases/PoseFormerV2.md 风险与未知 |
| 复杂度计算 O(4131²)≈17.1M、O(459²)≈210K、STC 1.07M | 算术验证正确 |
| 代码级复杂度对比 (27×17)²=210,849 vs 20,196 vs 2,601 | codebases/STCFormer.md §4 + 算术验证 |
