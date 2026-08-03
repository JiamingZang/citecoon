# 审校报告: 惊讶度门控测试时记忆_时序平滑与消歧的分离建模

> 对抗性审校 · 2026-07-22 · 对象: report.md（2026-07-21 版，含 GDN 双门控升级）
> 核查依据：cards/*.json、papers/*.md、codebases/*.md、ideas/*.md
> 审校员：对抗性审校（与作者利益对立）

---

## 问题清单

### [P2] §1.1 第2段 — TCPFormer "卡片局限栏"字段归属错误

> 报告原文：「TCPFormer（Liu et al., 2025, W4409366800）在卡片局限栏明确记录："随着输入帧数增加，性能提升趋于饱和（如从 243 帧扩到 351 帧仅降低 0.2–0.3 mm 误差）"」

**核查过程：**
- Read `cards/tcpformer_learning_temporal_correlation_with_implicit_pose_proxy_for_3d_human_po.json`。
- Grep "243帧扩到351"：匹配到 `"problem"` 字段，原文为"且随着输入帧数增加，性能提升趋于饱和（如从243帧扩到351帧仅降低0.2-0.3mm误差）"。
- 检查 `"limitation"` 字段：讨论代理长度敏感性、2D 检测依赖、数据集范围、计算开销、PAM 可扩展性，**不含此句**。

**问题：** 数字完全正确，但报告称"局限栏"而实际在"问题栏"（problem）。

**建议修改：** 改为"在卡片问题栏明确记录"或"在卡片中明确记录"。

---

### [P2] §1.2.2 MotionBERT 段 — Table 5 消融描述暗示 2D 数据有助于降至 37.4 mm，实际 2D 数据使 3D 姿态 MPJPE 微升

> 报告原文：「3D 姿态估计消融（Table 5）：从头训练 39.2 mm，加入噪声/遮挡/2D 数据后逐步降至 37.4 mm」

**核查过程：**
- Read `papers/motionbert_a_unified_perspective_on_learning_human_motion_representations.md`，定位 Table 5（"Comparison of pretraining strategies"）。
- 实际数据：无预训练 39.2 → +pretrain 38.8 → +noise 38.1 → +noise+mask **37.4** → +noise+mask+2D **37.5**。
- 37.4 mm 在 noise+mask 阶段（未加 2D 数据）即已达到；加入 2D 重投影数据后 3D 姿态 MPJPE 微升至 37.5 mm（2D 数据主要帮助动作识别和 mesh 重建任务）。

**问题：** "加入噪声/遮挡/2D 数据后逐步降至 37.4 mm"的表述暗示三者共同贡献了 39.2→37.4 的下降，但 2D 数据对 3D 姿态精度无正向贡献（甚至微负）。

**建议修改：** 改为"从头训练 39.2 mm，加入预训练/噪声/遮挡后逐步降至 37.4 mm（加入 2D 重投影数据后 3D 姿态 MPJPE 持平于 37.5 mm，主要增益来自动作识别等下游任务）"。

---

### [P2] §1.2.5 / §2.2.2 — AugLift 引文将两个独立陈述合并为单一论断

> 报告原文（§1.2.5）：「AugLift（Warner et al., 2025, W4414457910）直接指出"时序方法过拟合运动动态，在分布外动作上退化"」
> 报告原文（§2.2.2）：「AugLift（W4414457910）明确指出"时序方法过拟合运动动态，在分布外动作上退化"」

**核查过程：**
- Read `cards/auglift_depth_aware_input_reparameterization_improves_domain_generalization_in_2.json` problem 字段。
- 原文为两个并列分句：① "时序方法过拟合运动动态，稠密图像特征则易学习背景伪相关"；② 前文"跨数据集泛化能力差（H3.6M上40-50mm MPJPE在3DPW上退化至>100mm）"。
- 卡片未将"在分布外动作上退化"直接归于时序方法——"退化至>100mm"是对所有 2D-to-3D 方法的泛化问题描述，"过拟合运动动态"是对时序方法的单独诊断。二者为并列关系而非修饰关系。

**问题：** 报告用引号呈现为直接引用，但实际是对两个分句的合并改写，语义有轻微偏移（将泛化问题特指为时序方法的后果）。

**建议修改：** 去掉引号改为间接引述："AugLift 指出时序方法过拟合运动动态，且 2D-to-3D 方法整体跨数据集泛化能力差（H3.6M 上 40–50 mm 在 3DPW 上退化至 >100 mm）"。或保留引号但只引原文分句："时序方法过拟合运动动态"。

---

### [P2] §4.2 外部依赖风险表 — flash-linear-attention 仓库 URL 与 codebases/ 卡记录不一致

> 报告原文：「GLA（arxiv:2312.06635，开源 github.com/sustcsonglin/flash-linear-attention）」

**核查过程：**
- Read `codebases/flash-linear-attention.md`，该卡记录的仓库来源为 `https://github.com/fla-org/flash-linear-attention`（fla-org 组织仓库）。
- 报告使用 `sustcsonglin/flash-linear-attention`（个人 fork/早期地址）。二者代码内容相同（sustcsonglin 为 fla 主要维护者），但项目内 codebases/ 卡的规范 URL 为 fla-org 版本。

**问题：** 不影响技术正确性，但与项目内来源库记录的规范 URL 不一致。

**建议修改：** 统一使用 `github.com/fla-org/flash-linear-attention`，或注明两个地址指向同一仓库。

---

## 已核查且确认无误的要点

以下为本次审校中逐项核对到来源的声明（覆盖所有数字密集段落、代码引用和文献引用）：

| 报告声明 | 核查来源 | 结果 |
|----------|----------|------|
| VideoPose3D: GT 37.8 mm (Table 5), 检测 46.8 mm (Table 1a), 16.95M/33.87M FLOPs, 150k FPS (GP100), 前 SOTA 52.8→46.8 (6 mm/11%), 27f 40.6/17.09M, 81f 38.7/25.48M, 半监督 1% S1 降 22.6 mm, 全监督差距 9 mm | `papers/...temporal_convolutions...md` Table 5, Table 1a, abstract, Fig 5c | 全部精确匹配 |
| TCPFormer: 37.9/31.7 mm, N=16/H=8/C=128/L=T/3, 2×4090, 3DHP T=81 15.0 mm, 243→351 仅降 0.2–0.3 mm | `cards/tcpformer_*.json` | 数字全部匹配（字段归属见 P2） |
| HDFormer: 243帧性能略降, 3.7M (MixSTE 1/10), 96f GT 21.6 mm, 6×速度, SPD=4 | `cards/hdformer_*.json` | 完全一致 |
| Attention-TCN: n=1029 深层不优, n=243/L=5, GT 34.7, CPN 45.1, 3000 FPS (0.3 ms/帧) | `cards/attention_mechanism_exploits_temporal_contexts*.json` | 完全一致 |
| Simple Baseline: 1024+BN+Dropout 0.5+ReLU, 残差×2, 单帧 3 ms (batch 64), ~300 fps, 1518 引用 | `cards/a_simple_yet_effective_baseline*.json` | 完全一致 |
| MixSTE: CPN 40.9/32.6, GT 21.6, 33.7M (dl=8, dm=512, T=243), dl=10→42.2M, SitD>50 mm | `cards/mixste_*.json` | 完全一致 |
| MotionBERT: DSTformer N=5, h=8, dim=512, T=243; Table 7 自适应 39.25/S-T 40.58/均值 39.87; 371 引用 | `cards/motionbert_*.json` + `papers/motionbert_*.md` Table 7 | 完全一致 |
| MotionBERT Table 5 基线 39.2 mm | `papers/motionbert_*.md` Table 5 | 一致（37.4 描述见 P2） |
| DSTformer 代码行号: :269-358, :340-351, :239-267, :188-200, :307-311 (bias=0.5), :336, :360-361 | `codebases/MotionBERT.md` | 15 项全部精确匹配 |
| learning.py:39-67 (load_pretrained_weights), :79-101 (load_backbone); train.py:257 DataParallel | `codebases/MotionBERT.md` | 完全一致 |
| MB_pretrain.yaml:21 depth=5; 162MB 权重; batch_size=32 (ft); lr_decay=0.99 | `codebases/MotionBERT.md` | 完全一致 |
| DiffPose: GMM M=5, K=50→DDIM 5步, N=5 样本, 检测 36.9 (前 SOTA 40.9), GT 18.9 (前 SOTA 21.6), 184 引用 | `cards/diffpose_*.json` | 完全一致 |
| AugLift: 6D (x,y,c,d,dmin,dmax), OOD 降 10.1%/ID 降 4.0%, 3DHP 62.4/3DPW 92.6 (14.5%/22.2%), Fit3D 15.7%, dref 3.5–6.0 m | `cards/auglift_*.json` | 完全一致 |
| PoseMamba-L: P1=38.1/15.6, +1.1/2.2 mm vs MotionBERT, 16% 计算量, 25 引用 | `cards/posemamba_*.json` | 完全一致 |
| BSTMamba: CPN T=81 41.7, GT T=81 22.5, 9.85M/13.57G, 5组关节硬编码, 2025/0 引用 | `cards/a_spatiotemporal_bidirectional_mamba*.json` | 完全一致 |
| SBAHGNet: 37.24/31.57, GT 12.38, 18.3M/88.9G, +376 参数, 13×17/6×17 超边, 2026/0 引用 | `cards/sbahgnet_*.json` | 完全一致 |
| ConvFormer: 核 (7,7,7), 2.56M–10.24M, 减 65.5%–83.4%, 差 0.2 mm 未达 SOTA | `cards/convformer_*.json` | 完全一致 |
| Fusionformer: 9 帧, 48.7 mm, STE 0.5/CTE 0.2 mm | `cards/fusionformer_*.json` | 完全一致 |
| MixTGFormer: P1=37.6, P2=15.7, 2026/6 引用 | `cards/dual_stream_spatio_temporal_gcn_transformer*.json` | 完全一致 |
| DDHPose: 39.0/31.2, +0.4 mm (1.3%) vs D3DP, 7 层 HSTDenoiser | `cards/disentangled_diffusion*.json` | 完全一致 |
| PoseFormer: 检测 44.3, GT 31.3, 243×17=4131 token, 692 引用 | `cards/3d_human_pose_estimation_with_spatial_and_temporal_transformers.json` | 完全一致 |
| Flowing ConvNets: 50→5 fps, 7 上半身关节, 583 引用 | `cards/flowing_convnets*.json` | 完全一致 |
| 3D=2D+Matching: 条件独立假设, ~20 万姿态库, 593 引用 | `cards/3d_human_pose_estimation_2d_pose_estimation_matching.json` | 完全一致 |
| Strided Transformer: "视频中相邻帧姿态高度冗余，可通过步幅卷积逐步合并而不丢失关键信息", W4225557002, 2022, 302 引用 | `cards/exploiting_temporal_contexts_with_strided_transformer*.json` | 逐字匹配 |
| MixSTE 代码: run.py:423 loss_diff 0.5/2.0; run.py:472-494 翻转平均; data_3d_h36m.npz / data_2d_h36m_cpn_ft_h36m_dbb.npz | `codebases/MixSTE.md` | 行号和内容完全一致 |
| MotionBERT 卡片 limitation: "方法性能上限受限于上游2D姿态估计器的精度与遮挡处理能力" | `cards/motionbert_*.json` limitation 字段 Grep 确认 | 匹配 |
| H3.6M: 360 万帧/11 受试者/15 类动作 | `cards/posemamba_*.json` + `cards/hdformer_*.json` eval_setup | 匹配 |
| H3.6M: 4 视角/50 fps | `papers/hdformer_*.md` 原文 "4 cameras, 50 Hz" | 匹配（报告归因含 HDFormer 论文原文，有效） |
| GDN 双门控递推公式 $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ | `codebases/flash-linear-attention.md` §8 naive 参考实现代码等价 | 公式与代码语义一致 |
| Titans/GDN/GLA/µP 跨领域参照 | Glob cards/ 确认无对应条目；报告明确标注"cards/中无对应条目，具体数字待验证" | 标注合规 |
| 附录文献索引：所有 OpenAlex ID、年份、引用数 | 逐一与 cards/*.json paper_id/year/citation_count 比对 | 全部匹配 |
| 内部一致性：§3.6 总计 72h = §4.4 总计 72h; $36 = 72×$0.5; 33 天 ≈ 5 周; 表格与正文无矛盾 | 交叉比对 | 一致 |
| ideas/惊讶度门控测试时记忆分离时序平滑与消歧.md 与报告一致性 | Read ideas/ | 方法设计（含 GDN 双门控升级）、三分解、matched-FLOPs 一致 |

---

## 装饰性论证检查

§1.3"根本性分析"含信息论/几何/优化三视角 + 三分解公式。逐段评估：

- **信息论视角**（条件互信息）：直接支撑惊讶度计算设计（仅写入 $I(Y;X_T|X_{1:T-1})$ 显著非零的帧），并以 TCPFormer 243→351 经验数据为佐证。删除后设计动机链断裂。**非装饰性。**
- **几何视角**（深度歧义流形）：引用 DiffPose 多模态分布证据，支撑 DFR 消歧指标的必要性。**非装饰性。**
- **优化视角**（容量-长度失配）：引用 HDFormer/Attention-TCN 经验证据，支撑 matched-FLOPs 对比框架的必要性。**非装饰性。**
- **三分解公式**：C2 贡献的核心形式化定义，不可删除。

**结论：** 未发现删掉后方案设计依据不受损的纯装饰段落。

---

## 统计与判定

| 等级 | 数量 |
|------|------|
| P0（编造/来源不符） | **0** |
| P1（无来源未标待验证/结论夸大） | **0** |
| P2（表述或一致性小问题） | **4** |

**总体判定：可发布。**

**依据：** 全报告 50+ 个可验证数字（指标/参数量/FLOPs/引用数/代码行号）经逐项核对，无一编造，全部与 cards/papers/codebases/ 来源一致或已标注"待验证"。4 条 P2 均为字段归属、引述精确度或 URL 规范等表述层面小问题，不影响任何数字的正确性，也不影响方案设计依据和可行性结论。代码行号级引用（15 项）全部精确匹配。
