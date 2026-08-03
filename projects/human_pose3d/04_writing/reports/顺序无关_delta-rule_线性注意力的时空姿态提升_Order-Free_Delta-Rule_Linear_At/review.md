# 审校报告: 顺序无关_delta-rule_线性注意力的时空姿态提升_Order-Free_Delta-Rule_Linear_At
> 对抗性审校 · 2026-07-21 16:49 · 对象: 顺序无关_delta-rule_线性注意力的时空姿态提升_Order-Free_Delta-Rule_Linear_At.md · model: qwen3.8-max-preview


## 问题清单

### [P1] §4.4 性能/成本量化表 — PoseMamba-B 训练时间"~8h"来源不符

> 报告原文：
> | 训练时间 (120 epochs, T=81) | ~8h (3090, PoseMamba 卡) | ~8–10h | 略增（delta-rule 循环不可并行化于 $J$ 维，但 $J=17$ 极短） |

**核查过程：** Read `cards/posemamba_monocular_3d_human_pose_estimation_with_bidirectional_global_local_spa.json`，其 `resources` 字段明确写道："训练硬件为单张NVIDIA RTX 3090 GPU；框架PyTorch；训练120 epochs，batch size 4，AdamW优化器，初始学习率2e-4，指数衰减因子0.99，权重衰减0.01；2D检测使用Stacked Hourglass；**未报告具体训练时长与显存峰值**。" Grep PoseMamba 论文原文（`papers/posemamba_*.md`）亦未找到任何训练时长数字。

**问题：** 报告将"~8h"标注为来自"PoseMamba 卡"，但该卡明确声明"未报告具体训练时长"。此数字在来源库中无出处，且未标注为经验估计。§3.6 中同一数字写为"~8–12h / run | 参照 PoseMamba: batch=4, 120 epochs（PoseMamba 卡）"——此处"参照"仅能推出配置参数，不能推出时长。

**建议修改：** 将"~8h (3090, PoseMamba 卡)"改为"~8–12h（经验估计，基于 PoseMamba 卡训练配置推算；原卡未报告训练时长）"，并在 §4.4 表中同步修正来源标注。

---

### [P2] §1.3 命题 2 — BASED Theorem 3.4 结论对象表述不精确

> 报告原文：
> **Theorem 3.4**：在 $\log c \le d \le 2(\log N)^{1-\epsilon}$ 编码下，数据无关 BaseConv 需要 $\Omega(\epsilon \log\log N)$ 层。

**核查过程：** Read `papers/simple_linear_attention_language_models_balance_the_recall_throughput_tradeoff.md`，第 652–660 行原文为："Theorem 3.4. Given an input u ∈{0, 1}^{N×d} to the **MQAR** with any encoding such that log c ≤ d ≤ 2(log N)^{1-ε} ... a data-independent BaseConv model with model parameters taking O(log N) bits needs Ω(ε log log N) layers to solve **AR**."

**问题：** 定理前提是"给定 MQAR 的输入编码"，但结论是"needs Ω(ε log log N) layers to solve **AR**"（ Associative Recall），而非"求解 MQAR"。报告附录来源核对表写"BASED Theorem 3.2/3.4：BaseConv 需 log(2d)/Ω(ε log log N) 层"，未区分 AR 与 MQAR，但正文 §1.3 的表述"求解 MQAR"与原文结论"solve AR"不完全一致。原文第 666–668 行后续讨论才将二者合并："Theorem 3.2 and Theorem 3.4 imply that we need Ω(max(log log c, log log N)) many BaseConv layers to solve MQAR."

**建议修改：** 将 Theorem 3.4 的结论改为"需要 Ω(ε log log N) 层才能求解 AR（关联回忆）"，或补充说明"结合 Theorem 3.2 可推出求解 MQAR 的层数下界"。

---

### [P2] §3.6 显存估算 — 记忆矩阵额外开销计算有误（4 KB 应为 16 KB）

> 报告原文：
> 本方法空间维 delta-rule 记忆矩阵额外开销为 $H \times (d_k/H) \times (d_v/H) = d_k d_v / H$ per sample，在 $d_k=d_v=64, H=4$ 下仅增加约 4 KB/sample，可忽略。

**核查过程：** 代数验算。$H$ 个头各维护 $(d_k/H) \times (d_v/H)$ 的记忆矩阵，总元素数 = $H \times (d_k/H) \times (d_v/H) = d_k \cdot d_v / H$。代入 $d_k=d_v=64, H=4$：$64 \times 64 / 4 = 1024$ 个 float32 = 4 KB。但报告公式本身写的是 "$H \times (d_k/H) \times (d_v/H) = d_k d_v / H$"，代数上 $H \times (d_k/H) \times (d_v/H) = d_k \cdot d_v / H$ 确实成立（= 1024 元素）。然而物理上，每个头的记忆矩阵为 $(d_k/H) \times (d_v/H) = 16 \times 16 = 256$ 元素，$H=4$ 个头共 $4 \times 256 = 1024$ 元素 = 4 KB。**但这是单层单帧的开销**；报告写"per sample"未明确是否含 20 层。若含 20 层则应为 $20 \times 4\text{ KB} = 80\text{ KB}$。此外，若 $d_k = d_v = 64$ 是总维度（非每头维度），则每头为 $16 \times 16$，总计 1024 元素 = 4 KB/layer/frame，数值本身正确。

**问题（修正）：** 经复核，若 $d_k=d_v=64$ 为总维度，则 $H \times (64/4)^2 = 4 \times 256 = 1024$ 元素 = 4 KB/layer/frame 的计算是正确的。但报告写"per sample"含义模糊——未说明是每层还是全模型。若为全模型（20 层），应为 ~80 KB。结论"可忽略"不变，但表述不够精确。

**建议修改：** 明确写为"每层每帧约 4 KB（20 层共约 80 KB），相对 18–20 GB 总显存可忽略"。

---

## 抽查覆盖记录（通查后无 P0 的核对点）

以下各项均逐一核对到来源，未发现不符：

| 核查点 | 来源 | 结果 |
|--------|------|------|
| PoseMamba-L: 6.714M / 27.9G / 38.1 / 15.6 mm / 16% MotionBERT | `papers/posemamba_*.md:365-367, 504-506`; card eval_setup | 完全一致 |
| PoseMamba-B: 3.358M / 13.9G / P1=40.8 | `papers/posemamba_*.md:378-382` Table 6 | 完全一致 |
| PoseMamba-B GT 2D T=81: 14.51 mm | `papers/posemamba_*.md:221-223` Table 3 | 数值一致（原文表头未显式标 GT/估计，但数值范围与 GT 2D 一致） |
| PoseMamba-S: 0.860M | `papers/posemamba_*.md:328-332` Table 6; Table 4 各行 | 完全一致 |
| 消融 Table 5: 43.7/36.5 → 43.5/36.2 → 42.1/35.1 → 41.8/35.0 | `papers/posemamba_*.md:297-304` | 完全一致 |
| Table 4: Bidirectional 42.4 → Global-Local 41.8 (0.6 mm) | `papers/posemamba_*.md:271-280` | 完全一致 |
| 训练超参: AdamW / lr=2e-4 / 衰减0.99 / wd=0.01 / bs=4 / 120ep | `papers/posemamba_*.md:486-491`; card resources | 完全一致 |
| H3.6M: 360万帧 / 11受试者 / 15动作 | `papers/posemamba_*.md:454-455`; card eval_setup | 完全一致 |
| 3DHP: 130万帧 | `papers/posemamba_*.md:468`; card eval_setup | 完全一致 |
| `assert W==17` at csms6s.py:153 | `codebases/PoseMamba.md` 对应段落 | 完全一致 |
| `indices=[0,0,1,2,3,0,4,5,6,8,11,12,13,8,14,15,16]` at :156/:187 | `codebases/PoseMamba.md` | 完全一致 |
| `xs[:,0]=(x+x[...,indices]).flatten(2,3)` at :157 | `codebases/PoseMamba.md` | 完全一致 |
| 4 扫描方向 :158-160; CrossMerge :170-192 | `codebases/PoseMamba.md` | 完全一致 |
| PoseMamba.py:66-74 STEblocks; :133-140 主前向 | `codebases/PoseMamba.md` 方案 B 及前向代码 | 完全一致 |
| mambablocks.py:619-685 BiSTSSMBlock._forward; :312-341 FORWARD_TYPES; :555-578 forwardv2 | `codebases/PoseMamba.md` | 完全一致 |
| 方案 B 接口 (B,F,N,C)→(B,F,N,C) | `codebases/PoseMamba.md` 改造接口点 | 完全一致 |
| 风险 #1 时间块语义不明 | `codebases/PoseMamba.md` 风险段 | 完全一致 |
| PyTorch 1.13.1+cu117 | `codebases/PoseMamba.md` 环境段 | 完全一致 |
| 数据路径 MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl | `codebases/PoseMamba.md` | 完全一致 |
| fla naive.py:50-59 递推语义 | `codebases/flash-linear-attention.md` | 完全一致 |
| chunk.py:535-536 chunk_size ∈ {16,32,64} | `codebases/flash-linear-attention.md` | 完全一致 |
| chunk_o.py:125-126 因果 mask | `codebases/flash-linear-attention.md` | 完全一致 |
| gla/chunk.py:363, :401 intra-chunk mask | `codebases/flash-linear-attention.md` | 完全一致 |
| chunk.py:397-588 chunk_gated_delta_rule | `codebases/flash-linear-attention.md` | 完全一致 |
| chunk_fwd.py:40-69 solve_tril | `codebases/flash-linear-attention.md` | 完全一致 |
| gate.py:96-104 gdn_gate_chunk_cumsum | `codebases/flash-linear-attention.md` | 完全一致 |
| 风险 #1 双向仓库未验证; 风险 #2 非因果封闭解 | `codebases/flash-linear-attention.md` | 完全一致 |
| §9 flash-bidirectional-linear-attention | `codebases/flash-linear-attention.md` | 完全一致 |
| MixSTE model_cross.py:468-483 STE_forward; :485-496 TTE_forward | `codebases/MixSTE.md` | 完全一致 |
| model_cross.py:474 Spatial_pos_embed | `codebases/MixSTE.md` STE_forward 代码块内 | 一致（在 468-483 范围内） |
| run.py:401 WMPJPE 权重 [1,1,2.5,2.5,...,4,4] | `codebases/MixSTE.md` | 完全一致 |
| run.py:423 时间一致性损失 | `codebases/MixSTE.md` | 完全一致 |
| GLA Eq.3 / Table 1 / α 参数化 / §4.4 多头 / 输出门控 / O(LCd+Ld²) | `papers/gated_linear_attention_*.md` | 完全一致 |
| GLA arxiv:2312.06635 / ICML 2024 | `papers/gated_linear_attention_*.md` 头部 | 完全一致 |
| DeltaNet §2.2 公式 / 在线回归 / key collision L>d / WY | `papers/parallelizing_linear_transformers_*.md` | 完全一致 |
| DeltaNet arxiv:2406.06484 / NeurIPS 2024 | `papers/parallelizing_linear_transformers_*.md` 头部 | 完全一致 |
| Gated DeltaNet Table 2 公式 / §3.4 L2 norm / memory collisions / Smolensky 1990 | `papers/gated_delta_networks_*.md` | 完全一致 |
| Gated DeltaNet arxiv:2412.06464 / ICLR 2025 | `papers/gated_delta_networks_*.md` 头部 | 完全一致 |
| Gated DeltaNet 1.3B params / 100B tokens / gradient clipping 1.0 | `papers/gated_delta_networks_*.md:515-517` | 完全一致 |
| BASED Theorem 3.1 Ω(N)-bit | `papers/simple_linear_attention_*.md:618-619` | 完全一致 |
| BASED Theorem 3.2 log(2d) layers | `papers/simple_linear_attention_*.md:634-636` | 完全一致 |
| BASED arxiv:2402.18668 / ICML 2024 Workshop | `papers/simple_linear_attention_*.md` 头部 | 完全一致 |
| Arora et al. 2023a attention constant layers | `papers/simple_linear_attention_*.md:661-662` | 完全一致 |
| VideoPose3D: 16.95M / 33.87M FLOPs / 150k FPS / GP100 | `cards/3d_human_pose_estimation_in_video_*.json` | 完全一致 |
| PoseFormer: W3136525061 / 44.3 / 31.3 mm | `cards/3d_human_pose_estimation_with_spatial_*.json` | 完全一致 |
| MixSTE: W4312417903 / 33.7M / dl=8 / dm=512 / 40.9 / 32.6 / 21.6 / 31% | `cards/mixste_*.json` | 完全一致 |
| ConvFormer: W4382892987 / 2.56–10.24M / 65.5%–83.4% / 核(7,7,7) | `cards/convformer_*.json` | 完全一致 |
| HDFormer: W4385767582 / 3.7M / 21.6mm / T=96 / SPD≤4 / 243帧略降 | `cards/hdformer_*.json` | 完全一致 |
| BSTMamba: W4413980847 / 9.85M / 13.57G / 41.7 / 22.5 / 5区域 / 硬编码局限 | `cards/a_spatiotemporal_*.json` | 完全一致 |
| TCPFormer: 243→351帧仅降0.2-0.3mm | `cards/tcpformer_*.json` problem 字段 | 完全一致 |
| _themes.json 全部 6 条母题引用 | `cards/_themes.json` | 逐条核对，完全一致 |
| PoseMamba 卡局限引文（启发式/理论最优性/SSM扫描顺序） | `cards/posemamba_*.json` limitation 字段 | 完全一致（为卡作者读出局限，非论文原文自述，报告引用时标注为"PoseMamba 卡局限"，归属正确） |
| BSTMamba 卡局限引文（5组关节硬编码/迁移需重定义） | `cards/a_spatiotemporal_*.json` limitation 字段 | 完全一致 |
| idea 评审记录: 查重"无" / 3/3 票支持 | `ideas/顺序无关delta-rule线性注意力的时空姿态提升.md` 评审记录段 | 完全一致 |

---

## 统计与判定

| 等级 | 数量 |
|------|------|
| P0（编造/来源不符） | 0 |
| P1（无来源未标待验证） | 1 |
| P2（表述或一致性小问题） | 2 |

**总体判定：需修订后发布。**

**依据：** 报告整体来源核对质量极高——数十项数字、公式、代码行号、定理引用均与 cards/papers/codebases 完全吻合，无编造或歪曲；唯一 P1 为训练时长数字的来源标注错误（卡明确写"未报告"但报告标为卡来源），修正标注即可；两条 P2 为定理结论措辞和显存估算精度问题，不影响核心论证。修订上述 3 处后可发布。
