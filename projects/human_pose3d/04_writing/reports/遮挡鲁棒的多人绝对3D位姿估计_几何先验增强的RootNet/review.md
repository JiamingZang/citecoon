# 审校报告: 遮挡鲁棒的多人绝对3D位姿估计_几何先验增强的RootNet
> 对抗性审校 · 2026-07-28 18:51 · 对象: 遮挡鲁棒的多人绝对3D位姿估计_几何先验增强的RootNet/report.md · model: qmodel_preview


审校员角色：对抗性审校（与作者利益对立）
核查来源：cards/*.json、papers/*.md、codebases/*.md、ideas/*.md
审校日期：2026-07-28

---

## 问题清单

### [P1] §1.1 + §1.2表格 + §3.1 — MRPE 289/178 mm 的数据集归属错误（MuPoTS-3D→实为H36M）

> 报告原文摘录：
> - §1.1："Root-GAST-Net（2022）**在MuPoTS-3D上的实验表明**，RootNet原版的MRPE达289 mm，而其改进版仍为178 mm"
> - §1.2表格行："3DMPPE (Moon) | 2019 | **MuPoTS-3D关键指标** | MRPE 289 mm"
> - §1.2表格行："Root-GAST-Net | 2022 | **MuPoTS-3D关键指标** | MRPE 178 mm, 3DPCKabs 56.8%"
> - §3.1指标表："MRPE (MuPoTS-3D) | 平均根位置误差 | 289 mm (RootNet) / 178 mm (Root-GAST-Net)"

**核查过程：**

Read `cards/top_down_system_for_multi_person_3d_absolute_pose_estimation_from_monocular_vide.json`（Root-GAST-Net卡），eval_setup字段原文为：

> "关键结果：**MuPoTS-3D上**3D-PCKabs 56.8%（超TDBU_Net 8.8 pp）、AP_root_25 58.9%（超12.6 pp）、MPJPE 101.9 mm；**Human3.6M上**MRPE 178 mm vs RootNet 289 mm；速度15 fps（GTX 1080）。"

卡明确以分号分隔两个数据集的结果。MRPE 178/289 mm 归属于 **Human3.6M**，而非 MuPoTS-3D。MuPoTS-3D上的指标为 3D-PCKabs、AP_root_25、MPJPE（101.9 mm），不含 MRPE。

进一步验证：Read `papers/camera_distance_aware_top_down_approach_for_3d_multi_person_pose_estimation_from.md`（3DMPPE原文），Table 2 区域（行488-512）显示 3DMPPE 在 MuPoTS-3D 上的 MRPE 约为 120.0 mm（Disjointed learning），远非 289 mm。289 mm 是 Root-GAST-Net 论文在 H36M 上复测 RootNet 的结果。

**影响范围：** 此错误贯穿报告多处（§1.1正文、§1.2表格列标题、§3.1指标表、§3.3基线表隐含引用），构成系统性数据集归属错误。报告的核心论证"根深度误差在绝对位姿总误差中占比超过50%"在 H36M 上可能成立（MRPE 289 vs MPJPE ~46），但在 MuPoTS-3D 上原框架 MRPE 仅~120 mm，论证力度不同。

**建议修改：** 将所有"MRPE 289/178 mm (MuPoTS-3D)"修正为"MRPE 289/178 mm (Human3.6M)"；§1.2表格列标题拆分为 H36M 与 MuPoTS-3D 两列；§3.1指标表中 MRPE 行注明数据集为 H36M，MuPoTS-3D 行改用 3DPCKabs/AP_root_25/MPJPE。

---

### [P2] §2 Contribution 1可回退设计 — "3DPCKabs约44%，camera_distance_aware卡"来源不精确

> 报告原文摘录：
> "性能下界即为3DMPPE原版精度（MRPE 289 mm / 3DPCKabs约44%，camera_distance_aware卡）"

**核查过程：**

Read `cards/camera_distance_aware_top_down_approach_for_3d_multi_person_pose_estimation_from.json`，全文搜索"44"、"3DPCKabs"：卡的 eval_setup 字段仅列出指标名称（"指标：MPJPE、PA-MPJPE、MRPE（新提出）、3DPCKrel、AUCrel、3DPCKabs、AP_root_25"），未给出任何具体数值。

Read `papers/camera_distance_aware_top_down_approach_for_3d_multi_person_pose_estimation_from.md`（行500-512），Table 2 区域显示 3DPCKabs 列下 R-50 对应值为 **43.8**，与"约44%"吻合。

**结论：** 数字本身正确（43.8≈44%），但出处应为论文原文 Table 2，而非"camera_distance_aware卡"（卡中无此数字）。

**建议修改：** 将来源标注改为"camera_distance_aware论文 Table 2"或直接删除来源标注（因数字可从论文验证）。

---

### [P2] §1.2路线二 — MuPoTS-3D场景描述前后不一致（"户外"vs"室内外"）

> 报告原文摘录：
> - §3.4："MuPoTS-3D：20个**户外**真实场景（测试），最多3人/场景，含室内外混合"

**核查过程：**

- `cards/camera_distance_aware_top_down_approach_for_3d_multi_person_pose_estimation_from.json` eval_setup："MuPoTS-3D（20个**户外**真实场景/最多3人测试）"
- `cards/top_down_system_for_multi_person_3d_absolute_pose_estimation_from_monocular_vide.json` eval_setup："MuPoTS-3D（20段**室内外**多人视频）"

报告§3.4在同一句中写"20个户外真实场景…含室内外混合"，自相矛盾。两张卡对 MuPoTS-3D 的描述本身不一致（3DMPPE卡说"户外"，Root-GAST-Net卡说"室内外"），报告未加辨别地混用了两种说法。

**建议修改：** 统一为"20段室内外多人视频"（Root-GAST-Net卡描述更完整），删除自相矛盾的表述。

---

## 抽查通过项（通查后无问题）

以下为核心抽查点及对应来源，均核对通过：

| 报告声明 | 核对来源 | 结果 |
|---------|---------|------|
| PoseMamba-L: 38.1 mm (CPN) / 15.6 mm (GT 2D), T=243 | `cards/posemamba_...json` eval_setup | 完全吻合 |
| PoseMamba较MotionBERT精度提升1.1 mm且仅用16%计算量 | 同上 | 吻合（1.1/2.2 mm, 16%） |
| PoseMamba训练：单张RTX 3090, 120 epochs, batch 4, AdamW lr=2e-4 | 同上 resources字段 | 完全吻合 |
| MixSTE: 40.9 mm / P-MPJPE 32.6 mm, ~33.7M参数, 645M FLOPs/帧 | `cards/mixste_...json` | 完全吻合 |
| VideoPose3D: 16.95M参数, 33.87M FLOPs, ~150k FPS/GP100 | `cards/3d_human_pose_estimation_in_video_...json` | 完全吻合 |
| VideoPose3D: GT 2D可再降22.6 mm | 同上 | 吻合 |
| VideoPose3D 46.8 mm标注"待验证" | 同上（卡未给CPN MPJPE绝对值） | 标注合规 |
| OAHPE: 43.5 mm, 2.6M参数, 12.5 GB显存, τ=η=0.5 | `cards/oahpe_...json` | 完全吻合 |
| VNect自遮挡PCK约48%（坐/躺） | `cards/vnect.json` | 吻合 |
| SMAP: ~57 ms/帧, RtError 23.3→67 cm（无内参） | `cards/smap_...json` | 完全吻合 |
| HMOR: PCKabs +12.3, CMU Panoptic 20.5 mm | `cards/hmor_...json` | 完全吻合 |
| 3DMPPE推理~0.141 s/帧 (TitanX Maxwell) | `cards/camera_distance_aware_...json` | 完全吻合 |
| HMR: SMPL 85维(23轴角+10PCA+相机), 25个判别器 | `cards/end_to_end_recovery_...json` | 吻合（K+2=25） |
| Ordinal Depth: 标注效率~1 min/图 (17问×3.5s) | `cards/ordinal_depth_...json` | 完全吻合 |
| BASED: Ω(N)比特下界, SWDE 48.06 vs 71.92, FDA 24.41 vs 73.23 | `cards/simple_linear_attention_...json` | 完全吻合 |
| GDN: 1.3B参数, 100B tokens, 公式 S_t = α_t S_{t-1}(I-β_t k_t k_t^T)+β_t v_t k_t^T | `cards/gated_delta_networks_...json` | 完全吻合 |
| Titans: 惊讶度=关联记忆损失梯度, 三超头架构 | `cards/titans_...json` | 吻合 |
| fla chunk_size∈{16,32,64} | `codebases/flash-linear-attention.md` chunk.py:535-536 | 完全吻合 |
| PoseMamba损失权重 λ_3d=1.0, λ_scale=0.5, λ_vel=20.0, λ_diff=0.5 | `codebases/PoseMamba.md` 行197 | 完全吻合 |
| HybrIK "Naive HybrIK隐含假设预测骨骼长度等于模板骨骼长度" | `papers/hybrik_...md` 行382-385 | 原文吻合 |
| 分离训练理由"两任务相关性不高" | `cards/camera_distance_aware_...json` method字段 | 原文吻合 |

**代码事实（file:line）全量核对：**

| 报告引用 | repo卡对应 | 结果 |
|---------|-----------|------|
| `data/Human36M/Human36M.py:122`: area=bbox[2]*bbox[3] | 3DMPPE卡 Fact 1 | 吻合 |
| `data/dataset.py:73`: k_value公式 | 3DMPPE卡 Fact 1 | 吻合 |
| `main/config.py:38`: bbox_real=(2000,2000) | 3DMPPE卡 硬编码参数表 | 吻合 |
| `main/model.py:67-72`: GAP→Conv1×1→γ*k | 3DMPPE卡 Fact 2 | 吻合 |
| `main/model.py:22-28`: depth_layer Conv2d(2048→1) | 3DMPPE卡 Fact 2 | 吻合 |
| `main/model.py:106-113`: L1 loss | 3DMPPE卡 Fact 4 | 吻合 |
| `data/Human36M/Human36M.py:155-160`: pixel2cam | 3DMPPE卡 Fact 5 | 吻合 |
| `common/utils/pose_utils.py:13-18`: pixel2cam实现 | 3DMPPE卡 Fact 5 | 吻合 |
| `train.py:49`: target dict | 3DMPPE卡 改造点 | 吻合 |
| `datareader_h36m.py:36`: 2D归一化 | PoseMamba卡 (1) | 吻合 |
| `mambablocks.py:676-680`: BiSTSSMBlock._forward | PoseMamba卡 (2) | 吻合 |
| `csms6s.py:170-192`: CrossMerge_plus_poselimbs | PoseMamba卡 (2) | 吻合 |
| `csms6s.py:186`: ys[:,0:2]+ys[:,2:4].flip | PoseMamba卡 (2) | 吻合 |
| `PoseMamba.py:37-141`: 主干无需修改 | PoseMamba卡 改造点 | 吻合 |
| `mambablocks.py:582`: BiSTSSM类 | PoseMamba卡 (2) | 吻合 |
| `train.py:199-229`: 损失计算 | PoseMamba卡 (4) | 吻合 |
| `loss.py:56-63`: L2 MPJPE | PoseMamba卡 (4) | 吻合 |
| `loss.py:71`: 输入形状(N,T,17,3) | PoseMamba卡 (4) | 吻合 |
| `train.py:181`: root-relative | PoseMamba卡 (1) | 吻合 |
| `train.py:79`: predicted_3d_pos[:,:,0,:]=0 | PoseMamba卡 (1) | 吻合 |
| `train.py:121-123`: 排除序列 | PoseMamba卡 (5) | 吻合 |
| `train.py:71-75`: 翻转增强 | PoseMamba卡 (5) | 吻合 |
| `datareader_h36m.py:31-33`: 分辨率1000×1002 | PoseMamba卡 (5) | 吻合 |
| `datareader_h36m.py:100-107`: stride=81/243 | PoseMamba卡 (5) | 吻合 |
| `fla/ops/gated_delta_rule/chunk.py:397-588`: API | fla卡 Fact 4 | 吻合 |
| `fla/ops/gated_delta_rule/naive.py:50-59`: 递推 | fla卡 Fact 8 | 吻合 |
| `fla/ops/gated_delta_rule/gate.py:96-104`: 门控衰减 | fla卡 Fact 6 | 吻合 |
| `fla/ops/common/chunk_o.py:125-126`: 因果mask | fla卡 Fact 7 | 吻合 |
| `fla/ops/gated_delta_rule/chunk.py:535-536`: chunk约束 | fla卡 Fact 10 | 吻合 |

全部29处 file:line 引用与 repo 卡一致，无编造。

**撞车查新判定：**

对照系统提供的11篇arXiv近期工作：
- 5篇"gated delta rule"命中（2607.20062, 2605.22791, 2512.19331, 2512.10252, 2504.14366）：均为NLP/线性注意力/医学影像/视频分割领域，无一涉及3D姿态估计或根深度修正，方法实质不同。
- 5篇"absolute root depth"命中（2607.11928, 2606.25619, 2606.03608, 2605.10675, 2604.19871）：分别为条纹投影、手部姿态、测试时RL、神经形态深度估计、量子纠错，与多人绝对位姿估计无关。
- 1篇"ordinal depth supervision"（1805.04095, 2018）：即报告已引用的Pavlakos原文，非撞车。

**结论：无撞车。** 报告核心方法主张（解耦观测质量门+状态惊讶门用于GDN时序lifting处理遮挡）在给定素材中无实质覆盖。报告未使用"首个/唯一"类表述，idea评审记录亦确认查重0命中。

**装饰性论证检查：**

§1.3"根本性分析"三步递推（面积退化→单帧无法修正→检测器恶性循环）直接支撑C2（面积修正）和C3（门控记忆保护）的设计动机，删去后方案设计依据受损，非装饰性段落。§1.2路线三（NLP理论）中BASED的Ω(N)下界和Titans惊讶度概念分别为门控必要性和状态惊讶门提供理论依据，非纯装饰。

---

## 统计与判定

| 等级 | 数量 |
|------|------|
| P0（编造/来源不符） | 0 |
| P1（无来源未标待验证/结论夸大） | 1 |
| P2（表述或一致性小问题） | 2 |

**总体判定：需修订后发布。**

一句话依据：核心数字与代码事实全部可溯源（29处file:line零偏差，数十项指标均与cards/papers吻合），但MRPE 289/178 mm被系统性错误归属于MuPoTS-3D（实为H36M），影响实验计划表的目标设定与论证逻辑，须修正后方可作为可信文档使用。
