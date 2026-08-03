# 审校报告: 先验-精度_Pareto_前沿_Model-Free_位姿估计的统一横向基准与参考视角蒸馏

> 对抗性审校 · 2026-07-21（第四轮，全量核查）
> 核查范围：cards/*.json（54 篇）、cards/_themes.json（5 条母题）、papers/*.md、codebases/foundpose.md、codebases/dust3r.md、ideas/先验知识-精度Pareto前沿_...md

---

## 问题清单

### [P1] §2.1/§3.4/§3.5/§3.6 — BOP-LM-O 物体数错误：报告写 15，实际为 8；且错误归因于 BOP 卡片

> **报告原文（§2.1）：** "数据集：BOP-LM-O（Linemod-Occlusion，15 个物体，涵盖低纹理与对称场景）"
> **报告原文（§3.4）：** "物体数：15 个（BOP 卡片·eval_setup 记载全部 8 数据集共 89 个物体）"
> **报告原文（§3.5）：** "逐物体配对，LM-O 15 物体为 15 对"
> **报告原文（§3.6）：** "FoundPose 模板渲染（800 视角 × 15 物体）"

**核查过程：**
1. Read `cards/bop_benchmark_for_6d_object_pose_estimation.json` eval_setup 字段："数据集：8个（LM、LM-O、IC-MI、IC-BIN、T-LESS、RU-APC、TUD-L、TYO-L），89个物体，62K测试RGB-D图像"。BOP 卡片仅给出 8 数据集总计 89 物体，**未给出 LM-O 单独的物体数**，不能用作"15 个物体"的出处。
2. Read `ideas/先验知识-精度Pareto前沿_...md` 最小实验设计段："BOP-LM-O（15个物体，覆盖低纹理与对称场景）"。报告的"15"来源于 idea 卡，但 idea 卡本身有误。
3. 领域事实：BOP-LM-O（Occlusion-LINEMOD）包含 **8 个物体**（ape, can, cat, driller, duck, eggbox, glue, holepuncher），是 LINEMOD（LM）15 物体的子集。"15 个物体"是 LM 的物体数，非 LM-O。来源库中 `cards/onepose_...json` eval_setup 记载 "LINEMOD（13 物体）"、`cards/learning_descriptors_...json` 记载 "LineMOD数据集（15个物体）"，均指 LM 而非 LM-O。
4. 该错误向下游传播：§3.5 统计检验设计（"15 对"→应为 8 对，统计效力进一步降低）、§3.6 资源估算（"800 视角 × 15 物体"→应为 × 8）、§3.5 可视化计划（"15 物体 × 多方法"雷达图）。

**建议修改：** 将所有"LM-O 15 个物体"修正为"LM-O 8 个物体"；删除对"BOP 卡片·eval_setup"的错误归因（BOP 卡片仅给出 8 数据集总计 89 物体，无逐数据集分解）；相应修正统计检验（n=8 对）、资源估算和可视化计划。若需更大物体数做统计检验，应补充 TUD-L/YCB-V 等辅助数据集（报告 §3.4 已提及）。

---

### [P2] §1.1 先验层级表 — GS-Pose "~200 张带位姿参考图"无出处

> **报告原文：** "| 多视角参考图 + 已知位姿 | Gen6D（W4320013905）、GS-Pose（W4392971958） | ~200 张带位姿参考图 |"

**核查过程：**
1. Read `cards/gen6d_...json` eval_setup："使用~200张参考图像"。Gen6D 的 200 张有出处。
2. Read `cards/gs_pose_...json` method："用分割后的多视角参考图像重建3DGS模型Gobj"；limitation(1)："离线构建3DGS模型需要已知位姿的多视角参考图像"；eval_setup 仅记载训练数据（MegaPose 合成 1M 图像）和测试集（LINEMOD 13 物体、OnePose-LowTexture 8 物体），**未给出测试时所需参考图像的具体数量**。
3. 将 Gen6D 的"~200"直接套用到 GS-Pose 无来源支持。

**建议修改：** 将表格中 GS-Pose 的先验描述改为"多视角带位姿参考图（数量待验证）"，不应与 Gen6D 共享"~200"这一具体数字。

---

### [P2] §3.6 资源估算表 — FoundPose 分步耗时（~2h 渲染、~3h 推理）无来源且未标"待验证"

> **报告原文：**
> "| FoundPose 模板渲染（800 视角 × 15 物体） | 1× RTX 3090 | ~2h | ~5 GB |"
> "| FoundPose 特征提取 + 推理（全 LM-O） | 1× RTX 3090 | ~3h | ~2 GB |"

**核查过程：**
1. Read `codebases/foundpose.md` 全文：未报告任何推理耗时或渲染耗时数据。
2. Read `cards/foundpose_...json` resources 字段为空字符串，无耗时信息。
3. Grep `papers/foundpose_...md` 搜索 "time|speed|latency|hour|minute"：无推理速度相关数字。
4. 报告对 Gen6D（~4h）和 GS-Pose（~6h）标注了"（待验证）"，但 FoundPose 的 ~2h 和 ~3h 未标注，给读者以"已验证"的错觉。
5. 800 模板数有出处（FoundPose 论文 L288："We rendered 800 templates per object"），但耗时和存储量纯属估算。

**建议修改：** 为 FoundPose 渲染和推理行添加"（待验证）"标注，与 Gen6D/GS-Pose 行保持一致；或注明"估算值，无实测出处"。

---

### [P2] §4.4 逐组件推理耗时预算表 — MegaPose Refiner "典型 3-5 步"中"3 步"无出处

> **报告原文：** "| MegaPose Refiner（单步） | 66.5 ms/步 | MegaPose 卡片·limitation(4) | 多步迭代，典型 3-5 步 → ~200-330 ms |"

**核查过程：**
1. Read `cards/megapose_...json` limitation(4)："精炼器每步需多视角渲染（66.5ms/步），多次迭代成本高"。卡片未给出具体迭代步数。
2. Grep `papers/foundpose_...md` L286："5 iterations of the MegaPose refiner"。FoundPose 论文使用 5 步。
3. "5 步"有出处（FoundPose 论文），但"3 步"无任何来源。"典型 3-5 步"的表述暗示 3 步也是常见配置，但来源库中无此证据。

**建议修改：** 改为"典型 5 步（FoundPose 论文配置）→ ~330 ms"，或标注"3 步"为推测值。

---

### [P2] §1.3 — PoseGAM 对比对象声称为"MegaPose/GigaPose 等 CAD 方法"无卡片出处

> **报告原文：** "PoseGAM（2025）在 5 个 BOP 数据集上评测了多视图基础模型路线，但其对比对象为 MegaPose/GigaPose 等 CAD 方法，未纳入零训练方法。"

**核查过程：**
1. Read `cards/posegam_...json` eval_setup："在LM-O、T-LESS、YCB-V、TUD-L、IC-BIN五个真实基准上测试。采用BOP标准的Average Recall（AR）...相比先前方法平均AR提升5.1%，部分数据集最高提升17.6%"。
2. 卡片中**未列出具体对比方法名称**。"MegaPose/GigaPose 等 CAD 方法"是报告作者的推断，非卡片记载的事实。

**建议修改：** 改为"PoseGAM（2025）在 5 个 BOP 数据集上评测，但其卡片未列出具体对比方法，无法确认是否纳入零训练方法（待验证）"；或查阅 PoseGAM 论文原文补充出处。

---

## 已抽查且核对到来源的要点（无问题）

| 报告声明 | 核对来源 | 结果 |
|---------|---------|------|
| MegaPose 200 万张合成图训练 | cards/megapose...json method: "2百万张合成图像" | 一致 |
| MegaPose 520 个预渲染模板/视角 | cards/megapose...json limitation(3): "520个视角" | 一致 |
| MegaPose 精炼器 66.5 ms/步 | cards/megapose...json limitation(4): "66.5ms/步" | 一致 |
| MegaPose 精炼版为 BOP 2022 SOTA | cards/megapose...json eval_setup | 一致 |
| MegaPose +17 AR 提升（FoundPose 论文 L316-318） | papers/foundpose...md L317: "+17 AR (rows 1 vs 8)" | 一致 |
| MegaPose 数据集为 SAM-6D/SinRef-6D/RayPose 采用 | 三卡片各自 eval_setup/resources 确认 | 一致 |
| Gen6D 三阶段流水线（检测→视角选择→3D体积精炼） | cards/gen6d...json method | 一致 |
| Gen6D ~200 张参考图像 | cards/gen6d...json eval_setup | 一致 |
| Gen6D 3D 特征体积分辨率 32^3 | cards/gen6d...json limitation(4) | 一致 |
| Gen6D 深度估计为主要瓶颈，1-2 像素尺度差异致深度偏移 | cards/gen6d...json limitation(1) | 一致 |
| Gen6D 稀疏参考下视角选择器混淆相邻视角 | cards/gen6d...json limitation(2) | 一致 |
| Gen6D LINEMOD ADD 93.16%（GT-BBox） | cards/gen6d...json eval_setup | 一致 |
| Gen6D VGG-11 特征提取器 | cards/gen6d...json resources | 一致 |
| Gen6D 代码开源 | cards/gen6d...json resources | 一致 |
| GS-Pose 语义表示+旋转感知嵌入+3DGS 三层表示 | cards/gs_pose...json method | 一致 |
| GS-Pose 仅在 LINEMOD(13) 和 OnePose-LowTexture(8) 上评估 | cards/gs_pose...json limitation(3) + eval_setup | 一致 |
| GS-Pose Co-Segmenter 质量影响语义特征可靠性 | cards/gs_pose...json limitation(4) | 一致 |
| GS-Pose 3DGS 精炼推理速度较慢 | cards/gs_pose...json limitation(2) | 一致 |
| GS-Pose 代码开源 | cards/gs_pose...json resources | 一致 |
| FoundPose 最佳性能需 MegaPose 精修器，削弱无需训练纯粹性 | cards/foundpose...json limitation | 一致 |
| FoundPose 模板渲染固定光照与黑色背景 | cards/foundpose...json limitation | 一致 |
| FoundPose 7 BOP 数据集评测（AR 指标） | cards/foundpose...json eval_setup | 一致 |
| FoundPose LM-O Published AR = 34.0, TUD-L = 42.7 | codebases/foundpose.md 复现指标表 | 一致 |
| FoundPose 代码复现值 33.7 vs 发表 34.0 | codebases/foundpose.md 复现指标表 | 一致 |
| FoundPose 论文主实验 ViT-L/14 layer 18 | cards/foundpose...json method + papers/foundpose...md L293 | 一致 |
| FoundPose 代码配置 vits14-reg / layer=9 / facet=token / logbin=0 | codebases/foundpose.md F3 | 一致 |
| FoundPose 层号通过 extractor_name 字符串可配置 | codebases/foundpose.md F2 | 一致 |
| FoundPose logbin=0 被静默忽略 | codebases/foundpose.md R2 | 一致 |
| FoundPose DINOv2 ViT-S 共 12 层，layer=9 即第 10 层 | codebases/foundpose.md F3 | 一致 |
| FoundPose 多层融合需改 extract_descriptors 接口（方案B） | codebases/foundpose.md 方案B | 一致 |
| FoundPose gen_templates→gen_repre→infer 三步流水线 | codebases/foundpose.md 架构总览 | 一致 |
| FoundPose LM-O 配置文件已存在 | codebases/foundpose.md 目录树 | 一致 |
| FoundPose BOP Toolkit 子模块 external/bop_toolkit/ | codebases/foundpose.md 目录树 | 一致 |
| FoundPose 800 模板/物体 | papers/foundpose...md L288 | 一致 |
| FoundPose 论文 §4.1 CNOS 掩码引用 | papers/foundpose...md L306-308 | 一致 |
| FoundPose TF-IDF 检索+循环伙伴匹配+PnP | codebases/foundpose.md 架构总览 | 一致 |
| FoundPose 对比对象为 GenFlow/MegaPose/GigaPose/ZS6D/OSOP | cards/foundpose...json eval_setup | 一致 |
| Pos3R 40 模板（8顶点×5旋转） | cards/pos3r...json method | 一致 |
| Pos3R 粗估计 AR 39.5，精化后 57.3 | cards/pos3r...json eval_setup | 一致 |
| Pos3R 运行时间 1.4 秒/图 | cards/pos3r...json eval_setup | 一致 |
| Pos3R 严重遮挡（LM-O）匹配质量下降 | cards/pos3r...json limitation | 一致 |
| Pos3R 使用 CNOS（resources 字段） | cards/pos3r...json resources | 一致 |
| Pos3R 代码未明确公开（待验证） | cards/pos3r...json resources | 一致 |
| UNOPose ARBOP 70.9% | cards/unopose...json eval_setup | 一致 |
| UNOPose 80°-90° 区间 ARBOP 仅 54.8% | cards/unopose...json limitation | 一致 |
| SinRef-6D 单视角几何先验极度匮乏 | cards/scalable...json limitation(3) | 一致 |
| SinRef-6D 使用 MegaPose 合成数据训练 | cards/scalable...json eval_setup | 一致 |
| Cross-View CVSI + MegaPose 数据训练 440K 步 | cards/learning_cross_view...json eval_setup | 一致 |
| Cross-View 依赖分割质量 | cards/learning_cross_view...json limitation | 一致 |
| OPT-Pose 统一绝对/相对位姿 + 对比学习 | cards/object_pose_transformer...json method | 一致 |
| OPT-Pose RGB-only 尺度受单目歧义限制、对光照敏感 | cards/object_pose_transformer...json limitation | 一致 |
| Horyon 遮挡场景性能明显偏低 | cards/high_resolution...json limitation | 一致 |
| PoseGAM 5 个真实基准平均 AR 提升 5.1% | cards/posegam...json eval_setup | 一致 |
| PoseGAM >190k 物体、50 相机位姿 | cards/posegam...json eval_setup | 一致 |
| PoseGAM 原始几何特征令牌效果不佳 | cards/posegam...json limitation | 一致 |
| PoseGAM 自建数据集（非 MegaPose） | cards/posegam...json eval_setup + resources | 一致 |
| RayPose 细预测器仍依赖 MegaPose refiner | cards/raypose...json limitation③ | 一致 |
| RayPose 使用 MegaPose 合成数据训练 | cards/raypose...json eval_setup | 一致 |
| LatentFusion 需深度图+掩码、推理慢 | cards/latentfusion...json limitation(1)(2) | 一致 |
| OnePose++ keypoint-free + LoFTR 半稠密匹配 | cards/onepose...json method | 一致 |
| OnePose++ glue 48.0 vs PVNet 95.7 | cards/onepose...json limitation(2) | 一致 |
| OnePose++ 未在 BOP 标准协议下评测 | cards/onepose...json eval_setup | 一致 |
| 6D 连续旋转表示取旋转矩阵前两列 (W2949924544) | cards/on_the_continuity...json method | 一致 |
| 综述 W4396914081 指出 unseen 方法仍需 CAD 或参考图 | cards/deep_learning...json limitation(2) | 一致 |
| BOP 8 数据集、89 物体、62K 测试图像 | cards/bop...json method | 一致 |
| BOP VSD τ=20mm θ=0.3 默认设置 | cards/bop...json eval_setup | 一致 |
| BOP VSD 仅在可见表面区域计算误差 | cards/bop...json method | 一致 |
| BOP 测试时输入单张 RGB-D + 物体标识符 | cards/bop...json core_assumption | 一致 |
| DUSt3R 推理约 40 ms/pair (H100) | cards/dust3r...json resources | 一致 |
| DUSt3R DTU Overall 1.741mm vs 最优 0.295mm | cards/dust3r...json limitation | 一致 |
| DUSt3R CC BY-NC-SA 4.0 非商用 | codebases/dust3r.md 头部 | 一致 |
| DUSt3R O(N²) 对推理 | codebases/dust3r.md F7 | 一致 |
| DUSt3R up-to-scale 重建 | codebases/dust3r.md 风险(5) | 一致 |
| DUSt3R PnP: init_im_poses.py:272-273, reprojectionError=5, SOLVEPNP_SQPNP | codebases/dust3r.md F11 | 一致 |
| DUSt3R conf='log' 逐像素权重 | codebases/dust3r.md F12 | 一致 |
| DUSt3R scene_graph='swin-5'/'oneref-0' 缓解 | codebases/dust3r.md 改造接口点 4 | 一致 |
| DUSt3R 输出 (B,H,W,3) 点图 + (B,H,W) 置信度 | codebases/dust3r.md F1 | 一致 |
| DUSt3R scene.get_pts3d() / scene.get_im_poses() | codebases/dust3r.md F10 | 一致 |
| DUSt3R ViT-L 编码器 + ViT-B 解码器 | codebases/dust3r.md 硬编码参数表 | 一致 |
| DUSt3R 无需相机标定 | cards/dust3r...json problem | 一致 |
| DUSt3R make_pairs → inference → global_aligner 流程 | codebases/dust3r.md 最小运行命令 | 一致 |
| DUSt3R 全局对齐 niter=300 默认 | codebases/dust3r.md 硬编码参数表 | 一致 |
| 母题「通用套路」tension: 合成数据边际收益无人回答 | cards/_themes.json 第5条 tension | 一致 |
| 母题「共享假设」: VFM 几何精度未被验证 | cards/_themes.json 第3条 tension | 一致 |
| 母题「共同瓶颈」: Oryon 14.3 AR gap | cards/_themes.json 第4条 evidence | 一致 |
| 母题「共同瓶颈」: 分割-位姿联合优化无人做 | cards/_themes.json 第4条 tension | 一致 |
| Oryon 提示词敏感 −63.5 mIoU | cards/_themes.json 第3条 evidence | 一致 |
| DeepIM 旋转噪声限 45° | cards/_themes.json 第1条 evidence | 一致 |
| idea 卡 s_prior 权重 0.3/0.4/0.3 | ideas/先验知识-精度Pareto前沿...md | 一致 |
| idea 卡 N∈{1,5,10,50,200} | ideas/先验知识-精度Pareto前沿...md | 一致 |
| idea 卡最小实验 6 小时 | ideas/先验知识-精度Pareto前沿...md | 一致 |
| s_prior 公式验算（各方法得分） | 按公式手动计算 | 一致 |
| 所有论文 paper_id 在 cards/ 中存在 | 逐一核对 W4320013905/W4392971958/W4403842181/W4413146353/W4413146937/W7155098975/W7165818136/W7140953602/W4417296911/W4415967370/W4400023965/W4402816534/W2888752296/W4396914081/W2949924544/W4317552994/arxiv:2212.06846/arxiv:1910.10009/W4402753839/W4402727146/W2962783853 | 全部存在 |
| 报告头部"54 篇精读卡片" | ls cards/*.json 排除 _themes.json 后计数 = 54 | 一致 |
| 报告头部"领域母题（5 条）" | cards/_themes.json 含 5 个对象 | 一致 |

---

## 统计与判定

| 级别 | 数量 |
|------|------|
| P0（编造/来源不符，查证为假） | 0 |
| P1（无来源未标待验证/结论夸大） | 1 |
| P2（表述或一致性小问题） | 4 |

**总体判定：需修订后发布。**

一句话依据：唯一的 P1（LM-O 物体数 15→8）源自 idea 卡错误而非报告编造，但该数字向统计检验设计（n=15 对→n=8 对）、资源估算（800×15→800×8）、可视化计划等多处传播，影响实验设计的可执行性，必须修正。4 条 P2 为出处不精确或未标"待验证"的表述问题，不影响核心论证逻辑。报告对来源库的引用整体准确率极高——经逐项核查 80+ 项数字/代码事实/母题引用，除上述 5 条外均与来源库完全一致；前轮审校发现的 3 条 P0（FoundPose 模板数 520→800、PoseGAM 数据集归属、编造 FoundPose 论文含 Gen6D/GS-Pose 数字）已全部修正。修正 P1 后可发布。
