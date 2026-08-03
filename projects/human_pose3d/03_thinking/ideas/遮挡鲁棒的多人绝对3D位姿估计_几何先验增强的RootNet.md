# 遮挡鲁棒的多人绝对3D位姿估计_几何先验增强的RootNet

> 状态: draft · 2026-07-28

## Gap 来源（结构依据）
源自三条母题张力的交叉：①【通用套路·遮挡选择性遗忘】'NLP侧已发展出写什么/删什么/何时惊讶的完整理论(Titans梯度惊讶度、RWKV-7解耦删除键、DeltaNet门控delta规则α→0快擦/α→1精准更新)，而姿态侧路由信号仅为2D置信度阈值τ=0.5——恰在严重遮挡时最不可靠(OAHPE)；VNect自遮挡PCK仅48%、XNect重度遮挡完全依赖静态先验；P-STMO发现90%时间掩码才有效暗示姿态序列高度冗余，但无人用惊讶度/delta误差自适应决定哪些帧值得保留'——本idea把GDN门控delta记忆+Titans惊讶度门控迁移到姿态时序建模，正面接这条张力。②【共同瓶颈·效率-精度Pareto】'姿态论文(PoseMamba/OAHPE)声称线性模型可达到甚至超越注意力，但NLP理论与实验(BASED的Ω(N)下界、ReGLA/DeltaNet仍逊于Transformer)表明固定状态在精确检索上有不可消除损失；核心未验证假设：2D→3D lifting是否真的不需要精确召回——关节轨迹是低频平滑信号(PoseFormerV2的DCT假设)还是遮挡突变帧需精确检索历史？无人对姿态任务做过类似MQAR的召回需求分析'——本idea在遮挡突变帧上检验GDN(delta规则提供比固定状态更强的选择性检索)是否优于纯SSM/膨胀卷积，直接压力测试这条假设。③【共享假设·几何先验耦合】'骨长/重投影等约束本质是相对几何量，对全局平移(根深度)无观测力(Cross-View U明确承认)；真正的开放问题：当先验本身不确定时(遮挡、截断、极端视角)，耦合权重应如何自适应？无人让网络学习何时信任几何、何时放弃'——本idea保留RootNet几何先验主线但把骨长/根深度/重投影先验改为按自身不确定性加权、并反馈进惊讶度门控。结构旁证：2D检测误差级联(母题4'越需要3D消歧时前端越不准'的恶性循环)与多人根深度歧义(母题5)由RootNet主线+几何不确定性加权承接。

## 动机
RootNet(Moon 2019)以DetectNet→RootNet(绝对根深度，距离度量k=√(αx·αy·A_real/A_img))→PoseNet(root-relative 3D)三模块top-down框架成为多人绝对3D位姿的标准管线，其几何先验(面积比根深度)是主线价值。但原PoseNet是单帧lifting，对每帧每人独立做2D→3D提升：单目深度歧义无时序消解、遮挡帧只能靠静态先验硬填(VNect自遮挡PCK仅48%)，且'越遮挡→2D越不准→3D越崩'的恶性循环无人结构性解决(母题4)。与此同时视频时序lifting已全面证明时序上下文的价值(TCN→MixSTE→MotionBERT→PoseMamba/BSTMamba线性SSM)，而GDN门控delta规则以线性复杂度提供'选择性记忆+快速遗忘'，其门控语义(α→0快擦/α→1精准更新)与遮挡恢复天然契合：记住可见时的可靠3D估计、在2D证据矛盾时快速改写。NLP侧惊讶度门控(Titans用关联记忆损失梯度衡量惊讶度)更给出'何时该写记忆'的自适应判据，恰好替代姿态侧脆弱的硬阈值路由(OAHPE τ=0.5在严重遮挡时失效)。然而，直接搬运NLP惊讶度语义到姿态侧存在一个被忽视的范畴错误：NLP中惊讶度高意味着"世界变了，旧记忆过时"，应擦除；而姿态遮挡时2D观测退化导致的"惊讶度"高意味着"传感器失效，观测不可信"，此时正确行为恰恰是冻结记忆、依赖惯性外推与几何先验。若将2D置信度跌落、delta预测残差、重投影误差混合为单一惊讶度标量驱动擦除，则遮挡最严重(2D最不可靠)时系统反而最激进地擦除可见期积累的可靠3D记忆——与母题4'越遮挡→前端越不准'的恶性循环形成正反馈而非打断它。本idea的关键修正：将惊讶度解耦为**观测质量门**(2D置信度/检测面积退化→控制是否信任当前输入)与**状态惊讶门**(仅在观测可信时计算的delta预测误差→控制记忆改写幅度)，使遮挡期记忆被保护而非被擦除。但此修正本身面临一个循环性风险：观测质量门β_obs若仅依赖2D检测器自身输出的置信度与框面积，则恰好复现了Gap来源①所诊断的病灶——"2D置信度恰在严重遮挡时最不可靠"。具体而言，当遮挡物(另一人、物体)经过时，现代检测器(CPN/HRNet)常对错误位置仍输出高置信度(幻觉关节)，此时β_obs不会关闭，被污染的观测照样写入记忆，解耦形同虚设。打破此循环的关键在于引入**不依赖检测器自报置信度的独立观测质量信号**：几何一致性残差(当前帧2D骨架的骨长应变、相邻帧速度突变、多关节重投影误差)——这些量可从2D坐标本身与时序记忆预测的对比中计算，无需检测器"承认自己错了"。因此β_obs必须由(2D置信度退化 ∨ 几何一致性违反)双通道驱动，后者在检测器被欺骗时提供兜底关断。现在能做：fla库已提供硬件高效的chunk_gated_delta_rule kernel(WY表示分块并行)，PoseMamba repo证明SSM时空块可在(B,T,J,2)→(B,T,J,3)lifting主干中即插即换，二者拼合的工程路径已被repo卡逐行锚定。

## 核心假设
如果在保留RootNet的DetectNet+RootNet绝对根深度几何先验主线前提下，把其单帧PoseNet替换为按人跟踪的2D骨架序列上的双向GDN(门控delta规则)线性注意力时序lifter，且将GDN门控解耦为两级——**观测质量门**β_obs(由双通道驱动：通道A为2D置信度与检测框面积退化，通道B为几何一致性违反信号即骨长应变+帧间速度异常+多关节重投影残差，任一通道触发即关断写入)与**状态惊讶门**g_state(仅在β_obs>θ_obs的可靠帧上计算delta预测残差，高残差→α→0定向替换过时记忆、低残差→α→1保留)——那么在遮挡/截断序列上被遮挡关节的恢复误差与多人绝对位姿(MuPoTS-3D MRPE/3DPCKabs)将显著优于固定窗口SSM/注意力基线(PoseMamba/MixSTE)与RootNet原单帧PoseNet，且在H36M标准协议MPJPE/P-MPJPE上至少持平线性SSM基线；进一步，当骨长/根深度/重投影几何先验按其自身不确定性加权(而非硬约束)并在β_obs<θ_obs期间作为记忆冻结时的唯一位置外推依据时，遮挡/截断下的根深度与末端关节误差额外下降。可证伪判据：(i)若解耦门控组相对单一混合惊讶度组在遮挡段关节误差上无≥3%改善，则"范畴错误"修正不成立，回退原方案；(ii)若双通道β_obs(置信度+几何一致性)相对纯置信度β_obs在"检测器高置信但位置错误"的幻觉遮挡子集上无≥5%改善，则几何兜底通道无效，观测质量门退化为原OAHPE式软阈值。

## 技术路线
保留RootNet三模块top-down结构(DetectNet人体检测+跟踪 / RootNet逐人逐帧绝对根深度Z_R / PoseNet根相对3D)，仅替换PoseNet为GDN时空lifter，组合卡片方法如下：(1)【主干·来自PoseMamba repo】输入每人跟踪的2D骨架序列(B,T,J=17,2，root-centered、归一化[-1,1])，交替堆叠空间GDN块(沿关节维双向扫描，顺序用骨骼拓扑局部重排序，承PoseMamba/BSTMamba)与时间GDN块(沿帧维双向扫描)；每个GDN块用fla的chunk_gated_delta_rule(q,k,v,g,beta)实现，因fla chunk kernel硬编码因果下三角mask，双向以前向+反向两次扫描后CrossMerge求和(与PoseMamba mambablocks.py:582的CrossScan/CrossMerge同构)实现，输出(B,T,J,3) root-relative序列，再与RootNet的Z_R反投影组合得相机坐标系绝对3D——几何先验主线原样保留。(2)【解耦双门控·核心修正】将原"混合惊讶度→单一门控"拆为两级：(2a)观测质量门β_obs采用**双通道关断**结构：通道A(检测器自报)β_conf=σ(w₁·conf_2d + w₂·bbox_area_ratio + w₃·visibility_flag + b)；通道B(几何一致性兜底)β_geo=σ(−w₄·bone_strain − w₅·‖v_t−v_{t−1}‖/σ_v − w₆·e_reproj + b')，其中bone_strain=|‖j_i−j_parent‖_t / ‖j_i−j_parent‖_{t−1} − 1|逐骨计算取max，e_reproj为当前2D与记忆3D经Z_R重投影的像素误差；最终β_obs=min(β_conf, β_geo)——即任一通道判定不可信即关断写入，确保检测器幻觉(高conf但骨长/重投影严重违反)时β_obs仍→0；β_obs∈[0,1]直接乘入delta规则的写入系数beta(即有效beta_eff=β_obs·β_learned)——当2D检测退化(遮挡/截断/检测器幻觉)时β_obs→0，delta规则退化为h_{t+1}=α·h_t(纯衰减保持)，记忆不被不可靠观测污染或擦除；(2b)状态惊讶门g_state仅在β_obs>θ_obs(可学习，初始化0.5)的帧上由delta残差r=‖v_new−h@k‖驱动：g_in=f_mlp(r, Δr/Δt, bone_strain)，高r且r持续上升(真运动突变)→α→0定向替换，高r但r单帧脉冲(检测噪声)→α保持；(2c)遮挡恢复逻辑：当β_obs从<θ_obs回升至>θ_obs(关节重新可见且几何一致)，第一帧强制β_eff=1且g_state由r决定——若r小(姿态未变)则α→1直接读回冻结期记忆(零代价恢复)，若r大(遮挡期间姿态改变)则α→0快速改写。此设计使"遮挡期保护记忆、恢复期按需读写"成为门控结构的自然涌现而非外加规则。(3)【测试时记忆·来自已过审惊讶度门控idea】维护逐关节3D测试时记忆缓冲，冻结期(β_obs<θ_obs)记忆不更新但可被几何先验外推修正(见(4))，开放期(β_obs>θ_obs)按g_state决定写入幅度，分离'遮挡期惯性保持'(冻结+几何外推)与'可见期消歧'(delta精准更新)。(4)【几何先验不确定性加权·来自母题3张力+MAR/RePos】骨长恒定、RootNet根深度、2D重投影先验均不做硬约束，而按各自不确定性(根深度用RootNet面积比k在遮挡/截断时的退化程度、骨长用跨帧方差)加权为软损失；关键修正：在β_obs<θ_obs的冻结期，几何先验(尤其骨长约束+速度惯性)作为记忆位置的唯一外推修正源(替代不可用的2D观测)，其修正幅度按不确定性倒数加权；在β_obs>θ_obs期几何残差反馈进g_state的r计算(辅助判断运动突变vs检测噪声)——形成'观测可信时几何辅助判惊讶、观测不可信时几何替代观测维持记忆'的双模式闭环。注意：几何一致性信号同时服务于β_obs通道B(关断判据)与(4)(外推修正)，但二者角色不同——前者是二值化门控触发(阈值化)，后者是连续值修正(加权)，不产生梯度冲突。

## 最小实验设计
数据/基线/指标/预期，收窄到一脚本+result.json：【数据】Human3.6M(S1/5/6/7/8训练,S9/11测试,17关节,Protocol#1 MPJPE/Protocol#2 P-MPJPE,CPN检测2D与GT 2D两组,T=81/243)；遮挡鲁棒性用合成遮挡增强(随机mask关键点/帧/区域,承3D HPE explicit occlusion training与P-STMO掩码思路)，另加**结构化遮挡**(模拟另一人 bbox 经过导致的连续关节丢失，持续8-32帧，以区分检测噪声型单帧丢失与真遮挡型连续退化)；**关键新增：幻觉遮挡子集**——在结构化遮挡帧上保留检测器对错误位置的高置信度输出(即conf>0.8但GT标记为occluded)，专门压力测试β_obs通道B的兜底能力(若仅用通道A则此子集上系统完全失效)；多人绝对位姿用MuPoTS-3D(MRPE/3DPCKabs/AP_root_25)验证完整RootNet+GDN管线。【基线】VideoPose3D、MixSTE、PoseMamba(同构SSM主干,直接对照)、MotionBERT、RootNet原单帧PoseNet；消融：(a)去解耦——回退为单一混合惊讶度门控(即原方案，直接验证"范畴错误修正"是否有效)、(b)去观测质量门(仅保留状态惊讶门，即遮挡期仍用delta残差驱动擦写)、(b2)**去几何一致性通道B**(β_obs仅由通道A即conf/area驱动，验证"检测器幻觉兜底"是否必要——预期在幻觉遮挡子集上大幅退化)、(c)去双向(纯因果)、(d)去几何不确定性加权(硬先验)、(e)GDN换纯Mamba/SSM(隔离delta规则贡献)、(f)θ_obs固定为0/0.5/1(退化为全开放/原方案/全冻结，检验可学习阈值的必要性)。【实现】PoseMamba代码库为骨架，把mambablocks.py:582 BiSTSSM的selective_scan替换为fla chunk_gated_delta_rule(chunk_size∈{16,32,64}，torch≥2.7/triton≥3.3)，双向以前向+反向扫描CrossMerge；β_obs=min(β_conf,β_geo)乘入beta参数前向传播；几何一致性信号(bone_strain, velocity, reproj)从输入2D序列与记忆3D预测实时计算，不引入额外网络参数(仅6个标量权重w₄-w₆+b')；RootNet根深度模块用其官方RELEASE权重。【指标与result.json】MPJPE/P-MPJPE(mm)、遮挡段被mask关节恢复误差(mm，按遮挡持续长度分桶：短<8帧/中8-32帧/长>32帧)、**幻觉遮挡子集恢复误差**(conf>0.8但GT occluded的帧，隔离通道B贡献)、遮挡期记忆保持率(冻结期记忆向量与遮挡前最后一帧的余弦相似度)、MuPoTS-3D MRPE/3DPCKabs、参数量/MACs/吞吐(fps)；result.json字段：{method, h36m_p1_cpn, h36m_p1_gt, h36m_p2, occ_recover_err_short, occ_recover_err_mid, occ_recover_err_long, occ_hallucination_err, occ_memory_cosine, mupots_mrpe, mupots_3dpckabs, params_M, macs_G, fps, ablations:{mixed_surprise, no_obs_gate, no_geo_channel_b, causal_only, hard_prior, pure_mamba, theta_fixed_0, theta_fixed_05, theta_fixed_1}}。【预期】H36M全序列至少持平PoseMamba(约38.1mm CPN/15.6mm GT@T=243量级)；核心判据(假设成立)：解耦门控组相对混合惊讶度组(消融a)在中/长遮挡段(≥8帧)关节恢复误差降幅≥5%且记忆保持率>0.85(混合组预期<0.5)；**通道B判据**：双通道β_obs相对纯通道A(消融b2)在幻觉遮挡子集上恢复误差降幅≥5%(若检测器幻觉帧占比>20%的结构化遮挡中此判据不满足，则几何兜底通道失效)；MuPoTS-3D绝对位姿优于RootNet原单帧PoseNet；附带'lifting召回需求'分析(对比结构化遮挡恢复帧vs平滑帧上GDN与纯SSM/膨胀卷积的差异)以回应母题2的核心未验证假设。

## 相关论文
- W2964784655
- arxiv:2412.06464 — Gated Delta Networks: Improving Mamba2 with Delta Rule
- arxiv:2501.00663
- arxiv:2312.06635 — Gated Linear Attention Transformers with Hardware-Efficient Training
- arxiv:2406.06484 — Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- arxiv:2503.14456
- arxiv:2502.01578 — ReGLA: Refining Gated Linear Attention
- arxiv:2501.12352
- arxiv:2402.18668 — Simple linear attention language models balance the recall-throughput tradeoff
- arxiv:2606.04048 — Unlocking Feature Learning in Gated Delta Networks at Scale
- W4409368373 — PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model
- W4413980847 — A Spatiotemporal Bidirectional Mamba Network with Global–Local Skeletal Enhancement for 3D Human Pose Estimation
- W4312417903 — MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video
- W4390874423 — MotionBERT: A Unified Perspective on Learning Human Motion Representations
- W2903549000 — 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training
- W7168163523 — OAHPE: occlusion-aware hybrid routing for efficient and robust 3D human pose estimation in mixed-occlusion videos
- W4386083126 — PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation
- W3117675859
- W3116592456
- W4304091600
- W7167749100 — RePos: Relative-to-Absolute Output Factorization for Cross-Environment WiFi-Bas

## 评审记录（critique_idea 自动写入）

### 查重（top 相近工作）
无
（检索词: gated delta network 3D pose lifting, occlusion robust 3D pose temporal memory gating, linear attention video 2D-to-3D pose estimation；共命中 0 条，未经逐篇判定过滤）

### 对抗评审 3/3 票支持
✅ 评审通过（多数派未能驳倒）
  - The idea presents a genuinely novel combination (decoupled observation-quality vs state-surprise gating in GDN for occlusion-robust temporal 3D pose lifting on RootNet), identifies a real and well-argued category error in naively transplanting NLP surprise semantics to pose, and has a credible engineering path via existing fla kernels and PoseMamba-style architectures; no closely related prior work was found.
  - The idea identifies a genuine gap (occlusion-robust temporal 3D lifting with adaptive geometric priors), proposes a non-trivial architectural contribution (decoupled observation-quality vs. state-surprise gating to avoid the category error of naive NLP-surprise porting), and is grounded in feasible, existing tooling (fla GDN kernels, RootNet pipeline); no clearly overlapping prior work was found, making it a plausible and worthwhile research direction.
  - The proposal identifies a genuine gap (hard-threshold routing vs. adaptive gating in occluded pose), offers a novel decoupled observation-quality/state-surprise gating mechanism with concrete geometric fallback signals, and is feasible given existing fla kernels and PoseMamba-style SSM lifting backbones; no closely related prior work was found.

## 可执行性评估：重
- 外部仓库: 4 个（sustcsonglin/flash-linear-attention, mks0601/3DMPPE_ROOTNET_RELEASE, Yunlongs/PoseMamba, leoxiaobin/deep-high-resolution-net.pytorch） · GPU: 需要 · 预训练权重: 需要 · 数据准备: 大 · 胶水复杂度: 高
- 风险点: 最大风险在于将fla的chunk_gated_delta_rule CUDA kernel嵌入RootNet逐人跟踪序列的PoseNet替换中，同时实现双通道观测质量门（几何一致性残差需逐帧骨长应变+速度异常+重投影计算）与不确定性加权几何先验的联合训练——四仓库拼接+自定义门控逻辑+多人跟踪序列构造，调试链路极长，必须人工逐步排障。
- 结论: 重工程实验，plan_experiment 出方案书交用户执行，不要自动跑。
- 【下一步必做】这些涉及仓库还没有 repo 卡：mks0601/3DMPPE_ROOTNET_RELEASE, leoxiaobin/deep-high-resolution-net.pytorch。定稿后、写方案书/报告前，先逐个 study_codebase 查证工程事实。
