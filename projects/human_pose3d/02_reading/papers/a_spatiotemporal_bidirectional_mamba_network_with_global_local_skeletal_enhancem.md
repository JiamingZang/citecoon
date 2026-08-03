# A Spatiotemporal Bidirectional Mamba Network with Global–Local Skeletal Enhancement for 3D Human Pose Estimation

> 2025 · id: W4413980847 · pdf: https://www.researchsquare.com/article/rs-7477209/latest.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

A Spatiotemporal Bidirectional Mamba Network
with Global–Local Skeletal Enhancement for 3D
Human Pose Estimation
Chuhan Wu 
Henan University
Zan Wang 
Henan University
Guixian Zhou 
Henan University
Jiahao Hua 
Henan University
Lianke Shi 
Henan University
Research Article
Keywords: 3D Human Pose Estimation, State-Space Models, Bidirectional Modeling, Dynamic Gating,
Structural Perturbation Enhancement
Posted Date: September 4th, 2025
DOI: https://doi.org/10.21203/rs.3.rs-7477209/v1
License:   This work is licensed under a Creative Commons Attribution 4.0 International License.  
Read Full License
Additional Declarations: No competing interests reported.
Version of Record: A version of this preprint was published at The Visual Computer on December 14th,
2025. See the published version at https://doi.org/10.1007/s00371-025-04212-0.

A Spatiotemporal Bidirectional Mamba Network
with Global–Local Skeletal Enhancement for 3D
Human Pose Estimation
Chuhan Wu1, Zan Wang1*, Guixian Zhou1, Jiahao Hua1
1School of Physics and Electronics, Henan University, Kaifeng City
Intelligent Manufacturing Engineering Technology Research Center,
Kaifeng, 475004, Henan Province, China.
*Corresponding author(s). E-mail(s): 10110137@vip.henu.edu.cn;
Contributing authors: wuchuhan0081@henu.edu.cn;
104754231448@henu.edu.cn; 104753231063@henu.edu.cn;
Abstract
3D human pose estimation (HPE) is a cornerstone task in computer vision
with diverse applications, where lifting 2D pose sequences to 3D represen-
tations has attracted signiﬁcant interest. Transformer-based approaches have
demonstrated robust performance but are hampered by quadratic computational
complexity and insuﬃcient bidirectional modeling capabilities. The recently
introduced Mamba model mitigates these limitations through state-space models
(SSMs) oﬀering linear complexity and eﬀective long-range dependencies; how-
ever, it falls short in modeling local skeletal interactions essential for human
motion. To address this, we present BSTMamba, a bidirectional spatiotempo-
ral SSM architecture designed speciﬁcally for monocular 3D HPE. BSTMamba
integrates eﬃcient global sequence modeling with localized convolutions and
dynamic gating mechanisms to capture intricate spatiotemporal dependencies.
For enhanced robustness and generalization, we introduce DisruptEnhance, a
residual-compensated joint-order perturbation module that randomly disrupts
joint orders at both global (full-skeleton) and local (body-part) scales, followed
by feature compensation via a lightweight residual subnet. Comprehensive evalu-
ations on the Human3.6M and MPI-INF-3DHP datasets reveal that BSTMamba
attains state-of-the-art accuracy while requiring fewer parameters and lower
multiply-accumulate operations (MACs) compared to prior methods.
Keywords: 3D Human Pose Estimation; State-Space Models; Bidirectional Modeling;
Dynamic Gating; Structural Perturbation Enhancement
1

1 Introduction
3D human pose estimation (HPE) from monocular observations is a fundamental
task in computer vision, with broad applications in action recognition [1], posture
correction and online guidance [2], augmented reality, and virtual reality [3]. As
application scenarios expand, the demand for highly accurate and eﬃcient pose esti-
mation models continues to grow. However, in practical settings, where standard
RGB cameras are prevalent, captured pose sequences are often available only in
2D form. Thus, eﬀectively lifting 2D poses into 3D space while ensuring accuracy
and robustness remains a key challenge. Researchers have addressed this challenge
using deep learning methods [4–7] that map 2D pose sequences to 3D. Neverthe-
less, the absence of depth information in 2D environments renders this an inherently
ill-posed problem. To alleviate this, many studies exploit spatial and temporal relation-
ships among joints in input videos. For instance, Transformer architectures, initially
proposed by Vaswani et al. [8], utilize self-attention to capture spatio-temporal depen-
dencies eﬀectively, demonstrating strong potential in 3D HPE. Prominent examples
include PoseFormer [9], which accurately estimates the center-frame pose from video
sequences; MHFormer [10], which learns spatio-temporal representations of multi-
ple pose hypotheses in an end-to-end fashion; and MixSTE [11], which employs
a Transformer-based sequence-to-sequence design to maintain sequential coherence.
Despite these advancements, Transformer-based methods suﬀer from quadratic com-
plexity relative to the number of frames, impeding deployment on resource-constrained
devices. Furthermore, eﬃciency-enhancing techniques, such as token pruning [12, 13],
often diminish global receptive ﬁelds, failing to resolve the inherent trade-oﬀbetween
accuracy and eﬃciency fundamentally. In contrast, state space models (SSMs) [14–
16] have exhibited considerable potential in surmounting these limitations. Recent
developments, particularly the structured state space sequence model (S4) [17], oﬀer
promising architectures for sequence modeling. Building on S4, the Mamba model [14]
incorporates time-varying parameters into SSMs and introduces a hardware-aware
algorithm that achieves global receptive ﬁelds with linear complexity. However, the
standard Mamba model [14] primarily emphasizes long-range dependencies, over-
looking critical local spatio-temporal interactions in human motion. To tackle this
limitation, we propose BSTMamba, a novel 3D HPE architecture that incorporates
a bidirectional global–local spatio-temporal modeling framework. BSTMamba utilizes
bidirectional SSMs [15] to capture global features across the entire pose sequence,
while local modeling emphasizes ﬁne-grained details of limb movements. Speciﬁcally,
we augment the standard Mamba [14] with local convolutions and dynamic gating
mechanisms to enhance joint interactions and facilitate adaptive global–local feature
fusion, thereby boosting pose estimation accuracy. Our experiments, however, indicate
that global modeling may memorize speciﬁc motion rhythms, whereas local model-
ing can over-rely on ﬁxed skeletal structures, resulting in overﬁtting and diminished
generalization to unseen actions, cross-domain data, or skeleton perturbations. To mit-
igate these issues, we introduce DisruptEnhance, a residual-compensated joint-order
perturbation module. DisruptEnhance randomly shuﬄes joint orders at both global
(full-skeleton) and part (body-region) levels and employs a lightweight residual subnet
to compensate for the perturbed features. This approach reduces dependence on ﬁxed
2

skeletal topology, alleviates memorization of motion rhythms in long sequences, and
improves cross-domain generalization while mitigating overﬁtting. Extensive experi-
ments on the Human3.6M and MPI-INF-3DHP datasets demonstrate that BSTMamba
outperforms state-of-the-art methods while requiring fewer parameters and MACs,
underscoring the substantial potential of SSMs in 3D human pose estimation. Our
main contributions are summarized as follows:
• We introduce a novel bidirectional global–local spatio-temporal modeling approach
within the Mamba framework for 3D HPE.
• We propose residual enhancement and joint-order perturbation strategies to enhance
robustness and generalization.
• BSTMamba achieves superior performance on Human3.6M and MPI-INF-3DHP
with fewer parameters and MACs.
2 Related Work
2.1 3D Human Pose Estimation
3D human pose estimation (HPE) can be categorized by the number of camera views
into monocular [10, 11, 18] and multi-view approaches [19–21]. Given that single
cameras are more readily deployable in real-world scenarios, monocular 3D HPE—
estimating 3D human poses from a single image or video—has garnered substantial
attention. Most monocular methods adopt a two-stage pipeline: (i) an oﬀ-the-shelf 2D
pose detector [22, 23] identiﬁes joint positions in the image, and (ii) a lifting module
maps these 2D poses to 3D. This paper focuses on the latter, the challenging 2D-to-3D
lifting step [18, 24].
Advances in 2D pose detection, such as Mask R-CNN [25], Cascaded Pyramid Net-
works [26], Stacked Hourglass [27], HRNet [28], YOLO-Pose [29], and MediaPipe [30],
have signiﬁcantly enhanced the performance of subsequent lifting methods that rely
on estimated 2D poses. These intermediate 2D estimations substantially reduce data
requirements and simplify the complexity of 3D HPE. Despite this progress, the lifting
task remains hindered by the inherent lack of depth information [31]. Consequently,
a substantial body of research leverages temporal cues across adjacent frames [32],
evolving from early temporal convolutional networks (TCNs) [33, 34] to graph convo-
lutional networks (GCNs) [27, 35] and, more recently, Transformers. Representative
methods include the TCN-based VideoPose3D, the GCN-based SemGCN and ST-
GCN, and the Transformer-based PoseFormer [9], MHFormer [10], and MixSTE [11].
While each paradigm oﬀers distinct advantages, they also exhibit notable limitations:
TCNs excel in eﬃcient temporal modeling but struggle with spatial skeletal structures;
GCNs eﬀectively incorporate skeleton topology yet rely on predeﬁned (ﬁxed) graphs,
constraining generalization; and Transformers adeptly capture long-range dependen-
cies through self-attention but incur quadratic complexity with respect to the number
of frames, complicating real-time or resource-limited deployment.
In contrast, PoseMamba [36] and PoseMagic [37] introduce SSM-based bidirec-
tional spatiotemporal modeling for 3D HPE. However, both approaches may be limited
by their over-reliance on GCNs for local detail modeling.
3

2.2 State Space Models
State space models (SSMs) have recently emerged as eﬃcient alternatives [17, 38–40]
to Transformers for sequence modeling. Gu et al. [17] proposed the structured state
space model S4 and its diagonal variant S4D, which mitigate the computational bottle-
necks of traditional Transformers on long sequences. Nonetheless, these models process
inputs in a largely content-agnostic manner, restricting their ability for content-aware
modeling.
To enhance content sensitivity, Mamba [14] incorporates time-varying parameters
into the SSM framework and introduces a hardware-aware algorithm for eﬃcient train-
ing and inference. As research advances, SSMs have been adapted to vision tasks. For
instance, Vim [41] utilizes a bidirectional SSM [42] architecture to achieve performance
comparable to Vision Transformers [43] while providing superior eﬃciency and scala-
bility. Mondal et al. [24] also applied S4D to 3D human pose estimation, enabling faster
training and real-time inference. However, SSMs are inherently limited in capturing
local spatial dependencies among joints: their design prioritizes long-range sequence
modeling and lacks explicit mechanisms for local skeletal topology, making it chal-
lenging to model ﬁne-grained geometric structures and interactions, which ultimately
hampers pose accuracy.
In this work, we extend Mamba to 3D pose estimation by proposing a bidirectional
global–local modeling framework that enhances accuracy while maintaining eﬃciency,
thereby achieving a favorable balance between eﬀectiveness and computational cost.
3 Method
3.1 Overall Architecture
As illustrated in Fig. 1, we present BSTMamba, an eﬃcient attention-free framework
for 3D human pose estimation. The model consists of three components: (i) a 2D joint
positional embedding layer, (ii) a stack of BSTBlock modules, and (iii) a regression
head.
4

Fig. 1 Overall architecture of BSTMamba.
For the 2D joint positional embedding, given an input 2D skeleton sequence X2D ∈
RB×T ×J×3, where B is the batch size, T is the sequence length (number of frames),
and J is the number of joints, the input is ﬁrst mapped to a hidden dimension d
through a linear projection (notation in this paper is uniﬁed as follows: d denotes
the input feature dimension, dinner the higher-dimensional feature space after linear
mapping, and dmodel the ﬁnal output dimension; these deﬁnitions remain consistent
throughout):
X(0) = GELU
 Linear(X2D)

∈RB×T ×J×d.
(1)
Subsequently, a positional encoding Ppos ∈RB×T ×J×d is added to the initial features
to form the input representation for subsequent modeling. The features are then fed
into multiple stacked BSTBlocks. Each BSTBlock, following the pipeline in Fig. 1,
comprises three submodules: ﬁrst, the DisruptEnhance module, which mitigates the
sensitivity of state space models (SSMs) to joint ordering in spatial modeling and
is placed before BSTMamba; its structure and role are detailed in Sec. 3.2. Next,
BSTMamba (Spatial) performs bidirectional state-space modeling along the spatial
dimension to capture local and global dependencies among skeletal joints; ﬁnally, BST-
Mamba (Temporal) performs bidirectional state-space modeling along the temporal
dimension to characterize cross-frame motion dynamics and temporal patterns.
After several stacked BSTBlocks, the resulting high-dimensional spatiotemporal
features are fed to the regression head to estimate 3D pose coordinates bP3D ∈
RB×T ×J×3. The network is trained by minimizing the mean squared error (MSE)
between predictions and ground truth:
L =
 bP3D −P3D

2
2 .
(2)
This architecture integrates spatial and temporal information without attention,
enabling eﬃcient long-range dependency modeling. Notably, DisruptEnhance both
5

perturbs and enriches features and, to some extent, compensates for SSM limitations
in spatial modeling; see Sec. 3.2 for the design rationale and implementation.
3.2 DisruptEnhance
Although Mamba is designed for one-dimensional sequence modeling and is
inherently sensitive to input order, this characteristic—appropriate for temporal
sequences—limits its adaptability when applied along the joint dimension, where
human skeletons may follow diverse topologies or annotation styles. Conventional
spatial modeling often relies on a ﬁxed skeletal graph, which can restrict generalization.
Inspired by shuﬄe-based perturbations [44–46], we introduce DisruptEnhance, a
spatial modeling strategy that couples structural perturbation with a lightweight
residual compensation subnetwork. Unlike methods dependent on predeﬁned skele-
tal graphs, our approach improves generalization by randomly shuﬄing joint orders
to encourage structural invariance while using residual compensation to recover
information potentially lost during perturbation. This reduces order dependence,
strengthens robustness to heterogeneous skeleton styles, and improves adaptability in
pose estimation tasks that demand strong spatial reasoning.
Fig. 2 Architecture of the DisruptEnhance module.
Given an input feature tensor X ∈RB×T ×J×d, we adopt a probability-driven,
training-time perturbation mechanism. With a ﬁxed probability ρskip, perturbation is
skipped and the original features are fed directly into the backbone module F
Yorig = F(X).
(3)
Otherwise, with probability ρdisrupt, a permutation along the joint dimension is applied
before the backbone.
To enhance generalization across diﬀerent skeletal topologies, we employ two
random perturbation strategies during training:
6

Global disrupt
A partial random permutation πglobal is sampled from joint indices
{0, 1, . . . , J −1},
where non-central joints (e.g., hip, spine, neck excluded) are shuﬄed with strength con-
trolled by ρglobal. This preserves the core skeletal topology while promoting invariance
to joint ordering. Perturbation is disabled at inference:
eX = X[:, :, πglobal, :].
(4)
Local disrupt
Following Human3.6M, joints are partitioned into ﬁve regions:
g0 = {hip, spine, thorax, neck, head},
g1 = {rightHip, rightKnee, rightFeet},
g2 = {leftHip, leftKnee, leftFeet},
g3 = {rightShoulder, rightElbow, rightWrist},
g4 = {leftShoulder, leftElbow, leftWrist}.
Independent random permutations are applied within each region and concatenated
to form πlocal; joints outside these groups retain their original order. A schematic of
the regional partition is shown in Fig. 3.
Fig. 3 Human joint diagram: each index corresponds to a speciﬁc joint.
Regardless of the chosen strategy, the perturbed features are passed through the
backbone to obtain eY = F( eX). While perturbation promotes invariance, it can dis-
rupt local semantic adjacency, especially under global permutations. To address this,
we introduce a lightweight residual compensation subnetwork, PoseEnhance, imple-
mented as a shallow MLP (input projection, several linear residual blocks, and an
output mapping):
bY = eY +
L
X
l=1
ReLU
 Wl eY

,
(5)
7

where Wl denotes a linear transformation (we use three stacked layers by default).
The enhanced features are then restored to the original joint order via the inverse
permutation π−1:
bYinv = bY[:, :, π−1, :].
(6)
To avoid over-reliance on the enhancement path, we further adopt mask-based
fusion. A random contiguous segment along the joint dimension is sampled to construct
a binary mask:
M ∈{0, 1}B×1×J×1.
This mask blends the enhanced and original outputs:
Yout = M ⊙bYinv +
 1 −M

⊙Yorig.
(7)
where ⊙denotes element-wise multiplication. The entire module is activated only
during training and adds no computational overhead at inference. By jointly mod-
eling structural perturbation and residual compensation, the model learns spatial
invariance, alleviates overﬁtting to ﬁxed joint orders, and improves pose estimation
robustness across diverse skeletal topologies and annotation styles.
3.3 BSTMamba
Fig. 4 Architecture of BSTMamba.
To further enhance the model’s ability to capture the spatiotemporal characteristics
of human poses—particularly for dynamic modeling of complex actions—we propose
8

the BSTMamba module. Unlike standard Mamba, BSTMamba incorporates a bidi-
rectional scanning mechanism, non-causal convolution, dynamic gating fusion, and
gated RMS normalization [47], enabling more eﬀective modeling of bidirectional spa-
tiotemporal dependencies in human pose estimation. The overall structure is shown
in Fig. 4.
Given the feature tensor X ∈RB×T ×J×d produced by DisruptEnhance (with B
the batch size, T the number of frames, J the number of joints, and d the feature
dimension), we apply bidirectional selective state-space modeling as follows.
In the temporal mode, the input is rearranged in a joint-major order and unfolded
to (B, J × T, d), where the spatial joint index changes more slowly and the temporal
index varies contiguously inside:
Xtemporal = rearrange
 X, ”B T J d →B (J T) d”

.
(8)
In the spatial mode, the input is rearranged in a time-major order and unfolded to
(B, T ×J, d), where the temporal index changes more slowly and the joint index varies
contiguously inside:
Xspatial = rearrange
 X, ”B T J d →B (T J) d”

.
(9)
In either mode, we obtain input features hidden states ∈RB×L×dmodel (with
sequence length L), which are linearly projected to a higher-dimensional space dinner:
Xproj = Linear(hidden states),
Xproj ∈RB×L×dinner.
(10)
The projected features are then rearranged and split channel-wise into two equal parts:
Xproj
rearrange
−−−−−−−→(B, dinner, L) →[x, z],
x, z ∈RB× dinner
2
×L.
(11)
We replace causal convolution with non-causal convolution, padding on both sides
of the sequence so that each position accesses past and future context. This generally
improves robustness for complex actions. In addition, we introduce a symmetric branch
without SSM—consisting of a 1D convolution followed by SiLU—to strengthen local
perception and compensate for potential information loss due to sequential modeling:
xfwd = SiLU
 Conv1D(x)

,
zfwd = SiLU
 Conv1D(z)

,
(12)
xbwd = SiLU
 Conv1D(ﬂip(x))

,
zbwd = SiLU
 Conv1D(ﬂip(z))

,
(13)
where ﬂip(·) reverses the sequence along the length dimension.
9

Fig. 5 Bidirectional spatiotemporal scanning mechanism.
Features are then processed by selective scanning in both directions (Fig. 5):
yfwd = SelectiveScan(xfwd),
yfwd ∈RB× dinner
2
×L,
(14)
ybwd = ﬂip
 SelectiveScan(xbwd)

,
ybwd ∈RB× dinner
2
×L.
(15)
The two directions are then summed:
ys = yfwd + ybwd,
zs = zfwd + ﬂip(zbwd).
(16)
To enhance expressiveness, we introduce a dynamic gating mechanism. Gates are
generated from the forward and backward channels and used to adaptively fuse local
features with SSM outputs:
gatef = fcD
 z⊤
fwd

,
gateb = fcDb
 z⊤
bwd

,
(17)
gate = gatef + gateb,
gate ∈RB× dinner
2
×L.
(18)
The gate is applied to fuse with the spatially enhanced features:
yhalf = ys + gate ⊙zfwd,
(19)
where ⊙denotes element-wise multiplication.
The fused features are concatenated with the temporal information zs to restore
the original channel dimension:
ycat = Concat(yhalf, zs),
zcat = Concat(zs, zs),
ycat, zcat ∈RB×dinner×L.
(20)
10

They are rearranged back to (B, L, dinner) and passed to a gated RMS normalization
to improve stability and generalization:
ynorm = RMSNormGated(ycat, zcat).
(21)
Finally, a linear layer maps features back to the model dimension dmodel:
yout = Linear(ynorm),
yout ∈RB×L×dmodel.
(22)
Summarizing the above steps, the complete workﬂow of BSTMamba can be written
as
yout = Linear
 RMSNormGated
 Concat(yhalf, zs), Concat(zs, zs)

.
(23)
4 Experiments
We evaluate our approach on two widely used 3D human pose estimation benchmarks:
Human3.6M [28] and MPI-INF-3DHP [48].
4.1 Datasets and Evaluation Metrics
Human3.6M (H36M) is a large-scale dataset collected in a controlled laboratory
environment. It contains about 3.6 million frames covering 15 typical actions performed
by 11 subjects (6 male and 5 female), such as walking, sitting, taking photos, and
talking on the phone. Each frame is captured simultaneously by eight cameras and is
annotated with high-precision 3D joints (32 in total, with 17 joints commonly used
for evaluation), together with synchronized 2D projections. The data are recorded at
50 fps with consistent clothing, clean backgrounds, and stable lighting, making H36M
suitable for standardized training and evaluation. Consistent with prior work [10,
11, 36], we adopt Protocol #1 (direct evaluation using MPJPE) and Protocol #2
(Procrustes-aligned MPJPE after rigid alignment between predictions and ground
truth), which respectively measure absolute accuracy and structural alignment ability.
MPI-INF-3DHP (3DHP) is a more challenging dataset designed to assess
generalization in complex environments. It includes 8 training subjects and 6 test-
ing subjects spanning daily and athletic activities across diverse indoor and outdoor
scenes. Compared with H36M, 3DHP exhibits greater variation in clothing, lighting,
camera viewpoints, and backgrounds. Annotations are generated for 17 3D joints using
a markerless capture system, and multi-view RGB videos with synchronized 2D pro-
jections are provided. Following prior work [10, 11, 36], we report PCK (Percentage
of Correct Keypoints) and AUC (Area Under Curve) as the primary metrics to quan-
tify accuracy under diﬀerent distance thresholds, and additionally report MPJPE as
a complementary measure.
4.2 Implementation Details
Our model is implemented in PyTorch and trained on an NVIDIA GTX 4090 GPU.
We consider two types of 2D pose sequences as input: (i) 2D predictions produced
11

by a pretrained CPN [23] and (ii) ground-truth 2D keypoints. During training, the
batch size is set to 128 and the network is optimized with Adam [49] for 20 epochs.
The initial learning rate is 0.001 and is decayed by a factor of 0.96 after each epoch.
The number of BSTMamba layers L is treated as a hyper-parameter and adjusted
according to the network scale (e.g., input sequence length).
4.3 Results on Human3.6M
Table 1 Performance comparison under Protocol #1 on Human3.6M using 2D poses detected by
CPN [23] as input. The best results are highlighted in red, and the runner-up results in blue.
Protocol #1
Publication
Dir
Disc
Eat
Greet
Phone
Photo
Pose
Purch
Sit
SitD
Smoke
Wait
WalkD
Walk
WalkT
Avg
RS-Net[50]
TIP’23
44.7
48.4
44.8
49.7
49.6
58.2
47.4
44.8
55.2
59.7
49.3
46.4
51.4
38.6
40.6
48.6
CFI-3DHPE[51]
PRL’2025
45.0
50.3
45.8
48.4
49.7
55.8
47.3
45.4
56.4
59.4
49.9
46.5
50.9
38.0
39.6
48.6
GraphMLP[52]
PR’2024
43.7
49.3
45.5
47.9
50.5
56.0
46.3
44.1
55.9
59.0
48.4
45.7
51.2
37.1
39.1
48.0
GLA-GCN[35]
ICCV’23
41.3
44.3
40.8
41.8
45.9
54.1
42.1
41.5
57.8
62.9
45.0
42.8
45.9
29.4
29.9
44.4
StrideFormer[53]
TMM’22
40.3
43.3
40.2
42.3
45.6
52.3
41.8
40.5
55.9
60.6
44.2
43.0
44.2
30.0
30.2
43.7
MHFormer[10]
CVPR’22
39.2
43.1
40.1
40.9
44.9
51.2
40.6
41.3
53.5
60.3
43.7
41.1
43.8
29.8
30.6
43.0
HSTFormer[54]
arXiv’23
39.5
42.0
39.9
40.8
44.4
50.9
40.9
41.3
54.7
58.8
43.6
40.7
43.4
30.1
30.4
42.7
HDFormer[55]
ICCV’23
38.1
43.1
39.3
39.4
44.3
49.1
41.3
40.8
53.1
62.1
43.3
41.8
43.1
31.0
29.7
42.6
STCFormer[56]
CVPR’23
40.6
43.0
38.3
40.2
43.5
52.6
40.3
40.1
51.8
57.7
42.8
39.8
42.3
28.0
29.5
42.0
Ours (T=9)
—
41.6
47.1
43.4
45.0
48.6
54.1
43.6
44.1
55.1
61.7
46.7
43.1
48.7
32.8
34.5
46.0
Ours (T=27)
—
40.4
45.8
41.7
41.6
46.3
50.8
42.5
40.2
55.0
60.4
44.7
41.3
44.1
29.3
29.9
43.6
Ours (T=81)
—
38.3
41.9
40.3
39.1
44.4
49.1
39.4
39.3
52.3
59.7
42.6
40.0
41.9
28.4
29.0
41.7
Table 2 Performance comparison under Protocol #2 on Human3.6M using 2D poses detected by
CPN [23] as input. The best results are highlighted in red, and the runner-up results in blue.
Protocol #2
Publication
Dir
Disc
Eat
Greet
Phone
Photo
Pose
Purch
Sit
SitD
Smoke
Wait
WalkD
Walk
WalkT
Avg
RS-Net[50]
TIP’23
35.5
38.3
36.1
40.5
39.2
44.8
37.1
34.9
45.0
49.1
40.2
35.4
41.5
31.0
34.3
38.9
SGNN[57]
ICCV’21
33.9
37.2
36.8
38.1
38.7
43.5
37.8
35.0
47.2
53.8
40.7
38.3
41.8
30.1
31.4
39.0
ConvFormer[58]
TVC’24
31.9
34.4
32.2
35.0
34.2
40.7
32.9
31.8
42.8
49.1
36.0
31.5
35.0
23.6
25.2
34.5
CFI-3DHPE[51]
PRL’25
35.5
38.1
35.9
40.4
39.9
43.7
36.0
34.7
46.1
48.4
40.5
35.7
41.3
30.2
33.7
38.7
GraphMLP[52]
PR’24
35.1
38.2
36.5
39.8
39.8
43.5
35.7
34.0
45.6
47.6
39.8
35.1
41.1
30.0
33.4
38.4
StrideFormer[53]
TMM’22
32.7
35.5
32.5
35.4
35.9
41.6
33.0
31.9
45.1
50.1
36.3
33.5
35.1
23.9
25.0
35.2
P-STMO[59]
ECCV’22
31.3
35.2
32.9
33.9
35.4
39.3
32.5
31.5
44.6
48.2
36.3
32.9
34.4
23.9
23.9
34.4
HDFormer[55]
ICCV’23
29.6
33.8
31.7
31.3
33.7
37.7
30.6
31.0
41.4
47.6
35.0
30.9
33.7
25.3
23.6
33.1
STCFormer[56]
CVPR’23
30.4
33.8
31.1
31.7
33.5
39.5
30.8
30.0
41.8
45.8
34.3
30.1
32.8
21.9
23.4
32.7
Ours (T=9)
—
32.9
36.2
34.6
36.2
36.9
41.4
33.6
32.9
45.0
48.8
37.6
32.7
37.3
25.6
28.2
36.0
Ours (T=27)
—
31.7
35.0
32.8
33.5
35.1
40.1
32.8
31.0
43.5
48.0
35.7
31.5
33.8
23.5
24.9
34.2
Ours (T=81)
—
30.3
33.2
32.1
31.6
33.5
38.0
29.7
30.4
42.6
47.2
34.1
30.2
32.0
22.1
23.3
32.7
Table 3 Performance comparison under Protocol #1 on Human3.6M using ground-truth 2D poses
as input. The best results are highlighted in red, and the runner-up results in blue.
Protocol #1
Publication
Dir
Disc
Eat
Greet
Phone
Photo
Pose
Purch
Sit
SitD
Smoke
Wait
WalkD
Walk
WalkT
Avg
GraphMLP[52]
PR’2024
32.2
38.2
29.3
33.4
33.5
38.1
38.2
31.7
37.3
38.5
34.2
36.1
35.5
28.0
29.3
34.2
CFI-3DHPE[51]
PRL’2025
29.1
37.1
29.5
31.8
33.2
41.1
36.0
29.8
38.2
39.3
33.3
36.2
35.8
27.3
28.6
33.7
MHFormer[10]
CVPR’22
27.7
32.1
29.1
28.9
30.0
33.9
33.0
31.2
37.0
39.3
30.0
31.0
29.4
22.2
23.0
30.5
P-STMO[59]
ECCV’22
28.5
30.1
28.6
27.9
29.8
33.2
31.3
27.8
36.0
37.4
29.7
29.5
28.1
21.0
21.0
29.3
GLA-GCN[35]
ICCV’23
26.5
27.2
29.2
25.4
28.2
31.7
29.5
26.9
37.8
39.9
29.9
27.0
27.3
20.5
20.8
28.5
StrideFormer[53]
TMM’22
27.1
29.4
26.5
27.1
28.6
33.0
30.7
26.8
38.2
34.7
29.1
29.8
26.8
19.1
19.8
28.5
HSTFormer[54]
arXiv’23
24.9
27.4
28.1
25.9
28.2
33.5
28.9
26.8
33.4
38.2
27.2
26.7
27.1
20.4
20.8
27.8
STCFormer[56]
CVPR’23
26.2
26.5
23.4
24.6
25.0
28.6
28.3
24.6
30.9
33.7
25.7
25.3
24.6
18.6
19.7
25.7
Ours (T=9)
—
28.9
33.4
27.3
28.4
32.3
32.9
32.9
29.7
36.3
39.1
31.7
29.4
28.4
21.3
23.3
30.4
Ours (T=27)
—
23.3
26.2
23.3
22.7
26.2
26.8
27.0
24.4
31.8
33.1
25.9
24.5
23.7
18.7
19.4
25.1
Ours (T=81)
—
22.3
23.1
20.9
21.1
23.4
24.5
23.4
21.3
28.8
32.3
22.8
21.3
20.4
14.9
16.5
22.5
We
conduct
a
comparative
analysis
against
representative
approaches
on
Human3.6M. To ensure fairness, we report results without additional pretraining data.
Tables 1 and 2 summarize results under Protocol #1 and Protocol #2, respectively.
As shown in Table 1, under Protocol #1 our method achieves strong performance
across diﬀerent input lengths. With 81-frame inputs, BSTMamba attains an MPJPE
12

Table 4 Comparison of MPJPE with diﬀerent input frame numbers on Human3.6M. The best
result in each column is in bold
Method
Frames
Parameters
MACs (G)
MPJPE (mm)
MixSTE[11]
27
33.61M
7.7
45.1
HGMamba-XS[60]
27
2.8M
1.14
44.96
MotionAGFormer-XS[61]
27
2.2M
1.00
45.1
Ours
27
9.85M
4.53
43.6
MixSTE[11]
81
33.61M
23.1
42.7
HGMamba-XS[60]
81
4.8M
6.6
42.5
MotionAGFormer-XS[61]
81
6.1M
8.02
42.84
Ours
81
9.85M
13.57
41.7
of 41.7 mm, which is 0.3 mm and 1.3 mm lower than STCFormer [56] (42.0 mm) and
MHFormer [10] (43.0 mm), respectively. The method performs consistently well for
both short and long sequences, validating the eﬀectiveness of the bidirectional spa-
tiotemporal Mamba design. Under Protocol #2 (Table 2), our approach achieves a
P-MPJPE of 32.7 mm, on par with the best reported result and outperforming other
mainstream baselines.
Furthermore, when using ground-truth 2D poses as input (Table 3), BSTMamba
achieves an MPJPE of 22.5 mm, surpassing all competing approaches in the com-
parison. Notably, BSTMamba achieves the lowest errors on action categories such
as WalkDog, Walking, and WalkTogether, highlighting its adaptability to complex
motions. These gains beneﬁt from the DisruptEnhance module—its random pertur-
bation and residual compensation strengthen robustness to diverse skeletal structures
and noisy 2D inputs—leading to tangible accuracy improvements in realistic 2D
settings.
Table 4 further compares parameters, MACs, and MPJPE under diﬀerent input
lengths. Compared with the previous state-of-the-art MixSTE [11], BSTMamba
improves MPJPE by 1.5 mm while using roughly 55% of its computational cost.
Although the dynamic gating and local convolutional enhancement introduce mod-
est overhead relative to some lightweight models (e.g., HGMamba-XS [60] and
MotionAGFormer-XS [61]), this cost yields substantial performance gains, achieving
a favorable accuracy–eﬃciency trade-oﬀ.
4.4 Results on MPI-INF-3DHP
As shown in Table 5, we evaluate our method on MPI-INF-3DHP with input sequence
lengths of 9 and 27. With 27-frame inputs, our approach achieves a PCK of 98.2%, an
AUC of 79.8%, and an MPJPE of 26.6 mm, outperforming prior methods across all
three metrics.
4.5 Ablation Study
We evaluate the eﬀectiveness of each component in BSTMamba via ablations on
Human3.6M using 27-frame CPN-estimated [23] 2D inputs. As shown in Table 6, a uni-
directional Mamba baseline (without DisruptEnhance) yields an MPJPE of 47.32 mm.
Adding only DisruptEnhance improves MPJPE by 2.71 mm, while adding a bidirec-
tional Mamba with adaptive gating improves it by 1.09 mm. Overall, DisruptEnhance
contributes larger gains than the bidirectional Mamba alone, and combining both
yields the best performance.
13

Table 5 Performance comparison on MPI-INF-3DHP in terms of MPJPE, PCK, and AUC. The
top two results in each column are highlighted in bold.
Method
MPJPE (mm)
PCK
AUC
STCFormer[56]
28.2
98.2
81.5
PoseFormer[9]
57.7
95.4
63.2
MHFormer[10]
58.0
93.8
63.3
MixSTE[11]
54.9
94.4
66.5
P-STMO[59]
32.2
97.9
75.8
STAFFormer[62]
32.1
97.5
77.1
DiﬀPose[63]
29.1
98.0
75.9
HSTFormer[54]
28.3
98.0
78.6
PoseFormerV2[64]
27.8
97.9
78.8
GLA-GCN[35]
27.7
98.5
79.1
Ours (T=9)
30.6
97.8
77.1
Ours (T=27)
26.6
98.2
79.8
Table 6 Contribution of diﬀerent components on Human3.6M (27-frame CPN inputs).
Method
DisruptEnhance
Bidirectional Mamba
MPJPE (mm)↓
Baseline
47.32
+ Bidirectional Mamba
✓
46.23
+ DisruptEnhance
✓
44.61
BSTMamba
✓
✓
43.59
4.6 Visualization Presentation
Fig. 6 Qualitative 3D pose estimates by STCFormer, MotionAGFormer, and our BSTMamba.
We further validate the eﬀectiveness of the proposed model through qualitative visu-
alizations on Human3.6M. Figure 6 depicts two representative actions—Walking and
Sitting—and compares our results with MotionAGFormer [61] and STCFormer [56].
14

In each example, we display both predictions and ground-truth skeletons. BSTMamba
yields more accurate structural reconstructions; for instance, in the challenging Sitting
action, alternative methods may exhibit knee collapse or leg misalignment, whereas
our model preserves natural leg folding. These beneﬁts stem from the bidirectional
global–local spatiotemporal modeling and the dynamic gating enhancement path.
In addition, Figure 6 includes examples on more challenging in-the-wild images.
Even under misdetections or occlusions, BSTMamba produces structurally coherent
and natural-looking 3D poses, demonstrating strong robustness. Overall, these visual-
izations corroborate our quantitative results, showing high accuracy, robustness, and
structural consistency across action types and input qualities.
5 Conclusion
To mitigate the limitations of Transformer-based methods for 3D human pose esti-
mation—namely the quadratic cost of self-attention, ineﬃciency in modeling complex
spatiotemporal relations, and constrained (often unidirectional) dependency model-
ing—we propose BSTMamba. The architecture comprises two key components: (i)
DisruptEnhance, and (ii) a bidirectional global–local spatiotemporal Mamba block.
The bidirectional global–local mechanism leverages the eﬃciency of state-space
models (SSMs) for sequence modeling while injecting local convolution and dynamic
gating to more eﬀectively capture long- and short-range dependencies. This design
substantially improves both accuracy and robustness. Meanwhile, DisruptEnhance
randomly perturbs joint orders and employs a lightweight residual compensation sub-
network, strengthening generalization across subjects and diverse skeletal structures.
Extensive experiments show that BSTMamba outperforms state-of-the-art meth-
ods on Human3.6M and MPI-INF-3DHP in terms of MPJPE, PCK, and AUC, and
maintains strong performance across varying input sequence lengths and diﬀerent 2D
pose sources. Ablation studies validate the contributions of both the bidirectional
Mamba block and DisruptEnhance, with the latter providing a larger standalone gain;
their combination yields further improvements. Qualitative visualizations corrobo-
rate these ﬁndings, highlighting accurate pose reconstruction and strong structural
consistency, and conﬁrming the eﬀectiveness of the proposed model and its core
components.
References
[1] Peng, K., Yin, C., Zheng, J., Liu, R., Schneider, D., Zhang, J., Yang, K., Sarfraz,
M.S., Stiefelhagen, R., Roitberg, A.: Navigating open set scenarios for skeleton-
based action recognition. In: Proceedings of the AAAI Conference on Artiﬁcial
Intelligence, vol. 38, pp. 4487–4496 (2024)
[2] Dittakavi, B., Bavikadi, D., Desai, S.V., Chakraborty, S., Reddy, N., Balasubra-
manian, V.N., Callepalli, B., Sharma, A.: Pose tutor: an explainable system for
pose correction in the wild. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 3540–3549 (2022)
15

[3] Yuan, Y., Makoviychuk, V., Guo, Y., Fidler, S., Peng, X., Fatahalian, K.: Learning
physically simulated tennis skills from broadcast videos. ACM Trans. Graph 42(4)
(2023)
[4] Liu, R., Shen, J., Wang, H., Chen, C., Cheung, S.-c., Asari, V.: Attention mech-
anism exploits temporal contexts: Real-time 3d human pose reconstruction. In:
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 5064–5073 (2020)
[5] Chen, T., Fang, C., Shen, X., Zhu, Y., Chen, Z., Luo, J.: Anatomy-aware 3d
human pose estimation with bone-based pose decomposition. IEEE Transactions
on Circuits and Systems for Video Technology 32(1), 198–209 (2021)
[6] Zeng, A., Sun, X., Huang, F., Liu, M., Xu, Q., Lin, S.: Srnet: Improving gener-
alization in 3d human pose estimation with a split-and-recombine approach. In:
European Conference on Computer Vision, pp. 507–523 (2020). Springer
[7] Wang, J., Yan, S., Xiong, Y., Lin, D.: Motion guided 3d pose estimation from
videos. In: European Conference on Computer Vision, pp. 764–780 (2020).
Springer
[8] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N.,
Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural
information processing systems 30 (2017)
[9] Zheng, C., Zhu, S., Mendieta, M., Yang, T., Chen, C., Ding, Z.: 3d human
pose estimation with spatial and temporal transformers. In: Proceedings of
the IEEE/CVF International Conference on Computer Vision, pp. 11656–11665
(2021)
[10] Li, W., Liu, H., Tang, H., Wang, P., Van Gool, L.: Mhformer: Multi-hypothesis
transformer for 3d human pose estimation. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 13147–13156
(2022)
[11] Zhang, J., Tu, Z., Yang, J., Chen, Y., Yuan, J.: Mixste: Seq2seq mixed spatio-
temporal encoder for 3d human pose estimation in video. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13232–
13242 (2022)
[12] Ma, H., Wang, Z., Chen, Y., Kong, D., Chen, L., Liu, X., Yan, X., Tang, H., Xie,
X.: Ppt: token-pruned pose transformer for monocular and multi-view human pose
estimation. In: European Conference on Computer Vision, pp. 424–442 (2022).
Springer
[13] Li, W., Liu, M., Liu, H., Wang, P., Cai, J., Sebe, N.: Hourglass tokenizer
for eﬃcient transformer-based 3d human pose estimation. In: Proceedings of
16

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
604–613 (2024)
[14] Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state
spaces. arXiv preprint arXiv:2312.00752 (2023)
[15] Wang, J., Zhu, W., Wang, P., Yu, X., Liu, L., Omar, M., Hamid, R.: Selective
structured state-spaces for long-form video understanding. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6387–
6397 (2023)
[16] Islam, M.M., Bertasius, G.: Long movie clip classiﬁcation with state-space
video models. In: European Conference on Computer Vision, pp. 87–104 (2022).
Springer
[17] Gu, A., Goel, K., R´e, C.: Eﬃciently modeling long sequences with structured
state spaces. arXiv preprint arXiv:2111.00396 (2021)
[18] Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., Wang, Y.: Motionbert: A uniﬁed
perspective on learning human motion representations. In: Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15085–
15099 (2023)
[19] Qiu, H., Wang, C., Wang, J., Wang, N., Zeng, W.: Cross view fusion for 3d human
pose estimation. In: Proceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 4342–4351 (2019)
[20] Zhang, L., Zhou, K., Lu, F., Zhou, X.-D., Shi, Y.: Deep semantic graph trans-
former for multi-view 3d human pose estimation. In: Proceedings of the AAAI
Conference on Artiﬁcial Intelligence, vol. 38, pp. 7205–7214 (2024)
[21] Zhang, X., Cui, Q., Bao, Q., Yang, W., Liao, Q.: Geometry-guided diﬀusion model
with masked transformer for robust multi-view 3d human pose estimation. In:
Proceedings of the 32nd ACM International Conference on Multimedia, pp. 681–
690 (2024)
[22] Newell, A., Yang, K., Deng, J.: Stacked hourglass networks for human pose
estimation. In: European Conference on Computer Vision, pp. 483–499 (2016).
Springer
[23] Chen, Y., Wang, Z., Peng, Y., Zhang, Z., Yu, G., Sun, J.: Cascaded pyramid
network for multi-person pose estimation. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pp. 7103–7112 (2018)
[24] Mondal, A., Alletto, S., Tome, D.: Hummuss: Human motion understanding using
state space models. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 2318–2330 (2024)
17

[25] Hagbi, N., Bergig, O., El-Sana, J., Billinghurst, M.: Shape recognition and pose
estimation for mobile augmented reality. IEEE transactions on visualization and
computer graphics 17(10), 1369–1379