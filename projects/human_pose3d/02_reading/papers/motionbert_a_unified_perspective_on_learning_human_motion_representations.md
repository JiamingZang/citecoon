# MotionBERT: A Unified Perspective on Learning Human Motion Representations

> 2023 · id: W4390874423 · arXiv: 2210.06551 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present a unified perspective on tackling various
human-centric video tasks by learning human motion rep-
resentations from large-scale and heterogeneous data re-
sources. Specifically, we propose a pretraining stage in
which a motion encoder is trained to recover the underly-
ing 3D motion from noisy partial 2D observations. The
motion representations acquired in this way incorporate
geometric, kinematic, and physical knowledge about hu-
man motion, which can be easily transferred to multiple
downstream tasks. We implement the motion encoder with
a Dual-stream Spatio-temporal Transformer (DSTformer)
neural network. It could capture long-range spatio-temporal
relationships among the skeletal joints comprehensively and
adaptively, exemplified by the lowest 3D pose estimation
error so far when trained from scratch. Furthermore, our
proposed framework achieves state-of-the-art performance
on all three downstream tasks by simply finetuning the pre-
trained motion encoder with a simple regression head (1-2
layers), which demonstrates the versatility of the learned
motion representations. Code and models are available at
https://motionbert.github.io/

## introduction
As discussed in Section 1, our approach consists of two
stages, namely unified pretraining and task-specific fine-
tuning. In the first stage, we train a motion encoder to
accomplish the 2D-to-3D lifting task, where we use the pro-
posed DSTformer as the backbone. In the second stage,
we finetune the pretrained motion encoder and a few new
layers on the downstream tasks. We use 2D skeleton se-
quences as input for both pretraining and finetuning because
they could be reliably extracted from all kinds of motion
sources [3, 10, 76, 86, 103], and is more robust to varia-
tions [19,32]. Existing studies have shown the effectiveness
of using 2D skeleton sequences for different downstream
tasks [25,32,89,109]. We will first introduce the architec-
ture of DSTformer, and then describe the training scheme in
detail.
3.2. Network Architecture
Figure 2 shows the network architecture for 2D-to-3D
lifting. Given an input 2D skeleton sequence x ∈RT ×J×Cin,
we first project it to a high-dimensional feature F0 ∈
RT ×J×Cf, then add learnable spatial positional encoding
PS
pos ∈R1×J×Cf and temporal positional encoding PT
pos ∈
RT ×1×Cf to it. We then use the sequence-to-sequence model
DSTformer to calculate Fi ∈RT ×J×Cf (i = 1, . . . , N)
where N is the network depth. We apply a linear layer with
tanh activation [30] to FN to compute the motion represen-
tation E ∈RT ×J×Ce. Finally, we apply a linear transfor-
mation to E to estimate 3D motion ˆX ∈RT ×J×Cout. Here,
T denotes the sequence length, and J denotes the number
of body joints. Cin, Cf, Ce, and Cout denote the channel
numbers of input, feature, embedding, and output respec-
tively. We first introduce the basic building blocks of DST-
former, i.e. Spatial and Temporal Blocks with Multi-Head
Self-Attention (MHSA), and then explain the DSTformer
architecture design.
Spatial Block.
Spatial MHSA (S-MHSA) aims at model-
ing the relationship among the joints within the same time
step. It is defined as
S-MHSA(QS, KS, VS) = [head1; ...; headh]WP
S ,
headi = softmax(Qi
S(Ki
S)′
√dK
)Vi
S,
(1)
where WP
S is a projection parameter matrix, h is the number
of the heads, i ∈1, . . . , h, and ′ denotes matrix transpose.
We utilize self-attention to get the query QS, key KS, and
value VS from input per-frame spatial feature FS ∈RJ×Ce
for each headi,
Qi
S = FSW(Q,i)
S
, Ki
S = FSW(K,i)
S
, Vi
S = FSW(V,i)
S
,
(2)
where W(Q,i)
S
, W(K,i)
S
, W(V,i)
S
are projection matrices, and
dK is the feature dimension of KS. We apply S-MHSA to
3

features of different time steps in parallel. Residual connec-
tion and layer normalization (LayerNorm) are used to the
S-MHSA result, which is further fed into a multilayer per-
ceptron (MLP), and followed by a residual connection and
LayerNorm following [112]. We denote the entire spatial
block with MHSA, LayerNorm, MLP, and residual connec-
tions by S.
Temporal Block.
Temporal MHSA (T-MHSA) aims at
modeling the relationship across the time steps for a body
joint. Its computation process is similar with S-MHSA ex-
cept that the MHSA is applied to the per-joint temporal
feature FT ∈RT ×Ce and parallelized over the spatial dimen-
sion.
T-MHSA(QT, KT, VT) = [head1; ...; headh]WP
T ,
headi = softmax(Qi
T(Ki
T)′
√dK
)Vi
T,
(3)
where i ∈1, . . . , h, QT, KT, VT are computed similar with
Formula 2. We denote the entire temporal block by T .
Dual-stream Spatio-temporal Transformer.
Given spa-
tial and temporal MHSA that captures the intra-frame and
inter-frame body joint interactions respectively, we assemble
the basic building blocks to fuse the spatial and temporal
information in the flow. We design a dual-stream architec-
ture with the following assumptions: 1) Both streams should
be capable of modeling the comprehensive spatio-temporal
context. 2) Each stream should be specialized in different
spatio-temporal aspects. 3) The two streams should be fused
together, with the fusion weights dynamically balanced de-
pending on the input spatio-temporal characteristics.
Hence, we stack the spatial and temporal MHSA blocks in
different orders, forming two parallel computation branches.
The output features of the two branches are fused using
adaptive weights predicted by an attention regressor. The
dual-stream-fusion module is then repeated for N times:
Fi = αi
ST◦T i
1 (Si
1(Fi−1))+αi
TS◦Si
2(T i
2 (Fi−1)),
i ∈1, . . . , N,
(4)
where Fi denotes the feature embedding at depth i, ◦denotes
element-wise production. Orders of S and T blocks are
shown in Figure 2, and different blocks do not share weights.
Adaptive fusion weights αST, αTS ∈RN×T ×J are given by
αi
ST, αi
TS = softmax(W([T i
1 (Si
1(Fi−1)), Si
2(T i
2 (Fi−1))])),
(5)
where W is a learnable linear transformation. [, ] denotes
concatenation.
3.3. Unified Pretraining
We address two key challenges when designing the uni-
fied pretraining framework: 1) How to learn a powerful
motion representation with a universal pretext task. 2) How
to utilize large-scale but heterogeneous human motion data
in all kinds of formats.
For the first challenge, we follow the successful practices
in language [12,30,90] and vision [7,36] modeling to con-
struct the supervision signals, i.e. mask part of the input and
use the encoded representations to reconstruct the whole
input. Note that such “cloze” task naturally exists in human
motion analysis, that is to recover the lost depth information
from the 2D visual observations, i.e. 3D human pose estima-
tion. Inspired by this, we leverage the large-scale 3D mocap
data [76] and design a 2D-to-3D lifting pretext task. We first
extract the 2D skeleton sequences x by projecting the 3D
motion orthographically. Then, we corrupt x by randomly
masking and adding noise to produce the corrupted 2D skele-
ton sequences, which also resemble the 2D detection results
as it contains occlusions, detection failures, and errors. Both
joint-level and frame-level masks are applied with certain
probabilities. We use the aforementioned motion encoder
to get motion representation E and reconstruct 3D motion
ˆX. We then compute the joint loss L3D between ˆX and GT
3D motion X. We also add the velocity loss LO following
previous works [89,134]. The 3D reconstruction losses are
thus given by
L3D =
T
X
t=1
J
X
j=1
∥ˆXt,j−Xt,j ∥2,
LO =
T
X
t=2
J
X
j=1
∥ˆOt,j−Ot,j ∥2,
(6)
where ˆOt = ˆXt −ˆXt−1, Ot = Xt −Xt−1.
For the second challenge, we notice that 2D skeletons
could serve as a universal medium as they can be extracted
from all sorts of motion data sources. We further incorporate
in-the-wild RGB videos into the 2D-to-3D lifting framework
for unified pretraining. For RGB videos, the 2D skeletons
x could be given by manual annotation [3] or 2D pose es-
timators [14, 103], and the depth channel of the extracted
2D skeletons is intrinsically “masked”. Similarly, we add
extra masks and noises to degrade x (if x already contains
detection noise, only masking is applied). As 3D motion GT
X is not available for these data, we apply a weighted 2D
re-projection loss which is calculated by
L2D =
T
X
t=1
J
X
j=1
δt,j∥ˆxt,j −xt,j∥2,
(7)
where ˆx is the 2D orthographical projection of the estimated
3D motion ˆX, and δ ∈RT ×J is given by visibility annota-
tion or 2D detection confidence.
The total pretraining loss is computed by
L = L3D + λOLO
|
{z
}
for 3D data
+
L2D
|{z}
for 2D data
,
(8)
where λO is a constant coefficient to balance the losses.
4

3.4. Task-specific Finetuning
The learned feature embedding E serves as a 3D-aware
and temporal-aware human motion representation. For down-
stream tasks, we adopt the minimalist design principle, i.e.
implementing a shallow downstream network and training
without bells and whistles. In practice, we use an extra linear
layer or an MLP with one hidden layer. We then finetune the
whole network end-to-end.
3D Pose Estimation.
As we utilize 2D-to-3D lifting as the
pretext task, we simply reuse the whole pretrained network.
During finetuning, the input 2D skeletons are estimated from
videos without extra masks or noises.
Skeleton-based Action Recognition.
We directly apply a
global average pooling over different persons and timesteps.
The result is then fed into an MLP with one hidden layer. The
network is trained with cross-entropy classification loss. For
one-shot learning, we apply a linear layer after the pooled
features to extract clip-level action representation. We intro-
duce the detailed setup of one-shot learning in Section 4.4.
Human Mesh Recovery.
We use SMPL [71] model to
represent the human mesh and regress its parameters. The
SMPL model consists of pose parameters θ ∈R72 and
shape parameters β ∈R10, and calculates the 3D mesh as
M(θ, β) ∈R6890×3. To regress the pose parameters for
each frame, we feed the motion embeddings E to an MLP
with one hidden layer and get ˆθ ∈RT ×72. To estimate
shape parameters, considering that the human shape over a
video sequence is supposed to be consistent, we first perform
an average pooling of E over the temporal dimension and
then feed it into another ML

## method
Input
T
Human3.6M
3DPW
MPVE↓
MPJPE↓
PA-MPJPE↓
MPVE↓
MPJPE↓
PA-MPJPE↓
HMR [41] CVPR’18
image
1
-
88.0
56.8
-
130.0
81.3
† SPIN [48] ICCV’19
image
1
82.3
59.4
39.3
129.1
100.9
59.1
Pose2Mesh [25] ECCV’20
2D pose
1
85.3
64.9
48.7
109.3
91.4
60.1
I2L-MeshNet [83] ECCV’20
image
1
-
55.7
41.7
110.1
93.2
58.6
† HybrIK [54] CVPR’21
image
1
58.1
47.4
30.1
82.4
71.3
41.9
METRO [58] CVPR’21
image
1
-
54.0
36.7
88.2
77.1
47.9
Mesh Graphormer [59] ICCV’21
image
1
-
51.2
34.5
87.7
74.7
45.6
PARE [47] ICCV’21
image
1
-
-
-
88.6
74.5
46.5
ROMP [105] ICCV’21
image
1
-
-
-
108.3
91.3
54.9
PyMAF [133] ICCV’21
image
1
-
57.7
40.5
110.1
92.8
58.9
ProHMR [50] ICCV’21
image
1
-
-
41.2
-
-
59.8
OCHMR [43] CVPR’22
image
1
-
-
-
107.1
89.7
58.3
3DCrowdNet [26] CVPR’22
image
1
-
-
-
98.3
81.7
51.5
CLIFF [57] ECCV’22
image
1
-
47.1
32.7
81.2
69.0
43.0
FastMETRO [23] ECCV’22
image
1
-
52.2
33.7
84.1
73.5
44.6
VisDB [128] ECCV’22
image
1
-
51.0
34.5
85.5
73.5
44.9
TemporalContext [6] CVPR’19
video
32
-
77.8
54.3
-
-
72.2
HMMR [42] CVPR’19
video
20
-
-
56.9
139.3
116.5
72.6
DSD-SATN [106] ICCV’19
video
9
-
59.1
42.4
-
-
69.5
VIBE [46] CVPR’20
video
16
-
65.6
41.4
99.1
82.9
51.9
TCMR [24] CVPR’21
video
16
-
62.3
41.1
102.9
86.5
52.7
† MAED [114] ICCV’21
video
16
84.1
60.4
38.3
93.3
79.0
45.7
MPS-Net [120] CVPR’22
video
16
-
69.4
47.4
99.7
84.3
52.1
∗PoseBERT [8] TPAMI’22 (+SPIN [48])
video
16
-
-
-
-
-
57.3 ↓2.3
∗SmoothNet [130] ECCV’22 (+SPIN [48])
video
32
-
67.5 ↓1.0
46.3 ↓0.2
-
86.7 ↓0.9
52.7 ↓0.6
Ours (scratch)
2D motion
16
75.7
62.8
41.0
99.1
85.5
50.2
Ours (finetune)
2D motion
16
65.5
53.8
34.9
88.1
76.9
47.2
Ours (finetune) + SPIN [48]
video
16
63.7 ↓18.6
52.2 ↓7.2
35.7 ↓3.6
92.8 ↓36.3
79.6 ↓21.3
48.2 ↓10.9
Ours (finetune) + MAED [114]
video
16
66.8 ↓17.3
54.8 ↓5.6
36.4 ↓1.9
84.4 ↓8.9
72.3 ↓6.7
42.3 ↓3.4
Ours (finetune) + HybrIK [54]
video
16
52.6 ↓5.5
43.1 ↓4.3
27.8 ↓2.3
79.4 ↓3.0
68.8 ↓2.5
40.6 ↓1.3
Table 3. Quantitative comparison of human mesh recovery on Human3.6M and 3DPW datasets. T denotes the clip length used by the
method. † denotes the results obtained with official model weights. The rest are all officially reported results. The gains in ∗correspond to
different re-implemented SPIN [48] results.
using only 1 labeled video for each class. The auxiliary
set contains the other 100 classes, and all samples of these
classes can be used. We train the model on the auxiliary set
using the supervised contrastive learning technique [44]. For
a batch of auxiliary data, samples of the same class are pulled
together, while samples of different classes are pushed away
in the action embedding space. During the evaluation, we
calculate the cosine distance between the test examples and
the exemplars, and use 1-nearest neighbor to determine the
class. Table 2 (right) illustrates that the proposed models out-
perform state-of-the-art by a considerable margin. Moreover,
it is noteworthy that our pretrained model achieves optimal
performance with only 1-2 epochs of fine-tuning. Our results
indicate that the pretraining stage is effective in learning a
robust motion representation that generalizes well to novel
downstream tasks, even with limited data annotations.
4.5. Human Mesh Recovery
We conduct experiments on Human3.6M [38] and 3DPW
[113] datasets and additionally add COCO [61] dataset dur-
ing training following [46,58,114]. We keep the same train-
ing and test split for both datasets as in [78] (Section 4.2)
7

Figure 3. Learning curves of finetuning and training from scratch.
Backbone
MPJPE ↓
MPVE ↓
Accuracy ↑
Accuracy↑
(frozen)
(3D pose)
(mesh)
(action x-view)
(action 1-shot)
Random
404.4mm
114.4mm
47.6%
46.8%
Pretrained
40.3mm
72.1mm
87.3%
60.7%
Table 4. Comparison of partial finetuning.
and [46,58,114], respectively. Following the common prac-
tice [41,46,49,114], we report MPJPE (mm) and PA-MPJPE
(mm) of 14 joints obtained by JM(θ, β). PA-MPJPE cal-
culates MPJPE after aligning with GT in translation, ro-
tation, and scale. We further report the mean per vertex
error (MPVE) (mm) of the mesh M(θ, β), which measures
the average distance between the estimated and GT ver-
tices after aligning the root joint. Note that most previous
works [24,41,46,48,54,58,73,114] use more datasets other
than COCO [61] during training, such as LSP [39], MPI-INF-
3DHP [79], etc., while we do not. Table 3 demonstrates that
our finetuned model delivers competitive results on both Hu-
man3.6M and 3DPW datasets, surpassing all the state-of-the-
art video-based methods, including MAED [114], especially
on the MPVE error. Nonetheless, we note that estimating
full-body mesh from sparse 2D keypoints alone [10,25] is an
ill-posed problem because it lacks human shape information.
In light of this, we propose a hybrid approach that lever-
ages the strengths of both our framework (coherent motion)
and RGB-based methods (accurate shape). We introduce
a refiner module that can be easily integrated with existing
image/video-based methods, similar to [8,130]. Specifically,
our refiner module is an MLP that takes the combination of
our pretrained motion representations and an initial predic-
tion, regressing a residual in joint rotations. Our approach
effectively improves the state-of-the-art methods [48,54,114]
and achieves the lowest error to date.
4.6. Ablation Studies
Finetune vs. Scratch.
We compare the training progress
of finetuning the pretrained model and training from scratch.
As Figure
3 shows, models initialized with pretrained
weights demonstrate superior performance and faster conver-
gence on all three tasks. This observation suggests that the
pretrained model learns transferable knowledge about hu-
Pretrain
Noise
Mask
2D
MPJPE↓
MPVE↓
Accuracy↑
(3D pose)
(mesh)
(action x-sub)
-
-
-
-
39.2mm
75.7mm
87.7%
✓
-
-
-
38.8mm
70.6mm
89.4%
✓
✓
-
-
38.1mm
68.4mm
90.7%
✓
✓
✓
-
37.4mm
67.8mm
91.9%
✓
✓
✓
✓
37.5mm
65.5mm
93.0%
Table 5. Comparison of pretraining strategies.
man motion, facilitating the learning of multiple downstream
tasks.
Partial Finetuning.
In addition to end-to-end finetuning,
we freeze the motion encoder backbone and only train the
regression head for each downstream task. To verify the
effectiveness of the pretrained motion representations, we
compared the pretrained motion encoder with a randomly
initialized motion encoder. We report results of 3D pose and
mesh on Human3.6M, action on NTU-RGB+D and NTU-
RGB+D-120 (same for the tables below). It can be seen in
Table 4 that based on the frozen pretrained motion representa-
tions, our method still achieves competitive performance on
multiple downstream tasks and shows a large improvement
compared to the baseline. Pretraining and partial finetuning
make it possible for all the downstream tasks to share the
same backbone, significantly reducing computation over-
head for applications requiring multi-task inference.
Pretraining Strategies.
We evaluate how different pre-
training strategies influence the performance of downstream
tasks. Starting from the scratch baseline, we apply the pro-
posed strategies one by one. As shown in Table 5, a vanilla
2D-to-3D pretraining stage brings benefits to all the down-
stream tasks. Introducing corruptions additionally improves
the learned motion embeddings. Unified pretraining with
in-the-wild videos (w. 2D) enjoys higher motion diversity,
which further helps several downstream tasks.
Pretraining with Different Backbones.
We further study
the universality of the proposed pretraining approach. We
8

Setting
MPJPE ↓
MPVE ↓
Accuracy ↑
Accuracy↑
(3D pose)
(mesh)
(action x-view)
(action 1-shot)
TCN (scratch)
50.1mm
92.6mm
91.5%
52.4%
TCN (finetune)
47.9mm
86.3mm
92.8%
59.9%
PoseFormer (scratch)
44.8mm
85.9mm
94.2%
57.4%
PoseFormer (finetune)
41.5mm
80.5mm
95.9%
60.7%
Table 6. Comparison of different backbones.
Arch.
(a)
(b)
(c)
(d)
(e)
(f)
Design
S-T
T-S
S + T
ST-MHSA
S-T + T-S
S-T + T-S
(Average)
(Adaptive)
MPJPE ↓40.58±0.31 41.05±0.24 41.76±0.22 41.54±0.35 39.87±0.32 39.25±0.27
Table 7. Comparison of model architecture variants. All the
methods are trained on Human3.6M from scratch over 5 runs and
measured by MPJPE (mm) with mean and standard deviation.
replace the motion encoder backbone with two variants:
TCN [89] and PoseFormer [135]. The models are slightly
modified to a seq2seq version, while all the configurations
for pretraining and finetuning are simply followed. Table 6
shows that the proposed approach consistently benefits dif-
ferent backbone models on different tasks.
Model Architecture.
Finally, we study the design choices
of DSTformer. From (a) to (f) in Table 8, we compare
different structure designs of the basic Transformer mod-
ule. (a) and (b) are single-stream versions with different
orders. (a) is conceptually similar to PoseFormer [135], MH-
Former [56], and MixSTE [134]. (c) limits each stream to
either temporal or spatial modeling before fusion and is sim-
ilar to MAED [114]. (d) directly connects S-MHSA and
T-MHSA without the MLP in between and is similar to the

## experiments
4.1. Implementation
We implement the proposed motion encoder DSTformer
with depth N = 5, number of heads h = 8, feature size
Cf = 512, embedding size Ce = 512. For pretraining, we
use sequence length T = 243. The pretrained model could
handle different input lengths thanks to the Transformer-
based backbone. During finetuning, we set the backbone
learning rate to be 0.1× of the new layer learning rate. We
introduce the experiment datasets in the following sections
respectively. Please refer to the appendix for more experi-
mental details.
4.2. Pretraining
We collect diverse and realistic 3D human motion from
two datasets, Human3.6M [38] and AMASS [76].
Hu-
man3.6M [38] is a commonly used indoor dataset for
3D human pose estimation which contains 3.6 million
video frames of professional actors performing daily ac-
tions. Following previous works [78, 89], we use subjects
1, 5, 6, 7, 8 for training, and subjects 9, 11 for testing.
AMASS [76] integrates most existing marker-based Mocap
datasets [1,2,5,11,15,18,34,37,52,69,72,77,84,97,110,111]
and parameterizes them with a common representation. We
do not use the images or 2D detection results of the two
datasets during pretraining as Mocap datasets usually do not
provide raw videos. Instead, we use orthographic projec-
tion to get the uncorrupted 2D skeletons. We further incor-
porate two in-the-wild RGB video datasets PoseTrack [3]
(annotated) and InstaVariety [42] (unannotated) for higher
motion diversity. We align the body keypoint definitions
with Human3.6M and calibrate the camera coordinates to
pixel coordinates following [27]. We randomly zero out 15%
joints, and sample noises from a mixture of Gaussian and
uniform distributions [17]. We first train on 3D data only
for 30 epochs, then train on both 3D data and 2D data for 60
epochs, following the curriculum learning practices [9,118].
4.3. 3D Pose Estimation
We evaluate the 3D pose estimation performance on Hu-
man3.6M [38] and report the mean per joint position er-
ror (MPJPE) in millimeters, which measures the average
distance between the predicted joint positions and the GT
after aligning the root joint. We also compute the mean
per-joint velocity error (MPJVE) to evalute the temporal
smoothness following previous works [134,135]. We use
the Stacked Hourglass (SH) networks [86] to extract the
2D skeletons from videos, and finetune the entire network
on Human3.6M [38] training set. In addition, we train a
separate model of the same architecture, but with random
initialization rather than pretrained weights. As shown in
Table 1 (top), the model trained from scratch outperforms
5

## related_work
Learning Human Motion Representations.
Early works
formulate human motion with Hidden Markov Models [53,
108] and graphical models [51, 99]. Kanazawa et al. [42]
design a temporal encoder and a hallucinator to learn rep-
resentations of 3D human dynamics. Zhang et al. [132]
predict future 3D dynamics in a self-supervised manner.
Sun et al. [102] further incorporate action labels with an
action memory bank. From the action recognition perspec-
tive, a variety of pretext tasks are designed to learn mo-
tion representations in a self-supervised manner, includ-
ing future prediction [100], jigsaw puzzle [60], skeleton-
contrastive [107], speed change [101], cross-view con-
sistency [62], and contrast-reconstruction [117]. Similar
techniques are also explored in tasks like motion assess-
ment [33,85] and motion retargeting [126,139]. These meth-
ods leverage homogeneous motion data, design correspond-
ing pretext tasks, and apply them to a specific downstream
task. In this work, we propose a unified pretrain-finetune
framework to incorporate heterogeneous data resources and
demonstrate its versatility in various downstream tasks.
3D Human Pose Estimation.
Recovering 3D human
poses from monocular RGB videos is a classical problem,
and the methods can be categorized into two categories.
The first is to estimate 3D poses with CNN directly from
images [82, 104, 136]. However, one limitation of these
approaches is that there is a trade-off between 3D pose
precision and appearance diversity due to current data col-
lection techniques. The second category is to extract the
2D pose first, then lift the estimated 2D pose to 3D with
a separate neural network. The lifting can be achieved via
Fully Connected Network [29,78], Temporal Convolutional
Network (TCN) [22, 89], GCN [13, 28, 116], and Trans-
former [56,94,134,135]. Our framework is built upon the
second category as we use the proposed DSTformer to ac-
complish 2D-to-3D lifting.
Skeleton-based Action Recognition.
The pioneering
works [74, 115, 127] point out the inherent connection be-
tween action recognition and human pose estimation. To-
wards modeling the spatio-temporal relationship among hu-
man joints, previous studies mainly employ LSTM [98,138]
and GCN [21, 55, 68, 96, 123].
Most recently, PoseC-
onv3D [32] proposes to apply 3D-CNN on the stacked 2D
joint heatmaps and achieves improved results. In addition to
the fully-supervised action recognition task, NTU-RGB+D-
120 [64] brings attention to the challenging one-shot action
recognition problem. To this end, SL-DML [81] applies deep
metric learning to multi-modal signals. Sabater et al. [92]
explores one-shot recognition in therapy scenarios with TCN.
We demonstrate that the pretrained motion representations
could generalize well to action recognition tasks, and the
pretrain-finetune framework is a suitable solution for the
one-shot challenges.
Human Mesh Recovery.
Based on the parametric human
models such as SMPL [71], many research works [41,75,83,
122,133] focus on regressing the human mesh from a single
image. SPIN [48] additionally incorporates fitting the body
2

Spatial MHSA
Temporal MHSA
Add + Norm + MLP
Adaptive Fusion
Spatial Pos. Encoding
Temporal Pos. Encoding
N ×
Spatial MHSA
Temporal MHSA
Temporal MHSA
Spatial MHSA
2D Skeletons
3D Motion
DSTformer
FC
S1
T1
T2
S2
ˆX
x
αTS
αST
E
FC
FC
Figure 2. Model architecture. We propose the Dual-stream Spatio-temporal Transformer (DSTformer) as a general backbone for human
motion modeling. DSTformer consists of N dual-stream-fusion modules. Each module contains two branches of spatial or temporal MHSA
and MLP. The Spatial MHSA models the connection among different joints within a timestep, while the Temporal MHSA models the
movement of one joint.
model to 2D joints in the training loop. Despite their promis-
ing per-frame results, these methods yield jittery and unsta-
ble results [46,130] when applied to videos. To improve their
temporal coherence, PoseBERT [8] and SmoothNet [130]
propose to employ a denoising and smoothing module to the
single-frame predictions. Several works [24,42,46,106] take
video clips as input to exploit the temporal cues. Another
common problem is that paired images and GT meshes are
mostly captured in constrained scenarios, which limits the
generalization ability of the above methods. To that end,
Pose2Mesh [25] proposes to first extract 2D skeletons using
an off-the-shelf pose estimator, then lift them to 3D mesh
vertices. Our approach is complementary to state-of-the-art
human mesh recovery methods and could further improve
their temporal coherence with the pretrained motion repre-
sentations.

## conclusion
In this work, we provide a unified perspective to tackling
various human-centric video tasks. We develop a pretrain-
ing approach to learn human motion representations from
large-scale and heterogeneous data sources. We also propose
DSTformer as a universal human motion encoder. Experi-
mental results on multiple benchmarks demonstrate the ver-
satility of the learned motion representations. Future work
could explore fusing the learned motion representations with
generic video architectures as a human-centric semantic fea-
ture and applying it to more tasks (e.g., action assessment,
segmentation).
Acknowledgement This work was partially supported by
MOST-2022ZD0114900. We would like to thank Hai Ci and
Jiefeng Li for their exceptional support.
9