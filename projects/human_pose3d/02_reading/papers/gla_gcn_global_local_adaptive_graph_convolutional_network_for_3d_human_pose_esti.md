# GLA-GCN: Global-local Adaptive Graph Convolutional Network for 3D Human Pose Estimation from Monocular Video

> 2023 · id: W4390873166 · arXiv: 2307.05853 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
3D human pose estimation has been researched for
decades with promising fruits. 3D human pose lifting is one
of the promising research directions toward the task where
both estimated pose and ground truth pose data are used
for training. Existing pose lifting works mainly focus on
improving the performance of estimated pose, but they usu-
ally underperform when testing on the ground truth pose
data. We observe that the performance of the estimated
pose can be easily improved by preparing good quality 2D
pose, such as fine-tuning the 2D pose or using advanced
2D pose detectors. As such, we concentrate on improving
the 3D human pose lifting via ground truth data for the fu-
ture improvement of more quality estimated pose data. To-
wards this goal, a simple yet effective model called Global-
local Adaptive Graph Convolutional Network (GLA-GCN)
is proposed in this work. Our GLA-GCN globally mod-
els the spatiotemporal structure via a graph representation
and backtraces local joint features for 3D human pose es-
timation via individually connected layers. To validate our
model design, we conduct extensive experiments on three
benchmark datasets: Human3.6M, HumanEva-I, and MPI-
INF-3DHP. Experimental results show that our GLA-GCN1
implemented with ground truth 2D poses significantly out-
performs state-of-the-art methods (e.g., up to 3%, 17%, and
14% error reductions on Human3.6M, HumanEva-I, and
MPI-INF-3DHP, respectively).

## introduction
3D Human Pose Estimation (HPE) in videos aims to
predict the pose joint locations of the human body in 3D
space, which can facilitate plenty of applications such as
video surveillance, human-robot interaction, and physio-
therapy [54].
3D human poses can be directly retrieved
1Code is available: https://github.com/bruceyo/GLA-GCN
from advanced motion sensors such as motion capture sys-
tems, depth sensors, or stereotype cameras [52, 74]. The
3D HPE task can be performed under either multi-view or
monocular view settings. Although state-of-the-art multi-
view methods [31, 53, 79, 27] generally show superior
performance than monocular ones [29, 78], ordinary RGB
monocular cameras are much cheaper than these off-the-
shelf motion sensors and more widely applied in real-world
surveillance scenarios. Hence, 3D HPE from a monocular
video is an important and challenging task, which has been
attracting increasing research interest. Recent monocular
view works can be grouped into model-based and model-
free methods [14]. Model-based methods [16, 24] incorpo-
rate parametric body models such as kinematic [68], pla-
nar [61], and volumetric models [2] for 3D HPE. Model-
free methods can be further grouped into single-stage and
2D to 3D lifting methods. Single-stage methods estimate
the 3D pose directly from images in an end-to-end manner
[37, 47, 13, 66, 44, 82]. 2D to 3D lifting methods have
an intermediate 2D pose estimation layer [45, 51, 42, 62].
Among these methods, 2D to 3D lifting methods imple-
mented with ground truth 2D poses achieved better perfor-
mance.
The advantages of 2D to 3D lifting methods can be sum-
marized as two main points: allowing make use of advances
in 2D human pose detection and exploiting temporal infor-
mation along multiple 2D pose frames [51, 32]. For the 2D
human pose detection, it has achieved remarkable progress
via detectors such as Mask R-CNN (MRCNN) [25], Cas-
caded Pyramid Network (CPN) [15], Stacked Hourglass
(SH) detector [48], and HR-Net [59]. The intermediate 2D
pose estimation stage via these 2D pose detectors signifi-
cantly reduces the data volume and complexity of the 3D
HPE task.
For the temporal information, existing main-
stream methods [51, 42, 62, 29, 38, 78] gained notice-
able improvements by feeding a long sequence of 2D pose
frames to their models, among which [78] achieved the
state-of-the-art performance via ground truth 2D poses. Re-
arXiv:2307.05853v2  [cs.CV]  22 Jul 2023

cent methods [78, 85] simply fine-tuned these 2D pose de-
tectors on the target datasets and achieved great improve-
ments for the performance of estimated 2D pose data but
remain far behind the results of using ground truth 2D pose,
which motivates us to concentrate on improving the 3D
HPE via ground truth 2D pose data for potential improve-
ments via future more quality estimated 2D pose data.
Given the promising performance and advantages of 2D
to 3D lifting methods, our work contributes the literature
along this direction. For 2D to 3D lifting approaches, since
[45] proposed Fully Connected Network (FCN), recent ad-
vanced models have three main groups: Temporal Convolu-
tional Network (TCN)-based [51, 42], Graph Convolutional
Network (GCN)-based [80, 62, 29], and Transformer-based
ones [39, 38, 78]. On the one hand, we observe that exist-
ing TCN- and Transformer-based methods can receive large
receptive fields (i.e., a long 2D pose sequence) with strided
convolutions. However, it can be difficult to make further
intuitive designs to backtrace local joint features based on
the pose structure, since the 2D pose sequence is flattened
and fed to the model. Meanwhile, the estimation of dif-
ferent pose joints relies on the same fully connected layer,
which lacks considering the independent characteristic of
different pose joints. On the other hand, GCN-based mod-
els can explicitly reserve the structure of 2D and 3D human
pose during convolutional propagation. However, this ad-
vantage of GCN remains under-explored. Existing GCN-
based methods [80, 62] also utilized a fully connected layer
for the estimation of different 3D pose joints, which does
not consider the structural features of GCN representations.
To this end, we propose Global-local Adaptive GCN
(GLA-GCN) for 2D to 3D human pose lifting. Our GLA-
GCN contains two modules: global representation and lo-
cal 3D pose estimation. In the global representation, we
use an adaptive Graph Convolutional Network (GCN) to re-
construct the global representation of an intermediate 3D
human pose sequence from its corresponding 2D sequence.
For the local 3D pose joint estimation, we temporally shrink
the global representation optimized by the reconstructed 3D
pose sequence with a strided design. Then, an individual
connected layer is proposed to locally estimate the 3D hu-
man pose joints from the shrunken global representation.
Our contributions can be threefold as follows:
• We propose a global-local learning architecture that
leverages the global spatialtemporal representation and lo-
cal joint representation in the GCN-based model for 3D hu-
man pose estimation.
• We are the first to introduce an individual connected
layer that has two components to divide joint nodes and in-
put the joint node representation for 3D pose joint estima-
tion instead of based on pooled features.
• Our GLA-GCN model performs better than corre-
sponding state-of-the-art methods [38, 78] with consider-
able margins e.g., up to 3% and 17% error reductions on
Human3.6M [30] and HumanEva-I [58], respectively.

## method
Given the temporal information of a 2D human pose se-
quence estimated from a video P = {pt,i ∈R2| t =
1, ..., T; i = 1, ..., N}, where T is the number of pose
frames and N is the number of pose joints, we aim to uti-
lize this 2D pose sequence P to reconstruct 3D coordinates
of pose joints ¯P = {¯pi ∈R3|i = 1, ..., N}. Figure 1
shows the learning architecture of our GLA-GCN, which
uses AGCN layers to globally represent the 2D pose se-
quence and locally estimate the 3D pose via an individual
connected layer. In the following of this section, we intro-
duce the detailed design of our GLA-GCN.
3.1. Global Representation
Adaptive Graph Convolutional Network.
An AGCN
block [36, 57] is based on the GCN with an adaptive de-
sign that improves the flexibility of a typical ST-GCN block
[70]. Let us represent the 2D pose sequence P as a spatial-
temporal graph G = {υt, εt|t = 1, ..., T}, where υt =
{υt,i|i = 1, ..., N} represents the pose joints and εt repre-
sents the corresponding pose bones. To implement a basic
ST-GCN block, a neighbor set Bi is first defined to indi-
cate the spatial graph convolutional filter for a specific pose
joint υt,i. Specifically, for the graph convolutional filter of a
vertex node, we apply three distance neighbor subsets: the
vertex itself, centripetal subset, and centrifugal subset. The
definitions of centripetal and centrifugal subsets are based
on the pose frame’s gravity center (i.e., the average coordi-
nate of all pose joints). Centripetal and centrifugal subsets
represent nodes that are closer and farther to the average
distance from the gravity center, respectively. Empirically,
similar with 2D convolution, we set the kernel size K to 3,
which will lead to 3 subsets in Bi. To implement the sub-
sets, a mapping ht,i →{0, ..., K −1} is used to index each

subset with a numeric label, where centripetal and centrifu-
gal subsets are respectively labeled as 1 and 2. Subsets that
have the average distance to gravity center is indexed to 0.
This graph convolutional operation can be written as
fout(υt,i) =
X
υt,j∈Bi
1
Zt,j
fin (υt,j) W(ht,i(υt,j)) (1)
where fin : vt,j →R2 is a mapping that gets the attribute
features of joint node vt,j and Zt,j is a normalization term
that equals to the subset’s cardinality. W(ht,i(vt,j)) is a
weight function W (υt,i, υt,j) : Bi →R2 implemented
by indexing a (2, K) tensor. For a pose frame, the deter-
mined graph convolution of a sampling strategy (e.g., cen-
tripetal and centrifugal subsets) can be implemented by an
N × N adjacency matrix. Specifically, with K spatial sam-
pling strategies PK
k=1 Ak and the adaptive design, Equa-
tion 1 can be transformed into
fout(υt) =
XK
k=1(Ak + Bk + Ck)finWk
(2)
where Λ
−1
2
k
¯AkΛ
−1
2
k
is a normalized adjacency matrix of
¯Ak with its elements indicating whether a vertex υt,j is in-
cluded in the neighbor set. Λii
k = P
j(¯Aij
k ) + α is a di-
agonal matrix with α set to 0.001 to prevent empty rows.
Wk denotes the weighting function of Equation 1, which is
a weight tensor of the 1×1 convolutional operation. Unlike
Ak that represents the physical structure of a human pose,
Bk represents learnable parameters that indicate the con-
nection strength between pose joints,which is implemented
with an N × N adjacency matrix initialized to 0. Ck per-
forms the similar function of Bk, which is implemented by
the dot product of two feature maps calculated by embed-
ding functions (i.e., θ and ϕ) to calculate the similarity be-
tween pose joints. Calculation of Ck can be represented
as
Ck = SoftMax(f T
inWT
θkWϕkfin)
(3)
where Wθ and Wϕ are learnable parameter of the two em-
bedding functions, which are initialized as 0.0. Then an
AGCN block is realized with a 1 × Γ classical 2D con-
volutional layer (Γ is the temporal kernel size that we set
to 9) and the defined adaptive graph convolution fout(υt),
which are both followed by a batch normalization layer and
a ReLU layer and a dropout layer in between them. Mean-
while, a residual connection [26] is added to the AGCN
block.
Reconstruct 3D Pose Sequence. Taking the inspiration of
recent works [62, 29, 39, 38], the introduced AGCN block
is then used to extract the spatiotemporal structural infor-
mation in the global graph representation, which is super-
vised by estimating the 3D pose sequence from the corre-
sponding 2D sequence (see Figure 1 [Reconstruct 3D Pose
Sequence]). Here, each AGCN block has three key parame-
ters: the number of input channels Cin, the number of out-
put channels Cout, and the stride S of the temporal convolu-
tion, while the other parameters are kept consistent (e.g., the
temporal convolution kernel size is three). Given an input
Cin-dim pose representation F(Cin, Tin, N), the AGCN
block derives the output Cout-dim pose F(Cout, Tout, N)
via convolution on the pose structure sequence, where Tout
depends on Nin and S.
To reconstruct the 3D pose se-
quence, we first use AGCN(2, 96, 1) to convert the 2D
pose sequence F(2, T, N) into a 96D pose representation
F(96, T, N). Following the settings of related work, we
set T to 243 and N to 17 for the Human3.6M dataset. That
is, the input 2D pose sequence of F(2, 243, 17) is converted
into a 96D pose sequence of F(96, 243, 17). Then, we stack
iterative layers of AGCN(96, 96, 1) to construct the deep
spatiotemporal structural representation of the 96D pose se-
quence. The output of the last AGCN block is fed into an
AGCN(96, 3, 1) to estimate the 3D pose sequence based
on the 96D joint representation and derive F(3, 243, 17).
Then, we let ...p t,i ∈R3 be the 3D position of the i-th joint
at time t, and minimize the difference between the estimated
3D pose sequence and the ground truth 3D pose sequence:
Lglobal = 1
T
1
N
T
X
t=1
N
X
i=1
 ...p t,i −pt,i

2
(4)
Strided Learning Architecture.
Inspired by the TCN-
based approaches [51, 42], we further adapt the strided
learning architecture to the AGCN model, using strided
convolution to reduce long time sequences and aggregate
temporal information near time t for pose estimation. The
gray module in Figure 1(Strided Learning) illustrates the
design of the strided AGCN modules. Each strided AGCN
module has two consecutive AGCN blocks, which are sur-
rounded by residual connections [26]. We perform strided
convolutions at the second AGCN block of each strided
AGCN module to gradually shrink the feature size at the
temporal dimension. The input of the first strided AGCN
module is the intermediate output in 3D pose sequence
reconstruction, i.e., the extracted F(96, 243, 17).
After
the propagation through the first strided AGAN module,
the 96D pose sequence will be shrunken to F(96, 81, 17).
Then, we repetitively perform subsequence AGCN layers
until the feature size is shrunken to the size of 96 × 1 × 17.
In this way, the pattern of the temporal neighbor in the
pose sequence will be aggregated for subsequent local 3D
pose joint estimation to estimate the 3D pose of the centric
timestep.
3.2. Local 3D Pose Joint Estimation
Based on the above-mentioned strided AGCN modules,
the input 2D pose sequence represented as F(96, 243, 17)

can be transformed into a feature map F(96, 1, 17). The
next step is to estimate the 3D position of joint nodes based
on the feature map.
Individually Connected Layers.
Existing TCN- and
GCN-based methods [51, 42, 80, 62] usually flatten the de-
rived feature maps and use a global skeleton representa-
tion consisting of all joint nodes to estimate every single
joint, neglecting the matching information between joints
and corresponding vectors in feature maps. Unlike existing
works, we believe the global knowledge of the temporal and
spatial neighborhoods has been aggregated via the proposed
global representation. Thus, it is crucial to scope at the spa-
tial information of the corresponding joint node to infer its
3D position. Based on this idea, this paper first proposes
an individual connected layer to estimate the 3D position
of every single joint based on the corresponding joint node
feature F(96, 1, 1), instead of the pooled representation of
all joint nodes F(96, 1, 17). Mathematically, the individual
connected layer can be denoted as:
˙p(unshared)
i
= viWi + bi
(5)
where the estimated 3D position of joint i is denoted by ˙pi
and vi represents the flattened features of F(96, 1, i) joint
node i. The weight parameters of the individual connected
layer is represented by Wi and Wi ∈R96×3, whose bias
parameter is bi and bi ∈R1×3.
Due to the weight Wi and bias bi are not shared between
joints, we name the above individually connected layers as
unshared individually connected layers. On top of that, we
find that individually connected layers in the unshared fash-
ion may ignore the shared rules between joints in 2D to 3D
pose lifting, resulting in overfitting joint-specific distribu-
tion. Therefore, we further designed shared individually
connected layers:
˙p(shared)
i
= viWs + bs
(6)
The weight parameters of the shared individual connected
layer is represe

## experiments
4.1. Datasets and Evaluation
Our experiments are based on three public datasets:
Human3.6M [30], HumanEva-I [58], and MPI-INF-3DHP
[46].
With respect to Human3.6M, the data of subjects
S1, S5, S6, S7, and S8 are applied for training, while that
of S9 and S11 are used for testing, which is consistent
with the training and validation settings of existing works
[51, 42, 80, 62]. In terms of HumanEva-I, following [45]
and [42], data for actions “walk” and “jog” from subjects
S1, S2, and S3 are used for training and testing. For MPI-
INF-3DHP, we follow the experimental setting of the recent
state-of-the-art [55] for a fair comparison.
Standard evaluation protocols: Mean Per-Joint Position
Error (MPJPE) and Pose-aligned MPJPE (P-MPJPE), re-
spectively known as Protocol#1 and Protocol#2, are
used for both datasets. The calculation of MPJPE is based
on the mean Euclidean distance between the predicted 3D
pose joints aligned to root joints (i.e., pelvis) and the ground
truth 3D pose joints collected via motion capture, which fol-
lows [84, 60, 50]. Comparing with MPJPE, P-MPJPE is
also based on the mean Euclidean distance but has an extra
post-processing step with rigid alignments (e.g., scale, ro-
tation, and translation) to the predicted 3D pose. P-MPJPE
leads to smaller differences with the ground truth and it fol-
lows [45, 28, 22].
4.2. Implementation Details
We introduce the implementation detail of our GLA-
GCN from three main perspectives: 2D pose detections,
model setting, and hyperparameters for the training process.
For fair comparison, we follow the 2D pose detections of
Human3.6M [30] and HumanEva-I [58] used in [51, 42],
which are detected by CPN [15] and MRCNN [25], respec-
tively. The CPN’s 2D pose detection has 17 joints while
the MRCNN’s 2D pose detection has 15 joints. Besides, we
also conduct experiments for the ground truth (GT) 2D pose
detections of the two datasets.
Based on the specific structure of 2D pose, we imple-
ment the graph convolutional operation filters of AGCN

## related_work
2D to 3D Lifting. 3D HPE is a traditional vision prob-
lem that has been studied for decades [19, 76, 8, 64, 23,
33, 65, 67, 63, 43].
Existing works of 3D HPE from a
monocular view usually target two main scenarios: single
person and multi-person [14]. This work aims to improve
the performance of single person 3D HPE. [34, 1, 71] rep-
resent early efforts that attempt to infer 3D position from
2D projections. They usually rely on manually chosen pa-
rameters based on assumptions about pose joint mobility.
Methods [24, 77] estimating 3D pose from less frames or
even a single frame has shown great progress but can be
a lack of considering temporal information.
Recent ad-
vances in 2D human pose estimation [48, 25, 15] enable
2D to 3D lifting approaches to achieve remarkable perfor-
mance over other counterparts. Inspired by [45], there has
been more well-designed learning architectures being pro-
posed to improve the performance, in particular, by utiliz-
ing temporal information. These methods are also known
as 2D to 3D lifting, which can be grouped into three direc-
tions: TCN-, GCN-, and Transformer-based architectures
[51, 42, 80, 62, 29, 38, 78].
TCN-based methods [51, 42] successfully push the per-
formance of 2D to 3D lifting forward with a strided de-
sign for their learning architectures built upon 1D CNN
layers. The strided design is on the temporal dimension
of the input, which allows the features to shrink from a
2D pose sequence to a feature embedding for the 3D pose
estimation via a final fully connected layer. The number
of channels for the fully connected layer is conventionally
set to 1024, which is shared to predict the 3D positions of
all pose joints. While varied numbers of input 2D pose
frames have been extensively investigated, which shows in-
put 2D pose frames with reasonable length can benefit the
3D pose reconstruction. The strided design can effectively
reduce the feature size by shrinking the number of tempo-
ral frames along propagation of several TCN blocks. Using
this strided structure, Transformer-based methods [38, 78]
show promising performance, especially [78] that takes ad-
vantage of weighted and temporal loss functions and helps
it outperform the GCN-based methods optimized with an
additional motion loss [62, 29]. The motion loss was shown
not very effective in [78]. These observations compel us
to explore effective models in the direction of GCN-based
models with the inspiring designs in mind but without rely-
ing on various novel loss functions.
Graph Convolutional Network. A popular method repre-
senting the pose data with GCN is Spatial Temporal GCN
(ST-GCN) [70], which is originally proposed to model large
receptive fields for the skeleton-based action recognition.

AGCN(96,3,1)
F(2,243,17)
F(96,3,17)
F(96,1,17)
Global Representation
2D Pose Sequence
F(3,17)
Individual Connected Layer
BatchNorm 1D
AGCN(2,96,1)
AGCN(96,96,1)
AGCN(96,96,3)
F(96,243,17)
Reconstructed 3D Pose
F(96,81,17)
P:
P:
AGCN(96,96,1)
Local 3D Pose Joint Estimation
Strided Learning
Reconstruct 3D Pose Sequence
F(3,243,17)
P:
Figure 1. Learning architecture of our GLA-GCN. AGCN(Cin, Cout, S) represents AGCN blocks with the specific values of the input
channel, output channel, and stride length. F(C′, T ′, N ′) represents the size of a feature map. The individual connected layer shows the
prediction process of four pose joint examples that use separate 1D CNN layers.
Following ST-GCN, advanced GCN models have been pro-
posed to advance 3D HPE [18, 80, 17, 62].
Regarding GCN-based models for 3D HPE, Ci et al. [18]
proposed Locally Connected Network (LCN) that takes the
advantages of FCN [45] and GCN [20]. LCN has the sim-
ilar design for the convolutional filters to ST-GCN [70],
which defines a neighbor set for a node based on the dis-
tance to perform convolutional operation. Zhao et al. [80]
proposed an architecture called SemGCN that stacks GCN
layers by flatten output to a fully connected layer. The op-
timization of SemGCN is based on both joint positions and
bone vectors. Choi et al. [17] also proposed to use GCN to
recover 3D human pose and mesh from a 2D human pose.
Liu et al. [41] investigated how weight sharing schemes
in GCNs affect the pose lifting task, which shows the pre-
aggregation method leads to relatively better performance.
The architecture in [41] is similar with that of SemGCN.
The above mentioned GCN-based methods achieved good
performance via a single pose frame input but they did not
take the advantage of temporal information in a 2D pose
sequence.
Taking multiple 2D pose frames as input, U-shaped
Graph Convolution Networks (UGCN) [62, 29] further im-
proves the performance of GCN-based methods by paying
attention to the temporal characteristics of a pose motion.
Specifically, UGCN utilizes spatial temporal GCN [70] to
predict a 3D pose sequence from a 2D pose sequence for
the reconstruction of a single 3D pose frame. A motion
loss term that regulates the temporal trajectory of pose joints
based on the prediction of a 3D pose sequence and its cor-
responding ground truth 3D pose sequence. Despite the im-
provements grained with novel loss terms in works such as
SemGCN and UGCN, we aim to contribute the literature
of 2D-3D lifting by using the consistent loss term used in
[51, 42]. In our model design, we propose to incorporate
the strided convolutions to a GCN-based model that rep-
resents global information of a 2D pose sequence. Based
on the structure of GCN representation, we explicitly uti-
lize the structured features of different pose joints to locally
predict their corresponding 3D pose locations.

## conclusion
This paper proposes a GCN-based method utilizing the
structured representation for 3D HPE in the 2D to 3D lift-
ing paradigm. The proposed GLA-GCN globally represents
the 2D pose sequence and locally estimates the 3D pose
joints via an individual connected layer. Results show that
our GLA-GCN outperforms corresponding state-of-the-art
methods implemented with GT 2D poses on datasets Hu-
man3.6M, HumanEva-I, and MPI-INF-3DHP. We verify the
properness of model design with extensive ablation studies
and visualizations. In the future, we will tackle the issue
of parameter efficiency of our model via tuning techniques
[72, 73].
Meanwhile, we will consider its effect on ap-
plication scenarios such as human behavior understanding
[6, 5, 4, 7, 3] and aim to improve the results of the esti-
mated 2D pose by preparing high-quality 2D pose data via
fine-tuned 2D pose detectors (e.g., SH detector [48], Open-
Pose [10], and HR-Net [59]), abd investigate the effects of
other loss terms (e.g., based on bone features [12] and mo-
tion trajectory [62]).