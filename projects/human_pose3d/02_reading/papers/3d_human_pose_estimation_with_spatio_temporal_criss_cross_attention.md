# 3D Human Pose Estimation with Spatio-Temporal Criss-Cross Attention

> 2023 · id: W4386076485 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

3D Human Pose Estimation with Spatio-Temporal Criss-cross Attention*
Zhenhua Tang, Zhaofan Qiu, Yanbin Hao, Richang Hong, Ting Yao
Hefei University of Technology, Anhui, China
HiDream.ai Inc
University of Science and Technology of China, Anhui, China
zhenhuat@foxmail.com, zhaofanqiu@gmail.com, haoyanbin@hotmail.com
hongrc.hfut@gmail.com, tingyao.ustc@gmail.com
Abstract
Recent transformer-based solutions have shown great
success in 3D human pose estimation. Nevertheless, to cal-
culate the joint-to-joint affinity matrix, the computational
cost has a quadratic growth with the increasing number
of joints. Such drawback becomes even worse especially
for pose estimation in a video sequence, which necessitates
spatio-temporal correlation spanning over the entire video.
In this paper, we facilitate the issue by decomposing cor-
relation learning into space and time, and present a novel
Spatio-Temporal Criss-cross attention (STC) block. Tech-
nically, STC first slices its input feature into two partitions
evenly along the channel dimension, followed by perform-
ing spatial and temporal attention respectively on each par-
tition. STC then models the interactions between joints in an
identical frame and joints in an identical trajectory simulta-
neously by concatenating the outputs from attention layers.
On this basis, we devise STCFormer by stacking multiple
STC blocks and further integrate a new Structure-enhanced
Positional Embedding (SPE) into STCFormer to take the
structure of human body into consideration. The embedding
function consists of two components: spatio-temporal con-
volution around neighboring joints to capture local struc-
ture, and part-aware embedding to indicate which part each
joint belongs to. Extensive experiments are conducted on
Human3.6M and MPI-INF-3DHP benchmarks, and supe-
rior results are reported when comparing to the state-of-
the-art approaches. More remarkably, STCFormer achieves
to-date the best published performance: 40.5mm P1 error
on the challenging Human3.6M dataset.
1. Introduction
3D human pose estimation has attracted intensive re-
search attention in CV community due to its great poten-
*This work is supported by the National Natural Science Foundation of
China under Grants 61932009.
S
T
C
Spatio-Temporal
Attention
MLP
Temporal
Attention
MLP
Spatial
Attention
MLP
MLP
C
Spatial
Attention
Temporal
Attention
S
T
C
S
T
C
Receptive Field
(a)
(b)
(c)
slicing
concat
x L
x L1
x L2
x L
Figure 1.
Modeling spatio-temporal correlation for 3D human
pose estimation by (a) utilizing spatio-temporal attention on all
joints in the entire video, (b) separating the framework into two
steps that respectively capture spatial and temporal context, and
(c) our Spatio-Temporal Criss-cross attention (STC), i.e., a two-
pathway block that models spatial and temporal information in
parallel. In the visualization of receptive field, the covered joints
of each attention strategy is marked as red nodes.
tial in numerous applications such as human-robot inter-
action [20, 43], virtual reality [11] and motion prediction
[27, 28].
The typical monocular solution is a two-stage
pipeline, which first extracts 2D keypoints by 2D human
pose detectors (e.g., [7] and [41]), and then lifts 2D coordi-
nates into 3D space [31]. Despite its simplicity, the second
stage is an ill-posed problem which lacks the depth prior,
and suffers from the ambiguity problem.
To mitigate this issue, several progresses propose to ag-
gregate the temporal cues in a video sequence to promote
pose estimation by grid convolutions [15,26,35], graph con-
volutions [4, 47] and multi-layer perceptrons [6, 21]. Re-
cently, Transformer structure has emerged as a dominant ar-
chitecture in both NLP and CV fields [8,24,45,49], and also
demonstrated high capability in modeling spatio-temporal
correlation for 3D human pose estimation [13, 22, 23, 25,
48, 52, 54]. Figure 1(a) illustrates a straightforward way
to exploit the transformer architecture for directly learning
spatio-temporal correlation between all joints in the entire
video sequence. However, the computational cost of calcu-
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
4790

lating the joint-to-joint affinity matrix in the self-attention
has a quadratic growth along the increase of number of
frames, making such solution unpractical for model train-
ing. As a result, most transformer structures employ a two-
step alternative, as shown in Figure 1(b), which encodes
spatial information for each frame first and then aggregates
the feature sequence by temporal transformer. Note that we
take spatial transformer as the frame encoder as an example
in the figure. This strategy basically mines the correlation
across frame-level features but seldom explores the relation
between joints across different frames.
In this paper, we propose a novel two-pathway attention
mechanism, namely Spatio-Temporal Criss-cross attention
(STC), that models spatial and temporal information in par-
allel, as depicted in Figure 1(c). Concretely, STC first slices
the input joint features into two partitions evenly with re-
spect to the channel dimension. On each partition, a Multi-
head Self-Attention (MSA) is implemented to encapsulate
the context along space or time axis. In between, the space
pathway computes the affinity between joints in each frame
independently, and the time pathway correlates the identi-
cal joint moving across different frames, i.e., the trajectory.
Then, STC recombines the learnt contexts from two path-
ways, and mixes the information across channels by Multi-
Layer Perceptrons (MLP). By doing so, the receptive field is
like a criss cross of spatial and temporal axes, and the com-
putational cost is O(T 2S) + O(TS2). That is much lower
than O(T 2S2) of fully spatio-temporal attention, where T
and S denote the number of frames and joints, respectively.
By stacking multiple STC blocks, we devise a new ar-
chitecture — STCFormer for 3D human pose estimation.
Furthermore, we delve into the crucial design of positional
embedding in STCFormer in the context of pose estimation.
The observations that joints in the same body part are either
highly relevant (static part) or not relevant but containing
moving patterns (dynamic part) motivate us to design a new
Structure-enhanced Positional Embedding (SPE). SPE con-
sists of two embedding functions for the static and dynamic
part, respectively. A part-aware embedding is to describe
the static part by indicating which part each joint belongs
to, and a spatio-temporal convolution around neighboring
joints aims to capture dynamic structure in local window.
We summarize the main contributions of this work as
follows. First, STC is a new type of decomposed spatio-
temporal attention for 3D human pose estimation in an eco-
nomic and effective way. Second, STCFormer is a novel
transformer architecture by stacking multiple STC blocks
and integrating the structure-enhanced positional embed-
ding. Extensive experiments conducted on Human3.6M and
MPI-INF-3DHP datasets demonstrate that STCFormer with
much less parameters achieves superior performances than
the state-of-the-art techniques.
2. Related Work
Monocular 3D human pose estimation. Monocular 3D
human pose estimation is to re-localize human body joints
in 3D space from the input single view 2D data, i.e., im-
age or 2D coordinates. The early works [1, 2, 17] develop
various graphical or restrictive methods to explore the de-
pendencies of human skeleton and perspective relationships
across spaces. With the development of deep learning, sev-
eral deep neural networks [5, 10, 19, 31, 34, 42, 44, 53] are
devised for 3D human pose estimation, and can be cate-
gorized into one-stage and two-stage directions. The one-
stage approaches directly regress the 3D pose from the input
image, and necessitate a large number of image-pose paired
data and powerful computing resources [19, 34, 42]. The
two-stage methods first exploit off-the-shelf 2D pose detec-
tors [7, 33, 41] to estimate 2D joint coordinates, and then
lift the 2D coordinates into 3D space by the fully-connected
network [31], grid convolutional network [5], recurrent neu-
ral network [10], or graph convolutional network [53]. Al-
though the two-stage methods alleviate the requirement of
image-pose pairs, they still heavily suffer from the depth
ambiguities problem, which is intrinsically ill-posed due to
the lack of depth information.
3D pose estimation from video sequence.
To over-
come the limitation of depth ambiguities, the advances in-
volve temporal context from neighboring frames to im-
prove 3D coordinates regression. For example, Pavllo et
al. [35] propose a temporal fully-convolutional network
(TCN) to model the local context by convoluting the neigh-
boring frames. Later, Liu et al. [26] extend the TCN by
introducing an attention mechanism to adaptively identify
the significant frames/poses over a sequence. After that,
Chen et al. [6] decompose the pose estimation into bone
length and bone direction prediction. Instead of the afore-
mentioned methods based on temporal aggregation, latter
works [4, 16, 46] utilize the spatio-temporal graph convo-
lutional network to model the spatial and temporal correla-
tions across joints simultaneously.
Transformer-based methods.
In addition to the tra-
ditional convolutional networks, transformer architectures
are also be exploited to model spatio-temporal correlation
[13,22,23,29,30,37,50,51,54]. In particular, Zheng et al.
[54] design a concatenation architecture of several spatial
transformer encoders and temporal transformer encoders in
PoseFormer. MHFormer [23] proposes to generate multi-
ple hypothesis representations for a pose with the spatial
transformer encoder and then model multi-level global cor-
relations with different temporal transformer blocks. Strid-
edFormer [22] and CrossFormer [13] introduce locality by
integrating the 1D temporal convolution and 1D spatial con-
volution, respectively. More recently, the joint-wise incon-
sistency of motion patterns is highlighted in [48, 52], and
encourages to model spatial and temporal information si-
4791

multaneously. PATA [48] groups the joints with similar mo-
tion patterns and calculates the intra-part temporal correla-
tion. Similarly, MixSTE [52] uses multiple separated spa-
tial transformer blocks and temporal transformer blocks to
model the spatial and temporal correlation iteratively.
Our work also falls into the category of transformer-
based method for 3D human pose estimation. The afore-
mentioned transformers mainly model spatial and temporal
information respectively in different stages of the networks.
In view that the joint motion is a state of coexistence of
space and time, such separation may result in insufficient
learning of moving patterns. In contrast, our STC block is
a two-pathway design that models spatial and temporal de-
pendencies in parallel, which are then mixed through MLP.
Moreover, a new positional embedding function is deliber-
ately devised to explore the local structure of human body.
3. Spatio-Temporal Criss-cross Transformer
3.1. Preliminary – Transformer
We begin this section by reviewing the transformer ar-
chitecture [45] as the basis of our proposal. Transformer is
a versatile representation learning architecture, and mainly
consists of two components:
Multi-head Self-Attention
module (MSA) and Feed-Forward Network (FFN). MSA
calculates the token-to-token affinity matrix and propagates
the information across different tokens. Formally, given N
input tokens with C channels, MSA is formulated as
 \l abe l {e q :msa} \
be g in
 
{
a
l i gned} MSA \left (\mathbf {Q}, \mathbf {K}, \mathbf {V} \right ) = Softmax\left ( \frac {\mathbf {Q}\cdot \mathbf {K} ^{T}}{\sqrt {C} } \right )\cdot \mathbf {V}~, \end {aligned} 
(1)
where Q, K, V ∈RN×C denote the queries, keys and val-
ues obtained by linearly mapping the input tokens. Note
that we omit the multi-head separation here for simplicity.
FFN contains a Multi-Layer Perceptrons (MLP), i.e., a non-
linear mapping with two linear layer plus a GELU [14] ac-
tivation in between. The output of MLP is computed by
 \l abe l {eq :m l p} \ sm all \begin {aligned} MLP\left (\mathbf {H}\right )=GELU\left (\mathbf {H}\cdot \mathbf {W_1}\right )\cdot \mathbf {W_2}~, \end {aligned} 
(2)
where H ∈RN×C is the input tokens of MLP, W1 ∈
RC× ˆ
C and W2 ∈R ˆ
C×C are the projection matrices. With
these, each transformer block is constructed by utilizing
MSA and MLP in order with shortcut connection:
 \ la b e l {eq :tra n
s f orm er} \ sm a l l
 \ beg in {ali g n ed} & \mathbf {Q}, \mathbf {K}, \mathbf {V} = FC\left (LN\left (\mathbf {X}\right )\right ) ~, \\ & \mathbf {Y} = MSA \left (\mathbf {Q}, \mathbf {K}, \mathbf {V} \right ) + \mathbf {X} ~, \\ & \mathbf {Z} = MLP\left (LN\left (\mathbf {Y}\right )\right ) + \mathbf {Y} ~, \end {aligned} 
(3)
where FC is linear projection of the input tokens X, and
LN denotes Layer Norm [3]. The output Z serves as the
input to the next block until the last one.
Figure 2. An overview of our proposed Spatio-Temporal Criss-
cross Transformer (STCFormer). (a) It mainly consists of L se-
quentrial STC blocks. Each block aggregates the context across
tokens by spatio-temporal criss-cross attention, and non-linearly
maps each token by Multi-Layer Perceptrons (MLP). (b) The ar-
chitecture of our STC block and the Structure-enhanced Positional
Embedding (SPE).
3.2. Overall Architecture
Figure 2 depicts an overview of the proposed STC-
Former, which mainly includes three stages: a joint-based
embedding, stacked STC blocks and a regression head. The
joint-based embedding projects the input 2D coordinates of
each joint into feature space.
STC blocks aggregate the
spatio-temporal context, and update the representation of
each joint. Based on the learnt features, the 3D coordinates
are estimated by a regression head.
Joint-based embedding. Given a 2D pose sequence as
P2D ∈RT ×N×2, where T and N denote the number of
frames and the number of body joints in each frame, respec-
tively, we first project P2D to high-dimensional embed-
dings by a joint-based embedding layer. This layer applies
an FC layer to each 2D coordinate independently followed
by a GELU activation. As such, the joint-based embedding
layer produces the features with the shape of T × N × C.
Note that in the previous transformer [22], the embedding
layer projects all joint coordinates in each frame into a sin-
gle vector, reducing the computational cost of the subse-
quent transformer blocks while losing the spatial discrimi-
nation. Ours is different in that the spatial dimension N is
maintained, and the computational cost is also reduced by
spatio-temporal criss-cross attention.
STC blocks. The STC block originates from the trans-
former block in Eq.(3), and replaces the original MSA
layer with spatio-temporal criss-cross attention. In addi-
tion, a new positional embedding function, i.e., Structure-
enhanced Positional Embedding (SPE), is integrated into
the STC block for better descriptive capability of local
structures. Section 3.3 and Section 3.4 will elaborate STC
and SPE, respectively.
4792

Regression head. A liner regression head is finally es-
tablished upon the STC blocks to estimate the 3D pose co-
ordinates ˆP3D ∈RT ×N×3. The whole architecture is op-
timized by minimizing the Mean Squared Error (MSE) be-
tween ˆP3D and the ground-truth 3D coordinates P3D as
 \
label { e q:l
oss
}
 \small \begin {aligned} \mathcal {L}= \left \| \mathbf {\hat {P}}_{3D}-\mathbf {P}_{3D} \right \|^2 ~. \end {aligned} 
(4)
3.3. Spatio-Temporal Criss-cross Attention
STC aims to model the spatio-temporal dependencies be-
tween joints in an efficient way to avoid the quadratic com-
putation cost of fully spatio-temporal attention. Inspired by
the group contextualization strategy [12] which separates
the channels into several paralleled groups and applies dif-
ferent feature contextualization operations to them respec-
tively, we propose to capture the spatial and temporal con-
text on different channels in parallel. Different from the ax-
ial convolution in [12,36,38], we exploit axis-specific multi-
head self-attention in STC for spatial or temporal context,
which is more powerful for correlation learning.
Concretely, the input embedding X ∈RT ×N×C are
firstly mapped to queries Q ∈RT ×N×C, keys K ∈
RT ×N×C, and values V ∈RT ×N×C, which are then
evenly divided into two groups along the channel dimen-
sion.
For notation clarity, we denote the divided fea-
ture matrix as time group {QT , KT , VT } and space group
{QS, KS, VS}. Next, the temporal and spatial correlations
are calculated in two separate self-attention modules.
Temporal correlation represents the relation between
the joints in an identical trajectory moving across differ-
ent frames. To achieve this, we implement an axis-specific
MSA, named MSAT , which computes the attention affini-
ties in Eq.(1) between joints across the temporal dimension.
Hence, the output of temporal attention is measured as
 \ l abel {eq :te m} \small \begin {aligned} \mathbf {H_T} = MSA_T \left (\mathbf {Q_T}, \mathbf {K_T}, \mathbf {V_T} \right ) ~. \\ \end {aligned} 
(5)
Spatial correlation is the connection between joints in
an identical frame.
These joints indicate different body
parts in one frame, which are intrinsically relevant due to
the prior of body skeleton. Similar to temporal attention,
we devise MSAS as an axis-specific MSA component on
spatial dimension. Therefore, the output of spatial attention
is formulated as
 \ l abel {eq :sp a1} \small \begin {aligned} \mathbf {H_S} = MSA_S \left (\mathbf {Q_S}, \mathbf {K_S}, \mathbf {V_S} \right ) ~. \\ \end {aligned} 
(6)
The above two correlation modules process in parallel
and follow the self-attention regime for feature contextual-
ization. They compute the token-to-token affinities by con-
textualizing from a specific axial perspective, and comple-
ment to each other. Thus, we concatenate the outputs from
both attention layers along the channel dimension:
 \ lab el { eq: spa2} \small \begin {aligned} \mathbf {H} = cat\left (\mathbf {H_T}, \mathbf {H_S}\right ) ~, \\ \end {aligned} 
(7)
[0] Hip
[1] Spine
[2] Thorax
[3] Neck
[4] Head
[5] R Hip
[6] R Knee
[7] R Foot
[8] L Hip
[9] L Knee
[10] L Foot
[11] R Shouler
[12] R Elbow
[13] R Wrist
[14] L Shouler
[15] L Elbow
[16] L Wrist
 
0
5
1
10
9
8
7
6
3
2
4
14
15
16
11
12
13
�0
�1
�2
�3
�4
�1
�0
�2
�3
�4
（a）
（b）
Figure 3. (a) The coefficient matrix of the motion trajectory of
different joints. (b) The body joints are divided into five parts, de-
noted as g∗. The part with high/low relevance is colored as light/-
dark blue, respectively. The motion data is generated by actor S6
performing greeting action in the training set of Human3.6M.
where cat performs the concatenation. The resultant recep-
tive field of STC is like a criss cross of spatial and temporal
axes, and stacking multiple STC blocks is able to approxi-
mate the fully spatio-temporal attention.
3.4. Structure-enhanced Positional Embedding
One of the crucial factor in transformer is positional em-
bedding, which indicates the position of each token abso-
lutely or relatively. For the positional embedding function
in STCFormer, we delve into the inherent property of joints,
i.e., the local structure, and propose Structure-enhanced Po-
sitional Embedding (SPE). Figure 3 depicts the motivation
of SPE. Here, we group the body joints into five parts ac-
cording to the dynamic chain structure of human body:
 \ l abel {eq:pa rt} \sm all \ begin
 { a ligned } g_ {0}&= \lef t \{ hip,s
pi n e,thr oax, neck ,head \ri ght \
} \ \ g_{1 }&= \left \{ r ight\_ hip,r ight\_
kn e e,rig ht\_feet \rig ht \}\ \ g_ {2}&= \left \{ left\_hip,left\_knee,left\_feet \right \} \\ g_{3}&= \left \{ right\_shoulder,right\_elbow,right\_wrist \right \}\\ g_{4}&= \left \{ left\_shoulder,left\_elbow,left\_wrist \right \} \end {aligned} 
(8)
The trajectories of joints in the static part (g0, g3 and g4
in the figure) are highly relevant. We devise a part-ware
positional embedding to indicate which part each joint be-
longs to. The joints in the same part are attached with the
same embedding vector. In particular, a learnable dictio-
nary is constructed to assign embedding vector to different
joints according to their group index. Given the group in-
dex g ∈[0, 1, 2, 3, 4]T ×N of joints, the learnable dictionary
D ∈R5× C
2 convert the indexes to embedding vectors as
 \la b el { eq:sepe1} \small \begin {aligned} \mathbf {SPE}_{1} = \mathbf {D}(\mathbf {g}) ~. \\ \end {aligned} 
(9)
Nevertheless, in the dynamic part, i.e., part with relative
movements (g1, g2 in the figure), the trajectories of joints
are not relevant. Simply assigning the same embedding vec-
tor to these joints ignores the motion patterns in the dynamic
part. Hence, we propose to exploit a spatio-temporal con-
volution around the neighboring joints to capture the local
structure. Formally, given the values V ∈RT ×N× C
2 in
4793

Algorithm 1 Pseudo-code of STC with SPE (PyTorch-like)
# x: input tensor of shape (B, T, N, C)
# p: part index (B, T, N) in [0, 4]
# MSA: axis-specific multi-head self-attention
self.linear = nn.Linear(C, 3C)
self.embed1 = nn.Embedding(5, C//2)
# the channel-last convolution
self.embed2 = nn.Conv2d(C//2, C//2, k=3, g=C//2)
def STC(x, p):
Q, K, V = self.linear(x).chunk(3, dim=3)
Q_t, Q_s = Q.chunk(2, dim=3)
K_t, K_s = K.chunk(2, dim=3)
V_t, V_s = V.chunk(2, dim=3)
H_t = MSA(Q_t, K_t, V_t, dim=1)
H_s = MSA(Q_s, K_s, V_s, dim=2)
H_t += self.embed1(p) + self.embed2(V_t)
H_s += self.embed1(p) + self.embed2(V_s)
H = torch.cat(H_t, H_s, dim=3)
return H
STC block, we treat V as 2D (i.e., space and time) feature
map , and utilize 2D convolution on the neighboring joints:
 \label {eq:sepe2 } \small \begin {aligned} \mathbf {SPE}_{2}(\mathbf {V}) = conv2d(\mathbf {V}) ~, \\ \end {aligned} 
(10)
where conv2d is a 3 × 3 convolution operation. Although
the two SPE functions are designed respectively for static
part and dynamic part, we utilize the two functions concur-
rently on all joints leaving out the requirement of static/dy-
namic judgment. The duet of two SPE functions is able to
deal with the parts with various moving patterns.
By injecting the proposed SPE function into STC, the
equation of STC is reformulated as
 \ l abel {eq :st ca} \sma l l \begin 
{a l igne d} & \m ath b f {H _ T} = MSA_
T \le ft ( \ma thbf {Q_T}, \mathbf {K_T}, \mathbf {V_T} \right ) + \mathbf {SPE}_{1} + \mathbf {SPE}_{2}(\mathbf {V_T}), \\ & \mathbf {H_S} = MSA_S \left (\mathbf {Q_S}, \mathbf {K_S}, \mathbf {V_S} \right ) + \mathbf {SPE}_{1} + \mathbf {SPE}_{2}(\mathbf {V_S}), \\ & \mathbf {H} = cat\left (\mathbf {H_T}, \mathbf {H_S}\right ). \end {aligned} 
(11)
Implementation.
The proposed STC plus SPE in
Eq.(11) can be readily implemented with a few lines of
codes in Python. We detail an example of the codes in Al-
gorithm 1 based on PyTorch platform. Here, we execute the
pre-defined MSA and MLP function in the standard trans-
former. The SPE is implemented by constructing the default
Embedding layer and Conv2d layer.
4. Experiments
We comprehensively evaluate the proposed STCFormer
architecture on two large-scale datasets, i.e., Human3.6M
[18] and MPI-INF-3DHP [32].
4.1. Datasets and Evaluation Metrics
Human3.6M is currently the most popular benchmark
for indoor 3D human pose estimation, which contains 11
subjects performing 15 typical actions, leading to 3.6 mil-
lion video frames in total. Following the standard protocol,
we use subjects 1, 5, 6, 7, and 8 for training, and subjects
9 and 11 for evaluation. The Mean Per Joint Position Er-
ror (MPJPE) is used to measure the error under two pro-
tocols: Protocol 1 (referred to as P1) computes MPJPE be-
tween the estimated pose and the ground truth after aligning
their root joints (hip); Protocol 2 (referred to as P2) cal-
culates Procrustes-MPJPE, where the ground truth and the
pose prediction are further aligned through a rigid transfor-
mation. We also compute the MPJPE distribution of pose to
evaluate the overall precision of the reconstructed skeletons.
MPI-INF-3DHP is a recently proposed large-scale dataset,
which consists of three scenes, i.e., green screen, non-green
screen, and outdoor.
By using 14 cameras, the dataset
records 8 actors performing 8 activities for the training
set and 7 activities for evaluation. Following the previous
works [6, 39, 54], we adopt the MPJPE (P1), percentage of
correct keypoints (PCK) with 150mm, and area under the
curve (AUC) results as the evaluation metrics.
4.2. Implementation Details
Our model is implemented with PyTorch toolkit and runs
on a server with one GTX 2080Ti GPU. In the experiments,
two kinds of input 2D pose sequences are utilized includ-
ing the pre-estimated 2D pose by the pre-trained CPN [7]
and the real 2D pose (ground truth). For model training,
we set each mini-batch as 128 sequences. The network pa-
rameters are optimized for 20 epochs by Adam optimizer
with basic learning rate of 0.001 and decayed by 0.96 af-
ter each epoch. We consider the repeat time L of modules,
the hidden embedding channel C, and the number of head
H in attention block as free parameters that we tailor to the
scale of network. The performances of the standard version
STCFormer with {L = 6, C = 256, H = 8} and the large
version STCFormer-L with {L = 6, C = 512, H = 8} are
both reported.
4.3. Performance Comparison on Human3.6M
We compare with several state-of-the-art techniques on
Human3.6M dataset. Table 1 summarizes the performance
comparisons in terms of P1 and P2 errors taking the pre-
estimated 2D poses (CPN) as input, and the number of sam-
pled frames T per video is also given for each method. In
general, the longer input sequence leads to the lower regres-
sion error. Overall, STCFormer-L with T=243 input frames
achieves the new state-of-the-art performances with P1 er-
ror of 40.5mm and P2 error of 31.8mm. Benefiting from
the proposed STC attention module, STCFormer-L outper-
forms StridedFormer [22], PATA [48] and MixSTE [52]
with T=243 frames, which are also based on transformer
architecture, by the P1 error drop of 3.2mm, 2.6mm and
0.4mm, respectively.
Comparing to the best competitor
4794

Table 1. Performance comparisons in terms of P1 error (mm) and P2 error (mm) with the state-of-the-art methods on Human3.6M dataset.
The 2D pose input is estimated by CPN [7]. The best result and runner-up result in each column are marked in red and blue, respectively.
“*” denotes the post-processing module proposed in [4]. T is the number of sampled frames from each video.
P1
Publication
Dir.
Dis.
Eat.
Gre.
Phone
Photo
Pose
Purch.
Sit.
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Liu et al. [26] (T=243)
CVPR’20
41.8
44.8
41.1
44.9
47.4
54.1
43.4
42.2
56.2
63.6
45.3
43.5
45.3
31.3
32.2
45.1
UGCN [46] (T=96) *
ECCV’20
40.2
42.5
42.6
41.1
46.7
56.7
41.4
42.3
56.2
60.4
46.3
42.2
46.2
31.7
31.0
44.5
PoseFormer [54] (T=81)
ICCV’21
41.5
44.8
39.8
42.5
46.5
51.6
42.1
42.0
53.3
60.7
45.5
43.3
46.1
31.8
32.2
44.3
Shan et al. [40] (T=243)
ACM MM’21 40.8
44.5
41.4
42.7
46.3
55.6
41.8
41.9
53.7
60.8
45.0
41.5
44.8
30.8
31.9
44.3
Anatomy3D [6] (T=243)
TCVST’21
41.4
43.5
40.1
42.9
46.6
51.9
41.7
42.3
53.9
60.2
45.4
41.7
46.0
31.5
32.7
44.1
Einfalt et al. [9] (T=351) *
arXiv’22
39.6
43.8
40.2
42.4
46.5
53.9
42.3
42.5
55.7
62.3
45.1
43.0
44.7
30.1
30.8
44.2
StridedFormer [22] (T=243) *
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
CrossFormer [13] (T=81)
arXiv’22
40.7
44.1
40.8
41.5
45.8
52.8
41.2
40.8
55.3
61.9
44.9
41.8
44.6
29.2
31.1
43.7
PATA [48] (T=243)
TIP’22
39.9
42.7
40.3
42.3
45.0
52.8
40.4
39.3
56.9
61.2
44.1
41.3
42.8
28.4
29.3
43.1
MHFormer [23] (T=351)
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
P-STMO [39] (T=243)
ECCV’22
38.9
42.7
40.4
41.1
45.6
49.7
40.9
39.9
55.5
59.4
44.9
42.2
42.7
29.4
29.4
42.8
MixSTE [52] (T=81)
CVPR’22
39.8
43.0
38.6
40.1
43.4
50.6
40.6
41.4
52.2
56.7
43.8
40.8
43.9
29.4
30.3
42.4
MixSTE [52] (T=243)
CVPR’22
37.6
40.9
37.3
39.7
42.3
49.9
40.1
39.8
51.7
55.0
42.1
39.8
41.0
27.9
27.9
40.9
STCFormer (T=81)
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
STCFormer (T=243)
39.6
41.6
37.4
38.8
43.1
51.1
39.1
39.7
51.4
57.4
41.8
38.5
40.7
27.1
28.6
41.0
STCFormer-L (T=243)
38.4
41.2
36.8
38.0
42.7
50.5
38.7
38.2
52.5
56.8
41.8
38.4
40.2
26.2
27.7
40.5
P2
Publication
Dir.
Dis.
Eat.
Gre.
Phone
Photo
Pose
Purch.
Sit.
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Liu et al. [26] (T=243)
CVPR’20
32.3
35.2
33.3
35.8
35.9
41.5
33.2
32.7
44.6
50.9
37.0
32.4
37.0
25.2
27.2
35.6
UGCN [46] (T=96) *
ECCV’20
31.8
34.3
35.4
33.5
35.4
41.7
31.1
31.6
44.4
49.0
36.4
32.2
35.0
24.9
23.0
34.5
PoseFormer [54] (T=81)
ICCV’21
34.1
36.1
34.4
37.2
36.4
42.2
34.4
33.6
45.0
52.5
37.4
33.8
37.8
25.6
27.3
36.5
Shan et al. [40] (T=243)
ACM MM’21 32.5
36.2
33.2
35.3
35.6
42.1
32.6
31.9
42.6
47.9
36.6
32.1
34.8
24.2
25.8
35.0
Anatomy3D [6] (T=243)
TCSVT’21
32.6
35.1
32.8
35.4
36.3
40.4
32.4
32.3
42.7
49.0
36.8
32.4
36.0
24.9
26.5
35.0
Einfalt et al. [9] (T=351) *
arXiv’22
32.7
36.1
33.4
36.0
36.1
42.0
33.3
33.1
45.4
50.7
37.0
34.1
35.9
24.4
25.4
35.7
StridedFormer [22] (T=243) *
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
MHFormer [23] (T=351)
CVPR’22
31.5
34.9
32.8
33.6
35.3
39.6
32.0
32.2
43.5
48.7
36.4
32.6
34.3
23.9
25.1
34.4
P-STMO [39] (T=243)
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
23.8
23.9
34.4
CrossFormer [13] (T=81)
arXiv’22
31.4
34.6
32.6
33.7
34.3
39.7
31.6
31.0
44.3
49.3
35.9
31.3
34.4
23.4
25.5
34.3
PATA [48] (T=243)
TIP’22
31.2
34.1
31.9
33.8
33.9
39.5
31.6
30.0
45.4
48.1
35.0
31.1
33.5
22.4
23.6
33.7
MixSTE [52] (T=81)
CVPR’22
32.0
34.2
31.7
33.7
34.4
39.2
32.0
31.8
42.9
46.9
35.5
32.0
34.4
23.6
25.2
33.9
MixSTE [52] (T=243)
CVPR’22
30.8
33.1
30.3
31.8
33.1
39.1
31.1
30.5
42.5
44.5
34.0
30.8
32.7
22.1
22.9
32.6
STCFormer (T=81)
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
STCFormer (T=243)
29.5
33.2
30.6
31.0
33.0
38.0
30.4
29.4
41.8
45.2
33.6
29.5
31.6
21.3
22.6
32.0
STCFormer-L (T=243)
29.3
33.0
30.7
30.6
32.7
38.2
29.7
28.8
42.2
45.0
33.3
29.4
31.5
20.9
22.3
31.8
Table 2. Performance comparisons in terms of P1 error (mm) with the state-of-the-art methods on Human3.6M dataset. The models take
the ground-truth 2D pose as input. The best result and runner-up result in each column are marked in red and blue, respectively. “*”
denotes the post-processing module proposed in [4]. T is the number of sampled frames from each video.
P1
Publication
Dir.
Dis.
Eat.
Gre.
Phone
Photo
Pose
Purch.
Sit.
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Liu et al. [26] (T=243)
CVPR’20
34.5
37.1
33.6
34.2
32.9
37.1
39.6
35.8
40.7
41.4
33.0
33.8
33.0
26.6
26.9
34.7
PoseFormer [54] (T=81)
ICCV’21
30.0
33.6
29.9
31.0
30.2
33.3
34.8
31.4
37.8
38.6
31.7
31.5
29.0
23.3
23.1
31.3
Shan et al. [40] (T=243)
ACM MM’21 29.5
30.8
28.8
29.1
30.7
35.2
31.7
27.8
34.5
36.0
30.3
29.4
28.9
24.1
24.7
30.1
MHFormer [23] (T=351)
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
P-STMO [39] (T=243)
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
StridedFormer [22] (T=243) *
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
CrossFormer [13] (T=81)
arXiv’22
26.0
30.0
26.8
26.2
28.0
31.0
30.4
29.6
35.4
37.1
28.4
27.3
26.7
20.5
19.9
28.3
PATA [48] (T=243)
TIP’22
25.8
25.2
23.3
23.5
24.0
27.4
27.9
24.4
29.3
30.1
24.9
24.1
23.3
18.6
19.7
24.7
MixSTE [52] (T=81)
CVPR’22
25.6
27.8
24.5
25.7
24.9
29.9
28.6
27.4
29.9
29.0
26.1
25.0
25.2
18.7
19.9
25.9
MixSTE [52] (T=243)
CVPR’22
21.6
22.0
20.4
21.0
20.8
24.3
24.7
21.9
26.9
24.9
21.2
21.5
20.8
14.7
15.7
21.6
STCFormer (T=81)
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
STCFormer (T=81) *
25.9
25.9
22.7
24.0
24.6
27.5
27.6
23.1
30.1
31.5
25.1
24.7
23.8
18.4
19.6
25.0
STCFormer (T=243)
21.4
22.6
21.0
21.3
23.8
26.0
24.2
20.0
28.9
28.0
22.3
21.4
20.1
14.2
15.0
22.0
STCFormer (T=243) *
20.8
21.8
20.0
20.6
23.4
25.0
23.6
19.3
27.8
26.1
21.6
20.6
19.5
14.3
15.1
21.3
MixSTE [52], our STCFormer consistently obtains better
precision across different numbers of input frames, and only
demands around half of the parameters (18.9M v.s. 33.6M).
The results verify the advantages of STC attention as an
economic and effective way to decompose the full spatio-
temporal attention. More importantly, the series of STC-
Former reaches to-date the best reported performances in
10 out of 15 categories.
Table 2 further details the comparisons between STC-
Former and the state-of-the-art models with the ground-
truth 2D pose as input. This setting excludes the noise from
2D pose estimation, and measures the upper bound of 2D-
to-3D lifting models. Accordingly, the P1 errors are obvi-
ously decreased across different methods by replacing the
CPN-estimated 2D pose with the ground-truth 2D pose, but
the performance trends are still similar. STCFormer with
post-processing attains the best P1 error of 21.3mm, which
is 0.3mm lower than the best competitor MixSTE, validat-
ing the impact of STCFormer with different types of input.
In addition to the mean error, we also compare the er-
ror distribution of STCFormer and baseline methods in Fig-
ure 4. In this experiment, the methods take the estimated
2D poses by CPN of 27 frames as input.
Compared to
the recent transformer-based approaches including Strided-
Former [22], P-STMO [39], and MHFormer [23], our STC-
Former leads to the highest number of samples with error
4795

5.83%
12.37%
16.04%
15.48%
12.98%
9.69%
6.90%
4.90%
3.71%
2.84%
9.26%
2.50%
4.50%
6.50%
8.50%
10.50%
12.50%
14.50%
16.50%
<25
25-30
30-35
35-40
40-45
45-50
50-55
55-60
60-65
65-70
>70
Proportion
Distribution
StridedFormer
P-STMO
MHFormer
STCFormer
Figure 4. Error distribution of the estimated 3D poses on Hu-
man3.6M. The horizontal axis represents the error interval, and the
vertical axis is the proportion of poses with error in the interval.
less than 35mm, and the lowest number of those with er-
ror larger than 45mm. This again confirm the advances of
STCFormer for not only obtaining the lowest average error
but also better distribution across different ranges of error.
4.4. Performance Comparison on MPI-INF-3DHP
To verify the generalization of 3D pose estimation mod-
els, we then test the performance on MPI-INF-3DHP
dataset, which contains more complex backgrounds. Fol-
lowing previous works [23, 39, 52], the ground-truth 2D
poses are taken as input. In view of the shorter video se-
quence, we set the number of input frames as 9, 27 or 81.
Table 3 lists the performance comparisons. Similar to the
observations on Human3.6M, our STCFormer with T=81
reaches the to-date best reported performance with PCK of
98.7%, AUC of 83.9% and P1 error of 23.1mm, outperform-
ing the current state-of-the-art models with a large margin
of 0.8% in PCK, 8.1% in AUC and 9.1mm in P1 error. In
particular, STCFormer shows better generalization ability
and surpasses MixSTE [52] by a much larger P1 error drop
(31.8mm) against 0.3mm on Human3.6M. This highlights
the efficacy of our method on the more complicated data.
4.5. Ablation Study
For a more in-depth analysis of our STCFormer, we fur-
ther conduct a series of ablation studies on Human3.6M
dataset using the CPN-estimated 2D poses as input.
The first group of experiments is to verify how well our
STCFormer works with different number of input frames.
Table 4 shows the detailed comparisons in terms of P1 error.
A general performance tendency is observed that increasing
T leads to monotonic performance improvement. Among
the competitive methods, our STCFormer constantly ex-
hibits the best results across 27-frame, 81-frame and 243-
frame settings. The leading performances demonstrate the
ability of STCFormer to deal with different length of video
sequence.
More remarkably, STCFormer-L has 43.7%
fewer parameters and spends 43.6% fewer FLOPs than the
runner-up MixSTE.
The second ablation study assesses the performance im-
pact of different design components. In this experiment, the
models take the estimated 2D poses by CPN of 27 frames
Table 3. Performance comparisons in terms of PCK, AUC and
P1 with the state-of-the-art methods on MPI-INF-3DHP dataset.
Here, the higher PCK, the higher AUC and the lower P1 indicate
the better regressions. The best result in each column is marked in
red. T is the number of sampled frames from each video.
Method
Publication PCK ↑AUC ↑P1(mm) ↓
UGCN [46](T=96)
ECCV’20
86.9
62.1
68.1
Anatomy3D [6] (T=81)
TCSVT’21
87.8
53.8
79.1
PoseFormer [54] (T=9)
ICCV’21
88.6
56.4
77.1
Hu et al. [16] (T=96)
ACM MM’21 97.9
69.5
42.5
CrossFormer [13] (T=9)
arXiv’22
89.1
57.5
76.3
PATA [48] (T=243)
TIP’22
90.3
57.8
69.4
MHFormer [23] (T=9)
CVPR’22
93.8
63.3
58.0
MixSTE [52] (T=27)
CVPR’22
94.4
66.5
54.9
Einfalt et al. [9] (T=81)
arXiv’22
95.4
67.6
46.9
P-STMO [39] (T=81)
ECCV’22
97.9
75.8
32.2
STCFormer (T=9)
98.2
81.5
28.2
STCFormer (T=27)
98.4
83.4
24.2
STCFormer (T=81)
98.7
83.9
23.1
Table 4. The P1 error comparisons with different number of sam-
pled frame (T) on Human3.6M dataset. The best result in each
column is marked in red.
Method
Frames T
Parameters
FLOPs (M)
P1(mm)
StridedFormer [22]
27
4.01M
163
46.9
P-STMO [39]
27
4.6M
164
46.1
MHFormer [23]
27
18.92M
1000
45.9
MixSTE [52]
27
33.61M
15402
45.1
STCFormer
27
4.75M
2173
44.1
StridedFormer [22]
81
4.06M
392
45.4
P-STMO [39]
81
5.4M
493
44.1
MHFormer [23]
81
19.67M
1561
44.5
MixSTE [52]
81
33.61M
46208
42.7
STCFormer
81
4.75M
6520
42.0
StridedFormer [22]
243
4.23M
1372
44.0
P-STMO [39]
243
6.7M
1737
42.8
MHFormer [23]
243
24.72M
4812
43.2
MixSTE [52]
243
33.61M
138623
40.9
STCFormer
243
4.75M
19561
41.0
STCFormer-L
243
18.91M
78107
40.5
as input. Spatial Attention and Temporal Attention solely
exploit the spatial pathway and temporal pathway, respec-
tively. STC only contains both pathways but without the
positional embedding. SPE1, SPE2 and SPE represent the
two SPE positional embeddings and their combination, re-
spectively. Table 5 details the contribution of each compo-
nent towards the overall performance. STC only by consid-
ering both spatial and temporal correlations leads to the er-
ror drop over solely utilizing spatial attention and temporal
attention by 218.5mm and 10.6mm, respectively. The result
indicates the importance of modeling the correlations along
two axes in parallel. The three positional embedding strate-
gies, i.e., SPE1, SPE2 and SPE, further contribute 0.6mm,
12.1mm and 12.9mm of error drop, respectively, proving
the advances of involving the structure information.
In addition to the proposed SPE1, we explore three
other positional embedding functions, i.e., Absolute Posi-
tional Embedding (APE), Centrality Positional Embedding
4796

Table 5. Performance contribution of each component in the pro-
posed STCFormer on Human3.6M dataset.
STC
SPE1
SPE2
P1 (mm)
Spatial Attention
#1
275.5
Temporal Attention
#2
67.6
STC only
#3
!
57.0
+SPE1
#4
!
!
56.4
+SPE2
#5
!
!
44.9
+SPE
#6
!
!
!
44.1
(CPE), and Symmetric Positional Embedding (SyPE). We
refer the readers to read the supplementary materials for
more details. In Table 6, we assess the performance impact
of different positional embedding functions. In this exper-
iment, the models take the estimated 2D poses by CPN of
9 frames as input. And the comparisons empirically show
the superiority of the used SPE1 (48.3mm vs.
48.7mm,
49.9mm, and 49.2mm).
Table 6. The P1 error comparisons with different positional em-
bedding functions on H