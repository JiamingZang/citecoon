# HDFormer: High-order Directed Transformer for 3D Human Pose Estimation

> 2023 · id: W4385767582 · arXiv: 2302.01825 · pdf: https://www.ijcai.org/proceedings/2023/0065.pdf · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Human pose estimation is a challenging task due
to its structured data sequence nature.
Existing
methods primarily focus on pair-wise interaction
of body joints, which is insufﬁcient for scenarios
involving overlapping joints and rapidly changing
poses. To overcome these issues, we introduce a
novel approach, the High-order Directed Trans-
former (HDFormer), which leverages high-order
bone and joint relationships for improved pose esti-
mation. Speciﬁcally, HDFormer incorporates both
self-attention and high-order attention to formu-
late a multi-order attention module. This module
facilitates ﬁrst-order ”joint↔joint”, second-order
”bone↔joint”, and high-order ”hyperbone↔joint”
interactions, effectively addressing issues in com-
plex and occlusion-heavy situations. In addi-
tion, modern CNN techniques are integrated into
the transformer-based architecture, balancing the
trade-off between performance and efﬁciency. HD-
Former signiﬁcantly outperforms state-of-the-art
(SOTA) models on Human3.6M and MPI-INF-
3DHP datasets, requiring only 1/10 of the param-
eters and signiﬁcantly lower computational costs.
Moreover, HDFormer demonstrates broad real-
world applicability, enabling real-time, accurate 3D
pose estimation.1
1

## introduction
Despite signiﬁcant strides in deep learning-based 3D pose es-
timation [Iskakov et al., 2019; Qiu et al., 2019; Pavllo et al.,
2018; Li et al., 2020; Zhu et al., 2021; Gong et al., 2021;
Ye et al., 2022], achieving stable, accurate pose sequences
remains elusive. The prevalent 3D pose estimation frame-
work takes 2D pose detection results [Chen et al., 2017;
Sun et al., 2019] as inputs and estimates depth information
via end-to-end Graph Convolutional Networks (GCNs)[Cai
et al., 2019; Pavllo et al., 2018] or Transformers[Zhang et
∗Denotes equal contributions
†W. Xiang is the corresponding author
1The source code is in https://github.com/hyer/HDFormer
(a) joint <-> joint
(b) bone <-> joint
(c) hyperbone <-> joint
Figure 1: Illustration of ﬁrst-order (joint↔joint) attention, second-
order (bone↔joint) and high-order (hyperbone↔joint) attention.
The ﬁrst-order attention models the connections between joints,
while the second-order attention focuses on the relationship between
joints and bones. The high-order attention, on the other hand, further
delves into the intricate relationships between joints and hyperbones.
al., 2022]. However, complex scenarios involving overlap-
ping keypoints, rapid pose changes, and varying scales pose
challenges to the depth estimation of 3D keypoints.
Face with these challenges, existing methods mainly uti-
lize ﬁrst-order ”joint ↔joint” and second-order ”bone ↔
joint” connections, often overlooking high-order interactions
among joint sets (referred to as hyperbones).
Different
from ﬁrst-order (“joint ↔joint”) and second-order (“bone
↔joint”) that focus on pair-wise joints/bones connections,
high-order relations could describe complex motion dynam-
ics. The high-order interactions contain rich semantic infor-
mation in motions, as skeletons often move in speciﬁc pat-
terns and involve multiple joints and bones simultaneously.
Learning high-order information without expensive costs
is a challenging problem in 3D pose estimation.
To ad-
dress this issue, we propose a novel framework named High-
order Directed Transformer (HDFormer), which coherently
exploits the multi-order information aggregation of skeleton
structure for 3D pose estimation.
Speciﬁcally, HDFormer
leverages the ﬁrst-order attention to learn the spatial seman-
tics among “joint ↔joint” relationships. Additionally, it in-
tegrates a robust high-order attention module to enhance 3D
pose estimation accuracy by capturing both second-order and
high-order information. To encode the hyperbone features,
the hyperbone representation encoding module is employed
under the constraints of a pre-deﬁned directed human skele-
ton graph. With innovative designs and modern deep learn-
ing techniques, HDFormer strikes a commendable balance
arXiv:2302.01825v2  [cs.CV]  22 May 2023

between efﬁcacy and efﬁciency. In summary, the key con-
tributions of this paper are summarized as follows:
• We investigate high-order attention module to learn both
the “bone↔joint” and “hyperbone↔joint” with an effec-
tive and efﬁcient cross-attention mechanism. To the best
of our knowledge, it is the ﬁrst end-to-end model to utilize
high-order information on a directed skeleton graph for 3D
pose estimation.
• We propose a novel High-order Directed Transformer (HD-
Former) for 3D pose estimation. It utilizes “joint↔joint”,
“bone↔joint” and “hyperbone↔joint” information with a
three-stage U-shape architecture design, which endows the
network with the ability to handle more complex scenarios.
• HDFormer is evaluated on popular 3D pose estimation
benchmarks Human3.6M and MPI-INF-3DHP with anal-
ysis of quantitative and qualitative results. Speciﬁcally, it
achieves 21.6% (96 frames) on Human3.6M without using
any extra data, which outperforms the existing SOTA work
MixSTE [Zhang et al., 2022] with only 1/10 parameters
and a fraction of computational cost.
2

## method
MPJPE[↓]
Params
Frames
1
HDFormer (w/o Ψ)
27.4
3.7 M
96
2
HDFormer (with pos encoding)
22.1
3.8 M
96
3
HDFormer (multi-head concat)
22.9
3.7 M
96
4
HDFormer (T=243)
21.8
4.7 M
96
5
HDFormer (proposed)
21.6
3.7 M
96
Exploration of Hyperbone Representation. The hyperbone
representation is a vital factor for the graph skeleton struc-
ture, and we exploit the way of the hyperbone feature repre-
sentation. We conducted experiments by adopting 4th order
HDFormer block with different instantiations modes, which
includes summation, multiplication, concatenation, and sub-
traction+concatenation as can be shown in Table 6. Com-
pared to the baseline method, we found that all the hyperbone
representation methods outperform the baseline, as they uti-
lize high-order information. Among them, subtraction + con-
catenation boosts the performance over baseline by 4.0mm.
It shows that bone feature concatenation with shortest path
aggregation is effective for hyperbone feature representation.
Role of Position Encoding and Multi-Head Attention. We
observed from our experimental results (line 2 of Table 7)
that incorporating absolute positional encoding led to a de-
crease in performance. This suggests that position coding
is not helpful in improving performance. Besides, we have
conducted an ablation study on the use of concatenation and
summation in the multi-head attention module, and we found
that summation resulted in better performance, which can be
shown in line 3 of Table 7. Consequently, we adopted the
summation in the multi-head attention in our proposed model.
Longer Frames as Input. Extending the input frame num-
bers to 243 led to a slight decline compared to 96 frames when
using 2D ground truth input as shown in line 4 of 7. We sus-
pect that the reason might be the small scale of our model
(1/10 compared to [Zhang et al., 2022]) may not well capture
temporal redundancy and noise in dense sequence.
5

## experiments
4.1
Datasets and Metric
Experiments are conducted on the 3D pose estimation bench-
mark dataset Human3.6M[Ionescu et al., 2014] and MPI-
INF-3DHP [Mehta et al., 2016a]. Human3.6M is the most
widely used evaluation benchmark, containing 3.6 million
video frames captured from four synchronized cameras with

X
Y
Z
2D pose sequence
+
+
+
+
+
HDFormer Block
First-order Attention Block
Temporal Upsampling
Temporal Downsampling
FC
DownSampling Stage
UpSampling Stage
Merging Stage
[B, 2, J, T]
[B, 16, J, T]
[B, 32, J, T/2]
[B, 64, J, T/4]
[B, 128, J, T/8]
[B, 128, J, T/8]
[B, 64, J, T/4]
[B, 32, J, T/2]
[B, 16, J, T]
[B, 16, J, T]
[B, 16, J, T]
[B, 16, J, T]
[B, 3, J, T]
[B, 16, J, T]
[B, 32, J, T/2]
[B, 64, J, T/4]
[B, 128, J, T/8]
[B, 16, J, T]
3D pose sequence
Figure 3: Overview of our framework: A High-order Directed Transformer with a U-shaped design for 3D human pose estimation. The
framework includes downsampling, upsampling, and merging stages, incorporating high-order attention and multi-scale temporal information.
different locations and poses at 50 Hz. 11 subjects are per-
forming 15 kinds of actions. MPI-INF-3DHP is a 3D human
body pose estimation dataset consisting of both constrained
indoor and complex outdoor scenes.
It consists of 1.3M
frames captured from the 14 cameras. For fair comparisons,
the evaluation metric MPJPE is adopted in this work, which
follows the setting of the previous works [Hu et al., 2021;
Cai et al., 2019; Zhang et al., 2022; Zhao et al., 2022]. Unlike
the 2D pose estimation task, MPJPE is proposed to evaluate
models comprehensively for accuracy and stability.
4.2
Implementation Details
The proposed HDFormer is implemented with the PyTorch
platform and all the experiments are conducted on a single
NVIDIA TITAN V100 GPU. We optimized the model by the
AdaMod optimizer [Ding et al., 2019] for 110 epochs with a
batch size of 256, and the base learning rate is 5 × 10−3 with
decayed by 0.1 at 80, 90, and 100 epochs. To avoid over-
ﬁtting, we set the weight decay factor to 10−5 for parameters
of convolution layers and the dropout rate to 0.3 at part of the
layers. Besides, we followed UGCN [Wang et al., 2020] to
apply the sliding window algorithm with a step length of 5
to estimate a variable-length pose sequence with ﬁxed input
length at inference time.
4.3
Quantitative Evaluation
Results on Human3.6M. The proposed approach is com-
pared with the state-of-the-art methods to evaluate the per-
formance. In this subsection, the reported performance in
their original paper is directly copied as their results. The per-
formance comparison with the state-of-the-art works on Hu-
man3.6M [Ionescu et al., 2014] is listed in Table 1, including
graph ConvNet-based and Transformer-based methods. For
fair comparisons with other SOTA methods, we consider not
only the effectiveness of the model but also the scale of pa-
rameters and latency in Table 2, which can comprehensively
demonstrate the real-world performance of the model. To our
knowledge, this is the ﬁrst comprehensive comparison ex-
periment on the benchmark dataset Human3.6M. The current
SOTA method MixSTE [Zhang et al., 2022], a transformer-
based model, achieves 25.9% and 21.6% MPJPE with the
input of 81 and 243-frame sequences, respectively.
Com-
pared to MixSTE, the proposed HDFormer achieves 21.6%
MPJPE with the input of only 96 frames. More importantly,
our model has only a 1/10 scale of 3.7 M vs. 33.8 M and six
times the speed. The graph ConvNet-based SOTA method
U-CondDGCN has a very small scale of parameters, latency,
and ideal performance. However, the proposed HDFormer
is a transformer-based method that achieves signiﬁcant im-
provement compared to U-CondDGCN with a very close
scale of parameters and same-level latency.
The compar-
isons powerfully demonstrate the effectiveness and efﬁciency
of HDFormer.
Results on MPI-INF-3DHP. In Table 3, we compared our
method with state-of-the-art methods on the MPI-INF-3DHP
benchmark to evaluate the generalization ability of the pro-
posed HDFormer.
We take the ground-truth 2D poses as
model input. Our method achieves the same trends as the
results on Human3.6M, which is also the SOTA performance
under the metric of PCK, AUC, and MPJPE.
4.4
Qualitative Results
As shown in Figure 4, we further conduct visualization on
the First-order attention and High-order attention. The se-
lected action (Eating of test set S9) is applied for visualiza-
tion. For the First-order attention map in Figure 4(a), the hor-
izontal and vertical axes are all joint indexes, and it can be
easily observed that the dependency between the spine node
and left/right elbow nodes are signiﬁcant for the “eating” se-
quence. Besides, the left shoulder node also plays an impor-
tant role in the spatial relationship with the left ankle node
when eating in the sitting pose. So the ﬁrst-order attention
with self-attention of joints can capture the joint spatial rela-
tionship effectively.
Furthermore, to demonstrate the effect of the proposed
high-order attention block, we further visualize the high-
order attention map for the action of eating from the test set
S9 in Figure 4(b), where the vertical axes were the index of
the joint while the horizontal axes were the index of hyper-
bones. In our experiments, the maximum SPD length was
4. As a result, the hyperbone sequence has 42 bone features
in the horizontal axes. From the attention map, we can ﬁnd

Table 1: Quantitative comparisons with state-of-the-art methods on Human3.6M under protocol #1 and protocol #2, where methods marked
with † are video-based; T denotes the number of input frames; and CPN and HR-Net denote the input 2D poses are estimated by [Chen et al.,
2017] and [Sun et al., 2019], respectively. The best results of CPN and HR-Net are marked in red and blue, respectively.
Protocol #1
Dir.
Disc
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Cai [Cai et al., 2019] (CPN, T=7)
44.6
47.4
45.6
48.8
50.8
59.0
47.2
43.9
57.9
61.9
49.7
46.6
51.3
37.1
39.4
48.8
Pavllo [Pavllo et al., 2018] (CPN, T=243)
45.2
46.7
43.3
45.6
48.1
55.1
44.6
44.3
57.3
65.8
47.1
44.0
49.0
32.8
33.9
46.8
Xu [Xu et al., 2020] (CPN, T=9)
37.4
43.5
42.7
42.7
46.6
59.7
41.3
45.1
52.7
60.2
45.8
43.1
47.7
33.7
37.1
45.6
Liu [Liu et al., 2020](CPN, T=243)
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
Wang [Wang et al., 2020] (CPN, T=96)
41.3
43.9
44.0
42.2
48.0
57.1
42.2
43.2
57.3
61.3
47.0
43.5
47.0
32.6
31.8
45.6
Hu [Hu et al., 2021] (CPN, T=96)
38.0
43.3
39.1
39.4
45.8
53.6
41.4
41.4
55.5
61.9
44.6
41.9
44.5
31.6
29.4
43.4
Zhang [Zhang et al., 2022] (CPN, T=81)
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
Wang [Wang et al., 2020] (HR-Net, T=96)
38.2
41.0
45.9
39.7
41.4
51.4
41.6
41.4
52.0
57.4
41.8
44.4
41.6
33.1
30.0
42.6
Hu [Hu et al., 2021] (HR-Net, T=96)
35.5
41.3
36.6
39.1
42.4
49.0
39.9
37.0
51.9
63.3
40.9
41.3
40.3
29.8
28.9
41.1
Zhang [Zhang et al., 2022] (HR-Net, T=243)
36.7
39.0
36.5
39.4
40.2
44.9
39.8
36.9
47.9
54.8
39.6
37.8
39.3
29.7
30.6
39.8
HDFormer(CPN, T=96)
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
HDFormer (HR-Net, T=96)
34.7
41.7
36.0
38.4
41.1
45.3
39.6
37.4
49.0
63.1
39.8
38.9
40.2
29.3
29.1
40.3
Protocol #2
Dir.
Disc
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Cai [Cai et al., 2019] (CPN, T=7)
35.7
37.8
36.9
40.7
39.6
45.2
37.4
34.5
46.9
50.1
40.5
36.1
41.0
29.6
33.2
39.0
Pavllo [Pavllo et al., 2018] (CPN, T=243)
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
Xu [Xu et al., 2020] (CPN, T=9)
31.0
34.8
34.7
34.4
36.2
43.9
31.6
33.5
42.3
49.0
37.1
33.0
39.1
26.9
31.9
36.2
Liu [Liu et al., 2020](CPN, T=243)
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
Wang [Wang et al., 2020] (CPN, T=96)
32.9
35.2
35.6
34.4
36.4
42.7
31.2
32.5
45.6
50.2
37.3
32.8
36.3
26.0
23.9
35.5
Hu [Hu et al., 2021] (CPN, T=96)
29.8
34.4
31.9
31.5
35.1
40.0
30.3
30.8
42.6
49.0
35.9
31.8
35.0
25.7
23.6
33.8
Zhang [Zhang et al., 2022] (CPN, T=81)
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
Wang [Wang et al., 2020] (HR-Net, T=96)
28.4
32.5
34.4
32.3
32.5
40.9
30.4
29.3
42.6
45.2
33.0
32.0
33.2
24.2
22.9
32.7
Hu [Hu et al., 2021] (HR-Net, T=96)
27.7
32.7
29.4
31.3
32.5
37.2
29.3
28.5
39.2
50.9
32.9
31.4
32.1
23.6
22.8
32.1
Zhang [Zhang et al., 2022] (HR-Net, T=243)
28.0
30.9
28.6
30.7
30.4
34.6
28.6
28.1
37.1
47.3
30.5
29.7
30.5
21.6
20.0
30.6
HDFormer (CPN, T=96)
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
HDFormer (HR-Net, T=96)
27.9
32.8
29.7
30.6
32.5
35.0
28.9
29.2
38.3
50.0
32.9
30.1
31.8
23.6
22.8
31.7
Table 2: Results on Human3.6M with ground-truth 2D poses as in-
put. Our method with subtraction feature representation is marked
with *. The latency is measured with batch size = 1.

## related_work
Directed Skeleton Graph. The directed skeleton graph G
shown in the right part of Figure 2 represents the human
skeleton structure, where the nodes are human skeleton joints
and the arrows are human skeleton bones.
Generally, the
human skeleton can be represented as a graph G = (V; E),
where the vertices are human skeleton joints and edges are
physical connections between two joints. Here V is the set
of N joints and E is characterized by the adjacency matrix
A ∈RN×N . The raw pose data, i.e., the joint keypoints
vector, is a set of 2D coordinates. In this way, the pose data
is transformed into a graph sequence and speciﬁcally repre-
sented as a tensor X ∈RT×N×C, where T, N, and C denote
the temporal length, numbers of joints and channels, respec-
tively. We use the directed graph because it allows for a con-
venient hyperbone deﬁnition.
Transformer. Our model’s attention mechanism is built
upon the original implementation of the classic Transformer
[Vaswani et al., 2017]. The attention computing with query,
key and value matrix Q, K, V in each head are depicted as:
Attn(Q, K, V ) = Softmax((QKT + A + Ψ)/
p
dm)V, (1)
where Q, K, V ∈RN×dm, N is the number of tokens, and
dm indicates the dimension of each token. The multi-head
attention of S heads is deﬁned as follows:
ℏi = Attn(Qi, Ki, Vi), i ∈{1, . . . , S},
(2)
MSA = Concat(ℏ1, ..., ℏS)Wo,
(3)
where ℏis the attention calculation result for a single head,
Wo ∈Rdm×dm is the linear projection weight. A is the ad-
jacency matrix and Ψ is a learnable adjacency matrix. The
matrix A is ﬁxed and represents the predetermined connec-
tions between joints, while the learnable adjacency matrix Ψ
adjusts the connection weights based on the input data, im-
proving the capturing of spatial relationships between differ-
ent joints. The ablation study on the impact of Ψ is presented
in line 1 of Table 7.

Linear
Linear
Linear
Linear
Hyperbone
Representation
x
Scale
x
Softmax
Joint Feature
High-order Directed Transformer
Linear
x
Softmax
Linear
Linear
x
+
A
+
res(1x1)
First-order
Attention
First-order attention block
Norm
Norm Joint Feature
Hyperbone Representation Learning
Norm+MLP
+
Joint Feature
Distance of Shortest Path
（SPD）
Linear Projection
...
...
...
Norm Joint Feature
Instantiation Function 𝝓(#)
Joint 
Feature 
Selection
：are the hyperbone
features generated by ℋ!,#
Skeleton Directed Graph 𝓖
ℬ!,# = SPD(𝒢, 𝑖, 𝑗)
ℋ!,#
………
𝝓(ℋ!,#)
Concatenate
Scale
Figure 2: The illustration of High-order Direction Transformer (HDFormer) block. HDFormer block consists of three major parts: (a)
First-order attention block to capture “joint↔joint” spatial relationship; (b) Hyperbone representation learning module to encode hyperbone
features; (c) High-order attention block to capture both second-order “bone ↔joint” and high-order “hyperbone↔joint” interactions.
3.2
High-order Directed Transformer
The
spatial
connections
between
“joint↔joint”
and
“joint↔bone” are referred to as ﬁrst-order and second-
order information in the 3D pose estimation, which is widely
studied in the previous works [Zhang et al., 2022]. Neverthe-
less, pairwise ﬁrst-order and second-order information alone
cannot fully describe the complex human skeleton dynamics
in the 2D to 3D mappings. For example, human skeletons
often move in speciﬁc patterns and involve multiple joints
and bones at the same time. This observation leads us to
further investigate the high-order information interaction of
the human skeleton by integrating the high-order attention
learning with directed graph and propose a High-order Direct
Transformer (HDFormer) for 3D pose estimation.
First-order Attention Modeling. The joint sets of the skele-
ton describe the rough posture of the human body, and the
global multi-head attention [Zhang et al., 2022] has demon-
strated its effectiveness in 3D pose estimation. Therefore,
the multi-head attention scheme is adopted in this work for
ﬁrst-order attention modeling. As illustrated in Figure 2(a),
A ∈RN×N is the adjacency matrix and Ψ is the learn-
able adjacency matrix which has the same dimension as A.
Speciﬁcally, given the joint token set Z = {z1 . . . zi} where
i ∈N denotes the index of skeleton joints, zi ∈RC, C
represents the feature channel. The ﬁrst-order self-attention
modeling and feature of joints can be obtained by following
Eq. 1, where query, key, and value matrices are generated by
three linear layers, Q = WqZ, K = WkZ, and V = WvZ,
respectively. S represents the number of heads and ℏk is the
output of each head. Unlike the traditional multi-head atten-
tion fuses the output of the attention module with concate-
nating (Eq. 3), we revamp the fusion scheme with a simple
accumulation:
ˆZ =
S
X
k=1
ℏk,
(4)
where ˆZ denote the ﬁnal output of ﬁrst-order attention.
Hyperbone Representation. In this section, we outline the
process of constructing and learning the hyperbone represen-
tation. A hyperbone is a series of joints and bones that are
connected sequentially. The human skeleton can be repre-
sented as a special type of graph without loops, allowing for
the unique determination of the shortest path between two
joints. Given a starting and ending joint, the corresponding
hyperbone can be identiﬁed using the distance of the short-
est path (SPD). Speciﬁcally, as shown in Figure 2, the human
skeleton can be described as a directed graph G. The “hip”
joint is deﬁned as the directed graph’s root node. Given two
joint nodes on the directed graph, we could follow the direc-
tion of edges to ﬁnd the shortest path from starting joint i to
j. For example, there is a shortest path from joint index 0 to
joint index 3 by [0, 1, 2, 3], which is done by moving from
index 0 to index 3 following the edges of the directed graph
(bone for human skeleton).
Formally, given the human skeleton-directed graph G and
the (start, end) joint indices (i, j), we could utilize the short-
est path algorithm (SPD) to discover the joint set belonging
to the hyperbone Bi,j = {vhi, vhi+1, . . . , vhj}, where v∗rep-
resents the human joint, |Bi,j| = n represents the number of
joints in hyperbone, and we call this the order of hyperbone,
h∗is the joint index:
Bi,j = SPD(G, i, j),
(5)
To encode hyperbone features, we propose a novel hy-
perbone encoding method. Speciﬁcally, the feature of hy-
perbone can be obtained by a function φ(·) that takes hy-
perbone joint set Bi,j’s corresponding features Hi,j
=
{zhi, zhi+1, . . . , zhj} as input and generate hyperbone fea-
tures, where zhi is the feature of joint vhi.
Instantiation. Previous works, e.g. Anatomy [Chen et al.,
2021], have used a simple subtraction of joint features to con-
struct bone features. In contrast, we propose a general process
for constructing both bone and hyperbone features, and offer
instantiation methods. Speciﬁcally, we investigate several in-
stantiations of the function φ(·).

Subtraction. φ(·) can be deﬁned as a subtraction operation.
As we use a directed graph to represent the human skeleton
when adopting subtraction for hyperbone representation, it is
equivalent to the subtraction of start and end joints:
φ(Hi,j) = f(zhi −zhj),
(6)
where z is the joint feature, f is a linear mapping. This repre-
sentation is easy to calculate and works ﬁne for second-order
bone representation, however, it loses information on bone
sequence for hyperbone with higher order.
Summation/Multiplication.
φ(·) can also be deﬁned as
element-wise summation or multiplication for joints:
φ(Hi,j) =
X
z∈Hi,j
f(z)/n,
(7)
φ(Hi,j) =
Y
z∈Hi,j
f(z),
(8)
where n is the number of joints, f is a linear mapping.
Concatenation. φ(·) can be deﬁned with concatenation and
linear mapping:
φ(Hi,j) = f([zh1, . . . , zhn]),
(9)
where the operator [·] represents the concatenation of features
in the shortest path, f maps the concatenated feature to the
same dimension as the joint feature.
Sub-Concat. To overcome the sequence information loss
issue in subtraction, we combine subtraction and concatena-
tion for a mixed function for φ(·):
φ(Hi,j) = f([zh1 −zh2, . . . , zhn−1 −zhn]),
(10)
where the second-order bone feature is calculated with sub-
traction and the high-order hyperbone feature is obtained with
concatenation and linear mapping.
High-order Directed Transformer. Figure 2(b) illustrates
the architecture of our proposed High-order Directed Trans-
former block, which consists of three components: ﬁrst-order
attention block, hyperbone encoding block, and high-order
attention block. The cross-attention fusion involves joint fea-
tures ˆZ from ﬁrst-order attention modeling block and hyper-
bone feature H = [Y2, ..Yo, ..Yn], where Yo represent hyper-
bone features with order o from hyperbone encoding block.
Formally, the cross-attention fusion can be expressed as:
Yo = [φ(Hi,j)], |Hi,j| = o,
H = [Y2, . . . , Yn],
Qh = Wqh ˆZ, Kh = WkhH, Vh = WvhH,
CrossAttn(Qh, Kh, Vh) = Softmax(QhKT
h /
p
dm)Vh,
(11)
where Wqh, Wkh, Wvh are learnable parameters. Since we
only use the joint feature in the query, the computation and
m

## conclusion
In this work, we propose a novel model named High-order
Directed Transformer (HDFormer), which considers both
“joint↔joint”, “bone↔joint” and “hyperbone↔joint” con-
nections.
Speciﬁcally, we propose a hyperbone represen-
tation learning module and a high-order attention module
to model complicated semantic relations between hyperbone
and joint. We conduct extensive experiments to provide both
quantitative and qualitative analysis. Our proposed method
achieves state-of-the-art performances with only 1/10 param-
eters and a fraction of computational cost compared to re-
cently published SOTA.
Acknowledgments
The research work of Zhi-Qi Cheng in this project received sup-
port from the US Department of Transportation, Ofﬁce of the As-
sistant Secretary for Research and Technology, under the Univer-
sity Transportation Center Program with Federal Grant Number
69A3551747111. Additional support came from the Intel and IBM
Fellowships. The views and conclusions contained herein represent
those of the authors and not necessarily the ofﬁcial policies or en-
dorsements of the supporting agencies or the U.S. Government.