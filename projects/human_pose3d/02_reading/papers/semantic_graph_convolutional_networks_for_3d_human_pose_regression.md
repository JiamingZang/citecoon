# Semantic Graph Convolutional Networks for 3D Human Pose Regression

> 2019 · id: W2964318832 · arXiv: 1904.03345 · pdf: https://arxiv.org/pdf/1904.03345 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this paper, we study the problem of learning Graph
Convolutional Networks (GCNs) for regression. Current ar-
chitectures of GCNs are limited to the small receptive ﬁeld
of convolution ﬁlters and shared transformation matrix for
each node. To address these limitations, we propose Seman-
tic Graph Convolutional Networks (SemGCN), a novel neu-
ral network architecture that operates on regression tasks
with graph-structured data. SemGCN learns to capture se-
mantic information such as local and global node relation-
ships, which is not explicitly represented in the graph. These
semantic relationships can be learned through end-to-end
training from the ground truth without additional supervi-
sion or hand-crafted rules. We further investigate applying
SemGCN to 3D human pose regression. Our formulation is
intuitive and sufﬁcient since both 2D and 3D human poses
can be represented as a structured graph encoding the re-
lationships between joints in the skeleton of a human body.
We carry out comprehensive studies to validate our method.
The results prove that SemGCN outperforms state of the art
while using 90% fewer parameters. The code can be found
at https://github.com/garyzhao/SemGCN.

## introduction
Convolutional Neural Networks (CNNs) have success-
fully tackled classic computer vision problems such as im-
age classiﬁcation [12, 29, 31, 52], object detection [19, 46,
55, 63, 74, 79] and generation [43, 58, 71, 73, 80], where the
input image has a grid-like structure. However, many real-
world tasks, e.g., molecular structures, social networks and
3D meshes, can only be represented in the form of irregular
structures, where CNNs have limited applications.
In order to address this limitation, Graph Convolutional
Networks (GCNs) [17, 28, 49] have been introduced re-
cently as a generalization of CNNs that can directly deal
with a general class of graphs. They have achieved state-
of-the-art performance when applied to 3D mesh defor-
mation [45, 64], image captioning [70], scene understand-
ing [68], and video recognition [66, 67]. These works uti-
lize GCNs to model relations of visual objects for classiﬁ-
cation. In this paper, we investigate using deep GCNs for
regression, which is another core problem of computer vi-
sion with many real-world applications.
However, GCNs cannot be directly applied to regression
problems due to the following limitations in baseline meth-
ods [28, 64, 67]. First, to handle the issue that graph nodes
may have various numbers of neighborhoods, the convolu-
tion ﬁlter shares the same weight matrix for all nodes, which
is not comparable with CNNs. Second, previous methods
are simpliﬁed by restricting the ﬁlters to operate in a one-
step neighborhood around each node according to the guid-
ance of [28]. The receptive ﬁeld of the convolution kernel
is limited to one due to this formulation, which severely
impairs the efﬁciency of information exchanging especially
when the network goes deeper.
In this work, we propose a novel graph neural network
architecture for regression called Semantic Graph Convo-
lutional Networks (SemGCN) to address the above limi-
tations. Speciﬁcally, we investigate learning semantic in-
formation encoded in a given graph, i.e., the local and
global relations of nodes, which is not well-studied in pre-
vious works. SemGCN does not rely on hand-crafted con-
straints [10, 13, 51] to analyze the patterns for a speciﬁc ap-
plication, and thus can be easily generalized to other tasks.
In particular, we study SemGCN for 2D to 3D human
pose regression. Given a 2D human pose (and the optional
relevant image) as input, we aim to predict the locations
of its corresponding 3D joints in a certain coordinate space.
Using SemGCN to formulate this problem is intuitive. Both
2D and 3D poses are able to be naturally represented by
a canonical skeleton in the form of 2D or 3D coordinates,
and SemGCN can explicitly exploit their spatial relations,
which are crucial for understanding human actions [67].
Our work makes the following contributions. First, we
propose an improved graph convolution operation called Se-
mantic Graph Convolution (SemGConv) which is derived
from CNNs. The key idea is to learn channel-wise weights
for edges as priors implied in the graph, and then com-
bine them with kernel matrices. This signiﬁcantly improves
the power of graph convolutions.
Second, we introduce
1
arXiv:1904.03345v3  [cs.CV]  8 Mar 2020

SemGCN where SemGConv and non-local [65] layers are
interleaved. This architecture captures both local and global
relationships among nodes. Third, we present an end-to-end
learning framework to show that SemGCN can also incor-
porate external information, such as image content, to fur-
ther boost the performance for 3D human pose regression.
The effectiveness of our approach is validated by com-
prehensive evaluation with a rigorous ablation study and
comparisons with state of the art on standard 3D bench-
marks. Our approach matches the performance of state-of-
the-art techniques on Human3.6M [24] using only 2D joint
coordinates as inputs and 90% fewer parameters. Mean-
while, our approach outperforms state of the art when in-
corporating image features. Furthermore, we also show the
visual results of SemGCN, which demonstrate the effec-
tiveness of our approach qualitatively. Note that the pro-
posed framework can be easily generalized to other regres-
sion tasks, and we leave this for future work.

## method
# of params
MPJPE (mm)
aGCN [68] / GAT [60]
0.16M
82.9
ST-GCN [67]
0.27M
57.4
FC [34]
4.29M
45.5 (62.9)
FC [34] w/ PG [13]
-
43.3 (60.4)
Ours
0.43M
43.8 (60.8)
Ours w/ PG [13]
-
42.5 (59.8)
Table 3. Evaluation of 2D to 3D pose regression on Human3.6M
datasets [24]. Errors within the parentheses are computed by using
the 2D estimations from HG [38] as inputs during training and test-
ing. Otherwise, 2D ground truth is utilized. Our method advances
other GCN-based approaches by 20% and achieves the state-of-
the-art performance using 90% fewer parameters than [34].
5.3. Ablation Study
We conduct the ablation study on the proposed method in
Sect. 3. Conﬁguration #1 is employed. Our SemGCN con-
sists of two main components: SemGConv and non-local
layers. To verify them, we train two variants of SemGCN:
one only uses SemGConv and the other only uses non-local
layers.
Then we evaluate them together with the base-
line method in Sect. 3.1 (ResGCN) and our full model in
Sect. 3.3 on Human3.6M. Note that in order to get rid of the
inﬂuence from the 2D pose detector, we report the results
using 2D ground truth for training and testing.
All models are trained based on the architecture shown
in Fig. 2 after 200 epochs. Results are shown in Table 1. We
also show their curves of training losses and testing errors
in Fig. 4. We can see that our model with more components
performs better than those with fewer components, which
indicates the efﬁcacy of each part of our algorithm. More-
over, our networks with SemGConv have much smoother
training curves which demonstrates that learning local rela-
tions among nodes stabilizes the training process as well.
5.4. Evaluation on 3D Human Pose Regression
2D to 3D pose regression. We ﬁrst evaluate our method
for 2D to 3D pose regression and only Conﬁguration #1 is
leveraged. We compared ours with three GCN-based meth-
ods: aGCN [68], GAT [60] and ST-GCN [67], and two
state-of-the-art approaches: FC [34] and PG [13]. As ST-
GCN [67] is designed for videos, we set its temporal di-
mension to one for images. PG proposed a framework to
reﬁne the 3D pose, which is complementary to FC and ours.
Therefore, we also report our results reﬁned by PG.
The results are reported in Table 3. Our approach out-
performs other GCN-based approaches by a large margin
(about 20%). More importantly, our method achieves the
state-of-the-art performance with around 90% fewer param-
eters than [34]. Meanwhile, the runtime of SemGCN re-
duces 10% compared with [34], which is around 1.8ms for
a forward pass on a Titan Xp GPU. After we reﬁned our
results by PG, our approach obtains the best performance.
Comparison with the state of the art. We show evalua-
tion results under Conﬁguration #1 and #2. Note that many
leading methods have sophisticated frameworks or learning
strategies. Some of them aim at in-the-wild images [54, 69,
75] or exploit temporal information [11, 18, 21, 57], while
some other approaches use complex loss functions [53, 69].
These methods are with different research targets compared
to ours. Therefore, we include some of them during evalua-
tion for completeness. Table 2 reports the results.
7

Figure 5. Visual results of our method on Human3.6M [24] and MPII [3]. The ﬁrst three rows show results on Human3.6M. Results on
MPII are drawn in the last three rows. The bottom row shows four typical failure cases. Best viewed in color.
We ﬁnd that our method using only 2D joints as inputs
is able to match the state-of-the-art performance. After in-
corporating image features, our network sets the new state
of the art. Especially, we improve previous methods by a
large margin for the action of directions, taking photo, pos-
ing, sitting down, walking dog and walking together. We
hypothesize that this is due to the severe self-occlusions in
these actions, while they can be effectively encoded by our
SemGCN using relations within graphs. The result of our
method trained and tested with ground truth 2D joint loca-
tions shows our upper bound.
Qualitative results. In Fig. 5, we show the visual re-
sults of our method on Human3.6M and the test set of MPII.
MPII contains in-the-wild images with novel human poses
which are not similar to the examples in Human3.6M. As
seen, our method is able to accurately predict 3D pose for
both indoor and most in-the-wild images. It indicates that
SemGCN can effectively encode relationships among joints
and further generalize them to some novel cases.
The bottom row of Fig. 5 also shows typical failure cases
of our method. These images include extreme poses which
are largely different from those in Human3.6M. Our method
failed to handle them but still yields reasonable 3D poses.

## experiments
In this section, we ﬁrst introduce settings and implemen-
tation details for evaluation, and then conduct an ablation
study on components in our method, and ﬁnally report our
results and comparisons with state-of-the-art methods.
5.1. Implementation Details
As suggested in the previous works [34, 53, 75], it is
impossible to train an algorithm to infer the 3D joint loca-
tions in an arbitrary coordinate space system. Therefore,
we choose to predict 3D pose in the camera coordinate sys-
tem [11, 32, 41, 57], which makes the 2D to 3D regression
problem similar across different cameras.
We make use of the ground truth 2D joint locations pro-
vided in the dataset to align the 3D and 2D poses following
the setting of [75]. This implies that we implicitly use the
camera calibration information. Then, we zero-center both
the 2D and 3D poses around the predeﬁned root joint, i.e.,
the pelvis joint, which is in line with previous works and the
standard protocol. Moreover, we do not use data augmenta-
tion during the training process for simplicity.
Network training.
We use ResNet50 in [54] as our
backbone network, which is compatible with the integral
loss and pre-trained on ImageNet [9]. During training, we
5

employ Adam [27] for optimization with a initial learning
rate of 0.001 and use mini-batches of size 64. The learning
rate is dropped with a decay rate of 0.5 when the loss on the
validation set saturates. We initialize weights of the graph
network using the initialization described in [16].
In our preliminary experiments, we observe that the di-
rect end-to-end training of the whole network from scratch
cannot achieve the best performance. We argue that this
is likely because of the highly non-linear dependency be-
tween the graph network and conventional deep convolu-
tional module for 2D pose estimation. Therefore, we utilize
a multi-stage training scheme which is more stable and ef-
fective in practice. We ﬁrst train the backbone network for
2D pose estimation from images using 2D ground truth. As
described in [54], the integral loss is used. Then we ﬁx the
2D pose estimation module and train the graph network for
2D to 3D pose regression using the output of 2D estima-
tion module and the 3D ground truth. In this stage, the loss
function deﬁned in Eq. 7 is employed. At last, the whole
network is ﬁne-tuned with all data. Both integral loss and
Eq. 7 are activated. Note that the ﬁnal stage is end-to-end.
5.2. Datasets and Evaluation Protocols
Our proposed approach is comprehensively evaluated on
the most widely used dataset for 3D human pose estimation:
Human3.6M [24], following the standard protocol.
Datasets. Human3.6M [24] is currently the largest pub-
licly available dataset for 3D human pose estimation. This
dataset contains 3.6 million of images captured by a MoCap
System in an indoor environment, where 7 professional ac-
tors perform 15 everyday activities such as walking, eating,
sitting, making a phone call and engaging in a discussion.
Both 2D and 3D ground truth are available for supervised
learning. Following the setting of [75], the videos are down-
sampled from 50fps to 10fps for both the training and test-
ing sets to reduce redundancy. We also use MPII dataset [3],
the state-of-the-art benchmark for 2D human pose estima-
tion, for pre-training the 2D pose detector and qualitatively
evaluation in the experiment.
Evaluation protocols. For Human3.6M [24], there are
two common evaluation protocols using different training
and testing data split in the literature. One standard pro-
tocol uses all 4 camera views in subjects S1, S5, S6, S7
and S8 for training and the same 4 camera views in sub-
jects S9 and S11 for testing. Errors are calculated after the
ground truth and predictions are aligned with the root joint.
We refer to this as Protocol #1. The other protocol makes
use of six subjects S1, S5, S6, S7, S8 and S9 for training,
and evaluation is performed on every 64th frame of S11. It
also utilizes a rigid transformation to further align the pre-
dictions with the ground truth. This protocol is referred as
Protocol #2. In this work, we use Protocol #1 in all the ex-
periments for evaluation, since it is more challenging and
20
40
60
80
100
120
140
# of Epochs
0
0.01
0.02
0.03
0.04
Training Loss
ResGCN
Ours w/o SemGConv
Ours w/o Non-Local
Ours (SemGCN)
20
40
60
80
100
120
140
# of Epochs
50
100
150
200
250
300
350
MPJPE (mm)
ResGCN
Ours w/o SemGConv
Ours w/o Non-Local
Ours (SemGCN)
Figure 4. Training curves (left) and testing errors (right) of our
networks with different settings. Our full model has lower and
smoother learning curves as well as better testing results.

## related_work
Graph convolutional networks. Generalizing CNNs to
inputs with graph-like structures is an important topic in the
ﬁeld of deep learning. In the literature, there have been
several attempts to use recursive neural networks to pro-
cess data represented in graph domains as directed acyclic
graphs [14]. GNNs were introduced in [17, 28, 49] as a
more common solution to handle arbitrary graph data. The
principle of constructing GCNs on graph generally follows
two streams: the spectral perspective and the spatial per-
spective. Our work belongs to the second stream [28, 39,
60], where the convolution ﬁlters are applied directly on the
graph nodes and their neighbors.
Recent studies on computer vision have achieved state-
of-the-art performance by leveraging GCNs to model the
relations among visual objects [68, 70] or temporal se-
quences [66, 67]. This paper follows the spirit of them,
while we explore applying GCNs for regression tasks, es-
pecially, 2D to 3D human pose regression.
3D pose estimation. Lee and Chen [30] ﬁrst investi-
gated inferring 3D joints from their corresponding 2D pro-
jections. Later approaches either exploited nearest neigh-
bors to reﬁne the results of pose inference [18, 25] or ex-
tracted hand-crafted features [1, 23, 47] for later regres-
sion.
Other methods created over-complete bases which
are suitable for representing human poses as sparse com-
binations [2, 4, 44, 62, 77]. More and more studies focus
on making use of deep neural networks to ﬁnd the map-
ping between 2D and 3D joint locations. A couple of al-
gorithms directly predicted 3D pose from the image [75],
while others combined 2D heatmaps with volumetric repre-
sentation [41], pairwise distance matrix estimation [36] or
image cues [56] for 3D human pose regression.
Recently, it has been proven that 2D pose information
is crucial for 3D pose estimation. Martinez et al. [34] in-
troduced a simple yet effective method which predicted 3D
key points purely based on 2D detections. Fang et al. [13]
further extended this approach through pose grammar net-
works. These works focus on 2D to 3D pose regression,
which are most relevant to the context of this paper.
Other methods use synthetic datasets which are gen-
erated from deforming a human template model with the
ground truth [8, 42, 48] or introduce loss functions involv-
ing high-level knowledge [40, 53, 69] in addition to joints.
They are complementary to the others. Remaining works
target at exploiting temporal information [11, 18, 21, 57]
for 3D pose regression. They are out of the scope of this
paper, since we aim at handling the 2D pose from one sin-
gle image. However, our method can be easily extended to
sequence inputs, and we leave it for future work.
3. Semantic Graph Convolutional Networks
We propose a novel graph network architecture to han-
dle general regression tasks involving data that can be rep-
resented in the form of graphs. We ﬁrst provide the back-
ground of GCNs and related baseline method. Then we in-
troduce the detailed design of SemGCN.
We assume that graph data share the same topological
structure, such as human skeletons [10, 26, 61, 67], 3D
morphable models [33, 45, 72] and citation networks [50].
Other problems which own different graph structures in
the same domain, e.g., protein-protein interaction [60] and
quantum chemistry [15], are out of the scope of this paper.
This assumption makes it possible to learn priors implied in
the graph structure, which motivates SemGCN.
3.1. ResGCN: A Baseline
We will start by brieﬂy recapping the ‘vanilla’ GCNs as
proposed in [28]. Let G = {V, E} denote a graph where V
is the set of K nodes and E are edges, while −→
x (l)
i
∈RDl
and −→
x (l+1)
i
∈RDl+1 are the representations of node i be-
fore and after the l-th convolution respectively. A graph
based convolutional propagation can be applied to node i
in two steps. First, node representations are transformed
by a learnable parameter matrix W ∈RDl+1×Dl. Sec-
ond, these transformed node representations are gathered to
node i from its neighboring nodes j ∈N(i), followed by
a non-linear function (ReLU [37]). If node representations
are collected into a matrix X(l) ∈RDl×K, the convolution
operation can be written as:
X(l+1) = σ

WX(l) ˜A

,
(1)
where ˜A is symmetrically normalized from A in conven-
tional GCNs. A ∈[0, 1]K×K is the adjacency matrix of G,
and we have αij = 1 for node j ∈N(i) and αii = 1.
2

𝑤"
𝑤#
𝑤$
𝑤%
𝑤&
𝑤'
𝑤(
𝑤)
𝑤*
𝑎"
𝑎#
𝑎$
𝑎%
𝑎&
𝑎'
𝑎(
𝑎)
𝑎*
≈𝑊∗
𝑤"
𝑤"
𝑤"
𝑤"
𝑤"
𝑤"
𝑊∗
𝑎"
𝑎#
𝑎&
𝑎$
𝑎'
𝑎%
𝑊∗
𝑎"
𝑎#
𝑎&
𝑎$
𝑎'
𝑎%
(a)
(c)
(b)
(d)
Figure 1. Illustration of the proposed Semantic Graph Convolutions. (a) The 3×3 convolution kernel of CNNs (highlighted in green) learns
a different transformation matrix wi for each position inside the kernel. We approximate it by learning a weighting vector ai for each
position and a shared transformation matrix W. (b) Conventional GCNs only learn a shared transformation matrix w0 for all nodes. (c)
The approximated formulation in (a) can be directly extended to (b): we add an additional learnable weight ai for each node in the graph.
(d) We further extend (c) to learn a channel-wise weighting vector ai for each node. After combining them with the vanilla transformation
matrix W in GCNs, we can obtain a new kernel operation for graphs which owns comparable learning capability with CNNs. The learned
weight vectors show the local semantic relationships of neighboring nodes implied in the graph.
Wang et al. [64] rephrased a very deep graph network
based on Eq. 1 with residual connections [20] to learn the
mapping between image features and 3D vertexes.
We
adopt its network architecture and treat it as our baseline
which is denoted as ResGCN.
There are two clear drawbacks in Eq. 1. First, in order to
make the graph convolution work on nodes with arbitrary
topologies, the learned kernel matrix W is shared for all
edges. As a result, the relationships of neighboring nodes,
or the internal structure in the graph, is not well exploited.
Second, previous works only collect features from the ﬁrst-
order neighbors of each node. This is also limited because
the receptive ﬁeld is ﬁxed to 1.
3.2. Semantic Graph Convolutions
We show that learning semantic relationships of neigh-
boring nodes implied in edges of the graph is effective to
address the limitation of the shared kernel matrix.
The proposed approach builds on concepts from CNNs.
Fig. 1(a) shows a CNN with a convolution kernel of size
3 × 3. It learns nine transformation matrices which are dif-
ferent from each other to encode features inside the kernel
in the spatial dimension. This makes the operation own ex-
pressive power to model feature patterns contained in im-
ages. We ﬁnd that this formulation can be approximated by
learning a weighting vector −→
a i for each position, and then
combining them with a shared transformation matrix W. If
we represent the image feature map as a square grid graph
whose nodes represent pixels, this approximated formula-
tion can be directly extended to GCNs as shown in Fig. 1(c).
To this end, we propose Semantic Graph Convolution
(SemGConv), where we add a learnable weighting matrix
M ∈RK×K to conventional graph convolutions. And then
Eq. 1 is transformed to:
X(l+1) = σ

WX(l)ρi
 M ⊙A

,
(2)
where ρi is Softmax nonlinearity which normalizes the in-
put matrix across all choices of node i; ⊙is an element-
wise operation which returns mij if aij = 1 or negatives
with large exponents saturating to zero after ρi; A serves as
a mask which forces that for node i in the graph, we only
compute the weights of its neighboring nodes j ∈N(i).
As illustrated in Fig. 1(d), we can further extend Eq. 2 by
learning a set of Md ∈RK×K, so that a different weighting
matrix is applied to each channel d of output node features:
X(l+1) =
Dl+1

d=1
σ
−→
wdX(l)ρi
 Md ⊙A

,
(3)
where ∥represents channel-wise concatenation, and −→
wd is
the d-th row of the transformation matrix W.
Comparison to previous GCNs. Both aGCN [68] and
GAT [60] follow a self-attention strategy [59] to compute
the hidden representations of each node in the graph by at-
tending over its neighbors. They aim to estimate a weight-
ing function depending on inputs for edges to modulate in-
formation ﬂow throughout the graph. By contrast, we target
at learning input-independent weights for edges which rep-
resent priors implied in the graph structures, e.g., how one
joint inﬂuences other body parts in human pose estimation.
The edge importance weighting mask introduced in ST-
GCN [67] is the most related work to ours but with follow-
ing two sharp differences. First, no Softmax nonlinearity
is leveraged after weighting by [67], while we ﬁnd it stabi-
lizes the training and obtains better results, since the contri-
butions of nodes to their neighbors are normalized by Soft-
max. Second, ST-GCN applies only one single learnable
mask to all channels, but our Eq. 3 learns channel-wise dif-
ferent weights for edges. As a r

## conclusion
We present a novel model for 3D human pose regression,
the Semantic Graph Convolutional Networks (SemGCN).
Our method has addressed the key challenges of GCNs by
learning local and global semantic relations among nodes
in the graph. The combination of SemGCN and features
pooled from image content further improves the perfor-
mance in 3D human pose estimation. Comprehensive eval-
uation results show that our network obtains state-of-the-
art performance with 90% fewer parameters compared with
the closest work. The proposed SemGCN also opens up
many possible directions for future works. For example,
how to incorporate temporal information, such as videos,
8

into SemGCN becomes a natural question.
Acknowledgments. This work was funded partly by grant
BAAAFOSR-2013-0001 to Dimitris Metaxas. This work
was also partly supported by NSF 1763523, 1747778,
1733843 and 1703883 Awards.
Mubbasir Kapadia was
funded partly by NSF IIS-1703883, NSF S&AS-1723869,
and DARPA SocialSim-W911NF-17-C-0098.