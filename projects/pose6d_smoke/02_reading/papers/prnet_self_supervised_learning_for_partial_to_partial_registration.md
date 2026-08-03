# PRNet: Self-Supervised Learning for Partial-to-Partial Registration

> 2019 · id: W2971088236 · arXiv: 1910.12240 · pdf: https://arxiv.org/pdf/1910.12240 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present a simple, ﬂexible, and general framework titled Partial Registration Net-
work (PRNet), for partial-to-partial point cloud registration. Inspired by recently-
proposed learning-based methods for registration, we use deep networks to tackle
non-convexity of the alignment and partial correspondence problems. While previ-
ous learning-based methods assume the entire shape is visible, PRNet is suitable for
partial-to-partial registration, outperforming PointNetLK, DCP, and non-learning
methods on synthetic data. PRNet is self-supervised, jointly learning an appropriate
geometric representation, a keypoint detector that ﬁnds points in common between
partial views, and keypoint-to-keypoint correspondences. We show PRNet predicts
keypoints and correspondences consistently across views and objects. Furthermore,
the learned representation is transferable to classiﬁcation.
1

## introduction
Registration is the problem of predicting a rigid motion aligning one point cloud to another. Algo-
rithms for this task have steadily improved, using machinery from vision, graphics, and optimization.
These methods, however, are usually orders of magnitude slower than “vanilla” Iterative Closest
Point (ICP), and some have hyperparameters that must be tuned case-by-case. The trade-off between
efﬁciency and effectiveness is steep, reducing generalizability and/or practicality.
Recently, PointNetLK [1] and Deep Closest Point (DCP) [2] show that learning-based registration
can be faster and more robust than classical methods, even when trained on different datasets. These
methods, however, cannot handle partial-to-partial registration, and their one-shot constructions
preclude reﬁnement of the predicted alignment.
We introduce the Partial Registration Network (PRNet), a sequential decision-making framework
designed to solve a broad class of registration problems. Like ICP, our method is designed to be
applied iteratively, enabling coarse-to-ﬁne reﬁnement of an initial registration estimate. A critical
new component of our framework is a keypoint detection sub-module, which identiﬁes points that
match in the input point clouds based on co-contextual information. Partial-to-partial point cloud
registration then boils down to detecting keypoints the two point clouds have in common, matching
these keypoints to one another, and solving the Procrustes problem.
Since PRNet is designed to be applied iteratively, we use Gumbel–Softmax [3] with a straight-through
gradient estimator to sample keypoint correspondences. This new architecture and learning procedure
modulates the sharpness of the matching; distant point clouds given to PRNet can be coarsely matched
using a diffuse (fuzzy) matching, while the ﬁnal reﬁnement iterations prefer sharper maps. Rather
than introducing another hyperparameter, PRNet uses a sub-network to predict the temperature [4] of
the Gumbel–Softmax correspondence, which can be cast as a simpliﬁed version of the actor-critic
method. That is, PRNet learns to modulate the level of map sharpness each time it is applied.
33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.
arXiv:1910.12240v2  [cs.LG]  29 Oct 2019

We train and test PRNet on ModelNet40 and on real data. We visualize the keypoints and corre-
spondences for shapes from the same or different categories. We transfer the learned representations
to shape classiﬁcation using a linear SVM, achieving comparable performance to state-of-the-art
supervised methods on ModelNet40.
Contributions.
We summarize our key contributions as follows:
• We present the Partial Registration Network (PRNet), which enables partial-to-partial point cloud
registration using deep networks with state-of-the-art performance.
• We use Gumbel–Softmax with straight-through gradient estimation to obtain a sharp and near-
differentiable mapping function.
• We design an actor-critic closest point module to modulate the sharpness of the correspondence
using an action network and a value network. This module predicts more accurate rigid transfor-
mations than differentiable soft correspondence methods with ﬁxed parameters.
• We show registration is a useful proxy task to learn representations for 3D shapes. Our representa-
tions can be transferred to other tasks, including keypoint detection, correspondence prediction,
and shape classiﬁcation.
• We release our code to facilitate reproducibility and future research. 1
2

## method
We establish preliminaries about the rigid alignment problem and related algorithms in §3.1; then, we
present PRNet in §3.2. For ease of comparison to previous work, we use the same notation as [2].
3.1
Preliminaries: Registration, ICP, and DCP
Consider two point clouds X = {x1, . . . , xi, . . . , xN} ⊂R3 and Y = {y1, . . . , yj, . . . , yM} ⊂R3.
The basic task in rigid registration is to ﬁnd a rotation RXY and translation tXY that rigidly align X
to Y. When M = N, ICP and its peers approach this task by minimizing the objective function
E(RXY, tXY, m)= 1
N
N
X
i=1
∥RXYxi + tXY −ym(xi)∥2.
(1)
Here, the rigid transformation is deﬁned by a pair [RXY, tXY], where RXY ∈SO(3) and tXY ∈R3;
m maps from points in X to points in Y. Assuming m is ﬁxed, the alignment in (1) is given in
closed-form by
RXY = V U ⊤
and
tXY = −RXYx + y,
(2)
where U and V are obtained using the singular value decomposition (SVD) H = USV ⊤, with
H = PN
i=1(xi −x)(ym(xi) −y)⊤. In this expression, centroids of X and Y are deﬁned as
x = 1
N
PN
i=1 xi and y = 1
N
PN
i=1 ym(xi), respectively.
We can understand ICP and the more recent learning-based DCP method [2] as providing different
choices of m:
Iterative Closest Point. ICP chooses m to minimize (1) with [RXY, tXY] ﬁxed, yielding:
m(xi, Y) = arg min
j
∥RXYxi + tXY −yj∥2
(3)
3

ICP approaches a ﬁxed point by alternating between (2) and (3); each step decreases the objective (1).
Since (1) is non-convex, however, there is no guarantee that ICP reaches a global optimum.
Deep Closest Point. DCP uses deep networks to learn m. In this method, X and Y are embedded
using learned functions FX and FY deﬁned by a Siamese DGCNN [22]; these lifted point clouds
are optionally contextualized by a Transformer module [59], yielding embeddings ΦX and ΦY. The
mapping m is then
m(xi, Y) = softmax(ΦYΦ⊤
xi).
(4)
This formula is applied in one shot followed by (2) to obtain the rigid alignment. The loss used to
train this pipeline is mean-squared error (MSE) between ground-truth rigid motion from synthetically-
rotated point clouds and prediction; the network is trained end-to-end.
3.2
Partial Registration Network
DCP is a one-shot algorithm, in that a single pass through the network determines the output for each
prediction task. Analogously to ICP, PRNet is designed to be iterative; multiple passes of a point
cloud through PRNet reﬁne the alignment. The steps of PRNet, illustrated in Figure 1, are as follows:
1. take as input point clouds X and Y;
2. detect keypoints of X and Y;
3. predict a mapping from keypoints of X to keypoints of Y;
4. predict a rigid transformation [RXY, tXY] aligning X to Y based on the keypoints and map;
5. transform X using the obtained transformation;
6. return to 1 using the pair (RXYX + tXY, Y) as input.
When predicting a mapping from keypoints in X to keypoints in Y, PRNet uses Gumbel–Softmax
[3] to sample a matching matrix, which is sharper than (4) and approximately differentiable. It has a
value network to predict a temperature for Gumbel–Softmax, so that the whole framework can be
seen as an actor-critic method. We present details of and justiﬁcations behind the design below.
Notation. Denote by X p = {xp
1, . . . , xp
i , . . . , xp
N} the rigid motion of X to align to Y after p
applications of PRNet; X 1 and Y1 are initial input shapes. We will use [Rp
XY, tp
XY] to denote the
p-th rigid motion predicted by PRNet for the input pair (X, Y).
Since our training pairs are synthetically generated, before applying PRNet we know the ground-
truth [R∗
XY, t∗
XY] aligning X to Y. From these values, during training we can compute “local”
ground-truth [Rp∗
XY, tp∗
XY] on-the-ﬂy, which maps the current (X p, Y) to the best alignment:
Rp∗
XY = R∗
XYR1...p⊤
XY
and
tp∗
XY = t∗
XY −Rp∗
XYt1...p
XY ,
(5)
where
R1...p
XY = Rp−1
XY . . . R1
XY
and
t1...p
XY = Rp−1
XY t1...p−1
XY
+ tp−1
XY .
(6)
We use mp to denote the mapping function in p-th step.
Synthesizing the notation above, X p is given by
xp
i = Rp−1
XY xp−1
i
+ tp−1
XY
(7)
where
Rp
XY = V pU p⊤
and
tp
XY = −Rp
XYxp + y.
(8)
In this equation, U p and V p are computed using (2) from X p, Yp, and mp.
Keypoint Detection. For partial-to-partial registration, usually N̸ = M and only subsets of X and
Y match to one another. To detect these mutually-shared patches, we design a simple yet efﬁcient
keypoint detection module based on the observation that the L2 norms of features tend to indicate
whether a point is important.
Using X p
k and Yp
k to denote the k keypoints for X p and Yp, we take
X p
k = X p(topk(∥Φp
x1∥2, . . . , ∥Φp
xi∥2, . . . , ∥Φp
xN ∥2))
Yp
k = Yp(topk(∥Φp
y1∥2, . . . , ∥Φp
yi∥2, . . . , ∥Φp
yM ∥2))
(9)
4

where topk(·) extracts the indices of the k largest elements of the given input. Here, Φ denotes
embeddings learned by DGCNN and Transformer.
By aligning only the keypoints, we remove irrelevant points from the two input clouds that are not
shared in the partial correspondence. In particular, we can now solve the Procrustes problem that
matches keypoints of X and Y. We show in §4.3 that although we do not provide explicit supervision,
PRNet still learns how to detect keypoints reasonably.
Gumbel–Softmax Sampler. One key observation in ICP and DCP is that (3) usually is not differen-
tiable with respect to the map m but by deﬁnition yields a sharp correspondence between the points in
X and the points in Y. In contrast, the smooth function (4) in DCP is differentiable, but in exchange
for this differentiability the mapping is blurred. We desire the best of both worlds: A potentially
sharp mapping function that admits backpropagation.
To that end, we use Gumbel–Softmax [3] to sample a matching matrix. Using a straight-through
gradient estimator, this module is approximately differentiable. In particular, the Gumbel–Softmax
mapping function is given by
mp(xi, Y) = one hot

arg max
j
softmax(Φp
YΦp⊤
xi + gij)

,
(10)
where (gi1, . . . , gij, . . . , giN) are i.i.d. samples drawn from Gumbel(0, 1). The map in (10) is not
differentiable due to the discontinuity of arg max, but the straight-through gradient estimator [60]
yields (biased) subgradient estimates with low variance. Following their methodology, on backward
evaluation of the computational graph, we use (4) to compute
∂L
∂Φp
∗, ignoring the one hot operator and
the arg max term.
Actor-Critic Closest Point (ACP). The mapping functions (4) and (10) have ﬁxed “temperatures,”
that is, there is no control over the sharpness of the mapping matrix mp. In PRNet, we wish to adapt
the sharpness of the map based on the alignment of the two shapes. In particular, for low values of
p (the initial iterations of alignment) we may satisﬁed with high-entropy approximate matchings
that obtain a coarse alignment; later during iterative evaluations, we can sharpen the map to align
individual pairs of points.
To make this intuition compatible with PRNet’s learning-based architecture, we add a parameter λ to
(10) to yield a generalized Gumbel–Softmax matching matrix:
mp(xi, Y) = one hot
"
arg max
j
softmax
 
Φp
YΦp⊤
xi + gij
λ
!#
(11)
When λ is large, the map matrix mp is smoothed out; as λ →0 the map approaches a binary matrix.
It is difﬁcult to choose a single λ that sufﬁces for all (X, Y) pairs; rather, we wish λ to be chosen
adaptively and automatically to extract the best alignment for each pair of point clouds. Hence, we
use a small network Θ to predict λ based on global features Ψp
X and Ψp
Y aggregated from Φp
X and
Φp
Y channel-wise by global pooling (averaging). In particular, we take λ = Θ(Ψp
X , Ψp
Y), where
Ψp
X = avgiΦp
xi and Ψp
Y = avgiΦp
yi. In the parlance of reinforcement learning, this choice can be
seen as a simpliﬁed version of actor-critic method. Φp
X and Φp
Y are learned jointly with DGCNN [22]
and Transformer [59]; then an actor head outputs a rigid motion, where (11) uses the λ predicted
from a critic head.
Loss Function. The ﬁnal loss L is the summation of several terms Lp, indexed by the number p of
passes through PRNet for the input pair. Lp consists of three terms: a rigid motion loss Lm
p , a cycle
consistency loss Lc
p, and a global feature alignment loss Lg. We also introduce a discount factor
γ < 1 to promote alignment within the ﬁrst few passes through PRNet; during training we pass each
input pair through PRNet P times.
Combining the terms above, we have
L =
P
X
p=1
γp−1Lp,
where
Lp = Lm
p + αLc
p + βLp
g.
(12)
The rigid motion loss Lm
p is,
Lm
p = ∥Rp⊤
XYRp∗
XY −I∥2 + ∥tp
XY −tp∗
XY∥2
(13)
5

## experiments
Our experiments are divided into four parts. First, we show performance of PRNet on a partial-to-
partial registration task on synthetic data in §4.1. Then, we show PRNet can generalize to real data in
§4.2. Third, we visualize the keypoints and correspondences predicted by PRNet in §4.3. Finally, we
show a linear SVM trained on representations learned by PRNet can achieve comparable results to
supervised learning methods in §4.4.
4.1
Partial-to-Partial Registration on ModelNet40
We evaluate partial-to-partial registration on ModelNet40 [62]. There are 12,311 CAD models
spanning 40 object categories, split to 9,843 for training and 2,468 for testing. Point clouds are
sampled from the CAD models by farthest-point sampling on the surface. During training, a point
cloud with 1024 points X is sampled. Along each axis, we randomly draw a rigid transformation; the
rotation along each axis is sampled in [0, 45◦] and translation is in [−0.5, 0.5]. We apply the rigid
transformation to X, leading to Y. We simulate partial scans of X and Y by randomly placing a point
in space and computing its 768 nearest neighbors in X and Y respectively.
We measure mean squared error (MSE), root mean squared error (RMSE), mean absolute error
(MAE), and coefﬁcient of determination (R2). Angular measurements are in units of degrees. MSE,
RMSE and MAE should be zero while R2 should be one if the rigid alignment is perfect. We compare
our model to ICP, Go-ICP [14], Fast Global Registration (FGR) [61], and DCP [2].
6

## related_work
Rigid Registration. ICP [5] and variants [6, 7, 8, 9] have been widely used for registration. Recently,
probabilistic models [10, 11, 12] have been proposed to handle uncertainty and partiality. Another
trend is to improve the optimization: [13] applies Levenberg—Marquardt to the ICP objective, while
global methods seek a solution using branch-and-bound [14], Riemannian optimization [15], convex
relaxation [16], mixed-integer programming [17], and semideﬁnite programming [18].
Learning on Point Clouds and 3D Shapes. Deep Sets [19] and PointNet [20] pioneered deep
learning on point sets, a challenge problem in learning and vision. These methods take coordinates
as input, embed them to high-dimensional space using shared multilayer perceptrons (MLPs), and
use a symmetric function (e.g., max or P) to aggregate features. Follow-up works incorporate local
information, including PointNet++ [21], DGCNN [22], PointCNN [23], and PCNN [24]. Another
branch of 3D learning designs convolution-like operations for shapes or applies graph convolutional
networks (GCNs) [25, 26] to triangle meshes [27, 28], exemplifying architectures on non-Euclidean
data termed geometric deep learning [29]. Other works, including SPLATNet [30], SplineCNN [31],
KPConv [32], and GWCNN [33], transform 3D shapes to regular grids for feature learning.
Keypoints and Correspondence. Correspondence and registration are dual tasks. Correspondence is
the approach while registration is the output, or vice versa. Countless efforts tackle the correspondence
problem, either at the point-to-point or part-to-part level. Due to the O(n2) complexity of point-to-
point correspondence matrices and O(n!) possible permutations, most methods (e.g., [34, 35, 36,
37, 38, 39, 40, 41]) compute a sparse set of correspondences and extend them to dense maps, often
with bijectivity as an assumption or regularizer. Other efforts use more exotic representations of
correspondences. For example, functional maps [42] generalize to mappings between functions on
shapes rather than points on shapes, expressing a map as a linear operator in the Laplace–Beltrami
eigenbasis. Mathematical methods like functional maps can be made ‘deep’ using priors learned from
data: Deep functional maps [43, 44] learn descriptors rather than designing them by hand.
For partial-to-partial registration, we cannot compute bijective correspondences, invalidating many
past representations. Instead, keypoint detection is more secure. To extract a sparser representation,
KeyPointNet [45] uses registration and multiview consistency as supervision to learn a keypoint
detector on 2D images; our method performs keypoint detection on point clouds. In contrast to our
model, which learns correspondences from registration, [46] uses correspondence prediction as the
training objective to learn how to segment parts. In particular, it utilizes PointNet++ [21] to product
point-wise features, generates matching using a correspondence proposal module, and ﬁnally trains
the pipeline with ground-truth correspondences.
Self-supervised Learning. Humans learn knowledge not only from teachers but also by predicting
and reasoning about unlabeled information. Inspired by this observation, self-supervised learning
1https://github.com/WangYueFt/prnet
2

(a) Network architecture for PRNet
(b) ACP
Figure 1: Network architecture for PRNet and ACP.
usually involves predicting part of an input from another part [47, 48], solving one task using features
learned from another task [45] and/or enforcing consistency from different views/modalities [49, 50].
Self-supervised pretraining is an effective way to transfer knowledge learned from massive unlabeled
data to tasks where labeled data is limited. For example, BERT [51] surpasses state-of-the-art
in natural language processing by learning from contextual information. ImageNet Pretrain [52]
commonly provides initialization for vision tasks. Video-audio joint analysis [53, 54, 55] utilizes
modality consistency to learn representations. Our method is also self-supervised, in the sense that
no labeled data is needed.
Actor–Critic Methods. Many recent works can be counted as actor–critic methods, including deep
reinforcement learning [56], generative modeling [57], and sequence generation [58]. These methods
generally involve two functions: taking actions and estimating values. The predicted values can be
used to improve the actions while the values are collected when the models interact with environment.
PRNet uses a sub-module (value head) to predict the level of granularity at which we should map two
shapes. The value adjusts the temperature of Gumbel–Softmax in the action head.
3

## conclusion
PRNet tackles a general partial-to-partial registration problem, leveraging self-supervised learning to
learn geometric priors directly from data. The success of PRNet veriﬁes the sensibility of applying
learning to partial matching as well as the speciﬁc choice of Gumbel–Softmax, which we hope can
inspire additional work linking discrete optimization to deep learning. PRNet is also a reinforcement
learning-like framework; this connection between registration and reinforcement learning may provide
inspiration for additional interdisciplinary research related to rigid/non-rigid registration.
Our experiments suggest several avenues for future work. For example, as shown in Figure 6, the
matchings computed by PRNet are not bijective, evident e.g. in the point clouds of cars and chairs.
One possible extension of our work to address this issue is to use Gumbel–Sinkhorn [72] to encourage
bijectivity. Improving the efﬁciency of PRNet when applied to real scans also will be extremely
valuable. As described in §4.2, PRNet currently requires inference-time ﬁne-tuning on real scans
to learn useful data-dependent representations; this makes PRNet slow during inference. Seeking
universal representations that generalize over broader sets of registration tasks will improve the speed
and generalizability of learning-based registration. Another possibility for future work is to improve
the scalability of PRNet to deal with large-scale real scans captured by LiDAR.
Finally, we hope to ﬁnd more applications of PRNet beyond the use cases we have shown in the paper.
A key direction bridging PRNet to applications will involve incorporating our method into SLAM or
structure-from-motion can demonstrate its value for robotics applications and robustness to realistic
species of noise. Additionally, we can test the effectiveness of PRNet for registration problems in
medical imaging and/or high-energy particle physics.
6
Acknowledgements
The authors acknowledge the generous support of Army Research Ofﬁce grant W911NF1710068, Air
Force Ofﬁce of Scientiﬁc Research award FA9550-19-1-031, of National Science Foundation grant
IIS-1838071, from an Amazon Research Award, from the MIT-IBM Watson AI Laboratory, from the
Toyota-CSAIL Joint Research Center, from a gift from Adobe Systems, and from the Skoltech-MIT
Next Generation Program. Any opinions, ﬁndings, and conclusions or recommendations expressed in
this material are those of the authors and do not necessarily reﬂect the views of these organizations.
The authors also thank members of MIT Geometric Data Processing group for helpful discussion and
feedback on the paper.