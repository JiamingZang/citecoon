# COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation

> 2026 · id: arxiv:2603.00493 · arXiv: 2603.00493 · pdf: https://arxiv.org/pdf/2603.00493 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised
Single-reference Novel Object Pose Estimation
Yuchen Che1
Jingtu Wu1
Hao Zheng2
Asako Kanezaki1,2,3
1Institute of Science Tokyo
2RIKEN
3Tohoku University
{che.y.99ea@m, wu.j.baa4@m, kanezaki@comp}.isct.ac.jp, hao.zheng@riken.jp
Query
Ref
Query
Ref
Query
Ref
(a)
(b)
(c)
Optimal Transport
(d)
Point-wise 
Confidence
as Marginals
Feature
Similarity
as Kernel
Cross-view Correspondence
Query
Ref
Query
Ref
Query
Ref
Confidence
1
0
Figure 1. Given a novel object’s query and reference RGB-D images (a), COG outputs point-wise confidence and cross-view soft corre-
spondence (b), to estimate the relative pose between query and reference (c). To achieve this, we formulate correspondence finding as an
optimal transport problem, with each point’s confidence as target marginals, and the point features’ similarity as an affinity kernel (d).
Abstract
Estimating the 6DoF pose of a novel object with a sin-
gle reference view is challenging due to occlusions, view-
point changes, and outliers. A core difficulty lies in find-
ing robust cross-view correspondences, as existing meth-
ods often rely on discrete one-to-one matching that is non-
differentiable and tends to collapse onto sparse keypoints.
We propose Confidence-aware Optimal Geometric Corre-
spondence (COG), an unsupervised framework that formu-
lates correspondence estimation as a confidence-aware op-
timal transport problem. COG produces balanced soft cor-
respondences by predicting point-wise confidences and in-
jecting them as optimal transport marginals, suppressing
non-overlapping regions. Semantic priors from vision foun-
dation models further regularize the correspondences, lead-
ing to stable pose estimation. This design integrates con-
fidence into the correspondence finding and pose estima-
tion pipeline, enabling unsupervised learning. Experiments
show unsupervised COG achieves comparable performance
to supervised methods, and supervised COG outperforms
them. Codes: https://github.com/YC-Che/COG
1. Introduction
Object pose estimation, which aims to recover an object’s
6DoF pose (rotation and translation) from RGB-D images,
is a fundamental task for robotics [20, 21, 62], augmented
reality [42, 57], and 3D scene understanding [30, 74]. To
enable general real-world deployment, improving the gen-
eralization capability of object pose estimation has long
been a central goal. Early instance-level methods [28, 64,
66, 71] assume access to CAD models or training objects
identical to those at test time. This assumption was relaxed
in category-level pose estimation [7, 32, 67, 68], where
models learn to estimate the poses of unseen instances
within a few predefined categories.
Recent research has advanced toward novel object pose
estimation [9, 27, 35, 39, 44, 58], which targets poses of ar-
bitrary objects with dataset-agnostic generalization. How-
ever, these methods often depend on CAD models or mul-
tiple reference views, which limits scalability in practice.
A more challenging setting considers only a single refer-
ence image [18, 40, 41, 46] (see Fig. 1), where large view-
point changes and partial observations require the network
to jointly infer valid overlapping regions and object pose,
1
arXiv:2603.00493v1  [cs.CV]  28 Feb 2026

making the problem particularly ill-posed. A key to solving
this task is to establish reliable cross-view correspondences,
since the pose can be recovered by aligning geometric struc-
tures between query and reference views. Yet, most existing
approaches [39, 41] construct correspondences via discrete
one-to-one assignments (e.g. argmax), which tend to col-
lapse onto a few dominant keypoints, leaving many of the
points unused. Moreover, such non-differentiable assign-
ments breaks differentiability and prevents the model from
being trained in an unsupervised manner.
We introduce Confidence-aware Optimal Geometric
Correspondence (COG), an unsupervised framework for
novel object pose estimation from a single reference image.
COG addresses the above issues by formulating soft cor-
respondence as an optimal transport (OT) problem, where
point-wise confidences are predicted beforehand and ex-
plicitly incorporated as target marginals of the transport
plan. Compared to OT-based methods [13, 55, 72] with
uniform marginals and only apply confidence post hoc,
This formulation yields globally balanced correspondences
that naturally suppresses outliers and non-overlapping re-
gions. Given these correspondences, corresponding points
are generated via convex combinations, and a weighted
SVD solver [63] is used to recover the pose transforma-
tion. The entire process forms an end-to-end correspon-
dence finding and pose estimation pipeline, enabling un-
supervised optimization of both correspondence and con-
fidence. To mitigate ambiguity in purely geometric match-
ing, we integrate semantic priors denoised from vision foun-
dation models such as DINO [5, 48], which softly encour-
age correspondences between semantically consistent parts.
Furthermore, for unsupervised confidence learning, Gaus-
sian RBF-style kernels of geometric and semantic consis-
tency are used to generate pseudo confidence labels, guid-
ing the network to down weight uncertain points without
discarding them entirely and to emphasize reliable regions
with high confidence. With these designs, COG naturally
extends to the unsupervised setting, where neither ground-
truth confidence nor pose supervision is available. Experi-
mental results demonstrate that COG achieves performance
comparable to leading supervised approaches, while the su-
pervised variant of COG further outperforms them.
Our contributions are summarized as follows:
1. We formulate correspondence finding as an OT problem
with confidence as marginals. Compared to OT with uni-
form marginals, our formulation yields balanced corre-
spondences by suppressing non-overlapping points.
2. We propose an end-to-end pipeline that jointly learns ob-
ject pose and point validity confidence without supervi-
sion from CAD models, poses, or overlap scores.
3. Unsupervised COG achieves performance competitive
with state-of-the-art supervised methods, and its super-
vised variant further outperforms them.
2. Related Work
Instance-level and Category-level Object Pose Estima-
tion.
Traditional object pose estimation methods, under
both instance-level [26, 28, 64, 66, 71] and category-
level [7, 22, 32, 38, 67, 68] settings, aim to recover the
6DoF poses of known objects or objects within predefined
categories. A common paradigm is to directly regress ob-
ject poses [10, 14, 77] using deep backbones. Alternatively,
correspondence-based approaches establish matches either
between CAD models and point clouds [31, 32, 61], or be-
tween query images and rendered CAD views [59, 65, 68],
followed by transformation estimation via PnP [36] or
Umeyama [63].
Recently, unsupervised or weakly su-
pervised frameworks have also gained attention.
Equi-
Pose [38] jointly learns canonical reconstruction and pose
using SE(3)-equivariant backbones [8, 79], while OP-
Align [7] extends this paradigm to articulated objects. Zero-
shot Pose [24] aligns semantic features across image se-
quences to establish correspondences without explicit pose
labels. However, these settings inherently limit generaliza-
tion, as they rely on prior knowledge of specific object in-
stances or categories, restricting their ability to handle novel
objects in open-world scenarios.
Novel Object Pose Estimation.
Novel object pose esti-
mation lifts the instance-level and category-level restric-
tion by targeting arbitrary objects beyond predefined cat-
egories. For such dataset-agnostic generalization, many ap-
proaches [4, 9, 39, 41] exploit vision foundation models [5,
48, 52] to extract semantic priors for correspondence learn-
ing. Further progress [9, 27, 35, 39, 44, 47, 58] leverages
additional references such as CAD models or multi-view
images to construct dense correspondences. For example,
SAM-6D [39] builds on CNOS [45] by using SAM [34, 76]
for segmentation and combining DINO [48] features with
geometric cues, while OnePose [27, 58] reconstructs object
point clouds via Structure-from-Motion and aligns them to
query views. MegaPose [35] retrieves the closest CAD ren-
dering to the query for subsequent refinement, and Gen-
Flow [44] iteratively refines poses by estimating optical
flow between rendered and observed views. More recently,
several methods attempt to reduce reference requirements to
a single image [18, 40, 41, 46]. POPE [18] and NOPE [46]
introduce feedforward frameworks with RGB inputs, While
SinRef-6D [40] models point-wise alignment using state-
space dynamics with RGB-D inputs, and the latest Uno-
Pose [41] constructs an SE(3)-invariant canonical frame
for consistent object representation. Yet, most existing ap-
proaches construct correspondences via discrete one-to-one
assignments, which tend to collapse onto a few dominant
keypoints, and breaks differentiability, preventing models
from being trained in an unsupervised manner.
2

Point Cloud Registration.
As a commonly used formu-
lation in object pose estimation, correspondence finding and
pose estimation can also be viewed from the perspective of
point cloud registration [11, 12, 16, 23, 37, 49, 54, 72],
which seeks to estimate the optimal rigid transformation
that aligns two partially overlapping point sets. A common
pipeline extracts discriminative local descriptors to estab-
lish correspondences, followed by robust solvers such as
RANSAC [19] and ICP [1, 2]. FCGF [11] and DGR [12]
enhance this process by predicting pairwise inlier prob-
abilities to filter unreliable matches.
More recent meth-
ods [13, 43, 55, 72, 73] formulate registration as an optimal
transport (OT) problem, treating correspondences as con-
tinuous probability distributions. RPM-Net [72] introduces
a differentiable Sinkhorn [56] layer to compute correspon-
dences under uniform marginals. Meanwhile, confidence
which represents the validity of correspondences has also
been explored—either as post-hoc calibration of pairwise
matches [73], as distribution-level weights [43], or as the
results of a global thresholding [13]. These OT-based meth-
ods typically assign such weights after the correspondences
have been established, making the correspondences still un-
balanced. And more importantly, the resulting confidence
cannot be optimized jointly with the correspondences in an
end-to-end manner. In contrast, we propose a learned point-
wise confidence that directly serves as the target marginal in
OT, yielding globally balanced transport plans and enable
end-to-end unsupervised learning.
3. Methodology
3.1. Problem Formulation
Given two RGB-D images of the same novel object cap-
tured from different viewpoints, one as the query and the
other as the reference, our goal is to estimate the relative
rigid transformation between them. As illustrated in Fig. 2,
we first employ a CNOS [45]-like segmentation model, Un-
oSeg [41], to obtain object masks from the RGB images.
The masked depth maps are then back-projected into 3D
space to generate point clouds.
In addition to geomet-
ric coordinates, we extract per-pixel RGB features using
DINO [48] (patch features with up-sampling), which serve
as RGB descriptors. We denote the query and reference
point clouds as P ∈Rn×3 and Q ∈Rn×3, and their as-
sociated RGB features as Fp ∈Rn×df and Fq ∈Rn×df,
where n is the number of points and df is the dimension
of the DINO features. Our objective is to find a rotation
Rpq ∈SO(3) and a translation tpq ∈R3 such that
Qval = PvalRpq + 1t⊤
pq,
(1)
where Pval ⊂P and Qval ⊂Q denote the valid overlap-
ping regions of the object that can be matched across views.
Since the observed point clouds P and Q are often partial
UnoSeg
DINO
Back
Projection
↑
Up-Sampling
Patch Features
As Filter
Mask
Point Cloud
&
RGB Features
Depth
RGB
Figure 2. Pre-processing pipeline of COG. Given RGB-D inputs,
an object is segmented, depth map is back-projected into point
clouds, and per-point RGB features are extracted from DINO to
form feature augmented inputs.
due to occlusions and viewpoint differences, this task in-
herently requires reasoning about both the relative pose and
the validity of each point. For clarity, throughout this paper
the subscripts pq and qp denote operations in the query-to-
reference and reference-to-query directions, respectively.
3.2. COG
3.2.1. Model Overview
As illustrated in Fig. 3, COG adopts a coarse-to-fine ar-
chitecture built upon a geometric transformer [51]. In the
coarse phase, we use farthest point sampling [17] to obtain
sparse point clouds and their DINO features as inputs (de-
noted with ˜·), while in the fine phase, the full point clouds
and DINO features are used for refinement. For clarity, we
use the fine phase notation throughout the following sec-
tions; the coarse phase shares the same formulation with all
terms marked by˜·.
Specifically, the geometric transformer encoder takes the
query and reference point sets P and Q as inputs to the
SE(3)-invariant feature encoding module (and to the po-
sition embedding module in the fine phase), producing ge-
ometric hidden features Hp and Hq. Meanwhile, DINO
features are processed through a semantic denoising mod-
ule to obtain semantic embeddings Sp ∈Rn×ds and Sq ∈
Rn×ds. The geometric decoder then applies alternating self-
attention and cross-attention layers to compute point-wise
geometric features Gp ∈Rn×dg and Gq ∈Rn×dg. Then a
lightweight MLP confidence head with a sigmoid activation
outputs per-point confidence scores cp, cq ∈[0, 1]n. These
confidence values are normalized into wp = cp/cp, wq =
cq/cq, and incorporated as target marginals in an optimal
transport formulation, while the cosine similarity between
geometric and semantic features forms the affinity kernel
K ∈Rn×n. We apply the Sinkhorn algorithm [56] to solve
for the transport plan Π ∈[0, 1]n×n and derive the row-
stochastic correspondence matrices Mpq, Mqp ∈[0, 1]n×n.
Using these correspondences as projection operators, we
compute soft matches MpqQ and MqpP via convex combi-
nation, and estimate the rigid transformation using weighted
SVD [63], where normalized confidence weights wp and
wq emphasize reliable correspondences. After the coarse
3

Geometric Decoder
Geometric Encoder
Semantic Denoising
𝑆𝐸(3)-inv Encoding
Geometric Decoder
Correspondence
Pose Estimation
Coarse Phase
Fine Phase
𝑷, 𝑭𝑝
𝑸, 𝑭𝑞
Sinkhorn
Fine
Pose
෡𝑹𝑝𝑞, ො𝒕𝑝𝑞
෩𝑮𝑝, ෤𝒄𝑝
෩𝑮𝑞, ෤𝒄𝑞
𝑮𝑝, 𝒄𝑝
Geometric Encoder
Semantic Denoising
Position Embedding
𝑆𝐸(3)-inv Encoding
T
Coarse
Pose
෩𝑹𝑝𝑞, ෤𝒕𝑝𝑞
SVD
𝑴𝑝𝑞, 𝒘𝑝
𝑴𝑞𝑝𝑷]
𝑷
𝑴𝑝𝑞𝑸
]
𝑸
𝒘𝑝
]
𝒘𝑞
෩𝑷, ෩𝑭𝑝
෩𝑸, ෩𝑭𝑞
Query
Ref
Self-attn
Cross-attn
Self-attn
Cross-attn
Rigid Transformation
T
[·|·] Point Concatenation
𝚷𝑝𝑞
𝒘𝑝
𝒘𝑞
𝑲𝑝𝑞
𝒄𝑝, 𝒄𝑞
𝑴𝑞𝑝, 𝒘𝑞
Weights
Marginals
Pose Estimation
෩𝑴𝑝𝑞, ෥𝒘𝑝
෩𝑴𝑞𝑝, ෥𝒘𝑞
Correspondence
Sinkhorn
෩𝚷𝑝𝑞
෥𝒘𝑝
෥𝒘𝑞
෩𝑲𝑝𝑞
෤𝒄𝑝, ෤𝒄𝑞Marginals
෩𝑴𝑞𝑝෩𝑷
෩𝑷
]
෩𝑴𝑝𝑞෩𝑸
෩𝑸
]
෥𝒘𝑝
෥𝒘𝑞]
SVD
Weights
Farthest Point Sampling
𝑯𝑝, 𝑺𝑝
…
…
…
…
𝑯𝑞, 𝑺𝑞
Iterative Refinement (inference only)
෩𝑯𝑝, ෩𝑺𝑝
෩𝑯𝑞, ෩𝑺𝑞
𝑮𝑞, 𝒄𝑞
Figure 3. Overview of the COG framework. The pipeline consists of coarse and fine phases, each using a geometric transformer to predict
point-wise confidences and features. A Sinkhorn-based OT module computes soft correspondences, and a weighted SVD solver estimates
the rigid transformation. The coarse pose is further refined in the fine phase using position embeddings for precise alignment.
phase, the estimated pose ( ˜Rpq,˜tpq) is applied to transform
the query cloud, producing a coarse aligned version used
in the fine phase to predict the final refined pose (ˆRpq ∈
SO(3), ˆtpq ∈R3). During inference, iterative refinement
is applied by repeatedly transforming the query cloud using
the estimated pose, further improving alignment accuracy.
We next introduce the optimal transport based correspon-
dence formulation in Sec. 3.2.2, followed by pose estima-
tion in Sec. 3.2.3, semantic priors in Sec. 3.2.4, and confi-
dence learning in Sec. 3.2.5.
3.2.2. Optimal Correspondence
In our formulation, the confidence of each point represents
its likelihood of finding a valid correspondence in the coun-
terpart point cloud. Thus, correspondence estimation be-
tween the query and reference can be interpreted as trans-
porting these confidence weights from one distribution to
the other. This perspective naturally casts the problem as an
optimal transport (OT) task, where the transport plan satis-
fies marginal constraints approximately consistent with the
predicted confidences. Such a formulation ensures globally
balanced soft correspondences and prevents over concentra-
tion on a few dominant keypoints.
We first construct an affinity kernel that combines geo-
metric and semantic similarities as
K[i,j] = exp
  1
τ ⟨Gp[i], Gq[j]⟩cos
 1 + ⟨Sp[i], Sq[j]⟩cos
λ/τ,
(2)
where τ and λ are temperature and semantic prior weight
hyper-parameters.
To handle cases where the two point
clouds exhibit different confidence distributions, we nor-
malize the predicted confidences by their means to form
target marginals wp = cp/cp and wq = cq/cq, ensur-
ing that P
i wp[i] = P
j wq[j] = n. This normalization
preserves global balance and allows sparse high-confidence
regions to distribute their mass over denser ones. The trans-
port plan Π = S(K, wp, wq) ∈[0, 1]n×n is obtained us-
ing the Sinkhorn algorithm [56] (denotes with S), where
P
j Π[i,j] ≈wp[i] and P
i Π[i,j] ≈wq[j]. We then normal-
ize Π row-wise to derive directional, row-stochastic cor-
respondence matrices, such as Mpq[i,j] =
Π[i,j]
P
k Π[i,k] , and
Mqp[i,j] =
Π⊤
[i,j]
P
k Π⊤
[i,k] . Here, Mpq and Mqp act as soft pro-
jection operators, mapping each point in one cloud to a con-
vex combination of points in the other. Accordingly, we
denote MpqQ as the corresponding points of P in the Q
space, and MqpP vice versa.
To regularize the correspondences, we enforce a cycle
consistency constraint inspired by CycleGAN [78]. A point
be projected to the opposite domain and back with Mpq and
Mqp should approximately reconstruct its original position.
The cycle consistency loss is defined as
Lcycl = 1
n
n
X
i=1
wp[i]

1 −ϕcycl(P, Prec)[i]

+ 1
n
n
X
j=1
wq[j]

1 −ϕcycl(Q, Qrec)[j]

,
(3)
where Prec = MpqMqpP and Qrec = MqpMpqQ. The
kernel ϕcycl measures geometric similarity via a Gaussian
RBF:
ϕcycl(X, Y)[i] = exp
 −αg∥X[i] −Y[i]∥2
2

,
(4)
where αg is a geometric scaling parameter and X, Y ∈
Rn×3 are point clouds.
4

3.2.3. Pose Estimation
Once the correspondence matrices and point-wise confi-
dences are obtained, we estimate the 6DoF rigid transfor-
mation between the query and reference using a confidence-
weighted SVD, Umeyama algorithm [63], also known as
corresponding point alignment [53]. This algorithm takes
two point sets with associated correspondences and per-
point weights as input, and outputs the optimal rotation and
translation minimizing the weighted least squares error.
For the consistency, we concatenate both sides corre-
spondences with the original point clouds, and perform joint
optimization as
ˆRpq, ˆtpq = U
 [P | MqpP], [MpqQ | Q], [wp|wq]

, (5)
where U(·) denotes the Umeyama solver, and [X|Y] ∈
R2n×3 represents concatenation along the point dimension.
The inverse transformation is naturally given by ˆRqp =
ˆR⊤
pq and ˆtqp = −ˆR⊤
pqˆtpq. The transformed point clouds are
thus Ppred = PˆRpq + 1ˆt⊤
pq and Qpred = QˆRqp + 1ˆt⊤
qp.
The estimated poses are optimized using a confidence-
weighted Chamfer loss that measures geometric alignment
between the transformed and target point clouds, defined as
Lpose = 1
n
n
X
i=1
wp[i]

1 −ϕpose(P, Qpred)[i]

+ 1
n
n
X
j=1
wq[j]

1 −ϕpose(Q, Ppred)[j]

,
(6)
where ϕpose is a Gaussian RBF kernel based on the Chamfer
distance [69]:
ϕpose(X, Y)[i] = exp

−αg
min
j∈{1,...,n} ∥X[i] −Y[j]∥2
2

,
(7)
with αg a geometric scaling hyperparameter and X, Y ∈
Rn×3 denoting point clouds.
3.2.4. Semantic Priors
Semantic features provide valuable cues for establishing
reliable correspondences, as they help constrain matches
within semantically consistent regions. To leverage such
cues, we incorporate semantic priors derived from vision
foundation models (VFMs) such as DINOv2 [48], which
produce patch level RGB embeddings capturing high level
visual semantics. However, we observe that raw DINO fea-
tures, while semantically rich, often encode mixed infor-
mation unrelated to object parts, resulting in feature incon-
sistencies across viewpoints. Points belonging to the same
semantic part may still exhibit noticeable differences in fea-
ture space, degrading cross-view matching quality. To im-
prove their robustness, we adopt the self label refinement
strategy of STEGO [25], which applies energy based clus-
tering for semantic denoising and yields more stable and
consistent features.
In our model, a lightweight seman-
tic head (an MLP) projects the DINO features Fp and Fq
into lower dimensional semantic embeddings Sp and Sq.
The semantic head is trained with the same loss as STEGO,
promoting feature consistency across corresponding regions
while filtering noise.
To softly constrain the correspondence using semantic
similarity, we define a semantic consistency loss that pe-
nalizes correspondences assigned to semantically dissimilar
points. Formally,
Lsem = 1
n
n
X
i=1
wp[i]

1 −ϕsem(Mpq, Sp, Sq)[i]

+ 1
n
n
X
j=1
wq[j]

1 −ϕsem(Mqp, Sq, Sp)[j]

,
(8)
where ϕsem measures feature similarity through a Gaussian
RBF kernel weighted by the correspondence matrix:
ϕsem(Muv, U, V)[i] =
n
X
j=1
Muv[i,j] exp
 −αf(1 −⟨U[i], V[j]⟩cos)

,
(9)
where Muv ∈[0, 1]n×n is a row-stochastic correspondence
matrix, U, V ∈Rn×d are semantic feature matrices, and
αf is a scaling hyperparameter. This loss encourages cor-
respondences to align semantically coherent regions while
maintaining the soft matching flexibility of OT.
3.2.5. Confidence Learning
The absence of ground-truth confidence labels poses a key
challenge for unsupervised training.
We address this by
constructing pseudo confidence labels derived from the
model’s own geometric and semantic consistency quality.
Intuitively, a reliable (high confidence) point should incur
low loss values, corresponding to high values of the Gaus-
sian RBF kernels ϕcycl, ϕpose, and ϕsem. Hence, these ker-
nel responses can be interpreted as soft inlier likelihoods
that jointly reflect geometric, semantic, and cyclic consis-
tency.
Formally, we define a composite pseudo likelihood as
ϕtot(·) = ϕcycl(·) ϕpose(·) ϕsem(·), to generate pseudo con-
fidence labels: zp = ϕtot(P, Prec, Qpred, Mpq, Sp, Sq),
and zq = ϕtot(Q, Qrec, Ppred, Mqp, Sq, Sp). These labels
provide graded supervision instead of binary inlier-outlier
signals, enabling the confidence branch to learn to down
weight uncertain points rather than discard them. By jointly
fusing cues from geometric reconstruction, pose alignment,
and semantic consistency, the pseudo labels capture com-
plementary aspects of correspondence reliability and yield
well calibrated confidence predictions. The confidence is
5

optimized using a binary cross entropy (BCE) loss:
Lconf = BCE
 cp, detach(zp)

+ BCE
 cq, detach(zq)

,
(10)
where detach indicates a stop-gradient operation to prevent
gradients from propagating into other loss terms, ensuring
that Lconf serves purely as supervision for confidence.
4. Experiments
4.1. Experimental Settings
Datasets and Benchmarks.
Following MegaPose [35],
we train COG on two large-scale datasets: Google Scanned
Objects [15] and ShapeNet [6]. Together, these datasets
contain approximately 2,000,000 RGB-D images of over
50,000 objects covering diverse shapes and materials. For
evaluation, we adopt BOP [60] benchmarks, TUD-L, LM-
O, and YCB-V [3, 29, 70], which test generalization to un-
seen objects and complex scenes. Specifically, TUD-L in-
cludes 600 images of 3 geometrically intricate objects; LM-
O contains 200 images of 8 objects in a cluttered tabletop
scene; and YCB-V provides 900 images of 21 household
objects, sometimes heavily occluded.
Implementation Details.
We use the same backbone
as our direct baseline UnoPose [41]:
the coarse phase
encoder-decoder is implemented as standard geometric
transformer [51], while the fine phase adopts a sparse-to-
dense variant introduced in [39]. We also re-implement Ro-
bust OT [55] and Dustbin OT [13] with these backbones.
For fair comparison, we use the same query-reference pair-
ing and the same segmentation masks as UnoPose for all
models. We train using the ADAM optimizer [33] with an
initial learning rate of 10−4. The model is trained for 3
epochs with a batch size of 32. The loss weights are set to
γcycl = 0.5, γpose = 1, γsem = 1, and γconf = 10. Hyper-
parameters are λ = 3, τ = 0.01, αf = 4, and αg = 60. We
randomly sample 1024 points from both query and refer-
ence point clouds as input to the fine phase, and 256 points
via farthest point sampling [17] for the coarse phase. Ad-
ditionally, we train a supervised variant of COG by only
replacing the Chamfer distance in Lpose with the point-
wise distance between the predicted and ground-truth trans-
formed points
 PRpq + 1t⊤
pq

[i] and
 PˆRpq + 1ˆt⊤
pq

[i].
Evaluation Metrics.
For pose evaluation, we follow the
BOP protocol, report mean Average Precision (mAP) un-
der three standard error metrics: VSD, MSSD, and MSPD
(see [60] for definitions). We also report the average infer-
ence time, including both segmentation and pose estima-
tion modules. For overlapping evaluation, we report the
Intersection-over-Union (IoU) between the predicted and
the GT overlapping regions.
4.2. Results
4.2.1. Object Pose Estimation
Quantitative and qualitative results for single-reference
novel object pose estimation are presented in Tab. 1 and
Fig. 4, respectively. COG, when trained in an unsupervised
manner, not only outperforms all other unsupervised base-
lines but also outperforms most supervised methods. Com-
pared with the state-of-the-art supervised approach Uno-
Pose [41], our unsupervised model achieves comparable
performance, with only a 2.1% gap on average. Notably,
on benchmark with relatively complicated object shapes,
TUD-L, our unsupervised method exceeds UnoPose for
2.8%. Meanwhile, on benchmarks characterized by clut-
tered scenes and heavily occlusions, LM-O and YCB-V,
performance gap increases, suggesting that unsupervised
training still has room for improvement in handling compli-
cated environments. When trained with pose supervision,
COG achieves the best overall performance across LM-O
and TUD-L, outperforming all existing methods. In particu-
lar, the large improvement on TUD-L highlights the model’s
robustness in accurately capturing fine-grained geometric
correspondences. Compared to other OT-based methods,
Robust OT [55], Dustbin OT [13], and RPM-Net [72], our
method exceeds them on all benchmarks, indicating the ef-
ficiency of confidence marginal OT and network-predicted
confidence rather than post hoc calibration. Overall, these
results demonstrate that COG achieves performance com-
petitive with supervised methods even without ground-truth
supervision, and that its supervised variant further estab-
lishes a new state of the art. This validates the effectiveness
of our confidence-aware OT formulation and the unsuper-
vised network-predicted confidence.
4.2.2. Overlapping Prediction
As an essential step toward accurate pose estimation, we
also evaluate the overlapping prediction derived from the
confidence values. Predicted confidence values greater than
0.5 are treated as positives, while others are negatives.
Tab. 2 reports the IoU between predicted and ground-truth
overlapping regions on the TUD-L dataset, and qualitative
examples are shown in the last 3 columns of Fig. 5. COG
effectively distinguishes overlapping from non-overlapping
areas by assigning high confidence to valid points and low
confidence to outliers and semantic or geometric incon-
sistent points. Even without supervision, our method ex-
ceeds the supervised UnoPose’s performance in average,
and the visualization demonstrate that the learned confi-
dence is both interpretable and robust for outliers.
4.3. Ablation Studies
4.3.1. Modules and Losses
To evaluate the contribution of different correspondence
formulations and loss terms, we conduct ablation studies
6

Method
Supervision
Modality
Reference
LM-O [3] ↑
TUD-L [29] ↑
YCB-V [70] ↑
Mean ↑
Time (s) ↓
PPF [16]
None
PC
Image
29.7
14.8
38.3
27.6
11.8
FPFH+MAC [54, 75]
None
PC
Image
22.5
22.1
49.6
31.4
136.9
FPFH+RANSAC [19, 54]
None
PC
Image
31.0
31.0
50.0
37.3
6.4
PPF+ICP [2, 16]
None
PC
Image
44.7
29.1
66.8
46.9
14.3
Robust OT [55]
None
PC+RGB
Image
45.5
66.3
66.0
59.3
4.0
FreeZe [4]
None
PC+RGB
Image
45.5
68.3
65.5
59.8
53.0
Dustbin OT [13]
None
PC+RGB
Image
50.2
67.6
65.4
61.1
4.2
COG (Unsupervised)
None
PC+RGB
Image
56.7
73.8
75.9
68.8
4.0
RPM-Net [72]
GT Pose
PC
Image
38.9
28.0
23.8
30.2
3.2
FCGF+MAC [11, 75]
GT Pose
PC
Image
33.9
48.3
51.0
44.4
60.5
FCGF+RANSAC [11, 19]
GT Pose
PC
Image
38.9
59.0
57.6
51.8
11.0
GeDi [50]
GT Pose
PC
Image
42.8
67.3
60.6
56.9
48.9
SAM-6D [39]
GT Pose
PC+RGB
Posed Image
54.5
29.7
68.1
50.8
4.2
UnoPose [41]
GT Pose
PC+RGB
Image
58.7
71.0
83.1
70.9
3.7
COG (Supervised)
GT Pose
PC+RGB
Image
60.8
80.0
80.5
73.8
4.0
Table 1. Quantitative comparison of single-reference novel object pose estimation methods on LM-O, TUD-L, and YCB-V. COG achieves
state-of-the-art performance in both supervised and unsupervised settings, with comparable inference speed.
YCB-V
TUD-L
LM-O
Figure 4. Qualitative results of unsupervised COG on LM-O, TUD-L, and YCB-V datasets. Blue bounding boxes represent the estimated
poses, while white boxes denote ground-truth poses.
Method
Dragon
Frog
Watering Can
Mean
UnoPose [41]
70.0
72.2
59.1
67.1
COG (Supervised)
72.9
68.3
83.9
75.0
COG (Unsupervised)
71.2
64.4
81.2
72.3
Table 2. Overlapping IoU results on TUD-L benchmark’s objects.
on the YCB-V benchmark.
As shown in Tab. 3, both
uniform-marginal OT and our confidence-marginal OT out-
perform the argmax and softmax baselines by roughly 2−
3%, demonstrating the advantage of globally balanced cor-
respondences over discrete or row-normalized mappings.
Among OT-based variants, the confidence-marginal for-
mulation further improves the mean performance, validat-
ing the effectiveness of incorporating learned confidences
as non-uniform marginals. And regarding the loss terms,
adding Lsem consistently improves the MSSD and MSPD
metrics, while Lcycl mainly benefits VSD, indicating that
semantic consistency enhances geometric alignment, and
cycle consistency improves visible-region correspondence.
Correspondence
Lsem
Lcycl
VSD ↑
MSSD ↑
MSPD ↑
Mean ↑
A1)
Argmax
71.3
78.4
64.4
71.4
A2)
Argmax
✓
72.1
78.5
64.2
71.6
A3)
Argmax
✓
68.3
79.1
64.0
70.5
A4)
Argmax
✓
✓
73.4
80.0
65.9
73.1
S1)
Softmax
71.9
78.6
64.0
71.5
S2)
Softmax
✓
72.1
78.5
64.1
71.6
S3)
Softmax
✓
68.6
78.3
63.5
70.1
S4)
Softmax
✓
✓
73.4
79.8
65.8
73.0
U1)
Uniform OT
74.6
80.2
67.8
74.2
U2)
Uniform OT
✓
75.0
80.0
67.0
74.0
U3)
Uniform OT
✓
74.1
82.1
69.0
75.0
U4)
Uniform OT
✓
✓
75.8
81.5
68.4
75.2
C1)
Confidence OT
74.8
80.3
67.9
74.3
C2)
Confidence OT
✓
75.5
80.3
67.4
74.4
C3)
Confidence OT
✓
74.9
82.5
69.6
75.6
C4)
Confidence OT
✓
✓
76.5
82.0
69.1
75.9
Table 3.
Modules and losses ablation on YCB-V benchmark.
Correspondence refers to whether use argmax, softmax, uniform
marginal OT, or confidence marginal OT to estimate correspon-
dence, Lsem and Lcycl refer to the use of these loss terms.
Overall, the combination of confidence marginal OT with
both auxiliary losses yields the best mean performance.
7

Query
Reference
Dragon
Frog
Watering Can
Confidence
1
0
Driller
Cat
Duck
Box
Figure 5. Visualization of predicted confidence from unsupervised COG. Our method effectively handles non-overlapping regions and
outlier points by assigning low confidence to unreliable points. Poses are aligned for visualization.
Sinkhorn Iteration
Semantic Priors
mAP ↑
ENT ↓
∆Marginal ↓
C4-1g)
1
73.1
21.1
0.35
C4-2g)
2
73.2
23.0
0.27
C4-4g)
4
73.2
24.1
0.21
C4-8g)
8
73.0
25.7
0.16
C4-1s)
1
✓
75.8
9.2
0.48
C4-2s)
2
✓
75.9
10.5
0.38
C4-4s)
4
✓
75.8
11.4
0.29
C4-8s)
8
✓
75.7
12.0
0.23
Table 4. OT parameters ablation on YCB-V benchmark. Effective
Number of Tokens (ENT) reflects the number of points receiving
attention, lower values indicate sharper focus. ∆Marginal indi-
cates the distance between target marginals and result marginals.
4.3.2. OT Parameters
We further analyze the influence of parameters in the OT
formulation, as summarized in Tab. 4. We use the Effec-
tive Number of Tokens (ENT), defined as the exponential of
correspondence entropy, to measure the concentration of the
transport plan—lower values indicate sharper, more focused
correspondences.
We also report ∆Marginal, the mean
distance between target and resulting marginals, which re-
flects the marginal alignment quality. When semantic priors
are injected into the affinity kernel, mAP improves notably
while ENT decreases, indicating that correspondences be-
come more compact and semantically coherent. This con-
firms that semantic guidance helps the OT solver focus on
meaningful, consistent regions across views. In contrast,
increasing the number of Sinkhorn iterations beyond two
slightly reduces performance.
Although additional itera-
tions reduce ∆Marginal, they also make the correspon-
dence map more diffuse, which weakens the geometric pre-
cision of the convex combination. Considering this trade-
off, we adopt two iterations in our final setting to balance
OT marginal accuracy and correspondence sharpness.
4.3.3. Iterative Refinement
We further evaluate the performance gain and computa-
tional overhead introduced by iterative refinement on a sin-
gle NVIDIA RTX 3090 GPU. As shown in Fig. 6, in-
creasing the number of refinement iterations consistently
Figure 6. Performance vs runtime (including segmentation, DINO
feature extraction, and pose estimation). t indicates the number of
refinement iterations.
improves mean performance, though with diminishing re-
turns. A single refinement already boosts both supervised
and unsupervised COG by over 1%, while the improvement
becomes marginal beyond two iterations. Considering the
trade-off between runtime and accuracy, we adopt one re-
finement iteration as our default setting.
5. Conclusion
We presented Confidence-aware Optimal Geometric Cor-
respondence (COG), an unsupervised framework for sin-
gle reference novel object pose estimation by solving
a confidence-aware optimal transport problem.
COG
achieves balanced and robust correspondences that improve
pose accuracy under occlusions by integrating point-wise
confidence as transport marginals. And incorporating se-
mantic priors from vision foundation models further en-
hances its semantic consistency. Furthermore, the confi-
dence learned in an unsupervised manner from geomet-
ric, semantic, and cycle consistency enables COG to down-
weight unreliable regions without external labels. Extensive
experiments demonstrate that COG achieves performance
comparable to state-of-the-art supervised approaches, val-
idating the effectiveness of our unsupervised formulation.
Overall, COG provides a principled and scalable direction
toward generalizable, unsupervised object pose estimation.
8

References
[1] K. S. Arun, T. S. Huang, and S. D. Blostein. Least-squares
fitting of two 3-d point sets.
IEEE Transactions on Pat-
tern Analysis and Machine Intelligence, PAMI-9(5):698–
700, 1987. 3
[2] PJ Besl and Neil D McKay. A method for registration of 3-d
shapes. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 14(2):239–256, 1992. 3, 7
[3] Eric Brachmann, Alexander Krull, Frank Michel, Stefan
Gumhold, Jamie Shotton, and Carsten Rother.
Learning
6d object pose estimation using 3d object coordinates. In
Proceedings of European Conference on Computer Vision
(ECCV), pages 536–551. Springer, 2014. 6, 7, 2, 3, 4
[4] Andrea Caraffa, Davide Boscaini, Amir Hamza, and Fabio
Poiesi. Freeze: Training-free zero-shot 6d pose estimation
with geometric and vision foundation models. In Proceed-
ings of European Conference on Computer Vision (ECCV),
pages 414–431. Springer, 2024. 2, 7
[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers. In Pro-
ceedings of International Conference on Computer Vision
(ICCV), pages 9650–9660, 2021. 2
[6] Angel X Chang, Thomas Funkhouser, Leonidas Guibas,
Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese,
Manolis Savva, Shuran Song, Hao Su, et al.
Shapenet:
An information-rich 3d model repository.
arXiv preprint
arXiv:1512.03012, 2015. 6
[7] Yuchen Che, Ryo Furukawa, and Asako Kanezaki. Op-align:
Object-level and part-level alignment for self-supervised
category-level articulated object pose estimation.
In Pro-
ceedings of European Conference on Computer Vision
(ECCV), pages 72–88. Springer, 2024. 1, 2
[8] Haiwei Chen, Shichen Liu, Weikai Chen, Hao Li, and Ran-
dall Hill. Equivariant point network for 3d point cloud analy-
sis. In Proceedings of IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), pages 14514–14523, 2021.
2
[9] Kai Chen, Yiyao Ma, Xingyu Lin, Stephen James, Jianshu
Zhou, Yun-Hui Liu, Pieter Abbeel, and Qi Dou. Vision foun-
dation model enables generalizable object pose estimation.
In Proceedings of Advances in Neural Information Process-
ing Systems (NeurIPS), pages 19975–20002, 2024. 1, 2
[10] Wei Chen, Xi Jia, Hyung Jin Chang, Jinming Duan, Linlin
Shen, and Ales Leonardis. Fs-net: Fast shape-based network
for category-level 6d object pose estimation with decoupled
rotation mechanism.
In Proceedings of IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pages
1581–1590, 2021. 2
[11] Christopher Choy, Jaesik Park, and Vladlen Koltun. Fully
convolutional geometric features. In Proceedings of Interna-
tional Conference on Computer Vision (ICCV), pages 8958–
8966, 2019. 3, 7
[12] Christopher Choy, Wei Dong, and Vladlen Koltun.
Deep
global registration. In Proceedings of IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2020. 3
[13] Zheng Dang, Fei Wang, and Mathieu Salzmann. Learning
3d-3d correspondences for one-shot partial-to-partial regis-
tration. arXiv preprint arXiv:2006.04523, 2020. 2, 3, 6, 7
[14] Yan Di, Ruida Zhang, Zhiqiang Lou, Fabian Manhardt, Xi-
angyang Ji, Nassir Navab, and Federico Tombari. Gpv-pose:
Category-level object pose estimation via geometry-guided
point-wise voting. In Proceedings of IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pages
6781–6791, 2022. 2
[15] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kin-
man, Ryan Hickman, Krista Reymann, Thomas B McHugh,
and Vincent Vanhoucke. Google scanned objects: A high-
quality dataset of 3d scanned household items. In Proceed-
ings of IEEE International Conference on Robotics and Au-
tomation (ICRA), pages 2553–2560. IEEE, 2022. 6
[16] Bertram Drost, Markus Ulrich, Nassir Navab, and Slobodan
Ilic.
Model globally, match locally: Efficient and robust
3d object recognition. In Proceedings of IEEE Conference
on Computer Vision and Pattern Recognition (CVPR), pages
998–1005. Ieee, 2010. 3, 7
[17] Y. Eldar, M. Lindenbaum, M. Porat, and Y.Y. Zeevi. The
farthest point strategy for progressive image sampling. IEEE
Transactions on Image Processing, 6(9):1305–1315, 1997.
3, 6
[18] Zhiwen Fan, Panwang Pan, Peihao Wang, Yifan Jiang, Dejia
Xu, and Zhangyang Wang. Pope: 6-dof promptable pose
estimation of any object in any scene with one reference. In
Proceedings of IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pages 7771–7781, 2024. 1, 2
[19] Martin A Fischler and Robert C Bolles.
Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartograp