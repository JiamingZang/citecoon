# UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image

> 2025 · id: W4413146937 · arXiv: 2411.16106 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Unseen object pose estimation methods often rely on CAD
models or multiple reference views, making the onboarding
stage costly. To simplify reference acquisition, we aim to
estimate the unseen object’s pose through a single unposed
RGB-D reference image. While previous works leverage ref-
erence images as pose anchors to limit the range of relative
pose, our scenario presents significant challenges since the
relative transformation could vary across the entire SE(3)
space. Moreover, factors like occlusion, sensor noise, and
extreme geometry could result in low viewpoint overlap. To
address these challenges, we present a novel approach and
benchmark, termed UNOPose1, for UNseen One-reference-
based object Pose estimation. Building upon a coarse-to-fine
paradigm, UNOPose constructs an SE(3)-invariant refer-
ence frame to standardize object representation despite pose
and size variations. To alleviate small overlap across view-
points, we recalibrate the weight of each correspondence
based on its predicted likelihood of being within the overlap-
ping region. Evaluated on our proposed benchmark based on
the BOP Challenge, UNOPose demonstrates superior perfor-
mance, significantly outperforming traditional and learning-
based methods in the one-reference setting and remaining
competitive with CAD-model-based methods. The code and
dataset are available at github.com/shanice-l/UNOPose.

## introduction
Localizing an object in Euclidean space by estimating its
6DoF pose, i.e., 3DoF orientation and 3DoF position, plays
a crucial role in augmented/virtual reality [59, 76], scene un-
derstanding [37, 63] and robotic manipulation [23, 26, 80].
The vast majority of works [42, 48, 68, 78, 83–86] focuses
on instance-level object pose estimation, where the train-
*Equal contributions.
1UNO (/"u:noU/) means one in Spanish and Italian.
ing and testing datasets consist of an identical set of known
object instances, and the CAD models of objects are often
required for generating training images and labels [16, 56].
More recently, Wang et al. [87] and its successors [6, 79, 88]
extended this paradigm to the category level, intending to
estimate poses for novel instances within predefined cat-
egories without requiring CAD models of target objects.
Both paradigms have limitations in open-world applications,
where annotating and training for new objects beyond known
categories would be very labor-intensive and sometimes pro-
hibitive.
To mitigate this problem, recent research efforts [3, 43, 51,
90] have shifted towards pose estimation of arbitrary novel
objects, where the target object is unseen during training.
This task presents a great challenge due to the inherent uncer-
tainty of the novel object’s canonical frame (defined in the
CAD model of the object). Current approaches [43, 57, 62]
often tackle this by using omnidirectional reference views
to cover the target object, separating pose estimation task to
viewpoint selection and relative pose estimation to known
pose anchors. If CAD models are available during the test
phase, reference views can be easily produced through image
rendering [43, 51, 61, 90]. Alternatively, without CAD mod-
els, reference views can be obtained by capturing multiple
images of the object [29, 31, 55, 77]. Both setups have short-
comings, as creating a CAD model of an object or labeling a
substantial number of object poses is limited in scalability.
For example, they can hardly adapt to in-the-wild scenarios
where novel objects are introduced unpredictably.
To sum up, while existing methods exhibit promising
transfer capability to novel objects, they are notably con-
strained by their reliance on the CAD model or multiple
reference views (Fig. 1 (a, b)). In this work, we aim to mini-
mize onboarding cost by estimating novel object poses using
a single unposed RGB-D reference image (Fig. 1). This
setup is different from existing methods mainly in two as-
pects. First, we focus on estimating the relative pose between
1
arXiv:2411.16106v2  [cs.CV]  17 Mar 2025

Query image
…
…
…
…
ΔP
Our Results
Our Reference
Others’ Reference
Seg.
(a) CAD model
(b) RGB(-D) images
…
…
…
Figure 1. Illustration of unseen object pose estimation. Given a query image presenting a target object unseen during training, we aim to
estimate its segmentation and 6DoF pose w.r.t. a reference frame. While previous methods [43, 57, 77, 90] often rely on the CAD model or
multiple RGB(-D) images for reference, we merely use one unposed RGB-D reference image.
two unposed viewpoints rather than the absolute pose. Note
that the absolute pose of a novel object becomes ill-posed
without a certain canonical frame, while relative pose estima-
tion remains well-defined regardless of the canonical frame’s
definition. Secondly, some existing relative pose estimation
methods [98, 105, 106] merely use RGB modality to predict
the 3DoF rotation, limiting their ability to predict relative
translation and generalize beyond their training datasets. In
contrast, we fully leverage RGB-D modality to predict the
6DoF relative pose, meanwhile enhancing the network’s
generalizability to novel objects and environments.
Nevertheless, estimating the pose of unseen objects with
only one reference view presents significant challenges from
various aspects. First, previous methods [3, 43] select the
most similar reference view as a pose anchor, distinctly re-
ducing the search space for relative poses. However, in
our scenario, where both the target and reference object
poses are unknown, the relative pose can vary across the
entire SE(3) space. Moreover, in partial-to-partial object
matching, factors such as occlusion, sensor noise, and ex-
treme geometry can severely interfere with the matching
process. To address these challenges, we propose a novel
approach and benchmark for UNseen One-reference-based
object Pose estimation (UNOPose). Our pipeline leverages
the strong generalization capabilities of vision foundation
models [40, 65] to produce an effective segmentation of the
unseen object. Then, UNOPose employs a coarse-to-fine
paradigm for estimating the relative pose between the ref-
erence and the query objects. To alleviate the challenge of
diverse pose and size variations, we introduce the SE(3)-
invariant global reference frame (GRF) to standardize object
representation. Subsequently, a hierarchical object encoding
paradigm based on the local reference frame (LRF) further
captures fine-grained geometric details. For achieving reli-
able correspondences in partial-to-partial object matching,
we harness an overlap predictor to identify and concentrate
on the overlapping region.
Moreover, we propose a new benchmark devised from
the BOP challenge [34] to facilitate evaluation and future
research of unseen object pose estimation with one reference.
Extensive experiments on YCB-V [92], LM-O [2] and TUD-
L [32] datasets demonstrate that our UNOPose surpasses
all compared methods on a single reference setting. To our
surprise, UNOPose with an unposed reference is even on par
with some SE(3)-invariant-feature-based methods relying
on CAD models (Ours 70.9% vs. ZTE-PPF [1] 69.0% vs.
Koenig-PPF [41] 75.1% w.r.t. ARBOP metric).
Our contributions can be summarized as follows:
• To the best of our knowledge, we are the first to conduct
unseen object 6DoF pose estimation leveraging a single
unposed RGB-D reference.
• Based on the BOP Challenge, we devise a new exten-
sive benchmark tailored for unseen object segmentation
and pose estimation with one reference. Additionally,
we evaluate several traditional and learning-based meth-
ods on this benchmark for completeness.
• We introduce UNOPose, a network for learning relative
transformation between reference and query objects.
To achieve this, we propose the SE(3)-invariant global
and local reference frames, enabling standardized ob-
ject representations despite variations in pose and size.
Furthermore, the network can automatically adjust the
confidence of each correspondence by incorporating an
overlap predictor.
2

## method
VSD MSSD MSPD ARBOP
C0
B3 + Soft Correspondence [95]
77.1
81.7
72.7
77.2
C1
B3 + Overlap Predictor
82.6
87.4
79.4
83.1
C2
A3 + Overlap Predictor
68.4
74.6
64.8
69.2
D0
C1: Seg. →CNOS [61]
77.4
80.9
73.6
77.3
D1
C1: Seg. →SAM-6D [51]
78.3
82.1
74.3
78.2
D2
C1: Seg. →MRCNN [28, 42]
81.2
85.7
77.9
81.6
D3
C1: Seg. →GT Seg.
86.6
89.2
82.1
86.0
E0
C1: Ref. Real →Rendered
83.7
89.6
83.1
85.5
Table 5. Ablation of overlap predictor, segmentation, and refer-
ence type on YCB-V.
10
20
30
40
50
60
70
80
90
Reference Rotation Distance (◦)
20
30
40
50
60
70
80
90
100
Average Recall (%)
ARBOP per Group
ARBOP
ARBOP AlignCenter
20
30
40
50
60
70
80
90
100
Overlap Ratio (%)
Overlap Ratio
Figure 3. Ablation of initial rotation distance between query
and reference objects. We categorize all testing objects into nine
groups according to the initial rotation distance, and evaluate the
ARBOP metric and overlap ratio for each group separately.
OPose. Specifically, GRF brings significant performance
enhancement in the coarse pose estimation (Tab. 4 A1 vs. A0,
A3 vs. A2), providing an accurate pose initialization for the
next stage. Meanwhile, LRF is constructed to exploit fine-
grained local information, primarily enhancing the fine pose
estimation results (Tab. 4 B0 vs. B2, B1 vs. B3). With both
GRF and LRF, our network achieves obvious performance
improvement (Tab. 4 A3 vs. A0, B3 vs. B0).
Ablation on Correspondence Loss.
In partial-to-partial
point cloud registration, only some of the points can find
their counterparts on the target point cloud. Therefore, the
introduction of overlap prediction helps to distinguish the
points in the overlap region and achieves good results (Tab. 4-
5 C1 vs. B3, C2 vs. A3). Moreover, we consider soft cor-
respondence loss in [95] as an alternative, which predicts
probabilities of overlap between each point pair. However,
the result did not meet the expectations (Tab. 5 C0).
Ablation on Segmentation.
Tab. 1 showcases the advan-
tage of our UNOSeg over other segmentation methods on
YCB-V. Furthermore, substituting UNOSeg with CNOS [61],
SAM-6D [51], or MRCNN [28, 42] results in a noticeable
decline in UNOPose’s performance (Tab. 5 D0–D2). More-
over, substituting UNOSeg with ground-truth segmentation
brings ≈3% performance enhancement (Tab. 5 D3 vs. C1).
Using Rendered References.
We replaced the real refer-
ence with a pose-identical rendered version and present the
results in Tab. 5. With reduced depth noise and domain gap
between training and testing, UNOPose works even better us-
ing the rendered reference (Tab. 5 E0 vs. C1). However, this
ablation requires the object model, which is more difficult to
acquire than an unposed real image.
Ablation on Reference Rotation Distance.
To investigate
the impact of the initial rotation distance between the query
and reference images, we randomly select the reference ob-
ject with a maximum rotation distance of 90◦for each query
object. Further, all testing objects are categorized into nine
groups according to initial rotation distances. We separately
evaluate each group and present the results in Fig. 3. Mean-
while, we calculate the overlap ratio between reference and
query instances of each group and visualize the mean and
standard deviation. Notably, performance significantly de-
creases when the rotation distance exceeds 50◦. This decline
is attributed to the reduced overlap between query and refer-
ence viewpoints. However, our network still shows favorable
transfer ability in extreme relative pose estimation (rotation
distance (80◦, 90 ◦], overlap ratio 45.4%, ARBOP 54.8%).
The appendix presents further implementation details,
visualization results, and more experimental analysis on
reference selection and extensive datasets.
5. Conclusion, Limitation and Future work
This work has introduced UNOPose, a new approach for
unseen object pose estimation with one reference. The key
idea is constructing the SE(3)-invariant reference frame
for tackling diverse pose and size variations. Moreover,
we propose an overlap predictor for handling low-overlap
scenarios. To evaluate the proposed method, we devise a
new benchmark based on the BOP challenge and compare
some state-of-the-art methods. Experimental results show
UNOPose surpasses all compared reference-based methods
significantly, and is competitive with some SE(3)-invariant-
feature-based methods relying on CAD models.
Limitation
and
Future
Work.
While
efficient,
correspondence-based methods rely on predicting robust and
dense correspondence between query and reference objects,
thereby limited in extreme two-view geometry applications.
Future work will focus on reconstructing the unseen object
from a single reference [53, 108] and concurrently estimat-
ing its object pose.
Acknowledgements.
This work was supported in part
by the National Natural Science Foundation of China
under Grant No. 62406169, and in part by the China Post-
doctoral Science Foundation under Grant No. 2024M761673.
8

## experiments
UNO Object Segmentation Results.
We compare our
UNOSeg with some existing CAD-model-based unseen ob-
ject segmentation methods [7, 51, 61] in Tab. 1. Leveraging
a single image as a reference, our UNOSeg achieves com-
parable results with state-of-the-art methods (Ours 54.2%
vs. SAM-6D 54.5%) at a faster speed (Ours 2.68s vs. SAM-
6D 4.53s). Notably, UNOSeg-FastSAM achieves the best
performance on YCB-V (mAP 67.3%). Compared to Fast-
SAM [107], SAM [40] achieves consistently better results
on three datasets. Therefore, we choose UNOSeg-SAM as
the default segmentation for subsequent experiments.
UNO Object Pose Estimation Results.
Since we are the
first to conduct relative 6DoF pose estimation for unseen
objects and build a brand-new benchmark, we re-implement
some traditional [22, 72, 103] and learning-based meth-
ods [4, 11, 12, 51, 69] for comparison. Note that we lever-
age identical data pre-processing, segmentation, query
and reference image pairs, and evaluation protocols for
all methods for fair comparison. Moreover, we also com-
pare UNOPose with two CAD-model-based pose estimators
leveraging SE(3)-invariant PPF feature [1, 41]. All results
are illustrated in Tab. 2. It clearly shows that our method
surpasses all image-reference-based methods by a large mar-
gin, and even achieves comparable results with CAD-model-
based methods using class-specific detectors [28, 52] (Ours
70.9% vs. ZTE-PPF [1] 69.0% vs. Koenig-PPF [41] 75.1%).
“Ref. AlignCenter” directly uses the reference rotation as
prediction, meanwhile leveraging the shift between query
and reference centers as relative translation, so it can be re-
garded as the baseline of all methods. Among all traditional
descriptors, PPF with Iterative Closest Point (ICP) refine-
ment [22] achieves the best results (AR 46.9%). Learning-
based methods demonstrate varying generalization abilities
on unseen objects. For example, the GeoTransformer-based
method UTOPIC, designed for registering object-level par-
ROW
GRF
LRF
FPE
VSD
MSSD
MSPD
ARBOP
A0
✗
✗
✗
64.6
68.1
59.2
63.9
A1
✓
✗
✗
66.4
72.8
63.2
67.5
A2
✗
✓
✗
66.2
69.5
60.4
65.4
A3
✓
✓
✗
66.9
73.2
63.5
67.9
B0
✗
✗
✓
79.8
85.0
75.5
80.1
B1
✓
✗
✓
79.4
85.2
76.6
80.4
B2
✗
✓
✓
81.1
85.1
76.6
80.9
B3
✓
✓
✓
80.7
86.0
77.7
81.5
Table 4. Ablation of GRF and LRF on YCB-V. FPE stands for
“fine pose estimation”.
tial point clouds, performs suboptimally in our setting. In
contrast, the adapted unseen object pose estimation meth-
ods, i.e. SAM-6D [51] and FreeZe [4], achieve satisfactory
results compared to their counterparts.
Run-time Analysis. Evaluated on a single RTX 3090 GPU,
our pipeline runs at 3.70s for one image, including 2.68s for
segmentation and 1.02s for pose estimation (Tab. 1 - 2).
4.3. One Reference for a Category
To minimize the efforts of acquiring references, we leverage
a single reference for all objects of the same category in
a dataset. Tab. 3 illustrates the qualitative and quantitative
experimental results on TUD-L. It is worth noting that the
target object in TUD-L is dynamic rather than static, prevent-
ing methods from leveraging background information. This
setup poses challenges for relative pose estimation methods
that rely on scene features, making them likely to fail in
this task. Despite this, UNOPose achieves impressive per-
formance (60.7% in terms of the ARBOP metric) with only
a single reference per category. The visualization results
in Tab. 3b further show UNOPose is capable of predicting
omnidirectional relative poses. Additional experiments on
reference selection are provided in the appendix.
4.4. Ablation Studies
Effectiveness of GRF and LRF.
As demonstrated in
Sec. 3.3.2, GRF transfers arbitrary object poses to an SE(3)-
invariant frame, while LRF assists with extracting precise
local descriptors for fine pose estimation. Tab. 4 shows that
the effectiveness of GRF and LRF is complementary in UN-
7

ROW

## related_work
Class-specific Pose Estimation.
Class-specific ob-
ject pose estimation aims at predicting 6DoF poses of
either instance of a known object (instance-level) or un-
seen objects within a known category (category-level).
Instance-level object poses are often solved by direct re-
gression [35, 47, 56, 85] using deep neural networks, or
by establishing 2D-3D [27, 48, 68] or 3D-3D correspon-
dences [30, 73, 83] which are then leveraged by RANSAC-
based PnP/Kabsch algorithms. The instance-level setting re-
quires expensive data generation/annotation [18, 19, 81, 102]
and training for every new object. To address this limitation,
category-level pose estimation [87] is proposed to estimate
9DoF poses of novel objects among specific categories with-
out CAD models. Mainstream approaches can also be cate-
gorized into direct regression [6, 9, 10, 17, 50, 54, 99–101]
or correspondence-based [8, 24, 79, 87] methods. However,
this setting remains constrained to a limited number of cate-
gories, given the additional challenge of aligning canonical
frames, managing symmetric objects, and categorizing ob-
jects in similar categories correctly (e.g. can and bottle).
Novel Object Pose Estimation.
Estimating the pose of
novel objects beyond known classes, i.e., objects at inference
time are unseen during training, is a useful yet challenging
task. Existing works solve this problem through image-based
matching [3, 43, 60, 61, 74, 90, 104] or feature-based match-
ing approaches [4, 29, 36, 51, 62, 66, 77]. Notably, given
a target image, OVE6D [3] and MegaPose [43] retrieve the
most similar viewpoint from a pre-rendered omnidirectional
image database as a coarse pose estimate. A customized
neural network then refines the coarse pose estimate. Foun-
dationPose [90] further builds a pose ranking network to
score each refined pose hypothesis. Feature-matching meth-
ods learn local feature descriptors to construct pixel-level
or point-level correspondences. For example, OnePose [77]
and OnePose++ [29] reconstruct the 3D point cloud of an
unseen object to establish 2D-3D correspondences, whereas
SAM-6D [51] builds 3D-3D correspondences. More re-
cently, some methods [4, 66] directly exploit the power of
foundation models for zero-shot pose estimation. Exemplar-
ily, FoundPose [66] extracts DINOv2 visual features [65],
while FreeZe [4] further utilizes generalizable geometric
features [69] to conduct feature matching.
Relative Pose Estimation.
Orthogonal to previous works
which rely on known instances or multiple reference views,
relative object pose estimation estimates a relative transfor-
mation of the unseen object with only one reference image.
One closely related topic is camera pose estimation from two
views. For example, RelPose [98] and RelPose++ [49] infer
a distribution of relative rotations by leveraging an energy-
based formulation. Moreover, iFusion [91] optimizes the
unknown relative pose by inverting the novel view synthesis
diffusion model [53]. Relative camera pose estimation relies
on static background to establish correspondence between
viewpoints. In the field of relative object pose estimation,
given two RGB images, 3DAHV [105] presents a hypothesis-
and-verification framework to score each relative pose hy-
pothesis. Extended by that, DVMNet [106] introduces a
hypothesis-free pipeline to compute relative poses via deep
voxel matching. The above works have limited capability in
inferring 3DoF relative translations. In this work, with the
introduction of depth data, we can estimate a complete 6DoF
pose including rotation and translation.
Point Cloud Registration.
Our approach follows the
paradigm of correspondence-based methods [14, 15, 45, 89],
which initially extract 3D local feature descriptors to es-
tablish correspondences between source and target point
clouds. Subsequently, they estimate the relative transfor-
mation using techniques like SVD, RANSAC, or Hough
Voting [44]. Some feature descriptors are proposed to ensure
rotational invariance. For example, FPFH [72] descriptor
is computed with geometric surface properties. PPF [22]
uses the distance and angles to describe the relation of point
pairs. TOLDI [93] proposes a robust local reference frame
using keypoint normals and neighboring projections. With
the advent of deep learning, neural networks are increas-
ingly employed to extract 3D local or global feature descrip-
tors [12, 38, 71, 89, 97].
3. UNO Object Segmentation and Pose Estima-
tion
3.1. Problem Formulation
Assuming an arbitrary unseen rigid object in a query image,
our goal is to estimate the object’s mask Mq and its 6DoF
relative pose ∆T ∈SE(3) with a masked RGB-D reference
image exhibiting the target object without major occlusion
or truncation. As illustrated in Fig. 1, the input is:
1) [Iq|Dq] ∈RH×W ×4: The query RGB-D image;
2) [Ip|Dp] ∈RH×W ×4 and Mp ∈RH×W : The reference
RGB-D image and the corresponding binary mask indi-
cating the target object;
3) Kq ∈R3×3 and Kp ∈R3×3: Camera intrinsics of the
query and reference images.
Optionally, if the pose Tp ∈SE(3) of the reference ob-
ject in the camera frame is known , the query object pose
Tq ∈SE(3) can be recovered by Tq = ∆TTp. Note that,
for practicality, the method should not rely on the absolute
pose of the reference object, since the world frame can vary
arbitrarily in different applications. We only use the refer-
ence object pose during the inference stage for evaluation on
standard object pose datasets.
3.2. UNO Object Segmentation
Firstly, we segment the query object from a cluttered back-
ground. Thanks to the great generalization ability of vision
3

X
Z
Y
𝐏𝑮
GRF
Geometry 
Encoder
Color 
Encoder
(𝑁𝑐+ 1)
X
Z
𝐐𝑮
Positional Encoding
X
Z
Y
𝐐𝑮
𝒄
X
Z
Y
𝐏𝑮
𝒄
(𝑁𝑓+ 1)
BG token
c
GRF
s
s
Geometry 
Encoder
Color 
Encoder
s
෠𝐹𝑃
𝑓, ෠𝑂𝑃
𝑓
෠𝐹𝑄
𝑓, ෠𝑂𝑄
𝑓
overlap
prediction
Correlation Matrix
Δ𝐓𝒊𝒏𝒊𝒕
Correlation Matrix
Pose 
Selection
Geometric 
Transformer
Decoder
Weighted
SVD
ΔT
𝐐𝒄𝒂𝒎
𝐏𝒄a𝒎
BG token
Coarse Pose Estimation
Fine Pose Estimation
concatenate
point sample
s
c
overlap
prediction
𝐏𝒄a𝒎
𝐐𝒄𝒂𝒎
𝐏𝑮
𝒇
X
Z
Y
𝐏𝑮
𝐏𝒄a𝒎
X
Z
Y
𝐐𝑮
𝒇
X
Z
Y
𝐐𝑮
෩𝐐𝒄𝒂𝒎
X
Z
Y
𝑁𝑓× 3
𝑁𝑐× 3
s
መ𝑓𝑃
𝑐
መ𝑓𝑄
𝑐
መ𝑓𝑃
𝑓
መ𝑓𝑄
𝑓
Y
Δ𝑻𝒊𝒏𝒊𝒕
c
s
෠𝐹𝑃
𝑐, ෠𝑂𝑃
𝑐
෠𝐹𝑄
𝑐, ෠𝑂𝑄
𝑐
Geometric 
Transformer
Decoder
× 𝟑
× 𝟑
LRF Encoding
Figure 2. The network architecture of UNOPose. Given the query and reference point clouds Qcam and Pcam in the camera frame,
UNOPose first transforms them into the SE(3)-invariant global reference frame (GRF). Then feature descriptors are extracted from sparse
point sets for constructing the coarse correlation matrix. For achieving precise correspondences, the fine pose estimation module exploits
structural details using positional encoding and local reference frame (LRF) encoding.
foundation models, methods like [7, 51, 61] can effectively
segment novel objects with CAD models. However, un-
like previous approaches, which generate diverse descrip-
tors from multiple rendered views, we can access only a
single reference image. To address this challenge, we use
the SAM model [40] to predict all possible mask proposals
from the query image, and then score each mask proposal
by comparing DINOv2 [65] descriptors between the query
and reference images using cosine similarity, identifying the
most similar mask Mq. Please refer to the appendix for more
details about UNO segmentation.
3.3. UNO Object Pose Estimation
3.3.1. Overview of UNOPose
Given the predicted mask Mq of the query image and mask
Mp of the reference image, we crop and back-project the
object of interest from depth maps Dq, Dp into the cam-
era space as two point sets Qcam ∈RNQ×3 and Pcam ∈
RN P ×3, where N Q and N P denotes the point numbers of
query and reference point clouds, respectively. Our goal is
to recover the relative transformation ∆T = {∆R, ∆t} by
minimizing the correspondence distance
min
X
(q, p)∈C
∥∆Rq + ∆t −p∥2,
(1)
where C is the predicted correspondence set between Qcam
and Pcam. We exploit both color and geometric cues to build
this correspondence and follow the broadly used coarse-to-
fine paradigm in point cloud registration to solve Eq. (1).
The network is demonstrated in Fig. 2.
3.3.2. Coarse-to-fine Pose Estimation
Constructing a Pose-invariant Reference Frame.
Given
only an unposed reference image, the relative pose to predict
is arbitrary in the SE(3) space, which renders a significant
challenge for achieving robust correspondences. Hence, we
introduce a pose-invariant global reference frame (GRF) and
transform Qcam and Pcam into GRF as QG and PG.
Concretely, taking Qcam as an example, transforming the
point cloud to GRF involves a 7DoF coordinate transforma-
tion {RG ∈SO(3), tG ∈R3, sG ∈R}:
QG = {R⊤
G(q −tG)/sG | q ∈Qcam}.
(2)
The origin of GRF is located at the object center cQ for
translation invariance, and the radius of the point cloud is
rescaled to 1 for size invariance, computed as
tG = cQ,
sG =
max
q∈Qcam ∥q −cQ∥2.
(3)
The key ingredient is to devise the rotation RG
=
[rGx|rGy|rGz], where rGx, rGy, rGz 