# BundleTrack: 6D Pose Tracking for Novel Objects without Instance or Category-Level 3D Models

> 2021 · id: W4293365527 · arXiv: 2108.00516 · pdf: https://arxiv.org/pdf/2108.00516 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Robot manipulation often requires information about the
pose of the manipulated object. In some cases, this can
be achieved through forward kinematics (FK), assuming
the object’s motion equivalent to the end-eﬀector’s motion.
Frequently, however, FK is insuﬃcient to accurately estimate
the object’s pose [1]. This can be due to slippage during
grasping or in-hand manipulation [2], or during handoﬀs or
due to the compliance of a suction cup (Fig. 1). In these
cases, dynamically estimating an object’s pose from visual
data is desirable. Single-image 6D pose estimation methods
have been studied extensively [3]–[7]. Some of them are
fast and can re-estimate poses from scratch for every new
frame [8], [9]. Nevertheless, this is redundant, less eﬃcient,
leading to less coherent estimations over consecutive frames
and negatively impacts planning and control. On the other
hand, given an initial pose estimate, tracking 6D object
poses over image sequences can improve estimation speed
while providing coherent and accurate poses by leveraging
temporal consistency [10]–[12].
Most existing 6D object pose estimation or tracking ap-
proaches assume access to an object instance’s 3D model [3],
[9]. Having access to such instance 3D models complicates
generalization to novel, unseen instances. To overcome this
limitation, recent eﬀorts have relaxed this assumption and
The authors are with the Computer Science Dept. of Rutgers in NJ, USA.
Email: {bw344,kostas.bekris}@cs.rutgers.edu. This work is supported by
NSF NRI award 1734492. The results do not express the sponsor’s positions.
t
Fig. 1: Top: NOCS Dataset [13] example: The target object exits the camera’s
frustum during tracking but BundleTrack maintains its estimate without re-initialization.
Bottom: YCBInEOAT Dataset [22] example: The object is successfully tracked during
pick and place manipulation by a robotic arm, despite the lack of texture, severe self-
occlusion and motions due to the arm and the compliant suction cup. Computing object
pose from forward kinematics is unreliable in this setup due to the end-eﬀector.
require only category-level 3D models for 6D pose estima-
tion [13]–[16] or tracking [17]. They often achieve this by
training over a large number of CAD models from the same
category. While promising results have been demonstrated for
previously seen object categories, there are still limitations.
These methods are constrained by the variety of categories
in the training database. Popular 3D model databases, such
as ShapeNet [18] and ModelNet40 [19], contain 55 and 40
categories respectively. This is still far from suﬃcient to
cover diverse object categories present in the real world.
Furthermore, 3D model databases often require nontrivial
manual eﬀort and expert domain knowledge to build, involv-
ing steps such as scanning [20], mesh reﬁnement [21] or
CAD design.
Another line of work from the SLAM literature has
moved to address dynamic, object-aware challenges [23]–
[26], where dynamic objects are being reconstructed on-the-
ﬂy while being tracked without the need for object 3D models
beforehand. However, tracking-via-reconstruction [24], [26]
tends to accumulate errors when fusing observations with
erroneous pose estimates into the global model. These errors
adversely impact model tracking in subsequent frames.
Motivated by the above limitations, this work aims for
accurate, robust 6D pose tracking that is generalizable to
novel objects without instance or category-level 3D models.
It exploits recent advances in video segmentation as well as
learning-based keypoint detection and matching for a coarse
pose estimate, followed by a memory-augmented pose-graph
optimization step to achieve spatiotemporal consistent pose
output. Instead of aggregating into a global model, represen-
tative historical observations are maintained as keyframes
in a memory pool, providing candidate nodes for future
graphs so as to enable multi-pair data association together
with the latest observation. An eﬃcient implementation of
this framework in CUDA allows to achieve competitive
running times. Extensive experiments have been conducted
arXiv:2108.00516v1  [cs.CV]  1 Aug 2021

on two large-scale public benchmarks, shown in Fig. 1. Both
qualitative and quantitative results demonstrate a signiﬁcant
improvement over existing state-of-art approaches, including
methods using instance or category-level 3D models or
SLAM-like methods.
In summary, this work’s contributions are the following:
1) A novel integration of methods that result in a 6D pose
tracking framework that generalizes to novel objects without
access to instance or category-level 3D models.
2) A memory-augmented pose graph optimization for
low-drift accurate 6D object pose tracking. In particular,
augmenting the memory pool with historical observations
enables multi-hop data association and ameliorate the dearth
of correspondences between a pair of consecutive frames.
Additionally, maintaining keyframes as raw nodes instead of
aggregating into a global model signiﬁcantly reduces tracking
drift.
3) An eﬃcient CUDA implementation, which allows to
execute online the computationally-heavy multi-pair feature
matching as well as pose-graph optimization for 6D object
pose tracking (for the ﬁrst time to the best of the authors’
knowledge).
These contributions result in a new state-of-art perfor-
mance by boosting the previous best accuracy from 33.3% to
87.4% under the “5°5cm” metric in the NOCS Dataset [13],
even when compared against approaches utilizing category-
level 3D models for training. They also result in compa-
rable performance on the YCBInEOAT dataset [22], even
when compared against approaches utilizing instance-level
3D models [22].

## method
An overview of the proposed BundleTrack framework is
depicted in Fig. 2. The currently observed RGB-D frame
𝐼𝑡and the object segmentation mask computed during the
last timestamp 𝑀𝑡−1 are forwarded to a video segmentation
network to compute the current object mask 𝑀𝑡. Based on
𝑀𝑡and 𝑀𝑡−1 respectively, the target object regions in both 𝐼𝑡
and 𝐼𝑡−1 are cropped, resized and sent to a keypoint detection
network to compute keypoints and feature descriptors. A
data association process consisting of feature matching and
outlier pruning in the manner of RANSAC [48] identiﬁes
feature correspondences. Based on these correspondences,
a registration between 𝐼𝑡−1 and 𝐼𝑡can be solved in closed-
form, which is then used to provide a coarse estimate ˜T𝑡
for the transform between the two snapshots. The estimate
˜T𝑡is used to initialize the current node T𝑡as part of a
pose graph optimization step. To deﬁne the rest of the nodes
of the pose graph, no more than K keyframes are selected
from a memory pool to participate in the optimization. The
choice of K is made to balance an eﬃciency vs. accuracy
tradeoﬀ. Pose graph edges include both feature and geometric
correspondences, which are computed in parallel on GPU.
Given this information, the pose graph step outputs online
the optimized pose for the current timestamp T𝑡∈𝑆𝐸(3). If
the last frame corresponds to a novel view, then it is also
included in the memory pool.
A. Propagating Object Segmentation
The ﬁrst step is to segment the object’s image region from
the background. Prior work [24] used Mask-RCNN [49] to
compute the object mask in every frame of the video. It deals
with each new frame independently, which is less eﬃcient
and results in temporal inconsistencies.
To avoid these limitations, this work adopts an oﬀ-the-
shelf transductive-VOS network [50] for video object seg-
mentation, which is trained on the Davis 2017 [51] and
Youtube-VOS [52] datasets. The network uses dense long-
term similarity dependencies between current and past fea-
ture embeddings to propagate the previous object mask to
the latest frame. The object mask needed by BundleTrack
is simply binary, i.e., 𝑀𝜏= {0,1}𝐻×𝑊,𝜏∈{0,1,...,𝑡} and
distinguishes the object region from the background. The
only requirement is an initial mask 𝑀0 of interest. Neither
the transductive-VOS network nor the following steps of
BundleTrack require 𝑀0 to come from semantic/instance
segmentation. Therefore, it can also be obtained in alternative
ways depending on the application, e.g., low-level image
segmentation [43], [53], point cloud segmentation/clustering
[46], [47], or plane ﬁtting and removal [46], etc.
While the current implementation uses transductive-VOS,
the following techniques do not depend on this speciﬁc
network. If the object mask can be computed via simpler
means, such as computing a region of interest (ROI) from
forward kinematics followed by point cloud ﬁltering in robot
manipulation scenarios [2], the segmentation module can be
replaced.
B. Keypoint Detection, Matching and Local Registration
Local registration is performed between consecutive
frames 𝐼𝑡−1 and 𝐼𝑡to compute a initial pose ˜T𝑡. To do so,
correspondence between keyframes detected on each image
is performed. Diﬀerent from prior work [17], which relies
on category-level 3D models to learn a ﬁxed number of
category-level semantic keypoints, this work aims to use
generalizable features not speciﬁc to certain instances or
categories. The LF-Net [54] is chosen given its satisfac-
tory balance between performance and inference speed. It
only requires training on general 2D images, such as the
ScanNet dataset [55] used here, and generalizes to novel
scenes. During testing, for the newly observed frame 𝐼𝑡, LF-
Net receives the segmented image (Sec. IV-A) as input. It
then outputs 𝑛keypoints 𝑥𝑖,𝑖∈{0,1,...,𝑛−1} along with
the feature descriptor 𝐷𝑖∈𝑅128, where 𝑛is 500 in all
experiments. Due to the potentially imperfect segmentation
in previous step, outlier keypoints can arise from the back-
ground. It is thus critical to perform feature matching and
outlier pruning via RANSAC [48], executed in parallel on
GPU in this work. Each registration sample consists of 3
pairs of keypoints matched between the two images. A pose
hypothesis is generated from a sample via least squares
[56]. When evaluating samples, inlier correspondences have

a distance between transformed point pairs below a threshold
𝛿and an angle formed by the normals within a threshold
𝛼. The values of 𝛿and 𝛼are empirically set to 5𝑚𝑚and
45° in all experiments. After RANSAC, a preliminary pose is
computed by ˜T𝑡= T𝑡−1𝑇𝑡−1
𝑡
where 𝑇𝑡−1
𝑡
is the best sampled
correspondence hypothesis.
C. Keyframe Selection
˜T𝑡is then reﬁned during a pose graph optimization step.
The number of keyframes participating in the optimization
is limited to 𝑘⩽K for the sake of eﬃciency, where K = 15
is the number used in the experiments. When the size of the
keyframe memory pool N is larger than K, the objective is
to ﬁnd the set of keyframes with the largest mutual viewing
overlap to make good use of multi-view consistency. This
challenge can be formulated as the minimum H-subgraph of
an edge-weighted graph problem [57]:
argmin
𝑥
∑︁
𝑖∈N
∑︁
𝑗∈N, 𝑗≠𝑖
𝑥𝑖𝑥𝑗· 𝑎𝑟𝑐𝑐𝑜𝑠
 
𝑡𝑟(𝑅𝑇
𝑖𝑅𝑗) −1
2
!
so that :
∑︁
𝑖∈N
𝑥𝑖= K and 𝑥𝑖∈{0,1},𝑖∈N,
where 𝑅𝑖is the rotation matrix of the corresponding
keyframe’s pose. The goal is to ﬁnd the optimal binary vector
𝑥∈𝑅𝑁that indicates the selections. The weight of the edge
between frame pair (𝑖, 𝑗) is the geodesic distance of their
rotations. Mutual viewing overlap is maximized when the
mutual rotation diﬀerence relative to the camera is min-
imized. Combinatorial optimization algorithms for solving
this problem have a complexity of 𝑂(N K/𝑙𝑜𝑔N) [57]. In
practice, an iterative greedy selection is followed by starting
with the keyframe set {𝐼0} until the number of selected
keyframes reaches K. 𝐼0 is chosen since the initial frame does
not suﬀer from any tracking drift and serves as the reference
frame. In each iteration, the keyframe with the smallest sum
of geodesic distances against 𝐼𝑡as well as all previously
selected keyframes is added. This reduces complexity to
𝑂(NK3 + NK2), making the selection practical (under a
millisecond) without degrading performance.
D. Online Pose Graph Optimization
The pose graph can be denoted as 𝐺= {𝑉,𝐸}, |𝑉| = 𝑘+1,
where each node corresponds to the object pose in the
camera’s frame at the current and 𝑘selected timestamps
𝜏∈{𝑡,𝑡−𝑡1,𝑡−𝑡2,...,𝑡−𝑡𝑘}. For simplicity, the subscripts
of graph nodes will be denoted as simple indices 𝑖∈|𝑉|
instead of the actual timestamp 𝑡−𝑡𝑖. Each node’s pose can
then be denoted as T𝑖,𝑖∈|𝑉|. Inspired by [58], for the edges
between each pair of nodes, two types of energies E 𝑓and
E𝑔are considered. The energy E 𝑓relates to the residuals
computed from feature correspondences and E𝑔relates to
the geometric residuals measured by dense pixel-wise point-
to-plane distance. The spatiotemporal consistency is achieved
when the total energy of the graph E is minimized:
E = Í
𝑖∈|𝑉|
Í
𝑗∈|𝑉|, 𝑗≠𝑖
(𝜆1E 𝑓(𝑖, 𝑗) +𝜆2E𝑔(𝑖, 𝑗))
(1)
E 𝑓(𝑖, 𝑗) =
Í
(𝑚,𝑛) ∈𝐶𝑖, 𝑗
𝜌
T−1
𝑖𝑝𝑚−T−1
𝑗𝑝𝑛

2

(2)
In order to compute E 𝑓, feature correspondences 𝐶𝑖, 𝑗
between each pair of nodes (𝑖, 𝑗) are determined. If 𝐶𝑖, 𝑗
has been built during a previous pose graph optimization,
it is reused. Otherwise, the data association process of Sec.
IV-B is performed to compute 𝐶𝑖, 𝑗. These multi-pair feature
correspondences are built in parallel on GPU. In Eq. (2) and
(3), 𝑝represents the unprojected 3D points in the camera’s
frame, 𝜌is the M-estimator, where Huber loss is used.
E𝑔(𝑖, 𝑗) =
Í
𝑝∈|𝐼𝑖|
𝜌
𝑛𝑖(𝑥) · (T𝑖T−1
𝑗𝜋−1
𝐷(𝜋(T 𝑗T−1
𝑖𝑝)) −𝑝)

2

(3)
For E𝑔, dense pixel-wise correspondences are associated
by point re-projection, while outliers are ﬁltered based on
the distance between the point pair and the angle formed by
their normals; 𝜋(·) is the perspective projection operation;
𝜋−1
𝐷(·) denotes the unprojection mapping, which recovers a
3D point in the camera’s frame by looking up the depth value
on the pixel location; 𝑛𝑖(·) returns the normal of the pixel
on the frame 𝐼𝑖,𝑖∈|𝑉|.
In Eq. (1), 𝜆1 and 𝜆2 are the weights balancing E 𝑓and
E𝑔. To emphasize the lack of sensitivity to the choice of
these values, 𝜆1 and 𝜆2 are set to 1 in all experiments unless
otherwise speciﬁed. Then, the goal is to ﬁnd the optimal
poses, such that:
𝜉∗= argmin
𝜉
𝜌( ¯E(𝜉))
where
¯E(𝜉) is the stacked energy residual vector, 𝜉=
(𝜉𝑡,𝜉𝑡−𝑡1,𝜉𝑡−𝑡2,...,𝜉𝑡−𝑡𝑘)𝑇∈𝑅6×(𝑘+1) is the stacked pose vec-
tor corresponding to the current frame and 𝑘selected past
keyframes, while the pose corresponding to the initial frame
𝐼0 is kept constant as reference. Each block 𝜉𝑖= 𝑙𝑜𝑔(T𝑖) ∈
𝔰𝔢(3) is parametrized in Lie Algebra [59], consisting of 3
parameters for translation and 3 parameters for rotation. A
common approach is to apply ﬁrst-order Taylor expansion
around 𝜉, such that the iteratively re-weighted nonlinear least
squares can be solved by a Gauss-

## experiments
This section evaluates the proposed approach and com-
pares against state-of-the-art 6D pose tracking and estima-
tion methods on two public benchmarks, the NOCS dataset
[13] and the YCBInEOAT dataset [22]. Experiments are
performed over diverse types of objects and various tracking
scenarios (e.g., moving camera or moving objects). Both
quantitative and qualitative results demonstrate that Bundle-
Track achieves comparable or even superior performance
relative to alternatives, although it does not require instance
or category-level 3D models. Concretely, no CAD models
or training data from a 3D object database are used by
BundleTrack. All experiments are conducted on a standard
desktop with Intel Xeon(R) E5-1660 v3@3.00GHz processor
and a single NVIDIA RTX 2080 Ti GPU.
A. Datasets
NOCS dataset [13]: Among existing datasets, this is the
closest to the setup here, where instance 3D models are not
provided during evaluation. The dataset contains 6 object
categories: bottle, bowl, camera, can, laptop, and mug. The
training set consists of: (1) 7 real videos containing 3
instances of each category in total, annotated with ground
truth poses; and (2) 275K frames of synthetic data generated
using 1085 instances from the above 6 categories using a 3D
model database ShapeNetCore [18] with random poses and
object combinations in each scene. The testing set has 6 real
videos containing 3 diﬀerent unseen instances within each
category, resulting in 18 diﬀerent object instances and 3,200
frames in total.
YCBInEOAT dataset [22]: This dataset helps verify the
eﬀectiveness of 6D pose tracking during robot manipulation.
It was originally developed to evaluate approaches relying on
CAD models. The available CAD models, however, are not
used by BundleTrack. In contrast to the NOCS dataset where
objects are statically placed on a tabletop and captured by
a moving camera, YCBInEOAT contains 9 video sequences
captured by a static RGB-D camera, while objects are dynam-
ically manipulated. There are three types of manipulation:
(1) single arm pick-and-place, (2) within-hand manipulation,
and (3) pick to hand-oﬀbetween arms to placement. These
scenarios and the end-eﬀectors used make directly computing
Assumption

## related_work
6D Object Pose Tracking - For setups where object CAD
models are available, signiﬁcant progress has been made in
6D pose tracking. This includes techniques based on hand-
crafted probabilistic ﬁltering [11], [27], [28], optimization
[12], [29]–[31], and machine learning [10], [22]. The require-
ments, however, of such instance-level 3D models, either for
training oﬄine or model-frame registration during tracking,
complicate generalization to novel instances. More recently,
a 6D pose tracking approach [17] relaxed the assumption
to category-level 3D models using 3D object CAD model
databases for training [18]. During testing, the target ob-
ject category needs to be identiﬁed and the corresponding
network for that category is utilized for tracking. Instead
of being limited to the number of categories such database
is able to include, this work employs deep features that in
principle can be trained on arbitrary 2D images. It allows
generalization to diverse novel objects, as shown in the
accompanying experiments.
Dynamic Object-aware SLAM - In order to track dy-
namic objects’ pose and decouple them from static back-
ground, frame-model Iterative Closest Point (ICP) combined
with color [23]–[26], probabilistic data association [32], or
3D level-set likelihood maximization [33] has been applied.
Object models are simultaneously reconstructed on-the-ﬂy by
aggregating the observed RGB-D data with the newly tracked
pose. Nevertheless, frame-model tracking can be challenging
for object reconstruction, since errors in pose estimation
transfer to the reconstructed model and adversely aﬀect the
subsequent tracking [34]. This work does not fuse observed
frames but instead maintains them as nodes in a pose
graph, allowing to correct previously erroneous estimates,
and reduces drift in long-term tracking. The aforementioned
SLAM-family approaches may also face challenges in robot
manipulation setups that involve small, textureless, ﬂat or
shiny objects due to the dearth of suﬃcient correspondences
between the pair of consecutive frames. To ameliorate this
issue, BundleTrack searches correspondences among current
and multiple historical frames, consisting of both feature and
geometric terms, as the edges in the pose graph. Its eﬀec-
tiveness has been shown in extensive experiments including
for such challenging manipulation scenarios.
3D Hand-held Object Scanning - Promising results have
been demonstrated in scanning dynamic hand-held objects
[35]–[39], where the object’s motion needs to be taken
into account similar to the current setup. In particular, a
framework for robot manipulation [37] performs simultane-
ous object reconstruction and tracking, which leads to sim-
ilar issues as the aforementioned dynamic SLAM methods.
In addition, forward kinematics is required in its Kalman
Filtering framework, preventing generalization in scenarios
when objects are not held by the robotic manipulator. While
estimating object poses is part of the scanning process, there
are key diﬀerences from online 6D pose tracking. For the
scanning application, external assistance including human
interaction or deliberate motion is acceptable [36], [38], [39]
but it is not assumed in the current work. Furthermore, time
consuming global-optimization steps are often adopted at
the end of scanning to polish the models and their poses
while intermediate erroneous pose estimations and associated
frames can be discarded and not fused into the global model
[36], [38], [39]. In contrast, this work aims to provide fast
and accurate pose tracking output online.
III. PROBLEM FORMULATION
𝑻𝟎
𝑪
𝑻𝟎
𝑪
𝑻𝟎→𝝉
𝒕ൌ𝟎
𝒕ൌ𝝉
Assume a rigid body
for which there is no
its
corresponding
3D
model, nor its category-
level 3D model database
for training. The objective is to continuously track its 6D
pose change relative to the start of tracking, i.e., the relative
transformation 𝑇0→𝜏∈𝑆𝐸(3),𝜏∈{1,2,...,𝑡} in the camera’s
frame 𝐶. The input is the following:
• 𝐼𝜏: A sequence of RGB-D data 𝐼𝜏,𝜏∈{0,...,𝑡}.
• 𝑀0: A binary mask on the ﬁrst image 𝐼0, indicating the
target object region to track in the image space.
• 𝑇𝐶
0
(optional): The initial pose in the camera’s frame 𝐶.
Used if the objective is to recover the object’s absolute
pose in 𝐶, otherwise set to identity.
The initial mask 𝑀0 can be obtained in multiple diﬀerent
ways to initialize tracking. For instance, via semantic seg-
mentation [40]–[42] or non-semantic methods, such as image

Data Association
. . . . . .
RGB‐D Input Sequence
Segmented Image
𝑰𝒕
𝑰𝒕ି𝟏
historical keyframe
current frame
video segmentation network
keypoint detection network
pose graph edges
𝐓𝒕ି𝒕𝟐
𝐓𝒕ି𝒕𝒌
Local Registration
Keypoints &
Descriptors
𝐓𝒕ି𝟏
𝐓෩𝒕
Keyframe 
Selection
Keyframe Memory Pool
Object Pose 
Graph
𝐓𝒕ି𝒕𝟏
Novel View?
𝑴𝒕
𝑴𝒕ି𝟏
𝐓𝒕
𝑴𝒕ି𝟐
. . . . . .
(1)
(2)
(3)
(4)
(5)
(6)
Fig. 2: BundleTrack framework from left to right: (1) an image segmentation network returns the object mask given the prior one; (2) a network detects keypoints and their
descriptors; (3) keypoints are matched and coarse registration is performed between consecutive frames to estimate an initial relative transform ˜T𝑡; (4) keyframes are selected
from a memory pool to participate in the pose graph optimization; (5) online pose graph optimization outputs a reﬁned spatiotemporal consistent pose T𝑡; and (6) the latest
frame is included in the memory pool, if it is a novel view to enrich diversity.
segmentation, [43]–[45], point cloud segmentation/clustering
[46], [47], or plane ﬁtting and removal [46], etc.
The object’s pose in the camera’s frame 𝐶can be
recovered
at
any
timestamp
by
applying
the
relative
transformation 𝑇0→𝜏in the camera’s frame T𝜏= 𝑇𝐶
𝜏=
𝑇𝐶
0 [(𝑇𝐶
0 )−1𝑇0→𝜏𝑇𝐶
0 ] = 𝑇0→𝜏𝑇𝐶
0 ∈𝑆𝐸(3). For simplicity, the
rest of this document will refer to T𝜏as the output of the
process but 𝑇0→𝜏is what is actually computed as tracking.

## conclusion
This work presents BundleTrack, a general framework for
tracking the 6D pose of novel objects without any assump-
tions on instance or category-level 3D models. Extensive
experiments demonstrate that it is able to perform long-term
accurate tracking under various challenging scenarios. It even

achieves comparable performance to state-of-art methods that
depend on the target object’s CAD model. Future research
includes the exploration of combining BundleTrack with
model-free grasping methods [65], [66], to perform robust
pick-and-place [67], [68] or in-hand dexterous manipulation
for a wide variety of novel objects.