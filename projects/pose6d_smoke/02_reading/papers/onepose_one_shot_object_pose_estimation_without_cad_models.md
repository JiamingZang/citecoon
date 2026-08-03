# OnePose: One-Shot Object Pose Estimation without CAD Models

> 2022 · id: W4385263554 · arXiv: 2205.12257 · pdf: https://arxiv.org/pdf/2205.12257 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

OnePose: One-Shot Object Pose Estimation without CAD Models
Jiaming Sun1,2∗
Zihao Wang1∗
Siyu Zhang2∗
Xingyi He1
Hongcheng Zhao3
Guofeng Zhang1
Xiaowei Zhou1†
1Zhejiang University
2SenseTime Research
3TUM
Abstract
We propose a new method named OnePose for object pose
estimation. Unlike existing instance-level or category-level
methods, OnePose does not rely on CAD models and can
handle objects in arbitrary categories without instance-
or category-specific network training. OnePose draws the
idea from visual localization and only requires a simple
RGB video scan of the object to build a sparse SfM model
of the object. Then, this model is registered to new query
images with a generic feature matching network.
To
mitigate the slow runtime of existing visual localization
methods, we propose a new graph attention network that
directly matches 2D interest points in the query image with
the 3D points in the SfM model, resulting in efficient and
robust pose estimation.
Combined with a feature-based
pose tracker, OnePose is able to stably detect and track
6D poses of everyday household objects in real-time. We
also collected a large-scale dataset that consists of 450 se-
quences of 150 objects. Code and data are available at the
project page: https://zju3dv.github.io/onepose/.
1. Introduction
Object pose estimation plays an important role in aug-
mented reality (AR). The ultimate goal of object pose es-
timation in AR is to use arbitrary objects as “virtual an-
chors” of AR effects, which demands the ability to estimate
poses of surrounding objects in our daily life. Most estab-
lished works in object pose estimation [16, 26, 46] assume
that the CAD model of the object is known a priori. Since
high-quality CAD models of everyday objects are often in-
accessible, the research on object pose estimation for AR
scenarios necessitates new problem settings.
To not rely on instance-level CAD models, many recent
methods have been working on category-level pose estima-
tion [4,43]. By training a network on different instances in
∗Equal contribution. The authors from Zhejiang University are affil-
iated with the State Key Lab of CAD&CG and the ZJU-SenseTime Joint
Lab of 3D Vision. †Corresponding author: Xiaowei Zhou.
Synthetic
 Images
Real 
Images
Pose
CAD 
Model
Cat. 1
Cat. N
Pose
SfM
Pose
Query Image
GATs
Video Scan
Instance Level
Category Level
One Shot (ours)
Figure 1.
Comparison of different problem settings of
instance/category-level object pose estimation and the one-shot
pose estimation proposed in this work. Unlike previous works that
rely on instance- or category-specific network training, the pro-
posed approach only requires a simple video scan of the object to
build a sparse SfM model of the object and uses a generic 3D-
2D feature matching network (GATs) to estimate its pose, without
CAD models or additional network training.
the same category, the network can learn a category-level
representation of object appearances and shapes and thus
be able to generalize to new instances in the same cate-
gory. However, such approaches require a large number of
training samples in the same category, which can be hard to
obtain and annotate. Furthermore, the generalization capa-
bilities of category-level methods are not guaranteed when
a new instance has a significantly different appearance or
shape. More importantly, training and deploying a network
for each category are unaffordable in many real world ap-
plications, e.g., mobile AR, when the number of object cat-
egories to be handled is huge.
To alleviate the demand for CAD models or category-
specific training, we go back to an “old” problem setting
for object pose estimation, but renovate the entire pipeline
with a new learning-based approach. Similar to the task
of visual localization, which estimates the unknown cam-
era pose given an SfM map of a scene, object pose esti-
mation has long been formulated in the localization-based
setting [21, 35]. Different from instance- or category-level
1

methods, this setting assumes that a video sequence of the
object is given, and a sparse point cloud model can be re-
constructed from the sequence. Estimating the object pose
is then equivalent to localizing the camera pose with re-
spect to the reconstructed point cloud model. At test time,
2D local features are extracted from the query image and
matched with the points in the SfM model to obtain 2D-3D
correspondences, from which the object pose can be solved
by PnP. Instead of learning instance- or category-specific
representations by neural networks, this traditional pipeline
leverages an explicit 3D model of the object that can be
built on-the-fly for a new instance, which brings better gen-
eralization capabilities to arbitrary objects while making the
system more explainable.
In this paper, we refer to this problem setting as one-
shot object pose estimation, where the objective is being
able to estimate 6D pose of an object in arbitrary category,
given only a few pose-annotated images of the object for
training. While this problem is similar to visual localiza-
tion, directly migrating existing visual localization methods
does not solve this problem. The modern visual localization
pipeline [31] produces 2D-3D correspondences by first per-
forming a 2D-2D matching between the query image and
the retrieved database images. To ensure a high success rate
of localization, matching to multiple image retrieval candi-
dates is necessary, so that the 2D-2D matching can be ex-
pensive especially for learning-based matchers [32,36]. As
a result, the runtime of existing visual localization methods
is often seconds and cannot satisfy the requirement to track
moving objects in real-time.
For the reasons above, we propose to directly perform
2D-3D matching between the query image and the SfM
point clouds. Our key idea is to use graph attention net-
works (GATs) [40] to aggregate the 2D features that cor-
respond to the same 3D SfM point (i.e., a feature track)
to form a 3D feature.
The aggregated 3D features are
later matched with 2D features in the query images with
self- and cross-attention layers. Together with the self- and
cross-attention layers, the GATs can capture the globally-
consented and context-dependent matching priors exhibited
in ground-truth 2D-3D correspondences, making the match-
ing more accurate and robust.
To evaluate the proposed method, we collected a large-
scale dataset for the one-shot pose estimation setting, which
contains 450 sequences of 150 objects. Compared with pre-
vious instance-level method PVNet [27] and category-level
method Objectron [4], OnePose achieves better precision
without training for any object instances or categories in the
validation set, while taking only 58 ms to process one frame
on GPU. To the best of our knowledge, when combined with
a feature-based pose tracker, OnePose is the first learning-
based method that can stably detect and track poses of ev-
eryday household objects in real-time (refer to the project
page).
Contributions.
• Renovating the visual localization pipeline for object pose
estimation that can handle novel objects without CAD
models or additional network training.
• A new architecture of graph attention networks for robust
2D-3D feature matching.
• A large-scale object dataset for one-shot object pose esti-
mation with pose annotations.
2. Related works
CAD-Model-Based Object Pose Estimation. The state-
of-the-art approaches for the object 6DoF pose estima-
tion can be broadly categorized into regression and key-
point techniques. Given an image, the first type of meth-
ods [15,16,18,46] directly regress pose parameters with fea-
tures within each Region of Interest (RoI). In contrast, the
latter type of methods first find correspondences between
image pixels and 3D object coordinates either by regres-
sion [22, 24, 25] or by voting [26, 27], and then compute
the pose with Perspective-n-Points (PnP). These methods
require high fidelity textured 3D models to generate auxil-
iary synthetic training data and for pose refinements [16,18]
to achieve high accuracy on trained instances.
Unlike the abovementioned methods that train a single
network for each instance, NOCS [43] proposes to establish
correspondences between pixels on the image and Normal-
ized Object Coordinates (NOCS) shared within each cate-
gory. With this learned category-level shape prior, NOCS
can eliminate the dependencies on CAD models during test
time. Some later works [17,38,38,42,44] follow the trend
of leveraging category-level prior to further recover a more
accurate shape of the object with NOCS representation. A
limitation of this line of work is that the shape and the ap-
pearance of some instances could vary significantly even
they belong to the same category, thus the generalization ca-
pabilities of trained networks over these instances are ques-
tionable. Moreover, accurate CAD models are still required
for ground-truth NOCS map generation during training, and
different networks need to be trained for different cate-
gories. Our proposed method does not require CAD models
both for training and test time and is category-agnostic.
CAD-Model-Free Object Pose Estimation. Recently, a
few attempts have been made to achieve CAD-model-free
object 6D pose estimation both at the training and test time.
Both Neural Object Fitting [8] and LatentFusion [23] tack-
led the problem via analysis-by-synthesis approaches where
differentiablly synthesized images are compared with tar-
get images to generate gradients for the object pose op-
timization. Neural Object Fitting [8] proposes to encode
category-level appereance prior with a Variational Auto En-
2

coder (VAE) trained with fully synthetic data, while Latent-
Fusion [23] builds a 3D latent space based object represen-
tation with posed RGB-D images for each unseen object.
However, the efficiency and accuracy of such methods are
highly limited by image synthesizing networks and are not
suitable for AR applications. RLLG [7] takes a different ap-
proach and learns correspondences from image pixels to ob-
ject coordinates [5] without CAD models. Although RLLG
can achieve comparable precision to its counterparts [24],
it works only on the instance level and requires highly ac-
curate instance masks to segment foreground pixels. Most
recently, Objectron [4] proposes a data-driven approach that
learns to regress pixel coordinates of projected box corners
for each category with a tremendous amount of annotated
training data. Such an approach is costly and only limited
to a few categories as the learned model is category-specific.
Moreover, it can only obtain up-to-scale poses without met-
ric scales since it uses a single-view image as input. On the
contrary, our method can leverage the visual-inertial odom-
etry to recover metric scales during the mapping stage, thus
being able to recover metric 6D poses at test time.
Feature-Matching-Based Pose Estimation. Visual local-
ization pipelines based on feature-matching have long been
studied. Traditionally, the localization problem is solved
by finding 2D-3D correspondences between input RGB
images and a 3D model from SfM with hand-crafted lo-
cal features like SIFT [20], FAST [29] and ORB [30].
Recently, learning-based local feature detection, descrip-
tion [10–12, 39] and matching [32, 36] surpass these hand-
crafted methods and have substituted the traditional coun-
terparts in the localization pipeline. Notably, Hierarchical
Localization (HLoc) [31] provides a complete toolbox for
running SfM with COLMAP [33] and feature extraction and
matching with SuperGlue [32]. Our method is inspired by
SuperGlue in terms of using self- and cross-attention layers
for feature matching. However, SuperGlue only focuses on
2D-2D matching between images and does not consider the
graph structure of the SfM map. Our method uses graph at-
tention networks [40] to process and aggregate 2D features
that correspond to a 3D SfM point (i.e., a feature track),
which preserves the graph structure of the SfM during 2D-
3D matching.
Many traditional methods for object recognition and
pose estimation also share the feature-based pipeline sim-
ilar to visual localization. These methods first build object
models by reconstructing sparse point clouds from matched
keypoints across the views [9,21,35,37], and localize with
the sparse point cloud model given a query image. Some
approaches [28, 45] propose to build a point cloud model
online with a framework similar to Simultaneous Localiza-
tion and Mapping (SLAM). Notably, BundleTrack [45] pro-
poses an online pose tracking pipeline without instance- or
category-level models, which resembles ours mostly. How-
ever, it uses 2D-2D feature matching instead of 2D-3D as
in ours. To recover the 3D information, it also takes depth
map as input which could limit its usage in AR.
3. Method
An overview of the proposed method is presented in
Fig. 2. In the setting of one-shot object pose estimation
introduced in Sec. 1, a video scan surrounding the object is
captured with a mobile device (e.g. iPhone or iPad). Given
the video scan and a test image sequence {Iq}, the objective
of one-shot object pose estimation is to estimate the object
poses {ξq} ∈SE(3) defined in the camera coordinate sys-
tem, where q is the key-frame index in the video. We use
bold letters (e.g. I) to denote tensors, calligraphy letters
(e.g. G) to denote graphs and {·} to denote a set of these
entities.
3.1. Preliminaries
Data Capture and Annotation. During the data capture,
the object is assumed to be set on a flat surface and remains
static during the capture. To define the canonical pose of
the object, an object bounding box B is annotated in AR,
with the camera poses {ξi} ∈SE(3) tracked by off-the-self
AR toolboxes like ARKit [2] or ARCore [1]. i is the frame
index. The capture interface is shown in Fig. 4. B is pa-
rameterized by the center location, dimensions and rotation
around the z-axis (yaw angle). After the data capture and
annotation, the pipeline of OnePose can be separated into
the offline mapping phase and the online localization phase.
Structure from Motion. In the mapping phase, given a
set of images {Ii} extracted from the video scan, we use
Structure from Motion (SfM) to reconstruct the sparse point
cloud {Pj} of the object, where j is the point index. Since
B is annotated, {Pj} can be defined in the object coordi-
nate system. A visualization of all correspondence graphs
of the object {Gj} can be found in Fig. 2 (3,4). Specifically,
2D keypoints and descriptors are first extracted from each
image and matched between images to produce 2D-2D cor-
respondences. Every reconstructed point Pj corresponds to
a set of matched 2D keypoints and descriptors {F2D
k } ∈Rd
where k is the keypoint index and d is the dimension of the
descriptor. The correspondence graphs {Gj}, which are also
called the feature tracks, are formed by keypoint indexes for
{P3D
j } as visualized in Fig. 2 (3,4).
Pose Estimation through Visual Localization. In the lo-
calization phase, a sequence of query images {Iq} are cap-
tured in real-time. Localizing the camera poses of the query
images {ξ−1
q } with respect to {Pj} produces the object
poses {ξq} defined in the camera coordinate.
For each Iq, 2D keypoints and descriptors {F2D
q } ∈Rd
are extracted and used for matching. In modern visual lo-
3

2. Structure from Motion
1. Capture and Annotate
3. Correspondence Graphs
?
GATs
4. 2D-3D Matching
?
sNAFL2pr1pfVZduBlvBVUkKostSNy4r2Ac0oUymk3boZBJmJkIJ+QL3bvUX3Ilbv8I/8DOctFlo64ELh3Pu5R6OH3OmtG1/WaWNza3tnfJuZW/4PCoenzSU1EiCe2Si
Edy4GNFORO0q5nmdBLikOf074/u839/iOVikXiQc9j6oV4IljACNZGcutuiPXUD9J2Vh9Va3bDXgCtE6cgNSjQGVW/3XFEkpAKThWaujYsfZSLDUjnGYVN1E0xmSGJ
3RoqMAhV6yJyhC6OMURBJM0Kjhfr7IsWhUvPQN5t5RLXq5eJ/3jDRwY2XMhEnmgqyfBQkHOkI5QWgMZOUaD43BPJTFZEplhiok1Nf74IRmhgjKxiqnFWi1gnvWbDu
WrY981aq12UVIYzOIdLcOAaWnAHegCgRie4QVerSfrzXq3PparJau4OYU/sD5/AEIpmno=</latexit>B
{⇠i}
{Ii}
{Gj}
M3D
{F2D
q }
{Pj}
{F3D
j }
⇠−1
q
Gj
hv+gZ/hAHtQsJOKlXd6e4KYs60cd0vJ7e2vrG5ld8u7Ozu7R8UD49aOkoUhSaNeKQ6AdHAmYSmYZDJ1ZARMChHYyvZ37AZRmkbw1kxh8QYaShYwSY6W7cu+R9dP7ablfLkVdw68SryMlFCGRr/43RtENBEgDeVE67nxsZPiTKMcpgWeomGmNAxGULXUkEaD+dHzFZ1YZ4DBStqTBc/X3REqE1hMR2E5BzEgvezPxP6+bmPDKT5mMEwOSL
haFCcmwrPv8YApoIZPLCFUMXsrpiOiCDU2oz9bJKMQWmNasNF4y0Gskla14l1U3JtqVbPQsqjE3SKzpGHLlEN1VEDNRFAj2jF/TqPDlvzrvzsWjNOdnMfoD5/MHKyOZXw=</latexit>⇠q
PnP
5. Pose Estimation
{F2D
k }
Figure 2. Overview of OnePose. 1. For each object, a video scan with RGB frames {Ii} and camera poses {ξi} are collected together
with the annotated 3D object bounding box B. 2. Structure from Motion (SfM) reconstructs a sparse point cloud {Pj} of the object. 3.
The correspondence graphs {Gj} (
,
) are built during SfM, which represent the 2D-3D correspondences in the SfM map.
represents
the camera to be localized in the object frame. 4. 2D descriptors {F2D
k } ( , ) are aggregated to 3D descriptors {F3D
j
} (
) with the
aggregration-attention layer. {F3D
j
} are later matched with 2D descriptors from the query image {F2D
q } ( ) to generate 2D-3D match
predictions M3D. 5. Finally, the object pose ξq is computed by solving the PnP problem with M3D. Grey background color denotes
offline processes. Best viewed in color.
calization pipeline [31], an image retrieval network is used
to extract image-level global features, which can be used
to retrieve the image candidates from the SfM database for
2D-2D matching. Increasing the number of image pairs to
be matched will significantly slow down the localization,
especially for learning-based matchers like SuperGlue [32]
or LoFTR [36]. Reducing the number of images retrieved
can result in a low localization success rate and thus a trade-
off has to be made between runtime and pose estimation
accuracy.
To remedy this problem, we propose to directly perform
2D-3D matching between the query image and the SfM
point clouds. Direct 2D-3D matching avoids the need of the
image retrieval module, and thus can maintain localization
accuracy while being fast. In the next section, we describe
how to obtain the 2D-3D correspondences M3D.
3.2. OnePose
Graph Attention Networks (GATs) for 2D-3D Match-
ing. Direct 2D-3D matching requires 3D feature descrip-
tors. Since a 3D point Pj is associated with multiple F2D
k′
in Gj, an aggregation operation is needed to update the 3D
descriptors, defined as {F3D
j } ∈Rd which are initialized by
averaging the corresponding 2D descriptors. The aggrega-
tion operation could cause information loss since it reduces
multiple descriptors to one. An ideal aggregation operation
should be able to adaptively preserve the most informative
2D features in {F2D
k } for the 2D-3D matching according to
different F2D
q .
We propose to use the graph attention layer in [40]
to achieve the adaptive aggregation.
We name it the
aggregation-attention layer. The aggregation-attention layer
operates on each individual Gj.
For every Gj, denoting
the weight matrix as W ∈Rd×d, the aggregation-attention
layer is defined as:
  \resize
bo x {0.
8 \ h siz
e
 
}
{!}{ $
 \oper
at orname {Aggr}(\{\mathbf {F}_{k^\prime }^{2D}\}, \mathbf {F}_j^{3D}) = \mathbf {F}_j^{3D} + \underset {\forall k^\prime \in \mathcal {G}_j}{\sum } \alpha _{k^\prime } \mathbf {{F}}_{k^\prime }^{2D}, $ } 
  \ r esizebo
x {0.7 4\hsiz e }{
!} { $ \a
l pha _{k^\prime } = \underset {\forall k^\prime \in \mathcal {G}_j}{\operatorname {softmax}}(\operatorname {sim} (\mathbf {W} \cdot \mathbf {F}_{k^\prime }^{2D}, \mathbf {W} \cdot \mathbf {F}_j^{3D})) $ } 
with sim(·, ·) = ⟨Rd, Rd⟩∈R computes the attention coef-
ficient, which measures the importance of the descriptors in
the aggregation operation.
Inspired by [32, 36], we further use self- and cross-
attention layers following the aggregation-attention layers
to process and transform the aggregated 3D descriptors and
query 2D descriptors. A set of aggregation-, self- and cross-
attention layers forms an attention group, specifically:
 
 
\
r
e
s
i
z
e
box {
0 . 8 \hsize }{
! }{ $ \
b egi
n {ca
s e s } \{\math
b f {\ha
t {F
}}_j^
{ 3 D }\} = \ope
r at ornam
e {A
ggr}(
\ {\ mathb
f { F}_{k}^{2D}
\ }, \{\m
a thbf {F}_j^{3D}\}),\\ \{\mathbf {\tilde {F}}_q^{2D}\} = \operatorname {Self}(\{\mathbf {F}_q^{2D}\}, \{\mathbf {F}_q^{2D}\}),\\ \{\mathbf {\tilde {F}}_j^{3D}\} = \operatorname {Self}(\{\mathbf {\hat {F}}_j^{3D}\}, \{\mathbf {\hat {F}}_j^{3D}\}), \\ \{\mathbf {F^\prime }_q^{2D}\}, \{\mathbf {F^\prime }_j^{3D}\} = \operatorname {Cross}(\{\mathbf {\tilde {F}}_q^{2D}\}, \{\mathbf {\tilde {F}}_j^{3D}\}). \end {cases} $ } 
The proposed architecture of graph attention networks
(GATs) is composed of N stacked attention groups. Intu-
itively, the aggregation-attention layers will adaptively at-
tend to different F2D
k
in Gj according to its relevance with
4

Aggregation-attention
Cross-attention
Self-attention
Figure 3. Different types of attention layers in GATs. {F3D
j
} :
, {F2D
q } :
, {Gj} : (
,
) {F2D
k }: ( , ). For clarity, the
complete relations of attentions in the figure are not shown.
F2D
q , thus preserving more descriminative information for
2D-3D matching. By interleaving the aggregation-attention
layers with self- and cross-attention layers, {F2D
k }, {F3D
j },
{F2D
q } can exchange information with each other, thus
making the matching globally-consented and context-
dependent.
Match Selection and Pose Calculation. We follow [36]
to use the dual-softmax operator to differentiablly extract
match confidence scores P3D. The score matrix S between
the transformed features is first calculated by S (q, j) =
⟨F′2D
q , F′3D
j ⟩. Formally, the matching confidence C3D is
obtained by:
  \mat hb f {C}_{3 D} (q, j) = \opera to rna me { softmax}\left (\mathbf {S}\left (q, \cdot \right )\right )_j \cdot \operatorname {softmax}\left (\mathbf {S}\left (\cdot , j\right )\right )_q. \label {eq:dual-softmax} 
After selecting a confidence threshold θ, C3D becomes
a permutation matrix M3D, which represents the 2D-3D
match predictions. With M3D, the object pose in the cam-
era coordinate ξq can be computed by the Perspective-n-
Point (PnP) algorithm with RANSAC.
Supervision. The supervision signal Mgt
3D can be directly
obtained from filtered 2D-3D correspondences in the SfM
maps in the training set. The loss function L is the focal
loss [19] over the confidence scores C3D returned by the
dual-softmax operator:
  L = - 
(1 - \mat hbf {
C} ^\p ri me _{3D}\left (q, j\right )) ^ \gamma \log \mathbf {C}^\prime _{3D}\left (q, j\right ), \label {eq:loss_coarse} 
 
 \
resiz eb o x {.8\ hs
iz e }
{!}{ $ \ b
eg
in {c as e s } \math bf
 { C}^
\prim e _{ 3D}(q, j) = \mathbf {C}_{3D}(q, j) &\text {if} \: \mathcal {M}_{3D}^{gt}(q, j) = 1 \\ \mathbf {C}^\prime _{3D}(q, j) = 1 - \mathbf {C}_{3D}(q, j) &\text {if} \: \mathcal {M}_{3D}^{gt}(q, j) \neq 1. \end {cases} $ } 
Online Feature-based Pose Tracking.
The above-
mentioned pose estimation module takes only sparse key-
frame images as input. To obtain stable object poses for
AR applications, we further equip OnePose with a feature-
based pose tracking module, which processes every frame
in the test sequence. Similar to a SLAM system, the pose
tracking module reconstructs a 3D map online and main-
tains its own key-frame pool. At each time-step, tracking
adopts a tightly-coupled approach and relies on both the
Sequence 1
Align
Sequence 2
Data Collection
Bundle Adjustment
Figure 4. Dataset collection and sequence registration by joint
bundle adjustment. The AR-based dataset collection and annota-
tion interface are shown on the left. Multiple collected sequences
of the same object are aligned according to the bounding box an-
notations. The final camera poses are optimized by bundle adjust-
ment with the camera poses from ARKit as prior.
prebuilt SfM map and the online-built 3D map to find 2D-
3D correspondences and solve for 6D poses. Since the pose
tracking module preserves 2D and 3D information of the
test sequence in the online-built map, it can be more stable
than the single-frame-based pose estimation module. The
pose estimation module helps to recover and re-initialize
the tracking module when it fails. We provide more de-
tails about the pose tracking module in the supplementary
material.
Remarks on the One-Shot Setting. Other than not using
CAD models or additional network training, the one-shot
setting of OnePose has many advantages compared with ex-
isting instance- or category-level pose estimation methods.
During the mapping phase, OnePose takes as input a sim-
ple video scan of an object and builds an instance-specific
3D representation of the object geometry. Similar to the
role of CAD models in instance-level methods, the 3D ge-
ometry of the object is crucial for recovering object poses
with metric scales. In the localization phase, learned local
feature matching in OnePose can handle large changes in
viewpoint, lighting and scale, making the system more sta-
ble and robust compared to category-level methods. The
local-feature-based pipeline also allows the pose estimation
module to be naturally coupled with a feature-based track-
ing module to realize efficient and stable pose tracking.
3.3. OnePose Dataset
Since there is no existing large-scale dataset that can
fit the setting of one-shot pose estimation, we collected a
dataset with multiple video scans of the same object put
in different locations. The OnePose dataset contains over
450 video sequences of 150 objects. For each object, mul-
tiple video recordings, accompanied camera poses and 3D
bounding box annotations are provided. These sequences
are collected under different background environments, and
each has an average recording-length of 30 seconds cover-
ing all views of the object. The dataset is randomly divided
5

into training and validation sets. For each object in the val-
idation set, we assign one mapping sequence for building
the SfM map, and use a test sequence for the evaluation.
To reduce the manual labor of data annotation, we pro-
pose a semi-automatic approach to simultaneously collect
and annotate the data in AR. To be specific, an adjustable
3D bounding box is rendered onto the image in AR, as
shown in Fig. 4. The only manual work is to adjust the
rotation and rough dimensions of the 3D bounding box.
Visualizations of the data capture interface and the post-
processing process are shown in Fig. 4.
The objective of the post-processing is to reduce the pose
drift error of ARKit for each sequence and ensure consis-
tent pose annotations across sequences. To achieve this, we
first align sequences with the annotated bounding boxes and
perform bundle adjustment (BA) with COLMAP [31, 33].
Feature matches used in the BA are extracted with Super-
Glue. As the backgrounds are different between sequences,
we extract matches only in the foreground (i.e., within the
2D object bounding boxes) between all matchable pairs of
images. For more details about our data collection and pro-
cessing pipeline, please refer to our supplementary material.
4. Experiments
In this section, we first introduce our selection of base-
line methods and evaluation protocols, as well as evaluation
metrics on our proposed OnePose dataset in Sec. 4.1, fol-
lowed by implementation details of our method in Sec. 4.2.
Experimental results and ablation studies are detailed in
Sec. 4.3 and Sec. 4.4, respectively.
4.1. Experiment Settings and Baselines
Baselines.
We compare our method with the following
baseline methods in three categories: 1) Visual Localiza-
tion methods are most relevant to the proposed method in
terms of estimating the pose based on local feature match-
ing. To be specific, we compare our method with HLoc [31]
using different keypoint descriptors (SIFT [20] and Super-
Point [10]), as well as matchers (Nearest Neighbour, Super-
Glue [32]). 2) Instance-level method PVNet [26, 27]. 3)
Category-level method Objectron [4]. To the best of our
knowledge, Objectron is the only method for category-level
object pose estimation with RGB image as input.
Evaluation Protocols. We apply per-frame pose estimation
with the proposed method without the pose tracking mod-
ule for a fair comparison in all the experiments. For our
Visual Localization baselines and the proposed method, we
use the same video scan to build the SfM map for the local-
ization. Note that the original image retrieval module used
for large scale scenes does not generalize well to objects,
thus we equally sample a subset of five images with equal
intervals from database images as retrieved images for fea-
ture matching. To train our instance-level baseline PVNet,
we use 3D box corners instead of sampled semantic points
from CAD models as keypoints to vote for, and further sup-
ply auxiliary mask supervision which is indispensable for
training PVNet. Due to the data demanding nature of the
category-level baseline Objectron [4], we directly use the
models provided by the authors, which are trained on the
original Objectron dataset.
Metrics. For evaluation metrics, we cannot directly adopt
the commonly used ADD metric [13] and 2D projection
metric [6] since CAD models are unavailable in our setting.
Another commonly used metric for evaluating the quality
of predicted object pose is the 5cm-5deg metric proposed
in [34] which deems a predicted pose as correct if the error
is below 5cm and 5°. We further narrow down the criteria to
1cm-1deg and 3cm-3deg following a similar definition to set
up more strict metrics for the pose estimation in augmented
reality application. We divide the objects to three splits by
their diameters with 40 cm and 25 cm as thresholds. When
comparing with instance-level baseline and category-level
baseline, we follow the metrics used in the original paper.
4.2. Implementation Details
During the mapping phase, to maintain a fast mapping
speed, we reuse {ξi} and use triangulation to reconstruct
the point cloud, without further optimization on the cam-
era poses by bundle adjustment. During the localization
phase, we assume the 2D bounding box of the object is
known, which can be easily obtained from an off-the-shelf
2D object detector (e.g. YOLOv5 [3]) in practice. To re-
duce possible mismatches in pose estimation, only the 3D
points inside the annotated 3D bounding box are preserved
during mapping, and only the 2D features inside the de-
tected 2D bounding box are preserved during localization.
For the network design, we use N = 4 attention groups in
GATs. Linear Attention [14, 41] is used in all the attention
layers following [36]. As the input of GATs, we randomly
sample or pad a set of eight features from {F2D
i
} associ-
ated with each F3D
i
for all experiments in the paper. The
{F3D
i
} are initialized by averaging all of the associated fea-
tures {F2D
i
}.
4.3. Evaluation Results
Comparison with Visual Localization Baselines.
We
compare our approach with visual localization baselines
with different feature extractors and matchers, and present
the results in Tab. 1.
HLoc (SPP + SPG) is the base-
line with learning-based feature extractor (SuperPoint) and
matcher (SuperGlue), which mostly resembles our method
among all the three variants.
Our method performs on-
par or slightly better compared with HLoc (SPP + SPG),
6

Large Objects
Medium Objects
Small Objects
Time (ms)
1cm-1deg
3cm-3deg
5cm-5deg
1cm-1deg
3cm-3deg
5cm-5deg
1cm-1deg
3cm-3deg
5cm-5deg
HLoc (SIFT + NN)
0.314
0.572
0.572
0.432
0.575
0.608
0.248
0.468
0.515
116.27
HLoc (SPP + NN)
0.357
0.675
0.675
0.508
0.659
0.706
0.342
0.612
0.687
136.98
HLoc (SPP + SPG)
0.435
0.813
0.813
0.643
0.793
0.831
0.432
0.739
0.837
618.29
Ours
0.471
0.856
0.856
0.629
0.816
0.858
0.405
0.729
0.832
58.31
Table 1. Comparison with the Visual Localization baselines. Our method are compared with HLoc [31] with different detectors including
SIFT [20] and SuperPoint (SPP) [10], and matchers including Nearest Neighbor (NN) and SuperGlue (SPG) [32].
Obj. ID
0447 0450 0488 0493 0494 0524 0594
PVNet
0.253 0.127 0.042 0.094 0.192 0.119 0.077
Ours
0.900 0.981 0.740 0.873 0.819 0.679 0.789
Table 2.
Comparison with the instance-level baseline.
Our
method is compared with PVNet [26] on selected objects from the
OnePose dataset with the 5cm-5deg metric.
while HLoc (SPP + SPG) takes ten times the runtime of
our method. We believe the improvement comes from the
ability of our method to selectively aggregate context from
multiple images benefited from our GATs design, instead of
only focusing on the two images being matched.
Comparison with the Instance-level Baseline PVNet.
The proposed method is compared with PVNet [26] with
5cm-5deg on selected objects from our OnePose dataset and
the results are as presented in Tab. 2. To obtain segmen-
tation masks for training PVNet, we need to additionally
apply dense 3D reconstruction and render the reconstructed
meshes to obtain masks on the data sequences. This process
is time-consuming and greatly limits our choices for objects
because of the quality of 3D reconstruction. Our method
achieves much higher precision than PVNet, which demon-
strates the superiority of our method. PVNet relies on mem-
orizing the mapping from image patches to object-specific
keypoints. Without pre-training on large-scale synthetic im-
ages (rendered with CAD models) that densely cover all
possible views, the performance of PVNet drops drastically.
Conversely, our method is able to leverage the learned local
features that are relatively viewpoint-invariant and thus gen-
eralize to unseen views while maintaining the precision.
Comparison with the Category-level Baseline Objec-
tron. We compare our method with Objectron [4] on all
objects in the Shoe and Cup categories with the metrics used
in the original paper and present the results in Tab. 3. For
mean pixel error of 2D projection, the results of Objectron
on our dataset are far from the reported results for the two
categories on Objectron dataset. This is because of the de-
viations in ground-truth annotations between the Objectron
dataset and our dataset. For a fair comparison, we further
apply scaling and center alignment operations to the predic-
tions of Objectron to alleviate this gap and provided results
respectively as Objectron (S) and Objectron (S+C) in Tab. 3.
Although the performances of Objectron do get boosted and
are comparable with the reported results in the original pa-
per, our method surpasses it by a large margin. Our method
Obj. ID
0415 0475 0476 Cup
0592 0593 0594 0595 Shoe
Mean pixel error of 2D projection
Objectron
0.269 0.474 0.483 0.054 0.189 0.183 0.118 0.124 0.039
Objectron (S)
0.170 0.340 0.347
-
0.123 0.115 0.092 0.089
-
Objectron (S+C) 0.158 0.331 0.342
-
0.103 0.098 0.084 0.079
-
Ours
0.047 0.022 0.013
-
0.089 0.016 0.026 0.012
-
Average precision at 15 ° Azimuth error
Objectron
0.364 0.131 0.217 0.644 0.677 0.733 0.774 0.945 0.586
Ours
1.0
1.0
1.0
-
0.855 0.998 0.984
1.0
-
Average precision at 10 ° Elevation error
Objectron
0.707 0.906 0.821 0.837 0.794 0.842 0.622 0.995 0.754
Ours
1.0
1.0
1.0
-
0.831 0.996 0.973
1.0
-
Table 3.
Comparison with the category-level baseline.
Our
method is compared with Objectron [4] with auxiliary scale ad-
justment (S) and center alignment (S+C). The category-level eval-
uation results reported in the original paper are provided in grey
background below the name of the category.
outperforms Objectron evidently in the average precision
of azimuth error and elevation error, especially for the ob-
jects of Cup category where the shape and appearance may
vary significantly between instances. These experiments il-
lustrate the limited generalization ability of category-level
methods to new object instances.
Runtime Analysis. We report the runtimes of our visual
localization baselines and our method in Tab. 1. The run-
time consist of feature extraction for the query image with
SuperPoint and the 2D-3D matching process without 2D de-
tection and PnP. Our method runs ∼10× faster than HLoc
(SPP + SPG). All the experiments are conducted on an
NVIDIA TITAN RTX GPU.
4.4. Ablation Studies
In this section. we conduct several ablation experiments
by substituting GATs with simpler counterparts of the fea-
ture aggregation and matching modules. All the results for
our ablation studies are presented in Tab. 4.
Effectiveness of the Aggregation-Attention Layer. We
validate the effectiveness of the proposed aggregation-
attention layers by substituting the corresponding aggrega-
tion layers in GATs by the averaging operation and report
the result in Tab. 4 as (i). Notice the 2D-3D matching is
still based on a GNN with self- and cross-attention layers,
which is similar to SuperGlue. Without the aggregation-
attention layers, the results dropped significantly for large
and medium objects, which indicates the effectiveness of
aggregation-attention layers. The simple averaging opera-
7

Ours (GATs)
Ablation (iv)
HLoc (SIFT + NN)
HLoc (SPP + SPG)
Objectron
PVNet/Objectron
PVNet
No inlier
No inlier
PVNet
Figure 5. Qualitative results. Every two rows show the results on one test image. Green bounding boxes denote predictions and the red
ones denote the ground truth. The 2D-3D matches are visualized by projecting the 3D matches of the detected 2D points (shown on the
left image) onto the image plane (shown on the right image). Perfectly horizontal lines indicate correct 2D-3D matches. Reprojection error
less than 10 px (in 512 × 512 images) is colored as green. RANSAC is applied to the matches to remove outliers. Our method is able to
produce a larger quantity of matches compared to baseline HLoc (SIFT + NN). The matches from our method are also more accurate and
less noisy compared to the baseline method Ablation (iv)). HLoc (SPP + SPG) achieves similar results with our method, but at a ∼10×
run-time cost. PVNet and Objectron only obtained reasonable results in the first and last examples, respectively. Best viewed in color.
Components
Large Objects Medium Objects Small Objects
2D Feat.
Aggr.
Matching
Ours
SPP
GATs
GNN
0.471
0.629
0.405
i
SPP
Avg.
GNN
0.449
0.602
0.390
ii
SPP
Avg.
NN
0.426
0.570
0.367
iii
SPP
K-Means
NN
0.431
0.595
0.379
iv
SIFT
Avg.
NN
0.355
0.470
0.256
v
SIFT
K-Means
NN
0.369
0.450
0.281
Table 4. Ablation studies. Different alternatives for components
in the proposed method are compared with the 1cm-1deg metric.
SPP stands for SuperPoint and NN stands for Nearest Neighbor.
tion cannot adaptively select relevant information from dif-
ferent viewpoints according to different query features.
Other variants with 2D-3D NN Matching. To provide
more comparisons with traditional pipelines [9,35,37] that
estimate object pose with local features and 2D-3D match-
ing, we also experimented with variants of our method
based on different local features, feature aggregration meth-
ods and matchers for 2D-3D matching. The results are re-
ported in Tab. 4 as (ii - v). (ii - v) are still unable to pro-
duce comparable results with our approach. Compared with
(ii) and (iv) that use averaging for feature aggregation, our
method consistently outperforms them by a significant mar-
gin. Similar to the analysis for (i), simply averaging the
features from different viewpoints loses view-dependent in-
formation. For (iii) and (v), substituting averaging with K-
Means clustering could provide richer 3D features but the
results are still not comparable with ours.
Qualitative Comparisons. We provide some qualitative
results to compare our method with baseline methods in
Fig. 5. Please read the caption for details.
5. Conclusion
In this paper, we propose OnePose for one-shot ob-
ject pose estimation.
Unlike existing instance-level or
category-level methods, OnePose does not rely on CAD
models and can handle objects in arbitrary categories with-
out instance- or category-specific network training. Com-
pared with localization-based baseline methods, instance-
level baseline method PVNet and category-level baseline
method Objectron, OnePose achieves better pos