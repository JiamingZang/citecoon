# Scalable Unseen Objects 6-DoF Absolute Pose Estimation With Robotic Integration

> 2026 · id: W7155098975 · arXiv: 2503.05578 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
U
NSEEN object 6-DoF robotic manipulation is a fun-
damental task underlying scalable robotic applications
across diverse domains [1]–[4]. At its core lies the challenge of
estimating the 6-DoF absolute pose of objects not encountered
arXiv:2503.05578v4  [cs.CV]  17 Apr 2026

IEEE TRANSACTIONS ON ROBOTICS, 2026
2
during training [5]–[10]. Typically, 6-DoF pose comprises of
3-DoF rotation and 3-DoF translation of an object coordinate
system relative to the camera coordinate system [11]–[17].
Object pose estimation methods can be divided into three
main categories. Instance-level methods [18]–[26] have at-
tained high precision but are limited to objects encountered
during training. In contrast, category-level methods [27]–[34]
can generalize to objects within the same category but still
necessitate retraining for novel object categories. Furthermore,
some unseen object pose estimation methods [35]–[41] have
been proposed recently that do not require retraining for novel
object categories, thereby exhibiting enhanced scalability.
Unseen object pose estimation methods can be further
divided into two categories: CAD model-based [42]–[45],
where a textured CAD model of the unseen object is required
during training and inference; manual reference view-based
[46]–[48], where a set of manually labeled reference views
of the unseen object are required. Accurate textured CAD
models can only be obtained with specialized equipment and
expert knowledge, which hinders scalability in mobile devices
[49], [50]. Since manual reference views are relatively easy
to acquire, methods in this category offer greater scalability.
Manual reference view-based methods typically solve pose
through 3D object reconstruction or directly obtain coarse pose
via template matching, as shown in Fig. 1 (a), where the switch
indicates whether 3D reconstruction is required based on dense
reference views. Dense reference views consume time for
acquisition and memory for storage. Furthermore, template
matching-based methods require the use of novel template gen-
eration techniques or an additional pose refinement overhead
which further increases the computational complexity.
To address the aforementioned challenges, our motivation
is to explore a CAD model-free, sparse reference view-based
unseen object 6-DoF absolute pose estimation framework,
eliminating the need for either 3D object reconstruction or
template-based retrieval. Specifically, we formulate the task as
the extreme case of sparse reference view, where only a single
reference is available. Motivated by robotic manipulation
scenarios, we design a scalable label collection pipeline where
each unseen object is annotated with a single RGB-D refer-
ence view in a semi-automatic manner, while absolute pose
recovery is obtained through the annotated reference view.
The overview of our task setup is shown in Fig. 1 (b). The
robot first captures the object from its default manipulation
viewpoint, and a custom-developed annotator provides the
corresponding 6-DoF pose label. Given only a single annotated
view as the sparse reference prior of an unseen object, our goal
is to accurately estimate its 6-DoF absolute pose from arbitrary
novel viewpoints in different scenes. However, using a single
reference introduces several unique challenges, including large
pose discrepancies and limited spatial information.
With the task setup, we propose a scalable SinRef-6D
framework. Our key idea is to iteratively establish point-wise
alignment between the single reference view and a query view
in a common coordinate system to solve the 6-DoF pose of
unseen objects. SinRef-6D introduces two key components: 1)
Iterative object-space point-wise alignment, which addresses
large pose discrepancies by leveraging geometric and spatial
consistency to refine pose estimation; 2) State Space Models
(SSMs), which efficiently capture long-range spatial depen-
dencies from single-view data, offering linear computational
complexity and strong spatial modeling capability. Specifi-
cally, we propose to align the reference and query point clouds
within the object coordinate system (Sec. III-C). Given the
importance of spatial information for point-wise alignment and
the need for a lightweight model for mobile deployment, we
introduce Point and RGB SSMs (Sec. III-D) to establish point-
wise alignment for pose solving (Sec. III-E). To handle the
potentially large pose discrepancies between the reference and
query views, we propose to iteratively refine the alignment in
the object coordinate system, which gives more accurate and
robust pose estimation (Sec. III-F). Furthermore, we develop a
complete hardware-software robotic system that integrates the
proposed SinRef-6D to evaluate its scalability in real-world
scenarios (Sec. IV), as shown in Fig. 1 (c) and (d). Our main
contributions are summarized as follows:
• We introduce an efficient and scalable task setup for
unseen object 6-DoF absolute pose estimation using only
a single reference view captured during robotic manip-
ulation, eliminating the need for computation-intensive
template matching and multi-view reconstruction. We
further develop an integrated hardware-software robotic
system tailored to the proposed task setup and framework,
validating their efficacy in real-world scenarios.
• We propose an object-space point-wise alignment strategy
with iterative refinement, facilitating direct alignment of
query and reference views while effectively handling
large pose discrepancies. This enhances geometric consis-
tency and spatial awareness, enabling unseen object pose
estimation without category-specific retraining.
• We propose Point and RGB SSMs to capture rich spatial
information for establishing point-wise alignment, en-
abling efficient long-range spatial modeling with linear
computational complexity.
• Extensive experiments demonstrate that our task setup
and framework enable highly scalable 6-DoF robotic
grasping of unseen objects in diverse environments.
The remainder of this paper is structured as follows. Sec.
II reviews recent advances in unseen object pose estimation.
Sec. III introduces the proposed task setup and corresponding
framework. Sec. IV describes the developed 6-DoF robotic
grasping system that integrates both hardware and software.
Sec. V presents comprehensive experimental results that vali-
date the scalability of SinRef-6D and the effectiveness of the
robotic system. Finally, Sec. VI summarizes the paper.

## method
We begin with an overview of the overall task setup and
framework, including its input and output (Sec. III-A). We
then describe the initialization process, which involves unseen
object segmentation from the input RGB-D image (Sec. III-B).
Next, we present the proposed point focalization strategy (Sec.
III-C), Point and RGB SSMs (Sec. III-D), and point-wise
alignment for pose solving (Sec. III-E). Finally, we explain
the training procedure and supervision scheme (Sec. III-F).
A. Task Setup and Framework Overview
Overall, our work is problem-driven, aiming to enable
scalable 6-DoF absolute pose estimation of unseen objects for
robotic manipulation with minimal prior information. To this
end, we present a unified system that operates with only a
single reference view, where each component is explicitly de-
signed to address the challenges arising from single-reference,
manipulation-oriented absolute pose estimation. Specifically,
for unseen objects which are not encountered during training,
SinRef-6D takes a pair of RGB-D images captured from
robotic manipulation viewpoints as input: a single reference
image and a query image. The reference image is selected
only once and annotated with a 6-DoF pose through a semi-
automatic manner using our custom-developed pose annotator.
The output of SinRef-6D is the estimated 6-DoF absolute pose
Algorithm 1 Overall Pipeline of SinRef-6D
1: Input: Reference RGB-D image; Query RGB-D image
2: Output: Estimated 6-DoF object pose [Rfinal | tfinal]
3: Segment reference and query images to obtain object RGB
masks Ir, Iq and corresponding depth masks via SAM-
driven similarity matching
4: Back-project depth masks to get object reference and
query point clouds Pr, Pq
5: Transform Pr into the object coordinate system using
known reference pose (obtained via a semi-automated
annotator) [Rr | tr]: P o
r = R⊤
r (Pr −tr)
6: Initialize query pose:
[R1 | t1] ←identity matrix, average coordinate(Pq)
7: for i = 1 to T do
8:
Transform Pq into the object coordinate system:
P i
q = R⊤
i (Pq −ti)
9:
Extract point-wise and RGB features:
10:
Fr ←Point SSM(P o
r ) ⊕RGB SSM(Ir)
11:
F i
q ←Point SSM(P i
q) ⊕RGB SSM(Iq)
12:
Perform point-wise feature alignment:
13:
¯Fr, ¯F i
q ←GeoTransformer(Fr, F i
q)
14:
Compute point-wise affinity: Ai = ¯F i
q ⊗¯F ⊤
r
15:
Estimate 6-DoF object pose via weighted SVD:
16:
[Ri+1, ti+1] = WSV D(Ai, P o
r , Pq)
17: end for
18: Return: [Rfinal, tfinal] ←[RT +1, tT +1]
of the unseen object in the arbitrary query image. Algorithm
1 shows the overall pipeline of the proposed framework.
Figure 2 shows the overall workflow that comprises four
main components: (A) Initialization segments the unseen
object in the input reference and query views. (B) Points
Focalization focalizes the unseen object in the reference and
query views into the object coordinate system using their cor-
responding poses. (C) Point & RGB SSMs employ state space
models to extract point-wise reference and query features.
(D) Point-wise Alignment & Pose Solving derives point-wise
alignment relationships using the features extracted in (C) to
solve the object pose in the query view. In addition, iterating
the process from (B) to (D) allows for further obtaining more
accurate point-wise alignment and object pose.
B. Initialization
Notably, randomly sampling arbitrary rendering viewpoints
may introduce extreme perspectives (e.g., near top-down
views) that deviate significantly from real-world reference
acquisition, whereas manually selecting viewpoints for all
objects would be labor-intensive and may reduce robustness to
viewpoint variations. From a practical real-world application
perspective, during training, the synthetic reference view is
sampled from a viewpoint range that approximates the robotic
manipulation viewpoint while introducing natural perturba-
tions. Importantly, this reference view is not carefully selected.
To support this claim, we additionally generate reference
views using the same rendering protocol as GigaPose [60].

IEEE TRANSACTIONS ON ROBOTICS, 2026
5
SAM & Cropping
Query View
Reference View
(A) Initialization
[𝑹𝑹𝒓𝒓|𝒕𝒕𝒓𝒓]
(B) Points Focalization
𝑪𝑪𝒐𝒐
𝑷𝑷𝒒𝒒𝒊𝒊
𝑷𝑷𝒓𝒓𝒐𝒐
𝑷𝑷𝒓𝒓
𝑷𝑷𝒒𝒒
𝑰𝑰𝒓𝒓, 𝑷𝑷𝒓𝒓
𝑰𝑰𝒒𝒒, 𝑷𝑷𝒒𝒒
𝑭𝑭𝒓𝒓𝑰𝑰
𝑭𝑭𝒓𝒓𝑷𝑷
(C) Point & RGB SSMs
(D) Point-wise Alignment
& Pose Solving
𝑭𝑭𝒓𝒓
𝑭𝑭𝒒𝒒𝒊𝒊
GeoTransformer
…
𝑭𝑭𝒓𝒓
𝑭𝑭𝒒𝒒𝑰𝑰
𝑭𝑭𝒒𝒒𝑷𝑷𝒊𝒊
𝑭𝑭𝒒𝒒𝒊𝒊
𝑪𝑪𝒐𝒐
…
Point-wise
Reference Feature
Point-wise
Query Feature
Point-wise 
Alignment 
𝑪𝑪𝒄𝒄
𝑪𝑪𝒄𝒄
…
…
ഥ𝑭𝑭𝒓𝒓
ഥ𝑭𝑭𝒒𝒒𝒊𝒊
Pose Solver
𝑨𝑨𝒊𝒊
× 𝑲𝑲
𝑷𝑷𝒓𝒓𝒐𝒐
𝑷𝑷𝒒𝒒
[𝑹𝑹𝒊𝒊|𝒕𝒕𝒊𝒊]
[𝑹𝑹𝒊𝒊+𝟏𝟏|𝒕𝒕𝒊𝒊+𝟏𝟏]
[𝑹𝑹𝒊𝒊+𝟏𝟏|𝒕𝒕𝒊𝒊+𝟏𝟏]
Class & Pose Labeling
Similarity Matching
Depth Back-projection
Frozen
Offline Mode
Token & Position Embedding
Point-wise Scan & KNN
…
Point-wise Scan & KNN
Token & Position Embedding
…
Iterative
Fig. 2. Our proposed SinRef-6D framework. Given a normal RGB-D reference view of an unseen object, we aim to predict its 6-DoF absolute pose from any
query view. SinRef-6D comprises four modules: (A) The reference view is labeled via a semi-automatic annotator, then the RGB-D images of the reference
and query views are segmented, and the segmented depth maps are back-projected into point clouds. (B) The corresponding point clouds of the reference and
query views are focalized from the camera coordinate system to the object coordinate system. (C) Leveraging the proposed Point and RGB SSMs (details are
shown in Fig. 3 and Fig. 4), features are extracted from the focalized point clouds and RGB images, forming point-wise reference and query features. (D)
These features are then used to establish point-wise alignment to solve the object pose. Finally, the computed pose is fed back into module (B) to iteratively
improve the accuracy of the point-wise alignment, yielding a more precise object pose.
Specifically, we randomly sample one viewpoint from the 50th
to the 120th in its rendering sequence, which is designed to ap-
proximate the robotic manipulation viewpoint. This simulates
manual reference view acquisition while introducing natural
pose perturbations. During the evaluation in real-world robotic
scenarios, we adopt a semi-automatic manner. The reference
view for each unseen object is captured by the robot from
an occlusion-free manipulation viewpoint and annotated using
our custom-developed annotator. The rotation is determined
using a calibration board, while the translation and size are
manually adjusted through keyboard control (some visualiza-
tions are shown in the first row of Fig. 8). For testing on
public benchmarks, we adopt both reference view acquisition
strategies to align with those used in training.
The pipeline of the initialization process is shown in part
(A) of Fig. 2. Since both the reference and query views
often contain cluttered backgrounds, we first segment the
background. For a fair comparison, we employ Mask R-CNN
[81] or zero-shot CNOS [82] with FastSAM to segment the
input images, and then back-project the segmented depth maps
into point clouds. This results in the segmented RGB images
and point clouds for both reference (Ir, Pr ∈RNr×3) and
query (Iq, Pq ∈RNq×3) views, where Nr and Nq denote the
number of points in the reference and query point clouds,
respectively. Notably, CNOS relies on object CAD models
for rendering template images, which contrasts with our CAD
model-free setup. Based on this, we also use only our single
reference view as the template image for similarity matching
in CNOS segmentation (see the first two rows of Tab. IV for
details) [82].
C. Points Focalization
Since SinRef-6D aims to iteratively align point clouds for
precise object pose solving, our first step is to focalize the
reference and query point clouds within a common coordinate
system. This focalization facilitates point-wise alignment, en-
sures geometric consistency during iterative refinement, and
inherently decouples pose estimation from category priors,
enhancing robustness to unseen objects. Specifically, as the
reference point cloud Pr has a pose annotation [Rr|tr], we
can transform it from the camera coordinate system Cc to the
object coordinate system Co as follows:
P o
r = R⊤
r (Pr −tr) ,
(1)
where tr and Rr denote the annotated translation and rotation,
respectively. ⊤denotes matrix transpose, P o
r
denotes the
reference point cloud in the object coordinate system.
For the query point cloud, we apply the same method to
transform it into the object coordinate system as follows:
P i
q = R⊤
i (Pq −ti) ,
(2)
where ti and Ri represent the translation and rotation of the
object in the i-th iteration. P i
q represents the query point cloud
in the object coordinate system after the i-th iteration. Since
the object pose in the query view is initially unknown, we
do not perform rotation transformation during the first points
focalization and instead set the translation t1 to the average
coordinate of the object. In subsequent iterations, we use
the object pose [Ri+1|ti+1] solved in the previous round for
coordinate transformation. The overall process is shown in part
(B) of Fig. 2.

IEE

## experiments
We first introduce the benchmarks and evaluation metrics
(Sec. V-A), followed by the implementation details (Sec. V-B).
We then compare SinRef-6D with both manual reference view-
based and CAD model-based methods on these real-world
benchmarks to validate its superior performance (Sec. V-C and
Sec. V-D). Next, we evaluate the effectiveness of our approach
in real-world robotic grasping scenarios by deploying it on
our integrated hardware-software robotic system to perform
grasping tasks (Sec. V-E). Finally, we present comprehensive
ablation studies to analyze the contributions of key compo-
nents, the influence of point cloud alignment iterations, and
the effect of random reference view selection (Sec. V-F).
A. Datasets and Evaluation Metrics
Datasets: We conduct extensive experiments on six bench-
mark datasets (LineMod [89], LM-O [90], TUD-L [91], IC-
BIN [92], HB [93], and YCB-V [94]) and real-world robotic
scenes. For a fair comparison, we follow the BOP Challenge
setting [91] to train on the synthetic dataset generated by
MegaPose [58] using the ShapeNet-Objects [95] and Google-
Scanned-Objects [96] datasets. This training dataset comprises
∼2 million images from ∼50K objects.
Evaluation Metrics: 1) Recall of the average point distance
(ADD) that is less than 10% of the object diameter (ADD-
0.1d) [97]. 2) Area under the curve (AUC) of ADD [94];
3) BOP metric: Average Recall (AR) of the visible sur-
face discrepancy (VSD), maximum symmetry-aware surface
distance (MSSD), and maximum symmetry-aware projection
distance (MSPD) metrics [91]. Specifically, we first perform
a quantitative comparison using the ADD-0.1d and AUC of
ADD metrics for each instance in the LineMod [89] and YCB-
V [94] datasets, respectively, aligning with manual reference
view-based methods [62]–[64], [67]–[69], [98]. Subsequently,

IEEE TRANSACTIONS ON ROBOTICS, 2026
9
TABLE I
COMPARISON OF SINREF-6D WITH OTHER MANUAL REFERENCE VIEW-BASED METHODS ON THE LINEMOD DATASET [89], EVALUATED USING THE
ADD-0.1D METRIC. “REF.” AND “RECON.” MEAN ”REFERENCE” AND ”RECONSTRUCTION”. † REPRESENTS GEN6D [69] WITHOUT FINE-TUNING. ∧
INDICATES THAT THE REFERENCE VIEW IS MANUALLY SELECTED FROM THE CORRESPONDING DATASET TO APPROXIMATE THE ROBOTIC MANIPULATION
VIEWPOINT DURING BOTH TRAINING AND TESTING.

## related_work
This section provides an overview of state-of-the-art meth-
ods in unseen object absolute (Sec. II-A and Sec. II-B) and
relative (Sec. II-C) pose estimation, followed by a discussion
on how our work differs from existing approaches.
A. CAD Model-based Methods
Research in the domain of CAD model-based methods first
require obtaining the precise CAD model of the unseen object,

IEEE TRANSACTIONS ON ROBOTICS, 2026
3
which is then used as prior knowledge for pose estimation.
These methods can be further categorized into 1) feature
matching-based and 2) template matching-based.
Feature matching-based methods [51]–[55] learn a model to
match features between the observed image and CAD model,
establishing 2D-3D or 3D-3D correspondences to estimate
object pose. Specifically, GCPose [52] proposes a geometry
correspondence-based approach that leverages generic, object-
agnostic geometric features to establish clear and robust 3D-
3D correspondences. SAM-6D [53] introduces a novel match-
ing score based on semantics, appearance, and geometry to
improve segmentation. For pose estimation, it employs a two-
stage point matching model to establish dense 3D-3D cor-
respondences. FreeZe [54] develops a method that combines
visual and geometric features from various pre-trained models
to improve pose prediction stability and accuracy. MatchU [55]
proposes a technique for predicting object pose from RGB-D
images by integrating 2D texture with 3D geometric cues.
Template matching-based methods [56]–[60] render multi-
ple template views of the object with different poses from
the CAD model. Then, they retrieve the template that best
matches the observed image to obtain a coarse pose, followed
by a refinement process to achieve accurate pose estimation.
For example, MegaPose [58] proposes a render-and-compare-
based method and a coarse-to-fine pose estimation strategy.
GenFlow [59] introduces a shape-constrained recurrent flow
framework that predicts optical flow between the query and
template images while iteratively refining the pose. GigaPose
[60] achieves fast and robust pose estimation by striking
an effective balance between template matching and patch
correspondences. FoundationPose [61] increases the quantity
and diversity of synthetic data based on diffusion model and
achieves superior performance through render-and-compare.
B. Manual Reference View-based Methods
To eliminate the need for a precise CAD model, manual
reference view-based methods employ manual reference views
as the prior knowledge for unseen objects. These methods
can also be categorized into 1) feature matching-based and
2) template matching-based.
Feature matching-based methods [62]–[66] aim to establish
3D-3D correspondences between the query view and reference
views, or 2D-3D correspondences between the query view
and the 3D object representation reconstructed from reference
views. Specifically, FS6D [62] proposes a dense prototype
matching method to explore geometric and semantic relations
between the query view and reference views, estimating the
pose of unseen objects using only a few reference views.
OnePose [63] first utilizes Structure from Motion (SfM) to
reconstruct the 3D representation of the unseen object using
all reference views, and then establishes 2D-3D correspon-
dences between the query view and the reconstructed 3D
representation using a graph attention network. OnePose++
[64] introduces a keypoint-free SfM method to reconstruct
a semi-dense 3D representation of textureless objects by
leveraging the detector-free feature matching approach LoFTR
[67], enhancing robustness against textureless objects.
Template matching-based methods [68]–[72] primarily uti-
lize a retrieval and refinement strategy. They directly use
labeled reference views as templates to retrieve a coarse pose,
followed by a refinement process to enhance accuracy. Specif-
ically, LatentFusion [68] reconstructs 3D object representation
and estimates translation using bounding boxes and depth val-
ues. Then, the initial rotation is determined by angle sampling
and further refined through gradient updates using render and
compare. Gen6D [69] first detects object bounding boxes,
then compares the query and reference images via similarity
scores to obtain an initial pose. Next, the pose is refined via a
proposed refiner. FoundationPose [61] introduces an object-
centric neural field to enable accurate 3D object modeling
and RGB-D rendering, achieving performance comparable
to instance-level methods. GS-Pose [70] joints segmentation
and introduces a 3D gaussian splatting-based refiner, which
simultaneously enhances the accuracy of object localization
and pose estimation.
C. Unseen Object Relative Pose Estimation Methods
Relative object pose estimation [73]–[78] refers to comput-
ing the pose transformation of an object between two different
views. 3DAHV [73] proposes a 3D-aware hypothesis-and-
verification framework for relative pose estimation of unseen
objects from a reference image, achieving robust generaliza-
tion under large pose variations without relying on dense
multi-view supervision. Building on this idea, DVMNet [74]
introduces an end-to-end voxel-based framework that bypasses
discrete hypothesis generation by directly aligning voxelized
3D features from two RGB images, resulting in improved
accuracy and reduced computational cost. In contrast, NOPE
[75] presents a fast, training-free method that estimates relative
pose by predicting pose-conditioned viewpoint embeddings
using an attention-enhanced U-Net, without requiring 3D
models. While these methods demonstrate strong scalability,
the absence of depth information limits their ability to estimate
the full 3-DoF relative translation.
More recently, some works [76]–[78] have explored pose
estimation using a single RGB-D reference view to reduce on-
boarding cost for unseen objects. UNOPose [76] incorporates
depth data and proposes a one-reference-based pose estimation
framework that constructs an SE(3)-invariant reference repre-
sentation and adaptively weights correspondences to handle
low viewpoint overlap. One2Any [77] further introduces a
category-agnostic method for 6-DoF object pose estimation
that leverages a reference-query RGB-D pair to generate pose
embeddings and decode object coordinates. Any6D [78] esti-
mates both object pose and size from an RGB-D anchor image
by leveraging joint object alignment and a render-and-compare
strategy. Despite their effectiveness, these methods primarily
focus on relative pose estimation between the reference and
query views, which is insufficient for robotic manipulation
scenarios where absolute object poses in a common coordinate
system are required for action execution. In contrast, our work
targets single-reference 6-DoF absolute pose estimation under
robotic manipulation settings. To this end, we introduce a
semi-automated reference acquisition and annotation pipeline,

IEEE TRANSACTIONS ON ROBOTICS, 2026
4
a single reference view-based point cloud focalization strategy
to establish a common coordinate system, and SSMs-based
feature extraction networks tailored for the limited geometric
and spatial information available from a single view. This
problem-driven design enables direct deployment in manipula-
tion pipelines while maintaining scalability to unseen objects.
Discussions: Overall, CAD model-based methods depend on
textured CAD models, and manual reference view-based meth-
ods require dense reference views, both adding manual effort
in real-world applications. Related works such as Founda-
tionPose [61] also employ transformer-based architectures for
iterative pose refinement; however, our SSM-based backbone
is explicitly designed to model long-range spatial dependen-
cies under severely limited geometric information, which is
particularly critical in our single-reference setting. Addition-
ally, relative pose estimation methods are not well-suited for
robotic manipulation tasks that require absolute poses for
action execution. Hence, this paper seeks to enable unseen
object 6-DoF absolute pose estimation with a single reference
view, reducing manual overhead and enhancing scalability
for robotic applications. Most recently, 3D foundation models
such as SAM 3D [79] and VGGT [80] suggest a clear trend to-
ward large-scale, data-driven geometric perception. However,
these advances do not diminish the importance of reliable
pose estimation; instead, they increase the demand for scalable
modules that can provide accurate geometric initialization
for annotation bootstrapping and downstream reasoning. In
this broader context, our method can also be viewed as a
complementary component: a practical and scalable solution
for unseen object 6-DoF pose estimation that remains valuable
even as 3D foundation models continue to evolve.