# AugLift: Depth-Aware Input Reparameterization Improves Domain Generalization in 2D-to-3D Pose Lifting

> 2025 · id: W4414457910 · arXiv: 2508.07112 · pdf: https://arxiv.org/pdf/2508.07112 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Estimating 3D human pose from monocular RGB is a long-standing problem
in computer vision, with applications in graphics, robotics, AR/VR, and sports
⋆Corresponding author.
arXiv:2508.07112v4  [cs.CV]  7 Apr 2026

2
N. Warner et al.
Fig. 1: AugLift enriches standard lifting inputs for better generalization.
Standard lifting uses only sparse 2D coordinates, creating depth ambiguity. AugLift
enriches the input with a monocular depth map and keypoint confidence, forming a
compact UADD (c, d, dmin, dmax) that generalizes to occluded and novel poses.
analysis. A widely adopted paradigm decomposes the task into two stages: a 2D
keypoint detector first localizes joints in the image, and a lifting model then
maps these 2D keypoints to 3D joint locations [14,19,27]. However, lifting mod-
els struggle to generalize beyond the lab-style conditions of their training sets,
especially in realistic detection settings with noisy 2D inputs. The root cause
is that 2D keypoints are an extremely sparse representation: they discard geo-
metric cues available in the image that could help disambiguate depth, occlu-
sion, and viewpoint. Combined with the limited diversity of current training
datasets, lifters overfit to the training distribution—state-of-the-art networks
achieving 40–50 mm MPJPE on Human3.6M degrade to over 100 mm on in-the-
wild datasets such as 3DPW [12,22].
A natural response is to enrich the lifting input. Some methods leverage
temporal context from video sequences [19,27], but overfit to motion dynamics
and degrade on out-of-distribution actions. Others condition on dense image
features [23, 26], but risk learning spurious correlations with backgrounds and
scene appearance [26]. Meanwhile, modern monocular depth estimators (MDEs)
have made striking progress and generalize well across scenes, even though they
provide a lower bound for occluded keypoints. MDEs are trained on RGB-D
data that requires only a commodity depth sensor to collect, in contrast to the
multi-camera MoCap setups needed for 3D pose annotation. Yet their potential
as a plug-in signal for 2D–3D lifting remains underexplored.
In this work we revisit the input to lifting and ask: can we improve robust-
ness by augmenting 2D keypoints with noisy, occluded depth cues? We pro-
pose AugLift, which consists of two modules: (1) an Uncertainty-Aware Depth
Descriptor (UADD)—for each keypoint with detector confidence c, we extract
depth statistics from a confidence-scaled neighborhood, forming a compact tuple
(c, d, dmin, dmax) that captures both local geometry and reliability; and (2) a scale
normalization component that handles train/test distance shifts via bounding-
box or depth-based rescaling. AugLift is designed to be DG-compatible: its point-
wise geometric cues do not depend on scene-level statistics, making it naturally
combinable with render-based augmentation methods. Crucially, AugLift is a
change to the representation format of lifting—from 2D coordinates to a 6D ge-
ometric descriptor—rather than a standalone module. It requires no new sensors,
data collection, or architectural changes beyond widening the input layer from
2K to 6K channels, and is composable with any lifting architecture or domain

AugLift: Depth-Aware Input Reparameterization for 2D-to-3D Pose Lifting
3
Fig. 2: Confidence-aware depth sampling. Left: 2D keypoint confidence (top;
blue=confident, red X=occluded) and monocular depth map (bottom; blue=near,
red=far). Right: Low-confidence joints use a wider sampling radius for robust depth
statistics; high-confidence joints use a tight radius for precise estimates.
generalization technique. Using MDE for lifting is not trivial: MDE estimates
are noisy and provide only the nearest visible surface—a lower bound on the
true joint depth, not the depth of occluded joints—and exhibit domain shift
across datasets. UADD’s specific design—confidence-modulated neighborhood
size that widens under occlusion, robust summary statistics (dmin, d, dmax) that
encode this lower bound alongside a central estimate—addresses each of these
challenges.
We validate AugLift across three complementary settings. In the detection
setting, across four datasets and four architectures, AugLift consistently im-
proves both OOD (10.1% avg) and ID (4.0%) performance; post-hoc analysis
shows gains concentrate on novel poses and occluded joints, where depth statis-
tics resolve front–back ambiguities and confidence regulates sampling neighbor-
hoods. In the GT 2D + DG setting, combining AugLift with PoseAug [4] via a
novel live depth generation pipeline achieves state-of-the-art cross-dataset results
(Section 5). AugLift also complements dense image features, with the compact
UADD providing comparable OOD gains to dense learned features while requir-
ing far fewer parameters (Section 6).
Contributions.
– C1. AugLift = UADD + Scale Normalization. We propose AugLift, a
change to the representation format of 2D→3D lifting from 2D coordinates to
a 6D geometric descriptor, consisting of: (1) UADD, a compact uncertainty-
aware depth descriptor from an off-the-shelf MDE, and (2) scale normalization
for train/test distance shifts. Because AugLift operates at the input level, it
composes with any lifting architecture or DG technique.
– C2. SOTA cross-dataset results in GT 2D setting. Combining AugLift
with PoseAug [4] achieves 62.4 mm on 3DHP and 92.6 mm on 3DPW (14.5%
and 22.2% over PoseAug), demonstrating that foundation model (FM) depth
provides genuine geometric signal complementary to explicit 3D augmenta-
tion.
– C3. Systematic detection-setting study. Across 4 datasets × 4 architec-
tures, AugLift consistently improves performance: 10.1% OOD and 4.0% ID
MPJPE reductions on average.

4
N. Warner et al.
– C4. We provide a detailed post-hoc analysis explaining when and why AugLift
works. We show that the largest gains occur on the most challenging cases:
novel poses not seen during training (e.g., a 15.7% error reduction on Fit3D)
and significantly occluded joints.
2
Literature Review
Research in monocular 3D human pose estimation has progressed along several
distinct avenues. In the following, we provide a structured review of the relevant
literature along these themes—2D-to-3D lifting, generalization strategies, and
weakly supervised learning—before discussing methods that enrich the lifting
input, which is most relevant to our work.
2D-to-3D Lifting. The paradigm of lifting 2D keypoints to 3D space gained
prominence after Martinez et al. [14] demonstrated that a simple, fully-connected
network could achieve competitive results by operating directly on 2D coordi-
nates. This approach, however, struggles with the inherent ambiguity of single-
frame inputs. To mitigate this, temporal models were introduced. For instance,
VideoPose3D [19] leverages temporal convolutions over sequences of 2D poses
to enforce motion consistency. More recently, Transformer-based architectures
like MotionBERT [27] have become state-of-the-art by jointly modeling spatial
and temporal relationships in human motion. Despite their success, these foun-
dational methods still face challenges with occlusions and depth ambiguities,
motivating the need for richer input signals.
Generalization Strategies. Improving generalization to unseen datasets and
real-world scenarios remains a primary challenge. Several strategies focus on data
augmentation and architectural enhancements. PoseAug [4], for example, uses
virtual camera augmentations to simulate multiple perspectives during training,
thereby improving robustness. Other works propose multitask learning; Wang
et al. [22] integrate 3D pose estimation with camera viewpoint prediction to
leverage complementary information. Rhodin et al. [21] use a geometry-aware
encoder-decoder framework to learn from multi-view images in a semi-supervised
fashion. While effective, these methods often introduce significant architectural
complexity or require specialized training data.
Weakly-Supervised and Unsupervised Learning. Given the high cost of
acquiring 3D annotations, methods that reduce reliance on labeled data are
crucial. Unsupervised approaches, such as the one by Chen et al. [2], enforce
geometric self-consistency through a lift-reproject-lift cycle, using a 2D pose dis-
criminator to ensure plausible skeletons without any 3D priors. Semi-supervised
techniques, explored by Pavllo et al. [19], combine limited 3D ground truth with
a larger corpus of 2D data by minimizing a reprojection loss. These methods
showcase how geometric constraints can substitute for explicit 3D labels.
Enriching Lifting with Image-Derived Cues. To overcome the limitations
of using only 2D coordinates, another key line of research has explored enrich-
ing the input with cues derived directly from the source image. One approach
involves creating hybrid models that condition the lifter on rich visual features

AugLift: Depth-Aware Input Reparameterization for 2D-to-3D Pose Lifting
5
from a CNN backbone

## method
3.1
Motivation
The design of AugLift was guided by preliminary analyses aimed at understand-
ing the failure modes of modern lifting models (full details in Appendix 7).
Motion cues can harm generalization. While longer motion sequences re-
duce in-distribution error, we found they typically degrade OOD performance
(Appendix Fig. 6). Models tested on novel motions composed of familiar poses
(e.g., reversed or sped-up actions) also showed significant drops, indicating over-
fitting to training motion dynamics rather than static pose geometry.
Per-Frame Depth Cues Offer a Robust Alternative. Motivated by this, we
investigated the potential of enriching the per-frame input instead. In contrast
to prior works that rely on generic image-derived cues (see Section 2), our anal-
ysis focused on depth cues as a particularly promising signal. To estimate their
upper bound, we conducted oracle experiments using privileged ground-truth or-
dinal depth information at varying granularities. The results were compelling:
augmenting the sparse (x, y) input with even coarse, three-bin ordinal depth in-
formation reduced cross-dataset error (H3.6M→3DPW) by approximately 25%
(see Appendix, Table 11). This suggests that providing the lifter with even basic
geometric context via depth cues is a powerful and robust path toward better
generalization.
3.2
The AugLift Method
AugLift is a lightweight pre-processing pipeline that transforms 2D keypoint in-
puts into a 6D representation incorporating confidence and local depth statistics,
without altering the core lifting architecture (Algorithm 1).
UADD. Given keypoints {(xj, yj, cj)}K
j=1 from a 2D detector and a depth map
D from an off-the-shelf MDE, we define a confidence-aware radius rj = rmin+(1−
cj)(rmax −rmin) for each joint. Within the circular neighborhood Nj of radius
rj, we compute three depth statistics—median dj, minimum dmin
j
, maximum
dmax
j
—forming the UADD: (cj, dj, dmin
j
, dmax
j
). The key intuition is that dmin
j
acts as a geometric lower bound on joint depth (the nearest visible surface),

6
N. Warner et al.
Input: Image I containing the human subject.
Step 1: Obtain 2D keypoints with confidence scores
This step is identical to standard lifting. Run a 2D keypoint detector on I to obtain
keypoints with confidence, {(xj, yj, cj)}K
j=1.
Step 2: Obtain a monocular depth map
Run an off-the-shelf monocular depth estimator on I to obtain a depth map D
defined on image pixels.
Step 3: Compute confidence-aware local depth statistics
for j = 1, . . . , K do
i. Convert confidence cj ∈[0, 1] to a sampling radius
rj = rmin + (1 −cj) (rmax −rmin).
ii. Clamp rj to [rmin, rmax].
iii. Define Nj = {(u, v) : ∥(u, v) −(xj, yj)∥2 ≤rj}.
iv. Collect depth values Dj = {D(u, v) | (u, v) ∈Nj}.
v. Compute depth statistics dj = median(Dj),
dmin
j
= min(Dj), dmax
j
= max(Dj).
end for
Step 4: Rescale 2D keypoints per normalized bounding box
i. Compute keypoint centroid: c =
  1
K
P
j xj,
1
K
P
j yj

.
ii. Compute box size (average of width and height):
b = 1
2

(maxj xj −minj xj) + (maxj yj −minj yj)

.
iii. Compute scale factor s = ¯b/b, where ¯b is the mean training-set box size.
iv. Update (xj, yj) := s ·
 (xj, yj) −c

+ c for all j.
Step 5: Normalize confidence and depth statistics
i. Rescale confidence to [−1, 1]: ˜cj = 2cj −1 for all j.
ii. Obtain root-relative depths: ˜dj = dj −droot for all j.
iii. Clip root-relative depths so ˜dj ∈[−˜dmax, ˜dmax] for all j.
Output: 6D feature vector ˜qj = (xj, yj, ˜cj, ˜dj, ˜dmin
j
, ˜dmax
j
) for keypoints j = 1, . . . , K.
These augmented features serve as inputs to the lifting model.
Algorithm 1: The AugLift module.
while confidence controls the neighborhood size: tight for visible joints, broad
for occluded ones. Full details of each statistic’s role are provided in Appendix 7.
Scale normalization. AugLift normalizes for train/test distance shifts via two
setting-specific instantiations of the same principle. In the detection setting,
bounding-box rescaling (Step 4 of Algorithm 1) normalizes 2D skeleton scale
by the ratio of box size to the mean training-set box size. In the GT+DG set-
ting, depth-based 2D rescaling (dscale2d, Section 3.3): (x′, y′) = (x, y)·droot/dref
compensates for distance-driven scale shifts. While DG methods like PoseAug
introduce implicit scale invariance through random RT augmentations, AugLift
makes this explicit—which explains the large additional gains on 3DPW where
the distance domain gap is most pronounced.
Integration. After normalizing confidence to [−1, 1] and making depth root-
relative (Appendix 7), the final per-joint input is ˜qj = (xj, yj, ˜cj, ˜dj, ˜dmin
j
, ˜dmax
j
).
Integration requires only widening the input layer from 2K to 6K channels; all
other layers, losses, and training remain unchanged.

AugLift: Depth-Aware Input Reparameterization for 2D-to-3D Pose Lifting
7
3.3
Combining AugLift with Domain Generalization
AugLift’s pointwise geometric cues are designed to be combinable with DG
and render-based augmentation. Dense learned image features encode scene-level
statistics—texture, lighting, background—which may differ between real images
and the synthetic SMPL renders used in augmentation. By contrast, AugLift’s
per-joint depth statistics are local and geometric: they depend only on the 3D
structure near each joint, not on global scene appearance, enabling them to re-
main effective even when the depth source shifts from real images to synthetic
renders (Section 5).
Integration with PoseAug. For these experiments we use the XYD input repre-
sentation (omitting the confidence channel, which is not critical in the GT 2D set-
ting). We integrate PoseAug’s differentiable augmentation framework [4], which
applies rotation-translation (RT), bone angle (BA), and bone length (BL) trans-
formations in 3D pose space before projection to 2D. The key challenge is gen-
erating geometrically consistent depth for every augmented pose—pre-computed
depth from original viewpoints cannot be reused after augmentation.
Depth extraction pipeline. Naively fitting SMPL [11] parameters to each aug-
mented skeleton via iterative optimization would make training prohibitively
slow. We develop an analytical BA/BL→SMPL mapping that converts aug-
mented skeletons into SMPL body model parameters. The resulting SMPL meshes
are rendered at 128×128 resolution via differentiable rasterization, then pro-
cessed by Depth Anything V2 [10] to extract monocular depth maps. Per-joint
depth values are sampled at 2D keypoint pixel locations via grid sampling,
forming the D channel concatenated with detected 2D keypoints (X, Y) as the
final XYD input to the lifting network. This pipeline enables live generation
of viewpoint- and anatomy-augmented depth signals during training, where the
depth channel adapts coherently to each augmented pose rather than using fixed
pre-computed depth from original viewpoints. Full details of the SMPL mapping
are provided in Appendix 7.
Depth-based 2D rescaling (dscale2d). The raw XYD representation provides
depth as an independent channel but does not account for the relationship be-
tween subject distance and 2D projection scale. Subjects closer to the camera
appear larger in 2D, creating a systematic domain gap: 3DPW subjects are on
average ∼30% closer than Human3.6M training data (mean root depth 3.6m
vs. 5.15m). We address this with depth-based 2D rescaling, which scales the
2D keypoint coordinates by the ratio of estimated root depth to a reference
depth: (x′, y′) = (x, y) · droot/dref. This normalizes 2D scale variations caused
by distance differences, allowing the lifting network to see distance-invariant 2D
patterns. The reference depth dref is a hyperparameter; we find dref = 4.5m
optimal via sweep over the range 3.5–6.0m.

8
N. Warner et al.
Fig. 3: Live depth generation pipeline. Left to right: (1) Original RGB frame,
(2) SMPL mesh from original pose, (3) SMPL mesh after analytical BA augmentation,
(4) Depth Anything V2 depth map from augmented render. Grid sampling extracts
per-joint depth to form the D channel, concatenated with 2D coordinates (XY) as
input to the lifting network.
Fig. 4: Qualitative results. Baseline (red) vs. AugLift (green) vs. ground truth
(black). The baseline fails on OOD poses with occlusion (sitting, crouching); AugLift’s
depth cues resolve front–back ambiguities.
4
Experiments in the Detection Setting
We evaluate AugLift across three complementary experimental settings. First,
we test AugLift in the detection setting across multiple datasets and architec-
tures (this section). Then, we combine AugLift with PoseAug’s differentiable
domain generalization in the GT 2D setting, demonstrating SOTA cross-dataset
performance (Section 5). Finally, we compare AugLift’s sparse geometric cues
with dense learned feature fusion, showing complementarity (Section 6).
4.1
Experimental Setup
Datasets. We conduct experiments on four diverse 3D HPE datasets: Hu-
man3.6M (H36M) [7], MPI-INF-3DHP (3DHP) [15],

## conclusion
We introduced AugLift, a change to the representation format of 2D→3D lifting
consisting of two modules: an Uncertainty-Aware Depth Descriptor (UADD) and
a scale normalization component. In the detection setting, AugLift reduces cross-
dataset error by 10.1% and in-distribution error by 4.0% across four datasets and
four lifting architectures, with gains concentrating on occluded joints and novel
poses. In the GT 2D setting, combining AugLift with PoseAug achieves state-of-
the-art cross-dataset performance (62.4 mm on 3DHP, 92.6 mm on 3DPW; 14.5%
and 22.2% over PoseAug). AugLift’s sparse geometric cues complement dense
image features but, crucially, remain stable under domain shift and synthetic
rendering—functioning as a geometry bottleneck for robust lifting. Together,
these results establish monocular depth as a scalable improvement lever for 3D
pose lifting: as MDE models improve with readily available RGB-D data, lifting
improves without additional 3D pose annotation.
Looking ahead, the depth estimator could be distilled into a lighter-weight
unified architecture for real-time deployment. Our findings on the brittleness
of motion priors highlight the need to understand how much temporal context
remains necessary when strong per-frame cues like UADD are available. Finally,
AugLift-style descriptors could be extended to multi-person scenes, other 3D
prediction tasks, and settings with weaker or self-supervised depth signals.

16
N. Warner et al.