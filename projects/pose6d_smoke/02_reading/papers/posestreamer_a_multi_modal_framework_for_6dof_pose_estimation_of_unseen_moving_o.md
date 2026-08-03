# PoseStreamer: A Multi-modal Framework for 6DoF Pose Estimation of Unseen Moving Objects

> 2025 · id: W7117850891 · arXiv: 2512.22979 · pdf: https://arxiv.org/pdf/2512.22979 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
6DoF pose estimation [1] aims to compute the rigid six-
dimensional transformation (3D translation and rotation) be-
tween an object and a camera. Traditional 6DoF pose esti-
mation methods [2]–[4] cannot be directly applied to unseen
objects during the training phase. In practical scenarios, dif-
ferent applications often provide diverse input objects. More
recent efforts have focused on real-time pose estimation for
arbitrary unseen objects. However, existing RGB camera-based
methods [5] are prone to motion blur, which particularly limits
their performance in high-speed low-lit scenarios.
Some studies introduce event cameras [6] which capture
motion information via luminance variations. It outputs a
sparse event stream with high temporal resolution. Such a
stream can act as a complementary modality to the motion
geometry information for RGB cameras. Currently, relevant
research on visual object tracking using event cameras remains
relatively limited. EventVOT [7] distills knowledge from an
RGB-Event dual-modal teacher network into a pure event-
driven student network. AFNet [8] fuses frame-based RGB
data and event-based temporal data through a multi-modal
†Corresponding author.
alignment and fusion module. STNet [9] integrates Trans-
former with spiking neural networks to build an event-driven
tracking framework. However, all the aforementioned methods
only validate 2D tracking performance and do not explore
object tracking in the 3D space. Our research aims to leverage
the RGB-Event fused modality to achieve accurate 6DoF pose
estimation for arbitrary moving objects. On the one hand,
event cameras only capture edge information from luminance
variations. They lack critical contextual details, such as texture,
color, and global semantics, which are essential for object
orientation estimation. Moreover, a significant distribution gap
exists between event data and RGB modality data [10], [11],
posing a barrier to direct fusion for information complemen-
tarity. On the other hand, event cameras output relatively
sparse spatial signals. This sparsity can lead to tracking failure
when target object contours are insufficiently distinct. [12]
In addition, the temporal resolution mismatch between event
and RGB cameras further amplifies the uncertainty of spatial
perception.
To address the aforementioned challenges, this paper pro-
poses PoseStreamer, a multi-modal 6DoF pose estimation
framework tailored for high-speed moving scenarios. Its
core architecture comprises three key components: (1) The
Adaptive Pose Memory Queue (AMQ), a sliding-window
module that leverages historical orientation cues to guide
current-frame orientation inference. (2) The Multi-modality
3D tracker (M3D), which supplies robust 2D priors to enhance
the recall of 3D object centers. (3) The Ray Pose Filter (RPF),
designed to refine pose estimates along camera ray directions.
Furthermore, to evaluate performance under high-speed mo-
tion, we introduce MoCapCube6D, a novel multi-modal pose
estimation dataset that facilitates comprehensive benchmark-
ing of our method against state-of-the-art approaches. Notably,
the proposed method exhibits strong generalizability and can
be extended to a template-free framework for unseen object
instances. Our contributions are summarized as follows:
• We propose PoseStreamer, a novel multi-modal 6DoF
pose estimation framework. By integrating the Adaptive
Pose Memory Queue, Multi-modality 3D tracker, and Ray
Pose Filter, our method effectively leverages historical
cues and 2D priors to achieve robust tracking on high-
speed moving scenarios.
arXiv:2512.22979v3  [cs.CV]  2 Jan 2026

Fig. 1: Overview of the PoseStreamer Architecture. The framework proceeds in three stages: (A) Unseen object initialization
via RGB-based reconstruction and the Adaptive Pose Memory Queue (AMQ). (B) High-speed 3D center estimation via the
Multi-modality 3D Tracker (M3D) on RGB images and Event streams. (C) Fine-grained 6DoF optimization via the Ray Pose
Filter (RPF). The filter samples and selects pose hypotheses along the camera ray.
• We demonstrate the strong generalizability of our ap-
proach by extending it to a template-free framework
capable of handling arbitrary unseen object instances
without retraining, addressing the limitations of tradi-
tional methods on unseen objects.
• We construct MoCapCube6D, a multi-modal benchmark
dataset for 6DoF pose estimation performance on high-
speed moving scenarios.

## method
The overall architecture of PoseStreamer is illustrated in
Fig. 1. By leveraging the complementary strengths of RGB
and event cameras, our framework achieves robust 6DoF pose
estimation for unseen moving objects. It consists of three core
components: First, we introduce the Adaptive Pose Memory
Queue (AMQ) in Section III-A, which utilizes historical cues
to maintain temporal orientation consistency. Next, to ad-
dress high-speed motion, we propose the Multi-modality 3D
tracker (M3D) in Section III-B. This module employs a stereo
configuration to provide reliable 3D center priors. Finally, the
Ray Pose Filter (RPF), detailed in Section III-C, refines the
estimation by sampling hypotheses along the camera ray and
selecting the optimal pose via a render-and-compare strategy.
A. RGB-based Instance Pose Initialization
The primary challenge in tracking moving objects is main-
taining orientation consistency across consecutive frames. To
address this, we introduce the Adaptive Pose Memory Queue

(AMQ), a module designed to stabilize pose estimation by
leveraging historical temporal cues. Initially, to handle unseen
objects, we capture surround-view images with a standard
RGB camera and reconstruct a reference CAD model using
BundleSDF [17]. An initial pose estimator is then employed
to populate a First-In-First-Out (FIFO) queue M of length
N. During the tracking phase, AMQ ensures smoothness
through an adaptive update strategy. Specifically, at each
iteration, historical poses stored in the queue are projected
into the Euler-angle space via the mapping E−1. We then
apply a confidence scaler α to assign decaying weights to
these poses, effectively balancing the contribution of historical
trends against current observations. The fused orientation is
subsequently mapped back to the rotation matrix domain via E.
By propagating temporal information through low-dimensional
pose parameters rather than high-dimensional feature maps F,
AMQ significantly improves orientation stability with negligi-
ble computational overhead. The detailed procedure is outlined
in Algorithm 1.
Algorithm 1: Adaptive Pose-Queue Update Strategy
Input: queue M; rotation R; 3D centers C;
Output: pivot rotation ˆR;
1 E: Euler-to-rotation mapping;
2 E−1: rotation-to-Euler mapping;
3 α: decay weight;
4 H: center-pose hypothesis;
5 // initialization at the first frame.
6 if i == 0 then
7
ˆR′ ←H(C);
8 end
9 N ←min(i, N);
10 // adaptive scaling orientation
11 for n = 0 to N −1 do
12
R′
n ←E−1(Rn);
13
a ←an+1;
14
ˆR′ ←α ˆR′ + (1 −α) R′
n;
15 end
16 // update pose
17 ˆR ←E( ˆR′);
18 return ˆR
B. Multi-modal 3D Center Tracking
Conventional pose estimators relying solely on RGB frames
often falter in high-speed scenarios due to severe motion blur
and large inter-frame displacements. To overcome this limi-
tation, we leverage complementary modalities: RGB images,
which captures high-frequency motion changes, and Event
stream, which highlights structural edges. Integrating these
inputs, we propose M3D, a lightweight and modality-agnostic
tracker. As illustrated in Fig. 2, the pipeline extracts features
from these stereo streams (RGB and event steam), clusters
them based on motion consistency, and robustly computes the
3D center C via triangulation.
The process begins by extracting synchronized features FL
and FR from left and right cameras. We initialize a set of
feature points P = {pt
i} on the image plane and employ Pyra-
midal Lucas–Kanade optical flow [18] to handle rapid motion.
This coarse-to-fine approach computes the displacement vector
∆pi = pt+1
i
−pt
i for each point across consecutive frames.
RGB and Event Stream
Fig. 2: Details of Multi-modality 3D tracker. Features FL and
FR are extracted from the left and right cameras, respectively,
with modalities including RGB images and event stream.
Feature points are clustered into groups according to motion
consistency, after which the 2D centroids CL
2D and CR
2D are
computed from the left and right views. Finally, the 3D object
center C is obtained via disparity-based stereo triangulation.
To distinguish the moving object from the background,
we calculate the consistency Cij between points i and j by
combining motion displacement and spatial proximity:
Cij =
(
1,
 z([∆pi, λˆpi]) −z([∆pj, λˆpj])

2 ≤τ,
0,
otherwise,
(1)
where z(·) denotes z-score normalization, and
ˆp
=
pt/(W, H) represents the normalized position. The factor λ
(set to 0.3) balances the spatial term to ensure clusters are
compact.
Finally, we identify the dominant cluster Sk as the target
object. We compute its 2D centroids CL
2D = (uL, vL) and
CR
2D = (uR, vR) from the left and right views, respectively.
Defining the stereo disparity as d = uL −uR, the 3D object
center C is derived via stereo triangulation:
C =


X
Y
Z

= b
d ·


uL −cL
x
vL −cL
y
f L
x

,
(2)
where b is the stereo baseline, and (cL
x, cL
y , f L
x ) are the intrinsic
parameters of the left camera.
C. Ray Pose Filter
To refine the coarse pose estimates provided by M3D and
AMQ, we introduce the Ray Pose Filter (RPF). The module

operates in three stages: constructing a camera ray from the
estimated center, generating pose hypotheses along this ray,
and selecting the optimal candidate via an attention-based
refinement decoder.
1) Camera Ray Construction: Given the initial 3D object
center C = [x, y, z]T and rotation ˆR from previous stages,
we first define the object-centric ray. This ray originates from
the camera’s optical center and passes through the object’s
projected center on the image plane. The 2D projection (u, v)
and the depth d of the object center are obtained via the
perspective projection:
[u · d, v · d, d]T = KC,
(3)
where K is the camera intrinsic matrix. The ray is thus defined
as the vector direction passing through pixel (u, v) in the
camera coordinate system. This geometric constraint allows
us to reduce the search space from a 3D volume to a 1D
manifold (the ray), significantly enhancing efficiency.
2) Ray Pose Generation: To handle depth uncertainty,
we generate a set of candidate poses along the established
ray. We introduce a depth perturbation mechanism to sample
hypotheses around the initial depth estimate d. The perturbed
depth ˆd is defined as:
ˆd = d + β · U(−1, 1),
(4)
where U(−1, 1) denotes a uniform distribution, and β is a
scale parameter set to the mean object diameter. Using these
sampled depths, we back-project the 2D center (u, v) to obtain
a set of 3D candidate centers { ˆCj}. For the j-th candidate,
the 3D position is reconstructed as:
ˆCj = ˆdj · K−1[u, v, 1]T .
(5)
Combining these positions with the pivotal rotation ˆR, we
form a batch of ray poses. A CUDA-accelerated renderer then
generates synthetic feature maps Fq for each candidate pose,
which serve as queries for the subsequent refinement.
3) Pose Refinement: This module functions as both a pose
selector and a refiner. In the attention mechanism [19], the
synthetic features Fq (derived from ray poses) serve as the
query q, while the observed image features F serve as both
the key k and value v. Formally, we first project the features
and then aggregate scale- and perspective-aware context via
Multi-Head Attention (MHA):
q = FFN(Fq + Norm(FFN(Fq))),
ˆq = MHA(PE(q), PE(FFN(F)), FFN(F)),
(6)
where FFN, Norm, and PE denote the Feed-Forward Net-
work, Layer Normalization, and Positional Encoding function,
respectively. Note that we substitute k and v with the obser-
vation feature F.
Next, we evaluate the quality of each hypothesis to identify
the optimal candidate. We compute a salience score vector
S ∈RN×1 (where N = 64) for the updated queries ˆq and
select the best match:
S = Softmax(FFN(ˆq)),
ˆq1 = Topk(ˆq, S, 1),
ˆC1 = Topk( ˆC, S, 1),
ˆR1 = Topk( ˆR, S, 1),
(7)
where Topk selects the candidate with the highest score.
Finally, two separate FFN heads utilize the selected feature
ˆq1 to predict the residual translation ∆C and rotation ∆R.
The final object center and orientation are updated as follows:
∆C = FFN(ˆq1),
˜C = ˆC1 + ∆C,
∆R = FFN(ˆq1),
˜R = ∆R · ˆR1,
(8)
where ˜C and ˜R are the final outputs enqueued into the AMQ
for the next iteration. Thanks to this data-driven design [20],
the refinement decoder is universally applicable to standard
RGB and event stream inputs.

## experiments
A. Dataset
1) MoCapCube6D: We present MoCapCube6D, a high-
precision benchmark for high-speed tracking featuring a Ru-
bik’s Cube with synchronized RGB and event streams. Ground
truth is obtained via a calibrated MoCap system, ensuring
sub-millimeter translation and sub-degree rotation accuracy.
The dataset includes three scenes with distinct motion patterns
(details in Tab. I) and is further partitioned by projected pixel
velocity v into Regular (v < 45 px/s), Medium (45 ≤v <
180), and Faster (v ≥180) to evaluate robustness under
diverse dynamics.
TABLE I: Scene definitions used in the benchmark.
Scene
Description
a
The cube is suspended and rotated rapidly around different axes and at varying
angular velocities.
b
The cube undergoes periodic oscillations resembling pendulum motion, en-
abling evaluation of phase and frequency stability.
c
Appearance degradation is introduced through perturbations such as random
noise, artificial occlusion, and pixel replacement.
2) YCB Object Set: We also adopt the standard YCB
object set [21]. Due to the lack of high-speed multi-modal
benchmarks, we collected and labeled real-world sequences of
YCB objects using the aforementioned MoCap system. This
enables comprehensive qualitative and quantitative validation
of our template-free framework on unseen objects.
B. Evaluation Metrics
To evaluate performance, we employ the standard ADD
metric for general objects and ADD-S for symmetric ones like
the cube. We report the average recall rate where the mean dis-
tance is within 10% of the object’s diameter (ADD(S)@0.1d).
Additionally, we provide the mean translation error ep (in cm)
and rotation error er (in degrees) to offer a more granular
analysis.

TABLE II: Pose estimation performance under different speed conditions, i.e., Regular, Medium, and Faster, defined by pixel-
per-second velocity v. Specifically, Regular: v < 45, Medium: 45 ≤v < 180, and Faster: v ≥180. In addition, ep(σ) denotes
the translation error (cm), and er(σ) denotes the rotation error (deg), reported for further evaluation.
Scenes

## related_work
A. 6DoF Pose Estimation and Tracking
6DoF pose estimation aims to infer the 3D translation and
rotation of a target object. Traditional methods often require
instance- or category-level CAD models for offline training
or template matching [2], which restricts their application to
unseen objects. Although recent generalizable works [13] relax
these assumptions, they typically rely on pre-captured refer-
ence views of the test object. In the tracking domain, methods
like BundleTrack [14] attempt to generalize to unseen objects
instantly without prior templates. However, these RGB-based
approaches rely heavily on clear textures and accurate feature
matching. Consequently, they suffer significant performance
degradation in high-speed scenarios where severe motion blur
[15] and artifacts sever the required 2D-3D correspondences.
B. Event Camera-based Detection and Tracking
Event cameras excel in high-speed scenarios due to their
microsecond-level latency. Early works like EKLT [16] fused
frames and events to track visual features asynchronously.
Recent data-driven methods focus on enhancing modality
interaction; for instance, Wang et al. [10] utilize cross-modality
Transformers to fuse RGB and event streams, while Tang et
al. [11] employ a unified backbone for simultaneous feature
extraction and correlation. However, these methods are primar-
ily confined to 2D tracking. In contrast, PoseStreamer extends
beyond 2D fusion, utilizing event data to provide robust
geometric priors that drive explicit 6DoF pose estimation in
3D space.

## conclusion
In this paper, we present PoseStreamer, a robust multi-
modal framework designed for 6-DoF pose estimation of
unseen moving objects in high-speed scenarios. To overcome
the limitations of standard RGB cameras under rapid motion,
we introduce three core components: the Multi-modality 3D
Tracker (M3D), which leverages stereo-based multi-modal
features to provide reliable 3D center priors. The Adaptive
Pose Memory Queue (AMQ), which ensures temporal orienta-
tion consistency by utilizing historical cues. And the Ray Pose
Filter (RPF), which effectively mitigates depth uncertainty
through geometric refinement along camera rays. Furthermore,
we construct MoCapCube6D, a high-precision multi-modal
benchmark dataset containing synchronized RGB and event
streams, to evaluate performance across varying speed profiles.
Extensive experiments demonstrate that PoseStreamer sig-
nificantly outperforms state-of-the-art methods in high-speed
settings and exhibits strong generalizability as a template-free
framework for unseen objects.