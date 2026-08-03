# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors

> 2024 · id: arxiv:2412.12392 · arXiv: 2412.12392 · pdf: https://arxiv.org/pdf/2412.12392 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present a real-time monocular dense SLAM system de-
signed bottom-up from MASt3R, a two-view 3D reconstruc-
tion and matching prior. Equipped with this strong prior,
our system is robust on in-the-wild video sequences despite
making no assumption on a fixed or parametric camera
model beyond a unique camera centre. We introduce ef-
ficient methods for pointmap matching, camera tracking
and local fusion, graph construction and loop closure, and
second-order global optimisation. With known calibration,
a simple modification to the system achieves state-of-the-art
performance across various benchmarks. Altogether, we
propose a plug-and-play monocular SLAM system capable
of producing globally consistent poses and dense geometry
while operating at 15 FPS.

## introduction
Visual simultaneous localisation and mapping (SLAM) is
a foundational building block for today’s robotics and aug-
mented reality products. With careful design of an inte-
grated hardware and software stack, robust and accurate
visual SLAM is now possible. However, SLAM is not yet
a plug-and-play algorithm as it requires hardware expertise
and calibration. For a minimal single camera setup without
additional sensing such as an IMU, in-the-wild SLAM that
provides both accurate poses and consistent dense maps does
not exist. Achieving such a reliable dense SLAM system
would open new research avenues for spatial intelligence.
Performing dense SLAM from only 2D images requires
reasoning over time-varying poses and camera models, as
well as 3D scene geometry. To solve such an inverse problem
of large dimensionality, a variety of priors, from handcrafted
to data-driven, have been proposed. Single-view priors, such
as monocular depth and normals, attempt to predict geome-
try from a single image, but these contain ambiguities and
lack consistency across views. While multi-view priors like
optical flow reduce ambiguity, decoupling pose and geom-
*Authors contributed equally to this work.
Two-View Pointmap 
Prediction Using MASt3R
Real-Time Monocular Dense SLAM Without a 
Known Camera Model
Figure 1. Reconstruction from our dense monocular SLAM system
on the Burghers sequence [56]. Using two-view predictions from
MASt3R shown on the left, our system achieves globally consistent
poses and geometry in real-time without a known camera model.
etry is challenging since pixel motion depends on both the
extrinsics and the camera model. Although these underlying
causes may vary across time and different observers, the
3D scene remains invariant across views. Therefore, the
unifying prior required to solve for poses, camera models,
and dense geometry from images is over the space of 3D
geometry in a common coordinate frame.
Recently, two-view 3D reconstruction priors, pioneered
by DUSt3R [50] and its successor MASt3R [21], have cre-
ated a paradigm shift in structure-from-motion (SfM) by
capitalising on curated 3D datasets. These networks output
pointmaps directly from two images in a common coordi-
nate frame, such that the aforementioned subproblems are
implicitly solved in a joint framework. In the future, these
priors will be trained on all varieties of camera models with
significant distortion. While 3D priors could take in more
views, SfM and SLAM leverage spatial sparsity and avoid
redundancy to achieve large-scale consistency. A two-view
architecture mirrors two-view geometry as the building block
of SfM, and this modularity opens the door for both efficient
decision-making and robust consensus in the backend.

In this work, we propose the first real-time SLAM frame-
work to leverage two-view 3D reconstruction priors as a
unifying foundation for tracking, mapping, and relocalisa-
tion as shown in Fig. 1. While previous work has applied
these priors to SfM in an offline setting with unordered
image collections [10], SLAM receives data incrementally
and must maintain real-time operation. This requires new
perspectives on low-latency matching, careful map main-
tenance, and efficient methods for large-scale optimisation.
Furthermore, inspired by both filtering and optimisation tech-
niques in SLAM, we perform local filtering of pointmaps in
the frontend to enable large-scale global optimisation in the
backend. Our system makes no assumption on each image’s
camera model beyond having a unique camera centre that all
rays pass through. This results in a real-time dense monoc-
ular SLAM system capable of reconstructing scenes with
generic, time-varying camera models. Given calibration, we
also demonstrate state-of-the-art performance in trajectory
accuracy and dense geometry estimation.
In summary, our contributions are:
• The first real-time SLAM system using the two-view 3D
reconstruction prior MASt3R [21] as a foundation.
• Efficient techniques for pointmap matching, tracking and
local fusion, graph construction and loop closure, and
second-order global optimisation.
• A state-of-the-art dense SLAM system capable of handling
generic, time-varying camera models.

## method
We provide an overview of the method in Fig. 3, which shows
our main components: MASt3R prediction and pointmap
matching, tracking and local fusion, loop closure, and global
optimisation.

## experiments
We evaluate our system on a wide range of real-world
datasets. For localisation, we evaluate monocular SLAM on
Table 2. Absolute trajectory error (ATE (m)) on 7-Scenes [36].
chess
fire
heads office pumpkin kitchen stairs
avg
NICER-SLAM 0.033 0.069 0.042 0.108
0.200
0.039
0.108 0.086
DROID-SLAM 0.036 0.027 0.025 0.066
0.127
0.040
0.026 0.049
Ours
0.053 0.025 0.015 0.097
0.088
0.041
0.011 0.047
Ours*
0.063 0.046 0.029 0.103
0.114
0.074
0.032 0.066
TUM RGB-D [38], 7-Scenes [36], ETH3D-SLAM [34], and
EuRoC [3], all under monocular RGB setting. For geometry
evaluation, we use the EuRoC Vicon room sequences as it
provides 3D structure scan ground truth, as well as 7-Scenes
since it has depth camera measurements.
We run our system on a desktop with Intel Core i9 12900K
3.50GHz and a single NVIDIA GeForce RTX 4090. As our
system runs at roughly 15 FPS, we subsample every 2 frames
of the datasets to simulate real-time performance. Note that
we use the full resolution outputs from MASt3R, which
resizes the largest dimension to size 512.
4.1. Camera Pose Estimation
For all datasets, we report the RMSE of the absolute trajec-
tory error (ATE) in metres. Since all systems are monocular,
we perform scaled trajectory alignment. We denote our sys-
tem without known calibration as Ours*.
TUM RGB-D: On the TUM dataset, we demonstrate
state-of-the-art trajectory error when using calibration as
shown in Tab. 1. Many of the previously best performing
algorithms, such as DROID-SLAM, DPV-SLAM, and GO-
SLAM, build on the foundational matching and end-to-end
system proposed by DROID-SLAM. In contrast, we propose
a unique system that takes an off-the-shelf two-view geomet-
ric prior and show that it can outperform other systems while
operating in real-time. Furthermore, our uncalibrated sys-
tem significantly outperforms a baseline, which we denote
DROID-SLAM*, that calibrates the intrinsics using Geo-
Calib [48] on the first image of a sequence, which is then
used by DROID-SLAM. We achieve this without assuming
a fixed camera model across the entire sequence, and demon-
strate the value of 3D priors for dense uncalibrated SLAM
over priors that solve subproblems. Our uncalibrated SLAM
results are also comparable to results from recent learned
techniques such as DPV-SLAM with known calibration.




	






	




ATE
AUC
ORB-SLAM3
0.135
16.661
DROID-SLAM
0.171
22.297
DPVO
0.137
22.628
DPV-SLAM
0.109
23.097
DPV-SLAM++
0.132
21.784
Ours
0.086
23.935
Figure 5. Number of successful trajectories below ATE threshold
on ETH3D-SLAM (train) benchmark. The corresponding table
shows the mean ATE across completed sequences, as well as the
AUC up to the threshold.
7-Scenes: We use the same sequences for evaluation fol-
lowing NICER-SLAM as shown in Tab. 2. Our calibrated
system outperforms both NICER-SLAM [58] and DROID-
SLAM. Furthermore, our real-time uncalibrated system us-
ing a single 3D reconstruction prior outperforms NICER-
SLAM, which uses multiple priors in depth, normal, and
optical flow networks and runs offline.
ETH3D-SLAM: Due to its difficulty, ETH3D-SLAM
has only been evaluated for RGB-D methods. Since the ATE
thresholds for the official private evaluation are too strict
for monocular methods, we evaluate several state-of-the-art
monocular systems on the train sequences and generate the
ATE curves. The dataset contains sequences with fast cam-
era motion, hence, for all methods, we do not subsample the
frames. While other methods can have more precise trajec-
tories, our method has a longer tail in terms of robustness,
resulting in both the best ATE and area-under-curve (AUC).
EuRoC: We report the average ATE across all 11 EuRoC
sequences in Tab. 3. For the uncalibrated case, we found
that the distortion was too significant as MASt3R was not
yet trained on such camera models, so we undistorted the
images but did not give calibration to the rest of the pipeline.
In general, our system is outperformed by DROID-SLAM,
but it explicitly augments its training with 10% greyscale
images. However, 0.041m ATE is still very accurate, and
from the comparisons in [22], all outperforming methods
build on top of the foundation from DROID-SLAM, while
we present a novel method using a 3D reconstruction prior.
4.2. Dense Geometry Evaluation
We evaluate our geometry against DROID-SLAM and
Spann3R [49] on the EuRoC Vicon room sequences and
Table 3. Reconstruction Evaluation on 7-Scenes and EuRoC with
all metrics in metres.
7-scenes
ATE Accuracy Completion Chamfer
DROID-SLAM 0.049
0.115
0.040
0.077
Spann3R @20
N/A
0.069
0.047
0.058
Spann3R @2
N/A
0.124
0.043
0.084
Ours
0.047
0.074
0.057
0.066
Ours*
0.066
0.068
0.045
0.056
EuRoC
ATE Accuracy Completion Chamfer
DROID-SLAM 0.022
0.173
0.061
0.117
Ours
0.041
0.099
0.071
0.085
Ours*
0.164
0.108
0.072
0.090
Figure 6. Reconstruction on EuRoC Machine Hall 04.
7-Scenes seq-01. For EuRoC, the alignment between the ref-
erence and the estimated point cloud is obtained by aligning
the estimated trajectory against the Vicon trajectory. Note,
that this setup favours DROID-SLAM which obtains lower
trajectory error. For 7-Scenes, we backproject the depth
images using poses provided by the dataset to create the
reference point cloud. It is then aligned to the estimated
point cloud using ICP as the extrinsic calibration between
RGB and depth sensor is not provided.
We report the RMSE for accuracy, which is defined as
the distance between each estimated point and its nearest
reference point, and completion, the distance between each
reference point and its nearest estimated point. Both metrics
are calculated with a maximum distance threshold of 0.5m
and averaged across all sequences. We also report Chamfer
Distance, the average of the two metrics.
Tab. 3 summarises the geometry evaluation on 7-Scenes
and EuRoC. For 7-Scenes, both our method with and without
calibration and Spann3R achieve more accurate reconstruc-
tion compared to DROID-SLAM, highlighting the advantage
of the 3D prior. We run Spann3R under two different set-
tings. In one, a keyframe is taken every 20 images and in the
other every 2 images. The discrepancy in the two settings
shows the challenges test-time optimisation-free approaches
face to generalise. Ours without calibration performs the
best in both Accuracy and Chamfer distance. This can be
attributed to the fact that the intrinsic calibration 7-Scenes
provides is the default factory calibration.
For EuRoC, Spann3R struggles as the sequences are not
object-centric and thus is excluded. As summarised in Tab. 3,
although DROID-SLAM outperforms our method in terms
of ATE, our method with/without calibration obtains better

Consecutive keyframes 
(1 second difference)
Figure 7. Dense uncalibrated SLAM with extreme zoom changes
shown by two consecutive keyframes for an outdoor scene.
geometry. DROID-SLAM obtains higher completion as it
estimates a large number of noisy points which surround
the reference point cloud, but our method has significantly
better accuracy. It is interesting to note that our uncalibrated
system has a noticeably larger ATE, but still outperforms
DROID-SLAM in Chamfer distance.
4.3. Qualitative Results
Fig. 1 shows a reconstruction of the challenging Burghers
sequence which has few matchable features on the spec-
ular figures. We show examples of pose estimation and
dense reconstructions for TUM in Fig. 4 and for EuRoC
in Fig. 6. Furthermore, we show an example with extreme
zoom changes between consecutive keyframes in Fig. 7.
4.4. Component Analysis
We compare matching techniques in Tab. 4. Our parallelised
projective matching with feature refinement achieves the
best accuracy with significantly faster runtime. Performing
MASt3R matching over all pixels takes 2 seconds, while
our matching takes 2ms and makes the entire system FPS
nearly 40x faster. Please refer to the supplementary for a full
runtime analysis of the system. In Tab. 5, we test different
methods for updating the canonical pointmap and report the
average ATE across TUM, 7-Scenes, and EuRoC. Selecting
the most recent and first pointmaps incur drift and lack suf-
ficient baseline, respectively. Given calibration, weighted
fusion performs on par with selecting the pointmap with
the highest median confidence, but it achieves the lowest
ATE without calibration and improves the ATE on EuRoC
by 1.3cm, indicating that fusing over camera models is im-
portant. In Tab. 6, the ray error formulation for uncalibrated
tracking and backend optimisation improves performance
over using the 3D point error which contains inaccurate
depth predictions. Tab. 7 shows that loop closure improves
both pose and geometry accuracy, with more significant
gains on longer sequences. This demonstrates that the out-
puts of MASt3R still contain bias and cause drift, which our
components are designed to mitigate.
Table 4. Matching comparison.
ATE [m] ATE [m] Matching System
w/ calib w/o calib Time [ms]
FPS
k-d tree
0.0

## related_work
DUSt3R takes in a pair of images Ii, Ij ∈RH×W ×3, and
outputs pointmaps Xi
i, Xj
i ∈RH×W ×3 along with their
confidences Ci
i, Cj
i ∈RH×W ×1. Here, we use notation
Xi
j to express the pointmap of image i represented in the
coordinate frame of camera j. In MASt3R, an additional
head is added to predict d-dimensional features for match-
ing Di
i, Dj
i ∈RH×W ×d and its corresponding confidences
Qi
i, Qj
i ∈RH×W ×1. We define FM(Ii, Ij) as the forward
pass of MASt3R that yields the previously discussed out-
puts, and throughout the text we will use MASt3R’s output
directly for conciseness.
While some of the data used to train MASt3R has metric
scale, we found that scale is often a large source of inconsis-
tency across predictions. To optimise over differently scaled
predictions, we define all poses as T ∈Sim(3) and updates
to the poses using Lie algebra τ ∈sim(3) and a left-plus
operator:
T =
 sR
t
0
1

,
T ←τ ⊕T ≜Exp(τ) ◦T,
(1)
where R ∈SO(3), t ∈R3, and scale s ∈R, following the
notation in [37, 44].
Our only assumption on the camera model is that of a
generic central camera [35], which means that all rays pass
through a unique camera centre. We define the function
ψ
 Xi
i

that normalises a pointmap Xi
i into rays of unit norm
such that each pointmap defines its own camera model. This
Figure 2. Overview of iterative projective matching: given the
two pointmap predictions from MASt3R, the reference pointmap is
normalised ψ
 Xi
i

to give a smooth pixel to ray mapping. For an
initial estimate of the projection p0 of 3D point x from pointmap
Xj
i, the pixel is iteratively updated to minimise the angular differ-
ence θ between the queried ray ψ
 [Xi
i]p

and the target ray ψ (x).
After finding the pixel p∗that achieves the minimum error, we
have a pixel correspondence between Ii and Ij.
enables handling both time-varying camera models, such as
zoom, and distortion in a unified manner.
3.2. Pointmap Matching
Correspondence is a fundamental component of SLAM that
is required for both tracking and mapping. In this case, given
the pointmaps and features from MASt3R, we need to find
the set of pixel matches between the two images, denoted by
mi,j = M(Xi
i, Xj
i, Di
i, Dj
i). Naive brute-force matching
has quadratic complexity since it is a global search over all
possible pairs of pixels. To avoid this, DUSt3R uses a k-d
tree over 3D points; however, construction is non-trivial to
parallelise and the nearest-neighbour search in 3D will find
many inaccurate matches if there are errors in the pointmap
predictions. In MASt3R, additional high-dimensional fea-
tures are predicted from the network to achieve wider base-
line matching and a coarse-to-fine scheme is proposed to
handle the global search. However, the runtime is on the
order of seconds for dense pixel matching, and sparse match-
ing is still slower than the k-d tree. Rather than focusing
on efficient methods for a global search over matches, we
instead find inspiration from optimisation as a local search.
Compared to feature matching, we are motivated by the
use of projective data-association commonly used in dense
SLAM. However, this requires a parametric camera model
with closed-form projection, while our only assumption is
that each frame has a unique camera centre. Given the out-
put pointmaps Xi
i, Xj
i, we can construct the generic camera
model of Ii with the rays ψ
 Xi
i

. Inspired by generic cam-
era calibration methods [32, 35] which lack closed-form
projection, we project each point x ∈Xj
i independently by
iteratively optimising the pixel coordinates p in the reference
frame that minimise the ray error:
p∗= arg min
p
ψ
 [Xi
i]p

−ψ (x)
2 .
(2)

MASt3R (Sec 3.1)
Pointmap Matching (Sec 3.2) 
Tracking and Pointmap Fusion (Sec 3.3)
Loop Closure (Sec 3.4) and Global Opt. (Sec 3.5)
New Image
Current Keyframe
Loop Closure Candidates
Figure 3. System diagram of MASt3R-SLAM. New images are tracked against the current keyframe by predicting a pointmap from MASt3R
and finding pixel matches using our efficient iterative projection pointmap matching. Tracking estimates the current pose and performs local
pointmap fusion. When new keyframes are added to the backend, loop closure candidates are selected by querying the retrieval database
using encoded MASt3R features. Candidates are then decoded by MASt3R and if a sufficient number of matches is found, edges are added
to the backend graph. Large-scale second-order optimisation achieves global consistency of poses and dense geometry.
We show a visual overview in Fig. 2, and note that minimis-
ing the Euclidean distance between normalised vectors is
equivalent to minimising the angle θ between two normalised
rays:
∥ψ1 −ψ2∥2 = 2(1 −cos θ),
cos θ = ψT
1 ψ2.
(3)
By using the nonlinear least-squares form similar to [35],
we can iteratively solve for updates to projected locations by
calculating analytical Jacobians and solving via Levenberg-
Marquardt. This can be done separately for each point and
converges for almost all valid pixels within 10 iterations as
the ray image is smooth. At the end of this process, we now
have initial matches mi,j. When there is no initial estimate
for the projection p, such as when tracking against a new
keyframe or when matching loop closure edges, all pixels are
initialised with the identity mapping. During tracking, since
we always have the matches from the previous frame, we can
use this as initialisation to further speed up the convergence.
To handle occlusions and outliers, we also invalidate matches
that have large distances in 3D space. Our matching is
massively parallel on GPU and additionally can leverage the
incremental nature of SLAM.
While these pixels give a good initial estimate of matches
using the geometry, MASt3R demonstrates that leveraging
per-pixel features greatly improves downstream performance
on pose estimation. Since we have a good initialisation from
the previous step, we conduct a coarse-to-fine image-based
search by updating the pixel location to the maximum feature
similarity in a local patch window.
We implement both the iterative projection and feature
refinement steps in custom CUDA kernels, as both are par-
allelisable for each pixel. For tracking this takes only 2
milliseconds and for constructing edges in the graph this
takes only a few milliseconds for all newly added edges
without any initial estimates of the projections. Note that
our matches are unbiased by our pose estimates as they rely
purely on the MASt3R outputs, which is atypical for projec-
tive data association.
3.3. Tracking and Pointmap Fusion
A key component of SLAM is low-latency tracking of the
current frame’s pose against the map. As a keyframe-based
system, we estimate the relative transformation Tkf between
the current frame If and the last keyframe Ik. To be effi-
cient, we would like to use only a single pass of the network
to estimate the transformation. Assuming we already have
the last keyframe’s pointmap estimate ˜Xk
k, we need points
in the frame of If to resolve Tkf. This can be obtained via
FM(If, Ik). One straightforward method to solve for pose
is minimising the 3D point error:
Ep =
X
m,n∈mf,k

˜Xk
k,n −TkfXf
f,m
w(qm,n, σ2p)

ρ
,
(4)
where qm,n =
p
Qff,mQkf,n is the match confidence weight-
ing proposed in MASt3R-SfM [10]. For robustness, in ad-
dition to the Huber norm ∥· ∥ρ, a per-match weighting is
applied:
w(q, σ2) =
(
σ2/q
q > qmin
∞
otherwise .
(5)
While Xk
f instead of Xf
f could also be aligned to Xk
k with
the benefit of no explicit matching required as they are pixel
aligned, we found that explicit matching with Xf
f had im-
proved accuracy for larger baseline scenarios. More impor-
tantly, although the 3D point error is suitable, it is easily
skewed by errors in the pointmap predictions as inconsis-
tent predictions in depth are relatively frequent. Since we

ultimately fuse predictions into a single pointmap that aver-
ages out all the predictions, error in tracking degrades the
keyframe’s pointmap that will also be used in the backend.
By again exploiting that the pointmap predictions can be
converted to rays under a central camera assumption, we
can calculate a directional ray error instead, which is less
sensitive to incorrect depth predictions. To calculate this, we
simply normalise both points from Eq. (4):
Er =
X
m,n∈mf,k

ψ

˜Xk
k,n

−ψ

TkfXf
f,m

w(qm,n, σ2r)

ρ
.
(6)
This results in a similar angular error as mentioned in Eq. (3)
and shown in Fig. 2, except that we now have many known
correspondences and wish to find the pose that minimises
all angular errors between canonical rays and corresponding
predicted rays from the current frame. Since angular errors
are bounded, ray-based errors are robust against outliers [30].
We also include an error term with a small weight on the
difference in distances from the camera centre. This prevents
the system from becoming degenerate under pure rotation,
while avoiding significant bias from error

## conclusion
We present a real-time dense SLAM system based on
MASt3R that handles in-the-wild videos and achieves state-
of-the-art performance. Much of the recent progress in
SLAM has followed the contributions of DROID-SLAM,
which trains an end-to-end framework that solves for poses
and geometry from a flow update. We take a different ap-
proach by building a system around an off-the-shelf geomet-
ric prior that achieves comparable pose estimation for the
first time, while also providing consistent dense geometry.
7. Acknowledgement
This research is supported by the Engineering and Physical
Sciences Research Council [grant number EP/W524323/1].