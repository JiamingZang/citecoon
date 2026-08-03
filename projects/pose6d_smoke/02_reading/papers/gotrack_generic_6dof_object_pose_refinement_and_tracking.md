# GoTrack: Generic 6DoF Object Pose Refinement and Tracking

> 2025 · id: arxiv:2506.07155 · arXiv: arxiv:2506.07155 · pdf: https://arxiv.org/pdf/2506.07155 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

arXiv:2506.07155v1  [cs.CV]  8 Jun 2025
GoTrack: Generic 6DoF Object Pose Refinement and Tracking
Van Nguyen Nguyen1∗
Christian Forster2
Sindi Shkodrani2
Vincent Lepetit1
Bugra Tekin2
Cem Keskin2
Tomas Hodan2
1LIGM, ´Ecole des Ponts
2Meta Reality Labs
Abstract
We introduce GoTrack, an efficient and accurate CAD-
based method for 6DoF object pose refinement and track-
ing, which can handle diverse objects without any object-
specific training. Unlike existing tracking methods that rely
solely on an analysis-by-synthesis approach for model-to-
frame registration, GoTrack additionally integrates frame-
to-frame registration, which saves compute and stabilizes
tracking. Both types of registration are realized by opti-
cal flow estimation. The model-to-frame registration is no-
ticeably simpler than in existing methods, relying only on
standard neural network blocks (a transformer is trained
on top of DINOv2) and producing reliable pose confi-
dence scores without a scoring network. For the frame-
to-frame registration, which is an easier problem as con-
secutive video frames are typically nearly identical, we em-
ploy a light off-the-shelf optical flow model. We demon-
strate that GoTrack can be seamlessly combined with ex-
isting coarse pose estimation methods to create a minimal
pipeline that reaches state-of-the-art RGB-only results on
standard benchmarks for 6DoF object pose estimation and
tracking. Our source code and trained models are publicly
available at https://github.com/facebookresearch/gotrack.
1. Introduction
Vision-based 6DoF object pose estimation, which is an
important task for robotics and AR/VR, has been signifi-
cantly improved over the past decade [21,32,34,35,37,40,
44,48,49,51,54,66,71,81]. Yet most methods are designed
for single-image input, whereas real-world applications of-
ten demand 6DoF object pose tracking, which involves es-
timating the object pose across consecutive frames.
Classical 6DoF object tracking methods rely on key-
points [52,56,57,61,69], edges [6,13,19,58,62], or direct
optimization [4, 7, 41, 59, 67]. More recent learning-based
tracking methods [8,17,42,72,75] demonstrate remarkable
*Work done during Nguyen’s internship at Meta.
time
Figure 1. Given a CAD model of the target object, our method can
reliably track the 6DoF object pose in challenging conditions such
as dynamic scenes with hand-object interactions. The method is
applicable to diverse objects and image types, without requiring
any domain-specific training. Each row of this figure shows con-
tours of the tracked model in videos from the HOT3D dataset [3]
captured by Quest 3 [43] and Project Aria [15] headsets.
performance but depend on RGB-D input images [17,75] or
require extensive training datasets specific to each target ob-
ject [8, 42, 72], which limits their applicability. A problem
that is closely related to pose tracking is pose refinement.
In both problems, the input is an image, camera intrinsics,
and an initial object pose, and the output is a refined ob-
ject pose. Recent methods [35,44,77] for 6DoF object pose
refinement rely on an analysis-by-synthesis approach, typ-
ically rendering a 3D mesh model in the initial pose and
predicting a relative pose w.r.t. the input frame. The pose
is predicted either by direct regression [35, 77] or via 2D-
1

3D correspondences established between the input image
and an RGB-D rendering of the object model [44]. In both
cases, the methods need to solve the model-to-frame regis-
tration, which is not a trivial problem due to the synthetic-
to-real domain gap and requires a relatively heavy network.
In simultaneous localization and mapping (SLAM) [14],
the pose between a 3D scene model and a camera is contin-
uously estimated from video frames via two types of regis-
tration: model-to-frame and frame-to-frame. The primary
role of model-to-frame registration is to correct drift ac-
cumulated over time by re-aligning the current frame with
the model, ensuring global consistency. On the other hand,
frame-to-frame registration focuses on providing smooth,
incremental motion estimation as the camera moves through
the environment, allowing SLAM methods to efficiently
compute the relative transformation between frames. While
the interplay of the two registration types have proven ef-
fective for SLAM, the 6DoF object tracking field has been
exclusively focused on the model-to-frame registration.
In this paper, we introduce GoTrack, a method for
generic 6DoF object tracking and pose refinement.
The
method adapts an analysis-by-synthesis approach for
model-to-frame registration, similarly to recent model-
based pose refinement methods, while drawing inspiration
from SLAM and introducing frame-to-frame registration to
reduce computational cost when tracking. Both registration
types are realized in GoTrack by optical flow estimation.
For the model-to-frame registration, we train a trans-
former decoder on top of DINOv2 to predict (1) an opti-
cal flow field between a synthetic object template and the
input image, and (2) a mask of template pixels that are visi-
ble in the input image. From these predictions we construct
2D-3D correspondences by linking visible pixels with 3D
points in the model coordinate frame, which are calculated
from the depth channel of the template. The 6DoF pose is
then calculated by the PnP-RANSAC algorithm [16,36].
For the frame-to-frame registration, we build on the ob-
servation that this is an easier problem as the differences
between consecutive video frames are typically minimal,
and solve it with a small off-the-shelf flow estimation net-
work [65]. The frame-to-frame flow is used to propagate
2D-3D correspondences to the next frame, and the more ex-
pensive model-to-frame registration is triggered only if the
frame-to-frame tracking is lost. Besides saving compute,
the use of the frame-to-frame registration is shown to yield
more stable tracking results, especially under occlusion.
Additionally, we show how the proposed method for
model-to-frame registration can be seamlessly combined
with a BoW-based template retrieval from FoundPose [51]
to create an efficient and accurate end-to-end pose estima-
tion pipeline. We demonstrate that the proposed method
reaches state-of-the-art results on the standard 6DoF object
pose refinement [28] and tracking [68,79] benchmarks.
In summary, we make the following contributions:
1. A flow-based 6DoF pose refiner that is simpler than ex-
isting methods, can correct large pose deviations, pro-
duces reliable pose confidence score without requiring
a scoring network, and reaches state-of-the-art RGB-
only results on both 6DoF object pose estimation and
tracking benchmarks.
2. A seamless 6DoF pose estimation pipeline that swiftly
generates coarse poses by template retrieval and opti-
mizes them by the flow-based refiner. The pipeline is
easy to implement as both stages rely on DINOv2.
3. A robust 6DoF pose tracking pipeline that extends the
refiner with frame-to-frame optical flow estimation,
which increases tracking efficiency, reduces jitter, and
improves robustness to occlusion.
2. Related work
In this section, we provide an overview of existing meth-
ods for the closely related problems of pose estimation, re-
finement and tracking of unseen objects.
Object pose estimation and refinement. Recent works
on 6DoF object pose estimation have introduced diverse
benchmarks [5, 10, 12, 22, 26, 31, 64, 80] which have pow-
ered many deep learning-based methods [29, 32, 34, 37, 38,
40,53,54,66,81]. As shown in the report of the BOP Chal-
lenge 2023 [28], the majority of state-of-the-art methods for
6DoF pose estimation rely on a three-stage pipeline: (1) 2D
detection/segmentation, (2) coarse pose estimation, and (3)
pose refinement. The top 16 methods for the seen object
task and top 13 methods for the unseen object task rely on
this pipeline. However, increased accuracy by the third re-
finement stage usually comes at a significant cost, e.g., the
coarse version of GenFlow [44], the winner of BOP 2023,
takes 3.8 seconds per image with around 6 objects on av-
erage. In contrast, methods based on rendered templates
and estimating coarse poses via nearest neighbor search of
frozen features of DINOv2 [50], such as ZS6D [2], and
FoundPose [51], have shown promising results while being
much faster. Motivated by these results, we build our pose
refiner on top of frozen DINOv2 [50], which also enables a
seamless integration with the recent coarse pose methods.
The refinement stage, which is crucial for achieving
state-of-the-art accuracy, is usually realized by an iterative
render-and-compare approach with direct regression (Cosy-
Pose [34] for seen objects, MegaPose [35] and Foundation-
Pose [77] for unseen objects), or with 2D-3D correspon-
dences (GenFlow [44]). A common property of these meth-
ods is that they require a pose ranking/selection step, which
requires an extra network that takes as input the test image
and renderings of the object in estimated poses, and outputs
pose scores. In contrast, our approach produces a reliable
pose score without requiring an additional scoring network.
2

Object pose tracking. Classical object tracking methods
can be categorized into keypoint-based [52, 56, 57, 61, 69],
edge-based [6, 13, 19, 58, 62], and direct optimization [4, 7,
41,59,67]. While keypoints and direct optimization are not
suitable for texture-less objects, edge-based methods typi-
cally struggle with background clutter and texture. To over-
come these issues, learning-based methods [8,17,42,72,75]
have been proposed. However, they either depend on RGB-
D input images [17, 75] or require extensive training data
specific to each target object [8, 42, 72]. To address these
issues, Nguyen et al. [46] proposed a method for track-
ing 6DoF pose of unseen objects. While their approach
only demonstrates results in simple scenarios or simple mo-
tions, we focus on more challenging conditions where ob-
jects can be heavily occluded or manipulated by hands. An-
other line of work includes category-level methods [39,70],
which focus on unseen instances belonging to training cat-
egories. Recently, several methods for instance-level track-
ing [8, 18, 37, 75], model-based [30, 63, 78], and model-
free tracking of unseen objects [74,76] have been proposed.
However, these methods either focus on seen objects or use
depth images, which limits their applicability.
Object pose refinement for tracking of unseen objects.
The most similar to our approach are methods introduced
in [35, 37, 44, 77]. A prominent example is DeepIM [37],
which introduced a deep-learning approach for iterative
render-and-compare pose refinement. While their method
shows generalization to unseen objects, the experiments are
limited to synthetic images. MegaPose [35] has further ex-
tended this approach and shown strong generalization to
unseen objects in real images by training on large-scale
datasets. In the BOP Challenge 2023 [28], GenFlow [44]
won the award by introducing a RAFT [65]-based architec-
ture to estimate the flow from the template to real input im-
ages. In this work, we also introduce a flow-based method
for unseen object pose refinement. However, our approach
does not require costly shape constraints, differentiable PnP,
specialized neural architecture, nor a pose scoring stage to
reach state-of-the-art results.
Moreover, all of the exist-
ing methods focus solely on model-to-frame registration.
When applied to tracking, they transition to the next frame
solely via pose parameters, discarding pixel-level informa-
tion about the established registration.
In this work, we
show that propagating 2D-3D correspondences via frame-
to-frame flow increases efficiency and reduces jitter.
3. Method
In this section we first propose a 6DoF pose refinement
method for unseen objects (Sec. 3.1), and then show how
this method can support efficient pipelines for estimating
accurate object pose from a single image (Sec. 3.2) and for
tracking the object pose in an image sequence (Sec. 3.3).
3.1. Flow-based object pose refinement
Problem formulation. Given a CAD model of an object, an
RGB image with known intrinsics that shows the object in
an unknown pose, and a coarse estimate of the object pose,
our objective is to refine the pose such as the 2D projection
of the model aligns closely with the object’s appearance in
the image. We assume the object is rigid and represent its
pose by a rigid transformation (R, t), where R is a 3D ro-
tation and t is a 3D translation from the model coordinate
frame to the camera coordinate frame.
Method overview. We adapt an analysis-by-synthesis ap-
proach, similarly to several recent methods for model-based
pose refinement [35,37,44,77]. In particular, we predict the
refined pose based on (i) an RGB-D template showing the
CAD model in the given coarse pose, and (ii) a crop of the
object region in the input RGB image (the region is deter-
mined by the CAD model and the coarse pose). Instead of
directly regressing a relative 6DoF pose between the ren-
dering and the crop (as in [35, 37, 77]), we train a network
to predict dense 2D-2D correspondences, represented as an
optical flow field, together with a mask of pixels that are vis-
ible in both images. The pose is then estimated by lifting the
correspondences to 2D-3D using the template depth, and by
applying PnP-RANSAC [16, 36]. The refinement method
can be applied in several iterations for increased accuracy.
Predicting 2D outputs (flow and visibility mask) in-
stead of the relative 6DoF pose makes the network eas-
ier to train, interpretable and versatile – we show how to
use the predicted correspondences to calculate a reliable
and controllable pose score and how to naturally prop-
agate them via frame-to-frame flow in case of tracking.
The correspondence-based approach could also support 2D
tracking (just by turning off the 3D reasoning part).
Predicting 2D-3D correspondences. Let Ic be a perspec-
tive crop of the input RGB image around a provided 2D
bounding box of the object. The crop is obtained as in [51]
by warping the image to a virtual pinhole camera whose
optical axis passes through the center of the 2D bounding
box. Then let It be an RGB-D template that is rendered us-
ing a pinhole camera and shows the object in an initial pose
(R0, t0), let Mt be a binary mask of the object silhouette in
It, and let O be a set of 3D points representing the surface
of the object model. It and Ic have the same resolution.
Given the crop Ic and the template It, we aim to establish
correspondences between their pixels. We deem pixels ut ∈
It and uc ∈Ic as corresponding if they are aligned with 2D
projections of the same 3D point x ∈O: ut = πt(R0x +
t0) and uc = πc( ¯Rx + ¯t), where π: R3 7→R2 is the 2D
projection operator, and ( ¯R,¯t) is the GT object pose.
To establish the correspondences, we train a neural net-
work to predict two outputs for each pixel ut ∈Mt: a like-
3

Figure 2. Overview of the proposed efficient pipeline for 6DoF pose estimation. Each object is first onboarded by rendering RGB-D
templates of its 3D model, and by extracting DINOv2 [50] feature maps from RGB channels of the templates. At inference, we first crop the
input image around a 2D object detection (from CNOS [47]) and retrieve a template that shows the object in a similar pose as in the crop by
the BoW-based approach from [51]. Then we use a transformer decoder [73] with a DPT head [55] to predict (1) 2D-2D correspondences
between the template and the input, and (2) a mask of template pixels that are visible in the input. Next we lift the 2D-2D correspondences
established at visible template pixels to 2D-3D using the depth channel of the template, and estimate the pose by PnP-RANSAC.
lihood vt→c(ut) that the corresponding pixel uc ∈Ic is
visible, and a 2D flow vector ft→c(ut) such that ut +
ft→c(ut) = uc. Given the network predictions, we collect a
set of weighted 2D-2D correspondences D = {(uc, ut, w)}
by linking pixels uu and ut if they are related by ft→c(ut)
and if vt→c(ut) is above a threshold τv. The weight w is de-
fined by vt→c(ut). Finally, we convert each linked template
pixel ut to a 3D point x ∈O using the template depth and
known camera intrinsics, to obtain a set of weighted 2D-3D
correspondences C = {(uc, x, w)}.
Pose fitting. The refined pose ( ˆR,ˆt) is estimated from the
established correspondences C by solving the Perspective-
n-Point (PnP) problem. As in [51], we solve this problem
by the EPnP algorithm [36] combined with the RANSAC
fitting scheme [16] for robustness. In this scheme, PnP is
solved repeatedly on a randomly sampled minimal set of 4
correspondences, and the output is defined by the pose hy-
pothesis with the highest number of inlier correspondences
C′ ⊆C, for which the 2D re-projection error [36] is below
a threshold τr. The final pose is then refined from all inliers
by the Levenberg-Marquardt optimization [45].
Quality of pose estimates. We define the quality of an es-
timated pose ( ˆR,ˆt) as: q = s′/s, where s′ is the sum of
weights w of inlier correspondences C′, and s is the sum of
weights w of all correspondences C. As shown in Sec. 4,
this simple quality measure is surprisingly effective for se-
lecting the best pose estimate in a multiple-hypotheses re-
finement setup. Note that existing methods [35,44,77] need
a special pose scoring network for this purpose.
Network architecture and training. We rely on a frozen
DINOv2 [50] backbone that we use to extract features from
the image crop Ic and the template image It.
We then
pass the features to a two-branch transformer-based decoder
from CroCov2 [73] (one branch for the template and one for
the crop), and finally obtain the predictions by applying the
DPT head [55] to the output of the template branch. The
decoder and the head are trained jointly by minimizing the
following loss averaged over all pixels ut ∈Mt:
L(ut) =BCE
 vt→c(ut), ¯vt→c(ut)

+ ¯vt→c(ut)
ft→c(ut) −¯ft→c(ut)

1,
where BCE is the binary cross entropy loss between the pre-
dicted visibility vt→c(ut) and the binary ground-truth vis-
ibility ¯vt→c(ut). As in RAFT [65], the optical flow pre-
diction is supervised by minimizing the L1 distance be-
tween the predicted flow vector ft→c(ut) and the ground-
truth flow vector ¯ft→c(ut). Note that the flow prediction
is supervised only at template pixels ut ∈Mt for which
the corresponding crop pixel uc ∈Mc is visible, i.e., when
¯vt→c(ut) = 1. The ground-truth visibility value uc ∈Mc
is defined by Mc(ut + ft→c(ut)), where Mc is a modal
mask defined in the crop (Mc is required only at training).
We train the network on the MegaPose-GSO dataset [35]
that offers 1M PBR images rendered with BlenderProc [9],
shows 1000 GSO objects [11] that are annotated with the
GT 6DoF poses and 2D segmentation masks. As in [35],
we apply heavy image augmentation during training, which
avoid overfitting to the synthetic image domain and enables
applying the network to real images at inference time. We
also randomly perturb the GT poses as in [35].
4

3.2. Efficient object pose estimation pipeline
Similarly to the state-of-the-art methods for 6DoF pose
estimation of unseen objects [28], we adopt a three-stage
pipeline: 2D object detection, coarse 6DoF pose estima-
tion, 6DoF pose refinement. While the first stage can be re-
alized by any 2D object detector (e.g. CNOS [47]), the key
difference of our pipeline is in the other two stages, where
we apply efficient template retrieval for coarse pose estima-
tion and our flow-based approach for pose refinement, both
based on frozen DINOv2 [50]. This streamlined pipeline
delivers competitive accuracy while being faster than exist-
ing methods (Sec. 4). Fig. 2 illustrates the pipeline.
Object onboarding. First, in an offline onboarding stage,
we pre-render a set of RGB-D templates (800 in our experi-
ments), similarly to [51]. Given a texture-mapped 3D object
model, we render templates showing the model under orien-
tations sampled to uniformly cover the SO(3) group of 3D
rotations [1], and the model is rendered using a standard ras-
terization technique [60] with a gray background and fixed
lighting. We also pre-extract DINOv2 feature maps from
RGB channels of the templates.
Coarse pose estimation by template retrieval. Given a
2D object detection (e.g., from CNOS [47]), we generate a
coarse object pose with the fast template retrieval approach
from FoundPose [51]. In Sec. 4, we show that poses as-
sociated with the retrieved templates are sufficient for our
flow-based refiner to converge to an accurate pose.
Seamless integration.
The template retrieval approach
from [51] leverages Bag-of-Words (BoW) descriptors that
are computed from patch features from frozen DINOv2.
Since our refiner utilizes the same backbone, the feature ex-
traction process needs to be performed only once per im-
age crop. The extracted DINOv2 features can then be used
both to retrieve templates and to refine the poses associated.
Besides sharing the same backbone, the two stages in fact
share the exact same template-based object representation.
3.3. Extension to object pose tracking
Limitation of tracking by refinement. The proposed re-
finer can be directly applied to track the 6DoF object pose
in an image sequence. This can be achieved as in [35, 77]
by taking the pose from the previous frame as an initial
pose for the next frame. Although straightforward, this ap-
proach tends to suffers from jitter, i.e., rapid fluctuations or
inconsistencies, often resulting in a shaky or unstable ap-
pearance in the tracking output. We argue that this limita-
tion arises because predictions between consecutive frames
are connected solely through pose parameters. Information
about the pixel-level model-to-image registration from the
previous frame is discarded. Consequently, the registration
process, a core challenge in all analysis-by-synthesis refine-
Figure 3. Frame-to-frame consistency for tracking. Besides
the model-to-frame registration (between a synthetic template and
the real input image), which is the main building block of exist-
ing 6DoF object tracking methods, we introduce frame-to-frame
registration, which reduces jitter and increases efficiency. As the
difference between consecutive video frames is typically minimal,
estimating frame-to-frame correspondences is a simpler problem
and can be solved by a light-weight optical flow network such as
RAFT [65]. The model-to-frame correspondences need to be es-
tablished only in case of a low frame-to-frame tracking confidence.
ment methods, must largely restart in each new frame. We
address this limitation by introducing frame-to-frame con-
sistency, which we introduce next.
Frame-to-frame consistency. Inspired by the SLAM liter-
ature [14], where the pose between a 3D scene model and
a camera is continuously estimated from video frames not
only by model-to-frame but also by frame-to-frame regis-
tration, we introduce the direct frame-to-frame connection
to our pipeline by predicting optical flow between the pre-
vious and the current video frames.
Specifically, we obtain a perspective crop of the current
frame j by constructing a virtual camera looking at the ob-
ject model in the current pose estimate, and use RAFT [65]
to predict optical flow between the previous and the current
crop. We then apply this flow to the 2D coordinates of the
inlier 2D-3D correspondences C′
i established in the previ-
ous frame i to obtain a set of propagated inliers C′
i→j, and
re-estimate the pose by the EPnP algorithm from C′
i→j. If
the inlier ratio w.r.t. the re-estimated pose is above a thresh-
old τi (typically 0.8), we use the re-estimated pose as the
final pose estimate at frame j and proceed to the next frame.
Otherwise, if the inlier ratio is below the threshold, we pre-
dict the model-to-frame flow (as described earlier) to obtain
a new set of 2D-3D correspondences Cj. In this case, to en-
courage temporal pose smoothness, we combine sets C′
i→j
5

Input image
Template
Predicted mask
Predicted flow
Warped template
Estimated pose
Figure 4. Example results of the GoTrack refiner on LM-O [5], YCB-V [79], and T-LESS [23]. The input image is shown in the first
column, and the template retrieved using the BoW-based approach [51] in the second. The third and fourth columns show the predictions
of our network, which can then be used to remap pixels from the template to the input image as shown in the fifth column. The last column
presents the final pose estimated by PnP-RANSAC from 2D-3D correspondences (the contour of the object model in the initial pose is
shown in blue, and in the estimated pose in red). As shown in the third column, our method can reliably predict which part of the object is
visible, despite never seeing the object during training.
and Cj by taking all correspondences from the first set and
randomly selecting up to r|C′
i→j| correspondences from the
second, i.e., such as the ratio of correspondences selected
from the two sets is up to r. The pose is then estimated
from the combined set by EPnP, and inlier correspondences
from the combined set are propagated to the next frame.
The parameter r, which controls the ratio of correspon-
dences selected from the previous and the current frame,
can be effectively used to control the trade-off between jitter
and the risk of drift (i.e., the risk of deviating from the ac-
tual object pose). For lower values of r the tracking tends to
be smoother but may suffer from drifting, while for higher
values of r the tracking may be jittery.
4. Experiments
In this section, we describe the experimental setup for
6DoF object pose refinement (Sec. 4.1) and for 6DoF
object pose tracking (Sec. 4.2), and compare GoTrack
with the state-of-the-art RGB-only methods for these tasks.
Specifically, we compare GoTrack with refinement methods
MegaPose [35] and GenFlow [44] on BOP datasets [28],
with tracking methods LDT3D [67], SRT3D [62] and
DeepAC [72] on the RBOT dataset [68], and tracking
methods PoseCNN [79] and PoseRBPF [8] on the YCB-V
dataset [79]. Among the tracking methods, only LDT3D
and SRT3D operate on unseen objects while DeepAC,
PoseCNN, and PoseRBPF are trained on the test objects.
Implementation details. The evaluated version of the pro-
posed refinement method used 800 templates (with approxi-
mately 25◦angle between depicted object orientations), the
template and crop resolution of 280×280 px, DINOv2 [50]
with the ViT-S architecture, the transformer decoder from
CroCov2 [73] with 12 blocks, the default DPT head [55],
the visibility threshold τv set to 0.3, up to 400 iterations
of PnP-RANSAC, and the re-projection threshold τr set to
4 px. For the pose refinement experiments, we use 5 re-
finement iterations as previous works [35, 44]. Although
our flow-based refiner could rely solely on templates pre-
rendered in the onboarding phase (at each refinement itera-
tion, we could use the template that best matches the orien-
tation of the current pose estimate), we found that this ap-
proach does not perform as well as online rendering, where
the 3D model is rendered in the current pose estimate and
therefore does not suffer from the pose quantization error.
Online rendering is used also in the prior works [35,44]. For
6

#
Method
LM-O
T-LESS
TUD-L
IC-BIN
ITODD
HB
YCB-V
Average↑
Time↓Nets↓
ARMSPD
AR
ARMSPD
AR
ARMSPD
AR
ARMSPD
AR
ARMSPD
AR
ARMSPD
AR
ARMSPD
AR
ARMSPD
AR
Coarse pose estimation:
1 FoundPose [51]
60.7 39.6
48.9 33.8
67.0 46.7
33.9 23.9
37.0 20.4
65.0 50.8
59.6 45.2
52.2 37.2
1.7 s
1
2 GigaPose [48]
51.0 29.6
47.9 26.4
51.7 30.0
34.8 22.3
31.4 17.5
52.5 34.1
52.5 27.8
45.4 26.8
0.4 s
1
3 BoW retrieval∗
25.9 11.6
28.8 15.4
25.3 13.7
17.4 10.1
31.6 17.4
55.1 42.4
15.2 11.6
28.5 17.5
0.3 s
1
Refinement using a single hypothesis from FoundPose [51]:
4 GoTrack
70.4 56.5
57.3 50.4
81.5 67.2
50.7 43.2
52.8 39.3
76.3 72.2
75.8 63.1
66.4 56.0
2.5 s
1
5 MegaPose [35]
67.7 55.4
55.6 51.0
78.4 63.3
47.4 43.0
45.2 34.6
73.6 69.5
76.0 66.1
63.4 54.7
4.4 s
2
Refinement using a single hypothesis from GigaPose [48]:
6 GoTrack
72.6 58.4
61.9 54.9
73.9 61.3
52.5 45.3
50.0 39.5
77.0 72.9
73.8 62.0
66.0 56.3
1.3 s
2
7 GenFlow [44]
71.1 59.5
60.1 55.0
72.4 60.7
52.1 47.8
49.5 41.3
74.7 72.2
73.2 60.8
64.7 56.8
2.2 s
2
8 MegaPose [35]
68.0 55.7
59.2 54.1
71.9 58.0
49.4 45.0
45.4 37.6
73.1 69.3
73.4 63.2
64.2 54.7
2.3 s
2
Refinement using BoW retrieval for coarse pose estimation (from row 3):
9 GoTrack
70.4 56.9
56.9 50.4
79.2 65.2
49.7 42.4
52.6 40.2
72.5 69.5
75.2 62.9
65.2 55.4
0.8 s
1
Pose refinement using 5 hypotheses from GigaPose [48]:
12 GoTrack
78.1 62.7
66.3 59.5
80.8 66.8
54.4 46.6
56.3 43.0
81.9 77.1
80.6 68.4
71.2 60.6
3.0 s
2
13 GenFlow [44]
75.3 63.1
63.7 58.2
79.4 66.4
55.1 49.8
55.1 45.3
78.3 75.6
78.8 65.2
69.4 60.5
10.6 s
3
14 MegaPose [35]
73.6 59.8
62.0 56.5
77.9 63.1
52.8 47.3
48.5 39.7
76.5 72.2
77.7 66.1
65.3 57.8
7.7 s
3
Table 1. Pose refinement performance on the seven core BOP datasets [28]. Reported are accuracy scores, the average time required to
estimate poses of all objects in an image (in seconds), and the number of different neural networks required by the pipelines.
the object tracking experiments, the parameter τi that trig-
gers the template-to-frame registration was set to 0.8, and
the mixing ratio r was set to 2. We trained the flow-based re-
finer until convergence on the MegaPose-GSO dataset [35]
using Adam [33], with DINOv2 weights being frozen.
4.1. 6DoF object pose refinement
Evaluation datasets. We evaluate our method on the seven
core BOP datasets [28]: LM-O [5], T-LESS [22], TUD-
L [26], IC-BIN [10], ITODD [12], [31], and YCB-V [80].
These datasets include the total of 132 different objects and
19048 annotated test object instances.
Evaluation metrics. We use the BOP evaluation proto-
col [27], which relies on three metrics: Visible Surface Dis-
crepancy (VSD), Maximum Symmetry-Aware Surface Dis-
tance (MSSD), and Maximum Symmetry-Aware Projection
Distance (MSPD). The final average recall (AR), is calcu-
lated by averaging the individual AR scores of these three
metrics across a range of error thresholds.
Baselines. We compare our method with MegaPose [35]
and GenFlow [44], which are currently the state-of-the-art
methods on the BOP benchmark. The results of MegaPose
and GenFlow are sourced from the BOP leaderboard [28].
Results.
When starting from coarse poses estimated by
FoundPose [51], our refiner outperforms the MegaPose
refiner [35] on the majority of BOP datasets, achieving
+3.0 ARMSPD and +1.3 AR on average (rows 4 and 5 in
Tab. 1). When starting from coarse poses estimated by Gi-
gaPose [48], our refiner outperforms MegaPose by similar
margins (rows 6 and 8). Compared to GenFlow [44], our
refiner achieves +1.3 ARMSPD and -0.5 AR (rows 6 and 7).
In terms of complexity of the model architecture, the
simplest from the evaluated methods is the pipeline pro-
posed in Sec. 3.2, which relies on the BoW-based template
retrieval for coarse pose estimation followed by our refiner.
This pipeline relies on only a single neural network (col-
umn “Nets”) that is shared by both stages, which makes
it faster and easier to deploy on mobile devices. Compar-
ing runtime of other methods is problematic as each was
evaluated on a different GPU (we used V100, while Mega-
Pose was evaluated on RTX 2080 and GenFlow on RTX
3090). However, the difference in complexity of the meth-
ods is clear. Both GenFlow and MegaPose pipelines rely
on different networks for coarse pose estimation and for re-
finement. In case of multi-hypotheses setup, where multiple
coarse poses are considered per object instance and their re-
fined versions need to be at the end ranked to select the final
output, the GenFlow and MegaPose pipelines need an addi-
tional scoring network, whereas our method uses reliable
pose confidence score can be obtained by simply calculat-
ing the weighted inlier ratio defined in Sec. 3.1.
4.2. 6DoF object pose tracking
Evaluation datasets. We evaluate our tracking method on
two popular datasets: YCB-V [79] and RBOT [68]. Addi-
tional tracking results on the recent HOT3D [3] dataset are
presented in the appendix.
Evaluation metrics.
We use the same standard evalua-
tion metrics as in [8, 62, 67, 72, 79]: Area Under the Curve
7

PoseCNN [79]
PoseRBPF [8]
GoTrack
GoTrack+f2f
Initialization
-
PoseCNN
FoundPose
FoundPose
Unseen objects
✗
✗
✓
✓
ADD
ADD-S
ADD
ADD-S
ADD
ADD-S
ADD
ADD-S
002 master chef can
50.9
84.0
58.0
77.1
71.0
86.3
71.3
86.3
003 cracker box
51.7
76.9
76.8
87.0
79.6
89.2
80.6
89.8
004 sugar box
68.6
84.3
75.9
87.6
80.4
89.9
80.5
90.0
005 tomato soup can
66.0
80.9
74.9
84.5
74.0
87.2
71.2
85.8
006 mustard bottle
79.9
90.2
82.5
91.0
83.7
91.9
84.4
92.2
007 tuna fish can
70.4
87.9
59.0
79.0
81.4
91.7
82.0
91.9
008 pudding box
62.9
79.0
57.2
72.1
82.8
90.4
83.5
90.7
009 gelatin box
75.2
87.1
88.8
93.1
83.4
90.9
84.4
91.5
010 potted meat can
59.6
78.5
49.3
62.0
80.5
91.3
83.5
92.8
011 banana
72.3
85.9
24.8
61.5
53.8
71.0
56.0
68.7
019 pitcher base
52.5
76.8
75.3
88.4
79.7
90.0
79.9
90.1
021 bleach cleanser
50.5
71.9
54.5
69.3
68.9
83.1
69.3
83.3
024 bowl
6.5
69.7
36.1
86.0
35.1
66.2
34.8
62.0
025 mug
57.7
78.0
70.9
85.4
70.1
87.1
50.8
87.4
035 power drill
55.1
72.8
70.9
85.0
81.9
91.2
82.1
91.3
036 wood block
31.8
65.8
2.8
33.3
0.7
30.4
5.3
40.9
037 scissors
35.8
56.2
21.7
33.0
58.3
76.5
59.7
77.8
040 large marker
58.0
71.4
48.7
59.3
60.0
67.3
65.0
73.9
051 large clamp
25.0
49.9
47.3
76.9
64.1
83.1
64.4
83.2
052 extra large clamp
15.8
47.0
26.5
69.5
83.5
93.5
82.8
93.0
061 foam brick
40.4
87.8
78.2
89.7
83.0
92.3
83.3
92.4
Average
53.7
75.9
59.9
77.5
69.3
82.9
69.3
83.6
Table 2. Tracking performance on the YCB-V dataset [79].
The accuracy is measured by AUC w.r.t. the ADD and ADD-S
pose error functions. Despite not being trained on the YCB-V ob-
jects, our method outperforms PoseCNN [79] and PoseRBPF [8]
which were trained on these specific objects. Adding the frame-to-
frame consistency (f2f) makes our method 6X more efficient while
slightly improving the accuracy (see text for details).
Method
with reset
w/o reset
Regular
Dynamic
Noisy
Occlusion
Mean
Resets ↓
Mean
SRT3D [62]
94.2
94.6
81.7
93.2
90.9
6575
19.0
LDT3D [67]
95.2
95.4
83.2
94.9
92.1
6228
18.8
DeepAC [72]*
95.6
95.6
88.0
94.0
93.3
4826
30.3
GoTrack
97.3
96.2
94.2
95.4
95.9
3021
86.7
Table 3. Tracking on the RBOT dataset [68] with/without reset
as in [62, 67, 72]. The accuracy is measured by 5cm-5°score and
the number of reset. Character * indicates that DeepAC is a su-
pervised method trained on the test objects. GoTrack significantly
outperforms existing methods across all settings and metrics.
(AUC) for the ADD and and ADD(-S) pose error functions
on YCB-V dataset, and 5cm-5°and the number of resets on
RBOT dataset. A reset is counted when the tracking method
fails, i.e., when the translation error exceeds 5 centimeters
or the rotation error exceeds 5 degrees, the method is re-
initialized with the ground-truth pose. More details about
these metrics are in the appendix.
Baselines. Since no RGB-only methods for tracking of un-
seen objects have been evaluated on the YCB-V dataset, we
compare our method with tracking methods for seen objects
(i.e., methods trained on the target objects), PoseCNN [79]
and PoseRBPF [8]. PoseCNN is a method for 6DoF ob-
ject pose estimation from a single image and was eval-
uated frame by frame.
PoseRBPF was initialized at the
first frame with a pose estimated by PoseCNN. Since the
PoseCNN estimates are not publicly available, we initial-
ized our tracking method with a pose estimate from Found-
Pose. On the RBOT dataset, we compare our method with
contour-based methods DeepAC [72] and SRT3D [62] and
the optimization-based method LDT3D [67].
Results on YCB-V [79]. As shown in Tab. 2, even though
our method was not trained on the YCB-V objects, it signifi-
cantly outperforms both PoseCNN [79] and PoseRBPF [8],
which were trained specifically for these objects. Adding
the frame-to-frame flow makes our method around 6X more
efficient while slightly improving the accuracy. The for-
ward pass of the flow-based refiner, which we use for the
template-to-frame registration, takes 40 ms while the for-
ward pass of RAFT-Small, which we use for the frame-to-
frame registration takes only 6 ms. The template-to-frame
flow is triggered only every 50th frame on average on se-
quences from the YCB-V dataset. Another effect of adding
the frame-to-frame flow is a reduced jitter, which can be
seen in the supplementary video (App. D).
Results on RBOT [68]. As shown in Tab. 3, GoTrack
outperforms LDT3D, SRT3D and DeepAC on all RBOT
sequence types while requiring 37–54% less resets. The
most noticeable difference is on the noisy sequences where
contour detection is less reliable. Furthermore, these meth-
ods work well only on objects with distinct contour – they
struggle if the same contour shape corresponds to multi-
ple different poses and the actual pose can be estimated
only based on texture (e.g., DeepAC achieves only 70.1%
5cm-5°accuracy on the textured and cylindrical soda object
from RBOT while GoTrack achieves 97.2%). Additionally,
Tab. 3 shows that our method significantly outperforms the
other methods in tracking without reset (86.7 vs 30.3 AUC).
5. Conclusion
We introduced a method for 6DoF pose refinement of
unseen objects, based on predicting template-to-frame opti-
cal flow along with a mask of visible template pixels. We
showed how our refinement method can be seamlessly com-
bined with a BoW-based template retrieval to create an ef-
ficient and accurate object pose estimation pipeline. Addi-
tionally, we proposed an extension for 6DoF pose tracking
of unseen objects by propagating 2D-3D correspondences
using frame-to-frame optical flow. Our approach achieves
state-of-the-art results on the standard 6DoF object pose re-
finement and tracking benchmarks. For future work, we
plan to draw further inspiration from SLAM literature and
explore extending our approach to a model-free setup.
8

References
[1] Marc Alexa. Super-fibonacci spirals: Fast, low-discrepancy
sampling of so (3). In CVPR, 2022. 5
[2] Philipp Ausserlechner, David Haberger, Stefan Thalhammer,
Jean-Baptiste Weibel, and Markus Vincze.
ZS6D: Zero-
Shot 6D Object Pose Estimation Using Vision Transformers.
2023. 2
[3] Prithviraj Banerjee,
Sindi
Shkodrani,
Pierre
Moulon,
Shreyas Hampali, Fan Zhang, Jade Fountain, Edward Miller,
Selen Basol, Richard Newcombe, Robert Wang, et al. Intro-
ducing hot3d: An egocentric dataset for 3d hand and object
tracking. arXiv preprint arXiv:2406.09598, 2024. 1, 7, 12,
13
[4] Selim Benhimane and Ezio Malis. Real-time image-based
tracking of planes using efficient second-order minimization.
In IEEE/RSJ International Conference on Intelligent Robots
and Systems, 2004. 1, 3
[5] Eric Brachmann, Alexander Krull, Frank Michel, Stefan
Gumhold, Jamie Shotton, and Carsten Rother. Learning 6D
Object Pose Estimation Using 3D Object Coordinates. In
ECCV, 2014. 2, 6, 7, 14
[6] Andrew I. Comport, Eric Marchand, Muriel Pressigout, and
Francois Chaumette.
Real-time markerless tracking for
augmented reality: The virtual visual servoing framework.
IEEE Transactions on Visualization and Computer Graph-
ics, 2006. 1, 3
[7] Alberto Crivellaro and Vincent Lepetit. Robust 3D tracking
with descriptor fields. In CVPR, 2014. 1, 3
[8] Xinke Deng, Arsalan Mousavian, Yu Xiang, Fei Xia, Timo-
thy Bretl, and Dieter Fox. PoseRBPF: A Rao–Blackwellized
particle filter for 6-D object pose tracking. IEEE Transac-
tions on Robotics, 2021. 1, 3, 6, 7, 8, 12
[9] Maximilian Denninger,
Martin Sundermeyer,
Dominik
Winkelbauer, Dmitry Olefir, Tom´aˇs Hodaˇn, Youssef Zidan,
Mohamad Elbadrawy, Markus Knauer, Harinandan Katam,
and Ahsan Lodhi. BlenderProc: Reducing the reality gap
with photorealistic rendering. RSS Workshops, 2020. 4
[10] Andreas Doumanoglou, Rigas Kouskourid