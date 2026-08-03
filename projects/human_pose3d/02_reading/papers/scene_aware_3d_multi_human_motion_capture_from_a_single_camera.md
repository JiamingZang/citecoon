# Scene‐Aware 3D Multi‐Human Motion Capture from a Single Camera

> 2023 · id: W4378083152 · arXiv: 2301.05175 · pdf: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/cgf.14768 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this work, we consider the problem of estimating
the 3D position of multiple humans in a scene as well as
their body shape and articulation from a single RGB video
recorded with a static camera.
In contrast to expensive
marker-based or multi-view systems, our lightweight setup
is ideal for private users as it enables an affordable 3D mo-
tion capture that is easy to install and does not require ex-
pert knowledge. To deal with this challenging setting, we
leverage recent advances in computer vision using large-
scale pre-trained models for a variety of modalities, in-
cluding 2D body joints, joint angles, normalized disparity
maps, and human segmentation masks. Thus, we introduce
the ﬁrst non-linear optimization-based approach that jointly
solves for the absolute 3D position of each human, their ar-
ticulated pose, their individual shapes as well as the scale
of the scene.
In particular, we estimate the scene depth
and person unique scale from normalized disparity predic-
tions using the 2D body joints and joint angles. Given the
per-frame scene depth, we reconstruct a point-cloud of the
static scene in 3D space. Finally, given the per-frame 3D
estimates of the humans and scene point-cloud, we perform
a space-time coherent optimization over the video to ensure
temporal, spatial and physical plausibility. We evaluate our
method on established multi-person 3D human pose bench-
marks where we consistently outperform previous methods
and we qualitatively demonstrate that our method is ro-
bust to in-the-wild conditions including challenging scenes
with people of different sizes. Code: https://github.
com/dluvizon/scene-aware-3d-multi-human

## introduction
Estimating the absolute 3D position, body shape, and ar-
ticulation of multiple people in a scene is a fundamental
research problem that has many applications in game devel-
opment, VR/AR, and HCI. Years of research went into de-
veloping sophisticated and expensive setups such as multi-
view systems, motion capture suits, and manually or semi-
automatically denoising of the tracked motions to then, for
example, animate CG characters with these captured mo-
tions. However, one ideally would like to obtain such an
absolute scene understanding from a capture setup that is
easy to install, affordable, and that does not require expert
knowledge, i.e. a single RGB camera. Such a lightweight
setup would enable 3D motion capture for private users, e.g.
avatar control via the smartphone, but it can also be applied
for post production in the movie industry where, for exam-
1
arXiv:2301.05175v3  [cs.CV]  27 Mar 2023

ple, one person should be replaced by another in a 3D con-
sistent manner. At the same time, it has to be stated that
performing motion capture given such limited data is ex-
ceptionally more difﬁcult compared to multi-view systems.
The major challenges for such a monocular setting, where
only a single static video of the entire scene with moving
persons is given, are the inherent depth ambiguity and oc-
clusions, among many others.
Therefore, recent monocular approaches focus on a sin-
gle human [33, 40] or even assume an actor template is
given [13,14,61]. Recently, some works started to research
the multi-person setting, but they either only learn a rela-
tive depth ordering of people in the scene [20] that is not
3D consistent over time or they directly predict absolute
depth, which is prone to overﬁt to the settings shown in the
training data [37]. Most of those works leverage recent ad-
vances in Computer Vision and take as input several types
of regressed data modalities obtained from models trained
on large-scale data. This involves 1) 2D body joints [4,11],
2) joint angles [51], 3) normalized disparity maps [27, 41],
and 4) human segmentation masks [8]. Interestingly, none
of those works jointly considers all of those modalities.
To this end, this work investigates how each of those
modalities can beneﬁt the task of multi-person absolute 3D
pose and shape estimation. A particular challenge, however,
is that each individual modality has, of course, advantages,
but also disadvantages. While 2D and 3D keypoint detec-
tions can help to infer the local 3D pose of a single person,
they cannot ensure 3D consistency across humans and the
scene. Joint angle estimates can be directly used to drive
CG characters, but they are usually less accurate than the
3D keypoint detectors due to error accumulation along the
kinematic chain. Normalized disparity maps provide global
reasoning of the entire scene as well as the humans in terms
of its scale-normalized depth, but they cannot provide abso-
lute depth and scale of the scene. Finally, human segmen-
tation masks can provide close to pixel-perfect and identity
preserving segmentations of humans in the scene, but they
lack a 3D understanding.
Now, to unite all the advantages of each of the modali-
ties while compensating for their potential limitations, we
propose the ﬁrst optimization-based approach that jointly
recovers the absolute 3D position of all humans in the im-
ages, their articulated pose, their individual shapes, as well
as the scale of the scene from a single video recorded with a
static camera; see Fig. 1. In particular, we propose a novel
energy formulation, which infers the absolute scene depth
and the person unique scale from scale-normalized dispar-
ity predictions by using the 2D and joint angle estimates of
the humans in the scene as a prior. Once the per-frame ab-
solute depth is known, we reconstruct a dense point cloud of
the static scene in absolute 3D space by segmenting out the
humans using the predicted segmentations and aggregating
per-frame depth over time. Finally, we perform a coherent
space-time optimization over the entire sequence to ensure
temporal and spatial consistency as well as physical plausi-
bility leveraging the aggregated scene estimate and the joint
angle predictions. Note that in each of those steps, the com-
bination of different data modalities is leveraged through
our method and only this speciﬁc approach achieves the de-
sired result in the considered setting, as extensively shown
in our results. In summary, our primary technical contribu-
tions are as follows:
• The ﬁrst monocular approach for multi-person abso-
lute pose and unique scale estimation that jointly esti-
mates multiple human poses and the 3D scene by com-
bining data modalities in a novel optimization frame-
work.
• A human body prior to disambiguate the scale of the
scene, which allows us to perform a coherent space-
time reasoning of the human motion in absolute space.
• We show that the estimated 3D human bodies can be
reﬁned in 3D space and time by ﬁltering body move-
ments in 3D coordinates and by penalizing implausible
poses w.r.t. the estimated scene, resulting in a more co-
herent ﬁnal prediction.
Since our approach estimates joint angles, global positions
and scale, the recovered 3D human poses can be directly
applied to CG characters enabling exciting applications as
shown in Section 4. Moreover, we demonstrate that the joint
reasoning of the human body shape, pose, and the dense
scene over the entire video sequence improves state of the
art in terms of 3D localization, scene and person scale, as
well as body pose compared to prior work, both, quanti-
tatively and qualitatively.
Finally, we show that several
downstream applications can be directly derived from our
method, like monocular human motion capture and avatar
control.

## method
The goal of our method is to estimate the absolute 3D
position of each human in the scene, i.e., up to a unique and
global scale, their proxy shape and pose, as well as the scene
scale solely from a monocular RGB video recorded with
a static camera for which we know the intrinsics. To this
end, we propose a uniﬁed approach that, for the ﬁrst time,
leverages all available data modalities, including 2D joint
detections, regressed SMPL parameters, estimated dispar-
ity maps, and human segmentations. As illustrated in Fig-
ure 2, our method is divided into two stages. The ﬁrst stage,
i.e. Image Modality Regression and Matching (Section 3.1),
extracts per-frame estimates and aggregates human-related
predictions to individuals throughout the video sequence.
The second stage, i.e. the proposed Optimization Frame-
work, estimates the person and per-frame scene scale, the
global 3D position of each person in the scene, as well as
the reﬁned articulated body pose in the form of joint angles
per frame.
The optimization framework is further subdivided into
two parts. The Scene Scale and Depth Disambiguation part
(Section 3.2) recovers a consistent and absolute 3D scene
depth per frame, the human scales, and their absolute 3D
position and body pose by jointly reasoning about multi-
ple humans and the scene. The second part, referred to as
Space-time Coherent Pose Optimization (Section 3.3), re-
ﬁnes the pose and position of the estimated humans in a
space-time coherent formulation, i.e. we enforce over the
entire sequence the estimated poses to be temporally sta-
ble and physically plausible. For this, we leverage a rough
scene geometry estimation, which is obtained by aggregat-
ing the absolute depth maps also estimated by our method.
This ﬁnal part signiﬁcantly reduces artifacts, such as foot
sliding, human-scene intersections, and jitter. Before we
explain our method in more detail, we introduce relevant
notations.
Notations.
The input of our framework is a video se-
quence It, with t ∈{1, . . . , T}, where T is the number of
frames. We leverage the skinned multi-person linear model
(SMPL) [30] to represent the humans in the scene. SMPL is
a differentiable parametric human model that takes as input
the pose parameters θ ∈R72, corresponding to the axis-
angles of 24 body joints and the global body rotation, and
PCA shape parameters β ∈R10, and produces a skinned
human mesh
fsmpl(θ, β) = V,
(1)
where V are the posed and shaped vertices of the human
4

body; for more details we refer to their paper [30]. The
mesh vertices regressed by SMPL can also be used to es-
timate a sparse 3D pose as J (V), where J (·) is a linear
regressor parameterized by a matrix W ∈RJ×6890, and J
denotes the total number of joints.
To account for translations in 3D space, we further add a
translation Γt,n ∈R3 to the SMPL representation, where n
is the person index. Furthermore, the 3D human pose mod-
els are overwhelmingly biased towards adult body sizes.
Thus, we explicitly model the person scale by sn ∈R+
and our ﬁnal human mesh can be deﬁned as
˜Vt,n = snVt,n + Γt,n.
(2)
This human mesh for person n at time t is then fully deter-
mined by the parameters θt,n, Γt,n, βn, and sn, which we
aim to recover in the following. Important to note is that the
person scale sn and shape βn are unique for each person
and consistent across the entire video sequence.
3.1. Input Modality Regression and Matching
To solve this underconstrained and challenging problem,
our idea is to unite the strength of all data modalities, which
recent state-of-the-art Computer Vision methods provide, in
a single algorithm. More precisely, we leverage data-driven
priors in the form of four off-the-shelf methods for each
frame of the input video sequence, as shown in Figure 2.
First, we obtain normalized disparity maps ˆdt from
the state-of-the-art DPT model [41], which are then post-
processed to enhance sharpness [58]. Note that these maps
only encode relative and normalized depth and they are not
consistent across frames, which becomes visible in the form
of depth jitter.
Second, 2D pose tracking is obtained by AlphaPose [11],
which coherently detects and tracks 2D joint positions
ˆP2d
t,n ∈RJ×2 in image space and over time. Although this
method is very robust due to training on large scale data, it
falls short in predicting 3D.
Third, we predict the body shape βt,n and joint angles
ˆθt,n for each person in each frame using ROMP [51]. Since
ROMP predicts varying shapes for a single person across
time, we average the predictions over the entire sequence to
obtain a temporally consistent body shape. Thus, the ver-
tices (Equation 2) are now only a function of the pose θt,n,
translation Γt,n, and scale sn, which will be important in
the next section. Moreover, to match the 2D AlphaPose
and the SMPL detections, we leverage ROMPs projection
model, compute the average Euclidean distance in image
space, and pair detections with the lowest distance based on
the Hungarian matching. It is worth mentioning that ROMP
cannot account for out-of-distribution body sizes, e.g. small
kids, neither it can predict the absolute 3D position of the
humans with respect to the scene.
Fourth, we also leverage human segmentation masks,
referred to as Ωt,n ∈RH×W , which are obtained from
Mask2Former [8]. Similarly, if we consider all the remain-
ing pixels for frame t that do not belong to a person mask,
we can also obtain a per-frame background segmentation
mask Bt ∈RH×W . To ensure that the 2D AlphaPose de-
tections, the SMPL detections, and the foreground masks
have a consistent person ID, we read the pixel values of
the segmented masks at the 2D joint detections for each de-
tected skeleton and apply a max-voting to retrieve the ID of
the person.
In summary, the inputs to our algorithm now are:
• ˆdt: Normalized disparity maps
• ˆP2d
t,n: 2D joint predictions
• ˆθt,n, ˆβn: Pose angle and shape estimates
• Ωt,n, Bt: Human and background segmentations
Note that none of these predictions individually or by a triv-
ial combination is discriminative enough to fully describe
the entire scene, i.e. absolute 3D position, pose, and scale
of the humans in the scene. Next, we demonstrate how our
proposed method solves this problem.
3.2. Scene Scale and Depth Disambiguation
In the ﬁrst part or our optimization process we focus on
jointly obtaining the joint angles θt,n, shape parameters βn,
global translation Γt,n, and scale sn of each person. Impor-
tantly, this step is performed jointly for the entire sequence,
where the global reference is in the static camera. How-
ever, estimating the height of a person and the scale given
only a single RGB video is, by itself, an ill-posed problem
as variations in scale can be compensated by a translation
along the depth and vice versa. As a result, inﬁnitely many
scale/translation combinations can lead to the same 2D im-
age projections.
So far, we only considered individual humans without
looking at the surrounding scene, although the scene itself
can provide an important prior that helps to solve the above
problem. Therefore, we leverage recent advances in monoc-
ular depth estimation [41], which regress per-pixel normal-
ized disparity maps ˆdt. It encodes the relative depth of each
person in the scene, but obtaining the absolute depth val-
ues solely from ˆdt is also an ill-posed problem, and fur-
ther these predictions are not consistent across frames. The
question remains, how the absolute scene depth or equiva-
lently the human scales and translations can be recovered.
Our idea is to set the two entities, i.e., the scene and the
humans, into a relation such that they constrain each other
in an absolute 3D space. While the humans can already
be represented in absolute space by means of their global
translation Γt,n and scale sn, we also require a per-frame
5

conversion of temporally inconsistent normalized disparity
maps to absolute depth maps, which can be deﬁned as
˜Dt =
zfar,tznear,t
ˆdt(zfar,t −znear,t) + znear,t
(3)
where znear,t and zfar,t are the near and far depth values,
respectively. Intuitively, this operation shifts and scales the
normalized disparity maps to convert them to absolute depth
values. Importantly, these near and far values are optimized
per-frame to compensate for the temporal inconsistencies in
the disparity maps.
Once both humans and the scene can be represented in
absolute 3D space, we now relate them to each other by
jointly solving for κt,n ∈{znear,t, zfar,t, θt,n, βn, Γt,n, sn}
by minimizing the energy
arg min
∀t∈{1,...,T },∀n∈{1,...,N}:κt,n
EI,
with
(4)
EI = Edepth + E2d + Esmpl + Ereg,
(5)
which is jointly optimized over the entire sequence. In par-
ticular, our energy is composed of a depth term Edepth, a
2D image evidence term E2d, a joint angle and shape term
Esmpl, and additional regularization terms Ereg. In the fol-
lowing, we explain each term in more detail.
3.2.1
Depth Consistency Energy
Most importantly, to ensure a coherent depth between t

## experiments
In this section, we present an empirical evaluation of our
method. We ﬁrst brieﬂy describe the datasets and metrics
used in our experiments in Sections 4.1 and 4.2, followed
by the implementation details in Section 4.3. Next, we com-
pare our approach with the most related works to ours in
Section 4.4. In Section 4.5, we perform a thorough abla-
tion study of the main components of our method and show
additional qualitative results in Section 4.6.
4.1. Datasets
MuPoTs-3D [35] is a test dataset composed of 20 video
sequences with multiple people, including different types of
cameras in indoor and outdoor environments. We followed
the evaluation protocol from [35] in our experiments. This
dataset is especially challenging due to the large amount
of interactions between humans and the various types of
scenes. Ground-truth 3D pose annotations are provided in
absolute coordinates.
CMU Panoptic [21] is a dataset recorded in the Panoptic
Studio with multiple people. As in preliminary work [20,
65], we use this dataset for evaluation considering the
sequences haggling1, ultimatum1, and pizza1,
which are performed by several adults.
In addition to the previous datasets, we also evaluated
our method quantitatively on Internet videos considering
challenging cases with multiple people of different sizes,
including adults and children.
4.2. Metrics
MRPE and AP. We quantitatively evaluate the predic-
tion of the absolute 3D location of a human using the widely
adopted mean root position error (MRPE), in millimeters,
and the average precision of the human root joint (AProot
25 )
[37], considering the standard threshold of 25 cm.
3DPCK. The quality of the articulated 3D pose prediction
is measured using root-relative 3DPCK [33], with the stan-
dard threshold of 15 cm. The 3DPCK metric enables mea-
suring the correctness of the pose, independently of the pre-
diction of the absolute 3D location of the human.
MPJPE. For a fair comparison with previous methods,
we also report root-relative mean per-joint position error
(MPJPE) in the CMU Panoptic dataset.
Jitter.
Finally, since we are targeting high-quality tem-
poral predictions in 3D coordinates, we also evaluate the
amount of jitter of our estimations, which is a critical in-
dicator for many downstream applications. For this eval-
uation, we adapted the temporal smoothness error esmooth
from [47] to evaluate the jitter in 3D coordinates.
4.3. Implementation Details
Our method is implemented in PyTorch [38] using Py-
Torch3D [43] for the rasterization (6) and silhouette render-
ing (10). The camera intrinsics are used in the 3D joint pro-
jection (9), rasterization (6), and rendering (10) parts, and
can be obtained from video metadata if not given. We ap-
ply the RMSprop [17] optimizer with the parameters α and
momentum set to 0.5 and 0.9, respectively, for all experi-
ments. In the optimization process, we initially minimize
the ﬁrst part (4) only for 30 iterations, then perform the full
optimization (15) for more 200 iterations. We use a learn-
ing rate initially set to 0.01 and exponentially decaying with
factor 0.99.
The weights λ(.) were empirically deﬁned to
balance the magnitude of the individual energy terms, and
ﬁxed in the method in all experiments, except when men-
tioned otherwise (ablation in Section 4.5).
The values
were deﬁned as λdepth = λspeed = 0.05, λsilhouette = 0.1,
λsmpl = λtemporal = 0.002, λscale = 0.0001, λcontact =
0.001, and λslip = 0.01.
For numerical stability, we con-
strain the variables sn, znear,t, and zfar,t to be non-zero and
positive. Both human and background segmentation masks
were post-processed with morphological erosion and dila-
tion ﬁlters of size 3×3 and 5×5, respectively. For the sake
of GPU memory efﬁciency, we use mini batches of ten im-
ages in the depth and silhouette losses. Our experiments run
on a workstation with one Nvidia Titan V GPU with 12 GB
of memory.
8

Table 1. Comparison of our method with previous approaches on
MuPoTs-3D in the MRPE (lower is better), AProot
25 , and 3DPCK
metrics (higher is better), considering the global 3D pose and the
normalized (univ) ground truths. Our approach is superior to all
compared methods on the absolute metrics (MRPE, AProot
25
and
3DPCK3d), i.e., the most expressive ones for 3D human motion
capture. “†” evaluated on samples with IK only; “∗” evaluated on
root-relative predictions without IK; “‡” results only possible with
an additional 2D ﬁtting stage, implemented as our baseline.

## related_work
3D human motion capture is an active research area, and
many works have been proposed in the past [6, 23, 31, 32,
36, 49, 50, 53, 55, 70]. Since we target a monocular setting,
we do not review multi-view- and depth-based methods. In-
stead, we review previous works that are most related to our
method.
2.1. 3D Human Pose Estimation
2.1.1
Single Person Pose Estimation
Estimating the human body pose in 3D from a single image
is a challenging problem that has been successfully handled
by learning a human body prior from MoCap data [19]. To
simplify the problem, previous methods usually predict 3D
2

coordinates relative to the root joint, assuming a normalized
human body size [33] and a ﬁxed bounding box around the
person in 3D space [36, 40]. However, when multiple peo-
ple are interacting with the environment, normalized and
root-relative predictions are not enough to disambiguate the
position and scale of individual persons in the scene. In
addition, directly estimating the 3D joint coordinates could
result in implausible poses, which is a problem that can be
mitigated by estimating joint angles instead [71].
Several works focus on estimating the full human mesh
deformation from videos [13,14,61], assuming that the ac-
tor mesh is provided in advance. Other works for single
human estimation [22,24,39] rely on SMPL [30] as a proxy
shape. Reconstructing shape proxies along with sparse 3D
skeletons is desirable in many scenarios (e.g., they can be
used for body parts segmentation). Moreover, SMPL serves
as a statistical prior on human body shapes and enables ad-
ditional supervisory terms such as human silhouette over-
lays in 2D, which can result in higher accuracy [39].
2.1.2
Multiple Person Pose Estimation
Estimating positions of each person w.r.t. the others is cru-
cial in multi-human pose estimation. Nonetheless, most of
the existing multi-person methods are by design perform-
ing root-relative predictions [1, 45, 46, 51]. Several tech-
niques predict translations of each person in the camera
reference frame. They either optimize the translation by
projecting and ﬁtting the estimated 3D poses into the im-
age plane [9, 34, 66] or by directly regressing the distance
of the root joint to the camera with a deep neural net-
work [28, 37, 56, 69]. The ﬁrst case can be more robust
to different camera setups, but is limited by the unknown
height of each person in the scene. The second strategy is
highly dependent on the training data and may not gener-
alize to camera conﬁgurations not present in the training.
Others explore human priors [26] to estimate a global tra-
jectory [64], but still fail to recover the body size.
Recent methods performing human depth estimation are
focused on penalizing depth ordering of multiple humans.
For instance, Jiang et al. [20] uses instance segmentation
masks to penalize depth inversion and Sun et al. [52] pro-
poses to infer the depth of each person based on an imagi-
nary bird’s-eye-view representation and to estimate the per-
son age as a proxy for the scale. Other approaches pre-
dict the relative depth among multiple persons by inferring
some scene properties. A possible scene simpliﬁcation is to
assume a parametric planar ﬂoor, in such a way that each
prediction can be positioned to respect a plausible human-
ﬂoor contact [54, 65]. The common limitation of such ap-
proaches is the dependency on a simpliﬁed ﬂoor represen-
tation, which is often not the case in real applications. Con-
trarily, we estimate a scene point cloud that can represent a
arbitrary ground ﬂoor.
The works from Jiang et al. [20] and Ugrinovic et al. [54]
are the most closely related to ours. Similarly to the for-
mer, we also render the estimated human models into the
image plane to provide additional supervision in the depth
dimension, and, related to the latter, we also disambiguate
body size and depth for each person by constraining pre-
dictions with an estimated scene geometry. But differently
from [20], that does not take the scene into account, and
from [54], that relies on a simpliﬁed scene representation
and operates in a single frame, our method represents the
scene as a frustum point cloud and performs optimization
over the entire video sequence. In our work, we also rely
on a human body proxy model [30] to estimate joint angles
and we propose a new formulation to optimize the position
of the humans and the scene in a joint optimization process.
Therefore, our model improves the prediction of human po-
sitions by relying on an estimated proxy scene geometry
that does not depend on a simpliﬁed parametric model.
2.2. Scene-aware Motion Capture
Predicting and understanding how humans interact in
3D has recently gained a lot of attention. Several current
methods focus on positioning humans in a pre-scanned 3D
scene [12, 15, 18] and on simultaneous estimation of hu-
man poses and objects humans interact with [7, 59, 62]. A
different setup assumes an RGB-D sensor [68] or a mov-
ing camera [16, 25, 29, 67] that facilitates estimating the
scene geometry. Recent methods integrate physics-based
constraints into monocular 3D human motion capture and
mitigate foot-ﬂoor penetration and other severe artefacts
[47,48]. Yu et al. [63] also support composite scenes in the
parcours and sports scenarios. Although there is a growing
interest in investigating the interactions of humans and ob-
jects [2, 10], 3D motion capture of multiple humans with
environmental awareness from a single monocular camera
remains underexplored.
Determining the absolute human scale in 3D is an ill-
posed and challenging task. Bieler et al. [3] estimate the
height of a single person from monocular videos by observ-
ing jumping people. Dabral et al. [10] require an interaction
with an object undergoing a free ﬂight to resolve the abso-
lute scene scale. Both methods assume motion inﬂuenced
by the universal law of gravity near the surface of Earth,
which allows them to relate the time spent in the air or the
form of the observed trajectory with absolute distances in
the metric units.
The downside is that jumping humans
or ﬂying objects are restrictive assumptions. In contrast,
we use a human body and 3D scene priors in 3D multi-
human motion estimation and do not make strong assump-
tions about the observed human motions.
3

Input: Video Sequence
Disparity
Map
2D Pose
SMPL 
Parameters
Instance
Segmentation
Image Modality Regression and Matching (Section 3.1)
Pre-processing and Matching
Scene Scale and Depth Disambiguation (Section 3.2)
Output: Humans and Scene
Aggregation
(median)
Optimization Framework
Space-time Coherent Pose Optimization (Section 3.3)
Eq. 3
SMPL
Estimates
Disparity
Maps
Depth
Maps
Segmentation
and 2D Pose
Renderer
Background Scene
Point Cloud
For each
person

## conclusion
Our method achieves low reconstruction errors, because
it can successfully leverage multi-modal inputs to disam-
biguate the relative depths between humans and human
scales better than previous works. Moreover, our results
evince signiﬁcantly less jitter and foot-ﬂoor penetrations
than the evaluated baselines for multi-human 3D pose esti-
mation and the ablative study conﬁrms that all components
of the method contribute to the ﬁnal accuracy. We have
demonstrated that the recovered 3D human motions can be
applied for virtual character animation, as one potential ap-
plication among the many others.
Limitations and Possible Extensions.
Although our
method outperforms competing methods and makes a step
forward in monocular multi-human 3D motion capture, it
has several limitations caused by the severe ill-posedness of
our monocular setting. All these limitations open possibili-
ties for future extensions and follow-up works as described
in the following.
First, our approach relies on multiple inputs from pre-
trained models (depth maps and 2D body joints) and, there-
fore, could also be negatively affected by the output of
those methods; for example if the estimated depth maps
contain signiﬁcant artefacts (e.g., when obtained on our-of-
distribution environments). On the other hand, this implies
that the performance of our approach has the potential to
keep increasing in the future with the progress in related
ﬁelds (cf. Table 5).
Our method also requires that people are entirely visi-
ble in most of the frames and move in the scene. Other-
wise, the setting becomes degenerate, and we do not get
enough cues for accurate reconstruction. Even though we
mitigate artefacts that appear as violations of physical laws
by geometric terms, some minor ones of this type remain.
Further improvements can be attained by methods explic-
itly modelling physical laws as in single-human 3D motion
capture [47,48,60].
Moreover, while the static camera assumption is practi-
cal, it is also very challenging, and a moving camera could
provide additional 3D reconstruction cues. Finally, the pro-
posed approach is an optimization method that can efﬁ-
ciently process an entire video sequence and extract rele-
vant information about the scene from all frames globally.
However, due to this characteristic, the method in its current
version does not allow real-time applications.