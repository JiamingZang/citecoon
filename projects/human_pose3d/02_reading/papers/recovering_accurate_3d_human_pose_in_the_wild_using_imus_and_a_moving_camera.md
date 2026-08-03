# Recovering Accurate 3D Human Pose in the Wild Using IMUs and a Moving Camera

> 2018 · id: W2895748257 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Recovering Accurate 3D Human Pose in The
Wild Using IMUs and a Moving Camera
Timo von Marcard1, Roberto Henschel1, Michael J. Black2, Bodo Rosenhahn1,
and Gerard Pons-Moll3
1 Leibniz Universit¨at Hannover, Germany
2 MPI for Intelligent Systems, T¨ubingen, Germany
3 MPI for Informatics, Saarland Informatics Campus, Germany
{marcard,henschel,rosenhahn}@tnt.uni-hannover.de, black@tue.mpg.de,
gpons@mpi-inf.mpg.de
Abstract. In this work, we propose a method that combines a single
hand-held camera and a set of Inertial Measurement Units (IMUs) at-
tached at the body limbs to estimate accurate 3D poses in the wild.
This poses many new challenges: the moving camera, heading drift, clut-
tered background, occlusions and many people visible in the video. We
associate 2D pose detections in each image to the corresponding IMU-
equipped persons by solving a novel graph based optimization problem
that forces 3D to 2D coherency within a frame and across long range
frames. Given associations, we jointly optimize the pose of a statisti-
cal body model, the camera pose and heading drift using a continu-
ous optimization framework. We validated our method on the TotalCap-
ture dataset, which provides video and IMU synchronized with ground
truth. We obtain an accuracy of 26mm, which makes it accurate enough
to serve as a benchmark for image-based 3D pose estimation in the
wild. Using our method, we recorded 3D Poses in the Wild (3DPW ),
a new dataset consisting of more than 51, 000 frames with accurate
3D pose in challenging sequences, including walking in the city, going
up-stairs, having coﬀee or taking the bus. We make the reconstructed
3D poses, video, IMU and 3D models available for research purposes
at http://virtualhumans.mpi-inf.mpg.de/3DPW.
Keywords: Human Pose, Video, IMUs, Sensor Fusion, 2D to 3D, People
Tracking, 3D Pose Dataset
1
Introduction
This paper addresses two inter-related goals. First, we propose a method capable
of accurately reconstructing 3D human pose in outdoor scenes, with multiple
people interacting with the environment, see Fig. 1. Our method combines data
coming from IMUs (attached at the person’s limbs) with video obtained from
a hand-held phone camera. This allows us to achieve the second goal, which is
collecting the ﬁrst dataset with accurate 3D reconstructions in the wild. Since

2
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
Fig. 1. We propose Video Inertial Poser (VIP), which enables accurate 3D human
pose capture in natural environments. VIP combines video obtained from a hand-
held smartphone camera with data coming from body-worn inertial measurement units
(IMUs). With VIP we collected 3D Poses in the Wild, a new dataset of accurate 3D
human poses in natural video, containing variations in person identity, activity and
clothing.
our system works with a moving camera, we can record people in their everyday
environments, for example, walking in the city, having coﬀee or taking the bus.
3D human pose estimation from un-constrained single images and videos has
been a longstanding goal in computer vision. Recently, there has been a signif-
icant progress, particularly in 2D human pose estimation [23, 4]. This progress
has been possible thanks to the availability of large training datasets and bench-
marks to compare research methods. While obtaining manual 2D pose anno-
tations in the wild is fairly easy, collecting 3D pose annotations manually is
almost impossible. This is probably the main reason there exist very limited
datasets with accurate 3D pose in the wild. Datasets such as HumanEva [32]
and H3.6M [8] have facilitated progress in the ﬁeld by providing ground truth
3D poses obtained using a marker-based motion capture system synchronized
with video. These datasets, while useful and necessary, are restricted to indoor
scenarios with static backgrounds, little variation in clothing and no environmen-
tal occlusions. As a result, evaluations of 3D human pose estimation methods
in challenging images have been made mainly qualitatively, so far. There exist
several options to record humans in outdoor scenes, none of which is satisfactory.
Marker-based capture outdoors is limited. Depth sensors like Kinect do not work
under strong illumination and can only capture objects near the camera. Using
multiple cameras as in [21] requires time consuming set-up and calibration. Most
importantly, the ﬁxed recording volume severely limits the kind of activities that
can be captured.
IMU-based systems hold promise because they are not bound to a ﬁxed space
since they are worn by the person. In practice, however, accuracy is limited
by a number of factors. Inaccuracies in the initial pose introduce sensor-to-
bone misalignments. In addition, during continuous operation, IMUs suﬀer from
heading drift, see Fig. 2. This means, that after some time, each IMU does

Accurate 3D Human Pose Using IMUs and a Moving Camera
3
Fig. 2. For accurate motion capture in the wild we have to solve several challenges:
IMU heading drift has accumulated after a longer recording session and the obtained
3D pose is completely oﬀ(left image pair). In order to estimate the heading drift,
we combine IMU data and 2D poses detected in the camera view. This requires the
association of 2D poses to the person wearing IMUs, which is diﬃcult when several
people are in the scene (middle image). Also, 2D pose candidates might be inaccurate
and should be automatically rejected during the assignment step (right image pair).
not measure relative to the same world coordinate frame. Rather, each sensor
provides readings relative to independent coordinate frames that slowly drift
away from the world frame. Furthermore, global position can not be accurately
obtained due to positional drift. Moreover, IMU systems do not provide 3D pose
synchronized and aligned with image data.
Therefore, we propose a new method, called Video Inertial Poser (VIP), that
jointly estimates the pose of people in the scene by using 6 to 17 IMUs attached
at the body limbs and a single hand-held moving phone camera. Using IMUs
makes the task less ambiguous but many challenges remain. First, the persons
need to be detected in the video and associated with the IMU data, see Fig. 2.
Second, IMUs are inaccurate due to heading drift. Third, the estimated 3D poses
need to align with the images of the moving camera. Furthermore, the scenes
we tackle in this work include complete occlusions, multiple people, tracked
persons falling out of the camera view and camera motion. To address these
diﬃculties, we deﬁne a novel graph-based association method, and a continuous
pose optimization scheme that integrates the measurements from all frames in
the sequence. To deal with noise and incomplete data, we exploit SMPL [14],
which incorporates anthropometric and kinematic constraints.
Speciﬁcally, our approach has three steps: initialization, association and data
fusion. During initialization, we compute initial 3D poses by ﬁtting SMPL to
the IMU orientations. The association step automatically associates the 3D
poses with 2D person detections for the full sequence by solving a single binary
quadratic optimization problem. Given those associations, in the data fusion
step, we deﬁne an objective function and jointly optimize for the 3D poses of the
full sequence, the per-sensor heading errors, the camera pose and translation.
Speciﬁcally, the objective is minimized when (i) the model orientation and ac-
celeration is close to the IMU readings and (ii) the projected 3D joints of SMPL
are close to 2D CNN detections [4] in the image. To further improve results, we
repeat association and joint optimization once.

4
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
With VIP we can accurately estimate 3D human poses in challenging natural
scenes. To validate the accuracy of VIP, we use the recently released 3D dataset
Total Capture [39] because it provides video synchronized with IMU data. VIP
obtains an average 3D pose error of 26mm, which makes it accurate enough to
benchmark methods that tackle in-the-wild data. Using VIP we created 3D Poses
in the Wild (3DPW): a dataset consisting of hand-held video with ground-truth
3D human pose and shape in natural videos.
We make 3DPW publicly available for research purposes, including 60 video
sequences (51, 000 frames or 1700 seconds of video captured with a phone at
30Hz), IMU data, 3D scans and 3D people models with 18 clothing variations,
and the accurate 3D pose reconstruction results of VIP in all sequences. We
anticipate that the dataset will stimulate novel research by providing a platform
to quantitatively evaluate and compare methods for 3D human pose estimation.
2
Related Work
Pose Estimation using IMUs. There exist commercial solutions for MoCap
with IMUs. The approach of [30] integrates 17 IMUs in a Kalman Filter to
estimate pose. The seminal work of [41] uses a custom made suit to capture
pose in everyday surroundings. These approaches require many sensors and do
not align the reconstructions with video; therefore they suﬀer from drift. The
approach of [42] ﬁts the SMPL body model to 5-6 IMUs over a full sequence,
obtaining realistic results. The method, however, is applied to only 1 person at
a time and the motion is not aligned with video. To compensate for drift, 4-8
cameras and 5 IMUs are combined in [17, 25]. Using particle-based optimization,
in [24] they use 4 cameras and IMUs to sample from a manifold of constrained
poses. Other works combine depth data with IMUs [6, 47]. In [39] a CNN-based
approach fuses information from 8 camera views and IMU data to regress pose.
Since these approaches also use multiple static cameras, recordings are restricted
to a ﬁxed recording volume. A recent approach [16] also combines IMUs and 2D
poses detected in one or two cameras but expects only a single person visible in
the cameras and does not account for heading drift.
3D Pose Datasets. The most commonly used datasets for 3D human pose
evaluation are HumanEva [32] and H3.6M [8], which provide synchronized video
with MoCap. These datasets however are limited to indoor scenes, static back-
grounds and limited clothing and activity variation. Recently, a dataset of single
people, including outdoor scenes, has been introduced [19]. The approach uses
commercial marker-less motion capture from multiple cameras (the accuracy of
the marker-less MoCap software used is not reported). The sequences show vari-
ation in clothing, but again, since it uses a multi-camera setup, the activities
are restricted to a ﬁxed recording volume. Another recent dataset is TotalCap-
ture [39], which features synchronized video, marker-based ground-truth poses
and IMUs. In order to collect 3D poses in the wild, in [11] they ask users to pick
“acceptable” results obtained using an automatic 3D pose estimation method.
The problem is that it is diﬃcult to judge a correct pose visually and it is not

Accurate 3D Human Pose Using IMUs and a Moving Camera
5
clear how accurate automatic methods are with in-the-wild images. We do not
see our proposed dataset as an alternative to existing datasets; rather 3DPW
complements existing ones with new, more challenging, sequences.
3D Human Pose. Several works lift 2D detections to 3D using learning
or geometric reasoning [18, 29, 35, 9, 26, 49, 33, 48, 44, 34, 13, 43, 45]. These works
aim at recovering the missing depth dimension in single-person images, whereas
we focus on directly associating the 3D to the 2D poses in cluttered scenes. For
multiple people, the work [1] infers the 3D poses using a tracking formulation
that is based on short tracklets of 2D body parts. Recently 2D annotations
have been leveraged to train networks for the task of 3D pose estimation [21,
28, 36, 38, 50]. Such works typically predict only stick ﬁgures or bone skeletons.
Some approaches directly predict the parameters of a body model (SMPL) from
a single image using 2D supervision [10, 22, 40]. Closer to our method are the
works [2, 11], which ﬁt SMPL [14] to 2D detections. The optimization problem
we solve, even though it integrates more sensors, is much more involved. Very
few approaches tackle multiple-person 3D pose estimation [31, 20]. 3DPW allows
a quantitative evaluation of all these approaches for in-the-wild images.
3
Background
SMPL Body Model. We utilize the Skinned Multi-Person Linear (SMPL)
body model [14], which is a statistical body model parameterized by identity-
dependent shape parameters and the skeleton pose. We optimize the shape pa-
rameters to the person to be tracked by ﬁtting SMPL to a 3D scan. Holding
shape ﬁxed, our aim is to recover the pose θ ∈R75, consisting of 6 parameters
for global translation and rotation, and 23 relative rotations represented by axis-
angle for each joint. We use the standard forward kinematics to map pose θ to
the rigid transformation GGB(θ) : R75 →SE(3) of bone B. The bone trans-
formation comprises the rotation and translation GGB = {RGB, tGB} to map
from the local bone coordinate frame F B to the global SMPL frame F G.
Coordinate Frames. Ultimately, we want to ﬁnd the pose θ that produces
bone orientations close to the IMU readings. IMUs measure the orientation of
the local coordinate frame F S (of the sensor box) relative to a global coordinate
frame F I. However, this frame F I is diﬀerent from the coordinate frame F G
of SMPL, see Fig. 5. The oﬀset GGI : F I →F G between coordinate frames
is typically assumed constant, and is calibrated at the beginning of a recording
session – but that is not enough. We also need to know the oﬀset RBS from
the sensor to the SMPL bone where it is placed. The SMPL bone orientation
RGB(θ0) can be obtained in the ﬁrst frame assuming a known pose θ0. Using this
bone orientation RGB(θ0) and the raw IMU reading RIS(0) in the ﬁrst frame,
we can trivially ﬁnd the oﬀset relating them as
RBS =
 RGB(θ0)
−1 RGIRIS(0)
(1)
where the raw IMU reading RIS(0) needs to be mapped to the SMPL frame
ﬁrst using RGI. We assume that sensors do not move relative to the bones, and

6
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
Model+IMUs
Video
2D Poses
3D Poses
Assignment
Joint Optimization
Θ
V
ˆΘ
Ψ
Γ
Fig. 3. Method overview: By ﬁtting the SMPL body model to the measured IMU
orientations we obtain initial 3D poses ˆΘ. Given all 2D poses V detected in the images
we search for a globally consistent assignment of 2D to 3D poses. We jointly optimize
camera poses Ψ, heading angles Γ and 3D poses Θ with respect to associated IMU
and image data. In a second iteration we feed back camera poses and heading angles
which provides additional information further improving the assignment and tracking
results.
hence compute RBS from the initial pose θ0 and IMU orientations in the ﬁrst
frame.
Heading Drift. Unfortunately, the orientation measurements of the IMUs
are deteriorated by magnetic disturbances, which introduce a time-varying ro-
tational oﬀset to GGI, also commonly known as heading error or heading drift.
This drift (GI′I : F I →F I′) shifts the original global inertial frame F I to a
disturbed inertial frame F I′. What is even worse, the drift is diﬀerent for every
sensor. While most previous works ignore heading drift or treat it as noise, we
model it explicitly and recover it as part of the optimization. Concretely, we
model it as a one-parameter rotation R(γ) ∈SO(3) about the vertical axis,
where γ is the rotation angle. The collection of all angles, one per IMU sensor,
is denoted as Γ. Since the heading error commonly varies slowly, we assume it is
constant during a single tracking sequence. Recovering heading orientation was
crucial in order to be able to perform long recordings without time-consuming
re-calibration.
4
Video Inertial Poser (VIP)
In order to perform accurate 3D human motion capture with hand-held video
and IMUs we perform three subsequent steps: initialization, pose candidate as-
sociation and video-inertial fusion. Fig. 3 provides an overview of the pipeline
and we describe each step in more detail in the following.
4.1
Initialization
We obtain initial 3D poses by ﬁtting the SMPL bone orientations to the measured
IMU orientations. For an IMU, the measured bone orientation ˆRGB is given by
ˆRGB = RGI′RI′I(γ)RIS  RBS−1 ,
(2)

Accurate 3D Human Pose Using IMUs and a Moving Camera
7
Fig. 4. Every 2D pose represents a node in the
graph which can be assigned to a 3D pose cor-
responding to person 1 or 2 (represented by col-
ors orange and blue). The graph has intra-frame
edges (shown in black) activated if two nodes
are assigned in a single frame and inter-frame
edges (shown in blue and orange) activated for
the same person across multiple frames.
F G
F I′
F I
F B
F S
GBS
GIS
GI′I
GGI′
GGB
GGI
Fig. 5. Coordinate frames: Global
tracking frame F G, global inertial
frame F I, shifted inertial frame
F I′, bone coordinate frame F B
and IMU sensor coordinate frame
F S.
where RBS represents the constant bone to sensor oﬀset (Eq. (1)), and the
concatenation of RGI′, RI′I and RIS describes the rotational map from sensor
to global frame, see Fig. 5. We deﬁne the rotational discrepancy between actual
bone orientation RGB(θ) and measured bone orientation ˆRGB as
erot(θ) = log

RGB(θ)

ˆRGB−1∨
,
(3)
where the log-operation recovers the skew-symmetric matrix from the relative
rotation between RGB(θ) and ˆRGB, and the ∨-operator extracts the correspond-
ing axis-angle parameters. We ﬁnd the 3D initial poses at frame t that minimize
the sum of discrepancies for all IMUs
θ∗
t = arg min
θ
1
Ns
Ns
X
s=1
||erot
s,t (θt)||2 + wpriorEprior(θt),
(4)
where Eprior(θ) is a pose prior weighted by wprior. Eprior(θ) is chosen as deﬁned
in [42], enforcing θ to remain close to a multivariate Gaussian distribution of
model poses and to stay within joint limits. During the ﬁrst iteration, we have
no information about the heading angles γ. To initialize them, we use the IMU
placement as a proxy to know how local sensor axes are aligned with respect to
the body. This gives us a rough estimate of the sensor to bone oﬀset ˆRBS, which
we use to compute initial heading angles by solving Eq. (1) for γ.
In the following, we will refer to this tracking approach simply as the inertial
tracker (IT), which outputs initial 3D pose candidates θ∗
t,l for every tracked
person l. Such initial 3D poses need to be associated with 2D detections in the
video in order to eﬀectively fuse the data – this poses a challenging assignment
problem.

8
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
4.2
Pose Candidate Assignment
Using the CNN method of Cao et al. [4], we obtain 2D pose detections v, which
comprise the image coordinates of Njoints = 18 landmarks along with corre-
sponding conﬁdence scores. In order to associate each 2D pose v to a 3D pose
candidate, we create an undirected weighted graph G = (V, E, c), with V com-
prising all detected 2D poses in a recording sequence. An assignment hypothesis,
denoted as H(l, v) = (θl
t, v), links the 3D pose θl
t of person l ∈{1, . . . , P} to the
2D pose v ∈V in the same frame t. We introduce indicator variables xl
v, which
take value 1 if hypothesis H(l, v) is selected, and 0 otherwise. The basic idea is
to assign costs to each hypothesis, and select the assignments for the sequence
that minimize the total costs. We cast the selection problem as a graph labeling
problem by minimizing the following objective
arg min
x∈F∩{0,1}|V|P
X
v∈V
l∈{1,...,P }
cl
vxl
v +
X
{v,v′}∈E
l,l′∈{1,...,P }
cl,l′
v,v′ xl
vxl′
v′,
(5)
where the feasibility set F is subject to:
(a)
P
X
l=1
xl
v ≤1
∀v ∈V;
(b)
X
v∈Vt
xl
v ≤1
∀t, ∀l ∈{1, . . . , P}.
(6)
The edge set E contains all pairs of 2D poses {v, v′} that are considered for the
assignment decision. Eq. (6)(a) ensures that a 2D pose v is assigned to at most 1
person, and Eq. (6)(b) ensures that each person is assigned to at most one of the
2D pose detections v ∈Vt ⊂V in frame t. The objective in (5) consists of unary
costs cl
v measuring 2D to 3D consistency, and pairwise costs cl,l′
v,v′ measuring
consistency across diﬀerent hypothesis. Our formulation automatically outputs
a globally consistent assignment and does not require manual initialization.
Next we describe the unaries and pairwise potentials – speciﬁcally, we intro-
duce consistency features which are mapped to the costs cl
v, cl,l′
v,v′ of the objective
in (5) via logistic regression. Details about the training process are described in
Section 5.1. Fig. 4 visualizes the graph for two example frames and also illustrates
the corresponding labeling solution.
Unary Costs. To measure 2D to 3D consistency of a hypothesis H := H(l, v),
we obtain a hypothesis camera MH by minimizing the re-projection error be-
tween 3D landmarks of θl
t and the 2D detected ones v. The per landmark re-
projection error, denoted by eimg,k(H, MH), is weighted by the conﬁdence scores
wk. The consistency is then measured as the average of all weighted residuals
eimg,k(H, MH), denoted by eimg(H, MH). This measure depends heavily on the
distance to the camera. To balance it, we scale it by the average 3D joint distance
to the camera center ecam(MH) and obtain the feature:
fun(H) = eimg(H, MH)ecam(H, MH).
(7)

Accurate 3D Human Pose Using IMUs and a Moving Camera
9
Pairwise Costs. We deﬁne features to measure the consistency of two hypoth-
esis H = (θl
t, v) and H′ = (θl′
t′, v′) in frames t and t′. In particular, two kinds of
edges connect hypothesis: (a) inter-frame, and (b) intra-frame.
a) Inter-frame: Consider two hypothesis H, H′ corresponding to the same person
and separated by fewer than 30 frames. Then, the respective root joint position
r(θl
t) and orientation R(θl
t) in camera hypothesis (MH) coordinates should not
vary too much. This variation depends on the temporal distance |t −t′|. Conse-
quently, we introduce the following features
ftrans(H, H′) = ||MHr(θl
t) −MH′r(θl′
t′)||2,
(8)
fori(H, H′) =
log

(RHR(θl
t))−1(RH′R(θl′
t′))
∨
2
,
(9)
ftime(H, H′) = ||t −t′||2,
(10)
where ftrans and fori measure root joint translation and orientation consistency,
and ftime is a feature to accommodate for temporal distance. Here, RH is the
rotational part of MH, and frot computes the geodesic distance between R(θl
t)
and R(θl′
t′), similar to Eq. (3).
b) Intra-frame: Now consider two hypothesis H, H′ for diﬀerent persons in the
same frame. The resulting camera hypothesis centers should be consistent. To
measure coherency, we compute a meta-camera hypothesis MH by minimizing
the re-projection error of both hypothesis at the same time. Then the feature
fintra(H, H) = ||c(θl
t, MH) −c(θl
t, MH)||2
(11)
measures the camera c(θl
t, MH) to meta-camera center c(θl
t, MH) diﬀerence.
Accordingly, we also use the feature fintra(H′, H) for intra-frame edges.
Graph Optimization. Although the presented graph labeling problem in (5)
is NP-Hard, it can be solved eﬃciently in practice [7, 12]. We use the binary LP
solver Gurobi [5] by applying it to the linearized formulation of (5), where we re-
place each product xl
vxl′
v′ by a binary auxiliary variable yl,l′
v,v′ and add correspond-
ing constraints such that xl
vxl′
v′ = yl,l′
v,v′ for all v, v′ ∈V, for all l, l′ ∈{1, . . . , P}.
4.3
Video-Inertial Data Fusion
Once the assignment problem is solved we can utilize the associated 2D poses to
jointly optimize model poses, camera poses and heading angles by minimizing
the following energy:
E(Θ, Ψ, Γ) = Eori(Θ, Γ) + waccEacc(Θ, Γ)+
wimgEimg(Θ, Ψ) + wpriorEprior(Θ),
(12)

10
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
where Θ is a vector containing the pose parameters for each actor and frame, Γ
is the vector of IMU heading correction angles and Ψ contains the camera poses
for each frame. Eori(Θ, Γ), Eacc(Θ, Γ) and Eimg(Θ, Ψ) are energy terms related
to IMU orientations, IMU accelerations and image information, respectively.
Eprior(Θ) is an energy term related to pose priors. Finally, every term is weighted
by a corresponding weight w.
Orientation Term The orientation term simply extends Eq. (4) by considering
all frames NT of a sequence according to
Eori(Θ, Γ) =
1
NT Ns
NT
X
t=1
Ns
X
s=1
||erot
s,t (θt, γs)||2.
(13)
This term also includes the camera IMU, where the camera rotation mapping
from camera coordinate system F C to the global coordinate frame F G is given
by the inverse rotational part of the camera pose M.
Acceleration Term The acceleration term enforces consistency of the mea-
sured IMU acceleration and the acceleration of the corresponding model vertex
to which the IMU is attached. The IMU acceleration in world coordinates for
sensor s at time t is given by
aG
s,t(γ) = RGI′RI′I(γs)RISaS
s,t −gG,
(14)
where gG is gravity in global coordinates. The corresponding SMPL vertex ac-
celeration ˆa(θt) is approximated by ﬁnite diﬀerences. Finally, the acceleration
term contains the quadratic norm of the deviation of measured and estimated
acceleration for all NS IMUs over all frames NT :
Eacc(Θ, Γ) =
1
NT NS
NT
X
t=1
NS
X
s=1
||ˆas(θt) −as,t(γs)||2.
(15)
This term also contains the measured acceleration of the camera IMU and the
corresponding acceleration of the camera center in global coordinates.
Image Term The image term simply accumulates the re-projection error over
all Njoints landmarks and all frames NT according to
Eimg(Θ, Ψ) =
1
NT Ncoco
NT
X
t=1
Njoints
X
i=k
wk||eimg,k(θt, Mt)||2,
(16)
where wk is the conﬁdence score associated with a landmark.
Prior Term The prior term is the same as in Eq. (4), now accumulated for all
poses Θ and scaled by the number of poses NΘ.

Accurate 3D Human Pose Using IMUs and a Moving Camera
11
4.4
Optimization
In order to solve the optimization problems related to obtaining initial 3D
poses in Eq. (4), obtaining camera poses to minimize re-projection error and
to jointly optimize all variables in Eq. (12), we apply gradient-based Levenberg-
Marquardt.
5
Results
To validate our approach quantitatively (Section 5.1 and Section 5.2), we use the
recent TotalCapture [39] dataset, which is the only one including IMU data and
video synchronized with ground-truth. In Section 5.3 we then provide details of
the newly recorded 3DPW dataset, demonstrate 3D pose reconstruction of VIP
in challenging scenes, and evaluate the accuracy of automatic 2D to 3D pose
assignment in multiple-person scenes.
5.1
Tracker Parameters
Pose assignment: In the graph G, edges e ∈E are created between any two nodes
that are at most 30 frames apart. The weights mapping from features to costs
are learned using 5 sequences from 3DPW dataset, which have been manually
labeled for this purpose. Given the features f deﬁned in Section 4.2 and learned
weights α from logistic regression, we turn features into costs via c = −⟨f, α⟩,
making the optimization problem (5) probabilistically motivated [37].
Video Inertial Fusion: Diﬀerent weighting parameters in Eq. (4) and Eq. (12)
produce good results as long as they are balanced. However, rather than setting
them by hand, we used Bayesian Optimization [3] in the proposed training set
of TotalCapture (seen subjects). The values found are wacc = 0.2, wimg = 0.0001
and wprior = 0.006 and are kept ﬁxed for all experiments. Note, that these are
very few parameters and therefore, there is very little risk of over-ﬁtting, which
is also reﬂected in the results.
5.2
Tracking Accuracy
We quantitatively evaluate tracking accuracy on the TotalCapture dataset. The
dataset consists of 5 subjects performing several activities such as walking, act-
ing, range of motions and freestyle motions – which are recorded using 8 cal-
ibrated, static RGB-cameras and 13 IMUs attached to head, sternum, waist,
upper arms, lower arms, upper legs, lower legs and feet. Ground-truth poses are
obtained using a marker-baser motion capture system. All data is synchronized
and operates at a framerate of 60Hz. The ground truth poses are provided as
joint positions, which do not contain information about pronation and supina-
tion angles; i.e. rotations about the bone’s long axis. To obtain full degree of
freedom pose, we ﬁt the SMPL model to the raw ground-truth markers using a
method similar to [15].

12
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
Approach
[39]
[16]
IT VIP-2D VIP-Cam VIP-IMU6 VIP-IT VIP
MPJPE
70.0
(62)
55.0
15.1
25.3
39.6
28.2
26.0
MPJAE
-
-
16.9
10.1
12.1
15.3
12.0
12.1
Table 1. Mean Joint Position Error (MPJPE) in mm and Mean Per Joint Angular
Error (MPJAE) in degrees evaluated on TotalCapture.
Error Metrics: We report: Mean Per Joint Position Error (MPJPE) and
Mean Per Joint Angular Error (MPJAE). MPJPE is the average Euclidean dis-
tance between ground-truth and estimated joint positions of hips, knees, ankles,
neck, head, shoulders, elbows and wrists; MPJAE is the average geodesic dis-
tance between ground-truth and estimated joint orientations for hips, knees,
neck, shoulders and elbows. In order to evaluate pose accuracy independently
of absolute camera position and orientation, we align our estimates with the
ground-truth. This is standard practice in existing benchmarks [8]. Thus, in our
case MPJPE is a measure of pose accuracy independent of global position and
orientation.
Results: Our tracking results on TotalCapture are summarized in Table 1.
We used only 1 camera and the 13 IMUs provided. The cameras in TotalCapture
are rigidly mounted to the building and are not equipped with an IMU – hence
we assumed a static camera with unknown pose. VIP achieves a remarkably low
average MPJPE of 26mm and a MPJAE of only 12.1◦.
Comparisons to state-of-the-art: We outperform the learning-based ap-
proach introduced in the TotalCapture dataset [39] by 44mm – the approach
uses all 8 cameras and fuses IMU data with a probabilistic visual hull. We also
outperform [16], who report a mean MPJPE of 62mm using 8 cameras and all
13 IMUs. Admittedly, it is diﬃcult to compare approaches, since [39] and [16]
process the data in a frame-by-frame manner which is an advantage w.r.t. VIP,
which jointly optimizes over all frames simultaneously. However, VIP uses only
a single camera with unknown pose whereas the competitors use 8 fully cali-
brated cameras. To understand better the inﬂuence of components of VIP we
also report the tracking accuracy for ﬁve tracker variants in Table 1.
Comparison to IMU only: The Inertial tracker (IT) corresponds to the
single frame approach of Section 4.1. It uses only raw IMU orientations and is
the initialization for VIP. Over all sequences, IT achieves a MPJPE of 55mm.
VIP decreases this error by more than 50%. This demonstrates the usefulness of
fusing image information and optimizing heading angles.
Heading drift and misalignments: We report results of VIP-IT to demon-
strate the inﬂuence of optimizing heading angles, and sensor-to-bone misalign-
ments originating from an inaccurate initial pose. VIP-IT is identical to IT,
but uses the heading angles and initial pose obtained with VIP. VIP-IT is only
slightly less accurate than VIP validating the importance of inferring drift and
accurate initial pose. More evaluations are shown in the supplementary material.
Robustness to 2D pose accuracy: VIP-2D is identical to the VIP but
utilizes ground-truth 2D poses obtained by projecting ground-truth joint posi-

Accurate 3D Human Pose Using IMUs and a Moving Camera
13
Fig. 6. We show example results obtained using VIP for some challenging activities.
With VIP we get accurate 3D poses aligning well with the images using the estimated
camera poses.
tions to the images. VIP-2D achieves a MPJPE of 15.1mm which indicates how
much VIP can improve if 2D pose estimation methods keep improving.
Robustness to camera pose: VIP-Cam is also almost identical to VIP, but
uses the ground-truth camera pose instead of estimating it. The MPJPE of VIP-
Cam is 25.3mm, which is only 0.7mm better compared to VIP.
Fewer sensors: We report the error of VIP using 6 IMUs similar to [42],
denoted as VIP-IMU6. The combination of only 6 IMUs and 2D pose informa-
tion achieves a MPJPE of 39.6mm, which is 13.6mm higher than VIP-13 IMUs
but still very accurate. This demonstrates our approach could be used for appli-
cations where a minimal number of sensors is required.
This quantitative evaluation demonstrates the accuracy of VIP. Ideally, we would
evaluate VIP quantitatively also in challenging scenes, like the ones in 3DPW.
However, there exists no dataset with a comparable setting and ground-truth,
which was one of the main motivations of this work.
5.3
3D Poses in the Wild Dataset
VIP allowed us to achieve the second goal of this work: recording a dataset
with accurate 3D pose in challenging outdoor scenes with a moving camera.
A hand-held smartphone camera was used to record one or two IMU-equipped
actors performing various activities such as shopping, doing sports, hugging,
discussing, capturing selﬁes, riding bus, playing guitar, relaxing. The dataset
includes 60 sequences, more than 51, 000 frames and 7 actors in a total of 18
clothing styles. We also scanned subjects and non-rigidly ﬁtted SMPL to obtain
3D models similar to [27, 46]. For single subject tracking, we attached 17 IMUs
to all major bone segments. We used 9-10 IMUs per person to simultaneously
track up to 2 subjects. During all recordings one additional IMU was attached
to the smartphone. Video and inertial data was automatically synchronized by
a clapping motion at the beginning of a sequence as in [24]. For every sequence,
the subjects were asked to start in an upright pose with closed arms. In Fig. 6
we show tracking results illustrating the 3D model alignment with the images.
Fig. 7 shows more tracking results, where we animated the 3D models with the
reconstructed poses. 3DPW is the most challenging dataset (with 3D pose an-
notation) for state-of-the-art 3D pose estimation methods as evidenced by the
results reported in the supplemental material.

14
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
Fig. 7. We show several example frames of sequences in the 3DPW. The dataset con-
tains large variations in person identity, clothing and activities. For a couple of cases
we also show animated, textured SMPL body models.
Assignment Accuracy: In comparison to TotalCapture, the additional chal-
lenges in 3DPW originate from multiple people in the scene. Hence, we assessed
the accuracy of our automatic assignment of 2D poses to 3D poses using man-
ually labelled 2D pose candidate IDs. VIP achieves an assignment precision of
99.3% and a recall rate of 92.2% demonstrating the method correctly identiﬁes
the tracked persons for the vast majority of frames. This is a strong indication
that VIP achieves a 3D pose accuracy on 3DPW comparable to the MPJPE of
26mm reported for TotalCapture.
6
Conclusions
Combining IMUs and a moving camera, we introduced the ﬁrst method that can
robustly recover pose in challenging scenes. The main challenges we addressed
are: person identiﬁcation and tracking in cluttered scenes, and joint recovery of
3D pose for 2 subjects, camera and IMU heading drift. We combined discrete
optimization to ﬁnd associations, with continuous optimization to eﬀectively fuse
the sensor information.
Using our method, we collected the 3D Poses in the
Wild dataset, including challenging sequences with accurate 3D poses that we
make available for research purposes. With VIP it is possible to record people in
natural video easily and we plan to keep adding to the dataset. We anticipate the
proposed dataset will provide the means to quantitatively evaluate monocular
methods in diﬃcult scenes and stimulate new research in this area.
Acknowledgements. We thank Jorge M´arquez, Senya Polikovsky, Matvey
Safroshkin and Andrea Keller for the technical support.

Accurate 3D Human Pose Using IMUs and a Moving Camera
15
References
1. Andriluka, M., Roth, S., Schiele, B.: Monocular 3d pose estimation and tracking by
detection. In: The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). pp. 623–630 (2010)
2. Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., Black, M.J.: Keep it
SMPL: Automatic estimation of 3D human pose and shape from a single image.
In: European Conference on Computer Vision (ECCV) (2016)
3. Bull, A.D.: Convergence rates of eﬃcient global optimization algorithms. Journal
of Machine Learning Research 12(Oct), 2879–2904 (2011)
4. Cao, Z., Simon, T., Wei, S.E., Sheikh, Y.: Realtime multi-person 2d pose estimation
using part aﬃnity ﬁelds. In: The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) (2017)
5. Gurobi Optimization, I.: Gurobi optimizer reference manual (2016)
6. Helten, T., Baak, A., Bharaj, G., Muller, M., Seidel, H.P., Theobalt, C.: Personal-
ization and evaluation of a real-time depth-based full body tracker. In: 3D Vision
(3DV) (2013)
7. Henschel, R., Leal-Taix´e, L., Cremers, D., Rosenhahn, B.: Fusion of head and
full-body detectors for multi-object tracking. In: Computer Vision and Pattern
Recognition Workshops (CVPRW) (2018)
8. Ionescu, C., Papava, D., Olaru, V., Sminchisescu, C.: Human3.6m: Large scale
datasets and predictive methods for 3d human sensing in natural environments.
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 36(7),
1325–1339 (2014)
9. Jahangiri, E., Yuille, A.L.: Generating multiple diverse hypotheses for human 3d
pose consistent with 2d joint detections. In: IEEE International Conference on
Computer Vision (ICCV) Workshops (PeopleCap) (2017)
10. Kanazawa, A., Black, M.J., Jacobs, D.W., Malik, J.: End-to-end recovery of hu-
man shape and pose. In: The IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) (2018)
11. Lassner, C., Romero, J., Kiefel, M., Bogo, F., Black, M.J., Gehler, P.V.: Unite the
people: Closing the loop between 3d and 2d human representations. In: The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR). vol. 2 (2017)
12. Levinkov, E., Uhrig, J., Tang, S., Omran, M., Insafutdinov, E., Kirillov, A., Rother,
C., Brox, T., Schiele, B., Andres, B.: Joint graph decomposition & node labeling:
Problem, algorithms, applications. In: CVPR. vol. 7. IEEE (2017)
13. Li, S., Zhang, W., Chan, A.B.: Maximum-margin structured learning with deep
networks for 3d human pose estimation. In: IEEE International Conference on
Computer Vision (ICCV). pp. 2848–2856 (2015)
14. Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.: SMPL: A
skinned multi-person linear model. ACM Trans. Graphics 34(6), 248:1–248:16
(2015)
15. Loper, M.M., Mahmood, N., Black, M.J.: MoSh: Motion and shape capture from
sparse markers. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia) 33(6),
220:1–220:13 (2014)
16. Malleson, C., Volino, M., Gilbert, A., Trumble, M., Collomosse, J., Hilton, A.: Real-
time full-body motion capture from video and imus. In: 2017 Fifth International
Conference on 3D Vision (3DV) (2017)
17. von Marcard, T., Pons-Moll, G., Rosenhahn, B.: Human pose estimation from
video and IMUs. IEEE Transactions on Pattern Analysis and Machine Intelligence
(TPAMI) 38(8), 1533–1547 (2016)

16
T. v. Marcard, R. Henschel, M. J. Black, B. Rosenhahn, G. Pons-Moll
18. Martinez, J., Hossain, R., Romero, J., Little, J.J.: A simple yet eﬀective baseline
for 3d human pose estimation. In: IEEE International Conference on Computer
Vision (ICCV) (2017)
19. Mehta, D., Rhodin, H., Casas, D., Fua, P., Sotnychenko, O., Xu, W., Theobalt, C.:
Monocular 3d human pose estimation in the wild using improved cnn supervision.
In: 3D Vision (3DV). IEEE (2017)
20. Mehta, D., Sotnychenko, O., Mueller, F., Xu, W., Sridhar, S., Pons-Moll, G.,
Theobalt, C.: Single-shot multi-person 3d body pose estimation from monocular
rgb input. arXiv preprint arXiv:1712.03453 (2017)
21. Mehta, D., Sridhar, S., Sotnychenko, O., Rhodin, H., Shaﬁei, M., Seidel, H.P., Xu,
W., Casas, D., Theobalt, C.: Vnect: Real-time 3d human pose estimation with a
single rgb