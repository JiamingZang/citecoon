# GS-Pose: Generalizable Segmentation-based 6D Object Pose Estimation with 3D Gaussian Splatting

> 2024 · id: W4392971958 · arXiv: 2403.10683 · pdf: https://arxiv.org/pdf/2403.10683 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

GS-Pose: Generalizable Segmentation-based 6D Object Pose Estimation
with 3D Gaussian Splatting
Dingding Cai
Tampere University, Finland
dingding.cai@tuni.fi
Janne Heikkil¨a
University of Oulu, Finland
janne.heikkila@oulu.fi
Esa Rahtu
Tampere University, Finland
esa.rahtu@tuni.fi
Abstract
This paper introduces GS-Pose, a unified framework for
localizing and estimating the 6D pose of novel objects. GS-
Pose begins with a set of posed RGB images of a previ-
ously unseen object and builds three distinct representa-
tions stored in a database.
At inference, GS-Pose oper-
ates sequentially by locating the object in the input image,
estimating its initial 6D pose using a retrieval approach,
and refining the pose with a render-and-compare method.
The key insight is the application of the appropriate ob-
ject representation at each stage of the process. In par-
ticular, for the refinement step, we leverage 3D Gaussian
splatting, a novel differentiable rendering technique that of-
fers high rendering speed and relatively low optimization
time.
Off-the-shelf toolchains and commodity hardware,
such as mobile phones, can be used to capture new objects
to be added to the database. Extensive evaluations on the
LINEMOD and OnePose-LowTexture datasets demonstrate
excellent performance, establishing the new state-of-the-
art. Project page: https://dingdingcai.github.io/gs-pose.
1. Introduction
Acquiring the 3D orientation and 3D location of an ob-
ject based on RGB images is a long-standing and important
problem in computer vision and robotics. This 6D pose in-
formation is vital in applications that interact with the phys-
ical world, such as robotic manipulation [9, 10] and aug-
mented reality [30, 44]. Popular pose estimation approaches
are based on training instance-specific models, and they of-
ten assume the availability of an external object detector for
detecting the object from input RGB images. While some
works have proposed approaches to circumvent this prob-
lem [26, 33, 43], they often rely on high-fidelity 3D CAD
models of the object, which can be expensive and time-
consuming to acquire.
Ideally, a new object should be learned from a casually
captured set of RGB reference images without requiring any
expensive model parameter optimization. Recently, Liu et
al. [28] introduced a method called Gen6D in this direc-
tion. Gen6D works by extracting 2D feature maps from
the reference images, which are subsequently utilized for
various sub-tasks, including object localization, initial pose
estimation, and pose refinement. However, relying only on
2D representation often leads to sub-optimal performance.
Alternatively, OnePose [47] and OnePose++ [15] explicitly
reconstruct a 3D point cloud from the reference images via
local feature matching. The 6D pose is obtained using 2D-
3D correspondence matching between the test image and
the reference point cloud. The practical challenge is to ob-
tain an accurate 3D point cloud representation, particularly
for texture-less and symmetric objects. Furthermore, both
approaches still rely on an external object detector for crop-
ping out the object of interest, limiting their applicability in
real-world scenarios.
The key ingredient in 6D pose estimation is the object
representation generated from the input images. Popular
choices include 2D feature maps [28], 3D point clouds[15,
47], latent 3D models [37], and 3D CAD models[43],
to name a few.
Generally, each representation exhibits
strengths in one aspect, e.g., object localization or fast ini-
tial 6D pose approximation, but performs poorly on other
parts of the pipeline. With these insights, we propose a
framework that applies multiple representations optimized
for the three key steps: 1) object localization, 2) fast ini-
tial 6D pose estimation, and 3) iterative pose refinement.
In particular, we leverage the recent advancements in so-
called Foundation models and co-segmentation paradigms
to construct powerful representations for object localiza-
tion using only a handful of reference images. Secondly,
we estimate a rough 6D pose using optimized template re-
trieval. Finally, the pose estimate is refined using an itera-
tive render-and-compare technique. To this end, we rely on
a novel inverse rendering method called 3D Gaussian Splat-
ting (3DGS) [18], which represents a scene by many differ-
entiable 3D Gaussian primitives with optimizable geomet-
ric and appearance properties. This explicit representation
enables real-time photorealistic rendering capabilities, ideal
1
arXiv:2403.10683v2  [cs.CV]  14 Aug 2024

Figure 1. Overview of GS-Pose. GS-Pose involves two distinct phases to achieve pose estimation for a novel object, i.e., reference database
creation and object pose inference. The first phase operates offline and occurs only once per object to construct multiple representations of
the object. These representations include an object semantic representation (F obj), a set of rotation-aware embedding vectors ({V obj
i
}Nr
i=1),
and a 3D Gaussian Object (Gobj). During inference, GS-Pose first employs an object detector to detect the object in a query image using
the semantic information F obj. Then, GS-Pose adopts a pose estimator to produce an initial pose (blue box) from the detection result
with the rotation-aware embeddings {V obj
i
}Nr
i=1, Finally, GS-Pose leverages a pose refinement module (GS-Refiner) with Gobj to obtain a
refined pose (green box). We indicate the ground-truth pose in red.
for 6D pose optimization.
We evaluate the proposed framework,
called GS-
Pose, on the LINEMOD[16] and OnePose-LowTexture[15]
datasets and obtain new state-of-the-art results on both
benchmarks. The contributions of our work are summarized
as follows:
• We present an integrated framework for 3D CAD model-
free 6D object pose estimation. For each stage, we pro-
pose an optimized representation obtained from a set of
posed RGB images of newly added objects.
• We present a generalizable co-segmentation approach for
extracting object segmentation masks jointly from the ref-
erence RGB images, facilitating representation learning.
• We present a robust 3D Gaussian splatting-based method
for 6D object pose refinement.
• We experimentally confirm that the proposed framework
achieves state-of-the-art performance on the LINEMOD
and OnePose-LowTexture datasets.
2. Related Works
Object-Specific Pose Estimation. Most existing pose es-
timation methods [2, 3, 6, 13, 17, 19, 25, 38, 45, 50, 53]
are object-specific pose estimators, which are specialized
for pre-defined objects and cannot generalize to previously
unseen objects without retraining. Some of them[2, 3, 6, 19,
50, 53] directly regress the 6D pose parameters from RGB
images by training deep neural networks on a large num-
ber of labeled images. While other approaches[6, 13, 17,
25, 36, 38, 45] establish 2D-3D correspondences between
2D images and 3D object models to estimate the 6D pose
by solving the Perspective-n-Point (PnP)[22] problem. To
relax the assumptions about each object instance, category-
level methods[5, 7, 8, 51] have recently been proposed to
handle unseen object instances of the same trained category
by assuming that objects within the same category share
similar shape priors. However, they are still incapable of
estimating the object pose of unknown categories.
Generalizable Object Pose Estimation.
This type of
work[1, 15, 28, 35, 43, 47, 48] removes the requirement
of the object specific-training and can perform pose estima-
tion for previously unseen objects during inference. There
are two mainstreams, i.e., object model-based and object
model-free. The model-based approaches [1, 43, 48] as-
sume access to the 3D CAD models for rendering the object
pose-conditioned images that are often utilized for template
matching [1, 48, 54], pose refinement [24], or correspon-
dence establishment [43]. To avoid 3D CAD models, recent
works [15, 28, 35, 47] resort to capturing object multi-view
images with known poses as reference data for pose estima-
tion. OnePose series [15, 47] utilize the posed RGB images
to reconstruct 3D object point clouds and establish explicit
2D-3D correspondences between 2D query images and the
reconstructed 3D point clouds to solve the 6D pose. How-
ever, reliance on correspondences becomes fragile when ap-
plied to objects with visual ambiguities, such as symmetry.
Besides, the above methods often assume that the 2D object
detection or segmentation mask is available given a query
image. In contrast, Gen6D [28] leverages the labeled refer-
ence images to detect the object in query images, initialize
its pose, and then construct a 3D feature volume for pose
refinement, which is the first work to simultaneously satisfy
the requirements of being fully generalizable, model-free,
2

and RGB-only. The follow-up works [35, 55] revisit the
Gen6D pipeline and improve the performance and robust-
ness in object localization and pose estimation.
2D Object Detection.
Commonly used object detection
methods [14, 41, 42] are category-specific detectors and
cannot generalize to untrained categories. To tackle this
issue, some approaches [23, 28, 34, 43, 56] leverage ob-
ject reference images to detect previously unseen objects
through template matching or feature correlation. However,
they often show limited generalizability to new domains.
3D Object Representation.
Most generalizable pose es-
timators [1, 26, 31, 33, 43] often assume that the 3D ob-
ject representations are available, such as 3D CAD models.
OnePose family [15, 47] explicitly reconstructs 3D object
point clouds from object multi-view RGB images, which
can easily fail with challenging symmetric or textureless
objects.
Moreover, LatentFusion [37] and Gen6D series
[28, 35] utilize the 2D image features to build the 3D ob-
ject feature volumes for pose refinement. In this work, we
instead exploit the differentiable 3D Gaussian Splatting [18]
technique to create 3D Gaussian Object representations for
pose estimation. To the best of our knowledge, GS-Pose is
the first work that leverages 3D Gaussian splatting for 6D
object pose estimation.
3. Approach
This section presents GS-Pose for estimating the 6D pose
of novel objects from RGB images. An overview of GS-
Pose is provided in Fig. 1. GS-Pose operates in two distinct
phases: object reference database creation and object pose
inference. The creation phase, requiring RGB images of a
novel object with known poses (e.g., captured with com-
modity devices like mobile phones), is performed offline
once per object. During inference, GS-Pose leverages the
pre-built object reference database to facilitate the 6D pose
estimation task in a cascaded manner. In the subsequent
subsections, we first present the reference database creation
process in Sec. 3.1. Next, we describe the pose inference
workflow in Sec. 3.2. Finally, we present the objective func-
tions for training GS-Pose in Sec. 3.3.
3.1. Reference Database Creation
This section describes the process for creating the reference
database of a novel object based on its reference data. This
database is primarily comprised of object semantic repre-
sentation Fobj, a set of 3D object rotation-aware embedding
vectors {V obj
i
}Nr
i=1, and a 3D Gaussian Object representa-
tion Gobj, where Nr is the number of reference examples.
The creation process involves three sub-steps: (1) semantic
representation extraction, (2) 3D object rotation-aware rep-
resentation encoding, and (3) 3D Gaussian Object (3DGO)
model reconstruction, as depicted in Fig. 2. In the following
paragraphs, we elaborate on each sub-step.
Semantic Representation Extraction.
To enable GS-
Pose for 2D object detection and segmentation, we first
extract a set of feature representation tokens that can ef-
fectively capture the semantic information of the target
object from reference images. We leverage DINOv2[32]
to extract these tokens from RGB images.
Essentially,
a Co-Segmenter is employed to segment the object from
the background, ensuring that only relevant feature tokens
within the object region are considered (see Fig. 2 top).
Given Nr reference images, we first select Nk (≪Nr)
keyframes using farthest point sampling (FPS) [40] based
on their corresponding 3D rotation labels. Then, we ex-
tract image feature tokens Ffps ∈RNk×L×C from these
keyframes using DINOv2, where L and C denote the to-
ken number and feature dimension of each frame. Next, we
feed these feature tokens into the proposed Co-Segmenter,
consisting of a transformer-like module and a mask decod-
ing head, to jointly predict the object segmentation masks.
Specifically, we reshape the keyframe feature tokens as fea-
ture maps (denoted as ˆFfps), from which we sample a set
of frame-wise center tokens ˆFfps
c
∈RNk×C located at the
2D center of these feature maps. Next, the transformer-like
module takes Ffps and ˆFfps
c
as input and sequentially per-
forms Lm stacked self- and cross-attention computations
(see Fig. 3 top). The process can be formulated as
Lm ×















F fps = SelfAttn(F fps) ∈RNk×L×C
F fps = Reshape(F fps) ∈R1×NkL×C
F fps = CrossAttn(F fps, ˆF fps
c
)
F fps = SelfAttn(F fps) ∈R1×NkL×C
F fps = Reshape(F fps) ∈RNk×L×C
,
(1)
where Lm is the depth of the module. The transformed
Ffps is then fed into the mask decoding head (two 3 × 3
convolutional layers followed by an upsampling layer) to
produce the keyframe segmentation masks. Finally, we ex-
tract the object-aware semantic feature tokens Fobj from
the keyframe feature maps ˆFfps using the predicted masks.
Rotation-Aware Representation Encoding.
This step
focuses on extracting the 3D object rotation-aware embed-
ding vectors from reference images, which enables GS-
Pose to estimate an initial pose via template retrieval.
To achieve this, we first adopt an Obj-Segmenter to seg-
ment the object from each reference image and then uti-
lize a Rotation-Aware Encoder (RA-Encoder) to extract an
image-level embedding vector from the segmented image
(see Fig. 2 middle). Obj-Segmenter includes the DINOv2
backbone, a transformer-like module, and a mask decod-
ing head (identical to the one in Co-Segmenter).
Con-
cretely, Obj-Segmenter first extracts the DINOv2 feature to-
kens F ref
i
∈RL×C from the ith reference image. Then, the
3

Figure 2. Overview of the reference database creation process. We begin by selecting a group of keyframes from reference images. (1).
These keyframes are processed through DINOv2 and Co-Segmenter to jointly predict object segmentation masks, which are then utilized
to extract the object semantic tokens (F obj) from the keyframe features. (2). Image-wise object segmentation is performed for all reference
images {Iref
i
}Nr
i=1 using an Obj-Segmenter with the obtained semantic information F obj. We then employ an RA-Encoder to extract the
rotation-aware embeddings {V obj
i
}Nr
i=1 from the segmented images. (3). Finally, we create a 3D Gaussian Object representation Gobj
(viewed as a 3D point cloud for simplicity) using all segmented images with the known poses.
image feature tokens (F ref
i
) along with the object semantic
tokens (Fobj) are fed into the transformer-like module to
perform Lm stacked self- and cross-attention computations
(see Fig. 3 middle). This process can be formulated as
Lm ×
(
F ref
i
= SelfAttn(F ref
i
)
F ref
i
= CrossAttn(F ref
i
, F obj)
.
(2)
Subsequently, the mask decoding head is utilized to produce
a segmentation mask M ref
i
from the transformed image
features F ref
i
. Finally, we extract an image-level represen-
tation vector V ref
i
∈R64 from the segmented image using
RA-Encoder. RA-Encoder includes the DINOv2 backbone,
four 3 × 3 convolutional layers with stride 2, a generalized
average pooling layer, and a fully connected layer with an
output dimension of 64 (see Fig. 3 bottom).
3D Gaussian Object Reconstruction.
The last step is to
create the 3DGO representation Gobj for pose refinement
(see Fig. 2 bottom). 3D Gaussian Splatting [18] represents a
3D structure as a set of 3D Gaussians. Each 3D Gaussian is
parameterized with a 3D coordinate µ ∈R3, a 3D rotation
quaternion r ∈R4, a scale vector s ∈R3, an opacity factor
α ∈R, and spherical harmonics coefficients h ∈Rk, where
k is the degrees of freedom.
Consequently, the 3DGO
model is represented as Gobj
=
{µi, ri, si, αi, hi}U
i=1,
where U is the number of 3D Gaussians. All segmented
reference images with the known poses are utilized to build
this 3DGO model. We kindly refer to [18] for more details.
3.2. Object Pose Inference
This section outlines the inference pipeline of GS-Pose,
a cascaded process consisting of three core components.
Figure 3. (1). Co-Segmenter includes a transformer-like module
and a mask decoder to produce the co-segmentation masks. (2).
Obj-Segmenter consists of the DINOv2 backbone, a transformer-
like module, and a mask decoder to predict the object mask. (3).
RA-Encoder contains the DINOv2 backbone, four 3 × 3 2D con-
volutional (Conv2D) layers with stride 2, a generalizable average
pooling layer, and a fully connected (FC) layer.
Firstly, GS-Pose employs an object detector for detection.
Secondly, GS-Pose obtains an initial pose using a pose es-
timator based on the detection.
Finally, a 3D Gaussian
Splatting-based pose refinement module (GS-Refiner) is
adopted to optimize the initial pose. Fig. 4 illustrates these
components, and we describe each one in detail below.
Detector.
We leverage a segmentation-based detector to
localize the target object (see Fig. 4 top). The detector con-
4

Figure 4. (1). Detector first employs an Obj-Segmenter to pro-
duce a mask from the input image using the semantic information
(F obj). Then, connected components are computed from the pre-
dicted mask to generate proposals, which are further processed by
a proposal selector to determine the final output. (2). Pose Esti-
mator utilizes an Obj-Segmenter to predict an object mask M que
(F obj is omitted for clarity). An embedding vector V que is then
extracted from the segmented image using RA-Encoder, followed
by a pose decoder for estimating an initial pose (Pinit) using both
V que and M que. (3). GS-Refiner starts by applying an optimiz-
able transformation T j−1
gs
to the 3D coordinates of the 3D Gaus-
sian Object (3DGO) Gobj, where j ≥1 is the refinement step.
Then, the 3D Gaussian Splatting-based renderer (3DGS-Renderer)
generates an RGB image (Irend
j
) using the initial pose (Pinit) and
the transformed 3DGO ( ˆGobj). Finally, the gradient ∆Ti is used
to update the transformation parameter T j
gs, minimizing the differ-
ence (Lgs) between the rendered and the segmented images.
sists of an Obj-Segmenter (as described in Sec. 3.1) and a
proposal selector. Specifically, given an input image, we
first apply Obj-Segmenter to predict a segmentation mask,
from which we generate a set of mask proposals {M que
i
}m
i=1
by finding the connected components, where m represents
the number of proposals.
Subsequently, a set of object-
centric RGB images {Ique
i
}m
i=1 are cropped from the in-
put image using the 2D bounding boxes derived from these
mask proposals. Next, we feed these RGB images into the
proposal selector to obtain the final detection result. Within
the proposal selector, we first extract the DINOv2 feature
tokens {F que
i
∈RL×C}m
i=1 from these cropped images and
then compute the image-level cosine similarities between
these image features and the object semantic representation
Fobj. We select the one with the highest similarity score as
the output, denoted as Ique.
Pose Estimator.
In the second stage, we estimate an ini-
tial pose using a template retrieval-based pose estimator
(see Fig. 4 middle). This pose estimator is comprised of an
Obj-Segmenter (identical to the one in Detector), an RA-
Encoder (as described in Sec. 3.1), and a pose decoder.
We first obtain a segmentation mask M que using Obj-
Segmenter as well as an image-level representation vector
V que ∈R64 using RA-Encoder from the detection. We
then input M que and V que into the pose decoder to com-
pute an initial 6D pose Pinit = [Rinit, tinit]. More specif-
ically, the pose decoder first computes the cosine similar-
ity scores {ci = ||V que|| · ||V obj
i
||}Nr
i=1 between the query
vector V que and the reference vectors within the set of 3D
object rotation-aware representations {V obj
i
}Nr
i=1. Conse-
quently, the reference rotation matrix Rref
j
with the highest
similarity score is retrieved as the initial 3D rotation esti-
mate (Rinit), where j denotes the index of the closest ref-
erence template. We then analytically infer the initial 3D
translation tinit using the query mask (M que) and the jth
reference mask (M ref
j
). Specifically, we calculate a rela-
tive scale factor δs ∈R and a relative 2D center offset ratio
∆xy ∈R2 between M que and M ref
j
as follows:
(
δs
=
q
Area(M que)/Area(M ref
j
),
∆xy
= (Cbbox(M que) −Cbbox(M ref
j
))/S,
(3)
where Area(M) =
SP
j=0
SP
i=0
M[i,j] denotes the mask area, S
is the mask scale, and Cbbox(M) denotes the 2D center of
the bounding box tightly surrounding the mask M. We then
compute the distance tque
z
∈R and the 2D center P que
xy
∈
R2 of the object in the query image by
tque
z
= tref
z
S/δs/Sque
box , P que
xy
= Sque
box ∆xy + Cque
box ,
(4)
where tref
z
is the pre-computed z-axis distance of the ob-
ject in the reference image (more details provided in the
supplementary materials), Cque
box and Sque
box are the 2D center
and scale of the 2D object bounding box predicted from the
original input image. Finally, we obtain the initial transla-
tion estimate by
tinit = tque
z
K−1 ¯P que
xy ,
where ¯P que
xy
∈R3 is the homogeneous form of P que
xy , and
K denotes the camera intrinsic matrix.
GS-Refiner.
The initial pose estimate is further refined by
leveraging the 3D object representation Gobj through an it-
erative render-and-compare optimization procedure. This
pose refinement stage, termed GS-Refiner, utilizes differen-
tiable 3D Gaussian Splatting-based rendering [18], which
facilitates the optimization of a learnable transformation
Tgs to minimize the discrepancy between the rendered ob-
ject and the observed query image. Formally, the optimal
transformation T ∗
gs is obtained by minimizing the following
objective:
T ∗
gs = arg min
Tgs
Lgs(Rgs(Tgs ⊙Gobj, Pinit), ¯Ique).
(5)
5

Here, Rgs denotes the differentiable rendering function,
Tgs ∈SE(3) represents the learnable transformation pa-
rameters, ⊙indicates applying a rigid transformation to the
3D coordinates of Gobj, and ¯Ique is the segmented query
image. The loss function Lgs is defined as a combination of
the losses based on the image structural similarity (SSIM)
and Multi-Scale SSIM [52]:
Lgs = LD−SSIM + LD−MSSIM
(6)
The optimization is initialized with an identity transforma-
tion and iteratively updates T ∗
gs using the AdamW optimizer
[29] with the cosine annealing learning rate schedule, start-
ing from 5 × 10−3 over a maximum of Ngs iterations with
10 warm-up steps. An early-stopping strategy is employed
when the refinement loss converges to a predefined thresh-
old η. The final refined pose is obtained as P = PinitT ∗
gs.
3.3. Training Objective Functions
We employ the Binary Cross Entropy (BCE) loss to train
both Co-Segmenter (Lcoseg) and Obj-Segmenter (Lobjseg)
for pixel-wise segmentation prediction, i.e.,
Lcoseg = LBCE(M, ¯
M),
Lobjseg = LBCE(M, ¯
M),
(7)
where M and
¯
M separately denote the predicted and
ground truth group-wise segmentation masks, M and ¯
M
are the predicted and ground-truth frame-wise segmentation
masks, respectively. Additionally, we adopt the Negative
Log-Likelihood (NLL) loss to train RA-Encoder for learn-
ing the 3D object rotation-aware representation, defined as:
Lrot = −log
exp(||V que|| · ||Vp||/τ)
PNs
j=1 exp(||V que|| · ||Vj||/τ)
,
(8)
where Ns is the number of the reference samples in a batch,
V que and Vj are the representation vectors of the query and
the jth reference samples, respectively, τ is the temperature,
and p is the index of the positive training sample determined
by measuring the geodesic distance of the 3D rotation ma-
trices, calculated as:
p = arg min
0≤j≤Ns
arccos trace(RqueRT
j ) −1
2
,
(9)
where Rque is the ground truth rotation matrix of the query
sample, and Rj is for the jth training sample.
Conse-
quently, the entire network is optimized through a combined
loss in an end-to-end manner,
Ltotal = λcLcoseg + λoLobjseg + λrLrot,
(10)
where λ{c,o,r} represent the balance weights.
4. Experiments
Datasets. We utilize the synthetic MegaPose dataset [20]
for training and the real-world datasets LINEMOD [16]
and OnePose-LowTexture [15] for evaluation. The Mega-
Pose dataset was generated using BlendProc [11] and 1000
diverse objects from the Google Scanned Objects dataset
[12] and includes one million synthetic RGB images. The
LINEMOD dataset [16] contains 13 objects and is com-
monly used for 6D object pose evaluation.
Following
[15, 28, 35, 47], the training split of LINEMOD is selected
as reference data, while the testing split is used for evalu-
ation. OnePose-LowTexture [15] is a challenging dataset
with low-texture or texture-less objects, from which eight
scanned objects are utilized for evaluation.
Each object
was captured by two video sequences with different back-
grounds. We follow OnePose++ [15] and select the first
video as the reference and the other as query data.
Baseline Methods.
For comparison, we assess GS-Pose
against several state-of-the-art methods:
Gen6D [28],
Cas6D [35], OnePose [47], OnePose++ [15], and MFOS
[21]. They take RGB reference images of novel objects
with known poses as input to define the object coordinate
system and then estimate the 6D pose of these objects from
query images without retraining the network parameters.
Metrics.
We adopt the widely used ADD [16] metric that
measures the average distance between 3D points after be-
ing transformed by the ground truth and predicted poses.
The ADDS metric is used for symmetric objects, which
measures the average distance to the closest point instead
of the ground truth point. Following the protocol[16], we
report the average recall rate of ADD(S) within 10% of
the object diameter, denoted as ADD(S)@0.1d. We also
compute the 2D projection errors of the points after be-
ing transformed by the ground truth and predicted poses.
We report the average recall rate within 5 pixels, denoted
Method
Type
cat
duck
bvise
cam
driller
Avg.
SRPN-P [23]
BBox
11.85
1.62
18.94
2.44
8.91
8.76
SRPN [23]
BBox
9.72
4.56
22.47
13.43
10.97
12.23
SRPN-D [23]
BBox
22.97
1.85
49.14
17.76
18.89
22.12
OSOP [43]
BBox
32.10
34.81
26.68
24.33
21.36
27.86
Gen6D [28]
BBox
76.99
42.15
63.33
72.92
48.78
60.83
Cas6D [35]
BBox
79.46
67.44
66.32
76.39
59.35
69.79
LocPoseNet [55]
BBox
81.68
61.80
79.45
80.50
68.31
74.35
GS-Pose (ours)
Mask
69.14
80.07
66.46
75.51
73.08
72.85
GS-Pose (ours)
BBox
84.44
86.88
71.76
79.04
80.60
80.54
Table 1.
Quantitative results of the 2D object localization
on LINEMOD [16] regarding the mAP@[0.5:0.95](%) metric.
”Type” indicates the detection type in the form of either bound-
ing boxes or segmentation masks. GS-Pose derives the minimum
2D object bounding box from the mask prediction for comparison.
We highlight the best in Bold.
6

Method
YOLOv5
ape
bwise
cam
can
cat
driller
duck
ebox*
glue*
holep.
iron
lamp
phone
Avg.
ADD(S)@0.1d
Gen6D[28]
-
62.1
45.6
-
40.9
48.8
16.2
-
-
-
-
-
-
-
Gen6D[28]†
-
77.0
66.7
-
60.7
67.4
40.5
98.3
87.8
-
-
89.8
-
-
Cas6D[35]†
-
86.3
70.1
-
60.6
84.8
51.3
98.8
88.5
-
-
93.4
-
-
OSOP[43]
26.1
55.6
36.2
52.2
42.5
49.6
22.2
72.4
52.3
18.6
72.3
27.9
39.6
43.6
OnePose[47]
✓
11.8
92.6
88.1
77.2
47.9
74.5
34.2
71.3
37.5
54.9
89.2
87.6
60.6
63.6
OnePose++[15]
✓
31.2
97.3
88.0
89.2
70.4
92.5
42.3
99.7
48.0
69.7
97.4
97.8
76.0
76.9
MFOS[21]
✓
47.2
73.5
87.5
85.7
80.2
92.4
60.8
99.6
69.7
93.5
82.4
95.8
51.0
78.4
PoseMatcher [4]
✓
59.2
98.1
93.4
96.0
88.0
98.4
54.1
97.8
91.5
73.4
97.9
98.1
92.1
87.5
GS-Pose (ours)
59.6
99.6
96.0
97.6
88.9
95.1
74.9
99.3
92.2
86.8
98.2
96.7
80.7
89.7
GS-Pose (ours)
✓
71.0
99.8
98.2
97.7
86.7
96.2
77.2
99.6
98.4
87.4
99.2
98.9
85.0
92.0
Proj@5pix
Gen6D [28]†
-
82.5
90.8
-
96.1
72.4
79.7
97.8
96.2
-
-
91.6
-
-
Cas6D [35]†
-
93.4
96.3
-
99.0
95.0
93.5
98.3
98.8
-
-
96.9
-
-
OnePose[47]
✓
35.2
94.4
96.8
87.4
77.2
76.0
73.0
89.9
55.1
79.1
92.4
88.9
69.4
78.1
OnePose++[15]
✓
97.3
99.6
99.6
99.2
98.7
93.1
97.7
98.7
51.8
98.6
98.9
98.8
94.5
94.3
GS-Pose (ours)
77.5
98.9
98.4
97.6
97.6
92.3
97.7
97.3
91.3
96.5
98.9
90.9
91.9
94.4
GS-Pose (ours)
✓
97.9
98.9
99.1
97.6
98.9
93.7
97.8
97.1
97.4
98.8
99.6
94.2
93.8
97.3
Table 2. Quantitative results on LINEMOD [16] regarding the ADD(S)@0.1d and Proj@5pix metrics. ✓indicates using the external
YOLOv5 [49] as the object detector. * indicates symmetric objects. † indicates that the method includes a subset of objects of LINEMOD
as training data. We highlight the best in bold. ”-” indicates unavailable results.
Method
GTBox Toy. Tea. Cat. Cam. Shin. Molie. David Marse. Avg.
PVNet [39]
12.3 90.0 68.1
67.6 95.6
57.3
49.6
61.3
62.7
Gen6D [28]
55.5 40.0 70.0
42.2 62.7
16.6
15.8
8.1
38.9
OnePose [47]
✓
65.6 89.0 39.7
90.9 87.9
31.2
42.7
30.4
59.7
OnePose++[15]
✓
89.5 99.1 97.2
92.6 98.5
79.5
97.2
57.6
88.9
GS-Pose init
55.0 75.7 82.6
69.7 95.1
63.4
65.7
57.5
70.6
GS-Pose (ours)
89.3 86.7 100.0 90.2 99.3
95.9
91.7
83.6
92.1
Table 3.
Quantitative results on each object in OnePose-
LowTexure [15] regarding the ADD(S)@0.1d metric. ”init” in-
dicates the initial pose estimation results of GS-Pose. ”GTBox”
indicates the ground truth 2D object bounding boxes. We high-
light the best in bold.
as Proj@5pix. In addition, the mean Average Precision
(mAP)[0.5:0.95](%) [27] is reported for evaluating the 2D
object localization performance.
Configurations.
In our experiments, we set the hyperpa-
rameters: Nk = 8, Ngs = 400, Lm = 4, η = 1 × 10−4,
τ = 0.1, λc = 1, λr = 1, λo = 1, Ns = 32, unless
otherwise specified. We use the AdamW [29] solver with
the cosine annealing learning rate schedule, starting from
1 × 10−4 to 1 × 10−6, to train our framework for 100,000
steps on an Nvidia RTX3090 GPU with batch size 2.
4.1. Object Detection
Experiment Setups. Given a set of object-centric RGB im-
ages as reference data, the task is to localize the object of
interest in query images without fine-tuning the model pa-
rameters.
Results on LINEMOD.
We report the quantitative results
of the 2D object detection on LINEMOD [16] regarding the
mAP@[0.5:0.95](%) metric in Table 1. We primarily com-
pare GS-Pose against Gen6D [28], Cas6D [35], and Loc-
PoseNet [55], which are the most similar works.
Over-
all, GS-Pose achieves 72.85% mAP and 80.54% mAP in
terms of 2D segmentation masks and the mask-induced 2D
bounding boxes, respectively. Our segmentation-based de-
tection approach outperforms all baseline methods. It is
worth noting that all methods include a subset of held-out
objects in LINEMOD as training data and evaluate on the
other 5 selected objects, except OSOP [43] and GS-Pose.
4.2. Object Pose Estimation
Experiment Setups. Given a set of reference RGB im-
ages of a novel object with known poses, the task is to
estimate the 6D pose of the object in query images with-
out fine-tuning the network parameters. We conduct exper-
iments under two settings: 1) pose estimation without pre-
existing 2D bounding boxes and 2) pose estimation within
pre-existing 2D bounding boxes.
The latter involves es-
timating the pose from cropped object-centric images ac-
quired either using the YOLOv5 detector [49] (Table 2) or
by projecting the 3D object bounding boxes using ground
truth poses (Table 3).
Results on LINEMOD.
Table 2 shows the quantita-
tive results in terms of the ADD(S)@0.1d and Proj@5pix
metrics.
Overall, GS-Pose achieves impressive 89.7%
ADD(S)@0.1d and 94.4% Proj@5pix recalls on average,
7

Variant
ADD(S)
@0.1d
w/o proposal selector
88.95
w/o LD−SSIM
89.44
w/o LD−MSSIM
89.29
GS-Pose (ours)
90.86
Table 4. Ablation studies on
the LINEMOD subset regard-
ing different variants.
Method
Number of reference images (Nr)
8
16
32
64
128
All (∼180)
Gen6D [28]
-
29.07 49.41
-
-
62.45
Cas6D [35]
-
32.43 53.90
-
-
70.72
OnePose++ [15]
-
31.38 54.98
-
-
78.10
GS-Pose (ours)
49.39 62.50 74.50 85.81 89.00
90.86
Table 5.
Results on the LINEMOD subset regarding
the varying number of reference images in terms of the
ADD(S)@0.1d metric.
Maximum refin-
ement steps (Ngs)
100
200
300
400
500
ADD(S)@0.1d
85.58 90.31 90.70 90.86 90.82
Runtime (ms)
851
936
950
958
966
Table 6. Results on the LINEMOD subset
in terms of the ADD(S)@0.1d metric. The
refinement process automatically terminates
when the loss converges.
outperforming all baseline approaches.
When using the
2D detection results predicted by YOLOv5 [49], as in
OnePose [47] and OnePose++ [15], GS-Pose further im-
proves the ADD(S)@0.1d metric to 92.0% and Proj@5pix
to 97.3%, setting new state-of-the-art performance on
LINEMOD. This advantage is largely attributed to the
low-textured or symmetric objects (e.g., ape, duck, glue),
where the correspondence-based methods like OnePose
[47], OnePose++[15], MFOS [21], and PoseMatcher [4] in-
herently struggle.
Results on OnePose-LowTexture.
We further compare
GS-Pose against the baselines [15, 28, 47] on OnePose-
LowTexture [15]. In addition, we also include PVNet [39],
which trains a single network per object using approxi-
mately 5000 rendered images. Table 3 reports the quantita-
tive results regarding ADD(S)@0.1d and shows new state-
of-the-art performance (92.1%) achieved by GS-Pose. The
keypoint-based approach OnePose [47] obtains an average
recall of 59.7%, which lags behind our initial result (70.6%)
by about 10% and our refined result (92.1%) by over 30%.
OnePose relies on local feature matching to establish the
keypoint-based 2D-3D correspondences, making it unreli-
able for low-textured or texture-less objects in this dataset.
To alleviate this, OnePose++ [15] employs the keypoint-
free LoFTR [46] for feature matching and significantly im-
proves the result to 88.9%. Even though OnePose++ neces-
sitates ground-truth 2D object bounding boxes for evalua-
tion, GS-Pose still outperforms it using our built-in detec-
tor. Compared to the object-specific pose estimator PVNet
[39], GS-Pose outperforms it by a substantial margin.
4.3. Additional Experiments
We conduct additional experiments on the LINEMOD sub-
set and report the results in Tab. 4, Tab. 5, and Tab. 6.
Ablation studies.
To assess the efficacy of the connected
component-based proposal selector in object detection, we
remove it from our object detector and then utilize the 2D
bounding box derived from the entire segmentation mask
as the output. As a result, the ADD(S)@0.1d metric de-
creases by about 2%, indicating the proposal selector’s ef-
ficacy. Besides, when either LD−SSIM or LD−MSSIM is
removed from GS-Refiner, the performance decreases, in-
dicating that both terms contribute positively to the pose re-
finement.
Number of reference images.
GS-Pose consistently
achieves better performance with more reference images.
When using only 32 reference images, GS-Pose obtains
74.5% recall, already comparable to or even outperform-
ing the results achieved by the baseline methods using all
reference images.
Maximum refinement steps.
As expected,
GS-Pose
achieves consistently better performance with more refine-
ment steps, reaching saturation at up to 400 steps. The re-
finement process terminates when the loss converges, thus
resulting in a nonlinear increase in runtime with more steps.
Runtime.
GS-Pose takes about one second to process a
single RGB image (with resolution 480 × 640) on a desk-
top with an AMD 835 Ryen 3970X CPU and an Nvidia
RTX3090 GPU, in which ∼0.16s for object detection,
∼0.01s for pose initialization, and ∼0.96s for refinement.
GS-Pose employs an iterative, gradient-based optimization
process for pose refinement, which improves accuracy but at
the cost of computational efficiency. In future work, we plan
to explore more efficient optimization algorithms, such as
the Levenberg-Marquardt algorithm, to accelerate the pose
refinement process for GS-Pose.
5. Discussion and Conclusion
This work presents GS-Pose, an integrated framework for
estimating the 6D pose of novel objects in RGB images.
GS-Pose leverages multiple representations of newly added
objects to facilitate cascaded sub-tasks: object detection,
initial pose estimation, and pose refinement.
GS-Pose
is trained once using synthetic RGB images and evalu-
ated on two real-world datasets, LINEMOD and OnePose-
LowTexture. The experimental results demonstrate that GS-
Pose achieves state-of-the-art performance on the bench-
mark datasets and shows promising generalization capabil-
ities to new datasets. However, objects with slender or thin
structures may pose challenges for GS-Pose due to poor
segmentation. Future work could be extending GS-Pose for
6D pose tracking of unseen objects.
8

6. Acknowledgement
This work was supported by the Academy of Finland
project #353139. We also acknowledge CSC - IT Center
for Science, Finland, for computational resources.
References
[1] Dingding Cai, Janne Heikkil¨a, and Esa Rahtu.
Ove6d:
Object viewpoint encoding for depth-based 6d object pose
estimation.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6803–
6813, 2022. 2, 3
[2] Dingding Cai, Janne Heikkil¨a, and Esa Rahtu.
Sc6d:
Symmetry-agnostic and correspondence-free 6d object pose
estimation. In 2022 International Conference on 3D Vision
(3DV), pages 536–546. IEEE, 2022. 2
[3] Dingding Cai, Janne Heikkil¨a, and Esa Rahtu.
Msda:
Monocular self-supervised domain adaptation for 6d object
pose estimation.
In Scandinavian Conference on Image
Analysis, pages 467–481. Springer, 2023. 2
[4] Pedro Castro and Tae-Kyun Kim. Posematcher: One-shot
6d object pose estimation by deep feature matching.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 2148–2157, 2023. 7, 8
[5] Dengsheng Chen, Jun Li, Zheng Wang, and Kai Xu. Learn-
ing canonical shape space for category-level 6d object pose
and size estimation. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
11973–11982, 2020. 2
[6] Hansheng Chen, Pichao Wang, Fan Wang, Wei Tian, Lu
Xiong, and Hao Li.
Epro-pnp: Generalized end-to-end
probabilistic perspective-n-points for monocular object pose
estimation.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 2781–
2790, 2022. 2
[7] Wei Chen, Xi Jia, Hyung Jin Chang, Jinming Duan, Linlin
Shen, and Ales Leonardis. Fs-net: Fast shape-based network
for category-level 6d object pose estimation with decoupled
rotation mechanism. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
1581–1590, 2021. 2
[8] Xu Chen, Zijian Dong, Jie Song, Andreas Geiger, and Otmar
Hilliges. Category level object pose estimation via neural
analysis-by-synthesis. In European Conference on Computer
Vision, pages 139–156. Springer, 2020. 2
[9] Alvaro Collet, Manuel Martinez, and Siddhartha S Srinivasa.
The moped framework: Object recognition and pose estima-
tion for manipulation. The international journal of robotics
research, 30(10):1284–1306, 2011. 1
[10] Xinke Deng, Yu Xiang, Arsalan Mousavian, Clemens Epp-
ner, Timothy Bretl, and Dieter Fox. Self-supervised 6d object
pose estimation for robot manipulation. In 2020 IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 3665–3671. IEEE, 2020. 1
[11] Maximilian Denninger,
Martin Sundermeyer,
Dominik
Winkelbauer, Youssef Zidan, Dmitry Olefir, Mohamad El-
badrawy, Ahsan Lodhi, and Harinandan Katam.
Blender-
proc. arXiv preprint arXiv:1911.01911, 2019. 6
[12] Laura Downs,
Anthony Francis,
Nate Koenig,
Bran-
don Kinman, Ryan Michael Hickman, Krista Reymann,
Thomas Barlow McHugh, and Vincent Vanhoucke. Google
scanned objects: A hi