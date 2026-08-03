# iG-6DoF: Model-free 6DoF Pose Estimation for Unseen Object via Iterative 3D Gaussian Splatting

> 2025 · id: W4413156710 · pdf: https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_iG-6DoF_Model-free_6DoF_Pose_Estimation_for_Unseen_Object_via_Iterative_CVPR_2025_paper.pdf · 来源: backfill-cvf
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

iG-6DoF: Model-free 6DoF Pose Estimation for Unseen Object via
Iterative 3D Gaussian Splatting
Tuo Cao1, Fei Luo1, Jiongming Qin1, Yu Jiang1, Yusen Wang1, and Chunxia Xiao1*
1School of Computer Science, Wuhan University, Wuhan, Hubei, China
{maplect,luofei,jiongming,jiangyu1181,wangyusen,cxxiao}@whu.edu.cn
http://graphvision.whu.edu.cn/
Abstract
Traditional methods in pose estimation often rely on pre-
cise 3D models or additional data such as depth and nor-
mals, limiting their generalization, especially when objects
undergo large translations or rotations. We propose iG-
6DoF, a novel model-free 6D pose estimation method using
iterative 3D Gaussian Splatting to estimate the pose of un-
seen objects. We first estimates an initial pose by leveraging
multi-scale data augmentation and the rotation-equivariant
features to create a better pose hypothesis from a set of
candidates. Then, we propose an iterative 3DGS approach
through iteratively rendering and comparing the rendered
image with the input image to further progressively improve
pose estimation accuracy. The proposed method consists of
an object detector, a multi-scale rotation-equivariant fea-
ture based initial pose estimator, and a coarse-to-fine pose
refiner. Such combination allows our method to focus on the
target object in a complex scene dealing with large move-
ment and weak textures. Our method achieves state-of-the-
art results on the LINEMOD, OnePose-LowTexture, Gen-
MOP datasets and our self-captured data, demonstrating
its strong generalization to unseen objects and robustness
across various scenes.
1. Introduction
Estimating the rotation and translation parameters of ob-
jects within images has been a longstanding and widely
studied problem in computer vision. It has extensive ap-
plications in virtual reality, robotic manipulation, and au-
tonomous driving. Early pose estimation methods [10, 11,
26, 48, 61, 62, 71] primarily focused on pose estimation
at instance-level, requiring the target object to be included
in the training set. They often lack generalization capabil-
ities and hinder the estimation of unseen objects. Subse-
quently, researchers introduced category-level pose estima-
*Chunxia Xiao is corresponding author.
Reference Images
Input Images
Predicted Mask
3DGS model
Predicted Pose
Rendered Object
Figure 1. Given a set of reference images and an input image, our
method outputs the object mask, constructs a 3D Gaussian model,
and estimates its 6D pose.
tion methods [15, 19, 63, 73, 77], which can estimate the
pose parameters of objects within the same category, even
if the specific instance is not present in the training set. They
demonstrate a degree of generalization.
Recently, research has increasingly focused on general-
izable pose estimation, aiming to develop a universal model
to estimate an object’s pose using only its CAD model or a
few specific-view images [41]. Existing generalizable pose
estimation methods can be primarily categorized into two
types. The first type is CAD model-based. These meth-
ods [14, 21, 22, 36, 50] typically utilize the 3D or tex-
ture information of a precise CAD model as prior knowl-
edge. They often employ feature-matching techniques to
obtain 2D-3D correspondences between the query image
and the CAD model. Then, they calculate pose parameters
using traditional numerical algorithms such as PnP [20] or
ICP [6]. The second type is model-free object pose estima-
tion. These methods [8, 13, 25, 27, 42, 59] do not require
precise CAD models but rely on a set of annotated reference
images of the object. Multi-view stereo geometry provides
geometric information about the object as prior knowledge.
Compared to CAD-based methods, model-free methods of-
fer greater potential for practical applications without the
need to acquire accurate CAD models.
This CVPR paper is the Open Access version, provided by the Computer Vision Foundation.
Except for this watermark, it is identical to the accepted version;
the final published version of the proceedings is available on IEEE Xplore.
6436

However, current model-free methods have certain lim-
itations. For instance, FS6D [27] requires additional depth
information for supervision, Gen6D [42] relies solely on 2D
representations and struggles with large object movements
and rotations. OnePose [59] necessitates establishing 2D-
3D correspondences, which can lead to suboptimal perfor-
mance in weak-texture regions. To address these issues, we
propose a pose estimation network based on the multi-scale
rotation-equivariant feature and the 3D Gaussian Splatting
(3DGS). The core idea is to utilize multi-scale information
to tackle challenges posed by large-scale movements and
leverage the high-quality rendering capabilities of 3DGS to
handle pose estimation for weak textures.
As illustrated in Figure 1, our method takes a set of refer-
ence images and an input image to output the object’s mask,
construct a 3D Gaussian model, and determine the object’s
6D pose. Unlike traditional methods that match the query
image to the closest reference image, which often results in
inaccurate initial poses due to sparse reference data, our ap-
proach employs multi-scale data augmentation of reference
images and builds a feature vector space on the icosahedral
group to estimate the initial pose. Then, we refine this pose
by iteratively searching the surrounding neighborhood, uti-
lizing the high-quality rendering capabilities of 3DGS [33].
The key contributions of this work can be summarized as
follows:
• We propose a novel end-to-end object pose estimation
method that enables direct pose estimation of unseen ob-
jects without retraining.
• To enhance initialization accuracy, we introduce a multi-
scale icosahedral group feature matching module, im-
proving initial pose estimation precision.
• Finally, we incorporate a 3DGS-based rendering-and-
comparison module for fast and accurate iterative pose
optimization.
2. Related works
2.1. Model-based Unseen Object Pose Estimation
CAD model-based methods incorporate detailed 3D object
models as prior knowledge to accurately determine the po-
sition and orientation of previously unseen instances within
a scene.
Pitteri et al. pioneered using CAD models for
3DoF pose estimation by approximating object geometry
with corner points [50]. However, this approach was lim-
ited to objects with distinct corners. To address this, they
subsequently introduced an embedding method to capture
local 3D geometry, enabling 2D-3D correspondence estab-
lishment and PnP+RANSAC-based pose estimation [49].
However, both methods were confined to estimating only
three degrees of freedom.
Building upon point cloud registration techniques for
unseen objects, Zhao et al. [75] introduced a geome-
try correspondence-based approach using generic, object-
agnostic features to establish robust 3D-3D correspon-
dences.
However, this method required external meth-
ods like Mask-RCNN [24] for object class and segmenta-
tion mask determination. To address this limitation, Chen
et al. [14] presented ZeroPose, a framework for joint in-
stance segmentation and pose estimation of unseen objects.
Leveraging SAM [34], they generated object proposals and
employed template matching for instance segmentation.
A hierarchical geometric feature matching network based
on GeoTransformer [53] was used to establish correspon-
dences. Expanding on ZeroPose, Lin et al. [40] introduced a
refined matching score considering semantics, appearance,
and geometry for improved segmentation. For pose esti-
mation, they developed a two-stage partial-to-partial point
matching model to effectively construct dense 3D-3D cor-
respondences. FoundPose [46] put forward a rapid template
retrieval approach which founded on visual words created
from DINOv2 [45] patch descriptors. As a result, it reduces
the dependence on large amounts of data and boosts the
matching speed. Freeze [12] represents the initial technique
that harnesses the synergy between geometric and vision
foundation models to estimate the pose of unseen objects.
2.2. Model-free Unseen Object Pose Estimation
In contrast to CAD model-based approaches, manual ref-
erence view-based methods bypass the need for object
CAD models by relying on manually labeled reference im-
ages. These methods primarily establish correspondences
between the query image and reference views, either in
3D-3D or 2D-3D space, to determine object pose. He et
al. [27] introduced a pioneering few-shot 6DoF pose es-
timation method using a transformer-based dense RGBD
prototype matching framework to correlate query and refer-
ence views without additional training. Corsetti et al. [32]
employed textual prompts for object segmentation and re-
formulated the problem as relative pose estimation between
scenes, solved through point cloud registration.
Sun et al. [59] adapted visual localization techniques
for pose estimation by constructing a Structure from Mo-
tion (SfM) model of the unseen object using reference view
RGB sequences. A graph attention network matched 2D
query image keypoints with 3D points in the SfM model.
However, this approach suffered from poor performance
on low-textured objects due to reliance on repeatable key-
points. He et al. [25] addressed this limitation by intro-
ducing a keypoint-free SfM method to reconstruct semi-
dense point cloud models of low-textured objects using the
detector-free feature matching method LoFTR [58]. Rec-
ognizing the suboptimal performance of pre-trained fea-
ture matching models [54, 58] for pose estimation, Castro
et al. [13] redesigned the training pipeline using a three-
view system for one-shot object-to-image matching. In ad-
6437

Masks
Posed Refrences Images
Pose Estimator
Object-3DGS
Differentiable 
Render
Input image
Initial 6D pose
Refiner Model
Refined 6D pose
Input image and mask
loss
①
②
Coarse stage: coarse pose estimator
Fine stage: Refine based on the coarse or refined pose
The first refinement of pose
Further refinement of pose
Render image
Figure 2. Overview of iG-6DoF. Our method employs a coarse-to-fine approach, where the pose estimator first estimates an initial pose
from the input image, and then the pose refiner is employed to achieve a precise final pose.
dition to this, FoundationPose [69] has constructed a unified
framework for handling both model-based and model-free
scenarios simultaneously.
2.3. Pose Estimation with Neural Rendering
Recently,
some methods based on Neural Rendering
(NeRF[44] and 3DGS[33]) have made significant strides in
representing three-dimensional scenes [4, 5, 7, 23, 28, 52,
65, 66].
These methods train a neural network to mini-
mize the errors between rendered images and real images,
thereby modeling the color and volumetric density of a
scene as a function of spatial position, thereby enabling
high expressiveness for complex three-dimensional envi-
ronments. Several efforts have applied this framework to
tasks such as pose estimation and Simultaneous Localiza-
tion and Mapping (SLAM) [16, 31, 36, 57, 76]. For in-
stance, the iNeRF method [72] assumes a given camera
pose, creates an image via the rendering process and pro-
ceeds to contrast the pixel variances with the query image.
The gradient data obtained in this way is then applied to
iteratively adjust the camera pose in a step-by-step manner
until the rendered image matches the query image precisely.
Similarly, Nerf-pose[38] makes use of NeRF’s implicit por-
trayal of 3D scenes and trains a pose regression network so
as to set up associations between 2D and 3D data. iComMa
[60] inverts 3DGS to achieve accurate pose estimation with-
out training, using a gradient-based framework and an end-
to-end matching module to improve robustness and preci-
sion under difficult conditions.
Although these methods
can achieve an accurate pose estimation using pixel-level
comparison losses, they encounter difficulties in achieving
effective convergence in complex situations. Specifically,
when there is a substantial disparity between the rendered
images and the query images, it becomes a bottleneck for
precise pose estimation.
3. Method
Given a set of reference images of an object with known
camera poses and intrinsics, our goal is to estimate the
6D pose (translation T = (tx, ty, tz) ∈R3 and rotation
R ∈SO3) of the same object in a query image. The pose
transformation maps points from the object coordinate sys-
tem to the camera coordinate system using the equation
Pcam = RPobj + T.
As illustrated in Figure 2, iG-6DoF comprises three pri-
mary modules: an object detector, an initial pose estimator,
and a pose refiner. The object detector segments the ob-
ject region within the image (Section 3.2). Subsequently,
the initial pose estimator determines an initial rotation and
translation by identifying the most similar feature within a
multi-scale SO(3) group feature space (Section 3.3). Upon
the initial translation and rotation, the 3DGS pose refiner
computes a precise pose estimate (Section 3.4).
3.1. Preliminaries
Data Acquisition.
To implement our method,
we
require
a
set
of
reference
images
with
parameters
{Iref
i
, Rref
i
, T ref
i
}Nr
i=1, where I, R, and T represent the im-
age and its corresponding camera extrinsics. Nr is the num-
ber of reference images. Owing to off-the-shelf toolboxes
provided by OnePose [59] and ARKit [3], we can easily
manually annotate the 3D bounding box of an object in a
video sequence and obtain camera parameters.
3D Gaussian Splatting. 3DGS is a recent and innovative
technique for representing and rendering 3D scenes. They
6438

Figure 3. Detector architecture: We use the features from ref-
erence images as kernels to convolve with query image features,
generating heap maps. This heap maps are then processed by a
CNN to produce a object mask.
first recover camera poses and sparse 3D point clouds of
the scene from a sequence of captured images using Struc-
ture from Motion (SfM), and then construct 3D Gaussian
spheres based on these point clouds. Each 3D Gaussian is
parameterized by a 3D coordinate µ ∈R3, a 3D rotation
quaternion r ∈R4, a scale vector s ∈R3, an opacity fac-
tor α ∈R, and spherical harmonic coefficients h ∈Rk,
where k denotes the number of degrees of freedom. Finally,
We can calculating the loss between the rendered image and
the real image, and using the backpropagation algorithm to
optimize the Gaussian parameters.
3.2. Object Detector
Our detector builds on the TGID [2] and Gen6D [42] frame-
works, which apply a correlation-based object detector.
Since we need to construct a 3DGS model of the object,
a more precise object mask is required, so we replace the
output bounding box with a segmentation mask. Specifi-
cally, we set a per-pixel confidence score, and pixels are
considered part of the target object when their confidence
exceeds a certain threshold. The core idea is to use TDID
embeddings to convolve the feature map of the reference
image over the query image features, calculating the cor-
relation for each pixel. A threshold is then set to identify
high-confidence pixels as belonging to the target object, re-
sulting in the object’s mask.
As shown in Figure 3, our detector architecture employs
a shared feature extractor, like VGG-11 [56], to extract fea-
tures from the target and scene images. These features are
subsequently combined in a joint embedding layer. Finally,
a set of convolutions predicts class scores and segmentation
mask regression parameters for a set of default anchor boxes
on the embedding feature map.
3.3. Initial Pose Estimator
The primary objective of the initial pose estimator is to se-
lect the most accurate pose hypothesis from a set of candi-
dates. Previous methods often relied on template matching,
where the closest match to the query image is selected from
a reference image database. However, due to the sparsity
of viewpoints in the reference image set, this approach can
lead to significant errors, particularly when the query im-
age’s viewpoint differs substantially from those in the refer-
ence set.
As shown in Figure 4, we first apply multi-scale data
augmentation to the reference images to enrich the candi-
date pose database. Specifically, each reference image is
rotated kπ/2 clockwise and scaled by factors of 2 and 0.5,
respectively. Inspired by RoReg [64] and GIFT [43], we uti-
lize rotation-equivariant feature to embed the reference im-
ages. Specifically, we treat the RGB color values as 3D co-
ordinates in a three-dimensional space, establishing a map-
ping from the color space to the 3D space, so that we can
apply point set feature extractor PointNet [51] as backbone
to extract 3D feature from 2D image. To prevent the same
color at different positions from being mapped to a single
3D point, we added positional encoding [44]. Subsequently,
we define a neighborhood space on the 2D image and em-
ploy a icosahedral group feature encoder to encode the ref-
erence images, yielding a multi-scale group feature space
{V ref
i
}Nr
i=1 ∈R60×Nr. In a similar manner, a feature vector
V que ∈R60 is extracted from the query image. To obtain
the initial pose parameters, we compute the cosine similar-
ity between V que and each reference feature vector V ref
i
. The reference vector with the highest similarity score is
selected, and its associated pose parameters are assigned as
the initial estimate.
Group Feature Space. Given a target image that has been
segmented using a mask, we employ our proposed method
to project each pixel within the segmentation mask onto a
corresponding 3D point in space, resulting in a set of 3D
points denoted as {Pi ∈R3}. To establish local neighbor-
hoods for each pixel, we define NP = {pi|∥pi −p∥< 5},
where NP represents the neighborhood of pixel p, and pi
denotes the position of a neighboring pixel located within a
5-pixel radius of p.
Given an input neighborhood point set NP , we apply an
element g of the icosahedral group G to generate rotated
point sets. Each rotated point set is processed by a shared
point set feature extractor, denoted as ϕ, to produce an n-
dimensional feature vector, expressed as:
  f_0 ( g) = \p hi (T_g \circ N_{P}), 
(1)
where f0 : G →Rn0 represents the output group feature for
point p, and Tg ◦NP denotes the application of rotation g to
the point set NP . Since the icosahedral group G comprises
60 rotations, the group feature f0 can be efficiently stored as
a 60 × n0 matrix. We apply PointNet [51] as backbone ϕ.
Then, we adopt a localized icosahedral group convolution
for feature embedding:
  [f_{l+1} (
g)
]
_
j=
\sum _i^{1 3 }w_{j,i}^Tf_l(h_ig)+b_j, 
(2)
6439

Similarity score map
Initial Pose
Detector
Extra.
Extra.
Extra.
…
Icosahedral group
Emb.
Emb.
Emb.
…
Extra.
Icosahedral group
Emb.
Multi-scale Data Augmentation
Group Feature Space
Group Feature Descriptor 
Extra.
Emb.
Refrences Images
Input image
Group Feature Embedder
Group Feature Extractor
Figure 4.
Architecture of the pose estimator.
We first ap-
ply multi-scale image augmentations to the reference images, in-
cluding rotations and scaling. Subsequently, we extract rotation-
equivariant features using the icosahedral group. Finally, the op-
timal initial pose is determined by comparing the similarity of the
feature vectors.
where l denote the layer index, fl(g) ∈Rnl and fl+1(g) ∈
Rnl+1 represent the input and output feature vectors, respec-
tively. [·]j extracts the j-th element from a vector. The
neighborhood set hi ∈H is denoted by where each is an
element of the group G. The trainable weight associated
with the i-th neighbor and j-th output feature is represented
by wj,i ∈Rnk, with being the corresponding bias bj. Note
that j ranges from 1 to n, indexing the output feature dimen-
sions. Given the group’s closure property, the composition
hig is also an element of G.
3.4. Pose Refiner
The pose refiner aims to refine an initial pose Tinit with an in-
put image. To achieve this, we leverage the high rendering
quality of 3DGS [33]. By iteratively rendering and com-
paring the rendered image with the input image, we pro-
gressively update the pose estimate until convergence. As
shown in Figure 5, the refiner takes as input T k
init and a 3DGS
model and predicts an updated pose T k+1
init
= T k+1
∆
T k
init and
a rendered images Ik+1
render. We iteratively refine the pose
parameters by minimizing the SSIM loss between the ren-
dered and input images Ique. Similar to [35, 36, 39], we
decompose T k+1
∆
into its rotational component Rk+1
∆
and
translational component T k+1
∆
(Note that T ∈SE(4) and
T ∈R3). To decouple the rotation and translation compo-
nents, the rotation center is shifted from the camera origin
to the object’s center, as determined by the current pose es-
timate. This modification ensures that applying a rotation
does not alter the object’s position within the camera frame.
The iterative optimization process of the refiner is as fol-
lows:
  \b
e
g in {sp
l it}
 
\m athcal {T
}
_ \ De lta ^ {k+1}
= \ar g \
min 
_
{{T}_\Delta
 
^ {k +1}
}
\ m ath cal { L}_{T}(\mathcal {R}_{gs}(T_{\Delta }^{k+1}+\mathcal {T}^{k},GSM),{I}_{que})\\ +\arg \min _{{R}_\Delta ^{k+1}}\mathcal {L}_{R}(\mathcal {R}_{gs}({R}_\Delta ^{k+1}\odot (T_{\Delta }^{k+1}+\mathcal {T}^{k}),GSM),I_{que}), \end {split} 
(3)
Initial Pose
3DGS 
Render
Refiner Model

1
k
T

1
k
R
Figure 5. Diagram of pose refiner. Given the pose from the pre-
vious time step T k
init, we decouple T k+1
∆
into Rk+1
∆
and T k+1
∆
for
separate estimation. We first estimate the translation vector, fol-
lowed by the rotation vector. This process is iterated until reaching
the specified number of steps or convergence.
where Rgs denotes the 3D gaussian renderer, ⊙signifies the
application of a rigid rotation and GSM is a 3DGS model.
3.5. Loss Functions
We use the widely adopted Binary Cross Entropy (BCE)
loss to train our detector for pixel-wise segmentation, de-
noted as Ldet:
  \m a thcal { L
}_{det} = \mathcal {L}_{BCE}(M, \bar {M}), \\ 
(4)
where M and ¯
M represent the predicted and ground truth
segmentation masks, respectively.
We
apply
the
descriptor
construction
loss
from
RoReg [64] to train pose estimator.
Given a batch of
ground-truth image pairs (Iq, Ir) and their corresponding
ground-truth rotations RIq, we compute the outputs of
the group feature embedder, which include the rotation-
invariant descriptors (dIq, d+
Ir), the rotation-equivariant
group features (fIq, f +
Ir), and the corresponding ground
truth coarse rotations g+
Ir. For every sample in the batch,
we compute the loss:
  \ma thc al {
L}_1(\bm { d },\
bm {d}^+,\bm {D
}^-)=\frac 
{
e^{||\bm {d}-\bm {d}^+||_2}-\underset {\bm {d}^- \in \bm {D}^-}{\min }e^{||\bm {d}-\bm {d}^-||_2}}{e^{||\bm {d}-\bm {d}^+||_2}+\underset {\bm {d}^- \in \bm {D}^-}{\sum }e^{||\bm {d}-\bm {d}^-||_2}} \label {invloss} 
(5)
  \ma t hc al { L}_2( \bm {f},\ bm
 
{f}
^+,g^+)= -l og(\frac {e^{\left <\bm {f},P_{g^+}\circ \bm {f}^+\right >}}{\underset {g \in G}{\sum }e^{\left <\bm {f},P_{g}\circ \bm {f}^+\right >}}) \label {eqvloss} 
(6)
  \mat h c a l {L} _{g rou p }=\la m bd a *\mathcal {L}_1(\bm {d},\bm {d}^+,\bm {D}^-)+\mathcal {L}_2(\bm {f},\bm {f}^+,g^+), 
(7)
where the subscript Ir is omitted for simplicity. Equa-
tion 5 supervises the rotation-invariant descriptor, where d
is the descriptor, d+ is the matched descriptor, D−are the
negative descriptors in the batch, and | · |2 is the L2 norm.
6440

Finally, based on the aforementioned Lpose defined as
  \ma t hc a l {L}_{pose} = \mathcal {L}_{R}+\mathcal {L}_{T} 
(8)
  \ mathcal {L}_{T} = \mathcal {L}_{SSIM}, 
(9)
  \ mathc a l {L}_{R} = \mathcal {L}_{SSIM}+ \mathcal {L}_{MS-SSIM}, 
(10)
where LSSIM
and LMS−SSIM
represent the SSIM-
based [68] and multi-scale SSIM-based [67] loss functions,
respectively. The overall loss function of our method is:
  \mat h cal {L } _{total} = \lambda _{1}\mathcal {L}_{det}+\lambda _{2}\mathcal {L}_{group}+\lambda _{3}\mathcal {L}_{pose}, 
(11)
where λ{1,2,3} represent the hyperparameters, which we set
to 0.3, 0.2, and 0.5, respectively.
4. Experiments
Training Data.
We employ the synthetic MegaPose
dataset [36] for training, which generated using Blender-
Proc [17] with 1,000 diverse objects from the Google
Scanned Objects dataset [18], comprising one million syn-
thetic RGB images.
Evaluation data.
We evaluate our proposed model on
three widely used benchmarks:
LINEMOD, OnePose-
LowTexture, and GenMOP, to demonstrate its generaliza-
tion ability across diverse object categories and scenes. The
LINEMOD dataset [29], comprising 13 objects, is a com-
monly employed benchmark for 6D object pose estimation.
Adhering to the established protocol [25, 37, 42, 47, 59],
the training partition of LINEMOD is designated as refer-
ence data, while the testing partition serves as the evalu-
ation set. The OnePose-LowTexture dataset [59] presents
a challenging scenario with objects exhibiting minimal or
absent texture, containing eight scanned objects for evalua-
tion. The GenMOP [42] dataset comprises ten distinct ob-
jects. For each object, two video sequences were captured
under varying environmental conditions, including back-
ground and lighting variations. Each video sequence is seg-
mented into approximately 200 individual images
Metrics. To evaluate our model, we employ the commonly
used Average Distance (ADD) metric [29] and projection
error. For ADD, we calculate both the recall rate at 10%
of the object diameter (ADD-0.1d) and the Area Under the
Curve (AUC) within a 10 cm radius (ADD-AUC). Regard-
ing projection error, we compute the recall rate at a pixel
threshold of 5 (Prj-5).
Setups.
We
primarily
compare
iG-6DoF
against
Gen6D [42], Cas6D [47], Onepose [59], GS-Pose [8] and
MFOS [37]. To ensure a fair comparison and demonstrate
the effectiveness of each module, we evaluated our initial
pose estimator and pose refiner on the aforementioned three
separate datasets.
4.1. Results on LINEMOD
We first evaluate iG-6DoF on a subset of LINEMOD objects
against OSOP [55], Gen6D [42], Cas6D [47], GS-Pose [8]
and LocPoseNet [74] and present quantitative results in
Table 1. Without pose refinement, iG-6DoF achieves an
ADD(S)-0.1d of 45.99%. After refinement, performance
improves to 83.22%.
Then, we compare our method against state-of-the-art
one-shot approaches, including Gen6D [42], OnePose [59],
OnePose++ [25] and MFOS [37], using ADD(S)-0.1d and
Proj2D metrics.
As indicated in Table 2, our method
consistently outperforms these baselines. Notably, unlike
OnePose and OnePose++ which rely on pre-reconstructed
3D shape models, our approach operates without requiring
prior 3D object knowledge. This leads to improvements of
8.2% and 2.3% on ADD-S and Proj2D, respectively, over
the strongest baseline.
4.2. Results on OnePose-LowTexture
We then evaluate iG-6DoF on the challenging OnePose-
LowTexture dataset [25], comparing it against state-of-the-
art baselines including OnePose [59], OnePose++ [25],
Gen6D [42], and the instance-specific PVNet [48].
Ta-
ble 3 presents quantitative standard cm-degree accuracy
for different thresholds, demonstrating the superior perfor-
mance of iG-6DoF. Specially, our method outperforms all
baseline methods at the threshold 1cm/1deg and 5cm/5deg.
OnePose++ eliminates reliance on local feature matching
by adopting the keypoint-free LoFTR [58], improving per-
formance Onepose to 72.1%, yet still falls short of iG-6DoF
despite requiring ground-truth bounding boxes.
4.3. Results on GenMOP
We finally compare iG-6DoF with generalizable image-
matching based ObjDesc [1], two instance-specific estima-
tors PVNet [48] and RLLG [9] and model-free method
Gen6D [42] on GenMOP dataset. To ensure a fair com-
parison, we adopt the same experimental setup as Gen6D,
using the original reference images without data augmenta-
tion. All testing data is unseen during the training of iG-
6DoF, Gen6D, and ObjDesc. For PVNet and RLLG, we
train a separate model for each object. Quantitative results
are shown in Table 5, our method essentially achieves the
current state-of-the-art performance.
4.4. Ablation Study
To verify the effectiveness of each module in our pro-
posed method, we conducted ablation studies on the widely
used LM [29] dataset. Performance is assessed using the
BOP [30] metric.
Ablation study on the pose estimator. To demonstrate
the designs in the initial pose estimator, we conduct ablation
studies on the LM dataset and results are shown in Table 4
6441

Method
Pose Refiner
cat
duck
bvise
cam
driller
Avg.
OSOP [55]
w/o
34.43
20.08
50.41
32.30
43.94
36.23
Gen6D [42]
15.97
7.89
25.48
22.06
17.24
17.73
LocPoseNet [74]
-
-
-
-
-
27.27
GS-Pose [8]
47.80
30.70
63.47
44.61
47.27
46.77
iG-6DoF (Ours)
46.53
31.61
61.97
41.55
48.31
45.99
OSOP [55]
w/
42.54
22.16
55.59
36.21
49.57
42.21
Gen6D [42]
60.68
40.47
77.03
66.67
67.39
62.45
Cas6D [47]
60.58
51.27
86.72
70.10
84.84
70.72
iG-6DoF (Ours)
80.89
66.39
95.88
87.23
85.69
83.22
Table 1. Quantitative results on a subset of objects from the LINEMOD dataset [29] in terms of ADD(S)-0.1d. The best performance is
highlighted in bold.
Method
Object Name
Avg.
ape
benchwise
cam
can
cat
driller
duck
eggbox∗
glue∗
holepuncher
iron
lamp
phone
ADD(S)-0.1d
Gen6D
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
OnePose
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
OnePose++
31.2
97.3
88.0
89.8
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
MFOS
47.2
73.5
87.5
85.4
80.2
92.4
60.8
99.6
69.7
93.5
82.4
95.8
51.6
78.4
Ours
64.3
96.3
88.6
92.1
83.2
88.6
73.3
99.6
81.3
94.3
81.3
88.6
73.1
85.1
Proj2D
OnePose
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
OnePose++
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
MFOS
97.1
94.1
98.4
98.2
98.4
95.7
96.3
99.0
94.8
99.3
94.6
94.2
88.9
96.1
Ours
97.8
99.2
97.8
98.2
99.1
91.5
97.6
99.3
95.1
98.9
95.2
95.6
90.3
96.6
Table 2. Results on LINEMOD and comparison with other model-free baselines. Symmetric objects are indicated by ∗. The best perfor-
mance is highlighted in bold, while the second best results are underlined.
OnePose-LowTexture
GT-Mask
1cm-1deg
3cm-3deg
5cm-5deg
HLoc (SPP + SPG)
!
13.8
36.1
42.2
HLoc (LoFTR∗)
!
13.2
41.3
52.3
PVNet
!
15.1
33.2
48.6
Gen6D
%
11.5
31.6
25.9
OnePose
!
12.4
35.7
45.4
OnePose++
!
16.8
57.7
72.1
MFOS
!
14.1
54.3
74.2
Ours
%
16.6
53.2
73.5
Ours
!
17.2
55.6
75.1
Table 3. Comparison with Baselines on OnePose-LowTexture. We
denote the methods relying on an GT object mask as ’GT-Mask’.
(C1 and C2). We select ObjDesc [70] and Gen6D [42] for
the comparison baseline. The results show that our method
is capable of achieving a more accurate initial pose be-
cause we search within a multi-scale pose hypothesis space,
whereas the baseline method only selects the most similar
candidate from the reference image as the initial pose.
Ablation study on the pose refiner. To highlight the
advantages of our 3DGS-based refiner for unseen objects
over other 6D pose estimation methods, such as those used
Row
Method
LM
ARV SD
ARMSSD
ARMSPD
A0
iG-6DoF
0.549
0.689
0.853
B1
A0: GS refiner→Gen6D refiner
0.538
0.672
0.812
B2
A0: GS refiner→DeepIM refiner
0.512
0.638
0.779
C1
A0: Pose Estimator →Objdesc selector
0.424
0.503
0.637
C2
A0: Pose Estimator →Gen6D selector
0.432
0.511
0.669
D1
A0: w/o data augmentation
0.521
0.624
0.801
D2
B1: w/o data augmentation
0.501
0.613
0.786
D3
B2: w/o data augmentation
0.478
0.601
0.732
E0
A0: Nr →16
0.432
0.492
0.766
E1
A0: Nr →32
0.446
0.624
0.789
E2
A0: Nr →64
0.533
0.657
0.834
E3
A0: Nr →128
0.587
0.712
0.866
Table 4. Ablation study under BOP setup on LM dataset.
in Gen6D and DeepIM [39, 42], we present results in Ta-
ble 4 (B1 and B2). For the baseline refiner, DeepIM [39],
we treat the reference image selected by our selector as the
rendered image and use DeepIM to match it with the query
image to update the pose. It is important to note that further
refinement using additional iterations of DeepIM is not fea-
sible, as there is no object model available to render a new
image based on the updated pose. All refiners, including
DeepIM, Gen6D, and our 3DGS-based refiner, are trained
on the same dataset. The results indicate that our 3DGS-
based refiner demonstrates superior generalization capabil-
6442

Input Image
Pred. Mask
3DGS Render
6D Pose
Figure 6. Qualitative results captured by us in real-world scenes. More visual results, discussion and analysis are provided in the supple-
mentary material.
Metrics
Method
Object Name
avg.
Chair
PlugEN
Piggy
Scissors
TFormer
ADD-0.1d
ObjDesc [70]
3.50
5.14
14.07
1.25
7.54
8.55
Gen6D w/o Ref.
14.00
7.48
39.70
16.81
11.51
17.90
Gen6D w Ref.
61.50
19.63
75.38
32.76
62.70
50.39
Ours w/o Ref.
46.32
17.93
71.84
29.57
55.92
44.32
Ours w Ref.
66.83
32.61
79.84
40.35
60.81
56.10
Proj2D
ObjDesc [70]
4.00
10.75
4.52
18.53
8.33
9.23
Gen6D w/o Ref.
11.50
40.65
33.17
34.05
64.29
36.73
Gen6D w Ref.
55.00
72.90
92.96
93.53
98.81
82.64
Ours w/o Ref.
48.91
65.93
84.6
81.34
81.61
72.49
Ours w Ref.
66.83
79.64
95.11
92.18
97.92
86.34
Table 5. Performance on the GenMOP dataset. “Ours w/o Ref.”
means not using the pose refiner in the iG-6DoF estimator.
ities on unseen objects compared to DeepIM and Gen6D.
Ablation study on data augmentation. To demonstrate
the impact of our data augmentation module, we selected
B0, B1, and B2 as baselines and compared the quantita-
tive results before and after removing the data augmenta-
tion module. As shown in Table 4 (D1, D2 and D3), the
results indicate that our data augmentation module signifi-
cantly improves overall performance.
Ablation study on number of reference images. Fi-
nally, we evaluated the impact of the number of reference
images on our method’s performance by setting the refer-
ence image count to 16, 32, 64, and 128 in Table 4(E0 to
E3). As expected, the model’s performance improves with
an increasing number of reference images, aligning with our
intuition. Thanks to the effectiveness of our data augmen-
tation module, even with a smaller number of reference im-
ages, our method still achieves commendable results.
Runtime.
iG-6DoF processes each image (resolu-
tion 480×640) in approximately 0.5 seconds on a desktop
equipped with an Intel Xeon Silver 4310 CPU @ 2.10GHz
and an Nvidia GeForce RTX 3090 GPU. This includes 0.12
seconds for object detection, 0.01 seconds for initial pose
estimation, and 0.4 seconds for pose refinement.
5. Conclusion
In this paper, we introduced a novel end-to-end pose esti-
mation method based on 3D Gaussian Splatting without the
object’s CAD model. Our method demonstrates strong gen-
eralization capabilities, effectively estimating the pose of
unseen objects with only a set of reference images. Unlike
previous work, which always relies on precise 3D models,
additional supervisory data, and struggles with significant
object translations or rotations, our method is robust and
versatile. Our method consistently achieves state-of-the-art
performance, as evidenced by results on the widely used
benchmarks. Furthermore, we conducted experiments on
our captured scenes, validating our method’s generalization
potential and efficacy in diverse scenarios.
6443

6. Acknowledgments
This work is partially supported by National Nature Science
Foundation of China (No.62372336 and No.62172309).
References
[1] Adel Ahmadyan and Liangkai Zhang. Objectron: A large
scale dataset of object-centric videos in the wild with pose
annotations. In CVPR, 2021. 6
[2] Phil Ammirato, Cheng-Yang Fu, Mykhailo Shvets, Jana
Kosecka, and Alexander C Berg. Target driven instance de-
tection. arXiv preprint arXiv:1803.04610, 2018. 4
[3] Apple.
Arkit.
https://developer.apple.com/
augmentedreality/, 2017. 3
[4] Gil Avraham, Julian Straub, Tianwei Shen, Tsun-Yi Yang,
Hugo Germain, Chris Sweeney, Vasileios Balntas, David
Novotny, Daniel DeTone, and Richard Newcombe. Nerfels:
renderable neural codes for improved camera pose estima-
tion. In CVPR, pages 5061–5070, 2022. 3
[5] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. In CVPR, pages 5470–
5479, 2022. 3
[6] P.J. Besl and Neil D. McKay. A method for registration of
3-d shapes. IEEE TPAMI, 1992. 1
[7] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and
Victor Adrian Prisacariu. Nope-nerf: Optimising neural ra-
diance field with no pose prior. In CVPR, pages 4160–4169,
2023. 3
[8] Dingding Cai and Janne Heikkil¨a. Gs-pose: Cascaded frame-
work for generalizable segmentation-based 6d object pose
estimation. arXiv preprint arXiv:2403.10683, 2024. 1, 6, 7
[9] Ming Cai and Ian Reid. Reconstruct locally, localize glob-
ally: A model free method for object pose estimation. In
CVPR, 2020. 6
[10] Tuo Cao and Fei Luo. Dgecn: A depth-guided edge convolu-
tional network for end-to-end 6d pose estimation. In CVPR,
2022. 1
[11] Tuo Cao, Wenxiao Zhang, Yanping Fu, Shengjie Zheng, Fei
Luo, and Chunxia Xiao. Dgecn++: A depth-guided edge
convolutional network for end-to-end 6d pose estimation via
attention mechanism.
IEEE Transactions on Circuits and
Systems for Video Technology, 34(6):4214–4228, 2023. 1
[12] Andrea Caraffa, Davide Boscaini, Amir Hamza, and Fabio
Poiesi. Freeze: Training-free zero-shot 6d pose estimation
with geometric and vision foundation models. In European
Conference on Computer Vision, pages 414–431. Springer,
2024. 2
[13] Pedro Castro and Tae-Kyun Kim. Posematcher: One-shot 6d
object pose estimation by deep feature matching. In ICCVW,
2023. 1, 2
[14] Jianqiu Chen and Mingshan Sun.
Zeropose:
Cad-
model-based zero-shot pose estimation.
arXiv preprint
arXiv:2305.17934, 2023. 1, 2
[15] Kai Chen and Qi Dou. Sgpa: Structure-guided prior adapta-
tion for category-level 6d object pose estimation. In ICCV,
2021. 1
[16] Shin-Fang Chng, Sameera Ramasinghe, Jamie Sherrah, and
Simon Lucey. Garf: gaussian activated radiance fields for
high fidelity reconstruction and pose estimation. arXiv e-
prints, pages arXiv–2204, 2022. 3
[17] Maximilian Denninger,
Martin Sundermeyer,
Dominik
Winkelbauer, Youssef Zidan, Dmitry Olefir, Mohamad El-
badrawy, Ahsan Lodhi, and Harinandan Katam.
Blender-
proc. arXiv preprint arXiv:1911.01911, 2019. 6
[18] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kin-
man, Ryan Hickman, Krista Reymann, Thomas B McHugh,
and Vincent Vanhoucke. Google scanned objects: A high-
quality dataset of 3d scanned household items. In 2022 In-
ternational Conference on Robotics and Automation (ICRA),
pages 2553–2560. IEEE, 2022. 6
[19] Zhaoxin Fan and Zhenbo Song. Object level depth recon-
struction for category level 6d object pose estimation from
monocular rgb image. In ECCV, 2022. 1
[20] Martin A. Fischler and Robert C. Bolles. Random sample
consensus. COMMUN ACM, 1981. 1
[21] Minghao Gou and Haolin Pan.
Unseen object 6d pose
estimation: A benchmark and baselines.
arXiv preprint
arXiv:2206.11808, 2022. 1
[22] Frederik Hagelskjær and Rasmus Laurvig Haugaard. Key-
matchnet:
Zero-shot pose estimation in 3d point clouds
by
generalized
keypoint
matching.
arXiv
preprint
arXiv:2303.16102, 2023. 1
[23] Huasong Han, Kaixuan Zhou, Xiaoxiao Long, Yusen Wang,
and Chunxia Xiao. Ggs: Generalizable gaussian splatting
for lane switching in autonomous driving.
arXiv preprint
arXiv:2409.02382, 2024. 3
[24] Kaiming He and Georgia Gkioxari. Mask r-cnn. In ICCV,
2017. 2
[25] Xingyi He and Jiaming Sun. Onepose++: Keypoint-free one-
shot object pose estimation without cad models. In NeurIPS,
2022. 1, 2, 6
[26] Yisheng He and Wei Sun.
Pvn3d: A deep point-wise 3d
keypoints voting network for 6dof pose estimation. In CVPR,
2020. 1
[27] Yisheng He and Yao Wang. Fs6d: Few-shot 6d pose estima-
tion of novel objects. In CVPR, 2022. 1, 2
[28] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm,
George Drettakis, and Gabriel Brostow. Deep blending for
free-viewpoint image-based rendering. ACM TOG, 37(6):1–
15, 2018. 3
[29] Stefan Hinterstoisser and Vincent Lepetit.
Model based
training, detection a