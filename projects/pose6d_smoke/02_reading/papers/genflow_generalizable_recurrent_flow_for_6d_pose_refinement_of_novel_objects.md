# GenFlow: Generalizable Recurrent Flow for 6D Pose Refinement of Novel Objects

> 2024 · id: arxiv:2403.11510 · arXiv: 2403.11510 · pdf: https://arxiv.org/pdf/2403.11510 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

GenFlow: Generalizable Recurrent Flow for 6D Pose Refinement of Novel
Objects
Sungphill Moon*
Hyeontae Son*
Dongcheol Hur
Sangwook Kim
NAVER LABS
{sungphill.moon, son.ht, dongcheol.hur, o.s.w.a.l.k}@naverlabs.com
Abstract
Despite the progress of learning-based methods for 6D
object pose estimation, the trade-off between accuracy and
scalability for novel objects still exists. Specifically, pre-
vious methods for novel objects do not make good use of
the target object’s 3D shape information since they focus
on generalization by processing the shape indirectly, mak-
ing them less effective. We present GenFlow, an approach
that enables both accuracy and generalization to novel ob-
jects with the guidance of the target object’s shape. Our
method predicts optical flow between the rendered image
and the observed image and refines the 6D pose iteratively.
It boosts the performance by a constraint of the 3D shape
and the generalizable geometric knowledge learned from an
end-to-end differentiable system. We further improve our
model by designing a cascade network architecture to ex-
ploit the multi-scale correlations and coarse-to-fine refine-
ment. GenFlow ranked first on the unseen object pose esti-
mation benchmarks in both the RGB and RGB-D cases. It
also achieves performance competitive with existing state-
of-the-art methods for the seen object pose estimation with-
out any fine-tuning.
1. Introduction
Estimating accurate 6D object pose from an RGB or
RGB-D image is crucial for robotic tasks and augmented
reality applications. In the standard setup of 6D object pose
estimation, 3D models of target objects are given and avail-
able in training and testing. Most state-of-the-art algorithms
are learning-based and require training the model per object
or dataset containing few objects. However, these methods
are difficult to handle novel objects since they memorize the
3D shapes of the targets and suffer from the training cost to
support them.
Recent works [5, 37, 51, 54] have proposed generalizable
models that can estimate the 6D pose of novel objects, i.e.,
*These authors contributed equally.
Figure 1. Examples of 6D pose estimation of novel objects. Our
method estimates correspondences between the input image and
rendered image, and 6D object pose.
objects unseen during training. Specifically, OSOP [54] and
MegaPose [37] show remarkable performance with the gen-
eralizable pose refinement. Their refinement methods have
the scalability to novel objects since the networks learn to
compare the rendered image and the observed image with-
out memorizing 3D shapes. However, their refiners avail
themselves of 3D shape information indirectly. OSOP re-
finer is trained by the surrogate 2D-2D matching loss, which
makes it suboptimal for 6D pose estimation. MegaPose re-
finer regresses the relative 6D pose between the rendered
image and the input, but the regression makes no use of the
projective geometry.
Our research question is Is there a better design for gen-
eralizable pose refinement that takes advantage of shape
information?.
Recently, 2D optical flow-based methods
[20, 31] show noticeable performance on 6D pose esti-
mation. These methods estimate dense 2D-2D correspon-
dences between the rendered image and the observed im-
age. The estimated correspondences are lifted to 2D-3D
correspondences on the 3D shape of the target object, then
the 6D pose is recovered using a PnP solver [31] or pose
regressor networks [20]. The optical flow-based pose re-
finement provides a clue for the generalization due to its
robustness to the domain shift [31] and shows the potential
to utilize 3D information in the model [20].
We propose GenFlow, the optical flow-based iterative
arXiv:2403.11510v1  [cs.CV]  18 Mar 2024

refinement for the generalizable object pose estimation
guided by the 3D shape information of target objects. Given
an image and initial 6D pose of an object in the image,
GenFlow estimates 2D dense correspondences between the
rendered image and the input image and refined 6D pose
through a PnP layer. This series of processes is performed
iteratively, so it can improve accuracy and robustness to out-
liers [43].
We adopt the shape-constraint recurrent flow framework
inspired by [20] for our model to use 3D information. This
framework utilizes the shape information implicitly to enjoy
both generalization and performance. Its end-to-end differ-
entiable system learns to estimate the optical flow, confi-
dence, and object pose. These estimations are updated itera-
tively complying with the 3D shape. We further introduce a
cascade network design to make use of the multi-scale cor-
relation volumes. It takes advantage of the coarse-to-fine
refinement and helps the model estimate a more accurate
pose.
We validate our method on various datasets provided in
the BOP challenge [28]. Our model is designed to estimate
6D pose for the RGB input but is also applicable to the
RGB-D case with the RANSAC-Kabsch algorithm [18, 34].
For both RGB and RGB-D cases, our method achieves the
overall best performance for the unseen object pose estima-
tion task using the default detection [50] provided in the
BOP challenge 2023. It is also notable that our method
achieves competitive performance with the state-of-the-art
methods for seen objects when used with better detection
results.
2. Related Work
Iterative Refinement DeepIM [41] pioneered the iterative
render-and-compare strategy for 6D pose estimation. It pre-
dicts the relative pose that better matches a rendered im-
age against the observed image. By iterative refinement,
the rendered and observed images become more similar.
CosyPose [38] improved the idea by introducing better pa-
rameterization for rotation [69], network architecture [60],
loss design [55, 62] and data augmentation. These iterative
methods show accurate results and the possibility of being
applied to the unseen objects [41]. In contrast to the previ-
ous works, CIR [43] introduces a novel differentiable PnP
layer to utilize the projective geometry with deep learning.
Unlike CIR, which assumes the RGB-D input, ours focuses
on the RGB input. For the RGB input case, CIR obtains the
rendered depth as the substitution of input depth for every
flow update, so the inference suffers from time inefficiency.
On the other hand, our refinement method requires fewer
renderings.
Optical Flow Estimation The classical approaches [2, 6,
29] dealt with optical flow estimation as a hand-crafted
optimization problem in pursuit of the balance of bright-
ness consistency and motion plausibility.
Deep learning
methods [12, 32, 33, 57, 66] emerged as promising alterna-
tives of the classical methods, achieving considerable per-
formance with faster inference times. Recently, RAFT [61]
introduced a recurrent deep network architecture which has
shown significant improvement over the previous methods
regarding accuracy, efficiency, and generalization. PFA [31]
applied the optical flow estimation to 6D pose refinement
for data-limited cases. Since learning to estimate dense op-
tical flow helps the networks to focus on the lower-level
information, it could be robust to domain shift. Follow-
ing PFA, SCFlow [20] further improved the performance
by introducing an end-to-end trainable system using a pose
regressor and shape-constraint lookup. Our method capital-
izes on a differentiable PnP layer to the framework in order
to make use of projective geometry.
Unseen Object Pose Estimation Category-level 6D pose
estimation [7, 40, 48, 62] tackled the case of unseen ob-
jects of which category is given in advance. Since these
methods depend on the prior of the 3D shapes belonging
to specific category, they struggle with handling the objects
of unknown categories. Recently, several works proposed
class-agnostic generalizable pose estimators that can handle
unseen objects. A few works attempted the CAD model-
free pose estimation [23, 24, 45, 53, 58] to cover every-
day objects, but the results were only evaluated on datasets
without occlusion and cluttered scenes. [49, 59, 64, 65] fo-
cused on estimating the orientation of novel objects by com-
paring the deep image features of the rendered image and
the observed image. ZePHyR [51] proposed a generaliz-
able scoring function combined with hand-crafted features
[15, 47]. However, it uses the features selectively according
to the richness of texture in the datasets. ZeroPose [5] es-
timates 6D pose by hierarchical feature matching which re-
quires the depth input by its nature and pose refinement for
better performance. Like ours, OSOP [54] and MegaPose
[37] proposed CAD-based pose estimation methods that can
process RGB and RGB-D inputs. Specifically, MegaPose
achieved state-of-the-art performance by its iterative pose
refiner. The pose refiner leverages the multi-view rendered
images containing surface normals for utilizing rich infor-
mation about the novel object’s shape, anchor point, and co-
ordinate system. Our method refines the pose with the guid-
ance of the target object’s shape without rendering multi-
view images per refinement step.
3. Method
This section describes our methodology for 6D pose es-
timation of novel objects. Given an RGB image and 3D
models of target objects, which are unseen during training,
our goal is to estimate the 6D poses of the objects with re-
spect to the camera. Our pose estimation pipeline consists
of 3 stages: object detection, coarse pose estimation, and

Figure 2.
Visualization of the scores predicted by the coarse
model. Each dot denotes the sampled rotation, and its size repre-
sents the relative score of the coarse model for the corresponding
rotation. The green circles are boundaries of the area where the
refiner is interested. Figures show that the high-scoring poses are
clustered in the vicinity of the green circles.
pose refinement. Note that the detection of novel objects is
out of the scope of this paper. We first provide an overview
of the whole process and then focus on the pose refinement
method, which is the main contribution of this work.
3.1. Preliminaries
Given a calibrated RGB image I and a set of 3D models
of the target objects {Mi}, we estimate the 6D poses of
the visible objects in the image. A 6D pose is defined by
a matrix P representing the rigid transformation from the
object space to the camera space. Note that we assume the
input image is calibrated, i.e., the intrinsic camera matrix K
is known.
2D Object Detection Given an RGB image, the object de-
tection method produces a list of object detections consist-
ing of 2D bounding boxes and associated labels. For the
unseen object detection, we apply the CNOS [50] detection
leveraging the foundation models [36, 52]. To show the ex-
tensive results of our work, we also apply the YOLOX de-
tector [19] from GDRNPP [44] which is used for seen ob-
ject detection. For each detection, the image I is cropped
and further resized to the specific resolution in the follow-
ing steps. The intrinsic parameters are adjusted according
to the resized resolution. For convenience, the following
explanation focuses on the 6D pose estimation of a single
detected object of which the 3D model is M.
Coarse Pose Estimation Given object detection, the coarse
pose estimator predicts rough initial poses. We used the
classification-based method proposed in [37] for better gen-
eralization. Given an input image and an arbitrary pose P
for the 3D model M, the coarse model classifies whether
or not the pose P could be improved to the pose in proxim-
ity to the ground-truth by the refiner. It uses a pre-defined
set of 3D rotations of which the cardinality is N to gen-
erate the various arbitrary pose hypotheses. For each rota-
tion, a rough 3D translation is estimated for the points of
the object’s 3D model to be approximately inside the 2D
bounding box. We extract scores of the pose hypotheses
from the model and select the top-n scoring poses. Such
a classification-based estimation can handle novel objects,
but it suffers from the cost of rendering a large number of
images for the pre-defined rotations. Inspired by the prop-
erty of high-scoring rotations to be sparse and clustered as
shown in Fig 2, we introduce sampling using a Gaussian
mixture model (GMM) [1] for efficiency. First, we choose
the top k-scoring samples {Ri = (ψi, θi, ϕi)}k
i=1 from M
pre-defined rotations (Fig 2b left). Then we create a den-
sity distribution p(X) for the Euler angle X using a GMM
which satisfies that
  p( X
)
 
= \
s
u m _{i=1 }^{ k} {\frac {1}{k} \mathcal {N}(X|\mu _{i}, \Sigma )}, \mu _{i}=R_{i}
(1)
Finally, we sample additional M rotations from the density
distribution p(X) (Fig 2b right) and choose the best-scoring
n hypotheses among the 2M samples. This method renders
2M < N images while achieving better performance. We
used M = 104 and k = 16 for our experiments.
Pose Refinement Given a coarse pose estimate, we refine
the pose for the rendered image to match against the input
image crop. Starting from the coarse estimate P0 as an ini-
tial pose, we iteratively refine the pose using our refiner. We
denote the kth 6D pose estimate of the refiner as Pk. For
the kth iteration, we obtain an image Ir,k by rendering the
3D model M for the pose Pk−1. Our refiner then estimates
the 2D optical flow Fk between the image Ir,k and the in-
put image crop and the confidence weights Wk which are
used to solve the pose Pk. A detailed description is in the
following section.
3.2. GenFlow Refinement
3.2.1
Overview
We describe the specific operation of GenFlow refiner
to estimate the refined pose Pk from the previous result
Pk−1 but hereafter, we omit the k notation if possible for
convenience. First, we obtain a synthetic image by ren-
dering the 3D model M for the pose P and adjusted in-
trinsic parameters. Specifically, the results of the rendering
consists of the RGB image, depth image and binary ren-
dered mask, denoted by Ir ∈RH×W ×3, Dr ∈RH×W , and
Mr ∈{0, 1}H×W respectively. Then, we extract features
from both images and refine the pose with the GenFlow
module.
Feature Extraction We extract the image features from
the input image crop and the rendered image Ir with
the same feature encoder and the context features from

Figure 3. Overview of GenFlow refinement. We visualize the process of feature extraction and GenFlow update in the kth refinement.
TC2O refers to a transformation that maps the coordinates from camera space to object space.
the Ir. The feature maps from the ith layer of the fea-
ture encoder are used to construct the correlation volumes
Ci ∈RHi×Wi×Hi×Wi by the dot product.
GenFlow Module For each correlation volume and con-
text feature map, we construct a module named “GenFlow
module”. A GenFlow module consists of a convolution-
based Gated Recurrent Unit (GRU) [9], a differentiable PnP
solver, and correlation lookup based on the pose-induced
flow as shown in the Fig 3. GenFlow module iteratively
updates the optical flow, confidence weights, and 6D pose.
For the jth update, the output of the module is the refined
6D pose Pj. Note that we use subscripts for the outer up-
dates and superscripts for the inner updates, i.e., GenFlow
updates. The detailed illustration of updating the 6D pose
with a GenFlow module for ith correlation volume is as fol-
lows.
3.2.2
GenFlow Update
GRU Update For the jth update, the GRU of the GenFlow
module consumes the previous hidden state hj−1, correla-
tion features rj−1 and context features. The output of the
GRU {hj, f j, uj, cj, sj} includes the updated hidden state
hj, optical flow f j, flow upsampling mask uj, certainty cj
and pose sensitivity sj. The flow f j ∈RHi×Wi×2 is up-
sampled to Fj to match the resolution of previous (i −1)th
feature map using the mask uj. We apply the convex com-
bination method of 3x3 neighbors for the optical flow pro-
posed in [61]. The certainty cj and pose sensitivity sj are
upsampled by bilinear interpolation, and then we multiply
them to obtain the confidence weights Wj. The head net-
works for certainty and pose sensitivity are supervised by
different losses, respectively. This explicit factorization of
the confidence weights helps to be robust to occlusion with
the help of certainty estimation as shown in Fig 4. The up-
sampled flow Fj and confidence weights Wj are used to
establish weighted 2D-3D correspondences.
Pose Update As we know the depth of synthetic image Dr
for the pose P0
k = [R0
k|t0
k] and the intrinsic camera matrix
K, we can compute the object-space 3D coordinate of each
pixel by
 
 
\
l
e
f
t ( 
\begin {ma tr
i
x
}
 
x
 
\ \ y \\ z \end {matrix} \right ) = \mathbf {R}_{k}^{\text {T}} (\mathbf {K}^{-1} \mathcal {D}_{r}(u, v) \left ( \begin {matrix} u \\ v \\ 1 \end {matrix} \right ) - \mathbf {t}_{k}) \label {eq:lifting}
(2)
where (u, v) is the 2D coordinate of a pixel and the Dr(u, v)
is depth of the pixel. Through this lifting process, we obtain
the 2D-3D correspondences for the flow Fj. We use Wj
for the confidence of correspondences and binary rendered
mask Mr to consider the object region only.
To obtain the 6D pose Pj, we utilize the iterative PnP
solver based on the Levenberg-Marquardt (LM) algorithm.

Figure 4. Visualization of the outputs of GRU. Our method factor-
izes the confidence weights into two terms: certainty and pose sen-
sitivity. Certainty estimation helps to the robustness to occlusion.
Pose sensitivity highlights the rich texture regions and extremities
of the object.
It optimizes the 6D pose by minimizing the sum of squared
weighted reprojection error:
  \beg
in 
{
s
p
l
i
t
} \ope ra t orna
m
e
 
*
{
a
r g mi
n}
_{
R
,
 t} \f rac {1}{2} \sum _{u} \sum _{v} \lVert \mathbf {W}^{j}(u,v) \times (\pi (R \left ( \begin {matrix} x \\ y \\ z \end {matrix} \right ) + t) \\ - (\left ( \begin {matrix} u \\ v \\ \end {matrix} \right ) + \mathbf {F}^{j}(u, v)) ) \rVert ^{2}, \end {split}
(3)
where π is the projection function with the intrinsic cam-
era parameters and (x, y, z)T is the 3D coordinate of the
pixel (u, v) computed by the equation 2. Due to its dif-
ferentiable nature, the PnP layer enables end-to-end train-
ing. However, the backpropagation by unrolling the itera-
tive optimization causes numerical instability, making the
training process hard. For stable end-to-end training, we
introduce the regularization method [4], which backpropa-
gates the gradient through the last step of a differentiable
optimization. Specifically, we first obtain a detached pose
P∗through 3 LM steps and update the pose via another iter-
ation of the Gauss-Newton algorithm to obtain the pose Pj.
Note that the gradient is not backpropagated through P∗.
Correlation Lookup We compute the pose-induced flow,
which is the 2D optical flow between the Ir and the image
with respect to the pose Pj. The pose-induced flow is calcu-
lated as the displacement of each pixel in the Ir by lifting
and reprojecting the pixel using the depth Dr. To embed
the 3D shape of the target object in the lookup operation
[20, 43], we use the pose-induced flow rather than the esti-
mated flow f j for indexing the correlation volume C. This
way of imposing a shape constraint on the model induces
the networks to predict dense matching to comply with the
target’s 3D shape information. From the lookup operation,
we obtain the correlation feature rj, which is the input of
Figure 5. Cascade design of GenFlow modules. Multiple Gen-
Flow modules are attached to each level on the feature pyramid.
The last updated flow FM
k and 6D pose PM
k from the higher-level
GenFlow module are used to initialize the flow and 6D pose of the
lower-level module. With the cascade architecture, the 6D pose is
recovered in a coarse to fine manner.
GRU for the next GenFlow update.
3.2.3
Cascade Architecture
Inspired by the previous methods that exploit the multi-
scale features [8, 42], we design a cascade architecture us-
ing multiple GenFlow modules. Given correlation volumes
{Ci}, we assign a GenFlow module to each volume. Then,
we use the 2D optical flow and refined 6D pose of the last
inner updates to initialize those of the next module as shown
in Fig 5. This cascade architecture takes advantage of pro-
gressive refinement, which makes use of the high-level se-
mantics to the low-level fine details. We found that it im-
proves the accuracy of 6D pose rather than using a single
GenFlow module.
3.3. Training
Dataset Both the coarse and refiner models are trained us-
ing the large dataset provided by MegaPose [37]. It is a
large synthetic dataset comprising 2 million RGB-D im-
ages generated by BlenderProc [11], which contains di-
verse 3D models with ground-truth 6D pose annotations.
The 3D shapes are collected from ShapeNet [3] and Google
Scanned Object (GSO) [14] datasets, and we trained our
model on both of the datasets.
Refiner Model Our refiner model is trained similarly to the
previous iterative refinement methods [37, 38]. Given an
image with the ground-truth 6D pose Pgt for the 3D model
M, we perturb the pose to generate an initial pose Pinit
by applying random noises to the translation and rotation of
the pose respectively. Translation noise is sampled from a
normal distribution with a standard deviation of (0.01, 0.01,
0.05) centimeters, and rotation noise is sampled from an-
other normal distribution of Euler angles with a standard
deviation of 15 degrees in each axis. The refiner model

learns to predict the pose Pgt from the perturbed initial pose
Pinit.
Coarse Model Given an input RGB image I and a pose
P, a coarse model is trained to classify whether the refiner
could estimate the 6D pose in proximity to the ground truth
pose from the initial pose P. We randomly generate the
positive and negative pose samples and train the network to
classify them using the binary cross entropy loss. For the
positive samples, we use the same method used to gener-
ate the perturbed initial pose in the refiner model. Negative
samples are generated using the method proposed in [37].
3.4. Implementation Details
Coarse Estimation We use 224×224 resolution for the re-
sized input crop and the synthetic image Ir. The coarse
model is implemented using the ConvNeXt backbone [46]
and fully connected head. We used the ConvNext model
pre-trained on ImageNet [10] classification to initialize the
networks’ weights. For inference, we select top-n hypothe-
ses using our GMM-based sampling strategy with M =
104.
GenFlow Refinement First, we obtain a synthetic image
Ir of which resolution is 256×256 and an image crop re-
sized to the same resolution as the synthetic image with
adjusted intrinsic parameters. Like the coarse model, we
use the pre-trained ConvNeXt for the feature and con-
text encoders.
The feature maps from the first two lay-
ers of the feature encoder are used to construct correlation
volumes C1 ∈R
H
4 × W
4 × H
4 × W
4 and C2 ∈R
H
8 × W
8 × H
8 × W
8 .
Our GRU is implemented with a ConvGRU unit [61] and
following four CNN-based heads for the 2D optical flow f,
flow upsampling mask u, certainty c and pose sensitivity
s. Two GenFlow modules are attached to the correlation
volumes for the cascade architecture. They are serially con-
nected so that the estimations from higher-level features are
utilized as initial values for the next module.
The optical flow and its upsampling mask produce the
upsampled flow, which is supervised by the L1 endpoint
error loss with the ground-truth optical flow. We also uti-
lize the disentangled point matching loss for the pose loss,
supervising the rotation, 2D object center, and depth of
the object individually.
Specifically, given an estimated
pose P = [R|[tx, ty, tz]T] and ground truth pose Pgt =
[R⋆|[t⋆
x, t⋆
y, t⋆
z]T], the pose loss is computed as follows:
  \be g in {spli
t} \
ma th
cal { L}_{
pose} = \ma thc al
 {D}( [\ma
thbf {R}|[
\m at
hb f {t}_ {x}^{\star }, \mathbf {t}_{y}^{\star }, \mathbf {t}_{z}^{\star }]^{\text {T}}], \mathbf {P}_{gt}) \\ + \mathcal {D}([\mathbf {R}^{\star }|[\mathbf {t}_{x}, \mathbf {t}_{y}, \mathbf {t}_{z}^{\star }]^{\text {T}}], \mathbf {P}_{gt}) \\ + \mathcal {D}([\mathbf {R}^{\star }|[\mathbf {t}_{x}^{\star }, \mathbf {t}_{y}^{\star }, \mathbf {t}_{z}]^{\text {T}}], \mathbf {P}_{gt}) \end {split}
(4)
where the distance D between two 6D poses P1 and P2
using the 3D points XM on the 3D model M is defined as
  \ma thc a
l
 {D}
(
\mat
hbf { P}_{1}, \mathbf {P}_{2}) = \frac {1}{\lvert \mathcal {X}_{\mathcal {M}}|} \sum _{x \in \mathcal {X}_{\mathcal {M}}} |\mathbf {P}_{1}x - \mathbf {P}_{2}x\rvert
(5)
To supervise the certainty c, we introduce the certainty
estimation by classifying the depth consistency similarly to
[17]. Given 6D poses Pinit and Pgt for the 3D model M
and intrinsic camera parameters K, we compute the pro-
jected depth Dr→t by warping the rendered image to the
target input image using the depth Dr. We also obtain the
depth map of warped pixels Dt given the synthetic depth of
the target image. Then we train the networks by minimiz-
ing the binary cross entropy loss Lcert between c and cr→t
which is a binary mask cr→t = |Dr→t −Dt| < dth where
dth is the distance threshold. It is worth noting that we place
a stop-gradient on the certainty mask cj when computing
the pose loss Lj
pose for disentangling the certainty c and
pose sensitivity s.
Given the RGB-D images with annotated ground truth
6D poses, the overall loss is defined as
  
\
m
ath
cal {L}
 = \ s um 
_{j= 1 }^{
N} \gamma ^{j-N}(\mathcal {L}_{flow}^{j} + \alpha \mathcal {L}_{cert}^{j} + \beta \mathcal {L}_{pose}^{j} )
(6)
following the strategy of exponentially increasing weights
[61] where we use the weight γ = 0.8 and the iteration
of GenFlow updates N = 8 in this work. In specific, the
GenFlow module assigned to C2 outputs {Pj}N/2
j=1 and the
another module outputs {Pj}N
j=N/2 from C1.
Multi-Hypotheses Strategy We select n best-scoring hy-
potheses in the coarse estimation. The selected hypothe-
ses are refined, and we choose the best estimate by evalu-
ating the refined hypotheses using the coarse model again.
This strategy requires more inference time for evaluating
n iterations of the rendering and refinement process, but it
can boost the accuracy by avoiding the risk of the “Winner-
Takes-All” strategy.
RGB-D Input If the observed image consists of depth in-
formation, we apply the depth refinement after the GenFlow
loop for every refinement step. First, we obtain the optical
flow F and upsampled certainty from the results of the last
GenFlow update. 3D-3D correspondences are established
by the optical flow and the input depth. We filter out the cor-
respondences of certainty lower than a specific threshold to
remove the outliers. Then we apply the RANSAC-Kabsch
[18, 34] for the depth refinement. Using certainty for filter-
ing out the outliers is more effective than using confidence
weights since the high-confidence pixels are sparse and the
input depth is prone to be noisy.

2D Localization
Pose Initialization
Pose Refinement
Average Recall (↑)
RGB-D
Input
Method
Novel
Objects
Method
Novel
Objects
Method
Novel
Objects LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V MEAN
1
✗
Mask-RCNN [22, 38]
✗
CosyPose [38]
✗
CosyPose [38]
✗
63.3
64.0
68.5
58.3
21.6 65.6 57.4
57.0
2
✗
Mask-RCNN [22, 38]
✗
SurfEmb [21]
✗
BFGS
✗
66.3
73.5
71.5
58.8
41.3 79.1 64.7
65.0
3
✗
YOLOX [19, 44]
✗
ZebraPose [56]
✗
-
✗
72.9
81.1
75.6
59.2
50.4 92.1 72.9
72.0
4
✗
YOLOX [19, 44]
✗
MegaPose [37]
✓
MegaPose+MH [37]
✓
64.8
78.1
74.1
56.9
42.2 86.3 70.2
67.5
5
✗
YOLOX [19, 44]
✗
Ours
✓
Ours+MH
✓
68.3
82.8
77.8
59.6
50.1 89.7 70.8
71.3
6
✓
YOLOX [19, 44]
✗
WDR-Pose [30]
✗
PFA [31]+Kabsch
✗
79.2
84.9
96.3
70.6
52.6 86.7 89.9
80.0
7
✓
YOLOX [19, 44]
✗
MegaPose [37]
✓
MegaPose+MH [37]+Teaserpp [67]
✓
70.4
71.8
91.6
59.2
55.3 87.2 85.5
74.4
8
✓
YOLOX [19, 44]
✗
Ours
✓
Ours+MH+Kabsch
✓
74.2
78.3
92.8
64.9
65.2 92.0 88.3
79.4
9
✗
OSOP [54]
✓
OSOP [54]
✓
OSOP+PnP+MH [54]
✓
31.2
-
-
-
-
49.2 33.2
-
10
✗
CNOS-det. [50]
✓
MegaPose [37]
✓
MegaPose+MH [37]
✓
56.0
50.8
68.7
41.9
34.6 70.6 62.0
54.9
11
✗
CNOS-det. [50]
✓
Ours
✓
Ours+MH
✓
57.5
53.0
69.1
45.6
40.8 74.5 63.9
57.8
12
✓
OSOP [54]
✓
OSOP [54]
✓
OSOP+Kabsch+MH [54]+ICP
✓
48.2
-
-
-
-
60.5 57.2
-
13
✓
CNOS-seg. [50]
✓
ZeroPose [5]
✓
MegaPose+MH [37]
✓
53.8
40.0
83.5
39.2
52.1 65.3 65.3
57.0
14
✓
CNOS-det. [50]
✓
MegaPose [37]
✓
MegaPose+MH [37]+Teaserpp [67]
✓
62.6
48.7
85.1
46.7
46.8 73.0 76.4
62.8
15
✓
CNOS-det. [50]
✓
Ours
✓
Ours+Kabsch+MH
✓
62.9
51.7
85.8
53.3
55.9 78.2 82.5
67.2
Table 1. 6D pose estimation results on the BOP challenge datasets. We report the Average Recall (AR) scores across the datasets for
various methods. The higher score the better. The best results among the comparable methods are in bold. We denote the multi-hypotheses
strategy as MH for simplicity.
4. Experiments
4.1. Datasets and Evaluation Metrics
Datasets We evaluate our method on seven datasets of the
BOP challenge [28]: LM-O [25], T-LESS [26], TUD-L
[27], IC-BIN [13], ITODD [16], HomebrewdDB [35] and
YCB-V [63]. These datasets contain cluttered real-world
scenes captured from different lighting and camera sensors.
Each scene consists of multiple objects including occlu-
sions between them. The objects vary in terms of texture
richness, symmetry, and usage.
Evaluation Metrics Following the evaluation methodology
of BOP challenge [28], we report the average recall (AR)
considering three pose-error functions: Visible Surface Dis-
crepancy (VSD), Maximum Symmetry-Aware Surface Dis-
tance (MSSD), and Maximum Symmetry-Aware Projection
Distance (MSPD). VSD measures the misalignment of the
visible part by computing the ratio of pixels in which the
difference between the estimated distance to the camera
center and ground truth is under a specific tolerance. In con-
sideration of the global symmetry information, MSSD com-
putes the maximum distance between the estimated camera-
space coordinates and corresponding ground truth. In con-
trast, MSPD computes the maximum distance between the
estimated image-space coordinates and the corresponding
ground truth. Concerning each pose-error function, the AR
is obtained by averaging recall calculated for multiple set-
tings of the correctness thresholds (and misalignment toler-
ances for VSD).
4.2. BOP Benchmark Results
Table 1 shows the results of our method on the BOP
datasets. For a fair comparison, all reported pose initializa-
Method # of renderings (↓)
Average Recall (↑)
LM-O T-LESS TUD-L IC-BIN YCB-V MEAN
Na¨ıve
576
22.5
25.6
30.6
17.5
16.4
22.5
Ours
144
28.8
35.5
33.8
29.7
28.9
31.3
208
32.9
40.4
40.0
31.1
34.7
35.8
Table 2. Ablation study of the hypotheses generation strategy for
the coarse estimation. The AR scores are reported for the best pose
hypothesis from the candidates from different sampling methods.
tion and pose refinement methods are trained with synthetic
data containing images rendered with BlenderProc [11]. We
reported the AR scores of MegaPose and ours by setting
the number of hypotheses n as 10 for the multi-hypotheses
strategy and the number of outer updates as 5. For ours, the
number of GenFlow updates per single outer update is set
to 8.
Unseen Objects For both RGB input (row 9-11) and RGB-
D input (row 12-15), the results show that our approach out-
performs all other methods for novel objects.
Seen Objects From row 1 to 8, the pose estimation results
on the 2D detection trained for target datasets are reported.
Our method still outperforms MegaPose for the same detec-
tion results. It is noticeable that our method provides com-
petitive results to the state-of-the-art methods, though it is
not the best, even without fine-tuning on the target datasets.
4.3. Ablation Study
We conduct ablation studies to validate the efficacy of
our method. The ablation results are reported for the RGB
inputs we mainly focus on. All experiments were performed
on 5 BOP datasets: LM-O, T-LESS, TUD-L, IC-BIN, and
YCB-V. We used the 2D detection results of YOLOX [19,

Method
Training
Average Recall (↑)
LM-O
T-LESS
TUD-L
IC-BIN
YCB-V
MEAN
1 No shape-constraint + RANSAC-PnP [39]
Lflow
61.7
77.6
72.4
54.2
65.7
66.3
2 Shape-constraint
Lflow, Lpose
64.9
81.8
75.1
57.0
70.1
69.8
3 Shape-constraint + Cascade
Lflow, Lpose
65.7
82.3
75.0
58.9
69.9
70.4
4 Shape-constraint + Confidence factorization
Lflow, Lcert, Lpose 64.5
81.8
75.3
57.3
70.5
69.9
5 Shape-constraint + Confidence factorization + Cascade Lflow, Lcert, Lpose 65.9
82.0
76.1
59.5
69.8
70.7
Table 3. Ablation study of GenFlow design. The best result is in bold. Our method accomplishes the best performance (row 5).
44] trained on target datasets to focus on the accuracy of 6D
pose estimation by preventing the performance bottleneck
caused by the detection methods for novel objects.
4.3.1
Coarse Pose Estimation
Table 2 shows the effectiveness of the GMM-based sam-
pling method for generating pose hypotheses. Pre-defined
rotations are generated following [68] for the Na¨ıve sam-
pling and for creating GMM of our method. We used the
N = 576 rotations for the Na¨ıve sampling, and M = 72
and M = 104 which conduct 2M renderings (144 and 208,
respectively) for ours. The results show that our method
can generate a better initial pose despite the fewer render-
ings and network inferences.
4.3.2
GenFlow design
Table 3 shows the results on different refiner designs.
All experiments are conducted using the same result of
coarse estimation, which uses our GMM-based sampling
with M = 104 without the multi-hypotheses strategy. We
run 5 iterations of GenFlow refinement, each containing the
8 iterations of the inner update. We evaluated the final 6D
pose from the GenFlow refiner.
Shape-Constraint Learning to estimate 2D correspon-
dences with flow loss only is suboptimal for 6D pose es-
timation since it makes no use of the target shape (row 1).
Imposing shape-constraint on the model benefits from the
guide of 3D information without loss of generalization per-
formance (row 2-5).
Cascade Architecture The proposed cascade architecture
with multiple GenFlow modules contributes to the perfor-
mance improvement (row 3, 5).
Confidence Factorization Factorizing the confidence
weights to certainty and pose sensitivity helps to increase
the accuracy of the output pose. Certainty estimation lever-
ages the training depth data and makes our method robust
to occlusion (row 4, 5).
Figure 6. GenFlow vs MegaPose refiner for the RGB inputs.
4.3.3
MegaPose Refiner vs Ours
We compare the performance of the MegaPose refiner
and ours for the RGB input. The same pose initializations
from our coarse estimation were used for a fair evaluation.
We reported the changes in AR score according to the it-
eration of the outer updates for both methods in Fig 6. The
reported score is computed as the arithmetic mean of the AR
scores for the five datasets. A single outer update takes 66.5
milliseconds for the MegaPose refiner and 98.2 millisec-
onds for ours, respectively, on an RTX 3090 GPU and Core
i9-10920X CPU. Although ours is slower than the Mega-
Pose refiner for a single refinement step, it achieves better
performance for the same and even fewer iterations. Note
that ours performs effectively by comparing a single-view
rendered image for an outer update, while MegaPose refiner
uses multiple rendered views as input.
5. Conclusion
In this paper, we present GenFlow, an optical flow-based
method for 6D pose refinement of novel objects. First, we
generate pose hypotheses using a GMM-based sampling
strategy for efficient and effective coarse estimation. Then,
we refine the high-scoring hypotheses with our pose refiner,
our main contribution. GenFlow iteratively refines the flow
and confidence with the guidance of 3D shape. Our method
achieves state-of-the-art performance of 6D pose estimation
for novel objects concerning RGB and RGB-D inputs.

References
[1] Christopher M. Bishop.
Pattern recognition and machine
learning, 5th Edition. Springer, 2007. 3
[2] Michael J. Black and P. Anandan. A framework for the ro-
bust estimation of optical flow. In Fourth International Con-
ference on Computer Vision, ICCV 1993, Berlin, Germany,
11-14 May, 1993, Proceedings, pages 231–236. IEEE Com-
puter Society, 1993. 2
[3] Angel X. Chang, Thomas A. Funkhouser, Leonidas J.
Guibas, Pat Hanrahan, Qi-Xing Huang, Zimo Li, Silvio
Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong
Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich
3d model repository. CoRR, abs/1512.03012, 2015. 5
[4] Hansheng Chen, Pichao Wang, Fan Wang, Wei Tian, Lu
Xiong, and Hao Li. Epro-pnp: Generalized end-to-end prob-
abilistic perspective-n-points for monocular object pose es-
timation. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
2781–2790, 2022. 5
[5] Jianqiu Chen, Mingshan Sun, Tianpeng Bao, Rui Zhao, Li-
wei Wu, and Zhenyu He. Zeropose: Cad-model-based zero-
shot pose estimation, 2023. 1, 2, 7
[6] Qifeng Chen and Vladlen Koltun. Full flow: Optical flow es-
timation by global optimization over regular grids. In 2016
IEEE Conference on Computer Vision and Pattern Recogni-
tion, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016,
pages 4706–4714. IEEE Computer Society, 2016. 2
[7] Xu Chen, Zijian Dong, Jie Song, Andreas Geiger, and Otmar
Hilliges. Category level object pose estimation via neural
analysis-by-synthesis. 2020. 2
[8] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang
Zhang, Gang Yu, and Jian Sun.
Cascaded pyramid net-
work for multi-person pose estimation. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR), 2018. 5
[9] Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau,
and Yoshua Bengio. On the properties of neural machine
translation: Encoder-decoder approaches. In Proceedings of
SSST@EMNLP 2014, Eighth Workshop on Syntax, Seman-
tics and Structure in Statistical Translation, Doha, Qatar,
25 October 2014, pages 103–111. Association for Compu-
tational Linguistics, 2014. 4
[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE Computer Society Conference on
Computer Vision and Pattern Recognition (CVPR 2009), 20-
25 June 2009, Miami, Florida, USA, pages 248–255. IEEE
Computer Society, 2009. 6
[11] Maximilian Denninger, Dominik Winkelbauer, Martin Sun-
dermeyer, Wout Boerdijk, Markus Knauer, Klaus H. Strobl,
Matthias Humt, and Rudolph Triebel.
Blenderproc2: A
procedural pipeline for photorealistic rendering. Journal of
Open Source Software, 8(82):4901, 2023. 5, 7
[12] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip
Hausser, Caner Hazirbas, Vladimir Golkov, Patrick van der
Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learn-
ing optical flow with convolutional networks. In Proceedings
of the IEEE International Conference on Computer Vision
(ICCV), 2015. 2
[13] Andreas Doumanoglou, Rigas Kouskouridas, Sotiris Malas-
siotis, and Tae-Kyun Kim. Recovering 6d object pose and
predicting next-best-view in the crowd. In 2016 IEEE Con-
ference on Computer Vision and Pattern Recognition, CVPR
2016, Las Vegas, NV, USA, June 27-30, 2016, pages 3583–
3592. IEEE Computer Society, 2016. 7
[14] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kin-
man, Ryan Hickman, Krista Reymann, Thomas Barlow
McHugh, and Vincent Vanhoucke. Google scanned objects:
A high-quality dataset of 3d scanned household items. In
2022 International Conference on Robotics and Automation,
ICRA 2022, Philadelphia, PA, USA, May 23-27, 2022, pages
2553–2560. IEEE, 2022. 5
[15] Bertram Drost, Markus Ulrich, Nassir Navab, and Slobodan
Ilic. Model globally, match locally: Efficient and robust 3d
object recognition. In 2010 IEEE computer society confer-
ence on computer vision and pattern recognition, pages 998–
1005. Ieee, 2010. 2
[16] Bertram Drost, Markus Ulrich, Paul Bergmann, Philipp
H¨artinger, and Carsten Steger. Introducing mvtec ITODD
- A dataset for 3d object recognition in industry. In 2017
IEEE International Con