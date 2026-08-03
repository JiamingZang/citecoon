# PoseGAM: Robust Unseen Object Pose Estimation via Geometry-Aware Multi-View Reasoning

> 2025 · id: W4417296911 · arXiv: 2512.10840 · pdf: https://arxiv.org/pdf/2512.10840 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
6D object pose estimation, which predicts the transforma-
tion of an object relative to the camera, remains challeng-
ing for unseen objects. Existing approaches typically rely
on explicitly constructing feature correspondences between
the query image and either the object model or template
images. In this work, we propose PoseGAM, a geometry-
aware multi-view framework that directly predicts object
pose from a query image and multiple template images,
eliminating the need for explicit matching. Built upon re-
cent multi-view-based foundation model architectures, the
method integrates object geometry information through two
complementary mechanisms: explicit point-based geome-
try and learned features from geometry representation net-
works. In addition, we construct a large-scale synthetic
dataset containing more than 190k objects under diverse
environmental conditions to enhance robustness and gen-
eralization. Extensive evaluations across multiple bench-
marks demonstrate our state-of-the-art performance, yield-
ing an average AR improvement of 5.1% over prior methods
and achieving up to 17.6% gains on individual datasets, in-
dicating strong generalization to unseen objects.

## introduction
6D object pose estimation, i.e., predicting an object’s rota-
tion and translation relative to the camera coordinate sys-
tem, has long been an important research topic with broad
applications in robotic manipulation [47, 57], augmented
and virtual reality [35, 56], autonomous driving [23, 68],
content creation [6, 74], etc. Early works primarily focused
on instance-specific [50, 72] or category-specific [40, 77]
object pose estimation. However, such approaches are lim-
ited in practical scenarios, as they often fail to generalize to
objects unseen during training.
To improve generalization, recent research has shifted
toward unseen object pose estimation. Existing methods
typically follow either a match-then-localize [45, 46, 78]
or match-then-refine [13, 30, 43, 49, 67] paradigm. These
methods explicitly establish feature correspondences be-
tween the query image and either the 3D model of the object
or a set of template images with known poses. Once these
correspondences are constructed, the object pose in the
query image is recovered using standard geometric solvers
such as least squares optimization or the PnP algorithm. Al-
though these approaches have achieved promising results,
their performance depends on the quality of the matching
stage, which leads to pose estimation inaccuracies when
matching is unreliable [41].
In this work, we explore whether unseen object pose es-
timation can be addressed through an end-to-end network,
eliminating the need for explicit feature matching and min-
imizing reliance on camera imaging priors. Inspired by re-
cent multi-view foundation models [28, 64, 66], which have
1
arXiv:2512.10840v2  [cs.CV]  15 Jun 2026

demonstrated the ability to directly infer 3D geometry with-
out traditional structure-from-motion steps, we adopt and
extend their architecture to the 6D pose estimation setting.
Specifically, we design a multi-view model that jointly pro-
cesses the query image and multiple template images with
known poses, enabling the network to reason about the ob-
ject’s pose directly.
Architecture inheritance enables us to exploit the power-
ful pretrained weights of multi-view foundation models and
potentially extend their success to the object pose estima-
tion task. However, existing multi-view foundation models
rely solely on visual image inputs. Although they can esti-
mate camera poses, they lack explicit information about the
3D object model, which is typically available in pose esti-
mation tasks. This omission limits their effectiveness for
object pose estimation. Moreover, these models typically
assume appearance consistency across views, making them
sensitive to appearance variations that frequently occur due
to the domain gap between renderings of CAD models and
real-world observations (see Appendix B for an analysis).
To address these limitations, we incorporate object ge-
ometry information into the multi-view architecture and
construct a large-scale dataset for object-centric pose esti-
mation. Specifically, for network design, we explore two
complementary approaches: (1) injecting explicit point-
based geometry and (2) integrating learned geometry fea-
tures through a geometry representation network. We ob-
serve that directly feeding raw sequential geometry tokens
into the multi-view structure hinders learning; therefore, we
project geometry features back into view map representa-
tions, which better align with the model’s multi-view rea-
soning process. For dataset construction, to enhance ro-
bustness to object variations and visual inconsistencies, we
build a large-scale and diverse synthetic dataset compris-
ing over 190k objects with corresponding images under a
wide range of challenging conditions, including variations
in lighting, appearance, and other scene factors. This di-
versity enables our model to generalize effectively across
different pose estimation scenarios.
Our main contributions are summarized as follows:
• We propose a multi-view feedforward network for object
pose estimation. The network directly takes the query
image and template images as input and predicts the ob-
ject pose in an end-to-end manner, eliminating the explicit
feature matching step used in prior works.
• We introduce object geometry into the multi-view frame-
work using explicit point maps and learned geometry rep-
resentations. Instead of using raw geometry feature to-
kens, we project the features into view-map representa-
tions, enabling the network to reason more effectively
about object poses and improving robustness across di-
verse scenarios.
• We construct a large-scale and diverse synthetic object
pose estimation dataset containing over 190k objects
across multiple challenging scenarios, including varying
environmental lighting conditions, appearance variations,
and other real-world complexities.

## method
Given an object M and a query image Iquery containing this
object, our goal is to estimate the object pose Tquery with
respect to the camera coordinate system:
Tquery = Network(Iquery; M).
(1)
Recent
multi-view
foundation
models,
such
as
VGGT [64], π3 [66], and RayZer [28], have achieved
remarkable results in geometric reasoning tasks. Motivated
by their success, we incorporate multi-view information
into our framework. This design offers two key advantages:
1) we can leverage recent successful architectures of these
recent multi-view networks; 2) we can initialize our model
using large-scale pretrained weights, facilitating faster
convergence and improved generalization.
Specifically, we render a set of multi-view RGB images
V = {Ii}i=1,··· ,N of the object M using manually defined
camera transformations T = {Ti}i=1,··· ,N around object
M. The queried pose is then estimated as:
Tquery = Network(Iquery; T , V)
(2)
3

The main multi-view RGB image network is detailed in
Sec. 3.1.
Furthermore, the inputs can be augmented with addi-
tional geometric information derived from the known object
model M, leading to the final formulation of our network:
Tquery = Network(Iquery; T , V , P , F )
transformations (cameras)
multi-view RGB images
point maps
point cloud features
(3)
where P denotes the point maps (Sec. 3.2) and F represents
the corresponding per-point features (Sec. 3.3). The overall
pipeline of our method is illustrated in Fig. 2.
3.1. Multi-View Network
The main network takes camera poses T and the corre-
sponding multiview RGB images V as input. To process
this multiview information, we first extract feature tokens
Xi = (x(1)
i , x(2)
i , · · · , x(L)
i
) for each image Ii using a pre-
trained network (DINOv2 [48]). Concurrently, we encode
camera poses Ti into a dedicated camera token ci using a
lightweight camera encoder:
(V, T ) →{(x(1)
i , x(2)
i , · · · , x(L)
i
, ci)}i=1,··· ,N
(4)
Similarly, for the query image Iquery, we extract its visual
tokens Xquery and append a learnable query camera token
cquery. The total number of tokens processed by the network
is therefore:
inter-frame
z
}|
{
(L + 1)
| {z }
intra-frame
×(N + 1)
(5)
Our network architecture is inspired by the design of
VGGT [64], a feed-forward multi-view foundation model.
We alternately apply inter- and intra-frame self-attention
layers to these tokens (named multiview tokens). Further-
more, we inject geometry tokens into the main network
to incorporate explicit 3D information. Specifically, before
each self-attention layer, we perform a cross-attention op-
eration, denoted as CA, between the multi-view tokens and
the geometry tokens:
CA(Q ←multiview tokens, KV ←geometry tokens)
(6)
The processing and construction of these geometry tokens
are detailed in Sec. 3.2 and Sec. 3.3.
After passing through multiple attention layers, the fea-
tures corresponding to the camera tokens are decoded by a
lightweight head to predict the camera pose. The resulting
output is supervised solely by the ground-truth camera pose.
3.2. Geometry Processing via Point Maps
We process the object geometry M through the following
steps.
First, we render the object into multi-view depth
maps using the camera poses T , and subsequently recon-
struct the point maps in world coordinates using the corre-
sponding camera intrinsics:
Object M
T→Depth Maps →Point Maps P
(7)
where P = {Pi}i=1,··· ,N. Each point map Pi is processed
by a lightweight convolutional neural network to produce a
set of point map tokens:
Pi
Conv
−→
n
p(1)
i , p(2)
i , · · · , p(L)
i
o
(8)
A naive approach would be to directly add the point map to-
kens to the multi-view RGB image tokens. However, such
direct fusion introduces a substantial modality gap from the
pretrained model’s original input distribution (which is pri-
marily based on natural images), thereby hindering effective
knowledge transfer (see Sec. 5). To mitigate this, we em-
ploy a cross-attention mechanism for geometry information
injection, as defined in Eq. (6).
3.3. Geometry Processing via Point Cloud Networks
To further enhance the network’s understanding of the ob-
ject model, we employ off-the-shelf geometry representa-
tion networks to extract a global representation of the 3D
object and inject it into our framework. Specifically, we
adopt existing point cloud architectures (e.g., PointTrans-
former v3 [69]). The input to the network is a point cloud
(with coordinates recovered from Eq. (7)) augmented with
per-point color and normal information. The network out-
puts a set of per-point feature embeddings. In practice, we
observe that directly injecting these features in their raw for-
mat is ineffective (see Sec. 5.4); the model struggles to uti-
lize the information efficiently. Instead, when the per-point
features are spatially reorganized into a view-map format,
the network can more effectively leverage the encoded geo-
metric structure. Motivated by this observation, we replace
the coordinate channels in the point maps with the extracted
feature vectors, thereby forming feature maps:
Point Clouds
PCNet
−→Per-Point Features
Disperse
−→Feature Maps F
(9)
where F = {Fi}i=1,··· ,N. Analogous to point maps, we ap-
ply a lightweight convolution network to these feature maps
to obtain feature tokens:
Fi
Conv
−→
n
f (1)
i
, f (2)
i
, · · · , f (L)
i
o
(10)
These feature tokens are then added to the corresponding
point map tokens, jointly forming the KV inputs in the
cross-attention layers.
4

Figure 3. Examples from the constructed object pose estima-
tion dataset. The leftmost column shows the object mesh after
texture rebaking. The four columns on the right illustrate the four
types of rendered image data.
4. Data Construction
To ensure the robustness of our method for object pose es-
timation, we construct a large-scale synthetic dataset com-
prising a diverse collection of 3D objects. Each object (M)
is paired with texture-rendered images (V) and geometry-
related maps (P) captured under a wide range of camera
poses (T ). The overall data construction pipeline consists
of two stages: geometry data preparation and image/map
generation. Additional details of the dataset, along with ab-
lation studies validating its effectiveness, are provided in
Appendix C & E.
4.1. Geometry Data Construction
We collect synthetic 3D object assets from multiple
publicly available datasets, including Toys4K [55], 3D-
FUTURE [16], ABO [9], HSSD [29], and Objaverse [12].
To ensure geometric integrity and overall asset quality, we
follow the filtering strategy of Xiang et al. [71], which
removes objects with low-quality geometry or unrealistic
mesh structures.
After filtering, we retain over 190,000
high-quality object assets.
Following the asset selection process, we further stan-
dardize the objects to ensure consistency during rendering.
Many assets contain complex shader graphs or procedural
materials, which can lead to undesirable variations across
different rendering platforms. To address this, we perform
texture re-baking. Specifically, we apply Smart UV Un-
wrap in Blender [2] to obtain consistent UV coordinates,
and then bake a consolidated texture that integrates the Dif-
fuse, Glossy, and Transmission components. This proce-
dure eliminates shader-dependent inconsistencies and pro-
duces a uniform material representation for all assets. Fi-
nally, each object is exported in GLB format with a single
base color texture map.
4.2. Image Data Construction
For each object asset, we define 50 camera poses distributed
uniformly around the object using a spherical Hammersley
sequence. At each pose, we render both the texture image
and a set of geometry-related maps, including depth maps,
normal maps, and mask maps. These rendered modalities
serve as the geometric inputs required by our model (see
Sec. 3). To prepare the query images, we render each as-
set under four distinct scenarios designed to introduce vary-
ing levels of difficulty. This diversity helps strengthen the
model’s robustness under realistic conditions. Examples of
the resulting dataset are shown in Fig. 3; for clarity, only
RGB renderings are partially displayed.
Centric Object Images.
In the first scenario, camera
poses are uniformly sampled around the object, and all
viewing directions point toward its centroid. The lighting
environment is fixed and consists of three sources: a top
area light, a bottom area light, and a front-top point light.
Rendering is performed using Blender EEVEE [2], which
offers a favorable balance between visual fidelity and com-
putational efficiency. This configuration represents the most
basic case, where the object remains centered in the frame
and is observed under consistent illumination.
Uncentric Object Images.
To simulate more natural im-
age compositions in which the object does not appear at the
center of the frame, we randomize both the camera posi-
tion and the corresponding look-at point. Starting from the
centric configuration, we first sample camera poses on 

## experiments
Please refer to Appendix D for implementation details, in-
cluding the network architecture, training, inference, and
evaluation settings. Additional quantitative results, includ-
ing ablation studies of our dataset, further comparisons with
existing methods, and visual comparisons across diverse
scenarios, are provided in Appendix E and F.
5.1. Experimental Setup
Training and Evaluation Datasets.
We train our model
on the synthetic dataset constructed as described in Sec. 4.
To evaluate its performance on unseen object pose esti-
mation, we benchmark our approach on five widely used
datasets: LM-O [3], T-LESS [20], YCB-V [73], TUD-
L [21], and IC-BIN [14]. For these benchmark experiments,
the model is trained on the full constructed dataset to ensure
comprehensive learning. In contrast, for ablation studies,
training and evaluation are performed on a subsampled ver-
sion of the dataset, which enables efficient experimentation.
Evaluation Metrics.
To compare our method with ex-
isting object pose estimation approaches, we adopt the
standard Average Recall (AR) metric from the Benchmark
for Pose Estimation (BOP) [22].
AR is computed us-
ing three pose-error functions: Visible Surface Discrep-
ancy (VSD), Maximum Symmetry-Aware Surface Distance
(MSSD), and Maximum Symmetry-Aware Projection Dis-
tance (MSPD). A pose is considered correct if its error falls
below a predefined threshold. The mean recall is calculated
for each error function across multiple thresholds, and the
overall AR is defined as: AR = (ARVSD + ARMSSD +
ARMSPD)/3. For the ablation studies, we adopt a variant
of the standard AUC@N metric [64], which combines Ab-
solute Rotation Accuracy (ARA) and Absolute Translation
Accuracy (ATA). ARA and ATA measure the angular errors
in rotation and translation, respectively, for each query im-
age. These errors are thresholded to compute per-threshold
accuracy scores, and the AUC is then calculated as the area
under the curve of the minimum values between ARA and
ATA across all thresholds.
6

## related_work
2.1. Object Pose Estimation
Object pose estimation aims to determine an object’s trans-
formation relative to the camera, typically given an ob-
served image and the object’s geometric model.
Tradi-
tional methods can be broadly categorized into instance-
level and category-level approaches. Instance-level meth-
ods [24, 34, 37, 50, 51, 58, 60, 72] are designed or
trained for a specific object instance. By employing tech-
niques such as correspondence prediction [37, 51], tem-
plate matching [34, 58], keypoint voting [50, 60], or di-
rect pose regression [24, 72], these methods can achieve
highly accurate pose estimation. However, their applicabil-
ity is limited, as each new object instance requires retrain-
ing or fine-tuning. This limitation has motivated the devel-
opment of category-level methods [8, 15, 27, 38, 59, 61–
63], which seek to generalize across unseen objects within
the same category. Many such methods [15, 59, 63] first
extract a category-specific shape prior and then align the
query object to this canonical shape before estimating its
pose. Other works [8, 61] attempt to directly regress the
pose without explicitly modeling the shape prior. Although
these approaches improve generalization within a category,
they still struggle to handle the diversity of real-world ob-
ject appearances. Recently, research has shifted toward un-
seen object pose estimation [5, 17, 25, 30, 39, 46], where
the goal is to estimate the poses of category-agnostic novel
objects. These methods typically train networks to extract
representative features from both geometry and images, and
then derive the pose through cross-modal correspondences.
Some works [1, 4, 49] further exploit powerful pretrained
feature extractors to obtain these representations directly.
2.2. Multi-View Foundation Models
Traditional geometric reasoning methods [10, 11, 52],
which reconstruct sparse 3D maps while jointly estimat-
ing camera parameters from multiple images, typically rely
on Structure-from-Motion (SfM). In these pipelines, pixel
correspondences are first obtained through keypoint match-
ing across images to establish geometric relationships, fol-
lowed by bundle adjustment to jointly optimize 3D coor-
dinates and camera parameters. Dense geometry can then
be reconstructed using Multi-View Stereo (MVS) [18]. To
avoid such complex intermediate steps, recent approaches
aim to predict 3D geometry directly from RGB images.
Since single-image reconstruction is inherently ill-posed,
these methods employ neural networks trained on large
2

Cross-Attention
Self-Attention
× 𝐾𝐾times
Point Map Encoder
Feature Map Encoder
Camera Encoder & Image Encoder
𝑰𝑰𝒒𝒒𝒒𝒒𝒒𝒒𝒒𝒒𝒒𝒒
Crop with 
mask
𝓣𝓣, 𝓥𝓥
𝓟𝓟
𝓕𝓕
Learnable
Token
Query
Key & Value (Shared by each Cross-Attn Layer)
Extract
Decode
Inverse
𝑻𝑻𝒒𝒒𝒒𝒒𝒒𝒒𝒒𝒒𝒒𝒒
Geometry Feature 
Extractor
𝓜𝓜
Multi-Modal 
Encoders
Fusion Transformer & Pose Estimation
Multi-Modal Input Data
𝑇𝑇Obj
Cam
Figure 2. Overview of PoseGAM. Given a query image Iquery and an object mesh M, the goal is to estimate the object-to-camera
transformation Tquery. A set of camera poses is sampled around M to render images V and corresponding point maps P. Both Iquery (with
its foreground segmented) and V are encoded into image tokens, each paired with a camera token. For the rendered views V, the camera
tokens are computed from known intrinsics and extrinsics, whereas for Iquery the camera token is a learnable embedding. A geometry
feature extractor produces a global object representation, which is distributed to camera views to form view-specific features F. These,
together with point maps P, are encoded as key–value tokens for cross-attention. The output camera tokens are decoded to predict the
camera-to-object transformation T Cam
Obj , from which the final object-to-camera pose Tquery is obtained by matrix inversion.
datasets to learn strong 3D priors, helping resolve ambigu-
ities. For example, DUSt3R [65] and its metric follow-up
MASt3R [33] predict relative point maps from image pairs.
Additional scene representations, such as camera poses and
depth, can then be recovered by iteratively processing mul-
tiple image pairs and applying post-optimization. Subse-
quent works like VGGT [64] and π3 [66] extend DUSt3R
to multi-view settings. VGGT constructs a multi-image net-
work that employs an alternating-attention transformer to
predict multi-view point maps, depth, camera poses, and
tracking features. π3 further refines VGGT by removing
the need for the first input frame as a reference coordinate.
Similarly, RayZer [28] processes unposed and uncalibrated
multi-view images to predict per-view camera parameters
along with a consistent scene representation.
2.3. Ours versus Others
Unlike existing unseen object pose estimation methods,
which rely on explicitly constructing feature correspon-
dences, our approach employs a multi-view network in-
spired by the success of such architectures in recent geomet-
ric reasoning fields. The network directly predicts the object
pose by jointly processing the query image and multiple ob-
ject template images as input. In contrast to typical multi-
view foundation models, we incorporate the object’s geo-
metric information into the network, which enhances both
precision and accuracy, making the architecture well-suited
for the object pose estimation task.

## conclusion
In this work, we present a geometry-aware multi-view
framework for unseen object pose estimation. Built upon
recent multi-view foundation model architectures, the pro-
posed approach incorporates object geometry through ex-
plicit point-based representations and learned geometry fea-
tures projected into view-map form, facilitating more ef-
fective use of object model geometry information.
Sup-
ported by our arranged large-scale synthetic dataset con-
taining more than 190k objects under diverse conditions,
the method demonstrates strong robustness and generaliza-
tion across different scenarios. We believe this work repre-
sents an important step toward direct object pose estimation
and toward integrating advances in 3D geometric reasoning
within a unified framework.
8

Acknowledgements
The research reported in this publication was supported
by funding from King Abdullah University of Science and
Technology (KAUST) – Center of Excellence for Genera-
tive AI, under award number 5940 and a gift from Google.