# GCE-Pose: Global Context Enhancement for Category-level Object Pose Estimation

> 2025 · id: arxiv:2502.04293 · arXiv: 2502.04293 · pdf: https://arxiv.org/pdf/2502.04293 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
A key challenge in model-free category-level pose estimation
is the extraction of contextual object features that generalize
across varying instances within a specific category.
Re-
cent approaches leverage foundational features to capture
semantic and geometry cues from data.
However, these
approaches fail under partial visibility. We overcome this
with a first-complete-then-aggregate strategy for feature ex-
traction utilizing class priors.
In this paper, we present
GCE-Pose, a method that enhances pose estimation for
novel instances by integrating category-level global context
prior. GCE-Pose performs semantic shape reconstruction
with a proposed Semantic Shape Reconstruction (SSR) mod-
ule. Given an unseen partial RGB-D object instance, our
SSR module reconstructs the instance’s global geometry and
semantics by deforming category-specific 3D semantic pro-
totypes through a learned deep Linear Shape Model. We
further introduce a Global Context Enhanced (GCE) fea-
ture fusion module that effectively fuses features from partial
RGB-D observations and the reconstructed global context.
Extensive experiments validate the impact of our global con-
text prior and the effectiveness of the GCE fusion module,
demonstrating that GCE-Pose significantly outperforms ex-
isting methods on challenging real-world datasets House-
Cat6D and NOCS-REAL275. Our project page is available
at https://colin-de.github.io/GCE-Pose/.

## introduction
The task of object pose estimation varies according to gen-
eralization level and input modality.
Instance-level pose
estimation methods [24, 58, 65] focus on specific object
instances that does not generalize to other objects, while
fully unseen object pose estimation methods [13, 26, 56] are
designed to handle novel objects, however requires object
model as prior. Unlike aforementioned methods, category-
level pose estimation methods [8, 38, 66] aim to generalize
*Equal contribution.
†Corresponding author: junwen.huang@tum.de
Global Context Enhanced Feature Fusion
Partial Feature Learning Methods
Input RGB-D
Partial Feature
Global Feature
NOCS Prediction
Partial Feature
NOCS Prediction
NN
NN
NN
SSR
GCE Fusion
Input RGB-D
Figure 1. Overview of the category-level Pose Estimation Pipeline:
(A) Previous methods, i.g. AG-Pose [41] and Second Pose [8], rely
on partial features extracted by a neural network (NN) to regress
object poses. (B) We introduce a novel approach that leverages
a semantic shape reconstruction (SSR) module for global feature
extraction. This global context enhances (GCE) the mapping from
partial features to NOCS features.
across unseen instances within a defined category that re-
quires only an RGB(-D) image of a new instance during
the inference, making the method model-free that does not
require predefined object models.
Current category-level approaches primarily estimate the
Normalized Object Coordinate Space (NOCS) [66] and em-
ploy a pose solver, such as the Umeyama algorithm [62], to
obtain the object pose [38]. To effectively extract category-
level features from RGB (and/or depth) inputs, researchers
have developed various neural network architectures to cap-
ture features from partial RGB(-D) observations.
Some
recent methods [2, 47, 49] leverage foundation models like
DINOv2 [48] for improved performance. Additionally, re-
search [4, 8, 25, 26, 40] highlights the importance of com-
bining semantic and geometric information to enhance fea-
ture robustness and distinguishability, aiding in better cor-
respondence and pose estimation. However, category-level
pose estimation being model-free and having only partially

observed RGB(-D) inputs limits the extraction of global
context information.
Some methods [6, 38, 61, 72, 80]
have introduced categorical geometric shape priors to re-
construct instance models from partial input points, solving
for object pose by establishing dense correspondence be-
tween partial input points and reconstructed models. How-
ever, these methods solely introduce shape priors neglect-
ing the semantic context of the category. More recently,
GS-Pose [71] selects one instance as a reference prototype
within a category and applies semantic feature matching be-
tween partial points and the reference instance. However,
this design struggles with intra-class shape variations and is
particularly vulnerable to noise in partial point cloud obser-
vations.
In this work, we propose GCE-Pose, a novel approach
that integrates global context incorporating both geometric
and semantic cues to enhance category-level object pose es-
timation. We propose two major modules named Semantic
Shape Reconstruction (SSR) and Global Context Enhanced
(GCE) feature fusion modules to facilitate pose estimation.
The SSR module is a first-complete-then-aggregate strat-
egy that reconstructs the input partial points into a complete
shape and smoothly aggregates the semantic prototype to
the instance. The GCE feature fusion model is proposed
to effectively fuse the reconstructed global context with
local cues. The efficacy of our proposed method is con-
firmed by extensive evaluation on the challenging real-world
datasets, achieving SOTA performance against the existing
approaches. Our main contributions are as follows:
• We propose GCE-Pose, a Global Context Enhancement
(GCE) approach that integrates global context with both
geometric and semantic cues for category-level object
pose estimation.
• We introduce a Semantic Shape Reconstruction (SSR)
strategy that addresses partially observed inputs by re-
constructing both object geometry and semantics through
learned categorical deformation prototypes.
• Extensive experiments demonstrate that our method
achieves robust pose estimation even under significant
shape variations and occlusions improving the general-
ization to unseen instances.

## method
The objective of GCE-Pose is to estimate the 6D object pose
and size from RGB-D data. Given a single RGB-D frame
and the category instance mask, we obtain the partial RGB
observation Ipartial and its corresponding partial point cloud
Ppartial derived from the depth map. Utilizing Ipartial and
Ppartial, the objective is to recover the 3D rotation R ∈SO(3),
the 3D translation t ∈R3, and the size s ∈R3 of the target
object.
GCE-Pose consists of four main modules (Fig. 2): Ro-
bust Partial Feature Extraction (Sec. 3.1), Semantic Shape
Reconstruction (Sec. 3.2), Global Context Enhanced Feature
Fusion (Sec. 3.3), and Pose & Size Estimator (Sec. 3.5).
3.1. Robust Partial Feature Extraction
Partial observations from RGB-D sensors often contain sig-
nificant noise and incomplete geometry, making dense cor-
respondence prediction unreliable. We address this chal-
lenge with a keypoint-based approach [41] that focuses on

the most discriminative and reliable object regions.
The 𝑁input points are put in order within Ppartial ∈R𝑁×3
and we extract point features FP ∈R𝑁×𝐶1 using Point-
Net++ [51].
For the RGB image Ipartial, we extract the
image feature FI ∈R𝑁×𝐶2 using DINOv2 [48] and concate-
nate FI to FP to obtain Fpartial ∈R𝑁×𝐶. We follow AG-
Pose [41] for keypoint detection. First, 𝑀keypoint features
are extracted using a learnable embedding Femb ∈R𝑀×𝐶,
which undergoes cross-attention with Fpartial to attend to
critical regions in Ppartial.
This process yields a feature
query matrix Fq = CrossAttention(Femb, Fpartial). We then
compute correspondences via cosine similarity, forming a
matrix A ∈R𝑀×𝑁, and select 𝑀keypoints from Ppartial as
Pkpt = softmax(A)Ppartial. To ensure keypoints lie on the ob-
ject surface and minimize outliers, an object-aware Chamfer
distance loss Locd is applied. With ground truth pose Tgt,
we filter outliers by comparing each point 𝑥∈Ppartial to the
instance model Mobj:
min
𝑦∈Mobj
Tgt(𝑥) −𝑦

2 < 𝜏1,
(1)
where 𝜏1 is an outlier threshold. The object-aware Chamfer
distance loss is then:
Locd =
1
|Pkpt|
∑︁
𝑥∈Pkpt
min
𝑦∈P∗
partial
∥𝑥−𝑦∥2.
(2)
To prevent keypoints from clustering, a diversity regulariza-
tion loss is added:
Ldiv =
∑︁
𝑥≠𝑦∈Pkpt
max{0, 𝜏2 −∥𝑥−𝑦∥2},
(3)
where 𝜏2 controls keypoint distribution. To enhance features
with geometric context, the Geometric-Aware Feature Ag-
gregation (GAFA) module [41] is applied. GAFA augments
each keypoint with(1)local geometricdetailsfrom K-nearest
neighbors and (2) global information from all keypoints, im-
proving feature discriminability for correspondence estima-
tion.
3.2. Semantic Shape Reconstruction
Intra-class variation is a key challenge in category-level pose
estimation. To tackle this issue, category-level shape pri-
ors have been extensively used in object pose estimation.
By representing the shape with mean shapes and deforma-
tions [61] or learning implicit neural representations for ge-
ometry recovery [27, 28], pose estimators can better learn
correspondences in NOCS space, benefiting from accurate
shape priors. While geometric shape reconstruction pro-
vides valuable priors, it cannot fully capture the rich seman-
tic information of object parts. Recent advances in 2D foun-
dation models, particularly DINO [48], have demonstrated
remarkable capabilities in extracting zero-shot semantic in-
formation from single RGB images. Building upon this in-
sight, we propose Semantic Shape Reconstruction (SSR)
to learn a per-category linear shape model similar to [43]
that describes an object using instance-specific geometry
and category-level semantic features.
Deep Linear Semantic Shape Model. To overcome the
challenges posed by partial observations from depth sensors,
such as occlusions and incomplete geometry, we employ a
variation of the deep linear shape model [43]. This approach
is motivated by the need to robustly and efficiently param-
eterize object shapes with shape parameters and produce a
completed 3D object representation, even when faced with
limited input data. We represent each point in our model as
a tuple (𝑥, 𝑓) where 𝑥∈R3 represents a spatial coordinate
and 𝑓∈R𝐶represents its semantic feature vector. For 𝐼
points of an object instance within category 𝑘, we learn a
linear shape model. The model for category 𝑘consists of (i)
a geometric prototype 𝑐𝑘∈R𝐼×3 with associated semantic
features 𝑐𝑘
sem ∈R𝐼×𝐶, (ii) a set of geometric deformation
basis vectors 𝑣𝑘= {𝑣𝑘
1, . . . , 𝑣𝑘
𝐷} where 𝑣𝑘
𝑖∈R𝐼×3, and (iii)
a scale parameter vector 𝑠𝑘∈R3. The key insight of our
approach is that semantic features remain coupled to their
corresponding points during geometric deformation. Any
semantic shape Uk in the model family is defined by:
Uk = (Xk, Fk) =
 
𝑠𝑘⊙(𝑐𝑘+
𝐷
∑︁
𝑖=1
𝑎𝑘
𝑖𝑣𝑘
𝑖), 𝑐𝑘
sem
!
(4)
where Xk ∈R𝐼×3 are the 𝐼points in shape prior 𝑘and Fk ∈
R𝐼×𝐶their associated features. The shape parameter vector
is given by 𝑎𝑘=  𝑎𝑘
1, . . . , 𝑎𝑘
𝐷
 ∈R𝐷, 𝑠𝑘∈R3 controls
scaling, and ⊙defines the element-wise Hadamard product.
We train two neural networks for each category 𝑘to predict
shape parameter 𝑎𝑘with network D𝑘and scale 𝑠𝑘with S𝑘.
To optimize the model, we minimize the Chamfer dis-
tance loss, LCD, which ensures accurate shape reconstruc-
tion through:
LCD =
∑︁
𝑥∈P
min
𝑘𝑑(𝑥, Uk) ,
(5)
with the Chamfer distance 𝑑, ground truth point clouds P
from category 𝑘and shape reconstruction Uk defined in
Eq. (4) .
Training with ground truth yields the optimal
parameters ¯𝑎𝑘, ¯𝑠𝑘, 𝑐𝑘, and 𝑣𝑘which allow to formulate an
additional loss to refine shape reconstruction under partial
observations Ppartial by freezing 𝑐𝑘, and 𝑣𝑘within
Lpara =
∑︁
𝑥′∈Ppartial
𝜆1
D𝑘(𝑥′) −¯𝑎𝑘 + 𝜆2
S𝑘(𝑥′) −¯𝑠𝑘 .
(6)
Finally, we combine the reconstruction and parameter loss
to formulate the overall loss
Lrec = 𝜆CD · LCD + 𝜆para · Lpara,
(7)
where 𝜆CD and 𝜆para are the hyperparameters that weight
the contributions of LCD and Lpara , respectively.

Partial 
Prototype
Feature
Averaging
(B) Building Semantic Prototype
Semantic 3D Lifting
Semantic Aggregation
Instance 
(Stage 1)
(A) Deep Linear Shape Reconstruction
Semantic Recon.
(C) Semantics Reconstruction
Instance 
(Stage 2 )
Partial 
Instance 
Prototype 
Figure 3. Illustration of Deep Linear Semantic Shape Model. A Deep Linear Semantic Shape model is composed of a prototype shape
𝑐, a scale network S, a deformation network D, a Deformation field V and a category-level semantic features 𝑐𝑘sem. At stage 1, we build
a Deep Linear Shape (DLS) model using sampled point clouds from all ground truth instances within each category, training a linear
parameterization network to represent each instance. At stage 2, we retrain the DLS model to regress the corresponding DLS parameters
from partial point cloud inputs using a deformation and scale network. During testing, the network predicts DLS parameters for unseen
objects and reconstructs their point clouds based on the learned deformation field to get semantic reconstruction.
Semantic Prototype Construction. To effectively integrate
rich semantic information into our 3D shape reconstruction,
we employ a process that begins by extracting dense se-
mantic features from multiple RGB images of each object
instance using the DINOv2 [48]. For each object instance
without texture, we position multiple virtual cameras around
the object to capture RGB images and depth maps from di-
verse viewpoints. This setup ensures full coverage of the
object’s surface and mitigates occlusion effects. The RGB
images are processed through the DINOv2 model to extract
dense 2D semantic feature maps. Using the corresponding
depth maps and known camera intrinsics and extrinsic, we
project the 2D semantic features into 3D space. For each
pixel (𝑢, 𝑣) in the image, we compute its 3D position 𝑃us-
ing the depth value 𝑧and project the associated semantic
feature f2D(𝑢, 𝑣) to this point 𝑃= 𝑧𝐾−1[𝑢, 𝑣, 1]𝑇,, where
𝐾is the camera intrinsic matrix. As a result, we obtain a
dense semantic point cloud Fsem. To ensure computational
efficiency and point-wise correspondence, we downsample
this dense semantic point cloud to 𝐼points aligned with our
geometric reconstruction.
For each point 𝑃𝑖in the deep
linear shape reconstruction, we aggregate semantic features
from its k nearest neighbors in the dense cloud:
Finstance(𝑃𝑖) = 1
𝑘
∑︁
𝑃𝑗∈𝑁𝑘(𝑃𝑖)
Fsem(𝑃𝑗).
(8)
The category-level semantic prototype 𝑐𝑘
sem is then con-
structed by averaging 𝑁instance features across the cate-
gory 𝑘while maintaining point-wise correspondence with
the geometric prototype 𝑐𝑘:
𝑐𝑘
sem = 1
𝑁
𝑁
∑︁
𝑖=1
Finstance(𝑃𝑘
𝑖)
(9)
Semantic Reconstruction. The key advantage of our ap-
proach is that semantic reconstruction becomes straightfor-
ward once the semantic prototype is established. Given a
partial point cloud 𝑥′, we first reconstruct its geometry and
then directly inherit the semantic feature from the prototype.
3.3. Global Context Enhanced Feature Fusion
Traditional pose estimation methods rely primarily on par-
tial observations, but they often struggle with challenges
such as

## experiments
4.1. Implementation Details
For the HouseCat6D dataset [29], cropped images are re-
sized to 224 × 224 for feature extraction, and 1024 points
are sampled from inputs. For Partial Feature Extraction, the
number of keypoints is 𝑀= 96, and the feature dimensions
for geometric and DINO features are 𝐶1 = 128, 𝐶2 = 128,
and 𝐶= 256. In deep linear shape reconstruction, we set the
basis dimension 𝐷to 5, and the number of points in the pro-
totype is 1024. The pose estimation network is trained with
batch size 36 with an ADAM [32] optimizer with a triangu-
lar2 cyclical learning rate schedule [57] on a single NVIDIA
4090 GPU for 150 epochs. We attach more implementation
details in our Appendix.
4.2. Evaluation Benchmarks
Datasets.
We evaluate our method on two challenging
real-world benchmarks:
HouseCat6D [29] and NOCS-
REAL275 [66]. HouseCat6D contains 21K images of 194

AG-Pose( DINO)
Ours
Figure 4. Visualization of category-level object pose estimation results on HouseCat6D dataset [29]. Predicted 3D bounding boxes are
shown in red, with ground truth in green. Challenging cases are highlighted in pink side squares. Leveraging our global context-enhanced
pose prediction pipeline, GCE-Pose outperforms the SOTA AG-Pose [41] (DINO), demonstrating robustness to occlusions and strong
generalization to novel instances.
Dataset

## related_work
2.1. Object Reconstruction for Pose Estimation
Object reconstruction is essential for object pose estima-
tion when CAD models are unavailable, as it captures ob-
ject geometry and appearance while establishing a canon-
ical space. Methods like OnePose [59], OnePose++ [21],
and CosyPose [33] employ Structure from Motion (SfM)
to match features across views, while approaches such as
NeRFPose [35], GS-Pose [3], and FoundationPose [73] uti-
lize Neural Radiance Fields or 3D Gaussian Splatting [31]
for flexible reconstruction.
For known object categories, semantic information can
enhance instance-level reconstructions.
Some methods
build semantic representations directly: Goodwin et al. [19]
align 3D views to a query view, and Zero123-6D [12] synthe-
sizes views via diffusion models to reduce reference views.
I2cNet [53] expands such techniques to categories by in-
tegrating a 3D mesh reconstruction module. Other meth-
ods use shape priors, such as SPD [61], RePoNet [17],
and Wang et al. [67], which learn instance reconstructions
from category-specific priors. SGPA [6] and RBP [80] dy-
namically adapt priors based on observed structures, while
SAR-Net [36] and ACR-Pose [15] further incorporate ge-
ometric and adversarial strategies. GS-Pose [71] projects
DINOv2 [48] features onto a 3D reference shape, aiding fea-
ture alignment and pose prediction. From here, we propose
a novel method that further improves the aforementioned
methods by integrating semantics to the reconstructions to
provide global contextual information for pose estimation.
2.2. Representation Learning for Pose Estimation
Learning effective feature representations from input modal-
ities is crucial to pose estimation, evolving alongside ad-
vancements in vision neural networks. Early visual feature
extractors relied on CNN backbones [30, 34, 50, 60, 74, 78]
to predict or refine object poses from single RGB im-
ages. Recently, foundational models like DINOv2 have been
widely adopted [2, 8, 40, 47, 49] to enhance robustness and
contextual understanding.
Beyond 2D-only approaches, many methods now com-
bine 2D image and 3D point cloud networks to jointly
extract semantic and geometric features, constructing ro-
bust embeddings for tasks such as direct pose regres-
sion [8, 22, 23, 64] or feature matching [4, 26, 40]. RGB-
D methods address feature fusion at multiple levels, such
as 2D-3D and local-global fusion.
Methods like Dense-
Fusion [64] and PVN3D [22] concatenate per-pixel geo-
metric and RGB features, while SecondPose [8] employs
MLPs to fuse DINOv2 [48] features with Point Pair Fea-
tures (PPF) [14].
More recent transformer-based approaches, including
SAM6D [40] and MatchU [26], integrate both local RGB-D
and global CAD model features [52, 76, 77], demonstrat-
ing the value of cross-modality fusion. In this work, we
not only incorporate categorical semantic priors into object
reconstruction but also effectively integrate global context
into local embeddings through our fusion module.
Occlusions and object symmetries introduce visual am-
biguities [44], necessitating the consideration of multiple
correct poses. To address this, various methods frame pose
prediction as distribution estimation, learning ambiguity-
aware representations [20, 55, 63]. We tackle the occlusion
challenge by employing a completion task that enables the

Pose & Size 
Prediction
(C) Global Context Enhanced Feature Fusion
Key Point 
Feature
Extraction 
(A) Robust Partial Feature Extractor
(B) Semantic Shape Reconstruction (SSR)
Cross-Attn.
Self-Attn.
NOCS Predictor
Object Pose
(D) Pose & Size Estimator
SSR
DINOv2
PointNet++
PointNet++
Figure 2. Illustration of GCE-Pose: (A) Semantic and geometric features are extracted from an RGB-D input. A keypoint feature detector
identifies robust keypoints and extracts their corresponding features. (B) An instance-specific and category-level semantic global feature is
reconstructed using our SSR module. (C) The global features are fused with the keypoint features to predict the keypoint NOCS coordinates.
(D) The predicted keypoints, NOCS coordinates, and fused keypoint features are utilized for pose and size estimation.
network to reason about full geometric representations de-
spite missing points.
2.3. Generalizing Object Pose Estimators
The constraints of 3D model-based pose estimation have
been relaxed to category-level estimation, where the task
is to predict the pose of an unknown instance within a
known category (e.g., a fork in ”cutlery”) on benchmarks
like NOCS [66], PhoCal [69], and HouseCat6D [29].
Category-level pose estimation aims to predict 9DoF
poses for novel instances within specified categories.
Wang et al. [66] introduced the Normalized Object Coor-
dinate Space (NOCS) framework, mapping observed point
clouds to a canonical space with pose recovery via the
Umeyama algorithm [62]. Subsequent methods improve ac-
curacy [5, 7, 11, 37, 41, 79, 81]. Some works adopt prior-free
methods, such as VI-Net [39], which separates rotation com-
ponents, and IST-Net [42], which transforms camera-space
features implicitly. AG-Pose [41] achieves state-of-the-art
results by learning keypoints from RGB-D without priors.
In contrast, we incorporate learned priors for geometry and
semantics to complete partial observations and map mean
shape semantics onto observed instances.
Self-supervised approaches are also popular in category-
level estimation, refining models without annotated real
data. CPS++ [45] uses a differentiable renderer to adapt
synthetic models with real, unlabeled RGB-D inputs, while
other works [68, 70] tackle photometric challenges us-
ing RGBP polarization [18, 54] and contextual language
cues [70]. Self-DPDN [38] employs a shape deformation
network for self-supervision, while our approach leverages
a categorical shape prior without network refinement.
In open-vocabulary settings, POPE [16] introduces
promptable object pose, (H)Oryon [9, 10] use vision-
language models and stereo matching, while NOPE [46]
and SpaRP [75] predict pose distributions or relative NOCS-
maps. These methods often treat pose estimation as corre-
spondence matching [10, 16] or reconstruction [46]. We
instead deform a mean shape within an absolute category
space to capture instance-specific correspondences.