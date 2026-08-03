# RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation

> 2025 · id: W4415967370 · arXiv: 2510.18521 · pdf: https://arxiv.org/pdf/2510.18521 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Typical template-based object pose pipelines estimate the
pose by retrieving the closest matching template and align-
ing it with the observed image.
However, failure to re-
trieve the correct template often leads to inaccurate pose
predictions. To address this, we reformulate template-based
object pose estimation as a ray alignment problem, where
the viewing directions from multiple posed template images
are learned to align with a non-posed query image. In-
spired by recent progress in diffusion-based camera pose es-
timation, we embed this formulation into a diffusion trans-
former architecture that aligns a query image with a set of
posed templates. We reparameterize object rotation using
object-centered camera rays and model object translation
by extending scale-invariant translation estimation to dense
translation offsets.
Our model leverages geometric pri-
ors from the templates to guide accurate query pose infer-
ence. A coarse-to-fine training strategy based on narrowed
template sampling improves performance without modify-
ing the network architecture. Extensive experiments across
multiple benchmark datasets show competitive results of
our method compared to state-of-the-art approaches in un-
seen object pose estimation.

## introduction
Multi-view vision is a core element for 3D perception [12].
Spatial understanding and measurements often depends on
multiple cameras or temporally-varied perspectives over
time to reason about the surrounding in 3D. Also for the
task of object pose estimation – the prediction of rotation
and translation of objects in space, multi-view constraints
can be beneficial [25]. In many computer vision applica-
tions, like robotic bin picking, augmented reality, and au-
tonomous driving, multiple cameras or acquisitions are not
Project page: https://demianhj.github.io/projects/RayPose
{frstname.lastname}@tum.de
Multiview Posed Templates
Query Image 
RayPose Diffusion
Noisy Rays
Clean Rays
Query Pose
Conditioning
Figure 1. Given a novel object query image, our method accu-
rately predicts the object’s 6D pose using a multiview diffusion
model conditioned on a set of template images with known poses.
Leveraging our proposed structured 2D pose maps, represented as
bundles of rays, the diffusion model recovers the query object’s
pose by progressively denoising these ray bundles.
available and the system needs to function even with a sin-
gle monocular RGB image.
In object pose estimation literature, much effort has been
put into learning other constraints, such as object appear-
ance from visual data during training. Instance-based ap-
proaches [48, 53] therefore get their constraint from access
to model appearance during training while category-level
approaches [7, 22, 28, 54] use object shape and seman-
tic priors. Despite the excellent results that benefit from
deep learning, these approaches require training for every
new object or object category from scratch and creating
synthetic training data from a CAD model is also compu-
tationally expensive. To overcome per-object training, re-
searchers have been working on unseen object pose estima-
tion with access to textured CAD models during inference
[6, 19, 26, 40, 42, 56, 60]. These advancements promise
to overcome the scalability and flexibility hurdles of object-
specific approaches.
These approaches are unable to access multiple views
by input design and template approaches typically solve a
1
arXiv:2510.18521v1  [cs.CV]  21 Oct 2025

classification task first: which is the best template given an
image query? Consecutive steps after template matching in-
volve correspondence estimation, pose prediction, and op-
tionally refinement [40, 46, 56]. Instead of finding the best
possible posed template and then building pairwise corre-
spondences, we think of the problem as an implicit bundle
agreement among multiple views, using multiple template-
query tuples to reason about 3D, with the advantage of hav-
ing the template already posed.
Learning to reason about 3D from multiview inputs has
been extensively studied in prior work [14, 25, 31, 45].
More recently, diffusion models have emerged as powerful
tools for 3D reasoning, demonstrating remarkable general-
ization capabilities [2, 35, 36, 51, 52, 55, 58, 59]. Among
them, PoseDiffusion [55] addresses the inverse problem of
structure-from-motion by directly diffusing camera poses
within a probabilistic diffusion framework, modeling the
conditional distribution of poses given input images. Build-
ing upon this, recent work [59] introduces an overparame-
terization of camera poses using Pl¨ucker coordinates [44],
representing a pose as 2D maps of ray direction and ray
moment. This formulation is shown to be more compat-
ible with diffusion processes and leads to improved accu-
racy in relative pose estimation. These approaches exhibit
strong generalization and can infer relative camera poses
even in novel scenes composed of entirely unseen images.
Motivated by this capability, we propose to leverage a set
of posed template images and a single query image to esti-
mate the 6D pose of an object in the query by building on
the strengths of multiview diffusion-based backbones.
Although diffusion models have shown success in rela-
tive camera pose estimation [55, 59], they are suboptimal
for object pose estimation due to scale differences: camera
poses are defined in a large world coordinate system, while
object poses reside in a compact, object-centric space. To
bridge this gap, we propose novel object-centric pose rep-
resentations tailored for 6D object pose estimation. For ro-
tation, we replace camera-centric Pl¨ucker coordinates with
an object-centered formulation where rays are structured
as a 2D image-aligned grid.
For translation, we extend
the Scale-Invariant Translation Estimation (SITE) frame-
work [29] to generate a dense translation map. This object-
centric parameterization enables more precise and disentan-
gled reasoning about object-level 6D pose within the diffu-
sion framework. Our structured pose diffusion framework
takes a query image of an unseen object cropped from the
scene and a set of posed images as templates, obtained by
synthetic rendering from a CAD model, and generates pre-
cise 6D object pose predictions. We also propose a coarse-
to-fine object pose estimation strategy by sampling the tem-
plate with a narrower distribution based on the inputs. We
evaluated our method on standard benchmark datasets from
the pose estimation benchmark [50] and compared it to re-
cent methods for unseen object pose estimation. The per-
formance of our method surpassed the results of the re-
lated works, and a detailed ablation study verified our de-
sign choices. This paper makes the following contributions:
• we formulate unseen object pose estimation as ray
bundling problem between multiview templates and RGB
query, which helps the network to capture the correlation
between query and templates in 3D space.
• we introduce object-centric orientation and translation
over-parameterization suitable for learning within diffu-
sion framework.
• we propose a flexible diffusion-based 6D object pose
framework for unseen object pose estimation that can be
extended to a coarse-to-fine prediction by using different
template sampling

## method
3.1. Method Overview
In this paper, we represent the 6D object pose using pose
maps M, which encode both orientation and translation. As
illustrated in Fig. 2, we adopt a multiview diffusion trans-
former framework that learns to estimate object pose by de-
noising noisy pose maps conditioned on an input query ob-
ject image and a set of reference images with known object
poses(termed posed templates). We extract a query embed-
ding FQ and the multiview template embedding FMV us-
ing the query and template encoders, respectively. Each en-
coder consists of an image encoder EI that extracts 2D im-
age features, and a view encoder EV that encodes 6D object
pose and/or 2D object location. Specifically, the multiview
template features are fused using a Multiview Fuser to form
the embedding FMV . A Diffusion Transformer Decoder is
then trained to reconstruct the clean pose maps M0 from
noisy inputs Mt, conditioned on both FQ and FMV . We
train our model with two different template sampling strate-
gies to obtain both coarse and fine pose predictors. For the
coarse predictor, template viewpoints are randomly sam-
pled independently of the query pose. For the fine predictor,
the same model is trained with templates sampled from a
narrower distribution centered around the query pose. This
strategy enables coarse-to-fine pose inference during testing
without any changes to the network architecture.
3

3.2. Object Pose Parameterization
The 6D object pose is defined by its rotation R ∈SO(3)
and translation t ∈R3, representing the transformation
from the object’s local coordinate frame to the camera co-
ordinate system. While compact pose regression is desir-
able, it remains challenging for neural networks, especially
in generic or cluttered scenes. Recent work [59] overpa-
rameterizes camera poses using ray directions and ray mo-
ments based on Pl¨ucker coordinates [44], which has proven
effective for scene-level camera pose estimation. However,
this formulation entangles camera intrinsics, rotation, and
translation, limiting its effectiveness for object-level pose
tasks. Specifically, inaccuracies in the predicted direction
map can propagate to the translation component, hinder-
ing the centimeter-level precision required in object pose
estimation. To overcome this, we propose a novel object-
centric representation that maps the 6D object pose into
separate 2D rotation and translation maps, enabling more
accurate and disentangled learning.
Rotation Parameterization.
Camera pose estimation or
novel view synthesis methods often model camera-centered
rays, where rays originate from the camera center and pass
through pixel coordinates in the image plane. In contrast,
we introduce an object-centered ray representation, where
the object center is treated as a virtual pinhole camera,
emitting rays toward the camera coordinate system. Given
the camera intrinsic matrix K ∈R3×3 and extrinsic pa-
rameters—rotation R ∈SO(3) and translation t ∈R3,
a 3D object point x is projected onto the image plane as
u = K[R | t]x. Instead of relying on this conventional
image-based projection, we define a structured representa-
tion in which object-centered rays are mapped onto a nor-
malized 2D square grid using a uniform intrinsic matrix, de-
noted as K = KI. The set of direction vectors originating
from the object center is represented as
MR = {d1, . . . , dn}
(1)
where each direction vector di is normalized to unit length.
This formulation enables us to map arbitrary rotation ma-
trices R onto a unique structured grid on the unit sphere
surface.
To construct the ray map, we uniformly select
{di}n
i=1 on the projected grid of the sphere surface, ensur-
ing that each vector passes through the center of its cor-
responding grid cell. Consequently, we obtain a 2D grid
map with the shape of (p × p × 3) as our rotation repre-
sentation in the diffusion process. The illustration is given
in the supplementary. Given the object-centered ray repre-
sentation, we recover the rotation matrix R by aligning the
predicted ray directions with a predefined canonical frame.
Let MR = {d1, . . . , dn} be the predicted ray set and
M∗
R = {d∗
1, . . . , d∗
n} the reference rays corresponding to
an identity rotation R = I. The optimal rotation matrix R∗
is obtained by solving:
Object centric ray bundles
2D Rotation map 
Figure 3. Visual illustration of the object-centric ray representa-
tion used for rotation prediction in our diffusion model. The rota-
tion map MR is defined as a bundle of rays originating from the
object center Oobj, encoded as a 3-channel 2D map.
R∗= arg min
R∈SO(3)
n
X
i=1
∥Rd∗
i −di∥2
(2)
where R is the relative rotation of the object with respect to
the canonical frame. This problem can be solved using the
Singular Value Decomposition (SVD) differentially, ensur-
ing a valid rotation by enforcing RT R = I. This formu-
lation allows for robust recovery of the object’s orientation
and enables the diffusion process on 3D rotations from a
structured 2D ray representation.
Translation Parameterization. A major challenge in es-
timating an object’s 6D pose from a single RGB image is
minimizing translation error, particularly for previously un-
seen objects and scenes. Earlier work, SSD6D [23], esti-
mates translation by locating the object centroid in 2D co-
ordinates and comparing the bounding box scale with a pre-
rendered template of the same rotation to determine object
distance. However, this approach assumes the object center
aligns with the bounding box center, making it sensitive to
occlusion. Instance-level regression-based methods [29, 53]
improve robustness by employing Scale-Invariant Transla-
tion Estimation (SITE), which predicts translation by com-
puting the offset between the bounding box center and the
object center.
More recently, generalizable RGB-based
methods [40, 42] estimate translation by establishing 2D
correspondences between query and template images us-
ing a pre-trained feature matcher. While template depth can
be rendered, these methods rely solely on one RGB image
pair for correspondence extraction. In this paper, we extend
SITE to a patch-level dense translation map. Given the ob-
ject translation t = [tx, ty, tz] and the camera intrinsic ma-
trix K, the projected object centroid [ox, oy, 1]T in image
coordinates is computed as:
[ox, oy, 1]T = Kt.
(3)
We estimate the offset from each pixel (u, v) in the detected
bounding box to the object centroid (ox, oy), forming a
dense normalized translation offset map:
MT =
u −ox
w
, v −oy
h
, tz
rz

,
(4)
4

where w and h denote the bounding box width and height,
and rz is the zoom-in ratio of the bounding box. Similar
to the rotation map, we uniformly sample the pixels in the
bounding box with the same shape of (p × p × 3) as the
2D translation map in the diffusion process. The 3D ob-
ject translation is then recovered by back-projecting the es-
timated centroid offset using the camera intrinsics:
t∗= rz · K−1[w · ∆ox + ox, h · ∆oy + oy, ∆oz]T . (5)
To this end, we represent object pose as 2D pose maps M =
(MR, MT ). This pose representation decouples rotation
and translation as well as the camera intrinsics, enabling the
model to predict rotation and translation independently and
enabling the use of a diffusion model to denoise the pose on
two dense 2D maps.
3.3. Multiview Template Conditioned Diffusion
In our framework, we employ a multiview diffusion model
to estimate object pose by conditioning it with the input
query image and the posed templates. This network for-
mulates the learning as a denoising process that gradually
refines noisy inputs into the proposed structured pose maps.
3.3.1. Diffusion Preliminaries
Diffusion process.
The diffusion process consists of a
forward (noising) and a reverse (denoising) process. Given
a clean pose representation M0 (either the rotation map
MR or the translation map MT ), the forward process adds
Gaussian noise over a fixed number of timesteps T. At each
timestep t ∈{1, . . . , T}, the pose map is perturbed as:
Mt = √αtM0 +
√
1 −αtϵ,
ϵ ∼N(0, I),
(6)
where αt is a noise schedule controlling the variance at
timestep t.
Denoising process. The reverse process aims to recover
the clean pose representation by learning to predict and re-
move the noise. A neural network ϵθ(Mt, t, FC) is trained
to estimate the noise ϵ conditioned on an embedding Fc that
encodes the query and template information. The predicted
pose is obtained by iteratively refining Mt using the learned
noise estimator:
Mt−1 =
1
√αt

Mt −
√
1 −αtϵθ(Mt, t, Fc)

+ σtz,
z ∼N(0, I).
(7)
where σt controls the stochasticity of the denoising step.
This iterative process gradually refines the noisy pose rep-
resentation into a structured output.
3.3.2. Network Architecture
Our diffusion-based framework for 6D object pose estima-
tion consists of three main blocks: (1) the Query Encoder,
which extract the query image features, (2) the Template
Encoder, which encodes and fuse the multiple pose

## experiments
4.1. Experimental Setup
Evaluation Metrics.
We adopt the metric Average Re-
call (AR) proposed by the Benchmark of Pose Estimation
(BOP) [50]. The AR score is calculated with 3 pose-error
functions: Visible Surface Discrepancy (VSD), Maximum
Symmetry-Aware Surface Distance (MSSD), and Maxi-
mum Symmetry-Aware Projection Distance (MSPD). A
pose is considered correct if the pose errors are within a pre-
defined error threshold. The mean recall on the each error
functions is computed over multiple error thresholds. The
overall accuracy of a method is given by the Average Recall
AR = (ARVSD + ARMSSD + ARMSPD)/3.
Training and evaluation datasets.
We train our model
on realistic synthetic datasets generated by Megapose [26],
comprising approximately 2 million images rendered with
BlenderProc [8] using objects from Google Scanned Ob-
jects [10] and ShapeNet [5]. For novel object pose estima-
tion, we evaluate our method on five benchmark datasets:
LM-O [3], T-LESS [15], YCB-V [57], TUD-L [16], and
IC-BIN [9]. Our evaluation is structured as follows: in Sec-
tion 4.2, we compare our method with baselines on novel
object pose estimation; in Section 4.3, we conduct abla-
tion studies where we analyze design components, where
we train and evaluate our method on LM-O dataset.
4.2. Compare to Baselines
We evaluate our method on five benchmark datasets: LM-
O, T-LESS, TUD-L, IC-BIN, and YCB-V, which are unseen
during training, and compare it with recent state-of-the-art
methods that use only RGB images as input. All the meth-
ods use the same detection and segmentation results gen-
erated from CNOS [39] by default, except for OSOP [46].
As shown in Table 1, we analyze different setups, consid-
ering whether refinement and multi-hypothesis predictions
are used.
Our method achieves the highest average AR
across all settings. In the single-prediction setting without
refinement, it improves over the previous best method by
3.4% on average, with notable gains of 6.3% on LM-O and
9.2% on T-LESS. With refinement, our method continues to
outperform the baselines, particularly excelling on TUD-L
and LM-O. In the multi-hypothesis setting, it achieves the
best performance on most datasets, especially with T-LESS
dataset being improved by 3.7%. These results highlight the
effectiveness of our approach in enhancing pose estimation
by leveraging robust pose representations and a diffusion-
based pipeline while ensuring strong generalization across
diverse datasets.
4.3. Ablation Study
We conduct an ablation study on four key components of
our approach: template ground-truth (GT) view embedding,
the multiview setup, the fine-level predictor, and relative
pose prediction. Each component is either removed or re-
placed with an alternative setup, and the results are summa-
rized in Table 2.
Fine predictor. In the refinement stage, we apply both our
fine predictor and MegaPose refinement. As shown in (1)
– (3) of Table 2, our fine predictor improves performance
by 6.3% compared to the coarse prediction. Notably, the
fine predictor does not modify the network itself but instead
utilizes a different template sampling strategy. Further im-
provements are achieved when incorporating an external re-
finer during the refinement stage.
Relative pose prediction. To enhance generalization across
different scenes, camera intrinsics, and viewing conditions,
we predict the relative pose between posed templates and
the query, using the ground-truth template pose to infer
the absolute query pose. In this ablation, we modify the
7

## related_work
The benchmark for object pose estimation (BOP) [16]
has long been dominated by traditional handcrafted fea-
ture matching methods based on point pair features (PPF).
In recent years, learning-based approaches such as GDR-
Net [53], ZebraPose [48], and SurfEmb [13] have surpassed
traditional methods in performance. However, these meth-
ods are instance-specific and require training on each target
object. More recently, the community has placed increasing
emphasis on unseen object pose estimation, which focuses
on estimating the pose of novel objects not encountered dur-
ing training. Below, we describe different pose estimation
pipelines used in this setting.
Model-free approaches. Without 3D model of target ob-
jects, Gen6D [34], OnePose [49], and OnePose++ [14] es-
timate its pose by flipping the structure from motion (SfM)
at its head and matching features to align a posed object im-
age to a test view. MFOS [27] uses posed template images
as a model representation and establishes correspondences
between the input query image patch and the rendered 3D
bounding box of the object associated with each template
image. While attractive, this leads to lower pose accuracy
caused by this rough bounding box approximation of the
object shape.
Template-based approaches. OSOP [47] and OVE6D [4]
utilize a template object representation for 2D segmenta-
tion and coarse to fine matching.
MegaPose [26] pro-
poses a generic render-and-compare refinement strategy.
GigaPose [40] performs a template-matching approach in
two stages: 1) estimates out-of-plane rotation (2 DoF) by
finding discriminative synthetic templates rendered from a
CAD model and then 2) establishes correspondences to es-
timate the four remaining 4 DoF of the object pose.
Foundation models with CAD model prior. The idea of
foundation models is a recent way to incorporate generic
prior knowledge into pose estimation pipelines. Due to the
need for an abundance of labeled data, all approaches are
2

Query Image
Multiview Fuser
（Self-Attentions）
Patch Embedder
View 
Encoder
View Condition
Image
 Encoder
Query Encoder
Template Encoder
Diffusion Transformer Decoder
at time step 
Templates
View Condition
Noisy Pose Maps
View 
Encoder
Image
 Encoder
at time step 
Denoised Pose Maps
Diffusion Transformer Decoder
Norm
Self-Attention
N ×
Patchify + Linear
Conditional Emb.
Norm + Linear
Cross-Attention
Norm + Linear
MLP Head
Time Emb.
Noisy Inputs
Linear
Pos. Emb.
Pos. Emb.
Figure 2. Pipeline overview of our method. We represent the 6D object pose using structured rotation and translation maps and employ
a diffusion model to estimate the pose from random inputs. Given a query image of an unseen object and multiple template images with
known object poses, our method first extracts query embeddings from a query encoder and multiview posed template embeddings from
a template encoder. These embeddings serve as conditioning inputs for a diffusion transformer decoder, which is trained to denoise the
object pose from random inputs. The model predicts the relative pose between the query and templates, from which the absolute 6D pose
of the query object is reconstructed based on the known poses for the templates.
trained on synthetic data. Several methods learn generic 3D
descriptors such as Zeropose [6] and GCPose [60]. Zero-
pose predicts poses utilizing the foundation models of Im-
ageBind [11] and SAM [24] together with 3D-3D feature
matching. GCPose [60] uses explicit knowledge of object
symmetries. FoundPose [42] combines features from the
foundation model DINOv2 [41] and bag-of-words retrieval
for coarse matching and then uses featuremetric alignment
for pose refinement. MatchU [18] and SAM6D [32] build
discriminative descriptors by fusing RGB and depth in-
formation using transformers. Diffusion in Pose Estima-
tion Diffusion models reconstruct a target distribution from
noise over multiple time steps, inherently capturing multi-
modal distributions. They are, by design, capable of cap-
turing multimodal distributions as different noisy initializa-
tions can lead to different predictions during inference in
the case of a multimodal distribution. RayDiffusion [59]
denoises camera poses using ray parameterization for multi-
view estimation, avoiding COLMAP [45] in NeRF training
but is unsuitable for object pose estimation. Object pose
diffusion [17] diffuses poses in SE(3) space, excelling in
synthetic data but struggling with unseen objects and real
datasets. PoseDiffusion [55] addresses the SfM problem by
diffusing camera poses across multiple images, implicitly
performing bundle adjustment. Other methods include Dif-
fusionNOCS [20], an RGB-D approach that diffuses NOCS
maps for pose estimation, and Diff9D [33], which estimates
9D pose by diffusing scale, translation, and rotation based
on image conditioning.

## conclusion
In this paper, we introduced a structured representation for
object pose that enables effective deployment of diffusion
models for object 6D pose estimation.
Instead of pair-
wise matching, we propose aligning object-centered rays
across multiple posed templates. Our multiview diffusion
model is conditioned on embeddings extracted from both
the query and multiple posed template images using ded-
icated encoders. A coarse-to-fine strategy refines pose ac-
curacy without architectural changes, allowing probabilistic
reasoning over multiview inputs without explicit 3D recon-
struction. While achieving competitive performance, the
approach relies on posed templates and accurate detections.
Future work may focus on relaxing these constraints for
broader generalization.
8