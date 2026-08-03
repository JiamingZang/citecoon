# MatchU: Matching Unseen Objects for 6D Pose Estimation from RGB-D Images

> 2024 · id: W4402727146 · arXiv: 2403.01517 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent learning methods for object pose estimation re-
quire resource-intensive training for each individual object
instance or category, hampering their scalability in real
applications when confronted with previously unseen ob-
jects. In this paper, we propose MatchU, a Fuse-Describe-
Match strategy for 6D pose estimation from RGB-D images.
MatchU is a generic approach that fuses 2D texture and 3D
geometric cues for 6D pose prediction of unseen objects. We
rely on learning geometric 3D descriptors that are rotation-
invariant by design. By encoding pose-agnostic geometry,
the learned descriptors naturally generalize to unseen ob-
jects and capture symmetries. To tackle ambiguous asso-
ciations using 3D geometry only, we fuse additional RGB
information into our descriptor. This is achieved through
a novel attention-based mechanism that fuses cross-modal
information, together with a matching loss that leverages
the latent space learned from RGB data to guide the de-
scriptor learning process. Extensive experiments reveal the
generalizability of both the RGB-D fusion strategy as well
as the descriptor efficacy. Benefiting from the novel designs,
MatchU surpasses all existing methods by a significant mar-
gin in terms of both accuracy and speed, even without the
requirement of expensive re-training or rendering.

## introduction
Object 6D pose estimation is a critical task in computer
vision applications, such as robotic manipulation [39, 64],
augmented reality [1, 36], and autonomous driving [22, 32].
While object 6D pose estimation with object-specific train-
ing has achieved impressive results on benchmarks [50],
handling unseen objects still remains a challenge.
Ap-
proaches like template matching [26], keypoint detection [8,
17, 53], surface mapping [14, 45], and reconstruction-based
frameworks [29, 31, 47] have achieved high accuracy for in-
dividual objects [14, 17, 18, 47, 53]. However, these meth-
ods are not designed to handle multiple objects or generalize
to objects not presented in the training data.
Dataset-level pose estimation methods [52] can han-
Figure 1. MatchU provides a pipeline to match a previously un-
seen 3D CAD model of an object to an RGBD image (Top left).
(Fuse) Information from RGB-D and CAD is fused. (Describe)
Consumes fused information and produces generic color-aware
rotation-invariant 3D descriptors. (Match) Further used for estab-
lishing correspondences as well as the 6D pose.
dle multiple objects in a dataset but struggle when faced
with new instances. Similarly, category-level pose estima-
tion methods [54] generalize to new instances within the
same category but struggle with new categories. These ap-
proaches do not apply to the challenging problem of unseen
object pose estimation in real-world applications where the
3D model is only available during inference time, since
the models are designed to overfit the specific distribution
of one object, category, or dataset. Some one-shot learn-
ing methods attempt to align object models using template
matching or capture the structure from motion (SfM) of
unseen objects [44, 47].
However, these approaches of-
ten require object-specific preprocessing steps. Classic ap-
proaches that target unseen object pose estimation employ
handcrafted features and correspondences between CAD
models and observed RGB-D images [11, 28, 42]. How-
ever, these methods introduce many pose hypotheses with
high ambiguity, which require to be rated and refined itera-
tively, resulting in computational overhead.
arXiv:2403.01517v2  [cs.CV]  8 May 2024

Recent works have extensively delved into generic pose
estimation through learning models that rate and refine
pose hypotheses [3, 30], predominantly utilizing time-
consuming render-and-compare strategies. This constraint
limits their utility in real-world applications. Alternatively,
some approaches formulate the generic 6D pose estimation
problem as a point cloud registration task, benefiting from
the point cloud representation backbones but neglecting
crucial texture information from RGB images [6, 65], lead-
ing to less distinct point cloud-based descriptors. There-
fore, they introduce ambiguities in correspondence extrac-
tion partially remedied by adding knowledge about the ob-
ject symmetry during training [65].
We present MatchU, a Fuse-Describe-Match strategy for
unseen object 6D pose estimation from single RGB-D im-
ages as shown in Figure 1. Our method is designed to ex-
tract rotation-invariant descriptors that can be shared across
a wide range of objects, facilitating generalization to un-
seen objects. The extraction of rotation-invariant descrip-
tors is crucial as it allows our method to inherently capture
and model the natural symmetry of objects without rely-
ing on explicit symmetry annotations. However, rotation
invariance still has some ambiguity where one point can
be matched to several geometrically similar points. To ad-
dress the ambiguity problem introduced by rotation invari-
ance, we introduce a novel 2D-3D fusion module termed
Latent Fusion Attention Module. This module effectively
combines texture and geometric information. This results
in extracting descriptors that describe both the appearance
and the shape features of an object in a complementary and
generic manner. Furthermore, we propose a novel Bridged
Coarse-level Matching loss that leverages RGB informa-
tion to enhance the learning of geometric descriptors. This
loss function strengthens the association between texture
and geometric features, leading to more precise and accu-
rate matching between CAD models and RGB-D images of
unseen objects. Our main contributions are:
• We propose MatchU, a 6D pose estimation fuse-
describe-match strategy that extracts fused RGB-D in-
put features targeted to register an unseen 3D CAD model
to an object in the scene.
• We introduce a novel Latent Fusion Attention Mod-
ule to effectively fuse texture and geometric features
for generic pose estimation from RGB-D data and train
MatchU with a Bridged Coarse-level Matching Loss.
• MatchU captures symmetries inherently by learning a
fused feature representation without additional annota-
tions thus reducing pose ambiguities.

## method
3.1. Problem Formulation
The task of unseen object pose estimation aims at estimating
the 6D pose between a CAD model, which is not available
during training, and its partial observation from the RGB
and/or depth image. In this paper, we estimate the pose of
unseen objects by matching the learned descriptors between
RGB-D data and its CAD model. The input of our method
includes an unseen CAD model represented as a point cloud
P = {pi ∈R3 | 1 ≤i ≤n} with n points, the partial point
cloud obtained from the depth channel, denoted as Q =
{qj ∈R3 | 1 ≤j ≤m} with m points, as well as its
corresponding RGB image crop K of the localized object.
Corresponding points pi ↔qj are collected in the predicted
correspondence point set C which is used to estimate the 6D
pose of the novel object, by optimizing the objective
  \ lab
el 
{ eq:opt
i
mization}
 \b eg i n {al
igned} \mathcal {T^{*}} = \arg \min _{\mathcal {T} \in \text {SE}(3) } \sum _{(p_{i}, q_{j}) \in \mathcal {C}} \|(\mathcal {T}p_{i} - q_{j})\|^{2}_{2} \end {aligned} 
(1)
of mutual 3D correspondences. T ∈SE(3) denotes the
6D pose of the novel object in the Special Euclidean group
SE(3) of rigid transformation in 3D space.
3.2. Method Overview
To solve for object pose, we first calculate correspondences
by extracting the generic descriptors ϕP and ϕQ for the
points in P and Q in latent space Rd. By calculating the
similarity between ϕP and ϕQ, we can construct the cor-
respondence set as C = {(pi, qj) | ϕP
i
↔ϕQ
j }, where
ϕP
i
↔ϕQ
j denotes matched descriptors of the point pi
and qj, respectively. The optimization problem (Eqn. 1)
is designed as a least square problem that can be robustly
solved with an outlier-aware consensus algorithm such as
RANSAC [12].
From the input CAD point cloud P, depth point cloud
Q, and the RGB image K, our proposed method estimates
a mapping function ψ that maps P and Q to generic de-
scriptors ϕP = ψ(P | (Q, K)) ∈Rn×d and ϕQ = ψ(Q |
(P, K)) ∈Rm×d by fusing the cross-modality informa-
tion from (Q, K) and (P, K), respectively. By matching
our learned generic descriptors, correspondences are estab-
lished between the unseen object and its partial observation,
and the object pose is finally estimated. An overview of our
framework is depicted in Fig. 2.
3.3. Encoding and Fusing Descriptors
We first introduce the extraction of 3D and 2D local fea-
tures, and then the cross-modality descriptor fusion.
Local 3D Feature Extraction.
We employ the recent
transformer-based architecture RoITr [61] as our encoder
backbone to extract rotation-invariant 3D local features

: RGB
: Depth
: CAD
Inputs
1. Encoding and Fusing  
2. Learning to Describe
3. Matching and Estimating 6D Pose
6D Pose
Latent Fusion Attention Module
Latent Fusion Attention Module
2D-to-3D Fusion Block
3D-to-2D Fusion Block
Global
Global
Fusion
Fusion
Fusion
Fusion
Fusion
Fusion
Fusion
Fusion
Global
Global
(a) Method Overview
(b) RGB-D Fusion 
Latent Fusion Attention Module
Figure 2. Overview of MatchU. Upon encountering an unseen object, we initially derive the segmented depth point cloud Q and the
corresponding RGB image crop K utilizing a pre-trained generic segmentation network. Subsequently, we procure both 3D and 2D local
features from the CAD point cloud P, depth point cloud Q, and the RGB image crop K. These extracted features are then amalgamated
within a latent space through our innovative Latent Fusion Attention Module, under the guidance of a Bridged Coarse-level Matching Loss
(BCM Loss) LcP KQ. The refined 3D descriptors eϕP ′ and eϕQ′ are fed into decoders, which enhance the resolution of the descriptors to
ϕP and ϕQ, this process being steered by a detailed matching loss LP Q
f
. In the final stage, the 6D pose of the novel objects is deduced by
aligning the descriptors within the latent space and aggregating the pose parameters T .
from the CAD point cloud P and the partially observed
point cloud Q from depth image. The inherent rotation-
invariance of the descriptor provides a robust feature ex-
traction for geometric cues and guarantees the generaliz-
ability for unseen objects. Given P and Q, our encoder
down-samples the input point clouds via Farthest Point
Sampling (FPS) to superpoints.
They represent a well-
distributed coarse representation of spatial structure from
the underlying dense point cloud defined as P ′ = {p′
i ∈
R3 | 1 ≤i ≤n′} and Q′ = {q′
j ∈R3 | 1 ≤j ≤m′},
where n′ and m′ stand for the number of superpoints in P ′
and Q′, respectively. Following [61], for each superpoint
p′
i and q′
j, we first extract the local geometric features from
neighboring points within a radius of r. The local geometric
cues are then projected into the latent space by a sequence
of attention blocks, from which we obtain the inherently
rotation-invariant local 3D geometric descriptors, denoted
as ϕp′
i ∈Rd, and ϕq′
j ∈Rd where d is the dimension of the
latent space.
Local 2D Feature Extraction.
A convolutional neural
network (CNN) is used for local visual feature extraction.
Following LoFTR [46], we adopt a modified encoder of
FPN [35] as our CNN backbone. This 2D encoder down-
samples the input image crop of size H × W to a feature
map of size H
8 × W
8 , while simultaneously projecting the
local textural information into a d-dimension latent space
consistent with the 3D geometry features. The image’s lo-
cal feature map is then flattened into ϕK′ = {kt ∈Rd1 ≤
t ≤H
8 × W
8 }, where we denote the 2D superpixels as K′
and the 2D superpixel features as ϕK′.
Latent Fusion Attention Module. After extracting 3D and
2D local features, we fuse the encoded 3D and 2D context in
latent space via our proposed Latent Fusion Attention Mod-
ule. To keep the generalizability of our network and avoid
overfitting on object-specific features, we propose to fuse
the 3D and 2D features in a coarse-level latent space and
leverage a 3D-to-2D Fusion Block as well as a 2D-to-3D
Fusion Block for fusing the information in two perspectives.
We leverage the Latent Fusion Transformers (green layers)
and Global Transformers (blue layers) in these two fusion
blocks as shown in Figure 2 (b).
Previous methods [17, 18, 52] usually interpolating fea-
tures w.r.t. their spatial relationships explicitly. In MatchU,
we use the positional encoding in the attention mechanism
to incorporate spatial awareness and implicitly align differ-
ent modalities. For 2D features, we follow DETR [5] to
encode the spatial information of the 2D feature map into
the feature space. As for 3D, instead of encoding the raw
position of the points [66], we propose to use the pose-
agnostic Point Pair Features (PPFs) [11] as the position rep-
resentation following [61], which guarantees the geomet-
ric rotation-invariance and generalizability for unseen ob-
ject pose estimation.
The Latent Fusion Transformer is designed to fuse the 2D
superpixel features and 3D superpoint features in the latent
space, which consists of a series of self-attention and cross-
attention layers. Following [46], we adopt the linear atten-
tion [25] for all the self- and cross attention layers with
the goal of lower computational complexity. We stack g
self- and cross-attention layers for each Latent Fusion trans-

former in practice. The Global Transformer is designed to
aggregate the global context of the 3D and 3D features, for
which we follow the design of RoITr [61].
Details for both 3D-to-2D Fusion Block and 2D-to-3D
Fusion Block are illustrated in Fig. 2. For the 3D-to-2D
Fusion Block, we first aggregate CAD ϕP ′ and depth ϕQ′
superpoint features with a Global Transformer. Then the
RGB feature ϕK′ is fused with the global-aware depth and
CAD feature sequentially to get the final cross-modal 2D
feature eϕK′ for each superpixel by Latent Fusion Trans-
former. For the 2D-to-3D Fusion Block, we first separately
enhance both CAD ϕP ′ and depth ϕQ′ superpoint features
with RGB features through Latent Fusion Transformer. A
Global Transformer then co-injects this information to pro-
vide the 2D-aware 3D superpoint features eϕP ′ and eϕQ′ for
both CAD and depth.
3.4. Learning to Describe
In order to guide the learning of the fused descriptors, we
propose several loss functions.
With the latent features
learned from RGB images as the bridge between the la-
tent spaces of the CAD and depth point clouds, we define
Bridged Coarse-level Matching Loss (BCM Loss), which
significantly facilitates the unification of two different 3D-
based latent spaces, and helps to generate more robust and
reliable correspondences between superpoints. Moreover, a
fine-level matching loss is also introduced to guide the re-
finement of superpoint matches to point correspondences.
Bridged Coarse-level Matching Loss. To ensure the effec-
tiveness of RGB-based 2D information in the latent space,
the key is to provide the supervision signal from both 2D
and 3D modality by establishing the cross-modal matches
between 2D and 3D features.
The alignment between
the superpoints P ′ and Q′ can be obtained

## related_work
The majority of related work focused on 6D pose estima-
tion of seen objects for which training data (real or syn-
thetic) is available. However, they need to be retrained for
any new object instance. There were extensions to object
category pose estimation, but they cannot generalize to new
unseen categories. Therefore, in recent two years several
approaches, that aim at generalizing to novel unseen objects
without retraining, were introduced.
2.1. Seen Object Pose Estimation.
The approaches for seen object pose estimation rely on
available real or synthetic training data and train one neu-
ral network model per object or per scene. They are usually
multi-stage pipelines where the core learning efforts are in
establishing image-to-model (2D-3D) correspondences fur-
ther used for pose estimation through PnP+RANSAC or
direct regression. A larger amount of learning-based ap-
proaches consume RGB as input and only a few of them
focus on RGB-D inputs facing the challenge of fusing RGB
and depth information in neural networks.
RGB-D Fusion Methods are important because they profit
from complementarity of two data sources and naturally
improve pose accuracy as demonstrated in early works
[19, 62]. In deep learning approaches, features are extracted
separately from two modalities with different neural net-
works and their fusion is not obvious. Early approache, like
PointFusion [58], extracts global RGB (CNN) and depth
(PointNet) features from the patch containing the object,
and fuses them with per-point depth features for 3D ob-
ject bounding box detection. Later, DenseFusion [52] per-
forms late per-point feature fusion strengthened with the
global information, allowing better discrimination at the lo-
cal level and resulting in better occlusion handling. Other
works like PVNet3D [17] rely on DenseFusion and esti-
mate sparse keypoints instead of dense correspondences.
FFB6D [18] instead uses bidirectional fusion modules to
combine modality information at earlier stages and produce
stronger per-pixel fused features. Recently, DFTr [66] uses
Transformers and improves the data fusion with the global
semantic similarity between RGB and depth. This fusion
strategy can help handle missing and noisy data caused by
reflections or low-texture information.
Symmetric Objects are problematic because they look the
same from different viewpoints.
Correspondence-based
methods have issues with visual ambiguities cause [38] as
one-to-many matches define multiple equally correct poses.
This has been tackled if symmetry information is known
beforehand and used for data preparation [63] or in loss
functions [53, 65].
Contrary to this SurfEmb [14] does
not require known symmetry and learns symmetry invari-
ant features with contrastive loss. Learned 2D-3D descrip-
tions from SurfEmb [14] are not guaranteed to be invari-
ant to rigid object transformations; it robustly learns quasi-
invariance from a large dataset for specific objects. Addi-
tionally, predicting pose distribution [15, 24, 40] instead of
a single estimate elegantly circumvents this problem.

2.2. Unseen Object Pose Estimation
Pose estimation of unseen objects considers that the neu-
ral network model is trained once and can generalize to
novel unseen objects without retraining. For long, hand-
crafted feature matching using point pair features(PPF) [11]
has been a competitive method in BOP challenge [50] en-
abling unseen object pose estimation. Its main disadvantage
is efficiency due to large voting spaces and adding RGB
to PPF [11] brought some benefits. Recently, Gen6D [37],
OnePose [47] and OnePose++[16], utilize SfM and feature
matching techniques to align a posed set of images of a
given object to a target view using refined nearest neighbor
image retrieval [37] or 2D-3D image matching [46].
Template-based

## conclusion
We present MatchU, a Fuse-Describe-Match framework for
unseen object pose estimation from single RGB-D images.
Our method first extracts rotation-invariant descriptors from
3D point clouds of CAD model and depth map.
Then,
the multi-modal fusion of texture and geometry is achieved
through a Latent Fusion Attention Module.
A Bridged
Coarse-Level Matching Loss is introduced to utilize latent
features from RGB images to connect descriptions of par-
tial observations and full object geometry. MatchU inher-
ently models object symmetry without explicit annotations.
MatchU surpasses all existing methods for unseen object
pose estimation by a large margin on standard benchmarks.
Certainly, it relies on external object localization and could
be could be affected by their erroneous results. In the fu-
ture, incorporating such modules into the pipeline to build
end-to-end training might further improve our results. We
believe that by closing the gap to object-specific baselines,
MatchU constitutes an important step forward to truly scal-
able 6D pose estimation of unseen objects.