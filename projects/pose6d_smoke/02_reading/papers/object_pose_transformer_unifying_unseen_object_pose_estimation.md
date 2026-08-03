# Object Pose Transformer: Unifying Unseen Object Pose Estimation

> 2026 · id: W7140953602 · arXiv: 2603.23370 · pdf: https://arxiv.org/pdf/2603.23370 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Learning model-free object pose estimation for unseen in-
stances remains a fundamental challenge in 3D vision. Ex-
isting methods typically fall into two disjoint paradigms:
category-level approaches predict absolute poses in a canon-
ical space but rely on predefined taxonomies, while rela-
tive pose methods estimate cross-view transformations but
cannot recover single-view absolute pose. In this work, we
propose Object Pose Transformer (OPT-Pose), a unified feed-
forward framework that bridges these paradigms through
task factorization within a single model. OPT-Pose jointly
predicts depth, point maps, camera parameters, and normal-
ized object coordinates (NOCS) from RGB inputs, enabling
both category-level absolute SA(3) pose and unseen-object
relative SE(3) pose. Our approach leverages contrastive
object-centric latent embeddings for canonicalization with-
out requiring semantic labels at inference time, and uses
point maps as a camera-space representation to enable
multi-view relative geometric reasoning. Through cross-
frame feature interaction and shared object embeddings,
our model leverages relative geometric consistency across
views to improve absolute pose estimation, reducing ambi-
guity in single-view predictions. Furthermore, OPT-Pose is
camera-agnostic, learning camera intrinsics on-the-fly and
supporting optional depth input for metric-scale recovery,
while remaining fully functional in RGB-only settings. Ex-
tensive experiments on diverse benchmarks (NOCS, House-
Cat6D, Omni6DPose, Toyota-Light) demonstrate state-of-
the-art performance in both absolute and relative pose esti-
mation tasks within a single unified architecture.

## introduction
Object pose estimation for unseen object instances, with-
out relying on prior CAD models, is fundamental in vi-
sion. It unlocks object understanding to enhance robotic
manipulation, augmented reality, and autonomous systems.
Existing model-free approaches follow two paradigms:
Project Page: https://colin-de.github.io/OPT-Pose/
Relative Pose
Absolute Pose
T r a n s f o r m e r
Depth
Pointmap
NOCS
Input
Trel
Tabs
Figure 1. Unified unseen object pose estimation. OPT-Pose utilizes
a feed-forward transformer to predict point map, depth, NOCS, and
camera parameters. Existing category-level methods predict canonical
absolute 9-DoF SA(3) poses (equivalent to Depth + NOCS), but re-
quire predefined category labels and calibrated cameras. Relative pose
methods align unseen objects across views in 6-DoF SE(3) (equiva-
lent to Pointmap + Depth), but do not support single-view absolute
pose prediction. OPT-Pose enables the simultaneous recovery of both
unseen-object relative and category-level absolute poses (right-most
column) for flexible single or multi-view RGB or RGB-D input, with-
out the need for CAD models or semantic labels.
category-level absolute pose estimation predicts canonical-
space 9-DoF SA(3) transforms for instances within known
categories [5, 12, 27–29, 31–33, 44, 49, 53] but relies
on predefined taxonomies and category labels; relative
pose estimation aligns unseen objects across views via
6-DoF SE(3) [6, 7, 15, 20, 21, 38, 40] but lacks canon-
icalization and cannot handle single-view absolute pose.
However, both paradigms remain constrained. Category-
level methods require explicit category names at inference
[5, 17, 28, 29, 31, 34, 66, 67], limiting their generalization to
open-vocabulary conditions. Additionally, relative methods
typically require multiple views and cannot handle single-
view absolute pose. To the best of our knowledge, no prior
work unifies these complementary tasks in a single category-
agnostic model while leveraging their interplay to improve
pose estimation and generalization to unseen objects.
We propose Object Pose Transformer (OPT-Pose), a uni-
fied feed-forward framework for model-free unseen object
pose estimation with task factorization. OPT-Pose unifies
arXiv:2603.23370v1  [cs.CV]  24 Mar 2026

category-level absolute pose and unseen-object relative pose
in a single model by predicting depth, point maps, camera
parameters alongside NOCS from RGB images. The core
design insight is a complementary geometric mechanism:
• Canonical-space grounding. Depth + NOCS align in-
stances into a shared canonical space, enabling absolute
SA(3) pose estimation without requiring category labels.
• Relative geometric reasoning. Depth + point maps repre-
sent objects in camera space, enabling multi-view SE(3)
reasoning across frames that provides additional geometric
constraints and improves absolute pose estimation.
This factorization bridges camera- and canonical-space rea-
soning without CAD models or predefined taxonomies.
OPT-Pose employs a multi-view transformer that ag-
gregates image tokens and dispatches them to lightweight
task heads.
A keypoint-centric attention module builds
soft correspondences over sampled pixels, while a visual–
geometric fusion block integrates local 3D neighborhoods
with global context to produce discriminative keypoint de-
scriptors. These descriptors are pooled into an object la-
tent embedding and used to FiLM-condition the NOCS
head [41]. We train this latent representation with a con-
trastive InfoNCE objective across views [51], enabling a
shared canonical space without requiring semantic labels at
inference time. Unlike methods that scale to hundreds of cat-
egories but still depend on predefined taxonomies, OPT-Pose
is category-agnostic and treats all objects uniformly. A dedi-
cated camera head estimates intrinsic parameters, enabling
camera-agnostic operation. A metric-recovery head aggre-
gates keypoint-level depth evidence when measured depth is
available, enabling metric-scale recovery in RGB-D mode
while remaining fully functional in RGB-only settings.
Extensive experiments across diverse datasets and tasks,
including category-level absolute pose (REAL275, House-
Cat6D, Omni6DPose [18, 53, 66]) and unseen-object relative
pose (REAL275, Toyota-Light)[16], show that OPT-Pose
achieves state-of-the-art performance on both absolute and
relative pose estimation, generalizing across object cate-
gories, camera types, and input modalities (RGB/-D).
We summarize our contributions as follows:
• Unified Object Poses. A unified model-free framework
for category-level absolute SA(3) and unseen-object rela-
tive SE(3) pose via complementary geometric mechanisms.
We leverage multi-view relative geometric reasoning to
improve absolute pose estimation, without CAD model.
• Category-Agnostic Canonicalization.
We learn a
category-agnostic canonicalization for inference via a con-
trastive objective without class labels.
• Flexible inputs and outputs. OPT-Pose supports RGB
and RGB-D inputs from single or multiple views, achiev-
ing state-of-the-art performance across both category-level
absolute and relative unseen-object pose tasks.

## method
3.1. Design and Task Factorization
We address model-free, unseen-object pose estimation, cov-
ering category-level absolute and relative pose estimation
between views, through a geometric observation: two com-
plementary correspondence pairs are sufficient to link canon-
ical and camera spaces.
1. Depth + NOCS ⇒category-level absolute SA(3) pose.
Depth provides metric 3D observations in camera space,
whereas NOCS yields canonical correspondences; align-
ing these recovers rotation, translation, and scales for
category-level reasoning.
2. Depth + Point Map ⇒unseen-object relative SE(3)
pose. The two 3D representations enable robust cross-
frame alignment without explicit canonicalization.
This factorization naturally supports single- and multi-view
cases with RGB-(D), and is category- and camera-agnostic.
3.2. Problem Formulation
Given a sequence of RGB frames {Ii}S
i=1, we predict per-
frame object geometry and poses, without CAD prior. Let
Ki denote camera intrinsics, Ti ∈SE(3) the camera-to-
world extrinsics (implicitly represented by our pose encod-
ing), and Pi ∈RH×W ×3 the point map (i.e., the 3D camera-
space coordinates). Let further Ii be a set of K sampled
pixels and Xobj
i,k ∈R3 their camera-space 3D points.
In a canonical object space (NOCS), object coordinates
lie in [−0.5, 0.5]3. For each frame, we predict M keypoints
with canonical coordinates Ci,m ∈R3, and corresponding
3D observations Xobj
i,m ∈R3 in the first (anchor) camera
space. The (absolute) category-level object pose can be
retrieved by aligning the canonical to the observed coordi-
nates via transformation in the rigid anisotropic similarity
transformation

SA(3) := R3 × SO(3) × Diag+(3)
(1)
with
Xobj
i,m = RiSCi,m + ti
(2)
and S = diag(si)3
i=1 with SE(3) ⊂SA(3) for S = I
and Sim(3) ⊂SA(3) for S = sI, s > 0. We also estimate
a relative pose ∆Ti ∈SE(3) aligning the two geometry
branches (i.e., depth-derived vs. point-map predictions).
3.3. Multiview Geometry and Feature Transformer
Our model is a single feed-forward multiview transformer
that produces all geometric outputs jointly. (see. Fig. 2). The
aggregator encodes each image into a sequence of tokens
across several refinement iterations, using a visual backbone
[39, 54] with frozen patch embedding for stable training.
From these tokens, the camera head predicts per-frame in-
trinsics through a field-of-view representation and extrinsics
as quaternions. The depth head estimates dense depth ˆDi
and confidence, which we convert to camera-space points
and normals; at sampled indices Ii, we gather points Xobj
i,k
together with colors and normals for keypoint reasoning. A
parallel point-map head predicts a dense 3D point map ˆPi,
and we extract Xpm
i,k as an independent structural cue. A
canonicalization head predicts keypoint-level NOCS coor-
dinates ˆCi,m based on object-centric features fused with a
global latent embedding zobj. Using these canonical coordi-
nates and the observed keypoints, the pose head estimates
(R, t, s), while relative SE(3) is computed after inference
through a weighted Umeyama solver [50]. To recover real-
world scale, the model supports a relative-scale head that
infers scale from RGB features and the object latent zobj, and
an absolute-scale head that uses sensor-depth point clouds
to predict translation and object size in the camera frame if
available. This unified design provides camera parameters,
depth maps, point maps, canonical coordinates, and both
absolute and relative pose within one coherent framework.
For the geometric supervision of camera extrin-
sics,
point
maps,
and
depth
maps
in
normalized
space, we follow [54].
The depth loss follows the
aleatoric-uncertainty
formulation
and
uses
the
pre-
dicted uncertainty map ΣD
i
to weight both the depth
residual and the spatial gradient residual.
The loss is
Ldepth =
N
X
i=1
ΣD
i ⊙( ˆDi −Di)
 +
ΣD
i ⊙(∇ˆDi −∇Di)
 −α log ΣD
i

where
⊙
denotes
channel-broadcast
element-wise
multiplication.
The point-map loss uses the same
structure,
but
with
the
point-map
uncertainty
ΣP
i :
Lpoint =
N
X
i=1

∥ΣP
i ⊙( ˆPi −Pi)∥+ ∥ΣP
i ⊙(∇ˆPi −∇Pi)∥−α log ΣP
i

.
3.4. Keypoint-level Multi-view Feature Fusion
Direct dense pixel-level NOCS regression with attention [52]
is expensive and sensitive to noise. Following keypoint-
based formulations [28, 32], we represent each object by a
compact set of M latent keypoints that attend to joint vi-
sual and geometric evidence. Given a foreground RGB-D
crop, we sample N pixels and lift them to camera-space
points Xk with associated colors and normals. A trans-
former backbone extracts image features f rgb
k
at the sampled
tokens, while a 3D backbone [61] processes (Xk, Ik, nk)
to produce geometric features f geo
k . Concatenation yields
local descriptors fk = [f rgb
k ∥f geo
k ]. A learnable query per-
forms cross-attention using cosine similarity, producing soft
heatmaps Hm,k. Each keypoint is Xobj
m = Hm,kXobj
k , key-
point feature is extracted as Fobj
m = Hm,k[f rgb
k ∥f geo
k ]. A
visual-geometric fusion block [32] then refines Fobj
m by per-
forming KNN grouping in 3D around Xobj
m , encoding rela-
tive offsets and absolute coordinates, and applying cosine-
similarity attention over the local neighborhoods. This aggre-
gation yields enhanced keypoint descriptors ˜Fobj
m that carry
both local geometric context and global object cues.
We further aggregate keypoint features across frames us-
ing a cross-frame attention module. Concretely, we augment
keypoint descriptors with frame-wise sinusoidal positional
encodings and process them with a transformer encoder,
allowing keypoints from different views to exchange infor-
mation at the feature level. In parallel, we pool the object
latent embedding across the input views and share it back
to each frame. This design enables multi-view geometric
reasoning, enforces cross-view consistency, and improves
absolute pose estimation by reducing single-view ambiguity.
To ensure geometric consistency, we constrain Xobj
m to lie on
the object surface using a Chamfer distance loss Lcd.
Lcd =
1
|Xobj
m |
X
x∈Xobj
m
min
y∈Xobj
k
⋆∥x −y∥2
2.
(3)
To prevent keypoints from collapsing into a small region, we
add a diversity regularization that balances surface adherence
and spatial diversity
Ldiv =
1
M(M −1)
X
x̸=y∈Xobj
m
max
 0, τ2 −∥x−y∥2
2, (4)
where τ2 controls the minimum separation between key-
points. To encourage keypoints to be representative of the
depth-lifted point cloud, we employ a lightweight reconstruc-
tion head that takes keypoint positions and features ˜Fobj
m as
input, applies positional encoding, and decodes per-point
displacement deltas to recover the object geometry. The
reconstruction loss is a one-sided Chamfer distance between
the reconstructed point cloud ˆXobj and the observed camera-
space points Xobj
k :
Lrec =
1
 ˆXobj

X
x∈ˆXobj
min
y∈Xobj
k
⋆∥x −y∥2 .
(5)
The keypoint regularization is Lkpt = Lcd + Ldiv + Lrec.

3.5. Canonical Correspondences & Absolute Poses
Given latent keypoint features, the NOCS head pre-
dicts canonical coordinates for each keypoint as ˆCm =
NOCS(˜Fobj
m , zobj). NOCS regression is addtionally condi-
tioned on FiLM based affine transformation with parameters
(γ, β) to intermediate features, so that canonicalization is
conditioned on the object code zobj but remains category-
agnostic. Absolute pose is estimated by an MLP-based pose
and size head that takes ( ˆCm, Xobj
m ) and the corresponding
keypoint features ˜Fobj
m as input. Rotation is represented in
6D [72] and mapped to SO(3) via orthogonalization, while
translation is predicted as a residual with respect to the point-
cloud center following [29, 31]. An isotropic scale ˆs is
obtained from a per-axis size vector ˆs by averaging its mag-
nitudes. The resulting homogeneous transform ˆS = [ˆs ˆR,ˆt]
maps canonical coordinates to the camera frame. To super-
vise the normalized scale prediction, we use
Lpose = ∥Rgt −R∥F + ∥tgt −t∥2 + ∥sgt −s∥2.
(6)
For Lnocs, we use the Smooth L1 loss with
Lnocs = ∥Cgt
m −Cnocs
m ∥SL1
(7)
3.6. Relative Poses from Depth and Point Map
We estimate the metric relative pose by aligning two inde-
pendently predicted 3D structures with a robust, weighted
Procrustes/Umeyama procedure. Let (a, q) denote the an-
chor and query frames. Our model predicts two-view point
maps Pa and Pq in the anchor coordinate system, while
depth and intrinsics yield camera-space point clouds Xa
cam
and Xq
cam. We proceed in two steps:
1. Anchor calibration (Sim(3)). We compute a weighted
Umeyama similarity transform Sa ∈Sim(3) that aligns
the predicted anchor point map to the depth-derived an-
chor camera points,
Sa = argmin
S∈Sim(3)
X
n
wn
S Pa
n −Xa
cam,n
2
2
(8)
where weights wn come from point map confidences.
We then apply Sa to both Pa and Pq, removing global
scale ambiguity due to projective geometry ambiguity for
uncalibrated camera.
2. Query alignment (SE(3)). We then align the calibrated
query point map SaPq to the query camera-s

## experiments
4.1. Implementation Details
We follow the image configuration from [54]. Images are
resized to 518 × 518 with a patch size 14. For each frame
we sample K=1024 pixels to obtain indices Ii used by all
heads. We predict M=128 keypoints. We set the softmax
temperatures in keypoint attention and contrastive InfoNCE
to 1.0, use a repulsion threshold 0.02 in the keypoint diver-
sity loss, pose weight of ϵ=10−3 for absolute pose, Smooth-
ℓ1 with threshold 0.1 for sparse NOCS and relative-scale
supervision (with β=0.1), and kn ∈{8, 16, 32} neighbors
per stage in the fusion block. The multiview transformer
encoder is frozen, leaving the attention to fine-tune the fea-
tures for canonicalization. The model is optimized with
AdamW and parameter groups: NOCS/pose/fusion modules
and projectors use a base learning rate of 5 × 10−4 with
weight decay of 1 × 10−2; the global optimizer uses a learn-
ing rate of 5 × 10−7 with weight decay of 0.05. We apply
5% linear warm-up then cosine decay, mixed precision, and
per-module gradient clipping. We provide more parameter
details in supplementary material.
4.2. Benchmarks and Protocols
We evaluate on three tasks to demonstrate unified categorical
absolute and unseen-object relative pose in a single frame-
work, with flexible RGB-(D) input and intrinsics:
• Category-level absolute pose (RGB-D): HouseCat6D
[18]. We report metrics (5◦, 2cm), (5◦, 5cm), (10◦, 2cm),
(10◦, 5cm) thresholds and 3D IoU. For ROPE [66], we
report: VUS (Volume Under Surface) with rotation thresh-
olds from 1◦to 15◦and translation thresholds from 1 cm
to 5 cm, and AUC (Area Under Curve), which evaluates
Intersection over Union (IoU) of 3D bounding boxes over
IoU thresholds from 0.25 to 0.95.
• Category-level absolute pose (RGB; scale-agnostic on
REAL275): Following prior work, we report normal-
ized IoU (NIoU) and distance thresholds on REAL275 in
RGB-only settings for scale-agnostic pose estimation.
• Unseen-object relative pose (RGB-D): NOCS-REAL,
Toyota-Light (TOYL). Trained on SOPE [66], these
benchmarks test SE(3) alignment across views for unseen
objects. We evaluate using ADD(-S), AR, MSSD, MSPD,
and VSD metrics; relative SE(3) is estimated post-hoc via
weighted Umeyama alignment of depth and point-map
structures (Sec. 3.6).
4.3. Comparison with the State of the Art
Category-level absolute pose (RGB; scale-agnostic
REAL275).
As shown in Tab. 1, our method substantially
outperforms prior work on REAL275 in scale-agnostic eval-
uation. We achieve great performance on all reported metrics
except NIoU75, where we are comparable to UniDet[10]
and surpass GIVEPose [17], DMSR[58], LaPose[70] by
in-degree/normalized distance thresholds. Note that these
methods explicitly use the predefined calculated size and
inference with category priors.
Category-level absolute pose (RGB-D).
Leveraging mea-
sured depth, OPT-Pose recovers metric scale via the absolute-
scale head and achieves strong performance on House-
Cat6D [18] (See Fig. 4). In Tab. 2, we obtain the best
results under strict thresholds (5◦2cm, 5◦5cm) and com-
petitive accuracy at looser thresholds and IoU compared to
GCE-Pose[28], while using no category, shape, semantic,
or calibration priors. In addition, we evaluate a multi-view
enhanced inference mode using S=2, 3, 4 frames without
retraining. As shown in Tab. 2, multi-view inference consis-
tently improves absolute pose accuracy across all metrics.
This supports our central design: relative geometric reason-
ing across views provides additional constraints that reduce
ambiguity in single-view predictions, leading to more stable
and accurate absolute pose estimation. To validate the prac-
tical utility of our unified framework for real-world down-
stream tasks, we evaluate OPT-Pose on the large-vocabulary
Omni6DPose benchmark, designed for robotic manipulation.
As shown in Tab. 3, OPT-Pose surpasses recent state-of-the-
art methods like GenPose++ [66] and CPPF++ [64] in strict
IoU metrics and achieves highly competitive accuracy at
5°/10° thresholds while using less prior and performing in-
ference with a category-agnostic setting. This demonstrates
that our category-agnostic canonicalization effectively scales
to large-vocabulary scenarios, which are critical for robotics.
Unseen-object relative pose.
For unseen objects pose es-
timation, OPT-Pose aligns depth and point-map branches
with a weighted Umeyama to yield robust SE(3) estimation

Anchor Image
Oryon
Ours
Figure 3. Qualitative result in relative pose estimation. We compare different object instances across different scenes with Oryon [7].
Visualization shows that our OPT-pose can estimate the relative object poses across different objects and scenes.
Figure 4. Qualitative result in absolute pose estimation with RGB-D input. We showcase some difficult instances in comparison with
AG-Pose[32]. We zoom in on the difficult object categories, the shoe and the box, for better visualization.
across frames. As shown in Tab. 4, our method outper-
forms all existing methods by large margins across ADD(-S),
AR, MSSD, MSPD, and VSD, especially on NOCS-REAL
(Fig. 3), demonstrating that task factorization within a single
model enables strong performance on both canonical-space
(absolute) and camera-space (relative) reasoning. The slight
performance gain in TOY-L is due to our method being more
sensitive to illumination changes, and to geometric align-
ment being less accurate under these conditions.
4.4. Ablation Studies
We analyze key design choices on HouseCat6D to validate
our unified model-free formulation with task factorization.
• Object latent embedding. Removing contrastive learn-
ing harms canonical correspondence stability and reduces
HouseCat6D accuracy (Tab. 2), showing that object la-
tent learning is key for category-agnostic behavior without
predefined category names during inference.
• Pointmap head. Removing the pointmap head causes
negligible change in absolute pose accuracy, which is ex-
pected: the pointmap branch serves relative SE(3) esti-
mation, not absolute pose. This confirms that the two
geometric pathways are decoupled by design — the abso-
lute pose pathway (Depth + NOCS) is not degraded by the
addition of the relative pose pathway, demonstrating that
task factorization enables both capabilities within a single
model at no accuracy cost to either task.
• Metric head Replacing the absolute metric head with a
test-time Umeyama algorithm for scale calculation weak-
ens pose accuracy under metric evaluation, indicating that

Table 2. Category-level absolute pose estimation (RGB-D) on HouseCat6D. Models are trained on respective training sets and evaluated
on the test set. Prior columns indicate: Category (predefined category input), Shape (shape prior), Semantic (semantic prior), and
Calibration (camera intrinsics). MV indicates whether the method supports multi-view pose reasoning. Bold and underlined denote the
best and second-best results, respectively. We additionally report OPT-Pose with multi-view enhanced inference (S=2, 3, 4), which is only
applicable to our method and demonstrates its ability to leverage cross-view geometric cues for absolute pose estimation.
Dataset

## related_work
Category-level, Model-free Absolute Pose Estimation.
Category-level methods lift objects into a canonical space
using normalized object coordinate systems (NOCS) [53]
and regress per-pixel correspondences or keypoints [28, 32]
to recover 9D poses. While recent category-level approaches
focus on improving accuracy in pose prediction using shape
or semantic priors [3, 4, 8, 29, 32, 68, 71] or extending the
existing category to large-vocabulary methods [2, 19, 66, 69],
they still require predefined taxonomies and explicit cate-
gory names at inference, limiting true open-vocabulary de-
ployment. Many existing methods with high accuracy, like
GCE-Pose[28] and AG-Pose[32], further design specialized
networks with explicit categorical shape and semantic pri-
ors [5, 28, 30, 32, 55]. These constraints prevent generic,
category-agnostic operations.
Feed-forward
Geometry
Transformers.
Recent
feed-forward vision transformers (e.g., DUSt3R [57],
MASt3R [26], CUT3R [56], VGGT [56]) jointly predict
geometric signals such as depth, camera parameters, and
point maps in a single forward pass.
They decouple
representation learning from task-specific heads, enabling
multi-view token aggregation and camera self-calibration.
OPT-Pose extends this paradigm to object-centric perception
by additionally learning keypoints, object-centric latent
embeddings,
and latent-conditioned NOCS, bridging
camera- and canonical-space reasoning for joint absolute
and relative pose estimation.
Relative Pose Estimation and Point Cloud Registration.
Two-view relative pose estimation is commonly solved
via feature matching and post optimization (e.g., essen-
tial matrix, PnP), or learned 2D/3D matching and registra-
tion [11, 15, 21, 25, 38, 47, 59]. For model-free methods,
(H)Oryon [6, 7] establish cross-view correspondences via
feature matching and point cloud registration. Any6D and
OnePoseViaGen [13, 24] instead generate object meshes us-
ing image-to-3D diffusion [36, 60, 62, 63] and align them via
render-and-compare. In contrast, we predict object-centric
point maps and leverage measured depth to directly estimate
SE(3) alignment with weighted Umeyama [50].
Metric Scale Recovery.
Monocular predictions are inher-
ently scale-ambiguous. Previous work leveraged category-
level size information and regressed offsets to real sizes
[9, 10, 17, 22, 58, 70]. In our setting, the optional sensor
depth acts as an external signal. OPT-Pose predicts absolute
translation/size from depth-derived points and derives the
scale by comparing against normalized predictions. Addi-
tionally, in RGB-only mode, a lightweight head estimates
per-frame log-scale.

Multiview
Transformer
Sensor Depth (optional)
RGB
Depth 
Map
Point 
Map
Feature 
Map
…
x 𝐿times
Camera 
Parameter
Object Latent 
Embedding
project
Cross-view 
Feature Fusion
Multi-view
Key-point 
Feature
Extraction
Camera Head
NOCS Head
Pose Head
Metric Head
DPT Head
Encoder
Figure 2. OPT-Pose overview. A multiview transformer aggregates image tokens and emits predictions from light heads: camera parameters,
depth, and point maps for camera-space geometry; a multi-view-keypoint-centric module fuses RGB and 3D features to discover object
keypoints, predict NOCS coordinates, and build an object latent embedding. Absolute pose (SA(3)) and relative pose (SE(3)) are recovered
in a single forward pass. Optional sensor depth provides metric scale, while the system remains fully functional in RGB-only mode.
Semantic Priors and Category-agnostic Canonicalization.
Vision and language foundation models [39, 43] provide
rich semantic features, but existing category-level meth-
ods [5, 28, 32, 67] still require category labels at inference.
In contrast, OPT-Pose projects object-centric latent embed-
dings, extracted from keypoint-level visual features [54] and
geometric representations from Sonata [61], into a shared
object-latent space. We train this space using a contrastive
objective across views, enabling category-agnostic canoni-
calization without requiring category names at inference.
Taxonomy of Unseen Pose Estimation and OPT-Pose.
Model-free object pose estimation falls into two paradigms:
(i) Category-level absolute pose estimation (e.g., GCE-Pose,
AG-Pose) [28, 32], which predicts SA(3) transforms in a
canonical space but relies on predefined taxonomies; and (ii)
Unseen-object relative pose estimation (e.g., OnePose++,
MegaPose, OSOP) [21, 47, 48], which estimates SE(3)
across views but cannot recover single-view absolute pose.
OPT-Pose unifies these paradigms in a model-free frame-
work via task factorization, jointly predicting canonical-
space absolute pose and camera-space relative pose through
complementary geometric mechanisms. This formulation
removes the need for CAD models, category labels, and test-
time optimization. In addition, OPT-Pose achieves category-
agnostic generalization through contrastive latent learning,
while remaining camera-agnostic and supporting flexible
RGB(-D) inputs across single and multi-view settings.