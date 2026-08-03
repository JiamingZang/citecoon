# Learning Cross-View Semantic Priors for Single-Reference Unseen Object Pose Estimation

> 2026 · id: W7165818136 · arXiv: 2606.22076 · pdf: https://arxiv.org/pdf/2606.22076 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
O
BJECT 6D pose estimation recovers the 3D rotation
and translation of an object from visual observations,
and is a fundamental capability for robotic manipulation [1–
4], embodied perception [5, 6], and augmented reality [7,
8]. Existing methods are commonly studied under instance-
level, category-level, and unseen-object settings. Instance-
level [9–14] and category-level methods often achieve strong
performance [15–19], but remain limited by supervision for
specific objects or predefined categories, making new object
Jiahong Chen, Jinghao Wang, Ziwen Wang, Zi Wang, Banglei Guan
and Qifeng Yu are with the College of Aerospace Science and Engi-
neering, National University of Defense Technology, Changsha 410073,
China, and also with the Hunan Provincial Key Laboratory of Im-
age
Measurement
and
Vision
Navigation,
Changsha
410073,
China
(e-mail: chenjiahong@nudt.edu.cn; wangjinghao16@nudt.edu.cn; wangzi-
wen24a@nudt.edu.cn; wangzi16@nudt.edu.cn; guanbanglei12@nudt.edu.cn;
yuqifeng@nudt.edu.cn). Corresponding authors: Zi Wang.
Query
VFM
Ref.
Query
Geometric Decoder
Cross-View 
Interaction
VFM
VFM
VFM
Point Cloud Matching
Point Cloud Matching
(a)
(b)
Noise 
Point
Parallel input
Current methods
Proj.
Proj.
Proj.
Proj.
Ours
Sparse point features
False correspondences
Geo embedding
Geo embedding
Image token
Ref.
Semantic prior
Geo embedding
Geo embedding
Reliable correspondences
Cross-view 
conditioned token
Independent intra-view descriptors
Dense cross-view semantic prior
Intra-view 
Geometric Decoder
Fig. 1: Comparison with existing correspondence-based methods.
(a) Existing methods [26, 28–30] mainly use VFM features as intra-
view descriptors for point cloud matching. (b) Our method performs
cross-view semantic interaction on dense VFM tokens, forming a
cross-view semantic prior for geometry-aware matching.
onboarding costly [20, 21]. These limitations motivate unseen
object pose estimation, which targets arbitrary novel objects
beyond known instances and predefined categories [22–27].
To further reduce object onboarding, recent studies have
moved toward the single-reference setting [28–38]. In this
setting, only one RGB-D observation of the target object is
available as the reference. A common solution is to follow a
correspondence-based formulation [26, 28–30], where the rela-
tive pose is estimated from correspondences between the query
and reference observations. The central challenge is to make
these correspondences reliable under large viewpoint changes
and noisy observations. Vision foundation models (VFMs) [39,
40] provide strong visual representations and have recently
been adopted in single-reference pose estimation. However,
existing methods typically use VFM features as intra-view
descriptors. As illustrated in Fig. 1(a), dense VFM feature
maps are extracted independently from the query and refer-
ence images, projected onto sampled foreground point clouds,
and then processed by a geometric decoder [41] for point
matching and pose recovery. This design has been effective for
geometry-aware matching, but it does not fully condition dense
visual-semantic cues across views before geometric decoding.
arXiv:2606.22076v2  [cs.CV]  25 Jun 2026

SUBMITTED TO IEEE TRANSACTIONS ON IMAGE PROCESSING
2
Fig. 2: A sample with large viewpoint changes. Although the
overlapping regions are very small, our method still achieves accurate
pose estimation. Blue and green contours denote GT and estimated
poses, respectively.
Appearance, part structure, and contextual relations encoded
by VFM tokens are therefore weakened when they are used
only as independent view-wise descriptors. As a result, the
decoded point features may lack joint semantic and geometric
discriminability: they can encode local 3D consistency, but
remain weak at distinguishing truly corresponding object re-
gions from geometrically plausible yet semantically unrelated
regions.
This limitation becomes evident in two typical cases. First,
under noisy segmentation masks, as shown in Fig. 1(a), the
sampled foreground point cloud may contain nearby objects
whose local geometry resembles the target. If the decoder
mainly relies on geometry at this stage, these distractor
points can receive high matching salience and lead to false
correspondences. Second, under large viewpoint changes, the
visible overlap between the query and reference observations
becomes sparse and spatially ambiguous, as shown in Fig. 2.
In both cases, reliable matching requires a cross-view semantic
prior before geometric decoding, so that point-level correspon-
dence learning can be guided by reference-conditioned visual
semantics rather than relying only on sparse geometric cues.
Motivated by this observation, we build the correspon-
dence pipeline around an early cross-view semantic prior.
As shown in Fig. 1(b), our method introduces cross-view
semantic interaction (CVSI) at the dense VFM token level
before projection and geometric decoding. Instead of treating
query and reference VFM features as independent intra-view
descriptors, CVSI allows dense tokens from the two views
to exchange semantic context and condition each other. The
resulting tokens form a visual semantic prior conditioned cross
views. When fused with geometry-aware embeddings in the
decoder, this prior encourages the decoded point features to
be both semantically selective and geometrically consistent for
correspondence estimation. However, learning such a prior is
non-trivial because the query and reference observations are
unposed, partially overlapping, and may contain mask-induced
distractors. Direct dense token interaction can therefore cause
excessive feature mixing and over-smooth the spatial organiza-
tion encoded by the VFM. In addition, the learned prior should
be translated into spatially consistent point features to benefit
rigid 3D correspondence estimation. To address these issues,
we introduce two complementary training-time constraints
that jointly make the CVSI prior reliable for 3D correspon-
dence learning. The intra-view structure preservation (IVSP)
loss preserves the original intra-view token affinity structure
during cross-view interaction, avoiding over-smoothed spatial
organization. The reference-anchored geometric consistency
(RAGC) loss enforces spatial representation consistency of
decoded point features in a shared reference frame, mak-
ing the semantic prior compatible with rigid correspondence
estimation. The final pose is recovered from learned corre-
spondences through weighted SVD. Together, CVSI, IVSP,
and RAGC are designed to improve correspondence reliability
under large viewpoint changes and noisy masks. To evaluate
this capability, we further construct a challenging view pair
protocol from the BOP Challenge datasets YCB-V [9] and
TUD-L [42]. Extensive experiments on LM-O [43], TUD-
L [42], YCB-V [9], REAL275 [15], Toyota-Light [42], and
LINEMOD [44] demonstrate that our method achieves state-
of-the-art performance across different view pair settings while
maintaining comparable inference speed.
The main contributions of this work are as follows:
• We identify the absence of an explicit cross-view seman-
tic prior as a key limitation of existing correspondence-
based methods, where VFM features are mainly used as
intra-view descriptors before geometric decoding.
• We introduce cross-view semantic interaction (CVSI) at
the dense VFM token level, forming an early cross-
view semantic prior for subsequent geometry-aware point
cloud matching.
• We introduce two complementary training-time con-
straints to make the CVSI prior reliable for 3D correspon-
dence learning. IVSP preserves the intra-view token affin-
ity structure during cross-view interaction, while RAGC
enforces spatial representation consistency of decoded
point features in a shared reference frame.
• We construct a challenging view-pair protocol from the
BOP Challenge datasets YCB-V and TUD-L, focusing on
robustness to noisy masks and large viewpoint changes
in single-reference unseen object pose estimation.

## method
ARBOP (%)
TUD-L [42] ↑
YCB-V [9] ↑
Mean ↑
UNOPose [28]
55.9
58.5
57.2
CoordAR [37]
36.7
50.2
43.5
SinRef-6D [30]
24.5
49.0
36.8
Ours
67.0
64.2
65.6
C. Qualitative Results Analysis
We present extensive qualitative results, including pose
estimation visual comparisons on each dataset, results under
the challenging view pair protocol, and visualizations of
cross-view interaction, to better analyze and demonstrate the
superiority of our method.
Qualitative Results on Six Datasets. Fig. 5 shows qualita-
tive results on all six evaluated datasets, with object-level visu-
alizations for each tested object. These examples cover diverse
challenges, including heavy occlusion, illumination variation,
textureless objects, and geometric ambiguity. Compared with
UNOPose [28] and SinRef-6D [30], our method produces
more accurate pose alignment and lower object-level depth
errors across different datasets.
Reference
SinRef-6D
Query
UNOPose
Ours
(a) Large viewpoint changes
(b) Cluttered observations
Fig. 6: Qualitative comparison under the challenging view pair
protocol. (a) Large viewpoint changes. (b) Cluttered observations
with heavy mask noise.
Qualitative Results under the Challenging View Pair
Protocol. Fig. 6 shows qualitative results under the challenging
view-pair protocol, including large viewpoint changes and
cluttered observations with heavy mask noise. Under large
viewpoint gaps, our projected contours remain better aligned
with the ground truth than UNOPose [28] and SinRef-6D [30],
even when the visible overlap is limited. Under heavy mask
noise, nearby objects with similar geometry and appearance
can introduce highly ambiguous correspondences. Our method
remains more stable in these cases because the cross-view
Query token
Structure cue
Contextual cue
Appearance cue
Contextual cue
Reference attention map
driller
cat
clamp
Fig. 7: Visualization of the attention maps in cross-view interac-
tion. We select one token in the query view (red box) and visualize
the responses of different attention heads.
semantic prior provides appearance, structure, and spatial
guidance before geometry-aware matching.
Visualization of Cross-View Interaction. Fig. 7 visualizes
the attention responses from a selected query token to refer-
ence tokens in different CVSI heads. Different heads attend to
complementary reference regions, such as local parts, object
boundaries, and broader body structures. For example, in the
driller case, the selected query token activates the handle
area, the elongated body, and their junction. Similar patterns
appear in the cat and clamp examples, suggesting that CVSI
aggregates appearance, structural, and contextual cues across
views rather than simply averaging features. These cross-view
responses provide a semantic prior for point-level matching
under viewpoint changes and local ambiguities.
D. Ablation Studies
We conduct ablation studies to verify the contribution of the
proposed cross-view semantic prior and its training constraints.
We further examine key architectural choices and robustness
under different reference viewpoint gaps. Unless otherwise
specified, all ablations are performed on YCB-V [9].
Impact of Backbone Networks. We first study the influ-
ence of the VFM backbone to separate backbone improve-
ments from the contribution of the proposed components. As
shown in Table VI, replacing DINOv2 [39] with DINOv3 [40]
improves the baseline ARBOP from 83.1% to 84.2% (A0
vs. B0). This indicates that a stronger VFM backbone pro-
vides better visual representations for correspondence learning.
However, after introducing the proposed components, the
gap between the two backbones becomes much smaller: the
DINOv2 variant reaches 86.0% (A1), while the corresponding
DINOv3 variant reaches 86.2% (B3). This suggests that the
performance gain is not mainly explained by the backbone
replacement. Since DINOv3 also uses a larger patch size than
DINOv2 and is computationally lighter in our setting, we use
DINOv3 as the default backbone in the remaining experiments.
Effectiveness and Complementarity of CVSI, RAGC,
and IVSP. Table VI analyzes the contribution of the proposed
components on YCB-V [9]. This ablation is designed to

SUBMITTED TO IEEE TRANSACTIONS ON IMAGE PROCESSING
10
Table VI: Ablation study of the backbone and proposed components
on the YCB-V [9] dataset. Results are reported with DINOv2 [39]
and DINOv3 [40] backbones. (*) denotes the variant with shared
cross-attention weights between the query and reference branches.
Row
Backbone
CVSI
RAGC
IVSP
VSD
MSSD
MSPD
ARBOP
A0
DINOv2
✗
✗
✗
82.6
87.4
79.4
83.1
A1
✓
✓
✓
83.0
90.4
84.5
86.0
B0
DINOv3
✗
✗
✗
82.4
88.4
81.7
84.2
B1
✓
✗
✗
82.9
90.1
84.0
85.7
B2
✗
✓
✗
83.0
89.0
82.2
84.7
B3
✓
✓
✓
83.5
90.7
84.5
86.2
B4
✓
✓
✓
83.9*
91.1*
85.1*
86.7*
B5
✓
✓
✗
82.9
90.2
84.1
85.7
separate the effects of early cross-view semantic interaction,
reference-anchored geometric supervision, and token structure
preservation.
Under the DINOv3 backbone, adding only CVSI improves
ARBOP from 84.2% to 85.7% (B1 vs. B0), giving a gain
of 1.5%. The improvement is also consistent on MSSD and
MSPD, which increase from 88.4% to 90.1% and from 81.7%
to 84.0%, respectively. This result shows that performing
cross-view interaction before geometric decoding provides
more discriminative features for correspondence estimation.
In particular, the larger gains on MSSD and MSPD suggest
that CVSI improves both 3D surface alignment and projec-
tion accuracy, which are directly affected by correspondence
quality.
The RAGC-only variant improves ARBOP from 84.2%
to 84.7% (B2 vs. B0). This moderate gain indicates that
reference-anchored geometric supervision provides a useful
geometric regularization signal, but it is not the main source
of the overall improvement. This comparison is important
because it controls for the effect of the auxiliary coordinate
prediction head: adding the RAGC head and loss alone cannot
explain the full gain of the proposed method. Instead, RAGC
is most useful when it works with cross-view semantic condi-
tioning, where it encourages the decoded point features to be
consistent in a shared 3D reference frame.
The role of IVSP is also clarified by comparing B1, B3,
and B5. When CVSI and RAGC are used without IVSP,
the ARBOP remains 85.7% (B5), the same as CVSI alone
(B1). After adding IVSP, the performance increases to 86.2%
(B3), with consistent gains on VSD, MSSD, and MSPD. This
shows that IVSP is not merely an additional auxiliary loss,
but a stabilizing regularizer for cross-view token interaction.
By preserving the intra-view token affinity structure, IVSP
prevents the interacted tokens from losing the original VFM
organization, allowing the semantic prior introduced by CVSI
and the geometric constraint imposed by RAGC to better
complement each other.
Finally, sharing the cross-attention weights between the
query and reference branches further improves ARBOP from
86.2% to 86.7% (B4 vs. B3). This shared design encourages
the two branches to follow a consistent interaction pattern
and reduces redundant parameters. Compared with the DI-
NOv3 baseline, the full model improves ARBOP by 2.5%,
from 84.2% to 86.7%. The improvements are also observed
Query
Reference
Query
Reference
Query
Reference
Only Dinov3
Ours
Fig. 8: Visualization of correspondence estimation. We compare
the predicted correspondences obtained using only the DINOv3 [40]
backbone and our full method. Green and red lines denote geomet-
rically consistent and inconsistent correspondences, respectively.
across all three BOP metrics, with gains of 1.5, 2.7, and
3.4% on VSD, MSSD, and MSPD, respectively. These results
support the complementary design of the proposed framework:
CVSI provides the main cross-view semantic prior, RAGC
grounds the decoded features in 3D reference coordinates, and
IVSP stabilizes the token structure during interaction. Since
RAGC and IVSP are used only during training, they improve
feature learning without adding inference time computation.
We therefore adopt the shared CVSI design with both RAGC
and IVSP as the default configuration.
Visualization of Correspondences. Fig. 8 visualizes the
correspondences produced by the DINOv3-only baseline and
our full model. The DINOv3-only baseline produces scattered
matches, often connecting non-corresponding or locally simi-
lar object parts under partial overlap and noisy observations.
In contrast, our method yields more spatially coherent corre-
spondences concentrated on geometrically compatible regions.
This indicates that the cross-view semantic prior improves
point-feature discriminability before matching and provides
cleaner correspondences for subsequent weighted SVD pose
estimation.
Intra-View Similarity Structure Analysis. Fig. 9 com-
pares the intra-view token similarity matrices before and after
cross-view interaction. Raw DINO features show clear local
affinity structures related to object parts, boundaries, and
spatial layouts. Without IVSP, cross-vi

## experiments
A. Experimental Setup
Implementation Details. Our method is implemented in
PyTorch [63]. For image encoding, we use a DINOv3 [40]
pretrained ViT-Base backbone [64]. Following [28, 29], the
network is trained on the standard MegaPose synthetic
dataset [23] for the BOP unseen object pose estimation
track [65]. We train the network for 440K steps with a batch
size of 8 on the RTX 4090 GPU. Optimization is performed
with Adam [66] and cosine annealing [67], using a base
learning rate of 10−4. Unless otherwise specified, the number
of CVSI blocks is set to L = 3, the number of attention heads
is set to H = 8, and the loss weights are fixed to λRAGC = 1
and λIVSP = 1. In the coarse stage, we sample Nc = 196
points and generate 300 pose hypotheses. In the fine stage,
we increase the number of sampled points to Nf = 2048.
Datasets. We conduct experiments on six benchmark
datasets: LM-O [43], TUD-L [42], YCB-V [9], Real275 [15],
Toyota-Light [42], and LINEMOD [44]. LM-O, TUD-L, and
YCB-V follow the predefined view-pair protocol of [28],
where SAM masks [46] introduce realistic segmentation noise
and the scenes cover clutter, occlusion, lighting variation,
and sensor noise. Real275 and Toyota-Light are evaluated
under the Oryon protocol [33], with 2,000 reference-query
pairs and ground-truth masks for each test set, covering
diverse indoor objects and challenging illumination conditions.
To further assess sensitivity to the reference view, we also
evaluate the first-frame single-reference setting on YCB-V [9]
and LINEMOD [44], following [25, 35], where LINEMOD
provides large viewpoint changes across object sequences.
Challenging Benchmark. We further construct a challeng-
ing view-pair benchmark based on YCB-V [9] and TUD-
L [42]. For each query object in the test split, we randomly
select a reference view from the training or validation split and
constrain the relative rotation angle to the range of 60 to 90
degrees. For the object masks, we start from the segmentation
results of UNOPose [28] and further inject noise by applying
random morphological dilation.
Evaluation Metrics. For LM-O, TUD-L, and YCB-V un-
der the predefined view-pair protocol, we follow the BOP
evaluation protocol and report ARBOP, averaged over VSD,
MSSD, and MSPD [42]. For Real275, Toyota-Light, and
LINEMOD, we report ADD(-S) at 0.1d with ADD-S for
symmetric objects [44, 68]. For YCB-V under the first-
frame reference setting, we report AUC of ADD and ADD-S
following PoseCNN [9].
B. Comparison with State-of-the-Art Methods
Since the checkpoint of the best-performing (supervised
version) COG model [29] is not publicly available, we report
its results only in Table I using the numbers from the original
paper.
Comparison under the Predefined View Pair Protocol.
Table I compares different methods under the predefined view-
pair protocol of UNOPose [28]. Pure point cloud registration
methods are limited in this setting, with the best one reaching
only 56.9% mean ARBOP. This indicates that geometry alone
is insufficient for mask-cropped partial point clouds with dis-
tractors and limited overlap. RGB-D single-reference methods
perform better by combining visual and geometric information,
with the strongest baselines UNOPose [28] and COG [29]
reaching 70.9% and 73.8% mean ARBOP, respectively. Our
method achieves the best performance on all three datasets,
reaching 76.6% mean ARBOP and improving over UNOPose
and COG by 5.7% and 2.8%, respectively. Compared with
the strongest baseline on each dataset, our gains are 0.4% on
LM-O, 2.0% on TUD-L, and 3.6% on YCB-V.
Runtime Analysis. Due to differences in hardware from the
original papers, we re-evaluate the inference time of SinRef-
6D [30], UNOPose [28], COG [29], and CoordAR [37] on a

SUBMITTED TO IEEE TRANSACTIONS ON IMAGE PROCESSING
7
Table I: Pose estimation results on LM-O, TUD-L, and YCB-V under the view-pair setting of UNOPose [28]. Object masks are obtained
by SAM [46]. The mean Average Recall (AR) of the BOP metric and the average time (s) per image are reported. The runtime includes all
instances from the SAM proposals.

## related_work
A. Novel Object Pose Estimation
Novel object pose estimation aims to recover the 6D pose of
arbitrary objects beyond known instances and predefined cate-
gories. One line of work addresses this setting by using object
models, rendered templates, or generalizable matching at test
time. MegaPose [23] follows a render-and-compare paradigm,
where CAD-based renderings are retrieved and refined for
pose estimation. GigaPose [45] improves CAD-based novel
object pose estimation with discriminative rendered templates
and a compact correspondence formulation. Recent methods
further exploit vision foundation models [39, 40] and general
segmentation priors [46]. SAM-6D [26] combines SAM-based
object proposal generation with semantic, appearance, and
geometric matching, and formulates pose estimation as partial-
to-partial point matching.
When CAD models are unavailable, another direction rep-
resents the target object using multiple reference observa-
tions [25, 47–50]. OnePose [47] reconstructs a sparse object
model from a video scan using SfM, matches 2D query points
to 3D SfM points, and estimates the object pose with PnP.

SUBMITTED TO IEEE TRANSACTIONS ON IMAGE PROCESSING
3
OnePose++ [48] further removes the dependence on repeat-
able keypoint detection by using keypoint-free matching and
reconstructing a semi-dense object point cloud. Gen6D [49]
assumes several posed reference images and estimates pose
through detection, viewpoint selection, and refinement. Foun-
dationPose [25] unifies model-based and model-free pose
estimation by using either a CAD model or a small set of
reference images. These methods improve generalization to
novel objects, but they still typically rely on either CAD
models or multiple reference views.
Recent works therefore move toward the more constrained
single-reference setting [28–31, 33–38]. NOPE [31] predicts
pose-conditioned discriminative embeddings from a single
reference image and estimates the query pose by match-
ing them to generated viewpoint embeddings. Oryon [33]
and Horyon [34] study open-vocabulary relative pose es-
timation, where a text prompt identifies the target object
across two scenes and visual-language features are used for
cross-scene matching and 3D registration. One2Any [35] and
CoordAR [37] predict reference object coordinates from the
query observation, converting single-reference pose estimation
into alignment in a shared reference coordinate space. UN-
OPose [28] casts single-reference pose estimation as point
cloud registration by combining VFM semantic descriptors
with an SE(3)-invariant geometric representation for matching.
COG [29] extends this correspondence-based formulation with
confidence-aware optimal transport to obtain soft correspon-
dences guided by point confidence. Unlike UNOPose [28]
and COG [29], which mainly improve geometric decoding or
correspondence assignment, we focus on feature construction
before matching. Our method performs cross-view semantic
interaction at the dense VFM token level, forming a cross-view
semantic prior for geometry-aware correspondence learning.
B. Point Cloud Registration
Point cloud registration is closely related to single-reference
unseen object pose estimation, as both require reliable corre-
spondences between partial 3D observations. Classical regis-
tration methods mainly rely on geometric matching. ICP and
its variants [51, 52] iteratively refine closest point associa-
tions, FPFH [53] describes local surface neighborhoods, and
PPF [54] encodes oriented point pair relations for pose hy-
pothesis generation. These methods provide useful geometric
baselines, but are sensitive to outliers, repeated local structures,
and limited overlap. Learning methods improve registration by
estimating more discriminative correspondence cues. Preda-
tor [55] predicts overlap-aware features to focus matching
on shared regions under low overlap. GeoTransformer [41]
encodes pairwise distances and angular relations to obtain
geometric features that are invariant to rigid transformations
for robust superpoint matching.
Unlike generic registration, single-reference unseen object
pose estimation operates on mask-cropped partial RGB-D
point clouds, where mask noise and large viewpoint changes
introduce distractors and sparse overlap. In such cases, ge-
ometry alone is often insufficient, and VFM cues [39, 40]
can provide complementary appearance, layout, and contextual
information. This motivates a cross-view semantic prior before
geometry-aware matching.
C. Cross-View Correspondence Learning
Cross-view correspondence learning has been studied in
image matching, object association, and visual geometry rea-
soning. LoFTR [56] shows that transformer feature interaction
can establish dense local correspondences without explicit key-
point detection. O-MaMa [57] formulates ego-exo object cor-
respondence as mask matching, using DINOv2 features [39]
and cross-attention for object level alignment. V2-SAM [58]
adapts SAM2 to cross-view object correspondence by com-
bining geometry-aware anchor prompts from DINOv3 [40]
with visual prompts for appearance alignment. VGGT [59]
further demonstrates that cross-view spatial consistency can
be encoded through unified visual geometry representations.
These studies show the value of cross-view interaction
for transferring appearance, structure, and spatial cues, but
mainly target image-level correspondence, mask association,
or general visual geometry reasoning. We study cross-view
cues for single-reference unseen object pose estimation, where
they must be grounded by geometric consistency for RGB-D
correspondence learning and rigid pose recovery.

## conclusion
This paper studied single-reference unseen object 6D pose
estimation and identified the insufficient cross-view exchange
of dense visual-semantic cues as a key limitation of existing
correspondence-based pipelines. To address this issue, we
proposed a cross-view semantic prior learning framework that
introduces dense token-level CVSI before geometric decoding,
preserves intra-view token structure with IVSP, and grounds
decoded features through RAGC. Together, these components
improve the semantic and geometric discriminability of cor-
respondence features for more reliable point cloud matching
and weighted SVD pose estimation. Extensive experiments on
six benchmark datasets and a challenging view-pair protocol
demonstrate state-of-the-art performance with comparable in-
ference speed.
Limitations and Future Work. Despite these improve-
ments, the proposed method remains limited by the input
evidence available in the single-reference setting. When the
query and reference views have near-zero overlap or the masks
remove discriminative object regions, reliable correspondences
remain difficult to establish. Future work will extend the
framework to multiple query observations and explicit multi-
view geometric reasoning, which may provide richer object
evidence and alleviate these failure cases.