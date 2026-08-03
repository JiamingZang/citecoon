# TSR-Ego: Temporally Guided Stereo Refinement Framework for Egocentric 3D Human Pose Estimation

> 2026 · id: W7168178757 · arXiv: 2607.09169 · pdf: https://arxiv.org/pdf/2607.09169 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Egocentric 3D human pose estimation from head-mounted stereo cameras is chal-
lenging due to fisheye distortion, severe self-occlusion, and frequent truncation of body
joints outside the camera field of view. Recent stereo egocentric methods have improved
performance through heatmap lifting, stereo correspondence, and transformer-based re-
finement, but they often rely heavily on frame-local evidence or use temporal information
only as auxiliary pose-level context. This limits robustness when current-frame stereo
cues are weak, occluded, or ambiguous. We propose TSR-Ego, a temporally guided
stereo framework that couples short-term motion evidence with projection-guided feature
sampling. The model first enriches dense stereo feature maps using a causal depthwise-
separable temporal convolution, allowing past visual evidence to influence the feature
space before deformable cross-attention. A single-stage causal stereo decoder then re-
fines learned 3D joint queries through temporal self-attention, joint self-attention, and
fisheye deformable stereo cross-attention, using the evolving pose estimate to generate
2D sampling references. Unlike methods that apply temporal reasoning mainly after pose
prediction, TSR-Ego uses motion context to shape both the sampled stereo features and
the joint representations while preserving online inference without future frames. Ex-
periments on UnrealEgo2 and UnrealEgo-RW show state-of-the-art performance, with
especially strong gains on real-world sequences where single-frame stereo observations
are unreliable. Code will be available at: https://github.com/.
1

## introduction
Egocentric 3D human pose estimation [1, 3, 10, 12, 13, 24, 29, 30, 34] from head-mounted
cameras is a key component for immersive AR/VR, telepresence, human-computer interac-
tion, and embodied motion understanding. Unlike third-person pose estimation [4, 8, 25], the
egocentric setting observes the body from an extreme first-person viewpoint, where fisheye
distortion, large perspective variation, self-occlusion, and limited field of view make many
joints only partially visible or completely out of view. These challenges are particularly
© 2026. The copyright of this document resides with its authors.
It may be distributed unchanged freely in print or electronic forms.

2
AZAM, QUARLES, DESAI: TSR-EGO
severe for lower-body joints, which are often weakly observed by head-mounted cameras
and must be inferred from incomplete visual evidence. Stereo fisheye cameras provide an
attractive sensing setup because they introduce cross-view geometric cues while remaining
compatible with compact wearable devices. However, effectively exploiting these cues re-
mains difficult when body parts are truncated, stereo evidence is weak or one-sided, and
visible joint evidence changes rapidly over time.
Recent datasets and methods have substantially advanced stereo egocentric 3D pose esti-
mation. Large-scale benchmarks such as UnrealEgo, UnrealEgo2, and UnrealEgo-RW pro-
vide synthetic and real-world head-mounted fisheye stereo data for studying full-body pose
estimation under severe occlusion and limited field of view [1, 2]. Building on these bench-
marks, early approaches often rely on intermediate 2D heatmaps and lift them to 3D, which
provides strong image-space localization but remains underconstrained when joints are oc-
cluded or outside the camera view. Recent stereo egocentric methods have improved 3D pose
estimation by lifting image-space heatmaps into 3D poses, exploiting stereo correspondence
and egocentric geometric cues, and refining joint representations with transformer-based at-
tention [2, 11, 12, 31]. Despite this progress, existing stereo egocentric methods still leave
an important gap between spatial stereo reasoning and temporal guidance. Heatmap-based
methods depend on intermediate 2D detections, geometry-aware methods rely on reliable
current-frame stereo cues, and transformer-based refinement methods often build their 3D
hypotheses primarily from the current stereo observation. Video-based approaches intro-
duce temporal context, but temporal information is commonly used to augment joint repre-
sentations or improve pose consistency rather than to condition the stereo features sampled
during refinement. This separation is limiting in egocentric views, where joints may be oc-
cluded, truncated by the headset field of view, or visible in only one fisheye camera. In such
cases, current-frame spatial evidence alone may produce an unreliable pose hypothesis, even
though recent frames provide useful cues about joint motion and visibility.
We address this limitation with TSR-Ego, a temporally guided stereo refinement frame-
work for egocentric 3D human pose estimation. Our key idea is to inject temporal evidence at
the feature level, so that deformable stereo cross-attention samples from temporally enriched
representations rather than only frame-local features. Given a causal window of stereo fish-
eye frames, TSR-Ego first applies a causal depthwise-separable convolution stack along the
time axis to enrich each per-view spatial feature map with past visual evidence while preserv-
ing the spatial layout required for deformable sampling. A single-stage causal stereo decoder
then refines learned 3D joint queries through interleaved causal temporal self-attention, joint
self-attention, and fisheye deformable stereo cross-attention. The temporal self-attention up-
dates each joint query using its own motion history, enabling online inference without future
frames, while stereo sampling references are projected from the decoder’s evolving pose
estimate.
Unlike approaches that treat temporal reasoning as a post-processing or pose-level refine-
ment step, TSR-Ego lets motion context condition both the stereo feature maps and the joint
queries inside the decoder. This is useful in egocentric stereo settings where joints may be
occluded, truncated, or visible in only one fisheye view, but remain predictable from recent
motion. Our contributions are summarized as follows:
• A Causal Temporal Feature Mixer (TFM), a causal depthwise-separable convolu-
tion over the time axis of stereo feature maps that enriches each spatial location with
past visual evidence before deformable stereo cross-attention, conditioning the feature
space itself rather than only the queries.

AZAM, QUARLES, DESAI: TSR-EGO
3
• A single-stage causal stereo decoder that integrates pixel-level temporal feature enrich-
ment with joint-local temporal reasoning for egocentric 3D human pose estimation.
We achieve state-of-the-art results on UnrealEgo2 and UnrealEgo-RW, validating the
benefit of embedding temporal evidence inside stereo feature refinement rather than applying
it only after pose prediction.
2

## method
Given a causal window of synchronized stereo observations It = {Iv
τ}t
τ=t−T+1, v ∈{1,...,V},
our model predicts a 3D pose sequence ˆP ∈RT×J×3.
TSR-Ego uses a single-stage causal stereo decoder initialized with learned per-joint
queries and learned 3D reference points, avoiding an explicit proposal stage. The decoder
refines these queries through temporal reasoning, body-structure reasoning, and geometry-
aware stereo cross-attention. Temporal evidence is introduced both at the pixel-feature level
before stereo sampling and at the query level after joint tokens are formed.
The architecture consists of three stages as depicted in Figure 1. First, a stereo visual
encoder extracts dense feature maps for each view and frame. Second, a causal temporal
feature mixer enriches each spatial feature location using only past frames. Third, a causal
stereo decoder iteratively projects the current 3D pose estimate into each fisheye view, sam-
ples image evidence through deformable cross-attention, and predicts an absolute 3D pose
at every decoder layer.

AZAM, QUARLES, DESAI: TSR-EGO
5
1×1 Feature 
Projection
Stereo Visual Encoder
 (per frame)
Resnet-18 Backbone
FPN Neck
Causal Temporal Feature Mixer 
(TFM)
 Per-view, per-location sequence
GELU
Depthwise Conv1d
( k=3, d = dil)
Layer Norm
Pointwise Conv1d
(1×1)
Identity at step 0 
(causal start)
dil = 2
dil = 1
FFN
F′
+
+ residual
Decoder Inputs
Q0𝜖ℝJ×C
(learned query tokens)
P0𝜖ℝJ×3
(layer 0 init)
age_embed
(learned)
Input: Stereo Window
Left            Right
t-7
t-1
t
Pose Output
P-7
P-1
P
Causal Stereo Decoder
(single layer shown)
Causal Time Self-Attn
(joint-local over T frames)
Joint Self-Attn
(per-frame over J joints)
Stereo Deformable 
Cross-Attn
Absolute Pose Head
Fisheye Proj. 
→2D Refs.
Prunning 
(init = P0)
stop_grad
P𝜄 𝜖 ℝJ×3 
(per-layer 3D pose)
× 2  
layers
F′
+
Figure 1: Overview of TSR-Ego. A causal stereo window is encoded by a shared visual
backbone, temporally enriched by the proposed TFM, and decoded by a single-stage causal
stereo decoder. The decoder refines learned 3D joint queries using temporal self-attention,
joint self-attention, and projection-guided deformable stereo cross-attention. Each layer pre-
dicts an absolute device-relative 3D pose, whose detached projection is used as the sampling
reference for the next layer.
3.1
Stereo Visual Feature Encoding
Each input image is encoded independently by a ResNet-18 backbone [9] followed by a
feature-pyramid neck [16]. The backbone provides features at strides {4,8,16,32}, which
are fused by the neck into a single stride-4 feature map. For an input resolution of 256×256,
this produces a 64 × 64 feature map. A 1 × 1 convolution projects the neck output to the
decoder hidden dimension C = 128:
Fv
τ = φ(Iv
τ) ∈RC×H×W
Stacking all frames and views gives
F ∈RB×T×V×C×H×W
We use a dense feature map rather than a global image descriptor because egocentric
pose estimation depends strongly on local geometric evidence. Hands, feet, and limbs often
appear near the image boundary under severe fisheye distortion, and their visibility varies
between the two cameras. Preserving a stride-4 spatial grid allows later deformable attention
to sample local features around the physically projected joint location instead of forcing all
visual evidence through a global bottleneck.
The encoder is initialized from heatmap pretraining. This gives the backbone an explicit
localization prior before the 3D decoder is trained. We then fine-tune the encoder jointly
with the full model, allowing the features to adapt from 2D heatmap localization to stereo-
temporal 3D pose estimation.
3.2
Causal Stereo-Temporal Feature Mixing
A central design choice is to inject temporal information into the dense feature maps before
decoding. Given F ∈RB×T×V×C×H×W, we treat each fixed sample, view, and spatial location

6
AZAM, QUARLES, DESAI: TSR-EGO
(b,v,y,x) as a temporal sequence:
Fb,:,v,:,y,x ∈RT×C
We apply a stack of causal depthwise-separable 1D convolutional blocks along the time axis.
For the m-th block,
F(m+1) = F(m) +PWConv

GELU
 DWConv←
k,d(LNC(F(m)))

,
where F(0) = F, ˜F = F(M), DWConv←
k,d denotes left-padded causal depthwise convolution
over time, PWConv is a pointwise channel-mixing convolution, and LNC applies Layer-
Norm over channels at each time step. This module is intentionally temporal-only. It never
mixes neighboring spatial locations. Therefore, the output feature at coordinate (y,x) re-
mains aligned with the same image coordinate. This is important because the following
deformable stereo attention uses projected 3D joints as geometric reference points. If tem-
poral modeling were to blur or pool spatial coordinates, the decoder would sample from
feature maps whose spatial positions no longer correspond cleanly to image locations.
The temporal mixer serves several purposes. It allows visual evidence from previous
frames to support the current frame before the model commits to joint-level queries. This
is useful under self-occlusion, motion blur, extreme fisheye distortion, and view-dependent
limb visibility. In addition, pixel-level temporal mixing avoids an early query bottleneck.
At the beginning of decoding, the joint queries are learned parameters and have not yet
interacted with the current image evidence. If temporal reasoning were performed only on
these queries, the model would be propagating relatively abstract tokens. By mixing dense
features first, each future cross-attention operation samples from a feature map that already
carries causal temporal context.
Causality is enforced by left padding only. Thus, the representation at frame τ depends on
frames ≤τ and never on future frames. This makes the model suitable for online inference.
We also avoid BatchNorm or GroupNorm in this module because normalization statistics
over the temporal dimension could leak future information. Per-frame channel LayerNorm
preserves causality.
In our default setting, we use two temporal convolution blocks with kernel size 3 and
dilations 1 and 2, giving a receptive field of seven frames. The pointwise convolution in
each block is zero-initialized, so the entire temporal mixer starts as an identity mapping.
This is important for stable optimization: the model initially behaves like a strong per-frame
heatmap-pretrained encoder, and temporal corrections are learned gradually.
3.3
Learned Joint Queries and 3D Reference Initialization
The decoder operates on one token per joint per frame. We initialize the query content with a
learned joint embedding Q0 ∈RJ×C (truncated-normal, σ=0.02) and pair it with a learnable
initial 3D reference pose
P0 ∈RJ×3,
which is zero-initialized. Both tensors are broadcast over the batch and temporal dimensions:
the same learned prior seeds every frame, and there is no separately predicted per-frame
initial pose. Q0 and P0 are optimized jointly with the rest of the model.
This initialization provides a compact joint-specific prior while letting the pose estimate
be shaped by visual evidence through the decoder. At each decoder layer, the current 3D

AZAM, QUARLES, DESAI: TSR-EGO
7
reference is projected into each fisheye view to obtain 2D sampling locations for deformable
stereo cross-attention; the sampled image features then update the joint tokens, and a per-
layer head regresses an absolute 3D pose that becomes the reference for the next layer. The
sampling locations are therefore not fixed but evolve with the decoder’s own pose prediction,
yielding a geometry-aware refinement process.
The zero initialization of P0 is deliberate. Combined with the zero-initialized final linear
of every pose head, it guarantees that at the first training step every layer predicts exactly P0,
so training begins from a clean identity-refinement state and the decoder learns corrections
on top of the learned prior rather than fighting an arbitrary starting pose.
3.4
Causal Stereo Decoder
The decoder contains L repeated layers. Each layer performs four operations: causal tem-
poral self-attention, joint self-attention, fisheye stereo deformable cross-attention, and feed-
forward refinement. After each layer, a pose head predicts a full absolute 3D pose.
Causal temporal self-attention.
For each joint independently, we apply self-attention over
the T query tokens corresponding to that joint across time. A strict upper-triangular causal
mask ensures that frame τ attends only to frames ≤τ:
Qτ,j ←TimeAttn(Q≤τ,j).
We add a learned age embedding before attention, where the current frame has age 0, the
previous frame has age 1, and so on.
This component models joint-specific motion history. It is especially useful for tempo-
rally ambiguous observations: a partially visible hand or foot in the current frame can be
disambiguated using its recent trajectory. The age embedding tells the model how old each
token is relative to the target frame, which is more appropriate than absolute frame indexing
for sliding-window inference. The causal mask preserve

## experiments
4.1
Datasets
We evaluate on two egocentric stereo fisheye datasets that differ in domain, scale, and camera
calibration.
UnrealEgo2 [2].
UnrealEgo2 is a large-scale synthetic dataset rendered in Unreal Engine
from motion-capture sequences. Each sample contains synchronized left/right head-mounted
fisheye images at 1024×1024 resolution, together with accurate 3D pose annotations. Al-
though the raw annotations contain a much larger skeleton, we follow our model setting and
use a 16-joint limb subset. The pelvis is used only as the root for pose normalization and
as the per-view projection origin; it is not itself a predicted joint. Ground-truth poses are
expressed in the head-mounted device frame and centered at the pelvis, i.e. device-relative
coordinates with the pelvis as origin (Pj = jointdev
j
−pelvisdev). The synthetic stereo cameras
share the Scaramuzza fisheye calibration used by the UnrealEgo rendering setup. We follow
the official train/validation/test split.
UnrealEgo-RW [2].
UnrealEgo-RW is the real-world counterpart, captured with two phys-
ically separate head-mounted fisheye cameras at 872×872 resolution. Unlike the synthetic
setting, the two cameras have distinct per-view Scaramuzza calibrations. The dataset pro-
vides 16 limb joints and no pelvis; we therefore define the root as the mid-hip, the mid-
point of LeftUpLeg and RightUpLeg. Ground-truth poses are expressed in the device
frame, relative to this mid-hip root, and all 16 joints are supervised. We use the official
train/validation/test split of 402, 96, and 93 sequences.
4.2
Training Details
We use a ResNet-18 backbone with an FPN neck as the visual encoder. The encoder is
initialized from a dataset-specific heatmap-regression checkpoint and then optimized jointly
with the 3D decoder. The model takes a causal stereo window of T=8 frames as input. We
sample training windows with stride 4 and validation/test windows with stride 8.
All models are trained for 12 epochs on a single GPU using AdamW [17]. The initial
learning rate is 10−3, weight decay is 5×10−4, and the learning rate is warmed up linearly
for the first 200 iterations. We decay the learning rate by a factor of 0.1 at epochs 8 and 10.

10
AZAM, QUARLES, DESAI: TSR-EGO

## related_work
Early monocular egocentric methods studied pose estimation from downward-facing head-
mounted cameras, often relying on synthetic supervision or optimization-based temporal
priors to handle missing body evidence [23, 27]. EgoPW [28] further moved toward in-
the-wild egocentric pose estimation using weak external supervision, while SceneEgo [29]
introduced scene-aware constraints by projecting image and depth features into a voxel rep-
resentation to improve physically plausible pose estimation during human-scene interaction.
These works demonstrate the importance of geometric cues in egocentric pose estimation,
but most monocular settings remain underconstrained when large body regions are occluded
or outside the field of view.
Egocentric pose datasets and benchmarks.
Egocentric human pose estimation has evolved
from controlled capture setups toward large-scale, in-the-wild, and multimodal benchmarks.
Early full-body egocentric datasets mainly used head-mounted monocular or stereo cam-
eras to study the camera wearer’s pose under severe viewpoint distortion, self-occlusion,
and limited field of view [21, 23, 30]. Later datasets improved realism and supervision by
introducing in-the-wild egocentric captures, weak external supervision, scene constraints,
and human-interaction settings [28, 29]. More recently, large-scale egocentric resources
such as Ego4D, Ego-Exo4D, and Nymeria have broadened the field from isolated pose es-
timation to first-person video understanding, paired ego-exo learning, multimodal sensing,
and daily human motion modeling [6, 7, 18]. However, many of these datasets are either
not designed specifically for estimating the camera wearer’s full-body 3D pose from stereo
fisheye input, or rely on external views and additional sensors for supervision. In contrast,
stereo fisheye benchmarks such as UnrealEgo, UnrealEgo2, and UnrealEgo-RW directly tar-
get head-mounted full-body pose estimation with complementary views and temporal video
evidence [1, 2]. These benchmarks highlight the central challenge addressed in this work:
exploiting both cross-view geometry and short-term temporal context to recover body joints
under truncation, fast motion, and egocentric self-occlusion.
Stereo and geometry-aware egocentric pose estimation.
Several stereo egocentric meth-
ods exploit intermediate 2D heatmaps, stereo correspondence, or perspective geometry to
recover 3D pose. UnrealEgo [1] uses stereo inputs with 2D keypoint estimation to improve
3D pose prediction, establishing the effectiveness of binocular fisheye observations for ego-
centric motion capture. Ego3DPose [12] identifies two important cues in stereo egocentric
views: cross-view correspondence and strong perspective variation of nearby limbs. It intro-
duces a limb-wise two-path architecture and a perspective-aware orientation representation
to better estimate 3D limb structure from binocular heatmaps. EgoTAP [11] further improves
heatmap-to-3D lifting by encoding stereo heatmaps with a Grid-ViT module and propagating

4
AZAM, QUARLES, DESAI: TSR-EGO
information through a skeleton-aware network, allowing visible joints to support the estima-
tion of occluded or weakly observed joints. These methods show that stereo geometry and
skeletal structure are critical for egocentric pose estimation. However, their predictions still
depend heavily on the quality of current-frame heatmaps or correspondence cues, which can
be unreliable when joints are truncated, occluded, or visible in only one fisheye view.
Transformer and video-based egocentric pose estimation.
Transformer architectures[26]
have improved 3D human pose estimation significantly in recent years. It enables flexible
aggregation across spatial regions, joints, views, and time [5, 14, 15, 20, 22, 32, 33, 36,
37, 38, 39]. Recent egocentric methods have adapted transformer-based attention to better
exploit spatial, temporal, and view-dependent cues. Ego-STAN [19] introduces domain-
guided spatio-temporal self-attention for monocular fisheye egocentric pose estimation, us-
ing feature-map tokens to model fisheye distortion and self-occlusion. In the stereo setting,
EgoPoseFormer [31] formulates egocentric 3D pose estimation as a two-stage transformer
framework, where a coarse 3D pose proposal is refined using deformable stereo attention
over fine-grained multi-view features. EgoTAP [11] also uses a ViT-style heatmap encoder
before skeleton-aware lifting, indicating the benefit of attention-based feature modeling for
stereo egocentric pose. More recently, Akada et al. [2] incorporate depth-aware scene fea-
tures and temporal context into a stereo-video framework, showing that scene geometry and
motion history are useful for challenging motions such as sitting, crouching, and strong self-
occlusion. These works show that attention and temporal context are effective for egocentric
perception. However, temporal information is often used to enrich joint features, aggregate
video representations, or improve pose consistency, while the stereo refinement process itself
remains largely dependent on current-frame spatial evidence.
Our work builds on stereo egocentric pose estimation, transformer-based joint refine-
ment, and temporal video reasoning. Unlike heatmap-lifting methods [11, 12], TSR-Ego
predicts 3D poses directly without intermediate 2D detections. Compared with EgoPose-
Former [31], which refines a current-frame proposal, TSR-Ego injects causal temporal con-
text into both stereo feature maps and joint queries before deformable cross-attention, en-
abling motion-aware stereo refinement.
3

## conclusion
We introduced TSR-Ego, a single-stage framework for egocentric 3D pose estimation from
head-mounted stereo fisheye cameras. TSR-Ego injects causal temporal context directly into
dense stereo features before pose decoding, enabling projection-guided deformable attention
to sample temporally enriched evidence while preserving spatial alignment. A lightweight
decoder then refines learned 3D joint queries through temporal, joint-wise, and stereo cross-
attention to predict 3D poses end-to-end.
Experiments on UnrealEgo2 and UnrealEgo-RW show consistent gains over strong ego-
centric baselines, especially in real-world settings where frame-local stereo cues are often
ambiguous. The ablation studies show consistent contributions from temporal feature mix-
ing, feed-forward refinement, age embedding, joint self-attention, and decoder depth, in-
dicating that TSR-Ego benefits from both temporal feature enrichment and iterative query
refinement.
TSR-Ego still relies on calibrated and synchronized stereo input, and distal joints remain
challenging under severe occlusion or truncation. Future work will explore calibration-robust
stereo attention, stronger kinematic and motion priors, and broader real-world egocentric
capture scenarios.

AZAM, QUARLES, DESAI: TSR-EGO
15
6
Acknowledgements
This material is partially based upon work supported by the National Science Foundation
under Grant No. 2316240 and 2403411. Any opinions, findings, and conclusions or recom-
mendations expressed herein are those of the author(s) and do not reflect National Science
Foundation views.