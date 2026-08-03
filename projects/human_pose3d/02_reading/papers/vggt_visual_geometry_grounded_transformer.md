# VGGT: Visual Geometry Grounded Transformer

> 2025 · id: W4413146238 · arXiv: 2503.11651 · pdf: https://arxiv.org/pdf/2503.11651 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

VGGT: Visual Geometry Grounded Transformer
Jianyuan Wang1,2
Minghao Chen1,2
Nikita Karaev1,2
Andrea Vedaldi1,2
Christian Rupprecht1
David Novotny2
1Visual Geometry Group, University of Oxford
2Meta AI
…
Figure 1. VGGT is a large feed-forward transformer with minimal 3D-inductive biases trained on a trove of 3D-annotated data. It accepts
up to hundreds of images and predicts cameras, point maps, depth maps, and point tracks for all images at once in less than a second, which
often outperforms optimization-based alternatives without further processing.
Abstract
We present VGGT, a feed-forward neural network that di-
rectly infers all key 3D attributes of a scene, including cam-
era parameters, point maps, depth maps, and 3D point
tracks, from one, a few, or hundreds of its views.
This
approach is a step forward in 3D computer vision, where
models have typically been constrained to and special-
ized for single tasks.
It is also simple and efficient, re-
constructing images in under one second, and still out-
performing alternatives that require post-processing with
visual geometry optimization techniques.
The network
achieves state-of-the-art results in multiple 3D tasks, in-
cluding camera parameter estimation, multi-view depth es-
timation, dense point cloud reconstruction, and 3D point
tracking. We also show that using pretrained VGGT as a
feature backbone significantly enhances downstream tasks,
such as non-rigid point tracking and feed-forward novel
view synthesis. Code and models are publicly available at
https://github.com/facebookresearch/vggt.
1. Introduction
We consider the problem of estimating the 3D attributes
of a scene, captured in a set of images, utilizing a feed-
forward neural network. Traditionally, 3D reconstruction
has been approached with visual-geometry methods, uti-
lizing iterative optimization techniques like Bundle Adjust-
ment (BA) [45]. Machine learning has often played an im-
portant complementary role, addressing tasks that cannot
be solved by geometry alone, such as feature matching and
monocular depth prediction. The integration has become
increasingly tight, and now state-of-the-art Structure-from-
Motion (SfM) methods like VGGSfM [125] combine ma-
chine learning and visual geometry end-to-end via differ-
entiable BA. Even so, visual geometry still plays a major
role in 3D reconstruction, which increases complexity and
computational cost.
As networks become ever more powerful, we ask if,
finally, 3D tasks can be solved directly by a neural net-
work, eschewing geometry post-processing almost entirely.
Recent contributions like DUSt3R [129] and its evolution
1
arXiv:2503.11651v1  [cs.CV]  14 Mar 2025

MASt3R [62] have shown promising results in this direc-
tion, but these networks can only process two images at
once and rely on post-processing to reconstruct more im-
ages, fusing pairwise reconstructions.
In this paper, we take a further step towards removing the
need to optimize 3D geometry in post-processing. We do
so by introducing Visual Geometry Grounded Transformer
(VGGT), a feed-forward neural network that performs 3D
reconstruction from one, a few, or even hundreds of input
views of a scene. VGGT predicts a full set of 3D attributes,
including camera parameters, depth maps, point maps, and
3D point tracks. It does so in a single forward pass, in sec-
onds. Remarkably, it often outperforms optimization-based
alternatives even without further processing. This is a sub-
stantial departure from DUSt3R, MASt3R, or VGGSfM,
which still require costly iterative post-optimization to ob-
tain usable results.
We also show that it is unnecessary to design a special
network for 3D reconstruction. Instead, VGGT is based
on a fairly standard large transformer [119], with no par-
ticular 3D or other inductive biases (except for alternating
between frame-wise and global attention), but trained on a
large number of publicly available datasets with 3D annota-
tions. VGGT is thus built in the same mold as large models
for natural language processing and computer vision, such
as GPTs [1, 29, 148], CLIP [86], DINO [10, 78], and Stable
Diffusion [34]. These have emerged as versatile backbones
that can be fine-tuned to solve new, specific tasks. Simi-
larly, we show that the features computed by VGGT can
significantly enhance downstream tasks like point tracking
in dynamic videos, and novel view synthesis.
There are several recent examples of large 3D neural net-
works, including DepthAnything [142], MoGe [128], and
LRM [49]. However, these models only focus on a sin-
gle 3D task, such as monocular depth estimation or novel
view synthesis. In contrast, VGGT uses a shared backbone
to predict all 3D quantities of interest together. We demon-
strate that learning to predict these interrelated 3D attributes
enhances overall accuracy despite potential redundancies.
At the same time, we show that, during inference, we can
derive the point maps from separately predicted depth and
camera parameters, obtaining better accuracy compared to
directly using the dedicated point map head.
To summarize, we make the following contributions: (1)
We introduce VGGT, a large feed-forward transformer that,
given one, a few, or even hundreds of images of a scene,
can predict all its key 3D attributes, including camera intrin-
sics and extrinsics, point maps, depth maps, and 3D point
tracks, in seconds. (2) We demonstrate that VGGT’s pre-
dictions are directly usable, being highly competitive and
usually better than those of state-of-the-art methods that use
slow post-processing optimization techniques. (3) We also
show that, when further combined with BA post-processing,
VGGT achieves state-of-the-art results across the board,
even when compared to methods that specialize in a sub-
set of 3D tasks, often improving quality substantially.
We make our code and models publicly available at
https://github.com/facebookresearch/vggt. We believe that
this will facilitate further research in this direction and bene-
fit the computer vision community by providing a new foun-
dation for fast, reliable, and versatile 3D reconstruction.
2. Related Work
Structure from Motion is a classic computer vision prob-
lem [45, 77, 80] that involves estimating camera parameters
and reconstructing sparse point clouds from a set of images
of a static scene captured from different viewpoints. The
traditional SfM pipeline [2, 36, 70, 94, 103, 134] consists
of multiple stages, including image matching, triangulation,
and bundle adjustment. COLMAP [94] is the most popu-
lar framework based on the traditional pipeline. In recent
years, deep learning has improved many components of the
SfM pipeline, with keypoint detection [21, 31, 116, 149]
and image matching [11, 67, 92, 99] being two primary ar-
eas of focus. Recent methods [5, 102, 109, 112, 113, 118,
122, 125, 131, 160] explored end-to-end differentiable SfM,
where VGGSfM [125] started to outperform traditional al-
gorithms on challenging phototourism scenarios.
Multi-view Stereo aims to densely reconstruct the geome-
try of a scene from multiple overlapping images, typically
assuming known camera parameters, which are often esti-
mated with SfM. MVS methods can be divided into three
categories: traditional handcrafted [38, 39, 96, 130], global
optimization [37, 74, 133, 147], and learning-based meth-
ods [42, 72, 84, 145, 157].
As in SfM, learning-based
MVS approaches have recently seen a lot of progress. Here,
DUSt3R [129] and MASt3R [62] directly estimate aligned
dense point clouds from a pair of views, similar to MVS
but without requiring camera parameters. Some concurrent
works [111, 127, 141, 156] explore replacing DUSt3R’s
test-time optimization with neural networks, though these
attempts achieve only suboptimal or comparable perfor-
mance to DUSt3R. Instead, VGGT outperforms DUSt3R
and MASt3R by a large margin.
Tracking-Any-Point was
first
introduced
in
Particle
Video [91] and revived by PIPs [44] during the deep learn-
ing era, aiming to track points of interest across video se-
quences including dynamic motions.
Given a video and
some 2D query points, the task is to predict 2D correspon-
dences of these points in all other frames. TAP-Vid [23]
proposed three benchmarks for this task and a simple base-
line method later improved in TAPIR [24]. CoTracker [55,
56] utilized correlations between different points to track
through occlusions, while DOT [60] enabled dense track-
ing through occlusions. Recently, TAPTR [63] proposed
2

DINO
Add  
camera token
Frame
Attention
Global
Attention
Camera Head
DPT
Point maps
Tracks
Cameras
Depth maps
× 𝐿 times
Input
Concat
Figure 2. Architecture Overview. Our model first patchifies the input images into tokens by DINO, and appends camera tokens for camera
prediction. It then alternates between frame-wise and global self attention layers. A camera head makes the final prediction for camera
extrinsics and intrinsics, and a DPT [87] head for any dense output.
an end-to-end transformer for this task, and LocoTrack [13]
extended commonly used pointwise features to nearby re-
gions. All of these methods are specialized point trackers.
Here, we demonstrate that VGGT’s features yield state-of-
the-art tracking performance when coupled with existing
point trackers.
3. Method
We introduce VGGT, a large transformer that ingests a set
of images as input and produces a variety of 3D quantities
as output. We start by introducing the problem in Sec. 3.1,
followed by our architecture in Sec. 3.2 and its prediction
heads in Sec. 3.3, and finally the training setup in Sec. 3.4.
3.1. Problem definition and notation
The input is a sequence (Ii)N
i=1 of N RGB images Ii ∈
R3×H×W , observing the same 3D scene. VGGT’s trans-
former is a function that maps this sequence to a corre-
sponding set of 3D annotations, one per frame:
f
 (Ii)N
i=1

= (gi, Di, Pi, Ti)N
i=1 .
(1)
The transformer thus maps each image Ii to its camera pa-
rameters gi ∈R9 (intrinsics and extrinsics), its depth map
Di ∈RH×W , its point map Pi ∈R3×H×W , and a grid
Ti ∈RC×H×W of C-dimensional features for point track-
ing. We explain next how these are defined.
For the camera parameters gi, we use the parametriza-
tion from [125] and set g = [q, t, f] which is the concatena-
tion of the rotation quaternion q ∈R4, the translation vec-
tor t ∈R3, and the field of view f ∈R2. We assume that
the camera’s principal point is at the image center, which is
common in SfM frameworks [95, 125].
We denote the domain of the image Ii with I(Ii) =
{1, . . . , H} × {1, . . . , W}, i.e., the set of pixel locations.
The depth map Di associates each pixel location y ∈I(Ii)
with its corresponding depth value Di(y) ∈R+, as ob-
served from the i-th camera. Likewise, the point map Pi
associates each pixel with its corresponding 3D scene point
Pi(y) ∈R3. Importantly, like in DUSt3R [129], the point
maps are viewpoint invariant, meaning that the 3D points
Pi(y) are defined in the coordinate system of the first cam-
era g1, which we take as the world reference frame.
Finally, for keypoint tracking, we follow track-any-
point methods such as [25, 57]. Namely, given a fixed query
image point yq in the query image Iq, the network outputs
a track T ⋆(yq) = (yi)N
i=1 formed by the corresponding 2D
points yi ∈R2 in all images Ii.
Note that the transformer f above does not output the
tracks directly but instead features Ti ∈RC×H×W , which
are used for tracking. The tracking is delegated to a sep-
arate module, described in Sec. 3.3, which implements a
function T ((yj)M
j=1, (Ti)N
i=1) = ((ˆyj,i)N
i=1)M
j=1. It ingests
the query point yq and the dense tracking features Ti output
by the transformer f and then computes the track. The two
networks f and T are trained jointly end-to-end.
Order of Predictions. The order of the images in the input
sequence is arbitrary, except that the first image is chosen as
the reference frame. The network architecture is designed
to be permutation equivariant for all but the first frame.
Over-complete Predictions. Notably, not all quantities
predicted by VGGT are independent.
For example, as
shown by DUSt3R [129], the camera parameters g can
be inferred from the invariant point map P, for instance,
by solving the Perspective-n-Point (PnP) problem [35, 61].
3

Ours 
Input
Single view
Two views
< 0.1s
< 0.1s
< 0.1s
< 0.1s
DUSt3R
…
…
32 views
> 200s
< 0.6 s
Figure 3. Qualitative comparison of our predicted 3D points to DUSt3R on in-the-wild images. As shown in the top row, our method
successfully predicts the geometric structure of an oil painting, while DUSt3R predicts a slightly distorted plane. In the second row, our
method correctly recovers a 3D scene from two images with no overlap, while DUSt3R fails. The third row provides a challenging example
with repeated textures, while our prediction is still high-quality. We do not include examples with more than 32 frames, as DUSt3R runs
out of memory beyond this limit.
Furthermore, the depth maps can be deduced from the point
map and the camera parameters. However, as we show in
Sec. 4.5, tasking VGGT with explicitly predicting all afore-
mentioned quantities during training brings substantial per-
formance gains, even when these are related by closed-form
relationships. Meanwhile, during inference, it is observed
that combining independently estimated depth maps and
camera parameters produces more accurate 3D points com-
pared to directly employing a specialized point map branch.
3.2. Feature Backbone
Following recent works in 3D deep learning [53, 129, 132],
we design a simple architecture with minimal 3D induc-
tive biases, letting the model learn from ample quantities
of 3D-annotated data.
In particular, we implement the
model f as a large transformer [119]. To this end, each
input image I is initially patchified into a set of K tokens1
tI ∈RK×C through DINO [78]. The combined set of im-
age tokens from all frames, i.e., tI = ∪N
i=1{tI
i }, is subse-
quently processed through the main network structure, al-
ternating frame-wise and global self-attention layers.
Alternating-Attention. We slightly adjust the standard
transformer design by introducing Alternating-Attention
1The number of tokens depends on the image resolution.
(AA), making the transformer focus within each frame and
globally in an alternate fashion. Specifically, frame-wise
self-attention attends to the tokens tI
k within each frame
separately, and global self-attention attends to the tokens
tI across all frames jointly. This strikes a balance between
integrating information across different images and normal-
izing the activations for the tokens within each image. By
default, we employ L = 24 layers of global and frame-wise
attention. In Sec. 4, we demonstrate that our AA architec-
ture brings significant performance gains. Note that our ar-
chitecture does not employ any cross-attention layers, only
self-attention ones.
3.3. Prediction heads
Here, we describe how f predicts the camera parameters,
depth maps, point maps, and point tracks. First, for each
input image Ii, we augment the corresponding image to-
kens tI
i with an additional camera token tg
i ∈R1×C′ and
four register tokens [19] tR
i ∈R4×C′. The concatenation of
(tI
i , tg
i , tR
i j)N
i=1 is then passed to the AA transformer, yield-
ing output tokens (ˆtI
i ,ˆtg
i ,ˆtR
i )N
i=1. Here, the camera token
and register tokens of the first frame (tg
1 := ¯tg, tR
1 := ¯tR)
are set to a different set of learnable tokens t
g, t
R than those
of all other frames (tg
i := t
g, tR
i
:= t
R, i ∈[2, . . . , N]),
4

Figure 4. Additional Visualizations of Point Map Estimation. Camera frustums illustrate the estimated camera poses. Explore our
interactive demo for better visualization quality.
which are also learnable. This allows the model to distin-
guish the first frame from the rest, and to represent the 3D
predictions in the coordinate frame of the first camera. Note
that the refined camera and register tokens now become
frame-specific—–this is because our AA transformer con-
tains frame-wise self-attention layers that allow the trans-
former to match the camera and register tokens with the cor-
responding tokens from the same image. Following com-
mon practice, the output register tokens ˆtR
i are discarded
while ˆtI
i , ˆtg
i are used for prediction.
Coordinate Frame. As noted above, we predict cameras,
point maps, and depth maps in the coordinate frame of the
first camera g1. As such, the camera extrinsics output for
the first camera are set to the identity, i.e., the first rotation
quaternion is q1 = [0, 0, 0, 1] and the first translation vector
is t1 = [0, 0, 0]. Recall that the special camera and register
tokens tg
1 := t
g, tR
1 := t
R allow the transformer to identify
the first camera.
Camera Predictions. The camera parameters (ˆgi)N
i=1 are
predicted from the output camera tokens (ˆtg
i )N
i=1 using four
additional self-attention layers followed by a linear layer.
This forms the camera head that predicts the camera intrin-
sics and extrinsics.
Dense Predictions. The output image tokens ˆtI
i are used
to predict the dense outputs, i.e., the depth maps Di, point
maps Pi, and tracking features Ti. More specifically, ˆtI
i are
first converted to dense feature maps Fi ∈RC′′×H×W with
a DPT layer [87]. Each Fi is then mapped with a 3 × 3 con-
volutional layer to the corresponding depth and point maps
Di and Pi. Additionally, the DPT head also outputs dense
features Ti ∈RC×H×W , which serve as input to the track-
ing head. We also predict the aleatoric uncertainty [58, 76]
ΣD
i ∈RH×W
+
and ΣP
i ∈RH×W
+
for each depth and point
map, respectively. As described in Sec. 3.4, the uncertainty
maps are used in the loss and, after training, are proportional
to the model’s confidence in the predictions.
Tracking. In order to implement the tracking module T ,
we use the CoTracker2 architecture [57], which takes the
dense tracking features Ti as input. More specifically, given
a query point yj in a query image Iq (during training, we al-
ways set q = 1, but any other image can be potentially used
as a query), the tracking head T predicts the set of 2D points
T ((yj)M
j=1, (Ti)N
i=1) = ((ˆyj,i)N
i=1)M
j=1 in all images Ii that
correspond to the same 3D point as y. To do so, the fea-
ture map Tq of the query image is first bilinearly sampled at
5

the query point yj to obtain its feature. This feature is then
correlated with all other feature maps Ti, i̸ = q to obtain a
set of correlation maps. These maps are then processed by
self-attention layers to predict the final 2D points ˆyi, which
are all in correspondence with yj. Note that, similar to VG-
GSfM [125], our tracker does not assume any temporal or-
dering of the input frames and, hence, can be applied to any
set of input images, not just videos.
3.4. Training
Training Losses. We train the VGGT model f end-to-end
using a multi-task loss:
L = Lcamera + Ldepth + Lpmap + λLtrack.
(2)
We found that the camera (Lcamera), depth (Ldepth), and
point-map (Lpmap) losses have similar ranges and do not
need to be weighted against each other. The tracking loss
Ltrack is down-weighted with a factor of λ = 0.05. We de-
scribe each loss term in turn.
The camera loss Lcamera supervises the cameras ˆg:
Lcamera = PN
i=1 ∥ˆgi −gi∥ϵ , comparing the predicted cam-
eras ˆgi with the ground truth gi using the Huber loss | · |ϵ.
The depth loss Ldepth follows DUSt3R [129] and im-
plements the aleatoric-uncertainty loss [59, 75] weighing
the discrepancy between the predicted depth ˆDi and the
ground-truth depth Di with the predicted uncertainty map
ˆΣD
i . Differently from DUSt3R, we also apply a gradient-
based term, which is widely used in monocular depth es-
timation. Hence, the depth loss is Ldepth = PN
i=1 ∥ΣD
i ⊙
( ˆDi −Di)∥+ ∥ΣD
i ⊙(∇ˆDi −∇Di)∥−α log ΣD
i , where ⊙
is the channel-broadcast element-wise product. The point
map loss is defined analogously but with the point-map un-
certainty ΣP
i : Lpmap = PN
i=1 ∥ΣP
i ⊙( ˆPi −Pi)∥+ ∥ΣP
i ⊙
(∇ˆPi −∇Pi)∥−α log ΣP
i .
Finally,
the tracking loss is given by Ltrack
=
PM
j=1
PN
i=1 ∥yj,i −ˆyj,i∥. Here, the outer sum runs over
all ground-truth query points yj in the query image Iq, yj,i
is yj’s ground-truth correspondence in image Ii, and ˆyj,i
is the corresponding prediction obtained by the application
T ((yj)M
j=1, (Ti)N
i=1) of the tracking module. Additionally,
following CoTracker2 [57], we apply a visibility loss (bi-
nary cross-entropy) to estimate whether a point is visible in
a given frame.
Ground Truth Coordinate Normalization. If we scale a
scene or change its global reference frame, the images of the
scene are not affected at all, meaning that any such variant
is a legitimate result of 3D reconstruction. We remove this
ambiguity by normalizing the data, thus making a canoni-
cal choice and task the transformer to output this particular
variant. We follow [129] and, first, express all quantities in
the coordinate frame of the first camera g1. Then, we com-
pute the average Euclidean distance of all 3D points in the
point map P to the origin and use this scale to normalize
the camera translations t, the point map P, and the depth
map D. Importantly, unlike [129], we do not apply such
normalization to the predictions output by the transformer;
instead, we force it to learn the normalization we choose
from the training data.
Implementation Details. By default, we employ L = 24
layers of global and frame-wise attention, respectively. The
model consists of approximately 1.2 billion parameters in
total. We train the model by optimizing the training loss (2)
with the AdamW optimizer for 160K iterations. We use a
cosine learning rate scheduler with a peak learning rate of
0.0002 and a warmup of 8K iterations. For every batch,
we randomly sample 2–24 frames from a random training
scene. The input frames, depth maps, and point maps are re-
sized to a maximum dimension of 518 pixels. The aspect ra-
tio is randomized between 0.33 and 1.0. We also randomly
apply color jittering, Gaussian blur, and grayscale augmen-
tation to the frames. The training runs on 64 A100 GPUs
over nine days. We employ gradient norm clipping with a
threshold of 1.0 to ensure training stability. We leverage
bfloat16 precision and gradient checkpointing to improve
GPU memory and computational efficiency.
Training Data. The model was trained using a large and
diverse collection of datasets, including: Co3Dv2 [88],
BlendMVS
[146],
DL3DV
[69],
MegaDepth
[64],
Kubric [41],
WildRGB [135],
ScanNet [18],
Hyper-
Sim [89], Mapillary [71], Habitat [107], Replica [104],
MVS-Synth [50], PointOdyssey [159], Virtual KITTI [7],
Aria Synthetic Environments [82], Aria Digital Twin [82],
and a synthetic dataset of artist-created assets similar
to Objaverse [20]. These datasets span various domains,
including indoor and outdoor environments, and encompass
synthetic and real-world scenarios.
The 3D annotations
for these datasets are derived from multiple sources,
such as direct sensor capture, synthetic engines, or SfM
techniques [95]. The combination of our datasets is broadly
comparable to those of MASt3R [30] in size and diversity.
4. Experiments
This section compares our method to state-of-the-art ap-
proaches across multiple tasks to show its effectiveness.
4.1. Camera Pose Estimation
We first evaluate our method on the CO3Dv2 [88] and
RealEstate10K [161] datasets for camera pose estimation,
as shown in Tab. 1. Following [124], we randomly select 10
images per scene and evaluate them using the standard met-
ric AUC@30, which combines RRA and RTA. RRA (Rela-
tive Rotation Accuracy) and RTA (Relative Translation Ac-
curacy) calculate the relative angular errors in rotation and
translation, respectively, for each image pair. These angu-
6

Methods
Re10K (unseen)
CO3Dv2
Time
AUC@30 ↑
AUC@30 ↑
Colmap+SPSG [92]
45.2
25.3
∼15s
PixSfM [66]
49.4
30.1
> 20s
PoseDiff [124]
48.0
66.5
∼7s
DUSt3R [129]
67.7
76.7
∼7s
MASt3R [62]
76.4
81.8
∼9s
VGGSfM v2 [125]
78.9
83.4
∼10s
MV-DUSt3R [111] ‡
71.3
69.5
∼0.6s
CUT3R [127] ‡
75.3
82.8
∼0.6s
FLARE [156] ‡
78.8
83.3
∼0.5s
Fast3R [141] ‡
72.7
82.5
∼0.2s
Ours (Feed-Forward)
85.3
88.2
∼0.2s
Ours (with BA)
93.5
91.8
∼1.8s
Table 1. Camera Pose Estimation on RealEstate10K [161] and
CO3Dv2 [88] with 10 random frames. All metrics the higher the
better. None of the methods were trained on the Re10K dataset.
Runtime were measured using one H100 GPU. Methods marked
with ‡ represent concurrent work.
Known GT
Method
Acc.↓
Comp.↓
Overall↓
camera
✓
Gipuma [40]
0.283
0.873
0.578
✓
MVSNet [144]
0.396
0.527
0.462
✓
CIDER [139]
0.417
0.437
0.427
✓
PatchmatchNet [121]
0.427
0.377
0.417
✓
MASt3R [62]
0.403
0.344
0.374
✓
GeoMVSNet [157]
0.331
0.259
0.295
✗
DUSt3R [129]
2.677
0.805
1.741
✗
Ours
0.389
0.374
0.382
Table 2.
Dense MVS Estimation on the DTU [51] Dataset.
Methods operating with known ground-truth camera are in the top
part of the table, while the bottom part contains the methods that
do not know the ground-truth camera.
Methods
Acc.↓
Comp.↓
Overall↓
Time
DUSt3R
1.167
0.842
1.005
∼7s
MASt3R
0.968
0.684
0.826
∼9s
Ours (Point)
0.901
0.518
0.709
∼0.2s
Ours (Depth + Cam)
0.873
0.482
0.677
∼0.2s
Table 3. Point Map Estimation on ETH3D [97]. DUSt3R and
MASt3R use global alignment while ours is feed-forward and,
hence, much faster. The row Ours (Point) indicates the results
using the point map head directly, while Ours (Depth + Cam) de-
notes constructing point clouds from the depth map head com-
bined with the camera head.
lar errors are then thresholded to determine the accuracy
scores. AUC is the area under the accuracy-threshold curve
of the minimum values between RRA and RTA across vary-
ing thresholds. The (learnable) methods in Tab. 1 have been
trained on Co3Dv2 and not on RealEstate10K. Our feed-
forward model consistently outperforms competing meth-
Method
AUC@5 ↑
AUC@10 ↑
AUC@20 ↑
SuperGlue [92]
16.2
33.8
51.8
LoFTR [105]
22.1
40.8
57.6
DKM [32]
29.4
50.7
68.3
CasMTR [9]
27.1
47.0
64.4
Roma [33]
31.8
53.4
70.9
Ours
33.9
55.2
73.4
Table 4. Two-View matching comparison on ScanNet-1500 [18,
92]. Although our tracking head is not specialized for the two-
view setting, it outperforms the state-of-the-art two-view matching
method Roma. Measured in AUC (higher is better).
ods across all metrics on both datasets, including those that
employ computationally expensive post-optimization steps,
such as Global Alignment for DUSt3R/MASt3R and Bun-
dle Adjustment for VGGSfM, typically requiring more than
10 seconds. In contrast, VGGT achieves superior perfor-
mance while only operating in a feed-forward manner, re-
quiring just 0.2 seconds on the same hardware. Compared
to concurrent works [111, 127, 141, 156] (indicated by ‡),
our method demonstrates significant performance advan-
tages, with speed similar to the fastest variant Fast3R [141].
Furthermore, our model’s performance advantage is even
more pronounced on the RealEstate10K dataset, which
none of the methods presented in Tab. 1 were trained on.
This validates the superior generalization of VGGT.
Our results also show that VGGT can be improved even
further by combining it with optimization methods from vi-
sual geometry optimization like BA. Specifically, refining
the predicted camera poses and tracks with BA further im-
proves accuracy.
Note that our method directly predicts
close-to-accurate point/depth maps, which can serve as a
good initialization for BA. This eliminates the need for tri-
angulation and iterative refinement in BA as done by [125],
making our approach significantly faster (only around 2 sec-
onds even with BA). Hence, while the feed-forward mode of
VGGT outperforms all previous alternatives (whether they
are feed-forward or not), there is still room for improvement
since post-optimization still brings benefits.
4.2. Multi-view Depth Estimation
Following MASt3R [62], we further evaluate our multi-
view depth estimation results on the DTU [51] dataset. We
report the standard DTU metrics, including Accuracy (the
smallest Euclidean distance from the prediction to ground
truth), Completeness (the smallest Euclidean distance from
the ground truth to prediction), and their average Overall
(i.e., Chamfer distance). In Tab. 2, DUSt3R and our VGGT
are the only two methods operating without the knowl-
edge of ground truth cameras. MASt3R derives depth maps
by triangulating matches using the ground truth cameras.
Meanwhile, deep multi-view stereo methods like GeoMVS-
7

Ours
CoTracker
+Ours  
Figure 5. Visualization of Rigid and Dynamic Point Tracking. Top: VGGT’s tracking module T outputs keypoint tracks for an
unordered set of input images depicting a static scene. Bottom: We finetune the backbone of VGGT to enhance a dynamic point tracker
CoTracker [56], which processes sequential inputs.
Net use ground truth cameras to construct cost volumes.
Our method substantially outperforms DUSt3R, reduc-
ing the Overall score from 1.741 to 0.382. More impor-
tantly, it achieves results comparable to methods that know
ground-truth cameras at test time. The significant perfor-
mance gains can likely be attributed to our model’s multi-
image training scheme that teaches it to reason about multi-
view triangulation natively, instead of relying on ad hoc
alignment procedures, such as in DUSt3R, which only av-
erages multiple pairwise camera triangulations.
4.3. Point Map Estimation
We also compare the accuracy of our predicted point cloud
to DUSt3R and MASt3R on the ETH3D [97] dataset. For
each scene, we randomly sample 10 frames.
The pre-
dicted point cloud is aligned to the ground truth using the
Umeyama [117] algorithm. The results are reported after
filtering out invalid points using the official masks. We re-
port Accuracy, Completeness, and Overall (Chamfer dis-
tance) for point map estimation. As shown in Tab. 3, al-
though DUSt3R and MASt3R conduct expensive optimiza-
tion (global alignment–—around 10 seconds per scene),
our method still outperforms them significantly in a simple
feed-forward regime at only 0.2 seconds per reconstruction.
Meanwhile, compared to directly using our estimated
point maps, we found that the predictions from our depth
and camera heads (i.e., unprojecting the predicted depth
maps to 3D using the predicted camera parameters) yield
higher accuracy.
We attribute this to the benefits of de-
composing a complex task (point map estimation) into sim-
pler subproblems (depth map and camera prediction), even
though camera, depth maps, and point maps are jointly su-
pervised during training.
We present a qualitative comparison with DUSt3R on in-
the-wild scenes in Fig. 3 and further examples in Fig. 4.
VGGT outputs high-quality predictions and generalizes
ETH3D Dataset
Acc.↓
Comp.↓
Overall↓
Cross-Attention
1.287
0.835
1.061
Global Self-Attention Only
1.032
0.621
0.827
Alternating-Attention
0.901
0.518
0.709
Table 5. Ablation Study for Transformer Backbone on ETH3D.
We compare our alternating-attention architecture against two
variants: one using only global self-attention and another employ-
ing cross-attention.
well, excelling on challenging out-of-domain examples,
such as oil paintings, non-overlapping frames, and scenes
with repeating or homogeneous textures like deserts.
4.4. Image Matching
Two-view image matching is a widely-explored topic [68,
93, 105] in computer vision. It represents a specific case of
rigid point tracking, which is restricted to only two views,
and hence a suitable evaluation benchmark to measure our
tracking accuracy, even though our model is not specialized
for this task. We follow the standard protocol [33, 93] on
the ScanNet dataset [18] and report the results in Tab. 4.
For each image pair, we extract the matches and use them
to estimate an essential matrix, which is then decomposed
to a relative camera pose. The final metric is the relative
pose accuracy, measured by AUC. For evaluation, we use
ALIKED [158] to detect keypoints, treating them as query
points yq. These are then passed to our tracking branch T
to find correspondences in the second frame. We adopt the
evaluation hyperparameters (e.g., the number of matches,
RANSAC thresholds) from Roma [33]. Despite not being
explicitly trained for two-view matching, Tab. 4 shows that
VGGT achieves the highest accuracy among all baselines.
4.5. Ablation Studies
Feature Backbone. We first validate the effectiveness of
our proposed Alternating-Attention design by comparing it
8

w. Lcamera
w. Ldepth
w. Ltrack
Acc.↓
Comp.↓
Overall↓
✗
✓
✓
1.042
0.627
0.834
✓
✗
✓
0.920
0.534
0.727
✓
✓
✗
0.976
0.603
0.790
✓
✓
✓
0.901
0.518
0.709
Table 6. Ablation Study for Multi-task Learning, which shows
that simultaneous training with camera, depth and track estimation
yields the highest accuracy in point map estimation on ETH3D.
against two alternative attention architectures: (a) global
self-attention only, and (b) cross-attention.
To ensure a
fair comparison, all model variants maintain an identical
number of parameters, using a total of 2L attention lay-
ers. For the cross-attention variant, each frame indepen-
dently attends to tokens from all other frames, maximiz-
ing cross-frame information fusion although significantly
increasing the runtime, particularly as the number of input
frames grows. The hyperparameters such as the hidden di-
mension and the number of heads are kept the same. Point
map estimation accuracy is chosen as the evaluation metric
for our ablation study, as it reflects the model’s joint under-
standing of scene geometry and camera parameters. Results
in Tab. 5 demonstrate that our Alternating-Attention archi-
tecture outperforms both baseline variants by a clear mar-
gin. Additionally, our other preliminary exploratory exper-
iments consistently showed that architectures using cross-
attention generally underperform compared to those exclu-
sively employing self-attention.
Multi-task Learning. We also verify the benefit of train-
ing a single network to simultaneously learn multiple 3D
quantities, even though these outputs may potentially over-
lap (e.g., depth maps and camera parameters together can
produce point maps). As shown in Tab. 6, there is a no-
ticeable decrease in the accuracy of point map estimation
when training without camera, depth, or track estimation.
Notably, incorporating camera parameter estimation clearly
enhances point map accuracy, whereas depth estimation
contributes only marginal improvements.
4.6. Finetuning for Downstream Tasks
We now show that the VGGT pre-trained feature extractor
can be reused in downstream tasks. We show this for feed-
forward novel view synthesis and dynamic point tracking.
Feed-forward
Novel
View
Synthesis is
progressing
rapidly [8, 43, 49, 53, 108, 126, 140, 155]. Most exist-
ing methods take images with known camera parameters as
input and predict the target image corresponding to a new
camera viewpoint. Instead of relying on an explicit 3D rep-
resentation, we follow LVSM [53] and modify VGGT to di-
rectly output the target image. However, we do not assume
known camera parameters for the input frames.
We follow the training and evaluation protocol of LVSM
Input Images
Ground Truth
Prediction
Figure 6. Qualitative Examples of Novel View Synthesis. The
top row shows the input images, the middle row displays the
ground truth images from target viewpoints, and the bottom row
presents our synthesized images.
Method
Known Input Cam
Size
PSNR ↑
SSIM ↑
LPIPS ↓
LGM [110]
✓
256
21.44
0.832
0.122
GS-LRM [154]
✓
256
29.59
0.944
0.051
LVSM [53]
✓
256
31.71
0.957
0.027
Ours-NVS∗
✗
224
30.41
0.949
0.033
Table 7.
Quantitative comparisons for view synthesis on
GSO [28] dataset. Finetuning VGGT for feed-forward novel view
synthesis, it demonstrates competitive performance even without
knowing camera extrinsic and intrinsic parameters for the input
images. Note that ∗indicates using a small training set (only 20%).
closely, e.g., using 4 input views and adopting Pl¨ucker rays
to represent target viewpoints. We make a simple modifi-
cation to VGGT. As before, the input images are converted
into tokens by DINO. Then, for the target views, we use
a convolutional layer to encode their Pl¨ucker ray images
into tokens. These tokens, representing both the input im-
ages and the target views, are concatenated and processed
by the AA transformer. Subsequently, a DPT head is used
to regress the RGB colors for the target views. It is impor-
tant to note that we do not input the Pl¨ucker rays for the
source images. Hence, the model is not given the camera
parameters for these input frames.
LVSM was trained on the Objaverse dataset [20]. We
use a similar internal dataset of approximately 20% the size
of Objaverse. Further details on training and evaluation can
be found in [53]. As shown in Tab. 7, despite not requir-
ing the input camera parameters and using less training data
than LVSM, our model achieves competitive results on the
GSO dataset [28]. We expect that better results would be
obtained using a larger training dataset. Qualitative exam-
ples are shown in Fig. 6.
Dynamic Point Tracking has emerged as a highly com-
petitive task in recent years [25, 44, 57, 136], and it serves
as another downstream application for our learned features.
Following standard practices, we report these point-tracking
metrics: Occlusion Accuracy (OA), which comprises the bi-
nary accuracy of occlusion predictions; δvis
avg, comprising the
9

Method
Kinetics
RGB-S
DAVIS
AJ
δvis
avg
OA
AJ
δvis
avg
OA
AJ
δvis
avg
OA
TAPTR [63]
49.0 64.4 85.2 60.8 76.2 87.0 63.0 76.1 91.1
LocoTrack [13]
52.9 66.8 85.3 69.7 83.2 89.5 62.9 75.3 87.2
BootsTAPIR [26] 54.6 68.4 86.5 70.8 83.0 89.9 61.4 73.6 88.7
CoTracker [56]
49.6 64.3 83.3 67.4 78.9 85.2 61.8 76.1 88.3
CoTracker + Ours 57.2 69.0 88.9 72.1 84.0 91.6 64.7 77.5 91.4
Table 8.
Dynamic Point Tracking Results on the TAP-Vid
benchmarks. Although our model was not designed for dynamic
scenes, simply fine-tuning CoTracker with our pretrained weights
significantly enhances performance, demonstrating the robustness
and effectiveness of our learned features.
mean proportion of visible points accurately tracked within
a certain pixel threshold; and Average Jaccard (AJ), mea-
suring tracking and occlusion prediction accuracy together.
We adapt the state-of-the-art CoTracker2 model [57] by
substituting its backbone with our pretrained feature back-
bone. This is necessary because VGGT is trained on un-
ordered image collections instead of sequential videos. Our
backbone predicts the tracking features Ti, which replace
the outputs of the feature extractor and later enter the rest of
the CoTracker2 architecture, that finally predicts the tracks.
We finetune the entire modified tracker on Kubric [41]. As
illustrated in Tab. 8, the integration of pretrained VGGT sig-
nificantly enhances CoTracker’s performance on the TAP-
Vid benchmark [23]. For instance, VGGT’s tracking fea-
tures improve the δvis
avg metric from 78.9 to 84.0 on the TAP-
Vid RGB-S dataset. Despite the TAP-Vid benchmark’s in-
clusion of videos featuring rapid dynamic motions from var-
ious data sources, our model’s strong performance demon-
strates the generalization capability of its features, even in
scenarios for which it was not explicitly designed.
5. Discussions
Limitations. While our method exhibits strong generaliza-
tion to diverse in-the-wild scenes, several limitations re-
main.
First, the current model does not support fisheye
or panoramic images. Additionally, reconstruction perfor-
mance drops under conditions involving extreme input ro-
tations. Moreover, although our model handles scenes with
minor non-rigid motions, it fails in scenarios involving sub-
stantial non-rigid deformation.
However, an important advantage of our approach is its
flexibility and ease of adaptation. Addressing these limi-
tations can be straightforwardly achieved by fine-tuning the
model on targeted datasets with minimal architectural modi-
fications. This adaptability clearly d