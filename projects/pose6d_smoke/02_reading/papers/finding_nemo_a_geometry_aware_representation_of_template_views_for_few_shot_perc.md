# Finding NeMO: A Geometry-Aware Representation of Template Views for Few-Shot Perception

> 2026 · id: arxiv:2602.04343 · arXiv: arxiv:2602.04343 · pdf: https://arxiv.org/pdf/2602.04343 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Finding NeMO: A Geometry-Aware Representation of Template Views for
Few-Shot Perception
Sebastian Jung
Leonard Kl¨upfel
Rudolph Triebel
Maximilian Durner
German Aerospace Center (DLR)
{Sebastian.Jung, Leonard.Kluepfel, Rudolph.Triebel, Maximilian.Durner}@dlr.de
RGB Template Views
Dense Predictions
Decoder
Encoder
Query Image
NeMO
Figure 1. Overview. Our method uses a multi-view encoder to generate an object-centric geometric encoding called Neural Memory Object
(NeMO) with its own coordinate system from a set of RGB images depicting an object unseen during training. A decoder uses the NeMO
to retrieve dense predictions allowing us to detect, segment, estimate the objects surface and determine the camera-to-object position on an
RGB query image. Even in cluttered scenes, our method is able to find the object, which we can use to crop the corresponding region of
interest, demonstrating that our method can be used for multi-stage perception pipelines. Images were captured using a normal smartphone.
Abstract
We present Neural Memory Object (NeMO), a novel
object-centric representation that can be used to detect, seg-
ment and estimate the 6DoF pose of objects unseen dur-
ing training using RGB images. Our method consists of
an encoder that requires only a few RGB template views
depicting an object to generate a sparse object-like point
cloud using a learned UDF containing semantic and geo-
metric information. Next, a decoder takes the object en-
coding together with a query image to generate a variety
of dense predictions. Through extensive experiments, we
show that our method can be used for few-shot object per-
ception without requiring any camera-specific parameters
or retraining on target data. Our proposed concept of out-
sourcing object information in a NeMO and using a single
network for multiple perception tasks enhances interaction
with novel objects, improving scalability and efficiency by
enabling quick object onboarding without retraining or ex-
tensive pre-processing. We report competitive and state-of-
the-art results on various datasets and perception tasks of
the BOP benchmark, demonstrating the versatility of our
approach. https://github.com/DLR-RM/nemo
1. Introduction
Objects play a central role in our daily lives, and recog-
nizing them in images is critical for applications such as
robotics, augmented reality, and autonomous systems. Re-
cent advances in deep learning and computer vision have
greatly improved object perception, especially in model-
based approaches that leverage 3D CAD models to train
deep networks for detection, segmentation, and pose esti-
mation [25]. These methods benefit from large-scale syn-
thetic training [11] and can specialize in specific objects
or categories, achieving impressive performance when 3D
models are available at test time. However, in many real-
world scenarios, it is impractical to assume access to a 3D
model for every object. As a result, model-free perception
has become a growing research focus, aiming to rapidly on-
board and recognize novel objects.
When a CAD model is unavailable during training but
provided at inference, recent methods adapt perception
models using rendered templates of the target object, lever-
aging the geometric and textural cues from these synthetic
views. Some approaches fine-tune object-specific networks
in minimal time [58], while others use general-purpose
1
arXiv:2602.04343v1  [cs.CV]  4 Feb 2026

features to recognize new objects without retraining [43].
A common strategy compares features between template
views and query images, followed by post-processing [29,
44].
However, these approaches scale poorly with the
number of templates and rely on pairwise local compar-
isons [12, 50, 53], without jointly reasoning over all views.
While still affected by the sim-to-real gap, such methods
enable fast onboarding of CAD models.
In model-free object perception – where no CAD model
is available at any stage – the challenge is greater. Synthetic
training tailored to the object is infeasible, and geometric
information must be extracted from real reference images,
where object-to-camera poses are typically unknown. Some
methods rely on template or local feature matching using
real images [22, 32], while others employ neural fields [38]
to reconstruct CAD-like geometry [61, 62], requiring ex-
trinsic pose information.
To overcome the limitations of existing perception sys-
tems in terms of generalization, scalability, and efficiency,
we propose a novel encoder-decoder architecture trained
on a large-scale synthetic dataset. Our method constructs
a geometry-aware representation, termed Neural Memory
Object (NeMO), from a set of unordered, object-centric
RGB images, without requiring camera calibration or pose
annotations. NeMO is formulated as a sparse, continuous
point cloud that encapsulates both semantic and geometric
features observed from multiple viewpoints. Unlike con-
ventional encoder-decoder models that compress inputs into
a single latent vector [51, 55], NeMO preserves a structured,
interpretable abstraction of object geometry, enabling trans-
formations such as translation, rotation, and scaling without
reliance on a 3D CAD model. Critically, the object-specific
information is disentangled from the decoder’s parame-
ters, enabling object-agnostic inference and robust gener-
alization to novel instances not seen during training. The
point-based representation supports incremental refinement
through the addition of new views, without necessitating re-
processing of previous observations. Furthermore, the de-
coupling of encoding and decoding facilitates efficient de-
ployment, as NeMO can be precomputed offline, rendering
inference time invariant to the number of input views and
improving scalability for real-world applications.
We extensively ablate the representation to analyze its
potential and quantitatively demonstrate its performance
on multiple object perception tasks, achieving competitive
and state-of-the-art results on model-free and model-based
unseen object benchmarks. Additionally, we qualitatively
show its potential for object surface reconstruction. Con-
cretely, our contributions are threefold:
• We propose the NeMO, a geometry-sensitive, and com-
pact representation of template views well-suited for few-
shot object perception tasks.
• We evaluate our encoder-decoder network on the task of
model-based and -free few-shot detection, segmentation
and pose estimation of unseen objects. Without training
on test objects, we perform competitively and partially
better against state-of-the-art methods.
• We
contribute
an
object-centric
diverse
synthetic
dataset, mimicking realistic cluttered scenes that bal-
ances the occurrences of different object classes to ad-
vance research on related topics.
2. Related Work
Our work focuses on the perception of previously unseen
objects, encompassing detection, segmentation, and pose
estimation. Unseen refers to objects not explicitly seen dur-
ing training, provided at runtime via 3D models (model-
based) or a set of RGB template images (model-free). While
designed for the model-free setting, our method also sup-
ports model-based input.
Unseen Object Segmentation and Detection. Given a tar-
get object, GFreeDet [35], a model-free method, detects and
segments objects through reconstructing a Gaussian object
and comparing generated templates to region proposal from
the query image. NOCTIS [17] similarly computes segmen-
tation masks by matching the appearance and semantic of
reference images and a query image as obtained through
Grounded SAM2 and DINOv2.
Instead of learning a presentation to render template images,
model-based approaches directly use the provided CAD-
model. To this end, CNOS [43] matches DINOv2 [45] to-
kens of template renderings with region proposals derived
from SAM/ FastSAM [28, 66]. Similarly, NIDS-Net [37]
also leverages SAM [28] and Grounding DINO [34] to ob-
tain proposal embeddings which are compared to averaged
and masked DINOv2’s template embeddings of the object.
OC-DiT [57] uses latent diffusion models conditioned on
template images to generate segmentation masks.
Unseen Object Pose Estimation. Model-free pose esti-
mation also commonly requires multiple reference regis-
tration images [31, 47, 54, 62]. To this end, FS6D [22]
predicts 6D poses by fusing features from support RGBD
images and the query scene to establish dense correspon-
dences. OnePose [54] and OnePose++ [21] employ pair-
wise image matching of dense 2D-3D correspondences hi-
erarchically in a coarse-to-fine fashion. RelPose [65] es-
timates the relative camera rotation between image pairs
through an energy-based formulation, which is extended to
the relative 6D pose based on multiple images in the exten-
sion RelPose++ [32]. In contrast, PIZZA [42] approaches
6D object tracking through either an image pair or multi-
ple template images between which the relative pose is esti-
mated. Similarly, BundleSDF [61] jointly tracks poses and
learns a Neural Object Field from a RGB-D stream, using
RGB renderings from the learned field to supervise texture
and geometry prediction. FoundationPose [62] estimates ei-
2

NeMO
Decoder
ViT
ViT
ViT
Template
Views
Geometric
Mapping
MLP
Image
Features
Attended
Features
x
y
z
NeMO Encoder
ViT
xN
DPT
Query Image
NeMO
Representation
Randomly-Sampled
Point Cloud
Figure 2. Overview of the NeMO approach. RGB template views are first processed by a ViT [13] and a multi-view encoder [27],
producing updated image features. To incorporate spatial information, a randomly-sampled point cloud is processed through a MLP and
attended to the image features in our proposed Geometric Mapping block, yielding feature-enhanced 3D points that form the NeMO. A
decoder attends a query image with the NeMO using multiple Cross- and Self-Attention blocks to generate multiple dense predictions.
We represent the NeMO as a point cloud and reduce the higher dimensional NeMO features to RGB using PCA [49]. The PCA reduction
shows a relation between the learned features and the objects geometry and semantics.
ther single-view poses for a model-based or a model-free
scenario and also performs pose tracking. The model-free
setup is similar to BundleSDF, whereas in the model-based
scenario templates are rendered based on the given model.
Analogously to detection and segmentation, model-based
pose estimation relies on a CAD model of the unseen target
object. GigaPose [44] relies on comparing and matching
the most similar RGB template to the query image. Mega-
Pose [29] applies a render-and-compare strategy to refine
the best reference rendering of the target object. Both meth-
ods rely on pre-training on a large-scale synthetic dataset
spanning 2 million images. OSOP [52] establishes dense
2D-3D correspondences based on a template match, simi-
lar to ZS6D [1], which also relies on a self-supervised pre-
trained ViT for feature extraction. SAM6D [33] predicts the
6D pose and the segmentation mask by applying a coarse-
to-fine 3D-3D correspondence matching strategy that builds
up on region proposal obtained from SAM and matched
with template renderings w.r.t. appearance, semantics and
geometry. Similarly, ZeroPose [9] also estimates the 6D
pose and the segmentation masks given the CAD model and
an RGB-D image as input. The method matchs the query
image with the feature embeddings derived from DINOv2
and prompted with the CAD model.
3. Method
In the following, we present our encoder-decoder network
as well as the proposed Neural Memory Object (NeMO)
representation. Figure 2 gives an overview of the complete
approach. Our core idea is to separate visual and geometric
object information from neural network weights, enabling
an object-agnostic approach that adapts to any demonstrated
target object without additional training. To this end, given
a set of RGB template images I = {Ii}K
i=1 of an object,
where K ≥2 and Ii ∈RH×W ×3 with H and W being
the image height and width, we aim to construct a unified,
object-centric representation without the need for intrinsic
or extrinsic camera parameters.
3.1. Network Architecture
NeMO Encoder. As Leap [27] has shown strong general-
izability to novel objects, we use a similar attention mech-
anism in our network. A Vision Transformer (ViT) [13] is
used to extract patch-wise image features from all template
images FI =

f I
i
	N
i=1 with f I
i ∈Rd and N = Hpatch ×
Wpatch × K where Hpatch, Wpatch are the number of verti-
cally and horizontally extracted patches respectively. Since
no prior information about the object coordinate frame or
the camera-to-object transformation is provided, we define
an anchor image IA. It serves as the object’s initial orienta-
tion in the NeMO space. During training, a random image
from I acts as anchor IA. As in [27], we employ a multi-
view encoder to generate updated image features bFI. This
part incorporates a strong bias towards the anchor image,
using cross- and self-attention blocks that facilitate interac-
tions between the anchor and non-anchor features.
Geometric Mapping. To enhance these features with the
3D geometry of the object oriented in the anchor coordinate
system, we first create Q = {qi}M
i=1, a set of sampled 3D
points qi ∈[−1, 1]3 with M ≥1. Next, we train a Multi-
Layer Perceptron (MLP) (see Supplementary Fig. 7) to act
as point encoder λ that maps each 3D point qi to a corre-
3

sponding feature vector FQ =

f Q
i
	M
i=1 = {λ (qi)}M
i=1
with f Q
i
∈Rd. We fuse the resulting set of point features
FQ and the updated image features bFI via our proposed
geometric mapping block. As shown in Fig. 3, the initial
point features FQ are updated with the information from
bFI through multiple Transformer Decoders:
bFQ =
n
bfi
QoM
i=1 = TransformerDec(FQ, bFI) ∈Rd,
(1)
where FQ acts as queries and bFI as key-value pairs. This
allows the initial 3D point cloud features to attend to object-
specific 2D image features. Besides the visual cues, we also
want to encode shape information of the object.
There-
fore, we jointly learn a Unsigned Distance Field (UDF)
U using an MLP that predicts the unsigned distance from
each point qi in the initial point cloud Q to its closest point
si ∈[−1, 1]3 on the estimated object surface S = {si}M
i=1
w.r.t. the coordinate system of IA. Note that |Q| = |S|.
Since bfi
Q depends on qi we can define the distance di and
direction vi as
di = U

bfi
Q (qi)

∈R
and
vi
=
dU

bfi
Q (qi)

dqi
∈R3,
(2)
such that si = qi −divi. Note that di is a scalar.
NeMO Representation. Given the M estimated surface
points from the set S and the processed 3D point fea-
tures bFQ, we define our NeMO representation as χ =
n
si, bfi
QoM
i=1.
Keeping si and bfi
Q separate, we can
transform the point cloud in NeMO space, allowing us to
free χ from the anchor coordinate system defined by IA.
This gives the advantage of being able to modify the ge-
ometric prediction of downstream tasks. For convenience,
we define the complete NeMO encoder (see Fig. 2) as a neu-
ral network Ψ such that χ = Ψ(I, Q). To fuse the informa-
tion of the predicted 3D points si and their corresponding
features we set efi
Q = bfi
Q + λ(si). Our continuous UDF
approach differs from Leap [27], which relies on a discrete,
fixed-size neural volume. We design our architecture to al-
low (i) variation in the number of points M, adjusting the
size and descriptiveness of χ, (ii) biased point sampling for
Q, such as surface points from a CAD model, (iii) transfor-
mations of the NeMO point cloud, allowing object modifi-
cations after the NeMO generation, and (iv) extension of an
existing NeMO with another set of NeMO points.
NeMO Decoder. When observing a query image Iq /∈I
the goal of our decoder Θ is to use the information stored
in a NeMO, which integrates information from all images
in I, to return a set of perception related dense predictions
for the query image. As a first step the query image features
Fq =

f q
i
	Hpatch×Wpatch
i=1
obtained by a ViT are updated by
Image   
Features
Transformer
Decoder
MLP
Geo. Features
Est. Surface
Points
NeMO
Representation
UDF
Geometric Mapping
Figure 3. Geometric Mapping Block. We fuse the updated im-
age features (key-value pairs) with the pre-processed geometric
features (queries) in multiple transformer decoder blocks. The
features are then forwarded to a UDF that estimates the unsigned
distance of the initial point cloud to the estimated object surface.
After further processing these points via a MLP, we combine them
with the updated geometric features, resulting in the NeMO.
the information stored in efi
Q using a combination of cross-
and self-attention layers as shown in Fig. 2. Inspired by the
recent advancements in camera pose estimation and dense
predictions [31, 59, 60], we use DPT [48] with multiple out-
put heads to upscale the updated query image features c
Fq to
multiple dense outputs Θ(χ, Iq) = (Pmodal, Pamodal, X, C),
where Pmodal ∈RH×W and Pamodal ∈RH×W are the pre-
dicted modal and amodal segmentation masks of the object,
X ∈RH×W×3 is the predicted dense pointmap between 2D
pixels in Iq and their corresponding 3D surface points in the
coordinate system defined by χ and C ∈RH×W is the as-
sociated learned confidence map to assess the 2D-3D map-
ping accuracy of X. After filtering the estimated pointmap
based on C, we utilize RANSAC [16] and Perspective-n-
Point (PnP) [20, 30] to estimate the pose of the object in the
query image.
3.2. Training
We train the encoder Ψ and decoder Θ jointly end-to-end
through multiple losses. During training, we have access
to a dataset of synthetically rendered RGB-D images with
ground truth object poses, amodal and modal masks and
camera intrinsic. Given a training sample of multiple im-
ages of the same object, we randomly select one image as
the anchor image IA and one as a query image Iq and use
the ground truth poses together with the masked depth and
camera intrinsic to build the ground truth surface points of
the initial orientation of the NeMO space.
Losses. To enforce the encoder to predict towards ground
truth surface points we define a simple regression loss that
minimizes the Euclidean distance between the estimated
surface point si = qi −eU(qi) × d e
U(qi)
dqi
and the ground truth
surface point ¯si:
Lχ =
PM
i=1∥si −¯si∥
M
,
(3)
with M = |χ|. We do not directly enforce any loss on
the NeMO features bFQ so that they can be freely learned
4

through backpropagation through the decoder. During train-
ing, after Ψ has predicted χ, we randomly rotate, translate
and scale the NeMO points si by a random transformation
T to teach the decoder to learn a coordinate system that is
independent of the anchor image IA. We define additional
losses on the dense predictions of the decoder Θ. For the
modal and amodal segmentation losses Lmodal and Lamodal
we use an equally weighted dice-loss [39] and binary cross-
entropy loss [26]. The pointmap loss L2D3D is a confidence
weighted L1 loss which uses the estimated confidence map
C to weight the loss between the estimated 2D-3D corre-
spondences X and the ground truth mapping ¯X. To learn
the confidence map C, we define a certainty loss Lcertain and
an uncertainty loss Luncertain:
Lcertain =
P
i∈DObj

1 −tanh
 exp(Ci)

|DObj|
Luncertain =
P
i∈DBg

tanh
 exp(Ci)

|DBg|
(4)
where DObj are the pixels belonging to the object and DBg
are all pixels belonging to the background. The total loss is
a weighted sum between all losses. More details about the
training in Supplementary Sec. 7.2.
4. Experiments
Synthetic Dataset. We create a new object-centric dataset
using BlenderProc [11] to generate Physically-Based Ren-
dering (PBR) images given the CAD object models as pro-
vided by a subset of Objaverse [10], GSO [14], and Om-
niObject3D [63], resulting in a total of 11077 different ob-
jects. We deem this necessary as there is – to the best of
our knowledge – no available dataset of comparable object
variety with sufficiently high but also balanced distribution
of views per object. For the following experiments, a single
network is trained on parts of Objaverse and all OmniOb-
ject3D models. We supervised the training by evaluating on
GSO objects, no fine-tuning or training on any of the objects
present in the BOP benchmark is performed. For additional
information we refer to Supplementary Sec. 7.1.
Implementation. We train our method on the aforemen-
tioned synthetic dataset and resize the respective, artificially
corrupted object bounding box crops to 224 × 224. We
train for 400k steps (roughly 2000 epochs) with a maxi-
mum learning rate of 1 × 10−4 on which we apply a linear
warm-up of 5000 steps followed by standard cosine anneal-
ing. The ViT backbone is trained with a separate maximum
learning rate of 1 × 10−5. As ViT we use DINOv2 [45] and
DPT [48] as regression head. Optimization is performed
using AdamW [36]. Training takes roughly 10 days on 16
A100 GPUs with an effective batch size of 128, whereas the
Method
HOPEv2
HANDAL
CNOS (SAM) - Static onboarding [43]
0.345
–
dounseen-SAM-CTL [18]
0.380
–
GFreeDet-FastSAM [35]
0.364
0.255
GFreeDet-SAM [35]
0.384
0.264
Ours
0.411
0.273
Table 1. Model-Free Detection. We compare AP on BOP test
splits of HOPEv2 and HANDAL against other methods published
on the public Model-Free Unseen Object 2D Detection leader-
board [5].
Method
Detections
HOPEv2
HANDAL
OPFormer†
CNOS [43]
0.335
0.204
Ours
CNOS [43]
0.307
–
Ours
GFreeDet-FastSAM [35]
0.329
0.213
Ours
NeMO
0.302
0.235
Table 2. Model-Free 6DoF Pose Estimation. We compare AP on
BOP test splits of HOPEv2 and HANDAL against other methods
published on the public Model-Free Unseen Object 6D Detection
leaderboard [6]. † indicates unpublished methods.
following experiments are run on a single A100 GPU.
Experimental Setup. We evaluate our method’s capability
to perform multiple few-shot perception tasks in a model-
free setting, i.e. no CAD model is given, and a model-
based setting, i.e. a CAD model is given only during in-
ference but not during training. Following the BOP chal-
lenge’s [25] dataset split we use the T-LESS [24], TUD-
L [23] and YCB-V [64] datasets for model-based evalua-
tion and the HOPEv2 [56] and HANDAL [19] datasets for
model-free evaluation. As metrics, we use Average Preci-
sion (AP) and Average Recall (AR) as defined in [25]. In
the model-free setup we use 32 randomly picked real RGB
templates from the static onboarding videos provided by
the BOP benchmark to generate the NeMO representation
while in the model-based setting 32 PBR rendered images
are used. The NeMO coordinate system is aligned with
the ground truth object coordinate system as described in
Supplementary Sec. 7.4 for evaluation purposes only. The
dense decoder outputs are used for detection, segmenta-
tion and 6DoF pose estimation as described in Supplemen-
tary Sec. 7.5. Note that in both settings, the network weights
are not changed, i.e. no finetuning is performed and the ob-
jects have never been seen during training. Additionally,
we extensively ablate the NeMO representation and its in-
fluence on the downstream tasks in Sec. 4.3 and show qual-
itative object surface reconstruction results of unknown ob-
jects in Sec. 4.1. The same network is used for all experi-
ments if not stated otherwise.
4.1. Model-Free Few-Shot Perception
Model-Free Detection.
Tab. 1 shows the AP results of
our model-free amodal detection on HOPEv2 and HAN-
5

Figure 4. Qualitative Example of Model-Free Few-Shot Detec-
tion and Pose Estimation on HOPEv2. Left shows the scene
without annotations, right shows NeMO detections in green and
pose estimations with refinement as rendered overlays. Even in
the underexposed scene the model predicts reasonable results.
DAL datasets compared to all other publicly listed results.
Our method achieves state-of-the art performance on both
datasets, outperforming the previous best method by 2.7pp
on HOPEv2 and 1.9pp on HANDAL. While all other meth-
ods rely on SAM [28, 66] segmentations of the scene for
their bounding box predictions, we are, to the best of our
knowledge, the first to use a single network to predict
amodal segmentations/detections in a model-free setting.
While SAM is able to give modal object segmentations, we
can predict amodal segmentations based on the NeMO rep-
resentation, which we use to create amodal bounding boxes.
Note that in the model-free category, the BOP benchmark
does only evaluate amodal detection, no segmentation.
Model-Free 6DoF Pose Estimation.
We evaluate our
model’s capability for 6DoF Pose Estimation in a model-
free setting on the HOPEv2 and HANDAL datasets using
different detections in Tab. 2. On HOPEv2 we use ICP [7]
between our estimated object surface and the depth infor-
mation while no refinement is used on HANDAL, since
no depth data is available.
An example can be seen in
Fig. 4. Compared to the only other method OPFormer we
achieve state-of-the-art results on HANDAL when using
NeMO detections while being on par when using GFreeDet-
FastSAM detections. When using the default CNOS de-
tection as provided by the BOP benchmark, we are 2.8pp
behind OPFormer. Note that as of the time of writing, no
default CNOS detections for HANDAL are available any-
more. Although our detections outperform previous meth-
ods on HOPEv2 they do not provide an AP gain for pose es-
timation. We hypothesize that the detection improvements
come from our models ability to predict amodal bounding
boxes, which is beneficial for the detection evaluation but
might not always be an improvement for pose estimation.
Model-Free Object Reconstruction.
We qualitatively
show object surface reconstruction on random objects in
different scenarios in Fig. 5.
4.2. Model-Based Few-Shot Perception
This section discusses results of our network on model-
based perception.
Although the network has never been
trained on rendered template images with black background
and real query images, it performs on par and partially out-
Method
T-LESS
TUD-L
YCB-V
CNOS(Fast Sam) [43]
0.395
0.534
0.568
SAM6D-FastSAM [33]
0.417
0.546
0.573
SAM6D [33]
0.458
0.573
0.589
F3Dt2D†
0.482
0.573
0.666
MUSE†
0.467
0.590
0.674
anonymity†
0.477
0.593
0.685
NIDS Net [37]
0.493
0.486
0.621
Ours
0.183
0.623
0.602
Table 3. Model-Based Detection. We compare AP on BOP test
splits of T-LESS, TUD-L and YCB-V against other methods pub-
lished on the public Model-Based Unseen Object 2D Detection
leaderboard [2]. † indicates unpublished methods.
Method
T-LESS
TUD-L
YCB-V
CNOS(Fast Sam) [43]
0.374
0.480
0.599
SAM6D-FastSAM [33]
0.420
0.517
0.621
SAM6D [33]
0.451
0.569
0.605
NOCTIS [17]
0.479
0.583
0.684
LDSeg†
0.488
0.587
0.647
MUSE†
0.451
0.565
0.672
Prisma-MPG + SG†
0.454
0.590
0.607
anonymity†
0.464
0.569
0.688
NIDS Net [37]
0.496
0.556
0.650
Ours
0.169
0.488
0.579
Table 4. Model-Based Segmentation. We compare AP on BOP
test splits of T-LESS, TUD-L and YCB-V against other methods
published on the public Model-Based Unseen Object 2D Segmen-
tation leaderboard [4]. † indicates unpublished methods.
Method
Detections
T-LESS
TUD-L
YCB-V
Ours
NeMO
0.082
0.466
0.493
Ours
ground truth
0.295
0.538
0.566
ZS6D [1]
CNOS [43]
0.210
–
0.324
MegaPose [29]
CNOS [43]
0.177
0.258
0.281
GenFlow [40]
CNOS [43]
0.215
0.300
0.277
GigaPose [44]
CNOS [43]
0.264
0.300
0.278
FoundPose [46]
CNOS [43]
0.338
0.469
0.452
Co-op [41]
CNOS [43]
0.592
0.642
0.626
Ours
CNOS [43]
0.190
0.476
0.504
Table 5. Model-Based 6D Localization of Unseen Objects with-
out Refinement.
We report Average Recall (AR) on 3 BOP
datasets and compare with current SOTA model-based RGB based
pose estimation methods without refinement. We use the default
CNOS [43] detections provided by the BOP challenge when indi-
cated. Data taken from [41].
performs previous methods specialized on this task.
Model-Based Detection. AP for amodal detection on T-
LESS, TUD-L and YCB-V datasets is reported in Tab. 3.
We achieve state-of-the-art results on TUD-L, outperform-
6

Figure 5. Object Surface Reconstruction and Camera Pose Estimation on Unseen Objects. We show object surface points and camera
poses as predicted by the decoder based on four images of randomly chosen objects in different scenarios: (Left) A static coffee machine
standing on a table, captured by a dynamic camera. (Middle) A label machine in different environments, including occlusions. (Right) An
espresso mug manipulated in hand, captured by a static camera. In all three scenarios, our model is able to predict object-centric camera
poses and surface points. We map RGB pixel color to corresponding 3D point to show correct 2D-3D mapping. Blue is the anchor image.
ing the previous best method by 3pp. On YCB-V we would
rank 6th out of 13 methods reported on the BOP leader-
board. On T-LESS we achieve a precision of 0.183, which
is probably due to the strong similarities between the ob-
jects, the lack of texture as well as the dataset containing
many cluttered scenes with objects of the same instance.
All these factors are not present in our training data.
Model-Based Segmentation. In addition to detection we
also report object segmentation AP on the three datasets.
We report the results in Tab. 4. Compared to other networks
our method is less precise on pixel level, which could be
due to border artifacts as a results of patch-scaling. As with
the detection results, the precision on T-LESS is low, which
we attribute to the same reasons as mentioned above.
Model-Based Refiner-Free 6DoF Pose Estimation.
To
evaluate the 6DoF Pose Estimation capabilities of our
method independent of the detection quality we report the
average recall AR on T-LESS, TUD-L and YCB-V with
default CNOS [43] detection and without additional re-
finement step in Tab. 5 as is standard practice in the lit-
erature. Although our network was not designed for the
model-based category we achieve high results in TUD-L
and YCB-V, while only Co-op [41] achieves better results.
Our method fails to handle the symmetric and textureless
objects in T-LESS, which is reflected on the low average
recall of 0.190. In addition we report the results of our
method using ground truth and NeMO detections. Surpris-
ingly, although the NeMO detections show higher precision
than the CNOS detections as reported in Tab. 3, the average
recall on pose estimation task is lower when using NeMO
detections. This is due to the difference in how average
precision and average recall are evaluated. For results on
model-based pose estimation with refinement we refer to
Supplementary Tab. 9.
# NeMO Points
AP ↑
APMSPD ↑
APMSSD ↑
Time/Image (s) ↓
10
0.004
0.005
0.002
0.402
50
0.112
0.130
0.094
0.671
100
0.214
0.231
0.196
0.460
200
0.290
0.297
0.282
0.393
500
0.380
0.379
0.381
0.340
1000
0.383
0.389
0.376
0.320
1500
0.378
0.37
0.385
0.336
Table 6. Number of NeMO Input Points vs AP. We report the
AP on YCB-V 6D pose estimation with ground truth detections
by varying number of randomly sampled input points.
4.3. Analyzing the NeMO Representation
In this section we analyze the properties of the NeMO rep-
resentation. For all experiments we report AP of 6DoF pose
estimation on the test split of YCB-V with real template im-
ages randomly chosen from the public real training set pro-
vided by the BOP benchmark [25]. We use ground truth
detections and no refinement unless stated otherwise. A
smaller decoder that only outputs pointmap and confidence
is used in this section. Additional experiments and analyses
can be found in Supplementary Sec. 7.6.
Varying number of template images. We show the rela-
tion between the number of template images used to gener-
ate a NeMO and its influence on 6DoF pose estimation and
the required memory footprint in Fig. 6. It shows that while
more template images enhance precision, acceptable perfor-
mance is already achieved with just three views. We empha-
size the fact that whereas the number of templates increases,
the runtime and memory consumption of our decoder model
stays quasi constant while the precision increases. Since we
generate our NeMOs before the inference task, we shift the
computational heavy part of attending all template features
with each other to the offline phase.
Varying number of NeMO Points. Although the model
was trained on a fixed sized number of input points Q we
show in Tab. 6 that the encoder and decoder can adapt to
7

3
4
8
16
32
Number of Template Images
3000
4000
5000
6000
7000
8000
Memory Consumption [MB]
3520
3336
3562
3336
4030
3336
4991
3336
6916
3336
0.26
0.28
0.30
0.32
0.34
0.36
0.38
0.40
Average Precision [AP]
0.300
0.298
0.352
0.354
0.378
Encoder Memory
Decoder Memory
AP
Figure 6. Memory consumption and AP on 6DoF Pose Estima-
tion vs. Number of template images. While the offline NeMO
generation requires more memory as the number of template im-
ages increases, the inference memory remains constant. Addition-
ally, more template images increase the AP on 6DoF Pose Estima-
tion on YCB-V with ground truth detections.
different point cloud sizes, allowing for a dynamic adapta-
tion of the models memory consumption and precision. We
observe that the precision increases as the number of NeMO
points increases up to 500 where it stagnates around 0.38.
A visualization of NeMO features can be seen in Fig. 2.
Transforming NeMO coordinate system. During train-
ing, we randomly transform our NeMO point cloud before
passing it to the decoder.
In this section we evaluate if
the decoder adapts its output based on the positions of the
NeMO points. To test the rotation-equivariance between the
NeMO point cloud and the predicted pointmaps X of the
template images, we rotate the NeMO point cloud around
the z-axis in 10 degree steps and evaluate the Chamfer dis-
tance [15] between the predicted pointmap and the ground
truth CAD model rotated by the same angle. As can be
seen in Supplementary Fig. 11, the Chamfer distance [15]
between the pointmaps and the ground truth CAD model
remains low while the Chamfer distance between a non-
rotating pointmap varies, indicating that the pointmap pre-
diction is rotating accordingly. This is an interesting prop-
erty of our network for future work, in which parts of the
NeMO could be transformed online during inference.
Extending NeMO. As the NeMO representation is based
on a set of points, it can easily be extended. To see if the ad-
dition of new points from a different NeMO of the same ob-
ject leads to better results, we combine two NeMOs with the
same anchor image. In Supplementary Tab. 10 we show the
Chamfer distance between the predicted pointmap and the
ground truth CAD model for the original NeMO and the ex-
tended one, observing that the Chamfer distance is lower for
the extended NeMO than for the original one. This shows
that we can combine two sets of NeMO points without dis-
turbing the decoder while improving its performance. This
is beneficial in scenarios, where the hardware is memory
limited and thus, fewer template images can be used.
5. Limitations
Despite promising results, our method exhibits several lim-
itations. The encoder is trained to predict surface points,
which leads to difficulties with symmetric objects, as
demonstrated on the T-LESS dataset. We attribute this not
to the pointmap representation itself, but to limitations in
the current training procedure; addressing this will require
further research. Additionally, the encoder performs poorly
on highly textureless objects, likely due to their underrep-
resentation in the training set. Future work will focus on
scaling the dataset to include a broader and more diverse
set of objects. Another limitation is that the encoder does
not directly predict bounding boxes; instead, segmentation
masks are used as a proxy. This can result in merged bound-
ing boxes when multiple instances of the same object are
present—an issue we plan to resolve in future work.
6. Conclusion
In this work, we presented an encoder and decoder archi-
tecture for Neural Memory Object (NeMO), a general and
versatile object-centric representation that can be used for
few-shot perception tasks such as object detection, segmen-
tation and pose estimation using only an unordered set of
RGB images. Through thorough experiments, we demon-
strated that our method can be used for few-shot, model-free
and model-based unseen perception task, partially outper-
forming state-of-the-art methods. Our approach differs to
alternative methods by (i) being capable to incorporate in-
formation from multiple RGB recorded images into a single
representation without any camera parameters, (ii) utilizing
a single network for multiple perception tasks, (iii) hav-
ing constant inference time and memory requirements re-
gardless of the number of template images, (iv) predicting
amodal segmentation masks without CAD-model, (v) be-
ing able to incorporate CAD-model information if given,
(vi) allowing dynamically changing memory usage and pre-
cision, based on hardware capabilities, (vii) outsourcing
the object’s information from the network weights, allow-
ing for quick adaptation to novel objects without retraining.
Furthermore, we contribute a realistic large-scale and bal-
anced object-centric dataset that we deem beneficial for the
broader research community. Future work includes combin-
ing multiple NeMOs for articulated objects. Additionally,
we strive to increase robustness of the applied task-specific
encoder w.r.t. symmetrical and textureless objects as mo-
tivated by the results on the T-LESS dataset. We hope this
work stimulates discussion on decoupling object knowledge
from model weights and helps advance model-free few-shot
perception for unseen objects.
8

References
[1] Philipp Ausserlechner, David Haberger, Stefan Thalhammer,
Jean-Baptiste Weibel, and Markus Vincze. ZS6D: Zero-shot
6D Object Pose Estimation using Vision Transformers. In
2024 IEEE International Conference on Robotics and Au-
tomation (ICRA), pages 463–469, 2024. 3, 6
[2] BOP Authors.
Model-based unseen object 2d detection
leaderboard bop-classic. https://bop.felk.cvut.
cz/leaderboards/detection-unseen-bop23/
bop-classic-core/, . Accessed: 2025-07-30. 6
[3] BOP Authors.
Model-based unseen object 6d detection
leaderboard bop-classic. https://bop.felk.cvut.
cz/leaderboards/pose- detection- unseen-
bop24/bop-classic-core/, . Accessed: 2025-07-
30. 14
[4] BOP Authors. Model-based unseen object 2d segmentation
leaderboard bop-classic. https://bop.felk.cvut.
cz/leaderboards/detection-unseen-bop23/
bop-classic-core/, . Accessed: 2025-07-30. 6
[5] BOP Authors.
Model-free unseen object 2d detection
leaderboard bop-h3.
https://bop.felk.cvut.
cz / leaderboards / modelfree - detection -
unseen-bop24/bop-h3/, . Accessed: 2025-07-30. 5
[6] BOP Authors.
Model-free unseen object 6d detection
leaderboard bop-h3. https://bop.felk.cvut.cz/
leaderboards / modelfree - pose - detection -
unseen-bop24/bop-h3/, . Accessed: 2025-07-30. 5
[7] P.J. Besl and Neil D. McKay. A method for registration of
3-d shapes. IEEE Transactions on Pattern Analysis and Ma-
chine Intelligence, 14(2):239–256, 1992. 6
[8] Andrea Caraffa, Davide Boscaini, Amir Hamza, and Fabio
Poiesi. Freeze: Training-free zero-shot 6d pose estimation
with geometric and vision foundation models. In European
Conference on Computer Vision (ECCV), 2024. 14
[9] Jianqiu Chen, Zikun Zhou, Mingshan Sun, Rui Zhao, Li-
wei Wu, Tianpeng Bao, and Zhenyu He. Zeropose: Cad-
prompted zero-shot object 6d pose estimation in cluttered
scenes. IEEE Transactions on Circuits and Systems for Video
Technology, 35(2):1251–1264, 2025. 3
[10] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi.
Obja-
verse: A universe of annotated 3d objects. arXiv preprint
arXiv:2212.08051, 2022. 5
[11] Maximilian Denninger, Dominik Winkelbauer, Martin Sun-
dermeyer, Wout Boerdijk, Markus Knauer, Klaus H. Strobl,
Matthias Humt, and Rudolph Triebel.
Blend