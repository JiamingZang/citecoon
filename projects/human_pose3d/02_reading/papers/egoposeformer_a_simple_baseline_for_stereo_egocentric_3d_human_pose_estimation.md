# EgoPoseFormer: A Simple Baseline for Stereo Egocentric 3D Human Pose Estimation

> 2024 · id: W4404720255 · arXiv: 2403.18080 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
In the rapid expansion of Virtual Reality (VR) and Augmented Reality (AR)
technologies [1, 2], the capability to accurately interpret and emulate human
actions becomes increasingly crucial. Central to this pursuit is the egocentric pose
estimation task [3,11,12,14,22,26,31,32,35–37,41,45], which aims to estimate
the 3D body pose from a vantage point inherent to the user, predominantly from
head-mounted cameras. Its precision is pivotal in a wide range of applications,
such as gaming and virtual meetings, making it essential for crafting an immersive
user experience for the next generation of VR/AR systems.
⋆Work done when working as a research scientist intern at Meta Reality Labs
arXiv:2403.18080v2  [cs.CV]  15 Aug 2024

2
C. Yang et al.
3D Body Pose
Auto
Encoder
Convs
2D Joint Heatmaps
3D Feature Volume
3D Heatmap
3D
Convs
Project
3D Pose Proposal
3D Refine Offset
Refine
PRFormer
PPN
Image Features
(a) Heatmap-based Methods
(b) 3D Volume-based Methods 
(c) EgoPoseFormer (Ours)
Fig. 1: Illustration of different egocentric pose estimation methods. While previous
approaches predict joints’ locations via 2D heatmaps or 3D feature voxels, EgoPose-
Former first estimates the coarse locations of each joint using a Pose Proposal Net-
work (PPN) and uses a transformer to refine the estimated pose.
Different from outside-in pose estimation, where the human bodies are well
covered by the images, a key challenge of egocentric pose estimation is the joint
invisibility problem, which usually results from two causes. First, the limited
field of view (FOV) of head-mounted cameras cannot fully capture the human
body [3,32], especially when hands and legs are stretched out. Another cause
arises from the self-occlusion of different body parts [32], especially the lower
body, which is very prone to be occluded by the main trunk. To overcome this
limitation, some recent works [3,32,45] directly regress the 3D joint locations
using the 2D heatmaps by employing an auto-encoder style architecture, which
allows the locations of invisible joints to be inferred from the global information
and other visible joints’ locations. These approaches, however, take 2D heatmaps
as input and cannot leverage the rich appearance information of the input image,
which limits its 3D regression capability. This also causes a poor scaling-up
ability to the developed model, i.e., even if the model is equipped with a larger
backbone network, its pose estimation accuracy cannot be improved [3]. In another
recent work, SceneEgo [37], a 3D feature voxel grid was first built using fish-eye
projection [28] with the help of depth and semantic maps, and 3D convolution
was employed to operate on the feature grid to estimate each joint’s location
in a monocular setting. Despite the fact that 3D joints outside the FOV can
be estimated by leveraging a sufficiently large voxel grid, 3D convolutions are
computationally expensive and the cost increases with the size of the voxel grid.
In this work, we propose EgoPoseFormer, a simple transformer-based model
for multi-view egocentric pose estimation. As illustrated in Fig. 1, our method
uses a two-stage framework [25] to overcome the joint invisibility challenge, and
the body pose is predicted in a coarse-to-fine manner [24, 38, 39]. Specifically,
the first stage of our model is the Pose Proposal Network (PPN), which is a
simple 2-layer MLP that leverages the global information of the input multi-view
feature maps to predict a joint’s coarse location. Similar to the recent heatmap-
based methods [3,32], the usage of global features allows our method to reason
about the locations of all joints, including the invisible ones. Surprisingly, we
found, with proper training settings, this simple MLP can already outperform
previous state-of-the-art methods in stereo inputs are available. Then, in the
second stage, we employed a DETR-style [4] transformer, Pose Refinement
Transformer (PRFormer), to predict 3D refinement offsets related to the first stage

EgoPoseFormer
3
estimations by exploiting the multi-view stereo features and human kinematic
information. Specifically, we embed each joint’s location and identity information
into a Joint Query Token (JQT). Each JQT interacts with the multi-view
image features and other JQTs through attention operations in each layer of
PRFormer; subsequently, the refinement offsets are predicted from updated JQTs.
Furthermore, we design a new Deformable Stereo Attention [48] to effectively
process the fine-grained multi-view stereo features, which allows us to accurately
estimate a joint’s 3D location. In summary, we make three contributions:
– We propose EgoPoseFormer, a simple transformer-based model for stereo
egocentric pose estimation. Our model composes an MLP-based pose proposal
network for computing coarse joint locations, which already demonstrates a
strong accuracy, and a transformer-based pose refinement network to further
improve the localization accuracy.
– Our method achieves state-of-the-art on the stereo UnrealEgo dataset [3] by
a huge advantage over previous arts with much lower computation costs.
– We demonstrate that our method can be easily extended to the monocular
egocentric pose estimation problem and achieve state-of-the-art performance
on the SceneEgo dataset [37].
2

## method
In this section, we introduce the proposed EgoPoseFormer in detail. We first
discuss the motivation of our two-stage framework in Sec. 3.1. We present the Pose
Proposal Network inSec. 3.2 and the Pose Refinement Transformer in Sec. 3.3.
Later, we introduce the loss function in Sec. 3.4 and the feature extractor
in Sec. 3.5. An overview of our model is illustrated in Fig. 2 (a).
3.1
Two-stage Pose Estimator
Fig. 2 (a) shows the proposed two-stage framework, which is designed to overcome
the joint invisibility challenge caused by self-occlusion or the limited FOV of
head-mounted cameras. In the first stage, we estimate the coarse location of each
joint, which we call pose proposal [25], by utilizing the global feature pooled
from the stereo features. This design enables the network to roughly localize all
joints, including the invisible ones, by jointly reasoning visual clues from visible
joints and background scenes. The global feature, however, does not preserve
fine-grained local details that enable more accurate joint localization. Motivated
by this observation, we added a second stage to refine the pose proposal, where
we use a transformer that exploits fine-grained stereo features and the body
kinematic information through the attention mechanism. Finally, the refined
poses are output as the final pose estimation results.
3.2
Pose Proposal Network (PPN)
Given the multi-view image features F ∈RV ×H×W ×C, the PPN computes
the pose proposal P 0 ∈RNj×3 using global information. Here V denotes the
number of views; H, W and C denotes the height, width and channel number
of the image feature maps; Nj denotes the number of body joints. Specifically,
PPN begins by applying a global average pooling to the feature maps of each view,
then the resultant averaged features are concatenated to form a unified feature
representation, which captures the salient global features and stereo information

6
C. Yang et al.
Concat 
& 
Linear
JQTs
Pose Proposal
Multi-view Features
Deformable
Attention
(a) 2D Heatmap Pre-training
Monocular Image
Feature Encoder
Monocular Features
Joint Heatmap
Heatmap
Predictor
Multi-view Images
Feature Encoder
Pose
Estimator
Multi-view Features
Predicted Pose
(b) 3D Pose Estimation
Fig. 3: Left: An illustration of our Deformable Stereo Attention. The 3D joints are
first projected to each view plane using camera parameters. Within each view, we
compute 2D deformable attention by querying the image features with the JQTs with
the projected points serving as reference points. Finally, the attention results for each
view are concatenated and fed into a linear layer to be projected into the original
dimension. Right: (a) The feature extractor is first pre-trained to predict 2D joint
heatmaps using monocular images. (b) The multi-view feature maps are computed
using the pre-trained feature extractor.
across views. We then predict each joint’s initial 3D location with a 2-layer MLP
with GELU activation:
  P ^0 = \
m
athrm {ML
P
}_{\mathrm {ppn}}\Bigl (\mathrm {Concat}_{\{k\}}\bigl (\mathrm {AvgPool}(F_k)\bigr )\Bigr )
(1)
where k ∈{1, ..., V }. As we will show in Sec. 4.3, this simple design can already
provide reasonable location estimation for each joint.
3.3
Pose Refinement Transformer (PRFormer)
The PRFormer takes the multi-view image features F ∈RV ×H×W ×C and the
pose proposal P 0 ∈RNj×3 as inputs. Structurally, the transformer comprises
S layers. At each layer s, a refinement offset ∆P s ∈RNj×3 relative to P 0 is
predicted, and the pose estimation is computed by adding the offset to the pose
proposals P s = ∆P s + P 0 where s = {1, ..., S}. During the inference phase, the
final layer’s output, P S, is used as the model’s final prediction.
Joint Query Tokens. Inspired by DETR [4], in PRFormer every joint is
characterized by a unique Joint Query Token (JQT). The JQTs will serve as the
queries in our transformer to interact with each other and the multi-view image
features through attention mechanisms. We compute the JQTs by embedding
each joint’s identity and initial location information with a Query Generation
MLP. Specifically, for the j-th joint, its JQT Qj ∈RC is computed by feeding its
initial location P 0
j = (x0
j, y0
j , z0
j ) and a scalar identifier σj into the MLP:
  Q _j= \m
a
thr m 
{M LP
} _ {\
m
athrm {JQT}}\bigl (\sigma _j, x^0_j, y^0_j, z^0_j\bigr )
(2)

EgoPoseFormer
7
In practice, we simply use the joint index j to serve as the scalar identifier σj.
PRFormer Layer. As shown in Fig. 2 (b), each layer of the PRFormer is a
transformer decoder layer [34]. Each input JQT undergoes a cross-attention op-
eration to interact with the fine-grained stereo image features, and a subsequent
self-attention operation to extract spatial and human kinematic information
from other JQTs. Note that here we follow [5] to put the cross-attention opera-
tion before the self-attention operation. Finally, we use a Feed-forward Network
(FFN) [34] to non-linearly transform the JQTs. A distinct characteristic of the
PRFormer layer vis-à-vis traditional transformer decoder layers is our replace-
ment of the conventional cross-attention with Deformable Stereo Attention, which
enables our transformer to effectively reason about multi-view stereo and thus
can accurately locate the joints in the 3D world.
Deformable Stereo Attention. As shown in Fig. 3 Left, the proposed De-
formable Stereo Attention has three steps. Firstly, we project each body joint’s
initial 3D location onto each view plane by leveraging the camera parameters [28].
Specifically, for joint j and its initial 3D location P 0
j , we compute its 2D location
on each view { ˜P k
j } = {(˜xk
j , ˜yk
j )} where k = {1, ..., V }. Secondly, with the joints’
projected 2D locations as reference points, we independently apply deformable
attention [48] in each view to let the JQTs extract useful information from image
features:
  
Z ^ k_j = \mat
h
rm {D e
f o rm
Attn}\bigl (Q_j, \tilde {P}^k_j, F_k\bigr )
(3)
where Qj is the j-th JQT and Fk is the image feature of the k-th view. Notably, if
a joint is out of the view’s FOV, we simply fill the computed result Zk
j with zeros
to make this information explicit to the model. Finally, the computed results
{Zk
j } from each view are concatenated and fed into a linear projection layer
to fuse multi-view stereo information and transform the result to the original
dimension:
  Z _j = \
m
athrm {Linea
r}\bigl ( \mathrm {Concat}_{\{k\}}(Z^k_j)\bigr )
(4)
In this way, the Deformable Stereo Attention becomes an atomic attention oper-
ation and can replace the normal cross-attention operation in a plug-and-play
manner.
Predicting Refinement Offsets. Each PRFormer layer empowers a JQT to
engage with the multi-view image features and other JQTs to exploit stereo and
human kinematic information, imbuing it with joint-specific knowledge. Based
on this enriched representation, we use a shallow MLP to predict the refinement
offset relative to the pose proposal. Specifically, the refinement offset for joint j
at the s-th PRFormer layer is computed as:
  \
D e lta 
P^s_j= \m
athrm {MLP}_{\mathrm {offset}}^s(Q^s_j )
(5)

8
C. Yang et al.
3.4
Loss Function
To keep our method concise, we employ a simple per-joint error loss [3] to train
our model. Specifically, the loss function is formulated as the following:
  
\
m
ath
ca
l
 {L
} = 
\ s um _{s=0}^{S} \sum _{j=1}^{N_j}{||P^{s}_{j}-\hat {P}_j||_{2}}
(6)
Here, different loss stages are indexed with s. Specifically, s = 0 corresponds
to the pose proposals generated by PPN, and s > 0 represents the subsequent
refinement predictions derived from the transformer layers. Nj is the total number
of joints; P ∗
j and ˆPj are the predicted and ground-truth 3D coordinates of the
j-th joint.
3.5
Feature Extractor
We follow UnrealEgo [3] to adopt a UNet [27] architecture as our visual feature
extractor to compute the multi-view image features. The key difference in our
method is the exclusion of multi-view concatenation in the decoder part [3]. As
the PRFormer can effectively process multi-view stereo features, we can extract
features from each view independently instead of aggregating multi-view features
within the feature extractor. As we show in Sec. 4.2, this modification significantly
reduces the model size and improves its computational efficiency.
Inspired by the heatmap-based methods [3,32], we further enhance our model’s
efficacy by pre-training its feature extractor to predict 2D joint heatmaps. As
shown in Fig. 3 right, during the pre-training phase, the model predicts 2D joint
heatmaps for each view using a light-weighted fully convolutional head [3], which
is removed after pre-training.
4

## experiments
4.1
Experiment Settings
Dataset Settings. We use the multi-view UnrealEgo [3] dataset to benchmark
our proposed method. The UnrealEgo dataset has 451k synthetic stereo views
collected using 30 different actions, which are captured by two head-mounted
cameras placed 1cm from the head. We follow the official dataset splits: the model
is trained using the training set (357k views) and evaluated using the test set (48k
views); The validation set (46k views) is used for tuning the hyperparameters.
We also test our method using the monocular SceneEgo [37] dataset to test its
generalization ability to monocular settings. SceneEgo is a real-human dataset
recorded by two actors with different daily actions. It has a total 28k images.
For both datsets, we report the Mean Per Joint Position Error (MPJPE) and
Procrustes Analysis MPJPE (PA-MPJPE) in millimeters as evaluation metrics
following their official papers [3,37].

EgoPoseFormer
9
Table 1: Comparison of MPJPE (PA-MPJPE) between our method and previous
state-of-the-art approaches on the UnrealEgo dataset.

## related_work
Egocentric Pose Estimation. Prior to our work, there have been several works
on egocentric pose estimation, which cover both monocular and stereo settings.
Most previous approaches are based on predicting 3D joints locations from 2D
heatmaps. For example, Mo2Cap2 [41] first predicts the 2D joint heatmaps and
their corresponding depth, and the 3D coordinates are computed with fish-eye
unprojection. In xR-EgoPose [32], the 3D joint coordinates are directly estimated
with an auto-encoder, whose input is the predicted joint heatmap. This allows it
to tackle the joint invisibility difficulty. SelfPose [31] improves xR-EgoPose by
introducing joint rotation loss and UNet [27] backbone. EgoSTAN [22] improves
the quality of visual features by introducing temporal modeling. EgoGlass [45]
extends the monocular heatmap-based methods to multi-view settings, where the
joints’ locations are estimated from the multi-view joint heatmaps. It also adds an
auxiliary segmentation loss to improve accuracy further. EgoPW [36] explores ex-
tending existing pose estimation methods to estimate body poses in a global space.
UnrealEgo [3] further improves EgoGlass by introducing cross-view information
exchange in the UNet decoder. The recently proposed Ego3DPose [12] improves
UnrealEgo by explicitly modeling limb heatmaps and orientations. Apart from
heatmap-based approaches, SceneEgo [37] was recently introduced to directly
predict joint locations by running 3D convolution on 3D feature voxels with the
help of scene depth and segmentation. There are also works [14, 21, 43] about
egocentric pose hallucination, where the headset wearer’s body pose is estimated
with front-facing cameras, in which the body is rarely observed. Different from
ours, the focus of those works is generating body poses that are harmonious with

4
C. Yang et al.
Deformable 
Stereo Attention
Add & Norm
Add & Norm
Self Attention
Add & Norm
FFN
Refinement
Prediction
PRFormer
Layer
Refinement
Prediction
JQT
Generation
PRFormer
Layer
PPN
(a) Architecture of EgoPoseFormer
(b) PRFormer Layer
Multi-view
features
JQTs
Updated JQTs
Further Updated JQTs
Pose
Proposal
Refined
Pose
Further
Refined Pose
MLP
Pool & Concat
Prediction
Ground-truth
Fig. 2: (a) An overview of the proposed EgoPoseFormer. The input of EgoPoseFormer is
the multi-view image features. In the first stage: Pose Proposal Network (PPN), the
multi-view features are globally pooled and concatenated, from which an MLP is
used to estimate the coarse location of each joint (pose proposal). Then the joints’
identity and location information are embedded into Joint Query Tokens (JQTs) to
serve as the queries in the second stage Pose Refinement Transformer (PRFormer).
In PRFormer, a JQT iteratively interacts with the stereo features and other JQTs
to update itself through the attention mechanism. The updated JQTs are used to
predict refinement offsets related to the pose proposal, yielding more accurate pose
estimations. (b) The architecture of PRFormer layer is similar to the transformer
decoder layer, which includes a cross-attention block, a self-attention block, and a
feed-forward network (FFN). However, in PRFormer, the cross-attention is replaced by
the proposed Deformable Stereo Attention to better exploit stereo visual features.
the background scene.
Transformer for Outside-in Pose Estimation. There have been lots of
successful attempts to apply transformers for the outside-in body pose esti-
mation. One line of work [8,16,29,44,46,47] aims to develop high-performing
transformer-based backbone networks for outside-in pose estimation. For example,
PoseFormer [47] introduces spatial and temporal attention mechanisms to gener-
ate high-quality visual features for 3D pose estimation, which is improved by the
followed-up PoseFormer v2 [46] by introducing frequency modelling. Furthermore,
improvements in model efficiency are also elaborated in [8,15]. The other line of
work, similar to ours, focuses on developing DETR-style [6,18,23,30,40,42] sparse
transformers for human body pose estimation. For example, PETR [30] introduces
inter-instance and intra-instance attention for accurate 2D multi-person pose
estimation. PSVT [23] introduces spatial-temporal encoder and decoder for 3D
pose and body shape estimation. On the other hand, transformer architecture is
also used for 2D human pose estimation [19,40]. For example, GroupPose [19]
uses keypoint and instance queries to directly estimate the 2D human poses in

EgoPoseFormer
5
a multi-person setting. Despite the huge success of transformer architectures
in outside-in pose estimation, applying it to egocentric settings requires non-
trivial adaptation due of the intrinsic difference between the two problems. For
instance, in outside-in pose estimation, the human body usually lies within the
camera’s FOV, while the out-of-FOV problems usually happen in egocentric pose
estimation. Another difference is the input of those two tasks: most outside-in
pose estimation models take regular images captured by pin-hole cameras as
input [10,17], while in egocentric settings the images are usually captured by
fish-eye cameras [3,37,45] to expand FOV, causing image distortions and posing
further difficulties to the task.
3

## conclusion
We introduce EgoPoseFormer, a new transformer-based egocentric pose estimation
method. Our two-stage method first infers each joint’s coarse from the global
features, then uses a DETR-style transformer Pose Refinement Transformer to
refine the coarse locations by exploiting fine-grained stereo features and human
kinematic information. Furthermore, we design Deformable Stereo Attention to
better exploit the multi-view stereo information. Our method achieves state-of-
the-art with significant advantages over previous arts on two pose estimation
datasets, including stereo and monocular settings. We hope our model can serve
as a strong baseline approach for future research in this field.

EgoPoseFormer
15