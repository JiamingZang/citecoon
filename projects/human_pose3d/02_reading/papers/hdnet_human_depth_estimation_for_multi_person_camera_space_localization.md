# HDNet: Human Depth Estimation for Multi-person Camera-Space Localization

> 2020 · id: W3117675859 · arXiv: 2007.08943 · pdf: https://arxiv.org/pdf/2007.08943 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Given a 2D image with an unknown number of persons, the task of camera-
space multi-person 3D pose estimation is to: (1) identify all person instances,

HDNet: Human Depth Estimation
5
0
0
0.2
0.7
0.1
Fpose
Fdepth
Backbone
Integral
Integral
...
GNN Layers
GNN Layers
Avg
Avg
Feature Extraction
Pose Estimation
Depth Estimation
...
Masks
Pose
Feature extraction
Pose
Feature extraction
Depth
Feature extraction
Depth
Feature extraction
fc
fc
Fig. 2. Our HDNet architecture. The framework takes an image together with the
bounding box of a target person as input. A Feature Pyramid Network backbone is
used for general feature extraction followed by separated multi-scale feature extraction
for the tasks of pose and depth estimation. Estimated heatmaps are used as attention
masks to pool depth features. A Graph Neural Network is utilized to propagate and
aggregate features for the target person depth estimation.
(2) estimate the 3D pose with respect to the root joint, i.e., pelvis, for each
person, and (3) localize each person by estimating the 3D coordinate of root
joint in the camera coordinate space.
Following the top-down approaches in the literature of multi-person pose
estimation, we assume that the 2D human bounding boxes for each person in
the input image are available from a generic object detector. Given the person
instances and detected bounding boxes, we propose an end-to-end depth estima-
tion framework to localize the root joint of each person in the camera coordinate
space as illustrated in Figure 2. The root joint localization is decoupled into two
sub-tasks: (1) localization of the root joint image coordinate (u, v), and (2) es-
timation of the root joint depth Z in the camera frame, which is then used to
back-project (u, v) to 3D space. We use an oﬀ-the-shelf single-person 3D pose
estimator to estimate the 3D joint locations of each person with respect to the
root joint. The ﬁnal absolute 3D pose of each person in the camera coordinate
system is obtained by the transformation of each joint location with the absolute
location of the root joint.
The details of our proposed root joint localization framework are introduced
in Section 3.2. The choices of speciﬁc object detector and single-person 3D pose
estimator used in our experiments are given in the implementation details in
Section 4.2.
3.2
Root Localization Framework
Our framework for monocular image single/multi-person depth estimation is
shown in Figure 2. The framework consists of a Feature Pyramid Network (FPN)-
based backbone, a heatmap-based human pose estimation branch, and a Graph
Neural Network (GNN)-based depth estimation branch.

6
Jiahao Lin and Gim Hee Lee
FPN Backbone
C=256
1x1 conv
1x1 conv
C=128
8x
4x
2x
Multi-scale Feature Extraction
(a)
(b)
Upscale
Upscale
C=128
C=64
3x3 conv
3x3 conv
Fig. 3. (a) ResNet-based Feature Pyramid Network Backbone for general feature ex-
traction. (b) Multi-scale feature extraction subnet architecture used for both Pose
feature and Depth feature extraction.
Backbone Network. We choose FPN [15] as our backbone network due to its
capability of explicitly handling features of multiple scales in the form of feature
pyramids. Hence, it is suitable for perceiving the scale of human body parts and
consequently enhances depth estimation of the human pose in an image. The
FPN network consists of a ResNet-50 [9] with feature blocks of four diﬀerent
scales C2, C3, C4, C5 (cyan layers in Figure 3(a)), where a reversed hierarchy
of feature pyramid P5, P4, P3, P2 is built upon (orange layers in Figure 3(a)).
Each of the four scales encodes hierarchical levels of feature representations,
which are then passed through two consecutive convolutional layers as shown
in Figure 3(b). An upsampling operation with corresponding upsample scale
factor is applied between the two convolutional layers to ensure matching spatial
resolution from the output of the four scales. Batch Normalization [11] and
ReLU operations are used after each convolution layer. Weights are not shared
across scales. Blocks of all scales are then concatenated to form the ﬁnal feature
block F. Since we ﬁnd that the downstream tasks of pose estimation and depth
estimation are not collaboratively correlated, we split the multi-scale feature
processing from the output feature pyramid P5, P4, P3, P2 of the backbone into
two parallel branches without shared weights as shown in Figure 2. We denote
the features as Fpose and Fdepth, respectively.
2D Pose Estimation Branch. We propose to use estimated 2D pose as a
guide to aggregate information from useful feature regions to eﬀectively distil
information from the image and discard irrelevant areas such as the background.
We ﬁrst regress NJ heatmaps ˆH that correspond to the NJ joints with a 1 × 1
convolution from feature block Fpose. Each of the NJ heatmaps are normalized
across all spatial locations with a softmax operation. A direct read out of the
coordinate from the local maximum limits the precision of the joint location
estimation due to the low resolution of the output heatmap (4x downsample
from input image in ResNet backbone). To circumvent this problem, we follow
the idea of “soft-argmax” in [29] and compute the “integral” version of estimated
coordinate (ˆu, ˆv) for each joint j using the weighted sum of coordinates:
(ˆuj, ˆvj) =
(W −1,H−1)
X
(u,v)=(0,0)
ˆH(j)
u,v · (u, v),
(1)

HDNet: Human Depth Estimation
7
where W and H are the width and height of output heatmap. The softmax op-
eration guarantees that the weights ˆHu,v form a valid distribution which sum up
to 1 over all spatial locations. To supervise the heatmap regression, we generate
a ground truth heatmap H(j)GT for each joint j. A Gaussian peak is created
around the ground truth joint location (uj, vj) with a preset standard deviation
that controls the compactness of the Gaussian peak. We use standard Mean
Squared Error (MSE) as the heatmap regression loss and L1 loss for the pose
after soft-argmax as follows:
Lhm =
1
NJHW
NJ
X
j
(W −1,H−1)
X
(u,v)=(0,0)
H(j)GT
u,v
−ˆH(j)
u,v

2
,
(2)
Lpose =
1
NJ
NJ
X
j
uGT
j
−ˆuj
 +
vGT
j
−ˆvj


.
(3)
To deal with multiple persons in the image, we focus on a target person by
zeroing out the regions of the heatmap outside the bounding box of that person
from the object detector.
Depth Estimation Branch. After we obtain the heatmaps, we use them as
attention masks to guide the network into focusing on speciﬁc regions of the im-
age related to the target person. More speciﬁcally, we only care about features
from pixel locations that are close to the joints of the target person. The intu-
ition behind our design choice is that joint locations contain more scale-related
information than the larger yet less discriminative areas such as the whole up-
per body trunk. Attention-guided feature pooling is also adopted in other tasks
such as action recognition [17] and hand pose estimation [13]. We compute the
weighted sum feature vector d for each joint j from the feature block Fdepth as:
d(j) =
(W −1,H−1)
X
(u,v)=(0,0)
ˆH(j)
u,v · Fdepthu,v.
(4)
To eﬀectively aggregate features corresponding to diﬀerent joint types, we for-
mulate a standard Graph Neural Network (GNN) where each node represents
one joint type, e.g., elbow, knee, etc. The aggregated features d(j) for each joint
type j is fed into the corresponding node X(j)
in in the graph as input. Each layer
of the GNN is deﬁned as:
X(i)
out = σ

˜aiifself(X(i)
in ; Θself) +
X
j̸=i
˜aijfinter(X(j)
in ; Θinter)

.
(5)
The feature of each input node Xin undergoes the linear mappings fself(.) and
finter(.) that are parametrized by Θself and Θinter, respectively. The output
of the node, i.e. X(i)
out is computed from a weighted aggregation of fself(.) and

8
Jiahao Lin and Gim Hee Lee
finter(.) of all other nodes. The weighting factor ˜aij is an element of the normal-
ized adjacency matrix ˜A ∈RNj×Nj that controls the extent of inﬂuence of the
nodes on each other. The original adjacency matrix is A ∈{0, 1}Nj×Nj; an ele-
ment aij equals 1 if there is a skeletal link between joint i and j, e.g. left knee to
left ankle, or otherwise 0. ˜A ∈RNj×Nj is obtained by applying L1-normalization
on each row of A. The non-linearity function σ(.) is implemented with a Batch
Normalization followed by a ReLU. We stack L GNN layers in total. After the
last GNN layer, we merge the feature output from each node with an average
pooling operation.
Target output formulation Inspired by the work [7] for scene depth estima-
tion, we formulate the depth estimation as a classiﬁcation problem instead of
directly regressing the numerical value of depth. We follow the practice in [7] to
discretize the log-depth space into a preset number of bins, NB. We compute:
b(d) = log d −log α
log β −log α · (NB −1),
(6)
where ⌊b⌉gives the bin index of the depth, and the depth d of a pose is assumed
to be within the range [α, β]. Here ⌊.⌉is the round-oﬀto the nearest integer
operator. To eliminate quantization 

## method
MRPEz(↓)
AProot
25 (↑)
RootNet [21]
108.1
31.0
Ours direct regression
94.5
27.3
Ours shared feature branch
72.0
31.9
Ours w/o GNN
72.9
32.7
Ours w/o HM pooling
71.8
26.0
Ours (full)
69.9
39.4
4.5
Ablation Studies
We conduct ablation studies to show how each component in our framework
aﬀects the root joint localization accuracy. We evaluate the depth estimation
accuracy MRPEz on Human3.6M dataset and the root joint localization AProot
25
on MuPoTS-3D dataset with diﬀerent variants of our framework in Table 6. The
state-of-the-art approach [21] is also included for comparison.
– “Ours direct regression”: Performance drop (by 24.6mm and 12.1%) with
directly regressing target depth instead of performing classiﬁcation over bin-
ning shows the eﬀectiveness of formulating the depth estimation as a classi-
ﬁcation task.
– “Ours shared feature branch”: One single multi-scale feature branch is kept
after FPN, which means Fpose and Fdepth use the same feature representa-
tion. This setting causes performance to drop (by 2.1mm and 7.5%), and
thus demonstrates that the features used for pose estimation and depth es-
timation are not highly correlated.
– “Ours w/o GNN”: We replace the GNN layers in our depth estimation branch
with same number of fully-connected layers and observe a performance drop
(by 3mm and 6.7%), showing the eﬀectiveness of the graph neural network in
propagating and reﬁning the features extracted for diﬀerent types of joints.
– “Ours w/o HM pooling”: We remove feature pooling with estimated heatmaps
as mask in the depth estimation branch and instead apply a global average
pooling to obtain a single feature vector. The GNN layers are replaced with
fully-connected layers since we do not explicitly diﬀerentiate between dif-
ferent joint types. We observe a performance drop (by 1.9mm and 13.4%),
which demonstrates the eﬀectiveness of utilizing estimated pose as attention
mask for useful feature aggregation.
4.6
Discussions
We analyze the root joint localization results on the challenging multi-person
dataset MuPoTS-3D and observe several sources of large errors as shown in Fig-
ure 4: (1) Bounding boxes for two persons tend to have overlapping areas when

14
Jiahao Lin and Gim Hee Lee
(a)
(b)
Fig. 4. Typical errors in multi-person root localization. (a) Close and overlapping
bounding box regions. (b) Diﬀerent sizes of target persons.
Fig. 5. Qualitative results on MuPoTS-3D dataset. Columns are: (1) image with bound-
ing boxes (2) left-front view (3) right-front view (4) top-down view
the person closer to the camera partially occludes the other person farther away
(Figure 4(a)). Masking the heatmaps with bounding box cannot eﬀectively re-
move undesired regions of information and consequently the depth estimation for
both persons are aﬀected. The problem of ﬁne-grained target person segmenta-
tion will be of interest for future research. (2) Since monocular depth estimation
relies on prior knowledge such as typical scale of human bodies, estimation tends
to be erroneous when the size of target person is far away from the “average”
size, e.g., the target is a child or a relatively short person (Figure 4(b)). Research
on person 3D size estimation may complement our depth estimation task and
improve the generalizability to persons of diﬀerent sizes.
5

## experiments
4.1
Datasets and Evaluation Metrics
Human3.6M dataset. Human3.6M dataset [12] is currently the largest pub-
licly available dataset for human 3D pose estimation. The dataset consists of 3.6
million video frames captured by MoCap system in a constrained indoor stu-
dio environment. 11 actors performing 15 activities are captured from 4 camera
viewpoints. 3D ground truth poses in world coordinate system and camera ex-
trinsic (rotation and translation with respect to world coordinate) and intrinsic
parameters (focal length and principal point) are available. We follow previous
works that ﬁve subjects (S1, S5, S6, S7, S8) are used in training and two sub-
jects (S9 and S11) are used for evaluation. We use every 5th and 64th frames in
each video for training and evaluation respectively. No extra 2D pose dataset is
used to augment the training. We follow the metric Mean Root Position Error
(MRPE) proposed in [21] to evaluate the root localization accuracy. Speciﬁcally,
we consider the Euclidean distance between the estimated and the ground truth
3D coordinate of the root joint.
MuCo-3DHP and MuPoTS-3D datasets. MuCo-3DHP and MuPoTS-3D
are two datasets proposed by Mehta et al. [20] to evaluate multi-person 3D pose
estimation performance. The training set MuCo-3DHP is a composite dataset
which merges randomly sampled 3D poses from single-person 3D human pose
dataset MPI-INF-3DHP [19] to form realistic multi-person scenes. The test set
MuPoTS-3D is a markerless motion captured multi-person dataset including
both indoor and outdoor scenes. We use the same set of MuCo-3DHP synthesized
images from [21] for a fair comparison. No extra 2D pose dataset is used to
augment the training. For evaluation of multi-person root joint localization, we
follow [21] to report the average precision and recall of 3D root joint location
under diﬀerent thresholds. A root joint with a smaller distance to the matched
ground truth root joint location than a threshold is considered a true positive
estimation. We follow [21] to report 3DPCKabs for evaluation of the root-aware
3D pose estimation, where 3DPCK (3D percentage of correct keypoints) for the
estimated poses is evaluated without root alignment. 3DPCK treats an estimated
joint as correct if it is within 15 cm distance from the matched ground truth joint.
Although our framework does not focus on root-relative 3D pose estimation, we
also report the root-aligned 3DPCKrel to show that accurate root localization
also beneﬁts the precision of 3D pose estimation.

10
Jiahao Lin and Gim Hee Lee
Table 1. MRPE results comparison with state-of-the-arts on the Human3.6M dataset.
MRPEx, MRPEy, and MRPEz are the average errors in x, y, and z axes, respectively.

## related_work
Human pose estimation has been an interesting yet challenging problem in com-
puter vision. Early methods use a variety of hand-crafted features such as silhou-
ette, shape, SIFT features, HOG for the task. Recently, with the power of deep
neural networks and well-annotated large-scale human pose datasets, increasing
learning-based approaches are proposed to tackle this challenging problem.
Single-person 2D pose estimation. Early works, such as Stacked Hour-
glass [23], Convolutional Pose Machines [30], etc., have been proposed to use
deep convolutional neural networks as feature extractors for 2D pose estima-
tion. Heatmaps of joints are the commonly used representation to indicate the
presence of joints at spatial locations with Gaussian peaks. More recent works
including RMPE [5], CFN [10], CPN [3], HRNet [28], etc., introduce various
framework designs to improve the joint localization precision.
Single-person 3D pose estimation. Approaches for 3D pose estimation can
be generally categorized into two groups. Direct end-to-end estimation of 3D
pose from RGB images regresses both 2D joint locations and the z-axis root-
relative depth for each joint. [25, 29] extend the notion of heatmap to the 3D
space, where estimation is performed in a volumetric space. Another group of
approaches decouples the task into a two-stage pipeline. The 2D joint locations

4
Jiahao Lin and Gim Hee Lee
are ﬁrst estimated, followed by a 2D-to-3D lifting. [6, 18] utilize Multi-Layer
Perceptron (MLP) to learn the mapping.
Multi-person 2D pose estimation. Top-down [3, 5, 8, 10, 28] and bottom-
up [2, 22, 24] approaches have been proposed to estimate poses for multiple
persons. Top-down approaches utilize a human object detector to localize the
bounding box, followed by a single-person pose estimation pipeline with image
patch cropped from the bounding box. Bottom-up approaches detect human
joints in a person-agnostic way, followed by a grouping process to identify joints
belonging to the same person. Top-down approaches usually estimate joint loca-
tions more precisely because bounding boxes of diﬀerent sizes are scaled to the
same size in the single-person estimation stage. However, top-down approaches
tend to be more computationally expensive due to the redundancy in bounding
box detections.
Multi-person 3D pose estimation. Several works [4,20,26,27,31] have been
conducted on multi-person 3D pose estimation. Rogez et al. [26] propose a LCR-
Net which consists of localization, classiﬁcation, and regression parts and esti-
mates each detected human with a classiﬁed and reﬁned anchor pose. Mehta
et al. [20] propose a bottom-up approach which estimates a specially designed
occlusion-robust pose map and readout the 3D poses given 2D poses obtained
with Part Aﬃnity Fields [2]. Dabral et al. [4] propose to incorporate hour-
glass network into Mask R-CNN detection heads for better 2D pose localiza-
tion, followed by a standard residual network to lift 2D poses to 3D. Zanﬁr et
al. [31] design a holistic multi-person sensing pipeline, i.e. MubyNet, to jointly
address the problems of multi-person 2D/3D skeleton/shape-based pose estima-
tion. However, these works only estimate and evaluate the 3D pose after root
joint alignment and ignore the global location of each pose. Recently, Moon et
al. [21] propose a multi-stage pipeline for multi-person camera-space 3D pose
estimation. The pipeline follows the top-down scheme and consists of a RootNet
which localizes the root joint for each detected bounding box. We also adopt
the top-down scheme pipeline and estimate the camera-space root joint location
and 3D pose for each detected bounding box. To our best knowledge, [21] and
our work are the only two works that focus on the estimation and evaluation of
multi-person root joint locations. Compared to [21] which relies on the size of
detected bounding box, we utilize the underlying features and design a human-
speciﬁc pose-based root joint depth estimation framework to signiﬁcantly boost
the root localization performance.
3
Our Approach
3.1

## conclusion
In this work, we proposed the Human Depth Estimation Network (HDNet), an
end-to-end framework to address the problem of accurate root joint localization
for multi-person 3D absolute pose estimation. Our HDNet utilizes deep features
and demonstrates the capability to precisely estimate depth of root joints. We
designed a human-speciﬁc pose-based feature aggregation process in the HDNet
to eﬀectively pool features from regions of human body joints. Experimental
results on multiple datasets showed that our framework signiﬁcantly outperforms
the state-of-the-art in both root joint localization and 3D pose estimation.

HDNet: Human Depth Estimation
15