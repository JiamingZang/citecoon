# Anatomy-Aware 3D Human Pose Estimation With Bone-Based Pose Decomposition

> 2021 · id: W3126541466 · arXiv: 2002.10322 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
3
D human pose estimation in videos has been widely stud-
ied in recent years. It has extensive applications in action
recognition, sports analysis and human-computer interaction.
Current state-of-the-art approaches [1], [2], [3] typically de-
compose the task into 2D keypoint detection followed by 3D
pose estimation. Given an input video, they ﬁrst detect the
2D keypoints of each frame, and then predict the 3D joint
locations of a frame based on the 2D keypoints.
When estimating the 3D joint locations from 2D keypoints,
the challenge is to resolve depth ambiguity, as multiple 3D
poses with different joint depths can be projected to the
same 2D keypoints. Exploiting temporal information from
the video has been demonstrated to be effective for reducing
this ambiguity. Typically, to predict the 3D joint locations
of a frame in a video, recent approaches [1], [4], [5] utilize
temporal networks that additionally feed the adjacent frames’
T. Chen, and J. Luo are with the Department of Computer Science,
University of Rochester, Rochester, NY, 14627 USA (e-mail: {zyang39,
tusharku, tchen45, jluo}@cs.rochester.edu).
C. Fang, X. Shen, Y. Zhu and Z. Chen are with Bytedance AI Lab,
Mountain View, CA, USA (e-mail: {fangchen, shenxiaohui, yiheng.zhu,
zhili.chen}@bytedance.com).
Corresponding author: Jiebo Luo.
Copyright©2021 IEEE. Personal use of this material is permitted. However,
permission to use this material for any other purposes must be obtained from
the IEEE by sending an email to pubs-permissions@ieee.org.
2D keypoints as input. These approaches consider the adjacent
local frames most associated with the current frame, and
extract their information as extra guidance. However, such
approaches are limited to exploiting information only from
the neighboring frames. Given a 1-minute input video with a
frame rate of 50, even though we choose the existing temporal
network with largest temporal window size (i.e 243 frames)
[1], it is limited to using a concentrated short segment (about
one-twelfth length of the video) to predict a single frame. Such
a design can easily make existing temporal networks fail when
the current frame and its adjacent input frames correspond to
a complex pose, because none of the input frames provide
reliable and high-conﬁdence information to the networks.
Considering this, our ﬁrst contribution is proposing a novel
approach that can effectively capture the knowledge from both
local and distant frames to estimate the 3D joint locations
of the current frame, by cleverly exploiting the anatomic
properties of the human skeleton. We refer to it as anatomy
awareness. Speciﬁcally, based on the anatomy of the human
skeleton, we decompose the task of 3D joint location predic-
tion into two sub-tasks – bone direction prediction and bone
length prediction. We demonstrate that the combination of the
two new tasks are essentially equivalent to the original task.
The motivation is based on the fact that the bone lengths of
a person remain consistent in a video over time (This can be
veriﬁed by 3D human pose datasets such as Human3.6M and
MPI-INF-3DHP). Hence, when we predict the bone lengths
of a particular frame, we can leverage the frames distributed
over the duration of the entire video for more accurate and
smooth prediction. Note that although Sun et al. [6] transform
the task into a generic bone-based representation, such a
generic representation does not allow them to utilize that
critical bone length consistency. In contrast, we decompose the
task explicitly into bone direction and bone length prediction.
We demonstrate that this explicit design leads to signiﬁcant
advantages over either the generic representation design in [6]
or imposing a bone length consistency loss across frames.
However, it is nontrivial to implement this explicit design.
One problem for training the proposed bone length prediction
network is that the training dataset typically contains only a
few skeletons. For example, the training set of Human3.6M
contains 5 actors corresponding to 5 bone length settings.
Directly training the network on the data from the 5 actors
leads to serious overﬁtting. Therefore, we adopt the fully-
connected residual network for bone length prediction and
propose two effective mechanisms to prevent overﬁtting via
a network design and data augmentation.
As for the bone directions, we adopt the temporal convolu-
arXiv:2002.10322v5  [cs.CV]  26 Jan 2021

2
tional network in [1] to predict the direction of each bone in
the 3D space for each frame. Motivated by [5], we believe
it is beneﬁcial to predict the directions of different bones
hierarchically, instead of all at once as in [1]. Following the
human skeleton anatomy, the directions of simple torso bones
(e.g. lumbar vertebra) with less motion variation should be
predicted ﬁrst, and then guide the prediction of challenging
limb bones (e.g. arms and legs). This strategy is applied
straightforwardly by a recurrent neural network (RNN) with
different joints predicted step by step in [5] for a single
frame. However, the high computation complexity of RNN
precludes the network from holding a large temporal window
which has been shown to improve performance. To solve this
issue, based on [1], we further propose a high-performance
fully-convolutional propagating architecture, which contains
multiple sub-networks with each predicting the directions
of all the bones. The hierarchical prediction is implicitly
performed via long skip connections between adjacent sub-
networks.
Additionally, motivated by [6], we create an effective joint
shift loss for the two sub-tasks (i.e., bone direction prediction
and bone length prediction) to learn jointly. The joint shift
loss penalizes the relative joint shift between all long-range
joint pairs, for example the left hand and right foot. Thus, it
provides an extra strong supervision for the two networks to
be trained to coordinate with each other and produce robust
predictions.
Last but not least, we propose a simple yet effective
approach to further reduce the depth ambiguity. Speciﬁcally,
we incorporate 2D keypoint visibility scores into the model
as a new feature, which indicates the probability of each
2D keypoint being visible in a frame and provides extra
knowledge of the depth relation between speciﬁc joints. We
argue that the scores are useful to those poses with body parts
occluded or when the relative depth matters. For example, if
a person keeps her/his hands in front of the chest in a frontal
view, our model will be confused on whether the hands are in
front of the chest (visible) or behind the back (occluded), since
the occluded 2D keypoints can still be predicted sometimes.
Furthermore, We adopt an implicit attention mechanism to
dynamically adjust the importance of the visibility scores for
better performance.
Our contributions are summarized as follows:
• We explicitly decompose the task of 3D joint estimation
into bone direction prediction and bone length prediction.
As such, the bone length prediction branch can fully
utilize frames across the entire video.
• We propose a new fully-convolutional architecture for
hierarchical bone direction prediction.
• We propose a high-performance bone length prediction
network, two mechanisms are created to effectively pre-
vent overﬁtting.
• We feed the visibility scores of 2D keypoint detection
into the model to better resolve the depth ambiguity.
• Our model is inspired by the human skeleton anatomy and
achieves the state-of-the-art performance on Human3.6M
and MPI-INF-3DHP datasets.

## method
In this section, we formally present our 3D pose estimation
model. In section III-A, we ﬁrst describe the overall anatomy-
aware framework that decomposes the 3D joint location pre-
diction task into bone length and direction prediction. In
section III-B, we present the fully-convolutional propagating
network for hierarchical bone direction prediction. In Sec-
tion III-C, the architecture and training details of bone length
prediction network are presented. In Section III-D, we describe
the framework’s overall training strategy. In section III-E,
an implicit attention mechanism is introduced to feed the
keypoint visibility scores into the model as extra guidance.
Our framework’s overall architecture is shown as Fig. 1.
A. Anatomy-aware Framework
As in [1], [4], [3], given the predicted 2D keypoints of
each frame in a video, we aim at predicting the normalized
3D locations of j pre-deﬁned joints for each frame. The 3D
location of joint “Pelvis” is commonly deﬁned as the origin
of the 3D coordinates. Given a human joint set that contains
j joints as in Fig. 2, they correspond to (j −1) directed
bones with each joint being the vertex of at least one bone.
This enables us to transform the 3D joint coordinates to the
presentation of bone lengths and bone directions.
Bone Direction 
Prediction Network
Bone Length 
Prediction Network
Bone Directions 
Bone Lengths
Bone Direction Loss
Bone Length Loss
Joint Shift Loss
MPJPE Loss
Consecutive 
Frames
Randomly 
Sampled 
Frames
Fig. 1. The overview of the proposed anatomy-aware framework. It predicts
the bone directions and bone lengths of the current frame using consecutive
local frames and randomly sampled frames across the entire video, respec-
tively.
Head
L.shoulder
R.shoulder
L.elbow
R.elbow
L.wrist
R.wrist
Spine
L.hip
R.hip
Pelvis
Neck
L.knee
R.knee
L.ankle
R.ankle
Joint:
Bone:
Fig. 2. The joint and bone representation of a human pose.
Formally, to predict the 3D joint locations of a speciﬁc (i.e.
current) frame, we decompose the task to predict the length
and direction of each bone. For the k-th joint, its 3D location
−→
Jk can be derived as:
−→
Jk =
X
b∈Bk
−→
Db · Lb
(1)
Here −→
Db and Lb are the direction and length of bone b,
respectively. Bk contains all the bones in the path from
“Pelvis” to the k-th joint.
We use two separate sub-networks to predict the bone
lengths and directions of the current frame, respectively, as
bone length prediction needs global input to ensure consis-
tency across all the frames, whereas bone directions should
be estimated within a local temporal window. Meanwhile,
to ensure consistency between predicted bone lengths and
directions, motivated by [6], we add a joint shift loss between
the two predictions in addition to their own losses, as shown
in Fig. 1. Speciﬁcally, the joint shift loss is deﬁned as follows:
LJS =
X
k1,k2∈P,k1<k2
Xk1,k2
JS
−Y k1,k2
JS

1
2
(2)
Here Y k1,k2
JS
is the 3-dimensional ground-truth relative joint
shift of the current frame from the k1-th joint to the k2-th
joint, Xk1,k2
JS
is the corresponding predicted relative joint shift
derived from the predicted bone lengths and bone directions
of the current frame. P contains all the joint pairs that are not
directly connected as a bone. With the joint shift loss, the two
sub-networks are connected and enforced to learn from each
other jointly. We describe the details of the two sub-networks
in the following two sections.

4
  Duplicate
  Duplicate
Bone Directions
Bone Direction Loss + 
Joint Shift Loss
Bone Direction Loss + 
Joint Shift Loss
Bone Direction Loss + 
Joint Shift Loss
Sub-network 3
Sub-network 2
Sub-network 1
Bone Directions
Bone Directions
C
C
C
C
C
Concatenate
(b×d, 2n)
(b×   , o)
d
(b×  , 2o)
d
s
 s
(b×   , o)
d
s
2
(b×   , 2o)
d
s
2
(b×  , 2o)
d
s
(b×   , 2o)
d
s
2
(b, 3(j-1))
(b, 3(j-1))
(b, 3(j-1))
Fig. 3. The architecture of the bone direction prediction network. Long skip
connections are added between adjacent sub-networks. We illustration the
dimension of each input/output. b is the batch size. o is the output channel
number of fully-connected layer. s is the stride of 1D convolution layer in the
network (s = 3). n is the size of the 2D keypoint set. d is the input frame
number of the bottom sub-network.
B. Bone Direction Prediction Network
We adopt the temporal fully-convolutional network pro-
posed by Pavllo et al. [1] as the backbone architecture of
our bone direction prediction network. Speciﬁcally, the 2D
keypoints of d consecutive frames are concatenated to form
the input to the network, with the 2D keypoints of the
current frame in the center. In essence, to predict the bone
directions of the current frame, the temporal network captures
the information of the current frame and the context from its
adjacent frames as well. A bone direction loss based on mean
squared error is applied to train the network:
LD = ∥XD −YD∥2
2
(3)
Here XD and YD represent the predicted and ground-truth
3(j −1)-dimensional bone direction vector of the current
frame, respectively.
It should be noted that the joint shift loss introduced in
Section III-A makes the predicted directions of different bones
mutually relevant. For example, if the predicted direction of
the left lower arm is inaccurate, the predicted direction of
the left upper arm will also be affected, since the model is
encouraged to regress a long range shift from left shoulder to
left wrist. Intuitively, it would beneﬁt the overall prediction if
we could ﬁrst predict those easy and high-conﬁdent cases, and
let them guide the subsequent prediction of other joints. As
poses may vary signiﬁcantly, it is difﬁcult to pre-determine the
hierarchy of the prediction. Motivated by [5], here we propose
a fully-convolutional propagating architecture with long skip
connections, and let the network itself to learn the prediction
hierarchy instead, as in Fig. 3.
Speciﬁcally, the architecture is a stack of several sub-
networks, with each sub-network being a temporal fully-
convolutional network with residual blocks proposed by [1].
The output of each sub-network is the predicted bone direc-
tions of the current frame. Except the top sub-network, we
temporally duplicate the output of each sub-network d
s times as
2D Keypoint 
of l Frames
(b×l, 2n)
(b×l, o)
(b×l, o)
(b×l, o)
Bone Length 
Self-attention 
Module
(b×l, 3j)
(b×l, j-1)
(b, j-1)
Bone Lengths
3D Joint 
Locations of 
l frames
Bone Length 
of l Frames
Fully-
conntect
Batch
Norm1d
ReLU + 
Dropout
Bone 
Length 
Deriving
×r
Fig. 4.
Detailed structure of the bone length prediction network. r is the
number of residual block.
the input to the next sub-network. For each residual block of a
speciﬁc sub-network, we concatenate its output with the output
of the corresponding residual block in the adjacent upper sub-
network on channel level. This forms the long skip connections
between adjacent sub-networks. We adopt an independent
training strategy for each sub-network, that is, we train each
sub-network by the loss of the bone direction prediction
network, the back propagation is blocked between different
sub-networks. By doing that, the bottom networks would not
be affected by the upper ones, and instead would propagate
high-conﬁdent predictions to guide subsequent predictions. In
the process, the model automatically learns the hierarchical
order of the prediction. In Section IV, we demonstrate the
effectiveness of the proposed architecture.
C. Bone Length Prediction Network
As discussed in Section III-A, the prediction of bone lengths
requires global inputs from the entire video. However, taking
too many frames as the input would make the computation
prohibitively expensive. To capture the global context efﬁ-
ciently, we choose to randomly sample l frames across the
entire video as the input to the network. The detailed structure
of the network is shown as Fig. 4.
We adopt the fully-connected residual network for bone
length prediction. Speciﬁcally, it has the same structure and
layer number as the bottom sub-network of the bone direction
prediction network. However, since the randomly sampled
frames do not have temporal connections, we replace each 1D
convolution layer by the fully-connected layer in the network.
This adapts the network for single-frame input instead of
multi-frame consecutive inputs. The fully-connected network
predicts the (j −1) bone lengths of each sampled frame.
Intuitively, we can average the predicted bone lengths of
each sampled frame as the predicted bone lengths of the
current frame. In such a way, a similar bone length loss can
be applied to train the fully-connected network:
LL = ∥XL −YL∥2
2
(4)
Here XL and YL are the predicted and ground-truth (j −1)-
dimensional bone length vector of the current frame.
However, since the training datasets usually only contain
very limited number of actors, and the bone lengths in the
videos performed by the same actor are identical. Such a
training loss would lead to severe overﬁtting. To solve this
problem, 

## experiments
A. Datasets and Evaluation
We evaluate the proposed model on two well established 3D
human pose estimation datasets: Human3.6M [30] and MPI-
INF-3DHP [31].
• Human3.6M contains 3.6 million video frames with the
corresponding annotated 3D and 2D human joint positions,
from 11 actors. Each actor performs 15 different activities
captured from 4 camera views. Following previous works
[1], [5], [4], [6], [17], the model is trained on ﬁve subjects
(S1, S5, S6, S7, S8) and evaluated on two subjects (S9
and S11) on a 17-joint skeleton. We follow the standard
protocols to evaluate the models on Human3.6M. The
ﬁrst one (i.e. Protocol 1) is the mean per-joint position
error (MPJPE) in millimeters that measures the mean
Euclidean distance between the predicted and ground-truth
joint positions without any transformation. The second one
(i.e. Protocol 2) is the normalized variant P-MPJPE after
aligning the predicted 3D pose with the ground-truth using
a similarity transformation. In addition, to measure the
smoothness of predictions over time, which is important
for video, we also report the joint velocity errors (MPJVE)
created by [1] corresponding to the MPJPE of the ﬁrst
derivative of the 3D pose sequences.
• MPI-INF-3DHP is a recently proposed 3D dataset con-
sisting of both constrained indoor and complex outdoor
scenes. It records 8 actors performing 8 activities from 14
camera views. Following [20], [31], on a 14-joint skeleton,
we consider all the 8 actors in the training set and select
sequences from 8 camera views in total (5 chest-high
cameras, 2 head-high cameras and 1 knee-high camera) for
training. Evaluation is performed on the independent MPI-
INF-3DHP test set that has different scenes, camera views
and relatively different actions from the training set. This
design implicitly covers the cross-dataset evaluation. We
report the Percentage of Correct Keypoints (PCK) within
150mm range, Area Under Curve (AUC), and MPJPE.
B. Implementation details
For Human3.6M, we use the predicted 2D keypoints re-
leased by [1] from the Cascaded Pyramid Network (CPN)
as the input of our 3D pose model. For MPI-INF-3DHP,
the predicted 2D keypoints are acquired from the pretrained
AlphaPose model [32]. In addition to the 2D keypoints, the
keypoint visibility scores for both datasets are also extracted
from the pretrained AlphaPose model.
We use the Adam optimizer to train our model in an end-
to-end manner. For each training iteration, the mini-batch
size is set to 1024 for both original samples and augmented
samples. We set λD = 0.02, λL = 0.05, λJ = 1 and λJS =
0.1 for the loss terms in Equation 7. For the bone length self-
attention module, we set γ = 10 in Equation 8. The sampled
frame number of the bone length prediction network l is set
to 50 for both the training and inference process. For the
proposed architecture in Section III-B, the number of sub-
networks is set to 2. As in [1], the output channel number
of each 1D convolution layer and fully-connected layer is
set to 1024. For actual implementation, instead of manually
deriving the 3D joint locations and relative joint shifts from
the predicted bone lengths and bone directions, we regress the
two objectives by feeding the concatenation of the predicted
bone length vector and bone direction vector into two fully-
connected layers, respectively. The fully-connected layers are
trained together with the whole network. This achieves slightly
better performance.
C. Experiment results
Table I shows the quantitative results of our proposed full
model and other baselines on Human3.6M. Following [1], we
present the performance of our 81-frame and 243-frame mod-
els which receive 81 and 243 consecutive frames, respectively,
as the input of the bone direction prediction network. We
also experiment with a causal version of our model to enable
real-time prediction. During the training/inference process, the
causal model only receives d consecutive and l randomly
sampled frames from the past/current frames for the current
frame’s estimation. Overall, our model has low average error
on both Protocol 1, Protocol 2 and MPJVE. On a great number
of actions, we achieve the best performance. Compared with
the baseline model [1] that shares the same 2D keypoint
detector, our model achieves more smooth prediction with
lower MPJVE and achieves signiﬁcantly better performance
on complex activities such as “Sitting” (-3.4mm in Protocol
1) and “Sitting down” (-5.6mm in Protocol 1). We attribute
it to the accurate prediction of the bone lengths for these
activities. Even though the person bends his/her body, based on
the predicted bone lengths, the joint shift loss can effectively
guide the model to predict high-quality bone directions. Fig. 6

7
TABLE I
QUANTITATIVE COMPARISONS BETWEEN THE ESTIMATED POSE AND THE GROUND-TRUTH ON HUMAN3.6M UNDER PROTOCOLS 1,2 AND MPJVE. (*)
WE REPORT THE RESULT WITHOUT DATA AUGMENTATION USING VIRTUAL CAMERAS.
Protocol 1
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg
Martinez et al. [17] ICCV’17
51.8
56.2
58.1
59.0
69.5
78.4
55.2
58.1
74.0
94.6
62.3
59.1
65.1
49.5
52.4
62.9
Sun et al. [6] ICCV’17
52.8
54.8
54.2
54.3
61.8
67.2
53.1
53.6
71.7
86.7
61.5
53.4
61.6
47.1
53.4
59.1
Pavlakos et al. [12] CVPR’18
48.5
54.4
54.4
52.0
59.4
65.3
49.9
52.9
65.8
71.1
56.6
52.9
60.9
44.7
47.8
56.2
Yang et al. [9] CVPR’18
51.5
58.9
50.4
57.0
62.1
65.4
49.8
52.7
69.2
85.2
57.4
58.4
43.6
60.1
47.7
58.6
Luvizon et al. [33] CVPR’18
49.2
51.6
47.6
50.5
51.8
60.3
48.5
51.7
61.5
70.9
53.7
48.9
57.9
44.4
48.9
53.2
Hossain & Little [4] ECCV’18
48.4
50.7
57.2
55.2
63.1
72.6
53.0
51.7
66.1
80.9
59.0
57.3
62.4
46.6
49.6
58.3
Lee et al. [5] ECCV’18
40.2
49.2
47.8
52.6
50.1
75.0
50.2
43.0
55.8
73.9
54.1
55.6
58.2
43.3
43.3
52.8
Chen et al. [3] CVPR’19
41.1
44.2
44.9
45.9
46.5
39.3
41.6
54.8
73.2
46.2
48.7
42.1
35.8
46.6
38.5
46.3
Pavllo et al. [1] (243 frames, Causal) CVPR’19
45.9
48.5
44.3
47.8
51.9
57.8
46.2
45.6
59.9
68.5
50.6
46.4
51.0
34.5
35.4
49.0
Pavllo et al. [1] (243 frames) CVPR’19
45.2
46.7
43.3
45.6
48.1
55.1
44.6
44.3
57.3
65.8
47.1
44.0
49.0
32.8
33.9
46.8
Lin et al. [2] BMVC’19
42.5
44.8
42.6
44.2
48.5
57.1
42.6
41.4
56.5
64.5
47.4
43.0
48.1
33.0
35.1
46.6
Cai et al. [34] ICCV’19
44.6
47.4
45.6
48.8
50.8
59.0
47.2
43.9
57.9
61.9
49.7
46.6
51.3
37.1
39.4
48.8
Cheng et al. [35] ICCV’19 (*)
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
44.8
Yeh et al. [36] NIPS’19
44.8
46.1
43.3
46.4
49.0
55.2
44.6
44.0
58.3
62.7
47.1
43.9
48.6
32.7
33.3
46.7
Xu et al. [18] CVPR’20
37.4
43.5
42.7
42.7
46.6
59.7
41.3
45.1
52.7
60.2
45.8
43.1
47.7
33.7
37.1
45.6
Ours (243 frames, Causal)
42.5
45.4
42.3
45.2
49.1
56.1
43.8
44.9
56.3
64.3
47.9
43.6
48.1
34.3
35.2
46.6
Ours (81 frames)
42.1
43.8
41.0
43.8
46.1
53.5
42.4
43.1
53.9
60.5
45.7
42.1
46.2
32.2
33.8
44.6
Ours (243 frames)
41.4
43.5
40.1
42.9
46.6
51.9
41.7
42.3
53.9
60.2
45.4
41.7
46.0
31.5
32.7
44.1
Protocol 2
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg
Martinez et al. [17] ICCV’17
39.5
43.2
46.4
47.0
51.0
56.0
41.4
40.6
56.5
69.4
49.2
45.0
49.5
38.0
43.1
47.7
Sun et al. [6] ICCV’17
42.1
44.3
45.0
45.4
51.5
53.0
43.2
41.3
59.3
73.3
51.0
44.0
48.0
38.3
44.8
48.3
Pavlakos et al. [12] CVPR’18
34.7
39.8
41.8
38.6
42.5
47.5
38.0
36.6
50.7
56.8
42.6
39.6
43.9
32.1
36.5
41.8
Yang et al. [9] CVPR’18
26.9
30.9
36.3
39.9
43.9
47.4
28.8
29.4
36.9
58.4
41.5
30.5
29.5
42.5
32.2
37.7
Hossain & Little [4] ECCV’18
35.7
39.3
44.6
43.0
47.2
54.0
38.3
37.5
51.6
61.3
46.5
41.4
47.3
34.2
39.4
44.1
Chen et al. [3] CVPR’19
36.9
39.3
40.5
41.2
42.0
34.9
38.0
51.2
67.5
42.1
42.5
37.5
30.6
40.2
34.2
41.6
Pavllo et al. [1] (243 frames, Causal) CVPR’19
35.1
37.7
36.1
38.8
38.5
44.7
35.4
34.7
46.7
53.9
39.6
35.4
39.4
27.3
28.6
38.1
Pavllo et al. [1] (243 frames) CVPR’19
34.1
36.1
34.4
37.2
36.4
42.2
34.4
33.6
45.0
52.5
37.4
33.8
37.8
25.6
27.3
36.5
Lin et al. [2] BMVC’19
32.5
35.3
34.3
36.2
37.8
43.0
33.0
32.2
45.7
51.8
38.4
32.8
37.5
25.8
28.9
36.8
Cai et al. [34] ICCV’19
35.7
37.8
36.9
40.7
39.6
45.2
37.4
34.5
46.9
50.1
40.5
36.1
41.0
29.6
33.2
39.0
Cheng et al. [35] ICCV’19 (*)
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
34.1
Xu et al. [18] CVPR’20
31.0
34.8
34.7
34.4
36.2
43.9
31.6
33.5
42.3
49.0
37.1
33.0
39.1
26.9
31.9
36.2
Ours (243 frames, Causal)
33.6
36.0
34.4
36.6
37.5
42.6
33.5
33.8
44.4
51.0
38.3
33.6
37.7
26.7
28.2
36.5
Ours (81 frames)
33.1
35.3
33.4
35.9
36.1
41.7
32.8
33.3
42.6
49.4
37.0
32.7
36.5
25.5
27.9
35.6
Ours (243 frames)
32.6
35.1
32.8
35.4
36.3
40.4
32.4
32.3
42.7
49.0
36.8
32.4
36.0
24.9
26.5
35.0
MPJVE
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg
Pavllo et al. [1] (243 frames) CVPR’19
3.0
3.1
2.2
3.4
2.3
2.7
2.7
3.1
2.1
2.9
2.3
2.4
3.7
3.1
2.8
2.8
Ours (243 frames)
2.7
2.8
2.0
3.1
2.0
2.4
2.4
2.8
1.8
2.4
2.0
2.1
3.4
2.7
2.4
2.5
shows the visualized qualitative results from the baseline and
our full model on “Sitting” and “Sitting down” poses.
TABLE II
QUANTITATI

## related_work
3D human pose estimation has received much attention
in recent years. To predict the 3D joint location from 2D
image input, previous works of 3D pose estimation typically
fall into two categories based on the training pipeline. For
the approaches of the ﬁrst category, they created an end-to-
end convolutional neural network (CNN) model to directly
predict the 3D joint location from the original input images.
To establish a strong baseline, Pavlakos et al. [7] integrated
the volumetric representation with a coarse-to-ﬁne supervision
scheme to ﬁgure out the 3D joint locations by the predicted 3D
volumetric heat maps. Based on the ConvNet pose estimator
and the volumetric heap map representation proposed by [7],
recent approaches mainly made progress from two aspects.
On the one hand, human-structure constraints such as the
human shape constraints [8], body articulation constraints
[9] and the joint angle constraints [10] were employed to
prevent invalid pose prediction. On the other hand, effective
training approaches were proposed, making the estimation
process differentiable [11] and enabling the model to learn
from weakly labeled data [12], [13]. To further enable the
pose estimator to predict full 3D human mesh instead of the
joint locations, Kanazawa et al. [14], [15] proposed end-to-
end CNN frameworks for reconstructing the full 3D mesh of
a human body from an image or a video. These approaches
based on image-level input can directly capture rich knowledge
contained in images. However, without intermediate feature
and supervision, the model’s performance will also be affected
by the image’s background, lighting and person’s clothing.
More importantly, the large dimension of image-level input
disables the 3D model from receiving a large number of
images as input, bottlenecking the performance of 3D pose
estimation in video.
For the approaches of the second category, they built a
3D joint estimation model on top of a high-performance 2D
keypoint detector. Given an input image, these approaches ﬁrst
utilized the 2D keypoint detector to predict the image’ 2D
keypoints. The predicted 2D keypoints were then lifted as the
3D joint estimation model’s input to predict the ﬁnal 3D joint
locations. As an earlier work, Chen et al. [16] regarded the 3D
pose estimation as a matching problem. They found the best
matching 3D pose of the 2D keypoint input from the 3D pose
pool by a nearest-neighbor (NN) model. Considering that the
ground-truth 3D pose of the input may be non-corresponding
to all the 3D poses in the pool, Martinez et al. [17] proposed
an effective fully-connected residual network to regress the 3D
joint locations from 2D keypoint input. In addition to utilizing
effective human-structure information as the approaches of
the ﬁrst category, based on [17], recent approaches of this
category further improved the pose estimation performance by
hierarchical joint prediction [5], 2D keypoint reﬁnement [18]
and view-invariant constraint [3], [19]. Overall, the approaches
in such a “image-2D-3D” pipeline outperform the end-to-end
counterparts. One important reason is that the 2D detector can
be trained by large-scale indoor/outdoor images. It provides
the 3D model a strong intermediate feature to build upon.
When estimating the 3D poses in a video, recent approaches

3
exploited temporal information into the model to alleviate
incoherent predictions. As an earlier work, Mehta et al. [20]
applied simple temporal ﬁltering across 2D and 3D poses from
previous frames to predict a temporally consistent 3D pose.
As Long Short Term Memory networks (LSTM) were created
to adaptively capture information from temporal input by the
well-designed input gate, output gate and forget gate, Lin et al.
[21] presented the LSTM-based Recurrent 3D Pose Sequence
Machine. It automatically learns the image-dependent struc-
tural constraint and sequence-dependent temporal context by a
multi-stage sequential reﬁnement. Similar to [21], Rayat et al.
[4] predicted temporally consistent 3D poses by learning the
temporal context of a sequence using sequence-to-sequence
LSTM-based network. Considering the high computational
complexity of LSTM, Pavllo et al. [1] further introduced
a temporal fully-convolutional model which enables parallel
processing of multiple frames and supports very long 2D
keypoint sequence as input. All these approaches essentially
leverage the adjacent frames to beneﬁt the current frame’s
prediction. Compared with them, we are the ﬁrst to make
all the frames in a video contribute to the 3D prediction.
Motivated by [22], [23], [14] that created effective sub-tasks
for human pose estimation and mesh recovery, we propose
a novel solution to decompose the 3D pose estimation task
into two bone-based sub-tasks. It should be noticed that
Sun et al. [6] also transformed the 3D joint into a bone-
based representation. They trained the model to regress short
and long range relative shifts between different joints. We
demonstrate that completely decomposing the task into the
bone length and bone direction prediction achieves the best
performance and makes better use of the relative joint shift
supervision.

## conclusion
We present a new solution to estimating the human 3D
pose. Instead of directly regressing the 3D joint locations,
we transform the task into predicting the bone lengths and
directions. For bone length prediction, we make use of the
frames across the entire video and propose an effective fully-
connected residual network with a bone length re-weighting
mechanism. For bone direction prediction, we add along skip
connections into a fully-convolutional architecture for hierar-
chical prediction. Extensive experiments have demonstrated
that the combination of bone length and bone direction is
an effective intermediate representation to bridge the 2D
keypoints and 3D joint locations.
In recent years, as 3D human pose estimation has become
a signiﬁcant research topic for researchers to study, multiple
directions are demonstrated to be promising for exploration.
First, effective data augmentation algorithms [35], [38] are
continuously proposed to guide the model to handle occluded
or complex pose inputs. Moreover, the creation of genera-
tive adversarial network (GAN) [39] enables a number of
approaches [9], [40] to utilize GAN for realistic and rea-
sonable pose prediction, even in a weakly supervised setting.
In addition, high-performance temporal models [1], [4] are
created, which support very long 2D keypoint sequence as
input and can adaptively capture signiﬁcant information from
keyframes. These directions are regarded as general directions
since the proposed temporal models, adversarial training, and
data augmentation algorithms can be generally applied to
different research tasks other than 3D human pose estimation.
In this paper, we focus on a more fundamental aspect of human
pose estimation and create an effective learning representation
for this task. We believe that exploring the human pose’s
learning representation is promising as the human body is a
special Kinematic Tree-based structure different from other
objects. The motion of the human body is drove by joint
rotation with ﬁxed bone lengths. We are delighted to see the
human pose’s learning representation evolved from the joint
level to the bone vector level to the bone length/direction
level that constantly improves human pose estimation. Based
on our work, it may be illuminating for future works to
keep exploring the relationship between the tasks of bone
length prediction and bone direction prediction. Currently, we
adopt two independent networks to predict bone directions
and bone lengths. It is valuable to study whether the model
can further improve the performance of each task by utilizing
the knowledge captured from the other task. Applying the
joint shift loss is one useful way. However, we believe that
capturing this relationship at the network level as [41], [42]
for visual question answering and image-text matching will
make extra improvement for accurate and smooth 3D human
pose estimation in video.

11