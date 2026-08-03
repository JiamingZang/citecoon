# 3D Human Pose Machines with Self-supervised Learning

> 2019 · id: W2963102968 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
1
3D Human Pose Machines with
Self-supervised Learning
Keze Wang, Liang Lin, Chenhan Jiang, Chen Qian, and Pengxu Wei
Abstract—Driven by recent computer vision and robotic applications, recovering 3D human poses has become increasingly important
and attracted growing interests. In fact, completing this task is quite challenging due to the diverse appearances, viewpoints, occlusions
and inherently geometric ambiguities inside monocular images. Most of the existing methods focus on designing some elaborate priors
/constraints to directly regress 3D human poses based on the corresponding 2D human pose-aware features or 2D pose predictions.
However, due to the insufﬁcient 3D pose data for training and the domain gap between 2D space and 3D space, these methods have
limited scalabilities for all practical scenarios (e.g., outdoor scene). Attempt to address this issue, this paper proposes a simple yet
effective self-supervised correction mechanism to learn all intrinsic structures of human poses from abundant images. Speciﬁcally, the
proposed mechanism involves two dual learning tasks, i.e., the 2D-to-3D pose transformation and 3D-to-2D pose projection, to serve
as a bridge between 3D and 2D human poses in a type of “free” self-supervision for accurate 3D human pose estimation. The
2D-to-3D pose implies to sequentially regress intermediate 3D poses by transforming the pose representation from the 2D domain to
the 3D domain under the sequence-dependent temporal context, while the 3D-to-2D pose projection contributes to reﬁning the
intermediate 3D poses by maintaining geometric consistency between the 2D projections of 3D poses and the estimated 2D poses.
Therefore, these two dual learning tasks enable our model to adaptively learn from 3D human pose data and external large-scale 2D
human pose data. We further apply our self-supervised correction mechanism to develop a 3D human pose machine, which jointly
integrates the 2D spatial relationship, temporal smoothness of predictions and 3D geometric knowledge. Extensive evaluations on the
Human3.6M and HumanEva-I benchmarks demonstrate the superior performance and efﬁciency of our framework over all the
compared competing methods. Please ﬁnd the code of this project at: http://www.sysu-hcp.net/3d pose ssl/
Index Terms—human pose estimation, convolutional neural networks, spatio-temporal modeling, self-supervised learning, geometric
deep learning.
!
1
INTRODUCTION
R
ECENTLY, estimating 3D full-body human poses from
monocular RGB imagery has attracted substantial aca-
demic interests for its vast potential on human-centric
applications, including human-computer interactions [1],
surveillance [2], and virtual reality [3]. In fact, estimating
human pose from images is quite challenging with respect to
large variances in human appearances, arbitrary viewpoints,
invisibilities of body parts. Besides, the 3D articulated pose
recovery from monocular imagery is considerably more
difﬁcult since 3D poses are inherently ambiguous from a
geometric perspective [4], as shown in Fig. 1.
Recently, notable successes have been achieved for 2D
pose estimation based on 2D part models coupled with
2D deformation priors [6], [7], and the deep learning tech-
niques [8], [9], [10], [11]. Driven by these successes, some
3D pose estimation works [12], [13], [14], [15], [16], [17]
attempt to leverage the state-of-the-art 2D pose network
architectures (e.g., Convolutional Pose Machines (CPM) [10]
and Stacked Hourglass Networks [18]) by combing the
image-based 2D part detectors, 3D geometric pose priors
and temporal models. These attempts mainly follow three
types of pipelines. The ﬁrst type [19], [20], [21] focuses on
K. Wang is with the School of Data and Computer Science, Sun Yat-sen
University, Guangzhou, China, and also with the Department of Comput-
ing, The Hong Kong Polytechnic University, Hong Kong (e-mail: keze-
wang@gmail.com).
L. Lin, P. Wei, and C. Jiang are with the School of Data and Computer Science,
Sun Yat-sen University, Guangzhou, China. L. Lin is the corresponding
author; e-mail: linlang@ieee.org
C. Qian is with SenseTime Group.
(a) Intermediate Prediction
(b) Final Refinement
(c) Ground-truth 
Fig. 1: Some visual results of our approach on the Hu-
man3.6M benchmark [5]. (a) illustrates the intermediate 3D
poses estimated by the 2D-to-3D pose transformer module,
(b) denotes the ﬁnal 3D poses reﬁned by the 3D-to-2D pose
projector module, and (c) denotes the ground-truth. The es-
timated 3D joints are reprojected into the images and shown
by themselves from the side view (next to the images). As
shown, the predicted 3D poses in (b) have been signiﬁcantly
corrected, compared with (a). Best viewed in color. Note
that, red and green indicate left and right, respectively.
directly recovering 3D human poses from 2D input images
by utilizing the state-of-the-art 2D pose network architecture
to extract 2D pose-aware features with separate techniques
and prior knowledge. In this way, these methods can em-
arXiv:1901.03798v2  [cs.CV]  15 Jan 2019

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
2
ploy sufﬁcient 2D pose annotations to improve the shared
feature representation of the 3D pose and 2D pose estima-
tion tasks. The second type [16], [22], [23] concentrates on
learning a 2D-to-3D pose mapping function. Speciﬁcally, the
methodsbelonging to this kind ﬁrst extract 2D poses from
2D input images and further perform 3D pose reconstruc-
tion/regression based on these 2D pose predictions. The
third type [24], [25], [26] aims at integrating the Skinned
Multi-Person Linear (SMPL) model [27] within a deep net-
work to reconstruct 3D human pose and shape in a full 3D
mesh of human bodies. Although having achieved a promis-
ing performance, all of these kinds suffer from the heavy
computational cost by using the time-consuming network
architecture (e.g., ResNet-50 [28]) and limited scalability for
all scenarios due to the insufﬁcient 3D pose data.
To address the above-mentioned issues and utilize the
sufﬁcient 2D pose data for training, we propose an effective
yet efﬁcient 3D human pose estimation framework, which
implicitly learns to integrate the 2D spatial relationship,
temporal coherency and 3D geometry knowledge by uti-
lizing the advantages afforded by Convolutional Neural
Networks (CNNs) [10] (i.e., the ability to learn feature
representations for both image and spatial context directly
from data), recurrent neural networks (RNNs) [29] (i.e., the
ability to model the temporal dependency and prediction
smoothness) and the self-supervised correction (i.e., the
ability to implicitly retain 3D geometric consistency between
the 2D projections of 3D poses and the predicted 2D poses).
Concretely, our model employs a sequential training to cap-
ture long-range temporal coherency among multiple human
body parts, and it is further enhanced via a novel self-
supervised correction mechanism, which involves two dual
learning tasks, i.e., 2D-to-3D pose transformation and 3D-to-
2D pose projection, to generate geometrically consistent 3D
pose predictions under a self-supervised correction mech-
anism, i.e., forcing the 2D projections of the generated 3D
poses to be identical to the estimated 2D poses.
As illustrated in Fig. 1, our model enables the grad-
ual reﬁnement of the 3D pose prediction for each frame
according to the coherency of sequentially predicted 2D
poses and 3D poses, contributing to seamlessly learning
the pose-dependent constraints among multiple body parts
and sequence-dependent context from the previous frames.
Speciﬁcally, taking each frame as input, our model ﬁrst
extracts the 2D pose representations and predicts the 2D
poses. Then, the 2D-to-3D pose transformer module is in-
jected to transform the learned pose representations from
the 2D domain to the 3D domain, and it further regresses
the intermediate 3D poses via two stacked long short-
term memory (LSTM) layers by combining the following
two lines of information, i.e., the transformed 2D pose
representations and the learned states from past frames.
Intuitively, the 2D pose representations are conditioned on
the monocular image, which captures the spatial appear-
ance and context information. Then, temporal contextual
dependency is captured by the hidden states of LSTM units,
which effectively improves the robustness of the 3D pose
estimations over time. Finally, the 3D joint prediction implic-
itly encodes the 3D geometric structural information by the
3D-to-2D pose projector module under the introduced self-
supervised correction mechanism. In speciﬁc, considering
that the 2D projections of 3D poses and the predicted 2D
poses should be identical, the minimization of their dis-
similarities is regarded as a learning objective for the 3D-
to-2D pose projector module to bidirectionally correct (or
reﬁne) the intermediate 3D pose predictions. Through this
self-supervised correction mechanism, our model is capable
of effectively achieving geometrically coherent 3D human
pose predictions without requesting additional 3D pose an-
notations. Therefore, our introduced correction mechanism
is self-supervised, and can enhance our model by adding the
external large-scale 2D human pose data into the training
process to cost-effectively increase the 3D pose estimation
performance.
The main contributions of this work are three-fold. i)
We present a novel model that learns to integrate rich
spatial and temporal long-range dependencies as well as 3D
geometric constraints, rather than relying on speciﬁc man-
ually deﬁned body smoothness or kinematic constraints; ii)
Developing a simple yet effective self-supervised correction
mechanism to incorporate 3D pose geometric structural
information is innovative in literature, and may also inspire
other 3D vision tasks; iii) The proposed self-supervised
correction mechanism enables our model to signiﬁcantly
improve 3D human pose estimation via sufﬁcient 2D hu-
man pose data. Extensive evaluations on the public chal-
lenging Human3.6M [5] and HumanEva-I [30] benchmarks
demonstrate the superiority of our framework over all the
compared competing methods.
The remainder of this paper is organized as follows. Sec-
tion 2 brieﬂy reviews the existing 3D human pose estimation
approaches that motivate this work. Section 3 presents the
details of the proposed model, with a thorough analysis
of every component. Section 4 presents the experimental
results on two public benchmarks with comprehensive eval-
uation protocols, as well as comparisons with competing
alternatives. Finally, Section 5 concludes this paper.
2
RELATED WORK
Considerable research has addressed the challenge of 3D hu-
man pose estimation. Early research on 3D monocular pose
estimation from videos involved frame-to-frame pose track-
ing and dynamic models that rely on Markov dependencies
among previous frames, e.g., [31], [32]. The main drawbacks
of these approaches are the requirement of the initialization
pose and the inability to recover from tracking failure. To
overcome these drawbacks, more recent approaches [12],
[33] focus on detecting candidate poses in each individual
frame, and a post-processing step attempts to establish tem-
porally consistent poses. Yasin et al. [22] proposed a dual-
source approach for 3D pose estimation from a single image.
They combined the 3D pose data from a motion capture
system with an image source annotated with 2D poses. They
transformed the estimation into a 3D pose retrieval problem.
One major limitation of this approach is its time efﬁciency.
Processing an image requires more than 20 seconds. Sanzari
et al. [34] proposed a hierarchical Bayesian non-parametric
model, which relies on a representation of the idiosyncratic
motion of human skeleton joint groups, and the consistency
of the connected group poses is considered during the pose
reconstruction.

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
3
Deep learning has recently demonstrated its capabilities
in many computer vision tasks, such as 3D human pose
estimation. Li and Chan [35] ﬁrst used CNNs to regress
the 3D human pose from monocular images and proposed
two training strategies to optimize the network. Li et al.
[36] proposed integrating the structure learning into a deep
learning framework, which consists of a convolutional neu-
ral network to extract image features and two following
subnetworks to transform the image features and poses into
a joint embedding. Tekin et al. [15] proposed exploiting
motion information from consecutive frames and applied
a deep learning network to regress the 3D pose. Zhou et al.
[14] proposed a 3D pose estimation framework from videos
that consists of a novel synthesis among a deep-learning-
based 2D part detector, a sparsity-driven 3D reconstruction
approach and a 3D temporal smoothness prior. Zhou et al.
[4] proposed directly embedding a kinematic object model
into the deep learning network. Du et al. [37] introduced
additional built-in knowledge for reconstructing the 2D
pose and formulated a new objective function to estimate
the 3D pose from the detected 2D pose. More recently, Zhou
et al. [19] presented a coarse-to-ﬁne prediction scheme to
cast 3D human pose estimation as a 3D keypoint localization
problem in a voxel space in an end-to-end manner. Moreno-
Noguer et al. [38] formulated the 3D human pose estimation
problem as a regression between matrices encoding 2D and
3D joint distances. Chen et al. [16] proposed a simple ap-
proach to 3D human pose estimation by performing 2D pose
estimation followed by 3D exemplar matching. Tome et al.
[20] proposed a multi-task framework to jointly integrate 2D
joint estimation and 3D pose reconstruction to improve both
tasks. To leverage the well-annotated large-scale 2D pose
datasets, Zhou et al. [23] proposed a weakly-supervised
transfer learning method that uses mixed 2D and 3D labels
in a uniﬁed deep two-stage cascaded structure network.
However, these methods oversimplify the 3D geometric
knowledge. In contrast to all these aforementioned methods,
our model can leverage a lightweight network architecture
to implicitly learn to integrate the 2D spatial relationship,
temporal coherency and 3D geometry knowledge in a fully
differential manner.
Instead of directly computing 2D and 3D joint locations,
several works concentrate on producing a 3D mesh body
representation by using a CNN to predict Skinned Multi-
Person Linear model [27]. For instance, Omran et al. [25]
proposed to integrate a statistical body model within a
CNN, leveraging reliable bottom-up semantic body part
segmentation and robust top-down body model constraints.
Kanazawa et al. [26] presented an end-to-end adversarial
learning framework for recovering a full 3D mesh model
of a human body by parameterizing the mesh in terms of
3D joint angles and a low dimensional linear shape space.
Furthermore, this method employs the weak-perspective
camera model to project the 3D joints onto the annotated
2D joints via an iterative error feedback loop [39]. Similar
to our proposed method, these approaches also regard the
in-the-wild images with 2D ground-truth as the supervision
to improve the model performance. The main difference is
that our self-supervised learning method is more ﬂexible
and robust without relying on the assumption of the weak-
perspective camera model.
Our approach is close to [20], which also used the pro-
jection from the 3D space to the 2D space to improve the 3D
pose estimation performance. However, there are two main
differences between [20] and our model: i) The deﬁnition
of the 3D-to-2D projection function and the optimization
strategy. Rather than explicitly deﬁning a concrete model,
our 3D-to-2D projection is implicitly learned in a completely
data-driven manner. However, the projection of 3D poses
in [20] is explicitly modeled by using a weak perspective
model, which consists of the orthographic projection ma-
trix, a known external camera calibration matrix and an
unknown rotation matrix. As claimed in [20], this explicit
model is prone to sticking in local minima during the train-
ing. Thus, the authors have to quantize over the space of
possible rotations. Through this approximation, their model
performance may suffer from the ﬁxed choices of rotations;
ii) The way of utilizing the projected 2D pose. In contrast
to [20] which learns to weightily fuse the projected 2D
and the estimated 2D poses for further regressing the ﬁnal
3D pose, our model exploits the 3D geometric consistency
between the projected 2D and the estimated 2D poses to
bidirectionally reﬁne the intermediate 3D pose predictions.
Self-supervised Learning. Aiming at training the feature
representation without relying on manual data annotation,
self-supervised learning (SSL) has ﬁrst been introduced in
[40] for vowel class recognition, and further extended for
object extraction in [41]. Recently, plenty of SSL methods
(e.g., [42], [43]) have been proposed. For instance, [42]
investigated multiple self-supervised methods to encourage
the network to factorize the information in its representa-
tion. In contrast to these methods that focus on learning
an optimal visual representation, our work considers the
self-supervision as an optimization guidance for 3D pose
estimation.
Note that a preliminary version of this work was pub-
lished in [21], which uses multiple stages to gradually
reﬁne the predicted 3D poses. The network parameters
in the multiple stages are recurrently trained in a fully
end-to-end manner. However, the multi-stage mechanism
results in a heavy computational cost, and the stage-by-
stage improvement is less signiﬁcant as the number of stages
increases. In this paper, we inherit its idea of integrating the
2D spatial relationship, temporal coherency as well as 3D
geometry knowledge, and we further impose a novel self-
supervised correction mechanism to further enhance our
model by bridging the domain gap between the 3D and
2D human poses. Speciﬁcally, we develop a 3D-to-2D pose
projector module to replace the multi-stage reﬁnement to
correct the intermediate 3D pose predictions by retaining the
3D geometric consistency between their 2D projections and
the predicted 2D poses. Therefore, the imposed correction
mechanism enables us to leverage the external large-scale
2D human pose data to boost 3D human pose estimation.
Moreover, more comparisons with competing approaches
and more detailed analyses of the proposed modules are
included to further verify our statements.
3
3D HUMAN POSE MACHINE
We propose a 3D human pose machine to resolve 3D pose
sequence generation for monocular frames, and we intro-

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
4
(a) 2D-to-3D Pose Transformer 
LSTM
FC
LSTM
Hidden states 
from past frames
128
5
5
46
46
128
128
23
11
23
11
5
5
FC
Intermediate 3D 
Prediction
1024
1024
3K
1024
FC
(b) 3D-to-2D Pose Projector
Final 3D 
Prediction
1024
FC
1024
3K
2K
2K
Predicted
2D pose 
1024
FC
2D Projection
of 3D Pose 
Self-supervised Correction
Loss
FC
2D Pose Representation
Input Frame
FC
1024
...
ˆp2d
t
p2d
t
f  2 d
t
ˆp3d
t
ˆp*3  d
t
p3d
t
Concatenation
Fig. 2: An overview of the proposed 3D human pose machine framework. Our model predicts the 3D human poses
for the given monocular image frames, and it progressively reﬁnes its predictions with the proposed self-supervised
correction. Speciﬁcally, the estimated 2D pose p2d
t
with the corresponding pose representation f 2d
t
for each frame of the
input sequence is ﬁrst obtained and further passed into two neural network modules: i) a 2D-to-3D pose transformer
module for transforming the pose representations from the 2D domain to the 3D domain to intermediately predict the
human joints p3d
t
in the 3D coordinates, and ii) a 3D-to-2D pose projector module to obtain the projected 2D pose ˆp2d
t
after
regressing p3d
t
into ˆp3d
t . Through minimizing the difference between p2d
t
and ˆp2d
t , our model is capable of bidirectionally
reﬁning the regressed 3D poses ˆp3d
t
via the proposed self-supervised correction mechanism. Note that the parameters of
the 2D-to-3D pose transformer module for all frames are shared to preserve the temporal motion coherence. 3K and 2K
denotes the dimension of the vector for representing the 3D and 2D human pose formed by K skeleton joints, respectively.
duce a concise self-supervised correction mechanism to en-
hance our model by retaining the 3D geometric consistency.
After extracting the 2D pose representation and estimating
the 2D poses for each frame via a common 2D pose sub-
network, our model employs two consecutive modules. The
ﬁrst module is the 2D-to-3D pose transformer module for trans-
forming the 2D pose-aware features from the 2D domain to
the 3D domain. This module is designed to estimate inter-
mediate 3D poses for each frame by incorporating temporal
dependency in the image sequence. The second module is
the 3D-to-2D pose projector module for bidirectionally reﬁning
the intermediate 3D pose prediction via our introduced self-
supervised correction mechanism. These two modules are
combined in a uniﬁed framework to be optimized in a fully
end-to-end manner.
As illustrated in Fig. 2, our model performs the sequen-
tial reﬁnement with self-supervised correction to generate
the 3D pose sequence. Speciﬁcally, the t-th frame It is
passed into the 2D pose sub-network ΨR, the 2D-to-3D pose
transformer module ΨT , and the 3D-to-2D projector module
{ΨC, ΨP } to predict the ﬁnal 3D poses. The 2D pose sub-
network is stacked by convolutional and fully connected
layers, and the 2D-to-3D pose transformer module contains
two LSTM layers to capture the temporal dependency over
frames. Speciﬁcally, given the input image sequence with N
frames, the 2D pose sub-network ΨR is ﬁrst employed to
extract the 2D pose-aware features f 2d
t
and predict the 2D
pose p2d
t
for the t-th frame of the input sequence. Then,
the extracted 2D pose-aware features f 2d
t
are further fed
into the 2D-to-3D pose transformer module ΨT to obtain
the intermediate 3D pose p3d
t
, where ΨT is composed of the
hidden states Ht−1 learned from the past frames. Finally, the
predicted 2D poses p2d
t
and intermediate 3D pose p3d
t
are fed
into the 3D-to-2D projector module with two functions, i.e.,
ΨC and ΨP , to obtain the ﬁnal 3D poses ˆp∗3d
t
. Considering
that most existing 2D human pose data are still images
without temporal orders, we additionally introduce a simple
yet effective regression function ΨC to transform the inter-
mediate 3D pose vector p3d
t
into a changeable prediction
ˆp3d
t . The projection function ΨP implies projecting the 3D
coordinate ˆp3d
t
into the image plane to obtain the projected
2D pose ˆp2d
t . Formally, f 2d
t , p3d
t , ˆp3d
t , and ˆp2d
t
are formulated
as follows:
{f 2d
t , p2d
t } = ΨR(It; ωR),
p3d
t
= ΨT (f 2d
t ; ωT , Ht−1),
ˆp3d
t
= ΨC(p3d
t ; ωC),
ˆp2d
t
= ΨP (ˆp3d
t ; ωP ),
(1)
where ωR, ωT , ωC and ωP are parameters of ΨR, ΨT , ΨC
and ΨP , respectively. Note that, H0 is initially set to be a
vector of zeros. After obtaining the predicted 2D pose p2d
t
via ΨR, and the projected 2D pose ˆp2d
t
via ΨP in Eq. (1), we
consider minimizing the dissimilarity between p2d
t
and ˆp2d
t

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
5
Input 3D Prediction
3K
1024
FC
1024
FC
1024
FC
FC
2K
FC
1024
3D Ground-truth
3K
(a) Training Phase
Input 3D Prediction
3K
2D Ground-truth
2D Prediction
1024
1024
1024
2K
1024
(b) Testing Phase
Final 3D 
Prediction
3K
3K
Regression Function
Projection Function
Bidirectionally
Refine 3D Pose
p3d
t
p3d (gt)
t
p2d (gt)
t
p3d
t
p2d
t
ˆp*3  d
t
ˆp3d
t
ˆp3d
t
Fig. 3: Detailed sub-network architecture of our proposed 3D-to-2D pose projector module in the (a) training phase and
(b) testing phase. The Fully Connected (FC) layers for the regression function are in blue, while those for the projection
function are in yellow. The black arrows represent the forward data ﬂow, while the dashed arrows denote the backward
propagation used to update the network parameters and perform gradual pose reﬁnement in (a) and (b), respectively.
as an optimization objective to obtain the optimal ˆp∗3d
t
for
the t-th frame.
In the following, we will introduce more details of our
model and provide comprehensive clariﬁcations to make the
work easier to understand. The corresponding algorithm for
jointly training these modules will also be discussed at the
end.
3.1
2D Pose Sub-network
The objective of the 2D pose sub-network is to encode
each frame in a given monocular sequence with a compact
representation of the pose information, e.g., the body shape
of the human. The shallow convolution layers often extract
the common low-level information, which is a very basic
representation of the human image. We build our 2D pose
sub-network by borrowing the architecture of the convo-
lutional pose machines [10]. Please see Table 1 for more
details. Note that other state-of-the-art architectures for 2D
pose estimation can be also utilized. As illustrated in Fig. 2,
the 2D pose sub-network takes the 368×368 image as input,
and it outputs the 2D pose-aware feature maps with a size
of 128 × 46 × 46 and the predicted 2D pose vectors with 2K
entries being the argmax positions of these feature maps.
3.2
2D-to-3D Pose Transformer Module
Based on the features extracted by the 2D pose sub-network,
the 3D pose transformer module is employed to adapt the
2D pose-aware features in an adapted feature space for
the later 3D pose prediction. As depicted in Fig. 2 (a),
two convolutional layers and one fully connected layer are
leveraged. Each convolutional layer contains 128 different
kernels with a size of 5 × 5 and a stride of 2, and a
max pooling layer with a 2 × 2 kernel size and a stride
of 2 is appended on the convolutional layers. Finally, the
convolution features are fed to a fully connected layer with
1024 units to produce the adapted feature vector. In this
way, the 2D pose-aware features are transformed into the
1024-dimensional adapted feature vector.
Given the adapted features for all frames, we employ
LSTM to sequentially predict the 3D pose sequence by
incorporating rich temporal motion patterns among frames
as [21]. Note that, LSTM [29] has been proven to achieve bet-
ter performance in exploiting temporal correlations than a
vanilla recurrent neural network in many tasks, e.g., speech
recognition [44] and video description [45]. In our model, we
use the LSTM layers to capture the temporal dependency in
the monocular sequence for reﬁning the 3D pose prediction
for each frame. As illustrated in Fig. 2 (b), our model
employs two LSTM layers with 1024 hidden cells and an
output layer that predicts the locations of K joint points
of the human. In particular, the hidden states learned by the
LSTM layers are capable of implicitly encoding the temporal
dependency across different frames of the input sequence.
As formulated in Eq. (1), incorporating the previous hidden
states imparts our model with the ability to sequentially
reﬁne the pose predictions.
3.3
3D-to-2D Projector Module
As illustrated in Fig. 3 (a), this module consists of six
fully connected (FC) layers containing ReLU and batch
normalization operations. As one can see from left to right
in Fig. 3(a), the ﬁrst two FC layers (denoted in blue) deﬁne
the regression function ΨC in which the intermediate 3D
pose predictions are regressed into the pose prediction ˆp3d
t ,
and the remaining four FC layers (denoted in yellow) with
1024 units represent the projection function ΨP that projects
ˆp3d
t
into the image plane to obtain the projected 2D pose
ˆp2d
t . Moreover, an identical mapping as ResNet [28] is used
inside ΨP to make the information pass through quickly to
avoid overﬁtting. Therefore, our 3D-to-2D projector module
is simple yet powerful for both regression and projection
tasks. Considering the self-corrected 3D pose may need to
be discarded sometimes, we regard the regression function
ΨC as a copy to be corrected for the intermediate 3D poses.
In the training phase, we ﬁrst initialize the module pa-
rameters {ωC, ωP } for ΨC and ΨP via the supervision of the
3D and 2D ground-truth poses from 3D human pose data
as illustrated in Fig. 3 (a), respectively. The optimization
function is:
min
{ωC,ωP }
N
X
t=1
ˆp3d
t
−p3d(gt)
t

2
2 +
ΨP (ˆp3d
t ; ωP ) −p2d(gt)
t

2
2 , (2)
where ˆp3d
t
is the regressed 3D pose via ΨC in Eq. (1), and
its 2D projection is ˆp2d
t
= ΨP (ˆp3d
t ; ωP ). Eq. (2) forces ΨC
to regress ˆp3d
t
from intermediate 3D poses p3d
t
to the 3D
pose ground-truth p3d(gt)
t
, and it further forces the output
of ΨP , i.e., the projected 2D poses, to be similar to the 2D

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
6
TABLE 1: Details of the convolutional layers in the 2D pose sub-network.
1
2
3
4
5
6
7
8
9
Layer Name
conv1 1
conv1 2
max 1
conv2 1
conv2 2
max 2
conv3 1
conv3 2
conv3 3
Channel (kernel-stride)
64(3-1)
64(3-1)
64(2-2)
128(3-1)
128(3-1)
128(2-2)
256(3-1)
256(3-1)
256(3-1)
10
11
12
13
14
15
16
17
18
Layer Name
conv3 4
max 3
conv4 1
conv4 2
conv4 3
conv4 4
conv4 5
conv4 6
conv4 7
Channel (kernel-stride)
256(3-1)
256(2-2)
512(3-1)
512(3-1)
256(3-1)
256(3-1)
256(3-1)
256(3-1)
128(3-1)
pose ground-truth p2d(gt)
t
. In this way, the 3D-to-2D pose
projector module can learn the geometric consistency to
correct intermediate 3D pose predictions. After initializa-
tion, we substitute the predicted 2D poses and 3D poses
for the 2D and 3D ground-truth to optimize ΨC and ΨP in
a self-supervised fashion. Considering that the predictions
for certain body joints (e.g., handleft, handright, footleft and
footright deﬁned in the Human3.6M dataset) may not be
accurate and reliable due to the challenging nature of the
rich ﬂexibilities and occlusions of body joints, we employ
the dropout trick [46] in the intermediate 3D pose estima-
tions p3d
t
and the predicted 2D pose p2d
t , i.e., the position
for each body joint has a probability δ to be zero. This trick
enables the regression function ΨC and the project function
ΨP to be insensitive to the outliers inside p3d
t
and p2d
t . As
reported in [46], the dropout trick can signiﬁcantly con-
tribute to alleviating the overﬁtting of the fully connected
layers inside ΨC and ΨP . Meanwhile, we also employ the
3D pose ground-truth to encourage the regression function
ΨC to learn to regress the 3D pose estimation ˆp3d
t . In our
experiments, δ is empirically set to be 0.3.
The
inference
phase
of
this
module
is
also
self-
supervised. Speciﬁcally, given the predicted 2D pose p2d
t , we
can obtain the initial ˆp3d
t
and the corresponding projected
2D pose ˆp2d
t
via forward propagation, as indicated in Fig. 3
(b). According to the 3D geometric consistency that the pro-
jected 2D pose ˆp2d
t
should be identical to the predicted 2D
pose p2d
t , we propose minimizing the dissimilarity between
ˆp2d
t
and p2d
t
by optimizing its speciﬁc ωt
P and ωt
C as follows:
{ω∗t
P , ω∗t
C } = arg min
{ˆp3d
t ,ωt
P }
∥p2d
t −ˆp2d
t ∥2
2
= arg min
{ˆp3d
t ,ωt
P }
∥p2d
t −ΨP (ˆp3d
t ; ωt
P )∥2
2
= arg min
{ωt
P ,ωt
C}
∥p2d
t −ΨP (ΨC(p3d
t ; ωt
C); ωt
P )∥2
2,
(3)
where the parameters {ωt
P , ωt
C} are initialized from the
well-optimized {ωP , ωC} from the training phase. Note that,
ω∗t
C and ω∗t
P are disposable and only valid for It. Since p2d
t
and p3d
t
are ﬁxed, we ﬁrst perform forward propagation
to obtain the initial prediction, and further employ the
standard back-propagation algorithm [47] to obtain ω∗t
C and
ω∗t
P via Eq. (3). Thus, the output 3D pose regression ˆp3d
t
is bidirectionally reﬁned to be the ﬁnal 3D pose prediction
during the the optimizing of ωt
C and ωt
P according to the
proposed self-supervised correction mechanism. At the end,
the ﬁnal 3D pose ˆp∗3d
t
is obtained according to Eq. (3) as
follows:
ˆp∗3d
t
= ΨC(p3d
t ; ω∗t
C ).
(4)
The hyperparameters (i.e., the iteration number and
learning rate) for ω∗t
P and ω∗t
C play a crucial role in effectively
and efﬁciently reﬁning the 3D pose estimation. In fact, a
large iteration number and small learning rate can ensure
that the model is capable of converging to a satisfactory
ˆp∗3d
t
. However, this setting results in a heavy computational
cost. Therefore, a small iteration number with large learning
rate is preferred to achieve a trade-off between efﬁciency
and accuracy. Moreover, although we can achieve high
accuracy on 2D pose estimation, the predicted 2D poses may
contain errors due to the heavy occlusion of human body
parts. Treating these inaccurate 2D poses as optimization
objectives to bidirectionally reﬁne the 3D pose prediction is
prone to a decrease in performance. To address this issue,
we utilize a heuristic strategy to determine the optimal
hyperparameters used for each frame in our implementa-
tion. Speciﬁcally, we can check the convergence of some ro-
bust skeleton joints (i.e., Pelvis, Shoulderleft, Shoulderright,
Hipleft and Hipright deﬁned in the Human3.6M dataset)
in each iteration. In practice, we ﬁnd that the predictions
of these reliable joints are generally less ﬂexible and have
lower probabilities of being occluded than other joints. If the
predicted 2D pose contains small errors, then these joints of
the reﬁned 3D pose ˆp∗3d
t
will have large and inconsistent
changes within the self-supervised correction. Hence, we
terminate the further reﬁnement when the positions of these
joints are converged (i.e., average changes < ϵmm), and dis-
card the self-supervised correction when the average change
in these joints are not within an empirical threshold τmm.
In our experiments, we empirically set {τ, ϵ} = {20, 5} and
employ two back-propagation operations to update ωP and
ωC before outputting the ﬁnal 3D pose prediction ˆp∗3d
t
.
3.4
Model Training
In the training phase, the optimization of our proposed
model occurs in a fully end-to-end manner, and we have de-
ﬁned several types of loss functions to ﬁne-tune the network
parameters: ωR, ωT and {ωC, ωP }, respectively. For the 2D
pose sub-network, we build an extra FC layer upon the
convolutional layers of the 2D pose sub-network to generate
2K joint location coordinates. We leverage the Euclidean
distances between the predictions for all K body joints and
the corresponding ground-truth to train ωR. Formally, we
have:
min
ωR
N
X
t=1
p2d
t (gt) −ΨR(It; ωR)

2
2 ,
(5)
where p2d
t (gt) denotes the 2D pose ground-truth for the t-th
frame It.
For the 2D-to-3D pose transformer module, our model
enforces the 3D pose sequence prediction loss for all frames,
which is also deﬁned as follows:
min
ωT
N
X
t=1
p3d
t −p3d(gt)
t

2
2
=
N
X
t=1
ΨT (f 2d
t ; ωT , Ht−1) −p3d(gt)
t

2
2 ,
(6)

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 2019.
7
Algorithm 1 The Proposed Training Algorithm
Input: 3D human pose data {I3d
t }N
t=1 and 2D human pose
data {I2d
i }M
i=1
1: Pre-train the 2D pose sub-network with {I2d
i }M
i=1 to
initialize ωR via Eq. (5);
2: Fixing ωR, initialize ωT with hidden variables H with
{I3d
t }N
t=1 via Eq. (6);
3: Fixing ωR and ωT , initialize {ωC, ωP } with {I3d
t }N
t=1 via
Eq. (2);
4: Fine-tune the whole model to further update {ωR, ωT ,
ωC, ωP } on {I3d
t }N
t=1 and {I2d
i }M
i=1 via Eq. (7).
5: return {ωR, ωT , ωC, ωP }.
where p3d(gt)
t
is the 3D pose ground-truth for the t-th
frame It. According to Eq. (6), we integrally ﬁne-tune the
parameters of the 2D-to-3D pose transformer module and
the convolutional layers of the 2D pose sub-network in
an end-to-end optimization manner. Note that, to obtain
sufﬁcient samples to train the 3D pose transformer module,
we propose decomposing one long monocular image se-
quence into several small equal clips with N frames. In our
experiments, we jointly feed our model with 2D and 3D pose
data after all the network parameters are well initialized. For
the 2D human pose data, this module regards their 3D pose
sequence prediction loss as zero.
After initializing the 3D-to-2D projector module via
Eq. (2), we ﬁne-tune the whole network to jointly optimize
the network parameters {ωR, ωT , ωC, ωP } in a fully end-to-
end manner as follows:
min
{ωR,ωT ,ωC,ωP } ∥p2d
t −ˆp2d
t ∥2
2.
(7)
Since our model consists of two cascaded modules, the
training phase can be divided into the following steps: (i)
Initialize the 2D pose representation via pre-training. To
obtain a satisfactory feature representation, the 2D pose
sub-network is ﬁrst pre-trained with the MPII Human Pose
dataset [48], which includes a larger variety of 2D pose
data. (ii) Initialize the 2D-to-3D pose transformer module.
We ﬁx the parameters of the 2D pose sub-network and
optimize the network parameter ωT . (iii) Initialize the 3D-
to-2D pose projector module. We ﬁx the above optimized
parameters and optimize the network parameter {ωC, ωP }.
(iv) Fine-tune the whole model jointly to further update
the network parameters {ωR, ωT , ωC, ωP } with the 2D
pose and 3D pose training data. For each of the above-
mentioned steps, the ADAM [49] strategy is employed for
parameter optimization. The entire algorithm can then be
summarized as Algorithm 1. Obviously, this algorithm is in
a good agreement with the pipeline of our model.
3.5
Model Inference
In the testing phase, every frame of the input image se-
quence is sequentially processed via Eq. (1). Note that each
frame It has its own {ωt
C, ωt
P } in the 3D-to-2D projector
module. {ωt
C, ωt
P } are initialized from the well trained {ωC,
ωP }, and they will be updated by minimizing the difference
between the predicted 2D poses p2d
t
and projected 2D poses
ΨP (ˆp3d
t ; ωP ) via Eq. (3). During the inference, the 3D pose
estimation is bidirectionally reﬁned until convergence is
achieved according to the hyperparameter settings. Finally,
we output the ﬁnal 3D pose estimation via Eq. (4).
4
EXPERIMENTS
4.1
Experimental Settings
We perform extensive evaluations on two publicly available
benchmarks: Human3.6M [5] and HumanEva-I [30].
Human3.6M dataset. The Human3.6M dataset is a re-
cently published dataset that provides 3.6 million 3D hu-
man pose images and corresponding annotations from a
controlled laboratory environment. This dataset captures
11 professional actors performing in 15 scenarios under 4
different viewpoints. Moreover, there are three popular data
partition protocols for this benchmark in the literature.
•
Protocol #1: The data from ﬁve subjects (S1, S5, S6,
S7, and S8) are for training, and the data from two
subjects (S9 and S11) are for testing. To increase
the number of training samples, the sequences from
different viewpoints of the same subject are treated
as distinct sequences. By downsampling the frame
rate from 50 FPS to 2 FPS, 62,437 human pose images
(104 images per sequence) are obtained for training
and 21,911 images are obtained for testing (91 images
per sequence). This is the widely used evaluation
protocol on Human3.6M, and it was followed by sev-
eral works [4], [15], [16], [36]. To be more general and
make a fair comparison, our model is trained both
on training samples from all 15 actions as previous
works [4], [15], [16], [36] and by exploiting individual
actions as [14], [36].
•
Protocol #2: This protocol only differs from Protocol
#1 in that only the frontal view is considered for test-
ing, i.e., testing is performed on every 5-th frame of
the sequences from the frontal camera (cam-3) from
trial 1 of each activity with ground-truth cropping.
The training data contain all actions and viewpoints.
•
Protocol #3: Six subjects (S1, S5, S6, S7, S8 and S9)
are used for training, and every 64-th frame of S11’s
video clips is used for testing. The training data
contain all actions and viewpoints.
HumanEva-I dataset. The HumanEva-I dataset contains
video sequences of four subjects performing six common
actions (e.g., walking, jogging, boxing