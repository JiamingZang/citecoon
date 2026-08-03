# Feature Boosting Network For 3D Pose Estimation

> 2019 · id: W2963225971 · arXiv: 1901.04877 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this paper, a feature boosting network is proposed
for estimating 3D hand pose and 3D body pose from a
single RGB image.
In this method, the features learned
by the convolutional layers are boosted with a new long
short-term dependence-aware (LSTD) module, which en-
ables the intermediate convolutional feature maps to per-
ceive the graphical long short-term dependency among dif-
ferent hand (or body) parts using the designed Graphical
ConvLSTM. Learning a set of features that are reliable and
discriminatively representative of the pose of a hand (or
body) part is difﬁcult due to the ambiguities, texture and
illumination variation, and self-occlusion in the real appli-
cation of 3D pose estimation. To improve the reliability of
the features for representing each body part and enhance
the LSTD module, we further introduce a context consis-
tency gate (CCG) in this paper, with which the convolu-
tional feature maps are modulated according to their con-
sistency with the context representations. We evaluate the
proposed method on challenging benchmark datasets for
3D hand pose estimation and 3D full body pose estimation.
Experimental results show the effectiveness of our method
that achieves state-of-the-art performance on both of the
tasks.

## introduction
3D pose estimation (estimating the locations of the joints
of the human hand or body in 3D space) is a challenging
and fast-growing research area, thanks to its wide applica-
tions in gesture recognition, activity understanding, human-
machine interaction, etc. [2]. Most of the existing works
make use of highly constrained conﬁgurations [4], such as
multi-view systems [15] and depth sensors [28], to infer the
3D poses. In this paper, we address the problem of 3D pose
estimation from a single RGB image that is much easier to
be captured in uncontrolled environments [23, 25, 32, 44].
This task is challenging due to the ambiguities in recover-
ing the 3D information from a single 2D image, the complex
articulations and frequent occlusions of the hand (or body)
parts, and the large variation of clothing textures, camera
viewpoints, and lighting conditions, etc.
Convolutional neural networks (CNNs) demonstrate
their superior performance in various machine vision tasks,
such as image classiﬁcation and video analysis [29]. Re-
cently, they have also been successfully applied to 3D pose
estimation [19,25,30,32,39,42,44]. In this paper, we con-
struct our framework based on a CNN architecture.
Previous work on 3D pose estimation has shown the ben-
eﬁts of using the connection information of the body parts
to reﬁne the pose estimation results or lift 2D pose to 3D
space [23]. In this paper, we incorporate the complex de-
pendency and correlation information among different parts
to the convolutional features that contain very rich and rep-
resentative information. Speciﬁcally, a novel long short-
term dependence-aware (LSTD) module is proposed, which
is embedded inside the CNN architecture to boost the inter-
mediate convolutional feature maps for 3D pose estimation.
Our LSTD module is constructed based on the designed
graphical convolutional long short-term memory (Graphical
ConvLSTM). In the image of a human hand (or body), there
are complex dependency patterns among different parts.
Some joints are physically connected and obviously cor-
related, while some others can have indirect correlation in
their motion and appearance. In order to utilize these com-
1
arXiv:1901.04877v2  [cs.CV]  15 May 2019

plex dependency patterns effectively, we design a Graphi-
cal ConvLSTM for the LSTD module, which enables the
feature maps of each part to learn the longer-term (indi-
rect) and shorter-term (direct) dependency relations to other
parts. By modeling the graphical long short-term depen-
dency information among the features of different hand (or
body) parts, the boosted features produced by our LSTD
module are very effective for 3D pose estimation.
The inputs of the proposed LSTD module for feature
boosting are convnet feature maps that represent the infor-
mation for each hand (or body) part. However, these feature
maps which are extracted by the convolutional layers from
a single 2D image, may be unreliable for representing the
corresponding part, due to the existence of ambiguities in
3D pose estimation, the frequent occlusions, and also the
texture and lighting condition variations. In order to mit-
igate this drawback, we further improve the design of the
LSTD module by adding a soft modulator, context consis-
tency gate (CCG), which assesses the consistency of the
convolutional features with their context information and
modulates these features accordingly for boosting.
In our method, multiple convolutional layers and LSTD
modules can be stacked sequentially to construct a deep fea-
ture boosting network. In the whole convolutional architec-
ture, the intermediate feature maps are boosted at multiple
levels of the network.
The main contributions of this paper are summarized as
follows: (1) We propose an LSTD module within the CNN
architecture to boost the convolutional feature maps by al-
lowing them to perceive the graphical long short-term de-
pendency with the designed Graphical ConvLSTM. (2) We
further improve the design of the LSTD module by adding
a gating mechanism, CCG, to analyze the context consis-
tency of the convolutional feature maps. The CCG acts as
a soft modulator to regulate the propagation of the feature
map information based on their context consistency, which
also gives the LSTD module better insight about how to
boost the feature maps. (3) The proposed end-to-end fea-
ture boosting network achieves state-of-the-art performance
on challenging datasets for 3D hand pose estimation and 3D
full body pose estimation.
The rest of this paper is organized as follows. The related
works are introduced in section 2. The proposed feature
boosting network is described in detail in section 3. The
experimental results are provided in section 4. Finally, we
conclude the paper in section 5.

## method
[42]
[20]
3D Pose Net
3D Pose Net with FB+
PCK
69.2%
64.7%
66.2%
69.6%
Table 7: Evaluation of involving more connections.
Dataset
Human3.6M
3DHandPose
(PCK@50)
(PCK@20)
Max. number of linked joints
4
7
4
8
Accuracy (%)
51.5%
51.7%
89.5%
89.5%
10 
15 
7 
8 
9 
16 
14 
11 
12 
1 
2 
3 
13 
6 
5 
4 
(a) 
1 
5 
4 
3 
2 
9 
8 
7 
6 
13 
12 
11 
10 
17 
16 
15 
14 
21 
20 
19 
18 
(b) 
Figure 7: Illustration of involving more connections. Extra
links (denoted as blue arrows) are added.
Table 8: Evaluation of using different recurrent models.
Dataset
Human3.6M
3DHandPose
(PCK@50)
(PCK@20)
3D Pose Net with FB (ConvRNN)
49.3%
87.0%
3D Pose Net with FB (ConvGRU)
49.7%
87.5%
3D Pose Net with FB (ConvLSTM)
49.8%
87.7%
by using different recurrent structures, namely ConvLSTM,
ConvRNN, and ConvGRU, and report the results in Table 8.
The results show that the accuracy of ConvLSTM is higher
than ConvRNN and ConvGRU. We also observe that the
3D Pose Net with FB using different recurrent structures all
outperforms 3D Pose Net.
2D pose. Since 2D pose is estimated in our network, we
also evaluate its performance and report the results in Ta-
ble 9. The standard PCK metric is used for evaluation, and
Table 9: 2D pose accuracy on the 3DHandPose dataset.

## experiments
The proposed approach is evaluated on the 3DHandPose
dataset [40] for hand pose estimation, and the Human3.6M
dataset [16] for body pose estimation. The MPI-INF-3DHP
[20] and MPII [1] datasets are also used for qualitative anal-
ysis. We conduct extensive experiments using the following
models to test our proposed method:
(1) 3D Pose Net. This is the baseline network model for
3D pose estimation. In this network, the CNN feature maps
without feature boosting are fed to the CNN layers for 2D
heatmap generation and depth regression.
(2) 3D Pose Net with FB. In this network, the LSTD module
proposed by us is used for feature boosting (FB). However,
the CCG is not added.
(3) 3D Pose Net with FB+. This is the proposed feature
boosting network for 3D pose estimation. The LSTD mod-
ule is embedded in the CNN framework for feature boost-
ing, and the CCG is also added to improve the design of the
LSTD module.
4.1. Implementation Details
In our experiment, the parameter γ in the objective func-
tion is set to 0.1, and ω2 in Eq (9) is set to 2. These hyper-
parameters are obtained by using cross-validation protocol
on the training sets, and the parameter set achieving the op-
timum performance is used. The hourglass CNN layers are
implemented by following [22]. Data augmentation is used
in our experiments, including random translation, scaling,
and rotation.
During training, the 3D pose is aligned to the 2D pose of
the image plane, i.e., aligning the root joint location and also
the human body scale. Then this aligned 3D pose (at the im-
age pixel level) is used for network training. In testing, the
estimated 3D pose is re-scaled to the size of a pre-deﬁned
canonical skeleton, as done in [42, 43]. Rigid transforma-
tion [23] is not used in our experiment. For evaluation, the
6

Figure 3: Visualization of feature maps before and after
boosting for different joints (labeled as red circles). The
four columns are respectively (a) input image, (b) feature
map for representing a joint before boosting, (c) CCG, and
(d) feature map after boosting.
estimated pose and ground truth pose are aligned based on
the root joint locations.
4.2. Experiments of 3D Hand Pose Estimation
The 3DHandPose dataset [40] is a large dataset for 3D
hand pose estimation. It is captured under varying illumi-
nation conditions with 6 different backgrounds. Different
from the NYU Hand Pose dataset [33], which is mainly de-
signed for hand pose estimation from depth images and the
registered color images contain lots of artifacts, the large
3DHandPose dataset is highly suitable for 3D hand pose es-
timation from a single RGB image, as analyzed in [44]. In
this dataset, the 2D and 3D annotations of 21 keypoints of
the human hand are provided for each frame. We follow the
evaluation protocol of [44] by using 30,000 hand images for
training and 6,000 hand images for testing.
The experimental results are shown in Figure 4 and Ta-
ble 1. We report the percentage of correct keypoints (PCK)
for different error thresholds on this dataset by following
[44]. The results show that our proposed method outper-
forms the other methods on this dataset.
The 3D Pose Net and the model proposed by Zimmer-
mann et al. [44] are both CNN-based methods without con-
sidering the dependency structure of the features of the
hand joints, thus their performances are inferior to the pro-
posed feature boosting network with the LSTD module. By
adding the CCG to the LSTD module, the performance of
our method (3D Pose Net with FB+) is further improved.
Since the graphical long short-term dependency relations
among the joints are modeled in our network, we also eval-
uate the performance of the network by using different de-
pendency connections, and report the results in Table 2. The
“Simple Sequence” means that the hand joints are linked
one by one as a sequential chain by following the enumer-
ation order. The “Physical Dependency” link indicates that
the real physical connections between the joints are used
(as shown by the solid lines in Figure 2). The “Symmet-
Error Threshold (mm)
20
30
40
50
PCK
0.85
0.9
0.95
PSO
ICPPSO
CHPR
Zhao et al.
Zimmermann et al.
Mueller et al.
3D Pose Net
3D Pose Net with FB
3D Pose Net with FB+
Figure 4: 3D hand pose estimation results on the 3DHand-
Pose dataset. The curves indicate the percentage of correct
keypoint (PCK) over the respective threshold in mm.
Table 1: Experimental results on the 3DHandPose dataset.
Numbers are percentage of correct keypoint (PCK) over re-
spective threshold in mm. Refer to Figure 4 for more results.
Error Threshold (mm)
PCK@20
PCK@25
PCK@30
PSO [40]
32.2%
54.0%
67.4%
Zhao et al. [41]
43.6%
56.8%
70.1%
ICPPSO [40]
52.0%
64.5%
71.7%
CHPR [40]
56.6%
71.7%
82.2%
Zimmermann et al. [44]
85.9%
90.7%
93.7%
Mueller et al. [21]
88.0%
92.5%
95.2%
3D Pose Net
85.7%
91.0%
94.2%
3D Pose Net with FB
87.7%
92.1%
94.6%
3D Pose Net with FB+
89.5%
93.3%
95.6%
rical Connections” means that the “symmetrical” relations
are used (dashed lines in Figure 2). The “Graphical De-
pendency” link indicates that both the physical and “sym-
metrical” connections are used, but only the forward pass
is enabled. The “Bi-directional Graphical Dependency” is
the proposed graphical long short-term dependency rela-
tionship with bidirectional passes, as shown in Figure 2.
The results in Table 2 show that the “Graphical Depen-
dency” is superior to the “Physical Dependency” only and
the “Symmetrical Connections” only, which indicates that
it is beneﬁcial to combine the “symmetrical” relation links
and the physical dependency links for pose estimation. Our
proposed “Bi-directional Graphical Dependency” yields the
best result for 3D hand pose estimation, as shown in Table 2.
We evaluate the performance of the proposed frame-
work with different numbers of the sub-networks for fea-
ture learning and boosting, and show the results in Table 3.
Table 2: Evaluation of using different connections for Con-
vLSTM on the 3DHandPose dataset.
Connections
Accuracy (PCK@20)
Simple Sequence
86.1%
Physical Dependency
87.5%
“Symmetrical” Connections
87.4%
Graphical Dependency
89.0%
Bi-directional Graphical Dependency
89.5%
7

The results show that our feature boosting network with two
sub-networks outperforms the single sub-network frame-
work. This indicates that by boosting the feature maps at
multiple levels, the 3D pose estimation performance can be
improved. Due to the memory limitation of our GPUs, we
were not able to try stacking more sub-networks.
We also visualize some examples of the feature maps in
our network, as illustrated in Figure 3. Speciﬁcally, we vi-
sualize the feature maps learned by the previous CNN layers
before feature boosting, and the boosted feature maps. The
results show that by using the LSTD module with CCG for
boosting, the produced feature maps are more reliable and
stable compared to the feature maps before boosting.
4.3. Experiments of 3D Body Pose Estimation
Human3.6M. The Human3.6M dataset [16] is a large-
scale and widely used dataset for 3D human body pose esti-
mation. This dataset contains 3.6 million human poses cap-
tured with a motion capture system. We follow the evalua-
tion protocol in [42] on this dataset, in which 5 subjects (s1,
s5, s6, s7, and s8) are used for training, and 2 subjects (s9
and s11) are adopted for testing. The videos in this dataset
are down-sampled from 50fps to 10fps. The training sam-
ple combination in [42] is adopted to train our network (half
Human3.6M data [16] and half MPII data [1]).
The experimental results (PCKs) on the Human3.6M
dataset are shown in Table 5. The results show that by using
the LSTD module with CCG for feature boosting, the “3D
Pose Net with FB+” achieves the best results. We also com-
pare the proposed feature boosting network with the state-
of-the-arts, and report the results in Table 4. We can observe
that the feature boosting network outperforms other meth-
ods for 3D human pose estimation.
We also follow the data processing and evaluation setting
of [30], and use the videos of 5 subjects for training, while
evaluating on 2 subjects by using 1 frame from every 64
frames. On this setting, the joint error of our method is 58.0
mm, which is lower than 59.1 mm of the method in [30].
Cross-dataset evaluation on MPI-INF-3DHP. We per-
form cross-dataset evaluation on the MPI-INF-3DHP [20]
dataset, i.e., only Human3.6M and MPII are used for train-
ing, while the testing is performed on MPI-INF-3DHP. We
follow the evaluation criteria in [42] and report the average
PCK in Table 6. The results show that our proposed feature
boosting network achieves good performance in this cross-
dataset evaluation scenario.
Table 3: Evaluation of the feature boosting network with
different numbers of sub-networks.
Network stacking
Accuracy (PCK@20)
One sub-network
87.4%
Two sub-networks
89.5%
(a) 3D Pose Net 
(b) 3D Pose Net with FB 
(c) 3D Pose Net with FB+ 
Figure 5: Qualitative results on MPII. The wrongly esti-
mated joi

## related_work
2.1. 3D Pose Estimation
Different aspects of human hand (and body) pose esti-
mation have been explored in the past few years [9,27]. We
limit our review to more recent CNN-based approaches for
3D pose estimation. These methods mainly fall into two
categories: 3D regression-based, and intermediate 2D pose-
based methods [32].
3D regression-based methods: Many previous meth-
ods directly regress the 3D locations of each joint using the
convolutional features. For example, Li and Chan [17] de-
signed a pretraining strategy, in which the 3D pose regressor
was initialized with a model trained for body part detection.
Tekin et al. [31] used auto-encoders to learn structured la-
tent representations for 3D pose regression from the images.
Park et al. [24] introduced a CNN framework by simultane-
ously training for both 2D joint classiﬁcation and 3D joint
regression. Ghezelghieh et al. [11] proposed to learn the
camera viewpoint based on CNNs to improve the perfor-
mance of 3D body pose estimation.
Intermediate 2D pose-based methods: A very recent
trend of works started to investigate a pipeline framework
to strengthen the estimation of 3D poses. In this pipeline
framework, heatmaps of the joints are estimated in the 2D
frames ﬁrst. These 2D poses are then regarded as the in-
termediate representations, and the 3D poses are estimated
based on them. For example, Chen et al. [4] combined the
2D pose estimation results and a 3D matching library, and
achieved promising performance for 3D human pose esti-
mation. Zimmermann et al. [44] adopted a PoseNet to infer
the 2D hand joint locations, and then used a PosePrior net-
work to estimate the most likely 3D structure of the hand.
Zhou et al. [42] augmented the 2D pose estimation sub-
network with a 3D depth regression sub-network to per-
form 3D human pose estimation. Tome et al. [32] proposed
to perform 2D joint estimation and 3D pose reconstruction
jointly to improve both tasks. Nie et al. [23] proposed to
predict the depth of joints based on the 2D joint locations
and the body part image features for 3D pose estimation.
Our proposed method is based on the pipeline frame-
work as mentioned above [42, 44], i.e., the intermediate
2D poses are estimated for the ﬁnal 3D pose estimation.
Different from these works on 3D pose estimation, in our
method, the feature maps within the convolutional network
are boosted by enabling them to perceive the long short-
term dependency patterns among different parts with the
proposed LSTD module. Besides, a soft modulator, CCG,
is added to analyze the reliability and context consistency of
the convolutional features, which encourages the network to
learn reliable features for 3D pose estimation.
2.2. Dependency Structure
The analysis of the correlation between parts of the hand
(or body) has been shown to be very useful for pose estima-
tion. Felzenszwalb et al. [10] proposed to represent the hu-
man body by a collection of parts arranged in a deformable
conﬁguration for pose estimation. Yang et al. [38] described
a method for articulated human detection and pose esti-
2

mation in static images based on the representation of de-
formable part models with a tree structure. Chu et al. [7]
introduced a structured feature learning method to reason
the relationships within the body joints for 2D pose estima-
tion. Chen et al. [5] proposed a graphical model of the body
joints as a post-processing step.
Different from the above-mentioned works, in this pa-
per, we propose a new LSTD module with Graphical Con-
vLSTM for feature boosting. By introducing the Graphical
ConvLSTM, we add an extra layer of feature analysis, to
model the graphical long short-term dependency relations
among different parts. We show that the boosted feature
maps derived from the LSTD module are more powerful
for 3D pose estimation than the features before boosting.
Speciﬁcally, LSTD modules can be added at different lay-
ers, thus the features in the whole CNN architecture can
be boosted layer by layer. Moreover, we introduce a gat-
ing mechanism (CCG) to ﬂexibly regulate the propagation
of the intermediate feature representations within the CNN
architecture by analyzing their reliability and context con-
sistency.
2.3. Gating Mechanism
Our proposed context consistency gate (CCG) is inspired
by the gating mechanism [6,14,18,34,37], which is shown
to be an important technique to improve the representation
strength of deep networks. Cho et al. [6] proposed a net-
work with gated units to modulate the information ﬂow for
machine translation. Xiong et al. [37] designed an atten-
tion gate to explore the important information for textual
and visual question answering. Liu et al. [18] introduced a
trust gating mechanism to deal with the noisy sequences for
activity analysis. Dauphin et al. [8] proposed gated linear
units within the deep network for language modeling.
Compared to the aforementioned methods, our soft mod-
ulator, CCG, is designed in a different context in terms of
both its purpose and architecture. The goal of the CCG is
to assess the reliability of the convolutional features, and
accordingly regulate the propagation of them in the CNN
architecture. To the best of our knowledge, the proposed
work is the ﬁrst of its nature in introducing gating mecha-
nisms [18] in a CNN architecture for modulating and propa-
gating the features by considering their context consistency
for 3D pose estimation.
3. The Proposed Method
Given a single RGB image of a human hand (or a full hu-
man body), our goal is to estimate the locations of the major
joints of the hand (or body) in 3D space. In this paper, we
propose a feature boosting network based on a CNN frame-
work for this task. A long short-term dependence-aware
(LSTD) module is proposed, which is embedded inside the
CNN framework, to boost the convolutional features by en-
abling them to perceive the graphical long short-term de-
pendency patterns among different parts. Moreover, the de-
sign of the LSTD module is further improved by adding a
context consistency gate (CCG), which acts as a soft mod-
ulator to adjust the propagation of features through the net-
work, according to the context consistency and reliability.
The overall architecture of the feature boosting network is
illustrated in Figure 1.
3.1. Long Short-Term Dependence-aware Module
There are direct and indirect kinematic dependency re-
lations among different parts of the human hand (or body).
For example, in Figure 2(a), the adjacent joints, 2 and 3,
are directly connected in the human body, while the joints 2
and 7 are indirectly connected. Utilizing these complex di-
rect and indirect dependency patterns as a feature analysis
step is beneﬁcial for 3D pose estimation.
Many existing CNN-based 3D pose estimation ap-
proaches do not explicitly use the dependency structure,
while some others often consider it at the result level, e.g.,
employ the dependency relations to reﬁne the 3D estima-
tions, or use them to lift the 2D coordinates of the joints
to 3D space at a post-processing stage [23].
In this pa-
per, we employ the direct and indirect dependency patterns
to boost the intermediate features at different levels of the
convolutional architecture for 3D pose estimation. Specif-
ically, we introduce a novel long short-term dependence-
aware (LSTD) module to enable the features of each part
of a hand (or a body) to discover its long short-term de-
pendency relations to other parts. Below we introduce the
mechanism of the proposed LSTD module in detail.
Graphical Dependency Relations. The major joints of
the human body and hand are illustrated in Figure 2(a) and
Figure 2(b), respectively. These joints are physically con-
nected in a tree-like structure (solid lines in Figure 2). Since
there are often correlation patterns among the “symmetri-
cal” joints, which can be useful for 3D pose estimation, we
also introduce direct links between them (dashed lines in
Figure 2). Therefore, the full dependency graph can be con-
structed for the human body as illustrated in Figure 2(a) and
hand in Figure 2(b).
Graphical ConvLSTM. As a successful extension of
the recurrent neural networks, long short-term memory
(LSTM) networks [14] can learn the complex long-term and
short-term context dependency relations over the sequential
input data. Due to the natural dependencies among different
parts of the human hand (or body), LSTM is highly suitable
for modeling the direct (“short-term”) and indirect (“longer-
term”) dependency patterns among different parts for 3D
pose estimation.
Since we aim to investigate the long short-term depen-
dencies for boosting the feature maps within the CNN
framework, we adopt the convolutional LSTM (ConvL-
3

Summation 
C1 
C2 
Cj 
CJ 
Hourglass 
CNN 
Feature Boosting 
H1 
H2 
HJ 
2D Loss 
1x1 Conv 
+ 
F1 
F2 
FJ 
C1 
C2 
Cj 
CJ 
Hourglass 
CNN 
Feature Boosting 
H1 
H2 
HJ 
2D Loss 
1x1 Conv 
+ 
F1 
F2 
FJ 
Depth Loss 
Depth regression module 
Fj 
Feature maps of Join

## conclusion
We propose a feature boosting network for 3D hand and
full body pose estimation in this paper. A novel LSTD mod-
ule is introduced to enable the convolutional features to per-
ceive the graphical long short-term dependency relationship
among different hand (or body) parts. The design of the
LSTD module is further enhanced by assessing the context
consistency of the features with the CCG. The proposed fea-
ture boosting network achieves state-of-the-art performance
on challenging datasets for 3D hand and body pose estima-
tion.