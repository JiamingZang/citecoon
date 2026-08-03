# TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation

> 2025 · id: W4409366800 · arXiv: 2501.01770 · pdf: https://ojs.aaai.org/index.php/AAAI/article/download/32583/34738 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent multi-frame lifting methods have dominated the 3D
human pose estimation. However, previous methods ignore the
intricate dependence within the 2D pose sequence and learn
single temporal correlation. To alleviate this limitation, we
propose TCPFormer, which leverages an implicit pose proxy
as an intermediate representation. Each proxy within the im-
plicit pose proxy can build one temporal correlation therefore
helping us learn more comprehensive temporal correlation of
human motion. Specifically, our method consists of three key
components: Proxy Update Module (PUM), Proxy Invocation
Module (PIM), and Proxy Attention Module (PAM). PUM first
uses pose features to update the implicit pose proxy, enabling
it to store representative information from the pose sequence.
PIM then invocates and integrates the pose proxy with the
pose sequence to enhance the motion semantics of each pose.
Finally, PAM leverages the above mapping between the pose
sequence and pose proxy to enhance the temporal correlation
of the whole pose sequence. Experiments on the Human3.6M
and MPI-INF-3DHP datasets demonstrate that our proposed
TCPFormer outperforms the previous state-of-the-art methods.
Code — https://github.com/AsukaCamellia/TCPFormer

## introduction
3D human pose estimation has always been a crucial prob-
lem in computer vision, which aims to locate the 3D joint
positions of a human body (Moon and Lee 2020; Pavlakos,
Zhou, and Daniilidis 2018; Chen et al. 2021). Nowadays,
3D human pose estimation finds widespread applications in
various scenarios, including motion prediction (Wang et al.
2023), action recognition (Zhang et al. 2022a), and human-
robot interaction (Gong et al. 2022; Ye et al. 2021). Given the
widespread usage of 2D human pose detectors (Chen et al.
2018; He et al. 2017; Newell, Yang, and Deng 2016; Sun et al.
2019) and the task-relatedness between 2D pose and 3D pose,
most research follows a 2D-to-3D lifting pipeline (Zheng
et al. 2021; Li et al. 2022a,b; Zhang et al. 2022b; Wang et al.
2024), where 2D keypoints are first detected and then lifted to
the 3D space. Despite the considerable success achieved, this
task remains an ill-posed problem and inherently suffers from
depth ambiguity. An extensive body of literature focuses on
*Corresponding author: liumengyuan@pku.edu.cn
Copyright © 2025, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
(a) Previous Methods
(b) Our  TCPFormer
Pose Sequence
Kth Pose
Pose Sequence
Kth Pose
Pose Proxy
Trainable
Figure 1: An illustration of our motivation. Given a pose
sequence of length T, we take the individual pose within
the pose sequence as an example. (a) In previous methods,
one pose establishes the temporal correlation with the pose
sequence only in one 1-to-T mapping. (b) We introduce an
implicit pose proxy to act as an intermediate representation.
Each proxy within the implicit pose proxy of length L can
establish one 1-to-T mapping, which facilitates learning more
comprehensive temporal correlation.
exploiting temporal information between adjacent frames to
mitigate this issue, ranging from earlier methods (Pavllo et al.
2019; Liu et al. 2020b; Chen et al. 2021) use temporal con-
volution and subsequent attempts (Cai et al. 2019; Hu et al.
2021; Wang et al. 2020) use graph convolution. Recently,
transformer (Vaswani et al. 2017) have achieved significant
success in both natural language preprocess (Brown et al.
2020; Devlin et al. 2018) and computer vision (Dosovitskiy
et al. 2021; Carion et al. 2020). For the 3D human pose esti-
mation task, many works (Li et al. 2022b; Zhang et al. 2022b;
Shan et al. 2022; Tang et al. 2023) leverage the powerful
sequence modeling capability of transformer to extend their
input from the limited neighboring frames to long-term se-
quences for advanced accuracy, e.g., 243 video frames for
MixSTE (Zhang et al. 2022b) and STCFormer (Tang et al.
2023); large as 351 frames for MHFormer (Li et al. 2022b).
Despite their achievements, a potential concern has grad-
ually emerged: with the massive increase in the number of
input frames, the performance improvement becomes slow.
For instance, PoseFormerV2 (Zhao et al. 2023b) achieved an
error reduction of 0.8mm when expanding the input from 81
arXiv:2501.01770v1  [cs.CV]  3 Jan 2025

frames to 243 frames. StridedTrans (Li et al. 2022a) achieved
a marginal 0.3mm error reduction when expanding the input
from 243 frames to 351 frames, while MHFormer (Li et al.
2022b) achieved an even smaller error reduction of 0.2mm
with the same input expansion. These key observations point
towards a problem that restricts most methods from effec-
tively modeling the temporal correlation within the 2D pose
sequence. In this work, we are trying to solve this problem.
As illustrated in Figure 1a, we discover that most of the
aforementioned multi-frame methods only establish one 1-
to-T mapping for each pose within the pose sequence, where
T denotes the length of pose sequence. However, due to
the extensive number of frames, only establishing one 1-
to-T mapping can not comprehensively reflect the complex
temporal correspondence within the pose sequence.
To address this limitation, we propose a novel method to
learn Temporal Correlation with Implicit Pose Proxy, dubbed
TCPFormer. As illustrated in Figure 1b, we introduce an
implicit pose proxy to act as the intermediate representation.
We first establish a 1-to-L mapping to build the relationship
between the individual pose and the implicit pose proxy,
where L denotes the length of the implicit pose proxy. Then,
each proxy within the implicit pose proxy will interact with
the pose sequence and build multiple 1-to-T mapping to help
model learning more comprehensive temporal correlation.
Moreover, our implicit pose proxy is trainable and will be
continuously optimized during the training process.
Specifically, we first propose a Proxy Update Module
(PUM). PUM adaptively encodes useful and representative
information from the pose sequence to update the pose proxy
through the cross-attention mechanism (Vaswani et al. 2017).
Although the information in the pose proxy has been updated,
we have not yet transmitted it to each pose within the pose
sequence. Therefore, we propose a Proxy Invocation Module
(PIM) that uses the pose proxy as the key and value to en-
hance the feature representation ability of the pose sequence.
In addition, we propose a Proxy Attention Module (PAM).
PAM skillfully leverages the two cross-attention matrices
of PUM and PIM to get an aggregation matrix and flexibly
fuses it with the original self-attention matrix to obtain a
more effective and comprehensive temporal correlation.
We extensively evaluate our TCPFormer on two widely
used benchmark datasets, Human3.6M (Ionescu et al. 2013)
and MPI-INF-3DHP (Mehta et al. 2017). Empirical eval-
uations show that our approach outperforms the previous
state-of-the-art methods. Comprehensive ablation studies are
also presented to evaluate the contribution of each component.
Our contributions can be summarized as follows:
• To the best of our knowledge, we are the first to introduce
the implicit pose proxy to 3D human pose estimation.
Our method leverages the implicit pose proxy as an inter-
mediate representation to effectively model the complex
temporal correlation within the pose sequence.
• We design three novel modules: Proxy Update Module,
Proxy Invocation Module, and Proxy Attention Module.
These three modules present a unique way to effectively
enhance the feature of pose sequence and learn a more
comprehensive temporal correlation.
• Extensive experiments conducted on Human3.6M and
MPI-INF-3DHP two challenging datasets for 3D human
pose estimation demonstrate that our method achieves
superior performances than the previous methods.

## method
Venue
T
Seq2Seq
PCK ↑
AUC ↑
MPJPE ↓
MHFormer (Li et al. 2022b)
CVPR’22
9
✗
93.8
63.3
58.0
MixSTE (Zhang et al. 2022b)
CVPR’22
27
✓
94.4
66.5
54.9
P-STMO (Shan et al. 2022)
ECCV’22
81
✗
97.9
75.8
32.2
STCFormer (Tang et al. 2023)
CVPR’23
81
✓
98.7
83.9
23.1
PoseFormerV2 (Zhao et al. 2023b)
CVPR’23
81
✗
97.9
78.8
27.8
GLA-GCN (Yu et al. 2023)
ICCV’23
81
✗
98.5
79.1
27.8
MotionBERT (Zhu et al. 2023)
ICCV’23
-
✓
-
-
-
KTPFormer (Peng, Zhou, and Mok 2024)
CVPR’24
81
✓
98.9
85.9
16.7
TCPFormer (Ours)
-
9
✓
98.3
84.4
20.4
TCPFormer (Ours)
-
27
✓
98.7
86.5
17.8
TCPFormer (Ours)
-
81
✓
99.0
87.7
15.0
Table 3: Results on MPI-INF-3DHP under three evaluation metrics. T is the number of input frames. Seq2seq refers to estimating
3D pose sequences rather than only the center frame. The best result is shown in bold, and the second-best result is underlined.
Ablation Study
All experiments were conducted on the Human3.6M dataset
with T = 243 as the number of input frames.
Step Proxy PUM PIM PAM MPJPE ↓P-MPJPE ↓
1
✓
-
-
-
42.2
34.6
2
✓
✓
-
-
39.5
32.6
3
✓
✓
✓
-
38.7
32.3
Ours
✓
✓
✓
✓
37.9
31.7
Table 4: The effectiveness of different components. All our
proposed novel components exhibit improvements.
Impact of Each Component. As shown in Table 4, we vali-
date the overall performance gain brought by the proposed
implicit pose proxy (Proxy), proxy update module (PUM),
proxy invocation module (PIM), and proxy attention module
(PAM). Our baseline, which only introduces an implicit pose
proxy without additional module design, achieves a result of
42.2mm MPJPE and 34.6mm P-MPJPE. By applying PUM,
our method decreases 2.7mm MPJPE and 2.0mm P-MPJPE.
Next, we integrate PIM into our method and achieve better
results with 38.7mm MPJPE and 32.3mm P-MPJPE. Finally,
we achieve the best performance with 37.9mm MPJPE and
31.7mm P-MPJPE by incorporating the PAM.
Length
Distribution
MPJPE ↓
P-MPJPE ↓
27
Gaussian
38.4
32.1
81
Random
39.1
32.5
81
Laplacian
38.2
32.0
81
Gaussian
37.9
31.7
243
Gaussian
38.6
32.7
Table 5: Analysis on implicit pose proxy. Distribution and
Length denote the temporal dimension and initial distribution
of our proposed implicit pose proxy.
Analysis on Implicit Pose Proxy. How to represent implicit
pose proxy is crucial for our methods. We investigated the
PIM
PUM
MPJPE ↓
P-MPJPE ↓
MLP
MLP
40.8
33.8
CrossAttention
MLP
39.6
32.6
MLP
CrossAttention
39.5
32.8
CrossAttention
CrossAttention
38.7
32.3
Table 6: Analysis of the various micro designs within proxy
update module and proxy invocation module.
Range
Strategy
MPJPE ↓
P-MPJPE ↓
0
Fixed
38.4
32.4
0
Trainable
38.2
32.0
(0, 1)
Trainable
37.9
31.7
(-1, 0)
Trainable
38.5
32.3
(-1, 1)
Trainable
37.9
32.0
Table 7: Analysis on the adaptive fusion. Range denotes the
sampling range of µ. Strategy denotes whether µ is trainable.
impact of the temporal dimension and initial distribution of
our implicit pose proxy. For the temporal dimension, we set
it to 27, 81, and 243 respectively. For the initial distribution,
we provided gaussian distribution, laplace distribution, and
random distribution. The results presented in Table 5 show
that our method achieves the best performance when setting
the temporal dimension of implicit pose proxy to 81 and
using the gaussian distribution initialization.
Analysis on Micro Design. In this section, we further explore
the effectiveness of various micro designs within the proxy
update module (PUM) and proxy invocation module (PIM).
As shown in Table 6, we achieve the best performance when
both PUM and PIM use cross attention.
Analysis on Proxy Attention Module. We extensively in-
vestigated the fusion strategies of adaptive fusion within our
proxy attention module. Specifically, we pay attention to the
sampling range of µ and whether it is trainable. As shown in
Table 7, we achieved the best performance when we allowed
µ to be trainable and sampled it from (0, 1).

0
50
100
150
200
0
50
100
150
200
Self Attention
0
50
100
150
200
0
50
100
150
200
Self Attention
0
50
100
150
200
0
50
100
150
200
Self Attention
0
50
100
150
200
0
50
100
150
200
Self Attention
0
50
100
150
200
0
50
100
150
200
Self Attention
0
50
100
150
200
0
50
100
150
200
Aggregation Attention
0
50
100
150
200
0
50
100
150
200
Aggregation Attention
0
50
100
150
200
0
50
100
150
200
Aggregation Attention
0
50
100
150
200
0
50
100
150
200
Aggregation Attention
0
50
100
150
200
0
50
100
150
200
Aggregation Attention
0
50
100
150
200
0
50
100
150
200
Proxy Attention
0
50
100
150
200
0
50
100
150
200
Proxy Attention
0
50
100
150
200
0
50
100
150
200
Proxy Attention
0
50
100
150
200
0
50
100
150
200
Proxy Attention
0
50
100
150
200
0
50
100
150
200
Proxy Attention
Figure 3: Visualizations of different attention matrices. The first row is the original self-attention matrix. The second row is the
aggregation attention matrix. The third row is our proxy attention matrix. As expected, our proxy attention matrix effectively
leverages the aggregation attention matrix to complement the missing parts of the original self attention matrix.
Figure 4: Qualitative comparisons of our TCPFormer with
MotionBERT on in-the-wild videos. The yellow arrows indi-
cate locations where our method achieves better results.
Qualitative Analysis. We visualized the original self atten-
tion matrix (first row), aggregation attention matrix (second
row), and proxy attention matrix (third row) in Figure 3. All
attention matrices are normalized to [0, 1]. As expected, our
proxy attention matrix effectively leverages the aggregation
attention matrix to complement the original self attention ma-
trix. Furthermore, we also present 3D human pose estimation
results by MotionBERT (Zhu et al. 2023) and our TCPFormer
on the Human3.6M dataset and in-the-wild videos. As shown
in Figure 4 and Figure 5, TCPFormer achieves better qualita-
Figure 5: Qualitative comparisons of our TCPformer with
MotionBERT on Human3.6M. The green circles indicate
locations where our method achieves better results.
tive results compared with MotionBERT (Zhu et al. 2023).

## experiments
Datasets and Evaluation Metrics
We comprehensively evaluate our model on two large-scale
3D human pose estimation datasets: Human3.6M (Ionescu
et al. 2013) and MPI-INF-3DHP (Mehta et al. 2017).
Human3.6M is the most popular benchmark for indoor 3D
human pose estimation, which contains approximately 3.6
million frames captured by 4 cameras at different views. This
dataset contains 11 subjects performing 15 typical actions.
MPI-INF-3DHP is a recently proposed large-scale challeng-
ing dataset with both indoor and outdoor scenes. The train-
ing set comprises 8 subjects, covering 8 activities, ranging
from walking and sitting to complex exercise poses and dy-
namic actions. The test set covers 7 activities, containing
three scenes: green screen, non-green screen, and outdoor
environments.
Evaluation Metrics. For the Human3.6M dataset, we use
two evaluation metrics: MPJPE and P-MPJPE. MPJPE (Mean
Per Joint Position Error) is computed as the mean Euclidean
distance between the estimated joints and the ground truth
in millimeters after aligning their root joints. P-MPJPE
(Procrustes-MPJPE) is the MPJPE after the estimated joints
align to the ground truth via a rigid transformation. For the
MPI-INF-3DHP dataset, following previous works (Shan
et al. 2022; Zhou, Yin, and Li 2024; Zhu et al. 2023), we use
ground truth 2D pose as input and report MPJPE, Percentage
of Correct Keypoint (PCK) with the threshold of 150mm, and
Area Under Curve (AUC) as the evaluation metrics.
Implementation Details
We consider the layers N of modules, the number H of
heads in attention block, the size C of hidden feature, the
temporal dimension L of implicit pose proxy, and the ini-
tialization distribution D of proxy as free parameters. The
performances of the versions with (N = 16, H = 8, C =
128, L = T/3, D = Gaussian) are reported. Our model
is implemented using PyTorch and executed on a server
equipped with 2 NVIDIA 4090 GPUs. We apply horizontal
flipping augmentation for both training and testing follow-
ing (Tang et al. 2023; Zhu et al. 2023; Foo et al. 2023; Zhao
et al. 2023a). For model training, we set each mini-batch
as 16 sequences. The network parameters are optimized us-
ing AdamW (Loshchilov and Hutter 2017) optimizer over
90 epochs with a weight decay of 0.01. The initial learning
rate is set to 5e-4 with an exponential learning rate decay
schedule and the decay factor is 0.99. In the experiments on
Human3.6M, two kinds of input are utilized including the 2D
ground truth and the Stacked Hourglass (Newell, Yang, and
Deng 2016) 2D pose detection, following (Zhu et al. 2023;
Ci et al. 2019). For MPI-INF-3DHP, 2D ground truth is used
following previous works (Cai et al. 2024; Zhang et al. 2022b;
Li et al. 2023; Zhu et al. 2023; Li et al. 2024).

## related_work
3D Human Pose Estimation
Early works (Ionescu, Carreira, and Sminchisescu 2014;
Ionescu et al. 2013; Ramakrishna, Kanade, and Sheikh 2012;
Agarwal and Triggs 2005; Andriluka, Roth, and Schiele 2009;
Ionescu, Li, and Sminchisescu 2011) of monocular 3D human
pose estimation primarily focus on exploiting spatial prior in-
formation in the form of human skeletal structure and motion
features. With the development of deep learning, more deep
neural network-based methods have been introduced and can
be divided into two mainstream types: one-stage manner and
two-stage manner. One-stages approaches (Kanazawa et al.
2018; Pavlakos et al. 2017; Sun et al. 2018) directly estimate
the 3D pose from the input image without the intermediate
2D pose representation. Different from the one-stage manner,
two-stage methods (Fang et al. 2018; Martinez et al. 2017;
Zhao et al. 2019; Liu et al. 2020a; Xu and Takano 2021) first
obtain 2D joint coordinates in the image and then leverage the
task-relevant positional information to lift the 2D joint coor-
dinates to 3D poses. With the reliable achievement of 2D hu-
man pose detectors (Chen et al. 2018; He et al. 2017; Newell,
Yang, and Deng 2016; Sun et al. 2019), these 2D-to-3D lifting
methods outperform one-stage approaches. However, they
still inherently suffer from the problem of depth ambigui-
ties. To address this problem, some studies (Liu et al. 2020b;
Pavllo et al. 2019) have made preliminary explorations in
utilizing temporal information. Liu et al. (Liu et al. 2020b)
extend the temporal convolutional network by introducing the
attention mechanism. The aforementioned methods utilize
limited temporal information, which is unable to effectively
facilitate 3D human pose estimation.
Transformer-based Methods
For the 3D human pose estimation task, PoseFormer (Zheng
et al. 2021) firstly introduces transformer architecture to
leverage spatial and temporal dependency. MHFormer (Li
et al. 2022b) addresses the depth ambiguity by learning
spatio-temporal representations of multiple pose hypothe-
ses. MixSTE (Zhang et al. 2022b) constructs a mixed spatio-
temporal transformer to capture the temporal motion of dif-
ferent body joints. P-STMO (Shan et al. 2022) is the first
approach that introduces the pre-training technique to 3D hu-
man pose estimation. PoseFormerV2 (Zhao et al. 2023b) im-
proves PoseFormer by utilizing a frequency-domain represen-
tation of input joint sequences. STCFormer (Tang et al. 2023)
decomposes spatio-temporal attention and integrates the
structure-enhanced positional embedding. MotionBERT (Zhu
et al. 2023) and UPS (Foo et al. 2023) both train a unified
model for multi-task. However, these methods still have limi-
tation in directly modeling the complex temporal correlation
of the pose sequence due to sequence length. We conduct
further exploration to address this limitation in this paper.

Attention
Matrix
Attention Matrix
QP
Initialization
( L × J × C ) 
Gaussian Distribution
Implicit
Pose
Proxy
Projection
( T × J × C ) 
Spatio-Temporal
Transformer Encoder
Input 2D Pose Sequence
......
KF
VF
Approximate the
Data Distribution
Proxy Update Module (PUM)
Proxy Invocation Module (PIM)
QF
KP
VP
Updated Distribution
Aggregation
Attention Matrix
( T × T )
( T × L )
( L × T )
Output 3D Pose Sequence
......
Regression Head
Layer Norm
Linear Embedding
Porxy Attention Module (PAM)
Layer Norm
Feedforward Network
Adaptive
Fusion
Q
V
K
Proxy Attention Matrix
Softmax
Matrix
Product
× N 
Pose Feature Token
Initial Proxy Token
Updated Proxy Token
Figure 2: Overview of our method. We first extract the spatio-temporal information through a spatio-temporal encoder. Then,
we introduce an implicit pose proxy which is initialized by Gaussian distribution. These features and proxy are then handed to
the proxy update module to update the implicit pose proxy. Next, the proxy invocation module uses the updated pose proxy
to enhance the feature of the pose sequence. We obtain an aggregation attention matrix through two cross attention matrices
and send it with the pose sequence feature to the proxy attention module to learn comprehensive temporal correlation. After
repeating the above processes N times, we use a regression head to obtain the 3D pose sequence.

## conclusion
In this paper, we present TCPFormer, a novel method to
learn temporal correlation with implicit pose proxy. Different
from previous methods that learn complex temporal corre-
lations only through single mapping, TCPFormer leverages
the implicit pose proxy as an intermediate representation
to skillfully model the complex temporal correlation within
the pose sequence and effectively use the temporal informa-
tion to facilitate 3D human pose estimation. The visualiza-
tion results provide empirical evidence that our TCPFormer
can build comprehensive temporal correlation within the 2D
pose sequence. Extensive experimental results also show that
our TCPFormer outperforms the previous state-of-the-art ap-
proaches on the Human3.6M and MPI-INF-3DHP datasets.

Acknowledgments
This work was supported by National Natural Science Foun-
dation of China (No. 62203476), Natural Science Foundation
of Guangdong Province (No. 2024A1515012089), Shenzhen
Innovation in Science and Technology Foundation for The
Excellent Youth Scholars (No. RCYX20231211090248064).