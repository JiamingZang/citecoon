# Cascaded Deep Monocular 3D Human Pose Estimation With Evolutionary Training Data

> 2020 · id: W3034217102 · arXiv: 2006.07778 · pdf: https://arxiv.org/pdf/2006.07778 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
End-to-end deep representation learning has achieved
remarkable accuracy for monocular 3D human pose esti-
mation, yet these models may fail for unseen poses with lim-
ited and ﬁxed training data. This paper proposes a novel
data augmentation method that: (1) is scalable for syn-
thesizing massive amount of training data (over 8 million
valid 3D human poses with corresponding 2D projections)
for training 2D-to-3D networks, (2) can effectively reduce
dataset bias. Our method evolves a limited dataset to syn-
thesize unseen 3D human skeletons based on a hierarchi-
cal human representation and heuristics inspired by prior
knowledge. Extensive experiments show that our approach
not only achieves state-of-the-art accuracy on the largest
public benchmark, but also generalizes signiﬁcantly better
to unseen and rare poses. Code, pre-trained models and
tools are available at this HTTPS URL1.

## introduction
Estimating 3D human pose from RGB images is crit-
ical for applications such as action recognition [36] and
human-computer interaction, yet it is challenging due to
lack of depth information and large variation in human
poses, camera viewpoints and appearances.
Since the
introduction of large-scale motion capture (MC) datasets
[61, 22], learning-based methods and especially deep rep-
resentation learning have gained increasing momentum in
3D pose estimation. Thanks to their representation learn-
ing power, deep models have achieved unprecedented high
accuracy [47, 44, 31, 37, 36, 65].
Despite their success, deep models are data-hungry and
vulnerable to the limitation of data collection. This prob-
lem is more severe for 3D pose estimation due to two fac-
tors. First, collecting accurate 3D pose annotation for RGB
images is expensive and time-consuming. Second, the col-
lected training data is usually biased towards indoor envi-
1https://github.com/Nicholasli1995/EvoSkeleton
Input Image
Li et al.
Before Evolution (Ours)
After Evolution (Ours)
Figure 1: Model trained on the evolved training data gener-
alizes better than [28] to unseen inputs.
ronment and selected daily actions. Deep models can easily
exploit these bias but fail for unseen cases in unconstrained
environments. This fact has been validated by recent works
[74, 71, 28, 69] where cross-dataset inference demonstrated
poor generalization of models trained with biased data.
To cope with the domain shift of appearance for 3D
pose estimation, recent state-of-the-art (SOTA) deep mod-
els adopt the two-stage architecture [73, 14, 15]. The ﬁrst
stage locates 2D human key-points from appearance infor-
mation, while the second stage lifts the 2D joints into 3D
skeleton employing geometric information. Since 2D pose
annotations are easier to obtain, extra in-the-wild images
can be used to train the ﬁrst stage model, which effectively
reduces the bias towards indoor images during data collec-
tion. However, the second stage 2D-to-3D model can still be
negatively inﬂuenced by geometric data bias, yet not stud-
ied before. We focus on this problem in this work and our
research questions are: are our 2D-to-3D deep networks in-
ﬂuenced by data bias? If yes, how can we improve network
1
arXiv:2006.07778v3  [cs.CV]  9 Apr 2021

generalization when the training data is limited in scale or
variation?
To answer these questions, we propose to analyze the
training data with a hierarchical human model and represent
human posture as collection of local bone orientations. We
then propose a novel dataset evolution framework to cope
with the limitation of training data. Without any extra an-
notation, we deﬁne evolutionary operators such as crossover
and mutation to discover novel valid 3D skeletons in tree-
structured data space guided by simple prior knowledge.
These synthetic skeletons are projected to 2D and form 2D-
3D pairs to augment the data used for training 2D-to-3D
networks. With an augmented training dataset after evolu-
tion, we propose a cascaded model achieving state-of-the-
art accuracy under various evaluation settings. Finally, we
release a new dataset for unconstrained human pose in-the-
wild. Our contributions are summarized as follows:
• To our best knowledge, we are the ﬁrst to improve 2D-
to-3D network training with synthetic paired supervi-
sion.
• We propose a novel data evolution strategy which can
augments an existing dataset by exploring 3D human
pose space without intensive collection of extra data.
This approach is scalable to produce 2D-3D pairs in
the order of 107, leading to better model generalization
of 2D-to-3D networks.
• We present TAG-Net, a deep architecture consisting
of an accurate 2D joint detector and a novel cascaded
2D-to-3D network. It out-performs previous monoc-
ular models on the largest 3D human pose estimation
benchmark in various aspects.
• We release a new labeled dataset for unconstrained hu-
man pose estimation in-the-wild.
Fig. 1 shows a 2D-to-3D network trained on our aug-
mented dataset can handle rare poses while others such
as [28] may fail.

## method
Training Data
MPJPE
P1
P1*
Problem Setting A: Weakly-supervised Learning
B
S1
71.5
66.2
B+C
S1
70.1↓2.0%
64.5↓2.6%
B+C+E
Evolve(S1)
60.8↓15.0%
50.5↓21.7%
Problem Setting B: Fully-supervised Learning
B
S15678
54.3
44.5
B+C
S15678
52.1↓4.0%
42.9↓3.6%
B+C+E
Evolve(S15678)
50.9↓6.2%
34.5↓22.4%
Table 5: Ablation study on H36M. B: baseline. C: add cas-
cade. E: add data evolution. Evolve() represents the data
augmentation operation. Same P1 and P1* as in Table 2.
Error reduction compared with the baseline follows the ↓
signs.
mation. There are many fruitful directions remaining to be
explored. Extension to temporal domain, multi-view setting
and multi-person scenarios are three examples. In addition,
instead of being ﬁxed, the operators can also evolve during
the data generation process.
Acknowledgments We gratefully acknowledge the support
of NVIDIA Corporation with the donation of one Titan Xp
GPU used for this research. This research is also supported
in part by Tencent and the Research Grant Council of the
Hong Kong SAR under grant no. 1620818.
8

Supplementary Material
This supplementary material includes implementation
details and extended experimental analysis that are not in-
cluded in the main text due to space limit. The detailed
MPJPE under different settings are shown Tab. 6 and Tab. 7.
Other contents are organized in separate sections as follows:
• Section 7 includes the implementation details of the
hierarchal human representation.
• Section 8 elaborates the model training, which in-
cludes the training algorithm of the cascaded model
and describes details of data pre-processing.
• Section 9 gives ablation study on data generation and
the evolutionary operators.
• Section 10 describes the new dataset U3DPW and its
collection process.
7. Hierarchical Human Model
7.1. Choice of Local Coordinate System
As mentioned at equation 2 in section 3.1, each global
bone vector is transformed into a local bone vector with
respect to a coordinate system attached at a parent joint.
In general, the choice of the coordinate system is arbitrary
and our evolutionary operators do not depend on it. In im-
plementation, we adopt the coordinate system proposed in
[2], where the computation of basis vectors depends on the
3D joint position. For the bone vectors representing upper
limbs (left shoulder to left elbow, right shoulder to right el-
bow, left hip to left knee, right hip to right knee), the basis
vectors are computed based on several joints belonging to
the human torso. For the bone vectors representing lower
limbs (left elbow to left wrist, right elbow to right wrist,
left knee to left ankle, right knee to right ankle), the basis
vectors are computed from the parent bone vectors.
Algorithm 2 is adapted from [2] and details the pro-
cess of computing basis vectors and performing coordi-
nate transformation. Bold name such as rightShoulder de-
notes the global position of the 3D skeleton joint. We de-
ﬁne a bone vector’s parent bone vector as the bone vec-
tor whose end point is the starting point of it.
An in-
dex mapping function M(i) is introduced here that maps
bone vector index i to the index of its parent bone vec-
tor. Consistent with the notations of the main text, we have
child(M(i)) = parent(i). In implementation, we found
that the joints used in [2] have slightly different semantic
meaning compared to the data provided by H36M. Thus we
use the bone vector connecting the spine and thorax joints to
approximate the backbone vector used in in [2] (backBone
in algorithm 2).
Algorithm 2 Computation of local bone vector
Input: ith global bone vector bg = bi
global, constant 3D vector a
Output: ith local bone vector bl = bi
local with its local coordinate
system Ri
1: backBone = Spine - Thorax
2: if bg is upper limb then
3:
v1 = rightShoulder - leftShoulder
4:
v2 = backBone
5: else if bg is lower limb then
6:
v1 = rightHip - leftHip
7:
v2 = backBone
8: else
9:
v1 = bM(i)
g
10:
v2 = RM(i)a × v1
11: end if
12: Ri = GramSchmidt(v1, v2, v1 × v2)
13: bl = RiT bg
14: return bl
7.2. Validity Function
To implement v(p), local bone vectors are ﬁrst computed
by Algorithm 2 and converted into spherical coordinates as
bi
local = (ri, θi, φi). A pose p is then considered as the
collection of bone orientations (θi, φi)w
i=1. A function is
provided by [2] to decide the validity of each tuple (θi, φi).
We deﬁne a pose p to be anthropometrically valid if every
tuple (θi, φi) is valid:
v(p) =
(
0,
if (θi, φi) is valid for i=1,2,...,w,
−∞,
else.
The original code released by [2] was implemented by
MATLAB and we provide a Python implementation on our
project website.
8. Model Training
8.1. Training Procedure of the Cascaded Model
We train each deep learner in the cascade sequentially as
depicted by algorithm 3. The TrainNetwork is a routine
representing the training process of a single deep learner,
which consists of forward pass, backward pass and network
parameter update using Adam optimizer. Starting from the
second deep learner, the inputs can also be concatenated
with the current estimates as {φ(xi), ˆpi}N
i=1, which results
in slightly smaller training errors while the change of testing
errors is not obvious in our experiments on H36M.
8.2. Data Pre-processing
To train the heatmap regression model A(x), we down-
load training videos from the ofﬁcial website of H36M. We
crop the persons with the provided bounding boxes and pad
the cropped images with zeros in order to ﬁx the aspect ra-
tio as 4:3. We then resize the padded images to 384 by 288.
9

Protocol #1
Dir.
Disc
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
Avg.
Martinez et al. (ICCV’17)[37]
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
Fang et al. (AAAI’18) [17]
50.1
54.3
57.0
57.1
66.6
73.3
53.4
55.7
72.8
88.6
60.3
57.7
62.7
47.5
50.6
60.4
Yang et al. (CVPR’18) [71]
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
Pavlakos et al. (CVPR’18) [46]
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
Lee et al. (ECCV’18) [27]
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
Zhao et al. (CVPR’19) [73]
47.3
60.7
51.4
60.5
61.1
49.9
47.3
68.1
86.2
55.0
67.8
61.0
42.1
60.6
45.3
57.6
Sharma et al. (ICCV’19) [59]
48.6
54.5
54.2
55.7
62.6
72.0
50.5
54.3
70.0
78.3
58.1
55.4
61.4
45.2
49.7
58.0
Moon et al. (ICCV’19) [41]
51.5
56.8
51.2
52.2
55.2
47.7
50.9
63.3
69.9
54.2
57.4
50.4
42.5
57.5
47.7
54.4
Liu et al. (ECCV’20) [33]
46.3
52.2
47.3
50.7
55.5
67.1
49.2
46.0
60.4
71.1
51.5
50.1
54.5
40.3
43.7
52.4
Ours: Evolve(S15678)
47.0
47.1
49.3
50.5
53.9
58.5
48.8
45.5
55.2
68.6
50.8
47.5
53.6
42.3
45.6
50.9
Protocol #2
Dir.
Disc
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
Avg.
Martinez et al. (ICCV’17) [37]
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
Fang et al. (AAAI’18) [17]
38.2
41.7
43.7
44.9
48.5
55.3
40.2
38.2
54.5
64.4
47.2
44.3
47.3
36.7
41.7
45.7
Pavlakos et al. (CVPR’18) [46]
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
Yang et al. (CVPR’18) [71]
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
Sharma et al. (ICCV’19) [59]
35.3
35.9
45.8
42.0
40.9
52.6
36.9
35.8
43.5
51.9
44.3
38.8
45.5
29.4
34.3
40.9
Cai et al. (ICCV’19) [9]
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
Liu et al. (ECCV’20) [33]
35.9
40.0
38.0
41.5
42.5
51.4
37.8
36.0
48.6
56.6
41.8
38.3
42.7
31.7
36.2
41.2
Ours: Evolve(S15678)
34.5
34.9
37.6
39.6
38.8
45.9
34.8
33.0
40.8
51.6
38.0
35.7
40.2
30.2
34.8
38.0
Table 6: Quantitative comparisons with the state-of-the-art fully-supervised methods on Human3.6M under protocol #1 and
protocol #2. Best performance is indicated by bold font.
Protocol #1
Dir.
Disc
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
Avg.
Kocabas et al. (CVPR’19) [25]
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
65.3
Pavllo et al. (CVPR’19) [49]
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
64.7
Li et al. (ICCV’19) [30]
70.4
83.6
76.6
78.0
85.4
106.1
72.2
103.0
115.8
165.0
82.4
74.3
94.6
60.1
70.6
88.8
Ours: Evolve(S1)
52.8
56.6
54.0
57.5
62.8
72.0
55.0
61.3
65.8
80.7
59.0
56.7
69.7
51.6
57.2
60.8
Protocol #2
Dir.
Disc
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
Avg.
Rhodin et al. (CVPR’18) [54]
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
64.6
Kocabas et al. (CVPR’19) [25]
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
57.2
Li et al. (ICCV’19) [30]
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
66.5
Ours: Evolve(S1)
40.1
43.4
41.9
46.1
48.2
55.1
42.8
42.6
49.6
61.1
44.5
43.2
51.5
38.0
44.4
46.2
Table 7: Quantitative comparisons with the state-of-the-art weakly/semi-supervised methods on Human3.6M under protocol
#1 and protocol #2. Best performance is indicated by bold font.
Algorithm 3 Cascaded Deep Networks Training
Input:
Training set {φ(xi), pi}N
i=1, cascade length T
Output: G(c) = PT
t=1 Dt(it, Θt)
1: Current estimate {ˆp

## experiments
To validate our data evolution framework, we evolve
from the training data provided in H36M and investigate
how data augmentation may affect the generalization ability
of 2D-to-3D networks. We conduct both intra- and cross-
dataset evaluation. Intra-dataset evaluation is performed on
H36M and demonstrates the model performance in an in-
door environment similar to the training data. Cross-dataset
evaluation is conducted on datasets not seen during training
to simulate a larger domain shift. Considering the avail-
ability of MC data may vary in different application sce-
narios, We vary the size of initial population starting from
scarce training data. These experiments help comparison
with other weakly/semi-supervised methods that only use
very few 3D annotation but do not consider data augmen-
tation.
Finally we present an ablation study to analyze
the inﬂuences of architecture design and choice of hyper-
parameters.
5.1. Datasets and Evaluation Metrics
Human 3.6M (H36M) is the largest 3D human pose es-
timation benchmark with accurate 3D labels. We denote a
collection of data by appending subject ID to S, e.g., S15
denotes data from subject 1 and 5. Previous works ﬁx the
training data while our method uses it as our initial popu-
lation and evolves from it. We evaluate model performance
with Mean Per Joint Position Error (MPJPE) measured in
millimeters. Two standard evaluation protocols are adopted.
Protocol 1 (P1) directly computes MPJPE while Protocol 2
(P2) aligns the ground-truth 3D poses with the predictions
with a rigid transformation before calculating it. Protocol
P1∗uses ground truth 2D key-points as inputs and removes
the inﬂuence of the ﬁrst stage model.
MPI-INF-3DHP (3DHP) is a benchmark that we use
to evaluate the generalization power of 2D-to-3D networks
in unseen environments. We do not use its training data
and conduct cross-dataset inference by feeding the provided
key-points to G(c). Apart from MPJPE, Percentage of Cor-
rect Keypoints (PCK) measures correctness of 3D joint pre-
dictions under a speciﬁed threshold, while Area Under the
Curve (AUC) is computed for a range of PCK thresholds.
Unconstrained 3D Poses in the Wild (U3DPW) We
collect by ourselves a new small dataset consisting of 300
challenging in-the-wild images with rare human poses,
where 150 of them are selected from Leeds Sports Pose
dataset [23]. The annotation process is detailed in our sup-
plementary material. This dataset is used for qualitatively
validating model generalization for unseen rare poses.
5.2. Comparison with state-of-the-art methods
Comparison with weakly-supervised methods. Here we
compare with weakly/semi-supervised methods, which only
use a small number of training data to simulate scarce data
scenario. To be consistent with others, we utilize S1 as
our initial population. While others ﬁx S1 as the training
dataset, we evolve from it to obtain an augmented train-
ing set. The comparison of model performance is shown
in Tab. 2, where our model signiﬁcantly out-performs oth-
ers and demonstrates effective use of the limited training
data. While other methods [54, 25] use multi-view consis-
tency as extra supervision, we achieve comparable perfor-
mance with only a single view by synthesizing useful su-
pervision. Fig. 2 validates our method when the training
data is extremely scarce, where we start with a small frac-
tion of S1 and increase the data size by 2.5 times by evolu-
tion. Note that the model performs consistently better after
dataset evolution. Compared to the temporal convolution
model proposed in [49], we do not utilize any temporal in-
formation and achieve comparable performance. This indi-
cates our approach can make better use of extremely limited
6

data.
Method (Reference)
Average MPJPE↓
P1
P1*
P2
Use Multi-view Images
Rhodin et al. (CVPR’18) [54]
-
-
64.6
Kocabas et al. (CVPR’19) [25]
65.3
-
57.2
Use Temporal Information from Videos
Pavllo et al. (CVPR’19) [49]
64.7
-
-
Use a Single RGB Image
Li et al. (ICCV’19) [30]
88.8
-
66.5
Ours (CVPR’ 20)
60.8
50.5
46.2
Table 2: Comparison with SOTA weakly-supervised meth-
ods. Average MPJPE over all 15 actions for H36M under
two protocols (P1 and P2) is reported. P1* refers to proto-
col 1 evaluated with ground truth 2d key-points. Best per-
formance is marked with bold font. Error for each action
can be found in Tab. 7.
Comparison with fully-supervised methods.
Here we
compare with fully-supervised methods that uses the whole
training split of H36M. We use S15678 as our initial popu-
lation and Tab. 3 shows the performance comparison. Un-
der this setting, our model also achieves competitive perfor-
mance compared with other SOTA methods, indicating that
our approach is not limited to scarce data scenario.
Method (Reference)
Average MPJPE↓
P1
P1*
P2
Martinez et al. (ICCV’17) [37]
62.9
45.5
47.7
Yang et al. (CVPR’18) [71]
58.6
-
37.7
Zhao et al. (CVPR’19) [73]
57.6
43.8
-
Sharma et al. (ICCV’19) [58]
58.0
-
40.9
Moon et al. (ICCV’19) [41]
54.4
35.2
-
Ours (CVPR’ 20)
50.9
34.5
38.0
Table 3: Comparison with SOTA methods under fully-
supervised setting. Same P1, P1* and P2 as in Tab. 2. Error
for each action can be found in Tab. 6.
5.3. Cross-dataset Generalization
To validate the generalization ability of our 2D-to-3D
network in unknown environment, Tab. 4 compares with
other methods on 3DHP. In this experiment we evolve from
S15678 in H36M to obtain an augmented dataset consist-
ing of 8 million 2D-3D pairs. Without utilizing any train-
ing data of 3DHP, G(c) achieves competitive performance
in this benchmark. We obtain clear improvements compar-
ing with [28], which also uses S15678 as the training data
but ﬁx it without data augmentation. The results indicate
that our data augmentation approach improves model gen-
eralization effectively despite we start with the same biased
training dataset. As shown in Fig. 7, the distribution of the
augmented dataset indicates less dataset bias. Qualitative
results on 3DHP and LSP are shown in Fig. 6. Note that
Figure 7: Dataset distribution for the bone vector connect-
ing right shoulder to right elbow. Top: distribution before
(left) and after (right) dataset augmentation. Bottom: distri-
bution overlaid with valid regions (brown) taken from [2].
these unconstrained poses are not well-represented in the
original training dataset yet our model still gives good in-
ference results. Qualitative comparison with [28] on some
difﬁcult poses in U3DPW is shown in Fig. 8 and our G(c)
generalizes better for these rare human poses.

## related_work
Monocular 3D human pose estimation.
Single-image
3D pose estimation methods are conventionally categorized
into generative methods and discriminative methods. Gen-
erative methods ﬁt parametrized models to image obser-
vations for 3D pose estimation.
These approaches rep-
resent humans by PCA models [2, 75], graphical mod-
els [8, 5] or deformable meshes [4, 34, 7, 45, 26]. The
ﬁtting process amounts to non-linear optimization, which
requires good initialization and reﬁnes the solution iter-
atively.
Discriminative methods [57, 1, 6] directly learn
a mapping from image observations to 3D poses.
Re-
cent deep neural networks (DNNs) fall into this category
and employ two mainstream architectures: one-stage meth-
ods [71, 74, 36, 47, 44, 31, 65, 18] and two-stage meth-
ods [42, 37, 51, 73]. The former directly map pixel inten-
sities to 3D poses, while the latter ﬁrst extract intermediate
representation such as 2D key-points and then lift them to
3D poses.
We adopt the discriminative approach and focus on the
2D-to-3D lifting network. Instead of using a ﬁxed training
dataset, we evolve the training data to improve the perfor-
mance of the 2D-to-3D network.
Weakly-supervised 3D human pose estimation. Super-
vised training of DNNs demands massive data while 3D
annotation is difﬁcult. To address this problem, weakly-
supervised methods explore other potential supervision to
improve network performance when only few training data
is available [48, 53, 54, 25, 12, 70, 30]. Multi-view con-
sistency [48, 53, 54, 25, 12] is proposed and validated as
useful supervisory signal when training data is scarce, yet a
minimum of two views are needed. In contrast, we focus on
effective utilization of scarce training data by synthesizing
new data from existing ones and uses only single view.
Data augmentation for pose estimation. New images can
be synthesized to augment indoor training dataset [55, 68].
In [68] new images were rendered using MC data and hu-
man models. Domain adaption was performed in [11] dur-
ing training with synthetic images. Adversarial rotation and
scaling were used in [50] to augment data for 2D pose es-
timation. These works produce synthetic images while we
focus on data augmentation for 2D-to-3D networks and pro-
duce synthetic 2D-3D pairs.
Pose estimation dataset. Most large-scale human pose es-
timation datasets [72, 32, 3] only provide 2D pose anno-
tations. Accurate 3D annotations [22, 61] require MC de-
vices and these datasets are biased due to the limitation of
data collection process. Deep models are prone to overﬁt
to these biased dataset [66, 67, 29], failing to generalize in
unseen situations. Our method can synthesize for free with-
out human annotation large amount of valid 3D poses with
larger coverage in human pose space.
3. Dataset Evolution
From a given input image xi containing one human sub-
ject, we aim to infer the 3D human pose ˆpi given the im-
age observation φ(xi). To encode geometric information as
other 2D-to-3D approaches [37, 73, 28], we represent φ(x)
as the 2D coordinates of k human key-points (xi, yi)k
i=1 on
the image plane. As a discriminative approach, we seek
a regression function F parametrized by Θ that outputs
3D pose as ˆpi = F(φ(xi), Θ).
This regression func-
tion is implemented as a DNN. Conventionally this DNN
is trained on a dataset collected by MC devices [61, 22].
This dataset consists of paired images and 3D pose ground
truths {(xi, pi)}N
i=1 and the DNN can be trained by gradi-
2

ent descent based on a loss function deﬁned over the train-
ing dataset L = PN
i=1 E(pi, ˆpi) where E is the error mea-
surement between the ground truth pi and the prediction
ˆpi = F(φ(xi), Θ).
Unfortunately, sampling bias exists during the data col-
lection and limits the variation of the training data. Hu-
man 3.6M (H36M) [22], the largest MC dataset, only con-
tains 11 subjects performing 15 actions under 4 viewpoints,
leading to insufﬁcient coverage of the training 2D-3D pairs
(φ(xi), pi). A DNN can overﬁt to the dataset bias and be-
come less robust to unseen φ(x).
For example, when a
subject starts street dancing, the DNN may fail since it is
only trained on daily activities such as sitting and walking.
This problem is even exacerbated for the weakly-supervised
methods [48, 54, 12] where a minute quantity of training
data is used to simulate the difﬁculty of data collection.
We take a non-stationary view toward the training data to
cope with this problem. While conventionally the collected
training data is ﬁxed and the trained DNN is not modiﬁed
during its deployment, here we assume the data and model
can evolve during their life-time. Speciﬁcally, we synthe-
size novel 2D-3D pairs based on an initial training dataset
and add them into the original dataset to form the evolved
dataset. We then re-train the model with the evolved dataset.
As shown in Fig. 2, model re-trained on the evolved dataset
has consistently lower generalization error, comparing to a
model trained on the initial dataset.
%0.1 S1 (245)
%1 S1 (2.42k)
%5 S1 (12.4k)
%10 S1 (24.8k)
Training data
70
80
90
100
110
90.8
78.1
72.5
65.2
113.1
81.8
71.3
71.0
106.8
76.4
64.2
63.5
MPJPE (mm)
Temporal convolution Pavllo et al. CVPR' 19
Before evolution
After evolution
Figure 2: Generalizing errors (MPJPE using ground truth
2D key-points as inputs) on H36M before and after dataset
evolution with varying size of initial population.
In the following we show that by using a hierarchical
representation of human skeleton, the synthesis of novel
2D-3D pairs can be achieved by evolutionary operators and
camera projection.
3.1. Hierarchical Human Representation
We represent a 3D human skeleton by a set of bones
organized hierarchically in a kinematic tree as shown in
Right Shoulder
Left Shoulder
Right Elbow
Right Wrist
Left Elbow
Left Wrist
Right Knee
Right Foot
Left Knee
Left Foot
Head
Neck
Thorax
Spine
Pelvis
Right 
Hip
Left 
Hip
Parent
Child
i
j
k
Bone 
Vector
Nose
Figure 3: Hierarchical human representation. Left: 3D key-
points organized in a kinematic tree where red arrows point
from parent joints to children joints. Right: Zoom-in view
of a local coordinate system.
Parents
Children
Mutation
Crossover
P
C
P
C
Figure 4: Examples of evolutionary operation. Crossover
and mutation take two and one random samples respectively
to synthesize novel human skeletons. In this example the
right arms are selected for crossover while the left leg is
mutated.
Fig. 3. This representation captures the dependence of ad-
jacent joints with tree edges.
Each 3D pose p corresponds to a set of bone vectors
{b1, b2, · · · , bw} and a bone vector is deﬁned as
bi = pchild(i) −pparent(i)
(1)
where pj is the jth joint in the 3D skeleton and parent(i)
gives the parent joint index of the ith bone vector. A local
coordinate system2 is attached at each parent node. For a
parent node pparent(i), its local coordinate system is repre-
sented by the rotation matrix deﬁned by three basis vectors
Ri = [ii, ji, ki]. The global bone vector is transformed into
this local coordinate system as
bi
local = RiTbi
global = RiT(pchild(i) −pparent(i)) (2)
For convenience, this local bone vector is further converted
into spherical coordinates bi
local = (ri, θi, φi). The posture
2The coordinate system is detailed in our supplementary material.
3

Algorithm 1 Data evolution
Input:
Initial set of 3D skeletons Dold = {pi}N
i=1, noise level σ, number of
generations G
Output: Augmented set of skeletons Dnew = {pi}M
i=1
1: Dnew = Dold
2: for i=1:G do
3:
Parents = Sample(Dnew)
4:
Children = NaturalSelection(Mutation(Crossover(Parents)))
5:
Dnew = Dnew ∪Children
6: end for
7: return Dnew
of the skeleton is described by the collection of bone orien-
tations {(θi, φi)}w
i=1 while the skeleton size is encoded into
{ri}w
i=1.
3.2. Synthesizing New 2D-3D Pairs
We ﬁrst synthesize new 3D skeletons Dnew = {pj}M
j=1
with an initial training dataset Dold = {pi}N
i=1 and project
3D skeletons to 2D given camera intrinsics K to form 2D-
3D pairs {(φ(xj), pj)}M
i=1 where φ(xj) = Kpj.
When adopting the hierarchical representation, a dataset
of articulated 3D objects is a population of tree-structured
data in nature.
Evolutionary operators [20] have con-
structive property [62] that can be used to synthesize new
data [16] given an initial population. The design of opera-
tors is problem-dependent and our operators are detailed as
follows.
Crossover Operator Given two parent 3D skeletons rep-
resented by two kinematic trees, crossover is deﬁned as a
random exchange of sub-trees. This deﬁnition is inspired
by the observation that an unseen 3D pose might be ob-
tained by assembling limbs from known poses. Formally,
we denote the set of bone vectors for parent A and B as
SA = {b1
A, b2
A, . . . , bw
A} and SB = {b1
B, b2
B, . . . , bw
B}.
A joint indexed by q is selected at random and the bones
rooted at it are located for the two parents. These bones
form the chosen sub-tree set Schosen
{bj : parent(j) = q ∨IsOff(

## conclusion
This paper presents an evolutionary framework to enrich
the 3D pose distribution of an initial biased training set.
This approach leads to better intra-dataset and cross-dataset
generalization of 2D-to-3D network especially when avail-
able 3D annotation is scarce.
A novel cascaded 3D hu-
man pose estimation model is trained achieving state-of-
the-art performance for single-frame 3D human pose esti-
1
3
5
7
9
Number of blocks R
0
10
20
MPJPE (mm) under P1*
Train: BE
Test: BE
Train: AE
Test: AE
Figure 10: MPJPE (P1*) before (BE) and after evolution
(AE) with varying number of blocks R. Evolved training
data can afford a deeper network. Best viewed in color.