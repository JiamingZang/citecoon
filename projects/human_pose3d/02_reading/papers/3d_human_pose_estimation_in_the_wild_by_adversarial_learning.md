# 3D Human Pose Estimation in the Wild by Adversarial Learning

> 2018 · id: W2795089319 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

3D Human Pose Estimation in the Wild by Adversarial Learning
Wei Yang1
Wanli Ouyang2
Xiaolong Wang3
Jimmy Ren4
Hongsheng Li1
Xiaogang Wang1
1 CUHK-SenseTime Joint Lab, The Chinese University of Hong Kong
2 School of Electrical and Information Engineering, The University of Sydney
3 The Robotics Institute, Carnegie Mellon University
4 SenseTime Research
Abstract
Recently, remarkable advances have been achieved in
3D human pose estimation from monocular images because
of the powerful Deep Convolutional Neural Networks (DC-
NNs). Despite their success on large-scale datasets col-
lected in the constrained lab environment, it is difﬁcult
to obtain the 3D pose annotations for in-the-wild images.
Therefore, 3D human pose estimation in the wild is still a
challenge. In this paper, we propose an adversarial learn-
ing framework, which distills the 3D human pose structures
learned from the fully annotated dataset to in-the-wild im-
ages with only 2D pose annotations. Instead of deﬁning
hard-coded rules to constrain the pose estimation results,
we design a novel multi-source discriminator to distinguish
the predicted 3D poses from the ground-truth, which helps
to enforce the pose estimator to generate anthropometri-
cally valid poses even with images in the wild. We also
observe that a carefully designed information source for the
discriminator is essential to boost the performance. Thus,
we design a geometric descriptor, which computes the pair-
wise relative locations and distances between body joints,
as a new information source for the discriminator. The ef-
ﬁcacy of our adversarial learning framework with the new
geometric descriptor has been demonstrated through exten-
sive experiments on widely used public benchmarks. Our
approach signiﬁcantly improves the performance compared
with previous state-of-the-art approaches.
1. Introduction
Human pose estimation is a fundamental yet challeng-
ing problem in computer vision. The goal is to estimate 2D
or 3D locations of body parts given an image or a video,
which provides informative knowledge for tasks such as ac-
tion recognition, robotics vision, human-computer interac-
tion, and autonomous driving. Signiﬁcant advances have
Real Samples
t
e
s
a
t
a
d 
D
3
2D dataset
3D Human Pose EsƟmator
MulƟ-source Discriminator
PredicƟon
Ground-truth
Real
Fake
(a) 3D dataset
(b) 2D dataset
(c) 3D dataset
Predicted Samples
Figure 1. Given a monocular image and its predicted 3D pose, the
human can easily tell whether the prediction is anthropometrically
plausible or not (as shown in b) based on the perception of image-
pose correspondence and the possible human poses constrained by
articulation. We simulate this human perception by proposing an
adversarial learning framework, where the discriminator is learned
to distinguish ground-truth poses (c) from the predicted poses gen-
erated by the pose estimator (a, b), which in turn is enforced to
generate plausible poses even on unannotated in-the-wild data.
been achieved in 2D human pose estimation recently be-
cause of the powerful Deep Convolutional Neural Networks
(DCNNs) and the availability of large-scale in-the-wild hu-
man pose datasets with manual annotations.
However, advances in 3D human pose estimation remain
limited. The reason is mainly from the difﬁculty to obtain
ground-truth 3D body joint locations in the unconstrained
environment. Existing datasets such as Human3.6M [19]
are collected in the constrained lab environment using mo-
cap systems, hence the variations in background, viewpoint,
and lighting are very limited. Although DCNNs ﬁt well on
these datasets, when being applied on in-the-wild images,
where only 2D ground-truth annotations are available (e.g.,
the MPII human pose dataset [1]), they may have difﬁculty
in terms of generalization ability due to the large domain
5255

shift [44] between the constrained lab environment images
and unconstrained in-the-wild images, as shown in Figure 1.
On the other hand, given a monocular in-the-wild im-
age and its corresponding predicted 3D pose, it is relatively
easy for the human to tell if this estimation is correct or not,
as demonstrated in Figure 1(b). Human makes such deci-
sions mainly based on the human perception of image-pose
correspondence and possible human poses constrained by
articulation. This human perception can be simulated by a
discriminator, which is a neural network that discriminates
ground-truth poses from estimations.
Based on the above observation, we propose an adver-
sarial learning paradigm to distill the 3D human pose struc-
tures learned from the fully annotated constrained 3D pose
dataset to in-the-wild images without 3D pose annotations.
Speciﬁcally, we adopt an state-of-the-art 3D pose estima-
tor [57] as a conditional generator for generating pose es-
timations conditioned on input images.
The discrimina-
tor aims at distinguishing ground-truth 3D poses from pre-
dicted ones. Through adversarial learning, the generator
learns to predict 3D poses that is difﬁcult for the discrim-
inator to distinguish from the ground-truth poses. Since the
predicted poses can be also generated from in-the-wild data,
the generator must predict indistinguishable poses on both
domains to minimize the training error. It provides a way to
train the generator, i.e., the 3D pose estimator, with in-the-
wild data in a weakly supervised manner, and leads a better
generalization ability.
To facilitate the adversarial learning, a multi-source dis-
criminator is designed to take the two key factors into con-
sideration: 1) the description on image-pose correspon-
dence, and 2) the human body articulation constraint. One
indispensable information source is the original images. It
provides rich visual information for pose-image correspon-
dence.
Another information source of the discriminator
is the relative offsets and distances between pairs of body
parts, which is motivated by traditional approaches based on
pictorial structures [14, 54, 5, 32]. This information source
provides the discriminator with rich domain prior knowl-
edge, which helps the generator to generalize well.
Our approach improves the state-of-the-art both qualita-
tively and quantitatively. The main contributions are sum-
marized as follows.
• We propose an adversarial learning framework to dis-
till the 3D human pose structures from constrained im-
ages to unconstrained domains, where the ground-truth
annotations are not available. Our approach allows the
pose estimator to generalize well on another domain in a
weakly supervised manner instead of hard-coded rules.
• We design a novel multi-source discriminator, which
uses visual information as well as relative offsets and
distances as the domain prior knowledge, to enhance the
generalization ability of the 3D pose estimator.
2. Related Work
2.1. 2D Human Pose Estimation
Conventional methods usually solved 2D human poses
estimation by tree-structured models, e.g., pictorial struc-
tures [32] and mixtures of body parts [54, 5]. These models
consist of two terms: a unary term to detect the body joints,
and a pairwise term to model the pairwise relationships be-
tween two body joints. In [54, 5], a pairwise term was de-
signed as the relative locations and distances between pairs
of body joints. The symmetry of appearance between limbs
was modeled in [35, 39]. Ferrari et al. [13] designed repul-
sive edges between opposite-sided arms to tackle the double
counting problem. Inspired by aforementioned works, we
also use the relative locations and distances between pairs
of body joints. But they are used as the geometric descrip-
tor in the adversarial learning paradigm for learning bet-
ter 3D pose estimation features. The geometric descriptor
greatly reduces the difﬁculty for the discriminator in learn-
ing domain prior knowledge such as relative limbs length
and symmetry between limbs.
Recently, impressive advances have been achieved by
DCNNs [42, 49, 29, 8, 3, 53, 9, 52, 56]. Instead of directly
regressing coordinates [42], recent state-of-the-art methods
used heatmaps, which are generated by a 2D Gaussian cen-
tered on the body joint locations, as the target of regres-
sion. Our approach uses the state-of-the-art stacked hour-
glass [29] as our backbone architecture.
2.2. 3D Human Pose Estimation
Signiﬁcant progress has been achieved for 3D human
pose estimation from monocular images due to the avail-
ability of large-scale dataset [2] and the powerful DCNNs.
These methods can be roughly grouped into two categories.
One-stage approaches directly learn the 3D poses from
monocular images.
The pioneer work [22] proposed a
multi-task framework that jointly trains pose regression
and body part detectors. To model high-dimensional joint
dependencies, Tekin et al. [37] further adopted an auto-
encoder at the end of the network. Instead of directly re-
gressing the coordinates of the joints, Pavlakoset al. [31]
proposed a voxel representation for each joint as the regres-
sion target, and designed a coarse-to-ﬁne learning strategy.
These methods heavily depend on fully annotated datasets,
and cannot beneﬁt from large-scale 2D pose datasets.
Two-stage approaches ﬁrst estimate 2D poses and then
lift 2D poses to 3D poses [58, 4, 2, 51, 28, 40, 25, 57, 30].
These approaches usually generalize better on images in
the wild, since the ﬁrst stage can beneﬁt from the state-
of-the-art 2D pose estimators, which can be trained on im-
ages in the wild. The second stage usually regresses the
3D locations from the 2D predictions. For example, Mar-
tinez et al. [25] proposed a simple fully connected residual
5256

networks to directly regression 3D coordinates from 2D co-
ordinates. Moreno-Noguer [28] learned a pairwise distance
matrix, which is invariant to image rotation, translation, and
reﬂections, from 2D to 3D space.
To predict 3D poses for images in the wild, a geometric
loss was proposed in [57] to allow weakly supervised learn-
ing of the depth regression module. [26] adopted transfer
learning to generalize to in-the-wild scenes. [27] built a
real-time 3D pose estimation solution with kinematic skele-
ton ﬁtting. Our framework can use existing 3D pose esti-
mation approaches as the baseline and is complementary
to previous works by introducing an adversarial learning
framework, in which the predicted 3D poses from in-the-
wild images are used for learning better 3D pose estimator.
2.3. Adversarial Learning Methods
Adversarial learning for discriminative tasks. Adversar-
ial learning has been proven effective not only for genera-
tive tasks [16, 33, 46, 59, 47, 10, 18, 55, 21, 20, 45, 23], but
also for discriminative tasks [48, 50, 7, 6, 36]. For example,
Wang et al. [48] proposed to learn an adversarial network
that generates hard examples with occlusions and deforma-
tions for object detection. Wei et al. [50] designed an adver-
sarial erasing approach for weakly semantic segmentation.
An adversarial network was proposed in [7, 6] to distinguish
the ground-truth poses from the fake ones for human pose
estimation. The motivation and problems we are trying to
tackle are completely different from these work. In [7, 6],
the adversarial loss is used to improve pose estimation ac-
curacy with the same domain of the data. In our case, we
are trying to use adversarial learning to distill the structures
learned from the constrained data (with labels) in lab envi-
ronments to the unannotated data in the wild. Our approach
is also very different. [7, 6] only trained the models in one
single domain dataset, but ours incorporates the unanno-
tated data into the learning process, which takes a large step
in bridging the gap between the following two domains: 1)
in-the-wild data without 3D ground-truth annotations and
2) constrained data with 3D ground-truth annotations.
Adversarial learning for domain adaptation.
Recently,
adversarial methods have become an increasingly popular
incarnation for domain adaptation tasks [15, 43, 24, 44,
17]. These methods use adversarial learning to distinguish
source domain samples from target domain samples. And
adversarial learning aims at obtaining features that are do-
main uninformative. Different from these methods, our dis-
criminator aims at discriminating ground-truth 3D poses
from the estimated ones, which can be generated either from
the same domain as the ground-truth, or an unannotated do-
main (e.g., images in the wild).
3. Framework
As illustrated in Figure 1, our proposed framework
can be formulated as the Generative Adversarial Networks
(GANs), which consist of two networks: a generator and a
discriminator. The generator is trained to generate samples
in a way that confuses the discriminator, which in turn tries
to distinguish them from real samples. In our framework,
the generator G is a 3D pose estimator, which tries to predict
accurate 3D poses to fool the discriminator. The discrimi-
nator D distinguishes the ground-truth 3D poses from the
predicted ones. Since the predicted poses can be generated
from both the images captured from the lab environment
(with 3D annotations) and unannotated images in the wild,
the human body structures learned from 3D dataset can be
adapted to in-the-wild images through adversarial learning.
During training, we ﬁrst pretrain the pose estimator G
on 3D human pose dataset. Then we alternately optimize
the generator G and the discriminator D. For testing, we
simply discard the discriminator.
3.1. Generator: 3D Pose Estimator
The generator can be viewed as a two-stage pose estima-
tor. We adopt the state-of-the-art architecture [57] as our
backbone network for 3D human pose estimation.
The ﬁrst stage is the 2D pose estimation module, which
is the stacked hourglass network [29]. Each stack is in an
encoder-decoder structure. It allows for repeated top-down,
bottom-up inference across scales with intermediate super-
vision attached to each stack. We follow the previous prac-
tice to use 256 × 256 as input resolution. The outputs are P
heatmaps for the 2D body joint locations, where P denotes
the number of body joints. Each heatmap has size 64 × 64.
The second stage is a depth regression module, which
consists of several residual modules taking the 2D body
joint heatmaps and intermediate image features generated
from the ﬁrst stage as input. The output is a P × 1 vector
denoting the estimated depth for each body joint.
A geometric loss is proposed in [57] to allow weakly su-
pervised learning of the depth regression module on images
in the wild. We discard the geometric loss for a more con-
cise analysis of the proposed adversarial learning, although
our method is complementary to theirs.
3.2. Discriminator
The predicted poses by the generator G from both the
3D pose dataset and the in-the-wild images are treated as
“fake” examples for training the discriminator D.
At the adversarial learning stage, the pose estimator
(generator G) is learned so that the ground-truth 3D poses
and the predicted ones are indistinguishable for the discrim-
inator D. Therefore, this adversarial learning enforces the
predictions from in-the-wild images to have similar distri-
5257

CNN
2D Heatmaps
Depthmaps
Image
CNN
CNN
Real
Fake
64
64
Geometric descriptor
[∆, ∆, ∆]
[∆
, ∆
, ∆
]
Fully Connected layers
256
(a)
(b)
(c)
Real or Fake samples
ConcatenaƟon
Figure 2. The multi-source architecture. It contains three informa-
tion sources, image, geometric descriptor, as well as the heatmaps
and depth maps.
The three information sources are separately
embedded and then concatenated for deciding if the input is the
ground-truth pose or the estimated pose.
butions with the ground-truth 3D poses. Although unanno-
tated in-the-wild images are difﬁcult to be directly used for
training the pose estimator G, their corresponding 3D poses
predictions can be utilized as “fake” examples for learning
better discriminator, which in turn is helpful for learning a
better pose estimator (generator).
Discriminator decides whether the estimated 3D poses
are similar to ground-truth or not. The quality of discrimi-
nator inﬂuences the pose estimator. Therefore, we design a
multi-source network architecture and a geometric descrip-
tor for the discriminator.
3.2.1
Multi-Source Architecture
In the discriminator, there are three information sources: 1)
the original image, 2) the pairwise relative locations and dis-
tances, and 3) the heatmaps of 2D locations and the depths
of body joints. The information sources take two key factors
into consideration: 1) the description on image-pose corre-
spondence; and 2) the human body articulation constraints.
To model image-pose correspondence, we treat the orig-
inal image as the ﬁrst information source, which provides
rich visual and contextual information to reduce ambigui-
ties, as shown in Figure 2(a).
To learn the body articulation constraints, we design
a geometric descriptor as the second information source
(Figure 2(b)), which is motivated by traditional approaches
based on pictorial structures. It explicitly encodes the pair-
wise relative locations and distances between body parts,
and reduces the complexity to learn domain prior knowl-
edge, e.g., relative limbs length, limits of joint angles, and
symmetry of body parts. Details are given in Section 3.2.2.
Additionally, we also investigate using heatmaps as an-
other information source, which is effective for 2D adver-
sarial pose estimation [7]. It can be considered as a rep-
resentation of raw body joint locations, from which the
network could extract rich and complex geometric rela-
tionships within the human body structure.
Originally,
heatmaps are generated by a 2D Gaussian centered on the
body part locations. In order to incorporate the depth infor-
mation into this representation, we created P depth maps,
which have the same resolution as the 2D heatmaps for
body joints. Each map is a matrix denoting the depth of
a body joint at the corresponding location. The heatmaps
and depth maps are further concatenated as the third infor-
mation source, as shown in Figure 2 (c).
3.2.2
Geometric Descriptor
Our design of the geometric descriptor is motivated by the
quadratic deformation constraints widely used in pictorial
structures [54, 32, 5] for 2D human pose estimation. It en-
codes the spatial relationships, limbs length and symmetry
of body parts. By extending it from 2D to 3D space, we
deﬁne the 3D geometric descriptor d(·, ·) between pairs of
body joints as a 6D vector
d(zi, zj) = [∆x, ∆y, ∆z, ∆x2, ∆y2, ∆z2]T ,
(1)
where zi = (xi, yi, zi) and zj = (xj, yj, zj) denote the 3D
coordinates of the body joint i and j. ∆x = xi −xj, ∆y =
yi −yj and ∆z = zi −zj are the relative locations of joint i
with respect to joint j. ∆x2 = (xi−xj)2, ∆y2 = (yi−yj)2
and ∆z2 = (zi −zj)2 are distances between i and j.
We compute the 6D geometric descriptor in Eq. (1) for
each pair of body joint, which results in a 6× P × P matrix
for P body joints.
4. Learning
GANs are usually trained from scratch by optimizing the
generator and the discriminator alternately [16, 33]. For our
task, however, we observe that the training will converge
faster and get better performance with a pretrained genera-
tor (i.e., the 3D pose estimator).
We ﬁrst brieﬂy introduce the notation.
Let I
=
{(In, zn)}N
n=1 denote the datasets, where N denote the
sample indexes. Speciﬁcally, N = {N2D, N3D}, where
N2D and N3D are sample indexes for the 2D and 3D
pose datasets. Each sample (I, z) consists of a monocu-
lar image I and the ground-truth body joint locations z,
where z = {(xj, yj)}P
j=1 for 2D pose dataset, and z =
{(xj, yj, zj)}P
j=1 for 3D pose dataset. Here P denote the
number of body joints.
4.1. Pretraining of the Generator
We ﬁrst pretrain the 3D pose estimator (i.e. the gen-
erator), which consists of the 2D pose estimation module
5258

Human3.6M
MPII
60k iters
IniƟalizaƟon
120k iters
Figure 3. The predicted 3D poses become more accurate along
with the adversarial learning process.
and the depth regression module. We follow the standard
pipeline [41, 49, 3, 29] and formulate the 2D pose estima-
tion as the heatmap regression problem. The ground-truth
heatmap Sj for body joint j is generated from a Gaussian
centered at (xj, yj) with variance Σ, which is set as an iden-
tity matrix empirically. Denote the predicted 2D heatmaps
and depth as ˆSj and ˆzj respectively. The overall loss for
training pose estimator is deﬁned as the squared error
Lpose =
P
X
j=1



X
n∈N
heatmap regression
z
}|
{
∥Sj
n −ˆSj
n∥2
2 +
X
n∈N3D
depth regression
z
}|
{
∥zj
n −ˆzj
n∥2
2


.
(2)
As in previous works [25, 57], we adopt a pretrained
stacked hourglass networks [29] as the 2D pose estimation
module. Then the 2D pose module and the depth regression
module are jointly ﬁne-tuned with the loss in Eq.(2).
4.2. Adversarial Learning
After pretraining the 3D pose estimator G, we alternately
optimize G and D. The loss for training discriminator D is,
LD =
X
n∈N3D
Lcls (D(In, E(Sn, zn)), 1)
+
X
n∈N
Lcls(D (In, E(G(In))), 0) ,
(3)
where E(Sn, zn) encodes the heatmaps, depth maps
and the geometric descriptor as described in Section 3.
D(In, E(Sn, zn))
∈
[0, 1] represents the classiﬁcation
score of the discriminator given input image In and the
encoded information E(Sn, zn). G(In) is a 3D pose es-
timator which predicts heatmaps ˆSj
n and depth values ˆzj
n
given an input image In. Lcls is the binary entropy loss
deﬁned as Lcls(ˆy, y) = −(y log(ˆy) + (1 −y) log(1 −ˆy)).
Within each minibatch, half of samples are “real” from the
3D pose dataset, and the rest (In, E(G(In))) are generated
by G given an image In from 3D or 2D pose dataset. Intu-
itively, LD is optimized to enforce the network D to classify
the ground-truth poses as label 1 and the predictions as 0.
On the contrary, the generator G tries to generalize an-
thropometrically plausible poses conditioned on an image
to fool D via minimizing the following classiﬁcation loss,
LG =
X
n∈N
Lcls(D (In, E(G(In))), 1) .
(4)
We observe that directly train G and D with the loss pro-
posed in Eq.(3) and Eq.(4) reduces the accuracies of the
predicted poses. To regularize the training process, we in-
corporate the regression loss Lpose in Eq.(2) into Eq.(4),
which results in the following loss function,
LG = λ
X
n∈N
Lcls(D (In, E(G(In))), 1) + Lpose,
(5)
where λ is a hyperparameter to adjust the trade-off between
the classiﬁcation loss and the regression loss. λ is set as
1e −4 in the experiments.
Figure 3 demonstrates the improvements of predicted 3D
poses with the adversarial learning process. The initial pre-
dictions are anthropometrically invalid, and are easily dis-
tinguishable by D from the ground-truth poses. A relatively
large error LG is thus generated, and G is updated accord-
ingly to fool D better and produce improved results.
5. Experiments
Datasets. We conduct experiments on three popular human
pose estimation benchmarks: Human3.6M [19], MPI-INF-
3DHP [26] and MPII Human Pose [1].
Human3.6M [19] dataset is one of the largest datasets
for 3D human pose estimation. It consists of 3.6 million
images featuring 11 actors performing 15 daily activities,
such as eating, sitting, walking and taking a photo, from
4 camera views. The ground-truth 3D poses are captured
by the Mocap system, while the 2D poses can be obtained
by projection with the known intrinsic and extrinsic camera
parameters. We use this dataset for quantitative evaluation.
MPI-INF-3DHP [26] is a recently proposed 3D dataset
constructed by the Mocap system with both constrained in-
door scenes and complex outdoor scenes. We only use the
test split of this dataset, which contains 2929 frames from
six subjects performing seven actions, to evaluate the gen-
eralization ability quantitatively.
The MPII Human Pose [1] is the standard benchmark
for 2D human pose estimation.
It contains 25K uncon-
strained images collected from YouTube videos covering a
wide range of activities. We adopt this dataset for the 2D
pose estimation evaluation and the qualitative evaluation.
Evaluation protocols. We follow the standard protocol on
Human3.6M to use the subjects 1, 5, 6, 7, 8 for training and
the subjects 9 and 11 for evaluation. The evaluation metric
is the Mean Per Joint Position Error (MPJPE) in millimeter
between the ground-truth and the prediction across all cam-
eras and joints after aligning the depth of the root joints. We
refer to this as Protocol #1. In some works, the predictions
are further aligned with the ground-truth via a rigid trans-
form [2, 28, 25], which is referred as Protocol #2.
Implementation details. We adopt the network architec-
ture proposed in [57] as the backbone of our pose estimator.
5259

Protocol #1
Direct. Discuss Eating
Greet
Phone
Photo
Pose
Purch.
Sitting
SittingD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
LinKDE PAMI’16 [19]
132.7
183.6
132.3
164.4
162.1
205.9
150.6
171.3
151.6
243.0
162.1
170.7
177.1
96.6
127.9
162.1
Tekin et al., ICCV’16 [38]
102.4
147.2
88.8
125.3
118.0
182.7
112.4
129.2
138.9
224.9
118.4
138.8
126.3
55.1
65.8
125.0
Du et al. ECCV’16 [11]
85.1
112.7
104.9
122.1
139.1
135.9
105.9
166.2
117.5
226.9
120.0
117.7
137.4
99.3
106.5
126.5
Chen & Ramanan CVPR’17 [4]
89.9
97.6
89.9
107.9
107.3
139.2
93.6
136.0
133.1
240.1
106.6
106.2
87.0
114.0
90.5
114.1
Pavlakos et al. CVPR’17 [31]
67.4
71.9
66.7
69.1
72.0
77.0
65.0
68.3
83.7
96.5
71.7
65.8
74.9
59.1
63.2
71.9
Mehta et al. 3DV’17 [26]
52.6
64.1
55.2
62.2
71.6
79.5
52.8
68.6
91.8
118.4
65.7
63.5
49.4
76.4
53.5
68.6
Zhou et al. ICCV’17 [57]
54.8
60.7
58.2
71.4
62.0
65.5
53.8
55.6
75.2
111.6
64.1
66.0
51.4
63.2
55.3
64.9
Martinez et al. ICCV’17 [25]
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
Fang et al. AAAI’18 [12]
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
Ours (Full-2s)
53.0
60.8
47.9
57.1
61.5
65.5
50.8
49.9
73.3
98.6
58.8
58.1
42.0
62.3
43.6
59.7
Ours (Full-4s)
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
Protocol #2
Direct. Discuss Eating
Greet
Phone
Photo
Pose
Purch.
Sitting
SittingD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Ramakrishna et al. ECCV’12 [34]
137.4
149.3
141.6
154.3
157.7
158.9
141.8
158.1
168.6
175.6
160.4
161.7
150.0
174.8
150.2
157.3
Bogo et al. ECCV’16 [2]
62.0
60.2
67.8
76.5
92.1
77.0
73.0
75.3
100.3
137.3
83.4
77.3
86.8
79.7
87.7
82.3
Moreno-Noguer CVPR’17 [28]
66.1
61.7
84.5
73.7
65.2
67.2
60.9
67.3
103.5
74.6
92.6
69.6
71.5
78.0
73.2
74.0
Pavlakos et al. CVPR’17 [31]
–
–
–
–
–
–
–
–
–
–
–
–
–
–
–
51.9
Martinez et al. ICCV’17 [25]
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
Fang et al. AAAI’18 [12]
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
Ours (Full-4s)
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
Table 1. Quantitative comparisons of Mean Per Joint Position Error (MPJPE) in millimetre between the estimated pose and the ground-truth
on Human3.6M under Protocol #1 and Protocol #2. Some results are borrowed from [12].
Speciﬁcally, for 2D pose module, we adopt a shallower ver-
sion of stacked hourglass [29], i.e. 2 stacks with 1 residual
module at each resolution, for fast training in ablation stud-
ies (Table 2). The ﬁnal results in Table 1 are generated with
4 stacks of hourglass with 1 residual module at each res-
olution (i.e. Ours (Full-4s)), which has approximately the
same number of parameters but better performance com-
pared with the structure (2 stacks with 2 residual module at
each resolution) used in [57]. The depth regression mod-
ule consists of three sequential residual and downsampling
modules, a global average pooling, and a fully connected
layer for regressing the depth. The discriminator consists of
three fully connected layers after concatenating the three (or
two) branches of features embedded from three information
sources, i.e. the image, the heatmaps and depth maps, and
the pairwise geometric descriptors.
Following the standard training procedure as in [57, 25],
we ﬁrst pretrain the 2D pose estimator on the MPII dataset
to match the performance reported in [29]. Then we train
the full pose estimator with the pretrained 2D module on
Human3.6M for 200K iterations. To distill the learned 3D
poses to the unconstrained dataset, we then alternately train
the discriminator and pose estimator for 120k iterations.
The batch size is 12 for all the steps. All the experiments
were conducted on a single Titan X GPU. The forward time
during testing is about 1.1 second for a batch of 24 images.
5.1. Results on Human3.6M
Table 1 reports the comparison with previous methods
on Human3.6M. Our method (i.e. Ours (Full-4s)) achieves
the state-of-the-art results. For Protocol #1, our method ob-
tains 58.6 of mm of error, which has 9.7% improvements
compared to our backbone architecture [57], although the
geometric loss used in [57] is not used in our model for
clearer analysis. Comparing to the recent best result [12],
our method still has 3.0% improvement.
Under Protocol #2 (predictions are aligned with the
ground-truth via a rigid transform), our method obtains
37.7mm error, which improves the previous best result [12],
45.7mm, on a large margin (17.5% improvement).
5.1.1
Ablation Study
To investigate the efﬁcacy of each component, we conduct
ablation analysis on Human3.6M under Protocol #1. For
fast training, we adopt a shallower version of the stacked
hourglass, i.e. 2 stacks with 1 residual module at each reso-
lution (Ours (Full-2s) in Table 1), as the backbone architec-
ture for the 2D pose module. Mean errors of all the joints
and four limbs (i.e., upper/lower arms and upper/lower legs)
are reported in Table 2. The notations are as follows:
• Baseline refers to the pose estimator without adversarial
learning. The mean error of our baseline model is 64.8
mm, which is very close to the 64.9 mm error reported
on our backbone architecture in [57].
• Map refers to the use of heatmaps and depth maps, as
well as the original images for the adversarial training.
• Geo refers to use our proposed geometric descriptors as
well as the original images for the adversarial training.
• Full refers to use all the information sources, i.e., orig-
inal images, heatmaps and depth maps, and geometric
descriptors, for adversarial learning.
• Fix 2D refers to training with 2D pose module ﬁxed.
• W/o pretrain refers to adversarial learning without pre-
training the depth regressor.
Geometric features: heatmaps or pairwise geometric de-
scriptor? From Table 2, we observe that all the variants
with adversarial learning outperform the baseline model. If
5260

GT
Ours
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
(i)
(j)
Baseline
GT
Ours
Baseline
Figure 4. Predicted 3D poses on the Human3.6M validation set. Compared with the baseline pose estimator, the proposed adversarial
learning framework (Ours) is able to reﬁne the anatomically implausible poses, which is more similar to the ground-truth poses (GT).
Method
U.Arms
L.Arms
U.Legs
L.Legs
Mean
Baseline (ﬁx 2D)
67.6
89.6
46.6
83.3
65.2
Baseline
66.6
90.0
47.1
83.7
64.8
Map
62.9
81.6
44.6
80.9
61.3
Geo
61.6
80.7
43.9
78.8
60.3
Full (ﬁx 2D)
63.9
84.4
45.8
85.1
63.1
Full (w/o pretrain)
65.2
84.2
46.7
82.5
63.4
Full
61.7
81.1
43.1
77.6
59.7
Table 2. Ablation studies on the Human3.6M dataset under Proto-
col #1 with 2 stacks of hourglass. The ﬁrst two rows refer to the
baseline pose estimator without adversarial learning. Rest of the
rows refer to variants with adversarial learning. Please refer to the
text for the detailed descriptor for each variant.
30
40
50
60
70
80
90
100
110
1
11
21
31
41
51
61
71
81
Training
w/o pretraining
55
60
65
70
75
80
85
90
95
1
11
21
31
41
51
61
71
81
ValidaƟon
Mean Error
with pretraining
Figure 5. Training and validation curves of MPJPE (mm) vs.
epoch on the Human3.6M validation set. Better convergence rate
and performance have been achieved with the pretrained generator.
we use the image, the heatmaps and the depth maps as the
information source (Map) for the discriminator, the predic-
tion error is reduced by 3.5 mm. From the baseline model,
the pairwise geometric descriptor (Geo) introduced in Sec-
tion 3.2.2 reduces the prediction error by 4.5 mm. The pair-
wise geometric descriptor provides 1 mm lower mean er-
ror compared to the heatmaps (Map). This validates the
Model
Head
Sho.
Elb.
Wri.
Hip
Knee
Ank.
Mean
Pretrain
96.3
95.0
89.0
84.5
87.1
82.5
78.3
87.6
Ours
96.1
95.6
89.9
84.6
87.9
84.3
81.2
88.6
Table 3. PCKh@0.5 score on the MPII validation set.
effectiveness of the proposed geometric features in learn-
ing complex constraints in the articulated human body. By
combining all the three information sources together (Full),
our framework achieves the lowest error.
Adversarial learning: from scratch or not?
The stan-
dard practice to train GANs is to learn the generator and
the discriminator alternately from scratch [16, 33, 46, 59].
The generator is usually conditioned on noise [33], text [55]
or images [59], and lacks of ground-truth for supervised
training. This may not be necessary for our case because
our generator is actually the pose estimator and can be pre-
trained in a supervised manner. To investigate which train-
ing strategy is better, we train our full model with or with-
out pretraining the depth regressor. We found that it is eas-
ier to learn when the generator is pretrained: It not only
obtains lower prediction error (59.7 vs. 63.4 mm), but also
converges much faster, as shown by the training and valida-
tion curves of mean error vs. epoch in Figure 5.
Shall we ﬁx the pretrained 2D module? Since the 2D
pose estimator is mature enough [29, 49, 3]. Is it still nec-
essary to learn our model end-to-end to the 2D pose module
with more computational and memory cost? We ﬁrst inves-
tigate this issue with the baseline model. For the baseline
model, the top rows of Table 2 show that end-to-end learn-
ing (Baseline) is similar in performance compared to the
learning of depth regressor with 2D module ﬁxed (Baseline
5261

Ours
Baseline
Ours
Baseline
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
(i)
(j)
Figure 6. Qualitative comparison of images in the wild (i.e. the MPII human pose dataset [1]). anatomically implausible bent of limbs is
corrected by the adversarial learning. The last column shows typical failure cases caused by unseen camera views.
(Fix-2D)). For adversarial learning, on the other hand, the
improvement from end-to-end learning is obvious, with 3.4
mm (around 5%) error reduction when compare Full (Fix
2D) with Full in the table. Therefore, end-to-end training is
necessary to boost the performance in adversarial learning.
Adversarial learning for 2D pose estimation.
One may
wonder the performance of 2D module after the adversar-
ial learning. Therefore, we reported the PCKh@0.5 scores
for 2D pose estimation on the MPII validation set in Ta-
ble 3. Pretrain refers to our baseline 2D module without
adversarial training. Ours refers to the the model after the
adversarial learning. We observe that adversarial learning
reduces the error rate of 2D pose estimation by 8.1%.
Qualitative comparison.
To understand how adversar-
ial learning works, we compare the poses estimated by the
baseline model to those generated with adversarial learn-
ing.
Speciﬁcally, the high-level domain knowledge over
human poses, such as symmetry (Figure 4 (b,c,f,g,i)) and
kinematics (Figure 4 (b,c,g,f,i)), are encoded by the adver-
sarial learning. Hence the generator (i.e. the pose estimator)
is able to reﬁne the anatomically implausible poses, which
might be caused by left-right switch (Figure 4 (a, e)), clut-
tered background (Figure 4 (b)), double counting (Figure 4
(c,d,g)) and severe occlusion (Figure 4 (f,h,i)).
5.2. Cross-Domain Generalization
Quantitative results on MPI-INF-3DHP.
One way to
show that our algorithm is learning to transfer between do-
mains is to test our model on another unseen 3D pose esti-
mation dataset. Thus, we add a cross-dataset experiment on
a recently proposed 3D dataset MPI-INF-3DHP [26]. For
training, only the H36M and MPII are used, while MPI-
INF-3DHP is not used. We follow [26] to use PCK and
AUC as the evaluation metrics. Comparisons are reported
in Table 4. Baseline and Adversarial denote the pose esti-
mator without or with the adversarial learning, respectively.
We observe that the adversarial learning signiﬁcantly im-
[26]
Baseline
Ours
PCK
64.7
50.1
69.0
AUC
31.7
21.6
32.0
Table 4. PCK and AUC on the MPI-INF-3DHP dataset.
proves the generalization ability of the pose estimator.
Qualitative results on MPII. Finally, we demonstrate the
generalization ability qualitatively on the validation split of
the in-the-wild MPII human pose [1] dataset. Compared
with the baseline method without adversarial learning, our
discriminator is able to identify the unnaturally bent limbs
(Figure 6(a-c,g-i)) and asymmetric limbs (Figure 6(d)), and
to reﬁne the pose estimator through adversarial training.
One common failure case is shown in Figure 6(e). The
picture is a high-angle shot, which is not covered by the four
cameras in the 3D pose dataset. This issue could be proba-
bly solved by involving more camera views during training.
6. Conclusion
This paper has proposed an adversarial learning frame-
work to transfer the 3D human pose structures learned from
the fully annotated dataset to in-the-wild images with only
2D pose annotations. A novel multi-source discriminator, as
well as a geometric descriptor to encode the pairwise rela-
tive locations and distances between body joints, have been
introduced to bridge the gap between the predicted pose
from both domains and the ground-truth poses. Experimen-
tal results validate that the proposed framework improves
the pose estimation accuracy on 3D human pose dataset. In
the future work, we plan to investigate the augmentation of
camera views for better generalization ability.
Acknowledgment:
This work is supported in part by
SenseTime Group Limited, in part by the General Research
Fund through the Research Grants Council of Hong
Kong under Grants CUHK14213616, CUHK14206114,
CUHK14205615,
CUHK419412,
CUHK14203015,
CUHK14239816,
CUHK14207814,
CUHK14208417,
CUHK14202217, in part by the Hong Kong Innovation and
Technology Support Programme Grant ITS/121/15FX.
5262

References
[1] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. 2d
human pose estimation: New benchmark and state of the art
analysis. In CVPR, 2014. 1, 5, 8
[2] F. Bogo, A. Kanazawa, C. Lassner, P. Gehler, J. Romero,
and M. J. Black. Keep it smpl: Automatic estimation of 3d
human pose and shape from a single image. In ECCV, 2016.
2, 5, 6
[3] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime multi-
person 2d pose estimation using part afﬁnity ﬁelds. CVPR,
2017. 2, 5, 7
[4] C.-H. Chen and D. Ramanan. 3d human pose estimation= 2d
pose estimation+ matching. CVPR, 2017. 2, 6
[5] X. Chen and A. L. Yuille. Articulated pose estimation by a
graphical model with image dependent pairwise relations. In
NIPS, 2014. 2, 4
[6] Y. Chen, C. Shen, X.-S. Wei, L. Liu, and J. Yang. Adversar-
ial learning of structure-aware fully convolutional networks
for landmark localization. arXiv preprint arXiv:1711.00253,
2017. 3
[7] Y. Chen, C. Shen, X.-S. Wei, L. Liu, and J. Yang. Adversarial
posenet: A structure-aware convolutional network for human
pose estimation. ICCV, 2017. 3, 4
[8] X. Chu, W. Ouyang, H. Li, and X. Wang. Structured feature
learning for pose estimation. In CVPR, 2016. 2
[9] X. Chu, W. Yang, W. Ouyang, C. Ma, A. L. Yuille, and
X. Wang. Multi-context attention for human pose estima-
tion. CVPR, 2017. 2
[10] E. L. Denton, S. Chintala, and R. Fergus. Deep generative
image models using a laplacian pyramid of adversarial net-
works. In NIPS, 2015. 3
[11] Y. Du, Y. Wong, Y. Liu, F. Han, Y. Gui, Z. Wang, M. Kankan-
halli, and W. Geng. Marker-less 3d human motion capture
with monocular image sequence and height-maps. In ECCV,
2016. 6
[12] H. Fang, Y. Xu, W.