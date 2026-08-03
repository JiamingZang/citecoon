# Compositional Human Pose Regression

> 2017 · id: W2604375920 · arXiv: 1704.00159 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Regression based methods are not performing as well as
detection based methods for human pose estimation. A cen-
tral problem is that the structural information in the pose
is not well exploited in the previous regression methods.
In this work, we propose a structure-aware regression ap-
proach. It adopts a reparameterized pose representation us-
ing bones instead of joints. It exploits the joint connection
structure to deﬁne a compositional loss function that en-
codes the long range interactions in the pose. It is simple,
effective, and general for both 2D and 3D pose estimation
in a uniﬁed setting. Comprehensive evaluation validates the
effectiveness of our approach. It signiﬁcantly advances the
state-of-the-art on Human3.6M [20] and is competitive with
state-of-the-art results on MPII [3].

## introduction
Human pose estimation has been extensively studied for
both 3D [20] and 2D [3]. Recently, deep convolutional neu-
tral networks (CNNs) have achieved signiﬁcant progresses.
Existing approaches fall into two categories: detection
based and regression based. Detection based methods gen-
erate a likelihood heat map for each joint and locate the joint
as the point with the maximum value in the map. These heat
maps are usually noisy and multi-mode. The ambiguity is
reduced by exploiting the dependence between the joints in
various ways. A prevalent family of state-of-the-art meth-
ods [11, 6, 31, 5, 47, 18] adopt a multi-stage architecture,
where the output of the previous stage is used as input to
enhance the learning of the next stage. These methods are
dominant for 2D pose estimation [1]. However, they do not
easily generalize to 3D pose estimation, because the 3D heat
maps are too demanding for memory and computation.
Regression based methods directly map the input image
to the output joints. They directly target at the task and they
are general for both 3D and 2D pose estimation. Never-
theless, they are not performing as well as detection based
∗Corresponding author.
methods. As an evidence, only one method [6] in the 2D
pose benchmark [1] is regression based. While they are
widely used for 3D pose estimation [54, 30, 28, 43, 24, 42,
33], the performance is not satisfactory. A central problem
is that they simply minimize the per-joint location errors in-
dependently but ignore the internal structures of the pose.
In other words, joint dependence is not well exploited.
In this work, we propose a structure-aware approach,
called compositional pose regression. It is based on two
ideas. First, it uses bones instead of joints as pose represen-
tation, because the bones are more primitive, more stable,
and easier to learn than joints. Second, it exploits the joint
connection structure to deﬁne a compositional loss function
that encodes long range interactions between the bones.
The approach is simple, effective and efﬁcient. It only
re-parameterizes the pose representation, which is the net-
work output, and enhances the loss function, which relates
the output to ground truth. It does not alter other algorithm
design choices and is compatible with such choices, such as
network architecture. It can be easily adapted into any exist-
ing regression approaches with little overhead for memory
and computation, in both training and inference.
The approach is general and can be used for both 3D and
2D pose regression, indistinguishably. Moreover, 2D and
3D data can be easily mixed simultaneously in the training.
For the ﬁrst time, it is shown that such directly mixed learn-
ing is effective. This property makes our approach different
from all existing ones that target at either 3D or 2D task.
The effectiveness of our approach is validated by com-
prehensive evaluation with a few new metrics, rigorous ab-
lation study and comparison with state-of-the-art on both
3D and 2D benchmarks. Speciﬁcally, it advances the state-
of-the-art on 3D Human3.6M dataset [20] by a large margin
and achieves a record of 59.1 mm average joint error, about
12% relatively better that state-of-the-art.
On 2D MPII
dataset [3, 1], it achieves 86.4% (PCKh 0.5). It is the best-
performing regression based method and on bar with the
state-of-the-art detection based methods. As a by-product,
our approach generates high quality 3D poses for in the wild
images, indicating the potential of our approach for transfer
learning of 3D pose estimation in the wild.
1
arXiv:1704.00159v3  [cs.CV]  2 Aug 2017

## method
BL
Ours (all)
BL
Ours (all)
BL
Ours (all)
BL
Ours (all)
BL
Ours (all)
Average
102.2
92.4↓9.8
75.0
67.5↓7.5
65.5
58.4↓7.1
26.4
21.7↓4.7
3.7%
2.5%↓1.2
Ankle(→Knee)
94.5
88.5↓6.0
81.5
75.8↓5.7
81.2
74.1↓7.1
32.9
32.0↓0.9
-
-
Knee(→Hip)
68.6
63.7↓4.9
69.2
62.9↓6.3
69.1
63.4↓5.7
21.7
22.8↑1.1
4.8%
3.8%↓1.0
Hip(→Pelvis)
29.9
25.0↓4.9
63.3
58.4↓4.9
29.9
25.0↓4.9
21.3
16.4↓4.9
0.6%
0.6%↓0.0
Thorax(→Pelvis)
97.2
90.1↓7.1
30.7
28.1↓2.6
97.2
90.1↓7.1
28.0
26.7↓1.3
-
-
Neck(→Thorax)
104.3
96.4↓7.9
36.7
35.5↓1.2
22.2
22.9↑0.7
12.4
11.7↓0.7
2.2%
1.3%↓0.9
Head(→Neck)
115.4
108.4↓7.0
42.8
41.1↓1.7
39.7
37.3↓2.4
15.3
14.8↓0.5
-
-
Wrist(→Elbow)
181.9
163.0↓18.9
130.2
115.2↓15.0
102.6
89.0↓13.6
40.6
30.6↓10.0
-
-
Elbow(→Shoulder)
168.8
146.9↓21.9
115.8
97.6↓18.2
96.9
81.4↓15.5
27.6
21.4↓6.2
8.5%
5.4%↓3.1
Shoulder(→Thorax)
115.6
104.4↓11.2
57.7
52.2↓5.5
55.1
48.5↓6.6
25.9
12.6↓13.3
1.9%
0.8%↓1.1
Table 3. Detailed results on all joints for Baseline (BL) and Ours (all) methods, only trained on Human3.6M data (top half in Table 2). The
relative performance gain is shown in the subscript. Note that the left most column shows the names for both the joint (and the bone).
We therefore tested two sets of training data: 1) only Hu-
man3.6M; 2) Human3.6M plus MPII.
Table 2 reports the results under Protocol 2, which is
more commonly used. We observe several conclusions.
Using 2D data is effective. All metrics are signiﬁcantly
improved after using MPII data. For example, joint error
is reduced from 102.2 to 64.2. This improvement should
originate from the better learnt feature from the abundant
2D data. See the contemporary work [53] for more discus-
sions. Note that adding 2D data in this work is simple and
not considered as a main contribution. Rather, it is consid-
ered as a baseline to validate our regression approach.
Bone representation is superior than joint representa-
tion.
This can be observed by comparing Baseline with
Ours (joint) and Ours (bone). They are comparable because
they use roughly the same amount of supervision signals in
the training. The two variants of ours are better on nearly all
the metrics, especially the geometric constraint based ones.
Compositional loss is effective. When the loss function
becomes better (Ours (both) and Ours (all)), further im-
provement is observed. Speciﬁcally, when trained only on
Human3.6M, Ours (all) improves the Baseline by 9.8 mm
(relative 9.6%) on joint error, 7.5 mm (relative 10%) on PA
joint error, 7.1 mm (relative 10.8%) on bone error, 4.7 mm
(relative 17.8%) on bone std, and 1.2% (relative 32.4%) on
illegal angle.
Table 3 further reports the performance improvement
from Ours (all) to Baseline on all the joints (bones).
It
shows several conclusions.
First, limb joints are harder
than torso joints and upper limbs are harder than lower
limbs.
This is consistent as Figure 1 (middle).
It indi-
cates that the variance is a good indicator of difﬁculty and
a per-joint analysis is helpful in both algorithm design and
evaluation. Second, our method signiﬁcantly improves the
accuracy for all the joints, especially the challenging ones
like wrist, elbow and ankle. Figure 2 shows the results on a
testing video sequence with challenging arm motions. Our
result is much better and more stable.
Comparison with the state-of-the-art There are abun-
dant previous works. They have different experiment set-
tings and fall into three categories. They are compared to
our method in Table 4, 5, and 6, respectively.
The comparison is not completely fair due to the dif-
ferences in the training data (when extra data are used),
the network architecture and implementation. Nevertheless,
two common conclusions validate that our approach is ef-
fective and sets the new state-of-the-art in all settings by
a large margin. First, our baseline is strong. It is simple
but already improves the state-of-the-art, by 3.9 mm (rela-
tive 7%) in Table 4, 2.7 mm (relative 4%) in Table 5, and

 Frame 24
 Frame 102 Frame 119
 Frame 180
 Frame  200
 Frame 0
 Frame 158
Ours
(all)
Bone
Error
Joint
Error
 Baseline
Frame
(mm)
 Frame 68
Test Result
Image and 3D 
Ground Truth
Figure 2. (best viewed in color) Errors of wrist joint/bone of Baseline and Ours (all) methods on a video sequence from Human3.6M S9,
action Pose. The average error over the sequence is shown in the legends. For this action, the arms have large motion and are challenging.
Our method has much smaller joint and bone error. Our result is more stable over the sequence. The 3D predicted pose and ground truth
pose are visualized for a few frames. More video results are at https://www.youtube.com/watch?v=c-hgHqVK90M.

## experiments
Our approach is evaluated on 3D and 2D human pose
benchmarks.
Human3.6M [20] is the largest 3D human
pose benchmark. The dataset is captured in controlled en-
vironment. The image appearance of the subjects and the
background is simple. Accurate 3D human joint locations
are obtained from motion capture devices.
MPII [3] is the benchmark dataset for 2D human pose
estimation. It includes about 25k images and 40k annotated
2D poses. 25k of them are for training and another 7k of
the remaining are for testing. The images were collected
from YouTube videos covering daily human activities with
complex poses and image appearances.
5.1. Comprehensive Evaluation Metrics
For 3D human pose estimation, previous works [7, 44,
30, 57, 21, 28, 34, 52, 40, 4, 56, 43, 54] use the mean per
joint position error (MPJPE). We call this metric Joint Er-
ror. Some works [52, 40, 7, 4, 30, 57] ﬁrstly align the pre-
dicted 3D pose and ground truth 3D pose with a rigid trans-
CNN prediction
loss function
Baseline
joints J
L(J ), Eq.(3)
Ours (joint)
bones B
L(B, Pjoint), Eq.(8)
Ours (bone)
L(B, Pbone), Eq.(8)
Ours (both)
L(B, Pboth), Eq.(8)
Ours (all)
L(B, Pall), Eq.(8)
Table 1. The baseline and four variants of our method.
formation using Procrustes Analysis [15] and then compute
MPJPE. We call this metric PA Joint Error.
For 2D human pose estimation in MPII [3], Percentage
of Correct Keypoints (PCK) metric is used for evaluation.
Above metrics only measures the accuracy of absolute
joint location. They do not fully reﬂect the accuracy of in-
ternal structures in the pose. We propose three additional
metrics for a comprehensive evaluation.
The ﬁrst metric is the mean per bone position error, or
Bone Error. It is similar to Joint Error, but measures the
relative joint location accuracy. This metric is applicable
for both 3D and 2D pose.
The next two are only for 3D pose as they measure the
validity of 3D geometric constraints. Such metrics are im-
portant as violation of the constraints will cause physically
infeasible 3D poses. Such errors are critical for certain ap-
plications such as 3D motion capture.
The second metric is the bone length standard deviation,
or Bone Std. It measures the stability of bone length. For
each bone, the standard deviation of its length is computed
over all the testing samples of the same subject.
The third metric is the percentage of illegal joint angle,
or Illegal Angle. It measures whether the rotation angles at a
joint are physically feasible. We use the recent method and
code in [2] to evaluate the legality of each predicted joint.
Note that this metric is only for joints on the limbs and does
not apply to those on the torso.
5.2. Experiments on 3D Pose of Human3.6M
For Human3.6M [20], there are two widely used evalua-
tion protocols with different training and testing data split.
Protocol 1 Six subjects (S1, S5, S6, S7, S8, S9) are used
in training. Evaluation is performed on every 64th frame of
Subject 11’s videos. It is used in [52, 40, 7, 30, 57]. PA
Joint Error is used for evaluation.
Protocol 2 Five subjects (S1, S5, S6, S7, S8) are used
for training. Evaluation is performed on every 64th frame
of two subjects (S9, S11). It is used in [56, 43, 54, 7, 44,
30, 57, 21, 28, 34]. Joint Error is used for evaluation.
Ablation study. The direct joint regression baseline and
four variants of our method are compared. They are brieﬂy
summarized in Table 1. As explained in Section 4, train-
ing can use additional 2D data (from MPII), optionally.

Training Data
Metric
Baseline
Ours (joint)
Ours(bone)
Ours (both)
Ours (all)
Human3.6M
Joint Error
102.2
103.3↑1.1
104.6↑2.4
95.2↓7.0
92.4↓9.8
PA Joint Error
75.0
74.3↓0.7
75.0↓0.0
68.1↓6.9
67.5↓7.5
Bone Error
65.5
63.5↓2.0
62.3↓3.2
59.1↓6.4
58.4↓7.1
Bone Std
26.4
23.9↓2.5
21.9↓4.5
22.3↓4.1
21.7↓4.7
Illegal Angle
3.7%
3.2%↓0.5
3.3%↓0.4
2.6%↓1.1
2.5%↓1.2
Human3.6M + MPII
Joint Error
64.2
62.9↓1.3
63.8↓0.4
60.7↓3.5
59.1↓5.1
PA Joint Error
51.4
50.6↓0.8
50.4↓1.0
48.8↓2.6
48.3↓3.1
Bone Error
49.5
49.3↓0.2
47.4↓2.1
47.2↓2.3
47.1↓2.4
Bone Std
19.9
19.3↓0.6
17.5↓2.4
17.6↓2.3
18.0↓1.9
Table 2. Results of all methods under all evaluation metrics (the lower the better), with or without using MPII data in training. Note
that the performance gain of all Ours methods relative to the Baseline method is shown in the subscript. The Illegal Angle metric for
“Human3.6M+MPII” setting is not included because it is very good (< 1%) for all methods.
Metric
Joint Error
PA Joint Error
Bone Error
Bone Std
Illegal Angle

## related_work
Human pose estimation has been extensively studied for
years. A complete review is beyond the scope of this work.
We refer the readers to [29, 41] for a detailed survey.
The previous works are reviewed from two perspectives
related to this work. First is how to exploit the joint depen-
dency for 3D and 2D pose estimation. Second is how to
exploit “in the wild” 2D data for 3D pose estimation.
3D Pose Estimation Some methods use two separate
steps. They ﬁrst perform 2D joint prediction and then re-
construct the 3D pose via optimization or search. There is
no end-to-end learning. Zhou et al. [56] combines uncer-
tainty maps of the 2D joints location and a sparsity-driven
3D geometric prior to infer the 3D joint location via an EM
algorithm. Chen et al. [7] searches a large 3D pose library
and uses the estimated 2D pose as query. Bogo et al. [4] ﬁt a
recently published statistical body shape model [27] to the
2D joints. Jahangiri et al. [21] generates multiple hypothe-
ses from 2D joints using a novel generative model.
Some methods implicitly learn the pose structure from
data.
Tekin et al. [42] represents the 3D pose with an
over-complete dictionary. A high-dimensional latent pose
representation is learned to account for joint dependencies.
Pavlakos et al. [34] extends the Hourglass [31] framework
from 2D to 3D. A coarse-to-ﬁne approach is used to ad-
dress the large dimensionality increase. Li et al. [24] uses
an image-pose embedding sub-network to regularize the 3D
pose prediction.
Above works do not use prior knowledge in 3D model.
Such prior knowledge is ﬁrstly used in [54, 55] by embed-
ding a kinematic model layer into deep neutral networks and
estimating model parameters instead of joints. The geomet-
ric structure is better preserved. Yet, the kinematic model
parameterization is highly nonlinear and its optimization in
deep networks is hard. Also, the methods are limited for a
fully speciﬁed kinematic model (ﬁxed bone length, known
scale). They do not generalize to 2D pose estimation, where
a good 2D kinematic model does not exist.
2D Pose Estimation Before the deep learning era, many
methods use graphical models to represent the structures in
the joints. Pictorial structure model [13] is one of the earli-
est. There is a lot of extensions [23, 50, 36, 35, 51, 26, 9].
Pose estimation is formulated as inference problems on the
graph. A common drawback is that the inference is usually
complex, slow, and hard to integrate with deep networks.
Recently, the graphical models have been integrated into
deep networks in various ways. Tompson et al. [46] ﬁrstly
combine a convolutional network with a graphical model
for human pose estimation. Ouyang et al. [32] joints fea-
ture extraction, part deformation handling, occlusion han-
dling and classiﬁcation all into deep learning framework.
Chu et al. [10] introduce a geometrical transform kernels in
CNN framework that can pass informations between differ-
ent joint heat maps. Both features and their relationships
are jointly learned in a end-to-end learning system. Yang et
al. [49] combine deep CNNs with the expressive deformable
mixture of parts to regularize the output.
Another category of methods use a multi-stage architec-
ture [11, 6, 31, 5, 47, 18, 14]. The results of the previous
stage are used as inputs to enhance or regularize the learn-
ing of the next stage. Newell et al. [31] introduce an Stacked
Hourglass architecture that better capture the various spatial
relationships associated with the body. Chu et al. [11] fur-
ther extend [31] with a multi-context attention mechanism.
Bulat et al. [5] propose a detection-followed-by-regression
CNN cascade. Wei et al. [47] design a sequential archi-
tecture composed of convolutional networks that directly
operate on belief maps from previous stages. Gkioxari et
al. [14] predict joint heat maps sequentially and condition-
ally according to their difﬁculties. All such methods learn
the joint dependency from data, implicitly.
Different to all above 3D and 2D methods, our approach
explicitly exploits the joint connection structure in the pose.
It does not make further assumptions and does not involve
complex algorithm design. It only changes the pose repre-
sentation and enhances the loss function. It is simple, effec-
tive, and can be combined with existing techniques.
Leveraging in the wild 2D data for 3D pose estimation
3D pose capturing is difﬁcult. The largest 3D human pose
dataset Human3.6M [20] is still limited in that the subjects,
the environment, and the poses have limited complexity and
variations. Models trained on such data do not generalize
well to other domains, such as in the wild images.
In contrast, in the wild images and 2D pose annotation
are abundant. Many works leverage the 2D data for 3D pose
estimation. Most of them consist of two separate steps.
Some methods ﬁrstly generate the 2D pose results (joint
locations or heat maps) and then use them as input for re-
covering the 3D pose. The information in the 2D images is
discarded in the second step. Bogo et al. [4] ﬁrst use Deep-
Cut [38] to generate 2D joint location, then ﬁt with a 3D
body shape model. Moreno et al. [30] use CPM [47] to
detect 2D position of human joints, and then use these ob-
servations to infer 3D pose via distance matrix regression.
Zhou et al. [57] use Hourglass [31] to generate 2D joint
heat maps and then coupled with a geometric prior and Ja-
hangiri et al. [21] also use Hourglass to predict 2D joint heat
maps and then infer multiple 3D hypotheses from them. Wu
et al. [48] propose 3D interpreter network that sequentially
estimates 2D keypoint heat maps and 3D object structure.
Some methods ﬁrstly train the deep network model on
2D data and ﬁne-tune the model on 3D data. The informa-
tion in 2D data is partially retained by the pre-training, but
not fully exploited as the second ﬁne-tuning step cannot use
2D data. Pavlakos et al. [34] extends Hourglass [31] model
for 3D volumetric prediction. 2D heat maps are used as in-

termediate supervision. Tome et al. [44] extends CPM [47]
to 3D by adding a probabilistic 3D pose model to the CPM.
Some methods train both 2D and 3D pose networks si-
multaneously by sharing intermediate CNN features [28,
33]. Yet, they use separate networks for 2D and 3D tasks.
Unlike the above methods, our approach treats the 2D
and 3D data in the same way and combine them in a uni-
ﬁed training framework. The abundant information in the
2D data is fully exploited during training. As a result, our
method achieves strong performance on both 3D and 2D
benchmarks. As a by-product, it generates plausible and
convincing 3D pose results for in the wild images.
Some methods use synthetic datasets which are gener-
ated from deforming a human template model with known
ground truth [8, 40]. These methods are complementary to
the others as they focus on data augmentation.
3. Compositional Pose Regression
Given an image of a person, the pose estimation problem
is to obtain the 2D (or 3D) position of all the K joints, J =
{Jk|k = 1, ..., K}. Typically, the coordinate unit is pixel
for 2D and millimeter (mm) for 3D.
Without loss of generality, the joints are deﬁned with re-
spect to a constant origin point in the image coordinate sys-
tem. For convenience, let the origin be J0. Speciﬁcally,
for 2D pose estimation, it is the top-left point of the im-
age. For 3D pose estimation, it is the ground truth pelvis
joint [54, 33].
For regression learning, normalization is necessary to
compensate for the differences in magnitude of the vari-
ables. We use the standard normalization by subtraction of
mean and division of standard deviation. For a variable var,
it is normalized as
˜
var = N(var) = var −mean(vargt)
std(vargt)
.
(1)
The inverse function for unnormalization is
var = N −1( ˜
var) = ˜
var·std(vargt)+mean(vargt). (2)
Note that both mean(∗) and std(∗) are constants and
calculated from the ground truth training samples. The pre-
dicted output from the network is assumed already normal-
ized. Both functions N(∗) and N −1(∗) are parameter free
and embedded in the network. For notation simplicity, we
use ˜
var for N(var).
3.1. Direct Joint Regression: A Baseline
Most previous regression based methods [6, 54, 33, 42,
43] directly minimize the squared difference of the pre-
dicted and ground truth joints. In experiments, we found
that the absolute difference (L1 norm) performs better. In
our direct joint regression baseline, the joint loss is
L(J ) =
K
X
k=1
||˜Jk −˜Jgt
k ||1.
(3)
Note that both the prediction and ground truth are nor-
malized.
There is a clear drawback in loss Eq.(3). The joints are
independently estimated. The joint correlation, or the inter-
nal structure in the pose, is not well exploited. For example,
certain geometric constraints (e.g., bone length is ﬁxed) are
not satisﬁed.
Previous works only evaluate the joint location accuracy.
This is also limited because the internal structures in the
pose are not well evaluated.
3

## conclusion
We show that regression based approach is competitive
to the leading detection based approaches for 2D pose esti-
mation once pose structure is appropriately exploited. Our
approach is more potential for 3D pose estimation, where
more complex structure constraints are critical.
Acknowledgement
This research work was supported by The National Sci-
ence Foundation of China No.
61305091, and the Fun-
damental Research Funds for the Central Universities No.
2100219054.