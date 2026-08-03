# Deep Kinematics Analysis for Monocular 3D Human Pose Estimation

> 2020 · id: W3034581612 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Deep Kinematics Analysis for Monocular 3D Human Pose Estimation
Jingwei Xu⋆1,2, Zhenbo Yu⋆1,2, Bingbing Ni†1,2,3, Jiancheng Yang1,2, Xiaokang Yang1,2, Wenjun Zhang1,2
1Shanghai Jiao Tong University, Shanghai 200240, China
2MoE Key Lab of Artiﬁcial Intelligence, AI Institute, Shanghai Jiao Tong University
3Huawei Hisilicon
{xjwxjw,yuzhenbo,nibingbing,jekyll4168,xkyang,zhangwenjun}@sjtu.edu.cn, nibingbing@hisilicon.com
Abstract
For monocular 3D pose estimation conditioned on 2D
detection, noisy/unreliable input is a key obstacle in this
task. Simple structure constraints attempting to tackle this
problem, e.g., symmetry loss and joint angle limit, could
only provide marginal improvements and are commonly
treated as auxiliary losses in previous researches. It still
remains challenging to fully utilize human prior knowledge
in this task. In this paper, we propose to address above
issue in a systematic view. Firstly, we show that optimiz-
ing the kinematics structure of noisy 2D inputs is critical to
obtain accurate 3D estimations. Secondly, based on cor-
rected 2D joints, we further explicitly decompose articu-
lated motion with human topology, which leads to more
compact 3D static structure easier for estimation. Finally,
we propose a temporal module to reﬁne 3D trajectories,
which obtains more rational results. Above three steps are
seamlessly integrated into deep neural models, which form
a deep kinematics analysis pipeline concurrently consider-
ing the static/dynamic structure of 2D inputs and 3D out-
puts. Extensive experiments show that proposed framework
achieves state-of-the-art performance on two widely used
3D human action datasets. Meanwhile, targeted ablation
study shows that each former step is critical for the latter
one to obtain promising results.
1. Introduction
Pose estimation is a hot-spot topic in computer vision
researches [35, 51, 2, 5]. Particularly, 3D pose estimation
for monocular video has drawn tremendous attention in the
past decades [21, 18, 37, 33, 1, 6], which involves estimat-
ing keypoint trajectories of human subject in 3D space. This
research topic possesses several valuable downstream appli-
cations, e.g., action recognition [3, 23], human body recon-
⋆Equal contribution.
†Corresponding author: Bingbing Ni
Projection Based 
2D Correction
Noisy/Unreliable 
2D Inputs
3D Trajectory
Completion
Step 1
Step 2
Step 3
Reliable 
Estimation
Unreliable
Estimation
Decomposed 3D 
Estimation
𝒚
𝒙
𝒛
Deep
CNN
Figure 1: Overview of proposed framework. Our model
pursues 3D pose estimation with more reasonable structure
and more compact output space, which incorporates kine-
matics analysis into deep models. Step 1: Noisy/unreliable
2D Inputs (denoted as red dots) are corrected with perspec-
tive projection. Step 2: 3D poses are further estimated in
a decomposed manner within more compact space. Step 3:
Unreliably estimated 3D poses are excluded from previous
outputs (denoted by red cross), which are ﬁnally reﬁned as
a completion task.
struction [16, 13, 47] and robotics manipulation [34].
Recently, many works [11, 5] have used 2D pose detec-
tors to facilitate the 3D human pose estimation task. Sev-
eral previous researches [28, 25, 46, 38, 7] take detected
2D keypoints as input and predict corresponding 3D joint
locations from monocular video, which have promising re-
sults and require much less training resources compared to
other works fed with RGB images [19, 42, 10, 54]. Our
work belongs to this branch with explicit incorporation of
kinematics analysis, which would be discussed in detail at
latter paragraphs.
Despite considerable progress in 2D-keypoint condi-
899

Decomposed 3D
Pose Estimation
Completed
3D Trajectory
Refined
2D Pose
3D Trajectory
Completion
2D Pose
Inputs
Estimated
3D Pose
𝜙3D
𝑙𝑒𝑛
𝜙3D
𝑑𝑖𝑟
ℒ3𝐷
𝑑𝑐𝑚+ ℒ3𝐷
𝑓𝑖𝑛
ሚ𝑙3𝐷
ǁ𝑟3𝐷
ℒ3𝐷
𝑐𝑡𝑛
ǁ𝑠3𝐷
Ƹ𝑠3𝐷
2D Pose Correction
ℒ2𝐷
𝑡𝑒𝑚
෤𝑝2𝐷
ሚ𝑓2𝐷
ℒ2𝐷
𝑝𝑟𝑜𝑗
ǁ𝑐2𝐷
𝜙2D
𝜙3D
𝑐𝑡𝑛
Figure 2: Detailed architecture of proposed framework. Three hierarchical modules (from left to right) correspond to 2D
pose correction (φ2D), 3D pose estimation (φlen
3D, φdir
3D) and 3D trajectory reﬁnement (φctn
3D ) respectively. ˜p2D is corrected 2D
pose and ˜f, ˜c refer to regressed camera parameters. ˜l3D and ˜r3D are estimated length and direction from φlen
3D, φdir
3D. ˜S3D and
ˆS3D stand for 3D pose before/after trajectory completion respectively.
tioned 3D pose estimation, several critical challenges re-
main yet to be solved.
On one hand, 2D detections are
generally noisy and unreliable due to motion blur and self-
occlusion contained in video sequences.
Current meth-
ods [25, 46, 33] adopt simple structure constraints, e.g.,
symmetric bone length [33] and limited joint angle [16],
to facilitate 3D joint predictions, which are insufﬁcient to
bring signiﬁcant improvements for this task. On the other
hand, the majority of existing approaches directly formu-
late this task as a coordinate regression problem, which do
not fully take the inherent kinematics structure of human
subject into consideration and commonly leads to invalid
results.
Real-world human motion obeys kinematics laws in-
volving 2D/3D correspondence and static/dynamic struc-
ture: (1) For camera-based view, 3D and projected 2D joints
should follow the constraint of perspective projection. (2)
For static structure, the length between two adjacent 3D
joints (deﬁned by skeleton) should be constant throughout
the whole motion sequence. (3) For dynamic structure, es-
timated 3D trajectory formed by the same joint should be
smooth and continuous. We are thus motivated to integrate
all above laws forming a systematical analysis pipeline from
correspondence to structure, which facilitates pursuing 3D
poses within more rational solution space.
In this paper, we propose to systematically incorporate
kinematics analysis into deep models for effective utiliza-
tion of human prior knowledge. As illustrated in Fig. 1, we
ﬁrstly reﬁne 2D inputs to be more reliable rather than only
consider the estimation accuracy of 3D keypoints [33, 1]. A
novel optimization scheme is designed for 2D keypoints un-
der the constraint of perspective projection, which mainly
facilitates kinematics structure correction of noisy 2D in-
puts. To our best knowledge, it is the ﬁrst attempt that per-
spective projection is used for 2D joints reﬁnement rather
than 3D counterparts.
Experimental results demonstrate
that above 2D optimization scheme is critical for the follow-
ing 3D pose estimation. Secondly, starting from the static
structure of human subject, we decompose articulated mo-
tion based on rigid body assumption, which breaks uncon-
strained 3D trajectories down to a tree-structured combina-
tion of 2D sphere curves with much lower dimension. More
speciﬁcally, we split 3D coordinate regression problem into
two sub-tasks, i.e., length and direction estimation, which
are complementary to each other and reduce the learning
difﬁculty by a large margin. Finally, we notice that not all
parts are equally estimated. To pursue valid dynamic struc-
ture we exclude those joints with low reliability from above
predictions and the whole 3D trajectory is completed based
on more reliable parts. Above three steps are seamlessly in-
tegrated into deep neural models, which form a systematic
analysis pipeline, i.e., our model simultaneously considers
the kinematics structure of 2D inputs and 3D outputs.
We conduct detailed ablation study to demonstrate the
contribution of each component of proposed framework.
Further extensive experiments show that our model achieves
state-of-the-art performance on two widely used 3D human
motion datasets.
2. Related Work
In this section, we discuss the approaches that are based
on deep neural networks for 3D pose estimation.
Holistic 3D pose estimation. With the excellent fea-
ture extraction capacity of deep neural networks, many ap-
proaches [19, 31, 44, 42, 26, 32, 27, 10, 54, 6] utilize Deep
Convolutional Neural Networks to estimate 3D poses from
the images or other sources (e.g., point clouds [50, 22, 49])
directly. In this paper we concentrate on the image one. Li
et al. [19] ﬁrstly apply CNNs to jointly estimate 3D poses
and detect body parts via a multi-task framework. Tekin et
al. [42] use an overcomplete auto-encoder to learn a high-
900

Projection Based 2D Correction
Camera
3D Joints
Unreliable 
2D Inputs
Corrected 
2D Joints
Figure 3: Example of perspective projection used to reﬁne
2D inputs. This essentially incorporates 3D supervision sig-
nal for 2D pose training, where 2D/3D correspondence is
better preserved.
dimensional latent pose representation and account for joint
dependencies. Differently, [10] and [54] both utilize inter-
mediate 3D representations and 2D counterparts to regress
3D poses. However, training deep models directly from im-
ages requires expensive computation resources and careful
hyper-parameter tuning. Differently, our model starts with
detected 2D joints as inputs whose pipeline is largely sim-
pliﬁed but with comparable performance.
Two-step pose estimation. To avoid collecting 2D-3D
paired data, a variety of works [55, 25, 43, 4, 9, 53, 52, 28,
38, 7] decouple the task of 3D pose estimation into two in-
dependent stages: (1) ﬁrstly predicting 2D joint location in
image space using off-the-shelf 2D pose estimation meth-
ods; (2) and then learning a mapping to lift them to 3D
space.
Moreno et al. [28] learn a mapping from 2D to
3D distance matrices. A simple yet effective method [25]
can directly predict 3D joint locations via deep CNNs. Re-
garding the prior knowledge of human structure, Wang et
al. [46] propose a progressive approach that explicitly ac-
counts for the distinct DOFs among the body parts. Sharma
et al. [38] synthesize diverse anatomically plausible 3D-
pose samples conditioned on the estimated 2D poses via a
deep conditional variational auto-encoder based model. The
generic combination [7] of Graph Convolutional Network
(GCN) and Connected Network (FCN) can also improve
the representation capability. These approaches mainly fo-
cus on the second stage and our method also belongs to this
branch. However, few researches have paid attention on the
inherent validity of 2D/3D poses in a systemic and compre-
hensive view, where kinematics structure and view corre-
spondence are generally ignored.
Video pose estimation.
Since most previous works
operate in a single-frame setting, recently more atten-
tion [14, 21, 18, 8, 37, 33, 1, 6, 20, 36, 48] has been
put on temporal information from monocular video clips.
LSTMs [21] have been applied to reﬁne 3D poses predicted
Table 1: Inﬂuence of 2D accuracy for 3D pose estimation.
Both are evaluated by mean squared error. M = 1 is no 2D
reﬁnement is applied.
Window Length
2D Joints
3D Joints
Train/Test
Train/Test
M=1
0.058 / 0.078
0.027 / 0.053
M=5
0.042 / 0.071
0.024 / 0.052
M=9
0.033 / 0.067
0.023 / 0.052
from single images. There has also been work on RNN
approach [18] which considers prior knowledge with body
part based structural connectivity. Rayat et al. [37] utilize
the temporal smoothness constraint across a sequence of 2D
joint locations to estimate a sequence of 3D poses. Pavllo et
al. [33] transform a sequence of 2D poses through tempo-
ral convolutions and make computational complexity inde-
pendent of key-point spatial resolution. Multi-scale features
for the graph-based representations [1] are critic to pose es-
timation by a local-to-global network architecture. Unlike
existing temporal based methods, our model explicitly in-
corporate a kinematics analysis pipeline for 3D pose esti-
mation.
3. Method
In this section, we present detailed description of pro-
posed method.
The overall framework is illustrated in
Fig. 2. In our method and experiments, we focus on pose
estimation over a short video clip (T ≤9).
3.1. Projection based 2D Pose Correction
2D Temporal Reﬁnement. Given a monocular video
clip with length of T time stamps, we ﬁrst apply pretrained
2D pose detector (e.g., CPN [5]) to obtain 2D joints. 2D
detections on single frame are generally noisy and unreli-
able due to motion blur and occlusion [11]. We ﬁrst utilize
a temporal CNN model (denoted as φ2D as shown in Fig. 2)
to reﬁne 2D initial inputs (denoted as ˜P = {˜pt}T
t=1, ˜K =
{˜kt}T
t=1, ˜pt ∈RJ×2, ˜kt ∈RJ.) Here J refers to the num-
ber of joints for single human. Speciﬁcally, ˜pt = [˜at, ˜bt]
where ˜at and ˜bt represent 2D coordinates and ˜kt is corre-
sponding conﬁdence score. We adopt MSE loss weighted
by conﬁdence score ˜K for trained as follows:
LT em
2D
=
T
X
t=1
˜kt
q
|at −˜at|2
2 + |bt −˜bt|2
2,
(1)
where at, bt refers to the ground truth 2D joints. Intuitively,
this procedure utilizes temporal smoothness to reﬁne de-
tected 2D joints.
901

Limitation: Train/Test Imbalance.
However, above
optimization does not directly facilitate performance boost-
ing for ﬁnal 3D pose estimation. We conduct veriﬁcation
experiments (on Human3.6M dataset [15]) to support our
statements. Speciﬁcally, we adopt single-frame 3D pose es-
timation model [33] with reﬁned 2D joints as inputs. As
shown in Tab. 1, estimation accuracy of 2D and 3D joints on
both train and test set are reported. We can observe that with
increase of window length, the training accuracy boosts for
both 2D and 3D joints. But the improvement on 3D test set
is marginal (last column). We attribute this for imbalanced
inputs on Train/Test set. The large accuracy gap between
train and test inputs (ﬁrst two columns) is unknown to the
following 3D pose estimation model, which leads to sub-
optimal results.
Solution: Projection Constraint. To this end, we pro-
pose to reﬁne the 2D inputs in a different point of view,
i.e., 2D/3D correspondence is critical for rational pose es-
timation (as shown in Fig. 3).
We denote 3D joints as
S = {st}T
t=1, st = [xt, yt, zt] ∈RJ×3. For each time
stamp t, projected 2D joints p and 3D joints s should obey
perspective projection as follows:
a = x
z fx + cx, b = y
z fy + cy,
(2)
where f = [fx, fy] and c = [cx, cy] are focal length and
point respectively. We omit subscript t for simplicity. Our
main idea is to recover f and c from reﬁned 2D inputs and
ground truth 3D joints during training. Intuitively, well es-
timated ˜f and ˜c indicate the projection correspondence is
generally preserved. The remaining problem is how to ob-
tain ˜f and ˜c efﬁciently.
Linear squares regression is a simple yet effective ap-
proach for above issue. Take the projection x-axis coor-
dinates for example. As indicated by Eqn. 2, paired data
points (a, x
z ) = {(aj, xj
zj )}J
j=1 involving single human sub-
ject follows a straight line whose intercept and slope corre-
spond to fx and cx respectively. For simplicity we omit the
dominator, i.e., xj
zj : xj. Given reﬁned 2D detections ˜p and
ground truth 3D joints s, we estimate fx and cx as follows:
˜fx =
¯a(PJ
j=1 xj)2 −¯x PJ
j=1 xj˜aj
PJ
j=1(xj)2 −J ¯x2
, ˜cx = ¯a −˜fx¯x,
(3)
where ¯a = 1
J
PJ
j=1 ˜aj, ¯x = 1
J
PJ
j=1 xj. We obtain ground
truth fx and cx the same as Eqn. 3 with ground truth p and
s as inputs. We adopt L1 loss for training as follows:
Lproj
2D
= |fx −˜fx| + |cx −˜cx|.
(4)
Furthermore, estimated ˜fx and ˜cx at each time stamp should
be constant for the whole monocular video clip, which is
also utilized for training. More speciﬁcally, we randomly
0
3
4
5
2
6
7
8
1
16
9
10
11
13
14
12
15
10
11
11
Parent Joint: #10
Child Joint: #11
𝑥
𝑦
𝑧
Figure 4: Illustration of articulated motion for a pair of
parent-children joints. The motion trajectory of children
joint relative to its parent degenerates to spherical curve.
select two time stamps, where estimated ˜fx and ˜cx are pe-
nalized with L1 difference. As shown in Fig. 2, φ2D adopts
an encoder-decoder architecture, the temporal length of out-
puts are equal to that of inputs. For detailed architecture
please refer to supplementary material.
Note that datasets used in our paper (Human3.6M [15]
and HumanEva [39] Dataset) are recorded by cameras with
ﬁxed and almost identical (f,c), which indicates a global
projection relation existing between all 2D and 3D poses.
The reﬁnement model is trained towards a single projec-
tion relation.
If video sequences are recorded by cam-
eras with varied (f,c), we could feed the (f,c) as inputs
to guarantee the generalization ability of proposed model.
As intrinsic parameters, (f,c) are generally low cost to ob-
tain [33]. This part is leaved as future work worth studying.
3.2. Decomposed 3D pose estimation
Based on reﬁned 2D joints, our second model (denoted
as φdcm
3D ) predicts corresponding 3D keypoints.
As dis-
cussed in Sec. 1, 3D articulated motion involved in monoc-
ular video clip is structure constrained. Fig. 4 depicts this
kinematics law more explicitly. The motion trajectory of
children joint relative to its parent (deﬁned by human skele-
ton) forms a spherical curve.
Explicit Pose Decomposition. Motivated by this, we
decompose original coordinate regression problem into two
complementary sub-tasks, i.e., length and direction regres-
sion. Speciﬁcally, for a 3D sequence S ∈RT ×J×3, we ﬁrst
obtain the relative coordinate according to predeﬁned skele-
ton topology, i.e., ∆Sjc = Sjc −Sjp, where jc refers some
child joint and jp indicates its parent joint (shown in Fig. 4).
Suppose the skeleton deﬁned by joint pair {jc, jp} is of
length ljc, ∆Sjc could be rewritten as ∆Sjc = {ljcrjc
t }T
t=1,
where rjc
t is the unit vector representing direction of skele-
ton between (jc, jp) at time stamp t. ljc is kept constant
across the whole video clip. Therefore we predict ljc and
rjc
t
as intermediate results, which are composed together
according to human skeleton for ﬁnal estimation (Eqn. 5).
Global-local Combined 2D inputs. We extend inputs
902

0
0.01
0.02
0.03
0.04
0.05
0.06
0.07
0.08
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
Joints Index
MPJPE/103
Mean MPJPE
Figure 5: Per-joint estimation error in terms of MPJPE.
Notably, the estimation error of four limbs (indexed by
0/1/4/5/6/7/10/11 with red bar) is signiﬁcantly higher than
other joints. Dashed black line indicates mean MPJPE for
all joints.
˜P with similar manner as S, i.e., [˜P, ∆˜P, ||∆˜P||2
2] ∈
RT ×J×5.
∆˜P ∈RT ×J×2 is obtained the same as ∆S
and ||∆˜P||2
2 ∈RT ×J×1 is calculated pixel length for each
skeleton. Above operation combines both global and local
information as inputs, which provides more structured in-
formation for 3D pose estimation.
Length & Direction Estimation.
We adopt a two-
stream architecture for length and direction estimation,
where two sub-modules are denoted as φlen
3D and φdir
3D re-
spectively. For length estimation, we ﬁrst extract pose fea-
ture for each time stamp, and then conduct feature aggre-
gation at middle level representation, which are ﬁnally used
to predict skeleton length ˜l = {˜lj}J
j=1. In contrast, skele-
ton direction is estimated independently at each time stamp
and the unit vector ˜rt = {˜rj
t}J
j=1 is obtained through a L2
normalization layer. For detailed architecture of φlen
3D and
φdir
3D, please refer to supplementary material. Up to now we
obtain ∆˜S = ˜l ˜R, where ˜R = {˜rt}T
t=1. Final estimation
˜S is obtained through iterative summarization over ∆˜S ac-
cording to human topology as follows:
˜Sjc = ∆˜Sjc + ˜Sjp.
(5)
Correspondingly, we adopt intermediate loss functions to
facilitate the training procedure. For length estimation, we
use L1 loss (denoted as Llen
3D = |˜l −l|) to obtain more ac-
curate results. For direction estimation, a cosine similarity
loss (denoted as Ldir
3D = ⟨˜r, r⟩−1) is applied on ˜R, which
penalizes angle difference rather than distance. For ﬁnal
composed prediction, we use L2 loss (denoted as Lfin
3D ) for
training. All three loss functions are shown as follows. For
detailed architecture of φdir
3D and φlen
3D please refer to supple-
mentary material.
Ldcm
3D = Llen
3D + Ldir
3D, Lfin
3D = ||˜S −S||2
2.
(6)
Discussion on Pose Decomposition. The most related
work for this part is Sun et al. [41], which conducts pose
estimation in an compositional way. However, our work
differentiates from Sun et al. [41] in following three as-
pects: (1) Our work concentrate on monocular pose esti-
mation rather than single image. (2) Instead of direct co-
ordinate regression, we explicitly decomposes output space
with rigid body structure, whose dimension is reduced from
3 × T × J to 2 × T × J + J, where the ﬁrst part is about
direction estimation while the second part corresponds to
length estimation. More compact output space facilitates
3D pose estimation model to obtain more rational results
by a large margin. (3) We design targeted losses Ldir
3D and
Llen
3D based on above length/direction decomposition, which
reduces the learning difﬁculty by a large margin.
3.3. Pose Reﬁnement as Trajectory Completion
Based on pose decomposition we estimate the skeleton
direction at each time stamp independently, which needs
further reﬁnement. In this section, we consider above prob-
lem as trajectory completion task, where reﬁnement is ap-
plied on unreliably estimated joints.
Which joints should be reﬁned? Not all joints are esti-
mated equally. As shown in Fig 5, we can observe that the
estimation error of four limbs (i.e., joints 0/1/4/5/6/7/10/11
also as shown in Fig. 4) is signiﬁcantly higher than oth-
ers. Therefore, we focus on four-limb regularization. Mean-
while, the conﬁdence score ˜K (mentioned in Sec. 3.1) pro-
duced by 2D detectors is an informative indicator for unreli-
able estimation of joints location. To this end, we optimize
four-limb joints assigned with low 2D conﬁdence score.
Trajectory Completion with Reliable Estimation.
Given estimated 3D joints ˜S ∈RT ×J×3, we ﬁrst apply a
dropout layer [40] directly on the joints associated with four
limbs. The dropout rate is 1−˜K rather than a constant value,
i.e., unreliable joints are excluded from ˜S ∈RT ×J×3 which
are further completed with reliable ones.
The comple-
tion model is denoted as φctn
3D consisting of a bi-directional
LSTM [12] network (shown in Fig. 2). We denote com-
pleted output as ˆS, which trained as follows:
Lctn
3D = ||ˆS−S||2
2 +||H(ˆS)−H(S)||2
2 +||F(ˆS)−F(S)||2
2,
(7)
where H and F refer to ﬁrst and second order difference
over temporal axis respectively. Intuitively, high-order con-
tinuity facilitates better modelling of dynamic structure for
human subject.
3.4. Implementation Details
Our model is end-to-end trainable and the overall loss
function is Ltem
2D + 0.1Lproj
2D
+ 0.1Lpel
2D + Ldcm
3D + Lfin
3D +
0.1Lctn
3D . We adopt PyTorch [29] to implement our proposed
framework. During training phase, learning rate, learning
decay and weight decay are set to 1e−3, 0.93, 1e−4 respec-
tively. Dropout rate is set to 0.25 except the one mentioned
903

Protocol 1: MPJPE
Dir. Disc.
Eat
Greet Phone Photo Pose Purch.
Sit
SitD. Smoke Wait WalkD. Walk WalkT. Avg
Martinez et al. [25] ICCV’17 (T=1)
51.8
56.2 58.1
59.0
69.5
78.4
55.2
58.1
74.0 94.6
62.3
59.1
65.1
49.5
52.4
62.9
Luvizon et al. [24] CVPR’18 (T=1)
49.2
51.6 47.6
50.5
51.8
60.3
48.5
51.7
61.5 70.9
53.7
48.9
57.9
44.4
48.9
53.2
Hossain & Little [37] ECCV’18 (T=5) 48.4
50.7 57.2
55.2
63.1
72.6
53.0
51.7
66.1 80.9
59.0
57.3
62.4
46.6
49.6
58.3
Lee et al. [18] ECCV’18 (T=3)
40.2
49.2 47.8
52.6
50.1
75.0
50.2
43.0
55.8 73.9
54.1
55.6
58.2
43.3
43.3
52.8
Pavllo et al. [33] CVPR’19 (T=1)
47.1
50.6 49.0
51.8
53.6
61.4
49.4
47.4
59.3 67.4
52.4
49.5
55.3
39.5
42.7
51.8
Pavllo et al. [33] CVPR’19 (T=9)
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
49.8
Cai et al. [1] ICCV’19 (T=1)
46.5
48.8 47.6
50.9
52.9
61.3
48.3
45.8
59.2 64.4
51.2
48.4
53.5
39.2
41.2
50.6
Cai et al. [1] ICCV’19 (T=7)
44.6
47.4 45.6
48.8
50.8
59.0
47.2
43.9
57.9 61.9
49.7
46.6
51.3
37.1
39.4
48.8
Ours, 1-frame
40.6
47.1 45.7
46.6
50.7
63.1
45.0
47.7
56.3 63.9
49.4
46.5
51.9
38.1
42.3
49.2
Ours, 7-frames
38.2
44.4 42.8
43.7
47.6
60.3
42.0
45.4
53.2 60.8
46.4
43.5
48.5
34.6
38.6
46.3
Ours, 9-frames
37.4
43.5 42.7
42.7
46.6
59.7
41.3
45.1
52.7 60.2
45.8
43.1
47.7
33.7
37.1
45.6
Protocol 2: PA-MPJPE
Dir. Disc.
Eat
Greet Phone Photo Pose Purch.
Sit
SitD. Smoke Wait WalkD. Walk WalkT. Avg
Sun et al. [41] ICCV’17 (T=1)
42.1
44.3 45.0
45.4
51.5
53.0
43.2
41.3
59.3 73.3
51.0
44.0
48.0
38.3
44.8
48.3
Fang et al. [9] AAAI’18 (T=1)
38.2
41.7 43.7
44.9
48.5
55.3
40.2
38.2
54.5 64.4
47.2
44.3
47.3
36.7
41.7
45.7
Pavlakos et al. [30] CVPR’18 (T=1)
34.7
39.8 41.8
38.6
42.5
47.5
38.0
36.6
50.7 56.8
42.6
39.6
43.9
32.1
36.5
41.8
Hossain & Little [37] ECCV’18 (T=5) 35.7
39.3 44.6
43.0
47.2
54.0
38.3
37.5
51.6 61.3
46.5
41.4
47.3
34.2
39.4
44.1
Pavllo et al. [33] CVPR’19 (T=1)
36.0
38.7 38.0
41.7
40.1
45.9
37.1
35.4
46.8 53.4
41.4
36.9
43.1
30.3
34.8
40.0
Cai et al. [1] ICCV’19 (T=1)
36.8
38.7 38.2
41.7
40.7
46.8
37.9
35.6
47.6 51.7
41.3
36.8
42.7
31.0
34.7
40.2
Cai et al. [1] ICCV’19 (T=7)
35.7
37.8 36.9
40.7
39.6
45.2
37.4
34.5
46.9 50.1
40.5
36.1
41.0
29.6
33.2
39.0
Ours, 1-frame
33.6
37.4 37.0
37.6
39.2
46.4
34.3
35.4
45.1 52.1
40.1
35.5
42.1
29.8
35.3
38.9
Ours, 7-frames
31.7
35.3 35.0
35.3
36.9
44.2
32.0
33.8
42.5 49.3
37.6
33.4
39.6
27.6
32.5
36.7
Ours, 9-frames
31.0
34.8 34.7
34.4
36.2
43.9
31.6
33.5
42.3 49.0
37.1
33.0
39.1
26.9
31.9
36.2
Table 2: Quantitative comparisons of Mean Per Joint Position Error (MPJPE) in millimeter between the estimated pose and
the ground-truth on Human3.6M under P1 and P2, where T denotes the number of input frames used in each method. Lower
is better and best is bold highlighted.
Protocol 2
Walk
Jog
Box
PA-MPJPE
S1
S2
S3
S1
S2
S3
S1
S2
S3
Pavlakos et al. [31] 22.3 19.5 29.7 28.9 21.9 23.8
–
–
–
Pavlakos et al. [30] 18.8 12.7 29.2 23.5 15.4 14.5
–
–
–
Lee et al. [18]
18.6 19.9 30.5 25.7 16.8 17.7 42.8 48.1 53.4
Pavllo et al. [33]
13.9 10.2 46.6 20.9 13.1 13.8 23.8 33.7 32.0
Ours,9-frames
13.2 10.2 29.9 12.6 12.3 13.0 13.2 18.1 20.4
Table 3: Prediction accuracy on HumanEva Dataset [39]
in terms of Protocol # 2 evaluation. Note that we train
one model with all three actions (i.e., Walk, Jog and Box)
models. Lower is better and best is bold highlighted.
in Sec. 3.3. We adopt the same strategy for BN momen-
tum decay as in [33]. Adam Optimizer [17] is used for all
modules. The whole model is trained with 200 epoches.
4. Experiments
4.1. Datasets & Evaluation Metrics
Human3.6M Datasets [15].
In our work, we follow
the experimental setup in previous researches [10, 45, 33].
More speciﬁcally, we use Subject 1 / 5 / 6 / 7 / 8 for train-
ing and Subject 9 / 11 for testing. Without access to action
labels and camera parameters, all video sequences are used
for training one single model.
MPJPE(P1) PA-MPJPE(P2)
Baseline
51.8
40.0
2D Reﬁne
49.9
39.4
2D Reﬁne + 3D Decompose
47.1
37.5
2D Reﬁne + 3D Decompose + 3D Completion
45.6
36.2
Table 4: Ablation study on the contribution of proposed
three modules in terms of both P1 and P2. Note that Base-
line refers to single-frame results of Pavllo et al. [33].
HumanEva-I Datasets [39]. Compared to Human3.6M
dataset [15], HumanEva-I [39] is more light-weight con-
taining three erect actions: Walk, Jog, Box. We follow the
same data preprocessing strategy used in [33] for train/test
split. We report the estimation accuracy with T = 9.
Evaluation Metrics.
Following the majority of pre-
vious works [24, 37, 18, 1, 33] we evaluate our model
in terms of mean per joint position error (MPJPE, P1 for
short) commonly denoted as protocol #1.
Several re-
searches [41, 9, 30, 37, 33, 1] estimate 3D pose after align-
ment involving rotation and translation (PA-MPJPE, P2 for
short), which is termed as protocol #2. Both protocols are
utilized in our work.
904

Seq 1
Seq 2
Seq 3
Figure 6: Visualization of predicted 3D poses on monocular video clip. Each row corresponds to one action sequence.
4.2. Quantitative Evaluation
Results on Human3.6M Dataset [15].
As shown in
Tab. 2, we report pose estimation results in terms of Pro-
tocol #1 and #2. Note that previous works evaluate this
task with different temporal length. For fairness we com-
pare with them under the same inputs. When T = 1 tra-
jectory completion model φctn
3D is nonfunctional. Beneﬁted
from 2D correction and 3D decomposition, our model out-
performs prior art (Cai et al. [1]) under both P1 and P2 eval-
uation when T = 1. Boosted by explicit kinematics anal-
ysis, the estimation accuracy outperforms Pavllo et al. [33]
by 8.4%(4.2mm) when T = 9. A notable performance im-
provement over Cai et al. [1] is also presented for T = 7,
i.e., 46.3mm v.s. 48.8mm. Regarding evaluation with action
classes, our ﬁnal model (T = 9) achieves best performance
on the majority of them. Especially on several relatively
harder actions, e.g, sitting and sitting down with severe oc-
clusion, our model is robust enough to obtain better results
compared to Pavllo et al. [33] which merely considers tem-
poral smoothness rather than estimation reliability. This is
further validated in our own model. From T = 1 to T = 9,
estimation accuracy is enhanced constantly under both P1
and P2 evaluation, which is mainly facilitated by decom-
posed 3D pose estimation and 3D trajectory completion.
Results on HumanEva Dataset [39].
As shown in
Tab. 3, we report pose estimation results in terms of Proto-
col #2. Compared to Human3.6M Dataset [15], HumanEva
Dataset [39] is relatively easier to learn, where estimation
accuracy is near saturated. Still, performance gain is ob-
served for all three actions over Pavllo et al. [33]. By ex-
plicitly incorporating kinematics analysis into deep models,
reasonable spatial-temporal structure is well preserved and
output space is more compact, which ﬁnally lead to higher
estimation accuracy.
4.3. Qualitative Evaluation
We further present direct visualization results of monoc-
ular 3D pose estimation. As illustrated in Fig. 6, three se-
quences with diverse actions are presented. Meanwhile, for
each time stamp we demonstrate both reﬁned 2D pose and
1-Frame 3-Frames 5-Frames 7-Frames 9-Frames
Pavllo et al. [33]
51.80
–
–
–
49.80
Cai et al. [1]
50.62
49.08
48.86
48.78
–
Ours
49.21
47.87
46.83
46.26
45.61
Table 5: Prediction accuracy with different input length and
in terms of MPJPE. We compare our model with Pavllo et
al. [33] and Cai et al. [1].
corresponding 3D prediction. Our model makes it to pro-
duce visually natural estimation which is mainly beneﬁted
from explicit kinematics constraint. For example, sitting
down sequence (second row in Fig. 6) is well estimated for
both 2D and 3D joints, where the structure of four limbs
is properly handled by our model (requiring awareness of
human topology). More visual results are provided in sup-
plementary material for reference.
4.4. Ablation Study
Analysis on all modules. Recall that our model con-
sists of three modules: φ2D, φdcm
3D
and φctn
3D . To validate
the contribution of each module, we present corresponding
ablation study on Human3.6M Dataset [15]. As shown in
Tab. 4, both accuracy on P1 and P2 are reported. Baseline
refers to single frame estimation model of Pavllo et al. [33].
We can notice that each module offers positive contribu-
tion under evaluation of P1 and P2. Notably, the most sig-
niﬁcant improvement comes from 3D decomposition mod-
ule φdcm
3D , which beneﬁts from φ2D with more reasonable
2D/3D correspondence and further leads to more compact
output space.
Analysis on temporal length. As shown in Tab. 5, we
report the estimation accuracy w.r.t. different input lengths.
We can notice that with the increase of temporal hori-
zon, our model constantly performs better than those with
shorter inputs. For the identical input length, our model also
produces more accurate results than prior arts, i.e., Pavllo et
al. [33] and Cai et al. [1].
Analysis on projection based 2D correction. One re-
maining but critical problem is: how does projection based
905

2D Joints
2D Joints
Intercept
Intercept
3D Joints (𝑠𝑥/𝑠𝑧)
3D Joints (𝑠𝑥/𝑠𝑧)
Slope
Slope
A B
D
C
Estimated Focal
GT Focal
Estimated Focal
GT Focal
Slp:2.29 / Inter: -0.11 (GT)
Slp:2.02 / Inter: -.036 (Input)
Slp:2.29 / Inter: -0.11 (GT)
Slp:2.17 / Inter: -0.16 (Refine)
Figure 7: Analysis on projection based 2D correction. A/C
correspond to fx, cx estimation without/with constraint of
perspective projection. B/D correspond to two typical ex-
amples of projection correspondence between 2D and 3D
joints without/with constraint of perspective projection.
0.46
0.45
0.44
0
10
20
30
40
50
60
70
80
Length/m
Direction
0.40
0.30
0
10
20
30
40
50
60
70
80
Time Stamp
Single-frame
Ground Truth
Pavllo et.al. [31]
Ours
Figure 8: Analysis on decomposed 3D estimation.
The
length (upper part) and direction (lower part) estimation of
left shank (skeleton between joint 4 and joint 5) are pre-
sented.
2D correction facilitate 3D pose estimation?
As shown
in Fig 7(A and C), we plot estimated/ground thruth focal
length fx and point cx on test set for evaluation. Fig 7A cor-
responds to training without Lproj
2D
while Fig 7C shows re-
sults trained with it. Orange dot corresponds to ground truth
and blue cross are estimated results. Boosted by Lproj
2D , the
estimated projection structure in camera view is more accu-
rate. Moreover, we present two typical examples of corre-
spondence between 2D and 3D joints in Fig. 7 (B and D).
Following Eqn. 2, X-axis represents x/z while Y-axis is a.
Notation is used the same as Fig. 7 (A and C). All joints be-
1-Frame 3-Frames 5-Frames 7-Frames 9-Frames
Full model w/o φctn
3D
7.80
5.11
4.03
3.57
3.29
Full model
7.80
3.70
2.45
2.13
2.01
Table 6: Prediction accuracy (Human3.6M Dataset [15])
with different input length in terms of MPJVE [33]. The
ﬁrst row corresponds to training without trajectory model
φctn
3D , while the second row refers to full model. Note that
φctn
3D is nonfunctional when T = 1, whose estimation accu-
racy is identical to the ﬁrst row.
long to one single frame. Orange dots referring ground truth
lie in a straight line. Similarly, Fig. 7D depicts test results
trained with Lproj
2D
while Fig. 7B not. It clearly shows that
2D correction based on projection correspondence is more
reasonable and accurate.
Analysis on decomposed 3D estimation. To analyze
the contribution of φdcm
3D , we present both length and direc-
tion results shown in Fig. 8. Red line corresponds to ground
truth, blue line is single-frame estimation, green line refers
to the results of Pavllo et al. [33] and orange line is ours.
For length estimation, our model is robust to the variation
of 2D poses. One the contrary, both single-frame estimation
and the model of Pavllo et al. [33] fails to produce valid
length estimation which should keep constant throughout
the whole video clip. Based on accurate length estimation,
our model is capable of ﬁnding joint angle within more
compact space. As shown in Fig. 8B where the direction
is calculated as angle between left shank and Y-axis, our
model performs better than all other baselines by a large
margin, i.e, estimated direction is closer to ground truth (red
line) with smaller variation.
Analysis on 3D trajectory completion.
Following
Pavllo et al. [33], we evaluate trajectory estimation accu-
racy in terms of MPJVE, i.e., velocity error, to further vali-
date the contribution of φctn
3D . As shown in Tab. 6, facilitated
by φctn
3D our model achieves lower velocity error with all ex-
perimented temporal lengths (from T = 3 to T = 9).
5. Conclusion
In this paper, we propose a deep kinematics analysis
framework for monocular 3D pose estimation. By explic-
itly incorporating kinematics regularization into deep mod-
els, we achieves more reliable estimation with noisy 2D
joints as inputs. Extensive experiments show that our model
achieves state-of-the-art performance on two widely used
3D human action datasets.
Acknowledgment This work was supported by National Sci-
ence Foundation of China (61976137, 61527804, U1611461,
U19B2035), STCSM(18DZ1112300). This work was also sup-
ported by National Key Research and Development Program of
China (2016YFB1001003). The authors would like to give a per-
sonal thanks to the Student Innovation Center of SJTU for provid-
ing GPUs.
906

References
[1] Yujun Cai, Liuhao Ge, Jun Liu, Jianfei Cai, Tat-Jen Cham,
Junsong Yuan, and Nadia Magnenat Thalmann.
Exploit-
ing spatial-temporal relationships for 3d pose estimation via
graph convolutional networks. In ICCV, October 2019.
[2] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh.
Realtime multi-person 2d pose estimation using part afﬁnity
ﬁelds. In CVPR, pages 1302–1310, 2017.
[3] Jo˜ao Carreira and Andrew Zisserman.
Quo vadis, action
recognition? A new model and the kinetics dataset. In CVPR,
pages 4724–4733, 2017.
[4] Ching-Hang Chen and Deva Ramanan. 3d human pose es-
timation= 2d pose estimation+ matching. In CVPR, pages
7035–7043, 2017.
[5] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang
Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network
for multi-person pose estimation.
In CVPR, pages 7103–
7112, 2018.
[6] Yu Cheng, Bo Yang, Bo Wang, Wending Yan, and Robby T.
Tan. Occlusion-aware networks for 3d human pose estima-
tion in video. In ICCV, October 2019.
[7] Hai Ci, Chunyu Wang, Xiaoxuan Ma, and Yizhou Wang. Op-
timizing Network Structure for 3D Human Pose Estimation.
In ICCV, October 2019.
[8] Huseyin Coskun, Felix Achilles, Robert DiPietro, Nassir
Navab, and Federico Tombari.
Long short-term memory
kalman ﬁlters: Recurrent neural estimators for pose regular-
ization. In ICCV, pages 5524–5532, 2017.
[9] Hao-Shu Fang, Yuanlu Xu, Wenguan Wang, Xiaobai Liu,
and Song-Chun Zhu. Learning pose grammar to encode hu-
man body conﬁguration for 3d pose estimation. In AAAI,
2018.
[10] Ikhsanul Habibie, Weipeng Xu, Dushyant Mehta, Gerard
Pons-Moll, and Christian Theobalt. In the wild human pose
estimation using explicit 2d features and intermediate 3d rep-
resentations. In CVPR, pages 10905–10914, 2019.
[11] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross B.
Girshick. Mask R-CNN. In YCCV, pages 2980–2988, 2017.
[12] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term
memory. NC, 9(8):1735–1780, 1997.
[13] Zhongyue Huang, Jingwei Xu, and Bingbing Ni.
Human
motion generation via cross-space constrained sampling. In
IJCAI, pages 757–763, 2018.
[14] Sergey Ioffe and Christian Szegedy. Batch normalization:
Accelerating deep network training by reducing internal co-
variate shift. In ICML, pages 448–456, 2015.
[15] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian
Sminchisescu. Human3.6m: Large scale datasets and predic-
tive methods for 3d human sensing in natural environments.
TPAMI, 36(7):1325–1339, 2014.
[16] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and
Jitendra Malik. End-to-end recovery of human shape and
pose. In CVPR, pages 7122–7131, 2018.
[17] Diederik P. Kingma and Jimmy Ba. Adam: A method for
stochastic optimization. In ICLR, 2015.
[18] Kyoungoh Lee, Inwoong Lee, and Sanghoon Lee. Propagat-
ing lstm: 3d pose estimation based on joint interdependency.
In ECCV, pages 119–135, 2018.
[19] Sijin Li and Antoni B Chan. 3d human pose estimation from
monocular images with deep convolutional neural network.
In ACCV, pages 332–347, 2014.
[20] Jiahao Lin and Gim Hee Lee. Trajectory space factoriza-
tion for deep video-based 3d human pose estimation. arXiv
preprint arXiv:1908.08289, 2019.
[21] Mude Lin, Liang Lin, Xiaodan Liang, Keze Wang, and Hui
Cheng. Recurrent 3d pose sequence machines. In CVPR,
pages 810–819, 2017.
[22] Jinxian Liu, Bingbing Ni, Caiyuan Li, Jiancheng Yang, and
Qi Tian.
Dynamic points agglomeration for hierarchical
point sets learning. In Proceedings of the IEEE International
Conference on Computer Vision, pages 7546–7555, 2019.
[23] Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan, and Alex C.
Kot. Global context-aware attention LSTM networks for 3d
action recognition. In CVPR, pages 3671–3680, 2017.
[24] Diogo C Luvizon, David Picard, and Hedi Tabia. 2d/3d pose
estimation and action recognition using multitask deep learn-
ing. In CVPR, pages 5137–5146, 2018.
[25] Julieta Martinez, Rayat Hossain, Javier Rome