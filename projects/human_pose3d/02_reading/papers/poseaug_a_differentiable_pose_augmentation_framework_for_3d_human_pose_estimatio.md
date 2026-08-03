# PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation

> 2021 · id: W3169891778 · arXiv: 2105.02465 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Existing 3D human pose estimators suffer poor gener-
alization performance to new datasets, largely due to the
limited diversity of 2D-3D pose pairs in the training data.
To address this problem, we present PoseAug, a new auto-
augmentation framework that learns to augment the avail-
able training poses towards a greater diversity and thus im-
prove generalization of the trained 2D-to-3D pose estima-
tor. Speciﬁcally, PoseAug introduces a novel pose augmen-
tor that learns to adjust various geometry factors (e.g., pos-
ture, body size, view point and position) of a pose through
differentiable operations. With such differentiable capacity,
the augmentor can be jointly optimized with the 3D pose
estimator and take the estimation error as feedback to gen-
erate more diverse and harder poses in an online manner.
Moreover, PoseAug introduces a novel part-aware Kine-
matic Chain Space for evaluating local joint-angle plausi-
bility and develops a discriminative module accordingly to
ensure the plausibility of the augmented poses. These elab-
orate designs enable PoseAug to generate more diverse yet
plausible poses than existing ofﬂine augmentation methods,
and thus yield better generalization of the pose estimator.
PoseAug is generic and easy to be applied to various 3D
pose estimators. Extensive experiments demonstrate that
PoseAug brings clear improvements on both intra-scenario
and cross-scenario datasets.
Notably, it achieves 88.6%
3D PCK on MPI-INF-3DHP under cross-dataset evalua-
tion setup, improving upon the previous best data augmen-
tation based method [22] by 9.1%. Code can be found at:
https://github.com/jfzhang95/PoseAug.

## introduction
3D human pose estimation aims to estimate 3D body
joints in images or videos. It is a fundamental task with
broad applications in action recognition [47, 39], human-
*Equal contribution; order determined by coin toss. †Work done during
an internship at Huawei international Pte Ltd.
Source dataset: H36M
SemGCN SimpleBL ST-GCN
VPose3D
36
38
40
42
44
46
MPJPE(mm)
44.4
43.3
41.7
41.8
41.5
39.4
36.9
38.2
Train w/o PoseAug
Train w/   PoseAug
Cross dataset: 3DHP
SemGCN SimpleBL ST-GCN
VPose3D
65
70
75
80
85
90
95
100
105
MPJPE(mm)
97.4
85.3
87.8
86.6
86.1
76.2
74.9
73.0
Train w/o PoseAug
Train w/   PoseAug
Figure 1: Estimation error (in MPJPE) on H36M (intra-
dataset evaluation) and 3DHP (cross-dataset evaluation) of
four well established models [52, 26, 33, 3] trained with
and without PoseAug. PoseAug signiﬁcantly improves their
performance for both the intra- and cross-dataset settings.
robot interaction [11], human tracking [29], etc. This task
is typically solved using learning-based methods [26, 52, 3,
32] with ground truth annotations that are collected in the
laboratorial environments [16]. Despite their success in in-
door scenarios, these methods are hardly generalizable to
cross-scenario datasets (e.g., an in-the-wild dataset). We ar-
gue that their poor generalization is mainly due to the lim-
ited diversity of training data, such as limited variations in
human posture, body size, camera view point and position.
Recent works explore data augmentation to improve the
training data diversity and enhance the generalization of
their trained models. They either generate data through im-
age composition [37, 29, 28] and synthesis [5, 42], or di-
rectly generate 2D-3D pose pairs from the available training
data by applying pre-deﬁned transformations [22]. How-
ever, all of these works regard data augmentation and model
training as two separate phases, and conduct data augmen-
tation in an ofﬂine manner without interaction with model
training. Consequently, they tend to generate ineffective
augmented data that are too easy for model training, lead-
ing to marginal boost to the model generalization. More-
over, these methods heavily rely on pre-deﬁned rules such
as joint angle limitations [1] and kinematics constraints [37]
arXiv:2105.02465v1  [cs.CV]  6 May 2021

for data augmentation, which limit the diversity of the gen-
erated data and make the resulting model hardly generalize
to more challenging in-the-wild scenes.
To improve the diversity of augmented data, we propose
PoseAug, a novel auto-augmentation framework for 3D hu-
man pose estimation. Instead of conducting data augmenta-
tion and network training separately, PoseAug jointly opti-
mizes the augmentation process with network training end-
to-end in an online manner. Our main insight is that the
feedback from the training process can be used as effec-
tive guidance signals to adapt and improve the data aug-
mentation. Speciﬁcally, PoseAug exploits a differentiable
augmentation module (the ‘augmentor’) implemented by a
neural network to directly augment 2D-3D pose pairs in the
training data. Considering the potential domain shift with
respective to geometry in pose pairs (e.g., postures, view
points) [36, 22, 50], the augmentor learns to perform three
types of augmentation operations to respectively control 1)
the skeleton joint angle, 2) the body size, and 3) the view
point and human position. In this way, the augmentor is
able to produce augmented poses with more diverse geo-
metric features and thus relieves the diversity limitation is-
sue. With its differentiable capacity, the augmentor can be
optimized together with the pose estimator end-to-end via
an error feedback strategy. Concretely, by taking increasing
training loss of the estimator as the learning target, the aug-
mentor can learn to enrich the input pose pairs via enlarging
data variations and difﬁculties; in turn, through combating
such increasing difﬁculties, the pose estimator can become
increasingly more powerful during the training process.
To ensure the plausibility of the augmented poses, we use
a pose discriminator module to guide the augmentation, to
avoid generating implausible joint angles [1], unreasonable
positions or view points that may hamper model training.
In particular, the module consists of a 3D pose discrimina-
tor for enhancing the joint angle plausibility and a 2D pose
discriminator for guiding the body size, view point and po-
sition plausibility. The 3D pose discriminator adopts the
Kinematic Chain Space (KCS) [44] representation and ex-
tends it into a part-aware KCS for local-wise supervision.
More concretely, it splits skeleton joints into several parts
and focuses on joint angles in each part separately instead
of the whole body pose, which yields greater ﬂexibility of
the augmented poses. By jointly training the pose augmen-
tor, estimator and discriminator in an end-to-end manner
(Fig. 2), PoseAug can largely improve the training data di-
versity, and thus boost model performance on both source
and more challenging cross-scenario datasets.
Our PoseAug framework is ﬂexible regarding the choice
of the 3D human pose estimator. This is demonstrated by
the clear improvements made with PoseAug on four rep-
resentative 3D pose estimation models [52, 26, 33, 3] over
both source (H36M) [16] and cross-scenario (3DHP) [29]
Figure 2: Overview of our PoseAug framework. The aug-
mentor, estimator and discriminator are jointly trained end-
to-end with an error-feedback training strategy. As such,
the augmentor learns to augment data with guidance from
the estimator and discriminator.
datasets (Fig. 1). Remarkably, it brings more than 13.1%
average improvement w.r.t. MPJPE for all models on 3DHP.
Moreover, it achieves 88.6% 3D PCK on 3DHP under
cross-dataset evaluation setup, improving upon the previous
best data augmentation based method [22] by 9.1%.
Our contributions are three-fold. 1) To the best of our
knowledge, we are the ﬁrst to investigate differentiable data
augmentation on 3D human pose estimation. 2) We pro-
pose a differentiable pose augmentor, together with the er-
ror feedback design, which generates diverse and realistic
2D-3D pose pairs for training the 3D pose estimator, and
largely enhances the model’s generalization ability. 3) We
propose a new part-aware 3D discriminator, which enlarges
the feasible region of augmented poses via local-wise super-
vision, ensuring both data plausibility and diversity.

## method
Fig. 2 summarizes our PoseAug architecture design. It
includes 1) a pose augmentor that augments the input pose
pair {x, X} to an augmented one {x′, X′} for pose esti-
mator P training; 2) a pose discriminator module with two
discriminators in 3D and 2D spaces, to ensure the plausibil-
ity of the augmented data; and 3) a 3D pose estimator, that
provides pose estimation error feedback.

Augmentor Given a 3D pose X ∈R3×J, the augmentor
ﬁrst obtains its bone vector B ∈R3×(J−1) via a hierar-
chical transformation1 B = H(X) [44, 22], which can be
further decomposed into a bone direction vector ˆ
B (repre-
senting the joint angle) and a bone length vector ∥B∥(rep-
resenting the body size).
Then the augmentor applies multi-layer perceptron
(MLP) for feature extraction from the input 3D pose X.
Additionally, a noise vector based on Gaussian distribution
is concatenated with X in the feature extraction process to
incur sufﬁcient randomness for enhancing the feature di-
versity. The extracted features are then used for regressing
three operation parameters (γba, γbl and (R, t)) to change
the joint angles, body size, as well as view point and posi-
tion as illustrated in Fig. 3. Among these parameters,
1) γba ∈R3×(J−1) is the bone angle residual vector that is
used for adjusting the Bone Angle (BA) as follows:
ˆ
B′ = ˆ
B + γba,
(BA operation).
(4)
Speciﬁcally, BA operation will rotate the input bone di-
rection vector ˆ
B by γba, generating a new bone direc-
tion vector ˆ
B′.
2) γbl ∈R1×(J−1) represents the bone length ratio vector
that is used for adjusting the Bone Length (BL):
∥B′∥= ∥B∥× (1 + γbl),
(BL operation).
(5)
BL operation modiﬁes the input bone length vector ∥B∥
by γbl to adjust the body size. Notably, to ensure bio-
mechanical symmetry, the left and right body parts share
the same parameters.
3) R ∈R3×3 and t ∈R3×1 denote the rotation and trans-
lation parameters respectively for Rigid Transformation
(RT) operation to control pose view point and position:
X′ = R[H−1(B′)] + t,
(RT operation),
(6)
where B′ = ∥B′∥× ˆ
B′ is the augmented bone vector
from the above BA and BL operations. H−1 is the in-
verse hierarchical conversion to transform B′ back to a
3D pose [44, 22].
By applying these operations, the augmentor can generate
the augmented 3D pose X′ with more challenging pose,
body size, view point and position from the original 3D pose
X (Fig. 3). The augmented pose is then re-projected to 2D
with x′ = Π(X′), where Π : R3 →R2 denotes perspective
projection [15] via the camera parameters from the original
data. The augmented 2D-3D pair {x′, X′} is then used for
further training the pose estimator.
1The hierarchical transformation converts the J joints of X into J −1
column vectors of B, each of which represents a line segment connecting
two adjacent joints.
Figure 3: Augmentation operations with PoseAug.
A
source 3D pose is augmented by modifying its posture (via
BA operation), body size (via BL operation) and view point
and position (via RT operation).
Figure 4: Illustrations of the difference between original
and part-aware KCS based discriminator. Given a novel
and valid augmented pose, the original KCS based discrimi-
nator would wrongly classify it as fake as it does not appear
in source data (H36M), while the part-aware KCS based dis-
criminator would recognize is as real and approve it, since
it inspects local joint relations. It can be seen the part-aware
KCS based discriminator can help the augmentor generate
more diverse and plausible pose augmentation.
Discriminator Due to lacking priors in the augmentation
procedure, the augmented poses may present implausible
joint angles that violate the bio-mechanical structure [1], or
unreasonable positions and view points. Though such poses
are indeed harder cases for the estimator, training on them
would not beneﬁt the model generalization ability.
To ensure the plausibility of the augmented poses, we
introduce a pose discriminator module to guide the aug-
mentation. Speciﬁcally, the module consists of a 3D pose
discriminator D3d for evaluating the joint angle plausibility
and a 2D discriminator D2d for evaluating the body size,
viewpoint and position plausibility.
The key to the 3D pose discriminator design is to ensure
the pose plausibility without sacriﬁcing the diversity. In-
spired by the Kinematic Chain Space (KCS) [44], we design
a part-aware KCS as input to the discriminator. Instead of
taking the whole body pose into consideration as in the orig-
inal KCS, our part-aware KCS only focuses on local joint
angle and thus enlarges the feasible region of the augmented

pose, ensuring both plausibility and diversity (Fig. 4).
Speciﬁcally, to compute the part-aware KCS of an input
pose, either X or its augmentation X′, we convert the pose
to its bone direction vector ˆ
B as above and separate it into
5 parts (torso and left/right arm/leg) [1], denoted as ˆ
Bi, i =
1, . . . , 5, respectively. We then calculate the following local
joint angle matrix KCSi
local for the i-th part:
KCSi
local = ˆ
B⊤
i ˆ
Bi,
(7)
which encapsulates the inter joint angle information within
the i-th part. Based on the above local KCS representation,
a 3D pose discriminator D3d is constructed which takes the
KCSi
local as input and is trained for distinguishing the orig-
inal and augmented 3D poses.
Besides the 3D discriminator, we also introduce a 2D
discriminator to guide the augmentor to generate real body
size, view points and positions. As the 2D poses contain
information such as view point (rotation), position (trans-
lation), and body size (bone length), the 2D discriminator
can learn such information through adversarial training and
guide the pose augmentor in generating realistic rotation R,
translation t, and bone length ratio γbl.
Estimator The pose estimator P estimates 3D poses from
2D poses. We use the original and augmented 2D-3D pose
pair {x, X} and {x′, X′} to train the pose estimator. The
pose estimator contains a feature extractor to capture inter-
nal features from 2D poses, and a regression layer to es-
timate the corresponding 3D poses. Moreover, any exist-
ing effective estimator can be implemented in our PoseAug
framework. In Sec. 4.3, we conduct experiments to check
robustness of PoseAug with different estimators, and the re-
sults show PoseAug can bring noticeable improvements on
both source and cross-scenario datasets for all models.
3.4. Training Loss
Pose estimation loss We adopt the mean squared errors
(MSE) of the ground truth (GT) X and predicted poses f
X
as the pose estimation loss, which is formulated as
LP = ∥X −f
X∥2
2.
(8)
We train the pose estimator using LP with both original and
augmented pose pairs jointly, which can signiﬁcantly boost
performance for the challenging in-the-wild scenes.
Pose augmentation loss To facilitate model training, aug-
mented data should harder than the original one, i.e.,
LP(X′) > LP(X), but not too hard to hurt the training
process. A simple way to design the loss function is to
let the difference between the pose estimation loss on aug-
mented and original data within a proper range. Inspired
by [25, 21], we implement a controllable feedback loss as
Lfb = |1.0 −exp[LP(X′) −βLP(X)]|,
(9)
where β > 1 controls the difﬁculty level for the generated
poses, making the value of LP(X′) stay within a certain
range w.r.t. LP(X). During training, as the pose estima-
tor becomes increasingly more powerful, we accordingly
increase β value to generate more challenging augmenta-
tion data for training it.
Additionally, to prevent extremely hard cases from caus-
ing training collapse, we introduce a rectiﬁed L2 loss for
regularizing the augmentation parameters γba and γbl:
Lreg(γ) =
(
0,
if ¯γ < threshold,
∥γ∥2,
otherwise,
(10)
where γ denotes γba and γbl, and ¯γ denotes the mean value
over all of its elements. Combining Eqn. (9) and Eqn. (10),
the overall augmentation loss LA is formulated as
LA = Lfb + Lreg.
(11)
Pose discrimination loss For the discrimination loss LD,
we adopt the LS-GAN loss [25] for both 3D and 2D spaces:
LD = E[(D3d(X) −1)2] + E[D3d(X′)2]
+E[(D2d(x) −1)2] + E[D2d(x′)2],
(12)
where {x, X} and {x′, X′} denote the original (real) and
the augmented (fake) pose pairs, respectively.
End-to-end training strategy With the differentiable de-
sign, the pose augmentor, discriminator and estimator can
be jointly trained end-to-end. We update them alternatively
by minimizing losses Eqn. (11), Eqn. (12) and Eqn. (8). In
addition, we ﬁrst pre-train the pose estimator P before train-
ing the whole framework end-to-end, which ensures stable
training and produces better performance.

## experiments
We study four questions in experiments. 1) Is PoseAug
able to improve performance of 3D pose estimator for both
intra-dataset and cross-dataset scenarios? 2) Is PoseAug
effective at enhancing diversity of training data?
3) Is
PoseAug consistently effective for different pose estima-
tors and cases with limited training data?
4) How does
each component of PoseAug take effect? We experiment
on H36M, 3DHP and 3DPW. Throughout the experiments,
unless otherwise stated we adopt single-frame version of
VPose [33] as pose estimator.
4.1. Datasets
Human3.6M (H36M) [16] Following previous works [26,
52], we train our model on subjects S1, 5, 6, 7, 8 of H36M
and evaluate on subjects S9 and S11. We use two evalu-
ation metrics: Mean Per Joint Position Error (MPJPE) in
millimeters and MPJPE over aligned predictions with GT
3D poses by a rigid transformation (PA-MPJPE).

MPI-INF-3DHP (3DHP) [29] It is a large 3D pose dataset
with 1.3 million frames, presenting more diverse motions
than H36M. We use its test set to evaluate the model’s gen-
eralization ability to unseen environments, using metrics of
MPJPE, Percentage of Correct Keypoints (PCK) and Area
Under the Curve (AUC).
3DPW [43] It is an in-the-wild dataset with more compli-
cated motions and scenes. To verify generalization of the
proposed method to challenging in-the-wild scenarios, we
use its test set for evaluation with PA-MPJPE as metric.
MPII [2] and LSP [17] They are in-the-wild datasets with
only 2D body joint annotations and used for qualitatively
evaluating model generalization for unseen poses.

## related_work
3D human pose estimation Recent progress of 3D human
pose estimation is largely driven by the deployment of var-
ious deep neural network models [41, 26, 12, 52, 31, 3,
38, 53]. However, they all highly rely on well-annotated
data for fully-supervised model training and hardly gener-
alize to the new scenarios that present unseen patterns in
the training dataset, such as new camera views and subject
poses. Thus some recent works explore to leverage exter-
nal information to improve their generalization ability. For
example, some methods [54, 48, 9, 44, 14, 45, 6, 33, 20]
utilize 2D pose data collected in the wild for model train-
ing, e.g., through exploring kinematics priors for regular-
ization or post-processing [54, 9, 33], and adversarial train-
ing [48, 44]. More recently, geometry-based self-supervised
learning [36, 10, 4, 19, 34, 23, 35] has been used to train
models with unlabeled data.
Though effective, applying
these methods is largely constrained by the availability of
suitable external datasets. Instead of focusing on complex
network architectures and learning schemes, we explore a
learnable pose augmentation framework to enrich the 3D
pose data at hand directly. Speciﬁcally, the proposed frame-

work can generate 2D-3D pose pairs with both diversity and
plausibility for training pose estimation models. In addition,
our framework is generic and can adapt to those methods to
further improve their performance.
Data augmentation on 3D human poses Data augmen-
tation is widely used to alleviate the bottleneck of train-
ing data diversity and improve model generalization ability.
Some works augment data by stitching image patches [37,
29, 51], and some generate new data with graphics en-
gines [5, 42]. More recently, Li et al., [22] directly augment
2D-3D pose pairs through randomly applying partial skele-
ton recombination and joint angle perturbation on source
datasets. To ensure data plausibility, several constraints are
imposed, including joint angle limitation [1] and ﬁxed aug-
mentation range on view point and human position. De-
spite the good results on source data, these pre-deﬁned rules
limit the data diversity expansion and harm the model ap-
plicability to more challenging in-the-wild scenarios. Un-
like all these methods, we make the ﬁrst attempt to explore
learnable data augmentation on 3D human pose estimation,
which is shown effective for improving model generaliza-
tion ability.

## conclusion
In this paper, we develop an auto-augmentation frame-
work, PoseAug, that learns to enrich the diversity of training
data and improves performance of the trained pose estima-
tion models. The PoseAug effectively integrates three com-
ponents including the augmentor, estimator and discrimina-
tor and makes them fully interacted with each other. Specif-
ically, the augmentor is designed to be differentiable and
thus can learn to change major geometry factors of the 2D-
3D pose pair to suit the estimator better by taking its training
error as feedback. The discriminator can ensure the plausi-
bility of augmented data based on a novel part-aware KCS
representation. Extensive experiments justify PoseAug can
augment diverse and informative data to boost estimation
performance for various 3D pose estimators.
Acknowledgement
This
research
was
partially
sup-
ported by AISG-100E-2019-035,
MOE2017-T2-2-151,
NUS ECRA FY17 P08 and CRP20-2017-0006. JZ would
like to acknowledge the support of NVIDIA AI Tech Center
(NVAITC) to this research project.