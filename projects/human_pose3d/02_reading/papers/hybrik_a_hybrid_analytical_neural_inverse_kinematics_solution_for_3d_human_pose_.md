# HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation

> 2021 · id: W3167491448 · arXiv: 2011.14672 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Model-based 3D pose and shape estimation methods re-
construct a full 3D mesh for the human body by estimat-
ing several parameters. However, learning the abstract pa-
rameters is a highly non-linear process and suffers from
image-model misalignment, leading to mediocre model per-
formance.
In contrast, 3D keypoint estimation methods
combine deep CNN network with the volumetric represen-
tation to achieve pixel-level localization accuracy but may
predict unrealistic body structure. In this paper, we ad-
dress the above issues by bridging the gap between body
mesh estimation and 3D keypoint estimation. We propose a
novel hybrid inverse kinematics solution (HybrIK). HybrIK
directly transforms accurate 3D joints to relative body-part
rotations for 3D body mesh reconstruction, via the twist-
and-swing decomposition. The swing rotation is analyti-
cally solved with 3D joints, and the twist rotation is derived
from the visual cues through the neural network. We show
that HybrIK preserves both the accuracy of 3D pose and
the realistic body structure of the parametric human model,
leading to a pixel-aligned 3D body mesh and a more accu-
rate 3D pose than the pure 3D keypoint estimation methods.
Without bells and whistles, the proposed method surpasses
the state-of-the-art methods by a large margin on various
3D human pose and shape benchmarks. As an illustrative
example, HybrIK outperforms all the previous methods by
13.2 mm MPJPE and 21.9 mm PVE on 3DPW dataset. Our
code is available at https://github.com/Jeff-sjtu/HybrIK.

## introduction
Recovering the 3D surface from a monocular RGB im-
age is a fundamentally ill-posed problem. It has a wide rage
of application scenarios [54, 33, 28, 29, 8]. With the de-
velopment of the parametric statistical human body shape
†Cewu Lu is the corresponding author. He is the member of Qing
Yuan Research Institute, Qi Zhi Institute and MoE Key Lab of Artiﬁcial
Intelligence, AI Institute, Shanghai Jiao Tong University, China.
3D Skeleton
Parametric Model
3D
Keypoint
Estimation
3D
Mesh
Estimation
Forward Kinematics
Inverse Kinematics
Figure 1. Closing the loop between the 3D skeleton and the
parametric model via HybrIK. A 3D skeleton predicted by the
neural network can be transformed into a parametric body mesh
by inverse kinematics without loss of accuracy. The parametric
body mesh can generate structural realistic 3D skeleton by forward
kinematics.
models [2, 31, 49], a realistic and controllable 3D mesh of
human body can be generated from only a few parameters,
e.g. shape parameters and relative rotations of body parts.
Recent studies develop the model-based methods [7, 18, 24]
to obtain these parameters from the monocular RGB input
and produce 3D pose and shape of human bodies.
Most of the model-based methods can be catego-
rized into two classes:
optimization-based approach
and learning-based approach.
Optimization-based ap-
proaches [13, 7, 49] estimate the body pose and shape by
an iterative ﬁtting process. The parameters of the statistical
model are tuned to reduce the error between its 2D projec-
tion and 2D observations, e.g. 2D joint locations and silhou-
ette. However, the optimization problem is non-convex and
takes a long time to solve. The results are sensitive to the
initialization. These issues shift the spotlight towards the
learning-based approaches. With a parametric body model,
learning-based approaches use neural networks to regress
the model parameters directly [18, 24, 20]. But the parame-
ter space in the statistical model is abstract, making it difﬁ-
cult for the networks to learn the mapping function.
This challenge prompts us to look into the ﬁeld of 3D
keypoint estimation. Instead of the direct regression, previ-
1
arXiv:2011.14672v4  [cs.CV]  27 Apr 2022

ous methods [50, 62] adopt volumetric heatmap as the target
representation to learn 3D joint locations and have achieved
impressive performance. This inspires us to build a collabo-
ration between the 3D joints and the body mesh (Fig. 1). On
the one hand, the accurate 3D joints facilitate the 3D body
mesh estimation. On the other hand, the shape prior in para-
metric body model in turn ﬁxes the unrealistic body struc-
ture issue of the 3D keypoint estimation methods. Since the
current 3D keypoint estimation methods lack explicit mod-
elling of the distribution of body bone length, it may predict
unrealistic body structures like left-right asymmetry and ab-
normal proportions of limbs. By leveraging the parametric
body model, the presented human shape better conforms to
the actual human body.
In this work, we propose a hybrid analytical-neural in-
verse kinematics solution (HybrIK) to bridge the gap be-
tween 3D keypoint estimation and body mesh estimation.
Inverse kinematics (IK) process is the mathematical process
of ﬁnding the relative rotations to produce the desired loca-
tions of body joints. It is an ill-posed problem because there
is no unique solution. The core of our approach is to pro-
pose an innovative IK solution via twist-and-swing decom-
position. The relative rotation of a skeleton part is decom-
posed into twist and swing, i.e. a longitudinal rotation and
an in-plane rotation. In HybrIK, we composite the entire
rotation recursively along the kinematic tree by analytically
calculating swing rotation and predicting twist rotation. A
critical characteristic of our approach is that the relative ro-
tation estimated by HybrIK is naturally aligned with the 3D
skeleton, without the need for additional optimization pro-
cedures in the previous approaches [7, 49, 24]. All oper-
ations in HybrIK are differentiable, which allows us to si-
multaneously train 3D joints and human body mesh in an
end-to-end manner. Besides, experiments indicate that Hy-
brIK raises the performance of body mesh estimation to the
same level as 3D keypoint estimation and takes a step for-
ward. The proposed approach is benchmarked in various 3D
human pose and shape datasets, and it signiﬁcantly outper-
forms state-of-the-art approaches [24, 40] by 21.9 mm PVE
on 3DPW [69], 6.6 mm PA-MPJPE on Human3.6M [16]
and 10.8 AUC on MPI-INF-3DHP [35].
The contributions of our approach can be summarized as
follows:
• We propose HybrIK, a hybrid analytical-neural IK so-
lution that converts the accurate 3D joint locations to
full 3D human body mesh. HybrIK is differentiable
and allows end-to-end training.
• Our approach closes the loop between the 3D skeleton
and the parametric model. It ﬁxes the alignment issue
of current model-based body mesh estimation methods
and the unrealistic body structure problem of 3D key-
point estimation methods at the same time.
• Our approach achieves state-of-the-art performance
across various 3D human pose and shape benchmarks.

## method
In this section, we present our hybrid analytical-neural
inverse kinematics solution that boosts 3D human pose and
shape estimation (Fig. 2). First, in §3.1, we brieﬂy describe
the forward kinematics process, the inverse kinematics pro-
cess and the SMPL model. In §3.2, we introduce the pro-
posed inverse kinematics solution, HybrIK. Then, in §3.3,
we present the overall learning framework to estimate the
pixel-aligned body mesh and realistic 3D skeleton. Finally,
we provide the necessary implementation details in §3.4.
3.1. Preliminary
Forward Kinematics.
Forward kinematics (FK) for hu-
man pose usually refers to the process of computing the re-
constructed pose Q = {qk}K
k=1, with the rest pose template
T = {tk}K
k=1 and the relative rotations R = {Rpa(k),k}K
k=1
as input:
Q = FK(R, T),
(1)
where K is the number the body joints, qk ∈R3 denotes the
reconstructed 3D location of the k-th joint, tk ∈R3 denotes
the k-th joint location of the rest pose template, pa(k) re-
turn the parent’s index of the k-th joint, and Rpa(k),k is the
relative rotation of k-th joint with respect to its parent joint.
FK can be performed by recursively rotating the template
body part from the root joint to the leaf joints:
qk = Rk(tk −tpa(k)) + qpa(k),
(2)
where Rk ∈SO(3) is the global rotation of the k-th joint
with respect to the canonical rest pose space. The global
rotation can be calculated recursively:
Rk = Rpa(k)Rpa(k),k.
(3)
For the root joint that has no parent, we have q0 = t0.
Inverse Kinematics.
Inverse kinematics (IK) is the re-
verse process of FK, computing relative rotations R that
can generate the desired locations of input body joints P =
{pk}K
k=1. This process can be formulated as:
R = IK(P, T),
(4)
where pk denotes the k-th joint of the input pose. Ideally,
the resulting rotations should satisfy the following condi-
tion:
pk −ppa(k) = Rk(tk −tpa(k))
∀1 ≤k ≤K.
(5)
Similar to the FK process, we have p0 = t0 for the root joint
that has no parent. While the FK problem is well-posed, the
IK problem is ill-posed because there is either no solution
or because there are many solutions to fulﬁll the target joint
locations.
3

CNN
deconv
HybrIK
Φ
β
θ
M(θ, β)
Input Image
Regressed Pose
Reconstructed Mesh
Rest Pose
P
T
Reconstructed Pose
M
FK or Regressor
Q
fully-connected
Figure 2. Overview of the proposed framework. A 3D heatmap is generated by the deconvolution layers and used to regress the 3D joints
P. The shape parameters β and the twist angle Φ are learned from the visual cues through the fully-connected layers. These results are
then sent to the HybrIK process to solve the relative rotation, i.e. the pose parameters θ. Finally, with the pose and shape parameters, we
can obtain the reconstructed body mesh M, and the reconstructed pose Q via a further FK process or linear regression.
SMPL Model.
In this work, we employ the SMPL [31]
parametric model for human body representation. SMPL
allows us to use shape parameters and pose parameters
to control the full human body mesh. The shape param-
eters β ∈R10 are parameterized by the ﬁrst 10 princi-
pal components of the shape space.
The pose parame-
ters θ are modelled by relative 3D rotation of K = 23
joints, θ = (θ1, θ2, · · · , θK). SMPL provides a differen-
tiable function M(θ, β) that takes the pose parameters θ
and the shape parameters β as input and outputs a triangu-
lated mesh M ∈RN×3 with N = 6980 vertices. Conve-
niently, the reconstructed 3D joints Qsmpl can be obtained
by an FK process, i.e. Qsmpl = FK(R, T). Also, the joints
of Human3.6M [16] can be obtained by a linear combina-
tion of the mesh vertices through a linear regressor W, i.e.
Qh36m = WM.
3.2. Hybrid Analytical-Neural Inverse Kinematics
Estimating the human body mesh by direct regression of
the relative rotations is too difﬁcult [18, 24, 20]. Here, we
propose a hybrid analytical-neural inverse kinematics solu-
tion (HybrIK) to leverage 3D keypoints estimation to boost
3D body mesh estimation. Since 3D joints cannot uniquely
determine the relative rotation, we decompose the original
rotation into twist and swing. The 3D joints are utilized to
calculate the swing rotation analytically, and we exploit the
visual cues by a neural network to estimate the 1-DoF twist
rotation. In HybrIK, the relative rotations are solved recur-
sively along the kinematic tree. We conduct error analysis
and further develop an adaptive solution to reduce the re-
construction error.
Twist-and-Swing Decomposition.
In the analytical IK
formulation, some body joints are usually assigned lower
degree-of-freedom (DoFs) to simplify the problem, e.g. 1
or 2 DoFs [22, 64, 17]. In this work, we consider a gen-
Twist
Swing
(a) Original Rotation
(b) Twist-and-Swing Decompostion⃗
t⃗
p⃗
p = R⃗t⃗
p = RswRtw⃗t
Figure 3. Illustration of the twist-and-swing decomposition. (a)
The original rotation turns the right palm-down hand to the front
and the palm to the left in one step. (b) With twist-and-swing
decomposition, the rotation can be divided into two steps: First,
turn the palm 90◦, and then move the entire hand to the front.
eral case where each body joint is assumed to have full 3
DoFs. As illustrated in Fig. 3, a rotation R ∈SO(3) can be
decomposed into a twist rotation Rtw and a swing rotation
Rsw. Given the start template body-part vector⃗t and the tar-
get vector⃗p, the solution process of R can be formulated
as:
R = D(⃗p,⃗t, φ) = Dsw(⃗p,⃗t)Dtw(⃗t, φ) = RswRtw,
(6)
where φ is the twist angle that estimated by a neural net-
work, Dsw(·) is a closed-form solution of the swing rota-
tion, and Dtw(·) transforms φ to the twist rotation. Here, R
should satisfy the condition in Eq. 5, i.e.⃗p = R⃗t.
- Swing:
The swing rotation has the axis⃗n that is per-
pendicular to⃗t and⃗p. Therefore, it can be formulated as:⃗
n =⃗
t ×⃗p
∥⃗t ×⃗p ∥,
(7)
4

and the swing angle α satisﬁes:
cos α =⃗
t ·⃗p
∥⃗t∥∥⃗p∥,
sin α = ∥⃗t ×⃗p ∥
∥⃗t∥∥⃗p∥.
(8)
Hence, the closed-form solution of the swing rotation Rsw
can be derived by the Rodrigues formula:
Rsw = Dsw(⃗p,⃗t) = I + sin α[⃗n]× + (1 −cos α)[⃗n]2
×, (9)
where [⃗n]× is the skew symmetric matrix of⃗n and I is the
3 × 3 identity matrix.
- Twist:
The twist rotation is rotating around⃗t itself.
Thus, with⃗t itself the axis and φ the angle, we can deter-
mine twist rotation Rtw:
Rtw = Dtw(⃗t, φ) = I + sin φ
∥⃗t∥[⃗t]× + (1 −cos φ)
∥⃗t∥2
[⃗t]2
×, (10)
where [⃗t]× is the skew symmetric matrix of⃗t.
Note that function Dsw and Dtw are fully differentiable,
which allows us to integrate the twist-and-swing decompo-
sition into the training process. Although we need a neural
network to learn the twist angle, the difﬁculty of learning
is signiﬁcantly reduced. Compared with the 3-DoF rotation
that is directly regressed in previous work [18, 24, 20], the
twist angle is only a 1-DoF variable. Moreover, due to the
physical limitation of the human body, the twist angle has a
small range of variation. Therefore, it is much easier for the
networks to learn the mapping function. We further analyze
its variation in §4.2.
Naive HybrIK.
The IK process can be performed recur-
sively along the kinematic tree like the FK process. First of
all, we need to determine the global root rotation R0, which
has a closed-form solution using the locations of spine,
left hip, right hip and Singular Value Decomposition
(SVD). Detailed mathematical proof is provided in the sup-
plemental document. Then, in each step, e.g. the k-th step,
we assume the rotation of the parent joint Rpa(k) is known.
Hence, we can reformulate Eq. 5 with Eq. 3 as:
R−1
pa(k)(pk −ppa(k)) = Rpa(k),k(tk −tpa(k)).
(11)
Let⃗pk = R−1
pa(k)(pk −ppa(k)) and⃗tk = (tk −tpa(k)), we
can solve the relative rotation via Eq. 6:
Rpa(k),k = D(⃗pk,⃗tk, φk),
(12)
where φk is the network predicting twist angle for the k-th
joint. The set of twist angle is denoted as Φ = {φk}K
k=1.
Since the rotation matrices are orthogonal, their inverse
equals to their transpose, i.e. R−1
pa(k) = RT
pa(k), which keeps
the solving process differentiable.
The whole process is named Naive HybrIK and summa-
rized in Alg. 1. Note that we solve the relative rotation
Algorithm 1: Naive HybrIK
Input: P, T, Φ
Output: R
1 Determine R0;
2 for k along the kinematic tree do
3⃗
pk ←R−1
pa(k)(pk −ppa(k));
4⃗
tk ←(tk −tpa(k));
5
Rsw
pa(k),k ←Dsw(⃗pk,⃗tk);
6
Rtw
pa(k),k ←Dtw(⃗tk, φk);
7
Rpa(k),k ←Rsw
pa(k),kRtw
pa(k),k;
Rpa(k),k instead of the global rotation Rk. The reason is
that if we directly decompose the global rotation, the result-
ing twist angle will depend on all ancestors’ rotations along
the kinematic tree, which increases the variation of the dis-
tal limb joints and the difﬁculty for the network to learn.
Adaptive HybrIK.
Although the Naive HybrIK process
seems effective, it follows an unstated hypothesis: ∥pk −
ppa(k)∥= ∥tk −tpa(k)∥. Otherwise, there is no solution for
Eq. 5. Unfortunately, in our case, the body-parts predicted
by the 3D keypoint estimation method are not always con-
sistent with the rest pose template. In Naive H

## related_work
3D Keypoint Estimation.
Many works formulate 3D hu-
man pose estimation as the problem of locating the 3D
joints of the human body. Previous studies can be divided
into two categories: single-stage and two-stage approaches.
Single-stage approaches [51, 56, 36, 76, 38, 63, 39, 71] di-
rectly estimate the 3D joint locations from the input im-
age. Various representations are developed, including 3D
heatmap [51], location-map [36] and 2D heatmap + z re-
gression [76].
Two-stage approaches ﬁrst estimate 2D
pose and then lift them to 3D joint locations by a learned
dictionary of 3D skeleton [1, 55, 66, 58, 78, 79] or re-
gression [48, 73, 42, 11, 34, 61]. Two-stage approaches
highly rely on the accurate 2D pose estimators, which have
achieved impressive performance by the combination of
powerful backbone network [60, 15, 44, 46, 47] and the 2D
heatmap.
These privileged forms of supervision contribute to the
recent performance leaps of 3D keypoint estimation. How-
ever, the human structural information is modelled implic-
itly by the neural network, which can not ensure the output
3D skeletons to be realistic. Our approach combines the ad-
vantages of both the 3D skeleton and parametric model to
predict accurate and realistic human pose and shape.
Model-based 3D Pose and Shape Estimation.
Pioneer
works on the model-based 3D pose and shape estimation
methods use parametric human body model [2, 31, 49] as
the output target because they capture the statistics prior
of body shape.
Compared with the model-free meth-
ods [67, 25, 40], the model-based methods directly predict
controllable body mesh, which can facilitate many down-
stream tasks for both computer graphics and computer vi-
sion. Bogo et al. [7] propose SMPLify, a fully automatic
approach, without manual user intervention [59, 13]. This
optimization paradigm was further extended with silhouette
cues [27], volumetric grid [67], multiple people [75] and
whole-body parametric model [49].
With the advances of the deep learning networks, there
are increasing studies that focus on the learning-based
methods, using a deep network to estimate the pose and
shape parameters.
Since the mapping from RGB image
to shape space and relative body-part rotation is hard to
learn, many works use some form of intermediate repre-
sentation to alleviate this problem, such as keypoints and
silhouettes [52], semantic part segmentation [45] and 2D
heatmap input [65]. Kanazawa et al. [18] use an adversarial
prior and an iterative error feedback (IEF) loop to reduce
the difﬁculty of regression. Arnab et al. [4] and Kocabas
et al. [20] exploit temporal context, while Guler et al. [14]
2

use a part-voting expression and test-time post-processing
to improve the regression network. Kolotouros et al. [24]
leverage the optimization paradigm to provide extra 3D su-
pervision from unlabeled images.
In this work, we address this challenging learning prob-
lem by a transformation from the pixel-aligned 3D joints to
the relative body-part rotations.
Body-part Rotation in Pose Estimation.
The core of our
approach is to calculate the relative rotation of human body
parts through a hybrid IK process. There are several works
that estimate the relative rotations in the 3D pose estima-
tion literature. Zhou et al. [77] use the network to predict
the rotation angle of each body joint, followed by an FK
layer to generate the 3D joint coordinates. Pavllo et al. [53]
switch to quaternions, while Yoshiyasu et al. [74] directly
predict the 3×3 rotation matrices. Mehta et al. [37] ﬁrst es-
timate the 3D joints and then use a ﬁtting procedure to ﬁnd
the rotation Euler angles. Previous approaches are either
limited to a hard-to-learn problem or require an additional
ﬁtting procedure. Our approach recovers the body-part ro-
tation from 3D joint locations in a direct, accurate and feed-
forward manner.
Inverse Kinematics Process.
The inverse kinematics
(IK) problem has been extensively studied during recent
decades.
Numerical solutions [6, 72, 12, 19, 70, 9] are
simple ways to implement the IK process, but they suf-
fer from time-consuming iterative optimization. Heuristic
methods are efﬁcient solutions to the IK problem. For ex-
ample, CDC[32], FABRIK[3] and IK-FA[57] have a low
computational cost for each heuristic iteration. In some spe-
cial cases, there exist analytical solutions to the IK prob-
lem. Tolani et al. [64] propose a reliable algorithm by the
combination of analytical and numerical methods. Kall-
mann et al. [17] solve the IK for arm linkage, i.e. a three-
joint system. Recently, researchers have been interested in
using neural networks to solve the IK problem in robotic
control [10], motion retargeting [68] and hand pose estima-
tion [43, 23].
In this work, we combine the interpretable characteristic
of analytical solution and the ﬂexibility of the neural net-
work, introducing a feed-forward hybrid IK algorithm with
twist-and-swing decomposition. Twist-and-swing decom-
position is introduced by Baerlocher et al. [5]. The twist
angles are limited based on the particular body joint. In our
works, the twist angles are estimated by neural networks,
which is more ﬂexible and can be generalized to all body
joints. Compared with previous analytical solutions [17]
designed for speciﬁc joint linkage, our algorithm can be ap-
plied to the entire body skeleton in a direct and differen-
tiable manner.

## conclusion
In this paper, we bridge the gap between 3D keypoint
estimation and body mesh estimation via a novel hybrid
analytical-neural inverse kinematics solution, HybrIK. It
transforms the 3D joint locations to a pixel-aligned accurate
human body mesh, and then obtains a more accurate and re-
alistic 3D skeleton from the reconstructed 3D mesh, closing
the loop between the 3D skeleton and the parametric body
model. Our method is fully differentiable and allows simul-
taneously training of 3D joints and human body mesh in
an end-to-end manner. We demonstrate the effectiveness of
our method on various 3D pose and shape datasets. The pro-
posed method surpasses state-of-the-art methods by a large
margin. Besides, comprehensive analyses demonstrate that
8

HybrIK is robust and has error correction capability. We
hope HybrIK can serve as a solid baseline and provide a
new perspective for the 3D human pose and shape estima-
tion task.
Acknowledgements
This work is supported in part
by the National Key R&D Program of China,
No.
2017YFA0700800, National Natural Science Foundation of
China under Grants 61772332 and Shanghai Qi Zhi Insti-
tute, SHEITC (018-RGZN-02046).