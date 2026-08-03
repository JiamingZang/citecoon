# Distill Knowledge From NRSfM for Weakly Supervised 3D Pose Learning

> 2019 · id: W2981691949 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Distill Knowledge from NRSfM for Weakly Supervised 3D Pose Learning
Chaoyang Wang
Chen Kong
Simon Lucey
Carnegie Mellon University
{chaoyanw, chenk, slucey}@cs.cmu.edu
Abstract
We propose to learn a 3D pose estimator by distilling
knowledge from Non-Rigid Structure from Motion (NRSfM).
Our method uses solely 2D landmark annotations. No 3D
data, multi-view/temporal footage, or object speciﬁc prior
is required. This alleviates the data bottleneck, which is one
of the major concern for supervised methods. The challenge
for using NRSfM as teacher is that they often make poor
depth reconstruction when the 2D projections have strong
ambiguity. Directly using those wrong depth as hard target
would negatively impact the student. Instead, we propose a
novel loss that ties depth prediction to the cost function used
in NRSfM. This gives the student pose estimator freedom to
reduce depth error by associating with image features. Val-
idated on H3.6M dataset, our learned 3D pose estimation
network achieves more accurate reconstruction compared
to NRSfM methods. It also outperforms other weakly super-
vised methods, in spite of using signiﬁcantly less supervi-
sion.
1. Introduction
Learning to estimate 3D pose from images is bottle-
necked by the availability of abundant 3D annotated data.
Weakly supervised methods that reduce the amount of re-
quired annotation is of high practical value. Prior works
approach this problem by supplementing their training set
with: (i) extra 2D annotated data [47]; (ii) aligning 3D mod-
els to 2D annotations [35, 43, 37]; (iii) exploiting geometric
cues from multi-view footage [33, 32, 38]; or (iv) utilizing
adversarial framework to impose a prior on the 3D struc-
ture [12]. These methods, however, are either restricted to
laboratory settings or still requires a 3D training set – which
limits the type of target objects they can work with. This
paper addresses a more general setting – we utilize image
datasets with solely 2D landmark annotations (i.e. no 3D
supervision). This allows our method to be applied to a
wider scope of objects, not limited by the availability of 3D
models, kinematic priors, or sequential/multi-view footage.
Our work is made possible by some recent advances in
Deep-NRSfM
Ours
GT.
Figure 1. NRSfM methods often achieve poor reconstructions
when the 2D projections have strong ambiguity. Our proposed
knowledge distilling method lets the student pose estimation net-
work (3rd column) correct some of the mistakes made by its
NRSfM teacher (2nd column).
Non-Rigid Structure from Motion (NRSfM). NRSfM meth-
ods reconstruct 3D shapes and camera positions from multi-
ple 2D projections of articulated 3D points. These points do
not have to belong to the same object, but can be from mul-
tiple instances of the same object category, which naturally
applies to our problem. Prior NRSfM methods are restricted
by the number of frames and the type of shape variabil-
ity they can handle, which limits their usage to many real
world problems. Kong and Lucey [21] recently proposed
a neural network architecture (Deep-NRSfM) interpreted as
solving a multi-layer block sparse dictionary learning prob-
lem, and can handle problems of unprecedented scale and
shape complexity. Our modiﬁed version of Deep-NRSfM
achieves state-of-the-arts accuracy on H3.6M [18] dataset,
outperforming other NRSfM methods by a signiﬁcant mar-
gin.
Despite this progress, NRSfM still has difﬁculty in pre-
dicting correct depth for shapes with strong ambiguity in
terms of 2D projection, e.g. identifying if a leg is stretching
1743

towards/away from the camera, even though these are dis-
tinguishable with texture features. Therefore, directly using
the depth output from NRSfM as labels to train a pose esti-
mation network is affected by those errors. Instead of this
hard assignment of training labels, we propose a softer ap-
proach – we want to penalize less when there’s high ambi-
guity in 2D projection, so as to leave room for the pose esti-
mation network to correct errors made by NRSfM through
associating image features (see Fig. 1).
To design our learning objective, we review the dictio-
nary learning problem used to solve NRSfM. Assuming the
camera matrix ﬁxed, a depth hypothesis deﬁnes a subspace
of codes – any codes in this subspace is to have the same
depth reconstruction as the hypothesis, but have different
cost (2D reprojection error + regularizer). A natural way to
characterize the quality of a depth hypothesis is by the mini-
mum cost of codes in its subspace. However, directly using
this as a learning objective leads to solving a constrained
optimization problem numerically per SGD iteration, which
is computationally intractable. Instead, we derive a convex
upper bound by evaluating the cost at the projection of the
NRSfM solution on the subspace. Experiments show that
pose network trained by this loss noticeably reduces error
on the training set compared to our already strong NRSfM
baseline, and consequently leads to lower validation error
as a weakly supervised learning task.
Another beneﬁt of the proposed knowledge distilling
loss is that, it poses no restriction on the architecture of
the student pose estimation network, as long as it outputs
the depth value for the landmarks. This is not the case for
some of the prior works [43, 13], where the pose estimation
network has to output the coefﬁcients associated to some
external shape dictionary.
In conclusion, contributions of this paper are:
• We propose a weakly supervised pose estimation
method using solely 2D landmark annotations. We do
not use any 3D labels, multi-view footage, or target
speciﬁc shape prior. In spite of using weaker super-
vision, we achieve the best results compared to other
weakly supervised methods.
• We establish a strong NRSfM baseline modiﬁed from
Deep-NRSfM [21], which outperforms current pub-
lished state-of-the-art NRSfM methods on H3.6M
dataset.
• We propose a new knowledge distilling algorithm ap-
plicable to NRSfM methods based on dictionary learn-
ing. We demonstrate that our learned network gets sig-
niﬁcantly lower error on the training set compared to
its NRSfM teacher.
2. Related Works
Non-rigid structure from motion
NRSfM is a classical
ill-posed problem since the 3D shapes can vary between
images, resulting in more variables than equations. To al-
leviate the ill-posedness, various constraints are exploited
including 1) temporal smoothness [2, 15, 24, 23], 2) ﬁxed
articulation [31] and more commonly used 3) shape priors.
The ﬁrst statistical shape prior—non-rigid objects can be
modeled by a local subspace in low rank—is ﬁrst proposed
by Bregler et al. [5] and later developed by Dai et al. [9].
Following this direction, increasing works are reported
to model more complex objects while still maintaining a
well-conditioned system. Among them, representatives are
union-of-subspaces [48, 1], and block-sparsity [20, 22]. Of
particular interest to this paper is the most recent work [21]
that introduces deep neural network to accurately solving
large scale NRSfM problem. Even though great success,
majority NRSfM algorithms rely heavily on 2D annotation-
based priors. However, as pointed in the introduction, much
broader information are embedded under image itself, under
pixel values. In this paper, we impose a novel image prior
such that NRSfM is no longer trapped at 2D coordinates of
landmarks but also learn from origin images.
Weakly supervised 3D pose learning
Most 3D pose es-
timation methods [36, 30, 29, 47, 45, 44, 28, 8, 26, 6] are
fully supervised. One bottleneck for the supervised meth-
ods is that data coming from multi-view motion capture sys-
tems [19, 18] includes limited number of human subject,
and has simple backgrounds. This would affect the gen-
eralization ability of a trained model. Weakly supervised
methods aim to alleviate this problem by limiting the re-
quirement for labeled data. They can be loosely categorized
as: using synthetic datasets [7, 40] to increase the training
set size. These methods face the problem of generalizing to
new motions and environments that are different from the
simulated data; On the other hand, given the existing large-
scale image datasets with 2D annotation, Zhou et al. [47]
train their model with 2D labeled images together with mo-
tion capture data. To further reduce dependency on paired
3D annotation, 3D interpreter network [43], multi-modal
model [37] and generative adversarial networks [13, 41] are
trained on external 3D data; multi-view footage is also used
to enforce geometric constraints [38, 33]; However, these
methods still require a large enough 3D training set to prop-
erly initialize and constraint their learning process.
Recently, Rhodin et al. [32] propose a method based
on geometric-aware representation learning, which requires
only a small amount of annotation. Its performance how-
ever is limited, which restricts its practical usage. A con-
current work of Drover et al. [12] propose to use adversar-
ial framework to impose a prior on the 3D structure, learned
744

solely from 2D projections. Yet they still utilize the ground-
truth 3D poses to generate a large number of synthetic 2D
poses for training, which augments the original 1.5M 2D
poses in Human3.6M by almost 10 times.
3. Non-rigid Structure from Motion
Under weak perspective camera assumption, 2D projec-
tion W ∈RP ×2 is the product of 3D shape S ∈RP ×3 and
camera matrix M ∈R3×2:
W = SM, W =


...
...
up
vp
...
...

, S =


...
...
...
xp
yp
zp
...
...
...

,
(1)
where (up, vp) and (xp, yp, zp) are the image and world co-
ordinate of p-th point, and M is required to be orthonormal.
The goal of NRSfM is to recover 3D shape S and camera
matrix M given the observed 2D projections W. This is
an inherent ill-posed problem. Finding a unique solution
requires sufﬁcient regularization and prior knowledge.
One type of NRSfM methods approach the problem
through dictionary learning. Denote s ∈R3P is the vec-
torization of S, it satisﬁes: s = Dϕ, where D ∈R3P ×K
is a dictionary with K bases; and ϕ ∈RK is a code vector.
Given multiple observation of 2D projections W(i) from an
articulated object deforming over time, or different objects
of the same category, these methods can be loosely inter-
preted as minimizing the following objective:
min
D,{ϕ(i)},{M(i)}
X
i
∥[Dϕ(i)]P ×3M(i) −W(i)∥+ h(ϕ(i))
(2)
where operator [ ]P ×3 is deﬁned as reshaping the vecorized
3D shape into matrix form with dimension P × 3; h(ϕ) is
a regularizer introduced to improve uniqueness of solution,
e.g. low rank [9], sparsity [20], etc.
Our knowledge distilling method (see Section 4) is de-
signed for this general type of NRSfM method, and in prin-
cipal, it is agnostic to the type of regularizor they use, as
long as the dictionary is overcomplete.
Deep NRSfM
Kong and Lucey[21] propose a prior as-
sumption that 3D shapes are compressible via multi-layer
sparse coding:
s = D1ϕ1,
∥ϕ1∥1 ≤λ1,
ϕ1 ≥0,
ϕ1 = D2ϕ2,
∥ϕ2∥1 ≤λ2,
ϕ2 ≥0,
...,
...
ϕn−1 = Dnϕn,
∥ϕn∥1 ≤λn,
ϕn ≥0,
(3)
where Di are hierarchical dictionaries, and code vectors
ϕi ∈RKi are constrained to be sparse and non-negative.
Compared to single level sparse coding, codes in multi-
layer sparse coding not only minimizes the reconstruction
error at their individual levels, but is also regularized by the
codes from other levels. This helps to impose more con-
straints on code recovery while maintaining similar shape
expressibility versus single level sparse coding with the
same dictionary size.
To recover sparse codes, one of the classical method
to use is Iterative Shrinkage and Thresholding Algorithm
(ISTA) [10, 4, 34].
Papyan et al. [27] ﬁnd that feed-
forward neural netorks can be interpreted as approximat-
ing one iteration of inferencing sparse codes by ISTA, and
the dictionaries D1, D2, . . . , Dn serves as the neural net-
work weights. Based on this insight, Chen et al. derived a
novel neural network architecture which approximates the
solution of sparse codes ϕ1 and camera matrix M.
In
this paper, we made signiﬁcant modiﬁcation to their orig-
inal architecture, which we ﬁnd important to get good re-
sult in experiment.
Limited by space, we put descrip-
tion about our version of camera matrix estimation network
qM(W) : RP ×2 7→R3×2, and sparse code estimation net-
work qϕ(W, M) : RP ×2 × R3×2 7→RK1 in the supple-
mentary material.
With the feed-forward code/camera estimation networks
parameterized by the dictionaries, we can now learn the dic-
tionaries through minimizing reprojection error of all sam-
ples in the dataset. Denote ˜ϕ(i)
1 , ˜M(i) to be the output of
networks qϕ, qM given ith 2D projection W(i), the loss
function is:
min
D1,D2,...,Dn
X
i
∥[D1 ˜ϕ(i)
1 ]P ×3 ˜M(i) −W(i)∥2 + λ∥˜ϕ1∥1.
(4)
In this loss function, in addition to reprojection error, we
add sparisity penalty using a small weighting, which we ﬁnd
helpful to improve results.
4. Distilling Knowledge from NRSfM
Problem setup:
Given an image dataset paired with
annotated 2D locations of landmarks on target objects:
{(I(i), W(i))}, we want to train a 3D pose estimation net-
work able to predict 3D landmark positions from image in-
put. The main difﬁculty of this task is how to learn to predict
depth of landmarks without any depth supervision. Our cue
is from dictionary learning-based NRSfM method (Deep-
NRSfM in our experiment), which gives us a 3D shape dic-
tionary D, and recovered camera matrices M(i) and codes
ϕ(i)
nrsfm.
With the dictionary, camera matrices and codes from
NRSfM, depth in the image coordinate can be computed by
simply rotating the 3D shape reconstruction Dϕ(i)
nrsfm. Given
this, a simple baseline for this task would be: we use the
depth reconstruction as labels to train the 3D pose estima-
745

𝜑"
𝜑#
𝝋∗(𝒛)
𝑑ℒ
𝑑𝝋
𝑺(𝒛)
𝝋,-./0
𝜑"
𝜑#
𝝋∗(𝒛)
𝝋,-./0
𝑺(𝒛)
𝑑ℒ
𝑑𝝋
Ground truth
NRSfM
Ours
𝑺(𝐳𝐧𝐫𝐬𝐟𝐦)
𝑺(𝒇𝒛(𝑰; 𝜽))
𝜑"
𝜑#
𝝋,-./0
𝝋∗(𝒛)
  𝝋<
𝑑ℒ
𝑑𝝋
𝑺(𝒛)
NRSfM
(a)
(b)
(c)
Figure 2. Illustration of the proposed knowledge distilling algorithm. (a) For illustration purpose, we assume the code ϕ is 2-dimensional.
We plot the cost function (Eq. 9) as a 2D heatmap. The NRSfM solution ϕnrsfm is approximately the minima of this heat map (represented
as red dot). Given a depth hypothesis z, all the codes satisﬁes z forms a subspace S(z), which is shown as the orange line. The quality of a
depth hypothesis is evaluated by the best point on its subspace, denoted as ϕ∗(z) (red cross). Given different depth hypothesis is equivalent
to parallel translate the line. Suppose z is free to have any value, then minimizing our loss function (Eq. 10) would push the line to cross
ϕnrsfm(see the dashed orange line). This gives the same wrong depth reconstruction as the NRSfM method. (b) Suppose we get another
image of similar pose but with less 2D projection ambiguity. In this case, NRSfM gives correct shape recovery. Since texture features are
similar for both images, the pose estimation network is implicitly constrained to make similar depth predictions. Then minimizing our loss
for both images would lead to a better solution for image 1 (shown as solid orange line), because gradients are larger from the 2nd image
due to the fact that it has less ambiguity. (c) We approximate the loss by evaluating at the projection of ϕnrsfm on the subspace (yellow
square). This approximation is a convex upper bound for the original loss. It would still reﬂect the degree of projection ambiguity, and
push the subspace (lines) torwards ϕnrsfm.
tion network. However, as shown in Fig. 1, we ﬁnd that
NRSfM tends to make wrong estimation due to strong am-
biguity in 2D projections. Using those as hard target for re-
gression would bottleneck the accuracy of learned pose es-
timation network. We propose a better approach - we want
to establish a direct relation between depth prediction and
the cost function (Eq. 2) we used in NRSfM, which is the
better metric to evaluate the quality of predicted 3D shapes.
In this way, we can avoid confusing our student network
with wrong labels, and allow them to implicitly associate
image features to disambiguate difﬁcult poses for NRSfM.
This intuition is inline with other geometric self-supervised
learning, e.g. self-supervised depth estimation [46, 14, 42],
in which photometric loss is used to train a depth estimation
network.
Outline: The core problem is how to design a loss function
which properly evaluates the quality of a depth hypothesis
produced by the pose estimator. To derive our loss function,
We ﬁrst show that a depth hypothesis associates with a sub-
space of codes (see Section 4.1). We then advocate that the
loss should be the minimum cost value of codes in the sub-
space (see Section 4.2). Finally, we derive a convex upper
bound for the loss, which is computationally trackable for
SGD training (see Section 4.3). A 2D illustration is given
in Fig. 2 to help decipher the text.
4.1. Depth hypothesis deﬁnes a subspace of codes
From NRSfM, we get the dictionary D, and per example
camera matrix M(i). We ﬁnd that the camera matrices from
our modiﬁed Deep-NRSfM are accurate, thus we treat them
as oracle and ﬁxed in our learning algorithm. With this, we
can simplify our notation by absorbing camera matrix into
dictionary through rotation. Rotation matrix R(i) ∈R3×3
is formed from camera matrix by:
R(i) = [m(i)
1 , m(i)
2 , m(i)
1 × m(i)
2 ],
(5)
where m(i)
1 , m(i)
2 are columns of camera matrix M(i). Then
the dictionary is rotated by multiplying every 3D coordi-
nates inside D with R(i):
B(i) =

[d1
x, d1
y, d1
z]R(i)
. . .
[dP
x , dP
y , dP
z ]R(i)T
(6)
We further split B(i) into two matrices – one matrix takes
all the x, y coordinate elements of B(i), while the other
takes all the rest z coordinate elements.
B(i)
xy =
h
b1
x
(i)
b1
y
(i)
. . .
bP
x
(i)
bP
y
(i)
iT
,
B(i)
z
=
h
b1
z
(i)
. . .
bP
z
(i)
iT
,
(7)
With this, B(i)
xyϕ(i) computes 2D projection of shape recon-
structed by code ϕ(i); and B(i)
z ϕ(i) is reconstructed depth
in the image coordinate.
For a depth hypothesis z′ = fz(I(i); θ) produced by the
pose estimation network, codes giving depth reconstruction
equal to z′ forms a subspace:
S(i)(z′) = {ϕ : B(i)
z ϕ = z′}.
(8)
The subspace is not empty assuming that dictionary is over-
complete. In Fig. 2, the subspaces are visualized as orange
lines in 2D.
746

4.2. Loss = minimum cost on subspace
The quality of a depth hypothesis z′ could be represented
by the best code inside its subspace. As in NRSfM, the qual-
ity of a code is measured by the cost function = reprojection
error + some regularizer, i.e.:
C(i)(ϕ) = ∥B(i)
xyϕ −w(i)∥+ h(ϕ),
(9)
where w(i) is the vectorization of W(i). To keep formu-
lation general, we don’t specify the type of norm and reg-
ularizer here. Thereby we have the following deﬁnition of
quality function for z′, which we use as the loss function for
knowledge distilling:
L(i)(z′) =
min
ϕ∈S(i)(z′) C(i)(ϕ).
(10)
This computes the minimum cost value of codes inside the
subspace deﬁned by the depth hypothesis z′.
To evaluate this loss function, we need to ﬁrst solve for
the minima ϕ∗of the constrained convex optimization prob-
lem in Eq. 10 (red cross in Fig. 2). Suppose we can express
ϕ∗as a differentiable function of z′, i.e. ϕ∗= q(i)(z′),
Eq. 10 becomes:
L(i)(z′) = ∥B(i)
xyq(i)(z′) −w(i)∥+ h(q(i)(z′)).
(11)
This loss is explicitly a function of z′, and thus allows the
gradients to be propagated to the pose estimation network.
As a side note, suppose the pose network has unlimited
capacity, in other words, able to overﬁt any depth values,
then the end result of minimizing this loss function would
be a network predicting the same depth as the NRSfM al-
gorithm (illustrated in Fig. 2(a)). We argue that this would
not be the case in practice, since convolution networks con-
strained by their structure, is equivalent to have a deep im-
age prior [39] imposed on their output. This image prior
provides extra constraint to disambiguate confusing 2D pro-
jections, thus is the key source for our improvement over the
NRSfM teacher.
4.3. Convex upper bound of Eq. 11
Using Eq. 11 requires to form the (sub)differentiable
function q(i)(z′) which produces the solution to the con-
strained optimization problem in Eq. 10. However, solv-
ing this constrained optimization problem requires iterative
numerical method due to the existence of regularizer. As
a result, it’s computationally intractable to solve it exactly
per SGD iteration during training. Therefore we derive an
approximate solution as follow:
Suppose ϕ(i)
nrsfm is the solution we get from NRSfM, and
it approximates the minima of the optimization problem in
Eq. 10 without the subspace constraint, then an approximate
solution for the constrained problem could be the projection
of ϕ(i)
nrsfm onto the subspace S(i)(z′):
˜ϕ(i)(z′) = arg
min
ϕ∈S(i)(z′)
1
2∥ϕ −ϕ(i)
nrsfm∥2
2
(12)
The closed form solution to Eq. 12 is:
˜ϕ(i)(z′) = ϕ(i)
nrsfm + (B(i)
z )†(z′ −B(i)
z ϕ(i)
nrsfm),
(13)
where (B(i)
z )† = B(i)
z
T (B(i)
z B(i)
z
T )−1 is the right inverse
of B(i)
z . Eq. 13 is implemented as a differentiable operator
thanks to modern deep learning library.
Substitute the exact solution q(i)(z′) in Eq. 11 by the
approximate solution ˜ϕ(i)(z′) gives a convex upper bound
of Eq. 11:
˜L(i)(z′) = ∥B(i)
xy ˜ϕ(i)(z′) −w(i)∥+ h(˜ϕ(i)(z′))
(14)
In our experiment, we ﬁnd that using this convex upper
bound as training loss, is sufﬁcient to give lower error on the
training set compared to our already strong NRSfM base-
line.
4.4. Learning the 3D pose estimator
We use the state-of-the-art integral regression net-
work [36] as our student pose estimator. The network di-
rectly predicts 3D coordinates of landmarks in the image
coordinate. During training, the (x, y) coordinate is directly
supervised by 2D landmark annotations; while z coordinate
is supervised by our knowledge distilling loss (Eq. 14). The
proposed learning objective is:
min
θ
X
i
∥fxy(I(i); θ) −w(i)∥1 + ˜L(i)(fz(I(i); θ)), (15)
where fxy,fz denote the output of the network at (x, y) and
z coordinates; and θ refers to the network weights. For
the knowledge distilling loss ˜L, we use L2 norm for the
reprojection error, and L1 norm for the regularizer in our
experiment. The regularizer is weighted by an empirically
found coefﬁcient, which is 0.3 in our experiment.
5. Experiment
5.1. Implementation details
Data preprocessing: We assume no knowledge of 3D label
in both training and testing. We crop the image according
to the 2D human bounding box, and then resize and pad
such that it is 256x256 resolution. The 2D points are then
represented by the patch coordinate.
In evaluation, we
follow the same procedure as in [36], which aligns the scale
of the prediction by average bone length before computing
the metrics.
747

Consensus
Deep-NRSfM
Weaksup-bs
Ours
GT.
Figure 3. Visual comparison of NRSfM methods versus methods which include image as extra constraint (i.e. our weakly supervised
baseline and our knowledge distilling method) on the training set. Our method shows signiﬁcant improvement over its teacher, i.e. deep-
NRSfM. Skeletons are rendered from side view for better visualization of the difference in depth reconstruction. We use red and magenta
to color left leg and arm, while blue and dodgerblue are used to color right leg and arm.
P-MPJPE
MPJPE
depth
error
Ranklet [11]
281.1
-
-
Sparse [20]
217.4
-
-
SPM(2k) [9]
209.5
-
-
SFC [22]
167.1
218.0
135.6
KSTA(5k) [16]
123.6
-
-
RIKS(5k) [17]
103.9
-
-
Consensus [25]
79.6
120.1
111.5
Deep-NRSfM∗[21]
73.2
101.6
76.5
Weaksup-bs
61.2
86.2
75.3
Ours
56.4
80.9
71.2
Table 1. Compare with NRSfM methods on the training set of
H3.6M ECCV18 challenge dataset. KSTA, RIKS are evaluated
on a subset of 5k images, and SPM is evaluated on 2k images.
∗Our implementation of Deep-NRSfM has signiﬁcant difference
compared to the original paper.
3D pose estimation network:
We select the integral
regression network [36] due to its state-of-the-art per-
formance in human pose estimation.
Throughout our
experiment, we use ResNet50 as the backbone for the
regression network, and the input image resolution is
2D
3D
MV
P-MPJPE
MPJPE
Sun et al. [36]
-
-
-
-
86.4
Rhodin et al. [32]
✓
✓
98.2
131.7
Tung et al. [38]
✓
✓
✓
98.4
-
3Dinterp. [43]
✓
✓
98.4
-
AIGN [13]
✓
✓
97.2
-
Tome et al. [37]
✓
✓
-
88.4
Drover et al. [12]
✓
✓
64.6
-
Weaksup-bs
✓
67.3
95.0
Ours
✓
62.8
86.4
+ MPII
✓
57.5
83.0
Table 2. Compare with weakly supervised methods on H3.6M val-
idation set. Supervision source used by each method is marked:
‘2D’ refers to 2D landmark annotation; ‘3D’ represents any train-
ing source with 3D annotation, including synthetic 3D dataset, ex-
ternal human 3D model, etc.; ‘MV’ is the abbreviation for multi-
view.
set as 256 × 256. Using deeper backbone network (e.g.
ResNet152) and higher image resolution would improve
result, as already shown in [36]. We choose this cheaper
setting for a fairer comparison with other weakly super-
vised methods which use ResNet50.
748

Direct.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD
Smoke
Wait
Walk
WalkD
WalkP
3Dinterp. [43]
78.6
90.8
92.5
89.4
108.9
112.4
77.1
106.7
127.4
139.0
103.4
91.4
79.1
-
-
AIGN [13]
77.6
91.4
89.9
88.0
107.3
110.1
75.9
107.5
124.2
137.8
102.2
90.3
78.6
-
-
Drover et al. [12]
60.2
60.7
59.2
65.1
65.5
63.8
59.4
59.4
69.1
88.0
64.8
60.8
64.9
63.9
65.2
Weaksup-bs
58.8
62.4
56.7
59.8
68.6
60.8
59.7
81.0
93.4
68.5
75.8
65.9
61.5
67.6
65.0
Ours
54.7
57.7
54.8
55.8
61.6
56.3
52.7
73.7
95.5
62.3
68.5
60.8
55.5
64.0
58.0
+MPII
50.3
48.9
52.7
53.9
59.9
50.7
48.3
70.9
82.6
58.0
65.3
54.7
50.8
57.7
55.6
Table 3. Per action PA-MPJPE reported on H3.6M validation set. Our approach performs favorably compared to other weakly supervised
methods.
During training, we follow most of the settings in [36],
i.e. the base learning rate is 1e-3, and it drops to 1e-5 when
the loss on the validation set saturates. Limited by our com-
putational resources, we use a smaller batch size of 32.
Deep-NRSfM: We use dictionaries with 6 levels. The size
for the dictionaries from lower level to higher is: 256, 128,
64, 32, 16, 8. When learning the dictionaries, the sparsity
weight (λ in Eq. 2) is selected through cross validation and
set as 0.01. For more details of our modiﬁed version of
Deep-NRSfM, we refer the reader to our supplementary ma-
terial.
5.2. Experiment setup
Dataset: We validate our method on Human3.6M dataset
(H3.6M) [18], which is the major dataset used in current 3D
human pose estimation research. Despite our experiment is
focused on human pose estimation, we’d like to emphasize
that the proposed method is a general algorithm. Unlike
other weakly supervised methods which are deeply coupled
with external 3D human model, our method doesn’t require
any target speciﬁc prior knowledge, thus should be applica-
ble to other type of objects without restriction.
H3.6M includes sequences of 11 actors performing 15
type of actions captured from 4 camera locations. Footage
of 7 out of 11 actors are released for training/validation. We
follow the experiment convention conducted by prior pa-
pers: 5 subjects (S1, S5, S6, S7, S8) are used as training
set, and 2 subjects (S9, S11) for testing. Although H3.6M
dataset comes with 3D annotation, we use only 2D annota-
tion during training, and 3D labels are kept for validation.
Strategies to sample frames from the training footage
can have a direct impact on validation accuracy. For re-
producibility, we use the subset (35k+ images) selected by
H3.6M ECCV18 Challenge for training. We augment the
training set through random image warping and perturba-
tion as in [36].
Evaluation metric: We follow the two common evaluation
protocols used in literature, and report both of them.
• MPJPE: mean per joint positioning error measures
the mean euclidean distance between the reconstructed
and ground truth joints after shifting them to have the
same root joint coordinate.
• PA-MPJPE: Align the reconstructed joints to the
ground truth through rigid transformation before eval-
uating MPJPE. This metric is more often used in
NRSfM to measure the correctness of the recon-
structed shape.
In addition, we also report ‘depth error’ which measures the
mean difference along z-axis. This is the most important
metric to validate our method, because the core problem of
weakly supervised learning is how to recover depth without
annotation.
Weakly supervised learning baseline: As previously men-
tioned, a simple weakly supervised learning baseline is us-
ing the depth output from our Deep-NRSfM method as
training labels. We use this baseline (refer as “Weaksup-
bs”) to validate the contribution of our novel knowledge
distilling loss. To train the pose estimation network, we
employ L1 regression loss which has been proven effective
in [36].
Weighting value for the L1 regularizer: We study the
effect of different weighting values for the L1 regularizer
in the propoesed knowledge distilling loss (Eq. 14).
As
shown in Table 4, under a reasonable range (0.1-0.5) of the
weights, our method consistently outperforms the baseline.
L1weight
0.01
0.1
0.3
0.5
Weaksup-bs
depth error (mm)
79.0
74.6
73.1
76.7
78.0
PA-MPJPE (mm)
73.0
73.6
70.5
71.0
75.8
Table 4. Comparing different weighting values for the L1 regular-
izer in Eq. 14. Numbers reported on the validation set of H3.6M
ECCV18 challenge.
Using extra data from MPII:
Prior works [47] has
shown that including external 2D data such as MPII [3]
as training source can improve generalization ability of the
learned 3D pose estimator. Thus, we also report result of
our method trained with H3.6M+MPII. Due to our cur-
rent method does not handle missing joints, we apply our
proposed knowledge distilling loss only to those MPII im-
ages with complete 2D skeleton annotation; for images with
occluded/out-of-view joints, we only use 2D regression loss
as in [36].
5.3. Compare with NRSfM methods
We compare with 7 state-of-the-art NRSfM methods on
our training set (35k+ images from H3.6M ECCV18 Chal-
749

Ours
GT.
Figure 4. Qualitative results of ours on H3.6M validation set. The right part shows some of our failure cases. Our method may fail under
severe occlusion and rare body poses.
lenge). We ﬁnd this dataset is challenging to the compared
methods due to: 1) large variation in camera positions; 2)
difﬁcult poses such as sitting and prone occupy a signif-
icant portion of the dataset; 3) variation in scale is large,
due to the fact that without the knowledge of 3D, we cannot
normalize 2D projections by distance or calculating bone
length. The best we can do is to normalize 2D points by the
size of 2D bounding box. This leads to certain pose e.g. sit-
ting appears larger compared to others after normalization;
4) some of the methods fails to cope with a large number of
samples (e.g. >5k). For those methods, we report result on
the largest subset they can handle. We also try to compare
with the recently proposed MUS [1], but their implemen-
tation fails to handle H3.6M dataset with large number of
frames.
Despite of these difﬁculties, our implementation of
Deep-NRSfM outperforms all of them. As shown in Ta-
ble. 1, it reduces depth error by more than 33% compared to
the second best. This means that switching to other NRSfM
method is bound to inferior result of training a 3D pose es-
timator.
More interestingly, although our weakly supervised
learning baseline (Weaksup-bs) is trained to reconstruct the
same depth value produced by deep NRSfM, it actually gets
slightly lower depth error compared to its regression target.
This indicates that the deep image prior is taking effect, but
still restricted by the noisy labels from Deep-NRSfM.
Finally, the pose estimation network learned by our
knowledge distilling loss reduces the depth error from
Deep-NRSfM’s 76.5mm to 71.2mm. As shown in Fig. 3
and 1, this 5.3mm average difference includes a huge im-
provement in cases such as identifying if a leg is stretching
towards or away from the camera.
5.4. Compare with weakly supervised methods
We compare with other weakly supervised 3D pose
learning methods on the H3.6M validation set. In Table. 2,
we ﬁrst list the performance of Integral regression net-
work by Sun et al. [36] as a supervised learning baseline.
We copied its MPJPE (corresponding to ResNet50 with
256 × 256 input size and I1 loss) from their paper. Since in
our experiment, we’re using exactly the same pose estima-
tion network architecture, this serves as the upper bound of
accuracy, which a weakly supervised learning method can
achieve.
Next, we list results from 7 weakly supervised methods,
and the type of their training source is marked. ‘2D’ refers
to 2D landmark annotation; ‘3D’ represents any external 3D
training source, including 3D human models, unpaired 3D
skeleton dataset, synthetic dataset with 3D annotations, etc.;
MV is the abbreviation for multi-view footage. We ﬁnd that
our method outperforms all the compared methods, while
using the least amount of supervision. We also experiment
with including MPII as extra training source, which leads to
more error reduction. Fig. 4 shows some qualitative results
of our method on the validation set. For per action error
break down, we list PA-MPJPE of 13 different actions in
Table 3.
6. Conlusion
In this paper, we presented a weakly supervised 3D
pose learning algorithm requires zero 3D annotation. We
proposed a novel loss to distill knowledge from a general
type of NRSfM method based on dictionary learning. We
also established a strong NRSfM baseline on a challeng-
ing dataset, beating all the state-of-the-arts. Despite its cur-
rent sucess, the limitations of our method are: 1) we re-
quire weak perspective projection, thus objects with strong
perspective change is not ideal for the proposed method;
2) we do not model missing labels yet, thus another iter-
ation is needed to extend the method to datasets with lots
of occluded/out-of-view objects. We leave these for future
work.
750

References
[1] Antonio Agudo, Melcior Pijoan, and Francesc Moreno-
Noguer.
Image collection pop-up: 3d reconstruction and
clustering of rigid and non-rigid categories.
In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 2607–2615, 2018. 2, 8
[2] Ijaz Akhter, Yaser Sheikh, Sohaib Khan, and Takeo Kanade.
Trajectory space: A dual representation for nonrigid struc-
ture from motion.
Pattern Analysis and Machine Intelli-
gence, IEEE Transactions on, 33(7):1442–1456, 2011. 2
[3] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and
Bernt Schiele. 2d human pose estimation: New benchmark
and state of the art analysis. In IEEE Conference on Com-
puter Vision and Pattern Recognition (CVPR), June 2014. 7
[4] Amir Beck and Marc Teboulle. A fast iterative shrinkage-
thresholding algorithm with application to wavelet-based im-
age deblurring. 2009. 3
[5] Christoph Bregler, Aaron Hertzmann, and Henning Bier-
mann. Recovering non-rigid 3d shape from image streams.
In Computer Vision and Pattern Recognition, 2000. Proceed-
ings. IEEE Conference on, volume 2, pages 690–696. IEEE,
2000. 2
[6] Ching-Hang Chen and Deva Ramanan. 3d human pose es-
timation = 2d pose estimation + matching.
In The IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR), July 2017. 2
[7] Wenzheng Chen, Huan Wang, Yangyan Li, Hao Su, Zhen-
hua Wang, Changhe Tu, Dani Lischinski, Daniel Cohen-
Or, and Baoquan Chen.
Synthesizing training images for
boosting human 3d pose estimation. In 2016 Fourth Inter-
national Conference on 3D Vision (3DV), pages 479–488.
IEEE, 2016. 2
[8] Rishabh Dabral, Anurag Mundhada, Uday Kusupati, Safeer
Afaque, Abhishek Sharma, and Arjun Jain. Learning 3d hu-
man pose from structure and motion. In The European Con-
ference on Computer Vision (ECCV), September 2018. 2
[9] Yuchao Dai, Hongdong Li, and Mingyi He. A simple prior-
free method for non-rigid structure-from-motion factoriza-
tion. International Journal of Computer Vision, 107(2):101–
122, 2014. 2, 3, 6
[10] Ingrid Daubechies, Michel Defrise, and Christine De Mol.
An iterative thresholding algorithm for linear inverse prob-
lems with a sparsity constraint.
Communications on
Pure and Applied Mathematics: A Journal Issued by the
Courant Institute of Mathematical Sciences, 57(11):1413–
1457, 2004. 3
[11] Alessio Del Bue, Fabrizio Smeraldi, and Lourdes Agapito.
Non-rigid structure from motion using ranklet-based track-
ing and non-linear optimization. Image and Vision Comput-
ing, 25(3):297–310, 2007. 6
[12] Dylan Drover,
Rohith MV, Ching-Hang Chen,
Amit
Agrawal, Ambrish Tyagi, and Cong Phuoc Huynh. Can 3d
pose be learned from 2d projections alone? In Proceedings
of the European Conference on Computer Vision (ECCV),
pages 0–0, 2018. 1, 2, 6, 7
[13] Hsiao-Yu Fish Tung, Adam W. Harley, William Seto, and
Katerina Fragkiadaki.
Adversarial inverse graphics net-
works: Learning 2d-to-3d lifting and image-to-image trans-
lation from unpaired supervision. In The IEEE International
Conference on Computer Vision (ICCV), Oct 2017. 2, 6, 7
[14] Cl´ement Godard, Oisin Mac Aodha, and Gabriel J Bros-
tow.
Unsupervised monocular depth estimation with left-
right consistency. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages 270–279,
2017. 4
[15] Paulo FU Gotardo and Aleix M Martinez.
Computing
smooth time trajectories for camera and deformable shape in
structure from motion with occlusion. Pattern Analysis and
Machine Intelligence, IEEE Transactions on, 33(10):2051–
2065, 2011. 2
[16] Paulo FU Gotardo and Aleix M Martinez. Kernel non-rigid
structure from motion. In Computer Vision (ICCV), 2011
IEEE International Conference on, pages 802–809. IEEE,
2011. 6
[17] Onur C Hamsici, Paulo FU Gotardo, and Aleix M Martinez.
Learning spatially-smooth mappings in non-rigid structure
from motion. In European Conference on Computer Vision,
pages 260–273. Springer, 2012. 6
[18] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian
Sminchisescu. Human3.6m: Large scale datasets and predic-
tive methods for 3d human sensing in natural environments.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 36(7):1325–1339, jul 2014. 1, 2, 7
[19] Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe,
Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser
Sheikh. Panoptic studio: A massively multiview system for
social motion capture.
In Proceedings of the IEEE Inter-
national Conference on Computer Vision, pages 3334–3342,
2015. 2
[20] Chen Kong and Simon Lucey. Prior-less compressible struc-
ture from motion. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 4123–
4131, 2016. 2, 3, 6
[21] Chen Kong and Simon Lucey. Deep interpretable non-rigid
structure from motion.
arXiv preprint arXiv:1902.10840,
2019. 1, 2, 3, 6
[22] Chen Kong, Rui Zhu, Hamed Kiani, and Simon Lucey.
Structure from category: a generic and prior-less approach.
International Conference on 3DVision (3DV), 2016. 2, 6
[23] Suryansh Kumar, Anoop Cherian, Yuchao Dai, and Hong-
dong Li. Scalable dense non-rigid structure-from-motion: A
grassmannian perspective. arXiv preprint arXiv:1803.00233,
2018. 2
[24] Suryansh Kumar, Yuchao Dai, and Hongdong Li.
Multi-
body non-rigid structure-from-motion. In 3D Vision (3DV),
2016 Fourth International Conference on, pages 148–156.
IEEE, 2016. 2
[25] Minsik Lee, Jungchan Cho, and Songhwai Oh. Consensus of
non-rigid reconstructions. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pages
4670–4678, 2016. 6
[26] Julieta Martinez, Rayat Hossain, Javier Romero, and James J
Little. A simple yet effective baseline for 3d human pose es-
timation. In Proceedings of the IEEE International Confer-
ence on Computer Vision, pages 2640–2649, 2017. 2
751

[27] Vardan Papyan, Yaniv Romano, and Michael Elad. Convo-
lutional neural networks analyzed via convolutional sparse
coding.
The Journal of Machine Learning Research,
18(1):2887–2938, 2017. 3
[28] Georgios Pavla