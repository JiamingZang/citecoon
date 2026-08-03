# Human Pose Regression with Residual Log-likelihood Estimation

> 2021 · id: W3184095310 · arXiv: 2107.11291 · pdf: https://arxiv.org/pdf/2107.11291 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Human Pose Regression with Residual Log-likelihood Estimation
Jiefeng Li1
Siyuan Bian1
Ailing Zeng2
Can Wang3
Bo Pang1
Wentao Liu3
Cewu Lu1
1Shanghai Jiao Tong University
2The Chinese University of Hong Kong
3SenseTime Research
Abstract
Heatmap-based methods dominate in the ﬁeld of hu-
man pose estimation by modelling the output distribu-
tion through likelihood heatmaps. In contrast, regression-
based methods are more efﬁcient but suffer from inferior
performance.
In this work, we explore maximum likeli-
hood estimation (MLE) to develop an efﬁcient and effec-
tive regression-based methods.
From the perspective of
MLE, adopting different regression losses is making differ-
ent assumptions about the output density function. A den-
sity function closer to the true distribution leads to a bet-
ter regression performance. In light of this, we propose
a novel regression paradigm with Residual Log-likelihood
Estimation (RLE) to capture the underlying output distri-
bution. Concretely, RLE learns the change of the distri-
bution instead of the unreferenced underlying distribution
to facilitate the training process. With the proposed repa-
rameterization design, our method is compatible with off-
the-shelf ﬂow models. The proposed method is effective,
efﬁcient and ﬂexible. We show its potential in various hu-
man pose estimation tasks with comprehensive experiments.
Compared to the conventional regression paradigm, regres-
sion with RLE bring 12.4 mAP improvement on MSCOCO
without any test-time overhead.
Moreover, for the ﬁrst
time, especially on multi-person pose estimation, our re-
gression method is superior to the heatmap-based methods.
Our code is available at https://github.com/Jeff-sjtu/res-
loglikelihood-regression.
1. Introduction
Human pose estimation has been extensively studied
in the area of computer vision [23, 24, 1, 32, 21].
Re-
cently, with deep convolutional neural networks, signiﬁcant
progress has been achieved. Existing methods can be di-
vided into two categories: heatmap-based [60, 59, 65, 4,
67, 57, 49, 55] and regression-based [61, 5, 56, 73, 45, 64].
Heatmap-based methods are dominant in the ﬁeld of hu-
man pose estimation. These methods generate a likelihood
heatmap for each joint and locate the joint as the point
(a) Heatmap-based
(c) Regression with RLE
CNN
ℓ1
ℓ2
CNN
ˆµ
or
(b) Standard Regression Paradigm
RLE
CNN
−log PΘ(x|I)

x=µg
Figure 1: Illustrasion of (a) heatmap-based method, (b)
standard regression paradigm, and (c) regression with the
proposed RLE.
with the argmax [59, 67, 49] or soft-argmax [43, 34, 57]
operations. Despite the excellent performance, heatmap-
based methods suffer from high computation and storage
demands. Expanding the heatmap to 3D or 4D (spatial +
temporal) will be costly. Additionally, it is hard to deploy
heatmap in modern one-stage methods.
Regression-based methods directly map the input to the
output joints coordinates, which is ﬂexible and efﬁcient for
various human pose estimation tasks and real-time applica-
tions, especially on edge devices. A standard heatmap head
(3 deconv layers) costs 1.4× FLOPs of the ResNet-50 back-
bone, while the regression head costs only 1/20000 FLOPs
of the same backbone. Nevertheless, regression suffer from
inferior performance. In challenging cases like occlusions,
motion blur, and truncations, the ground-truth labels are in-
herently ambiguous. Heatmap-based methods are robust to
these ambiguities by leveraging the likelihood heatmap. But
current regression methods are vulnerable to these noisy la-
bels.
In this work, we facilitate human pose regression by ex-
ploring maximum likelihood estimation (MLE) to model
the output distribution. From the perspective of MLE, stan-
dard Euclidean distance loss (ℓ1 or ℓ2) can be viewed as a
particular assumption that the output conforms to a distribu-
tion family (Laplace or Gaussian distribution) with constant
1
arXiv:2107.11291v3  [cs.CV]  31 Jul 2021

variance. Intuitively, the regression performance can be im-
proved if we construct the likelihood function with the true
underlying distribution instead of the inappropriate hypoth-
esis.
To this end, we propose a novel and effective regres-
sion paradigm, named Residual Log-likelihood Estimation
(RLE), that leverages normalizing ﬂows to estimate the
underlying distribution and boosts human pose regression.
Given a tractable preset assumption of the likelihood func-
tion, RLE estimates the residual log-likelihood, i.e. the
change of the distribution. It is easier to be optimized com-
pared to the original unreferenced underlying distribution.
Besides, we design a reparameterization strategy for the
ﬂow model to learn the intrinsic characteristics of the un-
derlying distribution. This strategy makes our regression
framework feasible and allows us to utilize the off-the-shelf
ﬂow model to approximate the distribution without a so-
phisticated network architecture.
During training, the regression model and the RLE mod-
ule can be optimized simultaneously. Since the form of the
underlying distribution is unknown, the RLE module is also
trained via the maximum likelihood estimation process. Be-
sides, the RLE module does not participate in the inference
phase. In other words, the proposed method can bring sig-
niﬁcant improvement to the regression model without any
test-time overhead.
The proposed regression framework is general. It can be
applied to various human pose estimation algorithms (e.g.
two-stage approaches [48, 15, 13, 67, 55], one-stage ap-
proaches [73, 45, 64]) and various tasks (e.g. single and
multi-person 2D/3D pose estimation [1, 32, 21, 38, 23, 24]).
We benchmark the proposed method on three pose estima-
tion datasets, including MPII [1], MSCOCO [32] and Hu-
man3.6M [21]. With a simple yet effective architecture,
RLE boosts the conventional regression method by 12.4
mAP and achieves superior performance to the heatmap-
based methods. Moreover, it is more computation and stor-
age efﬁcient than heatmap-based methods.
Speciﬁcally,
on the MSCOCO dataset [32], our regression-based model
with ResNet-50 [16] backbone achieves 71.3 mAP with
4.0 GFLOPs, compared to 71.0 mAP with 9.7 GFLOPs of
heatmap-based SimplePose [67]. We hope our method will
inspire the ﬁeld to rethink the potential of regression-based
methods.
The contributions of our approach can be summarized as
follows:
• We propose a novel and effective regression paradigm
with the reparameterization design and Residual Log-
likelihood Estimation (RLE). The proposed method
boosts human pose regression without any test-time
overhead.
• For the ﬁrst time, regression-based methods achieve
superior performance to the heatmap-based methods,
and it is more computation and storage efﬁcient.
• We show the potential of the proposed paradigm by
applying it to various human pose estimation methods.
Considerable improvements are observed in all these
methods.
2. Related Work
Heatmap-based Pose Estimation.
The idea of utilizing
likelihood heatmaps to represent human joint locations is
proposed by Tompson et al. [60].
Since then, heatmap-
based approaches dominate in the ﬁeld of 2D human pose
estimation.
Pioneer works [60, 59, 65, 42] design pow-
erful CNN models to estimate heatmaps for single-person
pose estimation. Many works [48, 15, 13, 67, 30, 55] ex-
tend this idea to multi-person pose estimation following the
top-down framework, i.e. detection and single-person pose
estimation. In the bottom-up framework [51, 20, 22, 4, 41,
47, 8], multiple body joints are retrieved from the heatmaps
and grouped into different human poses. Pavlakos et al. [49]
ﬁrst extend the heatmap to 3D space. The 3D heatmap rep-
resentation is followed by several works [57, 39, 7, 72, 62,
31]. Sun et al. [57] leverage the soft-argmax operation to re-
trieve joint locations from heatmaps in a differentiable man-
ner, which allows end-to-end training. It prevents quantiza-
tion error, but the model is still required to generate high-
resolution features and heatmaps.
Regression-based Pose Estimation.
In the context of hu-
man pose estimation, only a few works are regression-
based. Toshev et al. [61] ﬁrst leverage the convolutional
network for human pose estimation. Carreira et al. [5] pro-
pose an Iterative Error Feedback (IEF) network to improve
the performance of the regression model. Zhou et al. [73]
and Tian et al. [58] propose direct pose regression in the
one-stage object detection framework. Nie et al. [45] factor-
ize the long-range displacement into accumulative shorter
ones. However, it is vulnerable to occlusions. Wei et al. [64]
regress the displacement w.r.t. the pre-deﬁned pose anchors.
In 3D pose estimation, Sun et al. [56] propose composi-
tional pose regression to learn the internal structures of 3D
human pose. Rogez et al. [53, 54] classify the human pose
into a set of K anchor-poses and a regression module is pro-
posed to reﬁne the anchor to the ﬁnal prediction. Two-stage
methods [36, 14, 50, 71, 63, 10, 33, 70] lift the 2D poses to
3D space by regression. But the 2D poses are still predicted
by the heatmap-based 2D pose estimator. Despite lots of
progress that have been made by previous works, there is
still a huge performance gap between the pure regression-
based approaches and the heatmap-based approaches.
In this work, for the ﬁrst time, we improve the perfor-
mance of the regression-based approach to a comparable
2

level of the heatmap-based approaches. Our method is ﬂex-
ible and can be applied to various human pose estimation
algorithms.
Normalizing Flow in Human Pose Estimation.
Some
recent works leverage normalizing ﬂows to build priors in
3D human pose estimation. Xu et al. [68] propose new 3D
human shape and articulated pose models with the kine-
matic prior based on normalizing ﬂows. Zanﬁr et al. [69]
use normalizing ﬂows to build a prior on SMPL joint an-
gles for their weakly-supervised method. Biggs et al. [3]
learn a pose prior by normalizing ﬂows to sample the best
output from the ambiguous image. Different from previous
methods, we leverage normalizing ﬂows to estimate the un-
derlying output distribution.
Adaptive Loss Function.
In our method, the output dis-
tribution is learnable, which resulting in a learnable loss
function. There have been several works towards adaptive
loss functions. Imani et al. [19] propose histogram loss,
which use histogram (i.e. heatmap) to represent the output
distribution. Some works deﬁne a superset of loss functions
and change the loss by tuning the parameters of the func-
tion. Wu et al. [66] using a teacher model to dynamically
change the loss function of the student model. Barron [2]
presents a generalization of common loss functions, which
automatically adapts itself during training. Different from
previous methods, we do not set the form of the distribu-
tion family in advance. The loss function can learn to be
arbitrary forms within the maximum likelihood estimation
framework.
3. Method
In this work, we aim at improving the performance of
the regression-based method to a competitive level of the
heatmap-based method. Compared with the heatmap-based
method, regression-based method has lots of merits: i) It
gets rid of the high-resolution heatmaps and has low com-
putation and storage complexity.
ii) It has a continuous
output and does not suffer from the quantization problem.
iii) It can be extended to a wide variety of scenarios (e.g.
one-stage methods, video-based methods, 3D scenes) at a
minimal cost. However, existing regression-based methods
suffer from poor performance, which is fatal and restricts its
wide usage.
In this section, before introducing our solution, we ﬁrst
review the general formulation of regression from the per-
spective of maximum likelihood estimation in §3.1. Then,
in §3.2, we present the Residual Log-likelihood Estimation
(RLE), an approach that leverages normalizing ﬂows to cap-
ture the underlying residual log-likelihood function and fa-
cilitate human pose regression. Finally, the necessary im-
plementation details are provided in §3.3.
3.1. General Formulation of Regression
The standard regression paradigm is to apply ℓ1 or ℓ2
loss to the regressed output ˆµ. Loss functions are empir-
ically chosen for different tasks. Here, we review the re-
gression problem from the perspective of maximum likeli-
hood estimation (MLE). Given an input image I, the regres-
sion model predicts a distribution PΘ(x|I) that indicates
the probability of the ground truth appearing in the location
x, where Θ denotes the learnable model parameters. Due
to the inherent ambiguities in the labels, the labelled loca-
tion µg can be viewed as an observation sampled near the
ground truth by the human annotator. The learning process
is to optimize the model parameters Θ that makes the ob-
served label µg most probable. Therefore, the loss function
of this maximum likelihood estimation (MLE) process is
deﬁned as:
Lmle = −log PΘ(x|I)

x=µg
.
(1)
In this formulation, different regression losses are es-
sentially different hypotheses of the output probability dis-
tribution.
For example, in some works of object detec-
tion [18, 29, 28] and dense correspondences [40], the den-
sity is assumed to be a Gaussian distribution. The model
needs to predict two values, ˆµ and ˆσ, to construct the den-
sity function PΘ(x|I) =
1
√
2π ˆσe−(x−ˆ
µ)2
2ˆ
σ2 . To maximize the
likelihood of the observed label µg, the loss function be-
comes:
L = −log PΘ(x|I)

x=µg
∝log ˆσ + (µg −ˆµ)2
2ˆσ2
.
(2)
If we assume the density function has a constant variance,
i.e. ˆσ is a constant, the loss degenerates to standard ℓ2 loss:
L = (µg −ˆµ)2. Further, if we assume the density fol-
lows the Laplace distribution with a constant variance, the
loss function becomes the standard ℓ1 loss. In the inference
phase, the value ˆµ used to control the location of distribu-
tion serves as the regressed output.
From this perspective, the loss function depends on the
shape of the distribution PΘ(x|I). Therefore, a more accu-
rate density function could lead to better results. However,
since the analytical expression of the underlying distribu-
tion is unknown, the model can not simply regress several
values to construct the density function like Eq. 2. To esti-
mate the underlying distribution and facilitate human pose
regression, in the following section, we propose a novel re-
gression paradigm by leveraging normalizing ﬂow.
3.2. Regression with Normalizing Flows
In this subsection, we introduce three variants of the pro-
posed paradigm that utilize normalizing ﬂows for regression
(see Fig. 2).
3

Basic Design.
The basic design of the proposed regres-
sion paradigm with normalizing ﬂows is illustrated in
Fig. 2(a). Here, normalizing ﬂows [52, 11, 26, 46, 25] learn
to construct a complex distribution by transforming a sim-
ple distribution through an invertible mapping. We consider
the distribution PΘ(z|I) on a random variable z as the ini-
tial density function. It is deﬁned by the output ˆµ and ˆσ
from the regression model Θ. For simplicity, we assume
PΘ(z|I) =
1
√
2π ˆσe−(z−ˆ
µ)2
2ˆ
σ2 , i.e. the Gaussian distribution.
A smooth and invertible mapping fφ : R2 →R2 is chosen
to transform z to x, i.e. x = fφ(z), where φ is the learnable
parameters of the ﬂow model.
The transformed variable x follows another distribution
PΘ,φ(x|I). The probability density function PΘ,φ(x|I) de-
pends on both the regression model Θ and the ﬂow model
fφ, which can be calculated as:
log PΘ,φ(x|I) = log PΘ(z|I) + log
det
∂f −1
φ
∂x
 ,
(3)
where f −1
φ
is the inverse of fφ and z = f −1
φ (x). In this way,
given arbitrary x, the corresponding log-probability can be
estimated through Eq. 3 by reversely computing z. Besides,
PΘ,φ(x|I) is learnable and can ﬁt arbitrary distribution as
long as fφ is complex enough. In practice, we can compose
several simple mappings successively to construct arbitrar-
ily complex functions, i.e. x = fφ(z) = fK◦· · ·◦f2◦f1(z).
The maximum likelihood process is performed on the
learned distribution PΘ,φ(x|I). Hence, the loss function
is formulated as:
Lmle = −log PΘ,φ(x|I)

x=µg
= −log PΘ(f −1
φ (µg) | I) −log
det
∂f −1
φ
∂µg
 .
(4)
Note that the underlying optimal distribution Popt(x|I)
is unknown. The ﬂow model is learned in an unsupervised
manner by maximizing the likelihood of the labelled loca-
tions. For example, for the challenging cases (e.g. occlu-
sions) with larger deviations in the labels from human an-
notators, the predicted distribution should have a large vari-
ance to maximize the log-probability.
Reparameterization.
Although the basic design seems
reasonable, it is not feasible in practice. The learning of fφ
relies on the terms log
det
∂f −1
φ
∂µg
 and f −1
φ (µg) in the loss
function (Eq. 4). Therefore, φ will learn to ﬁt the distribu-
tion of µg across all images. Nevertheless, the distribution
that we want to learn is about how the output deviates from
the ground truth conditioning on the input image, not the
distribution of the ground truth itself across all images.
ˆµ
ˆσ
Regression
Normalizing
Flow
PΘ,φ(x|I)
PΘ(z|I)
Image
x = fφ(z)
(a) Basic Design
×
Normalizing
Flow
PΘ,φ(x|I)
Q(¯x)
Pφ(¯x)
Reparameterization
Residual
Log-likelihood
Gφ(¯x)
ˆµ
ˆσ
Regression
Image
x = ¯x · ˆσ + ˆµ
¯x = fφ(z)
(c) Residual log-likelihood estimation with reparameterization
(b) Direct likelihood estimation with reparameterization
ˆµ
ˆσ
Regression
Reparameterization
Normalizing
Flow
PΘ,φ(x|I)
Pφ(¯x)
N(0, I)
Image
x = ¯x · ˆσ + ˆµ
¯x = fφ(z)
Figure 2: Illustrasion of the proposed regression frame-
works. (a) The basic design. (b) Direct likelihood esti-
mation with reparameterization. (c) Residual log-likelihood
estimation with reparameterization.
Here, to make our regression framework feasible and
compatible with the off-the-shelf ﬂow models, we further
design the regression paradigm with the reparameterization
strategy. The new paradigm is illustrated in Fig. 2(b). We
assume all the underlying distribution share the same den-
sity function family but with different mean and variance
conditioning on the input I.
Firstly, the ﬂow model fφ
is leveraged to map a zero-mean initial distribution ¯z ∼
N(0, I) to a zero-mean deformed distribution ¯x ∼Pφ(¯x).
Then the regression model Θ predicts two values, ˆµ and
ˆσ, to control the position and scale of the distribution.
The ﬁnal distribution PΘ,φ(x|I) is obtained by shifting and
rescaling ¯x to x, where x = ¯x · ˆσ + ˆµ.
Therefore, the loss function with reparameterization can
be written as:
Lmle = −log PΘ,φ(x|I)

x=µg
= −log Pφ(¯µg) −log
det ∂¯µg
∂µg

= −log Pφ(¯µg) + log ˆσ,
(5)
where ¯µg = (µg −ˆµ)/ˆσ, and ∂¯µg/∂µg = 1/ˆσ. With the
reparameterization design, now the ﬂow model can focus on
learning the distribution of ¯µg, which reﬂects the deviation
of the output from the ground truth.
4

Residual Log-likelihood Estimation.
After reparameter-
ization, the regression framework can be trained in an end-
to-end manner. The training of the regressed value ˆµ and
the ﬂow model fφ are coupled together, depending on the
term log Pφ(¯µg) in the loss function (Eq. 5). However, there
are intricate dependencies between these two models. The
training of the regression model entirely relies on the dis-
tribution estimated by the ﬂow model fφ. At the beginning
stage of training, the shape of the distribution is far from
correct, which increases the difﬁculty to train the regression
model and might degrade the model performance.
To facilitate the training process, we develop a gradient
shortcut to reduce the dependence between these two mod-
els. Formally, the distribution estimated by the ﬂow model
Pφ(¯x) is trying to ﬁt the optimal underlying distribution
Popt(¯x), which can be split into three terms:
log Popt(¯x) = log

Q(¯x) · Popt(¯x)
s · Q(¯x) · s

= log Q(¯x) + log Popt(¯x)
s · Q(¯x) + log s,
(6)
where the term Q(¯x) can be a simple distribution, e.g. Gaus-
sian distribution Q(¯x) = N(0, I), the term log Popt(¯x)
s·Q(¯x) is
what we call residual log-likelihood, and the constant s is
to make sure the residual term is a distribution. We assume
that Q(¯x) can roughly match the underlying distribution but
not perfectly.
The residual log-likelihood is to compen-
sate for the difference. Thus, we split the log-probability
of Pφ(¯x) the same way as Eq. 6:
log Pφ(¯x) = log Q(¯x) + log Gφ(¯x) + log s,
(7)
where Gφ(¯x) is the distribution learned by the ﬂow model.
The value of s =
1
R
Gφ(¯x)Q(¯x)d¯x can be approximated by
the Riemann sum. The derivation of s is provided in the
supplemental document.
In this way, Gφ(¯x) will try to ﬁt the underlying resid-
ual likelihood Popt(¯x)
s·Q(¯x) instead of learning the entire distri-
bution. Finally, combining the reparameterization design
(Eq. 5) and residual log-likelihood estimation (Eq. 7), the
total loss function can be deﬁned as:
Lrle = −log PΘ,φ(x|I)

x=µg
= −log Pφ(¯µg) + log ˆσ
= −log Q(¯µg) −log Gφ(¯µg) −log s + log ˆσ.
(8)
This process is illustrated in Fig. 2(c).
During training, the backward propagated gradients from
Q(¯µg) do not depend on the ﬂow model, which accelerates
the training of the regression model. Besides, as the hypoth-
esis of ResNet [16], it is easier to optimize the residual map-
ping than to optimize the original unreferenced mapping.
To the extreme, if the preset approximation Q(¯x) is opti-
mal, it would be easier to push the residual log-probability
to zero than to ﬁt an identity mapping by a stack of invert-
ible mappings in fφ. The effectiveness of the residual log-
likelihood estimation is validated in §4.1.
3.3. Implementation Details
In the training phase, the regression model and the ﬂow
model are simultaneously optimized in an end-to-end man-
ner. We replace the standard regression loss (ℓ1 and ℓ2)
with the proposed residual log-likelihood estimation loss
Lrle. The initial density is set to Laplace distribution by
default. In the testing phase, the predicted mean ˆµ serves
as the regressed output. Therefore, the ﬂow model does not
need to be run during inference. This characteristic makes
the proposed method ﬂexible and easy to apply to various
regression algorithms without any test-time overhead. Be-
sides, the prediction conﬁdence can be obtained from ˆσ:
ˆc = 1 −1
K
K
X
i
ˆσi,
(9)
where ˆσi is the learned deviation of the ith joint, and K
denotes the total number of joints. The deviation ˆσi is pre-
dicted with a sigmoid function. Hence we have ˆσi ∈(0, 1)
and ˆc ∈(0, 1).
Flow Model.
The proposed regression paradigm is agnos-
tic to the ﬂow models. Hence, various off-the-shelf ﬂow
models [52, 11, 26, 46, 25] can be applied. In the experi-
ments, we adopt RealNVP [11] for fast training. We denote
the invertible function with Lfc fully-connected layers with
Nn neurons as Lfc × Nn. We set Lfc = 3 and Nn = 64
by default. The ﬂow model is light-weighted and barely af-
fects the training speed. More detailed descriptions of the
ﬂow model architecture are provided in the supplemental
document (§A).
Tasks.
The proposed regression paradigm is general and
is ready for various human pose estimation tasks. In the ex-
periments, we validate the proposed regression paradigm on
seven different algorithms in ﬁve tasks: single-person 2D
pose estimation, top-down 2D pose estimation, one-stage
2D pose estimation, single-stage 3D pose estimation and
two-stage 3D pose estimation. Detailed training settings are
provided in §4 and §5. The experiments on single-person
2D pose estimation are provided in the supplemental docu-
ment.
4. Experiments on COCO
We ﬁrst evaluate the proposed regression paradigm on a
large-scale in-the-wild 2D human pose benchmark COCO
Keypoint [32].
5

Method
# Params
GFLOPs
AP
AP50
AP75
Direct Regression (with ℓ1)
23.6M
4.0
58.1
82.7
65.0
Regression with DLE
23.6M
4.0
62.7
86.1
70.4
Regression with RLE
23.6M
4.0
70.5
88.5
77.4
*Regression with RLE
23.6M
4.0
71.3
88.9
78.3
Table 1: Comparison with the conventional regression
paradigm. RLE provides signiﬁcant improvements with
the same test-time computational complexity.
Method
AP
AP50
AP75
(a)
SimplePose [67]
71.0
89.3
79.0
Integral Pose [57]
63.0
85.6
70.0
*Regression with RLE
71.3
88.9
78.3
(b)
HRNet-W32 [55]
74.1
90.0
81.5
HRNet-W32 + RLE (Regression)
74.3
89.7
80.8
(c)
Mask R-CNN [15]
66.0
86.9
71.5
Mask R-CNN + RLE
66.7
86.7
72.6
(d)
PointSet Anchor [64]
67.0
87.3
73.5
PointSet Anchor + RLE
67.4
87.5
73.9
Table 2: Comparison with heatmap-based methods on
COCO validation set.
The proposed paradigm achieves
competitive performance to the heatmap-based methods.
Implementation Details.
We embed RLE into the top-
down approaches and a one-stage approach. For the top-
down approach, we adopt a simple architecture consisting
of a ResNet-50 [16] backbone, followed by an average pool-
ing layer and an FC layer. The FC layer consists of K × 4
neurons, where 4 is for ˆµ and ˆσ, and K = 17 denotes the
number of body keypoints. For human detection, we use the
person detectors provided by SimplePose [67] for both the
validation set and the test-dev set. Data augmentations and
training settings follow previous work [55]. The end-to-end
approach, Mask R-CNN [15], is also adopted for ablation
study. Implementation is based on Detectron2 [12]. The
keypoint head is a stack of 8 convolutional layers, followed
by an average pooling layer and an FC layer. We train for
270, 000 iterations, with 4 images per GPU and 4 GPUs in
total. Other parameters are the same as the original Detec-
tron2.
For the one-stage approach, we adopt the state-of-the-art
method [64]. We replace its 2K-channel regression head
with a 4K-channel head for the prediction of both ˆµ and ˆσ.
Implementation is based on the ofﬁcial code of [64]. The
other training details are the same as them.
4.1. Main Results
Comparison with Conventional Regression.
To study
the effectiveness of the proposed regression paradigm, we
compare it with the conventional direct regression method.
The direct regression model has the same “ResNet-50 +
FC” architecture, and ℓ1 loss is adopted. The experimen-
tal results on COCO validation set are shown in Tab. 1.
Method
Backbone
AP
AP50
AP75
APM
APL
Heatmap-based
CMU-Pose [4]
3CM-3PAF
61.8
84.9
67.5
57.1
68.2
Mask R-CNN [15]
ResNet-50
63.1
87.3
68.7
57.8
71.4
G-RMI [48]
ResNet-101
64.9
85.5
71.3
62.3
70.0
RMPE [13]
PyraNet
72.3
89.2
79.1
68.0
78.6
AE [41]
Hourglass-4
65.5
86.8
72.3
60.6
72.6
PersonLab [47]
ResNet-152
68.7
89.0
75.4
64.1
75.5
CPN [6]
ResNet-Inception
72.1
91.4
80.0
68.7
77.2
SimplePose [67]
ResNet-152
73.7
91.9
81.1
70.3
80.0
Integral [57]
ResNet-101
67.8
88.2
74.8
63.9
74.0
HRNet [55]
HRNet-W48
75.5
92.5
83.3
71.9
81.5
EvoPose [37]
EvoPose2D-L
75.7
91.9
83.1
72.2
81.5
Regression-based
CenterNet [73]
Hourglass-2
63.0
86.8
69.6
58.9
70.4
SPM [45]
Hourglass-8
66.9
88.5
72.9
62.6
73.1
PointSet Anchor [64]
HRNet-W48
68.7
89.9
76.3
64.8
75.3
ResNet + RLE (Ours)
ResNet-152
74.2
91.5
81.9
71.2
79.3
*ResNet + RLE (Ours)
ResNet-152
75.1
91.8
82.8
72.0
80.2
HRNet + RLE (Ours)
HRNet-W48
75.7
92.3
82.9
72.3
81.3
Table 3: Comparison with the SOTA on COCO test-dev.
As shown, the proposed method brings signiﬁcant improve-
ment (12.4 mAP) to the regression-based method. Then we
compare the result with direct likelihood estimation (DLE)
to study the effectiveness of residual log-likelihood esti-
mation. The DLE model only adopts the reparameteriza-
tion strategy and no residual log-likelihood estimation. It is
seen that the residual manner provides 7.8 mAP improve-
ments. Like previous work [57], we further adopt the net-
work backbone that pre-trained by the heatmap loss. This
model achieves the best performance with 71.3 mAP, which
is denoted with ∗.
Note that the ﬂow model does not participate in the infer-
ence phase. Therefore, no extra computation is introduced
in testing. Besides, the training overhead of the ﬂow model
is negligible. Detailed results are reported in the supple-
mental document (§B). These experiments demonstrate the
superiority of the proposed regression paradigm.
Comparison with Heatmap-based Methods.
We further
compare our regression method with heatmap-based meth-
ods. As shown in Tab. 2(a), our regression method outper-
forms Integral Pose [57] by 7.5 mAP, and the heatmap su-
pervised SimplePose [67] by 0.3 mAP. For the ﬁrst time, the
direct regression method achieves superior performance to
the heatmap-based method.
In Tab. 2(b), we also implement RLE with HRNet [55] to
show our approach is ﬂexible and can be easily embedded
into various backbone networks. Since HRNet maintains
high resolution throughout the whole process, we adopt
soft-argmax to produce coordinates ˆµ and an FC layer to
produce ˆσ. It shows that integral with RLE surpasses con-
ventional heatmap by 0.2 mAP.
Tab. 2(c) shows the superiority of RLE on Mask R-CNN,
the end-to-end top-down approach.
Our regression ver-
sion outperforms the heatmap-based Mask R-CNN by 0.7
mAP. In Tab. 2(d), RLE brings 0.4 mAP improvement to
the state-of-the-art one-stage approach. Note that the out-
6

Figure 3: Qualitative results on COCO dataset: containing crowded scenes, occlusions and appearance change.
Method
Correlation
SimplePose (Heatmap)
0.479
Regression with Gaussian
0.476
Regression with Laplace
0.522
RLE (Q(¯x) ∼Gaussian)
0.540
RLE (Q(¯x) ∼Laplace)
0.553
Table 4: Correlation testing on dif-
ferent methods.
Method
#Params GFLOPs
GFLOPs
of Net. Head
AP
Heatmap-based
Integral Pose [57]
34.0M
9.7
5.7
63.4
SimplePose [67]
34.0M
9.7
5.7
71.0
Regression-based
*Regression with RLE
23.6M
4.0
0.0002
71.3
Table 5: Computation complexity and model
parameters.
Lfc × Nn
AP
AP50
AP75
3 × 64
70.5
88.5
77.4
3 × 128
70.2
88.5
77.3
3 × 256
69.6
87.9
76.5
5 × 32
70.0
88.2
76.8
5 × 64
70.3
88.7
77.4
Table 6: Different RealNVP
Architecture.
Distribution
AP
AP50
AP75
Const. Variance Gaussian (ℓ2)
36.6
70.9
33.6
Const. Variance Laplace (ℓ1)
58.1
82.7
65.0
Gaussian
60.2
82.9
66.6
Laplace
67.4
86.8
74.2
RLE (Q(¯x) ∼Gaussian)
70.0
88.1
76.7
RLE (Q(¯x) ∼Laplace)
70.5
88.5
77.4
Table 7: Initial Density: Performance with different hy-
potheses of the output distribution.
put of PointSet Anchor [64] relies on the heatmap predic-
tions. The regressed values are used for joint association.
The superiority of RLE is demonstrated by embedding it in
various approaches.
Comparison with the SOTA on COCO test-dev
In this
experiment, we compare the proposed RLE with the state-
of-the-art methods on COCO test-dev.
Quantitative re-
sults are reported in Tab. 3.
The proposed regression
paradigm signiﬁcantly outperforms other regression-based
methods by 7.0 mAP and achieves state-of-the-art perfor-
mance. Compared to the same backbone heatmap-based
methods, our regression method is 1.4 mAP higher with
ResNet-152 and 0.2 mAP higher with HRNet-W48. We
demonstrate that heatmap is not the only solution to human
pose estimation. Regression-based methods have great po-
tential and can achieve superior performance than heatmap-
based methods. Qualitative results are shown in Fig. 3.
Correlation with Prediction Correctness.
The esti-
mated standard deviation ˆσ establishes the correlation with
the prediction correctness. The model will output a larger
ˆσ for a more uncertain result. Therefore, ˆσ plays the same
role as the conﬁdence score in the heatmap. We transform
the deviation to conﬁdence with Eq. 9. To analysis the cor-
relation of ˆσ and the prediction correctness, we calculate
the Pearson correlation coefﬁcient between the conﬁdence
and the OKS to the ground truth on the COCO validation
set. The conﬁdence of the heatmap-based prediction [67] is
the maximum value of the heatmap. Tab. 4 reveals that RLE
has much a stronger correlation to OKS than the heatmap-
based method (relative 15.2% improvement). In real-world
applications and other downstream tasks, a reliable conﬁ-
dence score is useful and necessary. RLE address the lack
of conﬁdence scores in regression-based methods and pro-
vide a more reliable score than the heatmap-based methods.
Computation Complexity.
The experimental results of
computation complexity and model parameters are listed
in Tab. 5.
The proposed method achieves comparable
results to the heatmap-based methods with signiﬁcantly
lower computation complexity and fewer model parame-
ters. Speciﬁcally, the total FLOPs are reduced by 58.8%,
and the parameters are reduced by 30.6%. We further cal-
culate the FLOPs of the network head to remove the inﬂu-
ence of the network backbone. It shows that the FLOPs of
the regression head is only 1/28500 of the heatmap head,
which is almost negligible. The computational superiority
of our proposed regression paradigm is of great value in the
industry.
4.2. Ablation Study
RealNVP Architecture.
In Tab. 11, we compare different
network architectures of the RealNVP [11] model. It shows
that the ﬁnal AP keeps stable with different RealNVP archi-
tectures. We argue that learning the residual log-likelihood
is easy for the ﬂow model. Thus the results are robust to the
change of the architecture.
7

Initial Density.
To examine how the assumption of the
output distribution affects the regression performance in the
context of MLE, we compare the results of different den-
sity functions with our method. The Laplace distribution
and Gaussian distribution will degenerate to standard ℓ1 and
ℓ2 loss if they are assumed to have constant variances. As
shown in Tab. 7, the learned distributions of our method
provide more than 21.3% improvements. Besides, we study
the baselines that assuming the output follows the Gaus-
sian and Laplace distributions with the learnable deviation
σ. The distributions with learnable σ outperform those with
constant variance, but are still inferior to RLE.
Moreover, different initial densities Q(¯x) for RLE are
also tested. There is a large gap between the original Gaus-
sian and Laplace distribution. However, with RLE to learn
the change of the density, the difference between these two
distributions is signiﬁcantly reduced. It demonstrates that
RLE is robust to different assumptions of the initial density.
5. Experiments on Human3.6M
Human3.6M [21] is an indoor benchmark for 3D pose es-
timation. For evaluation, MPJPE and PA-MPJPE are used.
Following typical protocols [57, 39], we use (S1, S5, S6,
S7, S8) for training and (S9, S11) for evaluation.
Implementation Details.
For the single-stage approach,
we adopt the same ResNet-50 + FC architecture. The input
image is resized to 256 × 256. Data augmentation includes
random scale (±30%), rotation (±30◦), color (±20%) and
ﬂip. The learning rate is set to 1 × 10−3 at ﬁrst and reduced
by a factor of 10 at the 90th and 120 epoch. We use the
Adam solver and train for 140 epochs, with a mini-batch
size of 32 per GPU and 8 GPUs in total. The 2D and 3D
mixed data training strategy (MPII + Human3.6M) is ap-
plied. The testing procedure is the same as the previous
works [57].
For the two-stage approach, we embed the proposed re-
gression paradigm into the classic baseline [36] and the
state-of-the-art model [70].
2D ground-truth poses are
taken as inputs. For data normalization, we follow previ-
ous works [70, 50, 71]. The initial learning rate is 1 × 10−3
and decays 5% after each epoch. We use the Adam solver
and train for 80 epochs, with a mini-batch size of 1024.
Ablation Study.
In Tab. 8, we report the performance
comparison between RLE and the baselines on both single-
stage and two-stage approaches. It is seen that RLE reduces
the error of single-stage regression baseline by 1.5 mm and
the heatmap-based Integral Pose [57] by 0.6 mm. Besides,
without 3D heatmaps, our regression method signiﬁcantly
reduces the FLOPs by 61.7% and the model parameters by
30.6%. For the two-stage approach, RLE brings 2.7 mm im-
Method
#Params
GFLOPs
MPJPE ↓
PA-MPJPE ↓
Single-stage
Direct Regression
23.8M
5.4
50.1
39.3
Integral Pose [57]
34.3M
14.1
49.2
39.1
Regression with RLE
23.8M
5.4
48.6
38.5
Two-stage
FC Baseline
4.3M
0.275
43.6
33.2
FC Baseline + RLE
4.3M
0.275
40.9
31.1
Table 8: Ablation study on Human3.6M.
Method
Sun
[56]
Nibali
[44]
Sun
[57]
Moon
[39]
Zhou
[72]
Ours
#Params
-
57.9M
34.3M
34.3M
49.6M
23.8M
GFLOPs
-
29.3
14.1
14.1
41.4
5.4
MPJPE
59.1
49.5
49.6
53.3
47.7
48.6
Table 9: Single-stage Results on Human3.6M.
Method
Choi
[10]
Martinez
[36]
Zhao
[71]
Fang
[14]
Liu
[33]
Zeng
[70]
Ours
MPJPE
55.5
45.5
43.8
42.5
37.8
36.5
36.3
Table 10: Two-stage Results on Human3.6M. Our result
is based on SRNet [70] with RLE.
provement to the regression baseline without any test-time
overhead.
Comparison with the State-of-the-art.
In this experi-
ment, we compare the proposed regression paradigm with
both single-stage and two-stage state-of-the-art methods in
Tab. 9 and Tab. 10. For single-stage, our method achieves
comparable performance to the state-of-the-art methods
while reducing the FLOPs by 86.7%. The model parameters
and FLOPs are calculated using the ofﬁcial code of these
methods. Note that [72] only releases the testing code. For
fare comparison, we re-train the model with the same train-
ing settings as ours. For two-stage, our method is based
on SRNet [70] with RLE. It achieves state-of-the-art per-
formance by 0.2 mm improvement to the original SRNet.
6. Conclusion
In this paper, we propose a novel and effective regres-
sion paradigm from the perspective of maximum likelihood
estimation. The learning process is to maximize the prob-
ability of the observation.
We leverage the normalizing
ﬂow model to learn the residual log-likelihood w.r.t. to the
tractable initial density function. Comprehensive experi-
ments are conducted to validate the efﬁcacy of the proposed
paradigm. For the ﬁrst time, the regression-based methods
achieve superior performance to the heatmap-based meth-
ods. Regression-based methods are efﬁcient and ﬂexible.
We hope our method would inspire the ﬁeld to rethink the
potential of regression.
8

References
[1] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and
Bernt Schiele. 2d human pose estimation: New benchmark
and state of the art analysis. In CVPR, 2014. 1, 2, 11, 12
[2] Jonathan T Barron. A general and adaptive robust loss func-
tion. In CVPR, 2019. 3
[3] Benjamin Biggs, S´ebastien Ehrhadt, Hanbyul Joo, Benjamin
Graham, Andrea Vedaldi, and David Novotny.
3d multi-
bodies: Fitting sets of plausible 3d human models to am-
biguous image data. In NeurIPS, 2020. 3
[4] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh.
Realtime multi-person 2d pose estimation using part afﬁnity
ﬁelds. In CVPR, 2017. 1, 2, 6
[5] Joao Carreira, Pulkit Agrawal, Katerina Fragkiadaki, and Ji-
tendra Malik. Human pose estimation with iterative error
feedback. In CVPR, 2016. 1, 2
[6] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang
Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network
for multi-person pose estimation. In CVPR, 2018. 6
[7] Zerui Chen, Yiru Guo, Yan Huang, and Liang Wang. Learn-
ing depth-aware heatmaps for 3d human pose estimation in
the wild. In BMVC, 2019. 2
[8] Bowen Cheng, Bin Xiao, Jingdong Wang, Honghui Shi,
Thomas S Huang, and Lei Zhang. Higherhrnet: Scale-aware
representation learning for bottom-up human pose estima-
tion. In CVPR, 2020. 2
[9] Stephanie J Chiu, Michael J Allingham, Priyatham S Mettu,
Scott W Cousins, Joseph A Izatt, and Sina Farsiu. Kernel re-
gression based segmentation of optical coherence tomogra-
phy images with diabetic macular edema. Biomedical optics
express, 2015. 14
[10] Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee.
Pose2mesh: Graph convolutional network for 3d human pose
and mesh recovery from a 2d human pose, 2020. 2, 8
[11] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.
Density estimation using real nvp. In ICRL, 2016. 4, 5, 7, 11
[12] facebookresearch. Detectron2. https://github.com/
facebookresearch/detectron2, 2021. 6
[13] Hao-Shu Fang, Shuqin Xie, Yu-Wing Tai, and Cewu Lu.
Rmpe: Regional multi-person pose estimation.
In ICCV,
2017. 2, 6
[14] Hao-Shu Fang, Yuanlu Xu, Wenguan Wang, Xi