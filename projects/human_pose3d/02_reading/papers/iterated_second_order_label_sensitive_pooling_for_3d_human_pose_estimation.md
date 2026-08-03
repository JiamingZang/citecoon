# Iterated Second-Order Label Sensitive Pooling for 3D Human Pose Estimation

> 2014 · id: W2052747804 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Iterated Second-Order Label Sensitive Pooling for 3D Human Pose Estimation
Catalin Ionescu†♮, Joao Carreira‡, Cristian Sminchisescu♯†
♯Department of Mathematics, Faculty of Engineering, Lund University
♮University of Bonn
†Institute of Mathematics of the Romanian Academy
‡University of California at Berkeley
catalin.ionescu@ins.uni-bonn.de, carreira@eecs.berkeley.edu, cristian.sminchisescu@math.lth.se
Abstract
Recently, the emergence of Kinect systems has demon-
strated the beneﬁts of predicting an intermediate body part
labeling for 3D human pose estimation, in conjunction with
RGB-D imagery.
The availability of depth information
plays a critical role, so an important question is whether
a similar representation can be developed with sufﬁcient
robustness in order to estimate 3D pose from RGB im-
ages. This paper provides evidence for a positive answer,
by leveraging (a) 2D human body part labeling in images,
(b) second-order label-sensitive pooling over dynamically
computed regions resulting from a hierarchical decomposi-
tion of the body, and (c) iterative structured-output mod-
eling to contextualize the process based on 3D pose es-
timates.
For robustness and generalization, we take ad-
vantage of a recent large-scale 3D human motion capture
dataset, Human3.6M[18] that also has human body part
labeling annotations available with images. We provide ex-
tensive experimental studies where alternative intermediate
representations are compared and report a substantial 33%
error reduction over competitive discriminative baselines
that regress 3D human pose against global HOG features.
1. Introduction
We focus on the inference of 3D human pose from
monocular intensity images (RGB). Recently,
Kinect
systems[16] based on discriminative learning methods and
large-scale datasets of synthetically generated depth maps
with dense body part annotations have achieved consider-
able success for 3D human pose prediction indoors. An
open question is whether similar advances would be possi-
ble for 3D human pose estimation based on standard inten-
sity images. The transition is by no means trivial, as depth
allows the design of distinctive, scale and illumination-
invariant local features and substantially constrains 3D in-
ference. For RGB, 3D needs to be aggregated from 2D cues
only, and 3D inference ambiguities can occur[32, 30].
Besides heavily relying on depth sensors, a novel as-
pect of Kinect was the use of an intermediate 2D part la-
beling stage within a 3D pose estimation pipeline trained
on unprecedentedly large human pose datasets. No pub-
licly available datasets have been available to match the
diversity and size of the proprietary ones used for learn-
ing Kinect models so far. However, a large data gather-
ing effort has resulted in the recently released Human3.6M
dataset[18]. The annotations provided with this data include
not only 3.6 million ground truth 3D pose and camera pa-
rameter conﬁgurations, ﬁgure-ground segmentation masks,
and human body scans, but also automatically generated
pixel-level body part labels, obtained using a volumetric hu-
man model, and the kinematic information available in the
dataset. We will use the data to train human body labeling
and 3D pose prediction models in multiple passes.
Speciﬁcally, our formulation decomposes the problem
into three stages: (1) dense 2D human body part labeling,
(2) second-order label-sensitive pooling over a hierarchical
decomposition of the body, and (3) human 3D pose predic-
tion by regressing against pooled descriptors, dynamically
contextualized by pose estimates. The success of such a
pipeline fundamentally depends on being able to label body
parts in intensity images, and on the way such inherently
noisy inferences are further processed. We target robust
descriptions based on body part labeling using two com-
plementary strategies.
On one hand we design partially
redundant global pose descriptors that operate on overlap-
ping regions resulting from a hierarchical decomposition of
the human body. We generalize recent region descriptors
for semantic segmentation under high-order statistics[7]
towards a novel description process, called second-order
label-sensitive pooling (O2LP), where regions are identi-
ﬁed dynamically based on their inferred labels. This allows
us to derive efﬁcient descriptors, with multiple levels of se-
lectivity and invariance, for robust human pose estimation.

Additionally, we iteratively inject 3D pose information into
the labeling process, thereby promoting globally consistent
estimates. For example, the leg on the right side of a body
in the image may be a right leg or left leg depending on
whether the person is facing the camera or not. Using a 3D
pose estimate as additional feature to contextualize labeling
allows to transfer information between distant body parts,
reducing ambiguity.
We conclude with experimental studies and a quanti-
tative analysis of the proposed components, showing that
our methodology outperforms state of the art discriminative
baselines that regress 3D pose against global HOG silhou-
ette features, by a substantial 33% margin. Fig. 1 illustrates
the proposed model.
1.1. Related work
Many of the early approaches to 3D human pose estima-
tion employed generative models to search the state space in
order to identify conﬁgurations of a body model that would
best align with image features[12, 32, 29, 34, 15]. While
powerful, these models required careful initialization, con-
siderable manual effort to design a realistic synthetic model
and were computationally expensive. The problem of joint
segmentation and pose prediction has also been attacked us-
ing top-down[6] and bottom-up[17] approaches. Discrimi-
native regression methods[1, 31, 28, 20, 17] focus on pre-
diction from image descriptors, but depend on the existence
of a sufﬁciently large and diverse training set, which until
recently has not been available. The design of sufﬁciently
stable and accurate descriptors is also a main challenge.
For example, the commonly adopted HOG ﬁlters[10] use
a ﬁxed grid of gradient histograms. Depending on the pose
of the person in the image, the same body parts are likely
to fall into very different cells. Because the input descriptor
is unstable across poses, the mapping between the feature
space and the pose space is irregular and complex to learn.
While hierarchical encodings computed over regular grids
have shown a degree of invariance to such factors in 3D
human pose estimation[20], they remain pose-independent.
An alternative would be to compute ‘cells’ that are pose-
centric and can be obtained, e.g., from an image estimate of
the 2D body part layout.
Here we pursue such a representation in the form of a
dense 2D body part labeling. Kinect[16] used fast feature
extraction based on depth differences inspired by [23], to
infer a dense part labeling of the human body from depth
images. Body labeling methods have also been proposed
in the context of person detection[4], to learn a model of
the body[13] and recently for 2D pose prediction[21, 11],
but not within models and algorithms carrying all the way
to 3D pose estimation. These models handle spatial label
dependencies using CRF formulations, which makes infer-
ence hard in loopy graphs. Others have recently proposed
to handle complex dependencies by means of a sequential
feedback mechanism, such as auto-context[33] and ﬁxed-
point structured labeling[25, 3]. Such approaches classify a
pixel using context, based on previously predicted labels of
neighboring pixels. They differ from us in that auto-context
learns a sequence of models where earlier ones of the same
type provide context for later ones. Fixed-point structured
labeling[25] learns a single model that feeds back outputs
into itself and is learned using both clean and purposively
corrupted ground truth context data. Our model is some-
what related to auto-context but we use a deeper, higher-
level form of context, a 3D pose estimate, as well as differ-
ent description methods based on label-sensitive pooling.
There is a large body of literature on 2D pose predic-
tion, with the dominant approaches employing pictorial
structures[14, 27, 2, 35], which impose tree-structured de-
pendencies between joint positions while maximizing an
appearance likelihood. Body part labeling at the level of
pixels operates at a different granularity. While the label-
ing of each individual pixel is performed under weaker spa-
tial constraints, the average process could be more robust
to individual component errors (e.g. estimating a pixel, but
not an entire body part incorrectly). This appears adequate
when pooling inherently noisy 2D label estimates for 3D
pose prediction. It also has the potential advantage of la-
beling arbitrary partial views of the person (a property also
shared by poselets[5], at their different level of bounding
box granularity), which would be more challenging for pic-
torial structures, where a spatial dependency among a set of
components, assumed visible, is sought. Ultimately, such
complementary methodologies may have a role, at different
levels, in a robust system in the long run.
2. Overview and Model Formulation
Our pose estimation methodology is based on descriptors
computed from body part labels with an iterative scheme.
The model we propose can be viewed as two coupled pre-
dictors: a human body part labeling method and a 3D pose
predictor operating on features extracted based on a 2D la-
beling process.
Let L be a discrete set of labels and P(L) its power-
set. Let I be an image consisting of N pixels, and x a
vector, containing the label at each pixel in the image i.e.
x = {xk ∈L|k = 1, . . . , N}. Let zI ∈R3×D be a vec-
tor of 3-dimensional positions, for a representation with D
joints, corresponding to the pose of the person in image I1.
Let R : {1, . . . , M} →P(L) be a mapping, not nec-
essarily onto, deﬁning the assignment of M regions to el-
ements of the power set of labels P(L). Let region Rj =
{k|xk ∈R(j)} be deﬁned as the set of pixels with labels
1Whenever not critical, the dependence of different variables on I will
be dropped.

Figure 1: Overview of our 3D human pose estimation methodology. Local descriptors are extracted densely over the image and given as
input to a 2D body part labeling model. A second order pooling process informed by a hierarchical decomposition of the body (individual
limbs, upper and lower body parts, full body, etc.) is used to construct a global representation by stacking descriptors computed over
overlapping subsets. 3D pose is obtained using regression, with estimates fed-back in order to further constrain the 2D labeling process.
in the set R(j).
Let s(I) = {sk|k = 1, . . . , N}, with
dim(sk) = S, descriptors extracted over neighborhoods
centered at pixel k (e.g. SIFT) in I. Let the statistical model
producing the labeling, at iteration i be p(i) and the pose
predictor at the same iteration be f (i). Let Ψ(x(i), I) be the
pooled feature vector computed based on the labeling pro-
duced at iteration i. The model can be written as:
x(0) ←arg max
x
p(x|I)
(1)
z(i) ←f (i)(Ψ(x(i), I)), ∀i ≥0
(2)
x(i) ←arg max
x
p(i)(x|z(i−1), I), ∀i ≥1
(3)
We train by alternation, initially setting up an estimator for
the labeling based on the image information alone (itera-
tion 0), as p(0) ≡p(x|I), then we iterate between training
a 3D pose model based on the current label predictions and
training a labeling model using predicted poses as global
context. At iteration i, p(i) ≡p(x|z(i−1), I). In training
we have to learn a set of models that use noisy inputs as
pose context. In testing, we iterate using the trained mod-
els, (p(i), f (i)). The scheme usually converges within a few
steps (see ﬁg.3).2
3. Dense 2D Human Body Part Labeling (p)
We use multi-class random decision forests for body part
labeling. The forest is a collection of T decision trees con-
structed over features φ, locally extracted in the image:
p(x|I) = 1
T
T
∑
t=1
pt(x|I)
(4)
The trees contain split nodes and leaf nodes. Each split is
associated to a component of the feature vector φ and a
2Structured prediction methods like SOAR[3] offer a different ap-
proach to iteratively feed the output (pose) estimate as a covariate together
with the image descriptor, although their analytical formulation would not
immediately extend to a dynamic, pose-dependent descriptor construction
process (pose constrained-labeling + label-sensitive feature extraction), as
in the model presented here.
threshold τ. To label the pixel xk, we extract its corre-
sponding feature vector φk, start at the root and evaluate
the feature components based on the corresponding thresh-
olds. By reaching a leaf node, we can use the empirical
distribution of stored labels (a majority vote or any other
learnt model) in order to make the labeling decision.
In our formulation, the feature vectors over which de-
cision trees are constructed are the previously introduced
local S-dimensional vectors φ(x, I) ≡s, for the label-
ing model p(x|I) and the (S + 3 × D)-dimensional vec-
tors φ(i)(x, I) = (s⊤, z(i−1)⊤)⊤for the contextual model
p(x|z(i−1), I). For the latter, the vector z is obtained as
the pose estimate of the model f, at the previous iteration
i −1. Contextual features are computed at different pixel
locations k for each image I, and will include a global pose
component, estimated for that image. The local descriptors
will not change, but the context will change, leading to ran-
dom forests that depend on the iteration index.3
Features and Encodings: Different features φ were used
by the labeling models we developed. The simplest one
described the local image patches si using SIFT (the RF
method). Our experiments (table 1) however showed that a
model based on plain SIFT did not perform very well. Dur-
ing analysis, we identiﬁed 3 problems: (1) inherent ambigu-
ity between limbs because locally arms and legs look sim-
ilar, (2) lack of repeatability due to descriptors computed
for images where people appear at different scales, and (3)
ambiguities between the left and right sides of the body.
We address (1) by adding the location of the pixel rel-
ative to the person bounding box coordinate system as an
additional feature (lRF) to φ. (2) is addressed by learning
a scale predictor based on simple features and using the es-
3During prototyping, we have also experimented with other struc-
tured random forests including models with Potts dependencies. None of
these models lead to signiﬁcant improvement in labeling over the random-
regression forest proposed. We have also experimented with depth features
in order to better understand what is achievable with RGB features. In that
case the estimates of a random forest regressor based on a depth feature
similar to Kinect[16], produced comparable results with our best model on
RGB features, further supporting the feasibility of our approach.

timate to rescale the image before extracting local features
(rRF and lRF, if respectively only SIFT, or both SIFT and
the pixel location feature is used). The scale predictor uses
simple features of the foreground mask: the bounding box
size (height and width), area, perimeter, eccentricity and so-
lidity and learns to predict the distance from the camera
to the pelvis bone of the body (which is available for our
dataset). Images are then rescaled by that factor. The am-
biguity between the left and right side (3) requires global
body orientation information, for which we propose 2 solu-
tions. The ﬁrst uses conﬁdences of a simple global direc-
tion classiﬁer, predicting 4 body orientations and augment-
ing the feature vector φ with 4 directional conﬁdence di-
mensions (rldRF). The second solution is based on a global
contextual feature and appends a pose estimate at the previ-
ous labeling iteration (16× 3D joint coordinates, total of 48
dimensions) to the local features just described. This model
is named rlpRF.4
To understand why context can be useful let us consider
an example. A left-forearm of a person seen frontally may
appear similar to a right-forearm if one observes the person
from a short distance behind, and the context of the other
body parts is not accessible (visible or used). Since the
dataset is balanced with respect to the proportion of views
from the front and from the back, this could create problems
to the classiﬁer. Our contextual term attempts to mitigate
such issues by providing global information – the current
pose estimate – to the labeling model, to further focus its
predictions. While this information only seems useful if the
pose estimate is accurate enough, it does work best in our
experiments. Intuitively, the classiﬁer learns not only the
direction the person is facing by inspecting the positions of
different joints in the image, but also learns a coarse repre-
sentation of the limbs and self-occlusions.
4. Second-Order Label-Sensitive Pooling (Ψ)
In this section we introduce methodology that allows us
to design region descriptors that are covariant with body
pose. We will rely on the intermediate 2D body part label-
ing (§3), aiming, at the same time, of being robust to a de-
gree of body part mis-segmentation and noise. Each region
descriptor can be obtained by appropriately dividing the re-
gion spatially into cells and pooling local descriptors, e.g.
SIFT, over each cell independently, then concatenating the
resulting vectors. Previous work has considered laying out
the cells in a ﬁxed, hand-picked conﬁguration (e.g. spatial
4Notice the signiﬁcant differences compared to RGB-D Kinect mod-
els: (1) we show that human body part labeling can be made feasible using
RGB images only, by considering different features and spatial encodings,
iteratively contextualized by 3D pose estimates (§3); (2) for RGB-D, given
a body labeling and a depth map, predicting the 3D body joints is rela-
tively easy. This is by no means the case for RGB, motivating our novel
development of a label-sensitive pooling descriptor (§4).
pyramids[22]), or in a ﬁxed but learned conﬁguration[19],
both independent of the image content observed.
In contrast, we consider descriptors that provide invari-
ance to imaging nuisance factors by pooling global statistics
of the local descriptors inside multiple image regions, com-
puted dynamically. Instead of creating a one-to-one map-
ping between body-parts and sets of pixels inferred to have
the same label, one can view the space of possible regions
to pool over as a hierarchy, where the coarsest level con-
tains a region for the entire body, and the ﬁnest level has
different regions for each body part. Assuming human sil-
houettes are available, the coarsest region is the same no
matter how poorly body parts are labeled. At the other end,
deﬁning one pooling region for each body part offers po-
tentially distinctive descriptors which may be more difﬁcult
to reliably obtain in practice. Intermediate levels of the de-
composition, e.g. individual arms and legs, or the upper and
lower body parts, offer different trade-offs between full and
part visibility, discriminative power and repeatability.
In this work we propose the novel concept of label-
sensitive pooling by deﬁning a ‘cell’ (or region) conﬁgu-
ration on the ﬂy, based on the 2D body part labeling in-
ferred with our learned model p.
The cells are deﬁned
as potentially overlapping collections of pixels in the im-
age, and are both free-form and input-dependent. Each re-
gion descriptor is computed by means of a second-order
label-sensitive pooling operation (O2LP), by generalizing
a method proposed recently in the different context of se-
mantic segmentation[7].5 The operator is deﬁned as
v(Rj, I) ≡vec
(
log
( 1
|Rj|
∑
k∈Rj
sk · s⊤
k
))
(5)
where log is the principal matrix logarithm, which is sym-
metric, hence the vectorization operator retains only the up-
per triangle. The dimensionality of v is therefore S2+S
2
.
Additionally, a power transformation is applied on each in-
dividual dimension v of v, as sign(v) · |v|h, with h ∈[0, 1].
The ﬁnal global descriptor is obtained by concatenating
the descriptors v(Rj, I), for j ∈[1, . . . , M], resulting in
M × dim(v) dimensions:
Ψ(x, I) = [v(R1, I)⊤, . . . , v(RM, I)⊤]⊤
(6)
Notice the subtle dependency Ψ(x), as an effect of v(Rj)
depending on x, due to labeling Rj = {k|xk ∈R(j)} (§2).
5. Human 3D Pose Estimation (f)
We investigate regression methods with simple and
structured outputs for 3D human pose estimation.
5While in [7] regions were identiﬁed based on a bottom-up segmenta-
tion method, CPMC[8], here regions are deﬁned as containing subsets of
the labels, and their spatial support will vary, for different inputs, as the
union of pixels with inferred labels in those subsets.

Ridge Regression (RR): In this formulation, the models for
each of the D output dimensions (3D joints of the human
body) will be trained independently:
f(Ψ(x, I)) = W⊤Ψ(x, I)
(7)
where W is a matrix of size dim(Ψ)×3×D, with columns
wd. Learning is performed using regularized least squares
arg min
W
∑
I
3×D
∑
d=1
{
∥w⊤
d Ψ(x, I)−zI(d)∥2
2+∥wd∥2
2
}
(8)
where zI(d) is the dimension d of the ground truth pose
vector zI in image I, and wd is the d-th row of W. Ridge
regression has closed form solution and is very efﬁcient to
train if the dimensionality of the input is not extremely high.
Kernel Dependency Estimation: For this structured pre-
dictor, we ﬁrst map the targets to a RKHS, de-correlate
them, then learn an input-to-target map, using ridge regres-
sion. For scalability, we use a linear kernel approximation
methodology[24] to represent the non-linear lifting explic-
itly using a ﬁnite-dimensional approximation of size M. To
learn the target map Γ : R3×D →RM we solve the follow-
ing optimization problem
arg min
W
∑
I
M
∑
d=1
{
∥w⊤
d Ψ(x, I) −Γd(zI)∥2
2 + ∥wd∥2
2
}
(9)
where Γd(zI) is the d-th dimension of Γ(zI). Inference
requires solving a pre-image problem[18]
f(Ψ(x, I)) = arg min
zI
∥W⊤Ψ(x, I) −Γ(zI)∥2
(10)
This can be obtained efﬁciently using an LBFGS optimizer,
initialized using RR predictions (7). The method we em-
ploy is essentially the one in [18] except that for O2LP we
use a linear input kernel, the correct metric being already
accounted for by the principal matrix logarithm (§4).
6. Experiments
Leveraging a large and diverse set of human poses within
trainable systems is key to obtaining robust results, as it has
become clear recently, for pose estimation systems based
on RGB-D data. For our experiments we use the recently
released Human3.6M (H3.6M) dataset[18], which delivers
3.6 million synchronized images and 3D body poses, as
well as camera parameters and body part labeling in im-
ages. We use the set of 24 labels provided with H3.6M, out
of which 20 are associated to limbs (3 for the joints of each
limb and 2 for the upper and lower bones, e.g. the arm con-
sists of the shoulder, elbow, wrist as well as humerus and
radius), 2 for the torso (chest and abdomen), 1 for the pelvis
and 1 for the head (see [18] for details). Our pose estima-
tor uses the standard 17 joint skeleton from H3.6M. Of the
entire dataset, we select a subset of 55,000 training exam-
ples and 25,000 testing examples. We refer to this training
and testing subset Human80K (H80K)6. It is obtained by
eliminating those 3D poses that are similar (< 100 mm),
in each video of H3.6M, then sampling from the remain-
ing data uniformly. We use image data collected from all
4 cameras in H3.6M, in order to make sure that we have a
sufﬁciently diverse set of viewing angles. We cover all the
available 15 scenarios in H3.6M in roughly equal propor-
tions of about 4,000 training examples for each motion for
training and 1,800 for testing.
Our predictive model has three components: body part-
labeling, label sensitive pooling and 3D human pose estima-
tion based on the resulting descriptor. In the sequel, we will
analyze each of these stages in isolation, as well as jointly.
Dense Human Body Part Labeling. As described in §3,
for this task we chose a random forest (RF) which we found
to outperform SVM and logistic regression classiﬁers, by
a large margin. We trained using 150 sample patches per
image, making sure that every visible label has at least
one sample, then extracting the corresponding local feature.
Different predictors were trained using data from the same
image locations (which vary across images) to ensure they
are not biased by sampling artifacts. The resulting dataset
has 8 million patch descriptor examples. RF classiﬁers with
T = 30 trees were trained, which took between 2.5h (our
simplest RF model) to 4.5h (the most complex one), on a
8-core 2.13Ghz machine (48Gb RAM).7
Our results are summarized in table 1. Confusion matri-
ces for different models are shown in ﬁg. 2. We noticed that
both introducing a pixel location and rescaling the bounding
box prior to feature extraction are operations of consistent
performance beneﬁt. The same can be said about the global
information: both direction and pose offer advantages, with
pose providing better overall performance. Notice that pose
carries a much richer context about labels than orientation
disambiguation as it accounts for self-occlusion and gives a
strong prior on the spatial label distribution.
Label-sensitive Pooling. Having shown that human body
part labeling based on RGB images can be made effective,
the next stage is the construction of descriptors Ψ for 3D
human pose estimation, by considering a region hierarchy
for label-sensitive pooling (§4). The regions were deﬁned
by considering a 4-level decomposition of the body: the
ﬁrst level (L1) has one region for the fully body (FB); the
second level (L2) has two regions: upper (UB) and lower
body (LB); the third level (L3) has 5 regions for the head
6Available for download and testing via the Human3.6M site.
7Using one compute core, in Matlab, the inference time on an image is:
.47s for labeling, .81s for O2PL feature extraction and less than .01s for
pose estimation. An iteration takes less than 1.3s (we usually need 2).

RF
rRF
lRF
rlRF
rldRF
rlpRF-RR
rlpRF-KDE
Average Accuracy (%)
58.93
60.05
63.23
63.87
69.53
72.03
73.99
Average Accuracy per Class (%)
38.84
40.08
42.70
43.55
48.92
50.40
53.10
Table 1:
Comparison of classiﬁcation performance (%) for random forest features. We study how augmenting our base SIFT feature
further improve performance. RF is the model based on SIFT, lRF augments it with pixel location coordinates. In all cases an ‘r’ preﬁx
indicate that images were scaled based on the predicted distance to the camera. The other two models are rldRF (which contains both
pixel coordinates and orientation features) and rlpRF which uses the pixel location and an estimate of the 3D pose (predicted using either
ridge regression (RR) or kernel dependency estimation (KDE[9, 18]) against O2LP descriptors computed based on labels from rlRF) as
an additional input feature.
RThigh
RTibia
LThigh
LTibia
LowBack
Chest
Head
LUpArm
LLowArm
RUpArm
RLowArm
RThigh
RTibia
LThigh
LTibia
LowBack
Chest
Head
LUpArm
LLowArm
RUpArm
RLowArm
RThigh
RTibia
LThigh
LTibia
LowBack
Chest
Head
LUpArm
LLowArm
RUpArm
RLowArm
RThigh
RTibia
LThigh
LTibia
LowBack
Chest
Head
LUpArm
LLowArm
RUpArm
RLowArm
RThigh
RTibia
LThigh
LTibia
LowBack
Chest
Head
LUpArm
LLowArm
RUpArm
RLowArm
RThigh
RTibia
LThigh
LTibia
LowBack
Chest
Head
LUpArm
LLowArm
RUpArm
RLowArm
L1
L2
L3
L4
0
0.1
0.2
0.3
0.4
0.5
Fraction of test set
Decomposition level
 
 
LElbow
LWrist
RElbow
RWrist
LKnee
LAnkle
RKnee
RAnkle
Figure 2: Confusion matrices of pixelwise 2D part labeling for different models. First plot shows the RF baseline, second the rlRF and
third the rlpRF model. By injecting global pose information as a contextual feature in the labeling process, the rlpRF model reduces the
ambiguities in assignments over the left-right side of the body, that affect the RF model. Fourth: Histograms showing how many times
models obtained by pooling over different regions performed best at predicting different components of the 3D pose (see text).
and torso (HT) as well as each limb (LA, RA, LL, RL);
the fourth level (L4) has 10 regions: torso, head and the
lower and upper regions for each limb (upper arm, lower
arm, thigh, calf).
This results in 18 regions over which
label-sensitive pooling will be performed. Since the O2P
descriptor is 10,000 dimensional8 this produces a 180,000
dimensional encoding. While it is possible to learn with the
high-dimensional feature vector using, e.g. online methods,
we instead performed PCA on each region descriptor set
independently, then thresholded the spectrum at the same
value for all regions to obtain a 2,500 dimensional descrip-
tor for each. This results in a 45,000 dimensional global
descriptor to work with. For the KDE experiments we use
a 4K dimensional approximation of a Gaussian kernel over
the targets. For eχ2-HoG experiments we use a 15K dimen-
sional approximation[24].
Automatic 3D Human Pose Prediction.
Given label-
sensitive descriptors Ψ, we can obtain the regression pa-
rameters for the 3D human pose predictor in closed-form
(§5). To gain insight into the effectiveness of our hierarchi-
cal representation (16 regions), we have also trained 3D hu-
man pose regressors on descriptors obtained only for those
regions. Separately analyzing different decompositions of-
fers potentially more invariant descriptors at coarser levels
8While the features φ used by the random forest for labeling are ob-
tained by appending additional information to SIFT, the label-induced re-
gion descriptor Ψ used for 3D human pose estimation is constructed by
pooling over SIFT descriptors only.
and more distinctive ones at ﬁner levels. Training regressors
for each region in the decomposition, R(j), separately, and
counting how often each performs best (ﬁg. 2, right) indi-
cates that lowest levels are the most relevant about 50% of
the time, whereas the coarsest ones are cumulatively more
accurate the other 50%. While this invites other contextual
output fusion methods based on model uncertainty (or e.g.
conditional mixture of experts[31]), we still found that a
combined (ﬂat) descriptor produces accurate results overall
while being fast in both training and testing.
Finally, we present the results of our automatic 3D pose
estimation methods together with several baselines, in ta-
ble 2 and ﬁg. 3. Our labeling models are appended ‘O2LP’
to their names in order to make clear that a label-sensitive
pooling process operates over their results in order to con-
struct the descriptors used for 3D pose estimation. The re-
sults show that given ground truth labels, all but about 10%
of our errors are larger than 100mm thus validating the ap-
proach. Our automatic model, RR-O2LP-rlpRF achieves
106mm which represents a substantial 33% error reduc-
tion over a RR-HoG baseline. This shows that improved
labeling through contextual 3D pose features translates in
17% improvement over models not using it.
Our rlpRF
model also offers signiﬁcant improvements with respect to
our simpler rldRF. Note that KDE consistently improves
performance.9 Automatic labeling and 3D reconstruction
9In order to compare our model with a state-of-the-art 2D pose esti-
mation method[35] we have used the 2D ground truth joint position an-

Initial 3D pose error (mm)
Error reduction (mm)
 
 
0
200
400
−100
−50
0
50
100
10−4
10−3
10−2
Initial 3D pose error (mm)
Label accuracy improvement
 
 
0
200
400
−0.2
0
0.2
10−4
10−3
10−2
60
100
140
180
220
0
0.1
0.2
0.3
0.4
3D pose error (mm)
Fraction of test set
 
 
RR−HoG(158)
RR−O2LP−rlRF(127)
RR−O2LP−rlpRF(106)
RR−O2LP−GT(83)
1
2
3
90
100
110
120
130
Iteration
Error (mm)
 
 
RR−O2LP−rlpRF
KDE−O2LP−rlpRF
Figure 3: Left-to-right 1st and 2nd: Impact of including 3D global pose estimates in the features used for body part labeling. The ﬁrst
plot shows the frequency of improvement in labeling and 3D pose estimates as a function of the initial pose error. 3rd: Error magnitude
histograms showing that with perfect labels, on 90% of the test set we obtain 3D pose estimation errors less than 120mm. It also shows
that our contextual labeling model enhanced with global 3D pose estimates produces ≈10% more instances where error is below 100mm.
4th: Pose estimation performance as a function of iteration.
Method/Features
HoG
eχ2-HoG
O2LP-GT
O2LP-rlRF
O2LP-rldRF
O2LP-rlpRF
RR
158
140
83
127
115
106
KDE
129
112
78
100
98
92
Table 2: 3D pose prediction error (in mm, joint position averages) for different models. Notice the substantial improvement of our methods
(fourth column based on GT labeling, last three columns automatic) over competitive regression baselines based on HOG.
-
Head
LShoulder
LUpArm
LElbow
LLowArm
LWrist
RShoulder
RUpArm
RElbow
RLowArm
RWrist
Chest
Abdomen
Pelvis
RHip
RThigh
RKnee
RTibia
RAnkle
LHip
LThigh
LKnee
LTibia
LAnkle
Figure 4: Examples of labeling inference and 3D human pose estimation for our different models. The columns represent from left to
right: the input image, the 3D pose predicted by a HoG baseline, then labeling and estimated 3D pose for our RF, rlRF and rlpRF models.
notations from H80K and trained [35] on the same 55,000 images with
the same foreground masks applied, as our predictors. The model had 26
parts as in the PARSE experiment[35] and otherwise all standard parame-
ters. For this model we have obtained a PCK score 74% at (the standard,
yet fairly liberal) .2 tolerance, on the test set of 25,000 images in H80K.
For our model O2P-rlpRF-KDE we have taken the 3D pose estimates and
projected into the image to obtain 2D joint positions, and obtained a PCK
score of 95.52% at .2 tolerance.
visualizations for different models are shown in ﬁg. 4.
7. Conclusions
We have proposed a 3D pose estimation model that
decomposes into three layers: 2D human body part la-
beling, label-sensitive pooling over a hierarchical region
decomposition of the body, and continuous-valued pose

regression. We employ an iterative structured prediction
formulation that incorporates label-sensitive second-order
pooling over local features in order to build stable and
robust pose descriptors that adapt to the human pose conﬁg-
uration. The diversity and annotation detail of the recently
introduced Human3.6M dataset[18] makes possible to train
large-scale human labeling and pose estimation methods,
at large scale, for the ﬁrst time. The proposed methodology
operates on intensity images and leads to excellent 2D
body part labeling results as well as 3D pose estimates
that are signiﬁcantly more accurate compared to existing
competitive discriminative regression methods based on
HOG. In future work, we plan to introduce additional label
structure in order to handle complex backgrounds and
multiple people.
Acknowledgements This work was supported in part
by
CNCS-UEFISCDI
under
CT-ERC-2012-1
and
PCE-2011-3-0438.
Research was performed in part
while Joao Carreira was at ISR-Coimbra,
supported
by
FCT
under
PTDC/EEA-CRO/122812/2010
and
SFRH/BPD/84194/2012.
The authors are grateful to
Elisabeta Marinoiu for her generous help with the 3D
human modeling and the visualizations used in the paper.
References
[1] A. Agarwal and B. Triggs. 3d human pose from silhouettes
by relevance vector regression. In CVPR, 2004.
[2] M. Andriluka, S. Roth, and B. Schiele. Pictorial structures
revisited: People detection and articulated pose estimation.
In CVPR, 2009.
[3] L. Bo and C. Sminchisescu. Structured Output-Associative
Regression. In CVPR, 2009.
[4] Y. Bo and C. Fowlkes.
Shape-based pedestrian parsing.
CVPR, 2011.
[5] L. Bourdev and J. Malik.
Poselets: Body Part Detectors
Trained Using 3D Human Pose Annotations. In ICCV, 2009.
[6] M. Bray, P. Kohli, and P. Torr. Posecut: Simultaneous seg-
mentation and and 3d pose estimation of humans using dy-
namic graph cuts. In ECCV, 2006.
[7] J. Carreira, R. Caseiro, J. Batista, and C. Sminchisescu. Se-
mantic segmentation with second-order pooling. In ECCV,
2012.
[8] J. Carreira and C. Sminchisescu.
CPMC: Automatic Ob-
ject Segmentation Using Constrained Parametric Min-Cuts.
PAMI, 2012.
[9] C. Cortes, M. Mohri, and J. Weston. A general regression
technique for learning transductions. In ICML, 2005.
[10] N. Dalal and B. Triggs. Histograms of oriented gradients for
human detection. In CVPR, 2005.
[11] M. Dantone, J. Gall, C. Leistner, and L. Van Gool. Human
pose estimation using body parts dependent joint regressors.
In CVPR, 2013.
[12] J. Deutscher, A. Blake, and I. Reid. Articulated Body Motion
Capture by Annealed Particle Filtering. In CVPR, 2000.
[13] S. M. A. Eslami, N. Heess, and J. Winn. The shape boltz-
mann machine: A strong model of object shape. In CVPR,
2012.
[14] V. Ferrari, M. Marin, and A. Zisserman. Pose Seach: retriev-
ing people using their pose. In CVPR, 2009.
[15] J. Gall, B. Rosenhahn, T. Brox, and H. Seidel. Optimiza-
tion and ﬁltering for human motion capture: A multi-layer
framework. IJCV, 2010.
[16] R. Girshick, J. Shotton, P. Kohli, A. Criminisi, and
A. Fitzgibbon. Efﬁcient regression of general-activity hu-
man poses from depth images. In ICCV, 2011.
[17] C. Ionescu, F. Li, and C. Sminchisescu. Latent Structured
Models for Human Pose Estimation. In ICCV, 2011.
[18] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Hu-
man3.6m: Large scale datasets and predictive methods for
3d human sensing in natural environments. PAMI, 2014.
[19] Y. Jia, C. Huang, and T. Darrell. Beyond spatial pyramids:
Receptive ﬁeld learning for pooled image features. In CVPR,
2012.
[20] A. Kanaujia, C. Sminchisescu, and D. Metaxas.
Semi-
supervised hierarchical models for 3d human pose recon-
struction. In CVPR, 2007.
[21] L. Ladicky, P. Torr, and A. Zisserman. Human pose estima-
tion using a joint pixel-wise and part-wise formulation. In
CVPR, 2013.
[22] S. Lazebnik, C. Schmid, and J. Ponce.
Beyond bags of
features: Spatial pyramid matching for recognizing natural
scene categories. In CVPR, 2006.
[23] V. Lepetit and P. Fua. Keypoint recognition using random-
ized trees. PAMI, 2006.
[24] F. Li, G. Lebanon, and C. Sminchisescu. Chebyshev Ap-
proximations to the Histogram χ2 Kernel. In CVPR, 2012.
[25] Q. Li, J. Wang, Z. Tu, and D. P. Wipf. Fixed-point model for
structured labeling. In ICML, 2013.
[26] S. Nowozin, C. Rother, S. Bagon, T. Sharp, B. Yao, and
P. Kohli. Decision tree ﬁelds. In ICCV, 2011.
[27] B. Sapp, A. Toshev, and B. Taskar. Cascaded Models for
Articulated Pose Estimation. In ECCV, 2010.
[28] L. Sigal, A. Balan, and M. J. Black. Combined discrimi-
native and generative articulated pose and non-rigid shape
estimation. In NIPS, 2007.
[29] L. Sigal, S. Bhatia, S. Roth, M. Black, and M. Isard. Track-
ing Loose-limbed People. In CVPR, 2004.
[30] C. Sminchisescu.
3D human Motion Reconstruction in
Monocular Video. Techniques and Challenges, volume 36
of Human Motion Capture: Modeling, Analysis, Animation.
Springer, October 2007. ISBN 978-1-4020-6692-4.
[31] C. Sminchisescu, A. Kanaujia, Z. Li, and D. Metaxas. Dis-
criminative Density Propagation for 3D Human Motion Es-
timation. In CVPR, 2005.
[32] C. Sminchisescu and B. Triggs. Kinematic Jump Processes
for Monocular 3D Human Tracking. In CVPR, 2003.
[33] Z. Tu. Auto-context and its application to high-level vision
tasks. In CVPR, 2008.
[34] R. Urtasun, D. Fleet, A. Hertzmann, and P. Fua. Priors for
people tracking in small training sets. In ICCV, 2005.
[35] Y. Yang and D. Ramanan. Articulated Human Detection with
Flexible Mixtures of Parts. PAMI, 2013.