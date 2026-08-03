# Kinematic-Structure-Preserved Representation for Unsupervised 3D Human Pose Estimation

> 2020 · id: W2997288107 · arXiv: 2006.14107 · pdf: https://ojs.aaai.org/index.php/AAAI/article/download/6792/6646 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Estimation of 3D human pose from monocular image has
gained considerable attention, as a key step to several human-
centric applications. However, generalizability of human pose
estimation models developed using supervision on large-scale
in-studio datasets remains questionable, as these models often
perform unsatisfactorily on unseen in-the-wild environments.
Though weakly-supervised models have been proposed to ad-
dress this shortcoming, performance of such models relies on
availability of paired supervision on some related tasks, such
as 2D pose or multi-view image pairs. In contrast, we pro-
pose a novel kinematic-structure-preserved unsupervised 3D
pose estimation framework1, which is not restrained by any
paired or unpaired weak supervisions. Our pose estimation
framework relies on a minimal set of prior knowledge that
deﬁnes the underlying kinematic 3D structure, such as skele-
tal joint connectivity information with bone-length ratios in
a ﬁxed canonical scale. The proposed model employs three
consecutive differentiable transformations named as forward-
kinematics, camera-projection and spatial-map transforma-
tion. This design not only acts as a suitable bottleneck stim-
ulating effective pose disentanglement, but also yields inter-
pretable latent pose representations avoiding training of an
explicit latent embedding to pose mapper. Furthermore, de-
void of unstable adversarial setup, we re-utilize the decoder
to formalize an energy-based loss, which enables us to learn
from in-the-wild videos, beyond laboratory settings. Compre-
hensive experiments demonstrate our state-of-the-art unsu-
pervised and weakly-supervised pose estimation performance
on both Human3.6M and MPI-INF-3DHP datasets. Qualita-
tive results on unseen environments further establish our su-
perior generalization ability.
1

## introduction
Building general intelligent systems, capable of understand-
ing the inherent 3D structure and pose of non-rigid humans
from monocular RGB images, remains an illusive goal in
the vision community. In recent years, researchers aim to
solve this problem by leveraging the advances in two key
aspects, i.e. a) improved architecture design (Newell, Yang,
and Deng 2016; Chu et al. 2017) and b) increasing collection
of diverse annotated samples to fuel the supervised learning
paradigm (Mehta et al. 2017b). However, obtaining 3D pose
∗equal contribution
1https://sites.google.com/view/ksp-human/
Table 1: Characteristic comparison of our approach against
prior unsupervised and weakly-supervised human 3D pose
estimation works, in terms of access to direct (paired) or in-
direct (unpaired) supervision levels (MV: Multi-View). Note
that, in the proposed framework the latent pose represen-
tation itself, is the 3D pose coordinates, thereby avoiding
training of a separate latent to 3D pose mapper (last column).

## method
As shown in Fig. 1A, we employ two encoder networks each
with a different architecture, EP and EA to extract the local-
kinematic parameters vk (see below) and FG-appearance, fa
respectively from a given RGB image. Additionally, EP also
outputs 6 camera parameters, denoted by c, to obtain coor-
dinates of the camera-projected 2D landmarks, p2D.
One of the major challenges in learning factorized rep-
resentations (Denton and others 2017) is to realize purity
among the representations. More concretely, the appearance
representation should not embed any pose related informa-
tion and vice-versa. To achieve this, we enforce a bottleneck
on the pose representation by imposing kinematic-structure
based constraints (in 3D) followed by an inverse-graphics
formalization for 3D to 2D re-projection. This introduces
three pre-deﬁned transformations i.e., a) Forward kinematic
transformation, Tfk and b) Camera projection transforma-
tion Tc, and c) Spatial-map transformation Tm.
a) Forward kinematic transformation, Tfk
Most of the
prior 3D pose estimation approaches (Chen et al. 2019a;
Rhodin et al. 2018) aim to either directly regress joint lo-
cations in 3D or depth associated with the available 2D
landmarks. Such approaches do not guarantee validity of
the kinematic structure, thus requiring additional loss terms
in the optimization pipeline to explicitly impose kinematic
constraints such as bone-length and limb-connectivity in-
formation (Habibie et al. 2019). In contrast, we formalize
a view-invariant local-kinematic representation of the 3D
skeleton based on the knowledge of skeleton joint connec-
tivity. We deﬁne a canonical rule (see Fig. 1B), by ﬁxing
the neck and pelvis joint (along z-axis, with pelvis at the
origin) and restricting the trunk to hip-line (line segment
connecting the two hip joints) angle, to rotate only about
x-axis on the YZ-plane(i.e. 1-DOF) in the canonical coordi-
nate system C (i.e. Cartesian system deﬁned at the pelvis as
origin). Our network regresses one pelvis to hip-line angle
and 13 unit-vectors (all 3-DOF), which are deﬁned at their
respective parent-relative local coordinate systems, LP a(j),
where Pa(j) denotes the parent joint of j in the skeletal
kinematic tree. Thus, vk ∈R40 (i.e. 1+13*3). These pre-
dictions are then passed on to the forward-kinematic trans-
formation to obtain the 3D joint coordinates p3D in C, i.e.
Tfk : vk →p3D where p3D ∈R3J, with J being the to-
tal number of skeleton joints. First, positions of the 3 root
joints, p(j)
3D for j as left-hip, right-hip and neck, are ob-
tained using the above deﬁned canonical rule after applying
the estimate of the trunk to hip-line angle, v(0)
k . Let len(j)
store the length of the line-segment (in a ﬁxed canonical
unit) connecting a joint j with Pa(j). Then, p(j)
3D for rest
of the joints is realized using the following recursive equa-
tion, p(j)
3D = p(P a(j))
3D
+ len(j)v(j)
k . See Fig. 1B (dotted box)
for a more clear picture.
b) Camera-projection transformation, Tc
As p3D is de-
signed to be view-invariant, we rely on estimates of the cam-
era extrinsics c (3 angles, each predicted as 2 parameters, the
sin and cos component), which is used to rotate and trans-
late the camera in the canonical coordinate system C, to
obtain 2D landmarks of the skeleton (i.e. using the rotation
and translation matrices, Rc and Tc respectively). Note that,
these 2D landmarks are expected to register with the cor-
responding joint locations in the input image. Thus, the 2D
landmarks are obtained as, p(j)
2D = P(Rc ∗p(j)
3D +Tc), where
P denotes a ﬁxed perspective camera transformation.
c) Spatial-map transformation, Tm
After obtaining co-
ordinates of the 2D landmarks p2D ∈R2J, we aim to ef-
fectively aggregate it with the spatial appearance-embedding
fa. Thus, we devise a transformation procedure Tm, to trans-
form the vectorized 2D coordinates into spatial-maps de-
noted by f2D ∈RH×W ×Ch, which are of consistent reso-
lution to fa, i.e. Tm : p2D →f2D. To effectively encode
both joint locations and their connectivity information, we
propose to generate two sets of spatial maps namely, a) heat-
map, fhm and b) afﬁnity-map, fam (i.e., f2D : (fhm, fam)).
Note that, the transformations to obtain these spatial maps
must be fully differentiable to allow the disentaglement of
pose using the cross-pose image-reconstruction loss, com-
puted at the decoder output (discussed in Sec. 3.3a). Keeping

Training pipeline for paired samples (in blue) from 
Progression of Forward Kinematics 
camera
Camera 
projection
Heat map
Affinity map,
A.
B.
Spatial-map 
transformer
1 DOF
3 DOF
3 DOF
Differentiable transformations
Figure 1: A. Illustration of the proposed framework indicating output notation of individual modules. B. An overview of the
three differentiable transformations, with step-wise progression of forward kinematics using local-kinematic parameters, vk.
this in mind, we implement a novel computational pipeline
by formalizing translated and rotated Gaussians to represent
both joint positions (i.e. fhm) and skeleton-limb connectiv-
ity (i.e. fam). We use a constant variance σ along both spa-
tial directions to realize the heat-maps for each joint j, as
f (j)
hm(u) = exp(−0.5||u −p(j)
2d ||2/σ2), where u : [ux, uy]
denotes the spatial-index in a H × W lattice (see Fig. 2A).
We formalize the following steps to obtain the afﬁnity
maps based on the connectivity of joints in the skeletal
kinematic tree (see Fig. 2A). For each limb (line-segment),
l with endpoints pl(j1)
2D
and pl(j2)
2D , we ﬁrst compute loca-
tion of its mid-point, µ(l) : [µ(l)
x , µ(l)
y ] and slope θ(l). Fol-
lowing this, we perform an afﬁne transformation to obtain,
u′ = Rθ(l) ∗(u −µ(l)), where Rθ(l) is the 2D rotation ma-
trix. Let, σ(l)
x
and σ(l)
y
denote variance of a Gaussian along
both spatial directions representing the limb l. We ﬁx σ(l)
y
from prior knowledge of the limb width. Whereas, σ(l)
x
is
computed as α ∗len(l) in the 2D euclidean space (see Sup-
plementary). Finally, the afﬁnity map is obtained as,
f (l)
am(u) = exp(−0.5||u′
x/σ(l)
x ||2 −0.5||u′
y/σ(l)
y ||2)
Tfk, Tc and Tm (collectively denoted as Tk) are de-
signed using perfectly differentiable operations, thus allow-
ing back-propagation of gradients from the loss functions
deﬁned at the decoder output. As shown in Fig. 1A, the de-
coder takes in a tuple of spatial-pose-map representation and
appearance (f2D and fa respectively, concatenated along the
channel dimension) to reconstruct an RGB image. To effec-
tively disentangle BG information in fa, we fuse the back-
ground image Bt towards the end of decoder architecture,
inline with (Rhodin et al. 2018).
3.2
Access to minimal prior knowledge
One of the key objectives of this work is to solve the unsu-
pervised pose estimation problem with minimal access to
prior knowledge whose acquisition often requires manual
annotation or a data collection setup, such as CMU-MoCap
. Adhering to this, we restrict the proposed framework from
accessing any paired or unpaired data samples as shown in
Table 1. Here, we list the speciﬁc prior information that has
been considered in the proposed framework,
• Kinematic skeletal structure (i.e. the joint connectivity
information) with bone-length ratios in a ﬁxed canoni-
cal scale. Note that, we do not consider access to the
kinematic angle limits for the limb joints, as such angles
are highly pose dependent particularly for diverse human
skeleton structures (Akhter and Black 2015).
• A set of 20 synthetically rendered SMPL models with di-
verse 3D poses and FG appearance (Varol et al. 2017). We
have direct paired supervision loss (denoted by Lprior)
on these samples to standardize the model towards the in-
tended 2D or 3D pose conventions (see Supplementary).
3.3
Unsupervised training procedure
In contrast to (Jakab et al. 2018), we aim to disentangle fore-
ground (FG) and background (BG) appearances, along with
the disentanglement of pose. In a generalized setup, we also
aim to learn from in-the-wild YouTube videos in contrast to
in-studio datasets, avoiding dataset-bias.
Separating paired and unpaired samples.
For an efﬁ-
cient disentanglement, we aim to form image tuples of the
form (Is, It, Bt). Here, Is and It are video frames, which
have identical FG-appearance with a nonidentical kinematic-
pose (pairs formed between frames beyond a certain time-
difference). As each video-clip captures action of an indi-
vidual in a certain apparel, FG-appearance remains identi-
cal among frames from the same video. Here, Bt denotes
an estimate of BG image without the human subject cor-
responding to the image It, which is obtained as the me-
dian of pixel intensities across a time-window including the
frame It. However, such an estimate of Bt is possible only
for scenarios with no camera movement beyond a certain
time window to capture enough background evidence (i.e.
static background with a moving human subject).
Given an in-the-wild dataset of videos, we cl

## experiments
In this section, we describe experimental details followed by
a thorough analysis of the framework for bench-marking on
two widely used datasets, Human3.6M and MPI-INF-3DHP.
We use Resnet-50 (till res4f) with ImageNet-pretrained
parameters as the base pose encoder EP , whereas the ap-
pearance encoder is designed separately using 10 Convolu-
tions. EP later divides into two parallel branches of fully-
connected layers dedicated for vk and c respectively. We use
J = 17 for all our experiments as shown in Fig. 1. The

Table 2: Results on Human3.6M following the standard protocol-II setup. Here, Sup. (2nd column) denotes the amount of
supervision accessed by the respective approaches. Accordingly, the table is divided into 4 row-groups, a) row 1-5 use full 3D
pose sup., b) row 6-10 use full 2D pose as weak sup. c) row 11-12: unsupervised approaches, and d) row 13: Ours(semi-sup.).
We outperform prior approaches in both weakly supervised and unsupervised setting (highlighted as boldface).
Protocol-II
Sup.
Direct. Disc.
Eat
Greet Phone Photo Pose Purch.
Sit
SitD Smoke Wait Walk WalkD WalkT Avg.(↓)
(Akhter et al. 2015)
Full-3D
199.2 177.6 161.8 197.8 176.2 186.5 195.4 167.3 160.7 173.7 177.8 181.9 198.6 176.2
192.7
181.1
(Zhou et al. 2016)
Full-3D
99.7
95.8
87.9 116.8 108.3 107.3
93.5
95.3
109.1 137.5 106.0 102.2 110.4 106.5
115.2
106.7
(Bogo et al. 2016)
Full-3D
62.0
60.2
67.8
76.5
92.1
77.0
73.0
75.3
100.3 137.3
83.4
77.3
79.7
86.8
87.7
82.3
(Moreno et al. 2017)
Full-3D
66.1
61.7
84.5
73.7
65.2
67.2
60.9
67.3
103.5 74.6
92.6
69.6
78.0
71.5
73.2
74.0
(Martinez et al. 2017)
Full-3D
44.8
52.0
44.4
50.5
61.7
59.4
45.1
41.9
66.3
77.6
54.0
58.8
35.9
49.0
40.7
52.1
(Wu et al. 2016)
Full-2D
78.6
90.8
92.5
89.4
108.9 112.4
77.1
106.7 127.4 139.0 103.4
91.4
79.1
-
-
98.4
(Tung et al. 2017)
Full-2D
77.6
91.4
89.9
88.0
107.3 110.1
75.9
107.5 124.2 137.8 102.2
90.3
78.6
-
-
97.2
(Chen et al. 2019a)
Full-2D
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
68.0
(Wandt et al. 2019)
Full-2D
53.0
58.3
59.6
66.5
72.8
71.0
56.7
69.6
78.3
95.2
66.6
58.5
63.2
57.5
49.9
65.1
Ours (weakly-sup.)
Full-2D
56.0
53.2
56.3
63.6
74.1
77.5
53.4
67.9
75.8
90.8
64.2
56.9
61.4
56.3
49.7
63.8
(Rhodin et al. 2018)
Multi-view
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
98.2
Ours (unsup.)
No sup.
80.2
81.3
86.0
86.7
94.1
83.4
87.5
84.2
101.2 110.9
86.0
87.8
86.9
94.3
90.9
89.4
Ours (semi-sup.)
5%-3D
46.6
54.5
50.1
46.4
81.3
42.4
41.1
56.4
86.7
82.9
49.0
47.7
64.1
48.2
44.3
56.1
Table 3:
Results for the MPI-INF-3DHP dataset. Here,
Trainset (2nd column) denotes access to 3DHP trainset im-
ages before evaluation. Sup. (3rd column) denotes supervi-
sion level on 3DHP image-pose pairs. 4 row-groups, a) row
1-2: Fully supervised, b) row 3-7: Weakly supervised, c) row
8-10: Unsupervised, d) row 11: Semi-supervised.
No. Method
Trainset
Sup.
PCK (↑) AUC (↑) MPJPE (↓)
1.
(Mehta et al. 2017c)
+3DHP Full-3D
76.6
40.4
124.7
2.
(Rogez et al. 2017)
+3DHP Full-3D
59.6
27.6
158.4
3.
(Zhou et al. 2017)
+3DHP Full-2D
69.2
32.5
137.1
4.
(Kanazawa et al. 2018) +3DHP Full-2D
77.1
40.7
113.2
5.
(Yang et al. 2018)
+3DHP Full-2D
69.0
32.0
-
6.
(Chen et al. 2019a)
+3DHP Full-2D
71.7
36.3
-
7.
Ours (weakly-sup.)
+3DHP Full-2D
80.2
44.8
97.1
8.
(Chen et al. 2019a)
-3DHP
-
64.3
31.6
-
9.
Ours (unsup.)
-3DHP
-
76.5
39.8
115.3
10. Ours (unsup.)
+3DHP No sup.
79.2
43.4
99.2
11. Ours (semi-sup.)
+3DHP 5%-3D
81.9
52.6
89.8
channel-wise aggregation of fam (16-channels) and fhm
(17-channels) is passed through two convolutional layers to
obtain f2D (128-maps), which is then concatenated with fa
(512-maps) to form the input for DI (each with 14×14 spa-
tial dimension). Our experiments use different AdaGrad op-
timizers (learning rate: 0.001) for each individual loss com-
ponents in alternate training iterations, thereby avoiding any
hyper-parameter tuning. We perform several augmentations
(color jittering, mirroring, and in-plane rotation) of the 20
synthetic samples, which are used to provide a direct super-
vised loss at the intermediate pose representations.
Datasets. The base-model is trained on a mixture of
two datasets, i.e. Human3.6M and an in-house collection
of YouTube videos (also refereed as YTube). In contrast to
the in-studio H3.6M dataset, YTube contains human sub-
jects in diverse apparel and BG scenes performing varied
forms of motion (usually dance forms such as western, mod-
ern, contemporary etc.). Note that all samples from H3.6M
contribute to the paired dataset Dp, whereas ∼40% sam-
ples in YTube contributed to Dp and rest to Dunp based
on the associated BG motion criteria. However, as we do
not have ground-truth 3D pose for the samples from YTube
(in-the-wild dataset), we use MPI-INF-3DHP (also refereed
as 3DHP) to quantitatively benchmark generalization of the
proposed pose estimation framework.
a) Evaluation on Human3.6M.
We evaluate our frame-
work on protocol-II, after performing scaling and rigid
alignment of the poses inline with the prior arts (Chen et
al. 2019a; Rhodin et al. 2018). We train three different
variants of the proposed framework i.e. a) Ours(unsup.),
b) Ours(semi-sup.), and c) Ours(weakly-sup.) as reported
in Table 2. After training the base-model on the mixed
YTube+H3.6M dataset, we ﬁnetune it on the static H3.6M
dataset by employing Lprior and Lp (without using any
multi-view or pose supervision) and denote this model as
Ours(unsup.). This model is further trained with full su-
pervision on the 2D pose landmarks simultaneously with
Lprior and Lp to obtain Ours(weakly-sup.). Finally, we also
train Ours(unsup.) with supervision on 5% 3D of the en-
tire trainset simultaneously with Lprior and Lp (to avoid
over-ﬁtting) and denote it as Ours(semi-sup.). As shown
in Table 2, Ours(unsup.) clearly outperforms the prior-
art (Rhodin et al. 2018) with a signiﬁcant margin (89.4
vs. 98.2) even without leveraging multi-view supervision.
Moreover, Ours(weakly-sup.) demonstrates state-of-the-art
performance against prior weakly supervised approaches.
b) Evaluation on MPI-INF-3DHP.
We aim to realize a
higher level of generalization in consequence of leverag-
ing rich kinematic prior information. The proposed frame-
work outputs 3D pose, which is bounded by the kinematic
plausibility constraints even for unseen apparel, BG and
action categories. This characteristic is clearly observed
while evaluating performance of our framework on unseen
3DHP dataset. We take Ours(weakly-sup.) model trained on
YTube+H3.6M dataset to obtain 3D pose predictions on un-
seen 3DHP testset (9th row in Table 3). We clearly outper-
form the prior work (Chen et al. 2019a) by a signiﬁcant mar-
gin in a fully-unseen setting (8th and 9th row with -3DHP in
Table 3). Furthermore, our weakly supervised model (with
100% 2D pose supervision) achieves state-of-the-art perfor-
mance against prior approaches at equal supervision level.
c) Ablation study.
In the proposed framework, our ma-
jor contribution is attributed to the design of differentiable
transformations and an innovative way to facilitate the us-

A. On H36M, in-studio dataset   (samples from           w/ paired BG sup.)
FG-appear. 
separation
Pose 
separation
view 
synthesis
On a new 
BG
Pose 
separation
B. On YTube, in-the-wild dataset  (samples from               w/o BG sup.)
(P1, A1, B1)
(P1, -, -)
(P1, A1, -)
(P2, A1, -)
(P2’, A1, -) (P2’, A1, B3)
(P2, A2, B2)
(P2, -, -)
(P2, A2, -)
(P1, A2, -)
(P1’, A2, -) (P1’, A2, B3)
FG-appear. 
separation
Pose 
separation
view 
synthesis
On a new 
BG
Pose 
separation
(P1, A1, B1)
(P1, -, -)
(P1, A1, -)
(P2, A2, B2)
(P2, -, -)
(P2, A2, -)
(P1, A2, -)
(P1’, A2, -) (P1’, A2, B3)
(P2, A1, -)
(P2’, A1, -) (P2’, A1, B3)
Figure 3: Qualitative results, showing disentanglement of Pose (ID’d as P1 and P2), FG (ID’d as A1 and A2) and BG (ID’d as B1, B2, and
B3). Images in ﬁrst column (of each panel) deﬁne the IDs which are later used for novel image synthesis. Devoid of a direct pixel-wise loss,
energy-based losses for samples from Dunp, help to clearly separate the FG person even in absence of a BG estimate (right panel).
C. Results of LSP dataset (unseen, in-the-wild samples)
D. Results of YTube dataset (in-the-wild)
A. Results of H36M dataset (in-studio)
B. Results of 3DHP dataset
GT
Pred.
GT
Pred.
GT
Pred.
GT
Pred.
GT
Pred.
GT
Pred.
Figure 4: Qualitative results on 4 different datasets. Note that, results on LSP is obtained in an unseen setting (i.e. not even unpaired unsup.
training). The pink box highlights some failure cases, speciﬁcally in presence of self-occlusion as a result of joint-position ambiguity.
Table 4: Results on ablations of the proposed framework. It
clearly highlights importance of Tfk, Tm, and use of Dunp
in the unsupervised training pipeline. Notice the improve-
ment in 3DPCK on the unseen 3DHP testset as a result of
incorporating Dunp in the unsupervised training pipeline.

## related_work
3D human pose estimation. There is a plethora of fully-
supervised 3D pose estimations works (Fang et al. 2018;
Mehta et al. 2017a; Mehta et al. 2017b), where the perfor-
mance is bench-marked on the same dataset, which is used
for training. Such approaches do not generalize on minimal
domain shifts beyond the laboratory environment. In ab-
sence of large-scale diverse outdoor datasets with 3D pose
annotations, datasets with 2D pose annotations is used as
a weak supervisory signal for transfer learning using var-
ious 2D to 3D lifting techniques (Tung et al. 2017; Chen
et al. 2017; Ramakrishna et al. 2012). However, these ap-
proaches still rely on availability of 2D pose annotations.
Avoiding this, (Kocabas et al. 2019; Rhodin et al. 2018)
proposed to use multi-view correspondence acquired by
synchronized cameras. But in such approaches (Rhodin
et al. 2018), the latent pose representation remains un-
interpretable and abstract, thereby requiring a substantially
large amount of 3D supervision to explicitly train a latent-
to-pose mapping mapper. We avoid training of such explicit
mapping, by casting the latent representation, itself as the 3D
pose coordinates. This is realized as a result of formalizing
the geometry-aware bottleneck.
Geometry-aware representations. To capture intrinsic
structure of objects, the general approach is to disentangle
individual factors of variations, such as appearance, camera
viewpoint and other pose related cues, by leveraging inter-
instance correspondence. In literature, we ﬁnd unsupervised
land-mark detection techniques (Zhang et al. 2018), that aim
to utilize a relative transformation between a pair of in-
stances of the same object, targeting the 2D pose estimation
task. To obtain such pairs, these approaches rely on either
of the following two directions, viz. a) frames from a video
with an acceptable time-difference (Jakab et al. 2018), or b)
synthetically simulated 2D transformations (Rocco, Arand-
jelovic, and Sivic 2017). However, such techniques fail to

capture the 3D structure of the object in the absence of multi-
view information. The problem becomes more challenging
for deformable 3D skeletal structures as found in diverse hu-
man poses. Recently (Jakab et al. 2018) proposed an un-
supervised 2D landmark estimation method to disentangle
pose from appearance using a conditional image generation
framework. However, the predicted 2D landmarks do not
match with the standard human pose key-points, hence are
highly un-interpretable with some landmarks even lying on
the background. Such outputs can not be used for a conse-
quent task requiring a structurally consistent 2D pose input.
Deﬁning structural constraints in 2D is highly ill-posed,
considering images as projections of the actual 3D world.
Acknowledging this, we plan to estimate 3D pose separately
with camera parameters followed by a camera-projection to
obtain the 2D landmarks. As a result of this inverse-graphics
formalization, we have the liberty to impose structural con-
straints directly on the 3D skeletal representation, where
the bone-length and other kinematic constraints can be im-
posed seamlessly using consistent rules as compared to the
corresponding 2D representation. A careful realization of
3D structural constraints not only helps us to obtain inter-
pretable 2D landmarks but also reduces the inherent uncer-
tainty associated with the process of lifting a monocular 2D
images to 3D pose (Chen et al. 2019a), in absence of any
additional supervision such as multi-view or depth cues.
3

## conclusion
We present an unsupervised 3D human pose estimation
framework, which relies on a minimal set of prior knowl-
edge regarding the underlying kinematic 3D structure. The
proposed local-kinematic model indirectly endorses a kine-
matic plausibility bound on the predicted poses, thereby lim-
iting the model from delivering implausible pose outcomes.
Furthermore, our framework is capable of leveraging knowl-
edge from video frames even in presence of background mo-
tion, thus yielding superior generalization to unseen environ-
ments. In future, we would like to extend such frameworks
for predicting 3D mesh, by characterizing the prior knowl-
edge on human shape, alongside pose and appearance.

Acknowledgements. This work was supported by a Wipro
PhD Fellowship (Jogendra) and in part by DST, Govt. of
India (DST/INT/UK/P-179/2017).