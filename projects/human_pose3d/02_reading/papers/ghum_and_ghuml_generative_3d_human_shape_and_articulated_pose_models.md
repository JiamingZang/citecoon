# GHUM &amp; GHUML: Generative 3D Human Shape and Articulated Pose Models

> 2020 · id: W3035581100 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models
Hongyi Xu
Eduard Gabriel Bazavan
Andrei Zanﬁr
William T. Freeman
Rahul Sukthankar
Cristian Sminchisescu
Google Research
{hongyixu, egbazavan, andreiz, wfreeman, sukthankar, sminchisescu}@google.com
Abstract
We present a statistical, articulated 3D human shape
modeling pipeline, within a fully trainable, modular, deep
learning framework. Given high-resolution complete 3D
body scans of humans, captured in various poses, together
with additional closeups of their head and facial expres-
sions, as well as hand articulation, and given initial, artist
designed, gender neutral rigged quad-meshes, we train all
model parameters including non-linear shape spaces based
on variational auto-encoders, pose-space deformation cor-
rectives, skeleton joint center predictors, and blend skin-
ning functions, in a single consistent learning loop. The
models are simultaneously trained with all the 3d dynamic
scan data (over 60, 000 diverse human conﬁgurations in
our new dataset) in order to capture correlations and en-
sure consistency of various components. Models support
facial expression analysis, as well as body (with detailed
hand) shape and pose estimation. We provide fully train-
able generic human models of different resolutions – the
moderate-resolution GHUM consisting of 10,168 vertices
and the low-resolution GHUML(ite) of 3,194 vertices –, run
comparisons between them, analyze the impact of different
components and illustrate their reconstruction from image
data. The models will be available for research.
1. Introduction
Human motion, action, and expression are of central
practical importance, and subject to continuous focus, as
well as creative capture in images and video. Immersive
photography, augmented and virtual reality, and physical
3D space reasoning would be next. Consequently, mod-
els that can accurately represent the full body detail at the
level of pose, shape, and facial expression, as well as hand
manipulation are essential in order to capture and deeply
analyze those subtle interactions that can only be fully un-
derstood in 3D. While considerable progress has been made
in localizing human stick ﬁgures in images and video, and
Figure 1. Illustration of accuracy of GHUM and GHUML on data
from GHS3D, with heatmaps of both models on the left. Render-
ings show registrations of different body poses of a subject (grey,
ﬁrst row), as well as GHUM and GHUML reconstructions in sec-
ond and third rows, respectively. Notice good level of detail cap-
ture for both models, with higher accuracy for GHUM.
– under certain conditions – lifting to equivalent 3D skele-
tons and basic shapes, the general quest for reconstructing
accurate models of the the human body at the level of se-
mantically meaningful surfaces, grounded in a 3D physical
space, is still on.
The potential for model construction advances, at least
in the medium term, appears to be at the incidence between
intuitive physical and semantic human modeling, and large-
scale datasets. While many expressive models for faces,
hands and bodies have been constructed over time, most –
if not all – were built in isolation rather than in the context of
a full human body. Hence, inevitably, they did not take ad-
vantage of the large scale data analysis and model construc-
tion process that recently emerged in the context of deep
learning. A number of recent full body models like Adam,
Frank, or SMPL-X[14, 31], combine legacy components for
face, body and hands, but usually focus on constructing a
consistent, joint parameterization with proper scaling on top
of already learnt components, rather than on training a full
16184

Figure 2. Overview of our end-to-end statistical 3D articulated human shape model construction. We are given a set of high-resolution
3D body scans including both ’A’ – and arbitrary – poses exposing a variety of articulation and soft tissue deformations. Additionally, we
also collect head closeup scans of detailed facial expressions and hand closeup scans to capture different gestures and object grabs. Body
landmarks are automatically identiﬁed by rendering the photorealistic 3D reconstructions in multiple virtual viewpoints, detecting them
in the generating images and triangulating. An artist designed full body articulated mesh is progressively registered to point clouds using
losses that combine sparse landmark correspondences and dense iterative closest point (ICP) residuals (implemented as point scan to mesh
facet distances), under as conformal as possible surface priors[41]. The model has non-linear shape spaces implemented as deep variational
auto-encoders (VAEs) for the body φb, and offset VAEs for the facial expressions φf, and includes trainable pose-space deformation
functions D, modulated by a skeleton K with J joints, centers predictor C, and blend skinning functions M. During training, all high-
resolution scans of the same subjects (both full-body and closeups for face and hands) are used (see ﬁg. 3), with residuals appropriately
masked by the ﬁlter F. For model construction, we use N captured subjects, with B full body scans, F closeup hand scans, and H closeup
head scans. During learning, we alternate between minimizing the loss function w.r.t. pose estimates in each scan θ, and optimizing it
with respect to the other model parameters (φ, γ, ψ, ω). In operation, e.g. for pose and shape estimation, the model is controlled by
parameters α = (θ, β), including kinematic pose θ and VAE latent spaces for body shape and facial expressions β = (βf, βb), with
encoder-decoders given by φ = (φf, φb).
body model, end-to-end, based on a large data repository.
This makes it difﬁcult to take full advantage of the structure
in all data simultaneously, experiment with alternative rep-
resentations for components or different losses, assess end
impact, and innovate.
In this paper we propose an end-to-end learning pipeline
to construct full body, statistical human shape and pose
models capable of actuating facial expressions, as well as
body and hand motion.
We design end-to-end pipelines
and uniﬁed loss functions based on deep learning, which
allow for the simultaneous training of all model compo-
nents, including non-linear shape spaces, pose-space de-
formation correctives, skeleton joint center estimators, and
blend skinning functions in the context of minimal human
skeleton parameterizations with anatomical joint angle con-
straints. The models are trained with high-resolution full
body scans, as well as closeups of moving faces and hands,
in order to capture maximum detail and ensure design con-
sistency between body components. Our new collected 3D
dataset of generic human shapes, GHS3D, consists of over
60,000 photo-realistic dynamic human body scans, and we
also use over 4,000 full body scans from Caesar. We intro-
duce both a moderate-resolution model, GHUM, and a spe-
cially designed (not down-sampled) low-resolution model
GHUML, assess their relative performance for registration
and constrained 3d surface ﬁtting, under different linear and
non-linear models (e.g. PCA or variational auto-encoders
for body shape and facial expressions), and illustrate recov-
ery of shape and pose from images.
Related work. There is a remarkable amount of work de-
voted to both constructing 3D articulated surface models for
body parts, i.e. faces, hands and full bodies[2, 4, 10, 24, 37,
6185

30, 11, 29, 16, 38, 3, 32, 7, 8, 21, 39], as well as, more re-
cently, integrating them into complete, more expressive rep-
resentations, as e.g. in Adam, Frank or SMPL-X[14, 31].
Many image and video-based pose and shape estimation
methods have also been proposed[33, 27, 25, 40, 22, 1, 12,
26, 34, 23].
The Frank model[15] is based on a simpliﬁed version of
the SMPL body[24], to which it connects an artist-designed
hand rig, and the public FaceWarehouse head[8]. The com-
bined asset has possibly inconsistent components grafted
together, resulting in a model that may lack realism. In
turn, SMPL-X attaches the FLAME[21] head to the SMPL-
H (body and hand) model[35] and reﬁts it to an additional
set of 5,586 scans. However, since those full body scans
have limited resolution for hands and faces, the authors use
the original, pre-trained parameters of MANO and FLAME
(pose space and pose corrective blendshapes of MANO[35]
for the hands, and the expression space of FLAME[21],
respectively), thus limiting the amount of data simultane-
ously used for learning the full model, and the potential re-
alism attainable by jointly reﬁning all parameters. In con-
trast to combining legacy components, we focus on using
all high-resolution data simultaneously – both full body and
closeup detail for faces and hands –, in order to construct
low-res and high-res models where all parameters are re-
ﬁned end-to-end from the onset. This allows us to experi-
ment with different resolutions, linear and non-linear shape
spaces, loss functions, and assess their impact seamlessly
for different tasks. Recent work focuses on building deep
learning pipelines to predict articulated meshes from point
clouds[19, 13]. These registration alternatives would be im-
mediately applicable in our framework, although here we
rely on direct optimization for registration with automatic
landmark detection for accuracy, robustness, and general-
ization to virtually any pose and human datapoint scan.
Considerable work has been devoted to estimating 3D
pose and shape from images acquired with one or several
cameras or from video[33, 27, 25, 40, 22, 1, 12, 26, 9].
Several models rely on feed-forward pose and shape pre-
diction based on different learning architectures, on pose
prediction followed by pose and shape reﬁnement to body
joints of semantic body part segmentations, or on multicam-
era fusion[28, 43, 44, 20, 36, 5, 18]. Most shape priors come
in the form of PCA as available in SMPL[24], Frank[15],
or SMPL-X[31], and the pose priors are usually Gaussian
Mixture Models [6], and more recently VAEs[31]. In con-
trast, our GHUM and GHUML rely on non-linear shape
spaces constructed from deep variational autoencoders for
body and facial deformation and on normalizing ﬂow rep-
resentations for skeleton (body and hand) kinematics[42].
Moreover our minimal skeleton parameterization supports
the seamless integration of anatomical joint angle limits
constraints during registration, learning and pose optimiza-
tion, which reduces the search space, and makes estimates
anatomically consistent and more robust.
While our primary goal in this paper is to introduce
new end-to-end learnable 3D statistical articulated human
body shape methodologies, the models we present are use-
ful in connection with most work aiming to recover pose
and shape from images.
Moreover, by creating both a
medium-resolution and a low-resolution model, we enable
lightweight mobile applications of 3d human sensing, or ap-
proaches where different level of detail and run-time con-
straints could make it adequate to dynamically switch be-
tween models of different complexity.
2. Overview
Given a training set of human body scans, represented
as unstructured point clouds {Y ∈R3P }, where the num-
ber of points P varies, we learn a statistical human model
X(α) ∈R3V representing the variability of body shapes
and natural deformation due to articulation.
The body
model X has consistent topology with V vertices, as spec-
iﬁed by an artist-provided (rigged) template mesh, and α
are variables that control the body deformation as a result
of both shape and articulation. As illustrated in ﬁg. 2, to
learn a data-driven human model from 3D scans Y, we ﬁrst
register the body template to point clouds in order to obtain
new meshes of the same topology, marked as {X∗∈R3V }
(see Sup. Mat. for details on our registration methodology).
We then feed the registered meshes X∗into an end-to-end
Figure 3. We estimate the full body shape at a neutral A pose by
fusing the body scan and the closeup hand and head scans. Com-
pared with body shape estimation from a single body scan, we can
thus take advantage of additional head and hand shape detail.
training network where model parameters α are adjusted to
produce outputs that closely match the input as a result of
both articulation and shape adjustment. In practice, we ex-
perimented with both direct model parameter adjustment to
the point cloud via iterative closest point (ICP) losses (iden-
tical to the ones used for registration) or with alignment to
the proxy meshes X∗. Since our registration process is ex-
tremely accurate, we haven’t noticed any signiﬁcant differ-
6186

ence between the two. In contrast, using target input meshes
X∗of the same model topology, makes the process consid-
erably faster and training losses are better behaved.
2.1. Human Model Representation
We represent the human model as an articulated mesh,
speciﬁed by a skeleton K with J joints and skin deformed
based on Linear Blending Skinning (LBS) to explicitly en-
code the motion of joints. In addition to skeletal articu-
lated motion, we use nonlinear models to drive facial ex-
pressions. A model X with J joints can be formulated as
M(α = (θ, β), φ, γ, ψ, ω), or in detail, as
X(α) = M(θ, ˜X(βb), ∆˜X(θ), ∆˜Xf(βf), C( ˜X), ω)
(1)
where ˜X(βb) ∈R3V is the identity-based rest shape in A-
pose (ﬁg. 2), with βb a low-dimensional embedding vector
encoding body shape variability (different low-dimensional
representations including PCA or VAEs will be used); sim-
ilarly, ∆˜Xf(βf), is the facial expression at neutral head
pose controlled by low-dimensional latent code βf; c =
C( ˜X) ∈R3J are skeletal joint centers dependent on the
body shape, θ ∈R3×(J+1) is a vector of skeleton pose
parameters consisting of (up to) 3 rotational DOFs in Eu-
ler angles for each joint and 3 translational variables at the
root, ω ∈RV ×I are per-vertex skinning weights inﬂuenced
by at most I = 4 (in our experiments) joints.
FInally,
pose-dependent corrective blend shapes ∆˜X(θ) are added
to the rest shape to correct for skinning artifacts. We initial-
ize our human models, GHUM and GHUML, using artist-
deﬁned rigged template meshes (Vghum = 10, 168, Vghuml =
3, 194, J = 63), respectively and our pipeline will estimate
all the parameters (θ, φ, γ, ψ, ω) while the mesh topology
and the joint hierarchy K are considered ﬁxed. The hierar-
chy is anatomically (minimally) parameterized in order to
take advantage of bio-mechanical joint angle limits during
optimization. Vertices xi ∈X can be written as
xi =
I
X
j=1
ωi,jTj(θ, c)Tj(¯θ, c)−1

˜xi + ∆˜xi + ∆˜xf
i
1

(2)
Tj(θ, c) =
Y
a∈K(j)

Ra(θa)
ca
0
1

∈SE(3),
(3)
where Tj(θ, c) is the world transformation matrix for joint
j, integrated by traversing the kinematic chain from the root
to j. The transformation from rest to posed mesh is con-
structed by multiplying with the inverse of the world trans-
formation matrix at rest pose ¯θ.
3. End-to-End Statistical Model Learning
In this section, we will provide an end-to-end neural
network-based pipeline where we optimize the skinning
Figure 4. Evaluation on Caesar. Left: Per-vertex Euclidean error
to the registration for GHUM and GHUML. Right: (top to bottom,
registrations, GHUM and GHUML) VAE-based models can repre-
sent body shape very well. Compared to GHUML additional, e.g.
muscle or waist, soft tissue detail is preserved by GHUM.
weights ω, and learn a rest shape embedding βb, a fa-
cial expression embedding βf, identity shape-dependent
joint centers estimator C(ψ), pose-dependent blend shapes
function D(γ) given multi-subject and multi-pose surface
meshes X∗registered to full body and close-up face and
hand scans (ﬁg. 3). As a result of ICP registration, we can
easily formulate reconstruction losses using per-vertex Eu-
clidean distance error under one-to-one correspondences as
Lr(X∗, X(α)) = 1
V
V
X
i=1
∥Fi(xi −x∗
i )∥,
(4)
where F is a ﬁlter that accounts for different types of data
(full body scans as opposed to closeups). In order to con-
struct X(α), we need to jointly estimate the pose θ and the
statistical shape parameters. We rely on block coordinate
descent, alternating between estimation of pose parameters
θ under the current shape parameters β, based on a BFGS
layer, and updating the other model parameters with θ ﬁxed.
We initialize skinning from the artist-provided defaults, all
other parameters to 0. In the sequel, we detail how each
sub-module updates the parameters α of the global loss (4).
3.1. Variational Body Shape Autoencoder
We obtain multi-subject shape scans by registering our
models to the Caesar dataset (4, 329 subjects) as well as
our captured scans in GHS3D, in neutral A-pose. For now,
given rest shapes ¯X estimated for multiple subjects, we
build a compact latent space for the body shape variation.
Instead of simply building a PCA subspace, here we choose
to represent body shape using a deep nonlinear variational
autoencoder with a lower-dimensional latent code. Because
we estimate mesh articulation, the input scans to our autoen-
coder ¯X are all well aligned at A-pose without signiﬁcant
perturbations from rigid transformations or pose articula-
tion. The encoder and decoder are using parametric ReLU
6187

activation functions, as they can model either an identity
transformation or a standard ReLU, for certain parameters.
As standard practice, the variational encoder will output a
mean and a variance (µ, Σ), which will be transformed to
the latent space through the re-parametrization trick [17], in
order to obtain the sampled code βb. We choose a simple
distribution, N(0, I), and integrate the Kullback-Leibler di-
vergence in the loss function, to regularize the latent space
˜X(βb) =
1
NB
NB
X
1
¯X + SD(βb)
(5)
βb = SE

¯X −
1
NB
NB
X
1
¯X

(6)
where the encoder SE captures the variance from the mean
body shape into the latent vector βb and the decoder SD
builds up the rest shape from βb to match the input target
rest shape. In particular, we initialize the ﬁrst and last layers
of the encoder and decoder, respectively, to the PCA sub-
space U ∈R3V ×L, where L is the dimensionality of the
latent space. All other fully-connected layers are initialized
to identity, including the PReLU units. We initialize the
sub-matrix of log-variance entries to 0, and set the bias to
a sufﬁciently large negative value. The network will thus,
effectively initialize from the linear model, while keeping
additional parameters to a minimum, compared to PCA.
3.2. Variational Facial Expression Autoencoder
The variational shape autoencoder can represent various
body proportions, including the variances of face shapes. To
additionally support complex facial expressions (as opposed
to just anthropometric head and face variations at rest) we
introduce additional facial modeling. We build the model
from thousands of facial expression motion sequence scans
available in GHS3D. In addition to a 3-DOF articulated jaw,
two 2-DOF eyelids and two 2-DOF eyeballs, the parame-
ters of the articulated joints on the head, including skinning
weights and pose space deformation, will be updated to-
gether with the rest of pipeline. For facial motions caused
by expression and not articulation, we build a nonlinear em-
bedding βf with the same network structure as the varia-
tional body shape autoencoder. The input to the VAE is a
facial expression ∆¯Xf ∈R3V f (V f = 2, 056 for GHUM
and 596 for GHUML) at neutral head pose by removing
all articulated joint motion (including neck, head, eyes and
jaw). To un-pose the registered head mesh to neutral, we
ﬁrst ﬁt the articulated joint motion θ for the neutral head
shape (without expression) that matches the registration as
much as possible c.f. (4). The displacement ﬁeld between
the posed head and the registration is accounted to facial
expressions, and before estimating it, we undo (unpose) the
effect of articulated joint motion θ.
Figure 5. Pose space deformation architecture sketch and illustra-
tion showing the beneﬁt of PSD, here around non-passive articu-
lation points, e.g. right hip and thigh, as well as chest and armpits.
For simplicity of illustration, here we use θ as the input feature,
instead of Ri(θi) −Ri(¯θi).
3.3. Skinning Model
Besides nonlinear shape and facial expression models,
we rely on optimal skinning functions estimated from multi-
subject and multi-pose mesh data. Speciﬁcally, we share the
same data term as in (4) but now the optimization variables
are parameters of the joint center predictor C(ψ) : ˜X →c,
pose-dependent corrections to body shape D(γ) : θ →
∆˜X, and skinning weights ω.
A natural choice for the
skeletonal joint centers is to place them at average positions
on the ring of boundary vertices connecting two mesh com-
ponents (segmentations) maximally inﬂuenced by a joint.
The average of boundary vertices, ¯C˜X ∈R3J, imposes
that the skeleton lies in the convex hull of the mesh sur-
face, thus adapting the center placement to different body
proportions. However, we observe downgraded skinning
quality when using such predictors. For better skinning,
we keep the estimate ¯C but on top build a linear regressor
∆C : R3V →R3J to learn joint center corrections given
the body shape
c(˜X) = ¯C˜X + ∆C˜X
(7)
Instead of learning joint centers globally by pooling over all
mesh vertices, we only estimate locally from those vertices
skinned by the joint. This leads to considerably fewer train-
able parameters going down from 3N ×3J to 3N ×3I, with
I = 4 in practice. We also encourage sparsity, through L1
regularization, and also alignment of the bone directions to
the template. To avoid singularities and prevent joint centers
from moving outside the surface, we regularize the magni-
tude of center corrections ∥∆C˜X∥2.
To correct skinning artifacts as a result of complex soft
tissue deformation, we learn a data-driven pose-dependent
corrector (PSD) ∆˜X(θ) applied to the rest shape. We esti-
mate a nonlinear mapping D : Ri(θi) −Ri(¯θi) ∈R9J →
6188

∆˜X(θ) ∈R3n. However, pose space corrections on a
mesh vertex should intuitively be sourced from neighbor-
ing joints. We therefore use a fully-connected ReLU acti-
vated layer to extract a much more compact feature vector
than the input (we use 32 units), from which we then lin-
early regress the pose space deformation. Moreover, our
˜X(θ) is sparse, and a joint can only generate local deforma-
tion correctives to its skinned mesh patch. Compared to the
dense linear regressor in SMPL [24], our network produces
similar quality deformations with considerably fewer (17×
fewer) trainable parameters. We regularize the magnitude
of pose space deformation to be small, preventing matching
the targets by over-ﬁtting through PSD corrections. This is
implemented by a simple L2 penalty as
Lp(∆˜X) = ∥∆˜X(θ)∥2.
(8)
High-frequency local PSD is often undesirable and most
likely due to overﬁtting. Therefore we encourage smooth
pose space deformations with
Ls(∆˜X) =
V
X
i=1
X
j∈N(i)
∥li,j(∆˜xi −∆˜xj)∥2,
(9)
where N(i) are the neighboring vertices to vertex i and li,j
are cotangent-based Laplacian weights.
Even with PSD regularizers and a reduced number of
trainable weights, overﬁting could still occur. Differently
from SMPL or MANO [35], where pose space deforma-
tion were built speciﬁcally for only certain regions (body or
hand), we construct a PSD model for the entire humanoid,
trained jointly based on high-resolution body, hand and
head data closeups. Consequently our body data has limited
variation on hand and head motions, whereas head and hand
data has no motion for the rest of the body. Hence, there is
a large articulation space where all joints can move without
an effect on the loss, which is undesirable. To prevent over-
ﬁtting, we ﬁlter (mask) the input pose feature vector into 4
feature vectors, consisting of the head, body, left hand and
right hand joints. Each feature vector will be taken into the
same ReLU layer and we sum up the outputs before the next
regressor (ﬁg. 5). We formulate a loss
Lf(∆˜X) = ∥F∆˜X −∆˜X∥2,
(10)
that enforces PSDs outside masked regions to be small, thus
biasing the correctives produced by the network towards
limited global impact. However, deformations of shared
surface regions corresponding to areas at the interface be-
tween the head, hand, and the rest of body, are learnt from
all relevant data.
To estimate skinning weights, at the end of the pipeline,
we create a linear blending layer which, given poses θ and
pose-corrected rest shape with facial expression ˜X+∆˜X+
Table 1. Registration error for GHUM and GHUML, on Caesar
and GHS3D, with detail for faces, hands, and the rest of the body.
ICP error (mm)
Chamfer distance (mm)
Dataset
GHUM
GHUML
GHUM
GHUML
Caesar
0.265
0.465
19.13
31.84
body
0.371
0.725
20.76
33.64
head
0.442
0.519
10.12
12.38
hand
0.164
0.423
14.88
22.01
∆˜Xf, outputs a posed mesh (2) controlled by trainable
skinning weight parameters ω. Each skinned vertex is max-
imally inﬂuenced by I = 4 joints in the template. We also
include a prior on ω based on the initial artist painted values
¯ω, ensure that weights are spatially smooth, and per-vertex
weight components are non-negative and normalized
Ls
ω(ω) =
V
X
i=1
X
j∈N(i)
I
X
k=1
∥li,j(ωi,k −ωj,k)∥2
Li
ω(ω) =
V
X
i=1
I
X
k=1
∥ωi,k −¯ωi,k∥2
s.t.
I
X
k=1
ωi,k = 1,
ωi,k ≥0.
(11)
We also weakly regularize the ﬁnal skinned mesh X to
be smooth by adding
Lm(X) =
V
X
i=1
X
j∈N(i)
∥li,j(xi −xj)∥2.
(12)
Pose Estimator. Given body shape estimates and current
skinning parameters, we re-optimize poses θ over the train-
ing set.
To limit the search space, enforce consistency,
and avoid unnatural local minimal, we leverage anatomical
joint angle limits available with our anthropometric skele-
ton.
The problem can be efﬁciently solved using an L-
BFGS solver with box constraints, and gradients evaluated
by (e.g. TensorFlow’s) automatic differentiation.
4. Experiments
Datasets.
In addition to Caesar, which contains diverse
body and face shapes (4, 329 subjects), we also use mul-
tiple proprietary systems operating at 60Hz to capture 48
subjects (24 females and 24 males) with 55 body poses, 60
hand poses and 40 motion sequences of facial expressions.1
The subjects have a BMI range from 17.5 to 39.2, height
from 148 cm to 192 cm and are aged from 21 to 56. For
all multi-pose data, we use 4 subjects for evaluation, and 4
subjects for testing, including a freestyle motion sequence
containing poses generally not in the training set. Each face
1Subject data was collected in a lab setting with informed consent.
6189

Figure 6. Sample registrations from Caesar (top left) as well as our GHS3D. Notice the quality of registration that captures subtle facial
detail, and the soft tissue deformation of the other body parts as a result of articulation.
Figure 7. Analysis of VAE and PCA models illustrate the advan-
tage of non-linear representations in the low-dimensional regime.
capture sequence starts from a neutral face to a designated
facial expression and each sequence lasts about 2s. Regis-
tration samples from the data are shown in ﬁg. 6.
Registration. In Table 1, we report registration to the point
clouds using ICP and the (extended) Chamfer distance [19].
ICP error is measured as point-to-plane distance to the near-
est registered mesh facet, whereas Chamfer distances are
estimated point to point, bidirectionally. Registration has
low error and preserves local point cloud detail (ﬁg. 6).
Model Evaluation. We build both a full resolution and a
low-resolution human model (GHUM and GHUML) using
our end-to-end pipeline. Both models share the same set
of skeleton joints but have 10, 168 vs. 3, 194 mesh ver-
tices (with 1, 932 vs. 585 vertices for facial expressions).
For both models, we evaluate the mean vertex-based Eu-
clidean distances of meshes X to registrations X∗on test-
ing data. Numbers are reported in Table 2 and visualizations
are shown in ﬁgs. 1, 4, and 9. We compare the outputs of
both models to registered meshes under their correspond-
Table 2. Mean vertex-based Euclidean reconstruction error from
registration (mm).
Dataset
Caesar
GHS3D →body
face
hand
GHUM
2.81
5.21
2.96
2.22
GHUML
3.27
6.32
3.28
2.81
ing topology. Both models can closely represent a diversity
of body shapes (modeled as VAEs, ﬁg. 4), produce natural
facial expressions (represented as facial VAEs), c.f. ﬁg. 5
in Sup. Mat., and pose smoothly and naturally without no-
ticeable skinning artifacts for a variety of shapes and poses
(resulting from optimized skinning parameters, c.f. ﬁg. 1).
GHUM vs GHUML. The low resolution model preserves
the global features of the body shape and correctly skins the
body and facial motion. Compared with GHUM, we ob-
serve that GHUML loses some detail for lip deformations,
muscle bulges at the arms and ﬁngers, and wrinkles due to
fat tissue. Performance-wise, GHUML is 2.0× faster, in
feed-forward evaluation mode, than GHUM.
VAE Evaluation. For body shape, our VAE supports both a
16-dim and a 64-dim latent representation where the former
has 1.72× higher reconstruction error (our report is based
on a 16-dim representation). We use a 20-dim embedding
for our facial expression VAE. Fig. 7 shows the reconstruc-
tion error of facial expressions as a function of the latent di-
mension, for both VAE and PCA. The 20-dimensional VAE
has a reconstruction error similar to the one that uses 96 lin-
ear PCA bases, at the cost of 1.4× slower performance.
GHUM vs SMPL. In ﬁg. 8, we evaluate the skinning qual-
ity of GHUM and SMPL, for multiple subjects and poses,
total of 1, 100 scans. We have different mesh and skeleton
6190

Figure 8.
From left to right, registration, GHUM, and SMPL.
GHUM produces skinning with fewer pelvis artefacts for this mo-
tion sequence (0.76 mm lower error on average).
typologies from SMPL and SMPL does not have hand and
facial joints. We therefore take a captured motion sequence
(all the poses, not in our training dataset) from GHS3D,
and register the captured sequence with SMPL and GHUM
mesh respectively. We use one-to-one point-to-plane Eu-
clidean distance for error calculations (to avoid sensitivity
to surface sliding during registration), and we only evaluate
error on the body (minus face and hands) for fair compari-
son with SMPL. GHUM’s mean reconstruction error is 4.23
mm whereas SMPL has 4.96 mm error.
Figure 9. Evaluation and rendering as in ﬁg.1 with emphasis on
the hand reconstruction of GHUM and GHUML. Notice additional
deformation detail around the ﬂexion region of the palm preserved
by GHUM over GHUML. See Sup. Mat. for facial expressions.
3D Pose and Shape Reconstruction from Monocular Im-
ages. We also illustrate image reconstruction using GHUM.
The kinematic prior (for hands and the rest of the body, ex-
cluding the face) is based on normalizing ﬂows and has been
trained using Human3.6M, CMU, and GHS3D [42]. We do
not use an image predictor for pose and shape, but initialize
at 6 different kinematic conﬁgurations and optimize α pa-
rameters under anatomical joint angle limits. As loss we use
the skeleton joints reprojection error and a semantic body-
part alignment c.f. [6, 43]. We show results in ﬁg. 10, see
Sup. Mat for more.
Application Use Cases: Our construction of GHUM/L
models is motivated by the breadth of transformative, im-
mersive 3D applications, that would become possible, in-
cluding clothing virtual apparel try-on, ﬁtness, personal
well-being, health or rehabilitation, AR and VR for im-
proved communication or collaboration, special effects,
human-computer interaction or gaming, among others. In
contrast, applications like visual surveillance and person
identiﬁcation would not be effectively supported currently,
given that model’s output does not provide sufﬁcient detail
or resolution for these purposes. The same is true for the
creation of potentially adversely-impacting deepfakes, as an
appearance model or a joint audio-visual model are not in-
cluded to support photorealistic visual and voice synthesis.
Figure 10. Monocular 3D human pose and shape reconstruction
with GHUM by relying on non-linear pose and shape optimization
under a semantic body part alignment loss.
5. Conclusions
We have presented GHUM and GHUML(ite), two new
generative 3D human shape and pose models of both
moderate resolution (10, 168 vertices) and low-resolution
(3, 194 vertices), respectively. The models are trained based
on a new dataset, GHS3D, of over 60, 000 human scans,
containing both full-body and closeups for faces and hands.
We present a new end-to-end deep learning framework that
supports – for the the ﬁrst time, and based on all data simul-
taneously – the combined training of all model component
parameters including non-linear shape spaces, pose-space
deformation correctives, skeleton joint center estimators,
and surface blend skinning functions.
We run extensive
experiments in the low-resolution and medium-resolution
regime for both registration and constrained articulated 3D
shape ﬁtting and illustrate 3D pose and shape estimation
from monocular images. A perhaps surprising conclusion
is that appropriately trained, a low resolution nonlinear
model of about 3, 000 vertices could have surprisingly good
human shape representation capacity. Models will be made
available for research.
Acknowledgements:
We thank Elisabeta Oneata, Alin
Popa, Mihai Zanﬁr, and Ana Padurariu for their outstand-
ing support with data collection and processing.
6191

References
[1] CMU graphics lab motion capture database. 2009. http:
//mocap.cs.cmu.edu/.
[2] Brett Allen, Brian Curless, Brian Curless, and Zoran
Popovi´c. The space of human body shapes: Reconstruction
and parameterization from range scans. ACM Trans. Graph-
ics, 2003.
[3] Brian Amberg, Reinhard Knothe, and Thomas Vetter. Ex-
pression invariant 3d face recognition with a morphable
model. In FG, 2008.
[4] Dragomir Anguelov, Praveen Srinivasan, Daphne Koller, Se-
bastian Thrun, Jim Rodgers, and James Davis. Scape: shape
completion and animation of people. ACM Trans. Graphics,
2005.
[5] Abdallah Benzine, Bertrand Luvison, Quoc Cuong Pham,
and Catherine Achard. Deep, robust and single shot 3d multi-
person human pose estimation from monocular images. In
ICIP, 2019.
[6] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter
Gehler, Javier Romero, and Michael J Black. Keep it SMPL:
Automatic estimation of 3d human pose and shape from a
single image. In ECCV, 2016.
[7] Alan Brunton, Augusto Salazar, Timo Bolkart, and Stefanie
Wuhrer.
Comparative analysis of statistical shape spaces.
CoRR, abs/1209.6491, 2012.
[8] Chen Cao, Yanlin Weng, Shun Zhou, Yiying Tong, and Kun
Zhou. Facewarehouse: A 3d facial expression database for
visual computing. IEEE TVCG, 2014.
[9] Mihai Fieraru, Mihai Zanﬁr, Elisabeta Oneata, Alin-Ionut
Popa, Vlad Olaru, and Cristian Sminchisescu.
Three-
dimensional reconstruction of human interactions. In CVPR,
2020.
[10] Nils Hasler, Carsten Stoll, Martin Sunkel, Bodo Rosenhahn,
and Hans-Peter Seidel. A statistical model of human pose
and body shape. Computer Graphics Forum, 2009.
[11] Nikolaos Kyriazis Iason Oikonomidis and Antonis Argyros.
Efﬁcient model-based 3d tracking of hand articulations using
kinect. In BMVC, 2011.
[12] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian
Sminchisescu. Human3.6M: Large scale datasets and predic-
tive methods for 3d human sensing in natural environments.
PAMI, 2014.
[13] Haiyong Jiang, Jianfei Cai, and Jianmin Zheng. Skeleton-
aware 3d human shape reconstruction from point clouds. In
ICCV, 2019.
[14] Hanbyul Joo, Tomas Simon, and Yaser Sheikh. Total cap-
ture: A 3d deformation model for tracking faces, hands, and
bodies. In CVPR, 2018.
[15] Hanbyul Joo, Tomas Simon, and Yaser Sheikh. Total cap-
ture: A 3d deformation model for tracking faces, hands, and
bodies. In CVPR, 2018.
[16] Isinsu Katircioglu, Bugra Tekin, Mathieu Salzmann, Vincent
Lepetit, and Pascal Fua. Learning latent representations of
3d human pose with deep neural networks. IJCV, 2018.
[17] Diederik P Kingma and Max Welling. Auto-encoding varia-
tional bayes. arXiv preprint arXiv:1312.6114, 2013.
[18] Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, and
Kostas Daniilidis. Learning to reconstruct 3D human pose
and shape via model-ﬁtting in the loop. In ICCV, 2019.
[19] Chun-Liang Li, Tomas Simon, Jason Saragih, Barnabas Poc-
zos, and Yaser Sheikh. Lbs autoencoder: Self-supervised
ﬁtting of articulated meshes to point clouds. In CVPR, 2019.
[20] Jiefeng Li, Can Wang, Hao Zhu, Yihuan Mao, Hao-Shu
Fang, and Cewu Lu. Crowdpose: Efﬁcient crowded scenes
pose estimation and a new benchmark. In CVPR, 2019.
[21] Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, and
Javier Romero. Learning a model of facial shape and ex-
pression from 4d scans. ACM Trans. Graphics, 2017.
[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context. In
ECCV. 2014.
[23] Yebin Liu, Carsten Stoll, Juergen Gall, Hans-Peter Seidel,
and Christian Theobalt. Markerless motion capture of inter-
acting characters using multi-view image segmentation. In
CVPR, 2011.
[24] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard
Pons-Moll, and Michael J. Black. SMPL: A skinned multi-
person linear model. ACM Trans. Graphics, 2015.
[25] Diogo C Luvizon, David Picard, and Hedi Tabia. 2d/3d pose
estimation and action recognition using multitask deep learn-
ing. In CVPR, 2018.
[26] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Ger-
ard Pons-Moll, and Michael J. Black. AMASS: Archive of
motion capture as surface shapes. In ICCV, 2019.
[27] Julieta Martinez,
Rayat Hossain,
Javier Romero,
and
James J. Little. A simple yet effective baseline for 3d hu-
man pose estimation. In ICCV, 2017.
[28] Dushyant
Mehta,
Oleksandr
Sotnychenko,
Franziska
Mueller, Weipeng Xu, Srinath Sridhar, Gerard Pons-Moll,
and Christian Theobalt. Single-shot multi-person 3D pose
estimation from monocular RGB. In 3DV, 2018.
[29] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko,
Helge Rhodin, Mohammad Shaﬁei, Hans-Peter Seidel,
Weipeng Xu, Dan Casas, and Christian Theobalt.
Vnect:
Real-time 3d human pose estimation with a single rgb cam-
era. ACM Trans. Graphics, 2017.
[30] Markus Oberweger, Paul Wohlhart, and Vincent Lepetit.
Training a feedback loop for hand pose estimation. In ICCV,
2015.
[31] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani,
Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and
Michael J. Black. Expressive body capture: 3d hands, face,
and body from a single image. In CVPR, 2019.
[32] Stylianos Ploumpis, Haoyang Wang, Nick Pears, William
A. P. Smith, and Stefanos Zafeiriou. Combining 3d mor-
phable models: A large scale face-and-head model.
In
CVPR, 2019.
[33] Alin-Ionut Popa, Mihai Zanﬁr, and Cristian Sminchisescu.
Deep multitask architecture for integrated 2d and 3d human
sensing. In CVPR, 2017.
[34] Helge Rhodin, Nadia Robertini, Dan Casas, Christian
Richardt, Hans-Peter Seidel, and Christian Theobalt. Gen-
6192

eral automatic human shape and motion capture using volu-
metric contour cues. In ECCV, 2016.
[35] Javier Romero, Dimitrios Tzionas, and Michael J. Black.
Embodied hands: Modeling and capturing hands and bod-
ies together. ACM Trans. Graphics, 2017.
[36] Kai Su, Dongdong Yu, Zhenqi Xu, Xin Geng, and Changhu
Wang. Multi-person pose estimation with enhanced channel-
wise and spatial information. In CVPR, 2019.
[37] Jonathan Taylor, Lucas Bordeaux, Thomas Cashman, Bob
Corish, Cem Keskin, Toby Sharp, Eduardo Soto, David
Sweeney, Julien Valentin, Benjamin Luff, Arran Topalian,
Erroll Wood, Sameh Khamis, Pushmeet Kohli, Shahram
Izadi, Richard Banks, Andrew Fitzgibbon, and Jamie Shot-
ton. Efﬁcient and precise interactive hand tracking through
joint, continuous optimization of pose and correspondences.
ACM Trans. Graphics, 2016.
[38] Dimitrios Tzionas, Luca Ballan, Abhilash Srikantha, Pablo
Aponte, Marc Polle