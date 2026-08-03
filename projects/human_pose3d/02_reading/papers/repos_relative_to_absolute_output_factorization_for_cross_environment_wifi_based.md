# RePos: Relative-to-Absolute Output Factorization for Cross-Environment WiFi-Based 3D Human Pose Estimation

> 2026 · id: W7167749100 · arXiv: 2607.02986 · pdf: https://arxiv.org/pdf/2607.02986 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
The authors are with the Faculty of Science and Technology, Keio Univer-
sity, Yokohama 223-8522, Japan (e-mail: zhangcheng@ohtsuki.ics.keio.ac.jp;
ohtsuki@ics.keio.ac.jp).
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible.
H
UMAN
pose
estimation
underpins
many
mobile
and ubiquitous computing applications, from human–
computer interaction and smart-home automation to ambient
healthcare monitoring. For these settings, WiFi sensing is an
appealing alternative to cameras: it preserves privacy, tolerates
poor or changing illumination, and works through visual occlu-
sions [1]–[3]. Commodity WiFi is already deployed throughout
indoor spaces, so estimating 3D human pose directly from
its Channel State Information (CSI) is an infrastructure-reuse
alternative to camera-based systems that turns the wireless
medium already surrounding a user into a pose sensor [4]–
[6]. The premise of this reuse is that one model serves the
many spaces a deployed network already covers (Fig. 1).
Cross-environment generalization evaluates whether this
premise holds in previously unseen environments. The key
challenge is therefore not achieving high in-room accuracy,
but maintaining reliable performance after deployment with-
out per-environment calibration. CSI is shaped by the entire
propagation channel, so a change of room layout, furniture,
or transmitter–receiver geometry alters the signal that a fixed
pose produces, and a model trained in one space degrades
sharply in another [7], [8]. The cost of recovering accuracy in
each new space, per-space data collection or fine-tuning, scales
with the number of spaces and undoes the infrastructure-reuse
premise that motivated WiFi sensing. A deployable system
must therefore hold up in an unseen target environment with-
out target-side tuning, the regime we formalize in Section III.
This cross-environment gap, not single-room performance, is
the obstacle we target.
Existing WiFi-based 3D pose methods fall into three fami-
lies, each limited against this gap. Direct-regression models
map raw CSI to absolute joint coordinates [9]. Because
CSI carries no explicit spatial correspondence to keypoints,
this mapping is ill-posed and the predicted geometry is of-
ten physically inconsistent. Graph-based models add skeletal
structure [8], [10] by porting graph attention from vision-
based pose estimation [11], [12], yet they operate on joint-
level graphs without an intermediate representation that im-
poses anatomy. Domain-adaptation and generalization meth-
ods instead reshape the feature distribution, aligning environ-
ments [13], [14] or learning environment-invariant represen-
tations [15]. Such alignment can suppress pose-discriminative
signal and rarely separates environment-specific variation from
body-related variation.
Despite their differences, these three families share a com-
mon limitation. Each learns body structure and absolute po-
sition in a single representation, so the two entangle: the
training rooms’ position cues leak into structure estimation,
a coordinate-overfitting effect also observed by Jia et al. [16].
arXiv:2607.02986v2  [cs.CV]  17 Jul 2026

This entanglement is harmless when training and deployment
share a room: the position cues are then consistent, and a
well-built direct model is hard to beat in-domain. However,
the same entanglement is exactly what fails to transfer to
an unseen environment. We instead address this issue at the
output level. Rather than reshape features or supply geometry,
our method RePos (Relative-to-absolute Pose) factors the
prediction into a structure component (the body’s relative joint
configuration) and a location component (its position in the
room). It learns the two under separate objectives, so the
structure branch receives no absolute-position supervision.
RePos realizes this factorization as a two-stage architec-
ture, with each branch matched to its sub-problem. Stage 1,
the structure branch, encodes CSI into Body-Part Latent
Queries (BP-LQs): anatomically grouped latent tokens that
a Skeleton Graph Attention (SGA) module refines under
skeletal-connectivity constraints before they are decoded to
the root-relative pose. Stage 2, the localization branch, is
the Amplitude-based Spatial Prior Network (ASPN), which
predicts the root position from CSI amplitude alone through
a differentiable spatial-decomposition head that needs no cali-
brated phase. We further analyze its scope and show that it is a
partly environment-dependent localizer rather than a physical
phase or angle estimator. Summing the outputs of the two
branches gives the absolute pose.
The contrast between the two deployment regimes makes
the case for this design. When only a single environment is
available, the relative/absolute split is unnecessary: a direct
variant, RePos-D, that regresses the absolute pose end-to-end
already reaches 86.9 mm MPJPE on the in-domain Person-in-
WiFi-3D benchmark, a 3.4% improvement over the previous
best WiFi-based method, DT-Pose [8]. Across environments,
however, regressing absolute coordinates ties body structure
to the training rooms’ position cues, and the same direct
model degrades. Under the strict MM-Fi cross-environment
protocol (train E01–E03, deploy to unseen E04), the factorized
RePos instead lowers MPJPE to 254.4–296.1 mm, a 10–21%
reduction over prior WiFi-only methods across all three ac-
tivity protocols. The gain holds under leave-one-environment-
out cross-validation and leakage-free few-shot transfer, and
feature and pseudo-phase analyses trace its source. Sharing
one backbone, the two designs show that the factorization is
what cross-environment deployment specifically needs, not a
generic accuracy boost.
Our main contributions are as follows.
• We propose RePos, a factorized framework that addresses
coordinate overfitting [16] on the output side: it separates
root-relative pose from root position into two indepen-
dently trained branches, so position cues never leak
into structure estimation and the source-trained model
transfers to unseen environments without calibration.
• We
introduce
Body-Part
Latent
Queries
(BP-LQs),
anatomically grouped latent tokens refined by Skeleton
Graph Attention (SGA), which impose skeletal structure
before joint decoding rather than on the final joints alone.
• We design the Amplitude-based Spatial Prior Network
(ASPN), a calibration-free root-localization branch, and
delimit its scope as a coarse, partly environment-
dependent localizer rather than a physical phase or angle
estimator.
• Extensive experiments demonstrate state-of-the-art per-
formance in both in-domain and cross-environment set-
tings.

## method
ZS
+FT
MetaFi++ [21] (IoT-J’23)
349.6
253.0
HPE-Li [22] (ECCV’24)
387.6
254.4
DT-Pose [8] (EIS’26)
343.8
246.6
GraphPose-Fi [10] (ICASSP’26)
339.4
244.4
RePos w/o ASPN (ours)
351.4
241.6
RePos (ours, full)
291.3
242.9
TABLE VII
ABLATION ON MM-FI PROTOCOL 3 (CROSS-ENVIRONMENT). EACH ROW
REMOVES ONE COMPONENT FROM FULL REPOS.
Variant
MPJPE↓
PA-MPJPE↓
Root↓
PCK50↑
w/o BP-LQs (direct reg.)
363.1
112.9
327.7
48.9
w/o Stage 2
349.4
104.7
321.3
52.6
w/o latent code
365.0
102.7
336.6
49.1
w/o spatial decomp.
359.6
102.7
332.9
50.0
RePos (full)
296.1
102.0
261.0
66.8
result (291.3 mm) differs slightly from the 296.1 mm in Ta-
ble III. We also report RePos without the ASPN (Stage 1 with
absolute supervision only). Results are in Table VI.
Two findings stand out. First, in the deployment-relevant
zero-shot regime (where labeled target motion-capture (Mo-
Cap) data is typically unavailable), full RePos beats every
published baseline by 14–25% (291.3 vs. 339.4 mm best base-
line). RePos without the ASPN matches MetaFi++ (351.4 vs.
349.6 mm), indicating that the zero-shot improvement mainly
comes from the ASPN-based position factorization. Second,
fine-tuning on a target subject exposes the model to the deploy-
ment room, so the test reverts from cross-environment to effec-
tively in-domain (a seen environment with held-out subjects),
and the ranking inverts accordingly. Both variants still beat
every baseline (241.6 and 242.9 vs. 244.4 mm). Yet where the
zero-shot environment was unseen and the factorized RePos
led (291.3 vs. 351.4 mm), the now-seen environment favors the
direct variant (241.6 vs. 242.9 mm), mirroring the in-domain
result on Person-in-WiFi-3D (Table II). The few-shot results
thus trace the same regime boundary the two designs target: the
factorization helps precisely while the environment is unseen,
and its advantage disappears once target supervision makes
the environment in-domain.
H. Ablation Study
We ablate each component under Protocol 3 (cross-
environment) in Table VII.
Removing BP-LQs and regressing joints directly from CSI
raises PA-MPJPE from 102.0 to 112.9 mm (+10.7%) and root
error from 261.0 to 327.7 mm. The root degradation arises
through training: the ASPN is optimized against the absolute-
pose loss with the frozen Stage-1 output as a constant, so
a weaker structure estimate also shifts the root optimum.
The anatomical grouping thus ultimately helps both branches.
Removing Stage 2 reverts the model to direct absolute re-
gression, forcing the Stage-1 representation to encode both
TABLE VIII
SGA ABLATION UNDER PROTOCOL 3 (STAGE 1 ONLY, ROOT-RELATIVE).
SGA variant
MPJPE↓
PA-MPJPE↓
PCK@50↑
Skeleton (ours)
119.45
102.03
90.8
Full attention
121.5
102.9
90.8
TABLE IX
TOKEN-ALLOCATION SENSITIVITY (STAGE 1 ONLY, ROOT-RELATIVE,
E04). ALL ALLOCATIONS TOTAL 150 TOKENS (CAPACITY-MATCHED).
Allocation (H,T,limb)
MPJPErel ↓
PA-MPJPE↓
PCK@50↑
20,30,25 (default)
123.3
104.3
90.8
25,25,25 (uniform)
122.8
103.7
90.8
16,42,23 (torso-heavy)
123.1
104.3
90.7
14,20,29 (limb-heavy)
122.5
103.9
90.9
anatomical configuration and environment-dependent localiza-
tion: root error rises to 321.3 mm (+23.1%) and PA-MPJPE
also worsens to 104.7 mm. This is the coordinate-overfitting
effect the factorization is meant to remove, now visible as
a structure penalty when the two tasks share one network.
This controlled ablation differs from the standalone RePos-D
of Table III: there, self-supervised pretraining keeps structure
intact (PA 102.5 mm) but position still overfits, whereas here
the shared network also penalizes structure, so the two single-
stage routes fail by different mechanisms, both confirming the
factorization. Removing either the latent code or the spatial
decomposition leaves PA-MPJPE unchanged (102.7 mm) but
inflates root error (336.6 and 332.9 mm), so the two ASPN
components are needed together for localization and neither
disturbs the structure branch.
1) Isolating the SGA Module: To isolate SGA from the
BP-LQ prior, we compare skeleton-masked attention against
unconstrained full self-attention on Stage 1 only (root-relative
evaluation), Protocol 3 (Table VIII). Skeleton masking im-
proves MPJPE by 2.1 mm and PA-MPJPE by 0.9 mm at equal
PCK@50; since PCK is unchanged, SGA improves structural
consistency rather than keypoint coverage. The moderate gain
is expected: BP-LQs already introduce an anatomical induc-
tive bias through explicit body-part decomposition, so SGA
mainly improves joint-level consistency rather than recovering
missing body structure.
2) Token-Allocation
Sensitivity:
A
natural
concern
is
whether the {20, 30, 25} token budget is a tuned heuristic. We
sweep four capacity-matched allocations (each summing to
150 tokens) and train the structure branch on each (Table IX).
Across distributions as different as uniform, torso-heavy, and
limb-heavy, PA-MPJPE varies by only 0.6 mm and PCK@50
is unchanged. The structure branch is relatively insensitive to
the allocation within the tested range: the default follows the
relative complexity of each body part, but the result does not
depend on it. These runs use a short Stage-1 schedule, so the
absolute values sit 2–4 mm above the main model (Table VIII).
The relevant quantity is the small spread across allocations, not
the offset.
9

TABLE X
ROBUSTNESS TO STAGE-1 ERROR ON E04. GAUSSIAN NOISE OF THE
GIVEN STANDARD DEVIATION IS ADDED TO THE STAGE-1 ROOT-RELATIVE
POSE BEFORE Jabs = Jrel + r; THE ROOT BRANCH IS UNTOUCHED.
Stage-1 noise std (mm)
0
10
20
40
80
Absolute MPJPE (mm)
296.1
296.5
297.8
302.7
321.4
Increase (mm)
0.0
0.4
1.7
6.6
25.3
I. Robustness to Stage-1 Error
A potential concern with factorized prediction is error
propagation from Stage 1 to the final pose. By construction it
cannot occur: the ASPN reads CSI amplitude rather than the
Stage-1 pose (Section IV), so the branches do not cascade.
We confirm this empirically: on held-out E04 we corrupt the
Stage-1 root-relative pose with Gaussian noise of increasing
standard deviation before the combine Jabs
= Jrel + r
(Table X). The root prediction is identical in every row, and
even a severe 80 mm corruption of Stage 1 raises the absolute
MPJPE by only 25 mm, less than the injected noise itself. A
failing Stage 1 therefore degrades the final pose gracefully
and never breaks Stage 2’s localization, which addresses the
two-stage error-accumulation concern.
J. Stage-1 Module Roles and Pseudo-Phase Analysis
We analyze two aspects of Stage 1: what its modules con-
tribute to the representation, and whether the learned pseudo-
phase captures real spatial direction.
What BP-LQ and SGA contribute (Fig. 4). To see what
each Stage-1 module does, we linearly probe how well the
features at three stages (the CSI encoder output, the BP-LQ
tokens, and the post-SGA tokens) decode the root-relative pose
and the absolute root position. We use RePos-D on Person-in-
WiFi-3D, where the subject moves across a 2.7×1.9 m area
so that position genuinely varies. The CSI encoder already
carries most of the pose information (R2 = 0.47). BP-
LQ then performs a 3× token compression, mapping the
features into 150 anatomically-grouped tokens (feature spread
0.47→0.14), and linear pose decodability dips to 0.37: BP-LQ
is a structured bottleneck that organizes features by body part,
not a pose extractor. SGA produces the largest single pose gain
in the pipeline (+0.23, to R2 = 0.60) as the representation re-
expands, confirming that the skeleton-graph attention is where
structural refinement happens. This matches the design intent:
BP-LQ commits to anatomical grouping and SGA imposes
skeletal connectivity on top. This organization, visualized in
Fig. 7(b,c), is confirmed functional by the BP-LQ and SGA
ablations (Tables VII and VIII). Absolute position stays fully
decodable at every stage (R2→0.99), as expected for the direct
RePos-D variant, which regresses absolute pose end-to-end.
Pseudo-phase (Fig. 5). Regressing the learned inter-antenna
phase difference ∆ˆϕ1,2 against the subject’s ground-truth
azimuth gives only a weak correlation (r = +0.17, ρ = +0.17,
p < 10−5, N = 4032). The learned phase carries little reli-
able directional information. MM-Fi’s constrained geometry
(azimuth std ∼12◦, root X/Y std ≤7 cm) under-samples the
angular range and therefore limits the power of this test; even
CSI
encoder
BP-LQ
(pre-SGA)
post-SGA
points
0.0
0.2
0.4
0.6
0.8
1.0
linear decodability R2
SGA refines
+0.23
position R2
pose R2
spread
0.0
0.1
0.2
0.3
0.4
0.5
feature spread (per-dim std)
3× compression
(bottleneck)
Fig. 4.
Stage-1 module roles (RePos-D on Person-in-WiFi-3D, where the
root spans 2.7×1.9 m). Linear decodability of pose (R2, bars) and abso-
lute position (R2, gray line), with feature spread (orange). BP-LQ is a
3× token-compression bottleneck (pose decodability dips and spread drops
0.47→0.14), while SGA refines pose the most (+0.23). Position stays fully
encoded throughout, as the direct variant intends.
4
6
8
Subject azimuth, atan2(x, z) [d

## experiments
A. Dataset and Evaluation Setup
We evaluate on two WiFi benchmarks. Our primary, cross-
environment benchmark is MM-Fi [45], a multi-modal dataset
with 40 subjects, 27 activities, and 4 indoor environments
(E01–E04); WiFi CSI is captured with a 1×3 multiple-input
multiple-output (MIMO) array across 114 subcarriers, and
17-joint ground truth comes from stereo infrared cameras.
For the in-domain regime we additionally use the single-
environment Person-in-WiFi-3D benchmark [9] (Section V-C).
Following Setting 3 (cross-environment), we train on E01–E03
and test on E04, under three activity protocols: Protocol 1
(14 daily activities), Protocol 2 (13 rehabilitation activities),
and Protocol 3 (all 27 activities). To further verify that the
observed gains are not specific to one target environment, we
additionally report leave-one-environment-out (LOEO) cross-
validation (Section V-F) and a leakage-free, subject-disjoint
few-shot transfer comparison (Section V-G).
We report three metrics following [10], [21]. MPJPE (mm)
is the mean per-joint position error and reflects both pose accu-
racy and localization. PA-MPJPE (Procrustes-aligned MPJPE)
reflects body configuration independent of global translation,
TABLE II
IN-DOMAIN RESULTS ON PERSON-IN-WIFI-3D (MM). ALL METHODS USE
WIFI CSI ONLY. LOWER IS BETTER; BOLD = BEST.

## related_work
A. WiFi-Based Human Pose Estimation
WiFi-based human pose estimation recovers 2D or 3D body
pose from the CSI of commodity transceivers [17], [18],
offering a privacy-preserving alternative to cameras. Early
methods regress pose from CSI with convolutional or recurrent
networks [19], [20], and recent advances mainly improve
the network architecture through transformers, graph neural
networks, and convolutional backbones [10], [21], [22]. A
parallel line pursues robustness and cross-environment transfer
through CSI denoising, counterfactual radio frequency (RF)
generation, and geometry-conditioned modeling [8], [16], [23],
[24]. The limitation common to these methods is that they
map CSI to joint coordinates directly, with no intermediate
representation between the signal and the body; even graph-
based models [8], [10] impose their skeleton prior on the
output joints rather than upstream of joint regression. RePos
departs from this direct-regression paradigm by learning a
structured intermediate representation between CSI and body
structure, so that anatomical grouping and skeletal connectivity
shape the features before joints are decoded.
B. Cross-Domain Generalization in WiFi Sensing
Because CSI changes sharply with room geometry, furni-
ture, and transceiver placement, cross-environment robustness
is typically pursued through domain adaptation or domain
generalization. Representative strategies align feature distri-
butions across domains [13], [14], learn domain-independent
representations through counterfactual or causal RF genera-
tion [24], or condition on calibrated transceiver geometry to
factor out layout [16]. Others adapt to new users through
transfer learning [7] or learn environment-independent features
via meta-learned one-shot adaptation [25], [26]. A broader
survey catalogues these directions [15]. These approaches
share two limitations for deployment: they act on the input or
feature side, leaving the entanglement between body structure
and absolute position in the output untouched, and their adap-
tation variants further assume access to target-environment
data. RePos instead acts on the output side and uses no
target data, factoring structure from position so that only the
environment-stable component is shared across rooms; it is
therefore complementary to these feature-side methods rather
than a replacement.
C. Relative and Absolute Pose Estimation
Separating what the body is doing from where it is located
is a long-standing idea across pose-estimation modalities.
In vision-based 3D pose estimation, predicting root-relative
pose is standard practice because it decouples body structure
from absolute position [27]–[30], and some methods further
2

separate absolute camera-space localization from the relative
pose [31]. Hierarchical, part-based decomposition of the pose
has likewise proven effective in both vision [32] and iner-
tial measurement unit (IMU)-based [33] settings. WiFi-based
methods, however, have largely overlooked this separation,
regressing absolute coordinates end-to-end [8], [10]: this is
effective within a single environment but, across environments,
ties body structure to environment-dependent position and
transfers poorly. RePos brings the relative/absolute split to
WiFi sensing, predicting root-relative pose and root position
in separate branches so that the environment-stable structure
is never supervised by absolute-position targets.
D. Intermediate Representations in WiFi Sensing
A further line narrows the semantic gap between CSI
and pose by introducing intermediate representations that
inject spatial or structural priors. RF-Pose [34] pioneered
through-wall pose estimation from radio signals via cross-
modal supervision, WiFi Vision [35] unifies sensing, recog-
nition, and detection with commodity WiFi, and recent work
converts temporal CSI into 3D point clouds through trans-
former networks [36], establishing WiFi-to-point-cloud feasi-
bility. Related RF systems instead build intermediate keypoint-
confidence volumes or angle-of-arrival body images before
decoding pose or mesh [37], [38]. The limitation here is one
of resolution: unlike radar and LiDAR (light detection and
ranging) systems that recover body geometry from explicit
point clouds [39]–[42], WiFi CSI offers far coarser spatial
resolution (∼6 cm wavelength at 5 GHz vs. ∼4 mm at 77 GHz)
and lacks the explicit range–Doppler representation of radar,
so committing to an explicit 3D point cloud is ill-posed. RePos
therefore relies on BP-LQs, anatomically grouped latent tokens
that impose body-part structure without committing to 3D
coordinates.
III. PROBLEM FORMULATION
We formulate WiFi-based 3D pose estimation as a deploy-
ment problem, where the primary challenge is generalization
beyond the training environment. As Fig. 1 shows, commodity
WiFi already present in a space is reused to estimate 3D pose
for downstream mobile applications, and the deployed model
must run, unchanged, in spaces it never saw during training
and with no per-environment calibration. The input is a short
window of commodity WiFi CSI, represented as a complex
tensor X over frames, antennas, and subcarriers, from which
we use the amplitude component Xamp. The output is the
absolute 3D body pose Jabs ∈R17×3 of the person in the
sensing area, given in room coordinates.
A. Cross-Environment Deployment Objective
A WiFi sensing channel is determined by the room layout,
the furniture, and the transmitter–receiver geometry, so the
same pose produces different CSI in different rooms [7], [8].
We therefore separate the environment in which a model is
trained from the one in which it runs. Training uses a set
of source environments Esrc, each providing CSI windows
WiFi CSI
CSI
Encoder
Direct
Regression
Absolute
Pose
WiFi CSI
BP-LQ
Relative
Pose
ASPN
Root
Position
Absolute
Pose
Existing
Ours
Fig. 2. Comparison between existing end-to-end methods and the proposed
RePos framework. RePos factors the problem into a root-relative pose stage
and an absolute root-position stage.
paired with ground-truth pose. Deployment runs in a target
environment e⋆/∈Esrc for which no labels are available, and
the model receives only CSI (Fig. 1). A method is admissible
only if it requires no calibration measurement, no fine-tuning
step, and no target-domain data at deployment time. The
parameters fixed on Esrc are the parameters that run in e⋆. The
objective is to learn a predictor fθ : X 7→Jabs that minimizes
the expected pose error in the unseen target environment,
min
θ
E(X,Jabs)∼e⋆
fθ(X) −Jabs
,
(1)
while θ is estimated only from Esrc. The expectation defines
the deployment goal only. The target pose Jabs is unavailable
during deployment. This is the setting realized by MM-Fi
Setting-3 (train on E01–E03, test on the unseen E04), and
it is the bottleneck that distinguishes deployable WiFi sensing
from in-room accuracy.
B. Pose Factorization
The difficulty in Eq. (1) is that the two quantities packed
into Jabs generalize differently. Body structure is primarily
determined by human kinematics and is expected to transfer
across rooms, whereas the absolute location of that body
is read out from how the room shapes the channel and is
environment-dependent. A predictor that regresses Jabs in one
piece must fit both at once, and the position statistics of the
source rooms leak into its structure estimate, a coordinate-
overfitting failure mode also noted by Jia et al. [16]. We adopt
instead an output-side factorization: predict the root-relative
pose Jrel ∈R17×3 and the root position r ∈R3 with separate
predictors and recombine them,
Jabs = Jrel + r,
(2)
where r denotes the pelvis location and is broadcast to all
joints.
The structure predictor is trained only against root-relative
targets, so it receives no absolute-position gradient, substan-
tially reducing the coordinate-overfitting pathway. We treat
Jrel as the environment-stable factor and r as the environment-
dependent one, predicting r from a separate branch that
acts as a learnable spatial prior over position (Fig. 2). The
probabilistic reading of this split, the extent to which the two
3

WiFi CSI
(a)
[B, T=5, 3, 3, 114]
CSI
Encoder
CNN+Transformer
(b)
BP-LQ
Skeleton
Graph
Attention
4 layers,
intra/inter-part
(c)
Part Features
Head Torso L Arm
R Arm L Leg R Leg
attention pooling
Part-to-
Joint
Decoder
GCN+Transformer
Relative
Pose
Jrel ∈R17×3
Absolute
Pose
Jabs ∈R17×3
[B, L, 256]
[B, 150, 256]
[B, 150, 256]
[B, 6, 256]
CSI
Amplitude
Xamp
Latent
Encoder
CNN →tanh
Spatial
Decomp.
beamforming + IFFT
Spatial
Heatmap
(d)
M∈R90×64
Root
Regressor
CNN+MLP,
XY/Z heads
Root
Position
r∈R3
[B, 9, 114, T]
˜H=amp⊙ejπˆϕ
[B, 1, 90, 64]
[B, 512]
Absolute
Pose
Jabs =Jrel+r (RePos)
Stage 1: Structured Pose Decoder / RePos-D
Stage 2: ASPN (Amplitude-based Spatial Prior Network)
RePos
Fig. 3. Overview of RePos. Stage 1 (top) is the Structured Pose Decoder: WiFi CSI (a) is encoded and transformed into BP-LQs (b), refined by SGA and
aggregated in