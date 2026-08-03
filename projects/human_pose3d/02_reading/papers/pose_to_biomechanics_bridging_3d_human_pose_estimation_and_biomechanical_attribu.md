# Pose-to-Biomechanics: Bridging 3D Human Pose Estimation and Biomechanical Attribute Prediction

> 2026 · id: W7167912662 · arXiv: 2607.08725 · pdf: https://arxiv.org/pdf/2607.08725 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent progress in 3D human pose estimation has made markerless recovery of
skeletal motion increasingly accurate and scalable. However, most pose estima-
tors remain optimized for geometric keypoint accuracy, while many real-world
applications in rehabilitation, sports science, ergonomics, and clinical move-
ment analysis require biomechanical quantities that describe how the body
moves, loads, and activates. In this work, we propose BioModule, a lightweight
plug-in temporal transformer that attaches downstream of any 3D pose estima-
tor and predicts biomechanical attributes from standard 17-joint 3D skeletons.
BioModule is estimator-agnostic and requires no modification of the upstream
pose model, enabling existing pose estimators to be extended toward physically
interpretable motion analysis.
To train and evaluate BioModule, we construct a large-scale aligned dataset pair-
ing Human3.6M video and 3D keypoints with the biomechanical label space of
Human3.6Mplus. We establish and verify anatomical correspondence between
coordinate systems of the two datasets, enabling frame-accurate cross-modal
supervision. Using this aligned supervision, BioModule predicts biomechanical
quantities. We further benchmark BioModule across seven state-of-the-art 3D
pose estimators, providing the first systematic analysis of how upstream pose
estimation quality propagates to downstream biomechanical prediction fidelity.
The results position BioModule as a compact, modular bridge between vision-
based pose estimation and biomechanically meaningful human motion analysis.
The complete source code and additional qualitative results are available at:https:
//utsa-virlab.github.io/BioModule/
Keywords: Human Pose Estimation, Vision-based biomechanics, Markerless
biomechanics, Musculoskeletal model
1
arXiv:2607.08725v1  [cs.CV]  9 Jul 2026

## introduction
We propose a pipeline that maps monocular RGB video to a biomechanical state repre-
sentation of the human body through two decoupled stages, as shown in Figure 1. First,
a 3D human pose estimator reconstructs a temporally ordered sequence of 3D skeletal
joint positions from the input video. Second, BioModule receives the root-centred
3D pose sequence and predicts biomechanical attributes derived from the H3.6Mplus
musculoskeletal simulation labels [11]. This separation allows BioModule to operate
downstream of different 3D pose estimators without architectural changes. BioMod-
ule is trained from scratch using ground-truth 3D poses and H3.6Mplus biomechanical
labels.
Let Pt ∈RJ×3 denote the 3D skeleton at frame t, where J=17 joints follow the
Human3.6M joint convention. The input to BioModule is the described sequence of
Euclidean 3D joint coordinates. To remove global translation, each frame is centered
at the pelvis joint pt,0 ∈R3:
¯Pt = Pt −1 p⊤
t,0
(1)
The centred pose is then flattened as xt = vec(¯Pt) ∈R51. The resulting input window
is
X = [x1, . . . , xW ] ∈RW ×51
(2)
6

where W is the temporal receptive field. In this work, W=81, which corresponds
to approximately 1.62 seconds at 50 fps. BioModule receives raw meter scale, root
centered 3D joint coordinates.
BioModule predicts C=17 biomechanical attributes { ˆYa}C
a=1, where
ˆYa
∈
RB×W ×da and da denotes the output dimension of attribute a. These attributes are
organized into three groups according to their interpretation in biomechanical muscu-
loskeletal analysis and their relationship to vision based pose estimation, as defined in
Eq. 3.
The kinematic attributes describe motion geometry and its temporal deriva-
tives. In biomechanics, kinematics refers to motion quantities without directly
considering the forces that caused the motion. In human pose estimation, this group
is closest to the information explicitly represented by pose sequences, because 3D
joint locations encode body configuration over time. In our output set, the kine-
matic attributes are coordinates, speed, and acceleration. Here, there are 2 sets of
coordinates: the output pose of H36M Euclidean joint positions, and the OpenSim
generalized marker coordinates used in H3.6Mplus based on which the degrees of free-
dom for each joint are defined. Therefore, the model must learn the mapping from
Human3.6M joint positions to OpenSim generalized marker coordinates as well. This
mapping depends on the OpenSim skeleton, joint degrees of freedom constraints,
coordinate conventions, and the anatomical model used to generate the H3.6Mplus
labels.
The kinetic attributes describe force and load related quantities. In biome-
chanics, kinetics refers to the quantities associated with producing or constraining
movement. In pose estimation research, these quantities are usually not predicted
directly, because standard pose benchmarks evaluate geometric joint accuracy rather
than physical loading. In our output set, the kinetic attributes are active torque,
passive torque, ideal torque, instantaneous power, instantaneous power raw, ground
reaction, seat reaction, and touch. The touch attribute is binary and represents
whether or not the right and left foot have contact with floor.
The neuromuscular attributes describe quantities associated with excitation,
activation, actuator scaling, and torque generation capacity which are derived from
biomechanical simulations. The term neuromuscular in the sense is used in neuromus-
culoskeletal modeling, where neural excitation and activation dynamics are linked to
muscle force and joint torque generation [5, 6]. In our output set, the neuromuscu-
lar attributes are activation signal, excitation signal, normalized active torque, angle
scaling, velocity scaling, and maximum joint torque. This grouping is used both to
interpret the predicted outputs and to define the weighted multitask objective and is
further used in the equations 15, 16, 17, and 18.
7

Fig. 1 End to end vision to biomechanics pipeline and BioModule architecture. A monocular video
sequence is first converted into a temporally ordered 3D kinematic pose sequence by a 3D human pose
estimator. BioModule then receives the root centred pose sequences, embeds the frame vectors into a
latent space, and processes the temporal window with multiple layers of transformer encoders. Inde-
pendent prediction heads estimate kinematic, kinetic, and neuromuscular biomechanical attributes,
which are optimized using a tiered multi task loss with weights 1.0, 0.5, and 0.3, respectively.
Akin = {coordinate, speed, acceleration},
Aknt = {active torque, passive torque, ideal torque, instantaneous power,
instantaneous power raw, ground reaction, seat reaction, touch},
Anmsc = {activation signal, excitation signal, normalized active torque,
angle scaling, velocity scaling, maximum joint torque}.
(3)
3.2 BioModule Architecture
BioModule is a temporal transformer module that maps a sequence of root centred
3D skeletons to biomechanical attributes. The model has three main components: a
framewise pose embedding, a temporal transformer encoder, and a set of independent
attribute prediction heads. Figure 1 gives an overview of the architecture.
8

3.2.1 Pose Embedding and Temporal Encoding
Each input frame vector xt ∈R51 is projected into a hidden representation using a
shared linear embedding:
et = We xt + be
et ∈Rd
(4)
where d=256. Since self attention does not encode temporal order by itself, we add a
fixed sinusoidal positional encoding [17]:
PE(t, 2i) = sin

t
100002i/d

PE(t, 2i+1) = cos

t
100002i/d

(5)
The transformer input is then
z(0)
t
= et + PE(t)
Z(0) ∈RB×W ×d
(6)
The positional encoding is fixed and introduces no additional trainable parameters.
3.2.2 Temporal Transformer Encoder
The temporal encoder contains L=4 transformer layers with pre layer normaliza-
tion [26]. Each layer applies multi head self attention across the full temporal window,
followed by a feed forward network. For layer ℓ, the update is:
˜Z(ℓ) = Z(ℓ−1) + MHA

LN

Z(ℓ−1)
(7)
Z(ℓ) = ˜Z(ℓ) + FFN

LN

˜Z(ℓ)
(8)
The attention block uses h=8 heads with per head dimension dk=32. Attention is
bidirectional over the full window. No causal mask is applied because BioModule pre-
dicts the biomechanical state associated with the centre frame rather than forecasting
an unseen future frame. This allows the representation to use both preceding and
following motion context.
The feed forward network uses a four times hidden expansion with GELU
activation:
FFN(u) = W2 GELU(W1 u + b1) + b2
(9)
where W1 ∈R4d×d and W2 ∈Rd×4d. Dropout with probability p=0.1 is used in the
encoder. After the final transformer layer, a layer normalization operation produces
the encoded sequence:
H = LN

Z(L)
H ∈RB×W ×d
(10)
9

3.2.3 Biomechanical Attribute Prediction Heads
The encoded sequence H is passed to C=17 independent attribute prediction heads.
Each head ga is a two layer MLP applied framewise:
ˆYa = ga(H)
ˆYa ∈RB×W ×da
(11)
For each attribute, the head has the form
Linear(d →d/2) →GELU →Dropout(0.1) →Linear(d/2 →da)
The shared encoder learns a temporal representation of skeletal motion, while the
separate heads allow each biomechanical attribute to learn its own mapping from that
representation. Continuous outputs are trained and predicted in normalized space. At
inference, they are converted back to physical units using
ˆy = ˜y σa + µa
(12)
where µa and σa are the training set mean and standard deviation for the corre-
sponding attribute dimension. The binary touch head outputs logits, and a sigmoid is
applied at inference to obtain contact probabilities.
3.3 Weighted Multi Task Loss
BioModule is trained in a per-joint manner over 17 biomechanical attributes with
different physical meanings, dimensionalities, and levels of uncertainty. A direct sum
over all outputs would make the objective sensitive to the number of dimensions and
noise level of each attribute. To reduce this effect, we use a tiered weighted multi task
loss. Each attribute belongs to exactly one of the three biomechanical groups defined
in Eq. 3.
For each continuous attribute a, the loss is the mean squared error over the full
output tensor:
La =
1
B W da
B
X
b=1
W
X
t=1
da
X
j=1
 ˆya
b,t,j −ya
b,t,j
2
(13)
For the binary foot contact attribute, we use binary cross entropy with logits:
Ltouch = BCEWithLogits

ˆYtouchYtouch

(14)
The three group losses are computed by averaging the individual attribute losses
within each group:
¯Lkin =
1
|Akin|
X
a∈Akin
La
(15)
¯Lknt =
1
|Aknt|
X
a∈Aknt
La
(16)
10

¯Lnmsc =
1
|Anmsc|
X
a∈Anmsc
La
(17)
The total training objective is:
Ltotal = 1.0 ¯Lkin + 0.5 ¯Lknt + 0.3 ¯Lnmsc
(18)
The weights reflect the relevance of each group to the observed pose sequence.
Kinematic attributes receive the highest weight because they are most directly con-
strained by skeletal motion. Kinetic attributes receive an intermediate weight because
torques, powers, and reaction forces depend on

## method
Act.Sig. Exc.Sig. N.Act.T. Ang.Sc. Vel.Sc. MaxJT.
Touch
MHFormer
0.069
0.062
0.064
0.070
0.159
4.635
67.600%
TCPFormer
0.071
0.067
0.065
0.070
0.167
5.809
68.600%
PoseMamba
0.069
0.062
0.064
0.070
0.158
4.791
67.600%
VideoPose3D
0.069
0.063
0.064
0.070
0.160
6.136
68.300%
MotionAGFormer
0.071
0.067
0.065
0.070
0.167
5.026
68.200%
KTPFormer
0.068
0.061
0.063
0.067
0.156
3.874
67.500%
D3DP
0.067
0.061
0.063
0.067
0.155
4.278
68.600%
inputs. In contrast, kinetic and neuromuscular quantities are more sensitive to subtle
changes in joint angle, velocity, and temporal coordination. Thus, the performance gap
among upstream pose estimators becomes more visible as the target variable moves
14

from geometric motion description toward physically and physiologically interpretable
quantities.
The comparison across the seven pose estimators further demonstrates that pose-
estimation accuracy alone is not sufficient to fully explain biomechanical prediction
quality. A pose estimator that performs well in terms of joint localization may still
introduce errors that are biomechanically meaningful, such as inconsistent knee flex-
ion, unstable hip orientation, or unnatural ankle positioning during walking. These
errors may have limited impact on conventional pose metrics but can strongly affect
torque and muscle-related predictions. This finding supports the central motivation of
BioModule: biomechanical evaluation requires attention not only to where the joints
are located, but also to whether the estimated motion preserves physically meaningful
relationships among body segments.
Another important observation is that BioModule remains functional across all
evaluated upstream models, which supports its estimator-agnostic design. Since
BioModule operates on standard 17-joint 3D skeletons, it does not require retrain-
ing or redesigning the original pose-estimation models. This makes the framework
practical for comparing different pose estimators. At the same time, the variation in
performance across models shows that modularity does not eliminate the influence
of upstream error. Instead, the downstream structure of this research makes that
influence measurable. The results therefore provide not only a benchmark of BioMod-
ule performance, but also an analysis of how pose-estimation quality propagates into
biomechanical inference.
5.2 Qualitative Results
Skeletal visualizations of four biomechanical attributes from test subject S9 during
a walking sequence are presented in Figure 2. The figure compares the BioModule
outputs obtained from ground-truth 3D poses and from seven upstream 3D pose
estimators for the same representative frame. The red spectrum indicates the value
magnitude of the corresponding attribute across body segments.
The walking frame provides an informative case because gait involves coordinated
loading across the hip, knee, and ankle, with different muscles and torques contributing
at different phases of the movement.In the ground-truth visualization, higher biome-
chanical responses are concentrated around the lower-limb joints, particularly the hip,
knee, and ankle, reflecting their dominant role in generating and controlling walk-
ing motion. This spatial distribution serves as a reference for assessing whether the
predicted biomechanical patterns remain anatomically plausible.
For active torque, the qualitative results are expected to reveal how well each pose-
estimator input preserves the joint-level loading pattern of the walking frame. Active
torque reflects the net muscular effort required to generate or control motion. Models
that produce less accurate pose input with local distortions around the lower limbs
may shift the predicted torque magnitude or produce unnatural concentration at the
wrong joint.
Unlike active torque, passive torque is strongly related to joint configuration and
soft-tissue resistance and hence more sensitive to joint-angle errors. If an upstream
pose estimator produces excessive or insufficient knee flexion in the walking frame,
15

GT
D3DP
KTPFormer
MHFormer
MotionAG
Former
PoseMamba TCPFormer VideoPose3D
Active Torque
7.21
10.24
9.23
9.18
5.20
4.97
1.60
9.91
Passive Torque
0.65
1.00
0.96
1.24
0.94
1.44
0.92
1.07
Muscle Activation
0.06
0.14
0.14
0.13
0.17
0.09
0.15
0.13
Neural Excitation
0.09
0.03
0.03
0.04
0.11
0.05
0.11
0.03
Fig. 2 Qualitative comparison of BioModule predictions for a sample walking frame from test subject
S9. Rows show four biomechanical attributes. Columns compare predictions obtained from ground-
truth 3D poses and from seven upstream 3D pose estimators. Bone coloring encodes the predicted
attribute magnitude and the number above each panel reports the mean predicted value across all
body segments.
16

the predicted passive torque may become exaggerated or suppressed relative to the
ground truth. This makes passive torque a useful qualitative indicator of whether the
estimated skeleton remains within plausible biomechanical ranges.
Muscle activation is not directly visible from the skeleton, but it is inferred from
the relationship between posture, motion, and the learned biomechanical supervision.
Scattered or misplaced activation patterns may indicate that the input skeleton lacks
the temporal coherence needed to support reliable muscle-level inference.
Neural excitation is expected to be among the more challenging outputs because
it represents a control signal rather than a directly observable geometric quantity.
Qualitative differences in neural excitation therefore provide insight into the limits
of downstream inference from 3D skeletons alone. If the predicted excitation pat-
terns remain spatially and functionally consistent with the ground truth, this supports
the ability of BioModule to infer higher-level biomechanical attributes from pose
sequences. If the predictions become noisy or anatomically inconsistent, this suggests
that some neuromuscular quantities may require richer input representations, stronger
temporal modeling, or additional physical constraints.
The qualitative results are expected to show that the best-performing upstream
models do not simply produce cleaner skeletons; they preserve biomechanically mean-
ingful structure which also depends highly on the action scenario. When the upstream
pose contains local errors, the downstream biomechanical maps may amplify those
errors in different manners.
The visual patterns in Figure 2 support the quantitative findings. They show that
the downstream formulation makes biomechanical consequences of pose-estimation
error visible. Rather than treating all pose errors as equally important, the visual-
izations reveal which errors matter more for physical interpretation. This distinction
is central to the purpose of BioModule: to connect pose-estimation outputs with
biomechanical meaning.

## experiments
5.1 Quantitative Results
The quantitative results indicates the influence of the quality and temporal consis-
tency of the upstream 3D pose sequence on biomechanical prediction accuracy. Across
the evaluated models, BioModule generally produces more reliable estimates when the
input skeletons preserve anatomically plausible joint relationships and stable temporal
motion patterns. This trend is expected because the predicted biomechanical vari-
ables are not independent frame-level labels. They are consequences of coordinated
motion over time. Therefore, even when two pose estimators have similar average
joint-position errors, their downstream biomechanical predictions may differ if one
estimator produces smoother trajectories, more consistent limb orientations, or fewer
local joint distortions.
The results also indicate that errors do not propagate uniformly across all biome-
chanical targets. Kinematic-related outputs are generally more directly tied to the
observed skeletal geometry and therefore tend to be more stable across pose-estimator
13

Table 1 Biomechanical attributes’ MAE with the frozen weights protocol (RF = 81), evaluated on
subjects S9 and S11 from Human3.6M. The testing on the ground-truth 3D poses obviously plays
as the upper bound to those of 3D poses elicited from pose estimation models.
Kinematic
Kinetic
3D Pose / Model Coord. Speed Accel.
Act.T.
Pass.T. Ideal T. Inst.P. Inst.P.r
GRF
Seat R.
H36M GT
0.228
0.193
1.240
9.529
5.452
4.420
3.516
7.235
28.900
21.700
MHFormer
0.658
0.403
2.377
15.400
6.927
10.600
4.502
7.524
39.000
68.400
TCPFormer
0.534
0.587
4.290
16.900
6.536
12.300
7.002
8.080
37.500 42.900
PoseMamba
0.655
0.328 1.938
16.000
7.008
10.800
4.120
8.047
38.600
105.400
VideoPose3D
0.661
0.397
2.344
15.400
6.953
10.600
4.476
7.704
38.900
68.000
MotionAGFormer
0.530
0.594
4.318
16.900
6.515
12.300
7.037
8.062
37.500
43.000
KTPFormer
0.658
0.399
2.343
15.400
6.939
10.600
4.484
7.537
39.100
67.100
D3DP
0.656
0.400
2.346
15.400
6.937
10.600
4.500
7.512
39.100
66.600
Neuromuscular
Binary
3D Pose / Model Act.Sig. Exc.Sig. N.Act.T. Ang.Sc. Vel.Sc. MaxJT.
Touch
H36M GT
0.058
0.056
0.057
0.054
0.117
2.049
67.500%
MHFormer
0.189
0.124
0.101
0.217
0.187
16.100
43.700%
TCPFormer
0.174
0.168
0.090
0.150
0.225
14.300
41.800%
PoseMamba
0.159
0.089
0.103
0.212
0.175
16.200
42.400%
VideoPose3D
0.190
0.124
0.101
0.218
0.186
16.200
43.600%
MotionAGFormer
0.174
0.167
0.090
0.150
0.226
14.300
42.100%
KTPFormer
0.190
0.124
0.101
0.218
0.186
16.100
43.700%
D3DP
0.190
0.124
0.100
0.218
0.186
16.100
43.600%
Table 2 Biomechanical attributes’ MAE with the fine-tuned protocol (RF = 81), evaluated on S9
and S11 from Human3.6M. Fine-tuning done through 10 epochs on the estimator’s training subject
poses (S1–S8) at lr = 10−5.
Kinematic
Kinetic

## related_work
2.1 Deep Learning-Based Human Pose Estimation
Human pose estimation has progressed from direct coordinate regression in single
images to increasingly structured models that exploit spatial, temporal, and kinematic
priors. Early deep learning approaches such as DeepPose formulated pose estima-
tion as direct regression from image evidence to body-joint locations [12]. Subsequent
methods improved localization by using convolutional heatmap representations and
multi-stage refinement, including convolutional pose machines [13], stacked hourglass
networks [14], and high-resolution representations such as HRNet [15]. These methods
established strong 2D pose detectors that later became the input backbone for many
monocular 3D human pose estimation systems.
A major line of 3D pose estimation first estimates 2D joints and then lifts them into
3D space. Martinez et al. showed that a simple fully connected residual network can be
highly effective for 2D-to-3D pose lifting when accurate 2D detections are available [2].
VideoPose3D extended this formulation with temporal convolutional networks, using
motion context across frame windows to improve robustness against depth ambiguity
and 2D detection noise [3]. Large-scale motion-capture datasets such as Human3.6M
have provided the standard benchmark for evaluating these models in controlled indoor
settings [1].
More recently, transformer-based models have become prominent for video-based
3D pose estimation. PoseFormer introduced a spatial-temporal transformer for mod-
eling joint relations within frames and temporal dependencies across frames [16],
building on the general attention mechanism introduced in Transformer architec-
tures [17]. Subsequent methods improved efficiency, ambiguity handling, and motion
representation. MHFormer introduced a multi-hypothesis transformer to address
monocular depth ambiguity [18], PoseFormerV2 used frequency-domain representa-
tions for efficient temporal modeling [19], and MotionAGFormer combined graph-
based skeletal structure with attention mechanisms [20]. More recent models further
explore diffusion-based pose aggregation [21], kinematic and trajectory priors [22],
implicit temporal pose proxies [23], state-space sequence modeling [24], and general
3

human motion representations [25]. Pre-layer normalization has also been shown to
improve transformer training stability [26].
These methods show that temporal and structural modeling are essential for accu-
rate 3D pose recovery. However, their objective remains primarily geometric: the
output is a 3D skeleton or pose representation optimized by joint-position error. Even
when such models implicitly encode motion dynamics, they do not directly supervise
or evaluate biomechanical quantities such as torques, reaction forces, contact, acti-
vation, or excitation. BioModule uses the 3D skeletons produced by such estimators
as input, but evaluates them through biomechanical prediction fidelity rather than
geometric accuracy alone.
2.2 Biomechanically Accurate Body Models and Motion
Representations
Beyond sparse keypoints, parametric body models provide richer representations of
human shape and pose. SMPL introduced a learned skinned body model that rep-
resents human body shape and articulation with a compact parameterization [27].
Image- and video-based mesh recovery methods such as HMR and VIBE estimate body
pose and shape from monocular visual input, enabling temporally coherent reconstruc-
tion of human motion in mesh space [28, 29]. These methods provide a more complete
geometric representation than a sparse 3D skeleton and have become widely used in
human motion analysis.
Recent work has further sought to make body models more anatomically and
biomechanically meaningful. Keller et al. introduced SKEL and BioAMASS to connect
surface body models with a biomechanically grounded skeleton [9]. Xia et al. recon-
structed humans with biomechanically accurate skeletons, further emphasizing the
importance of anatomical structure in human reconstruction [10]. SKEL-CF extends
this direction through coarse-to-fine recovery of biomechanical skeleton and surface
mesh representations [30]. These works are important because they move beyond
visual surface reconstruction toward body representations that are more consistent
with human anatomy.
However, body models and mesh recovery methods primarily address how the body
is represented or reconstructed. They do not directly provide a general estimator-
agnostic mechanism for converting the standard 17-joint outputs of existing 3D pose
estimators into a broad biomechanical state space. BioModule is complementary to
these approaches: rather than proposing a new body model, it learns to infer kinematic,
kinetic, contact, and neuromuscular attributes from the sparse skeleton representation
already produced by contemporary 3D pose estimators.
2.3 Markerless Biomechanics and Video-to-Biomechanics
Pipelines
Biomechanical movement analysis traditionally relies on marker-based motion cap-
ture, force plates, and musculoskeletal modeling to estimate kinematics, kinetics, and
muscle-related quantities. Recent markerless systems have attempted to reduce this
dependence on laboratory instrumentation. OpenCap uses videos from smartphones to
4

estimate human movement kinematics and dynamics through a pipeline that combines
pose estimation, musculoskeletal modeling, and simulation [7]. OpenCap Monocu-
lar further extends this direction toward single-video biomechanical analysis [31].
OpenCapBench explicitly frames the gap between pose estimation and biomechan-
ics by evaluating whether pose-estimation outputs preserve biomechanically relevant
correctness, not only geometric accuracy [4].
Other video-to-biomechanics and markerless motion-capture studies examine dif-
ferent parts of this pipeline. Cotton et al. studied trajectory optimization and inverse
kinematics for biomechanical analysis of markerless motion-capture data [32]. Ruescas-
Nicolau et al. investigated keypoint augmentation for markerless motion capture in
biomechanical applications [33]. Auer et al. evaluated markerless motion capture com-
bined with musculoskeletal models for kinematic analysis [34], while Barzyk et al.
studied smartphone-based markerless capture of lower-limb joint angles during coun-
termovement jumps [35]. Rode et al. assessed monocular human pose estimation
models for clinical movement analysis [36]. Together, these studies show that mark-
erless motion capture is increasingly relevant for biomechanics, rehabilitation, sports
science, and clinical assessment.
Recent methods also integrate vision models more directly with biomechanical con-
straints. BioPose estimates biomechanically accurate 3D pose from monocular video
by combining mesh recovery with biomechanical constraints [8]. Lin et al. used biome-
chanical models and synthetic training data to estimate 3D kinematics from video [37].
Miller et al. showed that integrating machine learning with musculoskeletal simula-
tion can improve OpenCap video-based dynamics estimation [38]. These approaches
demonstrate the value of combining learned visual representations with biomechanical
modeling.
Most existing markerless biomechanics systems are designed as complete pipelines
involving video processing, pose recovery, trajectory refinement, inverse kinematics,
musculoskeletal modeling, or simulation. BioModule addresses a different setting: given
the 17-joint 3D skeleton output of an existing pose estimator, it learns a compact
temporal mapping to multiple biomechanical attributes. This makes the proposed
framework suitable for comparing different upstream pose estimators under a shared
biomechanical prediction interface.
2.4 Musculoskeletal Simulation, Physics-Informed Modeling,
and Biomechanical Datasets
Musculoskeletal simulation provides the physical foundation for estimating biome-
chanical variables that are not directly visible from video. OpenSim is a widely used
framework for creating and analyzing dynamic simulations of movement [5], and later
extensions support musculoskeletal dynamics and neuromuscular control modeling for
human and animal movement [6]. AddBiomechanics automates model scaling, inverse
kinematics, and inverse dynamics from motion-capture data and musculoskeletal mod-
els [39]. Its associated dataset captures the physics of human motion at scale, providing
a broader source of motion and biomechanical supervision [40].
5

Human3.6Mplus provides another important form of biomechanical supervision by
pairing Human3.6M motion with physically consistent musculoskeletal labels, includ-
ing kinematic, dynamic, and muscle-related quantities [11]. This type of dataset is
critical for learning mappings from pose sequences to biomechanical attributes because
quantities such as joint torques, reaction forces, activation, and excitation are not
directly annotated in conventional computer-vision pose datasets.
Related work has also explored p

## conclusion
This study evaluates BioModule as a modular temporal transformer for predicting
biomechanical attributes from 3D human pose sequences. In this work, biomechan-
ical prediction is treated as a downstream task as a methodological design choice
which allows BioModule to attach to existing 3D pose estimators without modify-
ing their architectures. biomechanical attributes can be learned from pose sequences
alone. In this sense, BioModule provides an initial bridge between the conventional 3D
pose-estimation setting and biomechanical analysis of human motion. As a byprod-
uct,this setting makes it possible to compare how different upstream pose models affect
downstream biomechanical prediction.
Benchmarking of BioModule on the outputs of SOTA 3D pose estimation model
sets an important milestone because most video-based human pose estimation
pipelines produce sparse skeletal representations rather than full musculoskeletal
states. Therefore, by learning the mapping from 3D pose sequences to simulation-
derived biomechanical attributes, BioModule shows that biomechanical interpretation
can be approached directly from video-compatible pose representations.
17

In qualitative walking visualizations, the skeletal maps of active torque, passive
torque, muscle activation provide insight beyond numerical error values by showing
where biomechanical demand is concentrated and whether the predicted distribution
remains consistent with expected movement behavior.
BioModule provides a baseline formulation for connecting standard 3D pose esti-
mation with biomechanical motion analysis. By predicting biomechanical attributes
from pose sequences, it treats the skeleton as an intermediate representation rather
than only a geometric output. This provides an initial step toward video-based biome-
chanical assessment without motion-capture systems, wearable sensors, force plates,
or separate inverse-kinematics pipelines.
Limitations and Future Work: The current study is intended as a baseline
rather than a complete solution for in-the-wild biomechanical analysis. BioModule
relies on aligned biomechanical supervision and therefore inherits limitations from
the underlying simulation-derived labels, musculoskeletal assumptions, and dataset
alignment process. In addition, the use of a reduced 17-joint skeleton improves com-
patibility with standard 3D pose estimators but limits anatomical detail compared
with full-body marker sets or subject-specific musculoskeletal models.
The evaluation is also limited to Human3.6M-based motion data, which is captured
in a controlled environment and does not fully represent outdoor videos, occlusion,
camera motion, clothing variation, clinical movement patterns, or complex sports
activities. Therefore, the present results should be interpreted as evidence that biome-
chanical attributes can be learned from pose sequences under controlled conditions,
and more data including pose and the corresponding biomechanical attributes is
needed to bring the field to the point where validation for unconstrained real-world
deployment is feasible.
Future work should extend this framework to richer skeletal representations,
subject-specific biomechanical modeling, uncertainty-aware prediction, and broader
activity domains. Further validation is needed on real-world video, sports move-
ments, rehabilitation tasks, and clinical populations. A longer-term direction is to
combine video-based pose estimation with BioModule-like biomechanical prediction
so that human motion analysis can be performed without intrusive sensors, expensive
laboratory equipment, or separate inverse-kinematics pipelines.