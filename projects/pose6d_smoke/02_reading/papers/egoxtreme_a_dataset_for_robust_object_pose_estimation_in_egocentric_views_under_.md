# EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions

> 2026 · id: arxiv:2603.25135 · arXiv: arxiv:2603.25135 · pdf: https://arxiv.org/pdf/2603.25135 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

EgoXtreme: A Dataset for Robust Object Pose Estimation
in Egocentric Views under Extreme Conditions
Taegyoon Yoon1
Yegyu Han1
Seojin Ji1
Jaewoo Park1
Sojeong Kim1
Taein Kwon2∗
Hyung-Sin Kim1∗
1Seoul National University
2VGG, University of Oxford
{taegyoun88, yegyuhan, seojinji23, 1qkrwodn1, kvia2230, hyungkim}@snu.ac.kr
taein@robots.ox.ac.uk
Figure 1. EgoXtreme, an egocentric dataset for robust 6D object pose estimation in extreme environments. The dataset provides 775.5
minutes of egocentric RGB video from 15 participants using Aria glasses. As illustrated, it spans three challenging scenarios—Sports,
Maintenance, and Emergency—featuring significant real-world visual degradations such as low light, smoke, and motion blur.
Abstract
Smart glass is emerging as an useful device since it pro-
vides plenty of insights under hands-busy, eyes-on-task sit-
uations. To understand the context of the wearer, 6D ob-
ject pose estimation in egocentric view is becoming essen-
tial. However, existing 6D object pose estimation bench-
marks fail to capture the challenges of real-world egocen-
tric applications, which are often dominated by severe mo-
tion blur, dynamic illumination, and visual obstructions.
This discrepancy creates a significant gap between con-
trolled lab data and chaotic real-world application.
To
bridge this gap, we introduce EgoXtreme, a new large-scale
6D pose estimation dataset captured entirely from an ego-
*Joint supervision and corresponding authors.
centric perspective. EgoXtreme features three challenging
scenarios—industrial maintenance, sports, and emergency
rescue—designed to introduce severe perceptual ambigui-
ties through extreme lighting, heavy motion blur, and smoke.
Evaluations of state-of-the-art generalizable pose estima-
tors on EgoXtreme indicate that their generalization fails
to hold in extreme conditions, especially under low light.
We further demonstrate that simply applying image restora-
tion (e.g., deblurring) offers no positive improvement for
extreme conditions. While performance gain has appeared
in tracking-based approach, implying using temporal infor-
mation in fast-motion scenarios is meaningful. We conclude
that EgoXtreme is an essential resource for developing and
evaluating the next generation of pose estimation models
robust enough for real-world egocentric vision. The dataset
and code are available at https://taegyoun88.github.
io/EgoXtreme/
1
arXiv:2603.25135v1  [cs.CV]  26 Mar 2026

Figure 2. Visualization of ground truth 6D pose annotations from our dataset, arranged sequentially from top-left to bottom-right.
The frames include the sports scenario (normal and middle light), the industrial maintenance scenario (low light, flashlight with smoke,
and headlight), and the emergency rescue scenario (warning light, exit green light, and high light with smoke).
1. Introduction
Lightweight smart glasses [7] provide a powerful interface
that can continuously capture and interpret the wearer’s ego-
centric view, enabling vision applications precisely when
users cannot free both hands or divert their gaze to a phone.
Examples include tightening a bolt while following a torque
sequence, navigating a dim corridor to locate an extin-
guisher, or executing a high-speed swing.
In these mo-
ments, the camera is rigidly mounted to a moving head; ob-
jects appear at close range and are often truncated by hands
or tools; and illumination is supplied or modulated by the
wearer (headlamp or flash) or by the environment (warning
beacons, exit-sign LEDs).
While 6D object pose estimation is fundamental to
understanding physical activities and environments—and
has achieved strong results on static third-person RGB
datasets [2, 5, 6, 15–17, 20, 42, 46]—egocentric settings in-
troduce unique challenges. In hands-busy, eyes-on-task,
always-moving scenarios, the data distribution departs
from stationary, uniformly lit third-person scenes and in-
stead shifts toward rapid and frequent head motion; a near-
field perspective at very short working distances; severe
truncation at image boundaries; and extreme visual con-
ditions (dynamic or narrow-spectrum illumination, smoke,
and clutter) that induce motion blur and visibility loss. The
very advantages of smart glasses as an interface therefore
create the sensing extremes we must target.
Despite these realities, the field of 6D object pose es-
timation largely relies on controlled, third-person datasets
such as YCB-Video [46], Honnotate [11], LineMod [15] ,
T-LESS [16], all collected under stable lighting and limited
motion. While the H2O [22] and HOT3D [1] introduced
egocentric benchmarks, they do not fully capture the diver-
sity and severity of real-world smart-glasses conditions—
particularly rapid object manipulations and dynamic light-
ing.
In such extremes, severe motion blur significantly
degrades the generalization of existing models. Attempts
to bridge the gap via simulation—synthetic data genera-
tion [4], frame averaging to simulate blur [38], and artificial
brightness variations [24]—remain insufficient: a substan-
tial discrepancy persists between simulated and real ego-
centric scenes. The absence of an egocentric dataset that di-
rectly captures these extreme conditions is therefore a crit-
ical bottleneck for advancing 6D object pose estimation in
egocentric vision.
This work aims to step toward bridging the gap between
controlled laboratory settings and extreme cases that can oc-
cur in real-world usage, analyzing where current pose esti-
mators fail under challenging egocentric conditions. To this
end, we introduce EgoXtreme, a novel dataset specifically
designed to evaluate robustness under dynamic lighting,
smoke, and severe motion blur—conditions that induce sig-
nificant perceptual ambiguity. The dataset comprises video
sequences collected from 15 participants over 775.5 min-
utes, and Figure 2 provides representative sample images
from EgoXtreme.
2

Table 1. Datasets for object pose estimations.
Extreme Condition
Dataset
Instance
Speed
Light
Smoke
Frames
Egocentric
Subjects
Objects
Annotation
LM [15]
No
-
1
No
18.2k
No
-
15
RGB-D
IC-BIN [46]
Yes
-
1
No
177
No
-
2
RGB-D
T-LESS [16]
No
-
1
No
49k
No
-
30
RGB-D
YCB-V [46]
No
Slow
1
No
0.1M
No
-
21
RGB-D
TUD-L [17]
No
-
8
No
62k
No
-
3
RGB-D
HOPE [42]
No
-
5
No
238
No
-
28
RGB-D
IPD [19]
Yes
-
3
No
30k
No
-
20
RGB-D
H2O [22]
No
Slow
1
No
572k
Yes
4
8
RGB-D
HOT3D [1]
No
Slow
1
No
1.5M
Yes
19
33
Mocap
EgoXtreme (Ours)
Yes
Fast
8
Yes
1.3M
Yes
15
13
Mocap
Our dataset presents three distinct scenarios:
• Industrial maintenance: Fine manipulation using tools
like hammers and drills, testing robustness to subtle, pre-
cise motions.
• Sports: Rapidly swung objects (e.g., table tennis rackets,
baseball bats), testing resilience to extreme motion blur
from high-speed rotational and translational movement.
• Emergency rescue: Recreating urgent situations, users
search for and interact with emergency items (e.g., first-
aid kits, extinguishers), testing object detection and track-
ing amid intense camera shake and visual obstructions.
Our extensive evaluation of 6D object pose on EgoX-
treme reveals that current state-of-the-art pose estima-
tors [28, 33, 45]—despite large-scale pretraining aimed
at generalizing to unseen objects in cluttered scenes—are
highly fragile under egocentric extremes.
This indicates
that the challenges captured by EgoXtreme are largely or-
thogonal to those addressed by existing generalizable meth-
ods, exposing a gap in current benchmarks for egocentric
applications.
Specifically, we identify two primary fail-
ure modes: (1) distribution shift induced by dynamic il-
lumination and smoke, and (2) feature loss caused by se-
vere truncation and motion blur. Critically, the degradation
persists even when perceptual image quality improves: vi-
sual restoration (e.g., deblurring, dehazing, and low-light
enhancement) offers little benefit, and model performance
remains significantly reduced.
In contrast, incorporating
temporal information via pose tracking improves accuracy
in dynamic, high-motion scenarios, underscoring the impor-
tance of temporal modeling for 6D pose in egocentric video.
The core contributions of this paper are as follows:
• A large-scale egocentric benchmark for robust pose
estimation: We release EgoXtreme, the first large-scale
egocentric 6D pose estimation dataset collected across
versatile scenarios under extreme conditions, including
severe motion blur, dynamic lighting, and visual obstruc-
tions. This dataset can serve as a challenging benchmark
for smart-glasses applications.
• Comprehensive baseline analysis: We evaluate state-of-
the-art generalizable 6D object pose estimators on EgoX-
treme, analyze their limitations and failure modes in ex-
treme egocentric environments, and study two mitigation
strategies—image restoration and temporal tracking—
demonstrating the critical importance of the latter for ro-
bust performance under severe motion.
2. Related work
This section reviews existing research pertinent to our work,
covering 6D object pose estimation models, relevant bench-
mark datasets, and methodologies aimed at improving per-
formance under adverse conditions such as blur, haze, and
low light.
2.1. Benchmarks for 6D object pose estimation
Standard benchmarks like YCB-Video [46] and LM/LM-
O [2, 15] advanced the field but they only controlled limited
conditions like lighting and motion. Datasets focusing on
specific challenges like multiple instances/symmetry (IC-
BIN [5], IC-MI [41], T-LESS [16]) or static lighting vari-
ations (TUD-L [17], HOPE [42], SenseShift6D [12]) exist,
but lack the dynamic or extreme lighting. Table 1 indicates
the difference between EgoXtreme and the other 6D pose
estimation benchmarks.
6D pose estimation methods are broadly divided into
instance-specific models (e.g., GDRN [44], ZebraPose [40],
HiPose [27]) and zero-shot models. Instance-specific mod-
els are impractical for our egocentric scenario as they re-
quire object-specific retraining for novel objects.
While
zero-shot models offer generalization, RGB-D approaches
(e.g., SAM6D [26], FoundationPose [45]) are often incom-
patible with lightweight smart glasses that lack depth sen-
sors.
We therefore focus on RGB-only zero-shot mod-
els. Two main strategies exist within this category. The
first one is a two-step pipeline where a coarse estimator
like GigaPose [33] and FoundPose [36] retrieves template
3

Table 2. EgoXtreme datasets configuration
Scenario
Lighting condition
Smoke
Object
Speed (m/s)
Standard
Extreme
Camera
Object
normal
middle
high
low
head
flash
warning
green
Maintenance
"
"
"
"
"
"
"
5
0.03
0.09
Sports
"
"
"
"
5
0.40
1.37
Emergency
"
"
"
"
"
"
"
3
0.47
0.10
candidates via feature matching for PnP-RANSAC [9, 25],
and a separate refinement module (e.g., MegaPose [23],
Gen-Flow [30]) iteratively improves the best hypothesis by
aligning dense features.
The second strategy uses inte-
grated models like PicoPose [28], which perform the en-
tire pipeline internally: they first find a single best-matched
template and then use dedicated multi-stage blocks to com-
pute 2D transformations and refine the final pose. Since
our egocentric scenario requires an RGB-only unseen ap-
proach, we analyze the robustness of these state-of-the-art
coarse-to-fine pipelines on our proposed extreme environ-
ment benchmark.
2.2. Egocentric pose estimation datasets
Recently, datasets capturing the first-person perspective
have become crucial for smart glass applications.
How-
ever, a significant portion of this research has centered on
human-centric aspects, such as full-body 3D pose (e.g.,
You2Me [31], EgoBody [48]) or detailed hand pose and
hand-object interactions (e.g., H2O [22]), rather than the
6D pose of general objects in the environment. HOT3D [1],
the most prominent large-scale egocentric object tracking
benchmark, was a significant step in addressing this. How-
ever, it was captured under relatively well-lit conditions and
similarly lacks the extreme motion, challenging illumina-
tion, or visual obstructions like smoke that we target. Con-
sequently, a significant gap remains in evaluating object
pose estimation robustness under the severe motion blur,
dynamic/extreme lighting, and drastic movements typical
of real-world smart glass usage.
2.3. Methodologies for adverse conditions
Various methods aim to enhance image quality under
adverse conditions.
Deblurring techniques range from
GAN-based approaches [21] to recent Transformer [47]
and efficient architectures [3].
Dehazing methods in-
clude classic approaches [13] and deep learning mod-
els [37] or Transformer-based [39].
Low-light enhance-
ment research offers specialized solutions, ranging from
efficient zero-reference methods to multi-task restoration
models. For instance, Zero-DCE [10] formulates enhance-
ment as a fast, image-specific curve estimation task, while
DarkIR [8] jointly addresses low-light, noise, and blurring
Figure 3. 3D models. This image shows the 13 object models
in the EgoXtreme dataset. From top to bottom five maintenance
scenarios, five sports scenarios and three emergency scenarios.
issues by employing advanced attention mechanisms within
efficient CNNs.
While potentially beneficial for detec-
tion/segmentation, the impact of these restoration methods
and potential artifacts on downstream pose estimation accu-
racy, especially in extreme conditions, is not well-studied.
We investigate the effectiveness of applying representative
enhancement techniques as preprocessing for robust 6D
pose estimation on our benchmark.
3. EgoXtreme dataset
EgoXtreme is a novel, large-scale dataset designed for ro-
bust egocentric 6D object pose estimation under extreme
conditions. Specifically, 8 illumination conditions are used
across three scenarios, and smoke is included in specific
scenes. The configurations of whole scenario is summa-
rized in Table 2. These conditions, combined with severe
motion blur, make accurate 6D object pose estimation ex-
tremely challenging.
Comprising approximately 1.3 million frames (775.5
minutes total) captured at 30fps (1650 frames/55 seconds
per sequence), the dataset is split into training (518.8 min),
validation (80.7 min), and test (176 min) sets. The dataset
includes three challenging scenarios:
industrial mainte-
nance (319 min), emergency rescue (165 min), and sports
(291.5 min). Videos were recorded using Aria glasses [7],
providing raw 1408 × 1408 fisheye RGB images and the
4

ARIA
Optitrack
R, t
Optitrack camera
Figure 4. Diagram for data collection.
corresponding undistorted version, enabling evaluation on
both data types. 15 participants were recorded perform-
ing diverse actions and object manipulations. The dataset
features 13 objects, including sports equipment, assembly
blocks, and emergency supplies, with corresponding 3D
CAD models as in Figure 3. Some scenarios involve multi-
ple instances of the same object (assembly blocks and ping-
pong racket), requiring instance-level disambiguation.
3.1. Scenario details
As shown in Table 2, the average speeds of the camera
and the object differ significantly depending on the scenario
which demonstrates the characteristics of each scenario.
Industrial maintenance scenario focuses on estimating
object pose during seated precision tasks involving manip-
ulating tools, including hammer, drill, saw, wrench and
brick, and assembling identical multiple blocks in vari-
ous industrial lighting conditions.
6 lighting conditions
include normal brightness, low light, medium brightness,
high brightness, headlamp, and flashlight. To mimic hazy
workplaces, smoke was introduced in a subset of the data.
This scenario allows for benchmarking model robustness
against fine-grained object movements and diverse illumi-
nation changes, particularly the movable light sources are
included.
Emergency rescue scenario involves searching for and
picking up three types of emergency supplies, such as first
aid kit, extinguisher, and flashlight, while moving through
different lighting environments. 6 lighting conditions com-
prise normal brightness, low light, medium brightness,
high brightness, emergency beacon, and exit sign lighting.
Smoke was added in a portion to simulate fire situations.
This setup evaluates the ability to detect objects and esti-
mate object poses under intense camera shake and restricted
visibility.
Sports scenario features five types of sports equipment, in-
cluding baseball bat, hockey stick, pingpong racket, ten-
nis racket, and golf club, being swung rapidly. To make
it more realistic, actual pingpong gameplay or self-practice
are included. 4 lighting conditions are used in this scenario:
normal brightness, low light, medium brightness, and high
brightness. This scenario evaluates the model’s ability to
handle extreme motion blur caused by high-speed rotational
and translational object movements.
3.2. Data collection and processing
RGB images were captured at 30fps using the Aria glasses,
synchronized with 1000fps SLAM data. A reflective marker
was attached to the Aria glasses, and rigid body marker
clusters were attached to all scenario objects, allowing si-
multaneous recording with a 120fps OptiTrack motion cap-
ture system. This system provides sub-millimeter accuracy
for the 6D pose of both the headset and the objects, serv-
ing as our ground truth (GT). In this setup, both the SLAM
trajectory and the motion capture marker trajectory were
collected concurrently. The trajectories were aligned using
the Umeyama method [43] to unify coordinate frames and
achieve precise temporal synchronization between the Aria
SLAM and the motion capture system. The whole process
is illustrated in Figure 4.
To compensate for SLAM drift that occurred during mo-
tion, a Kalman filter [18] was applied to refine the trajec-
tory alignment. Once the coordinate frames were unified
through this process, various objects tracked by the motion
capture system could be represented in the same global co-
ordinate system, enabling accurate object pose annotation.
Finally, to correct minor temporal misalignments, on the or-
der of a few milliseconds, between the synchronized Aria
timestamp and the true physical capture time of the RGB
image caused by the RGB camera’s exposure delay, a man-
ual time-offset adjustment was performed based on visual
inspection of projection results before final interpolation.
3.3. Dataset configuration
The lighting and environmental conditions within the
dataset are structured for multi-dimensional evaluation of
model robustness, as detailed in Table 2. The conditions
are classified into two main categories: Standard (common
illumination) and Extreme (limited illumination), compris-
ing a total of 8 unique lighting types. The Standard category
includes environments with consistent, ample light sources
such as normal fluorescent light (normal), bright lamp light
(middle), and floodlight illumination (high). Conversely,
the extreme category features setups designed to challenge
pose estimation: low-intensity lighting (low), dynamic il-
lumination coupled with head movement (head, flash), and
critical emergency sources (warning - rotating beacons and
green - emergency exit signs). Furthermore, to accurately
model visual degradation in maintenance and emergency
scenarios, simulated smoke was intentionally introduced us-
ing a fog machine. This smoke was dispersed to maximize
5

Table 3. 6D object pose estimation on EgoXtreme. Performance is evaluated using ADD(S) recall at thresholds of 0.1d, 0.2d, 0.3d (↑)
and spatial accuracy metrics MSSD/MSPD (↓), where d denotes the object diameter.
Scenario
Light
Smoke
FoundPose [36]
GigaPose [33]
PicoPose [28]
0.1d
0.2d
0.3d
MSSD
MSPD
0.1d
0.2d
0.3d
MSSD
MSPD
0.1d
0.2d
0.3d
MSSD
MSPD
Sports
Standard
0.53
1.55
4.72
0.59
7.65
4.12
11.77
24.64
5.21
10.61
3.13
9.48
24.61
5.00
12.87
Extreme
0.18
0.78
2.42
0.23
6.87
3.11
9.19
19.04
4.15
8.31
1.80
6.61
17.86
3.07
10.10
Maintenance
Standard
21.02
30.53
37.61
12.94
18.78
33.64
48.84
62.77
15.97
24.17
39.27
62.42
76.84
24.76
33.20
Extreme
13.78
22.94
30.03
9.92
16.40
19.78
32.20
45.52
10.44
19.89
26.44
47.18
64.09
18.37
28.12
Standard
"
14.44
22.49
30.00
8.23
13.11
23.01
37.87
52.86
12.24
19.83
26.37
46.05
59.87
18.26
25.48
Extreme
"
11.19
18.57
25.63
7.01
13.07
17.56
30.35
45.11
8.92
18.38
20.97
38.50
52.30
14.25
25.53
Emergency
Standard
6.31
11.96
12.88
9.73
13.08
22.03
40.29
46.34
32.08
43.52
22.67
59.11
67.83
46.40
50.81
Extreme
0.10
0.29
0.56
8.27
9.61
9.40
14.75
21.30
12.44
29.02
9.18
27.59
36.23
22.01
31.24
Standard
"
3.52
9.62
12.12
0.21
1.66
16.25
35.28
44.91
28.34
35.99
19.66
61.35
72.82
48.10
50.64
Extreme
"
0.11
0.60
0.76
0.44
2.05
7.07
15.26
21.54
12.00
32.81
9.45
24.19
31.54
19.37
29.62
atmospheric scattering and visual obstruction, ensuring that
objects were not contaminated directly, thereby isolating the
effects of airborne particulates on visibility.
4. Experiments
In this section, we evaluate the three state-of-the-art RGB-
based models and analyze the performance of 6D object
pose estimation under the proposed extreme conditions.
Since real-world scenarios demand high generalizability,
we focus our evaluation on state-of-the-art zero-shot 6D ob-
ject pose estimation models. Specifically, we define the test-
ing pipelines for each as follows: FoundPose [36] is used as
a coarse alignment module with its output refined by Mega-
Pose [23]. GigaPose [33] also utilize the MegaPose as a
refinement module. Conversely, PicoPose [28] is employed
as an integrated coarse-to-fine model performing its own in-
ternal refinement. For evaluation, we explicitly utilize the
official test set. First, we establish baselines on the EgoX-
treme dataset using state-of-the-art 6D object pose estima-
tion models (Sec. 4.1). Second, we investigate how image
restoration affects 6D object pose estimation performance
(Sec. 4.2). Finally, we benchmark and evaluate tracking of
the object pose under severe motion blur (Sec. 4.3).
4.1. Baseline evaluation
We first establish a baseline with recent state-of-the-
art RGB-only zero-shot models, FoundPose [36], Giga-
Pose [33] and PicoPose [28], on the three scenarios of the
EgoXtreme dataset. To rigorously evaluate the pose estima-
tion performance decoupled from detection errors, we uti-
lized GT bounding boxes. For evaluation metrics, we em-
ploy the ADD(-S) [2, 46] recall, along with the standard
BOP metrics [35]: Maximum Symmetry-Aware Surface
Distance (MSSD) and Maximum Symmetry-Aware Projec-
tion Distance (MSPD). End-to-end evaluation results using
CNOS [32] detections are provided in Appendix C1.
As shown in Table 3, the results demonstrate a significant
performance degradation in extreme lighting and smoke
conditions compared to standard light without smoke con-
ditions. Specifically, PicoPose showed a 31.6%p perfor-
mance degradation in the emergency scenario (measured
@0.3d) when comparing the extreme lighting condition to
the standard lighting condition.
Furthermore, in the in-
dustrial maintenance scenario, GigaPose’s performance de-
graded by 9.91%p with the addition of smoke. The sports
scenario characterized by severe object cropping and mo-
tion blur proved to be the most challenging with models
failing to achieve meaningful recall overall. FoundPose ex-
hibited the lowest recall, which we attribute to the brittle-
ness of its direct sparse feature matching approach. Feature
extraction and matching under egocentric cropping and mo-
tion blur frequently failed to secure the minimum number of
correspondences required for PnP-RANSAC [9, 25], result-
ing in a high rate of failure where no 6D pose output was
generated. These baselines show that current state-of-the-
art models are highly vulnerable to real-world challenges,
particularly motion blur and extreme lighting that are not
captured in existing datasets. It highlights the necessity of
the EgoXtreme dataset in filling the gap caused by these
real-world challenges. Figure 5 visualizes pose estimation
results in standard and extreme cases, respectively.
4.2. Object pose estimation with pre-processing
We investigate whether image restoration techniques can
mitigate the performance drop under adverse conditions
such as motion blur, smoke, and dynamic lighting. To this
end, we apply representative preprocessing methods for de-
blurring [3], dehazing [39], and low-light enhancement [8]
directly to the images before running the baseline pose esti-
mator.
The recall values under full lighting and smoke condi-
tions are detailed in Table 4. The result of image restora-
tion techniques ultimately fail to help, and often hurt perfor-
mance. Applying a single preprocessing method generally
did not increase performance, and combining two methods
resulted in an even greater performance drop, decreasing re-
call by approximately 5%p (measured @0.3d) threshold in
6

Figure 5. Example 6D Pose estimation results on baseline models. The red line is prediction and green is GT. (a), (b), and (c) are the
industry maintenance, sports, and emergency rescue scenarios, respectively. The top row indicates standard light condition, and the bottom
row indicates extreme light condition.
Table 4. 6D object pose estimation with pre-processing for Pi-
coPose.
Scenario
Deblur
Dehaze
Light enhance
PicoPose
0.1d
0.2d
0.3d
Sports
2.81
8.80
23.02
"
2.61
8.52
22.24
"
2.87
9.05
22.05
"
"
2.58
8.22
20.99
Maintenance
28.22
48.51
63.32
"
26.28
43.70
57.55
"
24.04
42.75
57.71
"
24.90
43.38
58.08
"
"
23.39
39.73
53.28
Emergency
15.12
42.69
51.70
"
14.13
37.74
45.51
"
4.74
22.49
37.74
"
13.78
38.51
46.75
"
"
13.72
35.58
43.65
the maintenance scenario and 8%p in the emergency sce-
nario.
Specifically, when light enhancement faces extreme con-
ditions such as high-contrast highlights or extremely low
light, these methods introduced significant noise making
prediction more difficult. The current deblurring and de-
hazing methods also showed no meaningful positive ef-
fect. In particular, dehazing method has severely failed in
the emergency scenario yielding an exceptionally low re-
call of only 4.74%p (measured @0.1d). The current dehaz-
ing method introduces significant noise artifacts under non-
uniform smoky conditions. This explains why our emer-
gency scenario characterized by partial smoke was not prop-
erly dehazed, making subsequent pose estimation extremely
difficult. Although human observers may perceive an im-
provement in the preprocessed images of some specific sce-
narios, the model performance decreased as illustrated in
Figure 6. Example 6D Pose estimation results with preprocess-
ing. The top row shows the original, non-preprocessed images.
The bottom row displays the corresponding images after applying
specific preprocessing: deblurring (left), light enhancement (mid-
dle), and dehazing (right).
Figure 6.
This result demonstrates that existing preprocessing
methods are insufficient for 6D object pose estimation on
our dataset and its extreme conditions. This failure under-
scores that the challenges posed by EgoXtreme cannot be
resolved through preprocessing alone and require deeper in-
vestigation using our dataset and benchmark.
4.3. Object pose tracking
For object pose tracking, we compare the tracking-based
approach against per-frame baseline (Sec. 4.1). We focus
on the highly dynamic sports scenarios. We evaluated three
distinct temporal strategies: (1) Direct temporal (using the
full pose of t −1 frame as initial input), (2) Fusion tempo-
ral (combining final rotation pose of t−1 frame with coarse
translation pose of t frame), and (3) Confidence-based hy-
brid temporal (selectively using the temporal prior based
on the current frame’s prediction score).
However, the introduction of the tracking strategy did
7

Table 5. 6D object pose tracking for GigaPose. Applied to sports
normal scenario.
Object
Method
GigaPose
0.1d
0.2d
0.3d
Pingpong
Per-frame
0.53
2.97
17.63
Direct
0.05
0.35
6.58
Fusion
0.56
1.57
10.81
Hybrid
0.49
1.52
16.53
Tennis
Per-frame
6.77
34.71
50.91
Direct
3.91
14.28
22.77
Fusion
5.47
30.64
49.56
Hybrid
6.93
34.34
50.55
Bat
Per-frame
17.95
41.29
60.55
Direct
4.66
9.84
14.29
Fusion
13.68
37.10
60.59
Hybrid
17.66
45.27
64.46
Golf
Per-frame
0.08
1.64
8.36
Direct
0.64
1.49
3.61
Fusion
0.45
5.76
13.98
Hybrid
0.29
4.08
14.35
Hockey
Per-frame
0.46
5.66
18.35
Direct
0.07
3.18
5.66
Fusion
1.57
12.31
24.47
Hybrid
1.08
10.72
26.13
not yield consistent performance gains across all temporal
strategies (Table 5), the dynamic nature of the sports sce-
nario clearly revealed the inherent limitations of direct tem-
poral approaches on our dataset. Specifically, the fast object
speed in the sports scenario caused large inter-frame dis-
placement. Therefore, the simple direct temporal propaga-
tion is inadequate and sometimes adverse to performance,
evidenced by a performance degradation of up to 46%p
(measured @0.3d). Furthermore, the fusion temporal ap-
proach was observed to perform worse than the per-frame
baseline in some cases (e.g., pingpong and tennis). This
failure is due to propagating the previous frame’s rotation,
which leads to inaccurate initial poses that the current frame
measurement struggles to correct. Conversely, the hybrid
temporal strategy successfully mitigated these failures, sig-
nificantly enhancing performance in this high-motion envi-
ronment. This robustness stems from its ability to selec-
tively ignore unstable rotational priors by utilizing the cur-
rent measurement’s prediction score, thereby maintaining
tracking robustness even under severe displacement.
These results strongly suggest that solving pose estima-
tion in extreme dynamic environments requires not merely
simple temporal propagation, but a hybrid temporal strategy
that uses dynamic quality assessment to selectively apply
the temporal prior. This inherent challenge, explicitly high-
lighted by the comparative analysis on our dynamic scenar-
ios, confirms the utility of EgoXtreme as a crucial and nec-
essary benchmark for future research. It serves not only to
evaluate conventional per-frame estimation accuracy, but,
more importantly, to drive the development of robust and
resilient pose tracking algorithms capable of operating reli-
ably under real-world extreme conditions.
5. Conclusion
In this paper, we filled the critical gap between existing 6D
object pose benchmarks and the demanding conditions of
real-world egocentric vision. We introduced EgoXtreme, a
new large-scale 6D pose dataset featuring three challeng-
ing scenarios specifically designed to test model robustness
against severe motion blur, dynamic/low lighting, and vi-
sual obstructions like smoke. Our extensive evaluation of
state-of-the-art RGB-only models revealed their significant
performance degradation on EgoXtreme, confirming that
existing methods lack robustness to these extreme condi-
tions. Our experiments showed that while image restoration
partially enhances visual fidelity, this improvement fails to
translate into meaningful gains for 6D object pose accuracy.
Moreover, we show that incorporating temporal information
through a tracking-based approach improves pose accuracy
and stability in the sports scenario, where motions are ex-
tremely fast. To enable performance improvements relevant
to these demanding real-world scenarios, a proper bench-
mark is required for an accurate evaluation, and EgoXtreme
dataset is specifically introduced to serve this vital role.
Limitations and future works.
While EgoXtreme pro-
vides high-fidelity 6D poses for objects, our methodology
involves a few trade-offs. Firstly, the reliance on the Opti-
Track system for high-accuracy GT confines data collection
to specialized indoor environments, limiting the evaluation
of true outdoor robustness. Secondly, we note the absence
of 3D hand pose annotations. Accurately labeling intricate
hand articulations, particularly due to the extreme motion
blur and speed prevalent in the sports scenario, remains a
significant independent challenge.
We believe, however,
that addressing these gaps presents valuable directions for
future work, especially by generating accurate 3D hand la-
bels through combining our existing motion capture data
with advanced parametric hand models.
Our findings suggest that future research on robust ego-
centric object pose estimation should move beyond single-
frame visual feature representation and focus on effective
temporal modeling and robustness against extreme condi-
tions to handle real-world scenarios. We believe EgoXtreme
will serve as a critical benchmark to drive the development
of the next generation of 6D pose estimation models.
8

Acknowledgment
This work was supported in part by an SNSF Postdoc.
Mobility Fellowship (P500PT 225450), the Institute of
Information & communications Technology Planning &
Evaluation (IITP) grant funded by the Korea government
(MSIT) (No.
RS-2023-00216821), and the National
Research Foundation (NRF) of Korea grant funded by
the Korea government (MSIT) (No. RS-2023-00222663).
We thank Meta for providing the Project Aria glasses
used in this research. We also gratefully acknowledge the
following colleagues for valuable discussions and support
of our project: Keondo Park, Eunsu Baek, Joopyo Hong,
Yoojin Kwon, Subeom Park, Wooseok Lee, Hun Heo,
Seojun Heo, Hongjun Suh, Suahn Bae, Dayeon Woo, Yejun
Ji, Wonjeong Lee, Dongik Park, and Boyeong Im.
References
[1] Prithviraj Banerjee,
Sindi
Shkodrani,
Pierre
Moulon,
Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang
Zhang, Jade Fountain, Edward Miller, Selen Basol, et al.
Hot3d: Hand and object tracking in 3d from egocentric
multi-view videos.
In Proceedings of the Computer Vi-
sion and Pattern Recognition Conference, pages 7061–7071,
2025. 2, 3, 4
[2] Eric Brachmann, Alexander Krull, Frank Michel, Stefan
Gumhold, Jamie Shotton, and Carsten Rother.
Learning
6d object pose estimation using 3d object coordinates. In
Computer Vision–ECCV 2014: 13th European Conference,
Zurich, Switzerland, September 6-12, 2014, Proceedings,
Part II 13, pages 536–551. Springer, 2014. 2, 3, 6
[3] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun.
Simple baselines for image restoration. In Computer Vision
– ECCV 2022, pages 17–33, 2022. 4, 6
[4] Maximilian Denninger, Dominik Winkelbauer, Martin Sun-
dermeyer, Wout Boerdijk, Markus Knauer, Klaus H. Strobl,
Matthias Humt, and Rudolph Triebel.
Blenderproc2: A
procedural pipeline for photorealistic rendering. Journal of
Open Source Software, 8(82):4901, 2023. 2
[5] Andreas Doumanoglou, Rigas Kouskouridas, Sotiris Malas-
siotis, and Tae-Kyun Kim. Recovering 6d object pose and
predicting next-best-view in the crowd. In Proceedings of
the IEEE conference on computer vision and pattern recog-
nition, pages 3583–3592, 2016. 2, 3
[6] Bertram Drost, Markus Ulrich, Paul Bergmann, Philipp
Hartinger, and Carsten Steger.
Introducing mvtec itodd-a
dataset for 3d object recognition in industry. In Proceed-
ings of the IEEE international conference on computer vision
workshops, pages 2200–2208, 2017. 2
[7] Jakob Engel, Kiran Somasundaram, Michael Goesele, Albert
Sun, Alexander Gamino, Andrew Turner, Arjang Talattof,
Arnie Yuan, Bilal Souti, Brighid Meredith, et al. Project aria:
A new tool for egocentric multi-modal ai research. arXiv
preprint arXiv:2308.13561, 2023. 2, 4, 1
[8] Daniel Feijoo, Juan C. Benito, Marcos V. Conde, and Alvaro
Garcia. Darkir: Robust low-light image restoration. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 10879–10889, 2025.
4, 6
[9] Martin A Fischler and Robert C Bolles.
Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 4, 6
[10] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy,
Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference
deep curve estimation for low-light image enhancement. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 1780–1789, 2020. 4
[11] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vin-
cent Lepetit. Honnotate: A method for 3d annotation of hand
and object poses.
In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
3196–3206, 2020. 2
[12] Yegyu Han, Taegyoon Yoon, Dayeon Woo, Sojeong Kim,
and Hyung-Sin Kim. Senseshift6d: Multimodal rgb-d bench-
marking for robust 6d pose estimation across environment
and sensor variations.
arXiv preprint arXiv:2507.05751,
2025. 3
[13] Kaiming He, Jian Sun, and Xiaoou Tang. Single image haze
removal using dark channel prior. In 2009 IEEE Conference
on Computer Vision and Pattern Recognition, pages 1956–
1963. IEEE, 2009. 4
[14] Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun
Bao, and Xiaowei Zhou. Onepose++: Keypoint-free one-
shot object pose estimation without CAD models. In Ad-
vances in Neural Information Processing Systems, 2022. 2
[15] Stefan Hinterstoisser, Vincent Lepetit, Slobodan Ilic, Ste-
fan Holzer, Gary Bradski, Kurt Konolige, and Nassir Navab.
Model based training, detection and pose estimation of
texture-less 3d objects in heavily cluttered scenes. In Asian
conference on computer vision, pages 548–562. Springer,
2012. 2, 3
[16] Tom´aˇs Hodan, Pavel Haluza, ˇStep´an Obdrˇz´alek, Jiri Matas,
Manolis Lourakis, and Xenophon Zabulis. T-less: An rgb-
d dataset for 6d pose estimation of texture-less objects. In
2017 IEEE Winter Conference on Applications of Computer
Vision (WACV), pages 880–888. IEEE, 2017. 2, 3
[17] Tomas Hodan, Frank Michel, Eric Brachmann, Wadim Kehl,
Anders GlentBuch, Dirk Kraft, Bertram Drost, Joel Vidal,
Stephan Ihrke, Xenophon Zabulis, et al. Bop: Benchmark for
6d object pose estimation. In Proceedings of the European
conference on computer vision (ECCV), pages 19–34, 2018.
2, 3
[18] Rudolph Emil Kalman. A new approach to linear filtering
and prediction problems. 1960. 5
[19] Agastya Kalra, Guy Stoppi, Dmitrii Marin, Vage Taa-
mazyan,
Aarrushi Shandilya,
Rishav Agarwal,
Anton
Boykov, Tze Hao Chong, and Michael Stark. Towards co-
evaluation of cameras hdr and algorithms for industrial-grade
6dof pose estimation. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
22691–22701, 2024. 3
[20] Roman Kaskman, Sergey Zakharov, Ivan Shugurov, and Slo-
bodan Ilic. Homebreweddb: Rgb-d dataset for 6d pose esti-
9

mation of 3d objects. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision Workshops, pages
0–0, 2019. 2
[21] Orest Kupyn, Tetiana Martyniuk, Junru Wu, and Zhangyang
Wang.
Deblurgan-v2:
Deblurring (orders-of-magnitude)
faster and better. In The IEEE International Conference on
Computer Vision (ICCV), 2019. 4
[22] Taein Kwon, Bugra Tekin, Jan St¨uhmer, Federica Bogo,
and Marc Pollefeys. H2o: Two hands manipulating objects
for first person interaction recognition. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 10138–10148, 2021. 2, 3, 4
[23] Yann Labb´e, Lucas Manuelli, Arsalan Mousavian, Stephen
Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpen-
tier, Mathieu Aubry, Dieter Fox, and Josef Sivic. Megapose:
6d pose estimation of novel objects via render & compare.
In Proceedings of the 6th Conference on Robot Learning
(CoRL), 2022. 4, 6
[24] Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim,
Byungju Woo, Haechan Lee, Sunghyun Cho, and Suha
Kwak. Human pose estimation in extremely low-light condi-
tions. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 7761–
7770, 2023. 2
[25] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua.
Ep n p: An accurate o (n) solution to the p n p problem. Inter-
national journal of computer vision, 81(2):155–166, 2009. 4,
6
[26] Jiehong Lin, Lihua Liu, Dekun Lu, and Kui Jia. Sam-6d:
Segment anything model meets zero-shot 6d object pose es-
timation.