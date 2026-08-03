# VNect

> 2017 · id: W2611932403 · arXiv: 1705.01583 · pdf: https://arxiv.org/pdf/1705.01583 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Our system is capable of obtaining a temporally consistent, full
3D skeletal pose of a human from a monocular RGB camera. Esti-
mating 3D pose from a single RGB camera is a challenging, under-
constrained problem with inherent ambiguities. Figure 2 provides
an overview of our method to tackle this challenging problem. It
consists of two primary components. The first is a convolutional
neural network (CNN) to regress 2D and 3D joint positions under
the ill-posed monocular capture conditions. It is trained on anno-
tated 3D human pose datasets [Ionescu et al. 2014b; Mehta et al.
2016], additionally leveraging annotated 2D human pose datasets
[Andriluka et al. 2014; Johnson and Everingham 2010] for improved
in-the-wild performance. The second component combines the re-
gressed joint positions with a kinematic skeleton fitting method to
produce a temporally stable, camera-relative, full 3D skeletal pose.
CNN Pose Regression: The core of our method is a CNN that predicts
both 2D, and root (pelvis) relative 3D joint positions in real-time. The
new proposed fully-convolutional pose formulation leads to results
on par with the state-of-the-art offline methods in 3D joint position
accuracy (see Section 5.2 for details). Being fully-convolutional, it
can operate in the absence of tight crops around the subject. The
CNN is capable of predicting joint positions for a diverse class of
activities regardless of the scene settings, providing a strong basis
for further pose refinement to produce temporally consistent full-3D
pose parameters
Kinematic Skeleton Fitting: The 2D and the 3D predictions from the
CNN, together with the temporal history of the sequence, can be
leveraged to obtain temporally consistent full-3D skeletal pose, with
the skeletal root (pelvis) localized in camera space. Our approach
uses an optimization function that: (1) combines the predicted 2D
and 3D joint positions to fit a kinematic skeleton in a least squares
sense, (2) ensures temporally smooth tracking over time. We further
improve the stability of the tracked pose by applying filtering steps
at different stages.
Skeleton Initialization (Optional): The system is set up with a default
skeleton which works well out of the box for most humans. For more
accurate estimates, the relative body proportions of the underlying
skeleton can be adapted to that of the subject, by averaging CNN
predictions for a few frames at the beginning. Since monocular
reconstruction is ambiguous without a scale reference, the CNN
predicts height normalized 3D joint positions. Users only need to
provide their height (distance from head to toe) once, so that we
can track the 3D pose in true metric space.
4
REAL-TIME MONOCULAR 3D POSE ESTIMATION
In this section, we describe in detail the different components of our
method to estimate a temporally consistent 3D skeletal motion from
monocular RGB input sequences. As input, we assume a continuous
stream of monocular RGB images {..., It−1, It }. For frame t in the
input stream, the final output of our approach is PG
t which is the
full global 3D skeletal pose of the person being tracked. Because
this output is already temporally consistent and in global 3D space,
it can be readily used in applications such as character control.
We use the following notation for the output in the intermediate
components of our method. The CNN pose regressor jointly esti-
mates the 2D joint positions Kt and root-relative 3D joint positions
PL
t (Section 4.1). The 3D skeleton fitting component combines the
2D and 3D joint position predictions to estimate a smooth, tempo-
rally consistent pose PG
t (θ, d), which is parameterized by the global
position d in camera space, and joint angles θ of the kinematic skele-
ton S. J indicates the number of joints. We drop the frame-number
subscript t in certain sections to aid readability.
4.1
CNN Pose Regression
The goal of CNN pose regression is to obtain joint positions, both,
in 2D image space and 3D. For 2D pose estimation with neural nets,
the change of formulation from direct regression of x,y body-joint
coordinates [Toshev and Szegedy 2014] to a heatmap based body-
joint detection formulation [Tompson et al. 2014] has been the key
driver behind the recent developments in 2D pose estimation. The
heatmap based formulation naturally ties image evidence to pose
estimation by predicting a confidence heatmap Hj,t over the image
plane for each joint j ∈{1..J}.
Existing approaches to 3D pose estimation lack such an image-
to-prediction association, often directly regressing the root-relative
joint locations [Ionescu et al. 2014a], leading to predicted poses
whose extent of articulation doesn’t reflect that of the person in the
image. See Figure 9. Treating pose as a vector of joint locations also
causes a natural gravitation towards networks with fully-connected
formulations [Mehta et al. 2016; Rogez and Schmid 2016; Tekin et al.
2016a; Yu et al. 2016], restricting the inputs to tight crops at a fixed
resolution, a limitation that needs to be overcome. These methods
assume the availability of tight bounding boxes, which necessitates
supplementation with separate bounding box estimators for actual
usage, which further adds to the run-time of these methods. The
fully-convolutional formulation of Pavlakos et al. [Pavlakos et al.
2016] seeks to alleviate some of these issues, but is limited by the
expensive per-joint volumetric formulation, which still relies on
cropped input and does not scale well to larger image sizes.
We overcome these limitations through our new formulation, by
extending the 2D heatmap formulation to 3D using three additional
location-maps Xj, Yj, Zj per joint j, capturing the root-relative lo-
cations xj, yj and zj respectively. To have the 3D pose prediction
linked more strongly to the 2D appearance in the image, the xj,
yj and zj values are read off from their respective location-maps
at the position of the maximum of the corresponding joint’s 2D
heatmap Hj, and stored in PL = {x, y, z}, where x ∈R1×J is a vec-
tor that stores the coordinate x location of each joint maximum.
The pose formulation is visualized in Figure 3. Networks using this
fully-convolutional formulation are not constrained in input image
size, and can work without tight crops. Additionally, the network
provides 2D and 3D joint location estimates without additional over-
head, which we exploit in subsequent steps for real-time estimation.
Section 5.2 shows the improvements afforded by this formulation.
Loss Term: To enforce the fact that we are only interested in xj,
yj and zj from their respective maps at joint j’s 2D location, the
joint location-map loss is weighted stronger around the joint’s 2D

VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera
•
1:5
Fig. 3. Schema of the fully-convolutional formulation for predicting root rel-
ative joint locations. For each joint j, the 3D coordinates are predicted from
their respective location-maps Xj, Yj, Zj at the position of the maximum
in the corresponding 2D heatmap Hj. The structure observed here in the
location-maps emerges due to the spatial loss formulation. See Section 4.1.
location. We use the L2 loss. For xj is the loss formulation is
Loss(xj) = ∥HGT
j
⊙(Xj −XGT
j
)∥2,
(1)
where GT indicates ground truth and ⊙is the Hadamard product.
The location maps are weighted with the respective ground truth 2D
heatmap HGT
j
, which in turn have confidence equal to a Gaussian
with a small support localized at joint j’s 2D location. Note that
no structure is imposed on the location-maps. The structure that
emerges in the predicted location-maps is indicative of the correla-
tion of xj and yj with root relative location of joint j in the image
plane. See Figure 3.
Network Details: We use the proposed formulation to adapt the
ResNet50 network architecture of He et al. [2016]. We replace the
layers of ResNet50 from res5a onwards with the architecture de-
picted in Figure 5, producing the heatmaps and location-maps for all
joints j ∈{1..J}. After training, the Batch Normalization [Ioffe and
Szegedy 2015] layers are merged with the weights of their preceding
convolution layers to improve the speed of the forward pass.
Intermediate Supervision: We predict the 2D heatmaps and 3D
location-maps from the features at res4d and res5a, tapering down
the weights of intermediate losses with increasing iteration count.
Additionally, similar to the root-relative location-maps Xj, Yj and
Zj, we predict kinematic parent-relative location-maps ∆Xj, ∆Yj
and ∆Zj from the features at res5b and compute bone length-maps
as:
BLj =
p
∆Xj ⊙∆Xj + ∆Yj ⊙∆Yj + ∆Zj ⊙∆Zj.
(2)
These intermediate predictions are subsequently concatenated with
the intermediate features, to give the network an explicit notion of
bone lengths to guide the predictions. See Figure 5.
Experiments showed that the deeper variants of ResNet offer only
small gains for a substantial increase (1.5×) in computation time,
prompting us to choose ResNet

## method
Direct Discuss Eating Greet Phone Posing Purch. Sitting Down Smoke Photo Wait Walk Dog
Pair
All
[Zhou et al. 2015b]
87.4
109.3
87.1
103.2 116.2
106.9
99.8
124.5
199.2
107.4
143.3 118.1 79.4 114.2 97.7
113.0
[Tekin et al. 2016c]
102.4
147.7
88.8
125.3 118.0
112.3
129.2
138.9
224.9
118.4
182.7 138.8 55.1 126.3 65.8
125.0
[Yu et al. 2016]
85.1
112.7
104.9
122.1 139.1
105.9
166.2
117.5
226.9
120.0
135.9 117.7 137.4 99.3 106.5 126.5
[Ionescu et al. 2014b]
132.7
183.6
132.4
164.4 162.1
150.6
171.3
151.6
243.0
162.1
205.9 170.7 96.6 177.1 127.9 162.1
[Zhou et al. 2016]
91.8
102.4
97.0
98.8
113.4
90.0
93.8
132.2
159.0
106.9
125.2
94.4
79.0 126.0 99.0
107.3
[Pavlakos et al. 2016]
58.6
64.6
63.7
62.4
66.9
57.7
62.5
76.8
103.5
65.7
70.7
61.6
69.0
56.4
59.5
66.9
[Mehta et al. 2016]
52.6
63.8
55.4
62.3
71.8
52.6
72.2
86.2
120.6
66.0
79.8
64.0
48.9
76.8
53.7
68.6
[Tekin et al. 2016b]
85.0
108.8
84.4
98.9
119.4
98.5
93.8
73.8
170.4
85.1
95.7
116.9 62.1 113.7 94.8
100.1
Ours (ResNet 100)
61.7
77.8
64.6
70.3
90.5
61.9
79.8
113.2
153.1
80.9
94.4
75.1
54.9
83.5
61.0
82.5
Ours (ResNet 50)
62.6
78.1
63.4
72.5
88.3
63.1
74.8
106.6
138.7
78.8
93.8
73.9
55.8
82.0
59.6
80.5
Fig. 14. Application to entertainment. The real-time 3D pose estimation
method provides a natural motion interface, e.g. for sport games.
Ubiquitous Motion Capture with Smartphones: Real-time monocular
3D pose estimation lends itself to application on low quality smart-
phone video streams. By streaming the video to a machine with
sufficient capabilities for our algorithm, one can turn any smart-
phone into a lightweight, fully-automatic, handheld motion capture
sensor, see Figure 15 and the accompanying video. Since smart-
phones are widespread, it enables the aforementioned applications
for casual users without requiring additional sensing devices.
6

## experiments
We show live applications of our system at 30 Hz. The reconstruction
quality is high and we demonstrate the usefulness of our method
3D character control, embodied virtual reality, and pose tracking
from low quality smartphone camera streams. See Section 5.3 and
Figure 1. Results are best observed in motion in the supplemental
video. The importance of the steps towards enabling these applica-
tions with a video solution are thoroughly evaluated in more than
10 sequences. Results are comparable in quality to depth-camera
based solutions like the Kinect [Microsoft Corporation 2013] and
significantly outperform existing monocular video-based solutions.
As the qualitative baseline we choose the state-of-the-art 2D to 3D
lifting approach of Zhou et al. [2015] and the 3D regression approach
of Mehta et al. [2016], which estimate joint-positions offline. The
accuracy improvements are further quantitatively validated on the
established H3.6M dataset [Ionescu et al. 2014b] and the MPI-INF-
3DHP dataset [Mehta et al. 2016]. The robustness to diverse persons,
clothing and scenes is demonstrated on several real-time examples

VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera
•
1:7
and community videos. Please see our project webpage for more
results and details1.
Computations are performed on a 6-core Xeon CPU, 3.8 GHz and
a single Titan X (Pascal architecture) GPU. The CNN computation
takes ≈18 ms, the skeleton fitting ≈7–10 ms, and preprocessing and
filtering 5 ms.
5.1
Comparison with Active Depth Sensors (Kinect)
We synchronously recorded video from an RGB camera and a co-
located Kinect sensor in a living room scenario. Figure 6 shows rep-
resentative frames. Although the depth sensor provides additional
information, our reconstructions from just RGB are of a similar qual-
ity. The Kinect results are of comparable stability to ours, but yield
erroneous reconstructions when limbs are close to scene objects,
such as when sitting down. Our RGB solution, however, succeeds
in this case, although is slightly less reliable in depth estimation.
A challenging case for both methods is the tight crossing of legs.
Please see the supplemental video for a visual comparison.
The video solution succeeds also in situations with direct sun-
light (Figure 7), where IR-based depth cameras are inoperable. More-
over, RGB cameras can simply be equipped with large field-of-view
(FOV) lenses and, despite strong distortions, successfully track hu-
mans [Rhodin et al. 2016a]. On the other hand, existing active sen-
sors are limited to relatively small FOVs, which severely limits the
tracking volume.
5.2
Comparison with Video Solutions
Qualitative Evaluation: We qualitatively compare against the 3D
pose regression method of Mehta et al. [2016] and Zhou et al. [2015]
on Sequence 6 (outdoor) of MPI-INF-3DHP test set. Our results are
comparable to the quality of these offline methods (see Figure 10).
However, the per frame estimates of these offline methods exhibits
jitter over time, a drawback of most existing solutions. Our full pose
results are temporally stable and are computed at real-time frame
rate of 30 Hz.
The kinematic skeleton fitting estimates global translation d. Fig-
ure 8 demonstrates that estimates are drift-free, the feet position
matches with the same reference point after performing a circular
walk. The smoothness constraint in depth direction limits sliding
of the character away from the character, as pictured in the supple-
mental video sequences.
Quantitative Evaluation: We compare our method with the state-
of-the-art approach of Mehta et al. [2016] on the MPI-INF-3DHP
dataset, using the more robust Percentage of Correct Keypoints
metric (3D PCK @150mm) on the 14 joints spanned by head, neck,
shoulders, elbow, wrist, hips, knees and ankles. We train both, our
model, as well as that of Mehta et al. on the same data (Human3.6m +
MPI-INF-3DHP), as detailed in Section 4.1, to be compatible in terms
of the camera viewpoints selected, and use ResNet100 as the base
architecture for a fair comparison. Table 1 shows the results of the
raw 3D predictions from our network on ground-truth bounding
box cropped frames. We see that the results are comparable to
that of Mehta et al. The slight increase in accuracy on going to a
1http://gvv.mpi-inf.mpg.de/projects/VNect/
Fig. 6. Side-by-side pose comparison with our method (top) and Kinect
(bottom). Overall estimated poses are of similar quality (first two frames).
Both the Kinect (third and fourth frames) and our approach (fourth and
fifth frames) occasionally predict erroneous poses.
Fig. 7. Our approach succeeds in strong illumination and sunlight (center
right and right), while the IR-based depth estimates of the Microsoft Kinect
are erroneous (left) and depth-based tracking fails (center left).
Fig. 8. The estimated 3D pose is drift-free. The motion of the person starts
and ends at the marked point (orange), both in the real world and in our
reconstruction.
50-layer network is possibly due to the better gradient estimates
coming from larger mini-batches that can be fit into memory while
training, on account of the smaller size of the network. Evidence
that our method ties the estimated 3D positions strongly to image
appearance than previous methods can also be gleaned from the
fact that our approach performs significantly better for activity
classes such as Standing/Walking, Sports and Miscellaneous without
significant self-occlusions. We do lose some performance on activity
classes with significant self-occlusion such as Sitting/Lying on the
floor. We additionally report the Mean Per Joint Position Error
(MPJPE) numbers in mm. Note that MPJPE is not a robust measure,
and is heavily influenced by large outliers, and hence the worse
performance on the MPJPE measure (124.7mm vs 117.6mm) despite
the better 3D PCK results (76.6% vs 75.7%).
We further investigate the nature of errors of our method. We first
look at the joint-wise breakup of accuracy of our fully-convolutional

1:8
•
D. Mehta et. al.
Table 1. Comparison of our network against state of the art on MPI-INF-3DHP test set, using ground-truth bounding boxes. We report the Percentage of
Correct Keypoints measure in 3D, and the Area Under the Curve for the same, as proposed by MPI-INF-3DHP. We additionally report the Mean Per Joint
Position Error in mm. Higher PCK and AUC is better, and lower MPJPE is better.
Stand/
Sit On
Crouch/
On the
Network
Scales
Walk
Exercise
Chair
Reach
Floor
Sports
Misc.
Total
PCK
PCK
PCK
PCK
PCK
PCK
PCK
PCK
AUC
MPJPE(mm)
Ours
(ResNet 100)
0.7, 1.0
87.6
76.4
71.4
71.6
47.8
82.5
78.9
75.0
39.5
127.8
1.0
86.4
72.3
68.0
65.4
40.7
80.5
76.3
71.4
36.9
142.8
Ours
(ResNet 50)
0.7, 1.0
87.7
77.4
74.7
72.9
51.3
83.3
80.1
76.6
40.4
124.7
1.0
86.7
73.9
69.8
66.1
44.7
82.0
79.4
73.3
37.8
138.7
[Mehta et al. 2016]
(ResNet 100)
0.7, 1.0
86.6
75.3
74.8
73.7
52.2
82.1
77.5
75.7
39.3
117.6
1.0
86.3
72.4
71.5
67.6
49.2
81.0
76.2
73.2
37.8
126.6
Fig. 9. A visual look at the direct 3D predictions resulting from our fully-
convolutional formulation vs Mehta et al. Our formulation allows the pre-
dictions to be more strongly tied to image evidence, leading to overall better
pose quality, particular for the end effectors. The red arrows point to mis-
predictions.
Fig. 10. Side-by-side comparison of our full method (left), against the of-
fline joint-position estimation methods of Mehta et al. [2016] (middle) and
Zhou et al. [2015] (right). Our real-time results are of a comparable quality
to these offline methods. 2D joint positions for Zhou et al. are generated
using Convolutional Pose Machines [2016].
Fig. 11. Joint-wise breakdown of the accuracy of Mehta et al. and Our
ResNet100 based CNN predictions on MPI-INF-3DHP test set.
ResNet100 CNN predictions vs Mehta et al. ’s formulation with fully-
connected layers. Figure 11 shows that the accuracy of ankles for
our formulation is significantly better, while the accuracy of the
head is markedly worse.
In Figure 9, we visually compare the two methods, further demon-
strating the strong tie-in to image appearance that our formulation
affords, and the downsides of the strong tie-in. We also show that
our method is prone to occasional large mispredictions when the
body joint 2D location detector misfires. It is these large outliers that
obfuscate the reported MPJPE numbers. Figure 12, which plots the
fraction of mispredicted joints vs. the error threshold on MPI-INF-
3DHP test set shows that our method has a higher fraction of per-
joint mispredictions beyond 300mm. It explains the higher MPJPE
numbers compared to Mehta et al. despite equivalent PCK perfor-
mance. The various filtering stages employed in the full pipeline
ameliorate these large mispredictions.
For Human3.6m, we follow the protocol as in earlier work [Pavlakos
et al. 2016; Tekin et al. 2016b,c], and evaluate on all actions and cam-
eras for subject number 9 and 11, and report Mean Per Joint Position
Error (mm) for root r

## related_work
Our goal is stable 3D skeletal motion capture from (1) a single
camera (2) in real-time. We focus the discussion of related work
on approaches from the large body of marker-less motion capture
research that contributed to attaining either of these properties.
Multi-view: With multi-view setups markerless motion-captue solu-
tions attain high accuracy. Tracking of a manually initialized actor
model from frame to frame with a generative image formation model
is common. See [Moeslund et al. 2006] for a complete overview. Most
methods target high quality with offline computation [Bregler and
Malik 1998; Howe et al. 1999; Loper and Black 2014; Sidenbladh
et al. 2000; Starck and Hilton 2003]. Real-time performance can be
attained by representing the actor with Gaussians [Rhodin et al.
2015; Stoll et al. 2011; Wren et al. 1997] and other approximations
[Ma and Wu 2014], in addition to formulations allowing model-to-
image fitting. However, these tracking-based approaches often lose
track in local minima of the non-convex fitting functions they opti-
mize and require separate initialization, e.g. using [Bogo et al. 2016;
Rhodin et al. 2016b; Sminchisescu and Triggs 2001]. Robustness
could be increased with a combination of generative and discrimina-
tive estimation [Elhayek et al. 2016], even from a single input view
[Rosales and Sclaroff 2006; Sminchisescu et al. 2006], and egocentric
perspective [Rhodin et al. 2016a]. We utilize generative tracking
components to ensure temporal stability, but avoid model projec-
tion through a full image formation model to speed up estimation.
Instead, we combine discriminative pose estimation with kinematic
fitting to succeed in our underconstrained setting.
Monocular Depth-based: The additional depth channel provided by
RGB-D sensors has led to robust real-time pose estimation solutions
[Baak et al. 2011; Ganapathi et al. 2012; Ma and Wu 2014; Shotton
et al. 2013; Wei et al. 2012; Ye and Yang 2014] and the availabil-
ity of low-cost devices has enabled a range of new applications.
Even real-time tracking of general deforming objects [Zollhöfer
et al. 2014] and template-free reconstruction [Dou et al. 2016; Inn-
mann et al. 2016; Newcombe et al. 2015; Orts-Escolano et al. 2016]
has been demonstrated. RGB-D information overcomes forward-
backwards ambiguities in monocular pose estimation. Our goal is a
video solution that overcomes depth ambiguities without relying
on a specialized active sensor.
Monocular RGB: Monocular generative motion capture has only
been shown for short clips and when paired with strong motion
priors [Urtasun et al. 2006] or in combination with discriminative
re-initialization [Rosales and Sclaroff 2006; Sminchisescu et al. 2006],
since generative reconstruction is fundamentally underconstrained.
Using photo-realistic template models for model fitting enables more
robust monocular tracking of simple motions, but requires more
expensive offline computation [de La Gorce et al. 2008]. Sampling-
based methods avoid local minima [Balan et al. 2005; Bo and Smin-
chisescu 2010; Deutscher and Reid 2005; Gall et al. 2010]. However,
real-time variants can not guarantee global convergence due to a
limited number of samples, such as particle swarm optimization
techniques [Oikonomidis et al. 2011]. Structure-from-motion tech-
niques exploit motion cues in a batch of frames [Garg et al. 2013],
and have also been applied to human motion estimation [Gotardo
and Martinez 2011; Lee et al. 2013; Park and Sheikh 2011; Zhu et al.
2011]. However, batch optimization does not apply to our real-time
setting, where frames are streamed sequentially. For some appli-
cations manual annotation and correction of frames is suitable,
for instance to enable movie actor reshaping [Jain et al. 2010] and
garment replacement in video [Rogge et al. 2014]. In combination

VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera
•
1:3
Fig. 2. Overview. Given a full-size image It at frame t, the person-centered crop Bt is efficiently extracted by bounding box tracking, using the previous
frame’s keypoints Kt−1. From the crop, the CNN jointly predicts 2D heatmaps Hj,t and our novel 3D location-maps Xj,t, Yj,t and Zj,t for all joints j. The 2D
keypoints Kt are retrieved from Hj,t and, after filtering, are used to read off 3D pose PL
t from Xj,t, Yj,t and Zj,t . These per-frame estimates are combined to
stable global pose PG
t by skeleton fitting. Information from frame t −1 is marked in gray-dashed.
with physical constraints, highly accurate reconstructions are pos-
sible from monocular video [Wei and Chai 2010]. Vondrak et al.
[2012] succeed without manual annotation by simulating biped-
controllers, but require batch-optimization. While these methods
can yield high-quality reconstructions, interaction and expensive
optimization preclude live applications.
Discriminative 2D human pose estimation is often an interme-
diate step to monocular 3D pose estimation. Pictorial structure
approaches infer body part locations by message passing over a
huge set of pose-states [Agarwal and Triggs 2006; Andriluka et al.
2009; Bourdev and Malik 2009; Felzenszwalb and Huttenlocher 2005;
Ferrari et al. 2009; Johnson and Everingham 2010] and have been
extended to 3D pose estimation [Amin et al. 2013; Balan et al. 2007;
Belagiannis et al. 2014; Sigal et al. 2012]. Recent approaches outper-
form these methods in computation time and accuracy by leveraging
large image databases with 2D joint location annotation, which en-
ables high accuracy prediction with deep CNNs [Belagiannis and
Zisserman 2016; Hu et al. 2016; Insafutdinov et al. 2016; Pishchulin
et al. 2016; Wei et al. 2016], on multiple GPUs, even at real-time
rates [Cao et al. 2016]. Given 2D joint locations, lifting them to 3D
pose is challenging. Existing approaches use bone length and depth
ordering constraints [Mori and Malik 2006; Taylor 2000], sparsity as-
sumptions [Wang et al. 2014; Zhou et al. 2015,a], joint limits [Akhter
and Black 2015], inter-penetration constraints [Bogo et al. 2016],
temporal dependencies [Rhodin et al. 2016b], and regression [Yasin
et al. 2016]. Treating 3D pose as a hidden variable in 2D estimation
is an alternative [Brau and Jiang 2016]. However, the sparse set
of 2D locations loses image evidence, e.g. on forward-backwards
orientation of limbs, which leads to erroneous estimates in ambigu-
ous cases. To overcome these ambiguities, discriminative methods
have been proposed that learn implicit depth features for 3D pose
directly from more expressive image representations. Rosales and
Sclaroff regress 3D pose from silhouette images with the specialized
mappings architecture [2000], Agarwal and Triggs with linear re-
gression [2006], and Elgammal and Lee through a joint embedding
of images and 3D pose [2004]. Sminchisescu further utilized tem-
poral consistency to propagate pose probabilities with a Bayesian
mixture of experts Markov model [2007]. Relying on the recent ad-
vances in machine learning techniques and compute capabilities,
approaches for direct 3D pose regression from the input image have
been proposed, using structured learning of latent pose [Li et al.
2015a; Tekin et al. 2016a], joint prediction of 2D and 3D pose [Li and
Chan 2014; Tekin et al. 2016b; Yasin et al. 2016], transfer of features
from 2D datasets [Mehta et al. 2016], novel pose space formula-
tions [Pavlakos et al. 2016] and classification over example poses
[Pons-Moll et al. 2014; Rogez and Schmid 2016]. Relative per-bone
predictions [Li and Chan 2014], kinematic skeleton models [Zhou
et al. 2016], or root centered joint positions [Ionescu et al. 2014a] are
used as the eventual output space. Such direct 3D pose regression
methods capture depth relations well, but 3D estimates usually do
not accurately match the true 2D location when re-projected to the
image, because estimations are done in cropped images that lose
camera perspective effects, using a canonical height, and minimize
3D loss instead of projection to 2D. Furthermore, they only deliver
joint positions, are temporally unstable, and none has shown real-
time performance. We propose a method to combine 2D and 3D
estimates in real-time along with temporal tracking. It is inspired
by the method of Tekin et al. [2016c], where batches of frames are
processed offline after motion compensation, and is related to the
recently proposed per-frame combination of 2D and 3D pose [Tekin
et al. 2016b].
Notably, only few methods target real-time monocular reconstruc-
tion. Exceptions are the regression of 3D pose from Haar features
by Bissacco et al. [2007] and detection of a set of discrete poses
from edge direction histograms in the vicinity of the previous frame
pose [Taycher et al. 2006]. Both only obtain temporally unstable,
coarse pose, not directly usable in our applications. Chai and Hod-
gins obtain sufficient quality to drive virtual avatars in real-time, but
require visual markers [Chai and Hodgins 2005]. The 

## conclusion
The availability of sufficient annotated 3D pose training data re-
mains an issue. Even the most recent annotated real 3D pose data
sets, or combined real/synthetic data sets [Chen et al. 2016; Ionescu
et al. 2014b; Mehta et al. 2016] are a subset of real world human
pose, shape, appearance and background distributions. Recent top
performing methods explicitly address this data sparsity by training

VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera
•
1:11
Fig. 15. Handheld recording with a readily available smartphone camera
(left) and our estimated pose (right), streamed to and processed by a GPU
enabled PC.
similarly deep networks, but with architectural changes enabling
improved intermediate training supervision [Mehta et al. 2016].
Our implementation only supports a single person, although
the proposed fully-convolutional formulation could be scaled to
multiple persons. Such an extension is currently precluded due to
the lack of multi-person datasets, required to train multi-person 3D
pose regressors. One possible approach is to adapt the multi-person
2D pose methods of Insafutdinov et al. [2016] and Cao et al. [2016].
We also analyze the impact of 2D joint position mispredictions
on the 3D joint position predictions from our fully-convolutional
formulation. We decouple the 3D predictions from the 2D predic-
tions by looking up the 3D joint positions from their location-maps
using the ground truth 2D joint positions. See Table 4. We see a 3D
PCK improvement of 2.8, which is congruent with the notion of a
stronger tie-in of the predicted joint positions with the image plane,
which causes the 3D joint predictions to be erroneous when 2D
joint detection misfires. The upside of this is that the 3D predictions
can be improved through improvements to 2D joint position predic-
tion. Alternatively, optimization formulations that directly operate
on the heatmaps and the location-maps could be constructed. Our
fully-convolutional formulation can also benefit from iterative re-
finement, akin to heatmap-based 2D pose estimation approaches
[Hu et al. 2016; Newell et al. 2016].
8