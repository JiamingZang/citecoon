# XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera

> 2019 · id: W2912199856 · arXiv: 1907.00837 · pdf: http://hdl.handle.net/21.11116/0000-0003-FE21-A · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Optical human motion capture is a key enabling technology in
visual computing and related fields [Chai and Hodgins 2005; Men-
ache 2010; Starck and Hilton 2007]. For instance, it is widely used
to animate virtual avatars and humans in VFX. It is a key compo-
nent of many man-machine interfaces and is central to biomedical
motion analysis. In recent years, computer graphics and computer
vision researchers have developed new motion capture algorithms
that operate on ever simpler hardware and under far less restrictive
constraints than before. These algorithms do not require special body
suits, dense camera arrays, in-studio recording, or markers. Instead,
Project Website: http://gvv.mpi-inf.mpg.de/projects/XNect/
ACM Trans. Graph., Vol. 39, No. 4, Article 1. Publication date: July 2020.
arXiv:1907.00837v2  [cs.CV]  30 Apr 2020

1:2
•
Mehta, D. et al.
theyonlyneedafewcalibratedcamerastocapturepeoplewearingev-
eryday clothes outdoors, e.g. Elhayek et al. [2016]; Fang et al. [2018];
Huang et al. [2017a]; Kanazawa et al. [2018]; Mehta et al. [2017b];
Omran et al. [2018]; Pavlakos et al. [2019]; Rhodin et al. [2016b]; Stoll
et al. [2011]; Xiang et al. [2019]. The latest approaches leverage the
power of deep neural networks to capture 3D human pose from a
single color image, opening the door to many exciting applications in
virtual and augmented reality. Unfortunately, the problem remains
extremely challenging due to depth ambiguities, occlusions, and the
large variety of appearances and scenes.
More importantly, most methods fail under occlusions and focus
on a single person. Some recent methods instead focus on the ego-
centric setting [Rhodin et al. 2016a; Tome et al. 2019; Xu et al. 2019].
Single person tracking in the outside-in setting (non-egocentric) is
already hard and starkly under-constrained; multi-person tracking
is incomparably harder due to mutliple occlusions, challenging body
part to person assignment, and is computationally more demanding.
Thispresentsapracticalbarrierformanyapplicationssuchasgaming
and social VR/AR, which require tracking multiple people from low
cost sensors, and in real time.
Prior work on multi-person pose estimation runs at best at inter-
active frame rates (10-15 fps) [Dabral et al. 2019; Rogez et al. 2019] or
offline [Moon et al. 2019], and produces per-frame joint position esti-
mates which cannot be directly employed in many end applications
requiring joint angle based avatar animations.
We introduce a real-time algorithm for motion capture of multiple
people in common interaction scenarios using a single color camera.
Our full system produces the skeletal joint angles of multiple people
in the scene, along with estimates of 3D localization of the subjects
in the scene relative to the camera. Our method operates at more
than 30 frames-per-second and delivers state-of-the-art accuracy and
temporal stability. Our results are of a similar quality as commercial
depth sensing based mocap systems.
To this end, we propose a new pose formulation and a novel neural
network architecture, which jointly enable real-time performance,
while handling inter-person and person-object occlusions. A sub-
sequent model-based pose fitting stage produces temporally stable
3D skeletal motions. Our pose formulation uses two deep neural
network stages that perform local (per body joint) and global (all
body joints) reasoning, respectively. Stage I is fully convolutional
and jointly reasons about the 2D and 3D pose for all the subjects in
the scene at once, which ensures that the computational cost does
not increase with the number of individuals. Since Stage I handles the
already complex task of parsing the image for body parts, as well as
associating the body parts to identities, our key insight with regards
to the pose formulation is to have Stage I only consider body joints for
which direct image evidence is available, i.e., joints that are themselves
visible or their kinematic parents are visible. This way Stage I does
not have to spend representational capacity in hallucinating poses for
joints that have no supporting image evidence. For each visible body
joint, we predict the 2D part confidence maps, information for asso-
ciating parts to an individual, and an intermediate 3D pose encoding
for the bones that connect at the joint. Thus, the 3D pose encoding
is only cognizant of the joint’s immediate neighbours (local) in the
kinematic chain. A compact fully-connected network forms Stage II,
which relies on the intermediate pose encoding and other evidence
extracted in the preceding stage, to decode the complete 3D pose. The
Stage II network is able to reason about occluded joints using the full
body context (global) for each detected subject, and leverages learned
pose priors, and the subject 2D and 3D pose evidence. This stage is
compact, highly efficient, and acts in parallel for all detected subjects.
Stage I is the most computationally expensive part of our pipeline,
and the main bottleneck in achieving real-time performance.
We achieve real-time performance by contributing a new convolu-
tional neural network (CNN) architecture in Stage I to speed up the
most computationally expensive part of our pipeline. We refer to the
new architecture as SelecSLS Net. Our proposed architecture depends
on far fewer features than competing ones, such as ResNet-50 [He
et al. 2016], without any accuracy loss thanks to our insights on selec-
tive use of short and long range concatenation-skip connections. This
enablesfastinferenceonthecompleteinputframe,withouttheadded
pre-orpost-processingcomplexityofaseparateboundingboxtracker
for each subject. Further, the compactness of our Stage II network,
which reconciles the partially incomplete 2D pose and 3D pose encod-
ing to a full body pose estimate, enables it to simultaneously handle
many people with minimal overhead on top of Stage I. Additionally,
we fit a model-based skeleton to the 3D and 2D predictions in order
to satisfy kinematic constraints and reconcile the 2D and 3D predic-
tions across time. This produces temporally stable predictions, with
skeletal angle estimates, which can readily drive virtual characters.
In summary, our technical innovations and new design insights
at the individual stages, as well as our insights guiding the proposed
multi-stage design enable our final contribution: a complete algo-
rithm for multi-people 3D motion capture from a single camera that
achievesreal-timeperformancewithoutsacrificingreliabilityoraccu-
racy. The run time of our system only mildly depends on the number
of subjects in the scene, and even crowded scenes can be tracked at
high frame rates. We demonstrate our system’s performance on a
variety of challenging multi-person scenes.
2

## experiments
In this section we evaluate the results of our real-time multi-person
motion capture solution qualitatively and quantitatively on various
benchmarks, provide extensive comparison with prior work, and
conduct a detailed ablative analysis of the different components of
our system. To ensure that the reported results on 3D pose bench-
marks are actually indicative of the deployed system’s performance,
there is no test-time augmentation applied for our quantitative and
qualitative results. We do not use procrustes alignment to the ground
truth, nor do we rely on ground truth camera relative localization of
the subjects to generate or modify our predictions.
For additional qualitative results, please refer to the accompanying
video.
Fig. 7. Live Interaction and Virtual Character Control: The temporally
smooth joint angle predictions from Stage III can be readily employed for
driving virtual characters in real-time. The top two rows show our system
driving virtual skeletons and characters with the motion captured in real
time. On the bottom, our system is set up as a Kinect-like game controller,
allowing subjects to interact with their virtual avatars live. Some images
courtesy Music Express Magazine (https://youtu.be/kX6xMYlEwLA, https://
youtu.be/lv-h4WNnw0g). See the accompanying video and the supplemental
document for further character control examples.
7.1
System Characteristics and Applications
First, we show that our system provides efficient and accurate 3D
motion capture that is ready for live character animation and other
interactive CG applications, rivaling depth-based solutions despite
using only a single RGB video feed.
Real-time Performance: Our live system uses a standard webcam
as input, and processes 512×320 pixel resolution input frames. For
a scene with 10 subjects, the system running on a Desktop with an
Intel Xeon E5 with 3.5 GHz and an Nvidia GTX 1080Ti is capable of
processing input at >30 fps, while on a laptop with an Intel i7-8780H
and a 1080-MaxQ it runs at ≈27 fps. On the laptop, Stage I takes
21.5 ms, part association and feature extraction take 1 ms, Stage II
takes 1 ms, and Stage III takes ≈9 ms (2.4 ms for identity matching
to the previous frame, 6.8 ms for skeleton fitting).
We compare our timing against the faster but less accurate ‘demo’
version of LCRNet++ [2019], but compare our accuracy against the
slower but more accurate version of LCRNet++. LCRNet++ demo
system uses ResNet-50 with less post-processing overhead than the
accuracy-benchmarked system, and we measured its forward pass

XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera
•
1:11
time on a TitanX-Pascal GPU (11 TFLOPs) to be 16 ms, while on a
K80 GPU (4.1 TFLOPs) it takes >100 ms. Our SelecSLS-based system
takes only 14 ms on TitanX-Pascal and 35 ms on a K80 GPU while
producing more accurate per-frame joint position predictions (Stage
II) than the slower version of LCRNet++, as shown in Table 7. An
additional CPU-bound overhead of ≈9 ms for Stage III results in tem-
porally smooth joint-angle estimates which can readily be used to
drive virtual characters. The accompanying video contains examples
of our live setup running on the laptop. Note that throughout the
manuscript we report the timings of the various stages of our system
on a set of GPUs with FP32 performance representative of current
low-end and high-end consumer GPUs (≈4−12TFLOPs).
Multi-Person Scenes and Occlusion Robustness: In Figure 9, we show
qualitative results of our full system on several scenes containing
multiple interacting and overlapping subjects, including frames from
MuPoTS-3D [2018b]

## related_work
We focus our discussion on relevant 2D and 3D human pose estima-
tion from monocular RGB methods, in both single- and multi-person
scenarios–for overview articles refer to Sarafianos et al. [2016]; Xia
et al. [2017]. We also discuss prior datasets, and neural network ar-
chitectures that inspired ours.
Multi-Person 2D Pose Estimation: Multi-person 2D pose es-
timation methods can be divided into bottom-up and top-down ap-
proaches. Top-down approaches first detect individuals in a scene
and fall back to single-person 2D pose approaches or variants for
pose estimation [Gkioxari et al. 2014; Iqbal and Gall 2016; Papandreou
et al. 2017; Pishchulin et al. 2012; Sun and Savarese 2011]. Reliable
detection of individuals under significant occlusion, and tracking of
people through occlusions remains challenging.
Bottom-up approaches instead first localize the body parts of all
subjects and associate them to individuals in a second step. Associa-
tions can be obtained by predicting joint locations and their identity
embeddings together [Newell and Deng 2017], or by solving a graph
cut problem [Insafutdinov et al. 2017; Pishchulin et al. 2016]. This in-
volves solving an NP-hard integer linear program which easily takes
hours per image. The work of Insafutdinov et al. [2017] improves

XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera
•
1:3
over Pishchulin et al. [2016] by including image-based pairwise terms
and stronger detectors based on ResNet [He et al. 2016]. This way
reconstruction time reduces to several minutes per frame. Cao et al.
[2017] predict joint locations and part affinities (PAFs), which are 2D
vectors linking each joint to its parent. PAFs allow quick and greedy
part association, enabling real time mutli-person 2D pose estimation.
Our Stage I uses similar ideas to localize and assign joints in 2D, but
we also predict an intermediate 3D pose encoding per joint which
enables our subsequent stage to produce accurate 3D body pose esti-
mates.Güler et al. [2018] compute dense correspondences from pixels
to the surface of SMPL [2015], but they do not estimate 3D pose.
Single-Person 3D Pose Estimation: Monocular single person
3Dposeestimationwaspreviouslyapproachedwithgenerativemeth-
ods using physics priors [Wei and Chai 2010], or semi-automatic
analysis-by-synthesis fitting of parametric body models [Guan et al.
2009; Jain et al. 2010]. Recently, methods employing CNN based learn-
ing approaches led to important progress [Ionescu et al. 2014; Li and
Chan 2014; Li et al. 2015; Pavlakos et al. 2017; Sigal et al. 2010; Sun
et al. 2017, 2018; Tekin et al. 2016]. These methods can broadly be
classified into direct regression and ‘lifting’ based approaches. Re-
gressing straight from the image requires large amounts of 3D-pose
labelled images, which are difficult to obtain. Therefore, existing
datasets are captured in studio scenarios with limited pose and ap-
pearance diversity [Ionescu et al. 2014], or combine real and synthetic
imagery [Chen et al. 2016]. Consequently, to address the 3D data
scarcity, transfer learning using features learned on 2D pose datasets
has been applied to improve 3D pose estimation [Mehta et al. 2017a,b;
Popa et al. 2017; Sun et al. 2017; Tekin et al. 2017; Zhou et al. 2017].
‘Lifting’ based approaches predict the 3D pose from a separately
detected 2D pose [Martinez et al. 2017]. This has the advantages that
2D pose datasets are easier to obtain in natural environments, and the
lifting can be learned from MoCap data without overfitting on the stu-
dio conditions. While this establishes a surprisingly strong baseline,
lifting is ill-posed and body-part depth disambiguation is often not
possible without additional cues from the image. Other work has pro-
posed to augment the 2D pose with relative depth ordering of body
jointsasadditionalcontexttodisambiguate2Dto3Dlifting[Pavlakos
et al. 2018a; Pons-Moll et al. 2014]. Our approach can be seen as a
hybrid of regression and lifting methods: An encoding of the 3D pose
of the visible joints is regressed directly from the image (Stage I), with
each joint only reasoning about its immediate kinematic neighbours
(local context). This encoding, along with 2D joint detection confi-
dences augments the 2D pose and is ‘decoded’ into a complete 3D
body pose by Stage II reasoning about all body joints (global context).
Some recent methods integrate a 3D body model [Loper et al. 2015]
within a network, and train using a mixture of 2D poses and 3D poses
to predict 3D pose and shape from single images [Kanazawa et al.
2018; Omran et al. 2018; Pavlakos et al. 2018b; Tung et al. 2017]. Other
approaches optimize a body model or a template [Habermann et al.
2019; Xu et al. 2018] to fit 2D poses or/and silhouettes [Alldieck et al.
2019, 2018a,b; Bhatnagar et al. 2019; Bogo et al. 2016; Guler and Kokki-
nos 2019; Kolotouros et al. 2019b,a; Lassner et al. 2017]. Very few are
able to work in real time, and none of them handles multiple people.
Prior real-time 3D pose estimation approaches [Mehta et al. 2017b]
designed for single-person scenarios fail in multi-person scenarios.
Recent offline single-person approaches [Kanazawa et al. 2019] pro-
duce temporally coherent sequences of SMPL [2015] parameters, but
work only for unoccluded single-person scenarios. In contrast, our
proposed approach runs in real time for multi-person scenarios, and
produces temporally coherent joint angle estimates at par with of-
fline approaches, while successfully handling object and inter-person
occlusions.
Multi-Person3DPose: Earlierworkonmonocularmulti-person
3D pose capture often followed a generative formulation, e.g. esti-
mating 3D body and camera pose from 2D landmarks using a learned
pose space [Ramakrishna et al. 2012]. We draw inspiration from and
improve over limitations of recent deep learning-based methods.
Rogez et al. [2017] use a Faster-RCNN [2015] based approach and
first find representative poses of discrete pose clusters that are sub-
sequently refined. The LCRNet++ implementation of this algorithm
uses a ResNet-50 base network and achieves non-real-time interac-
tive 10−12fps on consumer hardware even with the faster but less
accurate ‘demo’ version that uses fewer anchor poses. Dabral et al.
[2019] use a similar Faster-RCNN based approach, and predict 2D
keypoints for each subject. Subsequently, the predicted 2D keypoints
are lifted to 3D pose. We show that incorporating additional infor-
mation, such as the keypoint confidence, and 3D pose encodings in
the ‘lifting’ step results in a much higher prediction accuracy. Moon
et al. [2019] employ a prior person detection step, and pass resized
image crops of each detected subject to the pose estimation network.
As prior work [Cao et al. 2017] has shown, such an approach results
high pose estimation accuracy, but comes at the cost of a significant
increase in inference time. Not only does such an approach work at
offline rates, the per-frame inference time scales with the number of
subjects in the scene, making it unsuitable for real-time applications.
The aforementioned detection based approaches predict multiple
proposals per individual and fuse them afterwards. This is time con-
suming, and in many cases it can either incorrectly merge nearby
individuals with similar poses, or fail to merge multiple proposals
for the same individual. Beyond the cost and potential errors from
fusing pose estimates, multiple detections of the same subject further
increase the inference time for the approach of Moon et al. [2019].
Our approach, being bottom-up, does not produce multiple detec-
tions per subject. The bottom-up approach of Mehta et al. [2018b]
predicts the 2D and 3D pose of all individuals in the scene using a
fixed number of feature maps, which jointly encode for any number
of individuals in the scene. This introduces potential conflicts when
subjects overlap, for which a complex encoding and read-out scheme
is introduced. The 3D encoding treats each limb and the torso as dis-
tinct objects, and encodes the 3D pose of each ‘object’ in the feature
maps at the pixel locations corresponding to the 2D joints of the ‘ob-
ject’. The encoding can thus handle partial inter-personal occlusion
by dissimilar body parts. Unfortunately, the approach still fails when
similar body parts of different subjects overlap. Similarly, Zanfir et al.
[2018b] jointly encode the 2D and 3D pose of all subjects in the scene
using a fixed number of feature maps. Different from Mehta et al.
[2018b], they encode the full 3D pose vector at all the projected pixels
of the skeleton, and not just at the body joint locations, which makes
the 3D feature space rife with potential encoding conflicts. For asso-
ciation, they learn a function to evaluate limb grouping proposals. A
3D pose decoding stage extracts 3D pose features per limb and uses

1:4
•
Mehta, D. et al.
Fig. 2. Overview: Computation is separated into three stages, the first two respectiv