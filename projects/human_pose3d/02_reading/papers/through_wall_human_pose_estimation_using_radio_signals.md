# Through-Wall Human Pose Estimation Using Radio Signals

> 2018 · id: W2799062425 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Through-Wall Human Pose Estimation Using Radio Signals
Mingmin Zhao
Tianhong Li
Mohammad Abu Alsheikh
Yonglong Tian
Hang Zhao
Antonio Torralba
Dina Katabi
MIT CSAIL
Figure 1: The ﬁgure shows a test example with a single person. It demonstrates that our system tracks the pose as the person enters the room and even
when he is fully occluded behind the wall. Top: Images captured by a camera colocated with the radio sensor, and presented here for visual reference.
Middle: Keypoint conﬁdence maps extracted from RF signals alone, without any visual input. Bottom: Skeleton parsed from keypoint conﬁdence maps
showing that we can use RF signals to estimate the human pose even in the presence of full occlusion.
Abstract
This paper demonstrates accurate human pose estima-
tion through walls and occlusions. We leverage the fact that
wireless signals in the WiFi frequencies traverse walls and
reﬂect off the human body. We introduce a deep neural net-
work approach that parses such radio signals to estimate
2D poses. Since humans cannot annotate radio signals, we
use state-of-the-art vision model to provide cross-modal su-
pervision. Speciﬁcally, during training the system uses syn-
chronized wireless and visual inputs, extracts pose informa-
tion from the visual stream, and uses it to guide the training
process. Once trained, the network uses only the wireless
signal for pose estimation. We show that, when tested on
visible scenes, the radio-based system is almost as accu-
rate as the vision-based system used to train it. Yet, unlike
vision-based pose estimation, the radio-based system can
estimate 2D poses through walls despite never trained on
such scenarios. Demo videos are available at our website.
1. Introduction
Estimating the human pose is an important task in
computer vision with applications in surveillance, activ-
ity recognition, gaming, etc.
The problem is deﬁned as
generating 2D skeletal representations of the joints on the
arms and legs, and keypoints on the torso and head. It has
recently witnessed major advances and signiﬁcant perfor-
mance improvements [30, 27, 28, 46, 31, 20, 10, 16, 33, 12,
47, 37, 45, 13]. However, as in any camera-based recogni-
tion task, occlusion remains a fundamental challenge. Past
work deals with occlusion by hallucinating the occluded
body parts based on the visible ones. Yet, since the human
body is deformable, such hallucinations are prone to errors.
Further, this approach becomes infeasible when the person
is fully occluded, behind a wall or in a different room.
This paper presents a fundamentally different approach
to deal with occlusions in pose estimation, and potentially
other visual recognition tasks. While visible light is eas-
ily blocked by walls and opaque objects, radio frequency
(RF) signals in the WiFi range can traverse such occlusions.
Further, they reﬂect off the human body, providing an op-
portunity to track people through walls. Recent advances
in wireless systems have leveraged those properties to de-
tect people [5] and track their walking speed through oc-
clusions [19]. Past systems however are quite coarse: they
either track only one limb at any time [5, 4], or generate a
static and coarse description of the body, where body-parts
observed at different time are collapsed into one frame [4].
Use of wireless signals to produce a detailed and accurate
17356

description of the pose, similar to that achieved by a state-
of-the-art computer vision system, has remained intractable.
In this paper, we introduce RF-Pose, a neural network
system that parses wireless signals and extracts accurate 2D
human poses, even when the people are occluded or be-
hind a wall. RF-Pose transmits a low power wireless sig-
nal (1000 times lower power than WiFi) and observes its
reﬂections from the environment. Using only the radio re-
ﬂections as input, it estimates the human skeleton. Fig. 1
shows an example output of RF-Pose tracking a person as
he enters a room, becomes partially visible through a win-
dow, and then walks behind the wall. The RGB images in
the top row show the sequence of events and the occlusions
the person goes through; the middle row shows the conﬁ-
dence maps of the human keypoints extracted by RF-Pose;
and the third row shows the resulting skeletons. Note how
our pose estimator tracks the person even when he is fully
occluded behind a wall. While this example shows a single
person, RF-Pose works with multiple people in the scene,
just as a state-of-art vision system would.
The design and training of our network present different
challenges from vision-based pose estimation. In particu-
lar, there is no labeled data for this task. It is also infeasible
for humans to annotate radio signals with keypoints. To ad-
dress this problem, we use cross-modal supervision. During
training, we attach a web camera to our wireless sensor, and
synchronize the the wireless and visual streams. We extract
pose information from the visual stream and use it as a su-
pervisory signal for the wireless stream. Once the system
is trained, it only uses the radio signal as input. The result
is a system that is capable of estimating human pose using
wireless signals only, without requiring human annotation
as supervision. Interestingly, the RF-based model learns to
perform pose estimation even when the people are fully oc-
cluded or in a different room. It does so despite it has never
seen such examples during training.
Beyond
cross-modal
supervision,
the
design
of
RF-Pose accounts for the intrinsic features of RF signals
including low spatial resolution, specularity of the human
body at RF frequencies that traverse walls, and differences
in representation and perspective between RF signals and
the supervisory visual stream.
We train and test RF-Pose using data collected in pub-
lic environments around our campus. The dataset has hun-
dreds of different people performing diverse indoor activ-
ities: walking, sitting, taking stairs, waiting for elevators,
opening doors, talking to friends, etc. We test and train
on different environments to ensure the network generalizes
to new scenes. We manually label 2000 RGB images and
use them to test both the vision system and RF-Pose. The
results show that on visible scenes, RF-Pose has an aver-
age precision (AP) of 62.4 whereas the vision-based system
used to train it has an AP of 68.8. For through-wall scenes,
RF-Pose has an AP of 58.1 whereas the vision-based system
fails completely.
We also show that the skeleton learned from RF signals
extracts identifying features of the people and their style of
moving. We run an experiment where we have 100 people
perform free walking, and train a vanilla-CNN classiﬁer to
identify each person using a 2-second clip of the RF-based
skeleton. By simply observing how the RF-based skeleton
moves, the classiﬁer can identify the person with an accu-
racy over 83% in both visible and through wall scenarios.
2. Related Work
(a) Computer Vision: Human pose estimation from RGB
images generally falls into two main categories: Top-down
and bottom-up methods. Top-down methods [16, 14, 29,
15] ﬁrst detect each people in the image, and then apply
a single-person pose estimator to each people to extract
keypoints. Bottom-up methods [10, 31, 20], on the other
hand, ﬁrst detect all keypoints in the image, then use post-
processing to associate the keypoints belonging to the same
person. We build on this literature and adopt a bottom-up
approach, but differ in that we learn poses from RF sig-
nals. While some prior papers use sensors other than con-
ventional cameras, such as RGB-D sensors [50] and Vicon
[35], unlike RF signals, those data inputs still suffer from
occlusions by walls and other opaque structures.
In terms of modeling, our work is related to cross-modal
and multi-modal learning that explores matching different
modalities or delivering complementary information across
modalities [8, 11, 36, 34].
In particular, our approach
falls under cross-modal teacher-student networks [8], which
transfer knowledge learned in one data modality to another.
While past work only transfers category-level discrimina-
tive knowledge, our network transfers richer knowledge on
dense keypoint conﬁdence maps.
(b) Wireless Systems: Recent years have witnessed much
interest in localizing people and tracking their motion using
wireless signals. The literature can be classiﬁed into two
categories. The ﬁrst category operates at very high frequen-
cies (e.g., millimeter wave or terahertz) [3]. These can ac-
curately image the surface of the human body (as in airport
security scanners), but do not penetrate walls and furniture.
The second category uses lower frequencies, around a
few GHz, and hence can track people through walls and
occlusions. Such through-wall tracking systems can be di-
vided into: device-based and device-free.
Device-based
tracking systems localize people using the signal gener-
ated by some wireless device they carry. For example, one
can track a person using the WiFi signal from their cell-
phone [44, 24, 40].
Since the tracking is performed on
the device not the person, one can track different body-
parts by attaching different radio devices to each of them.
27357

On the other hand, device-free wireless tracking systems
do not require the tracked person to wear sensors on their
body. They work by analyzing the radio signal reﬂected
off the person’s body. However, device-free systems typi-
cally have low spatial resolution and cannot localize mul-
tiple body parts simultaneously. Different papers either lo-
calize the whole body [5, 23], monitor the person’s walking
speed [43, 19], track the chest motion to extract breathing
and heartbeats [6, 51, 52], or track the arm motion to iden-
tify a particular gesture [32, 26]. The closest to our work is
a system called RF-Capture which creates a coarse descrip-
tion of the human body behind a wall by collapsing multiple
body parts detected at different points in time [4]. None of
the past work however is capable of estimating the human
pose or simultaneously localizing its various keypoints.
Finally, some prior papers have explored human identiﬁ-
cation using wireless signals [49, 43, 18]. Past work, how-
ever, is highly restrictive in how the person has to move, and
cannot identify people from free-form walking.
3. RF Signals Acquisition and Properties
Our RF-based pose estimation relies on transmitting a
low power RF signal and receiving its reﬂections. To sep-
arate RF reﬂections from different objects, it is common to
use techniques like FMCW (Frequency Modulated Contin-
uous Wave) and antenna arrays [4]. FMCW separates RF
reﬂections based on the distance of the reﬂecting object,
whereas antenna arrays separate reﬂections based on their
spatial direction. In this paper, we introduce a radio similar
to [4], which generates an FMCW signal and has two an-
tenna arrays: vertical and horizontal (other radios are also
available [1, 2]). Thus, our input data takes the form of
two-dimensional heatmaps, one for each of the horizontal
and vertical antenna arrays. As shown in Fig. 2, the hori-
zontal heatmap is a projection of the signal reﬂections on a
plane parallel to the ground, whereas the vertical heatmap is
a projection of the reﬂected signals on a plane perpendicu-
lar to the ground (red refers to large values while blue refers
to small values). Note that since RF signals are complex
numbers, each pixel in this map has a real and imaginary
components. Our radio generates 30 pairs of heatmaps per
second.
It is important to note that RF signals have intrinsically
different properties than visual data, i.e., camera pixels.
• First, RF signals in the frequencies that traverse walls
have low spatial resolution, much lower than vision data.
The resolution is typically tens of centimeters [5, 2, 4],
and is deﬁned by the bandwidth of the FMCW signal and
the aperture of the antenna array. In particular, our radio
has a depth resolution about 10 cm, and its antenna ar-
rays have vertical and horizontal angular resolution of 15
degrees.
x
y
z
Figure 2: RF heatmaps and an RGB image recorded at the same time.
• Second, the human body is specular in the frequency
range that traverse walls [9]. RF specularity is a physical
phenomenon that occurs when the wavelength is larger
than the roughness of the surface. In this case, the object
acts like a reﬂector - i.e., a mirror - as opposed to a scat-
terer. The wavelength of our radio is about 5cm and hence
humans act as reﬂectors. Depending on the orientation
of the surface of each limb, the signal may be reﬂected
towards our sensor or away from it. Thus, in contrast
to camera systems where any snapshot shows all unoc-
cluded key-points, in radio systems, a single snapshot has
information about a subset of the limbs and misses limbs
and body parts whose orientation at that time deﬂects the
signal away from the sensor.
• Third, the wireless data has a different representation
(complex numbers) and different perspectives (horizon-
tal and vertical projections) from a camera.
The above properties have implications for pose estima-
tion, and need to be taken into account in designing a neural
network to extract poses from RF signals.
4. Method
Our model, illustrated in Fig. 3, follows a teacher-student
design. The top pipeline in the ﬁgure shows the teacher net-
work, which provides cross-modal supervision; the bottom
pipeline shows the student network, which performs RF-
based pose estimation.
4.1. Cross-Modal Supervision
One challenge of estimating human pose from RF sig-
nals is the the lack of labelled data. Annotating human pose
by looking at RF signals (e.g., Fig. 2) is almost impossi-
ble. We address this challenge by leveraging the presence
of well established vision models that are trained to predict
human pose in images [25, 7].
We design a cross-modal teacher-student network that
transfers the visual knowledge of human pose using syn-
chronized images and RF signals as a bridge. Consider a
synchronized pair of image and RF signals (I, R), where
37358

L
. . .
. . .
. . .
Horizontal Heatmaps
Vertical Heatmaps
RGB Frames
Keypoint Conﬁdence Maps
from Visual Inputs
from RF Signals
Keypoint Conﬁdence Maps
supervision
. . .
Horizontal RF Encoder Eh
Vertical RF Encoder Ev
Pose Decoder D
Student Network S
Teacher Network T
Figure 3: Our teacher-student network model used in RF-Pose. The upper pipeline provides training supervision, whereas the bottom pipeline learns to
extract human pose using only RF heatmaps.
R denotes the combination of the vertical and horizontal
heatmaps, and I the corresponding image in Fig. 2. The
teacher network T(·) takes the images I as input and pre-
dicts keypoint conﬁdence maps as T(I). These predicted
maps T(I) provide cross-modal supervision for the student
network S(·), which learns to predict keypoint conﬁdence
maps from the RF signals. In this paper, we adopt the 2D
pose estimation network in [10] as the teacher network. The
student network learns to predict 14 keypoint conﬁdence
maps corresponding to the following anatomical parts of the
human body: head, neck, shoulders, elbows, wrists, hips,
knees and ankles.
The training objective of the student network S(·) is to
minimize the difference between its prediction S(R) and
the teacher network’s prediction T(I):
min
S
X
(I,R)
L(T(I), S(R))
(1)
We deﬁne the loss as the summation of binary cross entropy
loss for each pixel in the conﬁdence maps:
L(T, S) = −
X
c
X
i,j
Sc
ij log Tc
ij + (1 −Sc
ij) log (1 −Tc
ij),
where Tc
ij and Sc
ij are the conﬁdence scores for the (i, j)-th
pixel on the conﬁdence map c.
4.2. Keypoint Detection from RF Signals
The design of our student network has to take into ac-
count the properties of RF signals. As mentioned earlier, the
human body is specular in the RF range of interest. Hence,
we cannot estimate the human pose from a single RF frame
( a single pair of horizontal and vertical heatmaps) because
the frame may be missing certain limbs tough they are not
occluded. Further, RF signals have low spatial resolution.
Hence, it will be difﬁcult to pinpoint the location of a key-
point using a single RF frame. To deal with these issues,
we make the network learn to aggregate information from
multiple snapshots of RF heatmaps so that it can capture
different limbs and model the dynamics of body movement.
Thus, instead of taking a single frame as input, we make the
network look at sequences of frames. For each sequence,
the network outputs keypoint conﬁdence maps as the num-
ber of frames in the input – i.e., while the network looks at
a clip of multiple RF frames at a time, it still outputs a pose
estimate for every frame in the input.
We also want the network to be invariant to translations
in both space and time so that it can generalize from visible
scenes to through-wall scenarios. Therefore, we use spatio-
temoral convolutions [22, 39, 42] as basic building blocks
for the student networks.
Finally, the student network needs to transform the in-
formation from the views of RF heatmaps to the view of
the camera in the teacher network (see Fig. 2). To do so,
the model has to ﬁrst learn a representation of the informa-
tion in the RF signal that is not encoded in original spatial
space, then decode that representation into keypoints in the
view of the camera. Thus, as shown in Fig. 3, our student
network has: 1) two RF encoding networks Eh(·) and Ev(·)
for horizontal and vertical heatmap streams, and 2) a pose
decoding network D(·) that takes a channel-wise concate-
nation of horizontal and vertical RF encodings as input and
predicts keypoint conﬁdence maps. The RF encoding net-
works uses strided convolutional networks to remove spa-
tial dimensions [48, 41] in order to summarize information
from the original views. The pose decoding network then
uses fractionally strided convolutional networks to decode
keypoints in the camera’s view.
47359

4.3. Implementation and Training
RF encoding network. Each encoding network takes 100
frames (3.3 seconds) of RF heatmap as input. The RF en-
coding network uses 10 layers of 9 × 5 × 5 spatio-temporal
convolutions with 1 × 2 × 2 strides on spatial dimensions
every other layer. We use batch normalization [21] followed
by the ReLU activation functions after every layer.
Pose decoding network. We combine spatio-temporal con-
volutions with fractionally strided convolutions to decode
the pose. The decoding network has 4 layers of 3 × 6 × 6
with fractionally stride of 1× 1
2 × 1
2, except the last layer has
one of 1 × 1
4 × 1
4. We use Parametric ReLu [17] after each
layer, except for the output layer, where we use sigmoid.
Training Details.
We represent a complex-valued RF
heatmap by two real-valued channels that store the real and
imaginary parts. We use a batch size of 24. Our networks
are implemented in PyTorch.
4.4. Keypoint Association
The student network generates conﬁdence maps for all
keypoints of all people in the scene. We map the keypoints
to skeletons as follows. We ﬁrst perform non-maximum
suppression on the keypoint conﬁdence maps to obtain dis-
crete peaks of keypoint candidates. To associate keypoints
of different persons, we use the relaxation method proposed
by Cao et al. [10] and we use Euclidean distance for the
weight of two candidates. Note that we perform association
on a frame-by-frame basis based on the learned keypoint
conﬁdence maps. More advanced association methods are
possible, but outside the scope of this paper.
5. Dataset
We collected synchronized wireless and vision data. We
attached a web camera to our RF sensor and synchronized
the images and the RF data with an average synchronization
error of 7 milliseconds.
We conducted more than 50 hours of data collection ex-
periments from 50 different environments (see Fig. 4), in-
cluding different buildings around our campus. The envi-
ronments span ofﬁces, cafeteria, lecture and seminar rooms,
stairs, and walking corridors. People performed natural ev-
eryday activities without any interference from our side.
Their activities include walking, jogging, sitting, reading,
using mobile phones and laptops, eating, etc. Our data in-
cludes hundreds of different people of varying ages. The
maximum and average number of people in a single frame
are 14 and 1.64, respectively. A data frame can also be
empty, i.e., it does not include any person. Partial occlu-
sions, where parts of the human body are hidden due to fur-
niture and building amenities, are also present. Legs and
arms are the most occluded parts.
Figure 4: Different environments in the dataset.
To evaluate the performance of our model on through-
wall scenes, we build a mobile camera system that has 8
cameras to provide ground truth when the people are fully
occluded. After calibrating the camera system, we construct
3D poses of people and project them on the view of the cam-
era colocated with RF sensor. The maximum and average
number of people in each frame in the through-wall testing
set are 3 and 1.41, respectively. This through-wall data was
only for testing and was not used to train the model.
6. Experiments
RF-Pose is trained with 70% of the data from visible
scenes, and tested with the remaining 30% of the data from
visible scenes and all the data from through-wall scenarios.
We make sure that the training data and test data are from
different environments.
6.1. Setup
Evaluation Metrics: Motivated by the COCO keypoints
evaluation [25] and as is common in past work [10, 29, 16],
we evaluate the performance of our model using the average
precision over different object keypoint similarity (OKS).
We also report AP50 and AP75, which denote the average
precision when OKS is 0.5 and 0.75, and are treated as loose
and strict match of human pose, respectively. We also report
AP, which is the mean average precision over 10 different
OKS thresholds ranging from 0.5 to 0.95.
Baseline: For visible and partially occluded scenes, we
compare RF-Pose with OpenPose [10], a state-of-the-art
vision-based model, that also acts as the teacher network.
Ground Truth: For visible scenes, we manually annotate
human poses using the images captured by the camera colo-
cated with our RF sensor. For through-wall scenarios where
the colocated camera cannot see people in the other room,
we use the eight-camera system described in 5 to provide
ground truth. We annotate the images captured by all eight
cameras to build 3D human poses and project them on the
57360

Methods
Visible scenes
Through-walls
AP
AP50
AP75
AP
AP50
AP75
RF-Pose
62.4
93.3
70.7
58.1
85.0
66.1
OpenPose[10]
68.8
77.8
72.6
-
-
-
Table 1: Average precision in visible and through-wall scenarios.
0.5
0.6
0.7
0.8
0.9
1.0
OKS
0.0
0.2
0.4
0.6
0.8
1.0
AP (%)
AP at Diﬀerent OKS
RFPose
OpenPose
Figure 5: Average precision at different OKS values.
Methods
Hea
Nec
Sho
Elb
Wri
Hip
Kne
Ank
RF-Pose
75.5
68.2
62.2
56.1
51.9
74.2
63.4
54.7
OpenPose[10]
73.0
67.1
70.8
64.5
61.5
71.4
68.4
68.3
Table 2: Average precision of different keypoints in visible scenes.
view of the camera colocated with the radio. We annotate
1000 randomly sampled images from the visible-scene test
set and another 1000 examples from the through-wall data.
6.2. Multi-Person Pose Estimation Results
We compare human poses obtained via RF signals with
the corresponding poses obtained using vision data. Ta-
ble 1 shows the performance of RF-Pose and the baseline
when tested on both visible scenes and through-wall sce-
narios. The table shows that, when tested on visible scenes,
RF-Pose is almost as good as the vision-based OpenPose
that was used to train it. Further, when tested on through-
wall scenarios, RF-Pose can achieve good pose estimation
while the vision-based baseline completely fail due to oc-
clusion.
The performance of RF-Pose on through-wall scenarios
can be surprising because the system did not see such ex-
amples during training. However, from the perspective of
radio signals, a wall simply attenuates the signal power, but
maintains the signal structure. Since our model is space in-
variant, it is able to identify a person behind a wall as similar
to the examples it has seen in the space in front of a wall.
An interesting aspect in Table 1 is that RF-Pose outper-
forms OpenPose for AP50, and becomes worse at AP75.
To further explore this aspect, we plot in Fig. 5 the av-
erage precision as a function of OKS values. The ﬁgure
shows that at low OKS values (< 0.7), our model outper-
forms the vision baseline. This is because RF-Pose predicts
less false alarm than the vision-based solution, which can
generate ﬁctitious skeletons if the scene has a poster of a
person, or a human reﬂection in a glass window or mirror.
In contrast, at high OKS values (> 0.75), the performance
of RF-Pose degrades fast, and becomes worse than vision-
based approaches. This is due to the intrinsic low spatial
resolution of RF signals which prevents them from pin-
pointing the exact location of the keypoints. The ability of
RF-Pose to exactly locate the keypoints is further hampered
by imperfect synchronization between the RF heatmaps and
the ground truth images.
Next, we zoom in on the various keypoints and com-
pare their performance.
Table 2 shows the average pre-
cision of RF-Pose and the baseline in localizing different
body parts including head, right and left shoulders, elbows,
wrists, hips, knees, and ankles. The results indicate that RF
signals are highly accurate at localizing the head and torso
(neck and hips) but less accurate in localizing limbs. This is
expected because the amount of RF reﬂections depends on
the size of the body part. Thus, RF-Pose is better at captur-
ing the head and torso, which have large reﬂective areas and
relatively slow motion in comparison to the limbs. As for
why RF-Pose outperforms OpenPose on some of the key-
points, this is due to the RF-based model operating over a
clip of a few seconds, whereas the OpenPose baseline oper-
ates on individual images.
Finally, we show a few test skeletons to provide a qual-
itative perspective. Fig. 6 shows sample RF-based skele-
tons from our test dataset, and compares them to the cor-
responding RBG images and OpenPose skeletons. The ﬁg-
ure demonstrates RF-Pose performs well in different envi-
ronments with different people doing a variety of everyday
activities. Fig. 7 illustrates the difference in errors between
RF-Pose and vision-based solutions. It shows that the errors
in vision-based systems are typically due to partial occlu-
sions, bad lighting 1, or confusing a poster or wall-picture
as a person. In contrast, errors in RF-Pose happen when
a person is occluded by a metallic structure (e.g., a metal-
lic cabinet in Fig. 7(b)) which blocks RF signals, or when
people are too close and hence the low resolution RF signal
fails to track all of them.
6.3. Model Analysis
We use guided back-propagation [38] to visualize the
gradient with respect to the input RF signal, and leverage
the information to provide insight into our model.
Which part of the RF heatmap does RF-Pose focus on?
Fig. 8 presents an example where one person is walking in
front of the wall while another person is hidden behind it.
Fig. 8(c) shows the raw horizontal heatmap. The two large
boxes are the rescaled versions of the smaller boxes and
zoom in on the two people in the ﬁgure. The red patch
indicated by the marker is the wall, and the other patches
are multipath effects and other objects.
The gradient in
Fig. 8(d) shows that RF-Pose has learned to focus its at-
1Images with bad lighting are excluded during training and testing.
67361

Figure 6: Pose estimation on different activities and environments. First row: Images captured by a web camera (shown as a visual reference). Second
row: Pose estimation by our model using RF signals only and without any visual input. Third row: Pose estimation using OpenPose based on images from
the ﬁrst row.
(a) Failure examples of OpenPose due to occlusioin, posters, and bad lighting.
(b) Failure examples of ours due to metal and crowd.
Figure 7: Common failure examples. First row: Images captured by a web camera (shown as a visual reference). Second row: Pose estimation by our
model using RF signals only and without any visual input. Third row: Pose estimation using OpenPose based on images from the ﬁrst row.
(a) RGB image
(b) Parsed poses
wall
(c) Horizontal Heatmap
(d) Gradients
Figure 8: Attention of the model across space
tention on the two people in the scene and ignore the wall,
other objects, and multipath.
How does RF-Pose deal with specularity? Due to the
specularity of the human body, some body parts may not re-
ﬂect much RF signals towards our sensor, and hence may be
RKnee
RAnkle
RShoulder
LWrist
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Time (s)
LElbow
t1
t2
t3
t4
Figure 9: Activation of different keypoints over time.
de-emphasized or missing in some heatmaps, even though
they are not occluded. RF-Pose deals with this issue by tak-
ing as input a sequences of RF frames (i.e., a video clip RF
heatmaps). To show the beneﬁt of processing sequences of
77362

−1.0
−0.5
0.0
0.5
1.0
Time (s)
Figure 10: Contribution of the neighbor
to the current frame.
# RF frames
AP
6
30.8
20
50.8
50
59.1
100
62.4
Table 3: Average precision
of pose estimation trained
on varying lengths of input
frames.
RF frames, we sum up the input gradient in all pixels in
the heatmaps to obtain activation per RF frame. We then
plot in Fig. 9 the activation as a function of time to visual-
ize the contribution of each frame to the estimation of var-
ious keypoints. The ﬁgure shows: that the activations of
the right knee (RKnee) and right ankle (RAnkle) are highly
correlated, and have peaks at time t1 and t2 when the per-
son is taking a step with her right leg. In contrast, her left
wrist (LWrist) gets activated after she raises her forearm at
t3, whereas her left elbow (LElbow) remains silent until t4
when she raises her backarm.
Fig. 9 shows that, for a single output frame, different RF
frames in the input sequence contribute differently to the
output keypoints. This emphasizes the need for using a se-
quence of RF frames at the input. But how many frames
should one use? Table 3 compares the model’s performance
for different sequence length at the input. The average pre-
cision is poor when the inout uses only 6 RF frames and
increases as the sequence length increases.
But
how
much
temporal
information
does
RF-Pose need?
Given a particular output frame, i,
we compute the contributions of each of the input frames
to it as a function of their time difference from i. To do
so, we back-propagate the loss of a single frame w.r.t. to
the RF heatmaps before it and after it, and sum up the
spatial dimensions. Fig. 10 shows the results, suggesting
that RF-Pose leverages RF heatmaps up to 1 second away
to estimate the current pose.
6.4. Identiﬁcation Using RF-Based Skeleton
We would like to show that the skeleton generated by
RF-Pose captures personalized features of the individuals
in the scene, and can be used by various recognition tasks.
Thus, we experiment with using the RF-based skeleton for
person identiﬁcation.
We conduct person identiﬁcation experiment with 100
people in two settings: visible environment, where the sub-
ject and RF device are in the same room, and through-wall
environment, where the RF device captures the person’s re-
ﬂections through a wall. In each setting, every person walks
naturally and randomly inside the area covered by our RF
device, and we collect 8 and 2 minutes data separately for
training and testing. The skeleton heatmaps are extracted
by the model trained on our pose estimation dataset, which
never overlaps with the identiﬁcation dataset. For each set-
ting, we train a 10-layer vanilla CNN to identify people
based on 50 consecutive frames of skeleton heatmaps.
Method
Visible scenes
Through-walls
Top1
Top3
Top1
Top3
RF-Pose
83.4
96.1
84.4
96.3
Table 4: Top1 and top3 identiﬁcation percent accuracy in visible and
through-wall settings.
Table 4 shows that RF-based skeleton identiﬁcation can
reach 83.4% top1 accuracy in visiable scenes. Interestingly,
even when a wall blocks the device and our pose extractor
never sees these people and such environments during train-
ing, the extracted skeletons can still achieve 84.4% top1 ac-
curacy, showing its robustness and generalizability regard-
less of the wall. As for top3 accuracy, we achieve more than
96% in both settings, demonstrating that the extracted skele-
ton can preserve most of the discriminative information for
identiﬁcation even though the pose extractor is never trained
or ﬁne-tuned on the identiﬁcation task.
7. Scope & Limitations
RF-Pose leverages RF signals to infer the human pose
through occlusions. However, RF signals and the solution
that we present herein have some limitations: First, the hu-
man body is opaque at the frequencies of interest – i.e., fre-
quencies that traverse walls. Hence, inter-person occlusion
is a limitation of the current system. Second, the operating
distance of a radio is dependent on its transmission power.
The radio we use in this paper works up to 40 feet. Fi-
nally, we have demonstrated that our extracted pose cap-
tures identifying features of the human body. However, our
identiﬁcation experiments consider only one activity: walk-
ing. Exploring more sophisticated models and identifying
people in the wild while performing daily activities other
than walking is left for future work.
8. Conclusion
Occlusion is a fundamental problem in human pose esti-
mation and many other vision tasks. Instead of hallucinat-
ing missing body parts based on visible ones, we demon-
strate a solution that leverages radio signals to accurately
track the 2D human pose through walls and obstructions.
We believe this work opens up exciting research opportuni-
ties to transfer visual knowledge about people and environ-
ments to RF signals, providing a new sensing modality that
is intrinsically different from visible light and can augment
vision systems with powerful capabilities.
Acknowledgments: We are grateful to all the human sub-
jects for their contribution to our dataset.
We thank the
NETMIT members and the anonymous reviewers for their
insightful comments.
87363

References
[1] Texas instruments. http://www.ti.com/. 3
[2] Walabot. https://walabot.com/. 3
[3] Reusing 60 ghz radios for mobile radar imaging. In In Pro-
ceedings of the 21st Annual International Conference on Mo-
bile Computing and Networking, 2015. 2
[4] F. Adib, C.-Y. Hsu, H. Mao, D. Katabi, and F. Durand. Cap-
turing the human ﬁgure through a wall. ACM Transactions
on Graphics, 34(6):219, November 2015. 1, 3
[5] F. Adib, Z. Kabelac, D. Katabi, and R. C. Miller. 3D track-
ing via body radio reﬂections. In Proceedings of the USENIX
Conference on Networked Systems Design and Implementa-
tion, NSDI, 2014. 1, 3
[6] F. Adib, H. Mao, Z. Kabelac, D. Katabi, and R. C. Miller.
Smart homes that monitor breathing and heart rate. In Pro-
ceedings of the 33rd Annual ACM Conference on Human
Factors in Computing Systems, 2015. 3
[7] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. 2D
human pose estimation: New benchmark and state of the
art analysis.
In Proceedings of the IEEE Conference on
computer Vision and Pattern Recognition, pages 3686–3693,
2014. 3
[8] Y. Aytar, C. Vondrick, and A. Torralba. Soundnet: Learning
sound representations from unlabeled video. In Advances in
Neural Information Processing Systems, NIPS, 2016. 2
[9] P. Beckmann and A. Spizzichino. The scattering of electro-
magnetic waves from rough surfaces. Norwood, MA, Artech
House, Inc., 1987. 3
[10] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime multi-
person 2D pose estimation using part afﬁnity ﬁelds. In Pro-
ceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, CVPR, 2017. 1, 2, 4, 5, 6
[11] L. Castrejon, Y. Aytar, C. Vondrick, H. Pirsiavash, and
A. Torralba. Learning aligned cross-modal representations
from weakly aligned data. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, CVPR,
2016. 2
[12] X. Chu, W. Ouyang, H. Li, and X. Wang. Structured feature
learning for pose estimation. In CVPR, 2016. 1
[13] X. Chu, W. Yang, W. Ouyang, C. Ma, A. L. Yuille, and
X. Wang. Multi-context attention for human pose estima-
tion. In CVPR, 2017. 1
[14] H. Fang, S. Xie, and C. Lu. Rmpe: Regional multi-person
pose estimation. arXiv preprint arXiv:1612.00137, 2016. 2
[15] G. Gkioxari, B. Hariharan, R. Girshick, and J. Malik. Us-
ing k-poselets for detecting people and localizing their key-
points. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, CVPR, 2014. 2
[16] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask r-cnn.
In Proceedings of the IEEE international conference on com-
puter vision, ICCV, 2017. 1, 2, 5
[17] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
rectiﬁers: Surpassing human-level performance on imagenet
classiﬁcation. In Proceedings of the IEEE international con-
ference on computer vision, ICCV, 2015. 5
[18] F. Hong, X. Wang, Y. Yang, Y. Zong, Y. Zhang, and Z. Guo.
Wﬁd: Passive device-free human identiﬁcation using WiFi
signal. In Proceedings of the 13th International Conference
on Mobile and Ubiquitous Systems: Computing, Networking
and Services. ACM, 2016. 3
[19] C.-Y. Hsu, Y. Liu, Z. Kabelac, R. Hristov, D. Katabi, and
C. Liu. Extracting gait velocity and stride length from sur-
rounding radio signals.
In Proceedings of the 2017 CHI
Conference on Human Factors in Computing Systems, CHI,
2017. 1, 3
[20] E. Insafutdinov, L. Pishchulin, B. Andres, M. Andriluka, and
B. Schiele. Deepercut: A deeper, stronger, and faster multi-
person pose estimation model. In Proceedings of the Euro-
pean Conference on Computer Vision, ECCV, 2016. 1, 2
[21] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
In Proceedings of the 32nd International Conference on Ma-
chine Learning, ICML, 2015. 5
[22] S. Ji, W. Xu, M. Yang, and K. Yu. 3D convolutional neural
networks for human action recognition. IEEE transactions
on pattern analysis and machine intelligence, January 2013.
4
[23] K. R. Joshi, D. Bharadia, M. Kotaru, and S. Katti. Wideo:
Fine-grained device-free motion tracing using rf backscat-
ter. In Proceedings of the USENIX Conference on Networked
Systems Design and Implementation, NSDI, 2015. 3
[24] M. Kotaru, K. Joshi, D. Bharadia, and S. Katti.
Spotﬁ:
Decimeter level localization using wiﬁ. In ACM SIGCOMM
Computer Communication Review, 2015. 2
[25] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-
manan, P. Doll´ar, and C. L. Zitnick. Microsoft COCO: Com-
mon objects in context. In Proceedings of the European con-
ference on computer vision, ECCV, 2014. 3, 5
[26] P. Melgarejo, X. Zhang, P. Ramanathan, and D. Chu. Lever-
aging directional antenna capabilities for ﬁne-grained ges-
ture recognition. In Proceedings of the 2014 ACM Interna-
tional Joint Conference on Pervasive and Ubiquitous Com-
puting, 2014. 3
[27] A. Newell and J. Deng.
Associative embedding: End-to-
end learning for joint detection and grouping. arXiv preprint
arXiv:1611.05424, 2016. 1
[28] A. Newell, K. Yang, and J. Deng. Stacked hourglass net-
works for human pose estimation. In European Conference
on Computer Vision, ECCV, 2016. 1
[29] G. Papandreou, T. Zhu, N. Kanazawa, A. Toshev, J. Tomp-
son, C. Bregler, and K. Murphy. Towards Accurate Multi-
person Pose Estimation in the Wild. In CVPR, 2017. 2, 5
[30] T. Pﬁster, J. Charles, and A. Zisserman. Flowing convnets
for human pose estimation in videos. In Proceedings of the
IEEE International Conference on Computer Vision, ICCV,
2015. 1
[31] L. Pishchulin, E. Insafutdinov, S. Tang, B. Andres, M. An-
driluka, P. V. Gehler, and B. Schiele. Deepcut: Joint sub-
set partition and labeling for multi person pose estimation.
In Proceedings