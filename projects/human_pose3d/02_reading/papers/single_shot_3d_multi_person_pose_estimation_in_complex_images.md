# Single-shot 3D multi-person pose estimation in complex images

> 2020 · id: W2986352270 · arXiv: 1911.03391 · pdf: https://arxiv.org/pdf/1911.03391 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this paper, we propose a new single shot method for multi-person 3D hu-
man pose estimation in complex images. The model jointly learns to locate
the human joints in the image, to estimate their 3D coordinates and to group
these predictions into full human skeletons. The proposed method deals with
a variable number of people and does not need bounding boxes to estimate
the 3D poses. It leverages and extends the Stacked Hourglass Network and
its multi-scale feature learning to manage multi-person situations. Thus, we
exploit a robust 3D human pose formulation to fully describe several 3D hu-
man poses even in case of strong occlusions or crops. Then, joint grouping
and human pose estimation for an arbitrary number of people are performed
using the associative embedding method. Our approach signiﬁcantly outper-
forms the state of the art on the challenging CMU Panoptic and a previous
single shot method on the MuPoTS-3D dataset. Furthermore, it leads to
good results on the complex and synthetic images from the newly proposed
JTA Dataset.
Keywords:
multi-person, 3D, human pose, deep learning

## introduction
3D human pose is a low dimensional and interpretable representation
which is used a lot in action recognition [1].
3D human pose estimation
based on RGB images is a challenging task from the computer vision per-
spective. Recent Convolution Neural Network (CNN) based approaches [2, 3]
Preprint submitted to Pattern Recognition
January 8, 2021
arXiv:1911.03391v2  [cs.CV]  7 Jan 2021

achieve excellent performance in 2D human pose estimation thanks to large
scale in the wild datasets. Nevertheless, methods for 3D human pose esti-
mation require 3D ground truth that is only available using Motion Capture
(Mocap) systems [4, 5, 6]. Therefore, these methods have good performance
in controlled environment but bad generalisation to real in the wild images.
Furthermore, most of the 3D pose estimation methods are restricted to a
single fully visible subject. In real-world scenarios, multiple people interact
in cluttered or even crowded scenes containing both self-occlusions of the
body and strong inter-person occlusions. Therefore, inferring the 3D pose
of all the subjects (without knowing in advance their number) from a single
and monocular RGB image is a harder problem and recent single-person 3D
human pose estimation methods fail in this case.
A natural approach is to decompose the multi-person ill-posed problem
into multiple single-person 3D estimations. These top-down approaches are
based on the generation of multiple pose proposals that are evaluated and
reﬁned in a second time [7]. Thus, they perform many redundant estimations
and scale badly for a large number of subjects.
Another way to solve this problem is bottom-up strategy [8, 9, 10] that
manages the whole scene in a single forward pass to give multi-person 3D hu-
man pose estimates. By their principle, they are more eﬀective in managing
occlusions between people and take advantage of context-related information
to predict the diﬀerent poses.
In the present article, we propose a new bottom-up approach that man-
ages the whole scene in a single forward pass to give multi-person 3D human
pose estimates.
Our method is based on the Stacked Hourglass architec-
ture [11] that has demonstrated its eﬀectiveness for 2D human pose estima-
tion. Single shot multi-person 3D human pose estimation is challenging as
it needs to properly locate human joints and to regroup these estimations
into ﬁnal 3D skeletons. By associating the Hourglass architecture with a
powerful joints grouping method named the associative embedding [3] and
a robust multi-person 3D pose description [10], we design an end-to-end ar-
chitecture that jointly performs 2D human joints detection, joints grouping
and full body 3D human pose estimation even when the subjects are par-
tially occluded or truncated by the image boundary. The proposed method
surpasses state of the art results on the CMU-Panoptic [12] dataset, achieves
higher accuracy than a state of the art single-shot method on the MuPoTS-
3D dataset [10], and shows good results on the Joint Track Auto dataset[13],
a synthetic but realistic dataset with a large number of people, various cam-
2

era viewpoints and backgrounds. So far, this dataset has only been used for
joint tracking.

## method
Nb of HG
ORPM
Haggling
Maﬁa
Ultimatum
Pizza
Mean
Ours, 1-HG
1
92.3
86.1
82.7
103.8
91.8
Ours, 2-HG
2
77.1
74.8
68.0
89.8
78.3
Ours, 3-HG
3
72.4
72.4
60.12
85.2
73.8
Ours, NR
4
×
101.5
124.2
105.7
130.3
118.8
Ours, full
4
70.1
66.6
55.6
78.4
68.5
Table 3: Mean per joint position error (MPJPE) in mm on the Panoptic Dataset following
Panoptic-1 protocol. (i-HG stands for i stacked hourglasses and NR for Naive Readout).
module to 68.5 mm for our full four hourglass modules model). This shows
the importance of the stacking scheme and the reﬁnement process in the
model architecture.
The penultimate line of this table shows the results
obtained with four hourglass modules and a Naive Readout (NR) in the
ORPM, that means when the 3D joint coordinates are read directly from
their 2D positions. Because of frequent crops and occlusions in the panoptic
dataset, this model has poor performance with an MPJPE of 118.8 mm.
This proves the importance of the ORPM storage redundancy to manage
occlusion. Our complete model(last row) with four hourglass modules and the
readout procedure described in Section 3.2 has the lowest MPJPE (68.5mm)
Examples of 3D human pose estimations on the Panoptic dataset are
shown in Figure 3. Our method can estimate the 3D pose of multiple people
even in case of truncation (1st, 2nd and last rows) or people overlap (2nd
and 4th rows)
Robustness to the number of training cameras : Protocols Panop-
tic 1 and 2 results are obtained by using a large number of training cameras.
What is the robustness of our model when using a reduced number of cam-
eras ? Table 4 provides Panoptic 3 protocol results. Panoptic 3a and 3b
results show that even by using only half and fourth of the training cam-
14

Protocol
Haggling
Maﬁa
Ultimatum
Pizza
Mean
Panoptic 2
78.3
60.7
84.2
78.3
68.1
Panoptic 3a
82.4
64.3
88.7
82.2
72.3
Panoptic 3b
84.0
74.2
87.4
92.0
76.4
Panoptic 3c
149.4
151.3
155.5
167.9
150.9
Panoptic 4
79.4
79.4
Table 4: Mean per joint position error (MPJPE) in mm on the Panoptic Dataset following
Panoptic-2, Panoptic-3 and Panoptic-4 protocols
eras, the MPJPE is only increased respectively by 6.9% and 12.2%. On the
other hand, where only 3 training cameras are used, the MPJPE is 2.2 times
greater than the Panoptic 2 MPJPE. This number of cameras is insuﬃcient
to learn such a complex task. Even single person 3D human pose models are
trained on datasets[5, 45] that provides images from four cameras or more.
Performance on an useen scenario:
Protocols Panoptic 1,2 and
3 show the ability of the model to generalise to unseen camera viewpoints.
Panoptic 4 results show the ability of the model to generalise to new scenarios.
The model is trained only on the Haggling, Maﬁa and Ultimatum scenarios
and evaluated on the unseen Pizza scenario. The Panoptic 4 MPJPE (79.4)
is close the MPJPE obtained on the Panoptic 2 protocol for the Pizza sce-
nario showing that model does not overfeat on the training scenarios and can
generalise to new ones.
4.3. Multi-person 3D pose estimation on JTA dataset
JTA (Joint Track Auto) is a dataset for human pose estimation and track-
ing in urban environment. It was collected from the realistic video-game the
Grand Theft Auto V and contains 512 HD videos of 30 seconds recorded at
30 fps. The collected videos feature a vast number of diﬀerent body poses,
in several urban scenarios at varying illumination conditions and viewpoints.
People perform diﬀerent actions like walking, sitting, running, chatting, talk-
ing on the phone, drinking or smoking. Each image contains a number of
people ranging between 0 and 60 with an average of more than 21 people.
The distance from the camera ranges between 0.1 to 100 meters, resulting
in pedestrian heights between 20 and 1100 pixels. None existing (virtual
or real) dataset with annotated 3D pose is comparable with JTA dataset
in terms of number of people per image, people and background variability.
As far as we know, we are the ﬁrst to demonstrate the ability of a trained
15

model to deal with such complex and rich environments with many people
at diﬀerent camera distances and with diﬀerent resolutions. 256 videos are
used for training and 128 for testing (the remaining 128 videos are used for
validation). From the testing videos, we take one frame every ten frames for
the evaluation.
Table 5 presents per camera distance results on the the JTA Dataset.
We evaluate our model on this dataset at diﬀerent resolutions (S1=512px,
S2=1024px and S3=1536px) and also with the multi-scale inference described
in section 3.5. The images from this dataset contain a large number of people
in various distances from the camera. The distance from the camera can
have a signiﬁcant impact on the performance of a 3D human pose estimator.
Indeed, distant people require higher image resolution and are more likely
to be occulted. For this reason, we provide in Table 5 results for people
in diﬀerent ranges of distance from the camera. Note that our testing set
contains 262510 people. Among these people, 10% have a distance from the
camera less than 10 meters, 23% have a distance from the camera between
10 and 20 meters, 21% have a distance from the camera between 20 and 30
meters, 14% have a distance from the camera between 30 and 40 meters and
31% have a distance from the camera greater than 40 meters.
The resolution having the best overall 3DPCKr is the resolution S2 with a
3DPCKr of 37.8%. This resolution performs a good compromise to estimate
the pose of the high resolution people that S3 cannot handle properly and
low resolution people that are too small from scale S1. Resolution S1 has the
best results for people that are close to the camera (less than 10 meters) with
an MPJPE of 165.2mm and a 3DPCKr of 68.5%. Resolution S2 has the best
results for people that have a distance from the camera between 10 and 20
meters with a 3DPCKr of 62.3% and an MPJPE of 194.50. Resolution S3
has the best results for people that are far from the cameras (greater than
20 meters). These results show that each resolution is adequate to a given
range of people distance and consequently to a resolution of people.
The multi-scale inference (MSI) improves the overall 3DPCKr and MPJPE.
The 3DPCKr goes from 37.8 to 43.9 for the MSI and the MPJPE goes from
258.9mm to 193.5mm. MSI has better results than scale S2 and S3 for close
to the camera people (less than 10 meters) taking advantages from poses es-
timated from scale S1 but without improving over this scale for these people.
It surpasses all the scales for people that have a distance from the camera
greater than 10 meters.
Joint-wise analysis (Table 6) shows that the results are unequal from one
16

joint to another one. Regardless of the distance to the camera, spines and
hips are always the best estimated joints. These articulations have a reduced
variability compared to the extremity joints like wrists and ankles that have
the worst MPJPE and 3DPCKr. Indeed, since the 3D joint coordinates are
expressed relatively to their parents joints in the kinematic tree and converted
to pelvis relative locations, errors in the estimation of a parent joint impact
the estimation of all its descendent in the kinematic tree. One way to solve
this issue could be to express the joints’ coordinates relatively to more stable
joints than their parent joints. For instance, the coordinates of the wrist
could be expressed relatively to the elbow but also to the shoulder which is
more stable and less prone to errors. A mechanism would be necessary to
fuse these predictions expressed relatively to diﬀerent joints and chose the
more precise one. We leave this for a future work.
Examples of 3D human pose estimations on the JTA dataset are shown in
Figure 4. Our method can estimate the 3D pose in several urban scenarios
at varying illumination conditions and viewpoints.
Nevertheless, very far
people are not detected and the method fails in case of crowded people.
4.4. Multi-person 3D pose estimation on MuPoTS-3D dataset
MuPoTS-3D [10] is a dataset containing 20 indoor and outdoor sequences
with ground truth 3D poses for up to three subjects. Like Mehta et. al [10],
our model is trained on the MuCo-3DHP dataset that has been generated by
compositing the existing MPI-INF-3DHP 3D single-person pose estimation
dataset [6] and the COCO-dataset [58] to ensure better generalisation. Each
mini-batch consists of half MuCo-3DHP and half COCO images. For COCO
data, the loss value for the ORPM is set to zero.
We compare our approach with the single-shot approach proposed by
Mehta et al. [10] and a recent two-stage approach [54]. Like [10], our model
is based on the ORPM formulation but diﬀers in the stacked architecture used
and in the bottom-up joints association method. Table 7 provides 3DPCKr
results on this dataset. Our model achieves higher accuracy with a 3DPCKr
of 67.5% ( 72.7% when evaluating only on well detected people) compared
to the approach of Mehta et.
al [10] that has a 3DPCKr of 6

## experiments
In this paper, we address the problem of single shot multi-person 3D
human pose estimation. To evaluate our method, we perform separate ex-
periments on:
• single-person 3D pose estimation in a controlled environment (Human
3.6M dataset [56])
• multi-person 3D pose estimation in a controlled environment (CMU-
Panoptic dataset [12]); some images are depicted in Figure 3.
• multi-person 3D pose estimation in outdoor and indoor scenes (MuPoTS-
3D dataset [10]).
10

• multi-person 3D pose estimation in virtual environments with many
people (JTA dataset [13]). This dataset is more complex and richer
than the previous one. Some images are shown in Figure 4. No previ-
ous method for 3D human pose estimation has been evaluated on this
dataset to the best of our knowledge.
Evaluation Metrics:
To evaluate our Multi-Person 3D pose approach,
we use two metrics.
The ﬁrst one is the Mean per Joint Position Error
(MPJPE) that corresponds to mean Euclidean distance between ground truth
and prediction for all people and all joint. The second one is the 3DPCK
which is the 3D extension of the Percentage of Correct Keypoints (PCK)
metric used for 2D Pose evaluation, as well. A joint is considered correctly
estimated if the error in its estimation is less than 150mm. If an annotated
subject is not detected by our approach, we consider all of its joints to be
incorrect in the 3DPCK metric. We distinguish between 3DPCKr that is
calculated after root joints alignment and 3DPCKa that is calculated in the
orginal camera 3D space.
Training Procedure: The method was implemented with PyTorch. The
hourglass component is based on the public code in [3]. We used four stacked
hourglasses in our model, each one outputting 2D heatmaps, ORPM and
associative embeddings. We trained the model using mini-batches of size 30
on 8 Nvidia Titan X GPU during 240k iterations. We used the Adam[57]
optimiser with an initial learning rate of 10−4.
4.1. Single-person 3D pose estimation on Human 3.6M
Human 3.6M [56] is a dataset containing 3.6 million single-person RGB
images with 3D human poses annotated by MoCap systems. We used the
standard protocol for the evaluation: S1, S5, S6, S7 and S8 subjects for
training and the subjects S9 and S11 for testing.
Table 1 provides results of our method on the Human 3.6M dataset. Let
us notice that all models that achieve high performance on Human3.6m are
single-person models that take as input cropped images containing a single
fully visible subject. This setting is not representative of real world images
where people can be anywhere in the image, at various scales, truncated and
occulted by other people. The proposed model treats this general case and
produces reliable results in a single person setting with an MPJPE of 66.4
mm on the Human 3.6M dataset, better than most compared approaches. In
particular, it has a lower error than [10] that also uses ORPM but diﬀers in
the architecture used and in the joint grouping method.
11

Direction

## related_work
Human pose estimation is more and more studied as it is very useful for
many applications (e.g. motion capture, human image synthesis, activity
recognition, sign language recognition, robotics vision, etc.). In this section,
we present recent deep learning approaches for 2D human pose estimation
and single/multi-person 3D human pose estimation.
2D human pose estimation: Most methods for single-person 2D pose
estimation extract probabilistic maps called heatmaps that estimates the
probability of each pixel to contain a particular joint. At inference time, the
2D joint positions correspond to the local maxima of the heatmaps. Most of
these methods [11, 14] are also iterative. A reﬁned estimate of the heatmap
is obtained from the previous estimates and the convolutional features. Wei
et al.
[14] reﬁne the predictions over successive stages with intermediate
supervision at each stage. The Stacked Hourglass networks [11] processes
and consolidates features across scales to capture the spatial relationships of
the human body. Bin et al. [15] extend the Stacked Hourglass networks with
a Pose Graph Convolutional Network to model the structural relationships
between body key points. Li et al. [16] introduce a Temporal Consistency
Exploration module that captures geometric transformations between frames.
Both top-down and bottom-up human approaches have been proposed
for multi-person 2D human pose estimation. Top down methods [17, 18] ﬁrst
detect human bounding boxes and then estimate 2D human poses. Never-
theless, these methods fail when the detector fails, in particular when there
are strong occlusions. Bottom-up approaches [2, 3] ﬁrst estimate the 2D lo-
cation of each joint and then associate them into full skeletons. Cao et al.
[2] regress aﬃnity between joints that means the direction of the bones in
the image. Unlike this approach that needs complex post-processing joints,
Newell et al. [3] propose to learn this association in an end-to-end network
thanks to the Associative Embeddings. Zhao et al. [19] exploit multi-level
contextual association with a cluster-wise feature aggregation network.
Single-person 3D human pose estimation: Motivated by the recent
advances in 2D human pose estimation, some existing approaches [20, 21,
22, 23, 24, 25, 26, 27, 28, 29] use only 2D human poses estimated by other
3

methods [11, 2] to predict 3D human poses. Atrevi et al. [29] perform 2D
body silhouette matching to assign 3D joints. Chen and Ramanan [26] per-
forms a nearest neighbour search on a given 3D pose library with a large
number of 2D projections. Moreno-Noguer [27] formulate the problem of the
3D human pose estimation as a 2D to 3D distance matrix regression. Nie
et al. [28] predict depth on joints using LSTM. Martinez et al. [20] lift 2D
joints to 3D space using a deep residual neural network. Nevertheless, these
approaches are limited by the 2D pose estimator performance and do not
take into account important images clues, such as contextual information, to
make the prediction.
Other methods predict 3D human poses from images features[30, 31, 32,
33, 34]. Recent methods make this prediction directly from monocular images
[35, 36, 37, 38, 39, 40, 41] or from sequences of images [42, 43] using Convo-
lutional Neural Networks. The learning procedure needs images annotated
with 3D ground-truth pose. Since no large scale 3D in the wild annotated
dataset exists, current approaches tend to overfeat on the constrained envi-
ronment they have been trained on. The existing in the wild approaches use
either synthetic data [38, 39, 44] or are trained on both 3D and in the wild
2D datasets [45, 46, 47, 48, 49, 50, 51, 52]. Mehta et al. [45] use a pretrained
2D pose network to initialize the 3D pose regression network. Zhou et al.
use geometric constraints [50] in a weakly supervised setting. Pavlakos et
al. [51] take another approach by relying on weak 3D supervision in form
of a relative 3D ordering of joints which can be easily annotated even for in
the wild images. Yang et al. [52] use an adversarial loss that transfers the
3D human pose structures learned from the indoor annotated dataset to the
in-the-wild images. Although performing well with a single fully visible sub-
ject, these methods fail with several interacting people that are at diﬀerent
image scale and that occult each other.
Multi-person 3D human pose estimation: In a top-down approach,
Rogez et al. [7, 53] generate human pose proposals that are further reﬁned
using a regressor. Moon et al. [54] propose a camera distance aware multi-
person top-down approach that performs human detection (DetectNet), ab-
solute 3D human localisation (RootNet) and root relative 3D human pose
estimation (PoseNet) for each person independently. Zanﬁr et al. [9] esti-
mate the 3D human shape from sequences of frames using a pipeline process
followed by a 3D pose reﬁnement based on a non-linear optimisation process
and semantic constraints. MubyNet [8] is a bottom-up multi-task network
4

that identiﬁes joints and learns to score their possible associations as limbs.
These scores are used to solve a global optimisation problem that groups the
joints into full skeletons following the human kinematic tree. Mehta et al.
[10] propose an approach that predicts 2D heatmaps, part aﬃnity ﬁelds [2]
and Occlusions Robust Pose Maps (ORPM). This approach manages multi-
person 3D human pose estimation even for occluded and cropped people.
Nevertheless, the architecture used in [10] is not a stacked architecture while
the stacking strategy [2, 3] performs well in the 2D context.
The proposed method deals with multi-person 3D human pose estimation.
Unlike [9], it does not need sequence of images to reﬁne the pose estimates.
It is based on the stacked hourglass networks [11] devoted to mono-person
2D pose estimation and showing very good performance on this task. Thus,
we extend this approach using the multi-person 3D poses description robust
to occlusions proposed in [10] and the associative embedding [3] to group
joints into full skeletons. The ﬁnal network architecture is notably trained in
an end-to-end manner and the inference requires a single forward pass. Our
work is similar to Mehta et al. [10] as both methods perform bottom-up 3D
multi-person pose estimation but diﬀer in two ways. First, a stacked archi-
tecture is used while a ResNet-50 [55] is used in [10]. Recent works show
the eﬀectiveness of such a reﬁnement strategy for 2D pose estimation [2, 11]
but also for 3D pose estimation [50]. Secondly, our work diﬀer in the group-
ing method used to group joints’ detections into full human skeletons. Part
Aﬃnity Fields are used in [10] which may be a sub-optimal way of grouping
joints because the grouping is performed by solving a bipartite graph match-
ing problem while the associative embedding method is a more direct way
to perform this grouping. Indeed, no multi-stage pipelines is required in our
model and the network simultaneously learn to perform pose estimation and
joints’ grouping. Furthermore, the experimental results detailed in Newell
et al. [3] show that the associative embedding method is more eﬀective than
Part Aﬃnity Fields in a 2D context.
3. Proposed Method
3.1. Description
Given a monocular RGB image I of size W × H, we seek to estimate
the 3D human poses PI = {Pi | i ∈[1, . . . , N]} where N is the number of
visible people, Pi ∈R3×K are the 3D joints locations and K is the number of
predicted joints. The 3D joint coordinates are expressed relatively to their
5

parents joints in the kinematic tree and converted to pelvis relative locations
for evaluation in a 3D coordinate reference oriented like the camera one. The
model is composed of several stacked hourglass networks. The image is ﬁrst
sub-sampled to images features I’ of size W ′×H′ by convolution and pooling
layers. Each hourglass module outputs heatmaps for 2D joints detection,
ORPM for 3D joints localisation and associative embeddings maps for joint
grouping, each map being of size W ′ × H′. Except for the ﬁrst hourglass
that takes as input only image features I’, other hourglasses takes as input
images features I’ and the prediction of the previous hourglass that is reﬁned.
Figure 1 1 depicts an overview of the proposed method.
3.2. Occlusions Robust Pose Maps
Suppose we have an image I and the corresponding 3D poses PI. A good
3D pose representation to train a Convolutional Neural Network should have
the following characteristics:
• a ﬁxed dimension regardless of the number of people in the image;
• being robust to occlusions and crops.
To address these two problems, we adopt the ORPM formulation. For
each joint, each hourglass network outputs three maps of dimensions W ′×H′,
one for each X,Y,Z dimension. The size of these maps does not depend on
the number of visible people which allows the estimation of the 3D pose of
an arbitrary number of people. In these maps, the 3D joint coordinates of
each person are stored at diﬀerent 2D

## conclusion
Eating
Greet
Phone
Photo
Pose
Purchase
[35]
67.4
71.9
66.7
69.1
72.0
77.0
65.0
68.3
[20]
51.8
56.2
58.1
59.0
69.5
78.4
55.2
58.1
[45]
52.5
63.8
55.4
62.3
71.8
79.8
52.6
72.2
[36]
62.6
78.1
63.4
72.5
88.3
93.8
63.1
74.8
[50]
54.8
60.7
58.2
71.4
62.0
65.5
53.8
55.6
[7]
76.2
80.2
75.8
83.3
92.2
79.9
105.7
71.7
[10]
58.2
67.3
61.2
65.7
75.82
84.5
62.2
64.6
[21]
50.1
54.3
57.0
57.1
66.6
73.3
53.4
55.7
Ours
50.1
66.4
56.4
65.0
69.4
81.5
55.6
52.1
Sitting
SittingD
Smoke
Wait
WalkD
Walk
WalkT
AVG
[35]
83.7
96.5
71.7
65.8
74.9
59.1
63.2
71.9
[20]
74.0
94.6
62.3
59.1
65.1
49.5
52.4
62.9
[45]
86.2
120.6
66.0
64.0
76.8
48.9
53.7
68.6
[36]
106.6
138.7
93.8
73.9
82.0
55.8
59.6
80.5
[50]
75.2
111.6
64.2
66.1
51.4
63.2
55.3
64.9
[7]
105.9
127.1
88.0
83.7
86.6
64.9
84.0
87.7
[10]
82.0
93.0
68.8
65.1
72.0
57.6
63.6
69.9
[21]
72.8
88.6
60.3
57.7
62.7
47.5
50.6
60.4
Ours
83.8
115.4
62.7
64.4
78.1
48.0
53.1
66.4
Table 1: Mean per joint position error (MPJPE) in mm on the Human3.6M dataset.
4.2. Multi-person 3D pose estimation on CMU-Panoptic
CMU Panoptic [12] is a dataset containing images with several people
performing diﬀerent scenarios (playing an instrument, dancing, etc.) in a
dome where several cameras are setup. This dataset is challenging because
of complex interactions and diﬃcult camera viewpoints. We evaluate our
model following these protocols:
• Panoptic-1 protocol: it is the protocol used in [9, 8]. The model is eval-
uated on 9600 frames from HD cameras 16 and 30 and for 4 scenarios:
Haggling, Maﬁa, Ultimatum, Pizza. The model is trained on the other
28 HD cameras of this dataset.
• Panoptic-2 protocol: This protocol is an extension of the previous one.
Instead of evaluating on a subset of arbitrary selected frames, we eval-
uate on the entire sequences from cameras 16 and 30. The training
12

dataset in this protocol is the frames from all the HD cameras (except
cameras 16 and 30) for the Haggling, Maﬁa, Ultimatum, Pizza scenar-
ios. The model is evaluated on the same scenarios by taking one frame
every ten frames from HD cameras 16 and 30.
• Panoptic-3 protocol: Previous protocols use a large number of training
cameras. To evaluate the robustness to the number of cameras and
to the amount of training data, we propose protocol Panoptic-3. The
model is trained on the Haggling, Maﬁa, Ultimatum, Pizza scenarios
but only a subset of the training cameras is used:
– Panoptic 3a: HD cameras 0, 2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24,
26 and 28 are used during training
– Panoptic 3b: HD cameras 0,4,8,12,20,24 and 28 are used during
training
– Panoptic 3c: HD cameras 0,8, and 24 are used during training
The test set is the same as Panoptic 2.
• Panoptic-4 protocol : In the previous protocols, the model is trained
and evaluated on the same scenarios. To evaluate the robustness to an
unseen scenario in new camera viewpoints, we propose the Panoptic-4
protocol. The training dataset in this protocol is the frames from all
the HD cameras (except cameras 16 and 30) from the Haggling, Maﬁa
and Ultimatum scenarios. The model is evaluated on the pizza scenario
by taking one frame every ten frames from HD cameras 16 and 30.
Comparison with prior work: On Panoptic-1 protocol, our model
improves the results over the recent state of the art methods on all the
scenarios (Table 2). It shows a global improvement of 5.0% compared to [8].
Note that unlike [8] we do not learn on any frame from the cameras 16 and
30 and on any external data. Actually, the proposed model does not need
a trained attention readout process thanks to the eﬀective ORPM readout
process.
Ablative studies:
Table 3 provides ablative results of our method
following Panoptic-1 protocol on the Haggling, Maﬁa, Ultimatum and Pizza
scenarios. Firstly, we present the results obtained by stacking one, two or
three hourglass modules. Each time an hourglass module is added, the Mean
per Joint Position Error (MPJPE) decreases (from 91.8 mm for one hourglass
13