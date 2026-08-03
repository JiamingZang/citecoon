# Camera Distance-Aware Top-Down Approach for 3D Multi-Person Pose Estimation From a Single RGB Image

> 2019 · id: W2964784655 · arXiv: 1907.11346 · pdf: https://arxiv.org/pdf/1907.11346 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Although signiﬁcant improvement has been achieved re-
cently in 3D human pose estimation, most of the previ-
ous methods only treat a single-person case. In this work,
we ﬁrstly propose a fully learning-based, camera distance-
aware top-down approach for 3D multi-person pose esti-
mation from a single RGB image. The pipeline of the pro-
posed system consists of human detection, absolute 3D hu-
man root localization, and root-relative 3D single-person
pose estimation modules. Our system achieves comparable
results with the state-of-the-art 3D single-person pose es-
timation models without any groundtruth information and
signiﬁcantly outperforms previous 3D multi-person pose es-
timation methods on publicly available datasets. The code
is available in 1,2.
1https://github.com/mks0601/3DMPPE_ROOTNET_
RELEASE
2https://github.com/mks0601/3DMPPE_POSENET_
RELEASE

## introduction
The goal of 3D human pose estimation is to localize
semantic keypoints of single or multiple human bodies in
3D space. It is an essential technique for human behavior
understanding and human-computer interaction. Recently,
many methods [26, 37, 43, 44, 49, 52] utilize deep convo-
lutional neural networks (CNNs) and have achieved no-
ticeable performance improvement on large-scale publicly
available datasets [16,28].
Most of the previous 3D human pose estimation meth-
ods [26, 37, 43, 44, 49, 52] are designed for single-person
case. They crop the human area in an input image with a
groundtruth bounding box or the bounding box that is pre-
dicted from a human detection model [11]. The cropped
patch of a human body is fed into the 3D pose estimation
module, which then estimates the 3D location of each key-
point. As their models take a single cropped image, es-
timating the absolute camera-centered coordinate of each
keypoint is difﬁcult.
To handle this issue, many meth-
ods [26,37,43,44,49,52] estimate the relative 3D pose to a
reference point in the body, e.g., the center joint (i.e., pelvis)
arXiv:1907.11346v2  [cs.CV]  17 Aug 2019

of a human, called root. The ﬁnal 3D pose is obtained by
adding the 3D coordinates of the root to the estimated root-
relative 3D pose. Prior information on the bone length [37]
or the groundtruth [44] has been commonly used for the lo-
calization of the root.
Recently, many top-down approaches [6, 13, 47] for the
2D multi-person pose estimation have shown noticeable
performance improvement. These approaches ﬁrst detect
humans by using a human detection module, and then esti-
mate the 2D pose of each human by a 2D single-person pose
estimation module. Although they are straightforward when
used in 2D cases, extending them to 3D cases is nontrivial.
Note that for the estimation of 3D multi-person poses, we
need to know the absolute distance to each human from the
camera as well as the 2D bounding boxes. However, exist-
ing human detectors provide 2D bounding boxes only.
In this study, we propose a general framework for 3D
multi-person pose estimation. To the best of our knowl-
edge, this study is the ﬁrst to propose a fully learning-based
camera distance-aware top-down approach of which com-
ponents are compatible with most of the previous human
detection and 3D human pose estimation methods.
The
pipeline of the proposed system consists of three modules.
First, a human detection network (DetectNet) detects the
bounding boxes of humans in an input image. Second, the
proposed 3D human root localization network (RootNet)
estimates the camera-centered coordinates of the detected
humans’ roots. Third, a root-relative 3D single-person pose
estimation network (PoseNet) estimates the root-relative 3D
pose for each detected human. Figures 1 and 2 show the
qualitative results and overall pipeline of our framework,
respectively.
We show that our approach outperforms previous 3D
multi-person pose estimation methods [29, 40] on several
publicly available 3D single- and multi-person pose estima-
tion datasets [16,29] by a large margin. Also, even without
any groundtruth information (i.e., the bounding boxes and
the 3D location of the roots), our method achieves compara-
ble performance with the state-of-the-art 3D single-person
pose estimation methods that use the groundtruth in the
inference time. Note that our framework is new but fol-
lows previous conventions of object detection and 3D hu-
man pose estimation networks. Thus, previous detection
and pose estimation methods can be easily plugged into
our framework, which makes the proposed framework quite
ﬂexible and generalizable.
Our contributions can be summarized as follows.
• We propose a new general framework for 3D multi-
person pose estimation from a single RGB image. The
framework is the ﬁrst fully learning-based, camera
distance-aware top-down approach, of which compo-
nents are compatible with most of the previous human
detection and 3D human pose estimation models.
• Our framework outputs the absolute camera-centered
coordinates of multiple humans’ keypoints. For this,
we propose a 3D human root localization network
(RootNet). This model makes it easy to extend the 3D
single-person pose estimation techniques to the abso-
lute 3D pose estimation of multiple persons.
• We show that our method signiﬁcantly outperforms
previous 3D multi-person pose estimation methods on
several publicly available datasets. Also, it achieves
comparable performance with the state-of-the-art 3D
single-person pose estimation methods without any
groundtruth information.

## method
Dir.
Dis.
Eat
Gre.
Phon.
Pose
Pur.
Sit
SitD.
Smo.
Phot.
Wait
Walk
WalkD.
WalkP.
Avg
With groundtruth information in inference time
Chen [5]
89.9
97.6
90.0
107.9
107.3
93.6
136.1
133.1
240.1
106.7
139.2
106.2
87.0
114.1
90.6
114.2
Tome [46]
65.0
73.5
76.8
86.4
86.3
68.9
74.8
110.2
173.9
85.0
110.7
85.8
71.4
86.3
73.1
88.4
Moreno [32]
69.5
80.2
78.2
87.0
100.8
76.0
69.7
104.7
113.9
89.7
102.7
98.5
79.2
82.4
77.2
87.3
Zhou [53]
68.7
74.8
67.8
76.4
76.3
84.0
70.2
88.0
113.8
78.0
98.4
90.1
62.6
75.1
73.6
79.9
Jahangiri [17]
74.4
66.7
67.9
75.2
77.3
70.6
64.5
95.6
127.3
79.6
79.1
73.4
67.4
71.8
72.8
77.6
Mehta [28]
57.5
68.6
59.6
67.3
78.1
56.9
69.1
98.0
117.5
69.5
82.4
68.0
55.3
76.5
61.4
72.9
Martinez [26]
51.8
56.2
58.1
59.0
69.5
55.2
58.1
74.0
94.6
62.3
78.4
59.1
49.5
65.1
52.4
62.9
Fang [7]
50.1
54.3
57.0
57.1
66.6
53.4
55.7
72.8
88.6
60.3
73.3
57.7
47.5
62.7
50.6
60.4
Sun [43]
52.8
54.8
54.2
54.3
61.8
53.1
53.6
71.7
86.7
61.5
67.2
53.4
47.1
61.6
63.4
59.1
Sun [44]
47.5
47.7
49.5
50.2
51.4
43.8
46.4
58.9
65.7
49.4
55.8
47.8
38.9
49.0
43.8
49.6
Ours (PoseNet)
50.5
55.7
50.1
51.7
53.9
46.8
50.0
61.9
68.0
52.5
55.9
49.9
41.8
56.1
46.9
53.3
Without groundtruth information in inference time
Rogez [40]
76.2
80.2
75.8
83.3
92.2
79.9
71.7
105.9
127.1
88.0
105.7
83.7
64.9
86.6
84.0
87.7
Mehta [29]
58.2
67.3
61.2
65.7
75.8
62.2
64.6
82.0
93.0
68.8
84.5
65.1
57.6
72.0
63.6
69.9
Rogez [41]∗
55.9
60.0
64.5
56.3
67.4
71.8
55.1
55.3
84.8
90.7
67.9
57.5
47.8
63.3
54.6
63.5
Ours (Full)
51.5
56.8
51.2
52.2
55.2
47.7
50.9
63.3
69.9
54.2
57.4
50.4
42.5
57.5
47.7
54.4
Table 4: MPJPE comparison with state-of-the-art methods on the Human3.6M dataset using Protocol 2. ∗used extra synthetic
data for training.
Disjointed pipeline. To demonstrate the effectiveness
of the disjointed pipeline (i.e., separated DetectNet, Root-
Net, and PoseNet), we compare MRPE, MPJPE, and run-
ning time of joint and disjointed learning of the RootNet
and PoseNet in Table 1. The running time includes De-
tectNet and is measured using a single TitanX Maxwell
GPU. For the joint learning, we combine the RootNet and
PoseNet into a single model which shares backbone part
(i.e., ResNet [12]). The image feature from the backbone
is fed to each branch of RootNet and PoseNet in a parallel
way. Compared with the joint learning, our disjointed learn-
ing gives lower error under a similar running time. We be-
lieve that this is because each task of RootNet and PoseNet
is not highly correlated so that jointly training all tasks can
make training harder, resulting in lower accuracy.
Effect of the DetectNet. To show how the performance
of the human detection affects the accuracy of the ﬁnal 3D
human root localization and 3D multi-person pose estima-
tion, we compare AProot
25 , AUCrel, and 3DPCKabs using
the DetectNet in various backbones (i.e., ResNet-50 [12],
ResNeXt-101-32 [48]) and groundtruth box in the second,
third, and fourth row of Table 2, respectively. The table
shows that based on the same RootNet (i.e., Ours), better
human detection model improves both of the 3D human
root localization and 3D multi-person pose estimation per-
formance. However, the groundtruth box does not improve
overall accuracy considerably compared with other Detect-
Net models. Therefore, we have sufﬁcient reasons to be-
lieve that the given boxes cover most of the person instances
with such a high detection AP. We can also conclude that
the bounding box estimation accuracy does not have a large
impact on the 3D multi-person pose estimation accuracy.
Effect of the RootNet. To show how the performance
of the 3D human root localization affects the accuracy of
the 3D multi-person pose estimation, we compare AUCrel
and 3DPCKabs using various RootNet settings in Table 2.
The ﬁrst and second rows show that based on the same
DetectNet (i.e., R-50), our RootNet exhibits signiﬁcantly

## experiments
8.1. Dataset and evaluation metric
Human3.6M dataset. Human3.6M dataset [16] is the
largest 3D single-person pose benchmark.
It consists of
3.6 millions of video frames. 11 subjects performing 15
activities are captured from 4 camera viewpoints.
The
groundtruth 3D poses are obtained using a motion capture
system. Two evaluation metrics are widely used. The ﬁrst
one is mean per joint position error (MPJPE) [16], which
is calculated after aligning the human root of the estimated
and groundtruth 3D poses. The second one is MPJPE af-
ter further alignment (i.e., Procrustes analysis (PA) [10]).
This metric is called PA MPJPE. To evaluate the localiza-
tion of the absolute 3D human root, we introduce the mean
of the Euclidean distance between the estimated coordinates
of the root R and the groundtruth R∗, i.e., the mean of the
root position error (MRPE), as a new metric:
MRPE = 1
N
N
X
i=1
||R(i) −R(i)∗||2,
(5)
where superscript i is the sample index, and N denotes the
total number of test samples.
MuCo-3DHP and MuPoTS-3D datasets.
These are
the 3D multi-person pose estimation datasets proposed by
Mehta et al. [29]. The training set, MuCo-3DHP, is gen-
erated by compositing the existing MPI-INF-3DHP 3D
single-person pose estimation dataset [28].
The test set,
MuPoTS-3D dataset, was captured at outdoors and it in-
cludes 20 real-world scenes with groundtruth 3D poses for
up to three subjects. The groundtruth is obtained with a
multi-view marker-less motion capture system. For evalua-
tion, a 3D percentage of correct keypoints (3DPCKrel) and
area under 3DPCK curve from various thresholds (AUCrel)
is used after root alignment with groundtruth.
It treats
a joint’s prediction as correct if it lies within a 15cm
from the groundtruth joint location. We additionally deﬁne
3DPCKabs which is the 3DPCK without root alignment to
evaluate the absolute camera-centered coordinates. To eval-
uate the localization of the absolute 3D human root, we use
Settings
MRPE
MPJPE
Time
Joint learning
138.2
116.7
0.132
Disjointed learning (Ours)
120.0
57.3
0.141
Table 1: MRPE, MPJPE, and seconds per frame compari-
son between joint and disjointed learning on Human3.6M
dataset.
DetectNet RootNet AP box
AP root
25
AUCrel
3DPCKabs
R-50
k
43.8
5.2
39.2
9.6
R-50
Ours
43.8
28.5
39.8
31.5
X-101-32 Ours
45.0
31.0
39.8
31.5
GT
Ours
100.0
31.4
39.8
31.6
GT
GT
100.0
100.0
39.8
80.2
Table 2: Overall performance comparison for different De-
tectNet and RootNet settings on the MuPoTS-3D dataset.
the average precision of 3D human root location (AP root
25
)
which considers a prediction is correct when the Euclidean
distance between the estimated and the groundtruth coordi-
nates is smaller than 25cm.
8.2. Experimental protocol
Human3.6M dataset. Two experimental protocols are
widely used. Protocol 1 uses six subjects (S1, S5, S6, S7,
S8, S9) in training and S11 in testing. PA MPJPE is used
as an evaluation metric. Protocol 2 uses ﬁve subjects (S1,
S5, S6, S7, S8) in training and two subjects (S9, S11) in
testing. MPJPE is used as an evaluation metric. We use
every 5th and 64th frames in videos for training and testing,
respectively following [43,44]. When training, besides the
Human3.6M dataset, we used additional MPII 2D human
pose estimation dataset [1] following [37,43,44,52]. Each
mini-batch consists of half Human3.6M and half MPII data.
For MPII data, the loss value of the z-axis becomes zero for
both of the RootNet and PoseNet following Sun et al. [44].
MuCo-3DHP and MuPoTS-3D datasets.
Following
the previous protocol, we composite 400K frames of which
half are background augmented.
For augmentation, we
use images from the COCO dataset [25] except for images
with humans. We use an additional COCO 2D human key-
point detection dataset [25] when training our models on
the MuCo-3DHP dataset following Mehta et al. [29]. Each
mini-batch consists of half MuCo-3DHP and half COCO
data. For COCO data, loss value of z-axis becomes zero for
both of the RootNet and PoseNet following Sun et al. [44].
8.3. Ablation study
In this study, we show how each component of our pro-
posed framework affects the 3D multi-person pose estima-
tion accuracy. To evaluate the performance of the Detect-
Net, we use the average precision of bounding box (AP box)
following metrics of the COCO object detection bench-
mark [25].

## related_work
2D multi-person pose estimation. There are two main
approaches in the multi-person pose estimation. The ﬁrst
one, top-down approach, deploys a human detector that esti-
mates the bounding boxes of humans. Each detected human
area is cropped and fed into the pose estimation network.
The second one, bottom-up approach, localizes all human
body keypoints in an input image ﬁrst, and then groups them
into each person using some clustering techniques.
[6,13,30,31,34,47] are based on the top-down approach.
Papandreou et al. [34] predicted 2D offset vectors and 2D
heatmaps for each joint.
They fused the estimated vec-
tors and heatmaps to generate highly localized heatmaps.
Chen et al. [6] proposed a cascaded pyramid network whose
cascaded structure reﬁnes an initially estimated pose by fo-
cusing on hard keypoints. Xiao et al. [47] used a simple
pose estimation network that consists of a deep backbone
network and several upsampling layers.
[3,14,21,33,38] are based on the bottom-up approach.
Cao et al. [3] proposed the part afﬁnity ﬁelds (PAFs)
that model the association between human body keypoints.
They grouped the localized keypoints of all persons in the
input image by using the estimated PAFs. Newell et al. [33]
introduced a pixel-wise tag value to assign localized key-
points to a certain human. Kocabas et al. [21] proposed a
pose residual network for assigning detected keypoints to
each person.
3D single-person pose estimation. Current 3D single-
person pose estimation methods can be categorized into
single- and two-stage approaches.
The single-stage ap-
proach directly localizes the 3D body keypoints from the
input image. The two-stage methods utilize the high accu-
racy of 2D human pose estimation. They initially localize
body keypoints in a 2D space and lift them to a 3D space.
[23, 37, 43–45] are based on the single-stage approach.
Li et al. [23] proposed a multi-task framework that jointly
trains both the pose regression and body part detectors.
Tekin et al. [45] modeled high-dimensional joint depen-
dencies by adopting an auto-encoder structure. Pavlakos et

3D root-relative pose
Input image
RootNet
PoseNet
Cropped humans
y
x
3D absolute human root 
z
DetectNet
3D multi-person pose
Figure 2: Overall pipeline of the proposed framework for 3D multi-person pose estimation from a single RGB image. The
proposed framework can recover the absolute camera-centered coordinates of multiple persons’ keypoints.
al. [37] extended the U-net shaped network to estimate a 3D
heatmap for each joint. They used a coarse-to-ﬁne approach
to boost performance. Sun et al. [43] introduced composi-
tional loss to consider the joint connection structure. Sun et
al. [44] used soft-argmax operation to obtain the 3D coor-
dinates of body joints in a differentiable manner.
[4, 5, 7, 26, 35, 49, 52] are based on the two-stage ap-
proach. Park et al. [35] estimated the initial 2D pose and
utilized it to regress the 3D pose. Martinez et al. [26] pro-
posed a simple network that directly regresses the 3D coor-
dinates of body joints from 2D coordinates. Zhou et al. [52]
proposed a geometric loss to facilitate weakly supervised
learning of the depth regression module with images in the
wild. Yang et al. [49] utilized adversarial loss to handle the
3D human pose estimation in the wild.
3D multi-person pose estimation.
Few studies have
been conducted on 3D multi-person pose estimation from a
single RGB image. Rogez et al. [40] proposed a top-down
approach called LCR-Net, which consists of localization,
classiﬁcation, and regression parts. The localization part
detects a human from an input image, and the classiﬁca-
tion part classiﬁes the detected human into several anchor-
poses. The anchor-pose is deﬁned as a pair of 2D and root-
relative 3D pose. It is generated by clustering poses in the
training set. Then, the regression part reﬁnes the anchor-
poses. Mehta et al. [29] proposed a bottom-up approach
system. They introduced an occlusion-robust pose-map for-
mulation which supports pose inference for more than one
person through PAFs [3].
3D human root localization in 3D multi-person pose
estimation. Rogez et al. [40] estimated both the 2D pose in
the image coordinate space and the 3D pose in the camera-
centered coordinate space simultaneously. They obtained
the 3D location of the human root by minimizing the dis-
tance between the estimated 2D pose and projected 3D
pose, similar to what Mehta et al. [28] did. However, this
strategy cannot be generalized to other 3D human pose esti-
mation methods because it requires both the 2D and 3D esti-
mations. For example, many works [37,44,49,52] estimate
the 2D image coordinates and root-relative depth values of
keypoints.
As their methods do not output root-relative
camera-centered coordinates of keypoints, such a distance
minimization strategy cannot be used. Moreover, contex-
tual information cannot be exploited because the image fea-
ture is not considered. For example, it cannot distinguish
between a child close to the camera and an adult far from
the camera because their scales in the 2D image is similar.
3. Overview of the proposed model
The goal of our system is to recover the absolute
camera-centered coordinates of multiple persons’ keypoints
{Pabs
j
}J
j=1, where J denotes the number of joints.
To
address this problem, we construct our system based on
the top-down approach that consists of DetectNet, Root-
Net, and PoseNet. The DetectNet detects a human bound-
ing box of each person in the input image. The RootNet
takes the cropped human image from the DetectNet and
localizes the root of the human R = (xR, yR, ZR), in
which xR and yR are pixel coordinates, and ZR is an ab-
solute depth value. The same cropped human image is fed
to the PoseNet, which estimates the root-relative 3D pose
Prel
j
= (xj, yj, Zrel
j
), in which xj and yj are pixel coordi-
nates in the cropped image space and Zrel
j
is root-relative
depth value. We convert Zrel
j
into Zabs
j
by adding ZR and
transform xj and yj to the original input image space. Then,
the ﬁnal absolute 3D pose {Pabs
j
}J
j=1 is obtained by simple
back-projection.
4. DetectNet
We use Mask R-CNN [11] as the framework of De-
tectNet. Mask R-CNN [11] consists of three parts. The
ﬁrst one, backbone, extracts useful local and global fea-
tures from the input image by using deep residual network
(ResNet) [12] and feature pyramid network [24]. Based
on the extracted features, the second part, region pro-
posal network, proposes human bounding box candidates.
The RoIAlign layer extracts the features of each proposal

0
2000
4000
6000
8000
10000
12000
k value (mm)
0
1000
2000
3000
4000
5000
6000
7000
8000
Depth value of the human center joint (mm)
Human3.6M (r=0.50)
MuCo-3DHP (r=0.71)
Figure 3: Correlation between k and real depth value of
the human root. Human3.6M [16] and MuCo-3DHP [29]
datasets were used. r represents Pearson correlation coefﬁ-
cient.
and passes them to the third part, which is the classiﬁca-
tion head network. The head network determines whether
the given proposal is a human or not and estimates the
bounding box reﬁnement offsets. It achieves the state-of-
the-art performance on publicly available object detection
datasets [25]. Due to its high performance and publicly
available code [9, 27], we use Mask R-CNN [11] as a De-
tectNet in our pipeline.
5. RootNet
5.1. Model design
The RootNet estimates the camera-centered coordinates
of the human root R = (xR, yR, ZR) from a cropped hu-
man image. To obtain them, RootNet separately estimates
the 2D image coordinates (xR, yR) and the depth value (i.e.,
the distance from the camera ZR) of the human root. The
estimated 2D image coordinates are back-projected to the
camera-centered coordinate space using the estimated depth
value, which becomes the ﬁnal output.
Considering that an image provides sufﬁcient informa-
tion on where the human root is located in the image space,
the 2D estimation part can learn to localize it easily. By
contrast, estimating the depth only from a cropped human
image is difﬁcult because the input does not provide infor-
mation on the relative position of the camera and human.
To resolve this issue, we introduce a new distance measure,
k, which is deﬁned as follows:
k =
s
αxαy
Areal
Aimg
,
(1)
where αx, αy, Areal, and Aimg are focal lengths divided by
the per-pixel distance factors (pixel) of x- and y-axes, the
(a) Different area, same distance
(b) Same area, different distance
Figure 4: Examples where k fails to represent the dis-
tance between a human and the camera because of incorrect
Aimg.
area of the human in real space (mm2), and image space
(pixel2), respectively. k approximates the absolute depth
from the camera to the object using the ratio of the actual
area and the imaged area of it, given camera parameters.
Eq 1 can be easily derived by considering a pinhole camera
projection model. The distance d (mm) between the camera
and object can be calculated as follows:
d = αx
lx,r

## conclusion
We propose a novel and general framework for 3D multi-
person pose estimation from a single RGB image.
Our
framework consists of human detection, 3D human root lo-
calization, and root-relative 3D single-person pose estima-
tion models. Since any existing human detection and 3D
single-person pose estimation models can be plugged into
our framework, it is very ﬂexible and easy to use. The pro-
posed system outperforms previous 3D multi-person pose
estimation methods by a large margin and achieves compa-

rable performance with 3D single-person pose estimation
methods without any groundtruth information while they
use it in inference time. To the best of our knowledge, this
work is the ﬁrst to propose a fully learning-based camera
distance-aware top-down approach whose components are
compatible with most of the previous human detection and
3D human pose estimation models. We hope that this study
provides a new basis for 3D multi-person pose estimation,
which has only barely been explored.
Acknowledgments
This work was partially supported by the Visual Turing
Test project (IITP-2017-0-01780) from the Ministry of Sci-
ence and ICT of Korea.

Supplementary Material of “Camera
Distance-aware Top-down Approach for 3D
Multi-person Pose Estimation from a Single
RGB Image”
In this supplementary material, we present more ex-
perimental results that could not be included in the main
manuscript due to the lack of space.
1. Derivation of Equation 1
We provide a derivation of Equation 1 of the main
manuscript with reference to Figure 6 ,which shows a pin-
hole camera model. The green and blue arrows represent
the human root joint centered x and y-axes, respectively.
The yellow lines show rays, and c is the hole. d, f, and
lsensor are distance between camera and the human root
joint (mm), focal length (mm), and the length of human
on the image sensor (mm), respectively.
According to the deﬁnition of tan,
tan θx = 0.5lx,real
d
= 0.5lx,sensor
f
,
Let px be per pixel distance factor in x-axis. Then,
d = f lx,real
lx,sensor
= fpx
lx,real
lx,sensorpx
= αx
lx,real
lx,img
,
Above equations are also valid in y-axis. Therefore,
d = f ly,real
ly,sensor
= fpy
ly,real
ly,sensorpy
= αy
ly,real
ly,img
,
Finally,
d =
s
αxαy
lx,real
lx,img
ly,real
ly,img
=
s
αxαy
Areal
Aimg
.
2. Comparison of 3D human root localization
with previous approaches
We compare previous absolute 3D human root localiza-
tion methods [28,40] with the proposed RootNet on the Hu-
man3.6M dataset [16] based on protocol 2.
Previous approaches [28,40] simultaneously estimate 2D
image coordinates and 3D camera-centered root-relative co-
ordinates of keypoints. Then, absolute camera-centered co-
ordinates of the human root are obtained by minimizing the
distance between 2D predictions and projected 3D predic-
tions. For optimization, linear least-squares formulation is
used. To measure the errors of their method, we imple-
mented and used ResNet-152-based model of Sun et al. [44]
as a 2D pose estimator and model of Martinez et al. [26]
as a 3D pose estimator, which are state-of-the-art methods.
In addition, to minimize the effect of outliers in 3D-to-2D
Figure 6: Visualization of a pinhole camera model.