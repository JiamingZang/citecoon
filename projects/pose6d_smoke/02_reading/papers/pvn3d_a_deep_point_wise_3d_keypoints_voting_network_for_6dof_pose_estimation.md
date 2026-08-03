# PVN3D: A Deep Point-Wise 3D Keypoints Voting Network for 6DoF Pose Estimation

> 2020 · id: W3034986117 · arXiv: 1911.04231 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this work, we present a novel data-driven method for
robust 6DoF object pose estimation from a single RGBD im-
age. Unlike previous methods that directly regressing pose
parameters, we tackle this challenging task with a keypoint-
based approach. Speciﬁcally, we propose a deep Hough
voting network to detect 3D keypoints of objects and then
estimate the 6D pose parameters within a least-squares ﬁt-
ting manner.
Our method is a natural extension of 2D-
keypoint approaches that successfully work on RGB based
6DoF estimation. It allows us to fully utilize the geomet-
ric constraint of rigid objects with the extra depth infor-
mation and is easy for a network to learn and optimize.
Extensive experiments were conducted to demonstrate the
effectiveness of 3D-keypoint detection in the 6D pose esti-
mation task. Experimental results also show our method
outperforms the state-of-the-art methods by large margins
on several benchmarks. Code and video are available at
https://github.com/ethnhe/PVN3D.git.

## introduction
To tackle this task, we develop a novel approach based
on a deep 3D Hough voting network, as shown in Figure 2.
The proposed method is a two-stage pipeline with 3D key-
point detection followed by a pose parameters ﬁtting mod-
ule. More speciﬁcally, taking an RGBD image as input, a
feature extraction module would be used to fuse the appear-
ance feature and geometry information. The learned fea-
ture would be fed into a 3D keypoint detection module MK
which was trained to predict the per-point offsets w.r.t key-
points. Additionally, we include an instance segmentation
module for multiple objects handling where a semantic seg-
mentation module MS predicts the per-point semantic la-
bel, and a center voting module MC predicts the per-point
offsets to object center. With the learned per-point offset,
the clustering algorithm [7] is applied to distinguish differ-
ent instances with the same semantic label and points on
the same instance vote for their target keypoints. Finally,
a least-square ﬁtting algorithm is applied to the predicted
keypoints to estimate 6DoF pose parameters.
3.2. Learning Algorithm
The goal of our learning algorithm is to train a 3D key-
point detection module MK for offset prediction as well
as a semantic segmentation module MS and center voting
module MC for instance-level segmentation. This naturally
makes training our network multi-task learning, which is
achieved by a supervised loss we designed and several train-
ing details we adopt.
3D Keypoints Detection Module. As shown in Figure
2, with the per-point feature extracted by the feature extrac-
tion module, a 3D keypoint detection module MK is used to
detect the 3D keypoints of each object. To be speciﬁc, MK
predicts the per-point Euclidean translation offset from visi-
ble points to target keypoints. These visible points, together
with the predicted offsets then vote for the target keypoints.
The voted points are then gathered by clustering algorithms
and centers of clusters are selected as the voted keypoints.
We give a deeper view of MK as follows. Given a set of
visible seed points {pi}N
i=1 and a set of selected keypoints
{kpj}M
j=1 belonging to the same object instance I, we de-
note pi = [xi; fi] with xi the 3D coordinate and fi the ex-
tracted feature. We denote kpj = [yj] with yj the 3D coor-
dinate of the keypoint. MK absorbs feature fi of each seed
point and generates translation offset {of j
i }M
j=1 for them,
where of j
i denotes the translation offset from the ith seed
point to the jth keypoint. Then the voted keypoint can be
denoted as vkpj
i = xi + of j
i . To supervise the learning of
of j
i , we apply an L1 loss:
Lkeypoints = 1
N
N
X
i=1
M
X
j=1
||of j
i −of j∗
i ||I(pi ∈I)
(1)
where of j∗
i
is the ground truth translation offset; M is the
total number of selected target keypoints; N is the total
number of seeds and I is an indicating function equates to 1
only when point pi belongs to instance I, and 0 otherwise.
Instance Semantic Segmentation Module. To handle
scenes with multi objects, previous methods [50, 53, 39]
utilize existing detection or semantic segmentation archi-
tecture to pre-process the image and obtain RoIs (regions

of interest) containing only single objects. Then build the
pose estimation models with the extracted ROIs as input to
simplify the problem. However, as we have formulated the
pose estimation problem to ﬁrst detect keypoints of objects
with a translation offsets to keypoints learning module, we
believe that the two tasks can enhance the performance of
each other. On the one hand, the semantic segmentation
module forces the model to extract global and local features
on instance to distinguish different objects, which helps to
locate a point on the object and does good to the keypoint
offset reasoning procedure. On the other hand, size infor-
mation learned for the prediction of offsets to the keypoints
helps distinguish objects with similar appearance but differ-
ent in size. Under such observation, we introduce a point-
wise instance semantic segmentation module MS into the
network and jointly optimized it with module MK.
To be speciﬁc, given the per-point extracted feature, the
semantic segmentation module MS predicts the per-point
semantic labels. We supervise this module with Focal Loss
[29]:
Lsemantic = −α(1 −qi)γlog(qi)
where
qi = ci · li
(2)
with α the α-balance parameter, γ the focusing parameter,
ci the predicted conﬁdence for the ith point belongs to each
class and li the one-hot representation of ground true class
label.
Meanwhile, the center voting module MC is applied to
vote for centers of different object so as to distinguish differ-
ent instance. We propose such module under the inspiration
of CenterNet [10] but further extend the 2D center point to
3D. Compared to 2D center points, different center points
in 3D won’t suffer from occlusion due to camera projection
in some viewpoints. Since we can regard the center point as
a special keypoint of an object, module MC is similar to the
3D keypoint detection module MK. It takes in the per-point
feature but predicts the Euclidean translation offset ∆xi to
the center of objects it belongs to. The learning of ∆xi is
also supervised by an L1 loss:
Lcenter = 1
N
N
X
i=1
||∆xi −∆x∗
i ||I(pi ∈I)
(3)
where N denotes the total number of seed points on the ob-
ject surface and ∆x∗
i is the ground truth translation offset
from seed pi to the instance center. I is an indication func-
tion indicating whether point pi belongs to that instance.
Multi-task loss. We supervise the learning of module
MK, MS and MC jointly with a multi-tasks loss:
Lmulti-task =λ1Lkeypoints + λ2Lsemantic + λ3Lcenter
(4)
where λ1, λ2 and λ3 are the weights for each task. Experi-
mental results shows that jointly training these tasks boosts
the performance of each other.
3.3. Training and Implementation
Network Architecture. The ﬁrst part in Figure 2 is a
feature extraction module. In this module, a PSPNet [55]
with an ImageNet [8] pretrained ResNet34 [16] is applied
to extract the appearance information in RGB images. A
PointNet++ [40] extracts the geometry information in point
clouds and their normal maps. They are further fused by
a DenseFusion block [50] to obtain the combined feature
for each point. After the process of this module, each point
pi has a feature fi ∈RC of C dimension. The following
module MK, MS and MC are composed of shared Multi-
Layer Perceptrons (MLPs) shown in Figure 2. We sample
N = 12288 points (pixels) for each frame of RGBD image
and set λ1 = λ2 = λ3 = 1.0 in Formula 4.
Keypoint Selection.
The 3D keypoints are selected
from 3D object models. In 3D object detection algorithms
[39, 53, 38], eight corners of the 3D bounding box are se-
lected. However, These bounding box corners are virtual
points that are far away from points on the object, making
point-based networks difﬁcult to aggregate scene context in
the vicinity of them. The longer distance to the object points
results in larger localization errors, which may do harm to
the compute of 6D pose parameters. Instead, points selected
from the object surface will be quite better. Therefore, we
follow [37] and use the farthest point sampling (FPS) al-
gorithm to select keypoints on the mesh. Speciﬁcally, we
initial the selection procedure by adding the center point of
the object model in an empty keypoint set. Then update
it by adding a new point on the mesh that is farthest to all
the selected keypoints repeatedly, until M keypoints are ob-
tained.
Least-Squares Fitting. Given two point sets of an ob-
ject, one from the M detected keypoints {kpj}M
j=1 in the
camera coordinate system, and another from their corre-
sponding points {kp
′
j}M
j=1 in the object coordinate system,
the 6D pose estimation module computes the pose parame-
ters (R, t) with a least-squares ﬁtting algorithm [1], which
ﬁnds R and t by minimizing the following square loss:
Lleast-squares =
M
X
j=1
||kpj −(R · kp
′
j + t)||2
(5)
where M is the number of selected keypoints of a object.

## experiments
4.1. Datasets
We evaluate our method on two benchmark datasets.
YCB-Video Dataset contains 21 YCB [4] objects of
varying shape and texture. 92 RGBD videos of the subset of
objects were captured and annotated with 6D pose and in-
stance semantic mask. The varying lighting conditions, sig-
niﬁcant image noise, and occlusions make this dataset chal-

Without Iterative Reﬁnement
With Iterative Reﬁnement
PoseCNN[52]
DF(per-pixel)[50]
PVN3D
PoseCNN+ICP[52]
DF(iterative)[50]
PVN3D+ICP
ADDS
ADD(S)
ADDS
ADD(S)
ADDS
ADD(S)
ADDS
ADD(S)
ADDS
ADD(S)
ADDS
ADD(S)
002 master chef can
83.9
50.2
95.3
70.7
96.0
80.5
95.8
68.1
96.4
73.2
95.2
79.3
003 cracker box
76.9
53.1
92.5
86.9
96.1
94.8
92.7
83.4
95.8
94.1
94.4
91.5
004 sugar box
84.2
68.4
95.1
90.8
97.4
96.3
98.2
97.1
97.6
96.5
97.9
96.9
005 tomato soup can
81.0
66.2
93.8
84.7
96.2
88.5
94.5
81.8
94.5
85.5
95.9
89.0
006 mustard bottle
90.4
81.0
95.8
90.9
97.5
96.2
98.6
98.0
97.3
94.7
98.3
97.9
007 tuna ﬁsh can
88.0
70.7
95.7
79.6
96.0
89.3
97.1
83.9
97.1
81.9
96.7
90.7
008 pudding box
79.1
62.7
94.3
89.3
97.1
95.7
97.9
96.6
96.0
93.3
98.2
97.1
009 gelatin box
87.2
75.2
97.2
95.8
97.7
96.1
98.8
98.1
98.0
96.7
98.8
98.3
010 potted meat can
78.5
59.5
89.3
79.6
93.3
88.6
92.7
83.5
90.7
83.6
93.8
87.9
011 banana
86.0
72.3
90.0
76.7
96.6
93.7
97.1
91.9
96.2
83.3
98.2
96.0
019 pitcher base
77.0
53.3
93.6
87.1
97.4
96.5
97.8
96.9
97.5
96.9
97.6
96.9
021 bleach cleanser
71.6
50.3
94.4
87.5
96.0
93.2
96.9
92.5
95.9
89.9
97.2
95.9
024 bowl
69.6
69.6
86.0
86.0
90.2
90.2
81.0
81.0
89.5
89.5
92.8
92.8
025 mug
78.2
58.5
95.3
83.8
97.6
95.4
94.9
81.1
96.7
88.9
97.7
96.0
035 power drill
72.7
55.3
92.1
83.7
96.7
95.1
98.2
97.7
96.0
92.7
97.1
95.7
036 wood block
64.3
64.3
89.5
89.5
90.4
90.4
87.6
87.6
92.8
92.8
91.1
91.1
037 scissors
56.9
35.8
90.1
77.4
96.7
92.7
91.7
78.4
92.0
77.9
95.0
87.2
040 large marker
71.7
58.3
95.1
89.1
96.7
91.8
97.2
85.3
97.6
93.0
98.1
91.6
051 large clamp
50.2
50.2
71.5
71.5
93.6
93.6
75.2
75.2
72.5
72.5
95.6
95.6
052 extra large clamp
44.1
44.1
70.2
70.2
88.4
88.4
64.4
64.4
69.9
69.9
90.5
90.5
061 foam brick
88.0
88.0
92.2
92.2
96.8
96.8
97.2
97.2
92.0
92.0
98.2
98.2
ALL
75.8
59.9
91.2
82.9
95.5
91.8
93.0
85.4
93.2
86.1
96.1
92.3
Table 1. Quantitative evaluation of 6D Pose (ADD-S AUC [52], ADD(S) AUC [19]) on the YCB-Video Dataset. Symmetric objects’ names
are in bold.
w/o iter. ref.
w/ iter. ref.
DF(p.p.)
PVN3D
DF(iter.)
PVN3D+ICP
large clamp
ADD-S
87.7
93.9
90.3
96.2
extra large clamp
ADD-S
75.0
90.1
74.9
93.6
ALL
ADD-S
93.3
95.7
94.8
96.4
ADD(S)
84.9
91.9
89.4
92.7
Table 2. Quantitative evaluation results on the YCB-Video dataset
with ground truth instance semantic segmentation result.
lenging. We follow [52] and split the dataset into 80 videos
for training and another 2,949 keyframes chosen from the
rest 12 videos for testing. Following [52], we add the syn-
thetic images into our training set. A hole completion al-
gorithm [25] is also applied to improve the quality of depth
images.
LineMOD Dataset [18] consists of 13 low-textured ob-
jects in 13 videos, annotated 6D pose and instance mask.
The main challenge of this dataset is the cluttered scenes,
texture-less objects, and lighting variations. We follow prior
works [52] to split the training and testing set. Also, we fol-
low [37] and add synthesis images into our training set.
4.2. Evaluation Metrics
We follow [52] and evaluate our method with the aver-
age distance ADD and ADD-S metric [52]. The average
distance ADD metric [19] evaluates the mean pair-wise dis-
tance between object vertexes transformed by the predicted
6D pose [R, t] and the ground true pose [R∗, t∗]:
ADD = 1
m
X
x∈O
||(Rx + t) −(R∗x + t∗)||
(6)
where x is a vertex of totally m vertexes on the object mesh
O. The ADD-S metric is designed for symmetric objects
and the mean distance is computed based on the closest
point distance:
ADD-S = 1
m
X
x1∈O
min
x2∈O ||(Rx1 + t) −(R∗x2 + t∗)||
(7)
For evaluation, we follow [52, 50] and compute the ADD-S
AUC, the area under the accuracy-threshold curve, which
is obtained by varying the distance threshold in evaluation.
The ADD(S)[19] AUC is computed in a similar way but cal-
culate ADD distance for non-symmetric objects and ADD-
S distance for symmetric objects.
4.3. Evaluation on YCB-Video & LineMOD Dataset
Table 1 shows the evaluation results for all the 21 ob-
jects in the YCB-Video dataset. We compare our model
with other single view methods.
As shown in the Ta-
ble, our model without any iterative reﬁnement procedure
(PVN3D) surpasses all other approaches by a large margin,
even when they are iterative reﬁned. On the ADD(S) met-
ric, our model outperforms PoseCNN+ICP [52] by 6.4%

RGB
RGBD
PoseCNN
DeepIM
[26, 52]
PVNet
[37]
CDPN
[27]
Implicit
ICP[45]
SSD-6D
ICP[22]
Point-
Fusion[50]
DF(per-
pixel)[50]
DF(ite-
rative)[50]
PVN3D
ape
77.0
43.6
64.4
20.6
65.0
70.4
79.5
92.3
97.3
benchvise
97.5
99.9
97.8
64.3
80.0
80.7
84.2
93.2
99.7
camera
93.5
86.9
91.7
63.2
78.0
60.8
76.5
94.4
99.6
can
96.5
95.5
95.9
76.1
86.0
61.1
86.6
93.1
99.5
cat
82.1
79.3
83.8
72.0
70.0
79.1
88.8
96.5
99.8
driller
95.0
96.4
96.2
41.6
73.0
47.3
77.7
87.0
99.3
duck
77.7
52.6
66.8
32.4
66.0
63.0
76.3
92.3
98.2
eggbox
97.1
99.2
99.7
98.6
100.0
99.9
99.9
99.8
99.8
glue
99.4
95.7
99.6
96.4
100.0
99.3
99.4
100.0
100.0
holepuncher
52.8
82.0
85.8
49.9
49.0
71.8
79.0
92.1
99.9
iron
98.3
98.9
97.9
63.1
78.0
83.2
92.1
97.0
99.7
lamp
97.5
99.3
97.9
91.7
73.0
62.3
92.3
95.3
99.8
phone
87.7
92.4
90.8
71.0
79.0
78.8
88.0
92.8
99.5
ALL
88.6
86.3
89.9
64.7
79.0
73.7
86.2
94.3
99.4
Table 3. Quantitative evaluation of 6D Pose on ADD(S) [19] metric on the LineMOD dataset. Objects with bold name are symmetric.
DF(RT)[50]
DF(3D KP)[50]
Ours(RT)
Ours(2D KPC)
Ours(2D KP)
PVNet[37]
Ours(Corr)
Ours(3D KP)
ADD-S
92.2
93.1
92.8
78.2
81.8
-
92.8
95.5
ADD(S)
86.9
87.9
87.3
73.8
77.2
73.4
88.1
91.8
Table 4. Quantitative evaluation of 6D Poses on the YCB-Video dataset with different formulations. All with our predicted segmentation.
VoteNet[38]
BBox 8
FPS 4
FPS 8
FPS 12
ADD-S
89.9
94.0
94.3
95.5
94.5
ADD(S)
85.1
90.2
90.5
91.8
90.7
Table 5. Effect of different keypoint selection methods of PVN3D.
Results of VoteNet[38], another 3D bounding box detection ap-
proach are added as a simple baseline to compare with our BBox8.
and exceeds DF(iterative) [50] by 5.7%. With iterative re-
ﬁnement, our model (PVN3D+ICP) achieves even better
performance. Note that one challenge of this dataset is to
distinguish the large clamp and extra-large clamp, on which
previous methods [50, 52] suffer from poor detection re-
sults. We also report evaluation results with ground truth
segmentation in Table 2, which shows that our PVN3D still
achieves the best performance. Some Qualitative results are
shown in Figure 3. Table 3 demonstrates the evaluation re-
sults on LineMOD dataset. Our model also achieves the
best performance.
Robust to Occlusion Scenes. One of the biggest advan-
tages of our 3D-keypoint-based method is that it’s robust
to occlusion naturally. To explored how different methods
are inﬂuenced by different degrees of occlusion, we follow
[50] and calculate the percentage of invisible points on the
object surface. Accuracy of ADD-S < 2cm under different
invisible surface percentage is shown in Figure 4. The per-
formance of different approaches is very close when 50%
of points are invisible. However, with the percentage of in-
visible part increase, DenseFusion and PoseCNN+ICP fall
faster comparing with ours. Figure 3 shows that our model
performs well even when objects are heavily occluded.
4.4. Ablation Study
In this part, we explore the inﬂuence of different formu-
lation for 6DoF pose estimation and the effect of keypoint
selection methods. We also probe the effect of multi-task
learning.
Comparisons to Directly Regressing Pose. To compare
our 3D keypoint based formulation with formulations that
directly regressing the 6D pose parameters [R, t] of an ob-
ject, we simply modify our 3D keypoint voting module MK
to directly regress the quaternion rotation R and the transla-
tion parameters t for each point. We also add a conﬁdence
header following DenseFusion [50] and select the pose with
the highest conﬁdence as the ﬁnal proposed pose. We super-
vise the training process using ShapeMatch-Loss [52] with
conﬁdence regularization term [50] following DenseFusion.
Experimental results in Table 4 shows that our 3D keypoint
formulation performs quite better.
To eliminate the inﬂuence of different network architec-
ture, we also modify the header of DenseFusion(per-pixel)
to predict the per-point translation offset and compute the
6D pose following our keypoint voting and least-squares ﬁt-
ting procedure. Table 4 reveals that the 3D keypoint formu-
lation, DF(3D KP) in the Table, performs better than the RT
regression formulation, DF(RT). That’s because the 3D key-
point offset search space is smaller than the non-linearity of
rotation space, which is easier for neural networks to learn,
ena

## related_work
2.1. Holistic Methods
Holistic methods directly estimate the 3D position and
orientation of objects in a given image. Classical template-
based methods construct rigid templates and scan through
the image to compute the best matched pose [21, 13, 17].
Such templates are not robust to clustered scenes. Recently,
some Deep Neural Network (DNN) based methods are pro-
posed to directly regress the 6D pose of cameras or ob-
jects [52, 50, 14]. However, non-linearity of the rotation
space making the data-driven DNN hard to learn and gener-
alize. To address this problem, some approaches use post-
reﬁnement procedure [26, 50] to reﬁne the pose iteratively,
others discrete the rotation space and simplify it to be a
classiﬁcation problem [49, 43, 45]. For the latter approach,
post-reﬁnement processes are still required to compensate
for the accuracy sacriﬁced by the discretization.
2.2. Keypoint-based Methods
Current keypoint-based methods ﬁrst detect the 2D key-
points of an object in the images, then utilize a PnP algo-
rithm to estimate the 6D pose. Classical methods [30, 42, 2]
are able to detect 2D keypoint of objects with rich texture
efﬁciently. However, they can not handle texture-less ob-
jects. With the development of deep learning techniques,
some neural-network-based 2D keypoints detection meth-
ods are proposed. [41, 47, 20] directly regress the 2D coor-
dinate of the keypoints, while [33, 24, 34] use heatmaps to
locate the 2D keypoints. To better deal with truncated and
occluded scenes, [37] proposes a pixel-wise voting network
to vote for the 2D keypoints location. These 2D keypoint
based methods aim to minimize the 2D projection errors of
objects. However, errors that are small in projection may be
large in the real 3D world. [46] extracts 3D keypoints from
two views of synthetic RGB images to recover 3D poses.
Nevertheless, they only utilize the RGB images, on which
geometric constraint information of rigid objects partly lost
due to projection, and different keypoints in 3D space may
be overlapped and hard to be distinguished after projected
to 2D. The advent of cheap RGBD sensors enables us to do
everything in 3D with captured depth images.
2.3. Dense Correspondence Methods
These approach utilize Hough voting scheme [28, 44, 12]
to vote for ﬁnal results with per-pixel prediction. They ei-
ther use random forest [3, 32] or CNNs [23, 9, 27, 35, 51]
to extract feature and predict the corresponding 3D object
coordinates for each pixel and then vote for the ﬁnal pose
results. Such dense 2D-3D correspondence making these
methods robust to occluded scenes, while the output space
is quite large. PVNet [37] uses per-pixel voting for 2D Key-
points to combine the advantages of Dense methods and
keypoint-based methods. We further extend this method to
3D keypoints with extra depth information and fully utilize
geometric constraints of rigid objects.

Figure 2. Overview of PVN3D. The Feature Extraction module extracts the per-point feature from an RGBD image. They are fed into
module MK, MC and MS to predict the translation offsets to keypoints, center point and semantic labels of each point respectively. A
clustering algorithm is then applied to distinguish different instances with the same semantic label and points on the same instance vote for
their target keypoints. Finally, a least-square ﬁtting algorithm is applied to the predicted keypoints to estimate 6DoF pose parameters.
3. Proposed Method
Given an RGBD image, the task of 6DoF pose estimation
is to estimate the rigid transformation that transforms an ob-
ject from its object world coordinate system to the camera
world coordinate system. Such transformation consists of a
3D rotation R ∈SO(3) and a translation t ∈R3.

## conclusion
We propose a novel deep 3D keypoints voting network
with instance semantic segmentation for 6DoF pose estima-
tion, which outperforms all previous approaches in several
datasets by large margins. We also show that jointly training
3D keypoint with semantic segmentation can boost the per-
formance of each other. We believe the 3D keypoint based
approach is a promising direction to explore for the 6DoF
pose estimation problem.

A. Appendix
A.1. Architecture Details.
The image embedding network in the Feature Extrac-
tion module consists of a standard ResNet-34 encoder pre-
trained with ImageNet and followed by 4 up-sampling lay-
ers as a decoder. The output RGB feature is of 128 chan-
nels. The point cloud embedding network is a PointNet++
[40] with Multi-scale Grouping (MSG), the output of which
is also of 128 channels. In the DenseFuion block, for each
sampled point, the 128 channels RGB feature is concate-
nated with the corresponding 128 channels point cloud fea-
ture and is fed into shared MLPs to raise the feature chan-
nels to 1024. The 128 channels RGB and point cloud fea-
ture are also fed into shared MLPs and raised to 256 chan-
nels respectively. They are then all concatenated together to
form the 1792 (128*2+256*2+1024) channels RGBD fea-
tures. The 3D keypoint offset voting module MK consists
of shared MLPs (1792-1024-512-128-Nkp*3) reducing the
1792 channels RGBD features to Nkp 3D keypoints offset.
The semantic segmentation module MS consists of shared
MLPs (1792-1024-512-128-Ncls) with Ncls the number of
object classes. The center voting module MC also consists
of shared MLPs (1792-1024-512-128-3). During training,
Adam optimization algorithm is employed, with a mini-
batch size of 24. We applied cyclical learning rates during
training, the range of which is from 1e-5 to 1e-3.
A.2.
Implementation:
Models
for
LineMOD
dataset.
In the LineMOD dataset, though there are multi objects
in one scene, only the label of one target object in each
scene is provided, making it hard to train a single model
for all objects. Therefore, we follow PVNet[37] and trained
models separately for each object. It means our semantic
segmentation module MS only predicts two semantic la-
bels for each object, label of the target object and the back-
ground.
A.3. Implementation: PVN3D+ICP
We introduce our implementation of PVN3D+ICP as fol-
lows. During training and inference of PVN3D, we ran-
domly sample 12288 points (pixels) from the whole scene
as input. Only these points are labeled with the semantic
label by our instance semantic segmentation module, which
is not enough for a good performance of the ICP algorithm.
Therefore, for each unlabeled point in the whole point cloud
scene, we ﬁnd its closest labeled point and assigned that
label to it. Points with the same instance semantic labels
are then selected. To eliminate the effect of noise points,
a MeanShift [7] clustering algorithm is also applied. The
biggest cluster of points is then selected as the input of the
ICP algorithm.
A.4. More Results
A.4.1
Instance semantic segmentation
We provide more results of instance semantic segmentation
in Table 8.
A.4.2
Visualization of detected keypoints
The visualization of some detected keypoints is shown in
Figure 6.
A.4.3
Evaluation on the Occlusion LineMOD Dataset
The Occlusion LineMOD dataset [3] is additionally anno-
tated from the LineMOD datasets, objects of which are
heavily occluded, making it harder for pose estimation.
Evaluation results in Table 9 show that our PVN3D outper-
forms previous methods by a large margin.
A.4.4
Inference time
It takes 0.17 seconds for network forward propagation and
0.02 seconds for pose estimation of each object during in-
ference. The overall runtime is 5 FPS on the LineMOD
dataset.