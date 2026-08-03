# Unseen Object 6D Pose Estimation: A Benchmark and Baselines

> 2022 · id: W4283453676 · arXiv: 2206.11808 · pdf: https://arxiv.org/pdf/2206.11808 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Estimating the 6D pose for unseen objects is in great
demand for many real-world applications. However, cur-
rent state-of-the-art pose estimation methods can only han-
dle objects that are previously trained. In this paper, we
propose a new task that enables and facilitates algorithms
to estimate the 6D pose estimation of novel objects during
testing. We collect a dataset with both real and synthetic
images and up to 48 unseen objects in the test set. In the
mean while, we propose a new metric named Inﬁmum ADD
(IADD) which is an invariant measurement for objects with
different types of pose ambiguity. A two-stage baseline so-
lution for this task is also provided. By training an end-
to-end 3D correspondences network, our method ﬁnds cor-
responding points between an unseen object and a partial
view RGBD image accurately and efﬁciently. It then cal-
culates the 6D pose from the correspondences using an al-
gorithm robust to object symmetry. Extensive experiments
show that our method outperforms several intuitive base-
lines and thus verify its effectiveness. All the data, code
and models will be made publicly available. Project page:
www.graspnet.net/unseen6d

## introduction
Object 6D pose estimation is an important task in com-
puter vision and robotics.
Many real-world applications
[1,35] such as grasping and VR/AR heavily rely on accurate
object 6D pose estimation result.
Researches on object pose estimation have been explored
for a long time.
Template-based methods such as point
cloud registration [46, 58] and template matching [11, 29]
mainly adopt handcrafted rules to encode geometry fea-
tures. Since the geometry encoding schema is model ag-
nostic, these methods are applicable to any objects in princi-
ple. However, due to the inferior expressive power of hand-
*Equal contribution
†Work done as an intern in Alibaba XR Lab
‡Cewu Lu is the corresponding author
crafted features, they cannot achieve satisfactory results in
cluttered scenes with noise and need many manual tuning
efforts. Recently, deep learning methods based on 2D im-
age [31, 41, 44, 56] or 3D point cloud [21, 22, 54] are pro-
posed to tackle this problem and yield better performances,
beneﬁting from the powerful feature extraction ability of
neural network.
However, in the current task setting of 6D pose esti-
mation [24, 26, 56], the same object set is shared in both
training and testing phase.
Taking such assumption that
the testing object is always available during the training
period, current state-of-the-art 6D pose estimation algo-
rithms [21, 22] follow the schema that directly models the
object’s texture and geometry features within the neural net-
works. Prior knowledge of object models such as keypoint
location [22,41] or voting offsets [22,41] is also encoded by
the networks. It turns out that these methods can only esti-
mate the 6D pose of known objects during training. In real-
world applications such as the ﬂexible robotic assembly,
novel objects appear frequently. To detect their 6D poses,
new data collection process including keypoints allocation
and synthetic image generation [12] needs to be repeated,
and the network needs to be retrained. This is labor inten-
sive and prevents the 6D pose estimation algorithms from
rapid deployment.
In this paper, we reconsider this problem and propose to
explore a new direction. In practice, the mesh model of an
object is easy to obtain. With a commercial 3D scanner, the
mesh model of an object can be retrieved within minutes.
The major bottlenecks for fast deployment of the aforemen-
tioned methods are the synthetic data generation and net-
work retraining processes. Thus, as shown in Fig. 1, we
propose a new task named unseen object 6D pose estima-
tion. After training on a ﬁnite set of objects, the algorithm
is required to estimate the 6D pose of any novel object in a
scene given their mesh models but without re-training. This
task is similar to the original 6D pose estimation problem
except that the mesh models of objects in the test set will
not be available during training.
To fulﬁll the task, we propose a new benchmark that con-
arXiv:2206.11808v1  [cs.CV]  23 Jun 2022

Ground Truth Poses 
...
Training 
Infinite Unseen Objects 
...
RGBD images 
...
Estimated Poses 
...
RGBD Images 
...
Finite Object Models 
...
3D Scanner

## method
In this section, we provide a baseline solution for the
unseen object 6D pose estimation task based on the archi-
tecture illustrated in Fig. 3. Given the point cloud of the
object and scene, we start by extracting high dimensional
features for the two inputs using a backbone network. Then
Object-Level Segmentation Proposal Network (SPN) pro-
poses candidates of the target object in the scene using these
features and we further obtains object ROIs(region of in-
terest) with a manually selected threshold. After that, an
Object-Scene Correspondence Network (OSCN) learns the
dense 3D correspondences between the object points and
the scene points in each selected ROI region. Finally, we
follow EPOS [25] to estimate the object 6D pose from the
3D correspondences.
4.1. Backbone Network
Our framework starts with point cloud feature extraction.
In our setting, As shown in Fig. 3a, given the object point
cloud Pobj of size N × 6 and scene point cloud Pscene of
size M × 6, the backbone network extracts point-wise high
dimensional feature vectors Fobj and Fscene of shape N ×
C and M × C for each input respectively. These features
are shared in the latter SPN and OSCN modules to segment
the point cloud and ﬁnd 3D correspondences.
4.2. Object-Level Segmentation Proposal Net-
work(SPN)
Finding correspondences in the whole scene is a difﬁcult
task. Inspired by [53], we introduce the attention mech-
anism by adding a point-wise segmentation network SPN.
Given the features of the target object Fobj and that of the
scene Fscene, SPN is designed to achieve a point-wise seg-
mentation of this object in the scene. This helps the OSCN
ﬁnd correspondences as it can focus on a small area, which
also saves the computational resources.
For the network structure, as shown in Fig. 3b, we ﬁrst
apply a mean pooling on the object features Fobj to obtain
the object’s global feature vector of shape 1 × C, which
serves as a descriptor for the object. Then, we concatenate
such feature vector with each point in the scene features
Fscene, resulting in a shape of M × 2C. These concate-
nated features are fed into a multi-layer perceptron (MLP).
The ﬁnal output is a segmentation heatmap of shape M ×1,
denoting whether each point in the scene belongs to the ob-
ject or not.
To cover as many points on the object as possible and
eliminate noise, we conduct a post-processing to reﬁne the
segmentation heatmap. As shown in Fig. 4, we ﬁrst project
the 3D heatmap to the 2D scene image and apply a Gaussian
smoothing to the 2D heatmap. This makes the discretized
heatmap more continuous. Then, to remove outliers, we
binarize the heatmap by Otsu’s method [3] and select the
connected components larger than a threshold as the ﬁnal
segmentation results which is shown in Fig. 4d.
4.3.
Object-Scene
Correspondence
Network
(OSCN)
After obtaining the target object segmentation candi-
dates, the OSCN module then ﬁnds 3D correspondences
between each segmented scene ROI and the object. This
module follows different strategy during training and test-
ing, and we ﬁrst introduce the testing stage.
During testing, the object level segmentation results are
used to segment the target object candidates in the scene.
For simplicity, we consider the case of only one target ob-
ject candidate, while the case of multiple targets can be pro-
cessed batch-wise similarly. Given the target segmentation,
we crop both the input scene point cloud and its features and
obtain Pseg and Fseg, which have a shape of M ′ × 6 and
M ′ ×C. Then, for each point on the object and the segment
scene, we concatenate their features and construct dense
pair-wise feature vectors with a shape of (M ′ × N) × 2C,
where (M ′ × N) denotes the amount of object-scene point
pairs. To save computation resources, we randomly sample
L pairs and feed them into an MLP, which estimates L × 1
scores ranging from 0 to 1 to denote the conﬁdence of input
pairs’ correspondences.
Among the L point pairs, we select those with conﬁ-
dence score larger than 0.8, resulting in K corresponding
point pairs. The point cloud of these corresponding points
as well as the correspondence’s scores are used for the ﬁnal
6D pose computation, which is detailed in Sec. 4.4.
During training, the segmentation results from SPN is
not used. Instead, we uniformly sample k1 pairs of match-

Heatmap
Points   Sampling
Fragmentation + PROSAC
MLP
Positive
Filtering 
Correspondences
Object Point 
Cloud
Scene Point 
Cloud
Point 
Decoder
Point 
Encoder
Point 
Decoder
Point 
Encoder
Object Level 
Segmentation
Repeat
Cat
MLP
M× 1
Post   Processing
Mean Pooling
M× 1
Scene 
Object 
6D Pose
Backbone
SPN
OSCN
N×C
M×C
N×6
M×6
1×C
M×C
M×C
M×C
M×C
L×12
K×13
L×1
L×2C
(a)
(b)
(c)
(d)
M`×C
M`×6
N×6
N×C
Pairing
Figure 3. Baseline Architecture: The method could be divided into two stages. The ﬁrst stage is an end-to-end neural network which
detects 3D correspondences between the object and the scene. It is composed of three parts, i.e., backbone, Object-Level Segmentation
Proposal Network (SPN) and Object-Scene Correspondence Network (OSCN). The second stage calculates the 6D pose from the 3D
correspondences with PROSAC algorithm.
(a) SPN’s result
(b) Heatmap
(c) Otsu threshold
(d) Segmentation
Figure 4. (a) is the output of SPN. (b) is the heatmap after Gaussian
smoothing. (c) is the result of Otsu method [3]. (d) is the ﬁnal
segmentation result.
ing points between the object and the scene using the
ground truth 6D pose and randomly sample k2 pairs of non-
matching points as negative samples. The ratio rk = k1 : k2
is a ﬁxed hyper parameter to ensure a balanced training set.
4.4. 6D Pose Computation
As discussed in Section 2, the traditional way to ob-
tain 6D pose from 3D correspondences is least square ﬁt-
ting with RANSAC [19]. However, such method would
fail when objects have keypoint ambiguity which is usually
caused by symmetry. In this paper, we adopt the 6D pose
ﬁtting module proposed in EPOS [25] for the ﬁnal 6D pose
computation. It adopts the PROSAC algorithm [38] instead
of RANSAC [19] to calculate the ﬁnal 6D pose. This algo-
rithm is a locally optimized RANSAC that ﬁrstly focuses on
correspondences with higher conﬁdence and progressively
turns to uniform sampling. For more details, we refer read-
ers to the original paper of EPOS [25].
4.5. Loss
The backbone, OSCN and SPN modules are trained si-
multaneously with multi-task loss:
L = (1 −λ)Lseg + λ Lcor, (0 < λ < 1),
(8)
where Lseg and Lcor are both binary cross entropy loss for
target classiﬁcation in SPN and correspondence classiﬁca-
tion in OSCN respectively.
5. Implementation Details
Dataset.
To construct a meaningful benchmark, it re-
quires a variational training set so that the networks can
learn representations general enough and a representative
test set that is close to the real-world setting. GraspNet-
1Billion [18], originally proposed for the problem of robotic
grasping, satisﬁes most of our requirements.
It contains
40 objects and 100 scenes for training, 76 objects and 90
scenes for testing. Its test set is further divided into 3 sub-
set, namely seen object set, similar object set and novel
object set, where each set contains 30 scenes consisted of

28 seen objects, 22 unseen but similar objects and 26 to-
tally novel objects respectively. Thus, we build our bench-
mark upon [18]. The only problem is that its training set
only contains 40 objects, which may be too few for the
network to learn model-agnostic geometry correspondence
features. Thus, we generate extra synthetic training data
with BlenderProc [9] simulator. The object mesh models
come from the Google Scanned Object dataset [20], which
consists of over 1000 real-world objects. In total, there are
1070 objects and 1500 scenes (1400 synthetic scenes and
100 real scenes from Graspnet-1Billion ) in our training set
and 76 objects and 90 scenes in the real data test set, in
which 48 objects are unseen during training.
To verify the effectiveness of our method, we also con-
duct pose estimation experiments on the YCB-Video [56]
without any retraining or ﬁne-tuning and compare the re-
sults of different algorithms.
Neural Network and Training.
For the backbone net-
work, we select ResUNet14 built on MinkowskiEngine [6]
which has great performance in processing the point cloud.
It can also be replaced by other point cloud networks such
as PointNet [42] and PointNet++ [43]. M and N are the
points number of the scene and object’s point cloud. C is
set to 512. L is set to 102400 during inference. k1 and
k2 are set to 100 and 600 during training. All the MLPs
are implemented using full connected layers with residual
blocks. The structure is illustrated in the supplementary
materials. The λ value in the loss layer is set to 0.6. To
reduce the size of the neural network, the backbone for ob-
ject branch and scene branch share the same structure and
weights. Our model is implemented with PyTorch and is
trained and tested on a server with 8 NVIDIA RTX 3090
GPUs. T

## experiments
We conduct extensive experiments to verify the effec-
tiveness and efﬁciency of our proposed method.
No  Axis
Finite Axes
Finite Poses
Infinite Axes
Infinite poses
Finite Axes
Infinite Poses
  
Sampling
Figure 5. The illustration of IADD. In the ﬁrst two cases, IADD
equals to ACPD and can be calculated by traversing all the poses
and ﬁnd the minimum ADD. In the third case where the object
has at least one rotational axis that has inﬁnite pose ambiguities,
we estimates the inﬁmum of ADD by uniform sampling. In the
last case where the object has inﬁnite rotational axes, IADD is the
distance between the ground truth center and estimated center.
6.1. Metric
The biggest challenge of 6D pose evaluation metric is
pose ambiguity [39]. As discussed in Section 2.3, the most
commonly used metrics are ADD [23] for objects with no
pose ambiguity and ADD-S [56] for object with pose am-
biguity. But the two metrics cannot be compared because
ADD-S is always numerically smaller than ADD [23]. In
other words, the pose estimation result for symmetric ob-
jects cannot be compared with asymmetric objects. ACPD
and MCPD [28] are proposed to solve this problem which
comprehensively evaluate all reasonable ground truth poses.
But neither the deﬁnition itself nor the implementation [27]
is able to handle objects with inﬁnite pose ambiguities.
We propose a new metric named Inﬁmum of ADD(IADD).
IADD extends ACPD when there are inﬁnite pose ambigu-
ities.
The previous metrics are given in Equation 9.
ADD = 1
m
X
v∈V
∥(Rv + T) −(R∗v + T ∗)∥,
ADD-S = 1
m
X
v1∈V
min
v2∈V ∥(Rv1 + T) −(R∗v2 + T ∗)∥,
ACPD = 1
m
X
v∈V
min
R∗∈R∗,T ∗∈T ∗∥(Rv + T) −(R∗v + T ∗)∥,
(9)
where V, v, R, and T are the vertex points set of the ob-
ject, vertex point, rotational matrix, translation respectively.
R∗, T ∗, R∗and T ∗denote the ground truth rotational ma-
trix, translation and the set of ground truth rotational ma-
trices and translations. The deﬁnition of IADD is given in
Equation 10.
IADD = 1
m
X
v∈V
inf
R∗∈R∗,T ∗∈T ∗∥(Rv + T) −(R∗v + T ∗)∥
(10)
Using this metric, both the pose for symmetric and asym-
metric objects can be evaluated in the same way even if

(a) Object
(b) Scene
Figure 6. t-SNE [52] visualization result of encoded features pre-
sented in RGB.
there are inﬁnite pose ambiguities. To further discuss the
implementation of this metric, we ﬁrstly discuss where the
pose ambiguity comes from.
Pose ambiguity occurs only when the object has a rota-
tional symmetry axis. Mirror symmetry brings problems of
keypoint ambiguity for 6D pose estimation algorithms. But
it leads to no pose ambiguity in evaluation. As shown in
Fig. 5, there are totally four cases.
1. Object has no rotational axis.
2. Object has ﬁnite rotational axes and each rotational
axis has ﬁnite equivalent poses.
3. Object has ﬁnite rotational axes and at least one rota-
tional axis has inﬁnite equivalent poses.
4. Object has inﬁnite rotational axes.
For the ﬁrst case, IADD equals to ADD and ACPD. For
the second case, IADD equals to ACPD. For the third case,
it is hard to ﬁnd an analytical solution. We sample n an-
gles around the axis with inﬁnite pose ambiguities in our
implementation. The number of n is a trade-off between
precision and efﬁciency. Although this is a numerical solu-
tion, it doesn’t destroy the overall science as ADD itself is a
numerical solution that samples points from a mesh model.
For the last case, although we can still take the numerical
solution, the sampling on two dimensions, i.e. the axis sam-
pling and the rotation angle sampling, results in a huge cost
for computation. Fortunately, the only object that has inﬁ-
nite rotational axes is a texture-less sphere. For this kind
of objects, IADD equals to the center distance between the
target pose and the estimated pose.
6.2. Experiment Results
6.2.1
Visualization of Extracted Features
We reduce the dimensions of extracted features to 3 and
colorize the point cloud by encoding the RGB channel with
these 3D features. As shown in the visualization result in
Fig. 6, both the features among different objects and those
among different parts within an object are clearly distin-
guishable.
GraspNet Similar
GraspNet Novel
YCB Video
FCGF+SPN
SuperGlue 
+ RANSAC
Our Method
Figure 7. The qualitative result of pose estimation on one object in
the scene. The selected object is painted green and identiﬁed by a
red bounding box. We can see that FCGF + SPN cannot generate
satisfactory results. SuperGlue + RANSAC fails to estimate the
pose for the image in YCB-Video dataset due to the target is too
small. Our method performs well across different scenes.
Synthetic
GraspNet
YCB Video
RGB image
Ground Truth Pose
Our Result
SuperGlue + RANSAC
Figure 8. The qualitative result of pose estimation on all objects in
the scene.
6.2.2
Qualitative and Quantitative Results on 6D Pose
As discussed in Section 2, no previous work proposes solu-
tion to this new task. We implement several baseline meth-
ods based on both deep-learning and conventional algo-
rithms. These baselines include point cloud clustering [17]
+ ICP registration [46] , SuperGlue [49] + RANSAC [19] +
least square ﬁtting and FCGF [7] + SPN.
The quantitative results using ADD, ADD-S and IADD
metrics are reported in Table 1. We can see that our method
outperforms other baselines by a large margin in all the test
subset. From the AUC scores of different metrics across
different test subset, we can see that IADD is more nu-
merical stable than ADD-S. For example, both ADD and
IADD reports a lower scores for our method on YCB-V
than on G.Novel subset, while ADD-S reports a better per-
formances. We also show the qualitative results of object
6D pose generated by different methods in Fig. 7 and Fig. 8.
Our method is more robust compared with other baselines.

Table 1. Quantitative results of different methods. g.t., w.o., G. and YCB-V are short for ground truth, without, GraspNet-1Billion [18]
and YCB-Video [56] respectively. The number is the area under curve(AUC) score of each methed using each metric. The upper bound of
AUC is set to 0.5× diagonal of the target object.

## related_work
In this section, we brieﬂy review previous researches on
object 6D pose estimation, 3D correspondence, and 6D pose
estimation metrics.
2.1. Object 6D Pose Estimation Algorithms
Current existing algorithms can be divided into mainly
three types.
Pose prediction methods tend to directly obtain the object
6D pose from image features. Some [31, 51, 54, 56] apply
classiﬁcation or regression to get the object 6D pose after
extracting pattern features by deep neural networks. Oth-
ers [34,40] iteratively optimize the object 6D pose by mini-
mizing the re-projection error. These algorithms work well
when objects are with rich texture but fail on texture-less
objects or occlusions. Other methods [13] requires an addi-
tional 2D detector. [14] focuses on category level pose esti-
mation and is not suitable for our task, since the test objects
are totally novel (see Figure 2).
Correspondences based methods aim to ﬁrstly detect 2D
or 3D object keypoints in the image and then solve a PnP
or ﬁtting problem to obtain the object 6D pose. In the for-
mer case, methods extract 2D keypoints [41, 44, 60, 61] of
the target objects and apply PnP-RANSAC [19] algorithms
to obtain their 6D poses. The latter ones [21, 22] ﬁnd 3D
keypoints of the target objects and calculate the 6D pose by
least square ﬁtting. These methods require a deﬁnition of
the keypoints as prior knowledge.
Beyond the former types of methods that require the net-
work to implicitly remember the target objects during train-

ing, registration based methods [2, 15, 16, 33, 46, 55, 59]
treat this task as point cloud registration and could estimate
the 6D pose between two novel inputs. However, they usu-
ally consider the registration between two similar-sized tar-
gets. In our cases, the intersection of union(IoU) between
the mesh model and the partial view scene point cloud is
small, which makes them difﬁcult to be registered.For ex-
ample, [15] mainly register two partial view point clouds
and [16] mainly register two full object meshes. As the IoU
between point clouds in these cases is high, the methods
work well. However, it fails in our cases when a partial view
point cloud needs to be aligned with a full object mesh. So
far, the most similar method with us is Scan2CAD [2] that
matches furniture CAD models with indoor RGB-D scan.
Our task and method differ from theirs in four aspects: (a)
our target scene is a single-view partial point cloud which
is closer to a practical setting, while [2] focuses on a 3D
reconstruction of an indoor scene, (b) our objects can be
precisely matched to the scene targets, while [2] considers
a CAD object set that can only be roughly matched with
the scene targets, (c) both the objects and scenes have color
information in our setting, while [2] only focuses on ge-
ometry matching and (d) we focus on table-top level object
pose estimation while [2] focuses on larger scale in-door
environment furniture alignment.
2.2. Keypoint Features and Matching.
Given two RGB images, conventional methods [30, 36,
50] use hand-crafted features such as SIFT [37], SURF [4]
and ORB [45] for corresponding keypoints detection and
matching. Recently, deep learning techniques have been ap-
plied to this long-standing area [10, 49] and show promis-
ing performances in both accuracy and efﬁciency. A sim-
ilar trend also appears in the 3D area. Researchers pro-
posed hand-crafted descriptors [47, 48] in early years for
point cloud registration. However, these methods are time-
consuming and limited in performance. Point cloud based
neural networks [7, 8, 42, 43] improved the performances
and found a balance between efﬁciency and accuracy. Other
researches [21,54,57] extracted multi-modal features by fu-
sion. The fusion of multi-modal information compensates
for the limitations of any single-modal features and thus im-
proves the overall performance.
2.3. 6D Pose Estimation Metrics.
So far, the most commonly used metrics in 6D pose es-
timation literature are ADD [23] and ADD-S [56]. ADD
measures the average point error between the estimated
pose and the ground truth pose. It is intuitive but not appli-
cable to rotational symmetric objects because of the prob-
lem of pose ambiguity. ADD-S metric is thus proposed to
solve the problem. However, ADD-S is not a good mea-
surement on pose error itself and can be problematic under
Figure 2. Examples of the unseen objects in the test set.
some circumstances. Examples will be given in supplemen-
tary materials. ACPD and MCPD metrics [28] are proposed
which can handle objects with ﬁnite pose ambiguities in a
uniﬁed manner. But, they still fail when objects have inﬁ-
nite ambiguous poses.
3. Task Deﬁnitions
Point Cloud is deﬁned by a matrix P
P =


x1
y1
z1
r1
g1
b1
x2
y2
z2
r2
g2
b2
· · ·
· · ·
xn
yn
zn
rn
gn
bn


(1)
in which xi, yi, zi and ri, gi, bi represent the 3D coordi-
nates and RGB values of the ith point respectively.
Object 6D Pose T is an element of the special Euclidean
group SE(3) that represents the object translation and rota-
tion in the scene.
T ∈SE(3)
(2)
For the task of unseen object 6D pose estimation, the
input of the task is a tuple I.
I = (s, o)
(3)
in which s and o represent the scene and object respectively.
The scene s is usually denoted by a colored point cloud
which is captured by indoor RGBD cameras such as Intel
RealSense or Lidar. The object o is usually denoted by a
triangle mesh model which can also be sampled and inter-
polated as a colored point cloud.
Unseen object 6D pose estimation algorithm F is a func-
tion that maps the input tuple to a 6D pose, through which
the object mesh can be transformed to its scene counterpart
in the camera frame.
F(I) = F(s, o) →T
(4)
The dataset D is composed of the training set Dtrain and
test set Dtest.
D = Dtrain ∪Dtest
Dtrain ∩Dtest = ∅
(5)

Each element d ∈D is a tuple di = (si, oi, T i) in which
T i is the ground truth 6D pose. Assume Otest is the object
set for the testing and Otrain is the one for the training.
Previous algorithms focus on the problem when Otrain ⊇
Otest. However, this unseen object 6D pose estimation task
requires novel object in the test set.
Otrain =

oi
train, i = 1, 2, · · · , ntrain
	
Otest =

oi
test, i = 1, 2, · · · , ntest
	
∃o ∈Otest, o /∈Otrain
(6)
Suppose TP, M(TP, T, o) are the predicted pose and
pose error metric. The task requires the algorithm F to min-
imize the average pose error on the test set given the training
set.
argmin
F
1
ntest
ntest
X
i=1
M(F(si
test, oi
test), T i
test, oi
test) |Dtrain
(7)

## conclusion
As discussed above, for the task of object 6D pose esti-
mation, ADD is a suitable metric for objects without pose
ambiguity but cannot be applied to those with pose ambi-
guity. ADD-S is problematic for objects without pose am-
biguity and has some drawbacks for objects with pose am-
biguity. As a result, neither ADD nor ADD-S can be used
to evaluate objects pose for those both with pose ambiguity
and without ambiguity in a uniﬁed manner.
8.5. MLP Structure
The stucture of the Multi Layer Perceptron (MLP) in the
main paper is shown in Fig. 13, which is composed of 4
blocks.
Block
Block
N × H
N × H
Block
Block
N × H
N × H
N × H
N × O
FC
Input
Output
N × C
FC
FC
Layer Norm
ReLU
Dropout
Block
: element-wise addition 
Figure 13. Structure of the Multi-Layer Perceptron (MLP) with
residual blocks used in our networks. FC represents Fullly Con-
nected Layer.