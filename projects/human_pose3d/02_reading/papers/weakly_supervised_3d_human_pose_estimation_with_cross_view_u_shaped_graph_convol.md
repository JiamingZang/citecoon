# Weakly-Supervised 3D Human Pose Estimation With Cross-View U-Shaped Graph Convolutional Network

> 2022 · id: W4287177665 · arXiv: 2105.10882 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
3D
HUMAN pose estimation aims to produce a 3-
dimensional ﬁgure that describes the spatial position
of the depicted person. This task has drawn tremendous
attention in the past decades [1]–[3], playing a signiﬁcant role
in many applications such as action recognition, virtual and
augmented reality, human-robot interaction, etc. Many recent
works [4]–[7] focus on estimating 3D human poses from
monocular inputs, either images or 2D keypoints. However, it
is ill-posed due to the inherent depth ambiguity since multiple
3D poses can map to the same 2D keypoints. As a result, most
monocular methods only estimate the relative positions to the
root joint and fail to estimate the absolute 3D poses, which
∗Equal contribution.
† Corresponding author.
G. Hua, H. Liu, W. Li, Q. Zhang, and R. Ding are with the Key Labora-
tory of Machine Perception, Peking University Shenzhen Graduate School,
Shenzhen 518055, China. E-mail: {glhua, hongliu, wenhaoli, qian.zhang,
dingrunwei}@pku.edu.cn.
Xin Xu is with the College of Intelligence Science and Technology,
National University of Defense Technology, Changsha, 410073, China. E-
mail: xinxu@nudt.edu.cn.
This
work
is
supported
by
National
Natural
Science
Foundation
of
China
(No.62073004,
No.61825305),
Shenzhen
Fundamental
Re-
search
Program
(No.
GXWD20201231165807007-20200807164903001,
JCYJ20190808182209321, JCYJ20200109140410340).
3D CNN / PSM
Input Images
2D Heatmaps
3D Poses
(a) Volumetric approach
Camera 1
Camera 2
Triangulation
2D Poses
Refined 3D Poses
Refined 
Network
Coarse 3D Poses
(b) Cross-view reﬁnement (ours)
Fig. 1.
(a) Most state-of-the-arts use multi-view images as input and follow
a pipeline that estimates 2D heatmaps and then directly recovers 3D poses
through volumetric convolutional neural networks or Pictorial Structure Model
(PSM). (b) Instead, we consider a 2D-3D lifting pipeline in a coarse-to-ﬁne
manner, which ﬁrst obtains coarse 3D poses through triangulation from cross-
view 2D joint detections and then reﬁnes the pose with a reﬁnement model.
greatly limits practical applications. Instead, exploiting multi-
view information is arguably the best way to achieve absolute
3D pose estimation [8].
Multi-view human pose estimation methods beneﬁt from
the complementary information from different camera views,
e.g. multi-view geometric constraints to resolve the depth
ambiguity and different views of the depicted person to deal
with the occlusion problem. Many existing multi-view based
methods [9]–[11] follow a pipeline that ﬁrst takes multi-view
images as input to predict 2D detection heatmaps and then
projects them to 3D poses through volumetric convolutional
networks or Pictorial Structure Model (PSM) [8], [12], as
shown in Figure 1 (a). However, using the convolutional
neural network to perform 2D-3D lifting requires quantities of
labelled 3D data as supervision, which is difﬁcult and costly
to collect. PSM discretizes the space around the root joint by
an N × N × N grid and assigns each joint to one of the
N 3 bins (hypotheses), therefore requiring no 3D ground truth.
However, the 2D-3D lifting accuracy of PSM based method is
subject to the number of grids, and the computation complexity
is of the order of O(N 6) which is computationally expensive.
To release the requirement of large quantities of 3D ground
truth and take the computation complexity into consideration,
arXiv:2105.10882v2  [cs.CV]  17 May 2022

2
Camera 1
2D Pose 
Estimation
Camera 2
Triangulation
CV-UGCN
2D Poses
Coarse 3D Poses
Refined 3D Poses
2D Pose 
Estimation
Weakly-supervised 
learning
Fig. 2.
The proposed pipeline for cross-view weakly-supervised 3D human pose estimation. Given the 2D poses estimated from RGB images of two
different camera views, the triangulation is ﬁrst performed to obtain coarse 3D poses. Then, a cross-view U-shaped graph network (CV-UGCN) trained in a
weakly-supervised manner is used to produce realistic and structurally plausible 3D poses.
a simple yet effective pipeline (see Figure 1 (b)) is proposed in
this paper for cross-view 3D human pose estimation. Different
from existing methods, our method estimates 3D human poses
from coarse to ﬁne and contains two steps: triangulation
and reﬁnement. Considering the increasing number of camera
views will bring more computation and reduce the ﬂexibility
of application in the wild, we only use two camera views for
training and inference. In the ﬁrst step, we perform the trian-
gulation between two camera views to lift 2D poses, which
can be obtained through any classic 2D keypoint detection
methods, to the 3D space. However, the triangulated 3D poses
are noisy and unreliable due to the errors of 2D keypoint
detection and camera parameters calibration, thus requiring
further reﬁnement.
In the reﬁnement progress, a lightweight cross-view U-
shaped graph convolutional network (CV-UGCN) is designed
to reﬁne the coarse 3D poses. By taking the cross-view
coarse 3D poses as input, CV-UGCN is able to exploit spatial
conﬁgurations and cross-view correlations to reﬁne the poses
to be more reasonable. Meanwhile, CV-UGCN is trained in
a weakly-supervised manner, requiring no 3D ground truth
but only 2D annotations. Speciﬁcally, by making full use of
the cross-view geometric constraints, geometric and structure-
aware consistency checks are introduced as the learning ob-
jective to train the network end-to-end.
We summarize our contributions as follows:
• A simple yet effective pipeline is proposed for cross-view
3D human pose estimation, which estimates the 3D hu-
man poses from coarse to ﬁne by using the triangulation
and the reﬁnement model.
• A cross-view U-shaped graph convolutional network (CV-
UGCN), which can take advantage of spatial conﬁgu-
rations and cross-view correlations, is proposed as the
reﬁnement model.
• A weakly-supervised learning objective containing geo-
metric and structure-aware consistency checks is intro-
duced, therefore releasing from the requirement of large
quantities of 3D ground truth for training.
Extensive experiments have been conducted on the bench-
mark dataset, Human3.6M, to verify the effectiveness of our
method. The Mean Per Joint Position Error (MPJPE) on the
benchmark dataset is 27.4 mm, which outperforms existing
state-of-the-art methods remarkably (27.4 mm vs 30.2 mm).

## method
V
3D
Model Size
MPJPE (mm)
Qiu et al. (CVPR’19) [10]
4
2.1 GB
31.2
Iskakov et al. (ICCV’19) [9]
4
✓
643 MB
20.8
Remelli et al. (CVPR’20) [15]
4
✓
251 MB
30.2
CPN + Ours
2
47 MB
27.4
2D-GT + Ours
2
20 MB
1.1
performance of estimating 3D poses [4]. We utilize Stack
Hourglass (SH) [40], Detectron [25], CPN [39], and 2D ground
truth (GT) with different levels of additive Gaussian noises to
explore the impact of different 2D detections. Figure 6 (a)
shows the relationship between the MPJPE of 3D poses and
two-norm errors of 2D detections. For both the triangulation
and the reﬁnement, the MPJPE of 3D poses increases lin-
early with the two-norm errors of 2D detections. However,
it can be observed that our reﬁnement model has a lower
incremental rate than triangulation, verifying the effectiveness
of our reﬁnement model to reﬁne the triangulated 3D poses.
Meanwhile, when the 2D ground truth is directly used for
triangulation, the results still have 9.1 mm MPJPE compared to
the 3D ground truth. It shows that the triangulation is sensitive
to the camera calibration noises. The proposed reﬁnement
model can effectively reﬁne the results to 1.1 mm MPJPE.
4) Hyperparameter evaluations: Due to the bone direction
consistency loss is calculated by measuring the cosine distance
between two bone vectors which is different from other losses
that calculate the Mean Squared Error (MSE), it is necessary
to ﬁnd a proper weighting factor for Lb. As shown in Figure 6
(b), the best performance is achieved when λb is equal to 0.1.
F. Generalization to Unseen Cameras
To validate the feasibility of applying our model to unknown
views, we train the model on some paired views and evaluate

9
Triangulation
Refined 3D Pose
Ground Truth
Triangulation
Refined 3D Pose
Ground Truth
Camera 2
Camera 1
Fig. 7.
Qualitative results of our approach on Human3.6M dataset (ﬁrst 3 rows) and HumanEva-I dataset (last 2 rows). The results of triangulation are noisy
and unreliable, while our model is able to produce realistic and structurally plausible 3D poses.
its performance on other unseen camera pairs. The results are
presented in Table V, where Pi denotes i-th paired views.
The results on unseen paired camera views are marked in
bold. It can be observed that CV-UGCN still can improve the
triangulated results in unseen camera views. For example, the
model trained only on P3 can also effectively reﬁne the trian-
gulation results on P1, P2 and P4. It veriﬁes the generalization
capability to unseen camera views of our method.
G. Model Size and Inference Time
We report the model size and inference time to show the
efﬁciency of our methods. Table VI exhibits the comparison of
the model size and performance with the recent methods [9],
[10], [15]. Because the comparison methods embed the 2D
detector into their model, we add the model size of CPN [39]
to ours for a fair comparison. It can be seen that our method
can achieve impressive performance with a lightweight model.
On a machine equipped with one NVIDIA RTX 2080 Ti
GPU, the 2D detector CPN requires about 0.02s to perform
the 2D detection, while our pipeline needs about another
0.01s to estimate the 3D poses. Consequently, when estimating
3D poses from images of two camera views, our method
could yield a real-time performance (∼33 fps). Note that if
a better and faster 2D detector is used with our method, the
performance and speed can have a further improvement.
H. Qualitative Results
3D reconstruction visualization.
Figure 7 shows some
visualization results of the triangulation and reﬁnement model.
The ﬁrst 3 rows are the results on Human3.6M dataset and the
last 2 rows are on HumanEva-I dataset. Because of the error of
2D detections and camera calibration errors, the triangulation
results are noisy and unreliable (see the poses in the red dotted
circles). Our reﬁnement model can efﬁciently improve the
coarse 3D poses to be more realistic and structurally plausible.
Visual results in the Real Scene. We apply our method to the
real scene to test its applicability. By using a stereo camera,
we take photos in a room and use the proposed method to
estimate the absolute 3D poses of the actor. Some visualization
results are given in Figure 8. It can be seen that the results are
reasonable, showing the feasibility of our method in real-scene
applications.

10
Camera 2
Camera 1
3D Pose
Fig. 8.
Visualization results of our method in the real scene. Camera 1 and Camera 2 are the left and right views of a stereo camera.

## experiments
A. Datasets and Evaluation Metrics
We evaluate our method on two standard benchmark
datasets, Human3.6M [34] and HumanEva-I [35].
Human3.6M. The Human3.6M dataset is the largest publicly
available benchmark dataset for 3D human pose estimation. It
consists of 3.6 million images captured from four synchronized
50 Hz cameras. There are 7 professional subjects performing
15 everyday activities. Following the standard protocol in prior
work [31], [32], we use 5 subjects (S1, S5, S6, S7, S8) for
training and 2 subjects (S9 and S11) for evaluation.
HumanEva-I. HumanEva-I is a smaller dataset with fewer
subjects and actions compared to Human3.6M, containing 3
subjects recorded from three synchronized camera views at 60
Hz. Following the train/test split in [25], [36], we train a single
model on all subjects for all actions and test on validation
sequences.
Evaluation Metrics. We report the Mean Per Joint Position
Error (MPJPE) and Procrustes analysis MPJPE (P-MPJPE) to
measure the 3D pose estimation accuracy. MPJPE is the eval-
uation metric referred to as protocol #1 in many works [11],
[37], which calculates the average Euclidean distance between
the ground truth and predictions. P-MPJPE reports the error
after the estimated 3D poses aligned to the ground truth in
translation, rotation, and scale, which is referred to as protocol
#2 [4], [38].
B. Implementation Details
In this work, all experiments are conducted on the PyTorch
framework with one NVIDIA RTX 2080 Ti GPU. Our model
is trained using Amsgrad optimizer with a mini-batch size of
256 for Human3.6M. For HumanEva-I, the mini-batch size is
set to 64. An initial learning rate of 0.001 is used and decreases
by 0.9 whenever the training loss does not decrease for every
10 epochs.
Different from other multi-view methods that use all the
camera views provided by the dataset, we only utilize two
camera views for training and inference due to the consider-
ation of computation complexity and implementing ﬂexibility
in the wild. For Human3.6M, adjacent camera pairs P1 (c1
and c2), P2 (c1 and c3), P3 (c2 and c4), and P4 (c3 and c4)
are used for training and testing. As for HumanEva-I, adjacent
camera pairs ˜P1 (c1 and c2), ˜P2 (c1 and c3), and ˜P3 (c2 and
c3) are used.
The 2D poses can be obtained by performing any classic
2D detection methods or directly using the 2D ground truth.
Following [25], we utilize the cascaded pyramid network
(CPN) [39] to obtain 2D poses of the Human3.6M dataset for
a fair comparison. For HumanEva-I, we directly use the 2D
ground truth to perform triangulation and then add Gaussian
noise with the different variances to the triangulated 3D poses
to validate the robustness of CV-UGCN.

7
TABLE III
ABLATION STUDIES ON EACH COMPONENT OF CV-UGCN. THE
EVALUATION IS PERFORMED ON HUMAN3.6M WITH MPJPE METRIC
UNDER PROTOCOL #1. ∆REPRESENTS THE PERFORMANCE GAP BETWEEN
THE METHODS AND OURS (CV-UGCN).

## related_work
Single-view 3D pose estimation. Current promising solutions
for monocular 3D pose estimation can be divided into two
categories. Methods of the ﬁrst category directly regress
the 3D poses from monocular images. Pavlakos et al. [13]
introduced a volumetric representation for 3D human poses,
while requiring a sophisticating deep network architecture
that is impractical in application. In the second category,
these works ﬁrst estimate 2D keypoints and then lift 2D
poses to the 3D space (2D-3D lifting). Martinez et al. [4]
predicted 3D poses via a fully-connected residual network and
showed low error rates when using 2D ground truth as input.
Cai et al. [14] presented a local-to-global GCN to exploit
spatial-temporal relationships to estimate 3D poses from a
sequence of skeletons. Meanwhile, they introduced a pose
reﬁnement step to further improve the estimation accuracy.
However, they only utilized the 2D detections to constrain
the depth-normalized poses, while ignoring the reﬁnement
for depth values. Different from [14], we perform both 3D
transformation and 2D reprojection consistency checks in our
reﬁnement model, so that the reﬁnement is more sufﬁcient.
Multi-view 3D pose estimation. In order to estimate the
absolute 3D poses, recent works seek to utilize information
from multiple synchronized cameras to solve the problem
of depth ambiguity. Most multi-view based approaches use
3D volumes to aggregate 2D heatmap predictions. Qiu et
al. [10] presented a cross-view fusion scheme to estimate 2D
heatmaps of multiple views and then used a recursive Pictorial
Structure Model to estimate the absolute 3D poses. Iskakov et
al. [9] proposed a learnable triangulation method to regress 3D
poses from multiple views. However, volumetric approaches
are computationally demanding. To recover 3D poses from
multi-view images without using compute-intensive volumet-
ric grids, Remelli et al. [15] exploited 3D geometry to fuse
input images into a uniﬁed latent representation of poses.
Different from these methods that embedded the improved 2D

3
S-GCN(3, 128)
S-GCN(128, 128)
S-GCN(3, 128)
S-GCN(128, 128)
M-GCN(128, 256)
M-GCN(256, 512)
Pooling
Upsampling
M-GCN(512, 512)
M-GCN(512, 512)
M-GCN(512, 512)
Pooling
FC(1024, 256)
Upsampling
FC(512, 3)
V×J×256
1×J×3
V×5×512
V×1×512
V×J×3
V×5×256
Fusion
S-GCN: Singe-view GCN Unit
M-GCN: Multi-view GCN Unit
Summation
Concatenation
Refined 3D Poses
Coarse 3D Poses
V×J×3
1×J×3
J: the number of joints
V: the number of camera views
Fused cross-view 
features
V×J×128
Fig. 3. A schematic of CV-UGCN. S-GCN units are ﬁrst utilized to preprocess the coarse 3D poses of each view to capture spatial conﬁgurations independently.
Then, the fused cross-view features are fed into the U-shaped architecture, where M-GCN units are utilized to explore additional cross-view correlations.
detector into their model to obtain more accurate 2D poses to
further improve the 3D pose estimation, our method focuses on
the task of 2D-3D lifting and can be easily integrated with any
2D detectors to achieve 3D pose estimation with a lightweight
reﬁnement model.
Weakly/self-supervised methods. Because 3D human pose
datasets are limited and collecting 3D human pose annota-
tions is costly, researchers have resorted to weakly or self-
supervised approaches. Zhou et al. [3] proposed a weakly-
supervised transfer learning method for in-the-wild images.
RepNet [16] proposed a weakly-supervised reprojection net-
work by using an adversarial training approach. Moreover,
in [17], a self-supervised learning method was proposed to
estimate 3D poses from unlabeled video frames via part guided
human image synthesis. Compared with previous methods, our
method has the advantage of decomposing the challenging 3D
human pose estimation task into two steps and making full
use of geometric and structure-aware consistency checks for
weakly-supervised learning.
III. CROSS-VIEW 3D HUMAN POSE ESTIMATION
Figure 2 depicts our pipeline for weakly-supervised cross-
view 3D human pose estimation. Given the estimated 2D poses
xi ∈RJ×2 from two different views, we aim at recovering
their absolute 3D poses Xi ∈RJ×3, where i is the index of
the camera views, and J is the number of joints. In particular,
we ﬁrst reconstruct coarse 3D poses through triangulation.
Then, a cross-view U-shaped graph convolutional network
(CV-UGCN) is proposed to reﬁne the coarse triangulated 3D
poses to obtain more precise estimations.
A. Triangulation
Assuming two cameras are synchronized and calibrated,
triangulation can be performed between two camera views to
lift 2D poses into the 3D space. Given the 2D joint locations
x1, x2 of two camera views, which can be obtained through
classic 2D keypoint detection methods, the triangulation is
solved through:

xj
1 ×

Tc1,w · ˜Xj
w

xj
2 ×

Tc2,w · ˜Xj
w


= 0,
(1)
where xj
1, xj
2 are the 2D coordinates of j-th joint, Tc1,w,
Tc2,w are the transformation matrixes between the camera ci
and world coordinate system. Since the origin of the world
coordinate system can be set in any positions, we select one
of the cameras as the origin to simplify the computation. Then,
the triangulation is deﬁned as:

xj
1 ×

I · ˜Xj
1

xj
2 ×

Tc2,c1 · ˜Xj
1


= 0,
(2)
where I is the identity matrix, and camera c1 is set as the origin
of the world coordinate system. By solving Eq. (2) through
Singular Value Decomposition (SVD), we can obtain the 3D
pose ˜X1 of the camera c1. Similarly, we can set c2 as the
origin to obtain the 3D pose ˜X2 of the camera c2.
Although triangulation is a straightforward way to achieve
2D-3D lifting, it is subject to the accuracy of the 2D joint
detections and the precision of calibrated camera parameters.
To solve this problem, Qiu et al. proposed a more robust
Recursive Pictorial Structure Model (RPSM) to replace the tri-
angulation [10]. Different from them, we propose to optimize
the initial triangulated 3D poses through a weakly-supervised
learning reﬁnement model, which is lightweight and requires
no 3D annotations to train.
B. Cross-view Reﬁnement Model
In order to reﬁne the coarse triangulated 3D poses, a cross-
view reﬁnement model is proposed. Speciﬁcally, we design a
cross-view U-shaped graph convolutional network, named CV-
UGCN, which can make full use of the spatial conﬁgurations
and cross-view correlations for reﬁnement. As speciﬁed in
Figure 3, the CV-UGCN takes the cross-view coarse 3D poses

4
Self-connection
Self-connection
Physically-connected
Physically-connected
Second-order
Second-order
Symmetrical
Symmetrical
View-connection
View-connection
(a)
(b)
(c)
Fig. 4.
(a) The human skeleton graph in kinematic connections and cross-view connections. (b) The adjacency matrix of single-view GCN. (c) The adjacency
matrix of multi-view GCN (2 views).
as input and outputs the residual shift added to the coarse 3D
poses to obtain the reﬁned 3D poses.
1) Graph modeling: Here, we ﬁrst give the deﬁnition of the
graph model of CV-UGCN. The cross-view skeletons are or-
ganized as an undirected graph in the spatial and camera-view
domains. The undirected graph G contains a set of vertices V
and edges E, where V = {vij | i = 1, . . . , V ; j = 1, . . . , J}
corresponds to J joints of the human body in V camera
views. For single-view GCN (S-GCN) that processes data
from a single view, the edge E only consists of kinematic
connections of spatial conﬁgurations. For multi-view GCN
(M-GCN) that embeds additional cross-view correlations into
the graph model, the edge E consists of two parts: kinematic
connections and cross-view connections.
2) Graph convolution: As presented in Figure 4, the deﬁned
undirected graph is represented in an adjacency matrix A ∈
RN×N, where N = V J (N = J for S-GCN). In detail, the
graph nodes are classiﬁed as the neighboring nodes according
to their semantic meanings in the human body structure, and
ﬁve kernels are used for different neighboring nodes: (i) self-
connection nodes; (ii) physical-connection nodes; (iii) second-
order connection nodes; (iv) symmetrical nodes; (v) view-
connection nodes. Note that S-GCN only processes single-
view data, thus having no view-connection nodes.
Given the input signal H ∈RN×C with C channels,
following [14], we update the graph convolution operation
in [18] by dismantling adjacent matrix into k sub-matrices
to:
Z
= C (H, A; W)
= P
k ˜
Dk
−1
2 ˜
Ak ˜
Dk
−1
2 HWk,
(3)
where Z ∈RN×F is the convolved signal matrix, C is
the function of the graph convolution. Wk ∈RC×F and
˜
Ak ∈RN×N are the learnable ﬁlter matrix and the normalized
adjacency matrix for the k-th type of neighboring nodes
respectively, and ˜
Dk
ii = P
j ˜
Ak
ij.
3) Network structure: As speciﬁed in Figure 3, S-GCN and
M-GCN units are the basic blocks to build our CV-UGCN.
Single-view GCN module. The triangulated results exist a lot
of noises that may disturb the accurate cross-view information
interaction. Therefore, the cross-view coarse 3D poses are ﬁrst
fed into S-GCN units independently to capt

## conclusion
In this paper, a simple yet effective pipeline is proposed
for cross-view 3D human pose estimation in a coarse-to-
ﬁne manner. Speciﬁcally, we ﬁrst exploit triangulation to lift
the 2D detections to coarse 3D poses and then utilize a
reﬁnement model to obtain precise results. In particular, a
new cross-view U-shaped graph convolutional network (CV-
UGCN) is designed as the reﬁnement model, which can take
advantage of spatial conﬁgurations and cross-view correlations
to accurately reﬁne the coarse 3D poses. Moreover, to release
the requirement of quantities of 3D ground truth as supervi-
sion, we introduce a weakly-supervised learning objective by
exploiting geometric and structure-aware consistency checks
in both single-view and cross-view optimizations. Extensive
experiments have been conducted on the benchmark dataset.
The results show that our method not only achieves state-of-
the-art performance but also is lightweight and could run in
real-time.