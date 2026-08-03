# OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models

> 2023 · id: W4317552994 · arXiv: 2301.07673 · pdf: https://arxiv.org/pdf/2301.07673 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We propose a new method for object pose estimation without CAD models. The
previous feature-matching-based method OnePose [48] has shown promising re-
sults under a one-shot setting which eliminates the need for CAD models or
object-speciﬁc training. However, OnePose relies on detecting repeatable im-
age keypoints and is thus prone to failure on low-textured objects. We propose
a keypoint-free pose estimation pipeline to remove the need for repeatable key-
point detection. Built upon the detector-free feature matching method LoFTR
[47], we devise a new keypoint-free SfM method to reconstruct a semi-dense
point-cloud model for the object. Given a query image for object pose estima-
tion, a 2D-3D matching network directly establishes 2D-3D correspondences
between the query image and the reconstructed point-cloud model without ﬁrst
detecting keypoints in the image. Experiments show that the proposed pipeline
outperforms existing one-shot CAD-model-free methods by a large margin and is
comparable to CAD-model-based methods on LINEMOD even for low-textured
objects. We also collect a new dataset composed of 80 sequences of 40 low-
textured objects to facilitate future research on one-shot object pose estimation.
The supplementary material, code and dataset are available on the project page:
https://zju3dv.github.io/onepose_plus_plus/.
1

## introduction
Object pose estimation is crucial for immersive human-object interactions in augmented reality
(AR). The AR scenario demands the pose estimation of arbitrary household objects in our daily
lives. However, most existing methods [39, 29, 38, 55, 2, 4, 37] either rely on high-ﬁdelity object
CAD models or require training a separate network for each object category. The instance- or
category-speciﬁc nature of these methods limits their applicability in real-world applications.
To alleviate the need for CAD models or category-speciﬁc training, OnePose [48] proposes a new
setting of one-shot object pose estimation. It assumes that only a video sequence with annotated
object poses is available for each object and aims for its pose estimation in arbitrary environments.
This setting eliminates the requirements for CAD models and the separated pose estimator training
for each object, and thus is more widely applicable for AR applications. OnePose adopts the feature-
matching-based visual localization pipeline for this problem setting. It reconstructs sparse object
point clouds with SfM [44] and establishes 2D-3D correspondences between keypoints in the query
image and the point cloud model to estimate the object pose. Being dependent on detecting repeatable
∗The ﬁrst two authors contributed equally. The authors from Zhejiang University are afﬁliated with the State
Key Lab of CAD&CG and the ZJU-SenseTime Joint Lab of 3D Vision.
†Corresponding author.
36th Conference on Neural Information Processing Systems (NeurIPS 2022).
arXiv:2301.07673v1  [cs.CV]  18 Jan 2023

Ref Images
Reconstructed SfM Point Cloud
Object Pose Estimation for Query Images
Ours
OnePose
Ours
OnePose
20113 points
20177 points
1193 points
972 points
Figure 1: Comparsion Between Our Method and OnePose [48]. For low-textured objects that
are challenging for OnePose, our method can reconstruct their semi-dense point clouds with more
complete geometry and thus achieves more accurate object pose estimation. Green and blue boxes
represent ground truth and estimated poses, respectively.
keypoints, OnePose struggles with low-textured objects whose complete point clouds are difﬁcult to
reconstruct with keypoint-based SfM. Without complete point clouds, pose estimation is prone to
failure for many low-textured household objects.
We propose to use a keypoint-free feature matching pipeline on top of OnePose to handle low-textured
objects. The keypoint-free semi-dense feature matching method LoFTR [47] achieves outstanding
performance on matching image pairs and shows strong capabilities for ﬁnding correspondences in
low-textured regions. It uses centers of regular grids on a left image as “keypoints”, and extracts
sub-pixel accuracy matches on the right image in a coarse-to-ﬁne manner. However, this two-view-
dependent nature leads to inconsistent “keypoints” and fragmentary feature tracks, which go against
the preference of modern SfM systems. Therefore, keypoint-free feature matching cannot be directly
applied to OnePose for object pose estimation. We will further elaborate this issue in Sec. 3.1.
To get the best of both worlds, we devise a novel system to adapt keypoint-free matching for one-shot
object pose estimation. We propose a two-stage pipeline for reconstructing a 3D structure, striving
for both accuracy and completeness. For testing, we propose a sparse-to-dense 2D-3D matching
network that efﬁciently establishes accurate 2D-3D correspondences for pose estimation, taking full
advantage of our keypoint-free design.
More speciﬁcally, to better adapt LoFTR [47] for SfM, we design a coarse-to-ﬁne scheme for accurate
and complete semi-dense object reconstruction. We disassemble the coarse-to-ﬁne structure of
LoFTR and integrate them into our reconstruction pipeline. In the coarse reconstruction phase, we use
less accurate yet repeatable LoFTR coarse correspondences to construct consistent feature tracks for
SfM and yield an inaccurate but complete semi-dense point cloud. Then, our novel reﬁnement phase
optimizes the initial point cloud by reﬁning “keypoint” locations in coarse feature tracks to sub-pixel
accuracy. As shown in Fig. 1, our framework can reconstruct accurate and complete semi-dense point
clouds even for low-textured objects, which lays the foundation for building high-quality 2D-3D
correspondences for pose estimation.
At test time, we draw inspiration from the sparse-to-dense matching strategy in visual localization [12],
and further adapt it to direct 2D-3D matching in a coarse-to-ﬁne manner for efﬁciency. Additionally,
we use self- and cross-attention to model long-range dependencies required for robust 2D-3D
matching and pose estimation of complex real-world objects, which usually contain repetitive patterns
or low-textured regions.
We evaluate our framework on the OnePose [48] dataset and the LINEMOD [16] dataset. The
experiments show that our method outperforms all existing one-shot pose estimation methods [48, 33]
by a large margin and even achieves comparable results with instance-level methods [39, 29] which are
trained for each object instance with a CAD model. To further evaluate and demonstrate the capability
of our method in real-world scenarios, we collect a new dataset named OnePose-LowTexture, which
comprises 80 sequences of 40 low-textured objects.
Contributions.
• A keypoint-free SfM method for semi-dense reconstruction of low-textured objects.
• A sparse-to-dense 2D-3D matching network for accurate object pose estimation.
• A challenging real-world dataset OnePose-LowTexture composed of 40 low-textured objects with
ground-truth object pose annotations.
2

2

## method
An overview of our method is shown in Fig. 2. Given a reference image sequence with known
object poses {Ii, ξi}, our objective is to estimate the object poses {ξq} for the test images, where
i and q denote the indices of the reference images and test images, respectively. To achieve this
goal, we propose a novel two-stage pipeline, which ﬁrst reconstructs the accurate semi-dense object
point cloud from reference images (Section 3.2), and then solves the object pose by building 2D-3D
correspondences in a coarse-to-ﬁne manner for test images (Section 3.3). Since our method is highly
related to the keypoint-free matching method LoFTR [47], we give it a short overview in Section 3.1.
3.1

## experiments
4.1
Datasets
OnePose and LINEMOD Datasets.
We validate our method on the OnePose [48] and
LINEMOD [16] datasets. The OnePose dataset is newly proposed, which contains around 450
real-world video sequences of 150 objects. LINEMOD is a broadly used dataset for object pose
estimation. For both datasets, we follow the train-test split in previous methods [48, 29].
OnePose-LowTexture Dataset.
Since the original OnePose evaluation set mainly comprises tex-
tured objects, we collected an additional test set, named OnePose-LowTexture, to supplement the
original OnePose dataset. The proposed dataset is composed of 40 household low-textured objects.
For each object, there are two corresponding videos captured with different backgrounds, one as
the reference video and the other for testing. Besides, to evaluate and compare our method with
CAD-model-based methods, we further obtain high-ﬁdelity 3D models of eight randomly selected
objects with a commercial 3D scanner. Some example images are shown in Fig. 5. Please refer to the
supplementary material for more details.
7

Table 1: Comparison with One-shot Baselines. Our method is compared with HLoc [40] combined
with different feature matching methods and OnePose [48], using the cm-degree pose success rate
with different thresholds.
OnePose dataset
OnePose-LowTexture
Time (ms)
1cm-1deg 3cm-3deg 5cm-5deg 1cm-1deg 3cm-3deg 5cm-5deg
HLoc (SPP + SPG)
51.1
75.9
82.0
13.8
36.1
42.2
835
HLoc (LoFTR∗)
39.2
72.3
80.4
13.2
41.3
52.3
909
OnePose
49.7
77.5
84.1
12.4
35.7
45.4
66.4
Ours
51.1
80.8
87.7
16.8
57.7
72.1
88.2
Table 2: Comparison with Instance-level Baseline. Our method is compared with PVNet[39] on
objects with CAD models in the OnePose-LowTexture dataset using the ADD(S)-0.1d metric.
Obj. ID
0700 0706
0714
0721
0727 0732
0736 0740
Avg.
PVNet
12.3
90.0
68.1
67.6
95.6
57.3
49.6
61.3
62.7
Ours
89.5
99.1
97.2
92.6
98.5
79.5
97.2
57.6
88.9
4.2
Experiment Settings and Baselines
Baselines.
We compare the proposed method with the following baselines in two categories: 1)
One-shot baselines [48, 33, 40] that hold the same setting as ours. OnePose [48] and HLoc [40] are
most relevant to our method in leveraging feature matching for reconstruction and pose estimation.
To be speciﬁc, we compare with HLoc combined with different feature matching methods including
SuperGlue [41] and LoFTR [47]. 2) Instance-level baselines [39, 29] that require CAD-models and
need to be trained separately for each object. These methods achieve high accuracy through training
on many rendered images with extensive data augmentation. We compare our method with them
to demonstrate that our method achieves competitive results while not relying on CAD models and
eliminating per-object pose estimator training.
Evaluation Protocols.
We compare our method with OnePose and HLoc using the same set of
reference images. Since HLoc’s original retrieval module is designed for the outdoor scenes, we
use uniformly sampled 10 reference views for 2D-2D matching for pose estimation, following [48].
For the comparison with PVNet [39], we follow its original training setting, which ﬁrst samples
8 keypoints on the object surface and then trains a network using 5000 synthetic images for each
object. In contrast, our method only uses around 200 reference images to reconstruct the object
point cloud. We evaluate our method and PVNet on the same real-world test sequences, while our
matching model has never seen the test objects before. As for the experiments on LINEMOD, we
compare our method with OnePose by running their open-source code. Our method and OnePose
share the same 2D bounding boxes from an off-the-shelf object detector YOLOv5 [1]. Note that the
object detector is trained on real-world images only to provide rough bounding boxes. We use the
real training images (∼180) for object reconstruction and all test images for evaluation. The results
of other baselines on LINEMOD are from the original papers.
Metrics.
We use metrics including the cm-degree pose success rate, the ADD(S)-0.1d average
distance with a threshold of 10% of the object diameter, and the 2D projection error Proj2D with a
threshold of 5 pixels. The deﬁnitions of these metrics are detailed in the supplementary material.
4.3
Results on the OnePose and OnePose-LowTexture Datasets
Comparison with One-shot Baselines.
The cm-degree success rate with different thresholds are
used for evaluation. As shown in Tab. 1, our method substantially outperforms OnePose [48]
and HLoc [40]. Objects in the OnePose dataset have rich textures, beneﬁting keypoint detection.
Therefore, keypoint-based methods OnePose and HLoc (SPP+SPG) perform reasonably well. Our
method achieves even higher accuracy thanks to the keypoint-free design, effectively utilizing both
texture-rich and low-textured object regions for pose estimation. On the OnePose-LowTexture dataset,
our method surpasses OnePose and HLoc by a large margin. This further demonstrates the capability
of our keypoint-free design for object reconstruction and the sparse-to-dense 2D-3D matching for
object pose estimation. HLoc (LoFTR∗) uses LoFTR coarse matches for SfM and uses full LoFTR to
match the query image and its retrieved images for pose estimation. It does not rely on keypoints,
similar to our design. Our method signiﬁcantly outperforms it on accuracy and runs ∼10× faster.
8

Table 3: Results on LINEMOD. Our method is compared with Instance-level and One-shot baselines.
Note that Gen6D is ﬁne-tuned on a selected subset of objects and uses the rest for testing. Gen6D† is
the version without ﬁne-tuning on LINEMOD. Symmetric objects are indicated by ∗.
Type
Name
Object Name
Avg.
ape benchwise cam
can
cat
driller duck eggbox∗glue∗holepuncher iron lamp phone
ADD(S)-0.1d
Instance-level
CDPN
67.3
98.8
92.8 96.6 86.6
95.1
75.2
99.6
99.6
89.7
97.9 97.8
80.7
91.4
PVNet
43.6
99.9
86.9 95.5 79.3
96.4
52.6
99.2
95.7
81.9
98.9 99.3
92.4
86.3
One-shot
Gen6D†
-
62.1
45.6
-
40.9
48.8
16.2
-
-
-
-
-
-
-
Gen6D
-
77.0
66.1
-
60.7
67.4
40.5
95.7
87.2
-
-
-
-
-
OnePose
11.8
92.6
88.1 77.2 47.9
74.5
34.2
71.3
37.5
54.9
89.2 87.6
60.6
63.6
Ours
31.2
97.3
88.0 89.8 70.4
92.5
42.3
99.7
48.0
69.7
97.4 97.8
76.0
76.9
Proj2D
Instance-level
CDPN
97.5
98.8
98.6 99.6 99.3
94.9
98.4
99.1
98.4
99.5
97.9 95.7
96.8
98.0
PVNet
99.2
99.8
99.2 99.9 99.3
96.9
98.0
99.3
98.5
100.0
99.2 98.3
99.4
99.0
One-shot
OnePose
35.2
94.4
96.8 87.4 77.2
76.0
73.0
89.9
55.1
79.1
92.4 88.9
69.4
78.1
Ours
97.3
99.6
99.6 99.2 98.7
93.1
97.7
98.7
51.8
98.6
98.9 98.8
94.5
94.3
The improved accuracy and speed come from the accurate point cloud reconstructed by our novel
SfM framework and the efﬁcient 2D-3D matching module.
Comparison with Instance-level Baseline PVNet.
On the OnePose-LowTexture dataset, the pro-
posed method is compared with PVNet [39] on the subset objects with scanned models. The ADD(S)-
0.1d results are presented in Tab. 2. Even though PVNet is trained on a large number (∼5000)
of rendered images covering almost all possible views, our method still outperforms it on most
objects without additional training. We attribute this to PVNet’s susceptibility to domain gaps and
our matching module’s robustness and generalizability, thanks to its large-scale pre-training.
4.4
Results on LINEMOD
We compare the proposed method with OnePose [48] and Gen6D [33] which are under the One-shot
setting, and Instance-level methods PVNet [39] and CDPN [29] on ADD(S)-0.1d and Proj2D metrics.
As shown in Tab. 3, our method outperforms existing one-shot baselines signiﬁcantly and achieves
comparable performance with instance-level methods. Notably, our method and OnePose are only
trained on the OnePose training set and tested on LINEMOD without additional training.
Since LINEMOD is mainly composed of low-textured objects, our method outperforms OnePose
signiﬁcantly thanks to the keypoint-free design. Gen6D [33] is CAD-model-free and can generalize
to unseen objects similar to our method. However, it relies on detecting accurate object bounding
boxes for pose initialization, which is hard on LINEMOD because of the poor image quality and
slight object occlusion. In contrast, our method only needs rough object detection to reduce possible
mismatches, which is more robust to detection error. Moreover, the performance of Gen6D drops
signiﬁcantly without training on a subset of LINEMOD, while our method requires no extra training
and achieves much higher accuracy than Gen6D. The experiment demonstrates the superiority of our
method over existing methods under the one-shot setting.
Our method has lower or comparable performance with instance-level methods [39, 29], which are
trained to ﬁt each object instance, and thus perform well naturally, at the expense of the tedious
training for each object. In contrast, our method is grounded in highly generalizable local features
and generalizes to unseen objects with comparable performances.
4.5
Ablation Studies
We conduct several experiments on the OnePose datas

## related_work
Keypoint-Free Feature Matching Method LoFTR [47].
Without a keypoint detector, LoFTR
builds semi-dense matches between image pairs (noted as left and right images) in a coarse-to-ﬁne
pipeline. First, dense matches between two coarse-level feature maps (1/8 resolution in LoFTR) are
built and upsampled, yielding coarse semi-dense matches in the original resolution. With the locations
of all left matches ﬁxed, the right matches are reﬁned to a sub-pixel level using ﬁne-level feature
maps. Thanks to the keypoint-free design and the global receptive ﬁeld of Transformers, LoFTR is
capable of building correspondences in low-textured regions.
Problem of Using LoFTR for Keypoint-Based SfM.
Directly combining LoFTR with modern
keypoint-based SfM systems such as COLMAP [44] is not applicable since they rely on ﬁxed
keypoints detected on each image to construct feature tracks for estimating 3D structures. However,
for LoFTR, its matching locations on a right image depend on its pairing left images. Therefore, the
right matching locations are not consistent when paired with multiple left images. Due to this reason,
keypoint-free feature matching cannot establish feature tracks across multiple views for effective 3D
structure optimization in SfM and is thus not directly applicable in OnePose.
4

1. Coarse Matching Pairs
2. Coarse Reconstruction
3. Feature Track Reﬁnement
4.Point Cloud Optimization
dr
ur
Pj
c
ur
T j
f
T j
c
ˆuk
s
˜uk
Pj
Figure 3: Keypoint-Free SfM. 1. We ﬁrst build repeatable coarse semi-dense 2D matches between
image pairs. 2. Then, we feed coarse matches to COLMAP [44] to build a coarse feature track T j
c
and a coarse 3D point Pj
c ( ). 3. To reﬁne T j
c , we ﬁx a reference node ur ( ) and search around the
local window ( ) of each source node ˜uk
s ( ) for sub-pixel correspondences ˆuk
s ( ). 4. Finally, we
optimize the depth dr of ur by minimizing reprojection errors. We back-project ur with its reﬁned
dr to the object coordinate to obtain an optimized accurate object point cloud Pj ( ).
3.2
Keypoint-Free Structure from Motion
To better adapt LoFTR for SfM, we design a coarse-to-ﬁne SfM framework leveraging the properties
of LoFTR’s coarse and ﬁne stages separately. Our framework constructs the coarse structure of the
feature tracks {T j
c } and point cloud {Pj
c} in the coarse reconstruction phase. Then in the reﬁnement
phase, the coarse structures are reﬁned to obtain the accurate point cloud {Pj}. For clarity, in this
part, we use ˜· to denote the coarse matching results and use ˆ· to denote ﬁne matching results. We
consider the feature track T j = {uk ∈R2|k = 1...Nj} as a set of matched 2D points observing a
3D point Pj ∈R3. j denotes the index of the feature track and its corresponding 3D point.
Coarse Reconstruction.
We ﬁrst strive for the completeness of the initially reconstructed 3D
structure. We propose to use the inaccurate yet repeatable coarse correspondences of LoFTR for
COLMAP [44] to reconstruct the coarse 3D structure. The coarse correspondences, as shown in
Fig. 3 (1), can be seen as pixel-wise dense correspondences on downsampled image pairs. Every
pixel in the downsampled image can be regarded as a “keypoint” in the original image. Therefore,
performing coarse matching can provide repeatable semi-dense correspondences for COLMAP to
reconstruct coarse feature tracks {T j
c } and semi-dense point cloud {Pj
c}, as shown in Fig. 3 (2).
Reﬁnement.
Due to the limited accuracy of performing matching on downsampled images, the
point cloud from the coarse reconstruction is inaccurate and thus insufﬁcient for the object pose
estimation. Therefore, we further reﬁne the object point cloud {Pj
c} with sub-pixel correspondences.
To achieve this, we ﬁrst ﬁx the position of one node for each feature track T j
c and reﬁne other nodes
within the track. Then, we use the reﬁned tracks {T j
f } to optimize the {Pj
c}.
For the reﬁnement of {T j
c }, we draw the idea from the ﬁne-level matching module in LoFTR and
adapt it to the multi-view scenario. As shown in Fig. 3 (3), we ﬁrst select and ﬁx one node in each T j
c
as the reference node ur, and then perform ﬁne matching with each of the remaining source node ˜uk
s.
The ﬁne matching searches within a local region around each ˜uk
s for a sub-pixel correspondence ˆuk
s,
so the nodes’ locations in the coarse feature track are reﬁned. We denote the reﬁned feature tracks as
{T j
f }. Details about the selection of reference nodes ur are provided in the supplementary material.
We now treat the reﬁned feature tracks {T j
f } as ﬁxed measurements, and optimize the 3D locations
of the coarse point cloud {Pj
c} using reprojection errors as shown in Fig. 3 (4). To accelerate the
convergence, inspired by SVO [11], we further decrease the DoF of each Pj
c by only optimizing the
depth dr of each reference node ur. Speciﬁcally, we transform each point Pj
c to the frame of ur
and use its coordinate of z-axis to initialize dr. Then we optimize each reference node depth dr by
minimizing the distance between each reprojected location and the reﬁned feature location ˆuk
s:
d∗
r = argmin
dr
P
k∈Nj−1
∥ˆuk
s −π
 ξr→sk · π−1 (ur, dr)

∥2.
(1)
where π is the projection determined by intrinsic camera parameters, and ξr→sk = ξsk · ξ−1
r
is the
relative pose between the frame of the reference node and k-th source node.
Finally, the optimized depth d∗
r of each reference node is transformed to the canonical object
coordinate to get the reﬁned 3D point Pj. Notably, when applying the proposed system in practical
AR applications, we can optimize inaccurate camera poses obtained from ARKit along with the 3D
points, i.e., solving a bundle adjustment problem. For the later 2D-3D matching at test time, we
calculate and store each 3D point feature by averaging the 2D features of its associated 2D points.
5

Query Image
PnP
Pose
Object Point Cloud
Self & 
Cross 
Attention
Correlation 
& 
Expectation
1. Coarse Matching
2. Fine Matching
Self
Cross
ˆF3D
˜F2D
˜F3D
ˆFcrop
LjxV9y4UMStP+HOv3HSZqGtBy4czrmXe+/xIkqENM1vrbCwuLS8Ulwtra1vbG7p2ztEcYc4RYKaci7HhSYEoZbkiKuxHMPAo7nijy8zvPGAuSMju5DjCTgAHjPgEQakV9+r2EMoEzuAcuj5yXWausnxVq9P6q4etmsmRMY8TKSRnkaLr6l90PURxgJhGFQvQsM5JOArkiOK0ZMcCRxCN4AD3FGUwMJj+
kxqFS+oYfclVMGhP190QCAyHGgac6s1vFrJeJ/3m9WPrnTkJYFEvM0HSRH1NDhkYWiNEnHCNJx4pAxIm61UBDyCGSKraSCsGafXmetOs167R2clsvNy7yOIpgHxyAKrDAGWiAG9AELYDAI3gGr+BNe9JetHftY9pa0PKZXfAH2ucPEDaXJA=</latexit>ˆF3D(j)
ˆuq
⇠q
Flatten
Mc
3D
ˆF
t
crop
+zJkC2/42CguLS8srxdXS2vrG5pa5vdNUSIJbZCIR7LtY0U5E7QBDhtx5Li0Oe05Q8vx37rgUrFInEHo5h6Ie4LFjCQUtdc8dYEhdoI/gB+l1lnXTk6vsHrpm2a7YE1jzxMlJGeWod80vtxeRJKQCMdKdRw7Bi/FEhjhNCu5iaIxJkPcpx1NBQ6p8tLJA5l1qJWeFURSlwBrov6eSHGo1Cj0dWeIYaBmvbH4n9dJIDj3UibiBKg0VBwi2IrHEaVo9JSoCPNMFEMn2rRQZYgI6s5IOwZl9eZ40jyv
OaV6Wy3XLvI4imgfHaAj5KAzVEM3qI4aiKAMPaNX9GY8GS/Gu/ExbS0Y+cwu+gPj8wdr65by</latexit>
ˆF
t
3D
ˆF2D
JiFv+LGhSJu/Q13/o2TNgtPTBwOde7pnjRYxKZVnfRmFhcWl5pbhaWlvf2Nwyt3daMowFJk0cslB0PCQJo5w0FVWMdCJBUOAx0vZGl5nfiBC0pDfqXFEnANOPUpRkpLrlX6QVIDTFiyU3qJsdX6X3ipxXLFtVawI4T+yclEGOhmt+9fohjgPCFWZIyq5tRcpJkFAUM5KWerEkEcIjNCBdTkKiHSf4UHmq
lD/1Q6McVnKi/NxIUSDkOPD2ZhZWzXib+53Vj5Z87CeVRrAjH0N+zKAKYVYG7FNBsGJjTRAWVGeFeIgEwkpXVtIl2LNfnietWtU+rZ7c1sr1i7yOItgHB+AI2OAM1ME1aIAmwOARPINX8GY8GS/Gu/ExHS0Y+c4u+APj8wefzZXb</latexit>
Mf
3D
{Pj}
Iq
Figure 4: Object Pose Estimation. At test time, we ﬁrst extract multi-scale query image features
{˜F2D, ˆF2D}. Coarse Matching module transforms coarse 2D and 3D features Nc times with self-
and cross-attention modules and then build their coarse 2D-3D correspondences Mc
3D. Next, we
crop the local window ˆFcrop on the ﬁne feature map around each coarse 2D match. Fine Matching
module transforms the 3D feature and cropped 2D features and calculates each 2D ﬁne match location
ˆuq with feature correlation and expectation. The object pose ξq is then solved using PnP with Mf
3D.
Note that we store coarse and ﬁne 3D features separately, which are extracted from multi-resolution
feature maps of LoFTR’s feature backbone.
3.3
Object Pose Estimation
At test time, we establish 2D-3D matches between the object point cloud {Pj} and the query image
Iq to estimate object pose ξq. Inspired by [47], we ﬁrst extract hierarchical feature maps of Iq and
then perform matching in a coarse-to-ﬁne manner for efﬁciency, as illustrated in Fig. 4.
Coarse 2D-3D Matching.
We ﬁrst perform dense matching between the pre-calculated coarse 3D
point features ˜F3D ∈RN×Cc and the extracted coarse image feature map ˜F2D ∈R
H
8 × W
8 ×Cc. This
phase globally searches for a rough correspondence of each 3D object point in the query image,
which also determines whether the 3D point is observable by Iq.
We augment 3D and 2D features {˜F3D, ˜F2D} with positional encodings, to make them position-
dependent, and thus facilitates their matching. Please refer to the supplementary material for more
details. Then we ﬂatten the 2D feature map and apply self- and cross-attention layers by Nc times to
yield the transformed features {˜Ft
3D, ˜Ft
2D}. Linear

## conclusion
We propose a keypoint-free SfM and pose estimation pipeline that enables pose estimation of both
texture-rich and low-textured objects under the one-shot CAD-model-free setting. Our method can
efﬁciently reconstruct accurate and complete 3D structures of low-textured objects and build robust
2D-3D correspondences with the test image for accurate object pose estimation. The experiments
show that our method achieves signiﬁcantly better pose estimation accuracy compared with existing
CAD-model-free methods, and even achieves comparable results with CAD-model-based instance-
level methods. Although we do not see the immediate negative societal impact of our work, we do
note that accurate object pose estimation can be potentially used for malicious purposes.
Limitations.
Being dependent on local feature matching, our method inherently suffers from very
low-resolution images and extreme scale and viewpoint changes. In the current pipeline, we still
need a separate object detector to provide rough regions of interest. In the future, we envision a more
tight integration with the object detector, where object detection can also be carried out through local
feature matching.
Acknowledgements.
The authors would like to acknowledge the support from the National Key
Research and Development Program of China (No. 2020AAA0108901), NSFC (No. 62172364), the
ZJU-SenseTime Joint Lab of 3D Vision, and the Information Technology Center and State Key Lab
of CAD&CG, Zhejiang University.
10