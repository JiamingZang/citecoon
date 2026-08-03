# MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation

> 2026 · id: arxiv:2601.06883 · arXiv: 2601.06883 · pdf: https://arxiv.org/pdf/2601.06883 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation
Xinhang Liu1,2
Jiawei Shi1,2
Zheng Dang3
Yuchao Dai1,2 *
School of Electronics and Information, Northwestern Polytechnical University1
Shaanxi Key Laboratory of Information Acquisition and Processing2
CVLab, EPFL, Switzerland3
{xinhangliu, sjw2018}@mail.nwpu.edu.cn, zheng.dang@epfl.ch, daiyuchao@nwpu.edu.cn
Abstract
We present MixRI, a lightweight network that solves the
CAD-based novel object pose estimation problem in RGB
images. It can be instantly applied to a novel object at test
time without finetuning. We design our network to meet the
demands of real-world applications, emphasizing reduced
memory requirements and fast inference time. Unlike ex-
isting works that utilize many reference images and have
large network parameters, we directly match points based
on the multi-view information between the query and ref-
erence images with a lightweight network. Thanks to our
reference image fusion strategy, we significantly decrease
the number of reference images, thus decreasing the time
needed to process these images and the memory required to
store them. Furthermore, with our lightweight network, our
method requires less inference time. Though with fewer refer-
ence images, experiments on seven core datasets in the BOP
challenge show that our method achieves comparable results
with other methods that require more reference images and
larger network parameters1.
1. Introduction
Six degrees-of-freedom (6DoF) object pose estimation pre-
dicts the orientation and location of a target object in 3D
space. This is crucial for embodied AI, as intelligent agents
must comprehend and interact with their environments to
perform tasks such as robotic manipulation and augmented
reality applications [13, 55]. In recent years, 6DoF object
pose estimation accuracy has been significantly improved
with deep learning [3, 29, 37, 38, 42, 44, 47, 64, 66, 70,
77, 85, 88, 89]. However, these methods require the same
object during training and test time and rely on generating
abundant synthetic data for each object during the train-
ing stage, which is tedious to deploy in practical situations.
* Yuchao Dai is the corresponding author.
This work was supported in part by the National Natural Science Foundation
of China under Grants 62271410 and 12150007.
1project page: https://npucvr.github.io/MixRI/
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
20.0
22.5
25.0
27.5
30.0
32.5
35.0
37.5
40.0
MixRI(12)
MixRI(24)
FoundPose
GigaPose
MegaPose
Genflow
12
798
参
考
图
像
数
量
AR Score
The number of reference images
Speed (Num Frames / Second)
small-parameter
large-parameter
Figure 1. Comparison of different methods. The area of each
bubble is proportional to the size of the network parameters, and
the color indicates the number of reference images used. The
detection[59] stage was removed in all speed evaluations. MixRI
achieves competitive results while using fewer reference images, a
smaller network, and providing shorter inference time.
To satisfy the practical need in industrial and daily life,
CAD-based novel object pose estimation (During the train-
ing stage, the specific object to be inferred at test time is
unknown.) was introduced [33] and more and more meth-
ods [1, 5, 10, 34, 39, 57, 58, 60, 62, 65, 69, 80, 84, 90] arise.
To enable methods to adapt to different objects with-
out retraining, recent works utilize reference images. These
images are typically rendered offline and stored in mem-
ory [1, 5, 10, 39, 57, 60, 62]. For edge AI and efficient
deployment on edge devices in practical embodied AI appli-
cations, a compact memory cache for both reference images
and network parameters is essential, along with the need
for fast inference time [41]. However, as shown in Fig. 1,
although recent works have successfully pursued high pose
estimation accuracy, they do so at the cost of requiring ex-
tensive reference images, large network parameters, and
slow inference time. As shown in Fig. 2 (a), existing meth-
1
arXiv:2601.06883v1  [cs.CV]  11 Jan 2026

Query Image
Reference Images Bank
Stage 1: 
View Selection
Render
Stage 2: 
Pose Estimation
Reference Images
Occluded
Not Occluded
Query Image
Step 1: Find Corresponding Points 
in All Reference Images 
Step 2: Find Corresponding Points on Query Image
Step 3: Solve Pose
(a): The Pipeline of Previous Methods  
(b): The Overview of Our Pipeline  
Figure 2. Comparison between our pipeline and previous methods. Unlike existing two-stage methods [1, 39, 60, 62] that first retrieve
the closest reference image from abundant reference images, we directly predict the location of sampled 3D object points on the query image
from their projections on all reference images. It constructs the 2D-3D correspondence and can solve the 6DoF pose. Hollow points indicate
invisibility, while solid points indicate visibility.
ods [1, 39, 57, 60, 62, 69] usually divide the pose estimation
into two stages: a view selection stage to retrieve the closest
reference image and a pose estimation stage to estimate the
6DoF pose of the query image by comparing it with the se-
lected reference image. As the number of reference images
for each object increases, this restricts the algorithm’s ability
to scale to a larger number of objects, as each object requires
rendering a substantial number of reference images to be
stored in memory [62]. Besides, these works mainly rely
on networks with a large number of parameters. The large
number of rendered reference images and extensive network
parameters present a significant challenge for devices with
limited computing power and memory. Additionally, some
previous works [60, 62] require the pre-extraction of image
features, which are stored in memory as well. These limita-
tions restrict their practical application in edge AI and limit
the number of unseen objects whose poses can be estimated,
as algorithms typically run in real time on edge devices with
limited memory.
In this paper, we aim to solve the novel pose estimation
problem using a lightweight network with a minimal number
of reference images, designed to support edge AI [41]. Addi-
tionally, our approach eliminates the need for pre-extraction
of features. Inspired by previous work that utilizes match-
ing to compute the pose [1, 60, 62], we also try to build
our method in a matching framework. However, when the
number of reference images is significantly reduced, the
closest reference image may still suffer from wide rotation
with the query image, known as a wide baseline in matching,
which is challenging to address [35]. To effectively solve the
matching problem, we design our method from the following
observation: 1) Given that multi-view geometry can provide
more information [24], we can use multi-view information
to enhance the matching procedures. 2) When the reference
images are few, the views become sparse, and occlusion be-
comes more common. This makes matching more necessary
to integrate multi-view information and handle occlusion
scenarios. Based on the above two observations, we design
our network by aggregating all the reference image features
and propose MixRI (Mix Reference Images), which is a
lightweight network and requires only few reference images
as input to solve the novel object pose estimation.
As shown in Fig. 2 (b), we gather all the 2D informa-
tion belonging to the same 3D object points in all reference
images and mix their features. We train a lightweight net-
work to fully fuse the feature of all reference images and
enhance the query image feature with the reference images,
which makes the matching more stable and accurate. We
also design the occlusion detection as part of the network’s
output, making our network handle the occlusion automat-
ically. In contrast to the usual feature matching problem
[15, 71, 86, 87] between paired images, our method requires
establishing correspondences from multiple reference im-
ages to a single query image. Specifically, the objective is to
determine the position and occlusion flag of each 3D object
point on the query image, based on its projections on all the
reference images. Upon establishing these correspondences,
the corresponding 2D points for the 3D object points are
then derived. With a sufficient number of 2D-3D correspon-
dences, the object pose can subsequently be computed using
the PnP algorithm [73] within the RANSAC [22] framework.
Our experiments demonstrate that, under the setting with-
out refinement, our method achieves comparable results de-
spite utilizing 33× fewer reference images than the most
accurate method [62] and being 2× faster than the fastest
method [60]. This suggests that a heavy reliance on a large
2

number of reference images and large network parameters
may not be the only way for achieving high performance.
In summary, our contributions can be summarized as:
• We present MixRI, a lightweight network for RGB-based
novel object pose estimation. It requires only 12 reference
images and does not necessitate the offline pre-extraction
of image features.
• We propose a View-Aggregated Point Matching module
that can find correspondences on the query image based
on multiple reference images simultaneously and give
occlusion prediction.
• MixRI achieves comparable performance to state-of-the-
art approaches while using significantly fewer reference
images, significantly fewer network parameters, shorter
preparation time for reference images, and faster inference
speed, making it more suitable for practical applications.
2. Related Work
Pose Estimation of Known Objects or Categories. 6DoF
object pose estimation has been widely studied for decades
as a fundamental vision problem [50, 51], solved from the
traditional methods [12, 19, 50, 51] to deep learning meth-
ods [29, 44, 47, 64, 66, 70, 82, 85, 89]. Early deep learning-
based methods for pose estimation focused on the seen
object pose estimation problem either based on template
matching [37] or feature matching followed by PnP [3, 9]
for 2D-3D correspondences [3, 44, 64, 66, 70, 89] or least
squares fitting for 3D-3D correspondences [7, 28, 29, 47, 77].
To compensate for the expensive retraining required for
a new object in seen object scenario, one approach in-
volves making the trained network generalize to known cate-
gories [11, 40, 43, 46, 54, 74, 78]. While these approaches
can estimate unseen objects, they assume all objects belong
to the same category, which still limits their application. In
contrast, our work can be generalized to arbitrary objects.
Pose Estimation of Unseen Categories. Recently, some
works have focused on novel object pose estimation, where
the target category is not seen during the training stage.
This can be addressed when an object model is avail-
able [1, 5, 10, 34, 39, 57, 58, 60, 60, 62, 65, 69, 80, 84, 90]
or with reference images [4, 27, 30, 49, 63, 72, 83, 84].
These works either use RGB images [1, 27, 39, 49, 57,
60, 63, 69, 72, 80, 91] or RGB-D images [5, 10, 34, 39,
45, 57, 61, 69, 84] as input. For scenes with available ob-
ject models, methods can be divided into feature-matching
methods [1, 5, 10, 34, 60, 62, 65] and template matching
methods [1, 39, 57, 58, 60, 62, 69, 80, 84, 90]. However,
for those feature-matching methods, they primarily use a
retrieval-based strategy, which actually applies a template
matching method to get the closest reference image and build
the correspondence between the closest reference image and
the query image [1, 10, 57, 60, 62]. Unlike those works, our
method is pure feature-matching. We directly build the cor-
respondences across all the reference images and the query
image, which significantly reduce the number of reference
images needed. This reduces the time needed to render refer-
ence images and avoids caching pre-computed features for
view selection, saving substantial memory.
Local Feature Matching & Point Tracking. Local fea-
ture matching establishes precise correspondences between
images, typically in paired image scenarios. Early works
use a detector-based approach first employing some well-
established handcrafted features such as SIFT [52] and
ORB [68] followed by a feature matching algorithm like
nearest-neighbor searches. With the help of deep learning,
more complex methods [15, 21, 48, 56, 67] arise. Recently,
more works [8, 71, 86, 87] are detector-free methods, and
they can efficiently handle extreme circumstances such as
texture-less regions or substantial viewpoint changes. Our
work focuses on feature matching, but we build the corre-
spondences between the query image and multiple reference
images of the same 3D object points, which differs from the
matching between two images. Those methods struggle to
handle occlusion and massive changes in viewpoint, which
can not predict the occlusion information as well.
Besides, some works focus on tracking points across con-
tinuous frames [16, 23, 53, 76, 79]. In those works, adjacent
frames, along with the initial points to be tracked, are input
into the network. The network will then output the points’
trajectories across the sequence, along with occlusion in-
formation. As with works in point tracking, we also build
the correspondence along all the images and output the oc-
clusion information. However, in our method, the reference
images are unordered and not consecutive. Furthermore, we
have the locations of the query 3D object points on all ref-
erence images, which are the projections of the same 3D
object points onto each reference image, and some can be
occluded. While in point tracking work, query points usually
belong to a single frame and must be visible.
3. Method
Our method is built on finding the correspondences between
the 2D pixel locations and 3D object point locations, fol-
lowed by a RANSAC-based PnP framework to compute the
6DoF pose. We break down the pose estimation problem
to a matching problem. However, different from the previ-
ous matching-based pose estimation, we neither directly find
2D-3D correspondences between the query image and 3D ob-
ject points [27, 64, 72, 89] nor find 2D-2D correspondences
between paired images as is done in classic local feature
matching [1, 60, 71, 87]. Instead, we focus on finding cor-
respondences between multiple reference images and the
query image. Given multiple projections of one 3D object
point, some of which may be occluded, we aim to find its
projection coordinate and occlusion flag on the query image.
3

Cost Volume
Query Image
Reference Images
Bilinear Sample
Dual-Attention Based Feature Mixer
Feature Tokens
3D CNN
ReLU
Occlusion Flags
Point Coordinates
oI
1I
iI
SI
Rotation Invariant Feature Extractor
D
W
H
Feature Maps
N
D
0
F
1
F
iF
S
F
1ˆF
ˆ
D
i
N 

F

ˆ
S
F
0
F
Mixed Query
Feature Maps
0
F
Fused Point
Feature Tokens
F
N
MLP
3D CNN
3D CNN
Mean Pool
0
N

O

Softmax
Heatmaps
2
0
N 

U

W
H
Figure 3. Overview of our network. Unlike existing methods [1, 39, 60, 62], which use two stages to compute the pose, we directly input all
reference images into our network without a view selection stage. After obtaining all features of the projection belonging to one 3D object
point pk, we use Dual-Attention Based Feature Mixer to fuse their features with the query image feature. Then, we build the cost volume
followed by two separate heads to predicate the projection of pk on the query image, including the occlusion flag as well as the coordinate.
3.1. Preliminary
Our task is CAD-based novel object pose estimation with
RGB images, where the primary goal is to estimate the pose
T0 of the query image I0. It is worth noting that the objects
observed in I0 used for testing are unseen in the training
dataset, i.e., Otrain ∩Otest = ∅[91]. After detecting and
segmenting the object in the query image, we can estimate
the pose of the novel object using a series of reference images
and depths rendered by the corresponding object model.
In other words, given S reference images {I1, I2, . . . , IS}
showing the same object under various viewpoints, for which
the object pose Ti = (Ri, ti) ∈SE(3), intrinsic matrix
Ki ∈R3×3 and depth Di are known, we focus on estimating
the novel object pose in the query image.
3.2. Correspondences between Reference Images
Before training and testing, to find the projections of each
3D object point across all reference images, we first sample
on each reference image i to obtain 2D pixel coordinates
ui,k ∈R2×1. Using the given ground truth pose Ti and
depth di,k, we can then recover the corresponding 3D object
point in the object model coordinate:
pk = di,kT−1
i K−1
i
˜ui,k,
(1)
where, ˜ui,k is homogeneous coordinate, 1 ≤k ≤N and
N is the total number of sampled points. Besides, we can
calculate new 2D coordinates on other reference image j by
projecting the 3D object points:
˜uj,k =
1
˜dj,k
KjTjpk,
(2)
where, ˜dj,k is the depth of point pk transformed to the j-
th camera coordinate. In this manner, we can obtain the
2D pixel coordinates {ui,k} of the N sampled points on
S reference images and their 3D object coordinates {pk}.
However, since each point may be occluded by the object
itself or other objects, we set the occlusion flag Oi,k ∈{0, 1}
for each ui,k. Similar to the Z-buffer algorithm [6], we mark
points that are projected outside the image or occluded, i.e.,
 ˜dj,k −dj,k
 > τ, as occlusion (Oj,k = 1).
Next, we aim to compute the corresponding point on
the query image. Supposing among S reference images, for
a given 3D object point pk, So of all its projections are
occluded and Sv are visible (where Sv + So = S), one
approach is to use Sv reference images to match pairwise
with it, and finally merge all the matching results to get u0,k,
which is the projection of pk on the query image I0. How-
ever, this approach can not handle the situation when u0,k
is occluded and fails to fully utilize the multi-view informa-
tion. To address this, for a 3D object points pk, we develop
a network to directly fuse all the projections’ information
around ui,k, 1 ≤i ≤S in the feature space and locate the
corresponding 2D pixel coordinate u0,k on the query image,
as shown in Section 3.3.
3.3. View-Aggregated Point Matching
Fig. 3 shows an overview of our network architecture. The
query image, along with the reference images, are sent to the
encoder fenc together. fenc is a ResNet-like backbone [25]
sharing the same weights between the query and reference
images. Since objects in images may have various poses
4

Self-Attention between Points
Permute
Self-Attention between Frames
Permute
Mix-Attention between Reference & Query
Permute
Permute
×n0
Position Encoding
!
!
!
!
Initial
Iteration
Output
Mix Attention Block
Mix-Attention between Reference & Query
Self-Attention between Frames
×n2
id
Feature Dim
S ∗ D
0
1
i
S
×n3
Self-Attention for Learnable Point Features
Self-Attention for Target Image Features
Cross-Attention for Feature Mixing
Order of Execution
1
i
N
Dim
(
)
!
"
#
′
′
∗
∗
Dim
! "
∗
1
2
H’W’
H’W’-1
1
2
H’W’-1
H’W’
1
i
N
!"!
!
!!
!
!
!
!
!
!
Self-Attention between Points
×n1
id
Feature Dim
N ∗ D 
1
i
N
Fused Point Feature Tokens !
Output:
Fused Query Image Feature
!
!!
!
Figure 4. Overview of the Dual-Attention Based Feature Mixer. The mixer consists of three modules: SAP (Self-Attention between
Points) , SAF (Self-Attention between Frames), and MARQ (Mix-Attention between Reference & Query). These modules perform attention
operations between N points within one reference image, between S frames, and between reference images and the query image, respectively.
and the features of the matching points are independent
of the poses, we utilize [81] in the encoder to ensure that
the extracted features are rotation-invariant. After feature
extraction, we obtain S + 1 feature maps:
Fi = fenc(Ii) ∈RH′×W ′×D, 0 ≤i ≤S,
(3)
where H′ =H/8, W ′ =W/8. With pre-sampled 2D projec-
tions {ui,k}S
i=1 and occlusion flags {Oi,k}S
i=1 on reference
images, we retrieve the feature tokens {ˆFi ∈RN×D}S
i=1
using bilinear interpolation. Next, for all 3D object points
{pk}N
k=1, we gather all feature tokens in the feature space
and fuse them with the query image feature F0 using at-
tention mechanism (see Section 3.4 for details). We denote
the final gathered feature and fused query image feature
as ¯F ∈RN×D, ˜F0 ∈RH′×W ′×D, respectively. Then we
can build the cost volume C ∈RH′×W ′×N with the query
image feature ˜F0.
For the given cost volume C, we first extract the informa-
tion with a Conv3D backbone. After that, we use two sepa-
rate heads to predict the corresponding 2D heatmap of the
3D object points and the associated occlusion flags. To con-
vert the heatmap to valid 2D coordinates, we use “spatial soft
argmax” [16], which computes the argmax of the heatmap
and then estimates the spatial average position within a ra-
dius around the argmax location. Lastly, we calculate the
6DoF pose directly using RANSAC-based SQ-PnP [73] al-
gorithm. Note that we only use the 2D location where the
predicted occlusion flag is False, i.e., where O0,k ≤τocc.
3.4. Dual-Attention Based Feature Mixer
Previous methods [1, 60, 62] select the reference image clos-
est to the query image for matching. Unlike them, we fuse
the feature tokens extracted from all reference images to ob-
tain the final tokens that contain more viewpoint information.
We design the Dual-Attention module, an overview of which
is illustrated in Fig. 4. Referring to [17], we first initialize the
fused feature tokens ¯F as learnable parameters, then input
the feature tokens ˆF ∈RN×S×D of all reference images,
occlusion flags O ∈RS×N of these tokens, and the feature
map F0 of the query image.
Firstly, in order to integrate the spatial location infor-
mation of N points, we permute ˆF as ˆFN = gN(ˆF) ∈
RS×N×D. Then we input ˆFN into the SAP (Self-Attention
between Points) module for spatial dimension mixing after
adding position encoding [71]. Secondly, the features mixed
on N-dim are permuted into frame-first form, i.e., ˆFS =
gS(ˆF) ∈RN×S×D, and then sent to SAF (Self-Attention be-
tween Frames) module to perform Self-Attention calculation
with learnable fused tokens ¯F on S-dim. These permutations
enables the first dimension to be merged with the batch size
dimension, thereby accelerating computation when multiple
objects are predicted concurrently. Finally, the adjusted to-
kens ˆF are input for the next iteration. To further augment the
features, we fuse ¯F with the permuted query image feature
F0 ∈R(H′W ′)×D using MARQ (Mix-Attention between
Reference & Query) module, which consists of two Self-
Attention and two Cross-Attention layers. It performs the
Self-Attention separately on the query image feature F0 and
the learnable fused token ¯F, and Cross-Attention between
5

the two [71]. The Dual-Attention module can be written as:



ˆF = SelfN(gN(ˆF), O), for n1 iters
ˆF, ¯F = SelfS(gS(ˆF), ¯F, O), for n2 iters
¯F, F0 = Mix(¯F, F0, O), for n3 iters
,
(4)
where SelfN, SelfS, Mix denote Self-Attention on N-dim,
Self-Attention on S-dim, and Mix-Attention, respectively.
Mask O is used in masked attention mechanism, introduced
to correct the weight matrices of attention, which avoids
fusing erroneous features corresponding to occluded points.
Iterate Equation (4) for n0 times and take the obtained ¯F
and ˜F0 as the final fused results. By combining Self/Cross-
Attention, Self-Attention on N/S-dim, our Dual-Attention
module can automatically perform fusion based on the de-
gree of similarity between point-to-point, frame-to-frame
and reference-to-query during feature mixing. This mod-
ule is crucial to accomplishing the view-aggregated point
matching task, as we will show in Section 4.4.
3.5. Training Losses
The total loss for each 3D object point consists of two com-
ponents: occlusion supervision and location supervision. The
occlusion loss is supervised using a BCE loss:
Locc = BCE(Ogt,k, O0,k),
(5)
where O0,k is network output for the kth 3D object point
and Ogt,k is the corresponding ground truth occlusion flag.
For location supervision, we use the Huber loss to regress
the 2D coordinates of the projection of the 3D object points
when the occlusion flag is False:
Lloc = Huber(Ugt,k, U0,k) · 1{Ogt,k = 0},
(6)
where Ugt,k and U0,k is the ground truth 2D coordinate and
predicted 2D coordinate for kth 3D object point. 1{·} is the
indicator function. The total loss is a weighted sum of the
occlusion and location losses:
L = Locc + λLloc,
(7)
with λ = 100. The final loss is the mean of the losses across
N sampled points.
4. Experiments
In this section, we first present the experiment setup. Then,
we compare our method with the state-of-the-art methods
for novel object pose estimation on seven core datasets of
the BOP challenge [33]. We also provide further ablation
studies to evaluate our method.
4.1. Datasets
We train our network entirely with synthetic images from the
GSO-Dataset [39]. Following previous work [39, 57, 60, 62],
we evaluate our method on seven core BOP datasets [33], in-
cluding LM-O [2], YCB-V [85], T-LESS [31], TUD-L [32],
IC-BIN [18], HomebrewedDB(HB) [36] and ITODD [20].
Detailed information is explained in the supplementary mate-
rial. We follow the official test splits of each dataset and use
the off-the-shelf object detector CNOS [59]. In practical situ-
ations, this detector can be replaced by any other lightweight
tracking method or an object-specific segmentation network.
4.2. Evaluation Metrics
For all experiments, we use the standard evaluation proto-
col in the BOP challenge [32], including Visible Surface
Discrepancy (VSD), Maximum Symmetry-Aware Surface
Distance (MSSD), and Maximum Symmetry-Aware Pro-
jection Distance (MSPD). The final average recall (AR) is
calculated by averaging the individual average recall scores
of these three metrics across a range of error thresholds. For
a more detailed explanation, refer to the BOP challenge [33].
4.3. Comparison with the State of the Art
We compare our method with OSOP [69], MegaPose [39],
ZS6D [1], GigaPose [60], Genflow [57] and FoundPose [62]
as shown in Table 1. In a setting without refinement, our
method achieves promising results. Specifically, compared
to FoundPose, our method achieves similar performance
while utilizing 33× fewer reference images, with fewer net-
work parameters and shorter inference time. When compared
to GigaPose, which uses the second fewest reference images,
our method improves accuracy by approximately 10% on the
challenging IC-BIN and HB datasets, and by around 25% on
the challenging YCB-V dataset. We also achieve comparable
results on LM-O, T-LESS, and TUD-L. However, the perfor-
mance on ITODD is not as good as GigaPose. This might
be because it is a dataset with grayscale images, whereas
our training is based on RGB images. In addition, it has se-
vere occlusions, reflections, and surfaces with weak textures,
which increases the difficulty of matching. However, we still
improve the mean AR of all seven core datasets by 6.5%
compared with GigaPose. When it comes to MegaPose, our
method achieves better results across all seven core datasets
and improves around 15% on average. Genflow utilizes a net-
work similar to MegaPose, and our method achieves better
results on most datasets, except for ITODD, where perfor-
mance is impacted due to differences in color modes. ZS6D
is a method that first retrieves the closest reference image
from 300 reference images and matches the query image
with the closest one. However, our method uses far fewer ref-
erence images but achieves better results. This demonstrates
that our multi-view fusion strategy can improve matching
accuracy and increase the AR score. Fig. 5 shows qualitative
results, which illustrate the accurate pose estimation results.
More qualitative results and comparisons with other methods
6

Method
Parm size Ref Num
Detection
LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V MEAN TIME
OSOP [69]
-
90 k
OSOP [69]
27.4
-
-
-
-
46.4
29.6
-
-
MegaPose [39]
21.6 M
520
Mask R-CNN [26]
18.7
19.7
20.5
15.3
8.0
18.6
13.9
16.4
-
ZS6D [1]
21.7 M
300
CNOS [59]
29.8
21.0
-
-
-
-
32.4
-
-
MegaPose [39]
21.6 M
520
CNOS [59]
22.9
17.7
25.8
15.2
10.8
25.1
28.1
20.8
15.5 s
GigaPose [60]
316.3 M
162
CNOS [59]
29.9
27.3
30.2
23.1
18.8
34.8
29.0
27.6
0.8 s
Genflow [57]
21.7 M
208
CNOS [59]
25.0
21.5
30.0
16.8
15.4
28.3
27.7
23.5
3.8 s1
FoundPose [62] 302.9 M
798
CNOS [59]
39.6
33.8
46.7
23.9
20.4
50.8
45.2
37.2
1.6 s
MixRI (ours)
5.3 M
12
CNOS [59]
27.0
25.4
29.3
29.7
10.9
44.9
52.8
31.4
0.5 s
MixRI (ours)
5.3 M
24
CNOS [59]
30.4
27.4
33.6
30.8
11.6
50.2
54.6
34.1
0.7 s
Table 1. Results on the seven core BOP datasets. The table compares methods in Average Recall (AR), network parameter size, reference
image count, and inference time. Bold denotes the best, and underline the second best.
Ref Nums
Method
YCB-V LM-O TUD-L MEAN
4
GigaPose [60]
8.1
10
9.1
9.1
MixRI (ours)
28.6
11.2
13.8
17.9
6
GigaPose [60]
7.9
12
13.1
11.0
MixRI (ours)
46.5
22.1
25.7
31.4
8
GigaPose [60]
15.4
15.6
12.3
14.4
MixRI (ours)
51.8
26.2
28.5
35.5
12
GigaPose [60]
14.9
13.9
15.3
14.7
MixRI (ours)
52.8
27.0
29.3
36.4
24
GigaPose [60]
15.9
18.2
20.8
18.3
MixRI (ours)
54.6
30.4
33.6
39.5
Table 2. Comparison with a limited number of reference images.
We compare our method with GigaPose in different number of
reference images.
are provided in the supplementary material.
Limited Number of Reference Images. Since GigaPose
uses the fewest reference images among existing works,
we compare it when using only a limited number of refer-
ence images. We conduct experiments with various settings
by changing the reference image number S and report the
results in Table 2. GigaPose experiences a significant per-
formance decrease when there are fewer reference images.
Because it’s a two-stage approach, which highly relies on
the closest reference image. When there are few reference
images, selecting a reference image with similar viewpoint
becomes difficult, and it is also challenging to provide suf-
ficient information for estimating the pose in their second
stage. It is worth noting that other methods [1, 39, 57, 62],
which are also based on the retrieval mechanism, can en-
counter the similar issue when using a limited number of
reference images. In contrast, our method is designed specif-
ically for limited reference images and achieve a better AR
score when using the same number of reference images.
Run-time & Memory. We report the average inference tim-
ings for each method in Table 1, measured using a single
4090 GPU. It is worth noting that the detection time included
in CNOS [59] is also taken into account, and each image
may contain multiple objects for pose estimation. In addition
1Since GenFlow’s code is not open-source, the reported time here is taken
from its public report in BOP, using a V100 GPU.
Figure 5. Qualitative results on YCB-V. We present the pose esti-
mation results obtained using MixRI. All the results are visualized
in error heatmap [75] which darker blue indicates lower error with
respect to the ground truth pose (legend: 0 cm
5 cm).
to reducing inference time, our method significantly saves
time in rendering the reference images and does not require
pre-extraction of features. However, GigaPose and Found-
Pose need pre-extracting all the reference image features
[60, 62], which also waste memory due to the need to cache
them. Compared to MegaPose, which also does not require
pre-extracting features, our method outperforms it, as we
eliminate the need for image rendering during inference [39].
We also report the network parameters compared with
others in Table 1. The parameter count of our network is
significantly smaller than that of all other methods. In par-
ticular, we have over 50× fewer parameters than GigaPose.
Combined with fewer reference images and no feature cache
needed, MixRI is very friendly for deployment on devices
with limited memory space. More discussions are shown in
A.7 in the supplementary material.
4.4. Ablation Study
Effectiveness of the Attention Module. Our main innova-
tion is to fuse the multiple reference points in the feature
SAP
MARQ
YCB-V
LM-O
TUD-L
MEAN
✗
✗
2.9
2.8
2.2
2.6
✓
✗
19.2
7.8
8.7
11.9
✗
✓
33.2
13.4
21.9
22.8
✓
✓
54.6
30.4
33.6
39.5
Table 3. Effectiveness of the attention module. We analyze the
impact of our different modules. Here SAP stands for Self-Attention
between Points in Section 3.4 and MARQ stands for Mix-Attention
between Reference & Query in Section 3.4.
7

Method
Ref Num Detection
Refinement
LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V MEAN TIME
MixRI+ (ours)
162
CNOS [59]
-
42.6
36.0
41.8
33.9
22.2
56.0
56.9
41.3
1.0 s
MegaPose [39]
520
CNOS [59] MegaPose [39]
49.9
47.7
65.3
36.7
31.5
65.4
60.1
50.9
17.0 s
GigaPose [60]
162
CNOS [59] MegaPose [39]
55.6
54.6
57.8
44.3
37.8
69.3
63.4
54.7
2.3 s
FoundPose [62]
798
CNOS [59] MegaPose [39]
55.4
51.0
63.3
43.0
34.6
69.5
66.1
54.7
4.4 s
MixRI (ours)
12
CNOS [59] MegaPose [39]
40.8
45.0
44.6
44.4
21.6
61.4
61.1
45.6
1.5 s
MixRI (ours)
24
CNOS [59] MegaPose [39]
44.8
46.1
48.2
44.6
21.3
61.4
62.0
46.9
1.7 s
MixRI+ (ours)
162
CNOS [59] MegaPose [39]
52.1
50.4
57.5
44.5
34.3
67.1
64.8
53.0
2.1 s
Table 4. Results on the seven core BOP datasets with refinement. The table shows Average Recall (AR) scores per dataset, the time and
the number of reference images. We also present MixRI+, the same structure but trained for more reference images available.
10
20
30
40
50
60
4
6
8
12
24
60
120
AR Score
(b) Number of Reference Images
LM-O
TUD-L
YCB-V
MEAN
10
20
30
40
50
60
60
120
240
480
1200
2400
6000
12000
AR Score
(a) Number of Correspondences
LM-O
TUD-L
YCB-V
MEAN
Figure 6. Impact of the number of correspondences and ref-
erence images. We report the AR score on LM-O, TUD-L, and
YCB-V, along with the average.
space. To demonstrate the effectiveness of the fusing strategy,
we conduct several ablation studies. As shown in Table 3,
through row 1 and row 2, adding SAP brings a 9.3% increase
in AR score. The considerable increase comes with MARQ,
which results in a 20.2% increase in AR score. Finally, with
both modules, we get the final AR score shown in row 4,
which has a 36.9% increase compared to row 1.
Number of Correspondences. Fig. 6 (a) shows the results
of different correspondences used in matching. Our method
is robust to correspondence, and increasing the correspon-
dences can increase the AR score until around 2400. It is
worth mentioning that, although there are only 60 correspon-
dences, for the YCB-V dataset, there is already an AR score
of 45%. When the number of correspondences reaches 240,
it is already comparable to other methods.
Number of Reference Images. Fig. 6 (b) ablates the num-
ber of reference images. Our network is trained using 12
reference images, but it does not require the same number
of reference images during inference. Therefore, we only
alter the number of reference images during the test stage.
However, we observe a performance degradation when the
number of reference images increases significantly, partic-
ularly when it exceeds 60. This is expected because our
lightweight network is less capable of handling the feature
fusion from a large number of reference images. As previ-
ous work [39, 60, 62], more reference images need a larger
network with more parameters to handle.
Effectiveness of Occlusion Flags. Table 5 demonstrates
the effectiveness of incorporating occlusion flags. It is worth
noting that when τocc = 1, the occlusion flag mechanism is
τocc LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V MEAN
1.0
27.6
24.1
30.9
28.5
10.4
44.1
49.4
30.7
0.8
30.4
27.4
33.6
30.8
11.6
50.2
54.6
34.1
0.5
27.8
26.7
28.4
28.6
10.2
46.8
53.8
31.8
Table 5. The effectiveness of occlusion flags. We compare different
occlusion flag settings.
effectively disabled. The results validate the robustness of
the method when occlusion prediction is enabled.
Refinement. While refinement can lead to improved pose ac-
curacy, it does so at the expense of speed, which goes against
our initial intention. However, to demonstrate the potential
of our lightweight network, we also present the results after
refinement in Table 4. For this, we use MegaPose [39] to
further refine our initial pose estimates. Additionally, we
include another MixRI variant (MixRI+) having the same
structure but trained for more reference images available.
To optimize the use of reference images, we follow the ap-
proach in [60], which retrieves the reference image with the
closest out-of-plane rotation. However, in our method, we
also select three additional reference images as input to the
network. Detailed information on this process is provided in
the supplementary material. Notably, with more reference
images available, the initial poses achieve an AR score of
41.3, surpassing FoundPose which is 37.2. Additionally, the
refinement performance is comparable, even we use fewer
network parameters and achieve faster processing speed.
5. Conclusion
In this paper, we proposed MixRI, a novel approach for
CAD-based novel object pose estimation using RGB images.
Designed with practical applications in mind, our method
features a lightweight network and requires far fewer refer-
ence images than existing approaches, eliminating the need
for feature caching. This reduces both pose inference time
and reference image preparation time while avoiding pre-
processing. Despite its simplicity, MixRI achieves perfor-
mance comparable with more complex networks that rely
on numerous reference images. In the future, we will further
improve the matching accuracy and extend our framework
to scenarios where object models are unavailable.
8

References
[1] Philipp Ausserlechner, David Haberger, Stefan Thalhammer,
Jean-Baptiste Weibel, and Markus Vincze. Zs6d: Zero-shot 6d
object pose estimation using vision transformers. In IEEE In-
ternational Conference on Robotics and Automation (ICRA),
pages 463–469. IEEE, 2024. 1, 2, 3, 4, 5, 6, 7
[2] Eric Brachmann, Alexander Krull, Frank Michel, Stefan
Gumhold, Jamie Shotton, and Carsten Rother. Learning 6d
object pose estimation using 3d object coordinates. In Eur.
Conf. Comput. Vis., pages 536–551, 2014. 6, 15, 18
[3] Yannick Bukschat and Marcus Vetter. Efficientpose: An effi-
cient, accurate and scalable end-to-end 6d multi object pose
estimation approach. arXiv preprint arXiv:2011.04307, 2020.
1, 3
[4] Dingding Cai, Janne Heikkil¨a, and Esa Rahtu. Gs-pose: Cas-
caded framework for generalizable segmentation-based 6d
object pose estimation. arXiv preprint arXiv:2403.10683,
2024. 3
[5] Andrea Caraffa, Davide Boscaini, Amir Hamza, and Fabio
Poiesi. Freeze: Training-free zero-shot 6d pose estimation
with geometric and vision foundation models. In Eur. Conf.
Comput. Vis., pages 414–431, 2024. 1, 3
[6] Edwin Catmull.
A hidden-surface algorithm with anti-
aliasing. ACM SIGGRAPH Computer Graphics, 12(3):6–11,
1978. 4
[7] Dengsheng Chen, Jun Li, Zheng Wang, and Kai Xu. Learning
canonical shape space for category-level 6d object pose and
size estimation. In IEEE Conf. Comput. Vis. Pattern Recog.,
pages 11973–11982, 2020. 3
[8] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Mingmin
Zhen, Tian Fang, David Mckinnon, Yanghai Tsin, and Long
Quan. Aspanformer: Detector-free image matching with adap-
tive span transformer. In Eur. Conf. Comput. Vis., pages 20–36,
2022. 3
[9] Hansheng Chen, Pichao Wang, Fan Wang, Wei Tian, Lu
Xiong, and Hao Li. Epro-pnp: Generalized end-to-end proba-
bilistic perspective-n-points for monocular object pose esti-
mation. In IEEE Conf. Comput. Vis. Pattern Recog., pages
2781–2790, 2022. 3
[10] Jianqiu Chen, Mingshan Sun, Tianpeng Bao, Rui Zhao, Liwei
Wu, and Zhenyu He. Zeropose: Cad-model-based zero-shot
pose estimation. arXiv preprint arXiv:2305.17934, 2023. 1, 3
[11] Yamei Chen, Yan Di, Guangyao Zhai, Fabian Manhardt,
Chenyangguang Zhang, Ruida Zhang, Federico Tombari,
Nassir Navab, and Benjamin Busam. Secondpose: Se (3)-
consistent dual-stream feature fusion for cate