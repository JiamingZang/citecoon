# SMAP: Single-Shot Multi-Person Absolute 3D Pose Estimation

> 2020 · id: arxiv:2008.11469 · arXiv: 2008.11469 · pdf: https://arxiv.org/pdf/2008.11469 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Recent years have witnessed an increasing trend of research on monocular 3D
human pose estimation because of its wide applications in augmented reality,
human-computer interaction, and video analysis. This paper aims to address
the problem of estimating absolute 3D poses of multiple people simultaneously
from a single RGB image. Compared to the single-person 3D pose estimation
problem that focuses on recovering the root-relative pose, i.e., the 3D locations
of human-body keypoints relative to the root of the skeleton, the task addressed
here additionally needs to recover the 3D translation of each person in the camera
coordinate system.
While there has been remarkable progress in recovering the root-relative 3D
pose of a single person from an image [36,18,12,8], it was not until recently
that more attention was paid to the multi-person case. Most existing methods
⋆Equal contribution.
arXiv:2008.11469v1  [cs.CV]  26 Aug 2020

2
Zhen et al.
Fig. 1. We propose a novel framework named SMAP to estimate absolute 3D poses of
multiple people from a single RGB image. The ﬁgure visualizes the result of SMAP on
an in-the-wild image. The proposed single-shot and bottom-up design allows SMAP to
leverage the entire image to infer the absolute locations of multiple people consistently,
especially in terms of the ordinal depth relations.
for multi-person 3D pose estimation extend the single-person approach with a
separate stage to recover the absolute position of each detected person separately.
They either use another neural network to regress the 3D translation of the
person from the cropped image [22] or compute it based on the prior about the
body size [6,41,42], which ignore the global context of the whole image. Another
line of work tries to recover body positions with a ground plane constraint [19],
but this approach assumes that the feet are visible, which is not always true,
and accurate estimation of ground plane geometry from a single image is still an
open problem.
We argue that the robust estimation of global positions of human bodies
requires to aggregate the depth-related cues over the whole image, such as the 2D
sizes of human bodies, the occlusion between them, and the layout of the scene.
Recent advances in monocular depth estimation have shown that convolutional
neural networks (CNNs) are able to predict the depth map from an RGB image
[16,13], which is particularly successful on human images [15]. This observation
motivates us to directly learn the depths of human bodies from the input image
instead of recovering them in a post-processing stage.
To this end, we propose a novel single-shot bottom-up approach to multi-
person 3D pose estimation, which predicts absolute 3D positions and poses of
multiple people in a single forward pass. We regress the root depths of human
bodies in the form of a novel root depth map, which only requires 3D pose

SMAP
3
annotations as supervision. We train a fully convolutional network to regress the
root depth map, as well as 2D keypoint heatmaps, part aﬃnity ﬁelds (PAFs),
and part relative-depth maps that encode the relative depth between two joints
of each body part. Then, the detected 2D keypoints are grouped into individuals
based on PAFs using a part association algorithm, and absolute 3D poses are
recovered with the root depth map and part relative-depth maps. The whole
pipeline is illustrated in Fig. 2.
We also show that predicting depths of human bodies is beneﬁcial for the
part association and 2D pose estimation. We observe that many association er-
rors occur when two human bodies overlap in the image. Knowing the depths
of them allows us to reason about the occlusion between them when assigning
the detected keypoints. Moreover, from the estimated depth, we can infer the
spatial extent of each person in 2D and avoid linking two keypoints with an un-
reasonable distance. With these considerations, we propose a novel depth-aware
part association algorithm and experimentally demonstrate its eﬀectiveness.
To summarize, the contributions of this work are:
– A single-shot bottom-up framework for multi-person 3D pose estimation,
which can reliably estimate absolute positions of multiple people by leverag-
ing depth-relevant cues over the entire image.
– A depth-aware part association algorithm to reason about inter-person oc-
clusion and bone-length constraints based on predicted body depths, which
also beneﬁts 2D pose estimation.
– The state-of-the-art performance on public benchmarks, with the general-
ization to in-the-wild images and the ﬂexibility for both whole-body and
half-body pose estimation. The code, demonstration videos and other sup-
plementary material are available at https://zju3dv.github.io/SMAP.
2

## method
Fig. 2 presents the pipeline of our approach, which consists of a single-shot
bottom-up framework named SMAP. With a single RGB image as input, SMAP
outputs 2D representations including keypoint heatmaps and part aﬃnity ﬁelds
[3]. Additionally, it also regresses 2.5D representations including root depth map
and part relative-depth maps, which encode depth information of human bodies.
Then, a depth-aware part association algorithm is proposed to assign detected
2D keypoints to individuals, depending on an ordinal prior and an adaptive

SMAP
5
…
Part Relative-Depth Maps
Root Depth Map
Heatmaps and PAFs
Shared Feature
Task-Specific Feature
Multi-Stage
Task-Specific Convolution
RefineNet
…
Supervision
Depth-Aware 
Part Association
(b)
(a)
(b)
Fig. 2. Overview of the proposed approach. Given a single image, our single-shot
network SMAP regresses several intermediate representations including 2D keypoint
heatmaps, part aﬃnity ﬁelds (PAFs), root depth map, and part relative-depth maps
(Red means the child joint has a larger depth than its parent joint, and blue means the
opposite). With a new depth-aware part association algorithm, body parts belonging
to the same person are linked. With all these intermediate representations combined,
absolute 3D poses of all people can be recovered. Finally, an optional ReﬁneNet can
be used to further reﬁne the recovered 3D poses and complete invisible keypoints.
bone-length constraint. Based on these results, the absolute 3D pose of each
person can be reconstructed with a camera model. Individual modules of our
system are introduced below.
3.1
Intermediate representations
Given the input image, SMAP regresses the following intermediate representa-
tions, based on which 3D poses will be reconstructed:
Root depth map. As the number of people in an input image is unknown,
we propose to represent the absolute depths for all human bodies in the image
by a novel root depth map. The root depth map has the same size as the input
image. The map values at the 2D locations of root joints of skeletons equal
their absolute depths. An example is shown in Fig. 2. In this way, we are able
to represent the depths of multiple people without predeﬁning the number of
people. During training, we only supervise the values of root locations. The

6
Zhen et al.
proposed root depth map can be learned together with other cues within the
same network as shown in Fig. 2 and only requires 3D poses (instead of full
depth maps) as supervision, making our algorithm very eﬃcient in terms of
both model complexity and training data.
It is worth noting that visual perception of object scale and depth depends
on the size of ﬁeld of view (FoV), i.e., the ratio between the image size and the
focal length. If two images are obtained with diﬀerent FoVs, the same person at
the same depth will occupy diﬀerent proportions in these two images and seem
to have diﬀerent depths for the neural network, which may mislead the learning
of depth. Thus, we normalize the root depth by the size of FoV as follows:
eZ = Z w
f ,
(1)
where eZ is the normalized depth, Z is the original depth, and f and w are the
focal length and the image width both in pixels, respectively. So w/f is irrelevant
to image resolution, but equals to the ratio between the physical size of image
sensor and the focal length both in millimeters, i.e., FoV. The normalized depth
values can be converted back to metric values in inference.
Keypoint heatmaps. Each keypoint heatmap indicates the probable locations
of a speciﬁc type of keypoints for all people in the image. Gaussian distribution
is used to model uncertainties at the corresponding location.
Part aﬃnity ﬁelds (PAFs). PAFs proposed in [3] include a set of 2D vector
ﬁelds. Each vector ﬁeld corresponds to a type of body part where the vector at
each pixel represents the 2D orientation of the corresponding body part.
Part relative-depth map. Besides the root depth, we also need depth values
of other keypoints to reconstruct a 3D pose. Instead of predicting absolute depth
values, for other keypoints we only regress their relative depths compared to their
parent nodes in the skeleton, which are represented by part relative-depth maps.
Similar to PAFs, each part relative-depth map corresponds to a type of body
part, and every pixel that belongs to a body part encodes the relative depth
between two joints of the corresponding body part. This dense representation
provides rich information to reconstruct a 3D skeleton even if some keypoints
are invisible.
Network architecture. We use Hourglass [24] as our backbone and modify it
to a multi-task structure with multiple branches that simultaneously output the
above representations as illustrated in Fig. 2. Suppose the number of predeﬁned
joints is J. Then, there are 4J −2 channels in total (heatmaps and PAFs: J +
2(J −1), root depth map: 1, part relative-depth map: J −1). Each output
branch in our network only consists of two convolutional layers. Inspired by [14],
we adopt multi-scale intermediate supervision. The L1 loss is used to supervise
the root depths and L2 losses on other outputs. The eﬀect of the network size
and the multi-scale supervision will be validated in the experiments.

SMAP
7
pelvis
neck
LKnee
LAnkle
(a)
(b)
(c)
(d)
Fig. 3.
Depth-aware part association. From left to right: candidate links, part
aﬃnity ﬁelds, pose estimation results from [3], and our results with depth-aware part
association. The example in the ﬁrst row shows the eﬀect of ordinal prior. Under this
circumstance, [3] assigns the pelvis to the occluded person while we give priority to
the front person. The example in the second row shows the eﬀect of the adaptive bone-
length threshold, indicated by the green circle. As a noisy response occurs at the right
ankle of the person it doesn’t belong to, [3] will induce a false connection while our
algorithm will not.
3.2
Depth-aware part association
Given 2D coordinates of keypoints from keypoint heatmaps after non-maximum
suppression, we need to associate detected joints with corresponding individuals.
Cao et al. [3] propose to link joints greedily based on the association scores
given by PAFs. Basically, we follow their method to calculate association scores.
However, PAFs scores might be unreliable due to occlusion between people.
Fig. 3 shows two examples where the above association strategy fails.
We propose to leverage the estimated depth maps to address the part asso-
ciation ambiguities caused by inter-person occlusion.
Ordinal prior. A key insight to solve the occlusion issue is to give priority to the
unoccluded person when assigning joints. The occlusion status can be inferred
from the depth map. Therefore, we sort root joints from near to far according
to the predicted root depth, instead of following the order of PAFs scores. Our
association process starts with the root and proceeds along the skeleton tree
successively.
Adaptive bone-length constraint. To avoid linking two keypoints with an
unreasonable distance, Cao et al. [3] constrain the association with a threshold
deﬁned as half of the image size. However, such a ﬁxed threshold is ineﬀective
as the 2D bone length depends on its depth. We adopt an adaptive bone-length

8
Zhen et al.
threshold to penalize the unreasonable association, depending on predicted body
depths. For each part, we compute the mean bone length in the training set in
advance. Then, its maximal length in 2D is computed and used as the distance
threshold for the corresponding part:
dcons = λ · Dbone · f
Z · w
= λ · Dbone
eZ
,
(2)
where Dbone is the 3D average length of a limb, eZ is the normalized root depth
predicted by our network, and λ is a relaxation factor. dcons is used to ﬁlter
unreasonable links. From Eq. 2 we can see that the adaptive threshold is not
aﬀected by camera intrinsic parameters or image resizing and is only related to
the depth value estimated by our network and the statistical bone length.
As the association can beneﬁt from depth information, we call it depth-aware
part association. Fig. 3(d) shows our qualitative results. The proposed scheme
will also be validated by experiments.
3.3
3D pose reconstruction
Reconstruction. Following the connection relations obtained from part asso-
ciation, relative depths from child nodes to parent nodes can be read from the
corresponding locations of part relative-depth maps. With the root depth and
relative depths of body parts, we are able to compute the depth of each joint.
Given 2D coordinates and joint depths, the 3D pose can be recovered through
the perspective camera model:
X, Y, ZT = ZK−1 x, y, 1T ,
(3)
where [X, Y, Z]T and [x, y]T are 3D and 2D coordinates of a joint respectively. K
is the camera intrinsic matrix, which is available in most applications, e.g., from
device speciﬁcations. In our experiments, we use the focal lengths provided by
the datasets (same as [22]). For internet images with unknown focal lengths, we
use a default value which equals 

## experiments
We evaluate the proposed approach on two widely-used datasets and compare
it to previous approaches. Besides, we provide thorough ablation analysis to
validate our designs.
4.1
Datasets
CMU Panoptic
[11] is a large-scale dataset that contains various indoor
social activities, captured by multiple cameras. Mutual occlusion between in-
dividuals and truncation makes it challenging to recover 3D poses. Following
[41], we choose two cameras (16 and 30), 9600 images from four activities (Hag-
gling, Maﬁa, Ultimatum, Pizza) as our test set, and 160k images from diﬀerent
sequences as our training set.
MuCo-3DHP and MuPoTS-3D [20]. MuCo-3DHP is an indoor multi-person
dataset for training, which is composited from single-person datasets. MuPoTS-
3D is a test set consisting of indoor and outdoor scenes with various camera
poses, making it a convincing benchmark to test the generalization ability.
4.2
Implementation details
We adopt Adam as optimizer with 2e-4 learning rate, and train two models for
20 epochs on the CMU Panoptic and MuCo-3DHP datasets separately, mixed
with COCO data [17]. The batch size is 32 and 50% data in each mini-batch is
from COCO (same as [20,22]). Images are resized to a ﬁxed size 832×512 as the
input to the network. Note that resizing doesn’t change FoV. Since the COCO
dataset lacks 3D pose annotations, weights of 3D losses are set to zero when the
COCO data is fed.
4.3
Evaluation metrics
MPJPE. MPJPE measures the accuracy of the 3D root-relative pose. It cal-
culates the Euclidean distance between the predicted and the groundtruth joint
locations averaged over all joints.
RtError. Root Error (RtError) is deﬁned as the Euclidean distance between
the predicted and the groundtruth root locations.
3DPCK. 3DPCK is the percentage of correct keypoints. A keypoint is declared
correct if the Euclidean distance between predicted and groundtruth coordinates
is smaller than a threshold (15cm in our experiments). PCKrel measures relative
pose accuracy with root alignment; PCKabs measures absolute pose accuracy
without root alignment; and PCKroot only measures the accuracy of root joints.
AUC means the area under curve of 3DPCK over various thresholds.

10
Zhen et al.
Table 1. Results on the Panoptic dataset. For [22], we used the code provided by the
authors and trained it on the Panoptic dataset. *The average of [42] is recalculated
following the standard practice in [41], i.e., average over activities.

## related_work
Multi-person 2D pose. Existing methods for multi-person 2D pose estima-
tion can be approximately divided into two classes. Top-down approaches detect
human ﬁrst and then estimate keypoints with a single person pose estimator
[5,7,27,39]. Bottom-up approaches localize all keypoints in the image ﬁrst and
then group them to individuals [3,9,23,31,10,26,25]. Cao et al. [3] propose Open-
Pose and use part aﬃnity ﬁelds (PAFs) to represent the connection conﬁdence
between keypoints. They solve the part association problem with a greedy strat-
egy. Newell et al. [23] propose an approach that simultaneously outputs detection
and group assignments in the form of pixel-wise tags.
Single-person 3D pose. Researches on single-person 3D pose estimation from
a single image have already achieved remarkable performances in recent years.
One-stage approaches directly regress 3D keypoints from images and can leverage
shading and occlusion information to resolve the depth ambiguity. Most of them
are learning-based [1,36,28,29,35,40,8,21]. Two-stage approaches estimate the
2D pose ﬁrst and then lift it to the 3D pose, including learning-based[18,30,43],

4
Zhen et al.
optimization-based [44,38] and exemplar-based [4] methods, which can beneﬁt
from the reliable result of 2D pose estimation.
Multi-person 3D pose. For the task of multi-person 3D pose estimation, top-
down approaches focus on how to integrate the pose estimation task with the
detection framework. They crop the image ﬁrst and then regress the 3D pose
with a single-person 3D pose estimator. Most of them estimate the translation of
each person with an optimization strategy that minimizes the reprojection error
computed over sparse keypoints [6,33,34] or dense semantic correspondences [41].
Moon et al. [22] regard the area of 2D bounding box as a prior and adopt a neural
network to learn a correction factor. In their framework, they regress the root-
relative pose and the root depth separately. Informative cues for inferring the
interaction between people may lose during the cropping operation. Another
work [37] regresses the full depth map based on the existing depth estimation
framework [16], but their ‘read-out’ strategy is not robust to 2D outliers. On the
other hand, bottom-up approaches focus on how to represent pose annotations
as several maps in a robust way [20,2,42,19]. However, they either optimize the
translation in a post-processing way or ignore the task of root localization. XNect
[19] extends the 2D location map in [21] to 3D ones and estimates the translation
with a calibrated camera and the ground plane constraint, but the feet may be
invisible in crowded scenes and obtaining the extrinsic parameters of the camera
is not practical in most applications. Another line of work tries to recover the
SMPL model [42,41], and their focus lies in using scene constraints and avoiding
interpenetration, which is weakly related to our task. Taking these factors into
consideration, a framework that both considers recovering the translation in a
single forward pass and aggregating global features over the image will be helpful
to this task.
Monocular depth estimation. Depth estimation from a single view suﬀers
from inherent ambiguity. Nevertheless, several methods make remarkable ad-
vances in recent years [16,13]. Li et al. [15] observe that Mannequin Challenge
could be a good source for human depth datasets. They generate training data
using multi-view stereo reconstruction and adopt a data-driven approach to re-
cover a dense depth map, achieving good results. However, such a depth map
lacks scale consistency and cannot reﬂect the real depth.
3

## conclusion
We proposed a novel single-shot bottom-up framework to estimate absolute
multi-person 3D poses from a single RGB image. The proposed framework uses
a fully convolutional network to regress a set of 2.5D representations for multiple
people, from which the absolute 3D poses can be reconstructed. Additionally,
beneﬁted from the depth estimation of human bodies, a novel depth-aware part
association algorithm was proposed and proven to beneﬁt 2D pose estimation in
crowd scenes. Experiments demonstrated state-of-the-art performance as well as
generalization ability of the proposed approach.
Acknowledgements: The authors would like to acknowledge support from
NSFC (No. 61806176), Fundamental Research Funds for the Central Universities
(2019QNA5022) and ZJU-SenseTime Joint Lab of 3D Vision.

SMAP
15
Supplementary Material:
SMAP: Single-Shot Multi-Person
Absolute 3D Pose Estimation
In this supplementary material, we provide more experimental details and
results. Additionally, qualitative results on in-the-wild images from the Internet
are shown in the supplementary video.
1
More details
1.1
Loss function
There are three output branches of the network, illustrated in Fig. 2. The ﬁrst
branch regresses keypoint heatmaps HJ and PAFs C simultaneously, while the
second branch regresses part relative-depth maps H∆Z. L2 loss is applied to
these two branches. The third branch predicts root depth map HRZ. According
to 2D location of detected root (xroot, yroot), we can get the predicted root depth
HRZ(xroot, yroot) and compared it with the groundtruth normalized depth eZ∗
using L1 loss. The total loss is computed by weighted summation of all losses.
Our loss functions are as follows.
Ltotal = w2D · L2D + w∆Z · L∆Z + wRZ · LRZ
L2D =
N
X
i=1
X
p
||HJ,i(p) −H∗
J,i(p)||2
2
+
2N−2
X
i=1
X
p
||Ci(p) −C∗
i (p)||2
2
L∆Z =
N−1
X
i=1
X
p
||H∆Z,i(p) −H∗
∆Z,i(p)||2
2
LRZ =
M
X
i=1
||HRZ(xroot
i
, yroot
i
) −eZ∗
i ||1 ,
where N, M are the number of joints, the number of detected people (root
joints) respectively, p means each pixel location and superscript ∗denotes the
groundtruth. The default settings are: w2D=0.1, w∆Z=5, wRZ=10.
1.2
Running time and memory
Table 5 provides detailed information about running time and memory of the
state-of-the-art top-down method [22] and our method. Note that our method
is almost not aﬀected by the number of people in the image.

16
Zhen et al.
Table 5. Running time and memory comparison.
3-people
20-people
Time(ms)
Memory(M)
Time(ms)
Memory(M)
[22]
DetectNet
120.0
899
120.0
899
PoseNet
14.7
815
71.8
1491
RootNet
13.0
803
58.9
1051
Ours
SMAP
57.0
1379
57.0
1379
DAPA
4.5
-
8.8
-
ReﬁneNet
0.80
∼0.5
0.83
∼0.5
2
More results compared with SOTA
Due to the limited space, only the average PCKabs is reported in the main
manuscript. Here we provide more thorough experimental results. Table 8 presents
sequence-wise PCKabs on the MuPoTS-3D dataset and demonstrates that our
PCKabs is higher than the state-of-the-art top-down method [22], especially for
outdoor scenarios (TS6-TS20). Table 7 shows that our model has higher PCKrel
compared with all bottom-up methods and most top-down methods except [22].
Note that we have higher AUCrel compared with [22] as we state in the main
manuscript. Table 10 shows the results on the Human3.6M dataset.
3
More ablation analysis
3.1
Eﬀect of the multi-task structure
SMAP simultaneously output 2D information (keypoint heatmaps and PAFs),
root depth map, and part relative-depth map. To analyze the impact of our
single-shot multi-task structure on root localization, we delete some of the output
branches and evaluate the performance, as indicated in Table 11. One variant is
only to regress the root position and its depth alone (row 2 of Table 11). This
variant can obtain an acceptable result, which reﬂects the signiﬁcance of our
bottom-up design for root localization. Another variant which adds the keypoint
heatmaps and PAFs branches (row 3 of Table 11) signiﬁcantly improves the
performance, indicating that 2D cues (pose, body size) are also beneﬁcial to root
depth estimation. Nevertheless, this variant is still inferior to the full model.
3.2
Inﬂuence of camera intrinsics
Here we make three comparisons:1) full model with known camera intrinsics. 2)
full model without camera intrinsics. 3) without normalization.
RtError of our full model reaches 23.3cm on the MuPoTS-3D dataset. If the
intrinsic parameter is not provided (use default intrinsics), RtError increases to
67cm. Note that the ordinal depth relation remains unchanged. If the model
lacks normalization, RtError is as high as 120cm.

SMAP
17
Table 6. Sequence-wise PCKabs on the MuPoTS-3D dataset for matched groundtruths.
S1
S2
S3
S4
S5
S6
S7
S8
S9
S10
Moon et al. [22]
59.5 45.3 51.4 46.2 53.0 27.4 23.7 26.4 39.1 23.6
Ours
42.1 41.4 46.5 16.3 53.0 26.4 47.5 18.7 36.7 73.5
S11
S12
S13
S14
S15
S16
S17
S18
S19
S20 Avg.
Moon et al. [22]
18.3 14.9 38.2 29.5 36.8 23.6 14.4 20.0 18.8 25.4 31.8
Ours
46.0 22.7 24.3 38.9 47.5 34.2 35.0 20.0 38.7 64.8 38.7
Table 7. PCKrel on the MuPoTS-3D dataset for matched groundtruths. ‘T’ denotes
top-down methods while ‘B’ denotes bottom-up methods.
S1
S2
S3
S4
S5
S6
S7
S8
S9
S10
Rogez et al. [33]
T
69.1 67.3 54.6 61.7 74.5 25.2 48.4 63.3 69.0 78.1
Rogez et al. [34]
T
88.0 73.3 67.9 74.6 81.8 50.1 60.6 60.8 78.2 89.5
Dabral et al. [6]
T
85.8 73.6 61.1 55.7 77.9 53.3 75.1 65.5 54.2 81.3
Moon et al. [22]
T
94.4 78.6 79.0 82.1 86.6 72.8 81.9 75.8 90.2 90.4
Mehta et al. [20] B
81.0 64.3 64.6 63.7 73.8 30.3 65.1 60.7 64.1 83.9
Mehta et al. [19] B
88.4 70.4 68.3 73.6 82.4 46.4 66.1 83.4 75.1 82.4
Ours
B
89.9 88.3 78.9 78.2 87.6 51.0 88.5 71.6 70.3 89.2
S11
S12
S13
S14
S15
S16
S17
S18
S19
S20 Avg.
Rogez et al. [33]
T
53.8 52.2 60.5 60.9 59.1 70.5 76.0 70.0 77.1 81.4 62.4
Rogez et al. [34]
T
70.8 74.4 72.8 64.5 74.2 84.9 85.2 78.4 75.8 74.4 74.0
Dabral et al. [6]
T
82.2 71.0 70.1 67.7 69.9 90.5 85.7 86.3 85.0 91.4 74.2
Moon et al. [22]
T
79.4 79.9 75.3 81.0 81.0 90.7 89.6 83.1 81.7 77.3 82.5
Mehta et al. [20] B
71.5 69.6 69.0 69.6 71.1 82.9 79.6 72.2 76.2 85.9 69.8
Mehta et al. [19] B
76.5 73.0 72.4 73.8 74.0 83.6 84.3 73.9 85.7 90.6 75.8
Ours
B
76.3 82.0 70.8 65.2 80.4 91.6 90.4 83.4 84.3 91.2 80.5
Table 8. Sequence-wise PCKabs on the MuPoTS-3D dataset.
Accuracy for all groundtruths
S1
S2
S3
S4
S5
S6
S7
S8
S9
S10
Moon et al. [22]
59.5 44.7 51.4 46.0 52.2 27.4 23.7 26.4 39.1 23.6
Ours
41.6 33.4 45.6 16.2 48.8 25.8 46.5 13.4 36.7 73.5
S11
S12
S13
S14
S15
S16
S17
S18
S19
S20 Avg.
Moon et al. [22]
18.3 14.9 38.2 26.5 36.8 23.4 14.4 19.7 18.8 25.1 31.5
Ours
43.6 22.7 21.9 26.7 47.1 32.5 31.4 18.0 33.8 47.8 35.4
Accuracy for matched groundtruths
S1
S2
S3
S4
S5
S6
S7
S8
S9
S10
Moon et al. [22]
59.5 45.3 51.4 46.2 53.0 27.4 23.7 26.4 39.1 23.6
Ours
42.1 41.4 46.5 16.3 53.1 26.4 47.5 18.7 36.7 73.5
S11
S12
S13
S14
S15
S16
S17
S18
S19
S20 Avg.
Moon et al. [22]
18.3 14.9 38.2 29.5 36.8 23.6 14.4 20.0 18.8 25.4 31.8
Ours
46.0 22.7 24.3 38.9 47.5 34.2 35.0 20.1 38.7 64.8 38.7

18
Zhen et al.
Table 9. PCKrel on the MuPoTS-3D dataset for all groundtruths. ‘T’ denotes top-
down methods while ‘B’ denotes bottom-up methods.
S1
S2
S3
S4
S5
S6
S7
S8
S9
S10
Rogez et al. [33]
T
67.7 49.8 53.4 59.1 67.5 22.8 43.7 49.9 31.1 78.1
Rogez et al. [34]
T
87.3 61.9 67.9 74.6 78.8 48.9 58.3 59.7 78.1 89.5
Dabral et al. [6]
T
85.1 67.9 73.5 76.2 74.9 52.5 65.7 63.6 56.3 77.8
Moon et al. [22]
T
94.4 77.5 79.0 81.9 85.3 72.8 81.9 75.7 90.2 90.4
Mehta et al. [20] B
81.0 59.9 64.4 62.8 68.0 30.3 65.0 59.2 64.1 83.9
Mehta et al. [19] B
88.4 65.1 68.2 72.5 76.2 46.2 65.8 64.1 75.1 82.4
Ours
B
88.8 71.2 77.4 77.7 80.6 49.9 86.6 51.3 70.3 89.2
S11
S12
S13
S14
S15
S16
S17
S18
S19
S20 Avg.
Rogez et al. [33]
T
50.2 51.0 51.6 49.3 56.2 66.5 65.2 62.9 66.1 59.1 53.8
Rogez et al. [34]
T
69.2 73.8 66.2 56.0 74.1 82.1 78.1 72.6 73.1 61.0 70.6
Dabral et al. [6]
T
76.4 70.1 65.3 51.7 69.5 87.0 82.1 80.3 78.5 70.7 71.3
Moon et al. [22]
T
79.2 79.9 75.1 72.7 81.1 89.9 89.6 81.8 81.7 76.2 81.8
Mehta et al. [20] B
67.2 68.3 60.6 56.5 69.9 79.4 79.6 66.1 66.3 63.5 65.0
Mehta et al. [19] B
74.1 72.4 64.4 58.8 73.7 80.4 84.3 67.2 74.3 67.8 70.4
Ours
B
72.3 81.7 63.6 44.8 79.7 86.9 81.0 75.2 73.6 67.2 73.5
Table 10. MPJPE Results on Human3.6M dataset. Note that there is no groundtruth
bounding box information in inference time.