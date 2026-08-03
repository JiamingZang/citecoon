# Mutual Adaptive Reasoning for Monocular 3D Multi-Person Pose Estimation

> 2022 · id: W4304091600 · arXiv: 2207.07900 · pdf: https://dl.acm.org/doi/pdf/10.1145/3503161.3548148 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Inter-person occlusion and depth ambiguity make estimating the
3D poses of monocular multiple persons as camera-centric coordi-
nates a challenging problem. Typical top-down frameworks suffer
from high computational redundancy with an additional detection
stage. By contrast, the bottom-up methods enjoy low computational
costs as they are less affected by the number of humans. However,
most existing bottom-up methods treat camera-centric 3D human
pose estimation as two unrelated subtasks: 2.5D pose estimation
and camera-centric depth estimation. In this paper, we propose
a unified model that leverages the mutual benefits of both these
subtasks. Within the framework, a robust structured 2.5D pose
estimation is designed to recognize inter-person occlusion based
on depth relationships. Additionally, we develop an end-to-end
geometry-aware depth reasoning method that exploits the mutual
benefits of both 2.5D pose and camera-centric root depths. This
method first uses 2.5D pose and geometry information to infer
camera-centric root depths in a forward pass, and then exploits
the root depths to further improve representation learning of 2.5D
pose estimation in a backward pass. Further, we designed an adap-
tive fusion scheme that leverages both visual perception and body
geometry to alleviate inherent depth ambiguity issues. Extensive
experiments demonstrate the superiority of our proposed model
over a wide range of bottom-up methods. Our accuracy is even
competitive with top-down counterparts. Notably, our model runs
much faster than existing bottom-up and top-down methods.
CCS CONCEPTS
• Computing methodologies →Activity recognition and un-
derstanding.
KEYWORDS
3D Human Pose Estimation, Depth Regression, Geometry-Aware
Depth Reasoning, Adaptive Fusion
*Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
MM ’22, October 10–14, 2022, Lisboa, Portugal.
© 2022 Association for Computing Machinery.
ACM ISBN 978-1-4503-9203-7/22/10...$15.00
https://doi.org/10.1145/3503161.3548148
ACM Reference Format:
Juze Zhang1,2,3, Jingya Wang1,4∗, Ye Shi1,4∗, Fei Gao1, Lan Xu 1,4, Jingyi
Yu 1,4. 2022. Mutual Adaptive Reasoning for Monocular 3D Multi-Person
Pose Estimation. In Proceedings of the 30th ACM International Conference on
Multimedia (MM ’22), Oct. 10–14, 2022, Lisboa, Portugal. ACM, New York,
NY, USA, 9 pages. https://doi.org/10.1145/3503161.3548148
1

## introduction
In recent years, 3D pose estimation has attracted a great deal of in-
terest from researchers due to its widespread application in a range
of fields, including video analysis, camera surveillance, human-
computer interaction, virtual/augmented reality, etc. Great suc-
cesses have been achieved in monocular 3D human pose estima-
tion, especially when dealing with a single person in an image[9,
20, 22, 36, 38, 41]. However, when it comes to real-world scenarios,
inferring poses can be much harder, especially when multiple per-
sons are present in the scene. This is because body joints are often
occluded by other objects or people. For this reason, monocular
3D multi-person pose estimation (3D-MPE) is still a challenging
problem, yet, robustness to such occlusions is critical to real-world
applications.
Camera-centric 3D-MPE tasks aim to recover each pose as coor-
dinates in the camera-centric coordinate system. This requires esti-
mating the absolute depth of each person in 3D. Generally, existing
methods for 3D-MPE can be divided into two categories: top-down
and bottom-up. Typical top-down approaches use the single-person
method with a 2D person detector to handle multi-person scenes
[29]. This involves a human detector and pose estimation in two
stages. Since each person in the image is treated individually, these
methods ignore out-of-patch contexts. They also suffer from high
computational redundancy. By contrast, bottom-up 3D-MPE [42]
does not need human detectors and can perceive global image
contexts in a single shot. Thus, multi-person interactions in pose
estimation tasks do not present a problem.
Most existing bottom-up methods treat camera-centric 3D-MPE
as two unrelated subtasks: 2.5D pose representation and depth esti-
mation as shown in Figure 1a. We argue that a robust 2.5D pose rep-
resentation, especially in crowded multi-person scenes, requires the
global depth cues over the whole image to be aggregated for disam-
biguation. Further, the geometric information associated with 2.5D
poses can lead to a closed-form solution for individual depths. This
observation motivated us to design adaptive reasoning between
pose and depth estimation that mutually benefit each other. To this
end, we developed a novel unified model that combines the two sub-
tasks as shown in Figure 1b. To improve the representation learning
of 2.5D pose estimation, we developed an end-to-end method using
arXiv:2207.07900v1  [cs.CV]  16 Jul 2022

MM ’22, October 10–14, 2022, Lisboa, Portugal.
Juze Zhang et al.
2.5D Pose
Depth
Mutual Adaptive 
Reasoning
(b)
2.5D Pose
Depth
(a)
Figure 1: (a) Existing methods treat camera-centric 3D-MPE
as two unrelated subtasks: 2.5D pose representation and ab-
solute depth estimation. (b) Our method can bridge the gap
between 2.5D pose representation and depth estimation and
achieve mutual benefit from them.
the information of geometry depths. To disambiguate the depths,
we designed a depth fusion scheme that leverages both direct visual
perception and geometry to deal with inherent depth ambiguity
issues in monocular human depth estimation. Thus, the model can
bridge the gap between 2.5D pose representation and depth esti-
mation. Moreover, the framework includes a process for robust
structured pose refinement to handle occlusive or out-of-image
joints. This means the model is capable of end-to-end learning with-
out needing an additional network in the post-processing stage to
complete any missing joints. The whole pipeline is illustrated in
figure 2. To summarize, our main contributions are:
• A unified bottom-up model that leverages the mutual ben-
efits of 2.5D pose and depth estimation. Exploiting each
strength yields greater robustness for disambiguation.
• An end-to-end geometry aware depth reasoning method
that first uses 2.5D pose and geometry information to in-
fer camera-centric root depths in a forward pass, and then
exploits the root depths to further improve representation
learning of 2.5D pose estimation in a backward pass.
• An adaptive depth fusion that leverages both direct visual
perception and geometry to deal with inherent depth ambi-
guity issues in monocular human depth estimation.
2

## method
Our task is to estimate the 3D poses for multiple people in a monoc-
ular RGB image as camera-centric coordinates considering the

Mutual Adaptive Reasoning for Monocular 3D Multi-Person Pose Estimation
MM ’22, October 10–14, 2022, Lisboa, Portugal.
Input image
Backbone
Robust 2.5D 
Structured Pose 
Estimation
Geometry-Aware Depth Reasoning
Geometry 
depth 𝐙root
geo
Regression
depth 𝐙root
reg
Adaptive Fusion for 
Depth Ambiguity
Geometry-based Backpropagation
Adaptive 
Fusion
Back
projection
𝐙root
geo = arg min
𝑍root
||𝑑𝑓𝒁root, 𝑷𝑛2.5𝐷, 𝑓𝒁root, 𝑷𝑚
2.5𝐷
−Ω||2
2
Depth
Regression
depth
Figure 2: Overview of the proposed Mutual Adaptive Reasoning (MAR) framework.
challenge of inter-person occlusion and depth ambiguity. In this
paper, we propose a Mutual Adaptive Reasoning (MAR) method
for this task. MAR is a unified bottom-up model that leverages
the mutual benefits of 2.5D pose and depth estimation. Within the
model, a robust structured 2.5D pose estimation is first designed
to recognize inter-person occlusion based on depth relationships.
Additionally, we develop an end-to-end geometry aware depth rea-
soning method that exploits the mutual benefits of both 2.5D pose
and camera-centric root depths. This method first uses 2.5D pose
and geometry information to infer camera-centric root depths in a
forward pass, and then exploits the root depths to further improve
representation learning of 2.5D pose estimation in a backward pass.
Further, we designed a depth fusion scheme that leverages both
visual perception and body geometry to alleviate inherent depth
ambiguity issues. An overview of the proposed Mutual Adaptive
Reasoning (MAR) method is depicted in Figure 2. More details on
the structured pose estimation module are provided in Section 3.1,
followed by the direct depth regression module in Section 3.2, the
end-to-end geometry depth aware reasoning module in Section 3.3
and the adaptive fusion for depth ambiguity in Section3.4.
3.1
Robust 2.5D Structured Pose Estimation
For monocular 3D multi-person pose estimation (3D-MPE), the
goal is to recover the absolute camera-centric coordinates of the
keypoint of multiple people {P3𝐷
𝑘}𝐽
𝑘=1, where P3𝐷
𝑘
= [𝑋𝑘,𝑌𝑘,𝑍𝑘]𝑇
and 𝐽represent the number of joints. A 3D pose can be decomposed
into representations of 2.5D pose {P2.5𝐷
𝑘
}𝐽
𝑘=1 = {
h
𝑢𝑘, 𝑣𝑘,𝑍𝑟
𝑘
i𝑇
}𝐽
𝑘=1
and the absolute depth of the root joint 𝑍root. Here, the coordinates
𝑢𝑘and 𝑣𝑘are the image pixel coordinates of the 𝑘-th keypoint,
𝑍𝑟
𝑘is its relative depth to the root keypoint and 𝑍root is the ab-
solute camera-centric depth of each person. Given a 2.5D pose
{P2.5𝐷
𝑘
}𝐽
𝑘=1, several elements can be derived from back-projection.
These are: the intrinsic camera parameters 𝐾∈R3×3; the absolute
human depth 𝑍root, and the final absolute 3D pose {P3𝐷
𝑘}𝐽
𝑘=1. Let
𝑓: (P2.5𝐷
𝑘
, Zroot) →P3D
k
be a back-projection function. Thus,
𝑓: P3𝐷
𝑘
=

Zroot + Zr
k

𝐾−1[𝑢𝑘, 𝑣𝑘, 1]𝑇,
(1)
where
𝐾=

𝑓𝑥
0
𝑐𝑥
0
𝑓𝑦
𝑐𝑦
0
0
1

,
𝑓𝑥, 𝑓𝑦is focal lengths divided by the per-pixel distance factors (pixel)
of x-axis and y-axis and 𝑐𝑥,𝑐𝑦is principal point of x-axis and y-axis.
In real application scenarios, the body joints may be occluded,
making it hard to infer occluded joints from heatmaps. Thus, we
designed the offset scheme for pose estimation, which handles
issues of occlusion as well as completely out-of-frame joints as
shown in Fig.3. Appearance-based heatmap representations have a
more precise spatial positioning accuracy, but they usually fail to
infer occluded joints. By contrast, 2.5D joint offset scheme estimates
its position from the offset between root joints, which is more
robust to occlusive joints. Here, we propose a robust structured
pose estimation that simultaneously estimates heatmaps and 2.5D
joint offsets.
A heatmap of keypoints encodes the center of a 2D human body
in the image, where the activation value at each position indicates
the degree to which the keypoint lies in position. The position of
each joint is represented as a gaussian distribution in the heatmap,
and we use PAFs to link the identity-agnostic joints from the key-
point heatmaps after non-maximum suppression following [11].
Then, we can get the 2D joints location (𝑢ℎ𝑚
𝑘, 𝑣ℎ𝑚
𝑘
)𝑇at image pixel
coordinate from heatmap scheme.

MM ’22, October 10–14, 2022, Lisboa, Portugal.
Juze Zhang et al.
Offset-based pose 
estimation
2.5D Structured Pose Estimation
2.5D Pose
2D offset
Depth offset
Heatmap and PAFs
Figure 3: Robust 2.5D Structured Pose Estimation
Moreover, we introduce 2.5D joint offset scheme that aims to
estimate (𝐽−1) human joint offsets with respect to a root joint. The
offset maps are encoded at the spatial location of each root joint
predicted by the keypoint heatmap. Formally, the 2.5D offset-based
pose estimation is formulated as follows:
(𝑢𝑑𝑖𝑠𝑝
𝑘
, 𝑣𝑑𝑖𝑠𝑝
𝑘
,𝑍𝑟
𝑘) = (𝑢root, 𝑣root, 0) + (Δ𝑢𝑘, Δ𝑣𝑘, Δ𝑍𝑘),
(2)
where (𝑢root, 𝑣root, 0) represents the spatial location of the root joint
as image coordinates and (Δ𝑢𝑘, Δ𝑣𝑘, Δ𝑍𝑘) represents the offset of
the 𝑘-th body joint position with respect to the root joint. Note that
the relative depth of the root joint with respect to itself must be
zero. Based on above observation, we use heatmap and part affinity
fields to estimate the major keypoints. Occlusive or out-of-image
joints can be inferred using the offset scheme. The final structured
2.5D pose P2.5𝐷
𝑘
is then inferred as follows:
P2.5𝐷
𝑘
=
(
(𝑢ℎ𝑚
𝑘, 𝑣ℎ𝑚
𝑘,𝑍𝑟
𝑘)𝑇
if 𝑐𝑘≥threshold,
(𝑢𝑑𝑖𝑠𝑝
𝑘
, 𝑣𝑑𝑖𝑠𝑝
𝑘
,𝑍𝑟
𝑘)𝑇
else,
(3)
where 𝑐𝑘is the local maximum value of heatmaps for joint 𝑘. It is
worth noting that Eq.3 was only used in inference stage. The 2.5D
pose estimation is trained in an end-to-end manner with the depth
estimation. The total training loss of the pose estimation includes
heatmap loss, PAF loss and keypoints offset loss.
3.2
Depth Regression
Generally, the human depth estimated from a single camera view
is inherently ambiguous. This ambiguity will make it hard to learn
an accurate regression model. To handle this issue, we exploit the
capability of probability prediction to capture the tolerance of depth
ambiguity by predicting a depth distribution instead of an absolute
depth, which has also been used in the field of object detection
[17]. Here, we assume that the depth distribution of each person
is independent and follows a Laplace distribution 𝐿(𝑧reg
root, 𝜆). The
probability density function of a Laplace random variable 𝑍∼
𝐿(𝑍reg
root, 𝜆) is:
𝑝reg(𝑍) = 1
2𝜆𝑒−
|𝑍−𝑍reg
root|
𝜆
(4)
where 𝜇and 𝜆are parameters of the Laplace distribution. The
ground-truth depth can also be formulated as a Laplace distribution.
Since it is deterministic, it can be further represented by a Dirac
delta function:
𝑝𝐷(𝑍) = 𝛿(𝑍−ˆ𝑍root),
(5)
where ˆ𝑍root represents the ground truth of depth. The distance
between the distribution of direct depth and the ground truth is
measured by the Kullback Leibler(KL) Divergence.
𝐿reg = 𝐷𝐾𝐿(𝑝𝐷(𝑍)∥𝑝reg(𝑍))
∝
|𝑍reg
root −ˆ𝑍root|1
𝜎1
+ log(𝜎1),
(6)
where 𝜎1 =
√
2𝜆is the standard deviation of the Laplace distribution
to indicate the depth uncertainty. If the model lacks confidence in
its prediction, it will output a larger 𝜎so that 𝐿reg can be reduced.
The term log(𝜎) avoids trivial solutions and encourages the model
to be optimistic about accurate predictions. Detailed derivation is
provided in Supplementary.
3.3
Geometry-Aware Depth Reasoning
The inherent ambiguity makes it hard to learn an accurate regres-
sion model. Given the difficulty to regress depth directly, we develop
a geometry depth reasoning method that exploits the mutual bene-
fits of both 2.5D pose and camera-centric root depths. This method
first uses 2.5D pose and geometry information to infer camera-
centric root depths in a forward pass, and then exploits the root
depths to improve representation learning of 2.5D pose estimation
in a backward pass.
3.3.1
Geometry-based Forward Pass. Once the 2.5D pose P2.5𝐷has
been determined by Section 3.1, the 2D image coordinates should
be back-projected to the camera-centred coordinate space using
the estimated depth value to get the final coordinates of P3𝐷. Given
a 2.5D pose P2.5𝐷, the intrinsic camera parameters 𝐾, and a torso
length Ω, the geometry reasoning depth can be inferred as follows:
𝑍geo
root = arg min
𝑍root
∥𝑑(𝑓(𝑍root, P2.5𝐷
n
), 𝑓(𝑍root, P2.5𝐷
m
)) −Ω∥2
2
(7)
where the function 𝑑(·, ·) measures the distance between the root
joint and neck joint in the camera-centric space. Eq.7 leads to a
closed-form solution as follows:
𝑍geo
root = −𝑏+
√
𝑏2 −4𝑎𝑐
2𝑎
,
(8)
where
𝑎= (𝑓−1
𝑥Δ𝑢𝑚)2 + (𝑓−1
𝑦Δ𝑣𝑚)2,
𝑏= 2𝑍𝑟
𝑚[𝑓−2
𝑥Δ𝑢𝑚(𝑢𝑚−𝑐𝑥) + 𝑓−2
𝑦Δ𝑣𝑚(𝑣𝑚−𝑐𝑦)],
𝑐= (𝑍𝑟
𝑚)2[𝑓−2
𝑥
(𝑢𝑚−𝑐𝑥)2 + 𝑓−2
𝑦(𝑣𝑚−𝑐𝑦)2 + 1]2 −Ω2.
Detailed derivation is provided in Supplementary. To optimize the
final geometry aware depth distribution, we apply the uncertainty
regression loss same as 3.2:
𝐿geo =
|𝑍geo
root −ˆ𝑍root|
𝜎2
+ log(𝜎2).
(9)

Mutual Adaptive Reasoning for Monocular 3D Multi-Person Pose Estimation
MM ’22, October 10–14, 2022, Lisboa, Portugal.
3.3.2
Geometry-bas

## experiments
4.1
Datasets and Evaluation Metrics
MuCo-3DHP and MuPoTS Datasets. MuCo-3DHP and MuPoTS-
3D, two datasets proposed by [28], were used to assess the frame-
work’s ability to estimate multi-person 3D poses. MuCo-3DHP, the
training set we used is a large-scale synthesized dataset, which
was generated from single person 3D pose estimation dataset MPI-
INF-3DHP [26] by randomly compositing the persons. We use the
same set of MuCo-3DHP synthesized images from [29] for a fair
comparison. 400K frames of MuCo-3DHP are used for training,
among which half are background augmented. We used MuPoTS-
3D datasets as our test set which is a real-world outdoor scenes
dataset captured by a marker-less motion capture system. Besides,
an additional 2D human keypoint dataset COCO is used to train
together with the MuCo-3DHP dataset following Mehta et al. [29].
Accordingly, we set the loss value of depth becomes zero when the
COCO dataset was imported.
Human3.6M Dataset. The Human3.6M dataset [19] is currently
the largest publicly available dataset for human 3D pose estima-
tion. Two experimental protocols are widely used for training and
testing. Protocol 1 uses S1, S5, S6, S7, S8, S9 in training and S11 in
testing, while Protocol 2 uses S1, S5, S6, S7, S8 in training and S9,
S11 in testing. Same as the configuration of MuCo-3DHP dataset,
additional 2D human keypoint dataset COCO is used to train to-
gether with the Human3.6M dataset. Accordingly, we set the loss
value of depth becomes zero when COCO dataset was imported.
Following previous work [29], we use Protocol 2 and sample every
5th and 64th frames in videos for training and testing respectively.
COCO Dataset. The COCO dataset [25] contains over 250, 000
person instances labeled and 200,000 images with 17 keypoints
annotations. COCO dataset is divided into three sets named train,
val and test-dev, containing 57k, 5k and 20k images respectively. In
this paper, we used train2017 as an additional 2D human keypoint
dataset. Accordingly, we set the loss value of depth becomes zero
followed previous work [29] when COCO dataset was imported.
Evaluation Metrics. Although our task is 3D-MPE at camera-
centric coordinates, we also perform person-centric 3D-MPE evalu-
ation metrics. We use Percentage of Correct 3D Keypoints(PCK) to
evaluate the performance of 3D-MPE on MuPoTS-3D, which calcu-
late the percentage of correct joints if it lies within 15cm from the
ground truth joint location. Following [29], We report the relative
3DPCKrel that with root alignment to evaluate the person-centric
3D-MPE, and absolute 3DPCKabs that without root alignment to
evaluate the camera-centric 3D-MPE. To compare the human depth
location ability, we evaluate PCKroot that only measures the accu-
racy of root joints and percentage of correct ordinal depth (PCOD)
that measures the accuracy of ordinal depth following [42].
4.2
Implementation Details
We use HRNet-w32 [35, 40] pre-trained on the ImageNet dataset as
our backbone network and Adam as our optimizer with a 5 × 10−4
learning rate and a 10−6 weight decay. All input images were padded
to the same size of 512 × 832. Every prediction head attached to
the backbone consists of one 3 × 3 × 256 conv layer, BatchNorm,
ReLU and another 1 × 1 × 𝑐0 conv layer, where 𝑐0 is the output size.
Following [6], we implemented multi-scale supervision from the
four scale output of HRNet-w32 at the heatmap and part affinity
field head. The model was trained for 50k iterations with a batch size
of 32 on four RTX 3090 GPUs; 50% of data in each mini-batch was
from COCO dataset [25]. The data augmentation process during
the training included rotation, horizontal flips, and color jittering.
4.3
Quantitative Evaluation on MuPoTS-3D
To evaluate the performance of 3D-MPE in complex scenarios, we
perform experiments on MuPoTS-3D and compare it to current
state-of-the-art methods, as shown in Table 1. After root alignment

MM ’22, October 10–14, 2022, Lisboa, Portugal.
Juze Zhang et al.
All people
Matched
Scheme

## related_work
Multi-Person 2D Pose Estimation. As mentioned, existing meth-
ods for multi-person 2D pose estimation can mainly be divided into
top-down and bottom-up approaches. Typical top-down frame-
works deal with human detection and pose estimation in two stages
[6, 13]. Despite their strength at handling scale variation, top-down
methods also suffer from high computational redundancy when
attempting to detect additional persons. Moreover, their perfor-
mance tends to degrade severely with heavy occlusion since they
have no awareness of out-of-patch contexts. Bottom-up approaches
[3–5, 14, 18, 21, 23, 30–32, 43] localize all key points in the image
first and then group them to an individual. Examples include Open-
Pose [5], a representative work, which was the first to present the
part-affinity field approach that links key points likely to lie in the
same person. Zhou et al. [43] were the first to propose a single-stage
offset-based method for 2D human pose estimation. However, the
above method only focuses on the 2D level and is careless about
3D information from the image.
Top-Down Multi-Person 3D Pose Estimation. Early works fo-
cused on human-centric tasks by using a top-down manner without
estimating individual depth [33, 34]. Only a few works tackle the
problem of camera-centric multi-person 3D pose estimation from a
monocular RGB image or video. Moon et al. [29] was the first to
suggest locating a person’s root absolute depth by learning a cor-
rection factor of the area of 2D bounding box, while Lin et al. [24]
proposed a pose-aware human depth estimation network to address
the problem of root joint localization for multi-person 3D pose esti-
mation in the camera-space. Similarly, Veges et al. [37] proposed to
use two separate networks, pose estimator and depth estimator, to
recover camera-centric multi-person 3D pose. To handle inherent
depth ambiguity, Wang et al. [39] proposed a novel hierarchical
multi-person ordinal relation. Another type of work utilizes tem-
poral information to recover 3D poses from a given video[7, 8]. By
applying a top-down scheme, the above method either directly re-
gresses the absolute 3D depth from a cropped image, or it computes
it based on a prior of the body size, ignoring global image contexts.
Moreover, they suffer from high computational redundancy with
an additional detection stage. By contrast, our methods can utilize
global image contexts and enjoy low computational costs as they
are less affected by the number of humans.
Bottom-Up Multi-Person 3D Pose Estimation. A few bottom-
up methods have been proposed for 3D-MPE[12, 27, 28, 42]. Mehta
et al. [28] first proposed a bottom-up method with an occlusion-
robust pose-map (ORPM) to represent the occlusive joints. Such
methods can alleviate the occlusion problem to some degree, but
they only focus on person-centric and do not infer multiple per-
sons as camera-centric coordinates. While Fabbri et al. [12] used an
encoder-decoder network to compress a heatmap and then decom-
press it back to the original resolution in the inference time for fast
HD image processing. They show superior results when tackling
crowded scenes but careless about the occlusion problem. Another
method called XNect [27], a framework that encodes a 3D location
map at the spatial location of each visible joint, which also focuses
on a person-centric problem. Zhen et al. [42] proposed an end-to-
end network with the depth-aware part association and bone-length
constraints that benefits the 2.5D pose estimation branch of the
scheme. However, these models heavily depend on post-processing
techniques or refinement. Above bottom-up methods treat camera-
centric 3D human pose estimation as two unrelated subtasks: 2.5D
pose estimation and camera-centric depth estimation. By contrast,
our model focus on exploring the mutual benefits 2.5D pose esti-
mation and camera-centric depths in an end-to-end manner.
3

## conclusion
In this paper, we developed a unified bottom-up model that lever-
ages the mutual benefits of both 2.5D pose and depth estimation
to handle the Monocular 3D-MPE problem. Different from existing
top-down or bottom-up methods that treat camera-centric 3D-MPE
as two unrelated subtasks: 2.5D pose representation and absolute
depth estimation, our method can bridge the gap between 2.5D pose
representation and depth estimation and thus benefit from each
other. First, we designed a robust structured 2.5D pose estimation
is designed to recognize inter-person occlusion based on depth
relationships. Additionally, we developed an end-to-end differen-
tiable geometry depth reasoning method that exploits the mutual
benefits of both 2.5D pose and camera-centric root depths. This
method first uses 2.5D pose and geometry information to infer
camera-centric root depths in a forward pass, and then exploits the
root depths to further improve representation learning of 2.5D pose
estimation in a backward pass. Further, we designed a depth fusion