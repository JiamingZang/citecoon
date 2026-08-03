# DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion

> 2019 · id: arxiv:1901.04780 · arXiv: 1901.04780 · pdf: https://arxiv.org/pdf/1901.04780 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
A key technical challenge in performing 6D object pose
estimation from RGB-D image is to fully leverage the two
complementary data sources. Prior works either extract in-
formation from the RGB image and depth separately or use
costly post-processing steps, limiting their performances in
highly cluttered scenes and real-time applications. In this
work, we present DenseFusion, a generic framework for
estimating 6D pose of a set of known objects from RGB-
D images.
DenseFusion is a heterogeneous architecture
that processes the two data sources individually and uses a
novel dense fusion network to extract pixel-wise dense fea-
ture embedding, from which the pose is estimated. Further-
more, we integrate an end-to-end iterative pose reﬁnement
procedure that further improves the pose estimation while
achieving near real-time inference. Our experiments show
that our method outperforms state-of-the-art approaches in
two datasets, YCB-Video and LineMOD. We also deploy our
proposed method to a real robot to grasp and manipulate
objects based on the estimated pose. Our code and video
are available at https://sites.google.com/view/densefusion/.

## introduction
6D object pose estimation is the crux to many important
real-world applications, such as robotic grasping and ma-
nipulation [7, 34, 43], autonomous navigation [6, 11, 41],
and augmented reality [18, 19]. Ideally, a solution should
deal with objects of varying shape and texture, show robust-
ness towards heavy occlusion, sensor noise, and changing
lighting conditions, while achieving the speed requirement
of real-time tasks. The advent of cheap RGB-D sensors
has enabled methods that infer poses of low-textured ob-
jects even in poorly-lighted environments more accurately
than RGB-only methods. Nonetheless, it is difﬁcult for ex-
isting methods to satisfy the requirements of accurate pose
estimation and fast inference simultaneously.
Classical approaches ﬁrst extract features from RGB-D
RGB-D
DenseFusion
Figure 1. We develop an end-to-end deep network model for 6D
pose estimation from RGB-D data, which performs fast and accu-
rate predictions for real-time applications such as robot grasping
and manipulation.
data and perform correspondence grouping and hypothesis
veriﬁcation [3, 12, 13, 15, 25, 32, 37]. However, the re-
liance on handcrafted features and ﬁxed matching proce-
dures have limited their empirical performances in presence
of heavy occlusion and lighting variation. Recent success
in visual recognition has inspired a family of data-driven
methods that use deep networks for pose estimation from
RGB-D inputs, such as PoseCNN [40] and MCN [16].
However, these methods require elaborate post-hoc re-
ﬁnement steps to fully utilize the 3D information, such
as a highly customized Iterative Closest Point (ICP) [2]
procedure in PoseCNN and a multi-view hypothesis ver-
iﬁcation scheme in MCN. These reﬁnement steps cannot
be optimized jointly with the ﬁnal objective and are pro-
hibitively slow for real-time applications. In the context of
autonomous driving, a third family of solutions has been
proposed to better exploit the complementary nature of
color and depth information from RGB-D data with end-
to-end deep models, such as Frustrum PointNet [22] and
PointFusion [41]. These models have achieved good per-
formances in driving scenes and the capacity of real-time
1
arXiv:1901.04780v1  [cs.CV]  15 Jan 2019

inference. However, as we demonstrate empirically, these
methods fall short under heavy occlusion, which is common
in manipulation domains.
In this work, we propose an end-to-end deep learning ap-
proach for estimating 6-DoF poses of known objects from
RGB-D inputs. The core of our approach is to embed and
fuse RGB values and point clouds at per-pixel level, as op-
posed to prior work which uses image crops to compute
global features [41] or 2D bounding boxes [22]. This per-
pixel fusion scheme enables our model to explicitly rea-
son about the local appearance and geometry information,
which is essential to handle heavy occlusion. Furthermore,
we propose an iterative method which performs pose re-
ﬁnement within the end-to-end learning framework. This
greatly enhances model performance while keeping the in-
ference speed real-time.
We evaluate our method in two popular benchmarks for
6D pose estimation, YCB-Video [40] and LineMOD [12].
We show that our method outperforms the state-of-the-art
PoseCNN after ICP reﬁnement [40] by 3.5% in pose ac-
curacy while being 200x faster in inference time. In par-
ticular, we demonstrate its robustness in highly cluttered
scenes thanks to our novel dense fusion method. Last, we
also showcase its utility in a real robot task, where the robot
estimates the poses of objects and grasp them to clear up a
table.
In summary, the contributions of this work are two-fold:
First, we present a principled way to combine color and
depth information from the RGB-D input.
We augment
the information of each 3D point with 2D information from
an embedding space learned for the task and use this new
color-depth space to estimate the 6D pose. Second, we in-
tegrate an iterative reﬁnement procedure within the neural
network architecture, removing the dependency of previous
methods of a post-processing ICP step.

## method
Our goal is to estimate the 6D pose of a set of known
objects present in an RGB-D image of a cluttered scene.
Without loss of generality, we represent 6D poses as ho-
mogeneous transformation matrix, p ∈SE(3). In other
2

object
segmentation
PointNet
image
crop
color
embeddings
geometry
embeddings
pixel-wise dense fusion
matching
point
...
pixel-wise feature
average
pooling
global
feature
...
(x1,y1)
(x1,y1)
(xN,yN)
(x2,y2)
(xN,yN)
rotation
translation
confidence
pixel (xi,yi)  i = 1...N
Ri
pose
predictor
ci
prediction per pixel
argmax(c)
6D pose estimation
per-pixel 
feature
masked 
point cloud
CNN
 ti
MLP
Figure 2. Overview of our 6D pose estimation model. Our model generates object segmentation masks and bounding boxes from RGB
images. The RGB colors and point cloud from the depth map are encoded into embeddings and fused at each corresponding pixel. The
pose predictor produces a pose estimate for each pixel and the predictions are voted to generate the ﬁnal 6D pose prediction of the object.
(The iterative procedure of our approach is not depicted here for simplicity)
words, a 6D pose is composed by a rotation R ∈SO(3)
and a translation t ∈R3, p = [R|t]. Since we estimate the
6D pose of the objects from camera images, the poses are
deﬁned with respect to the camera coordinate frame.
Estimating the pose of a known object in adversarial
conditions (e.g.
heavy occlusion, poor lighting, ...)
is
only possible by combining the information contained in
the color and depth image channels. However, the two data
sources reside in different spaces. Extracting features from
heterogeneous data sources and fusing them appropriately
is the key technical challenge in this domain.
We address this challenge with (1) a heterogeneous ar-
chitecture that processes color and depth information dif-
ferently, retaining the native structure of each data source
(Sec. 3.3), and (2) a dense pixel-wise fusion network that
performs color-depth fusion by exploiting the intrinsic map-
ping between the data sources (Sec. 3.4). Finally, the pose
estimation is further reﬁned with a differentiable iterative
reﬁnement module (Sec. 3.6). In contrast to the expensive
post-hoc reﬁnement steps used in [16, 40], our reﬁnement
module can be trained jointly with the main architecture and
only takes a small fraction of the total inference time.
3.1. Architecture Overview
Fig. 2 illustrates the overall proposed architecture. The
architecture contains two main stages. The ﬁrst stage takes
color image as input and performs semantic segmentation
for each known object category. Then, for each segmented
object, we feed the masked depth pixels (converted to 3D
point cloud) as well as an image patch cropped by the
bounding box of the mask to the second stage.
The second stage processes the results of the segmenta-
tion and estimates the object’s 6D pose. It comprises four
components: a) a fully convolutional network that processes
the color information and maps each pixel in the image crop
to a color feature embedding, b) a PointNet-based [23] net-
work that processes each point in the masked 3D point cloud
to a geometric feature embedding, c) a pixel-wise fusion
network that combines both embeddings and outputs the es-
timation of the 6D pose of the object based on an unsuper-
vised conﬁdence scoring, and d) an iterative self-reﬁnement
methodology to train the network in a curriculum learning
manner and reﬁne the estimation result iteratively. Fig. 2
depicts a), b) and c) and Fig. 3 illustrates d). The details
our architecture are described below.
3.2. Semantic Segmentation
The ﬁrst step is to segment the objects of interest in the
image. Our semantic segmentation network is an encoder-
decoder architecture that takes an image as input and gener-
ates an N +1-channelled semantic segmentation map. Each
channel is a binary mask where active pixels depict objects
of each of the N possible known classes. The focus of this
work is to develop a pose estimation algorithm. Thus we
use an existing segmentation architecture proposed by [40].
3.3. Dense Feature Extraction
The key technical challenge in this domain is the correct
extraction of information from the color and depth channels
and their synergistic fusion. Even though color and depth
present a similar format in the RGB-D frame, their infor-
mation resides in different spaces. Therefore, we process
3

them separately to generate color and geometric features
from embedding spaces that retain the intrinsic structure of
the data sources.
Dense 3D point cloud feature embedding: Previous ap-
proaches have used CNN to process the depth image as an
additional image channel [16]. However, such method ne-
glects the intrinsic 3D structure of the depth channel. In-
stead, we ﬁrst convert the segmented depth pixels into a 3D
point cloud using the known camera intrinsics, and then use
a PointNet-like architecture to extract geometric features.
PointNet by Qi et al. [23] pioneered the use of a symmet-
ric function (max-pooling) to achieve permutation invari-
ance in processing unordered point sets. The original archi-
tecture takes as input a raw point cloud and learns to encode
the information about the vicinity of each point and of the
point cloud as a whole. The features are shown to be effec-
tive in shape classiﬁcation and segmentation [23] and pose
estimation [22, 41]. We propose a geometric embedding
network that generates a dense per-point feature by map-
ping each of the P segmented points to a dgeo-dimensional
feature space. We implement a variant of PointNet architec-
ture that uses average-pooling as opposed to the commonly
used max-pooling as the symmetric reduction function.
Dense color image feature embedding: The goal of the
color embedding network is to extract per-pixel features
such that we can form dense correspondences between 3D
point features and image features. The reason for form-
ing these dense correspondences will be clear in the next
section. The image embedding network is a CNN-based
encoder-decoder architecture that maps an image of size
H × W × 3 into a H × W × drgb embedding space. Each
pixel of the embedding is a drgb-dimensional vector repre-
senting the appearance information of the input image at the
corresponding location.
3.4. Pixel-wise Dense Fusion
So far we have obtained dense features from both the
image and the 3D point cloud inputs; now we need to fuse
the information. A naive approach would be to generate a
global feature from the dense color and depth features from
the segmented area. However, due to heavy occlusion and
segmentation errors, the set of features from previous step
may contain features of points/pixels on other objects or
parts of the background. Therefore, blindly fusing color and
geometric features globally would degrade the performance
of the estimation.
In the following we describe a novel
pixel-wise1 dense fusion network that effectively combines
the extracted features, especially for pose estimation under
heavy occlusion and imperfect segmentation.
Pixel-wise dense fusion: The key idea of our dense fu-
sion network is to perform local per-pixel fusion instead
1Since the mapping between pixels and 3D points is unique, we will
use interchangeably pixel-fusion and point-fusion.
of global fusion so that we can make predictions based on
each fused feature. In this way, we can potentially select
the predictions based on the visible part of the object and
minimize the effects of occlusion and segmentation noise.
Concretely, our dense fusion procedure ﬁrst associates the
geometric feature of each point to its corresponding image
feature pixel based on a projection onto the image plane us-
ing the known camera intrinsic parameters. The obtain pairs
of features are then concatenated and fed to another network
to generate a ﬁxed-size global feature vector using a sym-
metric reduction function. While we refrained from using a
single global feature for the estimation, here we enrich each
dense pixel-feature with the global densely-fused feature to
provide a global context.
We feed each of the resulting per-pixel features into a
ﬁnal network that predicts the object’s 6D pose. In other
words, we will train this network to predict one pose from
each densely-fused feature. The result is a set of P pre-
dicted poses, one per feature. This deﬁnes our ﬁrst learning
objective, as we will see in Sec. 3.5. We will now explain
our approach to learn to choose the best prediction in a self-
supervised manner, inspired by the work by Xu et al. [41].
Per-pixel self-supervised conﬁdence: We would like to
train our pose estimation network to decide which pose es-
timation is likely to be the best hypothesis based on the spe-
ciﬁc context. To do so, we modify the network to output
a conﬁdence score ci for each prediction in addition to the
pose estimation predictions. We will have to reﬂect this sec-
ond learning objective in the overall learning objective, as
we will see at the end of the next section.

## experiments
In the experimental section, we would like to answer the
following questions: (1) How does the dense fusion net-
work compare to naive global fusion-by-concatenation? (2)
Is the dense fusion and prediction scheme robust to heavy
occlusion and segmentation errors? (3) Does the iterative
reﬁnement module improve the ﬁnal pose estimation? (4)
Is our method robust and efﬁcient enough for downstream
tasks such as robotic grasping?
To answer the ﬁrst three questions, we evaluate our
method on two challenging 6D object pose estimation
datasets: YCB-Video Dataset [40] and LineMOD [12]. The
YCB-Video Dataset features objects of varying shapes and
texture levels under different occlusion conditions. Hence
it’s an ideal testbed for our occlusion-resilient multi-modal
fusion method.
The LineMOD dataset is a widely-used
dataset that allows us to compare with a broader range of
existing methods. We compare our method with state-of-
the-art methods [14, 30] as well as model variants. To an-
swer the last question, we deploy our model to a real robot
platform and evaluate the performance of a robot grasping
task that uses the predictions from our model.
4.1. Datasets
YCB-Video Dataset. The YCB-Video Dataset Xiang et
al. [40] features 21 YCB objects Calli et al. [5] of varying
shape and texture. The dataset contains 92 RGB-D videos,
where each video shows a subset of the 21 objects in differ-
ent indoor scenes. The videos are annotated with 6D poses
5

Table 1. Quantitative evaluation of 6D pose (ADD-S[40]) on YCB-Video Dataset. Objects with bold name are symmetric.
PointFusion
[41]
PoseCNN+ICP
[40]
Ours (single)
Ours (per-pixel)
Ours (iterative)
AUC
<2cm
AUC
<2cm
AUC
<2cm
AUC
<2cm
AUC
<2cm
002 master chef can
90.9
99.8
95.8
100.0
93.9
100.0
95.2
100.0
96.4
100.0
003 cracker box
80.5
62.6
92.7
91.6
90.8
98.4
92.5
99.3
95.5
99.5
004 sugar box
90.4
95.4
98.2
100.0
94.4
99.2
95.1
100.0
97.5
100.0
005 tomato soup can
91.9
96.9
94.5
96.9
92.9
96.7
93.7
96.9
94.6
96.9
006 mustard bottle
88.5
84.0
98.6
100.0
91.2
97.8
95.9
100.0
97.2
100.0
007 tuna ﬁsh can
93.8
99.8
97.1
100.0
94.9
100.0
94.9
100.0
96.6
100.0
008 pudding box
87.5
96.7
97.9
100.0
88.3
97.2
94.7
100.0
96.5
100.0
009 gelatin box
95.0
100.0
98.8
100.0
95.4
100.0
95.8
100.0
98.1
100.0
010 potted meat can
86.4
88.5
92.7
93.6
87.3
91.4
90.1
93.1
91.3
93.1
011 banana
84.7
70.5
97.1
99.7
84.6
62.0
91.5
93.9
96.6
100.0
019 pitcher base
85.5
79.8
97.8
100.0
86.9
80.9
94.6
100.0
97.1
100.0
021 bleach cleanser
81.0
65.0
96.9
99.4
91.6
98.2
94.3
99.8
95.8
100.0
024 bowl
75.7
24.1
81.0
54.9
83.4
55.4
86.6
69.5
88.2
98.8
025 mug
94.2
99.8
95.0
99.8
90.3
94.7
95.5
100.0
97.1
100.0
035 power drill
71.5
22.8
98.2
99.6
83.1
64.2
92.4
97.1
96.0
98.7
036 wood block
68.1
18.2
87.6
80.2
81.7
76.0
85.5
93.4
89.7
94.6
037 scissors
76.7
35.9
91.7
95.6
83.6
75.1
96.4
100.0
95.2
100.0
040 large marker
87.9
80.4
97.2
99.7
91.2
88.6
94.7
99.2
97.5
100.0
051 large clamp
65.9
50.0
75.2
74.9
70.5
77.1
71.6
78.5
72.9
79.2
052 extra large clamp
60.4
20.1
64.4
48.8
66.4
50.2
69.0
69.5
69.8
76.3
061 foam brick
91.8
100.0
97.2
100.0
92.1
100.0
92.4
100.0
92.5
100.0
MEAN
83.9
74.1
93.0
93.2
88.2
87.9
91.2
95.3
93.1
96.8
and segmentation masks. We follow prior work [40] and
split the dataset into 80 videos for training and 2,949 key
frames chosen from the rest 12 videos for testing and in-
clude the same 80,000 synthetic images released by [40]
in our training set. In our experiments, we compare with
the result of [40] after depth reﬁnement(ICP) and learning-
based depth method [41].
LineMOD Dataset. The LineMOD dataset Hinterstoisser
et al. [12] consists of 13 low-textured objects in 13 videos.
It is widely adopted by both classical methods [4, 8, 36]
and recent learning-based approaches [17, 30, 33]. We use
the same training and testing set as prior learning-based
works [17, 24, 33] without additional synthetic data and
compare with the best ICP-reﬁned results of the state-of-
the-art algorithms.
4.2. Metrics
We use two metrics to report on the YCB-Video Dataset.
The average closest point distance (ADD-S) [40] is an
ambiguity-invariant pose error metric which takes care of
both symmetric and non-symmetric objects into an over-
all evaluation. Given the estimated pose [ ˆR|ˆt] and ground
truth pose [R|t], ADD-S calculates the mean distance from
each 3D model point transformed by [ ˆR|ˆt] to its closest
neighbour on the target model transformed by [R|t]. We
report the area under the ADD-S curve (AUC) following
PoseCNN [40]. We follow prior work and set the maximum
threshold of AUC to be 0.1m. We also report the percent-
age of ADD-S smaller than 2cm (<2cm), which measures
the predictions under the minimum tolerance for robot ma-
nipulation (2cm for most of the robot grippers).
For the LineMOD dataset, we use the Average Distance
of Model Points (ADD) [13] for non-symmetric objects and
ADD-S for the two symmetric objects (eggbox and glue)
following prior works [13, 30, 33].
4.3. Implementation Details
The image embedding network consists of a Resnet-
18 encoder followed by 4 up-sampling layers as the de-
coder. The PointNet architecture is an MLP followed by
an average-pooling reduction function. Both color and geo-
metric dense feature embedding are of dimension 128. We
choose w = 0.01 for Eq. 3 by empirical evaluation. The
iterative pose reﬁnement module consists of a 4 fully con-
nected layers that directly output the pose residual from the
global dense feature. We use the 2 reﬁnement iterations for
all experiments.
4.4. Architectures
We compare four model variants that showcase the ef-
fectiveness of our design choices.
• PointFusion [41] uses a CNN to extract a ﬁxed-size fea-
ture vector and fuse by directly concatenating the image fea-
ture with the geometry feature. The rest of the network is
similar to our architecture. The comparison to this baseline
demonstrates the effectiveness of our dense fusion network.
• Ours (single) uses our dense fusion network, but instead
6

Figure 4. Qualitative results on the YCB-Video Dataset. All three methods shown here are tested with the same segmentation masks as
in PoseCNN. Each object point cloud in different color are transformed with the predicted pose and then projected to the 2D image frame.
The ﬁrst two rows are former RGB-D methods and the last row is our approach with dense fusion and iterative reﬁnement (2 iterations).
of performing per-point prediction, it only outputs a single
prediction using the global feature vector.
• Ours (per-pixel) performs per-pixel prediction based on
each densely fused feature.
• Ours (iterative) is our complete model that uses the iter-
ative reﬁnement (Sec. 3.6) on top of Ours (per-pixel).
4.5. Evaluation on YCB-Video Dataset
Table 1 shows the evaluation results for all the 21
objects in the YCB-Video Dataset.
We report the
ADD-S AUC(<0.1m) and the ADD-S<2cm metrics on
PoseCNN [40] and our four model variants. To ensure a fair
comparison, all methods use the same segmentation masks
as in PoseCNN [40].
Among our model variants, Ours
(Iterative) achieves the best performance. Our method is
able to outperform PoseCNN + ICP[40] even without itera-
tive reﬁnement. In particular, Ours (Iterative) outperforms
PoseCNN + ICP by 3.5% on the ADD-S<2cm metric.
Effect of dense fusion Both of our dense fusion baselines
(Ours (single) and Ours (per-pixel)) outperform PointFu-
sion by a large margin, which shows that dense fusion has
a clear advantage over the global fusion-by-concatenation
method used in PointFusion.
Effect of iterative reﬁnement Table 1 shows that our iter-
ative reﬁnement improves the overall pose estimation per-
formance. In particular, it signiﬁcantly improves the per-
formances for texture-less symmetric object, e.g., bowl
(29%), banana (6%), and extra large clamp (6%)
which suffer from orientation ambiguity.
Robustness towards occlusion The main advantage of our
dense fusion method is its robustness towards occlusions.
To quantify the effect of occlusion on ﬁnal performance,
we calculate the visible surface ratio of each object instance
(further detail available in supplementary material). Then
we calculate how the accuracy (ADD-S<2cm percentage)
changes with extent of occlusion. As shown in Fig. 5, the
performances of PointFusion and PoseCNN+ICP degrade
signiﬁcantly as the occlusion increases. In contrast, none of
our methods experiences notable performance drop. In par-
ticular, the performance of both Ours (per-pixel) and Ours
(iterative) only decrease by 2% overall.
Time efﬁciency We compare the time efﬁciency of our
model with PoseCNN+ICP in Table 3.
We can see
that our method is two order of magnitude faster than
PoseCNN+ICP. In particular, PoseCNN+ICP spends most
of time on the post processing ICP. In contrast, all of
our computation component, namely segmentation (Seg),
pose estimation (PE), and iterative reﬁnement (Reﬁne), are
equally efﬁcient, and the overall runtime is fast enough
for real-time application (16 FPS, about 5

## related_work
Pose from RGB images. Classical methods rely on detect-
ing and matching keypoints with known object models [1, 7,
9, 26, 43]. Newer methods address the challenge by learn-
ing to predict the 2D keypoints [3, 21, 31, 33, 34] and solve
the poses by PnP [10]. Though prevail in speed-demanding
tasks, these methods become unreliable given low-texture
or low-resolution inputs. Other methods propose to directly
estimate objects pose from images using CNN-based archi-
tectures [27, 35]. Many such methods focus on orientation
estimation: Xiang et al. [38, 39] learns a viewpoint-aware
pose estimator by clustering 3D features from object mod-
els. Mousavian et al. [20] predicts 3D object parameters and
recovers poses by single-view geometry constraints. Sun-
dermeyer et al. [30] implicitly encode orientation in a latent
space and in test time ﬁnd the best match in a codebook as
the orientation prediction. However, pose estimation in 3D
remains a challenge for the lack of depth information. Our
method leverages both image and 3D data to estimate object
poses in 3D in an end-to-end architecture.
Pose from depth / point cloud. Recent studies have pro-
posed to directly tackle the 3D object detection problem in
discretized 3D voxel spaces. For example, Song et al. [28,
29] generate 3D bounding box proposals and estimate the
poses by featuring the voxelized input with 3D ConvNets.
Although the voxel representation effectively encodes ge-
ometric information, these methods are often prohibitively
expensive: [29] takes nearly 20 seconds for each frame.
More recent 3D deep learning architectures have en-
abled methods that directly performs 6D pose estimation
on 3D point cloud data.
As an example, both Frustrum
PointNets [22] and VoxelNet [42] use a PointNet-like [23]
structure and achieved state-of-the-art performances on the
KITTI benchmark [11]. Our method also makes use of sim-
ilar architecture. However, unlike urban driving applica-
tions for which point cloud alone provides enough informa-
tion, generic object pose estimation tasks such as the YCB-
Video dataset [40] demands reasoning over both geometric
and appearance information. We address such a challenge
by proposing a novel 2D-3D sensor fusion architecture.
Pose from RGB-D data. Classical approaches extract 3D
features from the input RGB-D data and perform corre-
spondence grouping and hypothesis veriﬁcation [3, 12, 13,
15, 25, 32, 37]. However, these features are either hard-
coded [12, 13, 25] or learned by optimizing surrogate ob-
jectives [3, 32, 37] such as reconstruction [15] instead of
the true objective of 6D pose estimation. Newer methods
such as PoseCNN [40] directly estimates 6D poses from im-
age data. Li et al. [16] further fuses the depth input as an
additional channel to a CNN-based architecture. However,
these approaches rely on expensive post-processing steps to
make full use of 3D input. In comparison, our method fuses
3D data to 2D appearance feature while retaining the geo-
metric structure of the input space, and we show that it out-
performs [40] on the YCB-Video dataset [40] without the
post-processing step.
Our method is most related to PointFusion [41], in which
geometric and appearance information are fused in a het-
erogeneous architecture. We show that our novel local fea-
ture fusion scheme signiﬁcantly outperforms PointFusion’s
naive fusion-by-concatenation method. In addition, we use
a novel iterative reﬁnement method to further improve the
pose estimation.

## conclusion
We presented a novel approach to estimating 6D poses of
known objects from RGB-D images. Our approach fuses a
dense representation of features that include color and depth
information based on the conﬁdence of their predictions.
With this dense fusion approach, our method outperforms
previous approaches in several datasets, and is signiﬁcantly
more robust against occlusions. Additionally, we demon-
strated that a robot can use our proposed approach to grasp
and manipulate objects.
Acknowledgement
This work has been partially supported by JD.com
American Technologies Corporation (“JD”) under the
SAIL-JD AI Research Initiative and by an ONR MURI
award (1186514-1-TBCJE). This article solely reﬂects the
opinions and conclusions of its authors and not JD or any
entity associated with JD.com.