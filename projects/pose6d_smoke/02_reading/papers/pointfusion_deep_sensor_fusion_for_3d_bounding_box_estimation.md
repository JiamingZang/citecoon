# PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation

> 2018 · id: W2962888833 · arXiv: 1711.10871 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present PointFusion, a generic 3D object detection
method that leverages both image and 3D point cloud in-
formation. Unlike existing methods that either use multi-
stage pipelines or hold sensor and dataset-speciﬁc assump-
tions, PointFusion is conceptually simple and application-
agnostic. The image data and the raw point cloud data are
independently processed by a CNN and a PointNet archi-
tecture, respectively. The resulting outputs are then com-
bined by a novel fusion network, which predicts multiple
3D box hypotheses and their conﬁdences, using the input
3D points as spatial anchors. We evaluate PointFusion on
two distinctive datasets: the KITTI dataset that features
driving scenes captured with a lidar-camera setup, and the
SUN-RGBD dataset that captures indoor environments with
RGB-D cameras. Our model is the ﬁrst one that is able to
perform better or on-par with the state-of-the-art on these
diverse datasets without any dataset-speciﬁc model tuning.

## introduction
We focus on 3D object detection, which is a fundamen-
tal computer vision problem impacting most autonomous
robotics systems including self-driving cars and drones.
The goal of 3D object detection is to recover the 6 DoF
pose and the 3D bounding box dimensions for all objects
of interest in the scene. While recent advances in convolu-
tional neural networks have enabled accurate 2D detection
in complex environments [25, 22, 19], the 3D object detec-
tion problem still remains an open challenge. Methods for
3D box regression from a single image, even including re-
cent deep learning methods such as [21, 36], still have rel-
atively low accuracy especially in depth estimates at longer
ranges. Hence, many current real-world systems either use
stereo or augment their sensor stack with lidar and radar.
The lidar-radar mixed-sensor setup is particularly popular
in self-driving cars and is typically handled by a multi-
stage pipeline, which preprocesses each sensor modality
separately and then performs a late fusion or decision-level
∗Work done as an intern at Zoox, Inc.
Figure 1. Sample 3D object detection results of our PointFusion
model on the KITTI dataset [10] (left) and the SUN-RGBD [30]
dataset (right). In this paper, we show that our simple and generic
sensor fusion method is able to handle datasets with distinctive
environments and sensor types and perform better or on-par with
state-of-the-art methods on the respective datasets.
fusion step using an expert-designed tracking system such
as a Kalman ﬁlter [4, 7]. Such systems make simplifying
assumptions and make decisions in the absence of context
from other sensors. Inspired by the successes of deep learn-
ing for handling diverse raw sensory input, we propose an
early fusion model for 3D box estimation, which directly
learns to combine image and depth information optimally.
Various combinations of cameras and 3D sensors are widely
used in the ﬁeld, and it is desirable to have a single algo-
rithm that generalizes to as many different problem settings
as possible. Many real-world robotic systems are equipped
with multiple 3D sensors: for example, autonomous cars
often have multiple lidars and potentially also radars. Yet,
current algorithms often assume a single RGB-D cam-
era [32, 16], which provides RGB-D images, or a single
lidar sensor [3, 18], which allows the creation of a local
front view image of the lidar depth and intensity readings.
Many existing algorithms also make strong domain-speciﬁc
assumptions. For example, MV3D [3] assumes that all ob-
jects can be segmented in a top-down 2D view of the point
cloud, which works for the common self-driving case but
does not generalize to indoor scenes where objects can be
placed on top of each other. Furthermore, the top-down
view approach tends to work well for objects such as cars,
but does not for other key object classes such as pedestrians
or cyclists. Unlike the above approaches, our architecture is
1
arXiv:1711.10871v2  [cs.CV]  25 Aug 2018

designed to be domain-agnostic and agnostic to the place-
ment, type, and number of 3D sensors. As such, it is generic
and can be used for a variety of robotics applications.
In designing such a generic model, we need to solve the
challenge of combining the heterogeneous image and 3D
point cloud data. Previous work addresses this challenge
by directly transforming the point cloud to a convolution-
friendly form.
This includes either projecting the point
cloud onto the image [11] or voxelizing the point cloud [32,
17]. Both of these operations involve lossy data quantiza-
tion and require special models to handle sparsity in the
lidar image [34] or in voxel space [27]. Instead, our so-
lution retains the inputs in their native representation and
processes them using heterogeneous network architectures.
Speciﬁcally for the point cloud, we use a variant of the re-
cently proposed PointNet [23] architecture, which allows us
to process the raw points directly.
Our deep network for 3D object box regression from im-
ages and sparse point clouds has three main components:
an off-the-shelf CNN [13] that extracts appearance and ge-
ometry features from input RGB image crops, a variant of
PointNet [23] that processes the raw 3D point cloud, and a
fusion sub-network that combines the two outputs to predict
3D bounding boxes. This heterogeneous network architec-
ture, as shown in Fig. 2, takes full advantage of the two
data sources without introducing any data processing bi-
ases. Our fusion sub-network features a novel dense 3D box
prediction architecture, in which for each input 3D point,
the network predicts the corner locations of a 3D box rela-
tive to the point. The network then uses a learned scoring
function to select the best prediction. The method is in-
spired by the concept of spatial anchors [25] and dense pre-
diction [15]. The intuition is that predicting relative spatial
locations using input 3D points as anchors reduces the vari-
ance of the regression objective comparing to an architec-
ture that directly regresses the 3D location of each corner.
We demonstrate that the dense prediction architecture out-
performs the architecture that regresses 3D corner locations
directly by a large margin.
We evaluate our model on two distinctive 3D object de-
tection datasets.
The KITTI dataset [10] focuses on the
outdoor urban driving scenario in which pedestrians, cy-
clists, and cars are annotated in data acquired with a camera-
lidar system.
The SUN-RGBD dataset [30] is recorded
via RGB-D cameras in indoor environments, with more
than 700 object categories. We show that by combining
PointFusion with an off-the-shelf 2D object detector [25],
we get comparable or better 3D object detections than the
state-of-the-art methods designed for KITTI [3] and SUN-
RGBD [16, 32, 26]. To the best of our knowledge, our
model is the ﬁrst to achieve competitive results on these
very different datasets, proving its general applicability.

## method
bathtub
bed
bkshelf
chair
desk
dresser
n. stand
sofa
table
toilet
mAP
runtime
DSS [32]
44.2
78.8
11.9
61.2
20.5
6.4
15.4
53.5
50.3
78.9
42.1
19.6s
COG [26]
58.26
63.67
31.80
62.17
45.19
15.47
27.36
51.02
51.29
70.07
47.63
10-30m
Lahoud et al. [16]
43.45
64.48
31.40
48.27
27.93
25.92
41.92
50.39
37.02
80.4
45.12
4.2s
rgbd
36.78
60.44
20.48
46.11
12.71
14.35
30.19
46.11
24.80
81.79
38.17
0.9s
Ours-dense-no-im
28.68
66.55
22.43
50.70
13.86
14.20
25.69
49.74
23.63
83.35
39.24
0.4s
Ours-ﬁnal
37.26
68.57
37.69
55.09
17.16
23.95
32.33
53.83
31.03
83.80
45.38
1.3s
4.6. Evaluation on SUN-RGBD
Comparison with our baselines As shown in Table 3, ﬁnal
is our best model variant and outperforms the rgb-d baseline
by 6% mAP. This is a much smaller gap than in the KITTI
dataset, which shows that the CNN performs well when it
is given dense depth information (rgb-d cameras provide a
depth measurement for every rgb image pixel). Further-
more, rgb-d performs roughly on-par with our lidar-only
model, which demonstrates the effectiveness of our Point-
Net subcomponent and the dense architecture.
Comparison with other methods We compare our model
with three approaches from the current state of the art. Deep
Sliding Shapes (DSS) [32] generates 3D regions using a
proposal network and then processes them using a 3D con-
volutional network, which is prohibitively slow. Our model
outperforms DSS by 3% mAP while being 15 times faster.
Clouds of Oriented Gradients (COG) by Ren et al. [26] ex-
ploits the scene layout information and performs exhaustive
3D bounding box search, which makes it run in the tens of
minutes. In contrast, PointFusion only uses the 3D points
that project to a 2D detection box and still outperforms
COG on 6 out of 10 categories, while approaching its over-
all mAP performance. PointFusion also compares favorably
to the method of Lahoud et al. [16], which uses a multi-
stage pipeline to perform detection, orientation regression
and object reﬁnement using object relation information.
Our method is simpler and does not make environment-
speciﬁc assumptions, yet it obtains a marginally better mAP
while being 3 times faster. Note that for simplicity, our
evaluation protocol passes all 300 2D detector proposals for
each image to PointFusion. Since our 2D detector takes
only 0.2s per frame, we can easily get sub-second evalua-
tion times simply by discarding detection boxes with scores
below a threshold, with minimal performance losses.
Qualitative results Fig. 6 shows some sample detection re-
sults from the ﬁnal model on 19 object categories.
Our
model is able to detect objects of various scales, orienta-
tions, and positions. Note that because our model does not
use a top-down view representation, it is able to detect ob-
jects that are on top of other objects, e.g., pillows on top of a
bed. Failure modes include errors caused by objects which
are only partially visible in the image or from cascading er-
rors from the 2D detector.
Figure 6. Sample 3D detection results from our ﬁnal model on the
SUN-RGBD test set. Our modle is able to detect objects of vari-
able scales, orientations, and even objects on top of other objects.
Detections with score > 0.7 are visualized.
5. Conclusions and Future Work
We present the PointFusion network, which accurately es-
timates 3D object bounding boxes from image and point
cloud information. Our model makes two main contribu-
tions. First, we process the inputs using heterogeneous net-
work architectures. The raw point cloud data is directly han-
dled using a PointNet model, which avoids lossy input pre-
processing such as quantization or projection. Second, we
introduce a novel dense fusion network, which combines
the image and point cloud representations. It predicts multi-
ple 3D box hypotheses relative to the input 3D points, which
serve as spatial anchors, and automatically learns to select
the best hypothesis. We show that with the same architec-
ture and hyper-parameters, our method is able to perform
on par or better than methods that hold dataset and sensor-
speciﬁc assumptions on two drastically different datasets.
Promising directions of future work include combining the
2D detector and the PointFusion network into a single end-
to-end 3D detector, as well as extending our model with a
temporal component to perform joint detection and tracking
in video and point cloud streams.

## experiments
We focus on answering two questions: 1) does PointFusion
perform well on different sensor conﬁgurations and envi-
ronments compared to models that hold dataset or sensor-
speciﬁc assumptions, and 2) do the dense prediction ar-
chitectures perform better than architectures that directly
regress the spatial locations. To answer 1), we compare our
model against the state of the art on two distinctive datasets,
the KITTI dataset [10] and the SUN-RGBD dataset [30]. To
answer 2), we conduct ablation studies for the model varia-
tions described in Sec. 3.
4.1. Datasets
KITTI
The KITTI dataset [10] contains both 2D and 3D
annotations of cars, pedestrians, and cyclists in urban driv-
ing scenarios. The sensor conﬁguration includes a wide-
angle camera and a Velodyne HDL-64E LiDAR. The ofﬁ-
cial training set contains 7481 images. We follow [3] and
split the dataset into training and validation sets, each con-
taining around half of the entire set. We report model per-
formance on the validation set for all three object categories.
SUN-RGBD
The SUN-RGBD dataset [30] focuses on in-
door environments, in which as many as 700 object cat-
egories are labeled. The dataset is collected via different
types of RGB-D cameras with varying resolutions.
The
training and testing sets contain 5285 and 5050 images, re-
spectively. We report model performance on the testing set.
We follow the KITTI training and evaluation setup with
one exception. Because SUN-RGBD does not have a di-
rect mapping between the 2D and 3D object annotations, for
each 3D object annotation, we project the 8 corners of the
3D box to the image plane and use the minimum enclosing
2D bounding box as training data for the 2D object detec-
tor and our models. We report 3D detection performance of
our models on the same 10 object categories as in [26, 16].
Because these 10 object categories contain relatively large
objects, we also show detection results on the 19 categories
from [32] to show our model’s performance on objects of
all sizes. We use the same set of hyper-parameters in both
KITTI and SUN-RGBD.
4.2. Metrics
We use the 3D object detection average precision metric
(AP3D) in our evaluation. A predicted 3D box is a true pos-
itive if its 3D intersection-over-union ratio (3D IoU) with
a ground truth box is over a threshold. We compute a per-
class precision-recall curve and use the area under the curve
as the AP measure.
We use the ofﬁcial evaluation protocol for the KITTI
dataset, i.e., the 3D IoU thresholds are 0.7, 0.5, 0.5
for Car, Cyclist, Pedestrian respectively. Fol-
lowing [30, 26, 16], we use a 3D IoU threshold 0.25 for
all classes in SUN-RGBD.
4.3. Implementation Details
Architecture We use a ResNet-50 pretrained on Ima-
geNet [29] for processing the input image crop. The output
feature vector is produced by the ﬁnal residual block (block-
4) and averaged across the feature map locations. We use
the original implementation of PointNet with all batch nor-
malization layers removed. For the 2D object detector, we
use an off-the-shelf Faster-RCNN [25] implementation [14]
pretrained on MS-COCO [20] and ﬁne-tuned on the datasets
used in the experiments. We use the same set of hyper-
parameters and architectures in all of our experiments.
Training and evaluation During training, we randomly re-
size and shift the ground truth 2D bounding boxes by 10%
along their x and y dimensions. These boxes are used as the
input crops for our models. At evaluation time, we use the
output of the trained 2D detector. For each input 2D box,
we crop and resize the image to 224 × 224 and randomly
sample a maximum of 400 input 3D points in both training
and evaluation. At evaluation time, we apply PointFusion to
the top 300 2D detector boxes for each image. The 3D de-
tection score is computed by multiplying the 2D detection
score and the predicted 3D bounding box scores.

Figure 4. Qualitative results on the KITTI dataset. Detection results are shown in transparent boxes in 3D and wireframe boxes in images.
3D box corners are colored to indicate direction: red is front and yellow is back. Input 2D detection boxes are shown in red. The top two
rows compare our ﬁnal lidar + rgb model with a lidar only model dense-no-im. The bottom row shows more results from the ﬁnal model.
Detections with score > 0.5 are visualized.
4.4. Architectures
We compare 6 model variants to showcase the effectiveness
of our design choices.
• ﬁnal uses our dense prediction architecture and the unsu-
pervised scoring function as described in Sec. 3.3.
• dense implements the dense prediction architecture with
a supervised scoring function as described in Sec. 3.3.
• dense-no-im is the same as dense but takes only the point
cloud as input.
• global is a baseline model that directly regresses the 8
corner locations of a 3D box, as shown in Fig. 2D.
• global-no-im is the same as the global but takes only the
point cloud as input.
• rgb-d replaces the PointNet component with a generic
CNN, which takes a depth image as input. We use it as
an example of a homogeneous architecture baseline. 1
1We have experimented with a number of such architectures and found
that achieving a reasonable performance requires non-trivial effort. Here
4.5. Evaluation on KITTI

## related_work
We give an overview of the previous work on 6-DoF object
pose estimation, which is related to our approach.
Geometry-based methods A number of methods focus on
estimating the 6-DoF object pose from a single image or
an image sequence. These include keypoint matching be-
tween 2D images and their corresponding 3D CAD mod-
els [1, 5, 37], or aligning 3D-reconstructed models with
ground-truth models to recover the object poses [28, 9].
Gupta et al. [12] propose to predict a semantic segmenta-
tion map as well as object pose hypotheses using a CNN and
then align the hypotheses with known object CAD models
using ICP. These types of methods rely on strong category
shape priors or ground-truth object CAD models, which
makes them difﬁcult to scale to larger datasets. In contrary,
our generic method estimates both the 6-DoF pose and spa-
tial dimensions of an object without object category knowl-
edge or CAD models.
3D box regression from images The recent advances in
deep models have dramatically improved 2D object detec-
tion, and some methods propose to extend the objectives
with the full 3D object poses. [33] uses R-CNN to propose
2D RoI and another network to regress the object poses.
[21] combines a set of deep-learned 3D object parameters
and geometric constraints from 2D RoIs to recover the full
3D box. Xiang et al. [36, 35] jointly learns a viewpoint-
dependent detector and a pose estimator by clustering 3D
voxel patterns learned from object models. Although these
methods excel at estimating object orientations, localizing
the objects in 3D from an image is often handled by impos-
ing geometric constraints [21] and remains a challenge for
lack of direct depth measurements. One of the key contri-
butions of our model is that it learns to effectively combine
the complementary image and depth sensor information.
3D box regression from depth data Newer studies have
proposed to directly tackle the 3D object detection problem
in discretized 3D spaces. Song et al. [31] learn to classify
3D bounding box proposals generated by a 3D sliding win-
dow using synthetically-generated 3D features. A follow-
up study [32] uses a 3D variant of the Region Proposal Net-
work [25] to generate 3D proposals and uses a 3D ConvNet
to process the voxelized point cloud. A similar approach by
Li et al. [17] focuses on detecting vehicles and processes
the voxelized input with a 3D fully convolutional network.
However, these methods are often prohibitively expensive
because of the discretized volumetric representation. As
an example, [32] takes around 20 seconds to process one
frame. Other methods, such as VeloFCN [18], focus on a
single lidar setup and form a dense depth and intensity im-
age, which is processed with a single 2D CNN. Unlike these
methods, we adopt the recently proposed PointNet [23] to
process the raw point cloud. The setup can accommodate
multiple depth sensors, and the time complexity scales lin-

n x 64
1 x 1024
n x 3136
ResNet
1 x 2048
point-wise feature
global feature
block-4
3D box corner 
offsets
n x 8 x 3
score
n x 1
[512->128->128]
dz
dx
dy
point-wise offsets to 
each corner
Predicted 
3D bounding box
[512->128->128]
PointNet 
Dense Fusion (final model)
Global Fusion (baseline model)
MLP
MLP
1 x 3072
argmax(score)
3D box corner 
locations
1 x 8 x 3
RGB image 
(RoI-cropped)
3D point cloud 
(n x 3)
[·, ·, ·]
[·, ·]
fusion feature
fusion feature
(A)
(B)
(D)
(C)
(E)
Figure 2. An overview of the dense PointFusion architecture. PointFusion has two feature extractors: a PointNet variant that processes
raw point cloud data (A), and a CNN that extracts visual features from an input image (B). We present two fusion network formulations: a
vanilla global architecture that directly regresses the box corner locations (D), and a novel dense architecture that predicts the spatial offset
of each of the 8 corners relative to an input point, as illustrated in (C): for each input point, the network predicts the spatial offset (white
arrows) from a corner (red dot) to the input point (blue), and selects the prediction with the highest score as the ﬁnal prediction (E).
early with the number of range measurements irrespective
of the spatial extent of the 3D scene.
2D-3D fusion Our paper is most related to recent methods
that fuse image and lidar data. MV3D by Chen et al. [3]
generates object detection proposals in the top-down lidar
view and projects them to the front-lidar and image views,
fusing all the corresponding features to do oriented box re-
gression. This approach assumes a single-lidar setup and
bakes in the restrictive assumption that all objects are on
the same spatial plane and can be localized solely from a
top-down view of the point cloud, which works for cars but
not pedestrians and bicyclists. In contrast, our method has
no scene or object-speciﬁc limitations, as well as no limita-
tions on the kind and number of depth sensors used.
3. PointFusion
In this section, we describe our PointFusion model, which
performs 3D bounding box regression from a 2D image
crop and a corresponding 3D point cloud that is typically
produced by lidar sensors (see Fig. 1). When our model is
combined with a state of the art 2D object detector supply-
ing the 2D object crops, such as [25], we get a complete 3D
object detection system. We leave the theoretically straight-
forward end-to-end model to future work since we already
get state of the art results with this simpler two-stage setup.
In addition, the current setup allows us to plug in any state-
of-the-art detector without modifying the fusion network.
PointFusion has three main components:
a variant of
the PointNet network that extracts point cloud features
(Fig. 2A), a CNN that extracts image appearance features
(Fig. 2B), and a fusion network that combines both and out-
puts a 3D bounding box for the object in the crop. We de-
scribe two variants of the fusion network: a vanilla global
architecture (Fig. 2C) and a novel dense fusion network
(Fig. 2D), in which we use a dense spatial anchor mech-
anism to improve the 3D box predictions and two scoring
functions to select the best predictions. Below, we go into
the details of our point cloud and fusion sub-components.
3.1. Point Cloud Network
We process the input point clouds using a variant of the
PointNet architecture by Qi et al. [23]. PointNet pioneered
the use of a symmetric function (max-pooling) to achieve
permutation invariance in the processing of unordered 3D
point cloud sets. The model ingests raw point clouds and
learns a spatial encoding of each point and also an aggre-
gated global point cloud feature. These features are then
used for classiﬁcation and semantic segmentation.
PointNet has many desirable properties: it processes the raw
points directly without lossy operations like voxelization or
projection, and it scales linearly with the number of input
points. However, the original PointNet formulation cannot
be used for 3D regression out of the box. Here we describe
two important changes we made to PointNet.
No BatchNorm
Batch normalization has become indis-
pensable in modern neural architecture design as it effec-
tively reduces the covariance shift in the input features. In
the original PointNet implementation, all fully connected
layers are followed by a batch normalization layer. How-
ever, we found that batch normalization hampers the 3D
bounding box estimation performance. Batch normaliza-
tion aims to eliminate the scale and bias in its input data, but
for the task of 3D regression, the absolute numerical values
of the point locations are helpful. Therefore, our PointNet
variant has all batch normalization layers removed.

Input point cloud
 Input point cloud 
(transformed)
Camera frame
Input RoI
Rc
x
y
z
Figure 3. During input preprocessing, we compute a rotation Rc
to canonicalize the point cloud inside each RoI.
Input normalization
As described in the setup, the cor-
responding 3D point cloud of an image bounding box is ob-
tained by ﬁnding all points in the scene that can be pro-
jected onto the box. However, the spatial location of the
3D points is highly correlated with the 2D box location,
which introduces undesirable biases.
PointNet applies a
Spatial Transformer Network (STN) to canonicalize the in-
put space. However, we found that the STN is not able to
fully correct these biases. We instead use the known camera
geometry to compute the canonical rotation matrix Rc. Rc
rotates the ray passing through the center of the 2D box to
the z-axis of the camera frame. This is illustrated in Fig. 3.
3.2. Fusion Network
The fusion network takes as input an image feature ex-
tracted using a standard CNN and the corresponding point
cloud feature produced by the PointNet sub-network. Its
job is to combine these features and to output a 3D bound-
ing box for the target object. Below we propose two fusion
network formulations, a vanilla global fusion network, and
a novel dense fusion network.
Global fusion network
As shown in Fig. 2C