# Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation

> 2019 · id: W2909314588 · arXiv: 1901.02970 · pdf: https://arxiv.org/pdf/1901.02970 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
The goal of this paper is to estimate the 6D pose and
dimensions of unseen object instances in an RGB-D im-
age. Contrary to “instance-level” 6D pose estimation tasks,
our problem assumes that no exact object CAD models are
available during either training or testing time. To han-
dle different and unseen object instances in a given cate-
gory, we introduce Normalized Object Coordinate Space
(NOCS)—a shared canonical representation for all possi-
ble object instances within a category. Our region-based
neural network is then trained to directly infer the corre-
spondence from observed pixels to this shared object repre-
sentation (NOCS) along with other object information such
as class label and instance mask. These predictions can be
combined with the depth map to jointly estimate the metric
6D pose and dimensions of multiple objects in a cluttered
scene.
To train our network, we present a new context-
aware technique to generate large amounts of fully anno-
tated mixed reality data. To further improve our model and
evaluate its performance on real data, we also provide a
fully annotated real-world dataset with large environment
and instance variation. Extensive experiments demonstrate
that the proposed method is able to robustly estimate the
pose and size of unseen object instances in real environ-
ments while also achieving state-of-the-art performance on
standard 6D pose estimation benchmarks.

## introduction
Detecting objects, and estimating their 3D position, ori-
entation and size is an important requirement in virtual
and augmented reality (AR), robotics, and 3D scene un-
derstanding. These applications require operation in new
environments that may contain previously unseen object
instances.
Past work has explored the instance-level 6D
pose estimation problem [37, 46, 27, 51, 6, 28] where ex-
act CAD models and their sizes are available beforehand.
 https://hughw19.github.io/NOCS_CVPR2019
Figure 1. We present a method for category-level 6D pose and
size estimation of multiple unseen objects in an RGB-D image. A
novel normalized object coordinate space (NOCS) representation
(color-coded in (b)) allows us to consistently deﬁne 6D pose at the
category-level. We obtain the full metric 6D pose (axes in (c)) and
the dimensions (red bounding boxes in (c)) for unseen objects.
Unfortunately, these techniques cannot be used in general
settings where the vast majority of the objects have never
been seen before and have no known CAD models. On
the other hand, category-level 3D object detection meth-
ods [43, 36, 9, 34, 49, 12] can estimate object class la-
bels and 3D bounding boxes without requiring exact CAD
models. However, the estimated 3D bounding boxes are
viewpoint-dependent and do not encode the precise orien-
tation of objects. Thus, both these classes of methods fall
short of the requirements of applications that need the 6D
pose and 3 non-uniform scale parameters (encoding dimen-
sions) of unseen objects.
In this paper, we aim to bridge the gap between these two
families of approaches by presenting, to our knowledge, the
ﬁrst method for category-level 6D pose and size estima-
tion of multiple objects—a challenging problem for novel
object instances. Since we cannot use CAD models for un-
seen objects, the ﬁrst challenge is to ﬁnd a representation
that allows deﬁnition of 6D pose and size for different ob-
jects in a particular category. The second challenge is the
unavailability of large-scale datasets for training and test-
ing. Datasets such as SUN RGB-D [41] or NYU v2 [40]
lack annotations for precise 6D pose and size, or do not
contain table-scale object categories—exactly the types of
objects that arise in table-top or desktop manipulation tasks
for which knowing the 6D pose and size would be useful.
1
arXiv:1901.02970v2  [cs.CV]  23 Jun 2019

To address the representation challenge, we formulate
the problem as ﬁnding correspondences between object pix-
els to normalized coordinates in a shared object descrip-
tion space (see Section 3).
We deﬁne a shared space
called the Normalized Object Coordinate Space (NOCS)
in which all objects are contained within a common normal-
ized space, and all instances within a category are consis-
tently oriented. This enables 6D pose and size estimation,
even for unseen object instances. At the core of our method
is a convolutional neural network (CNN) that jointly esti-
mates the object class, instance mask, and a NOCS map
of multiple objects from a single RGB image. Intuitively,
the NOCS map captures the normalized shape of the visible
parts of the object by predicting dense correspondences be-
tween object pixels and the NOCS. Our CNN estimates the
NOCS map by formulating it either as a pixel regression or
classiﬁcation problem. The NOCS map is then used with
the depth map to estimate the full metric 6D pose and size
of the objects using a pose ﬁtting method.
To address the data challenge, we introduce a spatially
context-aware mixed reality method to automatically gener-
ate large amounts of data (275K training, 25K testing) com-
posed of realistic-looking synthetic objects from ShapeNet-
Core [8] composited with real tabletop scenes. This ap-
proach allows the automatic generation of realistic data with
object clutter and full ground truth annotations for class la-
bel, instance mask, NOCS map, 6D pose, and size. We
also present a real-world dataset for training and testing
with 18 different scenes and ground truth 6D pose and size
annotations for 6 object categories, and in total 42 unique
instances. To our knowledge, ours is the largest and most
comprehensive training and testing datasets for 6D pose and
size, and 3D object detection tasks.
Our method uses input from a commodity RGB-D sen-
sor and is designed to handle both symmetric and asymmet-
ric objects, making it suitable for many applications. Fig-
ure 1 shows examples of our method operating on a tabletop
scene with multiple objects unseen during training. In sum-
mary, the main contributions of this work are:
• Normalized Object Coordinate Space (NOCS), a uni-
ﬁed shared space that allows different but related ob-
jects to have a common reference frame enabling 6D
pose and size estimation of unseen objects.
• A CNN that jointly predicts class label, instance mask,
and NOCS map of multiple unseen objects in RGB im-
ages. We use the NOCS map together with the depth
map in a pose ﬁtting algorithm to estimate the full met-
ric 6D pose and dimensions of objects.
• Datasets:
A spatially context-aware mixed reality
technique to composite synthetic objects within real
images allowing us to generate a large annotated
dataset to train our CNN. We also present fully anno-
tated real-world datasets for training and testing.

## method
Figure 3 shows our method for 6D pose and size estima-
tion of multiple previously unseen objects from an RGB-D
image. A CNN predicts class labels, masks, and NOCS
maps of objects. We then use the NOCS map and the depth
map to estimate the metric 6D pose and size of objects.
5.1. NOCS Map Prediction CNN
The goal of our CNN is to estimate class labels, instance
masks, and NOCS maps of objects based purely on RGB
images.
We build upon the region-based Mask R-CNN
framework [23] since it has demonstrated state-of-the-art
performance on 2D object detection and instance segmen-
tation tasks, is modular and ﬂexible, fast, and can easily be
augmented to predict NOCS maps as described below.
5.1.1
NOCS Map Head
Mask R-CNN builds upon the Faster R-CNN architec-
ture [38] and consists of two modules—a module to propose
regions potentially containing objects, and a detector to de-
tect and classify objects within regions. Additionally, it also
predicts the instance masks of objects within the regions.
Figure 5. NOCS map head architecture. We add three additional
heads to the Mask R-CNN architecture to predict the x, y, z coor-
dinates of the NOCS map (colored boxes). These heads can either
be used for direct pixel regression or classiﬁcation (best). We use
ReLU activation and 3×3 convolutions.
Our main contribution is the addition of 3 head architec-
tures to Mask R-CNN for predicting the x, y, z components
of the NOCS maps (see Figure 5). For each proposed region
of interest (ROI), the output of a head is of size 28×28×N,
where N is the number of categories and each category con-
taining the x (or y, z) coordinates for all detected objects in
that category. Similar to the mask head, we use the object
category prior to look up the corresponding prediction chan-
nel during testing. During training, only the NOCS map

component from the ground truth object category is used
in the loss function. We use a ResNet50 [25] backbone to-
gether with Feature Pyramid Network (FPN).
Regression vs. Classiﬁcation:
To predict the NOCS
map, we can either regress each pixel value or treat it as a
classiﬁcation problem by discretizing the pixel values (de-
noted by (B) in Figure 5). Direct regression is presumably a
harder task with the potential to introduce instability during
training. Similarly, pixel classiﬁcation with large number
of classes (e.g., B = 128, 256) could introduce more pa-
rameters making training even more challenging than direct
regression. Our experiments revealed that pixel classiﬁca-
tion with B = 32 performed better than direct regression.
Loss Function: The class, box, and mask heads of our
network use the same loss functions as described in [23].
For the NOCS map heads, we use two loss functions: a
standard softmax loss function for classiﬁcation, and the
following soft L1 loss function for regression which makes
learning more robust.
L(y, y∗) = 1
n
(
5 (y −y∗)2,
|y −y∗| ≤0.1
|y −y∗| −0.05,
|y −y∗| > 0.1 ,
∀y ∈N, y∗∈Np,
where y ∈R3 is the ground truth NOCS map pixel value,
y∗is the predicted NOCS map pixel value, n is the number
of mask pixels inside the ROI, I and Ip are the ground truth
and predicted NOCS maps.
Object Symmetry:
Many common household objects
(e.g., bottle) exihibit symmetry about an axis. Our NOCS
representation does not take symmetries into account which
resulted in large errors for some object classes. To mit-
igate this issue, we introduce a variant of our loss func-
tion that takes symmetries into account.
For each cate-
gory in our training data, we deﬁne an axis of symme-
try. Pre-deﬁned rotations about this axis result in NOCS
maps that produce identical loss function values. For in-
stance, a cuboid with a square top has a vertical symmetry
axis. Rotation by angles, θ = {0◦, 90◦, 180◦, 270◦} on this
axis leads to identical NOCS maps and therefore have the
same loss. For non-symmetric objects, θ = 0◦is unique.
We found that a |θ| ≤6 is enough to handle most sym-
metric categories. We generate ground truth NOCS maps,
{˜y1, . . . , ˜y|θ|}, that are rotated |θ| times along the symme-
try axis. We then deﬁne our symmetric loss function, Ls as
Ls = mini=1,...,|θ| L (˜yi, y∗) , where y∗denotes the pre-
dicted NOCS map pixel (x, y, z).
Training Protocol:
We initialize the ResNet50 back-
bone, RPN and FPN with the weights trained on 2D in-
stance segmentation task on the COCO dataset[33]. For all
heads, we use the initialization technique proposed in [24].
We use a batch size of 2, initial learning rate of 0.001, and
an SGD optimizer with a momentum of 0.9 and a 1×10−4
weight decay. In the ﬁrst stage of training, we freeze the
ResNet50 weights and only train the layers in the heads, the
RPN and FPN for 10K iterations. In the second stage, we
freeze ResNet50 layers below level 4 and train for 3K iter-
ations. In the ﬁnal stage, we freeze ResNet50 layers below
level 3 for another 70K iterations. When switching to each
stage, we decrease the learning rate by a factor of 10.
5.2. 6D Pose and Size Estimation
Our goal is to estimate the full metric 6D pose and di-
mensions of detected objects by using the NOCS map and
input depth map. To this end, we use the RGB-D camera
intrinsics and extrinsics to align the depth image to color
image. We then apply the predicted object mask to obtain a
3D point cloud Pm of the detected object. We also use the
NOCS map to obtain a 3D representation of Pn. We then
estimate the scales, rotation, and translation that transforms
the Pn to Pm. We use the Umeyama algorithm [47] for this
7 dimensional rigid transformation estimation problem, and
RANSAC [16] for outlier removal. Please see the supple-
mentary materials for qualitative results.
6. Experiments and Results
Metrics: We report results on both 3D object detection,
and 6D pose estimation metrics. To evaluate 3D detection
and object dimension estimation, we use the intersection
over union (IoU) metric with a threshold of 50% [17]. For
6D pose estimation, we report the average precision of ob-
ject instances for which the error is less than m cm for trans-
lation and n◦for rotation similar to [39, 30]. We decouple
object detection from 6D pose evaluation since it gives a
clearer picture of performance. We set a detection thresh-
old of 10% bounding box overlap between prediction and
ground truth to ensure that most objects are included in the
evaluation. For symmetric object categories (bottle, bowl,
and can), we allow the predicted 3D bounding box to freely
rotate around the object’s vertical axis with no penalty. We
perform special processing for the mug category by making
it symmetric when the handle is not visible since it is hard to
judge its pose in such cases, even for humans. We use [52]
to detect handle visibility for CAMERA data and manually
annotate for real data.
Baselines:
Since we know of no other methods for
category-level 6D pose and size estimation, we built our
own baseline to help compare performance. It consists of
the Mask R-CNN network trained on the same data but
without the NOCS map heads. We use the predicted in-
stance mask to obtain a 3D point cloud of the object from
the depth map. We align (using ICP [4]) the masked point
cloud to one randomly chosen model from the correspond-
ing category. For instance-level 6D pose estimation, we
present results that can readily be compared with [51].

Evaluation Data: All our experiments use one or both
of these evaluation datasets:
(1) the CAMERA valida-
tion dataset (CAMERA25), and (2) a 2.75K real dataset
(REAL275) with ground truth annotations. Since real data
is limited, this allows us to investigate performance without
entangling pose estimation and domain generalization.
6.1. Category-Level 6D Pose and Size Estimation
Test on CAMERA25: We report category-level results
for our method with the CNN trained only on the 275K
CAMERA training set (CAMERA*). We test performance
on CAMERA25 which is composed of objects and back-
grounds completely unseen during training. We achieve a
mean average precision (mAP) of 83.9% for 3D IoU at 50%
and an mAP of 40.9% for the (5◦, 5 cm) metric. (5◦, 5 cm)
is a strict metric for estimating 6D pose even for known in-
stances [51, 6, 37]. See Figure 6 for more details.
Figure 6. 3D detection and 6D pose estimation results on CAM-
ERA25 when our network is trained on CAMERA*.
Test on REAL275: We then train our network on a com-
bination of CAMERA*, the real-world dataset (REAL*),
with weak supervision from COCO [33], and evaluate it on
the real-world test set. Since COCO does not have ground
truth NOCS maps, we do not use NOCS loss during train-
ing. We use 20K COCO images that contain instances in
our categories. To balance between these datasets, for each
minibatch we select images from the three data sources,
with a probability of 60% for CAMERA*, 20% for COCO,
and 20% for REAL*. This network is the best performing
model which we use to produce all visual results (Figure 8).
In the real test set, we achieved an mAP of 76.4% for
3D IoU 

## related_work
In this section, we focus on reviewing related work
on category-level 3D object detection, instance-level 6D
pose estimation, category-level 4 DoF pose estimation from
RGB-D images, and different data generation strategies.
Category-Level 3D Object Detection:
One of the
challenges in predicting the 6D pose and size of objects
is localizing them in the scene and ﬁnding their physical
sizes, which can be formulated as a 3D detection prob-
lem [54, 22, 21, 31, 14]. Notable attempts include [43, 55]
who take 3D volumetric data as input to directly detect
objects in 3D. Another line of work [36, 20, 10, 29] pro-
poses to ﬁrst produce 2D object proposals in 2D image
and then project the proposal into 3D space to further re-
ﬁne the ﬁnal 3D bounding box location. The techniques
described above reach impressive 3D detection rates, but
unfortunately solely focus on ﬁnding the bounding volume
of objects and do not predict the 6D pose of the objects.
Instance-Level 6 DoF Pose Estimation: Given its prac-
tical importance, there is a large body of work focusing on
instance-level 6D pose estimation. Here, the task is to in-
fer the 3D location and 3D rotation of objects (no scale),
assuming exact 3D CAD models and size of these objects
are available during training. The state of the art can be
broadly categorized as template matching or object coordi-
nates regression techniques. Template matching techniques
align 3D CAD models to observed 3D point clouds with al-
gorithms such as iterative closest point [4, 53], or use hand
crafted local descriptors to further guide the alignment pro-
cess [26, 11]. This family of techniques often suffer from
inter- and intra-object occlusions, which is typical when we
have only partial scans of objects. The second category of
approaches based on object coordinates regression aim to
regress the object surface position corresponding to each
object pixel. Such techniques have been successfully em-
ployed for body pose estimation [45, 18], camera relocal-
ization [39, 48] and 6D object pose estimation [5].
Both the above approaches need exact 3D models of the
objects during training and test time. Besides the practi-
cal limitation in storing all 3D CAD models or learned ob-
ject coordinate regressors in memory at test time, capturing
high-ﬁdelity and complete 3D models of a very large array
of objects is a challenging task. Although our approach is
inspired by object coordinate regression techniques, it also
signiﬁcantly differs from the above approaches since we no
longer require complete and high-ﬁdelity 3D CAD models
of objects at test time.
Category-Level 4 DoF Pose Estimation:
There has
been some work on category-level pose estimation [20, 42,
19, 35, 7], however they all make simplifying assumptions.
First, these algorithms constrain the rotation prediction to
be only along the gravity direction (only four degrees of
freedom). Second, they focus on a few big room-scale ob-

ject categories (e.g., chairs, sofa, beds or cars) and do not
take object symmetry into account [20, 42, 19]. On the
contrary, we estimate the pose of a variety of hand-scale
objects, which are often much more challenging than the
bigger room-scale objects due to larger pose variation. Our
method also predicts full 6D pose and size without assum-
ing the objects gravity direction. Finally, our method runs
at interactive frame rates (0.5 s per frame), which is signif-
icantly faster than alternative approaches (∼70 s per frame
for [20], 25 mins per frame for [42]).
Training Data Generation:
A major challenge with
training CNNs is the lack of training data with sufﬁcient cat-
egory, instance, pose, clutter, and lighting variation. There
have been several efforts aimed at constructing real-world
datasets containing object labels (e.g., [40, 41, 50]). Unfor-
tunately, these datasets tend to be relatively small, mostly
due to the high cost (time and money) associated with
ground truth annotation. This limitation is a motivator for
other works (e.g., [35, 44, 51]) which generate data that
is exclusively synthetic allowing the generation of large
amounts of perfectly annotated training data at a smaller
cost. For the sake of simplicity, all these datasets ignore a
combination of factors (material, sensor noise, and lighting)
which creates a de-facto domain gap between the synthetic
and real data distributions. To reduce this gap, [13] have
generated datasets that mix real and synthetic data by ren-
dering virtual objects on real backgrounds. While the back-
grounds are realistic, the rendered objects are ﬂying mid-
air and out of context [13], which prevent algorithms from
making use of important contextual cues.
We introduce a new mixed reality method to automati-
cally generate large amounts of data composed of synthetic
renderings of objects and real backgrounds in a context-
aware manner which makes it more realistic. This is sup-
ported by experiments that show that our context-aware
training data enables the model to generalize better to real-
word test data. We also present a real-world dataset to fur-
ther improve learning and for evaluation.
3. Background and Overview
Category-Level 6D Object Pose and Size Estimation:
We focus on the problem of estimating the 3 rotation, 3
translation, and 3 scale parameters (dimensions) of object
instances. The solution to this problem can be visualized
as a tight oriented bounding box around an object (see
Figure 1).
Although not previously observed, these ob-
jects come from known object categories (e.g., camera) for
which training samples have been observed during training.
This task is particularly challenging since we cannot use
CAD models at test time and 6D pose is not well-deﬁned
for unseen objects. To overcome this, we propose a new
representation that deﬁnes a shared object space enabling
the deﬁnition of 6D pose and size for unseen objects.
Figure 2. The Normalized Object Coordinate Space (NOCS) is a
3D space contained within a unit cube. For a given object cate-
gory, we use canonically oriented instances and normalize them to
lie within the NOCS. Each (x, y, z) position in the NOCS is vi-
sualized as an RGB color tuple. We train our network on the per-
spective projection of the NOCS on the RGB image, the NOCS
map (bottom left inset). At test time, the network regresses the
NOCS map which is then used together with the depth map for 6D
pose and size estimation.
Normalized Object Coordinate Space (NOCS): The
NOCS is deﬁned as a 3D space contained within a unit cube
i.e., {x, y, z} ∈[0, 1]. Given a shape collection of known
object CAD models for each category, we normalize their
size by uniformly scaling the object such that the diagonal
of its tight bounding box has a length of 1 and is centered
within the NOCS space (see Figure 2). Furthermore, we
align the object center and orientation consistently across
the same category. We use models from ShapeNetCore [8]
which are already canonicalized for scale, position, and ori-
entation. Figure 2 shows examples of canonicalized shapes
in the camera category. Our representation allows each ver-
tex of a shape to be represented as a tuple (x, y, z) within
the NOCS (color coded in Figure 2).
Our CNN predicts the 2D perspective projection of the
color-coded NOCS coordinates, i.e., a NOCS map (bottom
left in Figure 2). There are multiple ways to interpret a
NOCS map: (1) as a shape reconstruction in NOCS of the
observed parts of the object, or (2) as dense pixel–NOCS
correspondences. Our CNN learns to generalize shape pre-
diction for unseen objects, or alternatively learns to pre-
dict object pixel–NOCS correspondences when trained on
a large shape collection. This representation is more robust
than other approaches (e.g., bounding boxes) since we can
operate even when the object is only partially visible.
Method Overview:
Figure 3 illustrates our approach
which uses an RGB image and a depth map as input. The
CNN estimates the class label, instance mask, and the
NOCS map from only the RGB image. We do not use the
depth map in the CNN because we would like to exploit
existing RGB datasets like COCO, which do not contain
depth, to improve performance. The NOCS map encodes

Figure 3. The inputs to our method are the RGB and depth images of a scene with multiple objects. Our CNN predicts the class label,
instance mask, and NOCS map (color-coded) for each object in the RGB image. We then use the NOCS maps for each object together with
the depth image to obtain the full metric 6D pose and size (axes and tight red bounding boxes), even if the object was never seen before.
the shape and size of the objects in a normalized space. We
can therefore use the depth map at a later stage to lift this
normalized space, and to predict the full metric 6D object
pose and size using robust outlier removal and alignment
techniques.
Our CNN is built upon the Mask R-CNN framework [23]
with improvements to jointly predict NOCS maps in addi-
tion to class labels, and inst

## conclusion
We presented a method for category-level 6D pose
and size estimation of previously unseen object instances.
We presented a new normalized object coordinate space
(NOCS) that allows us to deﬁne a shared space with consis-
tent object scaling and orientation. We propose a CNN that
predicts NOCS maps that can be used with the depth map to
estimate the full metric 6D pose and size of unseen objects
using a pose ﬁtting method. Our approach has important
applications in areas like augmented reality, robotics, and
3D scene understanding.
Acknowledgements: This research was supported by a grant
from Toyota-Stanford Center for AI Research, NSF grant IIS-
1763268, a gift from Google, and a Vannevar Bush Faculty Fel-
lowship. We thank Xin Wang, Shengjun Qin, Anastasia Dubrov-
ina, Davis Rempe, Li Yi, and Vignesh Ganapathi-Subramanian.

A. Implementation and Computation Times
Our network is implemented on Python 3, Keras and
Tensorﬂow. The code is based on MatterPort’s Mask RCNN
implementation[3]. The network uses Feature Pyramid Net-
work (FPN)[32] and a ResNet50 backbone[25].
Our network takes images with a resolution of 640×360
as input. We achieve an interactive rate of around 4 fps on
an Intel Xeon Gold 5122 CPU @ 3.60GHz desktop with a
NVIDIA TITAN Xp. Our implementation takes an average
time of 210 ms for neural network inference and 34 ms for
pose alignment using Umeyama algorithm.
B. Scanned Real Instances
Our real dataset contains 6 object categories and 42 real
scanned unique instances. For each category, we collect 7
instances with 4 for training and validation and the rest 3
for test. Figure 10 show a subset of our instances where one
can see a large intra-category shape variance in the dataset.
The ﬁrst row are instances used in training. The second and
third rows are held-out instances for testing.
C. Result Visualization
Here we provide more visual results of the 6D pose and
size estimation. Due to sufﬁcient training data, our method
achieves very promising performance on CAMERA25 val-
idation set as shown in Figure11. On REAL275 test set, we
still observe decent performance even though the amount of
real training data is small. We observe several failure modes
on real data, including missing detection, wrong classiﬁca-
tion, and inconsistency in predicted coordinate maps.
D. Comparisons on the OccludedLINEMOD
Dataset
The comparison between our method and other existing
methods with 2D projection metric on OccludedLINEMOD
dataset[6] is shown in Figure 13.