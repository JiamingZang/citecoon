# Multi-view self-supervised deep learning for 6D pose estimation in the Amazon Picking Challenge

> 2017 · id: W2963678509 · arXiv: 1609.09475 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
The last two decades have seen a rapid increase in ware-
house automation technologies, satisfying the growing de-
mand of e-commerce and providing faster, cheaper delivery.
Some tasks, especially those involving physical interaction,
are still hard to automate. Amazon, in collaboration with the
academic community, has led a recent effort to deﬁne two
such tasks: 1) picking an instance of a given a product ID out
of a populated shelf and place it into a tote; and 2) stowing
a tote full of products into a populated shelf.
In this paper we describe the vision system of the MIT-
Princeton Team, that took 3rd place in the stowing task
and 4th in the picking task at the 2016 Amazon Picking
Challenge (APC), and provide experiments to validate our
design decisions. Our vision algorithm estimates the 6D
poses of objects robustly under challenging scenarios:
· Cluttered environments: shelves and totes may have
multiple objects and could be arranged as to deceive
vision algorithms (e.g., objects on top of one another).
The authors would like to thank the MIT-Princeton APC team members
for their contribution to this project, and ABB Inc. for hardware and
technical support. This project is also supported by the Google Faculty
Award and Intel Gift Fund to Jianxiong Xiao. Andy Zeng and Daniel Suo
are supported by the Gordon Y.S. Wu fellowship. Shuran Song is supported
by the Facebook fellowship. Kuan-Ting Yu is supported by award [NSF-
IIS-1427050] through the National Robotics Initiative. Alberto Rodriguez
is supported by the Walter Henry Gale (1929) Career Development Profes-
sorship.
Fig. 1.
Top: The MIT-Princeton robotic picking system. Bottom-left: The
gripper mounted with an Intel RealSense camera (outlined in red). Bottom-
right: Predicted 6D object poses from our vision system during the stow-task
ﬁnals of the APC 2016. Each prediction is highlighted with a colored 3D
bounding box.
· Self-occlusion: due to limited camera positions, the
system only sees a partial view of an object.
· Missing data: commercial depth sensors are unreliable
at capturing reﬂective, transparent, or meshed surfaces,
all common in product packaging.
· Small or deformable objects: small objects provide
fewer data points, while deformable objects are difﬁcult
to align to prior models.
· Speed: the total time dedicated to capturing and pro-
cessing visual information is under 20 seconds.
Our approach makes careful use of known constraints
in the task—the list of possible objects and the expected
background. The algorithm ﬁrst segments the object from
a scene by feeding multiple-view images to a deep neural
network and then ﬁts a 3D model to a segmented point cloud
to recover the object’s 6D pose. The deep neural network
provides speed, and in combination with a multiple-view
approach boosts performance in challenging scenarios.
arXiv:1609.09475v3  [cs.CV]  7 May 2017

Fig. 2.
Overview of the vision algorithm. The robot captures color and depth images from 15 to 18 viewpoints of the scene. Each color image is fed into
a fully convolutional network [2] for 2D object segmentation. The result is integrated in 3D. The point cloud will then go through background removal
and then aligned with a pre-scanned 3D model to obtain its 6D pose.
Training a deep neural network for segmentation requires
a large amount of labeled training data. We have developed a
self-supervised training procedure that automatically gener-
ated 130,000 images with pixel-wise category labels of the 39
objects in the APC. For evaluation, we constructed a testing
dataset of over 7,000 manually-labeled images.
In summary, this paper contributes with:
· A robust multi-view vision system to estimate the 6D
pose of objects;
· A self-supervised method that trains deep networks by
automatically labeling training data;
· A benchmark dataset for estimating object poses.
All code, data, and benchmarks are publicly available [3].

## experiments
We evaluate variants of our method in different scenarios
in the benchmark dataset to understand (1) how segmentation
performs under different input modalities and training dataset
sizes and (2) how the full vision system performs.
A. Benchmark Dataset
Our benchmark dataset, ‘Shelf&Tote’, contains over 7,000
RGB-D images spanning 477 (Figure 6) scenes at 640 ×
480 resolution. We collected the data during practice runs
and competition ﬁnals for the APC and manually labeled 6D
object poses and segmentations using our online annotator
(Figure 7). The data reﬂects various challenges found in the
warehouse setting: reﬂective materials, variation in lighting
conditions, partial views, and sensor limitations (noisy and
missing depth) over cluttered environments.

Fig. 7.
The 3D online annotation tool used to label the benchmark. The
drag-and-drop UI allows annotators to navigate in 3D space and manipulate
point clouds with ease. Annotators are instructed to move and rotate a
pre-scanned object model to its ground truth location in a 3D point cloud
generated from RGB-D data. Labeling one object takes about 1 minute.
Tables I and II summarize our experimental results and
highlight the differences in performance over different over-
lapping scene categories:
· cptn: during competition at the APC ﬁnals.
· environment: in an ofﬁce (off); in the APC competition
warehouse (whs).
· task: picking from a shelf bin or stowing from a tote.
· clutter: with multiple objects.
· occlusion: with % of object occluded by another object,
computed from ground truth.
· object properties: with objects that are deformable,
thin, or have no depth from the RealSense F200 camera.
B. Evaluating Object Segmentation
We test several variants of our FCN on object segmenta-
tion to answer two questions: (1) can we leverage both color
and depth segmentation? (2) is more training data useful?
Metrics. We compare the predicted object segmentation from
our trained FCNs against the ground truth segmentation
labels of the benchmark dataset using pixel-wise precision
and recall. Table I displays the mean average F-scores (F =
2 · precision·recall
precision+recall).
Depth for segmentation. We use HHA features [23] to
encode depth information into three channels: horizontal
disparity, height above ground, and angle of local surface
normal with the inferred direction of gravity. We compare
AlexNet trained on this encoding, VGG on RGB data, and
both networks concatenated in Table I.
We ﬁnd that adding depth does not yield any notable
improvements in segmentation performance, which could be
in part due to the noisiness of the depth information from
our sensor. On the other hand, we observe that the FCN
performs signiﬁcantly better when trained on color data, with
the largest disparity for deformable objects and thin objects,
whose textures provide more discriminative power than their
geometric structure.
Size of training data. Deep learning models have seen
signiﬁcant success, especially if given large amounts of
training data. However in our scenario—instance-level ob-
ject segmentation on few object categories—it is not clear
whether such a large dataset is necessary. We create two
new datasets by randomly sampling 1% and 10% of the
original and use them to train two VGG FCNs (Table I).
We conﬁrm marked improvements in F-score across all
benchmark categories going from 1% to 10% to 100% of
training data.
C. Evaluating Pose Estimation
We evaluate several key components of our vision system
to determine whether they increase performance in isolation.
Metrics. We report the percentage of object pose predictions
with error in orientation smaller than 15◦, and the percentage
with error in translation smaller than 5cm. The metric also
recognizes the structural invariance of several objects, some
of which are axially-symmetric (cuboids), radially-symmetric
(bottles, cylinders), or deformable (see web page [3] for
further details). We have observed experimentally that these
bonds of 15◦and 5cm are sufﬁcient for picking with sensor-
guarded motions.
Multi-view information. With multiple views the system
overcomes missing information due to self-occlusions, other-
object occlusions, or clutter. Multi-view information also
alleviates problems with illumination on reﬂective surfaces.
To quantify the effect of the multiple-view system, we test
the full vision system on the benchmark with three different
subsets of camera views:
· [Full] All 15 views for shelf bins a1shelf = {0 ...14} and
all 18 views for the tote a1tote = {0 ...17}.
· [5v-10v] 5 views for shelf a2shelf = {0,4,7,10,14} and 10
views for tote a2tote ={0,2,4,6,8,9,11,13,15,17}, with a
sparse arrangement and a preference for wide-baseline
view angles.
· [1v-2v] 1 view for shelf bins a3shelf = {7} and 2 views
for the tote a3tote ={7,13}.
The viewpoint ids are zero-indexed in row-major order as
depicted in Figure 3. Our results show that multiple views
robustly address occlusion and heavy clutter in the warehouse
setting (Table II [clutter] and [occlusion]). They also present
a clear contrast between the performance of our algorithm
using a single view of the scene, versus multiple views of
the scene (Table II [Full] v.s [1v-2v]).
Denoising. The denoising step described in Section V proves
important for achieving good results. With this turned off, the
accuracy in estimating the translation and rotation decreases
by 6.0% and 4.4% respectively (Table II).
ICP improvements. Without the pre-processing steps to
ICP, we observe a drop in prediction accuracy of 0.9% in
translation and 3.1% in rotation (Table II).
Performance upper bound. We also evaluated how well
the model-ﬁtting part of our algorithm alone performs on
the benchmark by using ground truth segmentation labels
from the benchmark as the performance upper bound.

Fig. 8.
Example results from our vision system. 6D pose predictions are highlighted with a 3D bounding box. For deformable objects (cloth in a,c,i) we
output the center of mass. We additionally illustrate successful pose predictions for objects with missing depth (water bottle, black bin, green sippy cup,
green bowl)
Fig. 9.
Several common failure cases. These include low-conﬁdence predictions due to severe occlusion (missing object labels in m,o,p), completely
incorrect pose predictions due to confusion in texture (m,p,r) or bad initialization (n,q), and model-ﬁtting errors (o).
D. Common Failure Modes
Here we summarize the most common failure modes of
our vision system, which are illustrated in Figure 9:
· The FCN segmentation for objects under heavy occlu-
sion or clutter are likely to be incomplete resulting in
poor pose estimation (Fig. 8.e), or undetected (Fig. 9.m
and p). This happens with more frequency at back of
the bin with poor illumination.
· Objects color textures are confused with each other.
Figure 9.r shows a Dove bar (white box) on top of a
yellow Scotch mail envelope, which combined have a
similar appearance to the outlet plugs.
· Model ﬁtting for cuboid objects often confuses corner
alignments (marker boxes in Fig. 9.o). This inaccuracy,
however, is still within the range of tolerance that the
robot can tolerate thanks to sensor-guarded motions.
Filtering failure modes by conﬁdence score. We compute
a conﬁdence score per object pose prediction that favors high
precision for low recall. Speciﬁcally, the conﬁdence score of
a pose prediction equals the mean value of conﬁdence scores
over all points belonging to the segmentation of the object.
We observe that erroneous poses (especially those due to
partial occlusions) more often have low conﬁdence scores.
The robot system uses this value to target only predictions
with high scores.
We evaluate the usefulness of the conﬁdence scores by
recalling the output of the perception system to only consider
predictions with conﬁdence scores larger than 10% and 70%
respectively (see Table II). These conﬁdence percentages
are important thresholds, because the full robot system,
predictions with < 10% conﬁdence (conf-10, at 78% recall)
are ignored during planning, and prediction with > 70%
conﬁdence (conf-70, at 23% recall) trigger a pick attempt.

## related_work
Vision algorithms for robotic manipulation typically out-
put 2D bounding boxes, pixel-level segmentation [4, 5], or
6D poses [6, 7] of the objects. The choice depends primarily
on manipulation needs. For example, a suction based picker
might have sufﬁcient information with a 2D bounding box or
with a pixel-level segmentation of the object, while a grasper
might require its 6D pose.
Object segmentation. While the 2015 APC winning team
used a histogram backprojection method [8] with manually
deﬁned features [5, 4], recent work in computer vision
has shown that deep learning considerably improves object
segmentation [2]. In this work, we extend the state-of-the-
art deep learning architecture used for image segmentation
to incorporate depth and multi-view information.
Pose estimation. There are two primary approaches for
estimating the 6D pose of an object. The ﬁrst aligns 3D
CAD models to 3D point clouds with algorithms such as
iterative closest point [9]. The second uses more elaborated
local descriptors such as SIFT keypoints [10] for color data
or 3DMatch [11] for 3D data. The former approach is mainly
used with depth-only sensors, in scenarios where lighting
changes signiﬁcantly, or on textureless objects. Highly tex-
tured and rigid objects, on the other hand, beneﬁt from local
descriptors. Existing frameworks such as LINEMOD [12] or
MOPED [13] work well under certain assumptions such as
objects sitting on a table top with good illumination, but
underperform when confronted with the limited visibility,
shadows, and clutter imposed by the APC scenario [14].
Benchmark for 6D pose estimation. To properly evalu-
ate our vision system independent from the larger robotic
system, we have produced a large benchmark dataset with
scenarios from APC 2016 with manual labels for objects’
segmentation and 6D poses. Previous efforts to construct
benchmark datasets include Berkeley’s dataset [15] with a
number of objects from and beyond APC 2015 and Rutgers’s
dataset [16] with semi-automatically labeled data.
III. AMAZON PICKING CHALLENGE 2016
The APC 2016 posed a simpliﬁed version of the general
picking and stowing tasks in a warehouse. In the picking task,
robots sit within a 2x2 meter area in front of a shelf populated
with objects, and autonomously pick 12 desired items and
place them in a tote. In the stowing task, robots pick all 12
items inside a tote and place them in a pre-populated shelf.
Before the competition, teams were provided with a list
of 39 possible objects along with 3D CAD models of the
shelf and tote. At run-time, robots were provided with the
initial contents of each bin in the shelf and a work-order
containing which items to pick. After picking and stowing
the appropriate objects, the system had to report the ﬁnal
contents of both shelf and tote. Competition details are in [1].
IV. SYSTEM DESCRIPTION
Our vision system takes in RGB-D images from multiple
views, and outputs 6D poses and a segmented point cloud
for the robot to complete the picking and stowing tasks.
The camera is compactly integrated in the end-effector of a
6DOF industrial manipulator ABB IRB1600id, and points at
the tip of the ﬁngers (Figure 1). This conﬁguration gives the
robot full controllability of the camera viewpoint, and pro-
vides feedback about grasp or suction success. The camera
of choice is the RealSense F200 RGB-D because its depth
range (0.2m–1.2m) is appropriate for close manipulation, and
because it is a consumer-level range sensor with a decent
amount of ﬂexibility on the data capture process.
Due to the tight integration of the camera, the gripper
ﬁngers, even when fully open, occupy a small portion of the
view frustum. We overcome this limitation by combining
different viewpoints, making use of the accurate forward
kinematic reported by the robot controller.

Fig. 3. Camera viewpoints of the RGB-D frames captured for bins and tote,
and captured color images from 6 selected viewpoints. The 15 viewpoints
of a shelf bin (upper-left) are arranged in a 3x5 grid. The 18 viewpoints of
a tote (upper-right) are arranged in a 3x6 grid.
V. 6D OBJECT POSE ESTIMATION
The algorithm estimates the 6D pose of all objects in
a scene in two phases (Figure 2). First, it segments the
RGB-D point clouds captured from multiple views into
different objects using a deep convolutional neural network.
Second, it aligns pre-scanned 3D models of the identiﬁed
objects to the segmented point clouds to estimate the 6D
pose of each object. Our approach is based on well-known
methods. However, our evaluations show that when used
alone, they are far from sufﬁcient. In this section we present
brief descriptions of these methods followed by in-depth
discussions of how we combine them into a robust vision
system.
A. Object Segmentation with Fully Convolutional Networks
In recent years, ConvNets have made tremendous progress
for computer vision tasks [17, 2]. We leverage these advance-
ments to segment camera data into the different objects in
the scene. More explicitly, we train a VGG architecture [18]
Fully Convolutional Network (FCN) [2] to perform 2D object
segmentation. The FCN takes an RGB image as input and
returns a set of 40 densely labeled pixel probability maps–
one for each of the 39 objects and one for the background–of
the same dimensions as the input image.
Object segmentation using multiple views. Information
from a single camera view and from a given object, is often
limited due to clutter, self-occlusions, and bad reﬂections.
We address the missing information during the model-ﬁtting
phase by combining information from multiple views so that
the object surfaces are more distinguishable. In particular, we
feed the RGB images captured from each viewpoint (18 for
stowing from the tote and 15 for picking from the shelf)
to the trained FCN, which returns a 40 class probability
Fig. 4.
Pose estimation for objects with no depth. 2D object segmentation
results from a fully convolutional network are triangulated between the
different camera views to create a 3D convex hull (green) of the object. For
simplicity, only two camera views (yellow) are illustrated. The centroid and
aspect ratio of the convex hull are used to estimate the geometric center of
the object and its orientation (from a predeﬁned set of expected orientations).
distribution for each pixel in each RGB-D image. After
ﬁltering by the list of expected objects in the scene, we
threshold the probability maps (three standard deviations
above the mean probability across all views) and ignore any
pixels whose probabilities for all classes are below these
thresholds. We then project the segmented masks for each
object class into 3D space and directly combine them into
a single segmented point cloud with the forward kinematic
feedback from the robot arm (note that segmentation for
different object classes can overlap each other).
Reduce noise in point cloud. Fitting pre-scanned models
to the segmented point cloud directly often gives poor
results because of noise from the sensor and noise from the
segmentation. We address this issue in three steps: First, to
reduce sensor noise, we eliminate spatial outliers from the
segmented point cloud, by removing all point farther than
a threshold from its k-nearest neighbors. Second, to reduce
segmentation noise, especially on the object boundaries, we
remove points that lie outside the shelf bin or tote, and those
that are close to a pre-scanned background model. Finally,
we further ﬁlter outlier points from each segmented group of
points by ﬁnding the largest contiguous set of points along
each principal axis (computed via PCA) and remove any
points that are disjoint from this set.
Handle object duplicates. Warehouse shelves commonly
contain multiple instances of the same object. Naively seg-
menting RGB-D data will treat two distinct object with
the same label as the same object. Since we know the
inventory list in the warehouse setting, we know the number
of identical objects we expect in the scene. We make use of
k-means clustering to separate the segmented and aggregated
point cloud into the appropriate number of objects. Each
cluster is then treated independently during the model-ﬁtting
phase of the algorithm.
B. 3D Model-Fitting
We use the iterative closest point (ICP) algorithm [19] on
the segmented point cloud to ﬁt pre-scanned 3D models of

Fig. 5.
To automatically obtain pixel-wise object labels, we separate the
target objects from the background to create an object mask. There are a
2D and a 3D component in this data process. Both use color and depth
information. The 2D pipeline is robust to thin objects and objects with no
depth, while the 3D pipeline is robust to an unstable background.
objects and estimate their poses. The vanilla ICP algorithm,
however, gives nonsensical results in many scenarios. We
describe here several such pitfalls along with our solutions.
Point clouds with non-uniform density. In a typical RGB-
D point cloud, surfaces perpend

## conclusion
Despite tremendous advances in computer vision, many
state-of-the-art well-known approaches are often insufﬁcient
for relatively common scenarios. We describe here two
observations that can lead to improvements in real systems:
Make the most out of every constraint. External constraints
limit what systems can do. Indirectly they also limit the
set of states in which the system can be, which can lead
to opportunities for simpliﬁcations and robustness in the
perception system. In the picking task, each team received
a list of items, their bin assignments, and a model of the
shelf. All teams used the bin assignments to rule out objects
from consideration and the model of the shelf to calibrate
their robots. These optimizations are straightforward and
useful. However, further investigation yields more opportu-
nity. By using these same constraints, we constructed a self-
supervising mechanism to train a deep neural network with
signiﬁcantly more data. As our evaluations show, the volume
of training data is strongly correlated with performance.
Designing robotic and vision systems hand-in-hand. Vi-
sion algorithms are too often designed in isolation. However,
vision is one component of a larger robotic system with
needs and opportunities. Typical computer vision algorithms
operate on single images for segmentation and recognition.
Robotic arms free us from that constraint, allowing us to
precisely fuse multiple views and improve performance in
cluttered environments. Computer vision systems also tend
to have ﬁxed outputs (e.g., bounding boxes or 2D segmenta-
tion maps), but robotic systems with multiple manipulation
strategies can beneﬁt from variety in output. For example,
suction cups and grippers might have different perceptual
requirements. While the former might work more robustly
with a segmented point cloud, the latter often requires

TABLE I
2D OBJECT SEGMENTATION EVALUATION (PIXEL-LEVEL OBJECT CLASSIFICATION AVERAGE % F-SCORES).
environment
task
clutter (# of objects)
occlusion (%)
object-speciﬁc properties
network
all
cptn
off
whs
shelf
tote
1 - 3
4 - 5
6 +
< 5
5 - 30
30 +
dfrm.
no depth
thin
color
45.5
42.7
46.8
44.2
47.7
43.7
53.0
46.0
42.2
49.9
41.4
33.3
54.0
47.9
41.7
color+depth
43.8
41.5
44.8
42.6
45.8
41.9
52.2
43.5
40.0
47.5
39.1
32.6
51.1
47.7
37.2
depth
37.1
35.0
38.6
35.5
39.8
34.9
45.5
37.0
33.5
40.8
33.2
26.3
44.1
42.3
29.1
10% data
20.4
18.8
19.5
21.3
21.7
20.3
36.0
21.6
18.0
21.2
25.5
0.0
41.9
17.2
33.3
1% data
8.0
9.2
7.2
8.8
15.8
6.5
17.3
7.5
6.0
7.7
8.3
7.8
10.1
5.7
3.5
TABLE II
FULL VISION SYSTEM EVALUATION (AVERAGE % CORRECT ROTATION AND TRANSLATION PREDICTIONS FOR OBJECT POSE)
environment
task
clutter (# of objects)
occlusion (%)
object-speciﬁc properties
algorithm
all
cptn
off
whs
shelf
tote
1 - 3
4 - 5
6 +
< 5
5 - 30
30 +
dfrm.
no depth
thin
Full (rot.)
49.8
62.9
52.5
47.1
50.4
49.3
56.1
54.6
45.4
56.9
43.2
33.9
-
55.6
54.7
Full (trans.)
66.1
71.0
66.3
65.9
63.4
68.1
76.7
66.7
61.9
79.4
57.4
27.3
75.4
63.3
58.1
5v-10v (rot.)
44.0
48.6
50.9
35.9
50.9
38.9
53.9
53.1
34.4
47.6
40.0
26.7
-
47.4
42.4
5v-10v (trans.)
58.4
50.0
63.7
52.1
61.0
56.5
69.4
63.0
50.3
66.2
49.8
21.3
54.7
67.3
35.4
1v-2v (rot.)
38.9
60.0
41.1
36.5
45.0
35.3
45.7
45.2
32.7
43.6
33.9
14.8
-
40.9
35.4
1v-2v (trans.)
52.5
50.0
56.3
48.2
53.8
51.8
60.4
56.5
46.7
58.2
47.8
16.7
52.9
55.9
33.3
conf-70 (rot.)
58.3
77.3
65.0
49.0
64.2
53.2
63.8
69.3
49.0
63.7
43.1
36.4
-
64.5
81.6
conf-70 (trans.)
84.5
95.5
84.7
84.2
82.6
86.1
86.2
84.1
83.2
87.1
77.1
72.7
83.1
77.4
85.7
conf-10 (rot.)
55.0
70.8
57.0
52.7
54.9
55.0
58.6
59.3
51.0
59.8
50.0
34.2
-
53.1
60.2
conf-10 (trans.)
76.5
81.2
76.7
76.3
73.4
79.1
80.8
74.4
75.4
84.0
70.0
40.0
78.1
72.0
70.1
no denoise (rot.)
43.8
45.6
46.9
40.6
45.3
42.7
52.0
46.7
39.5
51.1
37.3
28.1
-
48.8
54.1
no denoise (trans.)
61.7
66.4
61.9
61.5
60.4
62.6
74.8
62.7
56.4
76.5
52.9
19.9
75.0
62.3
53.8
no ICP+ (rot.)
48.9
60.8
51.2
46.7
49.1
48.8
55.4
54.1
44.4
55.8
41.9
36.2
-
53.6
52.5
no ICP+ (trans.)
63.0
67.2
63.2
62.9
59.7
65.4
72.1
64.4
59.1
75.2
57.0
24.6
67.3
62.8
53.2
gt seg rot.
63.4
74.4
65.8
60.9
68.1
60.1
69.1
68.8
59.1
67.6
60.0
53.5
-
58.0
74.1
gt seg trans.
88.1
90.4
85.7
90.4
86.9
88.9
88.3
88.0
88.0
90.7
90.3
71.4
90.5
71.5
79.8
knowledge of the object pose and geometry.