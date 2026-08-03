# Recovering 6D Object Pose and Predicting Next-Best-View in the Crowd

> 2015 · id: arxiv:1512.07506 · arXiv: 1512.07506 · pdf: https://arxiv.org/pdf/1512.07506 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Object detection and 6D pose estimation in the crowd
(scenes with multiple object instances, severe foreground
occlusions and background distractors), has become an im-
portant problem in many rapidly evolving technological ar-
eas such as robotics and augmented reality. Single shot-
based 6D pose estimators with manually designed features
are still unable to tackle the above challenges, motivat-
ing the research towards unsupervised feature learning and
next-best-view estimation. In this work, we present a com-
plete framework for both single shot-based 6D object pose
estimation and next-best-view prediction based on Hough
Forests, the state of the art object pose estimator that per-
forms classiﬁcation and regression jointly. Rather than us-
ing manually designed features we a) propose an unsuper-
vised feature learnt from depth-invariant patches using a
Sparse Autoencoder and b) offer an extensive evaluation
of various state of the art features. Furthermore, taking
advantage of the clustering performed in the leaf nodes of
Hough Forests, we learn to estimate the reduction of un-
certainty in other views, formulating the problem of select-
ing the next-best-view. To further improve pose estimation,
we propose an improved joint registration and hypotheses
veriﬁcation module as a ﬁnal reﬁnement step to reject false
detections. We provide two additional challenging datasets
inspired from realistic scenarios to extensively evaluate the
state of the art and our framework. One is related to domes-
tic environments and the other depicts a bin-picking sce-
nario mostly found in industrial settings. We show that our
framework signiﬁcantly outperforms state of the art both on
public and on our datasets.

## introduction
Detection and pose estimation of everyday objects is a
challenging problem arising in many practical applications,
such as robotic manipulation [18], tracking and augmented
reality. Low-cost availability of depth data facilitates pose
estimation signiﬁcantly, but still one has to cope with many
challenges such as viewpoint variability, clutter and oc-
c)
b)
d)
a)
Figure 1: Sample photos from our dataset. a) Scene containing
objects from a supermarket, b) our system’s evaluation on a), c)
Bin-picking scenario with multiple objects stacked on a bin, d)
our system’s evaluation on c).
clusions. When objects have sufﬁcient texture, techniques
based on key-point matching [22, 30] demonstrate good re-
sults, yet when there is a lot of clutter in the scene they
depict many false positive matches which degrades their
performance. Also, holistic template-based techniques pro-
vide superior performance when dealing with texture-less
objects [14], but suffer in cases of occlusions and changes
in lighting conditions, while the performance also degrades
when objects have not signiﬁcant geometric detail. In or-
der to cope with the above issues, a few approaches use
patches [31] or simpler pixel based features [5] along with
a Random Forest classiﬁer. Although promising, these tech-
niques rely on manually designed features which are difﬁ-
cult to make discriminative for the large range of everyday
objects. Last, even when the above difﬁculties are partly
solved, multiple objects present in the scene, occlusions and
distructors can make the detection very challenging from a
single viewpoint, resulting in many ambiguous hypotheses.
When the setup permits, moving the camera can be proved
very beneﬁcial for accuracy increase. The problem is how
1
arXiv:1512.07506v2  [cs.CV]  19 Apr 2016

to select the next best viewpoint, which is crucial for fast
scene understanding.
The above observations motivated us to introduce a com-
plete framework for both single shot-based 6D object pose
estimation and next-best-view prediction in a uniﬁed man-
ner based on Hough Forests, a variant of Random Forest
that performs classiﬁcation and regression jointly [31]. We
adopted a patch-based approach but contrary to [14, 31, 5]
we learn features in an unsupervised way using deep Sparse
Autoencoders. The learnt features are fed to a Hough Forest
[12] to determine object classes and poses using 6D Hough
voting. To estimate the next-best-view, we exploit the capa-
bility of Hough Forests to calculate the hypotheses entropy,
i.e. uncertainty, at leaf nodes. Using this property we can
predict the next-best-viewpoint based on current view hy-
potheses through an object-pose-to-leaf mapping. We are
also taking into account the various occlusions that may ap-
pear from the other views during the next-best-view estima-
tion. Last, for further false positives reduction, we introduce
an improved joint optimization step inspired by [1]. To the
best of our knowledge, there is no other framework jointly
tackling feature learning, classiﬁcation, regression and clus-
tering (for next-best-view) in a patch-based inference strat-
egy.
In order to evaluate our framework, we do an exten-
sive evaluation for single shot detection of various state
of the art features and detection methods, showing that
the proposed approach demonstrates a signiﬁcant improve-
ment compared to the state of the art techniques, on many
challenging publicly available datasets. We also evaluate
our next-best-view selection to various baselines and show
its improved performance, especially in cases of occlu-
sions. To demonstrate more explicitly the advantages of our
framework, we provide an additional dataset consisting of
two realistic scenarios shown in Fig. 1. Our dataset also
reveals the weaknesses of the state of the art techniques to
generalize to realistic scenes. In summary, our main contri-
butions are:
• A complete framework for 6 DoF object detection that
comprises of a) an architecture based on Sparse Autoen-
coders for unsupervised feature learning, b) a 6D Hough
voting scheme for pose estimation and c) a novel active
vision technique based on Hough Forests for estimating
the next-best-view.
• Extensive evaluation of features and detection methods
on several public datasets.
• A new dataset of RGB-D images reﬂecting two usage
scenarios, one representing domestic environments and
the other a bin-picking scenario found in industrial set-
tings.
We provide 3D models of the objects and, to
the best of our knowledge, the ﬁrst fully annotated bin-
picking dataset.

## experiments
The experiments regarding the patch size and feature
evaluation were performed on a validation set of our own
dataset. Object detection accuracy is measured using the
F1-score and is averaged over the whole set of objects.
When comparing with the state of the art methods, we use
the public datasets and the evaluation metrics provided by
the corresponding authors. When evaluating on our own
dataset, we exclude the aforementioned evaluation set.
Patch Size Evaluation A patch in our framework is deﬁned
over 2 parameters: dp is the actual size measured in meters,
and V ×V is the number of cells a patch contains, which can
be considered as the patch resolution. We used six different
conﬁgurations shown in Fig. 5a. The maximum patch size
used was limited to the 2/3 of the smallest object dimen-
sions. The network architecture used for patch-size exper-
iments is 2 layers (the encoder part) of 1000 and 400 hid-
den units respectively. Fig. 5a shows that an increase in
the patch size signiﬁcantly increases the accuracy, while on
the other hand, an increase of the resolution offers a slight
improvement, and that comes at the expense of additional
computational cost. Another important factor is the stride s
during patch extraction. Fig. 5b shows that the smaller the
stride the more accurate the detection becomes.
Feature Evaluation using Hough Forests In order to eval-
uate our unsupervised feature we created 9 different net-
work conﬁgurations to test the effect of both the number
of features and the number of layers on the accuracy. We
used 1-3 layers as the encoder of the network with the last
layer of the encoder forming the feature vector used in the
Hough Forest. We varied the length of this feature vector
to be 100, 400 and 800. When we use 2 layers, the ﬁrst
has 1000 hidden units, while when we use 3 layers, the
ﬁrst two have 1500 and 1000 hidden units respectively. The
patch size used for these experiments is dp = 48mm with
V = 16, creating an input vector of 1024 dimensions. Us-
ing the same Hough Forest conﬁguration, we evaluate three
state of the art features: a) the feature of [31], a variant
of LineMOD[14] designed for Hough Forests, along with
its split function, b) the widely used pixel-tests [5] and c)
K-means clustering, the unsupervised single-layer method
that performed best in [7]2 with 100, 400 and 800 clusters.
Pixel-tests have been conducted inside the area of a patch
for comparison purposes, however in the next subsection
we compare the complete framework of [5] with ours. Re-
sults are shown in Fig. 5c. The 3-layer Sparse Autoencoder
shown the best performance. Regarding the Autoencoder,
we notice that the accuracy increases if more features are
used, but when the network becomes deeper, the difference
diminishes. However, it can be seen that deeper features sig-
niﬁcantly outperform shallower ones. K-means performed
slightly better than single-layer SAE, while pixel-tests had
worse performance. The feature of [31] had on average
worse performance than Autoencoders and K-means, which
is due to low performance on speciﬁc objects of the datasets.
We further provide a visualization of the ﬁlters of the ﬁrst
layer learned by a network with a 3-layer encoder (Fig. 5d).
The ﬁrst two rows are ﬁlters in the RGB channel, where it
can be seen a bias towards the objects used for the evalua-
tion. Filters in the depth channel resemble simple 3D edge
and corner detectors. Last, we have tried to pre-train each
layer as in [15], without signiﬁcantly inﬂuencing the results.
State of the Art Evaluation In the experiments described
in this subsection, we used an encoder of 3 layers with 1500,
1000 and 800 hidden units, respectively. The patch used has
V = 8 and dp = 48mm, which was found suitable for a
variety of object dimensions. The forests contain four trees
limiting only the number of samples per leaf to 30. For a
2We used the K-means (triangle) as described in [7]

(a) Patch-grid size
(b) stride
(c) feature evaluation
(d) 1st layer ﬁlters
Figure 5: Patch extraction parameters
fair comparison, we do not make use of joint registration or
active vision except when speciﬁcally mentioned.
We tested our solution on the dataset of [5], which con-
tains 20 objects and a set of images regarded as background.
The test scenes contain only one object per image, there is
no occlusion or clutter, and are captured with different illu-
mination from the training set, so one can check the general-
ization of a 6 DoF algorithm to different lighting conditions.
To evaluate our framework we extracted the ﬁrst K = 5 hy-
potheses from the Hough voting space and chose the one
with the best local ﬁtting score. The results are shown in
Table 1 where for simplicity we show only 6 objects and the
average over the complete dataset. Authors provided com-
parison with [14] only with one object, because it was difﬁ-
cult to get results using their method. This dataset was gen-
erally difﬁcult to evaluate, mainly because some pose an-
notations were not very accurate, resulting in having some
better estimations from the ground truth exceeding the met-
ric threshold of acceptance. More details and results are in-
cluded in the supplementary material. Our method showed
that it can generalize well on different lighting conditions,
even without the need of modifying the training set with
Gaussian noise as suggested by the authors.
Table 1: Results on the dataset of [5] (More on supplementary)
Object
[14] (%)
[5] (%)
Our (%)
Hole Puncher
-
98.1
94.3
Duck
-
81.6
87.7
Owl
-
60.5
90.27
Sculpture 1
-
82.7
89.5
Toy (Battle Cat)
70.2
91.8
92.4
...
-
...
Avg.
-
88.2
89.1
We have also tested our method on the dataset presented
in [31], which contains multiple objects of one category
per test image, with much clutter and some cases of oc-
clusion. Authors adopted one-class training, thus, avoiding
background class images during training. For comparison,
we followed the same strategy. Since there are multiple ob-
jects in the scene, we extract the top K = 10 modes of the
{x, y, z} Hough space, and for each mode, we extract the
H = 5 modes of the {yaw, pitch, roll} Hough space and
put a threshold on the local ﬁtting of the ﬁnal hypotheses to
produce the PR curves. Table 2 shows the results in the form
of F1-score (metric authors used) for each of the 6 objects.
The results of methods [14, 10] are taken from [31].
Table 2: Results on the dataset of [31]
Object
[14]
[10]
[31]
Our
F1 score
Coffee Cup
0.819
0.867
0.877
0.932
Shampoo
0.625
0.651
0.759
0.735
Joystick
0.454
0.277
0.534
0.924
Camera
0.422
0.407
0.372
0.903
Juice Carton
0.494
0.604
0.870
0.819
Milk
0.176
0.259
0.385
0.51
Average
0.498
0.511
0.633
0.803
In this dataset we see that our method signiﬁcantly out-
performs the state of arts, especially regarding the Camera
which is small and looks similar with the background ob-
jects, and the Joystick, which has a thin and a thick part. Our
features showed better performance on Milk that contains
other distracting objects on it. It is evident that our learnt
features are able to handle a variety of object appearances
with stable performance and at the same time being robust
to destructors and occluders. Note that without explicitly
training a background class, all the patches in the image are
classiﬁed as belonging to one of our objects. While [31]
designed a speciﬁc technique to tackle this issue, our fea-
tures seem informative enough to produce good modes in
the Hough spaces.
We have also tested [31] and [5] on our own dataset. We
also tried [14], but although we could produce the reported
results on their dataset, we were not able to get meaningful
results on our dataset and so we do not report them. This
is mainly because this method is not intended to be used in
textured objects with simple geometry. We provide results
both with and without using joint object optimization. Our
dataset contains 3D models of six training objects, while
the test images may contain other objects as well. More on
our dataset and evaluation can be found in the supplemen-
tary material. Table 3 shows the results on our database.
The work of [5] is designed to work only with one object
per image and it is not evaluated on the bin-picking dataset.
Our method outperforms all others even without joint op-
timization, but we can clearly see the advantages of such
optimization on the ﬁnal performance.
Active Vision Evaluation We tested our active vision
method on our dataset, using two different types of scenes.
One is the crowded scenario used for single-shot evaluation,
and the other depicts a special arrangement of objects, one

(a) Colgate
(b) Oreo
(c) Softkings
(d) Coffecup
(e) Juice
(f) Camera
(g) Joystick
Figure 6: Qualitative results of our framework. Image 6g is the next best view of image 6f.
Table 3: Results on our dataset
Object
[31]
[5]
Our
Our
joint optim.
scenario 1 (supermarket objects)
amita
26.9
60.8
64.3
71.2
colgate
22.8
11.1
26.1
28.6
elite
10.1
71.9
74.9
77.6
lipton
10.5

## related_work
Unsupervised feature learning has recently received the
attention of the computer vision community. Hinton et al.
[15] used a deep network consisting of Restricted Boltz-
mann Machines for dimensionality reduction and showed
that deep networks can converge to a better solution by
greedy layer-wise pre-training. Jarrett et al. [16] showed
the merits of multi-layer feature extraction with pooling and
local contrast normalization over single-layer architectures,
while Le et al. [19] used a 9-layer Sparse Autoencoder to
learn a face detector only from unsupervised data.
Fea-
ture learning has also been used for classiﬁcation[27] using
RNNs, and detection[3] using sparse coding, trained with
holistic object images and patches, respectively. Coates et
al. [7] investigated different single-layer unsupervised ar-
chitectures such as k-means, Gaussian mixture modes, and
Sparse Autoencoders achieving state of the art results when
parameters were ﬁne-tuned. Here, we use the Sparse Au-
toencoders of [7] but in a deeper network architecture, ex-
tracting features from raw RGB-D data. In turn, in [13] and
[34] it was shown how CNNs could be trained for super-
vised feature learning, while in [23] and [24] CNNs were
trained to perform classiﬁcation and regression jointly for
2D object detection and head pose estimation, respectively.
Object detection and 6 DoF pose estimation is also
frequently addressed in the literature.
Most represen-
tative are techniques based on template matching, like
LINEMOD [14], its extension [25] and the Distance Trans-
form approaches [21]. Point-to-Point methods [10, 26] form
another representative category where emphasis is given on
building point pair features to construct object models based
on point clouds. Tejani et al. [31] combined Hough For-
est with [14] using a template matching split function to
provide 6 DoF pose estimation in cluttered environments.
They provided evidence that, using patches instead of the
holistic image of the object, can boost the performance of
the pose estimator in cases of severe occlusions and clut-
ter. Brachmann et al. [5] introduced a new representation
in form of a joint 3D object coordinate and class labelling,
which, however suffers in cases of occlusions. Addition-
ally, Song et al. [28] proposed a computationally expensive
approach to the 6 DoF pose estimation problem that slides
exemplar SVMs in the 3D space, while in [4] shape priors
are learned by soft labelling Random Forest for 3D object
classiﬁcation and pose estimation. Lim et al. [20] achieved
ﬁne pose estimation by representing geometric and appear-
ance information as a collection of 3D shared parts and ob-
jectness, respectively. Wu et al. [33] designed a model that
learns the joint distribution of voxel data and category la-
bels using a Convolutional Deep Belief Network, while the
posterior distribution for classiﬁcation is approximated by
Gibbs sampling. The authors in [32] tackle the 3D object
pose estimation problem by learning discriminative feature

descriptors via a CNN and then passing them to a scalable
Nearest Neighbor method to efﬁciently handle a large num-
ber of objects under a large range of poses. However, com-
pared to our work, this method is based on holistic images
of the objects, which is prone to occlusions [31] and only
evaluated on a public dataset that contains no foreground
occlusions.
Hypotheses veriﬁcation is employed as a ﬁnal reﬁnement
step to reject false detections. Aldoma et al. [1] proposed
a cost function-based optimization to increase true positive
detections. Fioraio et al. [11] showed how single-view hy-
potheses veriﬁcation can be extended to multi-view ones in
order to facilitate SLAM through a novel Bundle adjustment
framework. Buch et al. [6] presented a two-stage voting
procedure for estimating the likelihood of correspondences,
within a set of initial hypotheses, between two 3D models
corrupted by false positive matches.
Regarding active vision, a recent work presented by Jia
et al. [17] makes use of the Implicit Shape Model com-
bined in a boosting algorithm to plan the next-best-view for
2D object recognition, while Atanasov et al. [2] proposed
a non-myopic strategy using POMDPs for 3D object detec-
tion. Wu et al. [33] used their generative model based on the
convolutional network to plan for the next-best-view but is
limited in the sense that the holistic image of the object is
needed as input. Since previous works are largely depen-
dent on the employed classiﬁer, more related to our work
is the recently proposed Active Random Forests [9] frame-
work, which, however (similar to [33]) requires the holistic
image of an object to make a decision, making it not appro-
priate for our patch-based method.
3. 6 DoF Object Pose & Next-Best-View Esti-
mation Framework
Our object detection and pose estimation framework
consists of two main parts: a) single shot-based 6D object
detection and b) next-best-view estimation. In the ﬁrst part,
we render the training objects and extract depth-invariant
RGB-D patches. The latter are given as input to a Sparse
Autoencoder which learns a feature vector in an unsuper-
vised manner. Using this feature representation, we train a
Hough Forest to recognize object patches in terms of class
and 6D pose (translation and rotation). Given a test image,
patches from the scene pass through the Autoencoder fol-
lowed by the Hough forest, where the leaf nodes cast a vote
in a 6D Hough space indicating the existence of an object.
The modes of this space represent our best object hypothe-
ses. The second part, next-best-view estimation, is based
on the previously trained forest. Using the training sample
distribution in the leaf nodes, we are able to determine the
uncertainty, i.e. the entropy, of our current hypotheses, and
further estimate the reduction in entropy when moving the
camera to another viewpoint using a pose-to-leaf mapping.
Fig. 2 shows an overview of the framework. In the follow-
ing subsections, we describe each part in detail.
3.1. Single Shot-based 6D Object Detection
State of the art Hough Forests Features In the literature
some of the most recent 6D object detection methods use
Hough Forests as their underlying classiﬁer. In [5] simple
two pixel comparison tests were used to split the data in the
tree nodes, while the location of the pixels could be any-
where inside the whole object area. In our experiments, we
also added the case where the pixel tests are restricted inside
the area of an image patch. A more sophisticated feature for
splitting the samples was proposed by Tejani et al. [31] who
used a variant of the template based LineMOD feature [14].
In comparison with the above custom-designed features, we
use Sparse Autoencoders to learn an unsupervised feature
representation of varying length and layers. Furthermore,
we learn features over depth-invariant RGB-D patches ex-
tracted from the objects, as described below.
Patch Extraction Our approach relies on 3D models of the
objects of interest. We render synthetic training images by
placing a virtual camera on discrete points on a sphere sur-
rounding the object. In traditional patch-based techniques
[12], the patch size is expressed directly in image pixels. In
contrast, we want to extract depth invariant, 2.5D patches
that cover the same area of the object regardless of the ob-
ject distance from the camera, similar to [29]. First, a se-
quence of patch centers ci, i = 1..N is deﬁned on a regular
grid on the image plane. Using the depth value of the under-
lying pixels these are back-projected to the 3D world coor-
dinate frame, i.e. ci = (x, y, z). For each such 3D point ci
we deﬁne a planar patch perpendicular to the camera, cen-
tered at ci and with dimensions dp×dp, measured in meters,
which is subdivided into V ×V cells. Then, we back-project
the center of each cell to the corresponding point on the im-
age plane, to compute its RGB and depth values via linear
interpolation1. Depth values are expressed with respect to
the frame centered at the center of the patch (Fig. 2). Also,
we truncate depth values to a certain range to avoid points
not belonging to the object. Depth-invariance is achieved
by expressing the patch size in metric units in 3D space.
From each training image we extract a collection of patches
P and normalize their values to the range [0, 1]. The ele-
ments corresponding to the four channels of the patch are
then concatenated into a vector of size V × V × 4 (RGBD
channels) and are given as input to the Sparse Autoencoder
for feature extraction.
Unsupervised Feature Learning We learn unsupervised
features using a network consisting of stacked, fully con-
nected Sparse Autoencoders, in a symmetric encoder-
decoder scheme. An autoencoder is a fully connected, sym-
1The cell values calculation can be done efﬁciently and in parallel using
texture mapping in gpu.

V
dv
RGB
DEPTH
. . .
V x V x 4
1500 1000
F
. . .
. . .
. . .
Optional
Layers
Feature
vector

## conclusion
In this paper we proposed a complete framework for 6D
object detection in crowded scenes, comprising of an un-
supervised feature learning phase, 6 DoF object pose esti-
mation using Hough Forests and a method for estimating
the next-best-view using the trained forest. We conducted
extensive evaluation on challenging public datasets, includ-
ing a new one depicting realistic scenarios, using various
state of the art methods. Our framework showed superior
results, being able to generalize well to a variety of objects
and scenes. As a future work, we want to investigate how
different patch sizes can be combined, and explore how con-
volutional networks can help in this direction.
Acknowledgment The major part of the work was under-
taken when A. Doumanoglou was at Imperial College Lon-
don within the frames of his Ph.D. A. Doumanoglou was
partially funded from the EU Horizon 2020 projects: RAM-
CIP (grant No 643433) and SARAFun (grant No 644938).