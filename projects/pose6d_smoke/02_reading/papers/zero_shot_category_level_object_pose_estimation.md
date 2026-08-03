# Zero-Shot Category-Level Object Pose Estimation

> 2022 · id: arxiv:2204.03635 · arXiv: 2204.03635 · pdf: https://arxiv.org/pdf/2204.03635 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Consider a young child who is presented with two toys of an object category they
have never seen before: perhaps, two toy aeroplanes. Despite having never seen
examples of ‘aeroplanes’ before, the child has the ability to understand the spatial
⋆These authors contributed equally
arXiv:2204.03635v2  [cs.CV]  2 Oct 2022

2
W. Goodwin et al.
relationship between these related objects, and would be able to align them if
required. This is the problem we tackle in this paper: the zero-shot prediction of
pose offset between two instances from an object category, without the need for
any pose annotations. We propose this as a challenging task which removes many
assumptions in the current pose literature, and which more closely resembles
the setting encountered by embodied agents in the real-world. To substantiate
this claim, consider the information existing pose recognition algorithms have
access to. Current methods make one (or more) of the following assumptions:
that evaluation is performed at the instance-level (i.e there is no intra-category
variation between objects) [49]; that we have access to labelled pose datasets for
all object categories [3, 10, 22, 33, 46, 50, 54]; and/or that we have access to a
realistic CAD model for each object category the model will encounter [9,21,52].
Meanwhile, humans can understand pose without access to any of this in-
formation. How is this possible? Intuitively, we suggest humans use an under-
standing of object parts, which generalise across categories, to correspond related
objects. This process can be followed by using basic geometric primitives to un-
derstand the spatial relationship between objects. Humans typically also have a
coarse depth estimate and can inspect the object from multiple viewpoints.
In this paper, we use these intuitions to build a solution to estimate the pose
offset between two instances of a given category. We perform ‘zero-shot’ pose-
estimation in the sense that our models have never seen pose-labelled examples
of the test categories, and neither do they rely on category-specific CAD mod-
els. We first make use of features extracted from a vision transformer (ViT [16]),
trained in a self-supervised manner on large scale data [7], to establish semantic
correspondences between two object instances of the same category. Prior work
has demonstrated that self-supervised ViTs have an understanding of object
parts which can transfer to novel instances and categories [4,44]. Next, using a
weighting of the semantic correspondences, we obtain a coarse estimate of the
pose offset by selecting an optimal viewpoint for one of the object instances. Hav-
ing obtained semantic correspondences and selected the best view, we use depth
maps to create sparse point clouds for each object at the corresponding semantic
locations. Finally, we align these point clouds with a rigid-body transform using
a robust least squares estimation [43] to give our final pose estimate.
We evaluate our method on the CO3D dataset [35], which provides high-
resolution imagery of diverse object categories, with substantial intra-category
appearance differences between instances. We find that this allows us to reflect a
realistic setting while performing quantitative evaluation in a controlled manner.
We consider and compare to a range of baselines which could be applied to this
task, but find that they perform poorly and often fail completely, demonstrating
the highly challenging nature of the problem.
In summary, we make the following contributions:
– We formalise a new and challenging setting for pose estimation, which is
an important component of most 3D vision systems. We suggest our setting
closely resembles those encountered by real-world embodied agents (Sec. 3).

Zero-Shot Category-Level Object Pose Estimation
3
– We propose a novel method for zero-shot, category-level pose estimation,
based on semantic correspondences from self-supervised ViTs (Sec. 4).
– Through rigorous experimentation on a devised CO3D benchmark, we demon-
strate that our method facilitates zero-shot pose alignment when the base-
lines often fail entirely (Sec. 5).
2

## method
In this section, we detail our method for zero-shot pose estimation. First, se-
mantic correspondences are obtained between the reference and target object
(Sec. 4.1). These correspondences are used to select a suitable view for pose
estimation from the N images in the target sequence (Sec. 4.2). Finally, using
depth information, the correspondences’ spatial locations are used to estimate
the pose offset between the reference and target object instances (Sec. 4.3).
4.1
Self-supervised semantic correspondence with cyclical distances
The key insight of our method is that semantic, parts-based correspondences
generalise well between different object instances within a category, and tend to
be spatially distributed in similar ways for each such object. Indeed, a parts-
based understanding of objects can also generalise between categories; for in-
stance, ‘eyes’, ‘ears’ and ‘nose’ transfer between many animal classes. Recent
work has demonstrated that parts-based understanding emerges naturally from
self-supervised vision transformer (ViT) features [4,7,44], and our solution lever-
ages such a network with large scale pre-training [7]. The ViT is trained over
ImageNet-1K, and we assume that it carries information about a sufficiently
large set of semantic object parts to generalise to arbitrary object categories.
As described in Sec. 3, the proposed setting for pose estimation considers a
relative problem, between a reference object (captured in a single image) and a
target object (with potentially multiple views available). We compare two images
(for now referred to as I1, I2) by building a ‘cyclical distance’ map for every pixel
location in I1 using feature similarities. Formally, consider Φ(Ii) ∈RH′×W ′×D
as the normalised spatial feature of an image extracted by a ViT. Letting u be
an index into Φ(I1) as u ∈{1...H′} × {1...W ′}, we find its cyclical point u′ as:
  u ' = \a
r
gmin _w d (\Phi (
I
_ 1 )_w, \
P
hi (I_2)_ v) \quad | \quad v = \argmin _w d(\Phi (I_1)_u, \Phi (I_2)_w) 
(2)
Here d(·, ·) is the L2-distance, and a cyclical distance map is constructed as
C ∈RH′×W ′ with Cu = −d(u, u′). Using the top-K locations in C, we take
features from Φ(I1) and their nearest neighbours in Φ(I2) as correspondences.
This process is illustrated in Fig. 2b.
The cyclical distance map can be considered as a soft mutual nearest neigh-
bours assignment. Mutual nearest neighbours [4] between I1 and I2 return a
cyclical distance of zero, while points in I1 with a small cyclical distance can

Zero-Shot Category-Level Object Pose Estimation
7
be considered to ‘almost’ have a mutual nearest neighbour in I2. The proposed
cyclical distance metric has two key advantages over the hard constraint. Firstly,
while strict mutual nearest neighbours gives rise to an unpredictable number of
correspondences, the soft measure allows us to ensure K semantic correspon-
dences are found for every pair of images. We find having sufficient correspon-
dences is critical for the downstream pose estimation. Secondly, the soft con-
straint adds a spatial prior to the correspondence discovery process: features
belonging to the same object part are likely to be close together in pixel space.
Finally, following [4], after identifying an initial set of matches through our
cyclical distance method, we use K-Means clustering on the selected features in
the reference image to recover points which are spatially well distributed on the
object. We find that well distributed points result in a more robust final pose
estimate (see supplementary). In practice, we select the top-2K correspondences
by cyclical distance, and filter to a set of K correspondences with K-Means.
4.2
Finding a suitable view for alignment
Finding semantic correspondences between two images which view (two instances
of) an object from very different orientations is challenging. For instance, it is
possible that images from the front and back of an object have no semantic parts
in common. To overcome this, an agent must be able to choose a suitable view
from which to establish semantic correspondences. In the considered setting,
this entails selecting the best view from the N target images. We do this by
constructing a correspondence score between the reference image, IR, and each
image in the target sequence, IT1:N . Specifically, given the reference image and an
image from the target sequence, the correspondence score is the sum the of the
feature similarities between their K semantic correspondences. Mathematically,
given a set of K correspondences between the jth target image and the reference,
{(uj
k, vj
k)}K
k=1, this can be written as:
  j ^{*} =
 \arg
m
a
x _
{j \in 1:N
} \quad \s
um _{k = 1}^{K} - d( \Phi (I_{\mathcal {R}})_{u^{j}_{k}} , \Phi (I_{\mathcal {T}_{j}})_{v^{j}_{k}} ) 
(3)
4.3
Pose estimation from semantic correspondences and depth
The process described in Sec. 4.1 gives rise to a set of corresponding points in
2D pixel coordinates, {(uk, vk)}K
k=1. Using depth information and camera intrin-
sics, these are unprojected to their corresponding 3D coordinates, {(uk, vk)}K
k=1,
where uk, vk ∈R3. In the pose estimation problem, we seek a single 6D pose
that describes the orientation and translation of the target object, relative to the
frame defined by the reference object. Given a set of corresponding 3D points,
there are a number of approaches for solving for this rigid body transform. As
we assume our correspondences are both noisy and likely to contain outliers, we
use a fast least-squares method based on the singular value decomposition [43],
and use RANSAC to handle outliers. We run RANSAC for up to 1,000 itera-
tions, with further details in supplementary. The least squares solution recovers

8
W. Goodwin et al.
a 7-dimensional transform: rotation R, translation t, and a uniform scaling pa-
rameter λ, which we found crucial for dealing with cross-instance settings. The
least-squares approach minimises the residuals and recovers the predicted 6D
pose offset, T ∗as:
  T ^{*} = ( \mathb
f {R}
^
{
*},
 \ m athbf {t}^{*}) = \argmin _{(\mathbf {R}, \mathbf {t})} \sum _{k = 1}^{K} \mathbf {v}_{k} - (\lambda \mathbf {R}\mathbf {u}_{k}+\mathbf {t}) 
(4)
5

## experiments
5.1
Evaluation Setup
Dataset, CO3D [35]: To evaluate zero-shot, category-level pose estimation meth-
ods, a dataset is required that provides images of multiple object categories,
with a large amount of intra-category instance variation, and with varied object
viewpoints. The recently released Common Objects in 3D (CO3D) dataset fulfils
these requirements with 1.5 million frames, capturing objects from 50 categories,
across nearly 19k scenes [35]. For each object instance, CO3D provides approx-
imately 100 frames taken from a 360º viewpoint sweep with handheld cameras,
with labelled camera pose offsets. The proposed method makes use of depth in-
formation, and CO3D provides estimated object point clouds, and approximate
depth maps for each image, that are found by a Structure-from-Motion (SfM)
approach applied over the sequences [38]. We note that, while other object pose
datasets exist [2,48,49], we find them to either be lacking in necessary meta-data
(e.g no depth information), have little intra-category variation (e.g be instance
level), contain few categories, or only provide a single image per object instance.
We expand on dataset choice in the supplementary.
Labels for evaluation: While the proposed pose estimation method requires no
pose-labelled images for training, we label a subset of sequences across the CO3D
categories for quantitative evaluation. We do this by assigning a category-level
canonical frame to each selected CO3D sequence. We exclude categories that
have infinite rotational symmetry about an axis (e.g ‘apple’) or have an insuffi-
cient number of instances with high quality point clouds (e.g ‘microwave’). For
the remaining 20 categories, we select the top-10 sequences based on a point cloud
quality metric. Point clouds are manually aligned within each category with a
rigid body transform. As CO3D provides camera extrinsics for every frame in a
sequence with respect to its point cloud, these alignments can be propagated to
give labelled category-canonical pose for every frame in the chosen sequences.
Further details are in the supplementary.
Evaluation setting: For each object category, we sample 100 combinations of
sequence pairs, between which we will compute pose offsets. For the first sequence
in each pair, we sample a single reference frame, IR, and from the second we
sample N target frames, IT1:N . We take N = 5 as our standard setting, with

Zero-Shot Category-Level Object Pose Estimation
9
results for different numbers of views in Tab. 2 and the supplementary. For each
pair of sequences, we compute errors in pose estimates between the ground truth
and the predictions. For the rotation component, following standard practise in
the pose estimation literature, we report the median error across samples, as well
as the accuracy at 15º and 30º, which are given by the percentage of predictions
with an error less than these thresholds. Rotation error is given by the geodesic
distance between the ground truth and predicted rotation [24].
‘Zero-shot’ pose estimation: In this work, we leverage models with large-scale,
self-supervised pre-training. The proposed pose estimation method is ‘zero-shot’
in the sense that it does not use labelled examples (either pose labels or category
labels) for any of the object categories it is tested on. The self-supervised fea-
tures, though, may have been trained on images containing unlabelled instances
of some object categories considered. To summarise, methods in this paper do
not require labelled pose training sets or CAD models for the categories they
encounter during evaluation. They do require large-scale unsupervised pre-
training, depth estimates, and multiple views of the target object. We assert
that these are more realistic assumptions for embodied agents (see Sec. 3).
5.2
Baselines
We find very few baselines in the literature which can be applied to the highly
challenging problem of pose-detection on unseen categories. Though some meth-
ods have tackled the zero-shot problem before, they are difficult to translate to
our setting as they require additional information such as CAD models for the
test objects. We introduce the baselines considered.
PoseContrast [50] : This work seeks to estimate 3D pose (orientation only) for
previously unseen categories. The method trains on pose-labelled images and
assumes unseen categories will have both sufficiently similar appearance and
geometry, and similar category-canonical frames, to those seen in training. We
adapt this method for our setting and train it on 86 of the 100 categories from
the ObjectNet3D dataset [51] (removing 14 categories that are present in our
CO3D setting, to ensure a zero-shot comparison). During testing, we extract
global feature vectors for the reference and target images with the model, and
use feature similarities to select a suitable view. We then run the PoseContrast
model on the reference and selected target image, with the model regressing to
an Euler angle representation of 3D pose. PoseContrast estimates pose for each
image independently, implicitly inferring the canonical frame for the test object
category. We thus compute the difference between the pose predictions for the
reference and chosen target image to arrive at a relative pose estimate.
Iterative Closest Point (ICP): ICP is a point cloud alignment algorithm that
assumes no correspondences are known between two point clouds, and seeks
an optimal registration. We use ICP to find a 7D rigid body transform (scale,
translation and rotation, as in Sec. 4.3) between the reference and target objects.

10
W. Goodwin et al.
We use the depth estimates for each image to recover point clouds for the two
instances, aggregating the N views in the target sequence for a maximally com-
plete target point cloud. We use these point clouds with ICP. As ICP is known
to perform better with good initialisation, we also experiment with initialising
it from the coarse pose estimate given by our ‘best view’ method (see Sec. 4.2)
which we refer to as ‘ICP + BV’.
Image Matching: Finally, we experiment with other image matching techniques.
In the literature, cross-instance correspondence is often tackled by learning
category-level keypoints. However, this usually involves learning a different model
for each category, which defeats the purpose of our task. Instead, we use category-
agnostic features and obtain matches with mutual nearest neighbours between
images, before combining the matches’ spatial locations with depth informa-
tion to compute pose offsets (similarly to Sec. 4.3). We experiment both with
standard SIFT features [31] and deep features extracted with an ImageNet self-
supervised ResNet-50 (we use SWaV features [6]). In both cases, we select the
best view using the strength of the discovered matches between the reference
and target images (similarly to Sec. 4.2).
5.3
Implementation Details
In this work we use pre-trained DINO ViT features [7] to provide semantic corre-
spondences between object instances. Specifically, we use ViT-Small with a patch
size of 8, giving feature maps at a resolution of 28 × 28 from square 224 × 224
images. Prior work has shown that DINO ViT features encode information on
generalisable object parts and correspondences [4,44]. We follow [4] for feature
processing and use ‘key’ features from the 9th ViT layer as our feature represen-
tation, and use logarithmic spatial binning of features to aggregate local context
at each ViT patch location. Furthermore, the attention maps in the ViT provide
a reasonable foreground segmentation mask. As such, when computing cyclical
distances, we assign infinite distance to any point which lands off the foreground
at any stage in the reference-target image cycle (Sec. 4.1), to ensure that all
correspondences are on the objects of interest. We refer to the supplementary
for further implementation details on our method and baselines.
5.4
Main Results
We report results averaged over the 20 considered categories in CO3D in the
leftmost columns of Tab. 1. We first highlight that the the baselines show poor
performance across the reported metrics. ICP and SIFT perform most poorly,
which we attribute to them being designed for within-instance matching. Align-
ment with the SWaV features, which contain more semantic information, fares
slightly better, though still only reports a 7.5% accuracy at 30º. Surprisingly,
we also found PoseContrast to give low accuracies in our setting. At first glance,
this could simply be an artefact of different canonical poses – between those in-
ferred by the model, and those imposed by the CO3D labels. However, we note

Zero-Shot Category-Level Object Pose Estimation
11
All Categories
Per Category (Acc30 ↑)
Med. Err (↓) Acc30 (↑) Acc15 (↑)
Bike
Hydrant M’cycle Teddy Toaster
ICP
111.8
3.8
0.7
3.0
6.0
1.0
1.0
7 .0
SIFT
129.4
4.0
1.5
3.0
11.0
1.0
1.0
0.0
SWaV
123.1
7.5
3.3
13.0
9.0
8.0
5.0
7.0
ICP+BV
109.3
5.4
1.2
6.0
5.0
8.0
5.0
5.0
PoseContrast
111.5
6.9
1.1
2.0

## related_work
2.1
Category-level pose estimation
While estimating pose for a single object instances has a long history in robotics
and computer vision [49], in recent years there has been an increased inter-
est in the problem of category-level pose estimation, alongside the introduction
of several category-level datasets with labelled pose [2, 46–48]. Approaches to
category-level pose estimation can be broadly delineated into: those defining pose
explicitly through the use of reference CAD models [21,36,39,39,52]; those which
learn category-level representations against which test-time observations can be
in some way matched to give relative pose estimates [8, 9, 12, 33, 41, 45, 46, 54];
and those that learn to directly predict pose estimates for a category from ob-
servations [3,10,22,50].
Most methods (e.g. [9, 29, 33, 41, 46]) treat each object category distinctly,
either by training a separate model per category, or by using different templates
(e.g. CAD models) for each category. A few works (e.g. [50, 54]) attempt to
develop category-agnostic models or representations, and several works consider
the exploitation of multiple views to enhance pose estimation [25,26]. In contrast
to existing works in category-level pose estimation, we do not require any pose-
labelled data or CAD models in order to estimate pose for a category, and tackle
pose estimation for unseen categories.
2.2
Few-shot and self-supervised pose estimation
There has been some recent work that notes the difficulty of collecting large,
labelled, in-the-wild pose datasets, and thus seeks to reduce the data burden
by employing few-shot approaches. For instance, Pose-from-Shape [52] exploits
existing pose-labelled RGB datasets, along with CAD models, to train an object-
agnostic network that can predict the pose of an object in an image, with respect
to a provided CAD model. Unlike this work, we seek to tackle an in-the-wild
setting in which a CAD model is not available for the objects encountered. Self-
supervised, embodied approaches for improving pose estimation for given object
instances have been proposed [15], but require extensive interaction and still
do not generalise to the category level. Few-shot approaches that can quickly
fine-tune to previously unseen categories exist [42, 50], but still require a non-
trivial number of labelled examples to fine-tune to unseen categories, while in
contrast we explore the setting in which no prior information is available. Fur-
thermore, recent works have explored the potential for unsupervised methods

4
W. Goodwin et al.
with equivariant inductive biases to infer category-level canonical frames with-
out labels [28,37], and to thus infer 6D object pose given an observed point cloud.
These methods, while avoiding the need for pose labels, only work on categories
for which they have been trained. Finally, closest in spirit to the present work
is [18], who note that the minimal requirement to make zero-shot pose estima-
tion a well-posed problem is to provide an implicit canonical frame through use
of a reference image, and formulate pose estimation as predicting the relative
viewpoint from this view. However, this work can only predict pose for single
object instances, and does not extend to the category level.
2.3
Semantic descriptor learning
A key component of the presented method to zero-shot category level pose es-
timation is the ability to formulate semantic keypoint correspondences between
pairs of images within an object category, in a zero-shot manner. There has been
much interest in semantic correspondences in recent years, with several works
proposing approaches for producing these without labels [1, 4, 7, 27]. Semantic
correspondence is particularly well motivated in robotic settings, where problems
such as extending a skill from one instance of an object to any other demand the
ability to relate features across object instances. Prior work has considered learn-
ing dense descriptors from pixels [19] or meshes [40] in a self-supervised manner,
learning skill-specific keypoints from supervised examples [32], or robust match-
ing at the whole object level [20]. The descriptors in [19, 32, 40] are used to
infer the relative pose of previously unseen object instances to instances seen
in skill demonstrations. In contrast to these robotics approaches, in our method
we leverage descriptors that are intended to be category-agnostic, allowing us to
formulate a zero-shot solution to the problem of pose estimation.
3
Zero-shot Category-Level Pose Estimation
In this section, we formalise and motivate our proposed zero-shot pose estimation
setting. To do this, we first outline the generic problem of object pose estimation.
6D object pose estimation entails estimating the offset (translation and rotation)
of an object with respect to some frame of reference, normally given just an
image of the object. This frame of reference can be defined implicitly (e.g in
the supervised setting, the labels are all defined with respect to some ‘canonical’
frame) or explicitly (e.g with a reference image). In either case, pose estimation
is fundamentally a relative problem. In the zero-shot setting we consider, the
frame of reference cannot be implicitly defined by labels: we do not have labelled
pose for any objects. Therefore, the pose estimation problem is that of aligning
(computing the pose offset between) two instances of a given category, i.e. a
reference and a target object.
In our proposed setting, we assume access to N views of a target object, as
well as depth information for both objects, and suggest that these constraints
reflect practical settings. For objects in the open-world, we are unlikely to have
realistic CAD models or labelled pose training sets. On the other hand, many

Zero-Shot Category-Level Object Pose Estimation
5
Target sequence
Reference image
(c) Top-K semantic correspondences
(b) Cyclical descriptor distances 
(a) Extract 
descriptors
(d) Optimal 
6D pose
(e) Alignment
Depth images
Inputs
Outputs
Fig. 2: Our method for zero-shot pose estimation between two instances of an
object, given a reference image and a sequence of target images. In our method,
we: (a) Extract spatial feature descriptors for all images with a self-supervised
vision transformer (ViT). (b) Compare the reference image to all images in the
target sequence by building a set of cyclical distance maps (Sec. 4.1). (c) Use
these maps to establish K semantic correspondences between compared images
and select a suitable view from the target sequence (Sec. 4.2). (d) Given the
semantic correspondences and a suitable target view, we use depth informa-
tion to compute a rigid transformation between the reference and target objects
(Sec. 4.3). (e) Given relative pose transformations between images in the target
sequence, we can align the point cloud of the reference image with the entire
target sequence.
embodied agents are fitted with depth cameras or can recover depth (up to a
scale) from structure from motion or stereo correspondence. Furthermore, real-
world agents are able to interact with the object and hence gather images from
multiple views.
Formally, we consider a reference image, IR, and a set of target images IT1:N =
{IT1...ITN }, where Ii ∈RH×W ×3. We further have access to depth maps, Di ∈
RH×W for all images. Given this information, we require a model, M, to output
a single 6D pose offset between the object in the reference image and the object
in the target sequence, as:
  T ^{*} = \ma t
hca l {M} (I_{\mathcal {R}}, I_{\mathcal {T}_{1:N}} | \quad D_\mathcal {R}, D_{\mathcal {T}_{1:N}}) 
(1)

6
W. Goodwin et al.
Finally, we note that, in practice, the transformations between the target
views must be known for the predicted pose offset to be most useful. These
transformations are easily computed by an embodied agent and can be used to
align the reference instance with the entire target sequence, given an alignment
between IR and any of the target views.
4

## conclusion
Consideration of limitations: We have have proposed a model which substan-
tially outperforms existing applicable baselines for the task of zero-shot category-
level pose detection. However, absolute accuracies remain low and we suggest fall
far short of human capabilities. Firstly, our performance across the considered
classes is 46.3% Acc30 with 5 views available. We imagine these accuracies to
be substantially lower than the a human baseline for this task. Secondly, though
single view novel category alignment is highly challenging for machines, humans
are capable of generalising highly abstract concepts to new categories, and thus
would likely be able to perform reasonably in a single view setting.
Final Remarks: In this paper we have proposed a highly challenging (but real-
istic) setting for object pose estimation, which is a critical component in most
3D vision pipelines. In our proposed setting, a model is required to align two in-
stances of an object category without having any pose-labelled data for training.

14
W. Goodwin et al.
Fig. 3: Example results for the categories Teddybear, Toybus, Car, Hydrant.
Depicted are the correspondences found between the reference image and the
best-matching frame from the target sequence found following Sec. 4.2. To the
right, the estimated pose resulting from these correspondences is shown as an
alignment between the reference object (shown as a rendered point cloud) and
the target sequence. Hydrant depicts a failure mode — while the result looks
visually satisfying, near rotational symmetry (about the vertical axis) leads to
poor alignment.
We further re-purpose the recently released CO3D dataset and devise a test set-
ting which reasonably resembles the one encountered by a real-world embodied
agent. Our setting presents a complex problem which requires both semantic and
geometric understanding, and we show that existing baselines perform poorly on
this task. We further propose a novel method for zero-shot, category-level pose
estimation based on semantic correspondences and show it can offer a six-fold
increase in Acc30 on our proposed evaluation setting. We hope that this work
will serve as a spring-board to foster future research in this important direction.
7
Acknowledgment
The authors gratefully acknowledge the use of the University of Oxford Advanced
Research Computing (ARC) facility http://dx.doi.org/10.5281/zenodo.22558.
Sagar Vaze is funded by a Facebook Research Scholarship. We thank Dylan
Campbell, Nived Chebrolu and Matias Mattamala for many useful discussions.

Zero-Shot Category-Level Object Pose Estimation
15