# Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images

> 2022 · id: W4224598146 · arXiv: 2204.10776 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Estimating the orientation and location of an object in 3D space is a prelimi-
nary and necessary step for many tasks involving interaction with the object. In
the last decade, 3D vision has witnessed tremendous development ranging from
robotics, games, to VR/AR. These applications raise new demands for the 6-DoF
object pose estimation, requiring a pose estimator to be generalizable, ﬂexible,
and easy-to-use. However, existing methods suﬀer from several restrictive condi-
tions. Most methods [28,70,64] can only be used for a speciﬁc object or category
same as the training data. Some methods [57,29,74,72,43,41,73] can generalize to
unseen objects, but they rely on high-quality target 3D models [57,29,74,72,43],
or additional depth maps [41] and masks [41,73] at test time. These requirements
severely limit the practical applications of the existing pose estimators.
To meet demands in practical applications, we argue that such a pose estima-
tor should have the following properties. 1) Generalizable. The pose estimator
can be applied on an arbitrary object without training on the object or its cat-
egory. 2) Model-free. When generalizing to an unseen object, the estimator
arXiv:2204.10776v2  [cs.CV]  27 Jan 2023

2
Y. Liu, Y. Wen, S. Peng, et al.
(a) Reference Images
(b) Query Images
(c) Object poses
Fig. 1. Given (a) reference images of an object with known poses and (b) query images
containing the same object with unknown poses, our pose estimator is able to accurately
estimate (c) their object poses in the query images, where green color means ground-
truth and blue color means estimation. Note that all objects are unseen in the
training set and the same estimator is applied for all objects.
only needs some reference images of this object with known poses to deﬁne the
object reference coordinate system, as shown in Fig. 1 (a), but does not rely on
a 3D model of the object. 3) Simple inputs. When estimating object poses, the
estimator only takes RGB images as inputs without requiring additional object
masks or depth maps.
To the best of our knowledge, there is no existing pose estimator satisfying
all the above three properties simultaneously. Thus, in this paper, we propose
a simple but eﬀective pose estimator, called Gen6D, which possesses the three
properties above. Given input reference images of an arbitrary object with known
poses, Gen6D is able to directly predict its object pose in any query images,
as shown in Fig. 1. In general, an object pose can be estimated by directly
predicting rotation/translation by regression [70,27,59], solving a Perspective-n-
Points (PnP) problem [42,47] or matching images with known poses [58,69,57].
Direct prediction of rotation and translation by regression is mostly limited to
a speciﬁc instance or category, which has diﬃculty in generalizing to unseen
objects. Meanwhile, due to the lack of 3D models, PnP-based methods do not
have 3D keypoints to build 2D-3D correspondences so that they are incompatible
with model-free setting. Hence, we apply image-matching in our framework for
pose estimation, which can generalize to unseen objects by learning a general
image similarity metric.
In Gen6D, we propose a novel image-matching based framework to estimate
the object pose in a coarse-to-ﬁne manner. The framework consists of an object

Gen6D Pose Estimator
3
Reference images
(a) Query image
(b) Detection
(c) Viewpoint selection
(d) Initial pose
(e) Refined pose
1. Detector
2. Selector
3. Refiner
In-plane
Rotation
Selected viewpoint
…
Fig. 2. Overview. The proposed pose estimator consists of a detector which detects
the object in the query image, a viewpoint selector which selects the most similar
viewpoint from reference images, and a pose reﬁner which reﬁnes the initial pose into
an accurate object pose.
detector, a viewpoint selector and a pose reﬁner, as shown in Fig. 2. Given
reference images and a query image, an object detector ﬁrst detects the object
regions by correlating the reference images with the query image, which is similar
to [1]. Then, a viewpoint selector matches the query image against the reference
images to produce a coarse initial pose. Finally, the initial pose is further reﬁned
by a pose reﬁner to search for an accurate object pose.
A challenge is how to design a viewpoint selector when the reference images
are sparse and contain cluttered background. Existing image-matching meth-
ods [58,69,57,22,2] have diﬃculty in handling this problem due to two problems.
First, these image-matching methods embed images into feature vectors and
compute similarities using distances of feature vectors, in which cluttered back-
ground interferes the embedded feature vectors and thus severely degenerates
the accuracy. Second, given a query image, there may not be a reference image
with exactly the same viewpoint as the query image. In this case, there will be
multiple plausible reference images and the selector has to select the one with
the nearest viewpoint as the query image, which usually are very ambiguous as
shown in Fig. 3.
To address these problems in viewpoint selection, we propose to use neural
networks to pixel-wisely compare the query image with every reference image
to produce similarity scores and select the reference image with highest simi-
larity score. This pixel-wise comparison enables our selector to concentrate on
object regions and reduces the inﬂuence of cluttered background. Furthermore,
we add global normalization layers and self-attention layers to share similarity
information across diﬀerent reference images. These two kinds of layers enable
every reference images to commute with each other, which provides context in-
formation for the selector to select the most similar reference image.
The main challenge of developing our pose reﬁner is the unavailability of
the object model. Existing pose reﬁners [29,74] are based on rendering-and-
comparison, which render an image on the input pose and then match the ren-

4
Y. Liu, Y. Wen, S. Peng, et al.
Query
Nearest
2nd Nearest
Query
Nearest
2nd Nearest
Fig. 3. Query images and reference images have cluttered background. The reference
image with the nearest viewpoint looks very similar as the one with second nearest
viewpoint, which brings challenges for the selector to correctly select the nearest one.
dered image with the query image to reﬁne the input pose. However, without
the object model, rendering high-quality images on arbitrary poses is diﬃcult,
which makes these reﬁnement methods infeasible in the model-free setting.
To address this problem, we propose a novel 3D volume-based pose reﬁnement
method. Given a query image and an input pose, we ﬁnd several reference images
that are near to the input pose. These reference images are projected back into
3D space to construct a feature volume. Then, the constructed feature volume is
matched against the features projected from the query image by a 3D CNN to
reﬁne the input pose. In comparison with previous pose reﬁners [29,74], our pose
reﬁner avoids rendering any new images. Meanwhile, the constructed 3D feature
volume enables our method to infer the 3D pose reﬁnement in the 3D space. In
contrast, previous pose reﬁners [29,74] only rely on 2D image features to regress
a 3D relative pose, which are less accurate especially for unseen objects.
To validate the eﬀectiveness of our generalizable model-free pose estimator,
we introduce a new dataset, called General Model-free Object Pose Dataset
(GenMOP), which contains video sequences of objects in diﬀerent environments
and lighting conditions. We choose one sequence as reference images and the rest
sequences of the same object as test query images. Experiments show that with-
out training on these objects, our method still outperforms instance-speciﬁc esti-
mator PVNet [42] on the GenMOP dataset and another model-free MOPED [41]
dataset. We also evaluate our method on the LINEMOD dataset [23], on which
our generalizable pose estimator achieves comparable results as instance-speciﬁc
estimators which needs to be trained with excessive rendered images.
2

## method
General
Object Name
avg.
Chair PlugEN Piggy Scissors TFormer
ADD-0.1d
PVNet [42]

49.50
2.33
77.89 44.40
19.84
38.79
RLLG [6]

0.70
1.28
1.01
3.45
0.79
2.71
ObjDesc [69]

3.50
5.14
14.07
1.25
7.54
8.55
Ours w/o Ref.

14.00
7.48
39.70
16.81
11.51
17.90
Ours

61.50 19.63
75.38
32.76
62.70
50.39
Prj-5
PVNet [42]

15.00
30.37
83.42
96.55
59.52
56.97
RLLG [6]

2.00
4.67
17.59
35.78
7.94
13.59
ObjDesc [69]

4.00
10.75
4.52
18.53
8.33
9.23
Ours w/o Ref.

11.50
40.65
33.17
34.05
64.29
36.73
Ours

55.00 72.90 92.96
93.53
98.81
82.64
training set for PVNet and RLLG. However, only ∼200 reference images are not
enough to produce reasonable results so we additionally label the object masks
on these reference images and cut the object to randomly paste on backgrounds
from COCO [32] to enlarge their training set. For PVNet, we use its 8 corners
of the 3D bounding box as keypoints for voting because no model is available.
Comparison with baselines. 1) Both ObjDesc [69] and “Ours w/o Ref”
select the most similar reference image to estimate the object pose. The results
show that our viewpoint selector is able to select more accurate viewpoint than
ObjDesc. However, only selecting the best reference viewpoint is not enough for
predicting accurate poses because the reference images do not cover all possible
viewpoints. 2) With further pose reﬁnement, our Gen6D estimator is able to
produce better results than the instance-speciﬁc methods PVNet and RLLG on
average. The main reason is that for PVNet and RLLG, these reference images
are not enough for training a very accurate pose estimator. In contrast, Gen6D
well adapts into this setting with limited reference images of a novel object. Our
pose reﬁner is able to learn generalizable features for accurate pose reﬁnement.
4.4
Results on LINEMOD [23]
We further report results in ADD-0.1d on the LINEMOD [23] dataset in Table 2
and show qualitative results in Fig. 8. For baselines, we include the instance-
speciﬁc pose estimators [58,62,74,6,42,70,29] and a generalizable estimator Pose-
From-Shape (PFS) [72]. The instance-speciﬁc estimators are either trained on the
synthetic data of the object (“synthetic training”) [58,62,74] or trained on both
the synthetic and real data of the object (“real training”) [42,70,74]. PFS [72] is
trained on ShapeNet [7], which embeds an object shape into a feature vector and
applies the embedded feature vector on a query image to predict an object pose.
For [72,74,70], we also include their reported performance using pose reﬁners
DeepIM [29] or DPOD [74], both of which are trained on the synthetic data or

12
Y. Liu, Y. Wen, S. Peng, et al.
Table 2. ADD-0.1d on LINEMOD [23] dataset. “Training” means what kind of training
set is used. “Synthetic” means the model only uses synthetic data of the given object
for training; “Real” means the model is trained on both the synthetic images and real
images of the given model; “No” means the model is not trained on any data of the
test object. “GT-BBox” means a model uses the ground-truth bounding box or not to
produce its performance. “Reﬁne” means the pose reﬁner.
Training
Name
GT-
Reﬁne
Object Name
Avg.
BBox
cat
duck
bvise
cam
driller
lamp
Synthetic
AAE [58]

No
17.90
4.86
20.92
30.47
23.99
60.47
26.44
Self6D [62]

No
57.90
19.60
75.20
36.90
67.00
68.20
54.13
DPOD [74]

No
32.36
26.12
66.76
24.22
66.60
57.26
45.55
DPOD [74]

DPOD [74]
65.10
50.04
72.69
34.76
73.32
74.27
61.70
Real
PFS [72]

DeepIM [29] 54.10
48.60
63.80
40.00
75.30
55.30
56.18
PVNet [42]

No
79.34
52.58
99.90
86.86
96.43
99.33
85.74
PoseCNN [70]

DeepIM [29] 82.10
77.70
97.40
93.50
95.00
96.84
95.19
DPOD [74]

DPOD [74]
94.71
86.29
98.45
96.07
98.80
97.50
90.53
Gen
PFS [72]

No
15.40
8.20
25.10
12.10
18.60
6.50
14.32
Ours

No
94.11
81.31
99.52
94.31
96.33
93.38
93.16
Ours

No
15.97
7.89
25.48
22.06
17.24
35.80
20.74
Ours

Volume
60.68
40.47
77.03
66.67
67.39
89.83
67.01
real data of the test object. Ground-truth bounding box is used in PFS to crop
the object region for the pose estimation. For baselines, we use the performance
reported in their paper for comparison.
The results in Table 2 show that: 1) In comparison with the generalizable pose
estimator PFS [72], Gen6D outperforms PFS [72] with or without subsequent
pose reﬁnement. Note the PFS [72] uses the DeepIM [29] reﬁner which actu-
ally is trained on the synthetic data of the test object while our volume-based
reﬁner is not trained on the test object at all. 2) In comparison with instance-
speciﬁc estimators [74,62,58] with synthetic training on the test object, Gen6D
clearly outperforms all these methods. 3) However, Gen6D performs worse than
instance-speciﬁc estimators [42,70,74] with real training. The main reason is the
inaccurate estimation of the depth. Since the object is usually very far away from
the camera and small scale diﬀerence (1-2 pixels) will result in a huge oﬀset in
the depth direction. Without training on the object, Gen6D cannot perceive such
subtle scale changes, which results in worse performance. 4) With ground-truth
bounding box, Gen6D achieves comparable results as the instance-speciﬁc esti-
mators [42,70,74] with real training because such ground-truth bounding boxes
provide correct depths.
4.5
Results on MOPED [41]
On the MOPED dataset, we compare Gen6D with Latent-Fusion [41] and PVNet
[42]. Latent-Fusion [41] is also a generalizable pose estimator which does not
require training on the test object but needs depth and object masks on query
images. We use the oﬃcial codes and the pretrained weights of Latent-Fusion [41]

Gen6D Pose Estimator
13
Table 3. ADD-AUC on the MOPED dataset with threshold 0-10cm. “LF” means
Latent-Fusion [41]. “General” means the pose estimator is trained on the speciﬁc object
or not. “Input” means the required type of query images at test time.

## experiments
Convolution
CNN
or
𝑁𝑠scales
Fig. 4. (a) Detection outputs. Depth can be computed from the bounding box size Sq,
which along with the 2D projection of the object center determine the location of the
object center. (b) Architecture of the detector. We use features of reference images as
kernels to convolve features of multi-scale query images to get score maps. Score maps
are further processed by a CNN to produce a heat map about the object center and a
scale map to determine the bounding box size.
Overview. As shown in Fig. 2, the proposed Gen6D pose estimator consists
of an object detector, a view selector and a pose reﬁner. The object detector
crops the object region and estimates an initial translation (Sec. 3.1). The view
selector ﬁnds an initial rotation by selecting the most similar reference image and
estimating an in-plane rotation (Sec. 3.2). The initial translation and rotation
are used in the pose reﬁner to iteratively estimate an accurate pose (Sec. 3.3).
3.1
Detection
The query image is usually very large and the object only occupies a small region
on the query image. To focus on the object, we apply a correlation-based instance
detector similar to [1]. We decompose the detection problem into two parts, i.e.
ﬁnding the 2D projection q of the object center and estimating a compact square
bounding box size Sq that encloses the unit sphere. As shown in Fig. 4 (a), such a
compact bounding box size is used in computing the depth of the object center
by d = 2 ˜f/Sq, where 2 is the diameter of the unit sphere and ˜f is a virtual
focal length by changing the principle point to the estimated projection q. The
projection q and the depth d will determine the location of the object center,
which provides an initial translation for the object pose.
The design of our detector is shown in Fig. 4 (b). We extract feature maps on
both the reference images and the query image by a VGG [53]-11 network. Then,
the feature maps of all reference images are regarded as convolution kernels to
convolve with the feature map of the query image to get score maps. To account
for scale diﬀerences, we conduct such convolution at Ns predeﬁned scales by
resizing the query images to diﬀerent scales. Based on the multi-scale score
maps, we regress a heat map and a scale map. We select the location with the
max value on the heat map as the 2D projection of the object center and use the

Gen6D Pose Estimator
7
Rotated reference images
…
…
…
Cropped query image
𝑁𝑎× 𝐹× 𝐻× 𝑊
𝐹× 𝐻× 𝑊
𝑁𝑎× 𝐹× 𝐻× 𝑊
…
Global 
Normalization
…
…
Max
Pooling
Conv
𝑁𝑎× 𝐹
CNN
CNN
Conv
Attention
Layers
…
…
In-plane 
rotation
Similarity
score
𝑁𝑎× 𝐹
𝑁𝑎rotations
Element-wise 
Product
Viewpoint 
Encoding
+
Similarity Network
Aligned
Fig. 5. Architecture of the viewpoint selector. We compute the element-wise product of
every reference image with the query image to get a score map, on which a similarity
network is applied to compute an in-plane rotation and a similarity score for this
reference image. Note that in the similarity network, we use global normalization layers
and a transformer to share information across reference images.
scale value s at the same location on the scale map to compute the bounding
box size Sq = Sr ∗s, where Sr is the size of reference images.
With the detected 2D projections and scales, we compute the initial 3D
translations and crop the object regions for subsequent processing. More de-
tails about architecture and training of detector networks can be found in the
supplementary materials.
3.2
Viewpoint Selection
Viewpoint selection aims to select a reference image whose viewpoint is the
nearest to the query image. Meanwhile, we will estimate an in-plane rotation
between the query image and the selected reference image. We approximately
regard the viewpoint of the selected reference image as the viewpoint of the
query image, which along with the estimated in-plane rotation forms an initial
rotation for the object pose.
As shown in Fig. 5, we design a viewpoint selector to compare the query
image with every reference image to compute similarity scores. Speciﬁcally, we
ﬁrst extract feature maps by applying a VGG [53]-11 on reference images and
the query image. Then, for every feature map of reference images, we compute
its element-wise product with the feature map of the query image to produce
a correlation score map. Finally, the correlation score map is processed by a
similarity network to produce a similarity score and a relative in-plane rotation
to align the query image with the reference image. In our viewpoint selector, we
have three special designs.
In-plane rotation. To account for in-plane rotations, every reference im-
age is rotated by Na predeﬁned angles and all rotated versions are used in the
element-wise product with the query image.
Global normalization. For every feature map produced by the similarity
network, we normalize it with the mean and variance computed from all feature
maps of reference images. Such a global normalization helps our selector select

8
Y. Liu, Y. Wen, S. Peng, et al.
the relatively most similar reference image because it allows the distribution
of feature maps to encode the context similarity and ampliﬁes the similarity
diﬀerences among diﬀerent reference images. For every reference image, max-
pooling is applied on its feature map to produce a similarity feature vector.
Reference view transformer. We apply a transformer on the similarity
feature vectors of all reference images, which includes the positional encoding of
their viewpoints and attention layers over all similarity feature vectors. Such a
transformer lets feature vectors communicate with each other to encode contex-
tual information [61,49,65], which is helpful to determine the most similar ref-
erence image. The outputs of reference view transformer will be used to regress
a similarity score and an in-plane rotation angle for each reference image. The
viewpoint of the reference image with highest score will be selected.
With the selected viewpoint and the estimated in-plane rotation, we esti-
mated an initial rotation for the object pose, which will be reﬁned by the pose
reﬁner. More details about the network and training can be found in the sup-
plementary materials.
3.3
Pose reﬁnement
Combining the translation estimated by the object detector and the rotation
estimated by the viewpoint selector, we get an initial coarse object pose. This
initial pose is further reﬁned by a 3D volume-based pose reﬁner.
Speciﬁcally, since the objects are already normalized inside an unit sphere at
the origin, we build a volume within the unit cube at the origin with S3
v = 323
vertices. As shown in Fig. 6 (a), to construct the features on these vertices, we
ﬁrst select Nn = 6 reference images that are near to the input pose. We extract
feature maps on these selected reference images by a 2D CNN. Then, these
feature maps are unprojected into the 3D volume and we compute the mean and
variance of features among all reference images as features for volume vertices.
For the query image, we also extract its feature map by the same 2D CNN,
unproject feature map into the 3D volume using the input pose and concatenate
the unprojected query features with the mean and variance of reference image
features. Finally, we apply a 3D CNN on the concatenated features of the volume
to predict a pose residual to update the input pose.
Similarity approximation. Instead of regressing the rigid pose residual
directly, we approximate it with a similarity transformation, as shown in Fig. 6
(b). The approximate similarity transformation consists of a 2D in-plane oﬀset, a
scale factor and a residual 3D rotation. The reason of using this approximation
is that it avoids direct regression of the 3D translation from the red circle to
the solid green circle in Fig. 6, which is out of the scope of the feature volume.
Instead, we regress a similarity transformation from red circle to dotted green
circle, which can be easily inferred from the features deﬁned in the volume. More
details can be found in the supplementary materials. In our implementation, we
apply the reﬁner iteratively 3 times by default.
Discussion. The key diﬀerence between our volume-based reﬁner and other
pose reﬁners [29,74,57] is that our pose reﬁner does not require rendering an

Gen6D Pose Estimator
9
Reference poses
Neighboring reference images
Input pose
…
…
2D CNN
2D CNN
Mean+Var
Feats
Unproject
Unproject
3D CNN
Pose Residual
Search
for
Update
Object under GT pose 
Object under input pose 
Constructed volume
Reference poses
Input pose
Legend
(a) Architecture of the pose refiner
(b) Similarity approximation
Fig. 6. (a) Architecture of our pose reﬁner. (b) A 2D diagram to illustrate the simi-
larity transformation approximation. Though the ground-truth pose residual from the
input object pose (solid red circle) to the ground-truth object pose (solid green circle)
is a rigid transformation, we can approximate this rigid transformation by a 

## related_work
2.1
Speciﬁc object pose estimator
Most object pose estimators [70,58,42,28,24,68,14,63,34,54,27,45,26,25,46,55] are
instance-speciﬁc, which cannot generalize to unseen objects and usually re-
quire a 3D model of the object to render extensive images for training. Recent
instance-speciﬁc pose estimators [40,6,33] reconstruct the object model implic-
itly in the pipeline so that they are model-free. Category-speciﬁc pose estima-
tors [64,11,67,13,30,10,60,8,31,9,16,15] can generalize to objects in the same cat-
egory and also do not require the object model. However, they are still unable to

Gen6D Pose Estimator
5
predict poses for objects in unseen categories. In comparison, Gen6D is general-
izable, which makes no assumption of the category or the instance of the object
and also does not need the 3D model of the object.
2.2
Generalizable object pose estimator
Generalizable pose estimators mostly require an object model either for shape
embedding [72,43,12,44] or template matching [22,2,69,57,21,37,75] or rendering-
and-comparison [29,74,38,4,57,18]. To avoid using 3D models, recent works [73,41]
utilize the advanced neural rendering techniques [36] to directly render from
posed images for pose estimation. However, current rendering methods are only
able to render images under exactly the same appearance like lighting conditions,
which degenerates the accuracy under varying appearance. To remedy this, these
methods have to resort to additional depth maps [41] or object masks [41,73]
to achieve robustness. There are also some works focusing on estimating poses
of unseen objects using RGBD sequences [66,38,52,20,5,17]. In contrast to these
methods, Gen6D is model-free and does not require depth maps or masks. There
are also concurrent works [56,51] of generalizable pose estimation.
2.3
Instance detection
Instance detection aims to detect a given object with some images of the ob-
ject [1,35,19,39]. There are some instance detection methods which also estimate
viewpoints [71,3] for novel category in one- or few-shot setting. The detector of
Gen6D is inspired by [1], which uses correlation to ﬁnd the object region. The
target of Gen6D is to estimate the 6-DoF object pose, which is diﬀerent from
these methods for detection or category-level viewpoint estimation.
3

## conclusion
In this paper, we propose an easy-to-use 6-DoF pose estimator Gen6D for un-
seen objects. To predict poses for unseen objects, Gen6D does not require the
object model but only needs some posed images of the object to predict its
pose in arbitrary environments. In Gen6D, we design a novel viewpoint selector
and a novel volume-based pose reﬁner. Experiments demonstrate the superior
performance of Gen6D estimator in predicting poses for unseen objects in the
model-free setting.

Gen6D Pose Estimator
15