# Learning descriptors for object recognition and 3D pose estimation

> 2015 · id: W1909903157 · arXiv: 1502.05908 · pdf: https://arxiv.org/pdf/1502.05908 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Detecting poorly textured objects and estimating their
3D pose reliably is still a very challenging problem. We
introduce a simple but powerful approach to computing de-
scriptors for object views that efﬁciently capture both the
object identity and 3D pose.
By contrast with previous
manifold-based approaches, we can rely on the Euclidean
distance to evaluate the similarity between descriptors, and
therefore use scalable Nearest Neighbor search methods to
efﬁciently handle a large number of objects under a large
range of poses. To achieve this, we train a Convolutional
Neural Network to compute these descriptors by enforcing
simple similarity and dissimilarity constraints between the
descriptors. We show that our constraints nicely untangle
the images from different objects and different views into
clusters that are not only well-separated but also structured
as the corresponding sets of poses: The Euclidean distance
between descriptors is large when the descriptors are from
different objects, and directly related to the distance be-
tween the poses when the descriptors are from the same
object. These important properties allow us to outperform
state-of-the-art object views representations on challenging
RGB and RGB-D data.

## introduction
Impressive results have been achieved in 3D pose estima-
tion of objects from images during the last decade. How-
ever, current approaches cannot scale to large-scale prob-
lems because they rely on one classiﬁer per object, or multi-
class classiﬁers such as Random Forests, whose complexity
grows with the number of objects. So far the only recog-
nition approaches that have been demonstrated to work on
large scale problems are based on Nearest Neighbor (NN)
classiﬁcation [23, 16, 8], because extremely efﬁcient meth-
ods for NN search exist with an average complexity of
O(1) [24, 21]. Moreover, Nearest Neighbor (NN) classi-
ﬁcation also offers the possibility to trivially add new ob-
jects, or remove old ones, which is not directly possible with
neural networks, for example. However, to the best of our
knowledge, such an approach has not been applied to the
3D pose estimation problem, while it can potentially scale
to many objects seen under large ranges of poses. For ex-
ample, [8] only focuses on object recognition without con-
sidering the 3D pose estimation problem.
For NN approaches to perform well, a compact and dis-
criminative description vector is required. Such representa-
tions that can capture the appearance of an object under a
certain pose have already been proposed [7, 14], however
they have been handcrafted. Our approach is motivated by
the success of recent work on feature point descriptor learn-
ing [5, 35, 20], which shows that it is possible to learn com-
pact descriptors that signiﬁcantly outperform handcrafted
methods such as SIFT or SURF.
However, the problem we tackle here is more complex:
While feature point descriptors are used only to ﬁnd the
points’ identities, we here want to ﬁnd both the object’s
identity and its pose. We therefore seek to learn a descrip-
tor with the two following properties: a) The Euclidean dis-
tance between descriptors from two different objects should
be large; b) The Euclidean distance between descriptors
from the same object should be representative of the sim-
ilarity between their poses. This way, given a new object
view, we can recognize the object and get an estimate of its
pose by matching its descriptor against a database of regis-
tered descriptors. New objects can also be added and exist-
ing ones removed easily. To the best of our knowledge, our
method is the ﬁrst one that learns to compute descriptors for
object views.
Our approach is related to manifold learning, but the key
advantage of learning a direct mapping to descriptors is that
we can use efﬁcient and scalable Nearest Neighbor search
methods. This is not possible for previous methods relying
on geodesic distances on manifolds. Moreover, while pre-
vious approaches already considered similar properties to
a) and b), to the best of our knowledge, they never consid-
ered both simultaneously, while it is critical for efﬁciency.
Combining these two constraints in a principled way is far
from trivial, but we show it can be done by training a Con-
volutional Neural Network [18] using simple constraints to
compute the descriptors. As shown in Fig. 1, this results in
arXiv:1502.05908v2  [cs.CV]  13 Apr 2015

Figure 1. Three-dimensional descriptors for several objects under many different views computed by our method on RGB-D data. Top-left:
The training views of different objects are mapped to well-separated descriptors, and the views of the same object are mapped to descriptors
that capture the geometry of the corresponding poses, even in this low dimensional space. Top-right: New images are mapped to locations
corresponding to the object and 3D poses, even in the presence of clutter. Bottom: Test RGB-D views and the RGB-D data corresponding
to the closest template descriptor.
a method that nicely untangles the views of different objects
into descriptors that capture the identities and poses of the
objects.
We evaluate our approach on instance recognition and
pose estimation data with accurate ground truth and show
signiﬁcantly improved results over related methods. Addi-
tionally we perform experiments assessing the ability of the
method to generalize to unseen objects showing promising
results.

## method
Given a new input image x of an object, we want to cor-
rectly predict the object’s class and 3D pose. Because of the
beneﬁts discussed above, such as scalability and ability to
easily add and remove objects, we formulate the problem as
a k-nearest neighbor search in a descriptor space: For each
object in the database, descriptors are calculated for a set of
template views and stored along with the object’s identity
and 3D pose of the view. In order to get an estimate for the
class and pose of the object depicted in the new input im-
age, we can compute a descriptor for x and search for the
most similar descriptors in the database. The output is then
the object and pose associated with them.
We therefore introduce a method to efﬁciently map an
input image to a compact and discriminative descriptor that
can be used in the nearest neighbor search according to the
Euclidean distance. For the mapping, we use a Convolu-
tional Neural Network (CNN) that is applied to the raw im-
age patch as input and delivers the descriptor as activations
of the last layer in one forward pass.
We show below how to train such a CNN to enforce the
two important properties already discussed in the introduc-
tion: a) The Euclidean distance between descriptors from
two different objects should be large; b) The Euclidean dis-
tance between descriptors from the same object should be
representative of the similarity between their poses.
3.1. Training the CNN
In order to train the network we need a set Strain of train-
ing samples, where each sample s = (x, c, p) is made of an
image x of an object, which can be a color or grayscale im-
age or a depth map, or a combination of the two; the identity
c of the object; and the 3D pose p of the object relative to
the camera.
Additionally, we deﬁne a set Sdb of templates where each
element is deﬁned in the same way as a training sample. De-
scriptors for these templates are calculated and stored with
the classiﬁer for k-nearest neighbor search. The template
set can be a subset of the training set, the whole training
set or a separate set. Details for the creation of training and
template data are given in the implementation section.
3.2. Deﬁning the Cost Function
We argue that a good mapping from the images to the de-
scriptors should be so that the Euclidean distance between
two descriptors of the same object and similar poses are
small and in every other case (either different objects or
different poses) the distance should be large. In particu-
lar, each descriptor of a training sample should have a small
distance to the one template descriptor from the same class
with the most similar pose and a larger distance to all de-
scriptors of templates from other classes, or the same class
but less similar pose.
We enforce these requirements by minimizing the fol-
lowing objective function over the parameters w of the
CNN:
L = Ltriplets + Lpairs + λ||w′||2
2 .
(1)
The last term is a regularization term over the parameters of
the network: w′ denotes the vector made of all the weights
of the convolutional ﬁlters and all nodes of the fully connect
layers, except the bias terms. We describe the ﬁrst two terms
Ltriplets and Lpairs below.
3.2.1
Triplet-wise terms
We ﬁrst deﬁne a set T of triplets (si, sj, sk) of training sam-
ples. Each triplet in T is selected such that one of the two
following conditions is fulﬁlled:
• either si and sj are from the same object and sk from
another object, or
3

• the three samples si, sj, and sk are from the same ob-
ject, but the poses pi and pj are more similar than the
poses pi and pk.
These triplets can therefore be seen as made of a pair of
similar samples (si and sj) and a pair of dissimilar ones (si
and sk). We introduce a cost function for such a triplet:
c(si, sj, sk) = max

0, 1 −
||fw(xi) −fw(xk)||2
||fw(xi) −fw(xj)||2 + m

,
(2)
where fw(x) is the output of the CNN for an input image x
and thus our descriptor for x, and m is a margin. We can
now deﬁne the term Ltriplets as the sum of this cost function
over all the triplets in T :
Ltriplets =
X
(si,sj,sk)∈T
c(si, sj, sk) .
(3)
It is easy to check that minimizing Ltriplets enforces our two
desired properties in one common framework.
The margin m serves two purposes. First, it introduces a
margin for the classiﬁcation. It also deﬁnes a minimum ratio
for the Euclidean distances of the dissimilar pair of samples
and the similar one. This counterbalances the weight regu-
larization term, which naturally contracts the output of the
network and thus the descriptor space. We set m to 0.01 in
all our experiments.
The concept of forming triplets from similar and
dissimilar pairs is adopted from the ﬁeld of metric
learning, in particular, the method of [37], where it
is used to learn a Mahalanobis distance metric.
Note
also that our deﬁnition of the cost is slightly differ-
ent from the one in [36], which uses c(si, sj, sk)
=
max

0, m + ||fw(xi) −fw(xj)||2
2 −||fw(xi) −fw(xk)||2
2

,
where m is set to 1. Our formulation does not suffer from
a vanishing gradient when the distance of the dissimilar
pair is very small (see suppl. material). Also the increase
of the cost with the distance of the similar pair is bounded,
thus putting more focus on local interactions. In practice,
however, with proper initialization and selection of m both
formulations deliver similar results.
3.2.2
Pair-wise terms
In addition to the triplet-wise terms, we also use pair-wise
terms. These terms make the descriptor robust to noise and
other distracting artifacts such as changing illumination. We
consider the set P of pairs (si, sj) of samples from the same
object under very similar poses, ideally the same, and we
deﬁne the Lpairs term as the sum of the squared Euclidean
distances between the descriptors for these samples:
Lpairs =
X
(si,sj)∈P
||fw(xi) −fw(xj)||2
2 .
(4)
64x64xC
input conv+max pool
28x28x16
16 x 8x8xC
conv+max pool
7 x 5x5x16
12x12x7
256
fully con.
fully con.
descriptor size
Figure 2. Network structure: We use a CNN made of two convo-
lutional layers with subsequent 2 × 2 max pooling layers, and two
fully connected layers. The activations of the last layer form the
descriptor for the input image.
This term therefore enforces the fact that for two images
of the same object and same pose, we want to obtain two de-
scriptors which are as close as possible to each other, even
if they are from different imaging conditions: Ideally we
want the same descriptors even if the two images have dif-
ferent backgrounds or different illuminations, for example.
As will be discussed in more detail in Section 4.1, this also
allows us to use a mixture of real and synthetic images for
training.
Note that we do not consider dissimilar pairs unlike work
in keypoint descriptors learning for example. With dissimi-
lar pairs the problem arises how strong to penalize a certain
distance between the two samples, given their individual la-
bels. Using triplets instead gives the possibility to only con-
sider relative dissimilarity.
3.3. Implementation Aspects
The exact structure of the network we train to compute
the descriptors is shown in Figure 2. It consists of two lay-
ers that perform convolution of the input with a set of ﬁlters,
max-pooling and sub-sampling over a 2 × 2 area and a rec-
tiﬁed linear (ReLU) activation function, followed by two
fully connected layers. The ﬁrst fully connected layer also
employs a ReLU activation, the last layer has linear output
and delivers the ﬁnal descriptor.
We optimize the parameters w of the CNN by Stochastic
Gradient Descent on mini-batches with Nesterov momen-
tum [32]. Our implementation is based on Theano [3].
The implementation of the optimization needs some spe-
cial care: Since we are working with mini-batches, the data
corresponding to each pair or triplet has to be organized
such as to reside within one mini-batch. The most straight-
forward implementation would be to place the data for each
pair and triplet after each other, calculate the resulting gra-
dient wrt. the network’s parameters individually and sum
them up over the mini-batch. However, this would be inef-
ﬁcient since descriptors for templates would be calculated
multiple times if they appear in more than one pair or triplet.
4

To assemble a mini-batch we start by randomly taking
one training sample from each object.
Additionally, for
each of them we add its template with the most similar pose,
unless it was already included in this mini-batch. This is it-
erated until the mini-batch is full. However, this procedure
can lead to very unequal numbers of templates per object
if, for instance, all of the selected training samples have the
same most similar template. We make sure that for each ob-
ject at least two templates are included by adding a random
one if necessary. Pairs are then formed by associating each
training sample with its closest template. Additionally, for
each training sample in the mini-batch we initially create
three triplets. In each of the

## experiments
We compare our approach to LineMOD and HOG on the
LineMOD dataset [13]. This dataset contains training and
test data for object recognition and pose estimation of 15
objects, with accurate ground truth. It comes with a 3D
mesh for each of the objects. Additionally, it also provides
sequences of RGB images and depth maps recorded with a
Kinect sensor.
4.1. Dataset Compilation
We train a CNN using our method on a mixture of syn-
thetic and real world data. As in [14], we create synthetic
training data by rendering the mesh available for each of the
objects in the dataset from positions on a half-dome over
the object, as shown in Fig. 1 on the left. The viewpoints
are deﬁned by starting with a regular icosahedron and re-
cursively subdividing each triangle into 4 sub-triangles. For
the template positions the subdivison is applied two times.
After removing the lower half-sphere we end up with 301
evenly distributed template positions. Additional training
data is created by subdividing one more time, resulting in
1241 positions.
From each pose we render the object standing on a plane
over an empty background using Blender1. We parameter-
ize the object pose with the azimuth and elevation of the
camera relative to the object. We store the RGB image as
well as the depth map.
For the real world data we split the provided sequences
captured with the Kinect randomly into a training and a test
set. We ensure an even distribution of the samples over
the viewing hemisphere by taking two real world images
close to each template, which results roughly in a 50/50 split
of the data into training and test. Preliminary experiments
showed very little to no variance over the different train/test
splits and, thus, all results presented here report runs on one
random split, ﬁxed for each experiment.
The whole training data set is augmented by making
multiple copies with added noise. On both RGB and depth
channel we add a small amount of Gaussian noise. Addi-
tionally, for the synthetic images, we add larger fractal noise
on the background, to simulate diverse backgrounds.
Note that the template views, which are ultimately used
in the classiﬁcation are purely synthetic and noise-free ren-
derings on clean backgrounds. The algorithm, thus, has to
learn to map the noisy and real world input data to the same
location in descriptor space as the clean templates.
As pointed out in [14] some of the objects are rotation-
ally invariant, to different degrees. Thus, the measure of
similarity of poses used for the evaluation and, in our case
to deﬁne pairs and triplets, should not consider the azimuth
of the viewing angle for those objects. We treat the bowl
object as fully rotationally invariant. The classes eggbox,
glue are treated as symmetric, meaning a rotation by 180◦
around the z-axis shows the same pose again. The cup is a
special case because it looks the same from a small range of
poses, but from sufﬁcient elevation such that the handle is
visible, the exact pose could be estimated. We also treat it
as rotationally invariant, mainly to keep the comparison to
LineMOD fair.
We extract a patch centered at the object and capturing
a ﬁxed size window in 3D at the distance of the object’s
center. In order to also address the detection part in a sliding
window manner, it would be necessary to extract and test
several scales. However, only a small range of scales needs
to be considered, starting with a maximal one, deﬁned by
the depth at the center point, and going down until the center
of the object is reached.
Before applying the CNN we normalize the input im-
ages. RGB images are normalized to the usual zero mean,
unit variance. For depth maps we subtract the depth at the
center of the object, scale down such that 20cm in front and
1http://www.blender.org
5

behind the object’s center are mapped to the range of [−1, 1]
and clip everything beyond that range.
The test sequences captured with the Kinect are very
noisy. In particular, there are many regions with undeﬁned
depth, introducing very large jumps for which the convolu-
tional ﬁlters with ReLU activation functions might output
overly strong output values. Therefore, we pre-process the
test data by iteratively applying median ﬁlters in a 3 × 3
neighborhood, but only on the pixels for which the depth is
available, until all gaps are closed.
4.2. Network Optimization
For the optimization we use the following protocol: We
initially train the network on the initial dataset for 400
epochs, an initial learning rate of 0.01 and a momentum
of 0.9. Every 100 epochs the learning rate is multiplied by
0.9. Then we perform two rounds of bootstrapping triplet
indices as explained in Section 3.3, and for each round we
train the CNN for another 200 epochs on the augmented
training set. In the end we train another 300 epochs with the
learning rate divided by 10 for ﬁnal ﬁne-tuning. The regu-
larization weight λ is set to 10−6 in all our experiments.
4.3. LineMOD and HOG
We compare our learned descriptors to the LineMOD
descriptor and HOG as a baseline, as it is widely used as
representation in the related work. For LineMOD we use
the publicly available source code in OpenCV. We run it
on the same data as our method, except for the median ﬁl-
ter depth inpainting and normalization: LineMOD handles
the missing values internally and performed better without
these pre-processing operations.
For HOG we also use the publicly available implemen-
tation in OpenCV. We extract the HOG descriptors from the
same data we use with our CNN. We use a standard setup
of a 64 × 64 window size, 8 × 8 cells, 2 × 2 cells in a block
and a block stride of 8, giving a 1764-dimensional descrip-
tor per channel. We compute descriptors on each RGB and
depth channel individually and stack them. For evaluation
we normalize all descriptors to length 1 and take the dot
product between test and template descriptors as similarity
measure.
4.4. Manifolds
Figure 1 plots the views of three objects after being
mapped into 3-dimensional descriptors, for visualization
purposes. As can be seen, not only the descriptors from the
different objects are very well separated, but they also cap-
ture the geometry of the corresponding poses. This means
that the distances between descriptors is representative of
the distances between the corresponding poses, as desired.
For longer descriptors we show an evaluation of the re-
lation between distances of descriptors and similarity be-
tween poses in Figure 3. For each object, we computed
the distances between every sample in the test set and every
template for the same object in the training set, as well as the
angles between their poses. We then plot a two-dimensional
histogram over these angle/distance pairs. Correlation be-
tween small angles and large distances indicates the risk of
missed target templates, and correlation between large an-
gles and small distances the risk of incorrect matches. Ide-
ally the histograms should therefore have large values only
on the diagonal.
The histograms for the descriptors computed with our
method clearly show that the distance between descriptors
increase with the angle between the views, as desired, while
the histograms for LineMOD and HOG show that these de-
scriptors are much more ambiguous.
Additionally, the ability of the descriptors to separate the
different classes is evaluated in Figure 4. For every test sam-
ple descriptor we compute the distance to the closest tem-
plate descriptor of the same object and the closest from any
other object and plot a histogram over those ratios. Clearly,
descriptors obtained with our method exhibit a larger ratio
for most samples and thus separate the objects better.
4.5. Retrieval Performance
What we ultimately want from the descriptors is that
nearest neighbors are from the same class and have similar
poses. In order to evaluate the performance we thus perform
the following comparisons.
The scores reported for LineMOD in [14] represent the
accuracy of the output of the whole processing pipeline, in-
cluding the descriptor calculation, retrieval of similar tem-
plates, pruning the set with heuristics and reﬁnement of the
pose for a set of candidate matches by aligning a voxel
model of the object. The contribution of this work is to
replace the descriptors for the retrieval of templates with
similar pose. Thus, we evaluate and compare this step in
separation of the rest of the pipeline.
Evaluation Metric
For each test sample we consider the
k-nearest neighbors according to the descriptors and simi-
larity metric of each method, the Euclidean distance in our
case, the dot product for HOG, and the matching score of
LineMOD. Among those k nearest templates we search for
the one with the best closest pose to the test sample’s pose,
assuming that this one would perform best in the subsequent
reﬁnement process and thus ﬁnally be selected. The pose
error is measured by the angle between the two positions
on the viewing half-sphere. We deﬁne the

## related_work
Our work is related to several aspects of Computer Vi-
sion, and we focus here on the most relevant and represen-
tative work. Our approach is clearly in the framework of
2D view speciﬁc templates [15], which is conceptually sim-
ple, supported by psychophysical experiments [33], and was
successfully applied to various problems and datasets over
the last years [22, 19, 13, 11, 8, 28].
However, most of these works rely on handcrafted rep-
resentations of the templates, for example HOG [7] or
LineMOD [14]. In particular, LineMOD was designed ex-
plicitly in the context of object detection and pose estima-
tion. However these handcrafted representations are subop-
timal compared to statistically learned features. [19, 11, 28]
show how to build discriminative models based on these
representations using SVM or boosting applied to training
data. [19, 28] do not consider the pose estimation problem,
while [11] focuses on this problem only, with a discrimina-
tively trained mixture of HOG templates. Exemplars were
recently used for 3D object detection and pose estimation
in [1], but still rely on a handcrafted representation.
As mentioned in the introduction, our work is inﬂuenced
by work developed for keypoint descriptor learning. Some
of these methods are applied to existing descriptors to make
them more discriminative, such as in [10, 31], but others are
trained directly on image data. [5] introduces datasets made
of “positive pairs” of patches corresponding to the same
physical points and “negative pairs” of patches correspond-
ing to different points. It is used for example in [35] to learn
a binary descriptor with boosting. [20] uses a “siamese” ar-
chitecture [6] to train a neural network to compute discrimi-
native descriptors. Our approach is related to this last work,
but the notion of pose is absent in their case. We show how
to introduce this notion by using triplets of training exam-
ples in addition to only pairs.
Instead of relying on rigid templates as we do, many
works on category recognition and pose estimation rely
on part-based models. [30] pioneered this approach, and
2

learned canonical parts connected by a graph for object
recognition and pose estimation.
[26] extends the De-
formable Part Model to 3D object detection and pose es-
timation. [25] uses contours as parts. One major drawback
of such approaches is that the complexity is typically linear
with the number of objects. It is also not clear how impor-
tant the “deformable” property really is for the recognition,
and rigid templates seem to be sufﬁcient [9].
Our approach is also related to manifold learning [27].
For example, [29] learns an embedding that separates ex-
tremely well the classes from the MNIST dataset of digit
images, but the notion of pose is absent. [12] learns either
for different classes, also on the MNIST dataset, or for vary-
ing pose and illumination, but not the two simultaneously.
More recently, [2] proposes a method that separates mani-
folds from different categories while being able to predict
the object poses, and also does not require solving an infer-
ence problem, which is important for efﬁciency. However, it
relies on a discretisation of the pose space in a few classes,
which limits the possible accuracy. It also relies on HOG
for the image features, while we learn the relevant image
features.
Finally, many works focus as we do on instance recog-
nition and pose estimation, as it has important applications
in robotics for example. [14] introduced LineMOD, a fast
but handcrafted presentation of template for dealing with
poorly textured objects.
The very recent [4, 34] do not
use templates but rely on recognition of local patches in-
stead. However they were demonstrated on RGB-D images,
and local recognition is likely to be much more challenging
on poorly textured objects when a depth information is not
available. [17] also expects RGB-D images, and uses a tree
for object recognition, which however still scales linearly in
the numbers of objects, categories, and poses.

## conclusion
We have shown how to train a CNN to map raw input im-
ages from different input modalities to very compact output
descriptors using pair-wise and triplet-wise constraints over
training data and template views. Our descriptors signiﬁ-
cantly outperform LineMOD and HOG, which are widely
used for object recognition and 3D pose estimation, both
in terms of accuracy and descriptor length. Our represen-
Figure 7. Generalization to objects not seen during training. Left:
Histogram of correlation between Pose Similarity and Descriptor
Distance for the duck that was not included during training for this
experiment. The recognition rate is only slightly reduced com-
pared to when the duck is used during training. Right: Difﬁcult
examples of correct and mis-classiﬁcations. The duck is mostly
confused with the ape, which looks similar in the depth image un-
der some angles.
tation therefore replaces them advantageously. Tests on the
capability to generalize to unseen objects also have shown
promising results. For further investigation we will make
our code available upon request.
8