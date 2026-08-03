# Maximum-Margin Structured Learning with Deep Networks for 3D Human Pose Estimation

> 2015 · id: W1905368000 · 来源: web-agent
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

Maximum-Margin Structured Learning with Deep Networks for 3D Human Pose
Estimation
Sijin Li
sijin.li@my.cityu.edu.hk
Weichen Zhang
wczhang4-c@my.cityu.edu.hk
Department of Computer Science
City University of Hong Kong
Antoni B. Chan
abchan@cityu.edu.hk
Abstract
This paper focuses on structured-output learning using
deep neural networks for 3D human pose estimation from
monocular images. Our network takes an image and 3D
pose as inputs and outputs a score value, which is high when
the image-pose pair matches and low otherwise. The net-
work structure consists of a convolutional neural network
for image feature extraction, followed by two sub-networks
for transforming the image features and pose into a joint
embedding. The score function is then the dot-product be-
tween the image and pose embeddings.
The image-pose
embedding and score function are jointly trained using a
maximum-margin cost function. Our proposed framework
can be interpreted as a special form of structured support
vector machines where the joint feature space is discrimi-
natively learned using deep neural networks. We test our
framework on the Human3.6m dataset and obtain state-of-
the-art results compared to other recent methods. Finally,
we present visualizations of the image-pose embedding
space, demonstrating the network has learned a high-level
embedding of body-orientation and pose-conﬁguration.
1. Introduction
Human pose estimation from images has been studies for
decades. Due to the dependencies among joint points, it can
be considered a structured-output task. In general, human
pose estimation approaches can be divided by two types:
1) prediction-based methods; 2) optimization-based meth-
ods. The ﬁrst type of approach views pose estimation as a
regression or detection problem [18, 31, 19, 30, 14]. The
goal is to learn the mapping from the input space (image
features) to the target space (2D or 3D joint points), or to
learn classiﬁers to detect speciﬁc body parts in the image.
This type of method is straightforward and usually fast in
the evaluation stage. Toshev et al. [31] trained a cascaded
network to reﬁne the 2D joint locations in an image stage
by stage. However, this approach does not explicitly con-
sider the structured constraints of human pose. Followup
work [14, 30] learned the pairwise relationship between 2D
joint positions, and incorporated them into the joint pre-
dictions. Limitations of prediction-based methods include:
the manually-designed constraints might not be able to fully
capture the dependencies among the body joints; poor scal-
ability to 3D joint estimation when the search space needs
to be discretized; prediction of only a single pose when mul-
tiple poses might be valid due to partial self-occlusion.
Instead of estimating the target directly, the second type
of approach learns a score function, which takes both an im-
age and a pose as inputs, and produces a high score for cor-
rect image-pose pairs and low scores for unmatched image-
pose pairs. Given an input image x, the estimated pose y∗
is the pose that maximizes the score function, i.e.,
y∗= argmax
y∈Y
f(x, y),
(1)
where Y is the pose space. If the score function can be
properly normalized, then it can be interpreted as a proba-
bility distribution, either a conditional distribution of poses
given the image, or a joint distribution over both images and
joints. One popular model is pictorial structures [9], where
the dependencies between joints are represented by edges
in a probabilistic graphical model [16]. As an alternative
to generative models, structured-output SVM [32] is a dis-
criminative method for learning a score function, which en-
sures a large margin between the score values for correct
input pairs and for incorrect input pairs [24, 10].
As the score function takes both image and pose as input,
there are several ways to fuse the image and pose informa-
tion together. For example, the features can be extracted
jointly according to the image and poses, e.g., the image
features extracted around the input joint positions could be
viewed as the joint feature representation of image and pose
[9, 26, 34, 8]. Alternatively, features from the image and
pose can be extracted separately and concatenated, and the
score function trained to fuse them together [11, 12]. How-
ever, with these methods, the features are hand-crafted, and
performance depends largely on the quality of the features.
On the other hand, deep neural networks have been
shown to be good at extracting informative high-level fea-
12848

tures [27, 3]. In this paper, we propose a uniﬁed framework
for maximum-margin structured learning with deep neural
network for human pose estimation.
Our uniﬁed frame-
work jointly learns the image and pose feature representa-
tions and the score function. In particular, our network ﬁrst
extracts separate feature embeddings from the image input
and from the 3D pose input. The score function is then the
dot-product between the image and pose embeddings. The
score function and feature embeddings are trained using
a maximum-margin criteria, resulting in a discriminative
joint-embedding of image and 3D pose. The dot-product
score function is efﬁcient to compute, and allows for fast
inference over a large set of candidate poses. In addition,
our proposed framework is quite general and can be applied
to a wide range of structured-output tasks.
2. Related work
Here we review recent related works in deep neural net-
work and structured learning.
2.1. 2D pose estimation via detection with deep net-
works
Traditional pictorial structure models usually apply lin-
ear ﬁlters on hand-crafted features, e.g., HoG and SIFT, to
calculate the probability of the presence of body parts or ad-
jacent body-joint pairs. As shown in [8], the quality of the
features are critical to the performance, and, while success-
ful for other tasks, these hand-crafted features may not be
necessarily optimal for pose estimation. Alternatively, with
sufﬁcient data, it is possible to learn the features directly
from training data. In recent years, deep neural networks,
especially convolutional neural networks (CNN), have been
shown to be effective in learning rich features [23, 17].
Jain et al. [14] trains a CNN as a sliding-window detec-
tor for each body part, and the resulting body-joint detec-
tion maps are smoothed using a learned pairwise relation-
ship between joints. Tompson et al. [30] extends [14] by
feeding the body-joint detection maps into a modiﬁed con-
volutional layer that performs pairwise smoothing, allowing
feature extraction and pairwise relationships to be jointly
optimized. Chen et al. [5] uses a deep CNN to predict the
presence of joints and the pairwise relationships between
joints, and the CNN output is then used as the input into a
pictorial structure model for 2D pose estimation.
The advantage of these approaches is that the features
extracted by deep networks usually lead to better perfor-
mance. However the detection-based methods for 2D pose
estimation are not directly applicable to 3d pose estima-
tion due to the need to discretize a large pose space – the
number of joint positions grows cubicly with the resolution
of the discretization, making inference computationally ex-
pensive [4]. In addition, it is difﬁcult to predict 3D coordi-
nates from only a local window around a joint, without any
other contextual information.
2.2. Pose regression via deep networks
In contrast to detection-based methods, regression-based
methods aim to directly predict the coordinates of the body-
joints in the image. Toshev et al. [31] trains a cascade CNN
to predict the 2D coordinates of joints in the image, where
the CNN inputs are the image patches centered at the co-
ordinates predicted from the previous stage. Li et al. [19]
use a multi-task framework to train a CNN to directly pre-
dict a 2D human pose, where auxiliary tasks consisting of
body-part detection guide the feature learning. This work
was later extended for 3D pose estimation from single 2D
images [18].
One disadvantage of regression-based methods is that
they can only predict one pose for a given image. This may
cause difﬁculties on images where the pose is ambiguous
due to partial-self occlusion, and hence several poses might
be valid. In contrast, our proposed model is better able to
handle ambiguities since several valid image-pose pairs can
have similar high scores.
2.3. Structured-output prediction and feature em-
bedding
Rodr´ıguez [24] represents the score function between
word labels and images as the dot-product between the
word-label feature and an image embedding, and trains
a structured SVM (SSVM) to learn the weights to map
the bag-of-words image features to the image embedding.
Dhungel et al. [7] uses structured learning and deep net-
works to segment mammograms. First, a network is trained
to generate a unary potential function. Next, a linear SSVM
score function is trained on the output of the deep network,
as well as other potential functions. Osadchy et al. [22] ap-
ply structured learning and CNN for face detection and face
pose estimation. The CNN was trained to map the face im-
age to a manually-designed face pose space. A per-sample
cost function is deﬁned with only one global minimum so
that the ground-truth pose has minimum energy. In contrast
to [7, 24, 22], we learn the feature embedding and score
prediction jointly within a maximum-margin framework.
Jaderberg et al. [13] proposed a deep structured-output
network for recognizing text in images. The score function
is a conditional random ﬁeld (CRF), where the input is an
image and the output is a word. The unary and higher-order
potential functions of the CRF are two CNNs, which are
trained to recognize single characters and n-grams in the im-
age, and the framework is jointly trained with a maximum
margin cost. In the context of pose recognition, [13] is a pic-
torial structure model with higher-order terms, whereas our
method is similar to learning a non-linear embedding with a
linear SSVM score function. In particular, the main differ-
ence is that we do not manually-design the score function
to encode the output structure as pairwise or higher-order
2849

Figure 1. Deep-network score function. The image input is fed through a set of convolutional layers for image feature extraction. Two
separate sub-networks are used to embed the image and the pose into a common space, and the score function is the dot-product between the
two embeddings. An auxiliary 3D body-joint prediction task is used to guide the network to ﬁnd good image features. Each convolutional
layer is followed by a max-pooling layer, which is not drawn to reduce clutter.
terms (i.e., the CRF), but instead train the network to learn
both image and pose embeddings such that a score function
can be represented as dot-product. Furthermore, the inter-
nal image representations in [13] are strongly supervised,
consisting of character/n-gram classiﬁers, whereas the inter-
nal representations (image/pose embeddings) in our method
are learned from the data. Although both methods use a
maximum-margin cost, [13] uses a ﬁxed margin for all in-
put/output pairs, whereas our method uses margin rescaling.
2.3.1
Unsupervised joint feature embedding
Deep networks have also been used to learn joint embed-
dings for multi-modal inputs.
Ngiam et al. [21] embed
audio-video pairs by jointly training autoencoders with a
shared middle layer.
Pereira et al. [28] build a genera-
tive model for image-text pairs by adding a binary hidden
layer on top of image-speciﬁc and text-speciﬁc deep Boltz-
mann machines. Andrew et al. [1] proposes deep canon-
ical correlation analysis (DCCA), where each input view
is passed through a separate deep network (implementing
a non-linear transformation), and the networks are jointly
trained so that the their outputs are maximally correlated. In
contrast to these works, our joint embedding is learned dis-
criminatively using a maximum-margin cost. In addition,
our embedding is loosely coupled, i.e., the image and pose
embeddings do not explicitly share latent variables (layers).
Rather the two embeddings are optimized through the dot-
product similarity and supervised cost function, similar to
learning a kernel embedding.
3. Maximum-margin structured learning
Our goal is to learn a score network that can assign max-
imum score to correct image-pose pairs and low scores to
other pairs. The network structure is illustrated in Figure 1.
Our network consists of two main components: an image
feature extraction sub-network and an image-pose embed-
ding sub-network. For the ﬁrst sub-network, a CNN ex-
tracts high-level image features from the raw image. For the
second sub-network, the image features and pose (3D joint
coordinates) are separately fed through fully-connected lay-
ers, mapping them into two embedding spaces. The score
function is then the dot-product between the two embed-
dings. Although the image/pose embeddings are calculated
from separate sub-networks, training the full network will
align the image/pose embeddings into a joint space, such
that their dot-product is a suitable score function.
To train the network, we use a maximum-margin cost
function that forces the score of the ground-truth image-
pose pair to be larger than other image-pose pairs by at least
a margin. We use a re-scaling margin, which is a function
of the distance between the ground-truth pose and the other
pose. In order to encourage image features that preserve
pose information, we add an auxiliary task consisting of 3D
body-joint prediction during training.
In the following, we use x to represent the image input,
y as the ground-truth matching pose (3D joint coordinates),
Y as the pose space, and θ as the network parameters.
3.1. Image feature extraction
The goal of the image extraction sub-network is to con-
vert the raw input image to a more compact representation
with pose information preserved. We use a deep CNN, con-
sisting of 3 sets of convolution and max-pooling layers, to
extract image features from the image. We use rectiﬁed lin-
ear units (ReLU) [20] as the activation function in the ﬁrst
2 layers, and the linear activation function in the 3rd layer.
The outputs of the pooling layers is a set of feature maps,
denoted as convj(x), where j is the layer number. Each
feature in the map has a receptive ﬁeld in the input image,
with higher layer features having larger receptive ﬁelds. In-
tuitively, the higher layer features will contain global in-
formation about the pose, which would be useful for dis-
2850

tinguishing between grossly different poses. On the other
hand, the lower layer features contain more detailed infor-
mation about the pose, which will be helpful in distinguish-
ing between similar poses.
3.2. Image-pose embedding
The image and pose inputs are in different spaces, and
the goal of the image-pose embedding sub-network is to
project the image features and the 3D pose into a joint
embedding space where they can be compared effectively.
The architecture of image and pose embedding network
is shown in Figure 1. Inspired by [29, 19], we use fea-
tures from both the middle- and top-convolutional layers.
The middle- and top-layer features are each passed through
separate fully connected layers, and then concatenated and
passed through two more fully connected layers to form the
image embedding fI(x). Speciﬁcally,
fI(x) = h4(h3(
h1(conv2(x))
h2(conv3(x))

)),
(2)
where the activation function hi(x) = ReLU(W T
i x+bi) is
a rectiﬁed linear unit with weight matrix Wi and bias bi.
The input pose y is represented by the 3D coordinates
of the body-joint locations, the dimensions of which are
strongly correlated due the dependencies among joints. The
pose is mapped into a non-linear embedding, so that it can
be more easily combined with the image embedding. We
use 2 fully connected layers for this transformation,
fJ(y) = h6(h5(y)).
(3)
3.3. Score prediction
We represent the score function between the image and
pose inputs fS(x, y) as the inner-product between the image
embedding fI(x) and pose embedding fJ(y), i.e.,
fS(x, y) = ⟨fI(x), fJ(y)⟩.
(4)
One advantage of using inner-product is that the corre-
sponding dimensions of the image/pose embedding vectors
interact directly, which makes aligning the two embeddings
easier. Another advantage is that it is very efﬁcient to cal-
culate. The calculation of the pose embedding does not de-
pend on the image features, which means it can be calcu-
lated ofﬂine if the set of candidate poses is ﬁxed.
Training the network will map the image and pose into
similar embedding spaces, where their dot-product simi-
larity serves as a suitable score function.
This can be
loosely interpreted as learning a multi-view “kernel” func-
tion, where the “high-dimensional” feature space is the
learned joint embedding.
Our score function can also be interpreted as a SSVM,
where the joint features are the element-wise product be-
tween the learned image and pose embeddings,
f ′
S(x, y) = ⟨w, fI(x) ◦fJ(y)⟩
(5)
where ◦indicates element-wise multiplication, and w is the
SSVM weight vector. The equivalence is seen by noting
that during network training the weights w can be absorbed
into the embedding functions {fI, fJ}. In our framework,
these embedding functions are discriminatively trained.
3.4. Maximum margin cost
Inspired by maximum-margin structured SVM [33], we
use a maximum margin cost to learn the score function.
The maximum margin cost ensures that the difference be-
tween the scores of two input pairs is at least a particular
value (i.e., the margin). Different from the standard SVMs,
with structured-SVM can have a margin that changes values
based on dissimilarity between the two input pairs.
Similar to the structured-SVM, we use the margin re-
scaling surrogate loss,
LM(x, y, ˆy) = max(0, fS(x, ˆy) + ∆(ˆy, y) −fS(x, y)),
(6)
where (x, y) is a training image-pose pair, ∆(y, y′) is a non-
negative margin function between two poses, and ˆy is the
pose that most violates the margin constraint1,
ˆy = argmax
y′∈Y
fS(x, y′) + ∆(y, y′) −fS(x, y).
(7)
Intuitively, a pose with a high predicted score, but that is
far from the ground-truth pose, is more likely to be the most
violated pose. For the margin function, we use the mean per
joint error (MPJPE), i.e.,
∆(y, y′) = 1
J
J
X
j=1
∥yj −y′
j∥,
(8)
where yj indicates the 3D coordinates of j-th joint in pose
y, and J is the number of body-joints.
When the loss function in (6) is zero, then the score of the
ground-truth image-pose pair (x, y) is at least larger than
the margin for all other image-pose pairs (x, y′),
fS(x, y) ≥fS(x, y′) + ∆(y′, y), ∀y′ ∈Y.
(9)
On the other hand, if (6) is greater than 0, then there exists at
least one pose y′ whose score f(x, y′) violates the margin.
3.5. Multi-task global cost function
Following [18, 19], in order to encourage the image em-
bedding to preserve more pose information, we include an
auxiliary training task of predicting the 3D pose. Speciﬁ-
cally, we add a 3D pose prediction layer after the penulti-
mate layer of the image embedding network,
fP (x) = g7(h3),
(10)
where h3 is the output of the penultimate layer of the image
embedding, and gi(x) = tanh(W T
i x + bi) is the tanh ac-
tivation function. The cost function for the pose prediction
1Note that ˆy depends on the input (x, y) and network parameters θ. To
reduce clutter, we write ˆy instead of ˆy(x, y, θ) when no confusion arises.
2851

network structure for ﬁnding the most-violated pose
network structure for maximum-margin training
Figure 2. (left) Network structure for calculating the most violated pose. For a given image, the score values are predicted for a set of
candidate poses. The re-scaling margin values are added, and the largest value is selected as the most-violated pose. Thick arrows represent
an array of outputs, with each entry corresponding to one candidate pose. (right) Network structure for maximum-margin training. Given
the most-violated pose, the margin cost and pose prediction cost are calculated, and the gradients are passed back through the network.
task is the square difference between the ground-truth pose
and predicted pose,
LP (x, y) = ∥fP (x) −y∥2.
(11)
Finally,
given
a
training
set
of
image-pose
pairs
{(x(i), y(i))}N
i=1, our global cost function consists the struc-
tured maximum-margin cost, pose estimation cost, as well
as a regularization term on the weight matrices,
cost(θ) = 1
N
N
X
i=1
LM(x(i), y(i), ˆy(i))
+ 1
N λ
N
X
i=1
LP (x(i), y(i)) + α
7
X
j=1
∥Wj∥2
F
(12)
where i is the index for training samples, λ is the weight-
ing for pose prediction error, α is the regularization param-
eter, and θ = {(Wi, bi)}7
i=1 are the network parameters.
Note that gradients from LP only affect the CNN and high-
level image features (FC1-FC3), and have no direct effect
on the pose embedding network or image embedding layer
(FC4). Therefore, we can view the pose prediction cost as a
regularization term for the image features. Figure 2 shows
the overall network structure for calculating the max-margin
cost function, as well as ﬁnding the most violated pose.
4. Training Algorithm
We use back-propagation [25] with stochastic gradient
descent (SGD) to train the network. Similar to SSVM [15],
our training procedure iterates between ﬁnding the most-
violated poses and updating the network parameters:
1. Find the most-violated pose ˆy for each training pair
(x, y) using the pose selection network with current
network parameters (Fig. 2 left);
2. Input (x, y, ˆy) into the max-margin training network
(Fig. 2 right) and run back-prop to update parameters.
We call the tuple (x, y, ˆy) the extended training data. The
training data is processed in mini-batches. We found that
using momentum between mini-batches, which updates the
parameters using the weighted average of the current gradi-
ent and previous update, always hinders convergence. This
is because the maximum-margin cost selects different most-
violated poses in each batch, which makes the gradient di-
rection change rapidly between batches. To speed up the
convergence of SGD, we use a line-search to ﬁnd the best
step-size for each mini-batch update. This was necessary
because the the back-propagated gradients have high dy-
namic range, which stems from the cost function consisting
of the difference between network outputs.
Although our score calculation is efﬁcient, it is still com-
putationally expensive to search the whole pose space to
ﬁnd the most-violated pose. Instead, we form a candidate
set YB for each mini-batch, and ﬁnd the most-violated poses
within the candidate set. The candidate set consists of C
poses sampled from the pose space Y. In addition, we ob-
served that some poses are selected as the most-violated
poses multiple times during training. Therefore, we also
maintain a working set of most-violated poses, and include
the top K most-frequent violated poses in the candidate set.
Our training procedure is summarized in Algorithm 1.
Note that the selection of the most-violated pose from a can-
didate set, along with the back-propagation of the gradient
for that pose, can be interpreted as a max-pooling operation
over the candidate set.
5. Experiments
In this section, we evaluate our maximum margin struc-
tured learning network on human pose estimation dataset.
5.1. Dataset
We evaluate on the Human3.6M dataset [12], which con-
tains around 3.6 million frames of video. The videos are
recorded with four RGB camera, along with a MoCap sys-
2852

Algorithm 1 Max-margin structured-network training
input: training set {(x(i), y(i))}N
i=1, pose space Y, num-
ber of iterations M, number of mini-batches B, number
of candidate poses C, number of most frequent violated
poses K.
output: network parameters θ.
V = ∅
{working set of most-violated poses}
for t = 1 to M do {loop over the whole training set}
for b = 1 to B do {loop over mini-batches}
B = ReadBatch()
{get the current set of candidate poses YB}
YB = UniformSample(Y, C) {get C poses}
YB = YB ∪KMostFrequent(V, K)
{build the extended training data D}
D = ∅
for all (x, y) ∈B do
{calculate the most violated pose for (x, y)}
ˆy = argmax
y′∈YB
⟨fI(x), fJ(y′)⟩+ ∆(y, y′)
D = D ∪(x, y, ˆy) {add to extended data}
V = V ∪ˆy {add to working set of violated poses}
end for
{update network parameters}
StepSize = LineSearch(cost, D, θ)
θ = SGD(cost, D, θ, StepSize)
end for
end for
tem for measuring the joint positions. We treat the four
RGB images separately, and project the MoCap coordinates
to each camera coordinate system as the ground-truth pose.
As in [12, 18], the image input is a cropped image around
the human. The training images are obtained by extracting
a square image according to the bounding box provided in
Human3.6M dataset [12], and resizing it to 128×128. As
in [17], we augment the image training set by local trans-
lations and by adding random pixel noise during training.
For local translations, a 112×112 sub-image is randomly
selected from the training image. For pixel noise, random
noise is added to all pixels according to the RGB covariance
matrix calculated over the whole training set. The 3D pose
input is a vector of the 3D coordinates of 17 body-joints.
5.2. Experiment setup
We follow the same protocol as in [18] for the training
and test set – we use 5 subjects (S1, S5, S6, S7, S8) for
training and validation, and 2 subjects (S9, S11) for testing.
Our structured-output network (denoted as StructNet) is
trained using the algorithm from Section 4. Given a test im-
age, ideally, the predicted pose should be found by search-
ing the entire pose space Y for the pose with maximum
score, as in (1). However, the pose space is continuous and
exhaustive search is computationally intractable. Instead,
we consider several approaches to approximate the search:
• StructNet-Max – the predicted pose is the pose in the
training set with maximum score.
• StructNet-Avg(A) – since the training and test sets
contain different subjects, the poses in the training set
will not perfectly match the subjects in the test set. To
allow for more pose variation, the predicted pose is the
average of the A training poses with highest scores.
• StructNet-Avg(A)-APF – the problem with using
StructNet-Avg is that the average pose is not guaran-
teed to be a valid pose. We use the annealing particle
ﬁltering (APF) [6] to generate a valid pose that best
matches the pose estimated with StructNet-Avg(A).
Speciﬁcally, APF adjusts the joint angles of a template
pose to minimize the MPJPE with the StructNet-Avg
pose. The template pose, which is a neutral “T” pose
from the test subject, is initialized with the joint-angles
from one of the top A poses. After APF converges, the
joint-angles are converted into 3D joint coordinates.
The pose estimates on the test set are evaluated using
MPJPE [12]. We also compare against multi-task deep net-
works (DconvMP-HML) [18], which trains a CNN using
the pose prediction cost (Eq. 11), and LinKDE, the best per-
forming method in [12].
5.3. Implementation details
The sizes of the network layers are shown in Figure 1.
We follow the multi-task framework in [18] to initialize the
weights for the convolutional layers. All the weight matri-
ces for other layers are randomly initialized. When training
the maximum-margin network, we ﬁx the weights in the
convolutional layers while still doing the data augmentation
of the input image. The line-search was performed over the
range [10−7, 102]. We approximate the pose space Y with
all the poses in the training set. The batch size is 128, and
the size of the sampled candidate set is C = 2000. The
number of most-frequent violated poses is K = 10. The
weight for the auxiliary prediction task is λ = 1, and the
regularization parameter is α = 0.0001. We use dropout
in the fully-connected layers {h1, h2}. The dropout rate is
75%. Our network is implemented in Theano [2].
5.4. Experiment results
Table 1 presents the MPJPE results on the test set for
each action, as well as the overall average. We ﬁrst compare
the different methods for estimating the pose from Struct-
Net. On all actions, StructNet-Avg (the average of the top
scoring poses) yields better results than StructNet-Max (the
maximum scoring pose), with overall reduction in error of
about 10% when A = 500. Figure 3 plots the error versus
different values of A. The error stabilizes between A = 500
and A = 1000, which represents ∼0.5% of the poses in
the training set. Furthermore, applying APF to the average
2853

0
500
1000
1500
2000
2500
A
60
80
100
120
140
160
MPJPE
Walking
TakingPhoto
WalkingDog
Greeting
Discussion
Eating
Figure 3. Pose error when averaging the top-A highest scoring
training poses (StructNet-Avg(A)). A = 500 represents ∼0.5%
of the poses in the training set.
pose from StructNet-Avg yields a valid pose with roughly
the same MPJPE as StructNet-Avg.
Comparing to previous works, the error for StructNet-
Avg is less than DconvMP-HML [18] and LinKDE [12] on
all actions. The overall error is reduced 9.2% (from 133.54
for DconvMP-HML to 121.31 for StructNet-Avg(500)-
APF). Also note that our method generates valid poses,
whereas DconvMP-HML and LinKDE do not.
Next, we consider the role of the auxiliary pose predic-
tion task in our network. We evaluate the performance of
the auxiliary pose prediction on the test set (denoted as
StructNet-Pred in Table 1). Overall, the performance of
the auxiliary pose prediction task is similar to that of [18],
which also uses a CNN for 3D pose estimation, but infe-
rior to the poses obtained using the score function. We also
test the effect of the auxiliary task on training the network.
When removing the auxiliary task, i.e., λ = 0, the pose er-
ror increases (denoted as StructNet∗-Max in Table 1). This
demonstrates that the auxiliary task helps the network to
converge to a good local optimum.
To justify the design choice of our pose embedding sub-
network, we trained the whole network with different forms
of pose embeddings: raw 3D joint coordinates, 1-layer net-
work with ﬁxed random weights, 1-layer network, and 2-
layer network. The results are presented in Table 3. The
network using no embedding (raw joint coordinates) has
the highest error, while the 2-layer pose embedding has the
lowest error, which suggests that embedding the pose in a
suitable high-dimensional space is necessary.
Finally, to demonstrate robustness of our framework, we
trained a network for each action category in Human3.6m
(using the same network parameters), and evaluated on the
online hidden test set2. The results are presented in Table 2.
On average, the proposed framework achieves 8.8% lower
error than LinKDE [12].
6. Visualization of image-pose embedding
In this section we visualize the latent features learned in
the image-pose embedding. We ﬁrst look at the 2 feature
dimensions of the image embedding with the highest vari-
2The action “Direction” is not included due to video corruption.
Embedding (dim.)
All
raw pose (51)
166.86 (92.63)
1-layer, random weights (1024)
145.95 (91.08)
1-layer (1024)
142.04 (87.98)
2-layer (1024)
135.63 (86.60)
Table 3. Comparison of different methods for pose embeddings.
ance over all the training images. Figure 4a plots the values
of these 2 features for each of the training images. To visu-
alize the meaning of the features, in each local region, we
show the average of the input images3 corresponding to the
feature points in that region. Figure 4b shows a similar plot
for the same 2 feature dimensions in the pose embedding,
with average poses over local regions of the space. The top-
2 features in the embedding correspond to the orientation of
the person. For example, in Figure 4a, the average image
in the upper-part of the plot is a frontal view of the person,
while the average image in the lower-part is the back view
(similarly for the average poses in Figure 4b).
Next, we look how the linear combination of embedding
features encodes the abstract attributes of the person. We
apply PCA on the image embedding vectors of all images in
the training set, and then project the image embeddings onto
two principal components. Figure 4c plots the two PCA co-
efﬁcients using the same local region visualization as Fig-
ure 4a. Figure 4d shows the corresponding plot for the pose
embedding. The ﬁrst PCA component (x-axis in Figs. 4c
and 4d) encodes the orientation (viewpoint) of the person,
while the second PCA component (y-axis) encodes the at-
tributes of the legs. For example, when the y-value is large
the left leg is closer to the camera, while when the y-value
is small, the right leg is closer to the camera.
Finally, these visualizations along with the supplemen-
tal video show that the learned embedding is smooth, even
though the temporal order of frames are not used. We be-
lieve this is because the score function is learned using a
max-margin constraint, which induces a topology of the
embedding. Speciﬁcally, since the margin is based on the
MPJPE between two poses, then the embedding vectors of
any two poses should be at least as far apart (according to
inner-product) as their MPJPE. In addition, the image and
pose embeddings are properly aligned; 97% of the max-
score poses for the training images are within 30 MPJPE of
the ground-truth pose.
7. Conclusion
In this paper, we propose a maximum-margin structured
learning framework with deep neural network for human
pose estimation. Our framework takes image and pose as
inputs and outputs a score value that represents a multi-
view similarity between the two inputs (whether they de-
pict the same pose). The network consists of a CNN for
image feature extraction, and two separate sub-networks
3For better visualization, we only use the images from a single subject.
2854

Action
Walking
Discussion
Eating
Taking Photo
Walking Dog
Greeting
All
LinKDE(BS) [12]
97.07 (37.14)
183.09 (116.74)
132.50 (72.53)
206.45 (112.61)
177.84 (122.65)
162.27 (88.43)
162.25 (104.43)
DconvMP-HML [18]
77.60 (23.54)
148.79 (100.49)
104.01 (39.20)
189.08 (93.99)
146.59 (75.38)
127.17 (51.10)
133.54 (81.31)
StructNet-Max
83.64 (27.44)
149.09 (108.93)
109.93 (51.28)
179.92 (93.50)
147.24 (85.62)
136.90 (64.71)
135.63 (86.60)
StructNet-Avg(20)
75.01 (25.60)
140.90 (110.07)
104.10 (51.39)
173.26 (93.71)
139.47 (86.67)
129.08 (65.11)
128.11 (87.18)
StructNet-Avg(500)
69.75 (21.42)
134.37 (110.04)
98.19 (49.49)
164.28 (90.60)
132.53 (85.91)
122.44 (61.83)
121.46 (85.65)
StructNet-Avg(500)-APF
68.51 (22.21)
134.13 (112.87)
97.37 (51.12)
166.15 (92.95)
132.51 (87.37)
122.33 (64.56)
121.31 (87.95)
StructNet-Avg(1500)
71.46 (19.75)
137.18 (110.91)
98.01 (47.20)
166.62 (88.89)
132.26 (83.34)
124.58 (60.64)
123.04 (85.17)
StructNet-Avg(1500)-APF
69.97 (20.66)
136.88 (113.93)
96.94 (49.03)
168.68 (91.55)
132.17 (85.37)
124.74 (63.92)
122.85 (87.77)
StructNet-Pred
84.85 (24.17)
148.82 (102.63)
121.57 (50.47)
179.39 (83.72)
151.92 (76.26)
133.79 (56.16)
133.79 (56.16)
StructNet∗-Max
87.15 (32.01)
161.62 (121.27)
119.50 (73.04)
196.24 (106.39)
154.91 (99.30)
145.30 (76.80)
145.74 (99.69)
Table 1. Results on Human3.6m: the MPJPE on the test set is calculated in millimeters (mm), with standard deviation parentheses.
Action
Discussion
Eating
Greeting
Phoning
Posing
Purchase
Sitting
SittingDown
Smoking
TakingPhoto
Waiting
Walking
WalkingDog
WalkingTogether
Avg
LinKDE(BS) [12]
108
91
129
104
130
134
135
200
117
195
132
115
162
156
133.81
DconvMP-HML [18]
103.11
91.68
108.38
109.49
116.45
145.24
145.14
329.96
110.35
174.97
112.43
99.16
153.29
116.44
136.47
StructNet-Avg(500)
92.97
76.70
98.16
92.70
106.86
140.94
135.46
260.75
98.03
170.83
105.11
99.40
138.53
109.30
122.03
StructNet-Avg(500)-APF
92.74
76.38
98.45
92.73
107.22
141.21
136.32
265.39
97.95
171.71
105.16
99.44
139.21
110.28
122.62
Table 2. Experimental results on the online (hidden) test set of Human3.6m.
(a)
0.2
0.4
0.6
0.8
1.0
1.2
0.0
0.2
0.4
0.6
0.8
1.0
(b)
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.0
0.1
0.2
0.3
0.4
(c)
−1.5
−1.0
−0.5
0.0
0.5
1.0
1.5
−1.5
−1.0
−0.5
0.0
0.5
(d)
−0.5
0.0
0.5
−0.2
0.0
0.2
0.4
0.6
0.8
Figure 4. Visualizations of the learned image-pose embedding: (a) visualization of the two highest-variance features in the learned image
embedding, and (b) the corresponding features in the pose embedding. (c) visualization of the two PCA coefﬁcients of the learned image
embedding and (d) pose embedding. In the pose plots, red/orange correspond to the right arm/leg, purple/green to the left arm/leg, and the
cyan “nose” points in the forward direction of the person.
for non-linear transformation of the image and pose into a
joint embedding, where the dot-product between the em-
beddings serves as the score function. We train the network
using a maximum-margin cost function, which enforces a
re-scaling margin between the score values of the ground-
truth image-pose pair and other image-pose pairs. This spe-
ciﬁc form of embedding and score function makes inference
computationally efﬁcient, by allowing the pose embedding
for a candidate set of poses to be calculated off-line. We
evaluate our proposed framework on Human3.6M dataset
and achieve signiﬁcant improvement over the state-of-art.
Finally, we show that the learned image-pose embedding
encodes semantic attributes of the pose, such as the orienta-
tion of the person and the position of the legs. Our proposed
framework is general, and future work will consider apply-
ing it to other structured-output tasks.
Acknowledgement. This work was supported by the Re-
search Grants Council of the Hong Kong Special Admin-
istrative Region, China (CityU 123212), and by a Strate-
gic Research Grant from City University of Hong Kong
(Project No. 7004417). We gratefully acknowledge the sup-
port of NVIDIA Corporation with the donation of the Tesla
K40 GPU used for this research.
2855

References
[1] G. Andrew, R. Arora, J. Bilmes, and K. Livescu.
Deep
canonical correlation analysis. In ICML, volume 28, pages
1247–1255, May 2013. 3
[2] F. Bastien, P. Lamblin, R. Pascanu, J. Bergstra, I. J. Good-
fellow, A. Bergeron, N. Bouchard, and Y. Bengio. Theano:
new features and speed improvements. In NIPS:Deep Learn-
ing and Unsupervised Feature Learning Workshop, 2012. 6
[3] Y. Bengio, G. Mesnil, Y. Dauphin, and S. Rifai. Better mix-
ing via deep representations. In ICML, pages 552–560, 2013.
2
[4] M. Burenius, J. Sullivan, and S. Carlsson. 3d pictorial struc-
tures for multiple view articulated pose estimation. In CVPR,
pages 3618–3625, 2013. 2
[5] X. Chen and A. Yuille.
Articulated pose estimation by a
graphical model with image dependent pairwise relations. In
NIPS, 2014. 2
[6] J. Deutscher and I. Reid. Articulated body motion capture
by stochastic search. IJCV, 61(2):185–205, 2005. 6
[7] N. Dhungel, G. Carneiro, and A. P. Bradley. Deep structured
learning for mass segmentation from mammograms. CoRR,
abs/1410.7454, 2014. 2
[8] M. Eichner and V. Ferrari.
Better appearance models for
pictorial structures. In BMVC, pages 1–11, 2009. 1, 2
[9] Felzenszwalb, P.F., Huttenlocher, and D.P. Pictorial struc-
tures for object recognition. IJCV, pages 55–79, 2005. 1
[10] C. Ionescu, L. Bo, and C. Sminchisescu. Structural SVM for
visual localization and continuous state estimation. In ICCV,
pages 1157–1164, 2009. 1
[11] C. Ionescu, F. Li, and C. Sminchisescu. Latent structured
models for human pose estimation. In ICCV, pages 2220–
2227, 2011. 1
[12] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Hu-
man3.6m: Large scale datasets and predictive methods for
3d human sensing in natural environments. IEEE TPAMI,
2014. 1, 5, 6, 7, 8
[13] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman.
Deep structured output learning for unconstrained text recog-
nition. ICLR, 2015. 2, 3
[14] A. Jain, J. Tompson, M. Andriluka, G. W. Taylor, and C. Bre-
gler. Learning human pose estimation features with convo-
lutional networks. In ICLR, April 2014. 1, 2
[15] T