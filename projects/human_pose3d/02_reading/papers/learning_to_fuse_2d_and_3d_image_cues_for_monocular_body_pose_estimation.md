# Learning to Fuse 2D and 3D Image Cues for Monocular Body Pose Estimation

> 2017 · id: W2758778552 · arXiv: 1611.05708 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Most recent approaches to monocular 3D human pose
estimation rely on Deep Learning. They typically involve
regressing from an image to either 3D joint coordinates di-
rectly or 2D joint locations from which 3D coordinates are
inferred. Both approaches have their strengths and weak-
nesses and we therefore propose a novel architecture de-
signed to deliver the best of both worlds by performing both
simultaneously and fusing the information along the way.
At the heart of our framework is a trainable fusion scheme
that learns how to fuse the information optimally instead of
being hand-designed. This yields signiﬁcant improvements
upon the state-of-the-art on standard 3D human pose esti-
mation benchmarks.

## introduction
Monocular 3D human pose estimation is a longstand-
ing problem of Computer Vision.
Over the years, two
main classes of approaches have been proposed:
Dis-
criminative ones that directly regress 3D pose from im-
age data [1, 8, 34, 46, 56, 67] and generative ones that
search the pose space for a plausible body conﬁguration
that aligns with the image data [21, 60, 68]. With the ad-
vent of ever larger datasets [30], models have evolved to-
wards deep architectures, but the story remains largely un-
changed. The state-of-the-art approaches can be roughly
grouped into those that directly regress 3D pose from im-
ages [30, 38, 64, 65] and those that ﬁrst predict a 2D pose
in the form of joint location conﬁdence maps and ﬁt a 3D
model to this 2D prediction [9, 76].
Since detecting the 2D image location of joints in eas-
ier than directly inferring the 3D pose, it can be done more
reliably. However, inferring a 3D pose from these 2D loca-
tions is fraught with ambiguities and the above-mentioned
methods usually rely on a database of 3D models to resolve
them, at the cost of a potentially expensive run-time ﬁtting
procedure. By contrast, the methods that regress directly to
3D avoid this extra step but also do not beneﬁt of the well-
posedness of the 2D joint detection location problem.
Con$idence Map stream
Image stream
FUSION
FCN
Figure 1: Overview of our approach. One stream of our
network accounts for the 2D joint locations and the corre-
sponding uncertainties. The second one leverages all 3D
image cues by directly acting on the image. The outputs
of these two streams are then fused to obtain the ﬁnal 3D
human pose estimate.
In this paper, we propose the novel architecture depicted
by Fig. 1 designed to deliver the best of both worlds. The
ﬁrst stream, which we will refer to as the Conﬁdence Map
Stream, ﬁrst computes a heatmap of 2D joint locations and
then infer the 3D poses from it. The second stream, which
we will dub the Image Stream, is designed to produce fea-
tures that complement those computed by the ﬁrst stream
and can be used in conjunction with them to compute the
3D pose, that is, guide the regression process given the 2D
locations.
However, for this approach to be beneﬁcial, effective fu-
sion of the two streams is crucial. In theory, it could hap-
pen at any stage of the two streams, ranging from early to
late fusion, with no principled way to choose one against
the other. We therefore also developed a trainable fusion
scheme that learns how to fuse the two streams.
Ultimately, our approach allows the network to still ex-
ploit image cues while inferring 3D poses from 2D joint
locations. As we demonstrate in our experiments, the fea-
tures computed by both streams are decorrelated and there-
fore truly encode complementary information. Our contri-
butions can be summarized as follows:
• We introduce a discriminative fusion framework to
1
arXiv:1611.05708v3  [cs.CV]  10 Apr 2017

simultaneously exploit 2D joint location conﬁdence
maps and 3D image cues for 3D human pose estima-
tion.
• We introduce a novel trainable fusion scheme, which
automatically learns where and how to fuse these two
sources of information.
We show that our approach signiﬁcantly outperforms the
state-of-the-art results on standard benchmarks and yields
accurate pose estimates from images acquired in uncon-
strained outdoors environments.

## method
Our goal is to increase the robustness and accuracy of
monocular 3D pose estimation by exploiting image cues to
the full while also taking advantage of the fact that 2D joint
locations can be reliably detected by modern CNN archi-
tectures. To this end, we designed the two stream archi-
tecture depicted by Fig. 1. The Conﬁdence Map Stream
shown at the top ﬁrst computes a heatmap of 2D joint loca-
tions from which feature maps can be computed. The Im-
age Stream shown at the bottom extracts additional features
directly from the image and all these features are fused to
produce a ﬁnal 3D pose vector.
As shown in Fig. 2, there is a whole range of ways to

perform the fusion of these two data streams, ranging from
early to late fusion with no obvious way to choose the best,
which might well be problem-dependent anyway. To solve
this conundrum, we rely on the fusion architecture depicted
by Fig. 3, which involves indroducing a third fusion stream
that combines the feature maps produced by the two data
streams in a trainable way. Each layer of the fusion stream
acts on a linear combination of the previous fusion layer
with the concatenation of the two data stream outputs. In
effect, different weight values for these linear combinations
correspond to different fusion strategies.
In the remainder of this section, we formalize this
generic architecture and study different ways to set these
weights, including learning them along with the weights of
the data streams, which is the approach we advocate.
3.1. Fusion Network
Let {Il}L
l=0 be the feature maps of the image stream and
{Xl}L
l=0 be the feature maps of the conﬁdence map stream.
As special cases, I0 : [1, 3]×[1, H]×[1, W] →[0, 1] is the
input RGB image, and X0 : [1, J] × [1, H] × [1, W] →R+
are the conﬁdence maps encoding the probability of observ-
ing each one of J body joints at any given image location.
The feature maps Il and Xl at each layer l must coincide
in width and height but can have different number of chan-
nels. In the following, we denote each feature map at level
l as both the output of layer l and the input to layer l + 1.
Let {Zl}L+1
l=0 be the feature maps of the fusion stream.
The feature map Zl is the output of layer l, but, unlike in the
data streams, the input to layer l + 1 is a linear combination
of Zl with Il and Xl given by
(1 −wl) · concat(Il, Xl) + wl · Zl,
1 ≤l ≤L,
(1)
where concat(·, ·) is the concatenation of the given feature
maps along the channel axis, and wl is the l-th element of
the fusion weights w ∈[0, 1]L controlling the mixture. For
this mixture to be possible, Zl must have the same size as
Il and Xl and a number of channels equal to the sum of
the number of channels of Il and Xl. As special cases,
Z0 = concat(I0, X0), and ZL+1 ∈R3J is the output of
the network, that is, the J predicted 3D joint locations.
In essence, the fusion weights w control where and how
the fusion of the data streams occurs. Different settings of
these weights lead to different fusion strategies. We illus-
trate this with two special cases below, and then introduce
an approach to automatically learn these weights together
with the other network parameters.
Early fusion.
If the fusion weights are all set to one,
w = 1, the two data streams are ignored, and only the fu-
sion one is considered to compute the output. Since the
fusion stream takes the concatenation of the image I0 and
the conﬁdence maps X0 as input, this is equivalent to the
early fusion architecture of Fig. 2(a).
Fusion at a speciﬁc layer.
Instead of fusing the streams
in the very ﬁrst layer, one might want to postpone the fu-
sion point to a later layer β ∈{0, · · · , L}. In our for-
malism, this can be achieved by setting the fusion weights
to wl = I[l > β], where I is the indicator function. For
example, when β = 4, our network becomes equivalent to
the one depicted by Fig. 2(b). The early and late fusion
architectures of Fig. 2(a, c) can also be represented in this
manner by setting β = 0 and β = L, respectively.
Ultimately, the complete fusion network encodes a func-
tion f(i, x; θ, w) = ZL+1|I0=i,X0=x mapping from an im-
age i and conﬁdence maps x to the 3D joint locations,
parametrized by layer weights θ and fusion weights w.
With manually-deﬁned fusion weights, given a set of
N training pairs (in, xn) with corresponding ground-truth
joint positions yn, the parameters θ can be learnt by mini-
mizing the square loss expressed as
L(θ) =
N
X
n=1
∥f(in, Xn; θ, w) −yn∥2
2 .
(2)
Trainable fusion.
Setting the weights manually, which in
our formalism boils down to choosing β, is not obvious;
the best value for β will typically depend on the network
architecture, the problem and the nature of the input data.
A straightforward approach would consist of training net-
works for all possible values of β to validate the best one,
but this quickly becomes impractical. To address this is-
sue, we introduce a trainable fusion approach, which aims
to learn β from data jointly with the network parameters.
To this end, however, we cannot directly use the indica-
tor function, which has zero derivatives almost everywhere,
thus making it inapplicable to gradient-based optimization.
Instead, we propose to approximate the indicator function
by a sigmoid function
wl =
1
1 + e−α·(l−β) ,
(3)
parameterized by α and β. As above, β determines the stage
at which fusion occurs and α controls how sharp the tran-
sition between weights with value 0 and with value 1 is.
When α →∞, the function in Eq. 3 becomes equivalent
to the indicator function1, while, when α = 0, the network
mixes the data and fusion streams in equal proportions at
every layer.
In practice, mixing the data and fusion streams at ev-
ery layer is not desirable. First, by contrast to having bi-
nary weights w, which deactivate some of the layers of each
stream, it corresponds to a model with a very large number
of active parameters, and thus prone to overﬁtting. Further-
more, after training, a model with binary weights can be
1Except at l = β.

w1
(1-w1)
(1-w2)
(1-w3)
(1-w4)
(1-w5)
(1-w6)
(1-w7)
w2
w3
w4
w5
w6
w7
conv
fc
concat
weighted
average
draw
weight
w1
(1-w1)
(1-w2)
(1-w3)
(1-w4)
(1-w5)
(1-w6)
(1-w7)
w2
w3
w4
w5
w6
w7
conv
fc
concat
weighted
average
draw
weight
weights
layers
Figure 3: Trainable fusion architecture. The ﬁrst two streams take as input the image and 2D joint location conﬁdence
maps, respectively. The combined feature maps of the image and conﬁdence map stream are fed into the fusion stream and
linearly combined with the outputs of the previous fusion layer. The linear combination of the streams is controlled by a
weight vector shown at the bottom part of the ﬁgure. The numbers below each layer represent the corresponding size of the
feature maps for convolutional layers and the number of neurons for fully connected ones.
pruned, by removing the inactive layers in each stream, that
is all layers l from the fusion stream where wl ≈0, and all
layers l from the data streams where wl ≈1. This yields a
more compact, and thus more efﬁcient network for test-time
prediction.
To account for this while learning where to fuse the in-
formation sources, we modify the loss function of Eq. (2)
by incorporating a term that penalizes small values of α and
favors sharp fusions. This yields a loss of the form
L(θ, α, β) =
N
X
n=1
∥f(in, Xn; θ, α, β) −yn∥2
2 + λ
α2 , (4)
with α and β as trainable parameters, in addition to θ, and
a hyperparameter λ weighing the penalty term. Altogether,
this loss lets us simultaneously ﬁnd the most suitable fusion
layer β for the given data and the corresponding network
parameters θ, while encouraging a sharp fusion function to
mimic the behavior of the indicator function.
In practice, we initialize α with a small value of 0.1 and
β to the middle layer of the complete network. We use the
ADAM [35] gradient update method with a learning rate
of 10−3 to guide the optimization. We set the regularization
parameter to 5 · 103, which renders the magnitude of both
the regularization term and the main cost comparable. We
use dropout and data augmentation to prevent overﬁtting.
3.2. 2D Joint Location Conﬁdence Map Prediction
Our approach depends on generating heatmaps of the 2D
joint locations that we can feed as input to the conﬁdence
map stream. To do so, we rely on a fully-convolutional net-
work with skip connections [43]. Given an RGB image as
input, it performs a series of convolutions and pooling op-
erations to reduce its spatial resolution, followed by upcon-
volutions to produce pixel-wise conﬁdence values for each
pixel. We employed the stacked hourglass network design
of [43], which carries out repeated bottom-up, top-down
processing to capture spatial relationships in the image. We
perform heatmap regression to assign high conﬁdence val-
ues to the most likely joint positions. In our experiments,
we ﬁne-tuned the hourglass network initially trained on the
MPII dataset [4] using the training data speciﬁc to each ex-
periment 

## experiments
In this section, we ﬁrst describe the datasets we tested
our approach on and the corresponding evaluation proto-

cols. We then compare our approach against the state-of-
the-art methods and provide a detailed analysis of our gen-
eral framework.
4.1. Datasets
We evaluate our approach on the Human3.6m [30],
HumanEva-I [61], KTH Multiview Football II [10] and
Leeds Sports Pose (LSP) [33] datasets described below.
Human3.6m is a large and diverse motion capture dataset
including 3.6 million images with their corresponding 2D
and 3D poses. The poses are viewed from 4 different cam-
era angles. The subjects carry out complex motions corre-
sponding to daily human activities. We use the standard 17
joint skeleton from Human3.6m as our pose representation.
HumanEva-I comprises synchronized images and motion
capture data and is a standard benchmark for 3D human
pose estimation. The output pose is a vector of 15 3D joint
coordinates.
KTH Multiview Football II provides a benchmark to eval-
uate the performance of pose estimation algorithms in un-
constrained outdoor settings. The camera follows a soccer
player moving around the pitch. The videos are captured
from 3 different camera viewpoints. The output pose is a
vector of 14 3D joint coordinates.
LSP is a standard benchmark for 2D human pose estima-
tion and does not contain any ground-truth 3D pose data.
The images are captured in unconstrained outdoor settings.
2D pose is represented in terms of a vector of 14 joint coor-
dinates. We report qualitative 3D pose estimation results on
this dataset.
4.2. Evaluation Protocol
On Human3.6m, we used the same data partition as in
earlier work [38, 39, 40, 65, 76] for a fair comparison. The
data from 5 subjects (S1, S5, S6, S7, S8) was used for train-
ing and the data from 2 different subjects (S9, S11) was
used for testing. We evaluate the accuracy of 3D human
pose estimation in terms of average Euclidean distance be-
tween the predicted and ground-truth 3D joint positions, as
in [38, 39, 40, 65, 76]. Training and testing were carried out
monocularly in all camera views.
In [9], [46]2 , and [58]3 the estimated skeleton was ﬁrst
aligned to the ground-truth one by Procrustes transforma-
tion before measuring the joint distances. This is therefore
what we also do when comparing against [9, 46, 58].
2While [46] also reports results without Procrustes analysis, the authors
conﬁrmed to us by email that their evaluation assumes the ground-truth
depth of the root joint to be known to go from their volumetric representa-
tion to 3D pose in metric space. Since this also sets the scale of the skele-
ton, we believe that a comparison using the full Procrustes transformation
for both their approach and ours is the right one to perform here.
3This it is not explicitly stated in [58], but the authors conﬁrmed this to
us by email.
On HumanEva-I, following the standard evaluation pro-
tocol [9, 62, 65, 72, 76], we trained our model on the train-
ing sequences of subjects S1, S2 and S3 and evaluated on
the validation sequences of all subjects. We pretrained our
network on Human3.6m and used only the ﬁrst camera view
for further training and validation.
On the KTH Multiview Football II dataset, we evalu-
ate our method on the sequence containing Player 2, as
in [7, 10, 46, 65]. Following [7, 10, 46, 65], the ﬁrst half
of the sequence from camera 1 is used for training and the
second half for testing. To compare our results to those
of [7, 10, 46, 65], we report accuracy using the percentage
of correctly estimated parts (PCP) score. Since the training
set is quite small, we propose to pretrain our network on
the recent synthetic dataset [12], which contains images of
sports players with their corresponding 3D poses. We then
ﬁne-tuned it using the training data from KTH Multiview
Football II. We report results with and without this pretrain-
ing.
4.3. Comparison to the State-of-the-Art
We ﬁrst compare our approach with state-of-the-art base-
lines on the Human3.6m [30], HumanEva [61] and KTH
Multiview Football [10] datasets.
Human3.6m.
In Table 1, we compare the results of our
trainable fusion approach with those of the following state-
of-the-art single image-based methods: KDE regression
from HOG features to 3D poses [30], jointly training a
2D body part detector and a 3D pose regressor [38, 45],
the maximum-margin structured learning framework of [39,
40], the deep structured prediction approach of [64], pose
regression with kinematic constraints [75], and 3D pose es-
timation with mocap guided data augmentation [53]. For
completeness, we also compare our approach to the follow-
ing methods that rely on either multiple consecutive images
or impose temporal consistency: regression from short im-
age sequences to 3D poses [65], ﬁtting a sparse 3D pose
model to 2D conﬁdence map predictions across frames [76],
and ﬁtting a 3D pose sequence to the 2D joints predicted by
images and height-maps that encode the height of each pixel
in the image with respect to a reference plane [17].
As can be seen from the results in Table 1, our approach
outperforms all the methods on all the action categories by a
large margin. In particular, we outperform the image-based
regression methods of [30, 38, 39, 40, 64, 45, 75], as well
as the model-ﬁtting strategy of [39, 40]. This, we believe,
clearly evidences the beneﬁts of fusing 2D joint location
conﬁdence maps with 3D image cues, as done by our ap-
proach. Furthermore, we also achieve lower error than the
method of [53], despite the fact that it relies on additional
training data. Even though our algorithm uses only indi-
vidual images, it also outperforms the methods that rely on
sequences [17, 65, 76].

Input

## related_work
The existing 3D human pose estimation approaches can
be roughly categorized into discriminative and generative
ones. In what follows, we review both types of approaches.
Discriminative methods aim at predicting 3D pose di-
rectly from the input data, may it be single images [28, 29,
37, 38, 39, 46, 52, 55, 64, 73], depth images [23, 50, 59],
or short image sequences [65]. Early approaches falling
into this category typically worked by extracting hand-
crafted features and learning a mapping from these fea-
tures to 3D poses [1, 8, 28, 29, 37, 56, 67]. Unsurpris-
ingly, the more recent methods tend to rely on Deep Net-
works [38, 64, 65, 75]. In particular, [38, 65] rely on 2D
poses to pretrain the network, thus exploiting the common-
alities between 2D and 3D pose estimation. In fact, [38]
even proposes to jointly predict 2D and 3D poses. However,
in such approaches, the two predictions are not coupled. By
contrast, [45] introduces a network that uses 2D information
for 3D pose estimation. This method, however, does not ex-
ploit pixelwise joint location uncertainty, and only makes
use of the 2D evidence late in the pose estimation process.
While these methods exploit the available 3D image cues,
they fail to explicitly model 2D joint location uncertainty,
which matters when addressing a problem as ambiguous as
monocular 3D pose estimation.
Since pose estimation is much better-posed in 2D than
in 3D, a popular way to infer joint positions is to use
a generative model to ﬁnd a 3D pose whose projection
aligns with the 2D image data.
In the past, this usu-
ally involved inferring a 3D human pose by optimizing
an energy function derived from image information, such
as silhouettes [6, 14, 21, 22, 25, 31, 44, 49, 60], tra-
jectories [74], feature descriptors [58, 62, 63] and 2D
joint locations [2, 3, 5, 20, 36, 51, 57, 68, 69]. Another
class of approaches retrieve the pose from a dictionary
of 3D poses based on similarity with the 2D image evi-
dence [18, 26, 39, 41, 42]. With the growing availability of
large datasets and the advent of Deep Learning, the empha-
sis has shifted towards using discriminative 2D pose regres-
sors [11, 13, 15, 16, 24, 27, 32, 43, 47, 48, 66, 70, 71] to ex-
tract the 2D pose and infer a 3D one from it [9, 19, 72, 76].
(a) Early fusion
(b) Fusion at a speciﬁc layer
(c) Late fusion
Figure 2: Three different instances of hard-coded fusion.
The fusion strategies combine 2D joint location conﬁdence
maps with 3D cues directly extracted from the input image.
The 2D joint locations are represented by heatmaps that en-
code the conﬁdence of observing a particular joint at any
given image location. A human body representation, such
as a skeleton [76], or a more detailed model [9] can then be
ﬁtted to these predictions. While this takes 2D joint posi-
tions into account, it ignores image information during the
ﬁtting process. It therefore discards potentially important
3D cues that could help resolve ambiguities.

## conclusion
In this paper, we have proposed to fuse 2D and 3D image
cues for monocular 3D human pose estimation. To this end,
we have introduced an approach that relies on two CNN
streams to jointly infer 3D pose from 2D joint locations and
from the image directly. We have also introduced an ap-
proach to fusing the two streams in a trainable way.
We have demonstrated that the resulting CNN pipeline
signiﬁcantly outperforms state-of-the-art methods on stan-
dard 3D human pose estimation benchmarks. Our frame-
work is general and can easily be extended to incorporate
CM stream 
channels
Image stream 
 channels
Image stream 
channels
CM stream 
 channels
Figure 7: Squared Pearson correlation coefﬁcients (R2) be-
tween each pair of the features learned at the last convolu-
tional layer of our trainable fusion network computed from
128 randomly selected images in Human3.6m. As can be
seen in the lower left and upper right submatrices, the fea-
ture maps of the image and the conﬁdence map streams are
decorrelated.
other modalities, such as optical ﬂow or body part segmen-
tation. Furthermore, our trainable fusion strategy could be
applied to other fusion problems, which is what we intend
to do in future work.

Figure 8: Pose estimation results on LSP. We trained our network on the recently released synthetic dataset of [12] and tested
it on the LSP dataset. The quality of the 3D pose predictions demonstrates the generalization of our method. Best viewed in
color.
A. Appendix
In this appendix, we analyze the inﬂuence of our regular-
ization term encouraging sharp fusion in Eq. 4, provide run-
ning time for our algorithm, and show additional qualitative
results on the Leeds Sports Pose [33], HumanEva-I [61],
Human3.6m [30] and KTH Multiview Football II [10]
datasets.
Effect of the regularization.
Below, we analyze the ef-
fect of the regularization term that encourages sharp fusion
in Eq. 4. In the absence of the regularization term, the net-
work mixes the data and fusion streams without necessarily
fusing them at a speciﬁc layer. As discussed in the main
paper, this corresponds to a model with many active param-
eters. Therefore it is prone to overﬁtting and computation-
ally less efﬁcient at test-time. In Table 6, we compare the
results of our approach with and without this regularization
term. For the latter, we do not parametrize the weights of
the network with a sigmoid function and do not constrain
the network to have a sharp fusion. The results conﬁrm that
encouraging sharp fusion yields both better accuracy and
faster prediction.
Running time.
We carried out our experiments on a ma-
chine equipped with an Intel Xeon CPU E5-2680 and an
NVIDIA TITAN X Pascal GPU. It takes 90 ms to compute
2D joint location conﬁdence maps and 6 ms to predict 3D
pose with our fusion network. Therefore, the total runtime
of our method is 0.096 sec/frame (over 10 fps), which com-