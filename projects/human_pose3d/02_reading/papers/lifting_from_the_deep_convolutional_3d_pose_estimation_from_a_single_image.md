# Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image

> 2017 · id: W2583585015 · arXiv: 1701.00295 · pdf: https://arxiv.org/pdf/1701.00295 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We propose a uniﬁed formulation for the problem of
3D human pose estimation from a single raw RGB image
that reasons jointly about 2D joint estimation and 3D pose
reconstruction to improve both tasks.
We take an inte-
grated approach that fuses probabilistic knowledge of 3D
human pose with a multi-stage CNN architecture and uses
the knowledge of plausible 3D landmark locations to reﬁne
the search for better 2D locations. The entire process is
trained end-to-end, is extremely efﬁcient and obtains state-
of-the-art results on Human3.6M outperforming previous
approaches both on 2D and 3D errors.

## introduction
Estimating the full 3D pose of a human from a single
RGB image is one of the most challenging problems in
computer vision. It involves tackling two inherently am-
biguous tasks. First, the 2D location of the human joints, or
landmarks, must be found in the image, a problem plagued
with ambiguities due to the large variations in visual ap-
pearance caused by different camera viewpoints, external
and self occlusions or changes in clothing, body shape or
illumination. Next, lifting the coordinates of the 2D land-
marks into 3D from a single image is still an ill-posed prob-
lem – the space of possible 3D poses consistent with the
2D landmark locations of a human, is inﬁnite. Finding the
correct 3D pose that matches the image requires injecting
additional information usually in the form of 3D geometric
pose priors and temporal or structural constraints.
We propose a new joint approach to 2D landmark de-
tection and full 3D pose estimation from a single RGB im-
age that takes advantage of reasoning jointly about the es-
timation of 2D and 3D landmark locations to improve both
tasks. We propose a novel CNN architecture that learns to
combine the image appearance based predictions provided
by convolutional-pose-machine style 2D landmark detec-
tors [44], with the geometric 3D skeletal information en-
coded in a novel pretrained model of 3D human pose.
Information captured by the 3D human pose model is
embedded in the CNN architecture as an additional layer
that lifts 2D landmark coordinates into 3D while impos-
ing that they lie on the space of physically plausible poses.
The advantage of integrating the output proposed by the 2D
landmark location predictors – based purely on image ap-
pearance – with the 3D pose predicted by a probabilistic
model, is that the 2D landmark location estimates are im-
proved by guaranteeing that they satisfy the anatomical 3D
constraints encapsulated in the human 3D pose model. In
this way, both tasks clearly beneﬁt from each other.
A further advantage of our approach is that the 2D and
3D training data sources may be completely independent.
The deep architecture only needs that images are annotated
with 2D poses, not 3D poses. The human pose model is
trained independently and exclusively from 3D mocap data.
This decoupling between 2D and 3D training data presents
a huge advantage since we can augment the training sets
completely independently. For instance we can take advan-
tage of extra 2D pose annotations without the need for 3D
ground truth or extend the 3D training data to further mocap
datasets without the need for synchronized 2D images.
Our contribution: In this work, we show how to integrate
a prelearned 3D human pose model directly within a novel
CNN architecture (illustrated in ﬁgure 1) for joint 2D land-
mark and 3D human pose estimation. In contrast to pre-
existing methods, we do not take a pipeline approach that
takes 2D landmarks as given. Instead, we show how such
a model can be used as part of the CNN architecture itself,
and how the architecture can learn to use physically plausi-
ble 3D reconstructions in its search for better 2D landmark
locations. Our method achieves state-of-the-art results on
the Human3.6M dataset both in terms of 2D and 3D errors.

## related_work
We ﬁrst describe methods that assume that 2D joint lo-
cations are provided as input and focus on solving the 3D
arXiv:1701.00295v4  [cs.CV]  11 Oct 2017

STAGE 1
2D joint prediction
3D lifting &
projection
Fusion
Feature extraction 
2D Loss
STAGE 2
2D joint prediction
3D lifting &
projection
Fusion
Feature extraction
2D Loss
STAGE 6
2D joint prediction
3D lifting &
projection
Fusion
Feature extraction
2D Loss
Probabilistic 3D
pose model
3D pose
3D/2D
projection
predicted
belief maps
projected pose
belief maps
9
9
9
9
9
9
9
9
1 1
1 1
predicted
belief maps
predicted
belief maps
projected pose
belief maps
2D
fusion
fused
belief maps
Input image
Probabilistic 3D
pose model
Output 2D pose
Final 3D pose
Figure 1: The multistage deep architecture for 2D/3D human pose estimation. Each stage produces as output a set of belief maps for the
location of the 2D landmarks (one per landmark). The belief maps from each stage, as well as the image, are used as input to the next
stage. Internally, each stage learns to combine: (a) belief maps provided by convolutional 2D joint predictors, with (b) projected pose
belief maps, proposed by the probabilistic 3D pose model. The 3D pose layer is responsible for lifting 2D landmark coordinates into 3D
and projecting them onto the space of valid 3D poses. These two belief maps are then fused into a single set of output proposals for the 2D
landmark locations per stage. The accuracy of the 2D and 3D landmark locations increases progressively through the stages. The loss used
at each stage requires only 2D pose annotations, not 3D. The overall architecture is fully differentiable – including the new projected-pose
belief maps and 2D-fusion layers – and can be trained end-to-end using back-propagation. [Best viewed in color.]
lifting problem and follow with methods that learn to esti-
mate the 3D pose directly from images.
3D pose from known 2D joint positions: A large body
of work has focused on recovering the 3D pose of people
given perfect 2D joint positions as input. Early approaches
[19, 34, 25, 6] took advantage of anatomical knowledge of
the human skeleton or joint angle limits to recover pose
from a single image. More recent methods [13, 28, 3] have
focused on learning a prior statistical model of the human
body directly from 3D mocap data.
Non-rigid structure from motion approaches (NRSfM)
also recover 3D articulated motion [8, 4, 14, 20] given
known 2D correspondences for the joints in every frame
of a monocular video. Their huge advantage, as unsuper-
vised methods, is they do not need 3D training data, instead
they can learn a linear basis for the 3D poses purely from
2D data. Their main drawback is their need for signiﬁcant
camera movement throughout the sequence to guarantee ac-
curate 3D reconstruction. Recent work on NRSfM applied
to human pose estimation has focused on escaping these
limitations by the use of a linear model to represent shape
variations of the human body. For instance, [10] deﬁned
a generative model based on the assumption that complex
shape variations can be decomposed into a mixture of prim-
itive shape variations and achieve competitive results.
Representing human 3D pose as a linear combination of
a sparse set of 3D bases, pretrained using 3D mocap data,
has also proved a popular approach for articulated human
motion [28, 43, 49], while [49] propose a convex relaxation
to jointly estimate the coefﬁcients of the sparse representa-
tion and the camera viewpoint [28] and [43] enforce limb
length constraints. Although these approaches can recon-
struct 3D pose from a single image, their best results come
from imposing temporal smoothness on the reconstructions
of a video sequence.
Recently, Zhao et al. [47] achieved state-of-the-art re-
sults by training a simple neural network to recover 3D pose
from known 2D joint positions. Although the results on
perfect 2D input data are impressive, the inaccuracies in 2D
joint estimation are not modeled and the performance of this
approach combined with joint detectors is unknown.
3D pose from images: Most approaches to 3D pose infer-
ence directly from images fall into one of two categories: (i)
models that learn to regress the 3D pose directly from image
features and (ii) pipeline approaches where the 2D pose is
ﬁrst estimated, typically using discriminatively trained part
models or joint predictors, and then lifted into 3D. While
regression based methods suffer from the need to annotate
all images with ground truth 3D poses – a technically com-
plex and elaborate process – for pipeline approaches the
challenge is how to account for uncertainty in the measure-
ments. Crucial to both types of approaches is the question
of how to incorporate the 3D dependencies between the dif-
ferent body joints or to leverage other useful 3D geometric
information in the inference process.
Many earlier works on human pose estimation from a
single image relied on discriminatively trained models to
learn a direct mapping from image features such as silhou-
ettes, HOG or SIFT, to 3D human poses without passing
through 2D landmark estimation [1, 12, 11, 24, 32].
Recent direct approaches make use of deep learning [21,
22, 40, 41]. Regression-based approaches train an end-to-
end network to predict 3D joint locations directly from the
image [41, 21, 22, 48]. Li et al. [22] incorporate model joint
dependencies in the CNN via a max-margin formalism, oth-
ers [48] impose kinematic constraints by embedding a dif-

ferentiable kinematic model into the deep learning architec-
ture. Tekin et al. [35] propose a deep regression architecture
for structured prediction that combines traditional CNNs for
supervised learning with an auto-encoder that implicitly en-
codes 3D dependencies between body parts.
As CNNs have become more prevalent, 2D joint estima-
tion [44] has become increasingly reliable and many recent
works have looked to exploit this using a pipeline approach.
Papers such as [9, 16, 40, 26] ﬁrst estimate 2D landmarks
and later 3D spatial relationships are imposed between them
using structured learning or graphical models.
Simo-Serra et al. [33] were one of the ﬁrst to propose
an approach that naturally copes with the noisy detections
inherent to off-the-shelf body part detectors by modeling
their uncertainty and propagating it through 3D shape space
while satisfying geometric and kinematic 3D constraints.
The work [31] also estimates the location of 2D joints be-
fore predicting 3D pose using appearance and the probable
3D pose of discovered parts using a non-parametric model.
Another recent example is Bogo et al. [7], who ﬁt a detailed
statistical 3D body model [23] to 2D joint proposals.
Zhou et al. [50] tackles the problem of 3D pose estima-
tion for a monocular image sequence integrating 2D, 3D
and temporal information to account for uncertainties in the
model and the measurements. Similar to our proposed ap-
proach, Zhou et al.’s method [50] does not need synchro-
nized 2D-3D training data, i.e. it only needs 2D pose an-
notations to train the CNN joint regressor and a separate
3D mocap dataset to learn the 3D sparse basis. Unlike our
approach, it relies on temporal smoothness for its best per-
formance, and performs poorly on a single image.
Finally, Wu et al. [45]’s 3D Interpreter Network, a recent
approach to estimate the skeletal structure of common ob-
jects (chairs, sofas, ...) bears similarities with our method.
Although our approaches share common ground in the de-
coupling of 3D and 2D training data and the use of projec-
tion from 3D to improve 2D predictions the network archi-
tectures are very different and, unlike us, they do not carry
out a quantitative evaluation on 3D human pose estimation.
3. Network Architecture
Figure 1 illustrates the main contribution of our ap-
proach, a new multi-stage CNN architecture that can be
trained end-to-end to estimate jointly 2D and 3D joint lo-
cations. Crucially it includes a novel layer, based on a prob-
abilistic 3D model of human pose, responsible for lifting
2D poses into 3D and propagating 3D information about
the skeletal structure to the 2D convolutional layers. In this
way, the prediction of 2D pose beneﬁts from the 3D infor-
mation encoded. Section 4 describes the new probabilistic
3D model of human pose, trained on a dataset of 3D mo-
cap data. Section 5 describes all the new components and
layers of the CNN architecture. Finally, Section 6 describes
experimental evaluation on the Human3.6M dataset where
we obtain state-of-the-art results. In addition we show qual-
itative results on images from the MPII and Leeds datasets.
4. Probabilistic 3D Model of Human Pose
One fundamental challenge in creating models of human
poses lies in the lack of access to 3D data of sufﬁcient va-
riety to characterize the space of human poses. To com-
pensate for this lack of data we identify and eliminate con-
founding factors such as rotation in the ground plane, limb
length, and left-right symmetry that l

## conclusion
Eating
Greeting
Phoning
Photo
Posing
Purchases
LinKDE [15]
132.71
183.55
132.37
164.39
162.12
205.94
150.61
171.31
Li et al. [22]
-
136.88
96.94
124.74
-
168.68
-
-
Tekin et al. [37]
102.39
158.52
87.95
126.83
118.37
185.02
114.69
107.61
Tekin et al. [35]
-
129.06
91.43
121.68
-
162.17
-
-
Tekin et al. [36]
85.03
108.79
84.38
98.94
119.39
95.65
98.49
93.77
Zhou et al. [50]
87.36
109.31
87.05
103.16
116.18
143.32
106.88
99.78
Sanzari et al. [31]
48.82
56.31
95.98
84.78
96.47
105.58
66.30
107.41
Ours - Single PPCA Model
68.55
78.27
77.22
89.05
91.63
110.05
74.92
83.71
Ours - Mixture PPCA Model
64.98
73.47
76.82
86.43
86.28
110.67
68.93
74.79
Sitting
Sitting Down Smoking
Waiting
Walk Dog Walking Walk Together
Average
LinKDE [15]
151.57
243.03
162.14
170.69
177.13
96.60
127.88
162.14
Li et al. [22]
-
-
-
-
132.17
69.97
-
-
Tekin et al. [37]
136.15
205.65
118.21
146.66
128.11
65.86
77.21
125.28
Tekin et al. [35]
-
-
-
-
130.53
65.75
-
-
Tekin et al. [36]
73.76
170.4
85.08
116.91
113.72
62.08
94.83
100.08
Zhou et al. [50]
124.52
199.23
107.42
118.09
114.23
79.39
97.70
113.01
Sanzari et al. [31]
116.89
129.63
97.84
65.94
130.46
92.58
102.21
93.15
Ours - Single PPCA Model
115.94
185.72
88.25
88.73
92.37
76.48
77.95
92.96
Ours - Mixture PPCA Model
110.19
173.91
84.95
85.78
86.26
71.36
73.14
88.39
Table 1: A comparison of the 3D pose estimation results of our approach on the Human3.6M dataset against competitors that follow
Protocol #1 for evaluation (3D errors are given in mm). We substantially outperform all other methods in terms of average error showing a
4.7mm average improvement over our closest competitor. Note that some approaches [37, 50] use video as input instead of a single frame.
which is then embedded in a belief map as
ˆbp
i,j =
(
1
if(i, j) = ˆYp
0
otherwise.
(6)
and then convolved using Gaussian ﬁlters.
5.5. 2D Fusion of belief maps
The 2D belief maps predicted by the probabilistic 3D
pose model are fused with the CNN-based belief maps bp
according to the following equation
f p
t = wt ∗bp
t + (1 −wt) ∗ˆbp
t
(7)
where wt ∈[0, 1] is a weight trained as part of the end-to-
end learning. This set of fused belief maps ft is then passed
to the next stage and used as an input to guide the 2D re-
estimation of joint locations, instead of the belief maps bt
used by convolutional pose machines.
5.6. The Objective and Training
Following [44], the objective or cost function ct min-
imized at each stage is the the squared distance between
the generated fusion maps of the layer f p
t , and ground-truth
belief maps bp
∗generated by Gaussian blurring the sparse
ground-truth locations of each landmark p
ct =
L+1
X
p=1
X
z∈Z
||f p
t −bp
∗||2
2
(8)
For end-to-end training the total loss is the sum over all
layers P
t≤6 ct.
The novel layers were implemented as
an extension of the published code of Convolutional Pose
Machines [44] inside the Caffe framework [17] as Python
layers, with weights updated using Stochastic Gradient De-
scent with momentum. Details of the novel gradient updates
used lifting estimates through 3d pose space are given in the
supplementary materials.
6. Experimental evaluation
Human3.6M dataset:
The model was trained and tested
on the Human3.6M dataset consisting of 3.6 million ac-
curate 3D human poses [15]. This is a video and mocap
dataset of 5 female and 6 male subjects, captured from 4 dif-
ferent viewpoints, that show them performing typical activ-
ities (talking on the phone, walking, greeting, eating, etc.).
2D Evaluation: Figure 5 shows how the 2D predictions are
improved by the projected pose model, reducing the over-
all mean error per landmark. The 2D error reduction using
our full approach over the estimates of [44] is comparable
in magnitude to the improvement due to the change of ar-
chitecture moving from the work Zhou et al. [50] to the
state-of-the-art 2d architecture [44] (i.e. a reduction of 0.59
pixels vs. 0.81 pixels). See Table 2 for details.
3D Evaluation: Several evaluation protocols have been
followed by different authors to measure the performance
of their 3D pose estimation methods on the Human3.6M
dataset. Tables 1 and 2 show comparisons of the 3D pose

Evaluation of 3D error (mm)
Protocol #2
Yasin et al. [46]
108.3
Rogez et al. [30]
88.1
Ours - Mixture PPCA Model
70.7
Evaluation of 3D error (mm)
Protocol #3
Bogo et al. [7]
82.3
Ours - Mixture PPCA Model
79.6
Evaluation of 2D pixel error
Zhou et al. [50]
10.85
Trained CPM [44] architecture
10.04
Ours using 3D reﬁnement
9.47
Table 2: Further evaluation on the Human3.6M dataset. Top two
tables compare our 3D pose estimation errors against competitors
on Protocols #2 or #3. Bottom table compares our 2D pose esti-
mation error against competitors. Our approach, which lifts the 2D
landmark predictions into a plausible 3D model and then projects
them back into the image, substantially reduces the error. Note
that [50] use video as input and knowledge of the action label.
estimation with previous works, where we take care to eval-
uate using the appropriate protocol.
Protocol #1, the most standard evaluation protocol on
Human3.6M, was followed by [15, 22, 37, 35, 36, 50, 31].
The training set consists of 5 subjects (S1, S5, S6, S7, S8),
while the test set includes 2 subjects (S9, S11). The orig-
inal frame rate of 50 FPS is down-sampled to 10 FPS and
the evaluation is on sequences coming from all 4 cameras
and all trials. The reported error metric is the 3D error i.e.
the Euclidean distance from the estimated 3D joints to the
ground truth, averaged over all 17 joints of the Human3.6M
skeletal model. Table 1 shows a comparison between our
approach and competing approaches using Protocol #1. Our
baseline method using a single unimodal probabilistic PCA
model outperforms almost every method in most action
types, with the exception of Sanzari et al. [31], which it
still outperforms on average across the entire dataset. The
mixture model improves on this again, offering a 4.76mm
improvement over Sanzari et al., our closest competitor.
Protocol #2, followed by [46, 30], selects 6 subjects (S1,
S5, S6, S7, S8 and S9) for training and subject S11 for
testing. The original video is down-sampled to every 64th
frame and evaluation is performed on sequences from all 4
cameras and all trials. The error metric reported in this case
is the 3D pose error equivalent to the per-joint 3D error up
to a similarity transformation (i.e. each estimated 3D pose
is aligned with the ground truth pose, on a per-frame basis,
using Procrustes analysis). The error is averaged over 14
joints. Table 2 shows a comparison between our approach
and other approaches that use Protocol #2. Although, our
model was trained using only the 5 subjects used for train-
ing in Protocol #1 (one fewer subject), it still outperforms
the other methods [30, 46].
Protocol #3, followed by [7], selects the same subjects
for training and testing as Protocol #1. However, evalua-
tion is only on sequences captured from the frontal camera
(“cam 3”) from trial 1 and the original video is not sub-
sampled. The error metric used in this case is the 3D pose
error as described in Protocol #2. The error is averaged
over a subset of 14 joints. Table 2 shows a comparison
between our approach and [7]. Our method outperforms
Bogo et al. [7] by almost 3mm on average, even though
Bogo et al. exploits a high-quality detailed statistical 3D
body model [23] trained on thousands of 3D body scans,
that captures both the variation of human body shape and
its deformation through pose.
MPII and Leeds datasets:
The proposed approach
trained exclusively on the Human3.6M dataset can be used
to identify 2D and 3D landmarks of images contained in
different datasets. Figure 4 shows some qualitative results
on the MPII dataset [5] and on the Leeds dataset [18], in-
cluding failure cases. Notice how the probabilistic 3D pose
model generates anatomically plausible poses even though
the 2D landmark estimations are not all correct. However,
as shown in bottom row, even small errors in 2D pose can
lead to drastically different 3D poses. These inaccuracies
could be mitigated without further 3D data by annotating
additional RGB images for training from different datasets.