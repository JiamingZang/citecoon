# Monocular Total Capture: Posing Face, Body, and Hands in the Wild

> 2019 · id: W2963873475 · arXiv: 1812.01598 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
We present the ﬁrst method to capture the 3D total mo-
tion of a target person from a monocular view input. Given
an image or a monocular video, our method reconstructs
the motion from body, face, and ﬁngers represented by a
3D deformable mesh model. We use an efﬁcient represen-
tation called 3D Part Orientation Fields (POFs), to en-
code the 3D orientations of all body parts in the common
2D image space. POFs are predicted by a Fully Convo-
lutional Network (FCN), along with the joint conﬁdence
maps.
To train our network, we collect a new 3D hu-
man motion dataset capturing diverse total body motion of
40 subjects in a multiview system. We leverage a 3D de-
formable human model to reconstruct total body pose from
the CNN outputs by exploiting the pose and shape prior in
the model. We also present a texture-based tracking method
to obtain temporally coherent motion capture output. We
perform thorough quantitative evaluations including com-
parison with the existing body-speciﬁc and hand-speciﬁc
methods, and performance analysis on camera viewpoint
and human pose changes. Finally, we demonstrate the re-
sults of our total body motion capture on various challeng-
ing in-the-wild videos. Our code and newly collected hu-
man motion dataset will be publicly shared.

## introduction
Human motion capture is essential for many applications
including visual effects, robotics, sports analytics, medi-
cal applications, and human social behavior understanding.
However, capturing 3D human motion is often costly, re-
quiring a special motion capture system with multiple cam-
eras. For example, the most widely used system [2] needs
multiple calibrated cameras with reﬂective markers care-
fully attached to the subjects’ body. The actively-studied
markerless approaches are also based on multi-view sys-
tems [18, 26, 16, 22, 23] or depth cameras [46, 7]. For this
reason, the amount of available 3D motion data is extremely
limited. Capturing 3D human motion from single images or
videos can provide a huge breakthrough for many applica-
tions by increasing the accessibility of 3D human motion
data, especially by converting all human-activity videos on
the Internet into a large-scale 3D human motion corpus.
Reconstructing 3D human pose or motion from a monoc-
ular image or video, however, is extremely challenging due
to the fundamental depth ambiguity. Interestingly, humans
are able to almost effortlessly reason about the 3D human
body motion from a single view, presumably by leverag-
ing strong prior knowledge about feasible 3D human mo-
tions. Inspired by this, several learning-based approaches
have been proposed over the last few years to predict 3D
human body motion (pose) from a monocular video (im-
1
arXiv:1812.01598v1  [cs.CV]  4 Dec 2018

age) [53, 41, 4, 54, 9, 32, 30, 64, 24, 33] using available
2D and 3D human pose datasets [5, 25, 1, 19, 22]. Re-
cently, similar approaches have been introduced to predict
3D hand poses from a monocular view [65, 34, 11]. How-
ever, fundamental difﬁculty still remains due to the lack of
available in-the-wild 3D body or hand datasets that provide
paired images and 3D pose data; thus most of the previous
methods only demonstrate results in controlled lab environ-
ments. Importantly, there exists no method that can recon-
struct motion from all body parts including body, hands, and
face altogether in a single view, although this is important
to fully understand human behavior.
In this paper, we aim to reconstruct the 3D total mo-
tions [23] of a human using a monocular imagery captured
in the wild. This ambitious goal requires solving challeng-
ing 3D pose estimation problems for different body parts
altogether, which are often considered as separate research
domains. Notably, we apply our method to in-the-wild sit-
uations (e.g., videos from YouTube), which has rarely been
demonstrated in previous work. We use a 3D representation
named Part Orientation Fields (POFs) to efﬁciently encode
the 3D orientation of a body part in the 2D space. A POF is
deﬁned for each body part that connects adjacent joints in
torso, limbs, and ﬁngers, and represents relative 3D orien-
tation of the rigid part regardless of the origin of 3D Carte-
sian coordinates. POFs are efﬁciently predicted by a Fully
Convolutional Network (FCN), along with 2D joint conﬁ-
dence maps [55, 60, 13]. To train our networks, we collect
a new 3D human motion dataset containing diverse body,
hands, and face motions from 40 subjects. Separate CNNs
are adopted for body, hand and face, and their outputs are
consolidated together in a uniﬁed optimization framework.
We leverage a 3D deformable model that is built for the to-
tal capture [22] in order to exploit the shape and motion
prior embedded in the model. In our optimization frame-
work, we ﬁt the model to the CNN measurements at each
frame to simultaneously estimate the 3D motion of body,
face, ﬁngers, and feet. Our mesh output also enables us to
additionally reﬁne our motion capture results for better tem-
poral coherency by optimizing the photometric consistency
in the texture space.
This paper presents the ﬁrst approach to monocular total
motion capture in various challenging in-the-wild scenarios
(e.g., Fig. 1). We demonstrate that our single framework
achieves comparable results to existing state-of-the-art 3D
body or hand pose estimation methods on public bench-
marks. Notably, our method is applied to various in-the-
wild videos, which has rarely been demonstrated in either
3D body or hand estimation area. We also conduct thor-
ough experiments on our newly collected dataset to quan-
titatively evaluate the performance of our method with re-
spect to viewpoint and body pose changes. The major con-
tributions of our paper are summarized as follows:
• We present the ﬁrst method to produce 3D total mo-
tion capture results from a monocular image or a
video in various challenging in-the-wild scenarios.
• We introduce an optimization framework to ﬁt a de-
formable human model on 3D POFs and 2D key-
point measurements for total body pose estimation,
and show comparable results to the state-of-the-art
methods in both 3D body and 3D hand estimation
benchmarks.
• We present a method to enforce photometric consis-
tency across time to reduce motion jitters.
• We capture a new 3D human motion dataset with 40
subjects to provide training and evaluation data for
monocular total motion capture.

## method
Pavlakos
[39]
Zhou
[64]
Luo
[28]
Martinez
[30]
Fang
[17]
Yang
[61]
Pavlakos
[38]
Dabral
[15]
Sun
[52]
*Kanazawa
[24]
*Metah
[32]
*Metah
[31]
*Ours
*Ours+
MPJPE
71.9
64.9
63.7
62.9
60.4
58.6
56.2
55.5
49.6
88.0
80.5
69.9
58.3
64.5
Table 1: Quantitative comparison with previous work on Human3.6M dataset. The ‘*’ signs indicate methods that show
results on in-the-wild videos. The evaluation metric is Mean Per Joint Position Error (MPJPE) in millimeter. The numbers
are taken from original papers. ‘Ours’ and ‘Ours+’ refer to our results without and with prior respectively.
3D body pose dataset, and Human3.6M for 165k iterations
with a batch size of 4. During testing time, we ﬁt Adam
model [23] onto the network output. Since Human3.6M
has a different joint deﬁnition from Adam model, we build
a linear regressor to map Adam mesh vertices to 17 joints in
Human3.6M deﬁnition using the training set, as in [24]. For
evaluation, we follow [39] to rescale our output to match the
size of an average skeleton computed from the training set.
The Mean Per Joint Position Error (MPJPE) after aligning
the root joint is reported as in [39].
The experimental results are shown in Table 1.
Our
method achieves competitive performance; in particular, we
show the lowest pose estimation error among all methods
that demonstrate their results on in-the-wild videos (marked
with ‘*’ in the table). We argue that this is important be-
cause methods are in general prone to overﬁtting to this
speciﬁc dataset. As an example, our result with pose prior
shows increased error compared to our result without prior,
although we ﬁnd that pose priors helps to produce good sur-
face structure and joint angles in the wild.
Ablation Studies. We investigate the importance of each
dataset through ablation studies on Human3.6M. We com-
pare the reconstruction error by training networks with: (1)
Human3.6M; (2) Human3.6M and our captured dataset; and
(3) Human3.6M, our captured dataset, and COCO. Note
that setting (3) is the method we used for the previous com-
parison. We follow the same evaluation protocol and metric
as in Table 1. The result is shown in Table 2. First, it is
worth noting that with only Human3.6M as training data,
we already achieve the best results among results marked
with ‘*’ in Table 1. Second, comparing (2) with (1), our
new dataset provides an improvement despite the difference
in background, human appearance and pose distribution be-
tween our dataset and Human3.6M. This veriﬁes the value
of our new dataset. Third, we see a drop in error when
we add COCO to the training data, which suggests that our
framework can take advantage of this dataset with only 2D
human pose annotation for 3D pose estimation.
7.2.2
3D Hand Pose Estimation.
We evaluate our method on the Stereo Hand Pose Tracking
Benchmark (STB) and Dexter+Object (D+O), and compare
our result with previous methods. For this experiment we
use the separate hand model of Frankenstein in [23].
STB. Since the STB dataset has a palm joint rather than
Training data
MPJPE
(1) Human3.6M
65.6
(2) Human3.6M + Ours
60.9
(3) Human3.6M + Ours + COCO
58.3
Table 2: Ablation studies on Human3.6M. The evaluation
metric is Mean Per Joint Position Error in millimeter.
20
30
40
50
Error Thresholds (mm)
0.8
0.85
0.9
0.95
1
3D PCK
STB dataset
Zimmermann et al. (0.948)
Mueller et al. (0.965)
Spurr et al. (0.983)
Iabal et al. (0.994)
Cai et al. (0.994)
Ours (0.994)
0
20
40
60
80
100
Error Thresholds (mm)
0
0.2
0.4
0.6
0.8
1
Dexter+Object dataset
+Sridhar et al. (0.81)
Mueller et al. (0.56)
Iqbal et al. (0.71)
Ours (0.70)
*Mueller et al. (0.70)
*Ours (0.84)
Figure 6: Comparison with previous work on 3D hand pose
estimation datasets. We plot PCK curve and show AUC
in bracket for each method in legend. Left: results on the
STB dataset [63] in 20mm - 50mm; right: results on Dex-
ter+Object dataset [49] in 0 - 100mm. Results with depth
alignment are marked with ‘*’; the RGB-D based method is
marked with ‘+’.
the wrist joint used in our method, we convert the palm joint
to wrist joint as in [65] to train our CNN. We also learn a
linear regressor using the training set of STB dataset. Dur-
ing testing, we regress back the palm joint from our model
ﬁtting output for comparison. For the evaluation, we follow
the previous work [65] and compute the error after aligning
the position of root joint and global scale with the ground
truth, and report the Area Under Curve (AUC) of the Per-
centage of Correct Keypoints (PCK) curve in the 20mm-
50mm range. The results are shown in the left of Fig. 6.
Our performance is on par with the state-of-the-art meth-
ods that are designed particularly for hand pose estimation.
We also point out that the performance on this dataset has
almost saturated, because the percentage is already above
90% even at the lowest threshold.
D+O. Following [34] and [20], we report our results us-
ing a PCK curve and the corresponding AUC, as shown
in the right of Fig. 6. Since previous methods are eval-
uated by estimating the absolute 3D depth of 3D hand
7

-180
180
0
90
-90
0
60
5
6
7
Elevation
Azimuth
Figure 7: Evaluation result in Panoptic Studio. Top: accu-
racy vs. view point; bottom: accuracy vs. pose. The metric
is MPJPE in cm. The average MPJPE for all testing samples
is 6.30 cm.
0
100
200
300
400
500
600
−20
−10
0
100
200
300
400
500
600
−30
−20
−10
0
100
200
300
400
500
600
260
280
coordinate
coordinate
coordinate
ground truth
after tracking
before tracking
(cm)
(cm)
(cm)
x
y
z
Figure 8: The comparison of joint location across time be-
fore and after tracking with ground truth.
The horizon-
tal axes show frame numbers (30fps) and the vertical axes
show joint locations in camera coordinate. The target joint
here is the left shoulder of the subject.
joints, we follow them by ﬁnding an approximate hand
scale using a single frame in the dataset, and ﬁx the
scale during the evaluation. In this case, our performance
(AUC=0.70) is comparable with the previous state-of-the-
art [20] (AUC=0.71). However, since there is fundamental
depth-scale ambiguity for single-view pose estimation, we
argue that aligning the root with the ground truth depth is
a more reasonable evaluation setting. In this setting, our
method (AUC=0.84) outperforms the previous state-of-the-
art method [34] (AUC=0.70) in the same setting, and even
achieves better performance than an RGB-D based method
[49] (AUC=0.81).
7.3. Quantitative Study for View and Pose Changes
Our new 3D pose data contain multi-view images with
the diverse body postures. This allows us to quantitatively
study the performance of our method in view changes and
body pose changes. We compare our single view 3D body
reconstruction result with the ground truth.
Due to the
scale-depth ambiguity of monocular pose estimation, we
align the depth of root joint to the ground truth by scaling
our result along the ray directions from the camera center,
and compute the Mean Per Joint Position Error (MPJPE) in
centimeter. The average MPJPE for all testing samples is
6.30 cm. We compute the average errors per each camera
viewpoint, as shown in the top of Fig. 7. Each camera view-
point is represented by azimuth and elevation with respect
to the subjects’ initial body location. We reach two interest-
ing ﬁndings: ﬁrst, the performance worsens in the camera
views with higher elevation due to the severe self-occlusion
and foreshortening; second, the error is larger in back views
compared to the frontal views because limbs are occluded
by torso in many poses. At the bottom of Fig. 7, we show
the performance for varying body poses. We run k-means
algorithm on the ground truth data to ﬁnd body pose groups
(the center poses are shown in the ﬁgure), and compute the
error for each cluster. Body poses with more severe self-
occlusion or foreshortening tend to have higher errors.
7.4. The Effect of Mesh Tracking
To demonstrate the effect of our temporal reﬁnement
method, we compare the result of our method before and
after this reﬁnement stage using Panoptic Studio data. We
plot the reconstructed left shoulder joint in Fig. 8. We ﬁnd
that the result after tracking (in blue) tends to be more tem-
porally stable than that before tracking (in green), and is
often closer to the ground truth (in red).
7.5. Qualitative Evaluation
Qualitative Results on Images: In this section we present
qualitative results of our method on individual images in
Fig.
9.
We show results on images with various back-
ground, human appearance and poses. Our method works
well for both indoor Mocap images (the ﬁrst row in Fig. 9)
and in-the-wild images (the latter 2 rows).
Qualitative Results on Video Sequences: We show results
of our method on video sequences. We test our method on
two kinds of videos. First, we take videos of human motion
using camera by ourselves; second, we use videos down-
loaded from Youtube. The results are presented in our sup-
plementary video. For videos where only the upper body of
the target person is vi

## experiments
We quantitatively evaluate the performance of our
method on public benchmarks for 3D body pose estimation
and hand pose estimation. We also thoroughly evaluate our
method on view point changes and human pose changes in
Figure 5: Illustration of our temporal reﬁnement algorithm.
The top row shows meshes projected on input images at
previous frame, current target frame, and after reﬁnement.
In zoom-in views a particular vertex is shown in blue, which
is more consistent after applying our tracking method.
our newly collected multi-view human pose dataset. For
all quantitative experiments, we use the camera intrincics
provided by the datasets. We ﬁnally show our total motion
capture results in various challenging videos recorded by us
or obtained from YouTube. Our qualitative results are best
shown in our supplementary videos.
7.1. Dataset
Body Pose Dataset: Human3.6M [19] is a large-scale in-
door marker-based human MoCap dataset, and currently the
most commonly used benchmark for 3D body pose estima-
tion. We quantitatively evaluate the body part of our algo-
rithm on it. We follow the standard training-testing protocol
as in [39].
Hand Pose Dataset: Stereo Hand Pose Tracking Bench-
mark (STB) [63] is a 3D hand pose dataset consisting of
30K images for training and 6K images for testing. Dex-
ter+Object (D+O) [49] is a hand pose dataset captured by
an RGB-D camera, providing about 3K testing images in 6
sequences. Only the locations of ﬁnger tips are annotated.
Newly Captured Total Motion Dataset:
We use the
Panoptic Studio [21, 22] to capture a new dataset for 3D
body and hand pose in a markerless way [23]. We use 31
HD cameras to capture 40 subjects. Each subject makes
a wide range of motion in body and hand under the guid-
ance of a video for 2.5 minutes. After cleaning out the erro-
neous frames, we obtain about 834K body images and 111K
hand images with corresponding 3D pose data. We split this
dataset into training and testing set such that no subject ap-
pears in both. This dataset will be publicly shared.
7.2. Quantitative Comparison with Previous Work
7.2.1
3D Body Pose Estimation.
Comparison on Human3.6M. We compare the perfor-
mance of our single-frame body pose estimation method
with previous state-of-the-arts. Our network is initialized
from the 2D body pose estimation network of OpenPose.
We train the network using COCO dataset [25], our new
6

## related_work
Single Image 2D Human Pose Estimation: Over the
last few years, great progress has been made in detecting
2D human body keypoints from a single image [56, 55,
10, 60, 35, 13] by leveraging large-scale manually anno-
tated datasets [25, 5] with deep Convolutional Neural Net-
work (CNN) framework. In particular, the major break-
through is boosted by using the fully convolutional archi-
tectures to produce conﬁdence scores for each joint with a
heatmap representation [55, 60, 35, 13], which is known
to be more efﬁcient than directly regressing the joint loca-
tions with fully connected layers [56]. A recent work [13]
similarly learns the connectivity between pairs of adjacent
joints, called the Part Afﬁnity Fields (PAFs) in the form of
2D heatmaps, to assemble 2D keypoints for different indi-
viduals in the multi-person 2D pose estimation problem.
Single Image 3D Human Pose Estimation:
Early
work [41, 4] models the 3D human pose space as an
over-complete dictionary learned from a 3D human motion
database [1]. More recent approaches rely on deep neural
networks, which are roughly divided into two directions:
two-stage methods and direct estimation.
The two-stage
methods take 2D keypoint estimation as input and focus on
lifting 2D human poses to 3D independently without input
image [9, 14, 30, 33, 36, 17]. These methods ignore rich
information in images that encodes 3D information, such as
shading and appearance, and also suffer from sensitivity to
2D localization error. Direct estimation methods predict 3D
human pose directly from images, in the form of direct coor-
dinate regression [42, 51, 52], voxel prediction [39, 29, 58]
or depth map prediction [64]. Similar to ours, a recent work
uses 3D orientation ﬁelds [28] as an intermediate represen-
tation for the 3D body pose. However, these models are
usually trained on MoCap datasets, with limited ability to
generalize to in-the-wild scenarios.
2

CNN 
(Sec. 4)
Model Fitting 
(Sec.5)
Deformable Human Model
Mesh Tracking 
(Sec. 6)
Input Image Ii
Joint Confidence Maps S
Part Orientation Fields L
Input Image Ii−1
Model Parameters Ψi
Model Parameters Ψ+
i
Model ParametersΨ+
i−1
Figure 2: An overview of our method. Our method is composed of CNN part, mesh ﬁtting part, and mesh tracking part.
Due to the above limitations, some methods have been
proposed to integrate prior knowledge about human pose
for better in-the-wild performance. Some work [38, 44, 59]
proposes to use ordinal depth as additional supervision for
CNN training. Additional loss functions are introduced in
[64, 15] to enforce constraints on predicted bone length and
joint angles. Some work [24, 61] uses Generative Adver-
sarial Networks (GAN) to exploit human pose prior in data-
driven approaches.
Monocular Hand Pose Estimation: Hand keypoint es-
timation is often considered as independent research do-
main from body pose estimation. Most of previous work is
based on depth image as input [37, 50, 45, 48, 57, 62], while
RGB-based method is introduced recently, for 2D keypoint
estimation [47] and 3D pose estimation [65, 11, 20].
3D Deformable Human Models: 3D deformable mod-
els are commonly used for markerless body [6, 27, 40] and
face motion capture [8, 12] to restrict the reconstruction out-
put to the parametric shape and motion spaces deﬁned by
the models. Although the outputs are limited by the ex-
pressive power of models (e.g., some body models cannot
express clothing and some face models cannot express wrin-
kles), they greatly simplify the 3D motion capture problem.
We can ﬁt the models based on available measurements by
optimizing cost functions with respect to the model param-
eters. Recently, a generative 3D model that can express
body and hands is introduced by Romero et al. [43]; the
Adam model is introduced by Joo et al. [23] to enable the
total body motion capture (face, body and hands), which we
adopt for monocular total capture.
3. Method Overview
Our method takes as input a sequence of images captur-
ing the motion of a single person from a monocular RGB
camera, and outputs the 3D total body motion capture (in-
cluding the motion from body, face, hands, and feet) of
the target person in the form of a deformable 3D human
model [27, 23] for each frame. Given a N-frame video se-
quence, our method produces the parameters of the 3D hu-
man body model [23], including body motion parameters
{θi}N
i=1, facial expression parameters {σi}N
i=1, and global
translation parameters {ti}N
i=1. The body motion parame-
ter θ includes hands and feet motions, as well as the global
rotation of the body. Our method also estimates shape co-
efﬁcients φ shared among all frames in the sequence, while
θ, σ, and t are estimated for each frame respectively. The
output parameters are deﬁned by the 3D deformable human
model Adam [23]. Note that our method can be also applied
to capture only a subset of total motions (e.g., body motion
only with the SMPL model [27] or hand motion only by
separate hand model of Frankenstein in [23]). We denote a
set of all parameters (φ, θ, σ, t) by Ψ, and denote the result
for the i-th frame by Ψi.
Our method is divided into 3 stages, as shown in Fig. 2.
In the ﬁrst stage, each image is fed into a Convolutional
Neural Network (CNN) obtain the joint conﬁdence maps
and the 3D orientation information of body parts, which we
call the 3D Part Orientation Fields (POFs). In the second
stage, we perform total body motion capture by ﬁtting a de-
formable human mesh model [23] on the image measure-
ments produced by the CNNs. We utilize the prior informa-
tion embedded in the human body model for better robust-
ness of results against the noise in CNN outputs. This stage
produces the 3D pose for each frame independently, rep-
resented by parameters of the deformable model {Ψi}N
i=1.
In the third stage, we additionally enforce temporal consis-
tency across frames to reduce motion jitters. We deﬁne a
cost function to ensure photometric consistency in the tex-
ture domain of mesh model, based on the initial ﬁtting out-
puts of the second stage. This stage produces reﬁned model
parameters {Ψ+
i }N
i=1. We demonstrate that this temporal
reﬁnement is crucial to obtain realistic body motion capture
output.
4. Predicting 3D Part Orientation Fields
The 3D Part Orientation Field (POF) encodes the 3D
orientation of a body part of an articulated structure (e.g.,
limbs, torso, and ﬁngers) in 2D image space. The same rep-
resentation is used in a very recent literature [28], and we
describe the details and notations used in our total motion
capture framework. We pre-deﬁne a human skeleton hier-
archy S in the form of a set of ‘(parent, child)’ pairs1. A
rigid body part connecting a 3D parent joint Jm ∈R3 and
a child joint Jn ∈R3, (m, n) ∈S, is denoted by P(m,n),
with Jm, Jn deﬁned in the camera coordinate. Its 3D ori-
entation ˆP(m,n) is represented by a unit vector from Jm to
1See the appendix for our body and hand skeleton deﬁnition.
3

Figure 3: An illustration of a Part Orientation Field. The
orientation ˆP(m,n) for body part P(m,n) is a unit vector
from Jm to Jn. In POFs, all pixels belong to this part are
assigned the value of this vector in x, y, z channels.
Jn in R3 :
ˆP(m,n) =
Jn −Jm
||Jn −Jm|| .
(1)
For a speciﬁc body part P(m,n), we deﬁne a Part Orienta-
tion Field L(m,n) ∈R3×h×w to represent its 3D orientation
ˆP(m,n) as a 3-channel heatmap (for x, y, z coordinates re-
spectively) in the image space, where h and w are the size
of image. The value of the heatmap at x in the POF L(m,n)
is deﬁned as,
L(m,n)(x) =
(ˆP(m,n)
if x ∈P(m,n),
0
otherwise.
(2)
Note that the POF values are deﬁned only for the pixel re-
gion belonging to the current target part P(m,n) and we fol-
low [13] to deﬁne the pixels belonging to the part as a rect-
angular (please refer to [13] for details). An example POF
of a body part (right lower arm) is shown in Fig. 3.
Implementation Details: We train a CNN to predict joint
conﬁdence maps S and Part Orientation Fields L. The input
image is cropped around the target person to 368×368, with
the bounding box given by OpenPose2 [13, 47] during test-
ing. We follow [13] for CNN architecture with minimum
change. We use 3 channels to estimate POF instead of 2
channels in [13] for every body part in S. L2 loss is applied
to network prediction on S and L. We also train on our net-
work on images with 2D pose annotations (e.g. COCO). In
this situation we only supervise the network with loss on S.
Two networks are trained for body and hands separately.
5. Model-Based 3D Pose Estimation
Ideally the joint conﬁdence maps S and POFs L pro-
duced by CNN provide sufﬁcient information to reconstruct
2https://github.com/CMU-Perceptual-Computing-Lab/
openpose
a 3D skeletal structure up to scale [28]. In practice, the S
and L can be noisy, so we exploit a 3D deformable mesh
model to more robustly estimate 3D human pose with the
shape and pose priors embedded in the model. In this sec-
tion, we ﬁrst describe our mesh

## conclusion
In this paper, we present a method to simultaneously re-
construct 3D total motion of a single person from an image
or a monocular video. We thoroughly evaluate the robust-
ness of our method on various benchmarks and demonstrate
monocular 3D total motion capture results on in-the-wild
videos. There are some limitations with our method. First,
8

Figure 9: Qualitative results of our method on in-the-wild images. For each example, we show input images and our predic-
tion with zoom-in views as well as side and top views.
we observe failure cases when a signiﬁcant part of the tar-
get person is invisible (out of image boundary or occluded
by other objects) due to erroneous network prediction. Sec-
ond, our hand pose detector fails in the case of insufﬁcient
resolution or severe motion blur. Third, our CNN requires
bounding boxes for body and hands as input, and cannot
handle multiple bodies or hands simultaneously. Solving
these problems points to interesting future directions.