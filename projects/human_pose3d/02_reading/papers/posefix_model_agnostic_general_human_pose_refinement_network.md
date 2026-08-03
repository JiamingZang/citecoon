# PoseFix: Model-Agnostic General Human Pose Refinement Network

> 2019 · id: W2904290346 · arXiv: 1812.03595 · pdf: https://arxiv.org/pdf/1812.03595 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

PoseFix: Model-agnostic General Human Pose Reﬁnement Network
Gyeongsik Moon
Department of ECE, ASRI
Seoul National University
mks0601@snu.ac.kr
Ju Yong Chang
Department of EI
Kwangwoon University
juyong.chang@gmail.com
Kyoung Mu Lee
Department of ECE, ASRI
Seoul National University
kyoungmu@snu.ac.kr
Abstract
Multi-person pose estimation from a 2D image is an es-
sential technique for human behavior understanding.
In
this paper, we propose a human pose reﬁnement network
that estimates a reﬁned pose from a tuple of an input image
and input pose. The pose reﬁnement was performed mainly
through an end-to-end trainable multi-stage architecture in
previous methods. However, they are highly dependent on
pose estimation models and require careful model design.
By contrast, we propose a model-agnostic pose reﬁnement
method. According to a recent study, state-of-the-art 2D
human pose estimation methods have similar error distri-
butions. We use this error statistics as prior information
to generate synthetic poses and use the synthesized poses
to train our model. In the testing stage, pose estimation
results of any other methods can be input to the proposed
method. Moreover, the proposed model does not require
code or knowledge about other methods, which allows it to
be easily used in the post-processing step. We show that
the proposed approach achieves better performance than
the conventional multi-stage reﬁnement models and consis-
tently improves the performance of various state-of-the-art
pose estimation methods on the commonly used benchmark.
The code is available in this https URL1.
1. Introduction
The goal of human pose estimation is to localize seman-
tic keypoints of a human body. It is an essential technique
for human behavior understanding and human-computer in-
teraction.
Recently, many methods [5, 7, 11, 14, 15, 18,
20, 22, 23, 26, 30] utilize deep convolutional neural net-
works (CNNs) and achieved noticeable performance im-
provement. They are also updating performance limits in
annual competitions for 2D human keypoint detection such
as MS COCO keypoint detection challenge [19].
In this paper, we propose a human pose reﬁnement net-
1https://github.com/mks0601/PoseFix_RELEASE
keypoints: 
(x1, y1),
(x2, y2),
(x3, y3), .. 
(xN, yN)
Pose result file of 
method A
Input pose 
Refined pose
Input image
keypoints: 
(x1, y1),
(x2, y2),
(x3, y3), .. 
(xN, yN)
Pose result file of 
method B
keypoints: 
(x1, y1),
(x2, y2),
(x3, y3), .. 
(xN, yN)
Pose result file of 
method C
(a) Pose estimation results of any methods
(b) Pose error fix procedure of the PoseFix
PoseFix
Figure 1: Testing pipeline of the PoseFix. It takes pose
estimation results of any other method with an input image
and outputs a reﬁned pose. Note that the PoseFix does not
require any code or knowledge about other methods.
work that estimates a reﬁned pose from a tuple of an in-
put image and a pose.
Conventionally, the pose reﬁne-
ment has been mainly performed by multi-stage architec-
tures [4,7,21,29]. In other words, the initial pose and image
features generated in the ﬁrst stage go through subsequent
stages, and each stage outputs a reﬁned pose. These multi-
stage architectures are usually trained in an end-to-end man-
ner.
However, the conventional multi-stage architecture-
based reﬁnement approach is highly dependent on the pose
estimation model and requires careful design for successful
reﬁnement. By contrast, in this work, we propose a model-
agnostic pose reﬁnement method that does not depend on
the pose estimation model.
Recent research by Ronchi et al. [24] gave us a clue on
how to design a general model-agnostic pose reﬁner. They
analyzed the results of the MS COCO 2016 keypoint detec-
tion challenge winners [5,22] by using new pose estimation
evaluation metrics, i.e., keypoint similarity (KS) and object
keypoint similarity (OKS). They taxonomized pose estima-
tion errors into several types such as jitter, inversion, swap,
1
arXiv:1812.03595v3  [cs.CV]  10 Mar 2019

and miss and described how frequently these errors occur
and how much they can negatively affect performance. Al-
though the winners [5, 22] used very different approaches,
their pose error distributions are very similar, which indi-
cates that common issues exist for more accurate pose esti-
mation.
Our basic idea is to use this error statistics as prior infor-
mation to generate synthetic poses and use the synthesized
poses to train the proposed pose reﬁnement model (Pose-
Fix). To train our model, we generate each type of the er-
rors (i.e., jitter, inversion, swap, and miss) based on the pose
error distributions from Ronchi et al. [24], and construct di-
verse and realistic poses. The generated input pose is fed to
the PoseFix with the input image, and the PoseFix learns to
reﬁne the pose. We design our PoseFix as a single-stage ar-
chitecture with a coarse-to-ﬁne estimation pipeline. It takes
the input pose in a coarse form and estimates the reﬁned
pose in a ﬁner form. The coarse input pose enables the pro-
posed model to focus not only on an exact location of the
input pose but also around it, allowing our model to ﬁx the
error of the input pose. Furthermore, the ﬁner form of the
output pose enables the proposed model to localize the loca-
tion of the pose more exactly compared to existing methods.
After training, our PoseFix can be applied to and reﬁne the
pose estimation results of any single- or multi-person pose
estimation method. Figure 1 shows such a pose reﬁnement
pipeline of the proposed PoseFix.
Our contributions can be summarized as follows.
• We show that model-agnostic general pose reﬁnement
is possible. The PoseFix is trained independently of
the pose estimation model. Instead, it is based on error
statistics obtained through empirical analysis.
• Our PoseFix can take the pose estimation result of any
pose detection method as the input. As the PoseFix
does not require any code or knowledge about other
methods, our model has very high ﬂexibility and ac-
cessibility.
• We design the PoseFix as a coarse-to-ﬁne estimation
system. We empirically observed that this coarse-to-
ﬁne pipeline is crucial for successful pose reﬁnement.
• Our PoseFix achieves a better result than the conven-
tional multi-stage architecture-based reﬁnement meth-
ods. Also, the PoseFix consistently improves the per-
formance of various state-of-the-art pose estimation
methods on the commonly used benchmark.
2. Related works
Single-person pose estimation. Toshev et al. [28] di-
rectly estimated the Cartesian coordinates of body joints
by using a multi-stage deep network and achieved state-of-
the-art performance.
Tompson et al. [27] jointly trained
a CNN and a graphical model. The CNN estimated 2D
heatmaps for each joint, and they were used as the unary
term for the graphical model. Liu et al. [29] used multi-
stage CNN which progressively enlarges receptive ﬁelds
and reﬁnes the pose estimation result. Newell et al. [21]
proposed a stacked hourglass network which repeats down-
sampling and upsampling to exploit multi-scale informa-
tion effectively. Carreria et al. [6] proposed an iterative er-
ror feedback-based human pose estimation system. Chu et
al. [8] enhanced the stacked hourglass network [21] by in-
tegrating it with a multi-context attention mechanism. Ke et
al. [16] proposed a multi-scale structure-aware network
which achieved leading position in the publicly available
human pose estimation benchmark [3].
Multi-person pose estimation. There are two main ap-
proaches in the multi-person pose estimation. The ﬁrst one,
top-down approach, relies on a human detector that predicts
bounding boxes of humans. The detected human image is
cropped and fed to the pose estimation network. The second
one, bottom-up approach, localizes all human body key-
points in an input image and assembles them using proposed
clustering algorithms in each work.
[7,11,14,22,26,30] are based on the top-down approach.
He et al. [11] proposed Mask R-CNN that can perform hu-
man detection and keypoint localization in a single model.
Instead of cropping the detected humans in the input image,
it crops human features from a feature map via the differen-
tiable RoIAlign layer. Chen et al. [7] proposed a cascaded
pyramid network (CPN) which consists of two networks.
The ﬁrst one, GlobalNet, is based on deep backbone net-
work and upsampling layers with skip connections. The
second one, ReﬁneNet, is built to reﬁne the estimation re-
sults from the GlobalNet by focusing on hard keypoints.
Xiao et al. [30] used a simple pose estimation network that
consists of a deep backbone network and several upsam-
pling layers. Although it is based on a simple network ar-
chitecture, it achieved state-of-the-art performance on the
commonly used benchmark [19].
[5,15,18,20,23] are based on the bottom-up approach.
DeepCut [23] assigned the detected keypoints to each per-
son in an image by formulating the assignment problem as
an integer linear program. DeeperCut [23] improves the
DeepCut [23] by introducing image-conditioned pair-wise
terms. Cao et al. [5] proposed part afﬁnity ﬁelds (PAFs)
that directly expose the association between human body
keypoints. They assembled the localized keypoints of all
persons in the input image by using the estimated PAFs.
Newell et al. [20] introduced a pixel-wise tag value to as-
sign localized keypoints to a certain human. Kocabas et
al. [18] proposed a pose residual network to assign detected
keypoints to each person. Their model can jointly handle
person detection, keypoint detection, and person segmenta-
tion.

Image 
Input pose 
(P)
Backbone + Upsampler
x-axis meshgrid
y-axis meshgrid
1  2  3  ..  w
1  2  3  ..  w
1  2  3  ..  w
..
1  2  3  ..  w
1  1  1  ..  1   
2  2  2  ..  2  
3  3  3  ..  3  
..
h  h  h  ..  h
Refined pose 
(C)
Refined pose 
(H)
Nose:Jitter
Hip: Good
Elbow: Jitter
        ..
Knee: Inv
Pose result file 
of any methods
GT pose
Testing
Synthesized errors
Training
~
Error distribution
Prob
Error type
keypoints
(x1, y1),
(x2, y2),
(x3, y3), .. 
(xN, yN)
Figure 2: Overall pipeline of the PoseFix. In the training stage, the input pose is generated by synthesizing the pose errors
based on the real pose error distributions on the groundtruth pose. In the testing stage, pose estimation results of any other
methods become the input pose. The heatmaps are visualized by performing max pooling along the channel axis.
Human pose reﬁnement.
Many methods attempted
to reﬁne the estimated keypoint for more accurate perfor-
mance. Newell et al. [21], Bulat and Tzimiropoulos [4],
Liu et al. [29], and Chen et al. [7] utilized an end-to-
end trainable multi-stage architecture-based network. Each
stage tries to reﬁne the pose estimation results of the previ-
ous stage via end-to-end learning. Carreria et al. [6] itera-
tively estimated error feedback from a shared weight model.
The output error feedback of the previous iteration is trans-
formed into the input pose of the next iteration, which is
repeated several times for progressive pose reﬁnement. All
of these methods combine pose estimation and reﬁnement
into a single model, and each reﬁnement module is depen-
dent on estimation. Therefore, the reﬁnement modules have
different structures, and they are not guaranteed to work
successfully when they are combined with other estimation
methods. On the other hand, our pose reﬁnement method
is independent of the estimation, and therefore the results
can be consistently improved regardless of the prior pose
estimation method.
Recently, Fieraru et al. [10] proposed a post-processing
network to reﬁne the pose estimation results of other meth-
ods, which is conceptually similar to ours. They synthe-
sized pose for training and employed simple network ar-
chitecture that estimates reﬁned heatmaps and offset vec-
tors for each joint. While their method follows ad-hoc rules
to generate input pose, our method is based on actual er-
ror statistics obtained through empirical analysis.
Also,
our network with coarse-to-ﬁne structure achieves a much
stronger reﬁnement performance than their simpler one.
3. Overview of the proposed model
The goal of the PoseFix is to reﬁne the input 2D coordi-
nates of the human body keypoints of all persons in an input
image. To address this problem, our system is constructed
based on the top-down pipeline which processes a tuple of
a cropped human image and a given pose estimation result
of that human instead of processing an entire image includ-
ing multiple persons. In the training stage, the input pose
is synthesized on the groundtruth pose realistically and di-
versely. In the testing stage, pose estimation results of any
other methods can be the input pose to our system. The
overall pipeline of the PoseFix is illustrated in Figure 2.
4. Synthesizing poses for training
To train the PoseFix, we generate synthesized poses us-
ing the groundtruth poses. As the PoseFix should cover
different pose estimation results from various methods in
the testing stage, synthesized poses need to be diverse and
realistic. To satisfy these properties, we generate synthe-
sized poses randomly based on the error distributions of
real poses as described in [24]. The distributions include
the frequency of each pose error (i.e., jitter, inversion, swap,
and miss) according to the joint type, number of visible key-
points, and overlap in the input image. There may also be
joints that do not have any error, which should be synthe-
sized very close to the groundtruth to simulate correct esti-

(a) Jitter
(b) Inversion
(c) Swap
(d) Miss
Figure 3: Visualization of synthesized pose errors for each
type. The keypoint with pose error is highlighted by a yel-
low rectangle, and the groundtruth keypoints are drawn in a
yellow circle.
mations. Ronchi et al. [24] called this status good. Consid-
ering most of the empirical distributions in [24], we com-
pute the probability that each joint will have one of the pose
errors or be in the good status.
The
detailed
error
synthesis
procedure
on
each
groundtruth keypoint θp
j of joint j which belongs to a per-
son p is described in below. For more clear description, we
deﬁne j′ as a left/right inverted joint from the j, and p′ as a
different person from the p in the input image. Also, dk
j is
deﬁned as a L2 distance that makes KS with the groundtruth
keypoint becomes k for joint j. Note that dk
j depends on the
type of joint j because the error distribution of each joint
has different scale [24]. For example, eyes require more
precise localization than hips to obtain the same KS k. Fig-
ure 3 visualizes examples of synthesized pose errors of each
type.
Good.
Good status is deﬁned as a very small dis-
placement from the groundtruth keypoint. An offset vector
whose angle and length are uniformly sampled from [0, 2π)
and [0, d0.85
j
), respectively, is added to the groundtruth θp
j .
The synthesized keypoint position should be closer to the
original groundtruth θp
j than θp
j′, θp′
j , and θp′
j′ .
Jitter. Jitter error is deﬁned as a small displacement from
the groundtruth keypoint. An offset vector whose angle and
length are uniformly sampled from [0, 2π) and [d0.85
j
,d0.5
j ),
respectively, is added to the groundtruth θp
j . Similar to the
good status, the synthesized keypoint position should be
closer to the original groundtruth θp
j than θp
j′, θp′
j , and θp′
j′ .
Inversion. Inversion error occurs when a pose estima-
tion model is confused between semantically similar parts
that belong to the same instance. We restrict the inversion
error to the left/right body part confusion following [24].
The jitter error is added to θp
j′. The synthesized keypoint
position should be closer to the θp
j′ than θp
j , θp′
j , and θp′
j′ .
Swap. Swap error represents a confusion between the
same or similar parts which belong to different persons. The
jitter is added to θp′
j or θp′
j′ . The closest keypoint from the
synthesized keypoint should be θp′
j or θp′
j′ , not any of θp
j and
θp
j′.
(a) GT pose
(b) Input pose in training stage
+ synthesized error
Figure 4: Visualization of the groundtruths and synthesized
input poses. The synthesized poses are generated by adding
errors to the groundtruth poses, which are used for training
PoseFix.
Miss. Miss error represents a large displacement from
the groundtruth keypoint position. An offset vector whose
angle and length are uniformly sampled from [0, 2π) and
[d0.5
j ,d0.1
j ), respectively, is added to one of θp
j , θp′
j , θp
j′, and
θp′
j′ . The synthesized keypoint position should be at least
d0.5
j
away from all of θp
j , θp
j′, θp′
j , and θp′
j′ .
Some examples of synthesized input poses are shown in
Figure 4.
5. Architecture and learning of PoseFix
5.1. Model design
We design the PoseFix to directly estimate a reﬁned pose
from a tuple of an input image and an input pose as shown in
Figure 2. The input image and the input pose provide con-
textual and structured information to the PoseFix, respec-
tively, and the PoseFix learns to use these information to ﬁx
pose errors in the input pose. Although some errors exist in
the input pose, it still provides useful structured information
because, as indicated by Ronchi et al. [24], most keypoints
in the input pose are in good status or have jitter error which
represent a small displacement from the groundtruth pose.
This rough structured information acts like attention which
tells the PoseFix where to focus on at the human body.
We observed that by learning to ﬁx the pose errors in
the input pose, the PoseFix learns where to focus on at the
human body as in Figure 5. As it shows, although some
errors exist in the input pose, the PoseFix initially focuses
well on the reliable keypoint locations of the input pose.
And then, it successfully localizes correct keypoints without
being inﬂuenced by the errors of the input pose.

Input pose
Refined pose
Image
Feature maps
Heatmap
Figure 5: Visualization of feature maps and ﬁnal heatmaps
of the PoseFix. The feature maps and heatmaps are reduced
into one channel by max pooling along the channel axis for
visualization. The order of the feature maps and heatmaps
in the ﬁgure is the same with that of the feedforward.
5.2. Coarse-to-ﬁne estimation
To make it more robust to errors, we design the proposed
PoseFix to operate in a coarse-to-ﬁne manner. We use the
terms “coarse” and “ﬁne” by the degree of uncertainty in
representing the pose. For example, in representing the po-
sition of each joint constituting a pose, a Gaussian blob has
a high uncertainty as much as the size of its standard de-
viation. On the other hand, a one-hot vector has relatively
low uncertainty up to the size of a quantized grid. The co-
ordinates of a keypoint has the least amount of uncertainty
because it provides the exact information about the location
itself. Therefore, in our work, the coarse-to-ﬁne estimation
implies that the coarse input pose (P = {Pn}N
n=1) repre-
sented by the set of Gaussian blobs is fed to the network,
producing the ﬁner pose in the form of the one-hot vector
(H = {Hn}N
n=1), and then the ﬁnest pose in terms of the
keypoint coordinates (C = {Cn}N
n=1) is generated as the ﬁ-
nal output as illustrated in Figure 2. N denotes the number
of keypoints. In this subsection, we describe this coarse-to-
ﬁne estimation in more detail.
The input pose is constructed in a coarse form by a
single-mode Gaussian heatmap representation as follows:
Pn(i, j) = exp

−(i −in)2 + (j −jn)2
2σ2

,
(1)
where Pn and (in,jn) are the input heatmap and 2D coor-
dinates of nth keypoint, respectively, and σ is the standard
deviation of the Gaussian peak. The generated input pose is
concatenated with the input image and fed into the PoseFix.
This Gaussian heatmap representation is suitable for sub-
sequent convolutional operations because it is pixel-wise
aligned with the input image. Moreover, as the input pose
can contain some errors, non-zero values around the center
of the blob can be used to encourage the PoseFix to focus
not only on the exact location of the input pose, but also
around it.
From the input Gaussian heatmap in a coarse form, the
proposed network generates the heatmap Hn and the key-
point coordinates Cn for the nth keypoint, sequentially. To
make Hn a ﬁner form, we supervise it using a one-hot vec-
tor.
Then, soft-argmax operation [26] is applied to Hn
to generate Cn in a differentiable manner.
Soft-argmax
is deﬁned as the element-wise product between the input
heatmap and the meshgrid followed by the summation, as
shown in Figure 2. More precisely, the 2D coordinates are
calculated from Hn as follows:
Cn =


w
X
i=1
h
X
j=1
iHn(i, j),
w
X
i=1
h
X
j=1
jHn(i, j)


T
,
(2)
where w and h are the width and height of Hn, respectively.
Our network is trained by minimizing the cross-entropy-
based integral loss [26], which is deﬁned as follows:
L = LH + LC,
(3)
where L is the cross-entropy-based integral loss, and two
losses LH and LC are described below.
The LH is a cross-entropy loss which is calculated after
applying the softmax function to the output heatmap along
the spatial axis. The deﬁnition of the LH is as follows:
LH = −1
N
N
X
n=1
X
i,j
H∗
n(i, j) log Hn(i, j),
(4)
where H∗
n and Hn are the groundtruth and estimated
heatmaps with softmax applied,
respectively.
The
groundtruth heatmap H∗
n is a one-hot vector if the
groundtruth keypoint coordinates are integers. Otherwise,
two grids for each x and y axis are selected by ﬂoor and
ceil operations and are ﬁlled with probabilities by linear ex-
trapolation. The LC is the sum of all L1 losses applied to
the coordinates as follows:
LC = 1
N
N
X
n=1
∥C∗
n −Cn∥1,
(5)
where C∗
n is the groundtruth coordinates vector for nth key-
point. The LH forces the PoseFix to select a single grid
point in the estimated heatmap, and the LC enables the
PoseFix to localize keypoints more precisely because it is
calculated in the continuous space which is free from quan-
tization errors.

5.3. Network architecture
We used network architecture of Xiao et al. [30] which
consists of a deep backbone network (i.e., ResNet [13]) and
several upsampling layers. The ﬁnal upsampling layer be-
comes heatmaps (H) after applying the softmax function.
The soft-argmax operation extracts coordinates (C) from
the heatmaps (H), and it becomes the ﬁnal estimation of
the PoseFix.
6. Implementation details
Training. The proposed PoseFix is trained in an end-to-
end manner. The weights of the backbone part are initial-
ized with the publicly released ResNet model pre-trained on
the ImageNet dataset [25], and the weights of the remaining
part are initialized from the zero-mean Gaussian distribu-
tion with σ = 0.01 and as in He et al. [12]. The weights are
updated by Adam optimizer [17] with a mini-batch size of
128. The initial learning rate is set to 5×10−4 and reduced
by a factor of 10 at 90 and 120th epoch. We perform data
augmentation including scaling (±30%), rotation (±40◦),
and ﬂip. To crop humans from an input image, groundtruth
human bounding boxes are extended to a ﬁxed aspect ratio
(i.e., height:width = 4:3) and then cropped without distort-
ing the aspect ratio. The cropped bounding box is resized
to a ﬁxed size, which becomes the input image. We train
the PoseFix 140 epochs with four NVIDIA 1080 Ti GPUs,
which took two days.
Testing. In the testing stage, the pose estimation result
of other pose estimation methods becomes the input pose.
To crop human bounding box from an image with multiple
persons, we calculate bounding box coordinates from the
keypoints coordinates of the input pose. Following [7,21],
we used testing time ﬂip augmentation.
Our model is implemented using TensorFlow [1] deep
learning framework.
7. Experiment
7.1. Dataset and evaluation metric
The proposed PoseFix is trained and tested on the MS
COCO [19] 2017 keypoint detection dataset, which consists
of training, validation, and test-dev sets. The training set
includes 57K images and 150K person instances. The vali-
dation set and the test-dev sets include 5K and 20K images,
respectively. The OKS-based AP metric is used to evaluate
the accuracy of the keypoint localization.
7.2. Ablation study
To validate each component of the PoseFix, we tested
the PoseFix on the validation set. The backbone of all the
models are ResNet-50, and the size of the input image is
Methods
AP
AP.50 AP.75 APM
APL
E2E-reﬁne
70.1
(+0.4)
87.3
(-1.0)
76.8
(-0.2)
66.8
(+0.6)
76.3
(+0.2)
MA-reﬁne (Ours)
72.1
(+2.4)
88.5
(+0.2)
78.3
(+1.3)
68.6
(+2.4)
78.2
(+2.1)
Table 1: AP comparison between the conventional end-
to-end trainable multi-stage reﬁnement model (E2E-reﬁne)
and the proposed model-agnostic reﬁnement model (MA-
reﬁne) on the validation set. The number in the parenthesis
denotes the AP change from the input pose (i.e., CPN).
F2F
C2F (Ours)
C2F-LC
C2F-LH
C2C
68
68.5
69
69.5
70
70.5
71
71.5
72
72.5
mAP
CPN
Figure 6: mAP comparison of various pipelines. The mAP
is calculated on the validation set.
set to 256×192. We used the CPN [7] which is a state-of-
the-art human pose estimation method to generate the input
poses.
Model-agnostic pose reﬁnement. We compared the ac-
curacy of the conventional end-to-end trainable multi-stage
architecture-based pose reﬁnement model (E2E-reﬁne) and
the proposed model-agnostic reﬁnement model (MA-reﬁne)
in Table 1. To train the E2E-reﬁne, we added a reﬁnement
module which has the same network architecture as the
PoseFix at the end part of the pre-trained CPN. And then,
we ﬁne-tuned it by additionally giving the cross-entropy-
based integral loss to the added module in an end-to-end
manner. Both the input image and the output pose of the
CPN are fed into the reﬁnement module similarly to the
PoseFix. We used a pre-trained CPN instead of training
it from scratch because ﬁne-tuning the pre-trained model
yielded better performance.
As Table 1 shows, the MA-reﬁne trained in a model-
agnostic manner improves the accuracy greatly more than
the conventional reﬁnement model does. We believe that
this is because the added reﬁnement module can be easily
overﬁtted to the output pose of the CPN when training the
E2E-reﬁne. In contrast, various input poses that are realisti-

0.5
0.6
0.7
0.8
0.9
1
OKS of the input pose
0.5
0.6
0.7
0.8
0.9
1
OKS of the refined pose
AE
Mask R-CNN
CPN
Figure 7: OKS change when the PoseFix is applied to state-
of-the-art methods. The dotted line denotes identity func-
tion. OKS is calculated on the validation set.
Figure 8: Frequency of each error type change when the
PoseFix is applied to the CPN. The frequency is calculated
on the validation set.
cally synthesized in the training stage of the PoseFix lead to
the effect of data augmentation, which makes the PoseFix
more robust to unseen input poses in the testing stage.
Instead of using the same network architecture of the
PoseFix like in the E2E-reﬁne, one can design their own
reﬁnement module. However, this approach requires care-
ful network design because the amount of GPU memory
available at one time is limited. By contrast, since the Pose-
Fix is a decoupled model in both of the training and testing
stages, it can serve as an add-on module, and thus provides
more ﬂexibility when building pose estimation models.
This analysis clearly demonstrates the beneﬁts of us-
ing the model-agnostic pose reﬁnement model compared
with the conventional end-to-end trainable multi-stage
architecture-based ones.
Coarse-to-ﬁne estimation. To demonstrate the validity
of the coarse-to-ﬁne estimation, we compared the perfor-
mance of ﬁne-to-ﬁne (i.e., F2F), coarse-to-ﬁne (i.e., C2F,
ours), and coarse-to-coarse (i.e., C2C) estimation pipelines
in Figure 6.
As described in Section 5.2, the Gaussian
heatmap and one-hot vector are used as coarse and ﬁne
forms of the input pose, respectively. To estimate the re-
ﬁned pose in a coarse form, the model learns to estimate the
Gaussian heatmap by minimizing mean square error follow-
ing [7,21,29]. For the ﬁne-form estimation, cross-entropy-
based integral loss is used as a loss function like ours.
As Figure 6 shows, the C2F (i.e., ours) exhibits a more
accurate performance than F2F, which indicates that coarse
input pose representation is more beneﬁcial than ﬁne input
pose representation. Also, C2C fails to improve the input
pose whereas F2F and C2F successfully reﬁne the input
pose. These results indicate that the ﬁne-form estimation is
crucial for a successful reﬁnement.
To further analyze the beneﬁt of the ﬁne-form estima-
tion, we additionally trained two models (C2F-LH and
C2F-LC). Instead of using both of the LC and LH like the
C2F does, they are trained by minimizing only either LH
or LC. The C2F-LH learns to estimate one-hot vector (H)
by minimizing LH, and C2F-LC is supervised to estimate
coordinate (C) by minimizing LC. Among C2C, C2F-LH,
and C2F-LC, the target form of the C2C is the most coarse
representation. On the other hand, that of C2F-LC is the
ﬁnest representation as described in Section 5.2. Figure 6
shows that C2C yields the worst performance while C2F-
LC achieves the best among them. This ﬁnding shows that
as the output representation of the PoseFix becomes a ﬁner
form, the performance improves. Thus, by integrating the
two loss functions (i.e., LH and LC) together, we can im-
prove the performance much, as C2F shows.
This analysis clearly shows the beneﬁt of the coarse-to-
ﬁne estimation pipeline.
7.3. Performance improvement of the state-of-the-
art methods by PoseFix
We report the performance improvement when the Pose-
Fix is applied to the recent state-of-the-art human pose es-
timation methods. PAFs [5], AE [20], Mask R-CNN [11],
CPN [7], and Simple [30] are used to generate the input
pose. To obtain the pose estimation results of the previ-
ous methods, we used their released codes and pre-trained
models. We tested them by ourselves without ensembling
and testing time augmentation. We also trained a pose esti-
mation model (IntegralPose) with the same network archi-
tecture and loss function with the PoseFix to show that the
PoseFix can improve a model trained from the same archi-
tecture. To analyze how the PoseFix changes the OKS and
frequency of each error type, we tested the PoseFix on the
validation set. We also report how much the PoseFix im-
proves AP on the test-dev set. The ResNet-152 is used as

Methods
AP
AP.50
AP.75
APM
APL
AR
AR.50
AR.75
ARM
ARL
AE [20]
56.6
81.7
62.1
48.1
69.4
62.5
84.9
67.2
52.2
76.5
+ PoseFix (Ours)
63.9
83.6
70.0
56.9
73.7
69.1
86.6
74.2
61.1
79.9
PAFs [5]
61.7
84.9
67.4
57.1
68.1
66.5
87.2
71.7
60.5
74.6
+ PoseFix (Ours)
66.7
85.7
72.9
62.9
72.3
71.3
88.0
76.7
66.3
78.1
Mask R-CNN (ResNet-50) [11]
62.9
87.1
68.9
57.6
71.3
69.7
91.3
75.1
63.9
77.6
+ PoseFix (Ours)
67.2
88.0
73.5
62.5
75.1
74.0
92.2
79.6
68.8
81.1
Mask R-CNN (ResNet-101)
63.4
87.5
69.4
57.8
72.0
70.2
91.8
75.6
64.3
78.2
+ PoseFix (Ours)
67.5
88.4
73.8
62.6
75.5
74.3
92.6
79.9
69.1
81.4
Mask R-CNN (ResNeXt-101-64)
64.9
88.6
71.0
59.6
73.3
71.4
92.4
76.8
65.9
78.9
+ PoseFix (Ours)
68.7
89.3
75.2
64.1
76.4
75.2
93.1
80.9
70.3
81.9
Mask R-CNN (ResNeXt-101-32)
64.9
88.4
70.9
59.5
73.2
71.3
92.2
76.7
65.8
78.9
+ PoseFix (Ours)
68.5
88.9
75.0
64.0
76.2
75.0
92.9
80.7
70.1
81.8
IntegralPose
66.3
87.6
72.9
62.7
72.7
73.2
91.8
79.1
68.3
79.8
+ PoseFix (Ours)
69.5
88.3
75.9
65.7
76.1
75.9
92.4
81.8
71.1
82.5
CPN (ResNet-50) [7]
68.6
89.6
76.7
65.3
74.6
75.6
93.7
82.6
70.8
82.0
+ PoseFix (Ours)
71.8
89.8
78.9
68.3
78.1
78.2
93.9
84.3
73.5
84.6
CPN (ResNet-101)
69.6
89.9
77.6
66.3
75.6
76.6
93.9
83.5
72.0
82.9
+ PoseFix (Ours)
72.6
90.2
79.7
69.0
78.9
78.9
94.1
85.0
74.2
85.1
Simple (ResNet-50) [30]
69.4
90.1
77.4
66.2
75.5
75.1
93.9
82.4
70.8
81.0
+ PoseFix (Ours)
72.5
90.5
79.6
68.9
79.0
78.0
94.1
84.4
73.4
84.1
Simple (ResNet-101)
70.5
90.7
78.8
67.5
76.3
76.2
94.3
83.7
72.1
81.9
+ PoseFix (Ours)
73.3
90.8
80.7
69.8
79.8
78.7
94.4
85.3
74.3
84.8
Simple (ResNet-152)
71.1
90.7
79.4
68.0
76.9
76.8
94.4
84.3
72.6
82.4
+ PoseFix (Ours)
73.6
90.8
81.0
70.3
79.8
79.0
94.4
85.7
74.8
84.9
Table 2: Improvement of APs when the PoseFix is applied to the state-of-the-art methods. The APs are calculated on the
test-dev set.
the backbone of the PoseFix, and the size of the input image
is set to 384×288.
OKS change. The graph in Figure 7 shows the change of
the OKS of the same instance when the PoseFix is applied
to the baseline state-of-the-art methods.
Error frequency change. Figure 8 shows how the fre-
quency of each status or error type changes when the Pose-
Fix is applied to the CPN.
AP improvement. Table 2 shows the improvements in
AP when the PoseFix is applied to the recent state-of-the-
art human pose estimation methods. We also included the
results of using different backbone networks [13,31] for the
Mask R-CNN, CPN, and Simple.
As Figures 7, 8 and Table 2 show, the PoseFix con-
sistently improves the performance of the state-of-the-art
methods. The PoseFix corrects not only the small displace-
ment error (i.e., jitter), but also the large displacement errors
(i.e., inversion, miss, and swap) as in Figure 8. Taking into
account the fact that the state-of-the-art methods used in the
experiments vary in structure and learning strategies, we be-
lieve that our model has generalizability that can be applied
to other pose estimation methods. It is also noticeable that
the PoseFix does not require any code or knowledge of the
pose estimation methods, which makes our model very easy
and convenient to use in practice.
8. Conclusion
We proposed a novel and powerful network, PoseFix,
for human pose reﬁnement. Unlike conventional end-to-
end multi-stage architecture models, the proposed PoseFix
is a model-agnostic pose reﬁnement network. To train the
PoseFix, we generate the input pose by synthesizing pose
errors according to empirical pose error distributions on
the groundtruth pose.
The PoseFix takes an input pose
in a coarse form and estimates the reﬁned pose in a ﬁner
form. Since PoseFix is model-agnostic, it does not require
any code or knowledge about the target models. So, it can
be used as a post-processing add-on module conveniently.
We showed that the PoseFix achieves better performance
than the conventional multi-stage architecture-based pose
reﬁnement module. Furthermore, the PoseFix consistently
improves the accuracy of other methods on the commonly
used pose estimation benchmark.

Supplementary Material of “PoseFix:
Model-agnostic General Human Pose
Reﬁnement Network”
In this supplementary material, we present more ex-
perimental results that could not be included in the main
manuscript due to the lack of space.
1. Comparison with conventional end-to-end
trainable multi-stage reﬁnement
In Table 1 of the main manuscript, we compared the ac-
curacy of the conventional end-to-end trainable multi-stage
reﬁnement model (E2E-reﬁne) and the proposed model-
agnostic reﬁnement model (MA-reﬁne). We tried to show
the effectiveness of the proposed model-agnostic reﬁnement
model by making the number of parameters of the E2E-
reﬁne and MA-reﬁne same.
However, as the conventional reﬁnement requires care-
ful model design, simply adding a reﬁnement module which
has the same network architecture with the PoseFix can re-
sult in sub-optimal performance. Therefore, we compare
the accuracy of the reﬁnement module of the state-of-the-art
reﬁnement-based method (i.e., CPN [7]) and the PoseFix.
The CPN consists of two parts. The ﬁrst one, GlobalNet, is
the baseline of the CPN. The second one, ReﬁneNet, reﬁnes
the pose estimation results of the GlobalNet. We use the
GlobalNet as the pose estimation model and compare the
accuracy improvement of the ReﬁneNet and PoseFix. We
trained and tested the CPN with GlobalNet only and both of
the GlobalNet and ReﬁneNet, using their released code.
Table 3 shows our PoseFix improves AP more than state-
of-the-art reﬁnement module (i.e., ReﬁneNet) by a large
margin. This comparison demonstrates the beneﬁt of the
model-agnostic reﬁnement over conventional end-to-end
trainable multi-stage reﬁnement more clearly.
2. Performance improvement of the state-of-
the-art methods by PoseFix
In Figure 8 of the main manuscript, we showed how the
frequency of each error type changes when the PoseFix is
applied to the state-of-the-art method (i.e., CPN [7]). We
additionally show the changes of the AE [20] and Mask R-
CNN [11] in Figure 9 and 10, respectively. As the Figures
show, our PoseFix improves the performance by ﬁxing all
types of pose errors.
We demonstrate more generalizability by showing per-
formance improvement on another 2D multi-person pose
estimation dataset (i.e., PoseTrack 2018 [2]). The Pose-
Track 2018 dataset includes 66K frames, and they are split
into training, validation and testing set. The state-of-the-
art human pose estimation method, Simple [30], is re-
Methods
AP
AP.50 AP.75 APM
APL
ReﬁneNet [7]
69.1
(+1.8)
87.9
(+0.4)
76.6
(+2.2)
65.7
(+1.6)
75.5
(+2.2)
PoseFix (Ours)
71.5
(+4.2)
88.0
(+0.5)
77.6
(+3.2)
68.0
(+3.9)
78.1
(+4.8)
Table 3: AP comparison between state-of-the-art conven-
tional end-to-end trainable multi-stage reﬁnement model
(ReﬁneNet [7]) and the proposed model-agnostic reﬁne-
ment model (PoseFix) on the MS COCO [19] validation set.
The number in the parenthesis denotes the AP change from
the input pose (i.e., GlobalNet of the CPN [7]).
Figure 9: Frequency of each error type change when the
PoseFix is applied to the AE. The frequency is calculated
on the MS COCO [19] validation set.
Figure 10: Frequency of each error type change when the
PoseFix is applied to the Mask R-CNN. The frequency is
calculated on the MS COCO [19] validation set.
implemented by ours 2 and its testing result is used as the in-
put pose of the PoseFix. Both of the Simple [30] and Pose-
Fix is pre-trained on the COCO dataset and trained again
on the PoseTrack 2018 training set without hyperparameter
changes following [30]. Figure 11 and Table 5 show perfor-
mance improvement on the PoseTrack 2018 validation set.
As they show, the PoseFix signiﬁcantly improves the perfor-
2https://github.com/mks0601/TF-SimpleHumanPose

Methods
AP
AP.50
AP.75
APM
APL
AR
AR.50
AR.75
ARM
ARL
RMPE [9]
61.0
82.9
68.8
57.9
66.5
-
-
-
-
-
PAFs [5]
61.8
84.9
67.5
57.1
68.2
66.5
87.2
71.8
60.6
74.6
Mask R-CNN [11]
63.1
87.3
68.7
57.8
71.4
-
-
-
-
-
AE [20]
65.5
86.8
72.3
60.6
72.6
70.2
89.5
76.0
64.6
78.1
Integral [26]
67.8
88.2
74.8
63.9
74.0
-
-
-
-
-
G-RMI [22]
64.9
85.5
71.3
62.3
70.0
69.7
88.7
75.5
64.4
77.1
G-RMI* [22]
68.5
87.1
75.5
65.8
73.3
73.3
90.1
79.5
68.1
80.4
MultiPoseNet [18]
69.6
86.3
76.6
65.0
76.3
73.5
88.1
79.5
68.6
80.3
CFN [14]
72.6
86.1
69.7
78.3
64.1
-
-
-
-
-
CPN [7]
72.1
91.4
80.0
68.7
77.2
78.5
95.1
85.3
74.2
84.3
CPN++ [7]
73.0
91.7
80.9
69.5
78.1
79.0
95.1
85.9
74.8
84.7
Simple [30]
73.7
91.9
81.1
70.3
80.0
79.0
-
-
-
-
Simple [30]
73.3
91.2
80.9
69.8
79.7
78.7
94.8
85.4
74.2
84.8
+ PoseFix (Ours)
74.9
91.2
81.9
71.1
81.2
79.9
94.8
86.3
75.5
86.0
Table 4: Comparison of APs with the state-of-the-art methods on the test-dev set. “*” means that the method involves extra
data for training. “++” indicates results using ensemble.
Figure 11: Frequency of each error type change when the
PoseFix is applied to the Simple. The frequency is calcu-
lated on the PoseTrack 2018 [2] validation set.
mance of the input pose. They show the proposed PoseFix
can improve the input pose on variable datasets.
3. Comparison with state-of-the-art methods
We compare the performance of the PoseFix with state-
of-the-art methods, which include PAFs [5], G-RMI [22],
AE [20], RMPE [9], Mask R-CNN [11], CFN [14],
CPN [7], Integral [26], MultiPoseNet [18], and Simple [30]
on the MS COCO [19] test-dev set. All the performance are
from their papers. We used Simple [30] as the input pose
of the PoseFix. As they did not release the human detec-
tion model and result, we used our human detection model
which achieves 57.2 AP for the human category on the test-
dev set. The Simple [30] wit