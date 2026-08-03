# 2D/3D Pose Estimation and Action Recognition Using Multitask Deep Learning

> 2018 · id: W2788865504 · arXiv: 1802.09232 · pdf: https://arxiv.org/pdf/1802.09232 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Action recognition and human pose estimation are
closely related but both problems are generally handled
as distinct tasks in the literature.
In this work, we pro-
pose a multitask framework for jointly 2D and 3D pose
estimation from still images and human action recogni-
tion from video sequences. We show that a single archi-
tecture can be used to solve the two problems in an efﬁ-
cient way and still achieves state-of-the-art results.
Ad-
ditionally, we demonstrate that optimization from end-to-
end leads to signiﬁcantly higher accuracy than separated
learning. The proposed architecture can be trained with
data from different categories simultaneously in a seam-
lessly way. The reported results on four datasets (MPII,
Human3.6M, Penn Action and NTU) demonstrate the effec-
tiveness of our method on the targeted tasks.

## introduction
Human action recognition and pose estimation have re-
ceived an important attention in the last years, not only be-
cause of their many applications, such as video surveillance
and human-computer interfaces, but also because they are
still challenging tasks. Pose estimation and action recog-
nition are usually handled as distinct problems [14] or the
last is used as a prior for the ﬁrst [57, 22]. Despite the fact
that pose is of extreme relevance for action recognition, to
the best of our knowledge, there is no method in the litera-
ture that solves both problems in a joint way to the beneﬁt
of action recognition. In that direction, our work proposes
unique end-to-end trainable multitask framework to handle
2D and 3D human pose estimation and action recognition
jointly, as presented in Figure 1.
One of the major advantages of deep learning is its capa-
bility to perform end-to-end optimization. As suggested by
Kokkinos [24], this is all the more true for multitask prob-
lems, where related tasks can beneﬁt from one another. Re-
cent methods based on deep convolutional neural networks
(CNNs) have achieved impressive results on both 2D and
Multitask CNN
 
t=1
t=T
...
RGB images
t=0
Aggregation
Action: "Baseball pitch"
Appearence
recognition
×
Pose
recognition
2D/3D estimated poses
...
Visual
features
...
t=0 t=1
t=T
Probability
maps
...
t=0 t=1
t=T
Figure 1.
The proposed multitask approach for pose estimation
and action recognition. Our method provides 2D/3D pose esti-
mation from single images or frame sequences. Pose and visual
information are used to predict actions in a uniﬁed framework.
3D pose estimation tasks thanks to the rise of new architec-
tures and the availability of large amounts of data [33, 35].
Similarly, action recognition has recently been improved by
using deep neural networks relying on human pose [3]. We
believe both tasks have not yet been stitched together to
perform a beneﬁcial joint optimization because most pose
estimation methods perform heat map prediction. These
detection based approaches require the non-differentiable
argmax function to recover the joint coordinates as a post
processing stage, which breaks the backpropagation chain
needed for end-to-end learning.
We propose to solve this problem by extending the dif-
ferentiable Soft-argmax [28, 58] for joint 2D and 3D pose
estimation. This allows us to stack action recognition on top
of pose estimation, resulting in a multitask framework train-
able from end-to-end. We present our contributions as fol-
lows: First, the proposed pose estimation method achieves
1
arXiv:1802.09232v2  [cs.CV]  21 Mar 2018

state-of-the-art results on 3D pose estimation and the most
accurate results among regression methods for 2D pose es-
timation. Second, the proposed pose estimation method is
based on still images, so it beneﬁts from images “in the
wild” for both 2D and 3D predictions.
This have been
proven a very efﬁcient way to learn visual features, which
is also very important for action recognition. Third, our
action recognition approach is based only on RGB images,
from which we extract pose and visual information. De-
spite that, we reached state-of-the-art results on both 2D
and 3D scenarios, even when compared with methods using
ground-truth poses. Fourth, the pose estimation method
can be trained with multiple types of datasets simultane-
ously, which makes it able to generalize 3D predictions
from 2D annotated data.
The rest of this paper is organized as follows. In sec-
tion 2 we present a review of the related work. The proposed
framework is presented in sections 3 and 4, respectively for
the regression method for pose estimation and human ac-
tion recognition. Our extensive experiments are shown in
section 5, followed by our conclusions in section 6.

## method
Year
PCKh
@0.2
AUC
@0.2
PCKh
@0.5
AUC
@0.5
Detection methods
Recurrent VGG [6]
2016
61.6
28.2
88.1
58.8
DeeperCut [20]
2016
64.0
31.7
88.5
60.8
Pose Machines [54]
2016
64.8
33.0
88.5
61.4
Heatmap regression [7]
2016
61.8
28.5
89.7
59.6
Stacked Hourglass [33]
2016
66.5
33.4
90.9
62.9
Fractal NN [34]
2017
–
–
91.2
63.6
Multi-Context Att. [16]
2017
67.8
34.1
91.5
63.8
Self Adversarial [15]
2017
68.0
34.0
91.8
63.9
Adversarial PoseNet[12]
2017
–
–
91.9
61.6
Pyramid Res. Module[56]
2017
–
–
92.0
64.2
Regression methods
LCR-Net [42]
2017
–
–
74.2
–
Iter. Error Feedback [9]
2016
46.8
20.6
81.3
49.1
Compositional Reg.[47]
2017
–
–
86.4
–
2D Soft-argmax
67.7
34.9
91.2
63.9
mean per joint position error (MPJPE), which is the most
challenging and the most common metric for this dataset.
We followed the common evaluation protocol [47, 35, 31,
11] by taking ﬁve subjects for training (S1, S5, S6, S7,
S8) and evaluating on two subjects (S9, S11) on one ev-
ery 64 frames. For training, we use the data equally bal-
anced as 50%/50% from MPII and Human3.6M. For the
multi-crop predictions we use ﬁve cropped regions and their
corresponding ﬂipped images. Our results compared to the
previous approaches are presented in Table 1 and show that
our approach is able to outperform the state of the art by a
fair margin. Qualitative results from our method are shown
in Figure 7, for both Human3.6M and MPII datasets, which
also demonstrate the capability of our method to general-
ize 3D pose predictions from data with only 2D annotated
poses.
5.4. Evaluation on action recognition
2D action recognition. We evaluate our action recogni-
tion approach on 2D scenario on the Penn Action dataset.
For training the pose estimation part, we use mixed data
from MPII (75%) and Penn Action (25%), using 16 body
joints. The action recognition part was trained using video
clips composed of T = 16 frames. We reached state of the
art classiﬁcation score among methods using RGB and esti-
mated poses. We also evaluated our method without consid-
ering the inﬂuence of estimated poses by using the manually
annotated body joints and are also able to improve over the
state of the art. Results are shown in Table 3.
3D action recognition. Since skeletal data from NTU
is frequently noisy, we train the pose estimation part with
only 10% of data from NTU, 45% from MPII, and 45%
from Human3.6M, using 20 body joints and video clips of
T = 20 frames. Our method improves the state of the art on
NTU signiﬁcantly using only RGB frames and 3D predicted
poses, as reported in Table 4. If we consider only RGB
frames as input, our method improves over [3] by 9.9%.
To the best of our knowledge, all the previous methods use
7

Table 3. Comparison results on Penn Action for 2D action recogni-
tion. Results given as the percentage of correctly classiﬁed actions.

## experiments
In this section we present the experimental evaluation of
our method in four different categories using four challeng-
ing datasets. We show the robustness and the ﬂexibility of
our proposed multitask approach. The four categories are
divided into two problems: human pose estimation and ac-
tion recognition. For both cases, we evaluate our approach
on 2D and 3D scenarios.
5.1. Datasets
We evaluate our method on four different datasets: on
MPII [1] and on Human3.6M [21] for respectively 2D and
3D pose estimation, and on Penn Action [59] and NTU
RGB+D [44] for 2D and 3D action recognition, respec-
tively. The characteristics of each dataset are given as fol-
lows.
MPII Human Pose Dataset. The MPII dataset for single
person pose estimation is composed of about 25K images of
which 15K are training samples, 3K are validation samples
and 7K are testing samples (which labels are withheld by
the authors). The images are taken from YouTube videos
covering 410 different human activities and the poses are
manually annotated with up to 16 body joints.
Human3.6M. The Human3.6M [21] dataset is composed
by videos with 11 subjects performing 17 different activities
and 4 cameras with different points of view, resulting in
more than 3M frames. For each person, the dataset provides
32 body joints, from which only 17 are used to compute
scores.
Penn Action . The Penn Action dataset [59] is composed
by 2,326 videos in the wild with 15 different actions, among
those “baseball pitch”, “bench press”, “strum guitar”, etc.
The challenge on this dataset is that several body parts are
missing in many actions and the image scales are very dis-
parate from one sample to another.
NTU RGB+D. The NTU dataset is so far the biggest and
a very challenging datasets for 3D action recognition. It
is composed of more than 56K videos in Full HD of 60
actions performed by 40 different actors and recorded by 3
cameras in 17 different positioning setups, which results in
more than 4M video frames.
5.2. Implementation details
For the pose estimation task, we train the network using
the elastic net loss function on predicted poses as deﬁned in
the equation bellow:
Lp =
1
NJ
NJ
X
n=1
 ∥ˆpn −pn∥1 + ∥ˆpn −pn∥2
2

,
(3)
where ˆpn and pn are respectively the estimated and the
ground truth positions of the nth joint. For training, we
crop bounding boxes centered on the target person by using
the ground truth annotations or the persons location, when
applicable. For the pose estimation task, on both MPII sin-
gle person and Human3.6M datasets it is allowed to use the
given persons location on evaluation. If a given body joint
falls outside the cropped bounding box on training, we set
the ground truth visibility ﬂag to zero, otherwise we set it
to one. The ground truth visibility information is used to
supervise the predicted joint visibility vector v with the bi-
nary cross entropy loss. When evaluating the pose estima-
tion task we show the results for single-crop and multi-crop.
In the ﬁrst case, one centered image is used for prediction,
and on the second case, multiple images are cropped with
small displacements and horizontal ﬂips and the ﬁnal pose
is the average prediction.
For the action recognition task, we train the network us-
ing the categorical cross entropy loss. On training, we ran-
domly select ﬁxed-size clips with T frames from a video
sample. On test, we report results on single-clip or multi-
clip. In the ﬁrst case, we crop a single clip in the middle
of the video. For the second case, we crop multiple clips
temporally spaced of T/2 frames from each other. The ﬁ-
nal scores on multi-clip is computed by the average result
on all clips from one video. To estimate the bounding box
on test, we do an initial pose prediction using the full im-
ages from the ﬁrst, middle, and last frames of a clip. Fi-
nally, we select the maximum bounding box that encloses
6

Table 1. Comparison with previous work on Human3.6M evaluated on the averaged joint error (in millimeters) on reconstructed poses.

## related_work
In this section, we present some of the most relevant
methods to our work, which are divided into human pose
estimation and action recognition. Since an extensive lit-
erature review is prohibitive here due to the limited size
of the paper, we encourage the readers to refer to the sur-
veys in [43, 19] for respectively pose estimation and action
recognition.
2.1. Human pose estimation
2D pose estimation. The problem of human pose esti-
mation has been intensively studied in the last years, from
Pictorial Structures [2, 17, 37] to more recent CNN ap-
proaches [34, 25, 38, 20, 41, 54, 5, 51, 52, 36]. From the
literature, we can see that there are two distinct families of
methods for pose estimation: detection based and regres-
sion based methods. Detection based methods handle pose
estimation as a heat map prediction problem, where each
pixel in a heat map represents the detection score of a cor-
responding joint [7, 18]. Exploring the concepts of stacked
architectures, residual connections, and multiscale process-
ing, Newell et al. [33] proposed the Stacked Hourglass Net-
work, which improved scores on 2D pose estimation chal-
lenges signiﬁcantly. Since then, methods in the state of the
art are proposing complex variations of the Stacked Hour-
glass architecture. For example, Chu et al. [16] proposed an
attention model based on conditional random ﬁeld (CRF)
and Yang et al. [56] replaced the residual unit by a Pyramid
Residual Module (PRM). Generative Adversarial Networks
(GANs) have been used to improve the capacity of learning
structural information [13] as well as to reﬁne the heat maps
by learning more plausible predictions [15],
However, detection approaches do not provide joint co-
ordinates directly. To recover the pose in (x, y) coordinates,
the argmax function is usually applied as a post-processing
step. On the other hand, regression based approaches use
a nonlinear function that maps the input directly to the de-
sired output, which can be the joint coordinates. Follow-
ing this paradigm, Toshev and Szegedy [52] proposed a
holistic solution based on cascade regression for body part
detection and Carreira et al. [9] proposed the Iterative Er-
ror Feedback. The limitation of regression methods is that
the regression function is frequently sub-optimal. In order
to tackle this weakness, the Soft-argmax function [28] has
been proposed to convert heat maps directly to joint coordi-
nates and consequently allow detection methods to be trans-
formed into regression methods. The main advantage of re-
gression methods over detection ones is that they often are
fully differentiable. This means that the output of the pose
estimation can be used in further processing and the whole
system can be ﬁne-tuned.
3D pose estimation. Recently, deep architectures have
been used to learn precise 3D representations from RGB
images [60, 50, 30, 49, 31, 39], thanks to the availability of
high quality data [21], and are now able to surpass depth-
sensors [32]. Chen and Ramanan [11] divided the problem
of 3D pose estimation into two parts. First, they handle the
2D pose estimation considering the camera coordinates and
second, the estimated poses are matched to 3D representa-
tions by means of a nonparametric shape model. A bone
representation of the human pose was proposed to reduce
the data variance [47], however, such a structural transfor-
mation might effect negatively tasks that depend on the ex-
tremities of the human body, since the error is accumulated
as we go away from the root joint. Pavlakos et al. [35] pro-
posed the volumetric stacked hourglass architecture. How-
ever, the method suffers from the signiﬁcant increase in the
number of parameters and in the required memory to store
all the gradients. In our approach, we also propose an in-
termediate volumetric representation for 3D poses, but we
use a much lower resolution than in [35] and still are able to
increase signiﬁcantly the state-of-the-art results, since our
method is based on a continuous regression function.
2.2. Action recognition
2D action recognition. Action recognition from videos
is considered a difﬁcult problem because it involves high
level abstraction, and furthermore the temporal dimension
is not easily handled. Previous approaches have explored
classical methods for features extraction [55, 23], where the
key idea is to use body joint locations to select visual fea-
tures in space and time. 3D convolutions have been stated
recently as the option that gives the highest classiﬁcation
2

scores [8, 10, 53], but they involve high number of parame-
ters, require an elevated amount of memory for training, and
cannot efﬁciently beneﬁt from the abundant still images for
training. Action recognition is improved by attention mod-
els that focus on body parts [4] and two-stream networks
can be used to merge both RGB images and the costly opti-
cal ﬂow maps [14].
Most 2D action recognition methods use the body joint
information only to extract localized visual features, as an
attention mechanism. The few methods that directly explore
the body joints do not generate it, therefore they are lim-
ited to datasets that provide skeletal data. Our approach re-
moves these limitations by performing pose estimation to-
gether with action recognition. As such, our model only
needs the input RGB frames while still performing discrim-
inative visual recognition guided by estimated body joints.
3D action recognition. Differently from video based ac-
tion recognition, 3D action recognition is mostly based on
skeleton data as the primary information [29, 40]. With re-
cently available depth sensors such as the Microsoft Kinect,
it is possible to capture 3D skeletal data without a complex
installation procedure frequently required for motion cap-
ture systems (MoCap). However, due to the use of infrared
projectors, these depth sensors are limited to indoor envi-
ronments. Moreover, they have a low range precision and
are not robust to occlusions, frequently resulting in noisy
skeletons.
To cope with noisy skeletons, Spatio-Temporal LSTM
networks have been widely used by applying a gating mech-
anism [26] to learn the reliability of skeleton sequences or
by using attention mechanisms [27, 46]. In addition to the
skeleton data, multimodal approaches can also beneﬁt from
the visual cues [45]. In that direction, Baradel et al. [3]
proposed the Pose-conditioned Spatio-Temporal attention
mechanism by using the skeleton sequences for both spatial
and temporal attention mechanisms, while action classiﬁ-
cation is based on pose and appearance features extracted
from patches on the hands.
Since our architecture predicts high precision 3D skele-
ton from the input RGB frames, we do not have to cope with
the noisy skeletons from Kinect. Moreover, we show in the
experiments that, despite being based on temporal convolu-
tion instead of the more common LSTM, our system is able
to reach state of the art performance on 3D action recogni-
tion.
3. Human pose estimation
Our approach for human pose estimation is a regression
method, similarly to [28, 47, 9]. We extended the Soft-
argmax function to handle 2D and 3D pose regression in
a uniﬁed way. The details of our approach are explained as
follows.
3.1. Regression-based approach
The human pose regression problem is deﬁned by the
input RGB image I ∈RW ×H×3, the output estimated pose
ˆp ∈RNJ×D with NJ body joints of dimension D, and a
regression function fr, as given by the following equation:
ˆp = fr(I, θr),
(1)
where θr is a set of trainable parameters of function fr. The
objective is to optimize the parameters θr in order to mini-
mize the error between the estimated pose ˆp and the ground
truth pose p. In order to implement this function, we use
a deep CNN. As the pose estimation is the ﬁrst part of our
multitask approach, the function fr has to be differentiable
in order to allow end-to-end optimization. This is made pos-
sible by the Soft-argmax, which is a differentiable alterna-
tive to the argmax function and can be used to convert heat
maps to (x, y) joint coordinates proposed in [28].
3.1.1
Network architecture
The network architecture has its entry ﬂow based on
Inception-V4 [48] that is used to provide basic features ex-
traction. Then, similarly to what is found in [28], K predic-
tion blocks are used to reﬁne estimations, from which we
use the last prediction p′
K as our estimated pose ˆp. Each
prediction block is composed of eight residual depth-wise
convolutions separated into three different resolutions. As
a byproduct, we also have access to low-level visual fea-
tures and to the intermediate joint probability maps that are
indirectly learned thanks to the Soft-argmax layer. In our
method for action recognition, both visual features and joint
probability maps are used to produce appearance features,
as detailed in section 4.2. A graphical representation of the
pose regression network is shown in Figure 2.
Input image
Prediction
block K
Incep

## conclusion
In this paper, we presented a multitask deep architecture
to perform 2D and 3D pose estimation jointly with action
recognition. Our model ﬁrst predicts the 2D and 3D loca-
tion of body joints from the raw RGB frames. These loca-
tions are then used to predict the action performed in the
video in two different ways: using semantic information
by leveraging the temporal evolution of body joint coor-
dinates and using visual information by performing an at-
tention based pooling on human body parts. Heavy shar-
ing of weights and features in our model allows us to solve
four different tasks - 2D pose estimation, 3D pose estima-
tion, 2D action recognition, 3D action recognition - with
a single model very efﬁciently compared to dedicated ap-
proaches. We performed extensive experiments that show
our approach is able to equal or even outperform dedicated
approaches on all these tasks.
7. Acknowledgements
This work was partially founded by CNPq (Brazil) -
Grant 233342/2014-1.
8

Appendix A: Network architecture
In our implementation of the proposed approach, we di-
vided the network architecture into four parts: the mul-
titask stem, the pose estimation model, the pose recogni-
tion model, and the appearance recognition model.
We
use depth-wise separable convolutions as depicted in Fig-
ure 10, batch normalization and ReLu activation. The archi-
tecture of the multitask stem is detailed in Figure 11. Each
pose estimation prediction block is implemented as a multi-
resolution CNN, as presented in Figure 12. We use Nd = 16
heat maps for depth predictions. The CNN architecture for
action recognition is detailed in Figure 13.
+
Output: W×H×Nfout
Input: W×H×Nﬁn
SC S×S, Nfout
C 1×1, Nfout
+
Output: W×H×Nfout
Input: W×H×Nﬁn
SC S×S, Nfout
Residual
connection
Figure 10. Separable residual module (SR) based on depth-wise
separable convolutions (SC) for Nfin̸ = Nfout (left), and Nfin =
Nfout (right), where Nfin and Nfout are the input and output
features size, W × H is the feature map resolution, and S × S
is the size of the ﬁlters, usually 3 × 3 or 5 × 5. C: Simple 2D
convolution.
C 3×3, 32
str. 2×2
C 3×3, 64
C 3×3, 32
C 3×3, 96
str. 2×2
MaxPooling
3×3 str. 2×2
Input: 256×256×3
Concat
C 1×1, 64
C 3×3, 96
C 1×1, 64
C 5×1, 64
C 1×5, 64
C 3×3, 96
Concat
C 3×3, 192
str. 2×2
MaxPooling
2×2 str. 2×2
Concat
SR 3×3, 576
Output: 32×32×576
Figure 11.
Shared network (entry ﬂow) based on Inception-V4.
C: Convolution, SR: Separable residual module.
Input: 32×32×576
Output: 32×32×576
+
+
SR 5×5, 576
MaxPooling
2×2 str. 2×2
C 1×1, 288
SR 5×5, 288
SR 5×5, 288
SR 5×5, 576
MaxPooling
2×2 str. 2×2
SR 5×5, 288
SR 5×5, 288
SR 5×5, 288
UpSampling
2×2
UpSampling
2×2
SC 5×5, 576
C 1×1, Nd*NJ
C 1×1, 576
+
+
Volumetric
heat maps
Soft-argmax
+
2D/3D
pose loss
Figure 12. Prediction block for pose estimation, where Nd is the
number of depth heat maps per joint and NJ is the number of body
joints. C: Convolution, SR: Separable residual module.
Appendix B: Training parameters
In order to merge different datasets, we convert the poses
to a common layout, with a ﬁxed number of joints equal to
the dataset with more joints. For example, when merging
the datasets Human3.6M and MPII, we use all the 17 joints
in the ﬁrst dataset and include one joint on MPII. All the
included joints have an invalid value that is not taken into
account in the loss function. Additionally, we use and al-
ternated human pose layout, similar to the layout from the
Penn Action dataset, which experimentally lead to better
scores on action recognition.
We optimize the pose regression part using the RMSprop
optimizer with initial learning rate of 0.001, which is re-
duced by a factor of 0.2 when validation score plateaus, and
batches of 24 images. For the action recognition task, we
train both pose and appearance models simultaneously us-
ing a pre-trained pose estimation model with weights ini-
tially frozen. In that case, we use a classical SGD optimizer
with Nesterov momentum of 0.98 and initial learning rate of
0.0002, reduced by a factor of 0.2 when validation plateaus,
and batches of 2 video clips. When validation accuracy
stagnates, we divide the ﬁnal learning rate by 10 and ﬁne
tune the full network for more 5 epochs. When reporting
9

Table 6. Our results on averaged joint error on reconstructed poses for 3D pose estimation on Human3.6 considering single dataset training
(Human3.6M only) and mixed data (Human3.6M + MPII). SC: Single-crop, MC: Multi-crop.