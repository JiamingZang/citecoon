# Flowing ConvNets for Human Pose Estimation in Videos

> 2015 · id: W602397586 · arXiv: 1506.02897 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
The objective of this work is human pose estimation in
videos, where multiple frames are available. We investigate
a ConvNet architecture that is able to beneﬁt from tempo-
ral context by combining information across the multiple
frames using optical ﬂow.
To this end we propose a network architecture with the
following novelties: (i) a deeper network than previously in-
vestigated for regressing heatmaps; (ii) spatial fusion lay-
ers that learn an implicit spatial model; (iii) optical ﬂow
is used to align heatmap predictions from neighbouring
frames; and (iv) a ﬁnal parametric pooling layer which
learns to combine the aligned heatmaps into a pooled con-
ﬁdence map.
We show that this architecture outperforms a number of
others, including one that uses optical ﬂow solely at the in-
put layers, one that regresses joint coordinates directly, and
one that predicts heatmaps without spatial fusion.
The new architecture outperforms the state of the art
by a large margin on three video pose estimation datasets,
including the very challenging Poses in the Wild dataset,
and outperforms other deep methods that don’t use a
graphical model on the single-image FLIC benchmark (and
also [5, 35] in the high precision region).

## introduction
Despite a long history of research, human pose estima-
tion in videos remains a very challenging task in computer
vision. Compared to still image pose estimation, the tem-
poral component of videos provides an additional (and im-
portant) cue for recognition, as strong dependencies of pose
positions exist between temporally close video frames.
In this work we propose a new approach for using op-
tical ﬂow for part localisation in deep Convolutional Net-
works (ConvNets), and demonstrate its performance for hu-
man pose estimation in videos. The key insight is that, since
for localisation the prediction targets are positions in the
image space (e.g. (x, y) coordinates of joints), one can use
dense optical ﬂow vectors to warp predicted positions onto
a target image. In particular, we show that when regressing
a heatmap of positions (in our application for human joints),
the heatmaps from neighbouring frames can be warped and
aligned using optical ﬂow, effectively propagating position
conﬁdences temporally, as illustrated in Fig 1.
We also propose a deeper architecture that has additional
convolutional layers beyond the initial heatmaps to enable
learning an implicit spatial model of human layout. These
layers are able to learn dependencies between human body
parts. We show that these ‘spatial fusion’ layers remove
pose estimation failures that are kinematically impossible.
Related work.
Traditional methods for pose estimation
have often used pictorial structure models [2, 8, 10, 27, 39],
which optimise a conﬁguration of parts as a function of lo-
cal image evidence for a part, and a prior for the relative
positions of parts in the human kinematic chain. An alter-
native approach uses poselets [1, 13]. More recent work has
tackled pose estimation holistically: initially with Random
Forests on depth data [12, 29, 31, 34] and RGB [3, 4, 24],
and most recently with convolutional neural networks.
The power of ConvNets has been demonstrated in a wide
variety of vision tasks – object classiﬁcation and detec-
tion [11, 21, 28, 40], face recognition [32], text recogni-
tion [15, 16], video action recognition [20, 30] and many
more [7, 22, 25].
For pose estimation, there were early examples of using
ConvNets for pose comparisons [33]. More recently, [37]
used an AlexNet-like ConvNet to directly regress joint co-
ordinates, with a cascade of ConvNet regressors to improve
accuracy over a single pose regressor network. Chen and
Yuille [5] combine a parts-based model with ConvNets (by
using a ConvNet to learn conditional probabilities for the
presence of parts and their spatial relationship with image
patches). In a series of papers, Tompson, Jain et al. devel-
oped ConvNet architectures to directly regress heatmaps for
each joint, with subsequent layers to add an Markov Ran-
dom Field (MRF)-based spatial model [17, 36], and a pose
reﬁnement model [35] (based on a Siamese network with
shared weights) upon a rougher pose estimator ConvNet.
1
arXiv:1506.02897v2  [cs.CV]  8 Nov 2015

conv1
SpatialNet
Input
...
...
t
t+n
t-n
...
...
conv9
1x1x7
Pose heatmaps
Warped
heatmaps
Optical ﬂow
Pooled
heatmap
for frame t
Output
Temporal
Pooler
...
...
+
+
Spatial
Fusion
Layers
conv2
conv3
conv4
conv5
conv6
conv7
conv8
conv1_f
conv2_f
conv3_f
conv4_f
conv5_f
Loss 1
Loss 2
Figure 1. Deep expert pooling architecture for pose estimation. The network takes as an input all RGB frames within a n-frame
neighbourhood of the current frame t. The fully convolutional network (consisting of a heatmap net with an implicit spatial model)
predicts a conﬁdence heatmap for each body joint in these frames (shown here with a different colour per joint). These heatmaps are then
temporally warped to the current frame t using optical ﬂow. The warped heatmaps (from multiple frames) are then pooled with another
convolutional layer (the temporal pooler), which learns how to weigh the warped heatmaps from nearby frames. The ﬁnal body joints are
selected as the maximum of the pooled heatmap (illustrated here with a skeleton overlaid on top of the person).
Temporal information in videos was initially used with
ConvNets for action recognition [30], where optical ﬂow
was used as an input motion feature to the network. Fol-
lowing this work, [18, 24] investigated the use of temporal
information for pose estimation in a similar manner, by in-
putting ﬂow or RGB from multiple nearby frames into the
network, and predicting joint positions in the current frame.
However, pose estimation differs from action recogni-
tion in a key respect which warrants a different approach to
using optical ﬂow: in action recognition the prediction tar-
get is a class label, whereas in pose estimation the target is
a set of (x, y) positions onto the image. Since the targets
are positions in the image space, one can use dense optical
ﬂow vectors not only as an input feature but also to warp
predicted positions in the image, as done in [4] for random
forest estimators. To this end, our work explicitly predicts
joint positions for all neighbouring frames, and temporally
aligns them to frame t by warping them backwards or for-
wards in time using tracks from dense optical ﬂow. This ef-
fectively reinforces the conﬁdence in frame t with a strong
set of ‘expert opinions’ (with corresponding conﬁdences)
from neighbouring frames, from which joint positions can
be more precisely estimated. Unlike [4] who average the
expert opinions, we learn the expert pooling weights with
backpropagation in an end-to-end ConvNet.
Our ConvNet outperforms the state of the art on three
challenging video pose estimation datasets (BBC Pose,
ChaLearn and Poses in the Wild) – the heatmap regres-
sor alone surpasses the state of the art on these datasets,
and the pooling from neighbouring frames using optical
ﬂow gives a further signiﬁcant boost.
We have released
the models and code at http://www.robots.ox.ac.
uk/˜vgg/software/cnn_heatmap.
2. Temporal Pose Estimation Networks
Fig 1 shows an overview of the ConvNet architecture.
Given a set of input frames within a temporal neighbour-
hood of n frames from a frame t, a spatial ConvNet re-
gresses joint conﬁdence maps (‘heatmaps’) for each input
frame separately.
These heatmaps are then individually
warped to frame t using dense optical ﬂow. The warped
heatmaps (which are effectively ‘expert opinions’ about
joint positions from the past and future) are then pooled
into a single heatmap for each joint, from which the pose
is estimated as the maximum.
We next discuss the architecture of the ConvNets in de-
tail. This is followed by a description of how optical ﬂow is
used to warp and pool the output from the Spatial ConvNet.
2.1. Spatial ConvNet
The network is trained to regress the location of the hu-
man joint positions. However, instead of regressing the joint
(x, y) positions directly [24, 37], we regress a heatmap of
the joint positions, separately for each joint in an input im-
age. This heatmap (the output of last convolutional layer,
conv8) is a ﬁxed-size i × j × k-dimensional cube (here
64 × 64 × 7 for k = 7 upper-body joints). At training
time, the ground truth label are heatmaps synthesised for
each joint separately by placing a Gaussian with ﬁxed vari-
ance at the ground truth joint position (see Fig 2). We then

k joints
Figure 2. Regression target for learning the Spatial ConvNet.
The learning target for the convolutional network is (for each of
k joints) a heatmap with a synthesised Gaussian with a ﬁxed vari-
ance centred at the ground truth joint position. The loss is l2 be-
tween this target and the output of the last convolutional layer.
use an l2 loss, which penalises the squared pixel-wise dif-
ferences between the predicted heatmap and the synthesised
ground truth heatmap.
We denote the training example as (X, y), where y
stands for the coordinates of the k joints in the image X.
Given training data N = {X, y} and the ConvNet regres-
sor φ (output from conv8), the training objective becomes
the task of estimating the network weights λ:
arg min
λ
X
(X,y)∈N
X
i,j,k
∥Gi,j,k(yk) −φi,j,k(X, λ)∥2
(1)
where Gi,j,k(yi) =
1
2πσ2 e−[(y1
k−i)2+(y2
k−j)2]/2σ2 is a Gaus-
sian centred at joint yk with ﬁxed σ.
Discussion.
As noted by [36], regressing coordinates di-
rectly is a highly non-linear and more difﬁcult to learn map-
ping, which we also conﬁrm here (Sect 5). The beneﬁts
of regressing a heatmap rather than (x, y) coordinates are
twofold: ﬁrst, one can understand failures and visualise the
‘thinking process’ of the network (see Figs 3 and 5); second,
since by design, the output of the network can be multi-
modal, i.e. allowed to have conﬁdence at multiple spatial
locations, learning becomes easier: early on in training (as
shown in Fig 3), multiple locations may ﬁre for a given
joint; the incorrect ones are then slowly suppressed as train-
ing proceeds. In contrast, if the output were only the wrist
(x, y) coordinate, the net would only have a lower loss if it
gets its prediction righ

## experiments
We ﬁrst describe the evaluation protocol, then present
comparisons to alternative network architectures, and ﬁ-
nally give a comparison to state of the art. A demo video is
online at https://youtu.be/yRLOid4XEJY.
5.1. Evaluation protocol and details
Evaluation protocol.
In all pose estimation experiments
we compare the estimated joints against frames with manual

ground truth (except ChaLearn, where we compare against
output from Kinect). We present results as graphs that plot
accuracy vs distance from ground truth in pixels, where a
joint is deemed correctly located if it is within a set distance
of d pixels from a marked joint centre in ground truth.
Experimental details.
All frames of the training videos
are used for training (with each frame randomly augmented
as detailed above). The frames are randomly shufﬂed prior
to training to present maximally varying input data to the
network. The hyperparameters (early stopping, variance σ
etc.) are estimated using the validation set.
Baseline method.
As a baseline method we include a Co-
ordinateNet (described in [23]). This is a network with sim-
ilar architecture to [28], but trained for regressing the joint
positions directly (instead of a heatmap) [24].
Computation time.
Our method is real-time (50fps on 1
GPU without optical ﬂow, 5fps with optical ﬂow).
5.2. Component evaluation
For these experiments the SpatialNet and baseline are
trained and tested on the BBC Pose and Extended BBC Pose
datasets. Fig 7 shows the results for wrists
With the SpatialNet, we observe a signiﬁcant boost in
performance (an additional 6.6%, from 79.6% to 86.1% at
d = 6) when training on the larger Extended BBC dataset
compared to the BBC Pose dataset. As noted in Sect 4, this
larger dataset is somewhat noisy. In contrast, the Coordi-
nateNet is unable to make effective use of this additional
noisy training data. We believe this is because its target
(joint coordinates) does not allow for multi-modal output,
which makes learning from noisy annotation challenging.
We observe a further boost in performance from using
optical ﬂow to warp heatmaps from neighbouring frames
(an improvement of 2.6%, from 86.1% to 88.7% at d = 6).
Fig 9 shows the automatically learnt pooling weights. We
see that for this dataset, as expected, the network learns to
weigh frames temporally close to the current frame higher
(because they contain less errors in optical ﬂow).
Fig 8 shows a comparison of different pooling types (for
cross-channel pooling). We compare learning a parametric
pooling function to sum-pooling and to max-pooling (max-
out [14]) across channels. As expected, parametric pool-
ing performs best, and improves as the neighbourhood n
increases. In contrast, results with both sum-pooling and
max-pooling deteriorate as the neighbourhood size is in-
creased further, as they are not able to down-weigh predic-
tions that are further away in time (and thus more prone to
errors in optical ﬂow). As expected, this effect is particu-
larly noticeable for max-pooling.
Failure modes.
The main failure mode for the vanilla
heatmap network (conv1-conv8) occurs when multiple
modes are predicted and the wrong one is selected (and the
0
2
4
6
8
10
12
14
16
18
20
0
10
20
30
40
50
60
70
80
90
100
Accuracy [%]
Distance from GT [px]
Wrists
 
 
CoordinateNet
CoordinateNet Extended
SpatialNet Extended
SpatialNet Flow Extended
SpatialNet
Figure 7. Comparison of the performance of our nets for wrists
on BBC Pose. Plots show accuracy as the allowed distance from
manual ground truth is increased. CoordinateNet is the network
in [23]; SpatialNet is the heatmap network; and SpatialNet Flow
is the heatmap network with the parametric pooling layer. ‘Ex-
tended’ indicates that the network is trained on Extended BBC
Pose instead of BBC Pose. We observe a signiﬁcant gain for the
SpatialNet from using the additional training data in the Extended
BBC dataset (automatically labelled – see Sect 4) training data,
and a further boost from using optical ﬂow information (and se-
lecting the warping weights with the parametric pooling layer).
0
2
4
6
8
10
12
14
70
72
74
76
78
80
82
84
Neighbourhood size (n)
Accuracy [%] at d=5px
Wrists
 
 
Sumpool
Maxout
Parametric pooling
Figure 8. Comparison of pooling types. Results are shown for
wrists in BBC Pose at threshold d = 5px. Parametric pooling
(learnt cross-channel pooling weights) performs best.
resulting poses are often kinematically impossible for a hu-
man to perform). Examples of these failures are shown in
Fig 18. The spatial fusion layers resolve these failures.
5.3. Comparison to state of the art
Training.
We investigated a number of strategies for
training on these datasets including training from scratch
(using only the training data provided with the dataset), or
training on one (i.e. BBC Pose) and ﬁne-tuning on the oth-
ers. We found that provided the ﬁrst and last layers of the
Spatial Net are initialized from (any) trained heatmap net-
work, the rest can be trained either from scratch or ﬁne-
tuned with similar performance. We hypothesise this is be-
cause the datasets are very different – BBC Pose contains

−15
−10
−5
0
5
10
15
0
0.05
0.1
0.15
0.2
0.25
pooling weight
frame number
Figure 9. Learnt pooling weights for BBC Pose with n = 15.
Weights shown for the right wrist. The centre frame receives high-
est weight. The jitter in weights is due to errors in optical ﬂow
computation (caused by the moving background in the video) – the
errors become larger further away from the central frame (hence
low or even negative weights far away).
long-sleeved persons, ChaLearn short-sleeved persons and
Poses in the Wild contains non-frontal poses with unusual
viewing angles. For all the results reported here we train
BBC Pose from scratch, initialize the ﬁrst and last layer
from this, and ﬁne-tune on training data of other datasets.
BBC Pose.
Fig 10 shows a comparison to the state of
the art on the BBC Pose dataset. We compare against all
previous reported results on the dataset.
These include
Buehler et al. [2], whose pose estimator is based on a pic-
torial structure model; Charles et al. (2013) [3] who uses
a Random Forest; Charles et al. (2014) [4] who predict
joints sequentially with a Random Forest; Pﬁster et al.
(2014) [24] who use a deep network similar to our Coor-
dinateNet (with multiple input frames); and the deformable
part-based model of Yang & Ramanan (2013) [39].
We outperform all previous work by a large margin, with
a particularly noticeable gap for wrists (an addition of 10%
compared to the best competing method at d = 6).
Chalearn.
Figs 19 & 12 show a comparison to the state
of the art on ChaLearn. We again outperform the state of
the art even without optical ﬂow (an improvement of 3.5%
at d = 6), and observe a further boost by using optical ﬂow
(beating state of the art by an addition of 5.5% at d = 6),
and a signiﬁcant further improvement from using a deeper
network (an additional 13% at d = 6).
Poses in the Wild.
Figs 13 & 14 show a comparison to
the state of the art on Poses in the Wild. We replicate the
results of the previous state of the art method using code
provided by the authors [6]. We outperform the state of the
art on this dataset by a large margin (an addition of 30% for
wrists and 24% for elbows at d = 8). Using optical ﬂow
yields a signiﬁcant 10% improvement for wrists and 13%
for elbows at d = 8. Fig 16 shows example predictions.
0
2
4
6
8
10
12
14
16
18
20
0
10
20
30
40
50
60
70
80
90
100
Accuracy [%]
Distance from GT [px]
Wrists
 
 
Charles et al. (2013)
Charles et al. (2014)
Yang & Ramanan (2013)
SpatialNet Flow
SpatialNet Fusion Flow
SpatialNet
Figure 11. Comparison to the state of the art on ChaLearn. Our
method outperforms state of the art by a large margin (an addition
of 19% at d = 4).
0
5
10
15
20
0
10
20
30
40
50
60
70
80
90
100
Accuracy [%]
Distance from GT [px]
Elbows
 
 
Charles et al. (2013)
Charles et al. (2014)
Yang & Ramanan (2013)
SpatialNet Flow
SpatialNet Fusion Flow
SpatialNet
0
5
10
15
20
0
10
20
30
40
50
60
70
80
90
100
Accuracy [%]
Distance from GT [px]
Shoulders
Figure 12. ChaLearn: elbows & shoulders.
0
2
4
6
8
10
12
14
16
18
20
0
10
20
30
40
50
60
70
80
Accuracy [%]
Distance from GT [px]
Wrists
 
 
Cherian et al. (2014)
Yang & Ramanan (2013)
SpatialNet Fusion Flow
SpatialNet Fusion
SpatialNet
Figure 13. Comparison to state of the art on Poses in the Wild.
Our method outperforms state of the art by a large margin (an ad-
dition of 30% at d = 8, with 10% from ﬂow).
FLIC.
Fig 15 shows a comparison to the state of the art
on FLIC. We outperform all pose estimation methods that
don’t use a graphical model, and match or even slightly out-
perform graphical model-based methods [5, 35] in the very

0
5
10
15
20
0
10
20
30
40
50
60
70
80
90
100
Accuracy [%]
Distance from GT [px]
Head
 
 
Buehler et al. (2011)
Charles et al. (2013)
Charles et al. (2014)
Pfister et al. (2014)
Yang & Ramanan (2013)
Ours
0
5
10
15
20
0
10
20
30
40
50
60
70
80
90
100
Accu

## conclusion
We have presented a new architecture for pose estimation
in videos that is able to utilizes appearances across multiple
frames. The proposed ConvNet is a simple, direct method
for regressing heatmaps, and its performance is improved
by combining it with optical ﬂow and spatial fusion lay-
ers. We have also shown that our method outperforms the
state of the art on three large video pose estimation datasets.
Further improvements may be obtained by using additional
inputs for the spatial ConvNet, for example multiple RGB
0
0.025
0.05
0.075
0.1
0.125
0.15
0.175
0.2
0
10
20
30
40
50
60
70
80
90
100
Normalised distance from GT
Accuracy [%]
Average PCK for wrist & elbow
 
 
Toshev et al.
Jain et al.
MODEC
Yang et al.
Sapp et al.
Tompson et al. *
Chen et al. *
Ours
Figure 15. Comparison to state of the art on FLIC. Solid lines
represent deep models; methods with a square (■) are without a
graphical model; methods with an asterisk (*) are with a graphical
model. Our method outperforms competing methods without a
graphical model by a large margin in the high precision area (an
addition of 20% at d = 0.05).
frames [24] or optical ﬂow [18] – although prior work has
shown little beneﬁt from this so far.
The beneﬁts of aligning pose estimates from multiple
frames using optical ﬂow, as presented here, are comple-
mentary to architectures that explicitly add spatial MRF and
reﬁnement layers [35, 36].
Finally, we have demonstrated the architecture for hu-
man pose estimation, but a similar optical ﬂow-mediated
combination of information could be used for other tasks in
video, including classiﬁcation and segmentation.
Acknowledgements:
Financial support was provided by
Osk. Huttunen Foundation and EPSRC grant EP/I012001/1.

Figure 16. Example predictions on a variety of videos in Poses in the Wild.
Figure 17. Example predictions on two videos in Poses in the Wild. Predictions and the corresponding heatmaps are shown.

Figure 18. Failures cases. As shown, failure cases contain multiple modes for the same joint in the heatmap (and the wrong mode has been
selected). Adding spatial fusion layers (an implicit spatial model) resolves these failures.

Figure 19. Example predictions on ChaLearn.

Figure 20. Example predictions on BBC Pose.