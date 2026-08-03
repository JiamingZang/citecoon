# 3D Human Pose Estimation in Video With Temporal Convolutions and Semi-Supervised Training

> 2019 · id: W2903549000 · arXiv: 1811.11742 · pdf: https://arxiv.org/pdf/1811.11742 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
In this work, we demonstrate that 3D poses in video
can be effectively estimated with a fully convolutional
model based on dilated temporal convolutions over 2D key-
points. We also introduce back-projection, a simple and
effective semi-supervised training method that leverages
unlabeled video data.
We start with predicted 2D key-
points for unlabeled video, then estimate 3D poses and
ﬁnally back-project to the input 2D keypoints.
In the
supervised setting, our fully-convolutional model outper-
forms the previous best result from the literature by 6 mm
mean per-joint position error on Human3.6M, correspond-
ing to an error reduction of 11%, and the model also
shows signiﬁcant improvements on HumanEva-I. More-
over, experiments with back-projection show that it comfort-
ably outperforms previous state-of-the-art results in semi-
supervised settings where labeled data is scarce.
Code
and models are available at https://github.com/
facebookresearch/VideoPose3D

## introduction
Our work focuses on 3D human pose estimation in video.
We build on the approach of state-of-the-art methods which
formulate the problem as 2D keypoint detection followed by
3D pose estimation [41, 52, 34, 50, 10, 40, 56, 33]. While
splitting up the problem arguably reduces the difﬁculty of
the task, it is inherently ambiguous as multiple 3D poses
can map to the same 2D keypoints. Previous work tack-
led this ambiguity by modeling temporal information with
recurrent neural networks [16, 27]. On the other hand, con-
volutional networks have been very successful in modeling
temporal information in tasks that were traditionally tack-
led with RNNs, such as neural machine translation [11],
language modeling [7], speech generation [55], and speech
recognition [6]. Convolutional models enable parallel pro-
cessing of multiple frames which is not possible with recur-
rent networks.
∗Work done while at Facebook AI Research.
Figure 1: Our temporal convolutional model takes 2D key-
point sequences (bottom) as input and generates 3D pose
estimates as output (top). We employ dilated temporal con-
volutions to capture long-term information.
In this paper, we present a fully convolutional architec-
ture that performs temporal convolutions over 2D keypoints
for accurate 3D pose prediction in video (see Figure 1). Our
approach is compatible with any 2D keypoint detector and
can effectively handle large contexts via dilated convolu-
tions. Compared to approaches relying on RNNs [16, 27],
it provides higher accuracy, simplicity, as well as efﬁciency,
both in terms of computational complexity as well as the
number of parameters (§3).
Equipped with a highly accurate and efﬁcient architec-
ture, we turn to settings where labeled training data is scarce
and introduce a new scheme to leverage unlabeled video
data for semi-supervised training. Low resource settings are
particularly challenging for neural network models which
require large amounts of labeled training data and collect-
ing labels for 3D human pose estimation requires an ex-
pensive motion capture setup as well as lengthy recording
sessions. Our method is inspired by cycle consistency in
unsupervised machine translation, where round-trip transla-
tion into an intermediate language and back into the original
language should be close to the identity function [46, 26, 9].
Speciﬁcally, we predict 2D keypoints for an unlabeled video
with an off the shelf 2D keypoint detector, predict 3D poses,
and then map these back to 2D space (§4).
arXiv:1811.11742v2  [cs.CV]  29 Mar 2019

In summary, this paper provides two main contributions.
First, we present a simple and efﬁcient approach for 3D
human pose estimation in video based on dilated temporal
convolutions on 2D keypoint trajectories. We show that our
model is more efﬁcient than RNN-based models at the same
level of accuracy, both in terms of computational complex-
ity and the number of model parameters.
Second, we introduce a semi-supervised approach which
exploits unlabeled video, and is effective when labeled
data is scarce.
Compared to previous semi-supervised
approaches, we only require camera intrinsic parameters
rather than ground-truth 2D annotations or multi-view im-
agery with extrinsic camera parameters.
In comparison to the state of the art our approach out-
performs the previously best performing methods in both
supervised and semi-supervised settings. Our supervised
model performs better than other models even if these ex-
ploit extra labeled data for training.

## method
Parameters
≈FLOPs
MPJPE
Hossain & Little [16]
16.96M
33.88M
41.6
Ours 27f w/o dilation
29.53M
59.03M
41.1
Ours 27f
8.56M
17.09M
40.6
Ours 81f
12.75M
25.48M
38.7
Ours 243f
16.95M
33.87M
37.8
Table 5: Computational complexity of various models un-
der Protocol 1 trained on ground-truth 2D poses. Results
are without test-time augmentation.
point operations (FLOPs) to predict one frame at inference
time (details in Appendix A.2). For the latter, we only con-
sider matrix multiplications and report the amortized cost
over a hypothetical sequence of inﬁnite length (to disregard
padding). MPJPE results are based on models trained on
ground-truth 2D poses without test-time augmentation. Our
model achieves a signiﬁcantly lower error even when the
number of computations are halved. Our largest model with
receptive ﬁeld of 243 frames has roughly the same com-
plexity as [16], but at 3.8 mm lower error. The table also
highlights the effectiveness of dilated convolutions which
increase complexity only logarithmically with respect to the
receptive ﬁeld.
Since our model is convolutional, it can be parallelized
both over the number of sequences as well as over the tem-
poral dimension. This contrasts to RNNs, which can only be
parallelized over different sequences and are thus much less
efﬁcient for small batch sizes. For inference, we measured
about 150k FPS on a single NVIDIA GP100 GPU over a
single long sequence, i.e., batch size one, assuming that 2D
poses were already available. Speed is largely independent
of the batch size due to parallel temporal processing.

.1% S1
49
1% S1
496
5% S1
2.48k
10% S1
4.97k
50% S1
24.8k
S1
49.7k
S15
129k
S156
179k
All
312k
Training data (downsampled to 10 FPS)
45
55
65
75
85
95
105
115
125
135
145
155
165
175
185
N-MPJPE (mm)
131.4
101.3
93.4
86.8
69.9
64.4
55.3
52.9
47.1
121
90.9
84.4
77.5
65.8
61.8
55.3
53.8
166.5
122.6
Rhodin supervised
Rhodin semi-supervised
Ours supervised
Ours semi-supervised
(a) Downsampled to 10 FPS under Protocol 3.
.1% S1
245
1% S1
2.42k
5% S1
12.4k
10% S1
24.8k
50% S1
124k
S1
248k
S15
645k
S156
895k
All
1.56M
Training data
35
45
55
65
75
85
95
105
115
125
MPJPE (mm)
114.4
106
96.6
88.3
73.2
67.6
65.5
57.6
49
102.2
91.3
84.5
78.1
67.3
64.7
63.9
57.6
Ours supervised
Ours semi-supervised
(b) Full framerate under Protocol 1.
.1% S1
245
1% S1
2.42k
5% S1
12.4k
10% S1
24.8k
50% S1
124k
S1
248k
S15
645k
S156
895k
All
1.56M
Training data
35
45
55
65
75
85
95
105
115
125
MPJPE (mm)
106.2
100.7
89.7
80.7
63
55.7
53.8
44.2
39
90.8
78.1
72.5
65.2
53.9
49.7
51.7
41.3
Ours supervised GT
Ours semi-supervised GT abl.
Ours semi-supervised GT
(c) Full framerate under Protocol 1 with ground-truth 2D poses.
Figure 5: Top: comparison with [45] on Protocol 3, using a
downsampled version of the dataset for consistency. Mid-
dle: our method under Protocol 1 (full frame rate). Bottom:
our method under Protocol 1 when trained on ground-truth
2D poses (full frame rate). The small crosses (“abl.” series)
denote the ablation of the bone length term.
6.2. Semi-supervised approach
We adopt the setup of [45] who consider various subsets
of the Human3.6M training set as labeled data and the re-
maining samples are used as unlabeled data. Their setup
also generally downsamples all data to 10 FPS (from 50
FPS). Labeled subsets are created by ﬁrst reducing the num-
ber of subjects and then by downsampling Subject 1.
Since the dataset is downsampled, we use a receptive
ﬁeld of 9 frames, equivalent to 45 frames upsampled. For
the very small subsets, 1% and 5% of S1, we use 3 frames,
and we use a single-frame model for 0.1% of S1 where only
49 frames are available. We ﬁne-tuned CPN on the labeled
data only and warm up training by iterating only over la-
beled data for a few epochs (1 epoch for ≥S1, 20 epochs
for smaller subsets).
Figure 5a shows that our semi-supervised approach be-
comes more effective as the amount of labeled data de-
creases. For settings with less than 5K labeled frames, our
approach achieves improvements of about 9-10.4 mm N-
MPJPE over our supervised baseline. Our supervised base-
line is much stronger than [45] and outperforms all of their
results by a large margin. Although [45] uses a single-frame
model in all experiments, our ﬁndings still hold on 0.1% of
S1 (where we also use a single-frame model).
Figure 5b shows results for our method under the more
common Protocol 1 for the non-downsampled version of the
dataset (50 FPS). This setup is more appropriate for our ap-
proach since it allows us to exploit full temporal informa-
tion in videos. Here we use a receptive ﬁeld of 27 frames,
except in 1% of S1, where we use 9 frames, and 0.1% of
S1, where we use one frame. Our semi-supervised approach
gains up to 14.7 mm MPJPE over the supervised baseline.
Figure 5c switches the CPN 2D keypoints for ground-
truth 2D poses to measure if we could perform better with
a better 2D keypoint detector. In this case, improvements
can be up to 22.6 mm MPJPE (1% of S1) which con-
ﬁrms that better 2D detections could improve performance.
The same graph shows that the bone length term is crucial
for predicting valid poses, since it forces the model to re-
spect kinematic constraints (line “Ours semi-supervised GT
abl.”). Removing this term drastically decreases the effec-
tiveness of semi-supervised training: for 1% of S1 the er-
ror increases from 78.1 mm to 91.3 mm which compares to
100.7 mm for the supervised baseline.

## experiments
6.1. Temporal dilated convolutional model
Table 1 shows results for our convolutional model with
B = 4 blocks and a receptive ﬁeld of 243 input frames for
both evaluation protocols (§5). The model has lower aver-
age error than all other approaches under both protocols,
and does not rely on additional data such as many other
approaches (+). Under protocol 1 (Table 1a), our model
outperforms the previous best result [27] by 6 mm on av-
erage, corresponding to an 11% error reduction. Notably,
[27] uses ground-truth boxes whereas our model does not.
The model clearly takes advantage of temporal infor-

Dir. Disc.
Eat Greet Phone Photo Pose Purch.
Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg
Pavlakos et al. [41] CVPR’17 (∗)
67.4
71.9 66.7
69.1
72.0
77.0 65.0
68.3 83.7
96.5
71.7 65.8
74.9
59.1
63.2 71.9
Tekin et al. [52] ICCV’17
54.2
61.4 60.2
61.2
79.4
78.3 63.1
81.6 70.1 107.3
69.3 70.3
74.3
51.8
63.2 69.7
Martinez et al. [34] ICCV’17 (∗)
51.8
56.2 58.1
59.0
69.5
78.4 55.2
58.1 74.0
94.6
62.3 59.1
65.1
49.5
52.4 62.9
Sun et al. [50] ICCV’17 (+)
52.8
54.8 54.2
54.3
61.8
67.2 53.1
53.6 71.7
86.7
61.5 53.4
61.6
47.1
53.4 59.1
Fang et al. [10] AAAI’18
50.1
54.3 57.0
57.1
66.6
73.3 53.4
55.7 72.8
88.6
60.3 57.7
62.7
47.5
50.6 60.4
Pavlakos et al. [40] CVPR’18 (+)
48.5
54.4 54.4
52.0
59.4
65.3 49.9
52.9 65.8
71.1
56.6 52.9
60.9
44.7
47.8 56.2
Yang et al. [56] CVPR’18 (+)
51.5
58.9 50.4
57.0
62.1
65.4 49.8
52.7 69.2
85.2
57.4 58.4
43.6
60.1
47.7 58.6
Luvizon et al. [33] CVPR’18 (∗)(+)
49.2
51.6 47.6
50.5
51.8
60.3 48.5
51.7 61.5
70.9
53.7 48.9
57.9
44.4
48.9 53.2
Hossain & Little [16] ECCV’18 (†)(∗) 48.4
50.7 57.2
55.2
63.1
72.6 53.0
51.7 66.1
80.9
59.0 57.3
62.4
46.6
49.6 58.3
Lee et al. [27] ECCV’18 (†)(∗)
40.2
49.2 47.8
52.6
50.1
75.0 50.2
43.0 55.8
73.9
54.1 55.6
58.2
43.3
43.3 52.8
Ours, single-frame
47.1
50.6 49.0
51.8
53.6
61.4 49.4
47.4 59.3
67.4
52.4 49.5
55.3
39.5
42.7 51.8
Ours, 243 frames, causal conv. (†)
45.9
48.5 44.3
47.8
51.9
57.8 46.2
45.6 59.9
68.5
50.6 46.4
51.0
34.5
35.4 49.0
Ours, 243 frames, full conv. (†)
45.2
46.7 43.3
45.6
48.1
55.1 44.6
44.3 57.3
65.8
47.1 44.0
49.0
32.8
33.9 46.8
Ours, 243 frames, full conv. (†)(∗)
45.1
47.4 42.0
46.0
49.1
56.7 44.5
44.4 57.2
66.1
47.5 44.8
49.2
32.6
34.0 47.1
(a) Protocol 1: reconstruction error (MPJPE).
Dir. Disc.
Eat Greet Phone Photo Pose Purch.
Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg
Martinez et al. [34] ICCV’17 (∗)
39.5
43.2 46.4
47.0
51.0
56.0 41.4
40.6 56.5
69.4
49.2 45.0
49.5
38.0
43.1 47.7
Sun et al. [50] ICCV’17 (+)
42.1
44.3 45.0
45.4
51.5
53.0 43.2
41.3 59.3
73.3
51.0 44.0
48.0
38.3
44.8 48.3
Fang et al. [10] AAAI’18
38.2
41.7 43.7
44.9
48.5
55.3 40.2
38.2 54.5
64.4
47.2 44.3
47.3
36.7
41.7 45.7
Pavlakos et al. [40] CVPR’18 (+)
34.7
39.8 41.8
38.6
42.5
47.5 38.0
36.6 50.7
56.8
42.6 39.6
43.9
32.1
36.5 41.8
Yang et al. [56] CVPR’18 (+)
26.9
30.9 36.3
39.9
43.9
47.4 28.8
29.4 36.9
58.4
41.5 30.5
29.5
42.5
32.2 37.7
Hossain & Little [16] ECCV’18 (†)(∗) 35.7
39.3 44.6
43.0
47.2
54.0 38.3
37.5 51.6
61.3
46.5 41.4
47.3
34.2
39.4 44.1
Ours, single-frame
36.0
38.7 38.0
41.7
40.1
45.9 37.1
35.4 46.8
53.4
41.4 36.9
43.1
30.3
34.8 40.0
Ours, 243 frames, causal conv. (†)
35.1
37.7 36.1
38.8
38.5
44.7 35.4
34.7 46.7
53.9
39.6 35.4
39.4
27.3
28.6 38.1
Ours, 243 frames, full conv. (†)
34.1
36.1 34.4
37.2
36.4
42.2 34.4
33.6 45.0
52.5
37.4 33.8
37.8
25.6
27.3 36.5
Ours, 243 frames, full conv. (†)(∗)
34.2
36.8 33.9
37.5
37.1
43.2 34.4
33.5 45.3
52.7
37.7 34.1
38.0
25.8
27.7 36.8
(b) Protocol 2: reconstruction error after rigid alignment with the ground truth (P-MPJPE), where available.
Table 1: Reconstruction error on Human3.6M. Legend: (†) uses temporal information. (∗) ground-truth bounding boxes.
(+) extra data – [50, 40, 56, 33] use 2D annotations from the MPII dataset, [40] uses additional data from the Leeds Sports
Pose (LSP) dataset as well as ordinal annotations. [50, 33] evaluate every 64th frame. [16] provided us with corrected results
over the originally published results 3. Lower is better, best in bold, second best underlined.
mation as the error is about 5 mm higher on average for
protocol 1 compared to a single-frame baseline where we
set the width of all convolution kernels to W = 1. The
gap is larger for highly dynamic actions, such as “Walk”
(6.7 mm) and “Walk Together” (8.8 mm). The performance
for a model with causal convolutions is about half way be-
tween the single frame baseline and our model; causal con-
volutions enable online processing by predicting the 3D
pose for the rightmost input frame. Interestingly, ground-
truth bounding boxes result in similar performance to pre-
dicted bounding boxes with Mask R-CNN, which suggests
that predictions are almost-perfect in our single-subject sce-
nario. Figure 4 shows examples of predicted poses includ-
ing the predicted 2D keypoints and we included a video
illustration in the supplementary material (Appendix A.7)
as well as at https://dariopavllo.github.io/
VideoPose3D.
Next, we evaluate the impact of the 2D keypoint de-
3All subsequent results for [16] in this paper were computed by us using
their public implementation.
tector on the ﬁnal result. Table 3 reports accuracy of our
model with ground-truth 2D poses, hourglass-network pre-
dictions from [34] (both pre-trained on MPII and ﬁne-tuned
on Human3.6M), Detectron and CPN (both pre-trained on
COCO and ﬁne-tuned on Human3.6M). Both Mask R-CNN
and CPN give better performance than the stacked hourglass
network. The improvement is likely to be due to the higher
heatmap resolution, stronger feature combination (feature
pyramid network [31, 44] for Mask R-CNN and ReﬁneNet
for CPN), and the more diverse dataset on which they are
pretrained, i.e. COCO [32]. When trained on 2D ground-
truth poses, our model improves the lower bound of [34] by
8.3 mm, and the LSTM-based approach of Lee et al. [27]
by 1.2 mm for protocol 1. Therefore, our improvements are
not merely due to a better 2D detector.
Absolute position errors do not measure the smoothness
of predictions over time, which is important for video. To
evaluate this, we measure joint velocity errors (MPJVE),
corresponding to the MPJPE of the ﬁrst derivative of the
3D pose sequences. Table 2 shows that our temporal model

Figure 4: Qualitative results for two videos. Top: video frames with 2D pose overlay. Bottom: 3D reconstruction.
Dir. Disc.
Eat Greet Phone Photo Pose Purch. Sit SitD. Smoke Wait WalkD. Walk WalkT. Avg
Single-frame 12.8
12.6 10.3
14.2
10.2
11.3 11.8
11.3 8.2
10.2
10.3 11.3
13.1
13.4
12.9 11.6
Temporal
3.0
3.1
2.2
3.4
2.3
2.7
2.7
3.1 2.1
2.9
2.3
2.4
3.7
3.1
2.8
2.8
Table 2: Velocity error over the 3D poses generated by a convolutional model that considers time and a single-frame baseline.

## related_work
Before the success of deep learning, most approaches
to 3D pose estimation were based on feature engineer-
ing and assumptions about skeletons and joint mobility
[48, 42, 20, 18]. The ﬁrst neural methods with convolutional
neural networks (CNN) focused on end-to-end reconstruc-
tion [28, 53, 51, 41] by directly estimating 3D poses from
RGB images without intermediate supervision.
Two-step pose estimation. A new family of 3D pose es-
timators builds on top of 2D pose estimators by ﬁrst pre-
dicting 2D joint positions in image space (keypoints) which
are subsequently lifted to 3D [21, 34, 41, 52, 4, 16]. These
approaches outperform the end-to-end counterparts, since
they beneﬁt from intermediate supervision. We follow this
approach.
Recent work shows that predicting 3D poses
is relatively straightforward given ground-truth 2D key-
points, and that the difﬁculty lies in predicting accurate 2D
poses [34]. Early approaches [21, 4] simply perform a k-
nearest neighbour search for a predicted set of 2D keypoints
over a large set of 2D keypoints for which the 3D pose
is available and then simply output the corresponding 3D
pose. Some approaches leverage both image features and
2D ground-truth poses [39, 41, 52]. Alternatively, the 3D
pose can be predicted from a given set of 2D keypoints by
simply predicting their depth [58]. Some works enforce pri-
ors about bone lengths and projection consistency with the
2D ground truth [2].
Video pose estimation. Most previous work operates in
a single-frame setting but recently there have been efforts
in exploiting temporal information from video to produce
more robust predictions and to be less sensitive to noise.
[53] infer 3D poses from the HoG features (histograms of
oriented gradients) of spatio-temporal volumes.
LSTMs
have been used to reﬁne 3D poses predicted from single
images [30, 24]. The most successful approaches, however,
learn from 2D keypoint trajectories. Our work falls under
this category.
Recently, LSTM sequence-to-sequence learning models
have been proposed, which encode a sequence of 2D poses
from a video into a ﬁxed-size vector that is then decoded
into a sequence of 3D poses [16]. However, both the in-
put and output sequences have the same length and a deter-
ministic transformation of 2D poses is a much more natu-
ral choice. Our experiments with seq2seq models showed
that output poses tend to drift over lengthy sequences. [16]
tackles this problem by re-initializing the encoder every 5
frames, at the expense of temporal consistency. There has
also been work on RNN approaches which consider priors
on body part connectivity [27].
Semi-supervised training. There has been work on mul-
titask networks [3] for joint 2D and 3D pose estimation
[36, 33] as well as action recognition [33]. Some works
transfer the features learned for 2D pose estimation to the
3D task [35]. Unlabeled multi-view recordings have been
used for pre-training representations for 3D pose estima-
tion [45], but these recordings are not readily available
in unsupervised settings. Generative adversarial networks
(GAN) can discriminate realistic poses from unrealistic
ones in a second dataset where only 2D annotations are
available [56], thus providing a useful form of regulariza-
tion. [54] use GANs to learn from unpaired 2D/3D datasets
and include a 2D projection consistency term. Similarly,
[8] discriminate generated 3D poses after randomly project-
ing them to 2D. [40] propose a weakly-supervised approach
based on ordinal depth annotations which leverages a 2D
pose dataset augmented with depth comparisons, e.g. “the
left leg is behind the right leg”.
3D shape recovery. While this paper and the discussed
related work focus on reconstructing accurate 3D poses, a
parallel line of research aims at recovering full 3D shapes
of people from images [1, 23]. These approaches are typi-
cally based on parameterized 3D meshes and give less im-
portance to pose accuracy.
Our work. Compared to [41, 40], we do not use heatmaps
and instead describe poses with detected keypoint coordi-
nates. This allows the use of efﬁcient 1D convolution over
coordinate time series, instead of 2D convolutions over in-
dividual heatmaps (or 3D convolutions over heatmap se-
quences). Our approach also makes computational com-
plexity independent of keypoint spatial resolution.
Our
models can reach high accuracy with fewer parameters
and allow for faster training and inference. Compared to
the single-frame baseline proposed by [34] and the LSTM
model by [16], we exploit temporal information by perform-
ing 1D convolutions over the time dimension, and we pro-
pose several optimizations that result in lower reconstruc-
tion error. Unlike [16], we learn a deterministic mapping

2J, 3d1, 1024
1024, 3d3, 1024
BatchNorm 1D
ReLU
Dropout 0.25
BatchNorm 1D
ReLU
Dropout 0.25
1024, 1d1, 1024
BatchNorm 1D
ReLU
Dropout 0.25
1024, 1d1, 3J
Slice
(241, 1024)
(243, 34)
(235, 1024)
(1, 51)
1024, 3d9, 1024
BatchNorm 1D
ReLU
Dropout 0.25
1024, 1d1, 1024
BatchNorm 1D
ReLU
Dropout 0.25
Slice
(235, 1024)
(217, 1024)
1024, 3d27, 1024
BatchNorm 1D
ReLU
Dropout 0.25
1024, 1d1, 1024
BatchNorm 1D
ReLU
Dropout 0.25
Slice
(217, 1024)
(163, 1024)
1024, 3d81, 1024
BatchNorm 1D
ReLU
Dropout 0.25
1024, 1d1, 1024
BatchNorm 1D
ReLU
Dropout 0.25
Slice
(163, 1024)
(1, 1024)
Figure 2: An instantiation of our fully-convolutional 3D pose estimation architecture. The input consists of 2D keypoints for
a recpetive ﬁeld of 243 frames (B = 4 blocks) with J = 17 joints. Convolutional layers are in green where 2J, 3d1,
1024 denotes 2 · J input channels, kernels of size 3 with dilation 1, and 1024 output channels. We also show tensor sizes
in parentheses for a sample 1-frame prediction, where (243, 34) denotes 243 frames and 34 channels. Due to valid
convolutions, we slice the residuals (left and right, symmetrically) to match the shape of subsequent tensors.
instead of a seq2seq model. Finally, contrary to most of
the two-step models mentioned in this section (which use
the popular stacked hourglass network [38] for 2D keypoint
detection), we show that Mask R-CNN [12] and cascaded
pyramid network (CPN) [5] detections are more robust for
3D human pose estimation.
3. Temporal dilated convolutional model
Our model is a fully convolutional architecture with
residual connections that takes a sequence of 2D poses as
input and transforms them through temporal convolutions.
Convolutional models enable parallelization over both the
batch and the time dimension while RNNs cannot be paral-
lelized over time. In convolutional models, the path of the
gradient between output and input has a ﬁxed length regard-
less of the sequence length, which mitigates vanishing and
exploding gradients which affect RNNs. A convolutional
architecture also offers precise control over the temporal re-
ceptive ﬁeld, which we found beneﬁcial to model temporal
dependencies for the task of 3D pose estimation. Moreover,
we employ dilated convolutions [15] to model long-term de-
pendencies while at the same time maintaining efﬁciency.
Architectures with dilated convolutions have been success-
ful for audio generation [55], semantic segmentation [57]
and machine translation [22].
The input layer takes the concatenated (x, y) coordi-
nates of the J joints for each frame and applies a tempo-
ral convolution with kernel size W and C output channels.
This is followed by B ResNet-style blocks which are sur-
rounded by a skip-connection [13]. Each block ﬁrst per-
forms a 1D convolution with kernel size W and dilation fac-
tor D = W B, followed by a convolution with kernel size
1. Convolutions (except the very last layer) are followed
by batch normalization [17], rectiﬁed linear units [37], and
dropout [49]. Each block increases the receptive ﬁeld expo-
nentially by a factor of W, while the number of parameters
increases only linearly. The ﬁlter hyperparameters, W and
D, are set so that the receptive ﬁeld for any output frame
forms a tree that covers all input frames (see §1). Finally, the
last layer outputs a prediction of the 3D poses for all frames
in the input sequence using both past and future data to ex-
ploit temporal information. To evaluate real-time scenarios,
we also experiment with causal convolutions, i.e. convolu-
tions that only have access to past frames. Appendix A.1
illustrates dilated convolutions and causal convolutions.
Convolutional image models typically apply zero-
padding to obtain as many outputs as inputs. Early experi-
ments however showed better results when performing only
unpadded convolutions while padding the input sequence
with replica of the boundary frames to the left and the right
(see Appendix A.5, Figure 9a for an illustration).
Figure 2 shows an instantiation of our architecture for a
receptive ﬁeld size of 243 frames with B = 4 blocks. For
convolutional layers, we set W = 3 with C = 1024 output
channels and we use a dropout rate p = 0.25.
4. Semi-supervised approach
We introduce a semi-supervi

## conclusion
We have introduced a simple fully convolutional model
for 3D human pose estimation in video. Our architecture ex-
ploits temporal information with dilated convolutions over
2D keypoint trajectories.
A second contribution of this
work is back-projection, a semi-supervised training method
to improve performance when labeled data is scarce. The
method works with unlabeled video and only requires in-
trinsic camera parameters, making it practical in scenarios
where motion capture is challenging (e.g. outdoor sports).
Our fully convolutional architecture improves the previ-
ous best result on the popular Human3.6M dataset by 6mm
average joint error which corresponds to a relative reduc-
tion of 11% and also shows improvements on HumanEva-I.
Back-projection can improve 3D pose estimation accuracy
by about 10mm N-MPJPE (15mm MPJPE) over a strong
baseline when 5K or fewer annotated frames are available.