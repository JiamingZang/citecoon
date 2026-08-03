# PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation

> 2023 · id: W4386083126 · arXiv: 2303.17472 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recently, transformer-based methods have gained sig-
nificant success in sequential 2D-to-3D lifting human pose
estimation.
As a pioneering work, PoseFormer captures
spatial relations of human joints in each video frame and
human dynamics across frames with cascaded transformer
layers and has achieved impressive performance. However,
in real scenarios, the performance of PoseFormer and its
follow-ups is limited by two factors: (a) The length of the
input joint sequence; (b) The quality of 2D joint detection.
Existing methods typically apply self-attention to all frames
of the input sequence, causing a huge computational burden
when the frame number is increased to obtain advanced es-
timation accuracy, and they are not robust to noise natu-
rally brought by the limited capability of 2D joint detectors.
In this paper, we propose PoseFormerV2, which exploits a
compact representation of lengthy skeleton sequences in the
frequency domain to efficiently scale up the receptive field
and boost robustness to noisy 2D joint detection. With min-
imum modifications to PoseFormer, the proposed method
effectively fuses features both in the time domain and fre-
quency domain, enjoying a better speed-accuracy trade-off
than its precursor. Extensive experiments on two benchmark
datasets (i.e., Human3.6M and MPI-INF-3DHP) demon-
strate that the proposed approach significantly outperforms
the original PoseFormer and other transformer-based vari-
ants.
Code is released at https://github.com/
QitaoZhao/PoseFormerV2.

## introduction
3D human pose estimation (HPE) aims at localizing
human joints in 3-dimensional space based on monocular
videos (without intermediate 2D representations) [25,28] or
2D human joint sequences (referred to as 2D-to-3D lifting
*Work was done while Qitao was an intern mentored by Chen Chen.
↓
Speedup 4.6x
0.9
1.0
↓
↓1.6
↓1.2
↓1.1
↓1.6
↓0.8
↓0.9
Figure 1. Comparisons of PoseFormerV2 and PoseFormerV1 [46]
on Human3.6M [13]. RF denotes Receptive Field and k×RF indi-
cates that the ratio between the full sequence length and the num-
ber of frames as input into the spatial encoder of PoseFormerV2
is k, i.e., the RF of the spatial encoder is expanded by k× with a
few low-frequency coefficients of the full sequence. The proposed
method outperforms PoseFormerV1 by a large margin in terms of
speed-accuracy trade-off, and the larger k brings more significant
improvements, e.g., 4.6× speedup with the k of 27.
approaches) [5,18,37,43]. With the large availability of 2D
human pose detectors [6,26] plus the lightweight nature of
2D skeleton representation of humans, lifting-based meth-
ods are now dominant in 3D human pose estimation. Com-
pared to raw monocular videos, 2D coordinates of human
joints in each video frame are much more memory-friendly,
making it possible for lifting-based methods to utilize a long
joint sequence to boost pose estimation accuracy.
Transformers [36] first gain huge success in the field of
natural language processing (NLP) [3, 8] and then extend
their capacity to the computer vision community, becom-
ing the de facto approach for several vision tasks, e.g., im-
age classification [9, 19, 35], object detection [4, 47] and
video recognition [1,2,42]. The discreteness of human joint
representation and the requirement for long-range temporal
1
arXiv:2303.17472v1  [cs.CV]  30 Mar 2023

…
2D pose sequence (e.g., 9 frames)
Temporal Transformer Encoder
Spatial Transformer
1
2
3
7
8
9
PE1
PE2
PE3
PE8
PE9
PE7
…
Regression Head
3D pose for the center frame
Spatial/Temporal 
Transformer Encoder
MLP
Layer Norm 
Layer Norm 
Multi-Head
Attention
L ×
Temporal
Positional
Embedding
Figure 2. Overview of PoseFormerV1. PoseFormerV1 mainly
consists of two modules: the spatial transformer encoder and the
temporal transformer encoder.
The temporal encoder of Pose-
FormerV1 applies self-attention to all frames given a 2D joint se-
quence for human motion modeling.
Table 1. The computational cost and performance drop brought
by replacing ground-truth 2D detection with CPN [6] 2D pose
detection for the SOTA transformer-based methods. The perfor-
mance drop is reported on Human3.6M dataset (Protocol 1) [13].
RF: Receptive Field, sharing the same meaning as that in Fig. 1.

## method
PCK ↑
AUC ↑
MPJPE ↓
Mehta et al. [23]
3DV’17
75.7
39.3
117.6
Mehta et al. [24]
ACM ToG’17
76.6
40.4
124.7
Pavllo et al. [29] (T =81)
CVPR’19
86.0
51.9
84.0
Pavllo et al. [29] (T =243)
CVPR’19
85.5
51.5
84.8
Lin et al. [17] (T =25)
BMVC’19
83.6
51.4
79.8
Li et al. [14]
CVPR’20
81.2
46.1
99.7
Chen et al. [5] (T =81)
TCSVT’21
87.9
54.0
78.8
PoseFormerV1 [46] (T =9)(†)
ICCV’21
95.4
63.2
57.7
MHFormer [16] (T =9)
CVPR’22
93.8
63.3
58.0
MixSTE [44] (T =27)
CVPR’22
94.4
66.5
54.9
P-STMO [32] (T =81)(*)
ECCV’22
97.9
75.8
32.2
PoseFormerV2 (T =81)
97.9
78.8
27.8
the proposed method outperforms other transformer-based
methods in terms of speed-accuracy trade-off. Note that the
methods with an additional pre-training stage and compu-
tationally heavy MixSTE [44] (3420 MFLOPs for only 3-
frame input) are not included. The improvements of Pose-
FormerV2 over PoseFormerV1 are provided in Fig. 1.
In order to demonstrate that the inclusion of low-
frequency DCT coefficients helps improve the robustness
of the proposed method, we make the lifting-based pose es-
timation task more challenging by adding zero-mean Gaus-
sian noise to the ground-truth 2D detection on the Hu-
man3.6M dataset [13] (Fig. 6). To ensure a fair comparison,
we keep the input sequence length the same for all meth-
ods (in this case, 27 frames). For our method, f = n = 3.
The experimental evidence reveals that PoseFormerV2 suf-
fers from less performance drop as the standard deviation
of Gaussian noise (sigma) increases while being more ef-
ficient. We observe that the performance of PoseFormerV1
drops drastically as sigma increases from 8 to 10. In con-
Table 4.
Ablation study on several modifications to Pose-
FormerV1. We show how a 9-frame PoseFormerV1 is converted
to PoseFormerV2 (with 9 DCT coefficients from an 81-frame se-
quence) step by step. The evaluation is performed on Human3.6M
(Protocol 1, in mm). RF indicates Receptive Field.
Step
Description
RF
MPJPE ↓
(0)
Original 9-frame PoseFormerV1.
9
49.9
(1)
Frames are sampled from a longer sequence.
9
49.9
(2)
Append the embedding of DCT coefficients.
81
47.1 (2.8↓)
(3)
Replace the vanilla MLP with FreqMLP.
81
46.0 (3.9↓)
Table 5. Ablation study on the number of frames and the number
of DCT coefficients that are used as input to PoseFormerV2. The
evaluation is performed on Human3.6M (Protocol 1, in mm).
Frame
Number (f)
Coefficient
Number (n)
Full
Length
MFLOPs
MPJPE
1
1
27
39.2
51.1
1
3
27
77.2
48.7 (2.4↓)
3
1
27
79.4
50.1 (1.0↓)
3
3
27
117.3
47.9 (3.2↓)
9
9
27
351.7
47.6 (3.5↓)
trast, the proposed method presents a more stable trend.
Moreover, our method even outperforms MHFormer [16]
that incorporates the uncertainty of 2D detectors into the
model design. Intriguingly, we find that minor noise may
improve the accuracy of 3D pose estimation (sigma = 3).
MPI-INF-3DHP. We also compare our method with oth-
ers on MPI-INF-3DHP [23] (Table 3). We use 9 central
frames and the first 9 DCT coefficients from the input 81-
frame sequence. The proposed method outperforms other
approaches including P-STMO [32] with masked joint pre-
training. This result verifies the effectiveness of our method.
Our implementation follows [32].
Qualitative comparisons. We provide qualitative com-
parisons of our method with competitive MHFormer [16]
and PoseFormerV1 [46] in Fig. 7.
All methods use 81-
frame 2D joint sequences as input. To further illustrate the
robustness of our approach, we make the pose estimation
task more difficult by adding Gaussian noise to the sequen-
tial 2D detection of a randomly selected joint (e.g., “left
wrist”, “right foot”). The proposed method obtains reliable
3D human pose even under highly-deviated 2D detection
(indicated by the light-yellow arrows). Note that our model
is ∼9× more efficient than MHFormer (3.12 GFLOPs vs.
0.35 GFLOPs) and ∼4× more efficient compared to Pose-
FormerV1 (1.36 GFLOPs vs. 0.35 GFLOPs).
4.4. Ablation Study
In this section, we show how a few modifications to
PoseFormerV1 bring significant improvements in a step-by-
step way. Moreover, to investigate more insights into the
frequency-domain representation of input sequences, we re-
veal the impact of the number of input frames and that of
kept DCT coefficients on our method.
Convert PoseFormerV1 into PoseFormerV2. We in-
herit the overall spatial-temporal architecture from Pose-
FormerV1 and introduce restrained modifications to its tem-
7

Input
MHFormer
PoseFormerV2
PoseFormerV1
Input
MHFormer
PoseFormerV2
PoseFormerV1
Figure 7. Qualitative comparisons of PoseFormerV2 with MHFormer [16] and PoseFormerV1 [46]. We randomly add Gaussian noise to
the 2D detection of a specific joint. We highlight the deviated 2D detection with light-yellow arrows and corresponding 3D pose estimations
with orange arrows. PoseFormerV2 shows better robustness to highly noisy input than existing methods.
poral transformer for better multi-domain feature fusion.
To exemplify, we illustrate how a 9-frame PoseFormerV1
is converted to PoseFormerV2 step by step: (1) the input
(i.e., 9 frames) is sampled from a longer sequence (e.g., 81
frames) at the sequence center. This step brings no perfor-
mance improvement or increase in the receptive field since
the input to the model is in fact unchanged. (2) The out-
put of the spatial encoder of PoseFormerV1, zT ime, is ap-
pended to the embedding of the first n DCT coefficients
(denoted by zF req) of the complete sequence (81 frames in
this case) as input into the temporal encoder. For conve-
nience, we set n to 9. (3) We replace the vanilla MLP for
zT ime (zT ime and zF req already use separate vanilla MLPs
before replacement) in the temporal encoder with FreqMLP
(details in Sec. 3.2.2). PoseFormerV1 is converted to Pose-
FormerV2 after these steps, with an enlarged receptive field
(from 9 to 81). We present the improvement brought by
each step in Table 4. It is worth noting that by introducing 9
DCT coefficients from a longer sequence (i.e., 81 frames),
the MPJPE of 9-frame PoseFormerV1 is reduced by 7.8%
(49.9mm vs. 46.0mm), which verifies the effectiveness of
the proposed DCT representation of input joint sequences.
Number of input frames and DCT coefficients. In Ta-
ble 5, we investigate the impact of the number of frames (f)
as input to the spatial encoder and the number of retained
DCT coefficients (n). Here we keep the length of the en-
tire joint sequence fixed, i.e., 27. The baseline model uses
only one central frame and one DCT coefficient (f = n =
1). Increasing both parameters brings consistent improve-
ments, and the increase in n translates to more error reduc-
tion (2.4↓for n = 3 vs. 1.0↓for f = 3) since only a few
DCT coefficients help capture the global characteristics of
the entire sequence. We empirically find that the matched f
and n with an expanding ratio of 9 (i.e., f = n = 3) achieve
a satisfactory speed-accuracy trade-off.
4.5. Generalization Ability
The proposed frequency-domain approach can general-
ize to other methods, e.g., MixSTE [44] and MHFormer
[16], as they also use transformers for temporal modeling.
We improve both methods by incorporating low-frequency
DCT coefficients. Details are in supplementary.

## experiments
4.1. Datasets and Evaluation Metrics
We conduct experiments on two 3D human pose esti-
mation datasets, i.e., Human3.6M [13] and MPI-INF-3DHP
[23] to demonstrate the effectiveness of our method. More
detailed descriptions of both datasets and their respective
evaluation metrics are in the supplementary.
4.2. Implementation Details and Analysis
The proposed method includes three important hyper-
parameters that are specific to experimental settings. These
include the number of frames (f) used as input in the spatial
encoder, the length of the entire input sequence (F) repre-
senting the enlarged receptive field, and the number of kept
DCT coefficients (n) utilized to incorporate long-range tem-
poral information. If not specified, we simply set n = f for
convenience. In practice, they can be further tuned for a
flexible speed-accuracy trade-off. When f equals 1, n is
set to 3 because a single DCT coefficient may be insuffi-
cient to encode temporal information from lengthy input se-
quences. As f and n are fixed, the computational complex-
ity of the model is predetermined (i.e., the token number for
the spatial encoder and that for the feature-fusion module
are fixed). We may vary F to effectively expand the model’s
receptive field from a limited f to an arbitrary value, bring-
ing no additional computational overhead. This enables us
to efficiently use long sequences to improve accuracy. We
provide details of the hyper-parameters for model architec-
ture and training in the supplementary.
4.3. Comparisons with State-of-the-art Methods
Human3.6M. We compare our method with Pose-
FormerV1 and other transformer-based methods on Hu-
Table 2.
Quantitative comparisons with previous transformer-
based methods on Human3.6M (in mm). f: number of frames as
input to the model, Seq. Len.: length of the entire input sequence
(i.e., the effective Receptive Field). The best scores are marked in
bold. (*) indicates using an additional pre-training stage and (†)
indicates our re-implementation.

## related_work
Our method is built on conceptually simple PoseFormer
[46], and we aim at improving its efficiency to operate long
sequences and its robustness to noisy joint detection from a
frequency-domain perspective. Therefore, here we mainly
focus on this line of works (transformer-based methods)
in 2D-to-3D lifting HPE and introduce applications of fre-
quency domain representations in computer vision litera-
ture, especially in skeleton-based tasks that are most related
to lifting-based 3D HPE.
2.1. Transformer-based 3D Human Pose Estimation
PoseFormer [46] is the first work to adopt the vision
transformer as the backbone network in lifting-based 3D
human pose estimation, and it outperforms previous CNN-
based methods by a large margin.
Though being com-
petitive, Zhang et al. [44] point out that the spatial-then-
temporal paradigm of PoseFormer may neglect distinct tem-
poral patterns for each joint, and propose to adopt alternate
spatial-temporal transformer layers for fine-grained joint-
specific feature extraction. MHFormer [16] further incor-
porates task-related prior knowledge into transformers for
3D HPE. Specifically, 2D-to-3D lifting is an inverse prob-
lem where more than one reasonable solutions exist, there-
fore they generate multiple hypotheses to model ambiguous
body parts and uncertainty in joint detectors, achieving ad-
vanced performance. Inspired by the progress of Masked
Image Modeling (MIM) in image classification [12,39,40],
P-STMO [32] applies Masked Joint Modeling to 3D HPE
with self-supervised learning.
Another line of works [10, 15] improves the efficiency
of transformer-based methods.
Taking advantage of the
temporal redundancy in 2D joint sequences, StridedTrans-
former [15] replaces the parameter-heavy fully-connected
layers with strided convolutions. Einfalt et al. [10] claim
that the per-frame 2D joint detection is even more computa-
tionally expensive than lifting models themselves and pro-
pose to downsample input video frames with a fixed interval
and adopt the 2D joint detector and lifting model only on
these sampled frames. While being more efficient than pre-
vious works, aforementioned methods [10, 15] reduce par-
ticipants in self-attention along the temporal dimension uti-
lizing only the consistency in adjacent video frames rather
than from a global view, and therefore they may suffer from
a considerable performance drop.
2.2. Frequency Representation in Vision
Since the human visual system is more sensitive to low-
frequency components of images, traditional image com-
pression algorithms, e.g., JPEG [30] and JPEG 2000 [33],
reduce memory cost to store 2D images by allocating more
storage budget to low-frequency Discrete Cosine Transform
(DCT) coefficients of the image. Following the same logic,
[41] proposes to adaptively remove uninformative channels
of DCT components for 2D images to boost image clas-
sification efficiency. More recently, some works [11, 31]
propose to replace the costly self-attention mechanism with
frequency transforms that can be accelerated by their fast
algorithms. GFN [31] proposes to efficiently mix visual to-
kens with learnable frequency filters, and AFNO [11] fur-
ther improves the performance of token mixer in the fre-
quency domain with operator learning. Moreover, Wang et
al. [38] utilize low-frequency Fast Fourier Transform (FFT)
components to compress vision transformers.
Skeleton-based tasks are more relevant to our work that
takes 2D skeleton sequences as input. In the human mo-
tion prediction literature, previous works [21,22] transform
the skeleton sequence from the time domain into DCT co-
efficients to encode human dynamics as compared to static
joint coordinates. They observe that discarding a few high-
frequency coefficients does not necessarily bring a perfor-
mance drop but even improves the smoothness of predicted
3

future motions.
However, frequency-domain representa-
tions of 2D joint sequences have not yet been explored in
lifting-based 3D human pose estimation.
Our approach is inspired by these former attempts of ap-
plying frequency transforms to vision tasks but from a dif-
ferent view. We include more details about our motivations
to choose the DCT coefficient representation in Sec. 3.2.1.

## conclusion
We present a solution to reconcile two seemingly un-
related or even contracted issues in lifting-based 3D hu-
man pose estimation – the efficiency of processing long-
sequence input and the robustness to noisy joint detection –
simultaneously from a barely explored frequency-domain
perspective.
The proposed method, PoseFormerV2, ex-
ploits a compact frequency representation of long 2D joint
sequences to efficiently enlarge the receptive field of the
model while improving its robustness. Experimental results
show that our method outperforms previous transformer-
based methods on Human3.6M and MPI-INF-3DHP.
8

Supplementary Material
A. Overview
The supplementary material includes sections as follows:
• Section B: A formal introduction to Discrete Cosine
Transform.
• Section C: Datasets and evaluation metrics.
• Section D: More implementation details.
• Section E: Comparisons of PoseFormerV2 and a sim-
ple baseline model purely in the frequency domain.
• Section F: Generalization of our approach to more
models.
• Section G: Visualizations and analysis.
• Section H: Broader impacts and limitations.
B. Discrete Cosine Transform
We now give a formal introduction to DCT. Given a 2D
joint sequence denoted by x ∈RF ×J×2, where F is the
sequence length and J is the joint number in each frame,
the trajectory of the x (or y) coordinate of the j-th joint de-
noted as xj,0 ∈RF (or xj,1 ∈RF , both denoted by ˆxj for
convenience) is a 1D time series and we apply DCT to each
trajectory (J ∗2 trajectories in total) individually.
For trajectory ˆxj, the i-th DCT coefficient is calculated
as
  \r e
s
i
z
eb
ox {0.9
\
linewi dth
 }
{! }{$ C _{j, i } 
=
 \sqrt {\frac {2}{F}}\sum _{f=1}^{F}x_{j,f}\frac {1}{\sqrt {1+\delta _{i1}}}\cos \left (\frac {\pi }{2F}(2f-1)(i-1)\right )\;,$} \label {eq:dct} \vspace {-0.2cm} (5)
where δi1 = 1 when i = 1, otherwise δi1 = 0. Each
time step in trajectory yields one DCT coefficient, i.e., i ∈
{1, 2, · · · , F}. DCT coefficients encode multiple levels of
temporal information in the input time series. Specifically,
low-frequency coefficients (i.e., when i is small) encode the
rough contour of the input sequence while high-frequency
coefficients (i.e., for the large i) encode details, e.g., jitters
or sharp changes in the input sequence. The original input
sequence in the time domain can be restored using Inverse
Discrete Cosine Transform (IDCT), which is given by
  \r e
s
i
z
eb
ox {0.9
\
linewi dth
 }
{! }{$ x _{j, f } 
=
\sqrt {\frac {2}{F}}\sum _{i=1}^{F}C_{j,i}\frac {1}{\sqrt {1+\delta _{i1}}}\cos \left (\frac {\pi }{2F}(2f-1)(i-1)\right )\;,$} \vspace {-0.2cm} (6)
and f ∈{1, 2, · · · , F}. DCT is lossless if we keep all its
coefficients intact. In practice, we can slightly lossily re-
cover the input sequence using only a few low-frequency
coefficients and set other coefficients to zero. It is worth
noting that the recovered curve would be smoother com-
pared to the original one since we discard some of the high-
frequency coefficients. This property of DCT is desirable –
only a small proportion of DCT coefficients are enough to
represent the whole input sequence, even in a cleaner man-
ner. This motivates us to use such representation to effi-
ciently operate long sequences while improving the robust-
ness of the model to low-quality 2D detection where high-
frequency noise often occurs.
C. Datasets and Evaluation Metrics
Human3.6M is the most widely used benchmark for 3D
human pose estimation. Over 3.6 million video frames are
captured indoors from 4 cameras at different places. This
dataset contains 11 subjects performing 15 different actions,
e.g., “Walking” and “Phoning”. We train our model on 5
subjects (S1, S5, S6, S7, S8) and use other 2 subjects (S9,
S11) for testing, following [5,18,29,46].
MPI-INF-3DHP is collected in both controlled indoor
environments and challenging outdoor environments.
It
also provides different subjects and actions from multiple
camera views similar to Human3.6M.
Evaluation Metrics. We report two common metrics,
MPJPE and P-MPJPE [45] on Human3.6M. MPJPE (Mean
Per Joint Position Error, referred to as Protocol 1) measures
the mean Euclidean distance between the estimated 3D pose
and the ground truth 3D pose. P-MPJPE (Protocol 2) ap-
plies a rigid transformation to the estimated 3D pose and
the distance is computed between the aligned estimated 3D
pose and the ground truth 3D pose.
For the MPI-INF-3DHP dataset, we report MPJPE, Per-
centage of Correct Keypoint (PCK) within the 150mm
range, and Area Under Curve (AUC) as in [5,17,37].
D. More Implementation Details
Our method is built upon PoseFormerV1 [46]. Aiming
at better demonstrating the effectiveness of our DCT co-
efficient representation of input sequences and providing
fair comparisons to PoseFormerV1, we directly adopt op-
timal hyper-parameters for model architecture from Pose-
FormerV1, although further investigation may bring addi-
tional improvements.
Model architecture hyper-parameters. The embedded
feature dimension c in the spatial transformer is 32 and the
layer number of the spatial transformer and feature-fusion
transformer is 4, following [46]. Plus, the design of Spatial-
Temporal Positional Embedding is also adopted from [46].
Experimental settings. Our experiments are conducted
with Pytorch [27] on a single NVIDIA RTX 3090. For both
training and testing, we apply horizontal flipping augmen-
tation following [5, 18, 29, 46]. We train our model using
the AdamW [20] optimizer for 80 epochs with a weight de-
cay of 0.1. The initial learning rate is set to 8e-4 with an
exponential learning rate decay schedule and the decay fac-
tor is 0.99. We adopt the CPN [7] 2D pose detection on
Human3.6M, following [5, 18, 29]. As for the MPI-INF-
3DHP dataset, we use ground truth 2D detection, follow-
ing [17,24].
E. Simple Baseline
In our approach, the temporal encoder of PoseFormerV1
[46] is reformulated as a Time-Frequency Feature Fusion
9

Table 6.
Comparisons of PoseFormerV2 and a simple base-
line. The evaluation is performed on Human3.6M (Protocol 1,
MPJPE) [13] and the Frame Number (f) is only applicable to
PoseFormerV2.
Frame
Number (f)
Coefficient
Number (n)
Full
Length
Baseline
PoseFormerV2
3
3
9
50.2
49.5 (0.7↓)
3
3
27
48.7
47.9 (0.8↓)
3
3
81
49.7
47.1 (2.6↓)
9
9
27
48.8
47.6 (1.2↓)
9
9
81
47.8
46.0 (1.8↓)
module and we show that the low-frequency coefficients of
the input sequence help improve the efficiency of the model
to process long sequences and its robustness against noisy
joint detection. Given the effectiveness of this representa-
tion, readers may raise a question: Why not entirely ex-
tract features from DCT coefficients of the input sequence
but additionally combine them with features in the time do-
main? Here we design a baseline model where we simply
replace the input to PoseFormerV1 [46] (joint coordinates
in the time domain) with low-frequency DCT coefficients
of the input sequence. The full sequence length and the
number of the retained DCT coefficients (denoted as n) are
kept the same for the baseline model and our approach. For
convenience, the number of frames (f) as input into the
spatial encoder of PoseFormerV2 is set to n. We provide
quantitative results to demonstrate that this straightforward
approach does not work well, especially when the ratio be-
tween the full sequence length and n is increased (see Table
6). The features of only a few central frames in the sequence
significantly boost accuracy, e.g., with 3 central frames of
the full input sequence of length 81, the MPJPE is reduced
from 49.7mm to 47.1mm (5.2%↓, the 3rd row in Table 6).
Intuitively, the spatial encoder of PoseFormerV2 that en-
codes joint coordinates of a few central frames in the time
domain helps capture the fine-grained human motions, ben-
efiting 3D pose estimation for the frame at the sequence
center. In contrast, low-frequency coefficients of the input
sequence filter out high-frequency noise and human mo-
tion details (e.g., fast motions) that may be informative to
human pose estimation (i.e., the over-smoothing problem).
Therefore, features from the time domain and frequency do-
main, i.e., the joint coordinate of central frames and low-
frequency coefficients of the sequence, carry complemen-
tary semantics. These considerations necessitate our pro-
posed Time-Frequency Feature Fusion design.
F. Generalization to More Models
In the main text, we focus on improving PoseFormerV1
[46] from a barely explored frequency-domain perspective.
In this part, we show that the proposed frequency-domain
approach also generalizes well to other existing state-of-the-
art methods, e.g., MixSTE [44] and MHFormer [16]. Since
these approaches [16, 44] also apply self-attention along
5
10
15
20
25
30
GFLOPs
46
47
48
49
MPJPE 
MixSTE
MixSTE-3xRF
MixSTE-9xRF
Figure 8. Comparisons of MixSTE [44] and its improved version
with frequency representations of the sequence on Human3.6M
[13].
RF: Receptive Field and k×RF indicate that the 