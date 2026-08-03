# P-STMO: Pre-trained Spatial Temporal Many-to-One Model for 3D Human Pose Estimation

> 2022 · id: W4312797994 · arXiv: 2203.07628 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## introduction
Fig. 1 depicts an overview of the proposed P-STMO method, which divides the
optimization process into two stages: pre-training and fine-tuning. Firstly, a part
of STMO model is pre-trained by solving the masked pose modeling (MPM) task
in a self-supervised manner. The goal of this stage is to recover the input sequence
from the corrupted 2D poses. Secondly, STMO model is fine-tuned to predict
the 3D pose in the current (middle) frame given a sequence of 2D poses obtained
by an off-the-shelf 2D pose detector. For both stages, we take a sequence of 2D
poses as input, denoted as
  X=\{x_n\}_{
n=-(N-1)/2}
^{ ( N-1)/
2},\quad x_n=\{p_i\}_{i=1}^J,\label {eq1} 
(1)
where pi ∈R2 is the 2D position of ith joint. N, J are the number of frames and
human joints in each frame, respectively. Usually, N is an odd number, which
means that the current frame and (N −1)/2 frames to the left and right of it
are used as inputs.

5
Spatial Temporal Masking
Temporal Encoding Module
Self-supervised
Encoded unmasked 
embedding
Spatial padding 
joint
Stage I (Pre-Training)
Stage II (Training & Inference)
Spatial Encoding Module
Temporal Encoding Module
Spatial Encoding Module
Many-to-One Frame 
Aggregator
Temporal padding 
embedding
Model loading
(a)
(b)
Decoder
single

Encoder
multiple

MOFA
Decoder
TEM
SEM
Transformer
MLP
Strided 
Transformer
Fig. 1. (a) The pre-training procedure for STMO. The 2D pose sequence is randomly
masked and fed to the encoder. The encoded unmasked embeddings as well as the
temporal padding embeddings are sent to the decoder to reconstruct the original 2D
poses in the input sequence. (b) Overview of our STMO model, which consists of SEM,
TEM, and MOFA in series.
Stage I: As shown in Fig. 1a, a proportion of frames as well as some joints
in the remaining frames are randomly masked. This spatially and temporally
masked input is denoted as XST. The whole network architecture consists of a
spatial temporal encoder (SEM+TEM) that maps the masked input XST to the
latent space, and a decoder that recovers the original 2D poses X from latent
representations. To predict the complete 2D poses, the model has to seek relevant
unmasked joints for help. In this way, SEM and TEM are enabled to learn 2D
spatial and temporal relationships. Since monocular motion contributes to depth
estimation [41], acquiring 2D spatio-temporal relationships is beneficial for 3D
human pose estimation.
Stage II: Fig. 1b shows the proposed STMO model. The pre-trained encoder is
loaded and fine-tuned in this stage to obtain knowledge about the 3D space. This
encoder is followed by MOFA that aggregates multiple frames to estimate the
3D pose in the middle frame of the sequence, denoted as Y = {y0}, y0 ∈RJ×3.

6
(a)
Masked pose
Masked joint
(b)
(c)
Fig. 2. Illustration of three masking strategies. (a) Temporal masking. (b) Spatial
masking. (c) Spatial temporal masking.
3.2
Pre-Training of STMO
The spatial and temporal dependencies of the pose sequence are mainly captured
by SEM and TEM in our STMO model. To improve the overall estimation
accuracy, we propose a self-supervised spatial temporal pre-training task, namely
MPM. As shown in Fig. 2, we use three masking strategies to mask the input
2D poses, which are described in detail below.
Temporal Masking. A portion of input frames are randomly masked, which is
called temporal masking. The masked frame is replaced with a temporal padding
embedding eT which is a shared learnable vector. Similar to [16], to improve
the efficiency of the model, we only use the unmasked frames as inputs to the
encoder, excluding the temporal padding embeddings. Instead, the decoder takes
the temporal padding embeddings as well as the encoded unmasked embeddings
as inputs and reconstructs the original 2D poses. In this way, the encoder models
temporal dependencies between two unmasked frames that are not adjacent to
each other in the original sequence, and then the decoder fills in the missing 2D
poses between these two frames. We denote the indices of the masked frames as
a set: MT ⊆{−N−1
2 , −N−1
2
+ 1, . . . , N−1
2 }, | M T |= qT · N, where qT ∈R is
the temporal masking ratio. We use a large temporal masking ratio (e.g., 90%),
so this task cannot be solved easily by interpolation. The input X to the encoder
(eq. 1) is modified to
  X ^\t
e x t { T}=\{x_n^{
\text {T}}:
n\
n o tin \
mathcal {M}^{\text {T}}\}_{n=-(N-1)/2}^{(N-1)/2},\quad x_n^{\text {T}}=\{p_i\}_{i=1}^J. 
(2)
After that, the input to the decoder is denoted as
  \ { h _n :n\notin \
mathcal {M
}
^\t e x t {T}\}_{n=
-(N-1)/2}^{(N-1)/2} \bigcup \{e^\text {T}:n\in \mathcal {M}^\text {T}\}_{n=-(N-1)/2}^{(N-1)/2}, 
(3)
where hn, eT ∈Rd are the encoded unmasked embedding and temporal padding
embedding respectively. d is the dimension of the latent features. The output is
Y = {yn}(N−1)/2
n=−(N−1)/2, where yn ∈RJ×2 is the recovered 2D pose in frame n.
Spatial Masking. A fixed number of joints in each frame are randomly masked,
which is called spatial masking. The masked joint is replaced with a spatial
padding joint eS that is a shared learnable vector. The spatial padding joints,
together with other unmasked joints, are sent to the encoder. Since the pose
sequence is not masked temporally, no temporal padding embedding is used

7
at the decoder side. In such manner, the encoder models spatial dependencies
between joints, and then the decoder is capable of recovering the contaminated
joints. The indices of the masked joints in frame n are denoted as a set: MS
n ⊆
{−N−1
2 , −N−1
2
+ 1, . . . , N−1
2 }, | MS
n |= mS, where mS ∈N is the number of
masked joints. The spatial masking ratio is qS = mS/J. Although the number
of masked joints is the same for each frame, the indices of the masked joints are
various in different frames to increase the diversity. The input X to the encoder
(eq. 1) is modified to
  X ^\t
ext {S}=\
{x_n^\text 
{S
} \ }_{ n = -( N-
1)/
2}^
{
(N- 1 ) / 2}
,\q
uad x_n^\text {S}=\{p_i :i\notin \mathcal {M}^\text {S}_n\}_{i=1}^J \bigcup \{e^\text {S} :i\in \mathcal {M}^\text {S}_n\}_{i=1}^J, 
(4)
where eS ∈R2 is the spatial padding joint.
Spatial Temporal Masking. To integrate the information on the spatio-
temporal domain, we propose a spatial temporal masking strategy, which is
a combination of the above two masking methods. Specifically, the temporal
masking is implemented on the input pose sequence, followed by the spatial
masking on the unmasked frames. This strategy is utilized in the proposed P-
STMO model. The total spatial temporal masking ratio is calculated by qST =
qT + (1 −qT) · qS. The input X to the encoder (eq. 1) is modified to
  X ^ {\te
x
t {S T}}&=\{x_n
^{\text {ST
}}:
n\n
o
t in \ m at hc
al 
{M}
^
\te x t {T
}\}
_{n=-(N-1)/2}^{(N-1)/2},\\ x_n^{\text {ST}}&=\{p_i :i\notin \mathcal {M}^\text {S}_n\}_{i=1}^J \bigcup \{e^\text {S} :i\in \mathcal {M}^\text {S}_n\}_{i=1}^J .
(6)
3.3
Spatial Temporal Many-to-One (STMO) Model
In Stage II, the pre-trained encoder is loaded to the proposed STMO model
and fine-tuned on 3D poses. The detailed architecture of STMO is illustrated in
Fig. 3. The input sequence will go through SEM, TEM, and MOFA to obtain
the final output. The role of each module is described in detail below.
Spatial Encoding Module (SEM). SEM aims to capture the characteristics
of each frame in the spatial domain. Zheng et al. [55] propose to use a Trans-
former [45] as the backbone network of SEM to integrate information across
all joints in a single frame. However, the self-attention operation brings a great
computational overhead, which limits the scalability of the network in the case
of using multiple frames as inputs. Therefore, we propose to use a simple MLP
block as the backbone network to establish spatial relationships between joints.
Compared to [55], this lightweight design allows the network to accommodate
more frames with the same computational budget. Each 2D pose in the input se-
quence is independently sent to the MLP block whose weights are shared across
all frames.

8
Temporal Encoding Module (TEM). As 2D keypoints are used as inputs to
the network, the amount of data per frame is small. Thus, we can take advantage
of the extra-long 2D input pose sequence (e.g., 243 frames). Since the objective
of this module is to exploit temporal information from the changes of human
posture within this sequence, it is inefficient to focus on the relationships between
highly redundant frames in a local region. Therefore convolution operations,
which introduce the inductive bias of locality, can be discarded. Instead, the
self-attention mechanism is exploited, which allows the network to easily focus
on correlations between frames that are far apart. We use a standard Transformer
architecture as the backbone network of TEM. It is capable of capturing non-
local self-attending associations. TEM takes a sequence of latent features from
SEM as input and treats the features in each frame as an individual token. As
the inputs are already embe

## method
PCK↑AUC↑MPJPE↓
Mehta et al. [32] 3DV’17 (N=1)
75.7
39.3
117.6
Pavllo et al. [37] CVPR’19 (N=81)
86.0
51.9
84.0
Lin et al. [26] BMVC’19 (N=25)
83.6
51.4
79.8
Zeng et al. [52] ECCV’20 (N=1)
77.6
43.8
-
Wang et al. [48] ECCV’20 (N=96)
86.9
62.1
68.1
Zheng et al. [55] ICCV’21 (N=9)
88.6
56.4
77.1
Chen et al. [6] TCSVT’21 (N=81)
87.9
54.0
78.8
Hu et al. [19] MM’21 (N=96)
97.9
69.5
42.5
P-STMO (N=81)
97.9
75.8
32.2
Fig. 4. Qualitative comparison between our P-STMO method and Poseformer [55] on
Human3.6M and MPI-INF-3DHP datasets.
quence length in this dataset is shorter than that in Human3.6M dataset, we
set the number of input frames to 81. Our method achieves significant improve-
ments in terms of AUC (9.1%) and MPJPE (24.2%). The PCK metric is at a
competitive level compared to [19]. The results suggest that our method has a
strong generalization ability.
Qualitative Results. We provide some qualitative results in Fig. 4. We com-
pare the proposed method with Poseformer [55] on Human3.6M and MPI-INF-
3DHP datasets. Our method achieves good qualitative performance in both in-
door scenes and complex outdoor scenes.
4.3
Ablation Study
To verify the effectiveness of each component, we conduct ablation experiments
on Human3.6M dataset using our P-STMO-S model. We utilize 2D keypoints
from CPN as inputs. MPJPE is reported for analysis.
Impact of Each Component. As shown in Table 3, we validate the contri-
butions of different modules in STMO model and the overall performance gain
brought by the proposed MPM and TDS methods. A network consisting of only
SEM achieves 51.0 mm under MPJPE. To evaluate the effect of TEM, we use a
vanilla Transformer as the backbone and yield a result of 49.6mm. After com-
bining SEM and TEM, MPJPE drops to 46.0mm. Finally, we end up with the
proposed STMO model by assembling SEM, TEM as well as MOFA, and achieve
44.2mm under MPJPE. Furthermore, MPM and TDS improve upon STMO by

13
Table 3. The effectiveness of different com-
ponents.
SEM TEM MOFA MPM TDS
Params(M) FLOPs(M) MPJPE↓
!
%
%
%
%
1.1
536
51.0
%
!
%
%
%
1.6
769
49.6
!
!
%
%
%
2.2
1094
46.0
!
!
!
%
%
6.2
1482
44.2
!
!
!
!
%
6.2
1482
43.2
!
!
!
!
!
6.2
1482
43.0
Table 4. Analysis on designs of SEM. L
is the number of sub-blocks in MLP.
Backbone
N
Params(M) FLOPs(M) MPJPE↓
fc
243
1.6
769
49.6
Transformer
81
7.2
1218
48.8
MLP (L = 1) 243
2.3
1094
46.0
MLP (L = 2) 243
2.8
1350
45.9
MLP (L = 3) 243
3.3
1608
46.1
0.1
0.3
0.5
0.7
0.9
Temporal masking ratio qT
6
8
10
12
14
16
18
20
Loss (Stage I)
6.62
10.97
11.27
11.89
19.86
Stage I
43.4
43.5
43.6
43.7
43.8
43.9
MPJPE (Stage II)/mm
43.86
43.9
43.45
43.49
43.36
Stage II
(a)
1
3
5
7
10
15
Spatial masking number mS
7
8
9
10
11
12
13
Loss (Stage I)
6.69
6.83
7.38
8.63
9.95
13.17
Stage I
43.4
43.5
43.6
43.7
43.8
MPJPE (Stage II)/mm
43.82
43.6
43.67
43.39
43.66
43.69
Stage II
(b)
0.5
0.6
0.7
0.8
0.9
Spatial temporal masking ratio qST
43.2
43.4
43.6
43.8
44.0
MPJPE (Stage II)/mm
43.06
qT=0.5
qT = 0.6
qT = 0.7
qT = 0.8
qT = 0.9
(c)
Fig. 5. Performance of three different masking strategies. (a) Temporal masking. (b)
Spatial masking. (c) Spatial temporal masking.
1.0mm and 0.2mm respectively. Previous works [37,29] observe decreasing re-
turns when increasing the temporal receptive field, so a small gain (0.2mm)
from TDS is to be expected in the case of 243 input frames.
Analysis on Masking Ratio of Pre-Training. Fig. 5 shows the influence of
the masking ratio of three masking strategies in the pre-training stage. As shown
in Fig. 5a, the optimal ratio is qT = 90% when only temporal masking is used.
This ratio is higher than MAE [16] in CV and BERT [11] in NLP, whose masking
ratios are 75% and 15%, respectively. This is because 2D pose sequences are more
redundant than images and sentences. Since two adjacent poses are very similar,
we need a high masking ratio to increase the difficulty of the task. We observe
a negative correlation between the loss of Stage I and the error of Stage II. This
implies that increasing the difficulty of the MPM task in Stage I does promote
the fitting ability of the encoder, thus aiding the training of Stage II.
The way of spatial masking is different from that of temporal masking. In
the case of temporal masking, the decoder perceives the indices of the masked
frames through positional embeddings. In the case of spatial masking, only the
latent features in each frame are visible to the decoder, so the indices of the
masked joints cannot be inferred directly. For the above reasons, it is clear that
the spatial masking task is more troublesome than the temporal masking task.
Therefore, the optimal masking ratio of the former should be smaller than that
of the latter. Fig. 5b shows that the optimal number is mS = 7 (qS = 41.1%)
when only spatial masking is used. When the masking number is greater than 7,
continuing to increase the difficulty of the task will result in too much noise in
the input data. Subsequently, the encoder is unable to obtain useful information.
We illustrate the effect of combining temporal masking and spatial masking
in Fig. 5c. Since this hybrid masking strategy is more challenging, the spatial

14
Table 5. Analysis on temporal
downsampling rate s. RF is the
temporal receptive field.
N
s RF
Params(M) FLOPs(M) MPJPE↓
27 1 27
4.6
163
48.2
27 3 81
4.6
163
46.8
27 9 243
4.6
163
46.1
81 1 81
5.4
493
45.6
81 3 243
5.4
493
44.1
243 1 243
6.2
1482
43.2
243 2 486
6.2
1482
43.0
Table 6. Analysis on computational complexity.
Top table: taken from [55]. Bottom table: our
implementation.

## experiments
4.1
Datasets and Evaluation Metrics
Human3.6M [21] is the most commonly used indoor dataset, which consists
of 3.6 million frames captured by four 50 Hz cameras. The dataset is divided
into 15 daily activities (e.g., walking and sitting) performed by 11 subjects.
Following [37,42,3], we use 5 subjects for training (S1, S5, S6, S7, S8), and 2
subjects for testing (S9, S11).
We report the mean per joint position error (MPJPE) and Procrustes MPJPE
(P-MPJPE) for Human3.6M dataset. The former computes the Euclidean dis-
tance between the predicted joint positions and the ground truth positions. The
latter is the MPJPE after the predicted results align to the ground truth via a
rigid transformation.
MPI-INF-3DHP [32] is a more challenging dataset with both indoor and
outdoor scenes. The training set comprises of 8 subjects, covering 8 activities.
The test set covers 7 activities, containing three scenes: green screen, non-green
screen, and outdoor. Following [55,6], we train the proposed method using all
activities from 8 camera views in the training set and evaluate on valid frames
in the test set.
We report MPJPE, percentage of correct keypoints (PCK) within 150mm
range, and area under curve (AUC) as the evaluation metrics for MPI-INF-
3DHP dataset.
4.2
Comparison with State-of-the-Art Methods
Results on Human3.6M. We compare our P-STMO with existing state-of-
the-art methods on Human3.6M dataset. Following [55,25,37,29], we use CPN [8]
as the 2D keypoint detector, and then train our networks on the detected 2D
pose sequence. As shown in Table 1 (top and middle), our method achieves
promising results under both MPJPE (42.8mm) and P-MPJPE (34.4mm) when

11
Table 1. Results on Human3.6M in millimeter under MPJPE and P-MPJPE.
Top&Middle table: 2D poses detected by CPN are used as inputs. Bottom table: the
ground truth of 2D poses are used as inputs. N is the number of input frames. (*)
- uses the refining module proposed in [3]. The best result is shown in bold, and the
second-best result is underlined.
MPJPE (CPN)
Dir. Disc. Eat Greet Phone Photo Pose Pur.
Sit
SitD. Smoke Wait WalkD. Walk WalkT. Avg
Pavllo et al. [37] CVPR’19 (N=243) 45.2 46.7 43.3
45.6
48.1
55.1
44.6 44.3 57.3
65.8
47.1
44.0
49.0
32.8
33.9
46.8
Lin et al. [26] BMVC’19 (N=50)
42.5 44.8 42.6
44.2
48.5
57.1
42.6 41.4 56.5
64.5
47.4
43.0
48.1
33.0
35.1
46.6
Xu et al. [49] CVPR’20 (N=9)
37.4 43.5 42.7
42.7
46.6
59.7
41.3 45.1 52.7 60.2
45.8
43.1
47.7
33.7
37.1
45.6
Wang et al. [48] ECCV’20 (N=96)
41.3 43.9 44.0
42.2
48.0
57.1
42.2 43.2 57.3
61.3
47.0
43.5
47.0
32.6
31.8
45.6
Liu et al. [29] CVPR’20 (N=243)
41.8 44.8 41.1
44.9
47.4
54.1
43.4 42.2 56.2
63.6
45.3
43.5
45.3
31.3
32.2
45.1
Zeng et al. [52] ECCV’20 (N=243)
46.6 47.1 43.9
41.6
45.8
49.6
46.5 40.0 53.4
61.1
46.1
42.6
43.1
31.5
32.6
44.8
Zeng et al. [53] ICCV’21 (N=9)
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
45.7
Zheng et al. [55] ICCV’21 (N=81)
41.5 44.8 39.8
42.5
46.5
51.6
42.1 42.0 53.3
60.7
45.5
43.3
46.1
31.8
32.2
44.3
Shan et al. [42] MM’21 (N=243)
40.8 44.5 41.4
42.7
46.3
55.6
41.8 41.9 53.7
60.8
45.0
41.5
44.8
30.8
31.9
44.3
Chen et al. [6] TCSVT’21 (N=243) 41.4 43.5 40.1
42.9
46.6
51.9
41.7 42.3 53.9
60.2
45.4
41.7
46.0
31.5
32.7
44.1
Hu et al. [19] MM’21 (N=96)
38.0 43.3 39.1 39.4
45.8
53.6
41.4 41.4 55.5
61.9
44.6
41.9
44.5
31.6
29.4
43.4
Li et al. [25] TMM’22 (N=351)(*)
39.9 43.4 40.0
40.9
46.4
50.6
42.1 39.8 55.8
61.6
44.9
43.3
44.9
29.9
30.3
43.6
P-STMO-S (N=81)
41.7 44.5 41.0
42.9
46.0
51.3
42.8 41.3 54.9
61.8
45.1
42.8
43.8
30.8
30.7
44.1
P-STMO-S (N=243)
40.0 42.5 38.3 41.5
45.8
50.8
41.6 40.9 54.2
59.3
44.4
41.9
43.6
30.3
30.1
43.0
P-STMO (N=243)
38.9 42.7 40.4
41.1
45.6
49.7
40.9 39.9 55.5
59.4
44.9
42.2
42.7
29.4
29.4
42.8
P-STMO (N=243)(*)
38.4 42.1 39.8
40.2
45.2
48.9 40.4 38.3 53.8 57.3
43.9
41.6
42.2
29.3
29.3
42.1
P-MPJPE (CPN)
Dir. Disc. Eat Greet Phone Photo Pose Pur.
Sit
SitD. Smoke Wait WalkD. Walk WalkT. Avg
Lin et al. [26] BMVC’19 (N=50)
32.5 35.3 34.3
36.2
37.8
43.0
33.0 32.2 45.7
51.8
38.4
32.8
37.5
25.8
28.9
36.8
Pavllo et al. [37] CVPR’19 (N=243) 34.1 36.1 34.4
37.2
36.4
42.2
34.4 33.6 45.0
52.5
37.4
33.8
37.8
25.6
27.3
36.5
Xu et al. [49] CVPR’20 (N=9)
31.0 34.8 34.7
34.4
36.2
43.9
31.6 33.5 42.3 49.0
37.1
33.0
39.1
26.9
31.9
36.2
Liu et al. [29] CVPR’20 (N=243)
32.3 35.2 33.3
35.8
35.9
41.5
33.2 32.7 44.6
50.9
37.0
32.4
37.0
25.2
27.2
35.6
Wang et al. [48] ECCV’20 (N=96)
32.9 35.2 35.6
34.4
36.4
42.7
31.2 32.5 45.6
50.2
37.3
32.8
36.3
26.0
23.9
35.5
Chen et al. [6] TCSVT’21 (N=243) 33.1 35.3 33.4
35.9
36.1
41.7
32.8 33.3 42.6
49.4
37.0
32.7
36.5
25.5
27.9
35.6
Shan et al. [42] MM’21 (N=243)
32.5 36.2 33.2
35.3
35.6
42.1
32.6 31.9 42.6 47.9
36.6
32.1
34.8
24.2
25.8
35.0
Zheng et al. [55] ICCV’21 (N=81)
32.5 34.8 32.6 34.6
35.3
39.5
32.1 32.0 42.8
48.5
34.8
32.4
35.3
24.5
26.0
34.6
P-STMO (N=243)
31.3 35.2 32.9 33.9
35.4
39.3
32.5 31.5 44.6
48.2
36.3
32.9
34.4
23.8
23.9
34.4
MPJPE (GT)
Dir. Disc. Eat Greet Phone Photo Pose Pur.
Sit
SitD. Smoke Wait WalkD. Walk WalkT. Avg
Pavllo et al. [37] CVPR’19 (N=243) 35.2 40.2 32.7
35.7
38.2
45.5
40.6 36.1 48.8
47.3
37.8
39.7
38.7
27.8
29.5
37.8
Lin et al. [26] BMVC’19 (N=50)
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
32.8
Liu et al. [29] CVPR’20 (N=243)
34.5 37.1 33.6
34.2
32.9
37.1
39.6 35.8 40.7
41.4
33.0
33.8
33.0
26.6
26.9
34.7
Zeng et al. [52] ECCV’20 (N=243)
34.8 32.1 28.5 30.7
31.4
36.9
35.6 30.5 38.9
40.5
32.5
31.0
29.9
22.5
24.5
32.0
Chen et al. [6] TCSVT’21 (N=243)
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
32.3
Zheng et al. [55] ICCV’21 (N=81)
30.0 33.6 29.9
31.0
30.2
33.3
34.8 31.4 37.8
38.6
31.7
31.5
29.0
23.3
23.1
31.3
Shan et al. [42] MM’21 (N=243)
29.5 30.8 28.8
29.1
30.7
35.2
31.7 27.8 34.5 36.0
30.3
29.4
28.9
24.1
24.7
30.1
P-STMO (N=243)
28.5 30.1 28.6 27.9
29.8
33.2 31.3 27.8 36.0
37.4
29.7
29.5
28.1
21.0
21.0
29.3
using 243 frames as inputs. We also train our model using the same refining
module as [25,3,48]. It achieves 42.1mm under MPJPE, which surpasses all other
methods. Our method yields better performance on hard poses (such as Photo
and SittingDown) than the previous works. This demonstrates the robustness of
our model in the case of depth ambiguity and severe self-occlusion. Additionally,
we propose a smaller model P-STMO-Small. The only difference between P-
STMO and P-STMO-S lies in the number of Transformer layers. For more details,
please refer to the supplementary materials. Compared with P-STMO, P-STMO-
S achieves 43.0mm MPJPE with a smaller number of FLOPs and parameters. To
explore the lower bound of the proposed method, we utilize the ground truth of
2D poses as inputs. In this way, the input noise is removed. As shown in Table 1
(bottom), P-STMO outperforms other methods with 29.3mm under MPJPE.
Results on MPI-INF-3DHP. Table 2 reports the performance of state-of-
the-art methods and the proposed method on MPI-INF-3DHP dataset. Follow-
ing [19,48,55], we adopt the ground truth of 2D poses as inputs. Since the se-

12
Table 2. Results on MPI-INF-3DHP under three evaluation metrics.

## related_work
2.1
3D Human Pose Estimation
Recently, there is a research trend that uses 2D keypoints to regress correspond-
ing 3D joint positions. The advantage is that it is compatible with any existing
2D pose estimation method. Our approach falls under this category. Extensive
works [49,47,43,31,36,14,33] have been carried out around an important issue
in videos, which is how to exploit the information in spatial and temporal do-
mains. Some works [43,31,53,14] only focus on 3D single-frame pose estimation
and ignore temporal dependencies. Other recent approaches [49,55,48,3,17] ex-
plore the way of integrating spatio-temporal information. The following are the
shortcomings of these methods. Approaches relying on recurrent neural net-
work (RNN) [17,23] suffer from high computational complexity. Graph convolu-
tional network (GCN)-based methods [48,19] perform graph convolutions on the
spatial temporal graph, and predict all poses in the sequence. This diminishes
the capability of the network to model the 3D pose in a particular frame. The
Transformer-based method [55] predicts the 3D pose of the current (middle)
frame by performing a weighted average at the last layer over the features of
all frames in the sequence. It ignores the importance of the current pose and its
near neighbors.
On the other hand, some works [37,29,6] focus on the process of many-
to-one frame aggregation. Pavllo et al. [37] first propose a temporal convolu-
tional network (TCN) that uses multiple frames to aid the modeling of one
frame by progressively reducing the temporal dimension. Since then many meth-
ods [52,6,42,29] have been proposed based on TCN. However, these methods do
not explicitly extract spatial and temporal features. Li et al. [25] alleviate this
problem by using a vanilla Transformer to capture long-range temporal depen-
dencies in the sequence. But they acquire spatial information by a single fc layer,
which has insufficient representation capability.

4
In contrast, we propose a Spatial Temporal Many-to-One (STMO) model.
The most important steps in 3D human pose estimation are represented as three
modules in STMO: SEM, TEM, and MOFA. Our method clearly delineates the
responsibility of each module and promotes the modeling of intrinsic properties.
2.2
Pre-Training of Transformer
Transformer [45] has become the de facto backbone in NLP [11,2,40,10,39] and
CV [4,12,44,15,7]. Transformer owes its widespread success to the pre-training
technique [51,38,11,1,50]. In NLP, Devlin et al. [11] propose a masked language
modeling (MLM) task, which triggers a wave of research on pre-training. They
randomly mask some words and aim to predict these masked words based only
on their context. Subsequent developments in CV have followed a similar trajec-
tory to NLP. Some works [12,16,1] transfer self-supervised pre-trained models to
image-based CV tasks, such as classification, object detection, semantic segmen-
tation, etc. They replace the words to be reconstructed in MLM with pixels or
discrete visual tokens. The pre-training task in CV is termed as masked image
modeling (MIM).
Inspired by these works, we apply the pre-training technique to 3D human
pose estimation and propose a masked pose modeling (MPM) task similarly. We
randomly mask some joints in spatial and temporal domains and try to recover
the original 2D poses. Note that METRO [27] also masks the input 2D poses,
but the goal is to directly regress 3D joint positions rather than recovering the
input. This approach is essentially a data augmentation method and therefore
different from the proposed pre-training task.
3

## conclusion
In this paper, we present P-STMO, a Pre-trained Spatial Temporal Many-to-
One model for 3D human pose estimation. A pre-training task, called MPM, is
proposed to enhance the representation capability of SEM and TEM. For SEM,
an MLP block is used as the backbone, which is more effective than Transformer
and fc. For TEM, a temporal downsampling strategy is introduced to mitigate
input data redundancy and increase the temporal receptive field. Comprehensive
experiments on two datasets demonstrate that our method achieves superior
performance over state-of-the-art methods with smaller computational budgets.
A
Supplementary Material
A.1
Other Related Work
Hu et al. [18] also explores the spatial temporal masking strategy in the field
of sign language recognition. In addition to being applied in different fields, our
work differs from theirs in the following ways: 1) They set the coordinates of the
masked joints to (0,0), which is a very common position (center of the image).
Thus, this approach will lead to confusion between the masked and unmasked
joints. In contrast, we replace the masked joints with learnable vectors. These
special identifiers enable the model to distinguish whether a joint is masked or
not. 2) They take all 2D poses as inputs, while we only take 2D unmasked poses
as inputs, which greatly reduces the computational complexity of the encoder.
Liu et al. [30] learn 2D human pose embeddings for downstream tasks, which
is similar to our work. Our method differs from theirs in these ways: 1) They
aim to learn view-invariant embeddings by metric learning, while we aim to learn
spatial temporal embeddings by solving the MPM task. 2) The masking strategy
is utilized to increase the difficulty of pre-training in our method, while they use
it to simulate the real-world occlusion situation. 3) We use Transformer, whose

16
token design is more suitable for masking. 4) We explore the spatial temporal
masking strategy, while they do not.
A.2
Implementation Details
All experiments are carried out on a single GeForce GTX 3080Ti GPU. The
proposed method is implemented using PyTorch [35]. For both stages, we train
our model using the Adam [22] optimizer for 80 epochs. The initial learning rate
is 1e−4 for Stage I and 7e−4 for Stage II. The learning rate decays by 3% after
each epoch. The batch size is set to 160. We use horizontal flipping as the data
augmentation approach during training and testing. The balance factor λ in the
loss function is set to 1 empirically.
For SEM, we utilize an MLP block as the backbone. The number of sub-
blocks is L = 1. The dimension of the latent features is set to 256. For TEM
as well as the decoder in Stage I, we utilize a vanilla Transformer [45] as the
backbone. The depth of Transformer for these two modules is set to 3,2 (for
P-STMO-S) or 4,3 (for P-STMO), respectively. Besides, the dimension of the
QKV matrices is 256. The number of heads in self-attention is 8. For MOFA,
STE [25] is used as the backbone. The basic settings are the same as TEM. For
N = 27, 81, 243, the depth of Transformer is set to 3, 4, 5, respectively. The
stride of 1D convolution is set to 3 for each Transformer layer.
A.3
Analysis on Model Hyperparameters
As shown in Table 7, we mainly investigate three hyperparameters in our method:
the depth of TEM (L1), the depth of the decoder in Stage I (LD), and the
dimension of latent features of all Transformers (d). Following [16], we adopt an
asymmetric encoder-decoder design. Unlike classical auto-encoders, the depth
of the decoder can be different from that of the encoder. Since we only care
about the speed of inference, the number of parameters and FLOPs of the pre-
training stage are not included in the calculation results. Thus, these results are
not affected by different settings of LD. The best performance is achieved when
L1 = 4, LD = 3, d = 256.
A.4
Reconstruction Results in the Pre-Training Stage
In Fig. 6, we give some qualitative results of the 2D pose reconstruction in the
pre-training stage. The 2D input sequence is masked with qT = 0.8, mS = 2,
which means only 18.7% of the joints are visible to the network. It can be seen
that the network is able to perform a rough recovery of the 2D poses in the
original sequence by virtue of only a small number of visible joints.
A.5
Visualization of Multi-Head Self-Attention
We perform a subjective analysis of the self-attention mechanism in Transformer
in both stages. The results are shown in Fig. 7-10. The x-axis (horizontal) and

17
Input
MPM_reconstruction
(a) Directions
Input
MPM_reconstruction
(b) Walking
Input
MPM_reconstruction
(c) SittingDown
Input
MPM_reconstruction
(d) Photo
Fig. 6. Qualitative results of the 2D pose reconstruction in Stage I.
y-axis (vertical) correspond to the index of key (K) and query (Q) in the multi-
head self-attention respectively. The output is the normalized attention weight.
We set the input frame number to N = 243.
Stage I. We set the temporal masking ratio to qT = 0.8 and spatial masking
number to mS = 2. Therefore, the number of unmasked frames is a = (1 −qT) ·
N = ⌊48.6⌋= 48. The number of masked frames is b = N −a = 195. The
unmasked frames are fed to the encoder (SEM+TEM). The attention maps of
TEM are shown in Fig. 7. Since the input is randomly masked, two adjacent
frames in the unmasked frame sequence are most likely to be non-adjacent in
the original sequence. Our goal is to recover all the masked frames between two
adjacent frames in the unmasked frame sequence. Therefore, most heads of TEM
focus on aggregating neighbouring information around the query frame. Others
(e.g., head 6) capture long-term dependencies.
The attention maps of the decoder are shown in Fig. 8. An obvious cross can
be seen in each attention map because we utilize the same implementation as [16].
Specifically, to improve the efficiency, we append a list of temporal padding
embeddings after the encoded unmasked embeddings at the decoder side. Then,
we add positional embeddings to these padding embeddings to restore their
original positions. Therefore, the first half of the sequence (before frame 48)
behaves differently from the second half (after frame 48). Each attention map
can be divided into four matrices as follows. 1) The matrix D1 ∈Ra×a in the
upper left corner measures the self-attention of unmasked frames. The patterns
are very similar to those in Fig. 7. 2) The matrix D2 ∈Ra×b in the upper
right corner measures the degree of attention paid to the masked frames by the

18
Table 7. Analysis on model hyperparameters. Only the number of parameters and
FLOPS of Stage II are included in the calculation results. L1, LD are the depth of
TEM and the decoder in Stage I respectively. d is the dimension of latent features of
all Transformers.
L1 LD
d
Params(M) FLOPs(M) MPJPE↓
4
2
256
6.7
868.5
43.0
4
3
256
6.7
868.5
42.8
4
4
256
6.7
868.5
42.9
4
5
256
6.7
868.5
43.3
2
3
256
5.7
613.7
44.4
3
3
256
6.2
741.4
43.1
4
3
256
6.7
868.5
42.8
5
3
256
7.3
995.9
43.4
4
3
64
1.1
121.8
44.9
4
3
128
2.6
307.0
43.7
4
3
256
6.7
868.5
42.8
4
3
512
19.5
2755.1
45.5
unmasked frames. The values of most heads are 0, which means that the masked
frames provide little help to the unmasked ones. A small number of heads behave
differently (e.g., head 0), which can be explained by the need to interpret the
continuity of the entire pose sequence. 3) The matrix D3 ∈Rb×a in the bottom
left corner measures the degree of attention paid to the unmasked frames by
the masked frames. For a particular masked frame, it resorts to the unmasked
frames that are close to it. As a result, some heads (e.g., head 4) exhibit a
locally relevant pattern. In addition, non-local patterns can also be observed
in other heads (e.g., head 3). 4) The matrix D4 ∈Rb×b in the bottom right
corner measures the self-attention of masked frames. Since the masked frames
mainly obtain information from the unmasked frames, the self-attention pattern
of masked frames is not pronounced. Nevertheless, we can still find some patterns
learned by a small number of heads, such as capturing long-term dependencies
(e.g., head 0) and short-term dependencies (e.g., head 1).
Stage II. After the pre-trained model is loaded into STMO, TEM is tuned
to acquire long- and short-term information that contributes to the overall 3D
pose sequence prediction. As illustrated in Fig. 9, Transformer learns a variety
of patterns. The local patterns (e.g., head 0) focus on a small region around the
query frame, while the global patterns (e.g., head 5) exploit relevant features
over a longer time span.
As we use STE [25], a Transformer-based method, as the backbone network
of MOFA, its attention maps can also be visualized in Fig. 10. For N = 243,
the depth of Transformer is 5. Only the first 4 layers are shown. STE uses 1D
convolution to reduce the temporal dimension layer by layer. As we set the stride
to 3, the number of frames in each layer is reduced to 1/3 of that in the previous

19
Table 8. Anal