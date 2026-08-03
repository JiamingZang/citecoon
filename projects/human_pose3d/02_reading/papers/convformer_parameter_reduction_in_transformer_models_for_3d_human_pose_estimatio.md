# ConvFormer: parameter reduction in transformer models for 3D human pose estimation by leveraging dynamic multi-headed convolutional attention

> 2023 · id: W4382892987 · arXiv: 2304.02147 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recently, fully-transformer architectures have replaced the defacto convolutional architecture for the 3D human pose
estimation task. In this paper we propose ConvFormer, a novel convolutional transformer that leverages a new dynamic
multi-headed convolutional self-attention mechanism for monocular 3D human pose estimation. We designed a spatial
and temporal convolutional transformer to comprehensively model human joint relations within individual frames and
globally across the motion sequence. Moreover, we introduce a novel notion of temporal joints proﬁle for our temporal
ConvFormer that fuses complete temporal information immediately for a local neighborhood of joint features. We have
quantitatively and qualitatively validated our method on three common benchmark datasets: Human3.6M, MPI-INF-3DHP,
and HumanEva. Extensive experiments have been conducted to identify the optimal hyper-parameter set. These experiments
demonstrated that we achieved a signiﬁcant parameter reduction relative to prior transformer models while
attaining State-of-the-Art (SOTA) or near SOTA on all three datasets. Additionally, we achieved SOTA for Protocol III on
H36M for both GT and CPN detection inputs. Finally, we obtained SOTA on all three metrics for the MPI-INF-3DHP
dataset and for all three subjects on HumanEva under Protocol II.
1

## introduction
Figure 1: Panel A depicts architecture of a ConvFormer Block. Panel B presents the overall pipeline for 3D HPE from a sequence of 2D
poses. The central component of a ConvFormer Block is DMHCSA which is depicted in the panel C. A curvy blue line at the bottom of
Panel C corresponds to a part of an extracted temporal joints proﬁle of the right elbow joint (for the temporal ConvFormer block).
Panel D presents an example of convolutions during generation of Queries, Keys, and Values in a Temporal ConvFormer Block. A ﬁlter
slides across the feature dimension eﬀectively convolving full temporal proﬁles of local joint neighborhoods.
The overall architecture of our methodology is described in Figure 1. Given a sequence of 2D poses P = {Pi}T
i=1 ⊂RJ×2
where T represents the number of frames in the sequence and J is the number of joints in the skeleton. We seek to reconstruct
the 3D poses in the root relative camera reference frame (i.e. the camera reference frame where the root joint sits at
the origin). Following [2], we predict the 3D pose for the central frame from any such sequence, i.e. ˆp⌈i/2⌉∈RJ×3. Our
network contains two Dynamic ConvFormer blocks, one with spatial attention and the other with temporal attention. More
speciﬁcally, we leverage a spatial attention mechanism to extract frame-wise inter-joint dependencies by analyzing sections of
joints that are related. The temporal attention mechanism extracts global inter-frame relationships by analyzing correlations
between the temporal proﬁles of joints. In contrast to [5], which queries latent pose representations for individual frames
and then computes attention with respect to the temporal axis, our temporal joints proﬁle mechanism fuses temporal
information at the querying level prior to computing self-attention with respect to the temporal axis.
3

3.2
Network Architecture
We employ two main components in our network architecture: a spatial and a temporal ConvFormer. The spatial ConvFormer
block extracts a high dimensional feature vector for a single-frames’ encoded joint correlations. We assume our input is a
2D pose with J joints that is represented by two coordinates (u, v). Following [5] we ﬁrst map the coordinate of each joint
into a higher-dimensional feature vector with a trainable linear layer. We then apply a learned positional encoding via
summation to retain joint position information. That is, given a sequence of poses {Pi}T
i=1 ⊂RJ×2 and W ∈R2×d and
Epos ∈RJ×d we encode Pi as follows:
xi = PiW + Epos,
i ∈{1, ..., T}.
(1)
and d represents the dimension of the embedding, W is the trainable linear layer, and Epos is the learned positional encoding.
Subsequently, the spatial feature sequence {xi}T
i=1 ⊂RJ×d are fed into the spatial ConvFormer which applies the attention
mechanism to the joint dimension to integrate information across the complete pose on a per frame basis. Q, K, V are
generated via convolutions with weights of the following dimension (d, d, k) where d is the encoded dimension and k is the
kernel size and the ﬁlter is slid over the joints dimension. The output for the i-th frame of the b-th spatial ConvFormer
block is denoted by zb
i ∈RJ×d for i = 1, ..., T.
While the spatial ConvFormer seeks to encode correlations between joints in a single frame we leveraged the temporal
model to localize sequence wise correlations between the encoded spatial features. This mechanism should be viewed
as extracting the temporal proﬁle of a neighborhood of joints, which we call temporal joints proﬁle (see Panel D in
Figure 1). An early work that leveraged this temporal fusion mechanism was [4] where Karpathy et al. studied diﬀerent
mechanisms for incorporating temporal information without convolving over the temporal dimension. To further clarify
the point, Q, K, V are generated via convolutions with weights of the following dimension (T, T, k) where k is the kernel
size and the 1D convolutions have depth the size of input sequence. Thus, one can view our network as fusing into the
queries the temporal evolution of a patch of deep joint features immediately. This is very distinct from the temporal
attention seen in [5] which attends complete pose encoding throughout the motion sequence. We note that the output
from the spatial ConvFormer block is a sequence {zB
i }i=1,...,T ⊂RJ×d where B is the number of spatial blocks and T is
the number of frames in the sequence. We note that zb
i can be represented in R1×J·d and thus concatenate these features
along the ﬁrst axis giving us X0 = Concatenate(zB
1 , ..., zB
T ) ∈RT ×J·d. Following this procedure we incorporate a learned
temporal embedding to retain information about the deep joint features evolution throughout time, i.e. Etemp ∈RT ×J·d
and X = X0 + Etemp is the input into our temporal transformer. We note that the output of the b-th ConvFormer block
with temporal attention is Zb ∈RT ×J·d where there are B such layers.
Since we follow many-to-one prediction scheme ﬁrst introduced in [2] we ﬁrst down sample the spatial axis with a linear
projection and then perform a temporal convolution with one output channel i.e. ˆp = ConvT,1(ZBW) where W ∈RJ·d×3J
and ConvT,1 denotes a temporal convolution with one output channel and T input channels.
We trained our network by minimizing the MPJPE (Mean Per Joint Position Error) during optimization. The loss
function is deﬁned as
L(p, ˆp) = 1
J
J
X
i=1
∥pi −ˆpi∥2
(2)
where p is the ground truth 3D pose and ˆp is the predicted pose and i is indexing speciﬁc joints in the skeleton.
3.3
Dynamic Multi-Headed Convolutional Self-Attention
A core novelty of this paper is the dynamic multi-headed convolutional self-attention mechanism. This is introduced to
reduce the over connectedness witnessed in classic transformer architectures while simultaneously extracting contexts at
diﬀerent scales. An additional novelty is the type of representations being queried in our temporal ConvFormer block.
Instead of generating queries, keys, and values, that are latent pose representations for individual frames and attending the
temporal axis; we query temporal joints proﬁles eﬀectively fusing temporal information prior to the attention mechanism.
Convolutional Scaled Dot Product Attention can be described as a mapping function that maps a query matrix Q, a
key matrix K, and a value matrix V to an output attention matrix – where the matrix entries are scores representing the
strength of correlation between any two elements in the dimension being attended. We note that Q, K, V ∈RN×d where N
is the length of the sequence and d is the dimension. In our Spatial ConvFormer N = J and in the Temporal ConvFormer
N = T. The output of the scaled dot product attention can be expressed as
Attention(Q, K, V ) = Softmax(QKT /
√
d)V .
(3)
The query, keys, and values, are computed in the same manner for a ﬁxed ﬁlter length. We demonstrate how Q can be
generated, and note that K and V are computed in an identical manner.
Q = Convn,dout(z) =
din
X
i=1
κ
X
k=1
wdout,i,k · zi,n−κ−1
2
+k
(4)
Here, κ denotes the kernel size and dout denotes output dimension. This is juxtaposed against the classic scaled dot product
attention introduced in [12] where queries, keys, and values are generated via a linear projection
Q = WQz
K = WKz
V = WV z
(5)
which provides global scope but causes redundancy due to the complete connectivity. In our dynamic convolutional
attention mechanism we introduce sparsity via convolutions to decrease connectivity while simultaneously fusing complete
4

temporal information prior to the scaled-dot-product-attention. ConvFormers’ ability to provide context at diﬀerent scales
is attributable to the dynamic feature aggregation method. Moreover, due to our convolution mechanism we query on
inter-frame level where we learn the temporal joints proﬁle. To this end, we use n convolutional ﬁlter sizes to extract
diﬀerent local contexts at scales {κi}n
i=1 and then perform an averaging operation to generate the ﬁnal query, keys, and
values that we apply attention to, following ideas presented in [36]:
Q = Concat(Q1, ..., Qn)ηQ =
n
X
i=1
ηQ(i)Qi
where
n
X
i=1
ηQ(i) = 1
(6)
where n is the number of convolution ﬁlters used, ηQ ∈Rn×1 is a learned parameter and Qi are generated as in equation 4.
Dynamic Multi-headed Convolutional Self-Attention (DMHCSA) leverages multiple heads to jointly model information
from multiple representation spaces. As seen in Figure 1 each head applies scaled dot-product self-attention in parallel. The
output of the DMHCSA block is the concatenation of h attention head outputs fed into a feed-forward network.
DMHCSA(Q, K, V ) = Concatenate(H1, ..., Hh)
where
Hi = Attention(Qi, Ki, Vi),
i ∈{1, ..., h}
(7)
where Qi, Ki, and Vi are computed via the procedure deﬁned above.
Then the ConvFormer block is deﬁned by the following equations:
X
′
b = DMHCSA(LN(Xb−1)) + Xb−1,
b = 1, ..., B
Xb = FFN(LN(X
′
b)) + X
′
b,
b = 1, ..., B
(8)
where LN(·) denotes layer normalization same as [21, 55]. and FFN de

## method
PCK ↑
AUC ↑
MPJPE (mm) ↓
Subject
S1
S2
S3
S1
S2
S3
S1
S2
S3
[10]
75.7
39.3
117.6
Martinez et al. [1]
19.7
17.4
46.8
26.9
18.2
18.6
–
–
–
Lin et al. [33]
83.6
51.4
79.8
Pavalkos et aal. [9]
22.3
19.5
29.7
28.9
21.9
23.8
–
–
–
Pavllo et al. [2]
86.0
51.9
84.0
Pavllo et al. [2]
13.9
10.2
46.6
20.9
13.1
13.8
23.8
33.7
32.0
Li et al. [43]
81.2
46.1
99.7
Zheng et al. [5]
16.3
11.0
47.1
25.0
15.2
15.1
–
–
–
Chen et al. [11]
87.6
54.0
78.8
ConvFormer (T=9)
12.5
10.1
25.4
13.3
12.9
22.6
31.7
28.6
29.0
Zheng et al. [5]
88.6
56.4
77.1
ConvFormer (T=27)
11.4
9.0
20.1
19.1
11.8
11.8
20.8
28.0
26.1
Li et al. [68]
93.8
63.3
58.0
ConvFormer (T=43)
10.7
7.9
16.0
16.7
9.3
10.0
18.2
25.0
24.3
ConvFormer
96.4
69.8
53.6
We report results for our 143 and 243 frame models on H3.6M and we report results for our 9, 27, and 43 frame model
for HumanEva. We report all 15 action results for both subjects S9 and S11 using GT and CPN detections as the 2D input
under protocol I, II, and III in Table. 1 and the last column represents the average. ConvFormer’s 143 and 243-frame
models substantially reduce the parameter count by 83.4% and 65.5% respectively, relative the previous
SOTA [68]. ConvFormer’s 143 and 243-frame models outperforms the previous SOTA on GT inputs – achieving a 2.3%
reduction of error. ConvFormer’s 243-frame model misses SOTA on CPN inputs for Protocol I by 0.2mm while having
7

substantially lowered parameters and achieving best or second best on 11 of the 15 actions. However, it outperforms the
SOTA on some challenging actions such as Sitting and WalkingDog which exhibit complex postures and rapid postural
changes. Under Protocol II ConvFormer achieves SOTA on 9 individual actions and on the average error. Lastly, for both
GT and CPN inputs ConvFormer reduces the MPJVE by 8.6% and 14.3% respectively, resulting in smoother predictions.
See Figure 2 for some qualitative results on H36M or see https://github.com/AJDA1992/ConvFormer for more examples
from challenging in-the-wild motions.
The left side of Table 2 shows the results of training ConvFormer from scratch on HumanEva. We note that our larger
receptive ﬁeld model, with 43 frames, achieves SOTA for every action, while our 27 frame receptive ﬁeld model achieves
second place for every action.
The right side of Table 2 reports the quantitative results of ConvFormer on MPI-INF-3DHP relative to other methods.
Following [5, 68], we use 2D pose sequences of 9 frames due to fewer samples and shorter video sequences. We note that
ConvFormer increases PCK by 2.7%, AUC by 10.2%, and decreases MPJPE by 7.6%.
5.2
Ablations
First, we study the contribution of individual hyper-parameters and tune them. Second, we assess the contribution
of convolutional self-attention relative to the baseline (vanilla transformer) and then the contribution of our dynamic
self-attention mechanism. To the ﬁrst point, we perform an extensive grid search using [3] and report some of the results in
Tables 3, 4. We ﬁne the following hyper-parameters to be optimal: d = 32, Bsp = Btemp = 2 and using the following kernel
sizes (7, 7, 7).
In Table 5 we analyze the eﬀect of receptive ﬁeld alongside parameter counts relative to other transformer based
methods. We ﬁx the optimal hyper-parameters found in Tables 3, 4. We ﬁnd across all receptive ﬁelds, ConvFormer reduces
parameters substantially relative [71, 5] while remaining extremely competitive on CPN inputs for Protocol I.
Finally, we analyze what improvement ConvFormer brings relative to a vanilla transformer architecture and the beneﬁt
of you using our Dynamic Multi-Headed attention mechanism. In Table 6 our baseline model following the same architecture
as ConvFormer except with class scaled dot product attention and fully-connected layers generating the queries, keys, and
values. We ﬁnd by using a single ﬁlter in our ConvFormer architecture improves on the baseline by 2mm and introducing
our Dynamic Multi-Headed Attention we reduce by another 1.1 mm.
Table 3: Analysis of spatial embedding dimension and number of spatial and temporal ConvFormer Blocks. We also perform a limited
analysis on number of attention heads. Optimal performance is marked by Red and assessed by MPJPE.
d
16
16
16
32
32
32
64
32
32
32
Bsp
2
2
4
2
2
4
2
2
2
2
Btemp
2
4
2
2
4
2
2
2
2
2
Params (M)
0.65
1.26
0.69
2.56
4.95
2.70
9.97
2.56
2.56
2.56
Heads
8
8
8
8
8
8
8
1
2
4
MPJPE (mm)
50.8
51.0
51.7
49.4
50.6
49.9
50.1
52.3
51.0
49.6
6

## experiments
4.1
Datasets and Evaluation Protocols
Our proposed method is evaluated on three common datasets: Human3.6M [7], HumanEva [8], and MPI-INF-3DHP [10].
Human3.6M consists of approximately 2.3 million images from 4 synchronized video cameras capturing video at 50 Hz.
There are 7 subjects performing 15 distinct actions and each action is performed twice per subject. We train on subjects
(S1, S5, S6, S7, S8) and validate on subjects (S9, S11) following previous works [11, 33, 10, 5, 68]. We evaluate our method
on H36M under three diﬀerent protocols. The mean per joint position error (MPJPE) which is referred to as Protocol
I in many works [6, 14, 2]. Procrustes analysis or rigid alignment denoted by P-MPJPE is calculated as the Euclidean
distance between the ground-truth and the optimal SE(3) transformation aligning the predicted pose with the ground-truth.
This is referred to as Protocol II as in [1, 15]. Lastly, we evaluate temporal smoothness via the mean per joint velocity
error, referred to as MPJVE (the mean across joints of the ﬁnite diﬀerence velocity approximations) or Protocol III as in
[2, 16]. HumanEva on the other hand is a much smaller dataset with less than 50k frames and only 3 subjects (S1, S2, S3)
performing three actions. We evaluate our method with respect to Protocol II following previous works (e.g. [2]. Lastly,
we evaluated on MPI-INF-3DHP to assess our model’s generalizability. MPI consists of roughly 1.3 million frames. This
dataset contains more diverse motions than the previous two datasets. Following the setting in [11, 33, 10, 5, 68] we report
the following metrics: MPJPE, Percentage of Correct Keypoint (PCK) with the threshold of 150mm, and Area Under
Curve (AUC) for a range of PCK thresholds.
4.2
Implementation Details
We implemented our proposed solution methodology with PyTorch [17] and trained using two NVIDIA RTX 3090 GPUs.
We trained on H3.6M using 5 diﬀerent frame sequence lengths when conducting our experiments, T = 9, 27, 81, 143, 243.
Following [2] we augment our datasets with ﬂipping poses horizontally. We train our models for 60 epochs with an initial
learning rate of 1e−3 and a weight decay factor of 0.95 after each epoch. We set the batch size to 1024 and utilize stochastic
depth [18] of 0.2. We also use a dropout [22] rate of 0.2 on the dynamic feature aggregation inside of the convolutional
self-attention mechanism. We benchmark on H3.6M using both CPN [24] detections following [2, 11, 16, 5] and ground-truth
2D poses. Furthermore, we benchmark on HumanEva using three diﬀerent frame sequence lengths of T = 9, T = 27, and
T = 43 following [13]. Lastly, following [5, 68] we further assess the generalization ability of our solution methodology
on MPI-INF-3DHP dataset. We use 2D pose sequences of length T = 9 as our model input and we evaluate using three
metrics, percentage of correct keypoints (PCK), area under the curve (AUC), and MPJPE.
5

(a) a
(b) b
(c) c
(d) d
Figure 2: Qualitative examples of S11 from H36M displaying ConvFormer’s eﬀectiveness: (a) Sitting Down action with heavy occlusion
on lower extremities, (b) demonstrates high quality reconstruction in the presence of slight occlusions, (c) heavy occlusion from camera
and ConvFormer still captures the correct pose from previous frame information (d) slight failure case in presence of occlusion from
right arm.
Figure 3: Qualitative results for ConvFormer on challenging In-The-Wild videos.
6

5
Results and Discussion
5.1
Comparison with State-of-the-Art
Table 1: The ﬁrst block reports MPJPE for GT-inputs and the second block is MPJPE for CPN detections. The third block reports
P-MPJPE for CPN detections. The fourth block reports MPJPV for CPN detections and the ﬁfth is MPJPV for GT-inputs. Best is in
Red and second is in Blue.
GT-MPJPE (mm)
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Hossain and Little [15]
35.2
40.8
37.2
37.4
43.2
44.0
38.9
35.6
42.3
44.6
39.7
39.7
40.2
32.8
35.5
39.2
Pavllo et al. [2]
–
–
–
–
–
–
–
–
–
–
–
–
–
–
–
37.8
Liu eet al. [13]
34.5
37.1
33.6
34.2
32.9
37.1
39.6
35.8
40.7
41.4
33.0
33.8
33.0
26.6
26.9
34.7
Zeng et al. [30]
34.8
32.1
28.5
30.7
31.4
36.9
35.6
30.5
38.9
40.5
32.5
31.0
29.9
22.5
24.5
32.0
Chen et al. [11]
–
–
–
–
–
–
–
–
–
–
–
–
–
–
–
32.3
Zheng et al. [5]
30.0
33.6
29.9
31.0
30.2
33.3
34.8
31.4
37.8
38.6
31.7
31.5
29.0
23.3
23.1
31.3
Li et al. [68] (T=351)
27.7
32.1
29.1
28.9
30.0
33.9
33.0
31.2
37.0
39.3
30.0
31.0
29.4
22.2
23.0
30.5
ConvFormer (T=143)
29.1
32.4
28.1
28.5
29.3
33.3
33.3
30.5
37.0
37.6
29.2
29.5
28.4
21.8
21.3
29.9
ConvFormer (T=243)
28.9
31.8
28.0
28.2
29.5
33.0
32.9
30.1
36.8
37.4
29.8
29.6
28.2
21.7
21.5
29.8
CPN-MPJPE (mm)
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Dabral et al. [31]
44.8
50.4
44.7
49.0
52.9
61.4
43.5
45.5
63.1
87.3
51.7
48.5
52.2
37.6
41.9
52.1
Cai et al. [32] (T=7)
44.6
47.4
45.6
48.8
50.8
59.0
47.2
43.9
57.9
61.9
49.7
46.6
51.3
37.1
39.4
48.8
Pavllo et al. [2] (T=243)
45.2
46.7
43.3
45.6
48.1
55.1
44.6
44.3
57.3
65.8
47.1
44.0
49.0
32.8
33.9
46.8
Lin and Lee [33] (T=50)
42.5
44.8
42.6
44.2
48.5
57.1
52.6
41.4
56.5
64.5
47.4
43.0
48.1
33.0
35.1
46.6
Yeh et al. [34]
44.8
46.1
43.3
46.4
49.0
55.2
44.6
44.0
58.3
62.7
47.1
43.9
48.6
32.7
33.3
46.7
Liu et al. [13] (T=243)
41.8
44.8
41.1
44.9
47.4
54.1
43.4
42.2
56.2
63.6
45.3
43.5
45.3
31.3
32.2
45.1
Zeng et al. [30]
46.6
47.1
43.9
41.6
45.8
49.6
46.5
40.0
53.4
61.1
46.1
42.6
43.1
31.5
32.6
44.8
Wang et al. [16] (T=96)
41.3
43.9
44.0
42.2
48.0
57.1
42.2
43.2
57.3
61.3
47.0
43.5
47.0
32.6
31.8
45.6
Chen et al. [11] (T=243)
41.4
43.2
40.1
42.9
46.6
51.9
41.7
42.3
53.9
60.2
45.4
41.7
46.0
31.5
32.7
44.1
Lin et al. [35] (T=1)
–
–
–
–
–
–
–
–
–
–
–
–
–
–
–
54.0
Zheng et al. [5] (T=81)
41.5
44.8
39.8
42.5
46.5
51.6
42.1
42.0
53.3
60.7
45.5
43.3
46.1
31.8
32.2
44.3
Li et al. [68] (T=351)
39.2
43.1
40.1
40.9
44.9
51.2
40.6
41.3
53.5
60.3
43.7
41.1
43.8
29.8
30.6
43.0
ConvFormer (T=143)
41.8
43.6
39.3
43.2
44.9
52.8
42.7
41.2
53.1
60.9
45.0
41.9
44.7
29.7
31.1
43.7
ConvFormer (T=243)
41.0
43.2
39.0
42.4
44.5
52.2
41.7
40.8
53.0
60.6
44.8
41.3
43.7
29.6
30.9
43.2
CPN-P-MPJPE (mm)
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Pavlakos et al. [9]
34.7
39.8
41.8
38.6
42.5
47.5
38.0
36.6
50.7
56.8
42.6
39.6
43.9
32.1
36.5
41.5
Rayat et al. [15]
35.7
39.3
44.6
43.0
47.2
54.0
38.3
37.5
51.6
61.3
46.5
41.4
47.3
34.2
39.4
44.1
Cai et al. [32] (T=7)
35.7
37.8
36.9
40.7
39.6
45.2
37.4
34.5
46.9
50.1
40.5
36.1
41.0
29.6
32.3
39.0
Lin and Lee [33] (T=50)
32.5
35.3
34.3
36.2
37.8
43.0
33.0
32.2
45.7
51.8
38.4
32.8
37.5
25.8
28.9
36.8
Pavllo et al. [2] (T=243)
34.1
36.1
34.4
37.2
36.4
42.2
34.4
33.6
45.0
52.5
37.4
33.8
37.8
25.6
27.3
36.5
Liu et al. [13] (T=243)
32.3
35.2
35.6
34.4
36.4
42.7
31.2
32.5
45.6
50.2
37.3
32.8
36.3
26.0
23.9
35.5
Wang et al. [16] (T=96)
32.9
35.2
35.6
34.4
36.4
42.7
31.2
32.5
45.6
50.2
37.3
32.8
36.3
26.0
23.9
35.5
Chen et al. [11] (T=243)
32.6
35.1
32.8
35.4
36.3
40.4
32.4
32.3
42.7
49.0
36.8
32.4
36.0
24.9
26.5
35.0
Zheng et al. [5] (T=81)
32.5
34.8
32.6
34.6
35.3
39.5
32.1
32.0
42.8
48.5
34.8
32.4
35.3
24.5
26.0
34.6
Li et al. [68] (T=351)
31.5
34.9
32.8
33.6
35.3
39.6
32.0
32.2
43.5
48.7
36.4
32.6
34.3
23.9
25.1
34.4
ConvFormer (T=143)
31.9
34.4
32.2
35.0
34.2
40.7
32.9
31.8
42.8
49.1
36.0
31.5
35.0
23.6
25.2
34.5
ConvFormer (T=243)
31.4
34.2
32.0
35.2
34.0
40.3
32.7
31.3
42.6
49.0
36.2
31.3
34.8
23.4
24.9
34.2
CPN-MPJVE
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Pavllo et al. [2] (T=243)
3.0
3.1
2.2
3.4
2.3
2.7
2.7
3.1
2.1
2.9
2.3
2.4
3.7
3.1
2.8
2.8
Chen et al. [11] (T=243)
2.7
2.8
2.0
3.1
2.0
2.4
2.4
2.8
1.8
2.4
2.0
2.1
3.4
2.7
2.4
2.5
Wang et al. [16] (T=96)
2.3
2.5
2.0
2.7
2.0
2.3
2.2
2.5
1.8
2.7
1.9
2.0
3.1
2.2
2.5
2.3
ConvFormer (T=143)
2.3
2.3
1.8
2.6
1.8
2.1
2.1
2.5
1.4
2.0
1.7
1.9
3.0
2.4
2.1
2.1
GT-MPJVE
Dir.
Disc.
Eat
Greet
Phone
Photo
Pose
Purch.
Sit
SitD.
Smoke
Wait
WalkD.
Walk
WalkT.
Avg.
Wang et al. [16] (T=96)
1.2
1.3
1.1
1.4
1.1
1.4
1.2
1.4
1.0
1.3
1.0
1.1
1.7
1.3
1.4
1.4
ConvFormer (T=143)
1.2
1.3
0.9
1.4
1.0
1.2
1.3
1.5
0.7
1.1
0.9
1.1
1.7
1.4
1.2
1.2
Table 2: Quantitative results on HumanEva under protocol 2 for the left part of the table and quantitative results on MPI-INF-3DHP
in the right part of the table. Best is in Red and second best is Blue.
Action
Walk
Jog
Box

## conclusion
In this paper we attempt to address the ever growing complexity of transformer models. For this, we introduce ConvFormer
which is based on three novel components: temporal fusion, convolutional self-attention, and dynamic feature aggregation.
To assess the eﬀectiveness of diﬀerent components we conducted extensive ablation studies. We reduced the parameter
counts relative to the previous SOTA by over 65% while achieving SOTA on H36M for Protocol I on GT inputs, Protocol
II for CPN detections, Protocol III for both GT and CPN inputs, HumanEva for all subjects, and lastly all three metrics of
MPI. Interestingly, even though graph convolutional networks and graph attention networks are light-weight and robustly
model spatial/temporal relationships, ConvFormer provides a better trade oﬀbetween error reduction and computational
complexity. We believe ConvFormer will provide more ready access to high quality 3D reconstruction networks by making
the training and inference process less computationally demanding.
8

Table 4: Analysis of diﬀerent kernel conﬁgurations where performance is evaluated relative to MPJPE on CPN detections for H36M.
Best marked in Red.
Bsp
Btemp
Kernels
MPJPE (mm)
Params (M)
2
2
3
51.7
2.44
2
2
3,3
51.6
2.46
2
2
3,3,3
50.8
2.48
2
2
5
50.5
2.46
2
2
5,5
52.0
2.49
2
2
5,5,5
51.9
2.52
2
2
7
50.5
2.47
2
2
7,7
50.1
2.51
2
2
7,7,7
49.4
2.56
2
2
9
51.5
2.48
2
2
9,9
50.8
2.54
2
2
9,9,9
50.6
2.60
2
2
3,5,7
50.6
2.52
2
2
5,7,9
51.6
2.56
Table 5: Parameter count and FLOPs results with MPJPE for diﬀerent transformer architectures and graph attention networks
separated by receptive ﬁeld. The last grouping is for models with largest receptive ﬁeld. Best and second best marked with Red and
Blue respectively.
T
Params (M)
FLOPs (M)*
MPJPE (mm)
Zheng et al. [5]
9
9.58
180
49.9
Li et al. [68]
9
19.09
340
47.8
ConvFormer
9
2.56
100
49.4
Zheng et al. [5]
27
9.59
540
47.0
Li et al. [68]
27
19.18
1040
45.9
ConvFormer
27
2.65
360
47.7
Zheng et al. [5]
81
9.60
1620
44.3
Li et al. [68]
81
19.84
3120
44.5
ConvFormer
81
3.43
1600
45.0
Liu et al. [77]
243
7.09
9700
44.9
Li et al. [68]
351
31.52
14160
43.0
ConvFormer
143
5.24
4220
43.7
ConvFormer
243
10.24
10000
43.2