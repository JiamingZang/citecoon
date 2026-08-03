# MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network

> 2024 · id: W4394597906 · arXiv: 2310.16288 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Recent transformer-based approaches have demon-
strated excellent performance in 3D human pose estima-
tion. However, they have a holistic view and by encoding
global relationships between all the joints, they do not cap-
ture the local dependencies precisely.
In this paper, we
present a novel Attention-GCNFormer (AGFormer) block
that divides the number of channels by using two parallel
transformer and GCNFormer streams. Our proposed GCN-
Former module exploits the local relationship between ad-
jacent joints, outputting a new representation that is com-
plementary to the transformer output. By fusing these two
representation in an adaptive way, AGFormer exhibits the
ability to better learn the underlying 3D structure.
By
stacking multiple AGFormer blocks, we propose Motion-
AGFormer in four different variants, which can be chosen
based on the speed-accuracy trade-off. We evaluate our
model on two popular benchmark datasets: Human3.6M
and MPI-INF-3DHP. MotionAGFormer-B achieves state-
of-the-art results, with P1 errors of 38.4 mm and 16.2 mm,
respectively. Remarkably, it uses a quarter of the parame-
ters and is three times more computationally efficient than
the previous leading model on Human3.6M dataset. Code
and models are available at https://github.com/
TaatiTeam/MotionAGFormer.

## introduction
Human pose estimation in 3D space is an active area of
research with significant implications for numerous appli-
cations, from augmented [21] and virtual reality [26] to
autonomous vehicles [2, 9, 43], human-computer interac-
tion [27] and beyond. With this vast range of applications,
the demand for more accurate and computationally effi-
cient pose estimation models continues to grow. In most
real-world scenarios, pose sequences are captured in 2D,
primarily due to the prevalent use of standard RGB cam-
Figure 1. Comparisons of recent pose uplifting methods on Hu-
man3.6M [14] (lower is better). MACs/frame denotes multiply-
accumulate operations per each output frame. The proposed Mo-
tionAGFormer presents different variants and attains superior re-
sults, while maintaining computational efficiency.
eras. Consequently, one of the pivotal challenges in the
field has been to effectively lift these 2D sequences into
a 3D space. Accurate 3D human pose estimation enables
the extraction of rich spatio-temporal information about hu-
man movements, and a deeper understanding of activities
and interactions. Recent 3D lifting models leverage the in-
herent spatial and temporal coherence of human movements
to enhance the precision of 3D pose predictions. Nonethe-
less, despite the considerable advancements, there are sev-
eral significant challenges that require attention.
The Transformer architecture [41], originally designed
for NLP tasks, has been adapted to various computer vision
problems, including pose estimation. Its ability to capture
long-range dependencies and its innate self-attention mech-
anism make it a promising candidate for this domain. How-
ever, a sole reliance on global attention mechanisms, as em-
ployed by standard Transformers, may not be optimal for
1
arXiv:2310.16288v1  [cs.CV]  25 Oct 2023

pose estimation tasks. Human motion is inherently struc-
tured with local spatial and temporal dependencies.
One primary concern is the modeling of skeleton rela-
tions over time. Existing methods predominantly rely ei-
ther on transformer architectures or graph-based models.
While transformers excel at capturing long-term dependen-
cies, graph models excel at local dependencies. So, there is
an opportunity for a unified architecture that integrates the
global perspective of transformers with the local precision
of graph models.
Additionally, the race for achieving SOTA accuracy has
often led to the development of increasingly complex mod-
els with a large number of parameters. Such models, de-
spite their good accuracy, often become impractical for
real-world applications where computational efficiency and
swift response times are pivotal. Moreover, a predominant
approach in recent models has been the prediction of a sin-
gle 3D pose only for the central frame from a sequence of
frames. This method, while seemingly efficient, leads to
computational redundancy as it requires the reprocessing of
several overlapping sequences. As a result, we instead em-
ploy a streamlined inference strategy that optimally exploits
sequential data. This approach minimizes redundancy by
predicting the complete 3D sequence of the input at a single
forward pass.
In this paper, we introduce the MotionAGFormer, a
novel transformer-graph hybrid architecture tailored for 3D
human pose estimation. At its core, the MotionAGFormer
harnesses the power of transformers to capture global in-
formation while simultaneously employing Graph Convolu-
tional Networks (GCNs) to integrate local spatial and tem-
poral relationships. We use an adaptive fusion to aggregate
features from the transformer and graph streams. By doing
so, we ensure a balanced and comprehensive representation
of human motion, leading to enhanced accuracy in the 3D
pose estimation (See Figure 1).
In summary, the main contributions of our paper are:
• Novel Design:
We propose the MotionAGFormer
model, in which we introduce a new GCNFormer mod-
ule that excels in representing local dependencies in-
herent in human pose sequences.
• Efficiency and Flexibility: i) Our MotionAGFormer
stands out due to its lightweight nature and faster
speed with fewer parameters compared to previous
SOTA methods, without compromising on accuracy.
ii) Recognizing diverse needs, we offer different vari-
ants of MotionAGFormer, granting users the flexibility
to make a balanced choice between accuracy and speed
based on their specific requirements.
• SOTA
Performance:
MotionAGFormer
achieves
SOTA on two challenging datasets, Human3.6M and
MPI-INF-3DHP.

## method
3.1. Preliminary
We begin this section by reviewing the concept of
MetaFormer [46], which forms the core of our encoders.
A MetaFormer can be described as a generalization of the
Transformer architecture [41], wherein the attention mod-
ule is substituted with any mixer capable of transform-
ing information among tokens. Specifically, for an input
X ∈RN×C, with N denoting the token numbers and C
representing the embedding dimension, the token mixer can
be formally expressed as
Y = TokenMixer(Norm(X)) + X,
(1)
where Norm(·) denotes a normalization method such as
batch or layer normalization [1,13], and TokenMixer(·) de-
notes a module that combines information among tokens.
Our approach uses two parallel token mixers: Multi-head
Self-Attention (MHSA) and Graph Convolutional Networks
(GCNs) [17], each contributing uniquely to the information
transformation process.
3.2. Overall architecture
Our objective is to lift a 2D (pixel coordinate) skele-
ton sequence to accurate 3D pose sequences. To this end,
we propose the MotionAGFormer architecture, which uses
both attention (Transformer) and graph convolutional (GC-
NFormer) to lift motion sequences. An overview of this
architecture is shown in Figure 2a.
The model takes a 2D input sequence with confidence
score X ∈RT ×J×3, where T and J refer to the number of
frames and joint numbers, respectively. It then proceeds to
map each joint in each time frame to a d-dimensional fea-
ture, F(0) ∈RT ×J×d, using a linear projection layer. Then
a spatial position embedding Ps
pos ∈R1×J×d is added to
the tokens. It is important to highlight here that our model
does not disregard the temporal token order as that informa-
tion is preserved in the GCNFormer stream (further discus-
sion in ablation studies 4.5).
Subsequent to position embedding, we use N blocks
of AGFormer (Section 3.3) to compute F(i) ∈RT ×J×d
(i = 1, ..., N) to effectively capture the underlying 3D
structure of the skeleton sequence. Finally, we map F(N)
to a higher dimension by applying a linear layer and tanh
activation to compute motion semantic M ∈RT ×J×d′ and
use a regression head to estimate 3D pose ˆP ∈RT ×J×3.
The lifting loss contains position (L3D) and velocity (L∆P)
terms defined as
L3D = ΣT
t=1ΣJ
j=1∥ˆPt,j −Pt,j∥,
L∆P = ΣT
t=2ΣJ
j=1∥∆ˆPt,j −∆Pt,j∥,
(2)
where ∆ˆPt = ˆPt −ˆPt−1, ∆Pt = Pt −Pt−1. The total
lifting loss is then defined as
L = L3D + λ∆PL∆P,
(3)
where the constant coefficient λ∆P is used to balance posi-
tion accuracy and motion smoothness.
3.3. AGFormer Block
The AGFormer block uses a dual-stream architec-
ture.
Each stream consists of two components: a Spa-
tial MetaFormer (Figure 2b) followed by a Temporal
MetaFormer (Figure 2c).
The Spatial MetaFormer pro-
cesses individual body joints as distinct tokens, effectively
capturing intra-frame relationships within a single frame.
The Temporal MetaFormer, on the other hand, treats each
frame as a single token, thus capturing inter-frame rela-
tionships over time. The key distinction between the two
streams lies in their token mixer type. While one stream
employs Transformers, the other stream uses GCNFormers.
Transformer stream.
This stream employs a Spa-
tial Multi-Head Self-Attention (S-MHSA) to capture spa-
tial relationships, followed by a Temporal Multi-Head Self-
Attention (T-MHSA) to capture temporal relationships. The
S-MHSA is defined as
S-MHSA(Qs, Ks, Vs) = Concat(headi, ..., headh)Ws
(O),
headi = softmax(Qs
(i)(Ks
(i))T
√dk
)Vs
(i),
(4)
3

Fully Connected Layer
SPE
Spatial Transformer
Temporal Transformer
Fully Connected Layer
Regression Head
1
2
3
T-1
T
SPE
Spatial Position Embedding
Adaptive Fusion
(a)
× 𝑁
AGFormer 
T-2
Temporal GCNFormer
Spatial GCNFormer
𝐅𝐓𝐅
𝐅𝐆𝐅
Reshape
Layer Normalization
Spatial Token Mixer
Layer Normalization
Channel MLP
1
2
3
16
𝐽
Layer Normalization
Temporal Token Mixer
Layer Normalization
Channel MLP
1
2
3
T-1
T
(b)
(c)
Figure 2. (a) MotionAGFormer is a novel architecture featuring N dual-stream spatio-temporal blocks, wherein one stream employs
Transformers and the other leverages GCNFormers. (b) Spatial MetaFormer. Each input token represents an individual joint of the
human body. (c) Temporal MetaFormer. Input tokens are frames of pose sequence.
where Ws
(O) is a projection parameter matrix, h is the
number of parallel attention heads, and dk is the feature di-
mension of Ks. For computing the query matrix Qs, the
key matrix Ks, and the value matrix Vs, we have
Qs
i = FsWs
(Q,i), Ks
i = FsWs
(K,i), Vs
i = FsWs
(V,i),
(5)
where Fs ∈RBT ×J×d is spatial feature and Ws
(Q,i),
Ws
(K,i), Ws
(V,i) are projection matrices and B is the
batch size.
The S-MHSA result is subsequently fed
into a multilayer perceptron (MLP), followed by a resid-
ual connection and LayerNorm. This completes the first
MetaFormer, i.e. the Spatial Transformer.
Next, we reshape Fs into FT ∈RBJ×T ×d to prepare
per-joint temporal feature as the input of T-MHSA. Here
we have
T-MHSA(QT, KT, VT) = Concat(headi, ..., headh)WT
(O),
headi = softmax(QT
(i)(KT
(i))T
√dk
)VT
(i),
(6)
where QT, KT, and VT are calculated similar to Eqn. (5).
GCNFormer stream. Unlike the Transformer stream,
which aggregates global information, the GCNFormer
stream focuses on local spatial and temporal relationships
present within the skeleton sequence. While the local infor-
mation is also available to the Transformers, the inclusion
of this parallel stream allows the model to more effectively
balance the integration of local and global information (see
ablation analysis 4.5). The customized GCN module [24]
used in our GCNFormer is defined as:
GCN(F(i)) = σ(V l + Norm( ˜D−1
2 ˜A ˜D−1
2 F(i)W1 + F(i)W2).
(7)
Where ˜A = A + IN represents the adjacency matrix with
self-connections added, IN stands for the identity matrix,
˜Dii = Σj ˜Ajj is defined as the sum of elements along the
diagonal of ˜A, and W1, W2 denote trainable weight matri-
ces specific to each layer. The activation function σ(·), such
as ReLU, is applied, along with Batch Normalization [13].
The GCN’s output is then passed through an MLP, followed
by residual connection and LayerNorm.
The difference between the Spatial GCNFormer and the
Temporal GCNFormer lies in their adjacency matrices and
input features.
The input features resemble that of the
Transformer stream. In the Spatial GCNFormer, the adja-
cency matrix represents the human topology (Figure 3a).
For Temporal GCNFormer, on the other hand, we calculate
the similarity between a single joint at different time frames
using Sim(FT
ti, FT
tj) = (FT
ti)T FT
tj and choose the
K nearest neighbors as the connected nodes in the graph
(Figure 3b). Hence, the graph topology in Temporal GCN-
Former is determined by the learned node features.
Adaptive Fusion. Similar to MotionBERT [53], we use
adaptive fusion to aggregate extracted features of the Trans-
former and GCNFormer streams. This is defined as:
F(i) = αT F (i) ◦FTF
(i−1) + αGF (i) ◦FGF
(i−1), (8)
where F(i) represents the feature embedding extracted at
depth i, the element-wise multiplication denoted by ◦, and
FTF
(i−1), FGF
(i−1) refer to the extracted Transformer
stream and GCNFormer stream features at depth i −1, re-
spectively. The adaptive fusion weights αT F and αGF are
4

𝑇= 1
Spatial
Temporal
(a)
(b)
1
17
17
1
Adjacency Matrix
y
x
y
x
y
x
𝑇= 0
𝑇= 2
Low
High
Low
High
0
1
0
1
𝑇= 0 𝑇= 1 𝑇= 2
𝑇= 0
𝑇= 1
𝑇= 2
Similarity
(Left Wrist)
K-NN
(K=1)
Similarity
(Right Ankle)
K-NN
(K=1)
Adjacency Matrix
Adjacency Matrix
Figure 3. GCNFormer module topology. (a) The Spatial GCN-
Former employs the human skeleton as its underlying topology.
(b) The Temporal GCNFormer uses K-nearest neighbor (K-NN)
to determine connected edges by considering the highest similarity
of each joint (e.g., left wrist and right ankle in the figure) across
the entire time frame. After K-NN, each row is connected to K
columns.
defined as
αT F (i), αGF (i) = softmax(W · Concat(FTF
(i−1), FGF
(i−1))), (9)
where W is a learnable linear transformation.

## experiments
We evaluate the performance of our proposed Motion-
AGFormer on two large-scale 3D human pose estimation
datasets, i.e., Human3.6M [14] and MPI-INF-3DHP [25].
4.1. Datasets and Evaluation Metrics
Human3.6M is a widely used indoor dataset for 3D hu-
man pose estimation. It contains 3.6 million video frames
of 11 subjects performing 15 different daily activities. To
ensure fair evaluation, we follow the standard approach and
train the model using data from subjects 1, 5, 6, 7, and 8,
and then test it on data from subjects 9 and 11. Following
previous works [40, 50], we use two protocols for evalu-
ation. The first protocol (referred to as P1) uses Mean Per
Joint Position Error (MPJPE) in millimeters between the es-
timated pose and the actual pose, after aligning their root
joints (sacrum). The second protocol (referred to as P2)
measures Procrustes-MPJPE, where the actual pose and the
estimated pose are aligned through a rigid transformation.
MPI-INF-3DHP is another large-scale dataset gathered
in three different settings: green screen, non-green screen,
and outdoor environments. Following previous works [40,
50], MPJPE, Percentage of Correct Keypoint (PC) within
150 mm range, and Area Under the Curve (AUC) are re-
ported as evaluation metrics.
4.2. Implementation Details
Model Variants. We build four different configurations
of our model, as summarized in Table 1. Our base model,
known as MotionAGFormer-B, strikes a balance between
accurate estimation and computational cost. The remain-
ing variants are named according to their parameter size and
computational demands and selection of each variant can be
based on an application’s requirements, such as choosing
between real-time processing or more precise estimations.
The motion semantic dimension is d′ = 512, the expan-
sion layer of each MLP is α = 4, the number of attention
heads is h = 8, and the number of temporal neighbours in
GCNformer stream is k = 2, for all experiments.
Table 1. Details of MotionAGFormer model variants. N: Number
of layers. d: Hidden size. T: Number of input frames.

## related_work
3D human pose estimation.
Current approaches to
tackle this problem can be understood from two perspec-
tives.
From one perspective, models can be categorized
based on the input video, which can be either multi-view
or monocular. Models that rely on multi-view inputs [7,
15, 34, 35, 48] necessitate the simultaneous use of multiple
cameras from different angles, which can be less feasible
in real-world scenarios. From another perspective, consid-
ering their methodology, these models can be categorized
as either direct 3D estimation approaches or 2D-3D lifting
approaches. Direct 3D estimation methods [31, 32, 39, 52]
infer the joints in 3D coordinate from the video frames with-
out any intermediate step. Inspired by the rapid develop-
ment and availability of accurate 2D pose estimation mod-
els, more recently, 2D-3D lifting methods first use an off-
the-shelf 2D pose detectors [4, 28, 38] then lift 2D coordi-
nates to 3D space [40, 47, 50, 51, 53]. In this work, we are
using 2D-3D lifting methods by having monocular video as
the input.
Transformer-based methods. Transformers [41] have
shown promising result in different visual tasks [10, 37,
45, 54]. In the field of 3D human pose estimation, Pose-
Former [51] was the first purely transformer-based model.
PoseFormerV2 [50] improved its computational efficacy
by employing a frequency-domain representation that also
made it robust against sudden movements in noisy data.
MHFormer [20] addressed the problem of self-occlusion
and depth ambiguity by proposing a module that learns
multiple plausible pose hypotheses.
P-STMO [36] pro-
posed masked pose modeling and reduced the final er-
ror by self-supervised pretraining the model.
Enfalt et
al. [11] decreased the computational complexity by lever-
aging masked token modeling. StridedFormer [19] replaced
fully-connected layers in the feed-forward network of the
transformer encoder with strided convolutions to progres-
sively shrink the sequence length and efficiently lift the cen-
tral frame. Unlike the abovementioned models that esti-
mate the 3D pose for only the center frame of a sequence,
MixSTE [47] provided 3D estimates for every frame in
the input sequence. STCFormer [40] decreased computa-
tional complexity by separating the correlation learning into
spatial and temporal components.
HSTFormer [33] pro-
posed hierarchical transformer encoders for better captur-
ing spatial-temporal correlations. In addition to joint-joint
attention which is commonly used, HDFormer [3] included
bone-joint and hyperbone-joint interactions. Some works
exploited the output motion representation for various tasks.
MotionBERT [53] fine-tuned the model learned for the task
of 3D human pose estimation for tasks such as action recog-
nition and 3D human mesh recovery, while UPS [12] trained
a unified model for action recognition, 3D pose estimation,
and early action prediction at the same time.
2

Graph Convolutional Network.
GCN-based meth-
ods have achieved remarkable success within the domain
of skeleton-based action recognition [5, 18, 22].
Despite
their computational efficiency in 3D human pose estima-
tion [6,8,42,49], they usually cannot show competitive error
compared to transformer-based counterparts. This is pri-
marily due to their focus on local joints alone. Recently,
GLA-GCN [44] introduced an adaptive GCN approach to
leverage global representation. By employing a strided de-
sign to reduce its temporal scope, they achieve competi-
tive 3D human pose estimation against various transformer-
based models, all while maintaining a lighter memory load.
However, the effectiveness of the proposed module in ex-
tracting global representation is not on par with that of an
attention module.
Hybrid methods. These methods use different modules
to capture distinct aspects of the input sequence and are not
extensively explored yet. Recently, DC-GCT [16] proposed
a Local Constraint Module based on GCN, and Global Con-
straint Module based onf self-attention to exploit both local
and global dependecies of the input sequence. However, as
their model is designed to operate with both individual input
frames and sequences of frames, it does not distinguish be-
tween temporal and spatial dimensions. As a result, it falls
short in delivering competitive outcomes when contrasted
with transformer-based methods.

## conclusion
We introduced MotionAGFormer, a novel approach that
leverages GCNFormer to capture intricate local joint rela-
tionships, and combines it with Transformer that effectively
Left  Ankle
Left Elbow
Left Knee
Layer 1
Layer 8
Layer 16
Figure 6. Temporal adjacency matrix of a random sequence on
three different joints, from the Human3.6M dataset, at the first
layer (left), middle layer (center), and last layer (right). K in K-
NN is set to 2.
Table 6. The P1 error comparison when using different positional
embedding.
Temporal Embedding
Spatial Embedding
P1
-
-
39.3
-
✓
38.4
✓
-
38.9
✓
✓
40.5
Table 7.
Comparison of different MetaFormer integra-
tion.
All the models are trained on Human3.6M with our
MotionAGFormer-B settings.