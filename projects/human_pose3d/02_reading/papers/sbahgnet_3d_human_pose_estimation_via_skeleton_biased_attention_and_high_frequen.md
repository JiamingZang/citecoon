# SBAHGNet:3D Human Pose Estimation via Skeleton-Biased Attention and High-Frequency Enhanced Graph Convolution

> 2026 · id: W7128608150 · pdf: https://www.researchsquare.com/article/rs-8548943/latest.pdf · 来源: pdf_url
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

SBAHGNet:3D Human Pose Estimation via
Skeleton-Biased Attention and High-Frequency
Enhanced Graph Convolution
Yu Wang 
Hefei University of Technology
Jiaqiu Ai 
Hefei University of Technology
Xinyu Sun 
Hefei University of Technology
Yong Zhang 
Hefei University of Technology
Jinyang Huang 
Hefei University of Technology
Research Article
Keywords: Monocular 3D human pose estimation, Spatio-temporal fusion, Graph convolutional network,
Skeletal-Biased Attention, High-frequency enhancement
Posted Date: February 11th, 2026
DOI: https://doi.org/10.21203/rs.3.rs-8548943/v1
License:   This work is licensed under a Creative Commons Attribution 4.0 International License.  
Read Full License
Additional Declarations: No competing interests reported.
Version of Record: A version of this preprint was published at Machine Vision and Applications on June
2nd, 2026. See the published version at https://doi.org/10.1007/s00138-026-01852-7.

 
1 
 
SBAHGNet:3D Human Pose Estimation via Skeleton-Biased 
Attention and High-Frequency Enhanced Graph Convolution 
Yu Wang1 · Jiaqiu Ai1 · Xinyu Sun1 · Yong Zhang1· Jinyang Huang1  
 
 
Abstract 
Monocular 3D human pose estimation is challenged by depth ambiguity and complex articulation, which complicate feature 
modeling and demand robust spatio-temporal representations. Although existing methods have advanced spatio-temporal modeling, 
limitations remain: graph convolutional network (GCN) exhibits low-pass behavior that, as depth increases, attenuates high-
frequency geometric details in joint trajectories and thus degrades depth accuracy; and standard self-attention does not explicitly 
encode skeletal topology, resulting in indirect modeling of bone connectivity. To address these issues, we propose SBAHGNet, a 
dual-branch spatio-temporal feature-fusion network. In the GCN branch, a Multi-Scale High-Frequency Enhancement (MSHFE) 
module—applied after feature aggregation-recovers high-frequency geometric cues lost to GCN smoothing, improving fine-grained 
depth representation. In the attention branch, a Skeletal-Biased Attention (SBA) module injects a learnable skeletal bias into spatial 
attention to explicitly encode skeletal topology and strengthen structural modeling. Complementary features from both branches 
are adaptively fused for final 3D pose regression. Extensive experiments on Human3.6M and MPI-INF-3DHP validate our approach. 
With detected 2D keypoints, SBAHGNet attains 37.24 mm MPJPE (P1) and 31.57 mm PA-MPJPE (P2) on Human3.6M (12.38 
mm with ground-truth 2D), and 13.83 mm MPJPE, 99.02% PCK@150mm, and 88.22 AUC on MPI-INF-3DHP. With only 18.3M 
parameters, the model achieves a favorable accuracy–efficiency trade-off and outperforms many comparable methods. 
Keywords  Monocular 3D human pose estimation  · Spatio-temporal fusion · Graph convolutional network · Skeletal-
Biased Attention · High-frequency enhancement  
 
1  Introduction 
 
Monocular 3D human pose estimation aims to recover the 
three-dimensional coordinates of human keypoints from a 
single RGB image or video sequence, serving as a 
foundational task for understanding human motion and 
behavior [1, 2].Owing to its reliance on inexpensive standard 
RGB cameras for capturing 3D poses, monocular pose 
estimation offers low cost and high practicality, finding 
widespread applications in action recognition [3], intelligent 
surveillance 
[4], 
human-computer 
interaction 
[5], 
augmented reality [6], and virtual reality [7]. However, the 
inherent depth ambiguity in monocular inputs (where poses 
at varying depths may yield similar 2D projections), coupled 
with motion blur from rapid movements, occlusions, and  
 
  Jiaqiu Ai 
aijiaqiu1985@hfut.edu.cn 
Yu Wang 
2024170780@mail.hfut.edu.cn 
Xinyu Sun 
2025010057@mail.hfut.edu.cn 
Yong Zhang 
yongzhang@hfut.edu.cn 
Jinyang Huang 
hjy@hfut.edu.cn 
1      School of Computer and Information, Hefei University of 
Technology, Hefei 230009, Anhui Province, China 
 
noise in 2D detections, poses significant challenges to 
effective spatio-temporal feature modeling [8, 9]. 
In engineering practice, the 2D-to-3D lifting paradigm, 
which first detects 2D keypoints and subsequently regresses 
3D coordinates, has emerged as the dominant approach [10-
12]. This paradigm capitalizes on mature 2D detectors while 
substantially reducing annotation and training costs. Its core 
challenge lies in recovering accurate and skeleton-consistent 
3D structures from noisy and potentially incomplete 2D 
keypoint sequences, imposing stringent requirements on the 
expressive capacity and robustness of spatio-temporal 
modeling strategies. Early works primarily relied on 
Temporal Convolutional Network (TCN) for sequence 
modeling. For instance, VideoPose3D [10] proposed by 
Pavllo et al. employed dilated temporal convolutions to 
achieve large receptive fields and incorporated semi-
supervised 2D-3D reprojection to exploit unlabeled videos, 
establishing a widely adopted benchmark. Although TCN is 
computationally efficient and straightforward to implement, 
it exhibits clear limitations in capturing extremely long-
range dependencies and complex cross-joint interactions [1, 
13]. To address these limitations, researchers have pursued 
two 
complementary 
directions: 
introducing 
Graph 
Convolutional Network (GCN) to explicitly encode skeletal 
topology and reinforce local kinematic priors on one hand, 
and adopting attention-based mechanisms to model global 
dependencies across joints and frames on the other [14-17].                     

 
 
2
GCN effectively captures structured inter-joint relationships 
through neighborhood aggregation (e.g., SemGCN [15] 
demonstrated high parameter efficiency and robustness), 
significantly enhancing skeleton consistency and local 
geometric representation. However, GCN inherently 
possesses low-pass filtering characteristics [18-20]. As 
network 
depth 
increases, 
features 
progressively 
homogenize, attenuating high-frequency details such as 
rapid joint trajectories and local geometric variations, 
thereby impairing depth recovery and subtle motion 
estimation. Subsequent studies have proposed multi-scale 
designs, spectral decompositions, and trainable bandpass 
filtering strategies to compensate for lost high-frequency 
components [21, 22]. To tackle this issue, we introduce a 
Multi-Scale High-Frequency Enhancement (MSHFE) 
module in the GCN branch. Applied after feature 
aggregation, this module selectively restores suppressed 
multi-scale 
high-frequency 
components, 
recovering 
geometric details in joint motions and improving depth 
estimation accuracy. 
Meanwhile, attention-based methods excel in global 
spatio-temporal modeling by directly capturing long-range 
interactions between arbitrary joints and frames [23-26]. A 
representative work, PoseFormer [27], pioneered the 
separation of spatial and temporal attention modules to 
encode intra-frame joint relationships and inter-frame 
motion dependencies, respectively. Follow-up approaches 
such as MixSTE [16] and PoseFormerV2 [28] further 
expand receptive fields and enhance robustness through 
refined token designs or frequency-domain representations 
(e.g., preserving critical high-frequency coefficients in 
Discrete Cosine Transform). Despite the significant 
advantages of attention mechanisms in long-range 
dependency modeling, they typically process flattened joint 
sequences and lack explicit encoding of physical skeletal 
connections, leading to degraded skeleton consistency under 
noisy inputs [29, 30]. To address this limitation, we propose 
a Skeletal-Biased Attention (SBA) module that injects 
learnable skeletal topological bias into the spatial attention 
distribution with extremely low parameter overhead, 
explicitly guiding the attention mechanism toward 
anatomically relevant joint pairs and thereby enhancing 
local structural consistency. 
Inspired by the complementary strengths of GCN's local 
structural modeling and attention mechanisms' global 
dependency capture, we propose SBAHGNet, a lightweight 
dual-branch 
spatio-temporal 
feature-fusion 
network 
designed to jointly achieve explicit skeletal topology 
injection, global long-range modeling, and high-frequency 
detail compensation. The graph branch incorporates the 
MSHFE module to recover high-frequency geometric cues 
lost due to GCN smoothing. The attention branch integrates 
the SBA module, which injects skeletal topological bias into 
the spatial attention distribution with negligible parameter 
overhead. Complementary features from both branches 
are adaptively fused for final 3D pose regression. 
Our main contributions are as follows: 
1.   We design the MSHFE module, which effectively 
compensates for high-frequency losses in the GCN 
branch, recovering geometric details in joint motions 
and enhancing depth estimation accuracy. 
2.   We propose the SBA module, which injects skeletal 
topological bias into the spatial attention distribution 
with extremely low parameter overhead (376 
additional parameters), improving local structural 
consistency. 
3.   Through extensive experiments on the Human3.6M 
and MPI-INF-3DHP datasets, we demonstrate that 
SBAHGNet 
achieves 
highly 
competitive 
performance among existing monocular 3D pose 
estimation methods. 
 
2  Related work 
Current research on monocular 3D human pose estimation 
primarily focuses on recovering 3D joint positions from 
single RGB images or video sequences, a task complicated 
by depth ambiguity, nonlinear motion, occlusion, and rapid 
movements [8, 9]. Recent approaches predominantly adopt 
the 2D-to-3D lifting paradigm, where 2D keypoints are first 
detected and then lifted to 3D via spatio-temporal modeling 
[11, 12]. Existing methods can be grouped by their core 
modeling components into four main categories. 
2.1  Temporal Modeling Based on Temporal 
Convolution and Recurrent Operators 
In video-based monocular 3D pose estimation, temporal 
information from 2D keypoint sequences is exploited to 
improve inter-frame consistency and robustness. Early 
methods employ recurrent networks, such as sequence-to-
sequence LSTMs [31], to model temporal joint evolution 
and reduce jitter. Some real-time systems integrate CNN-
based regression with skeletal fitting or temporal filtering 
for online stability, exemplified by VNect [32]. More 
recently, dilated temporal convolutional networks (TCNs) 
gain prominence due to their large receptive fields and 
parallelism, with VideoPose3D [10] introducing semi-
supervised back-projection to utilize unlabeled data. Recent 
works further explore the potential of temporal modules in 
modeling motion dynamics and global consistency; for 
example, You et al. [33] propose PMCE, a dual-stream “co-
evolution” architecture where one branch lifts 2D joint 
sequences to the mid-frame 3D pose and the other 
aggregates cross-time image features using a temporal 
convolutional network; Zheng et al. [34] introduce the 
Retentive Network (RetNet), leveraging a large window of 
past frames and a few future frames to capture long-range

 
 
3
dependencies, with a non-causal variant (NC-RetNet) and a 
knowledge-transfer training scheme; Hsu and Jang [35] 
utilize an RNN to predict bone lengths over the entire 
sequence and adjust 3D poses accordingly to enforce 
physical consistency. These approaches offer simple 
architectures, efficient training/inference, and strong 
deployability, but often face limitations in explicitly 
encoding skeletal topological priors and recovering high-
frequency geometric details from rapid local depth 
variations. 
2.2  Skeletal Topology Modeling Based on Graph 
Convolutional Network 
To incorporate human body priors, graph-based methods 
represent joints and bones as graphs and apply GCN for 
spatial aggregation of dependencies. ST-GCN [36] pioneers 
spatio-temporal graph convolution in skeleton-based tasks, 
while SemGCN [15] introduces learnable or semantically 
guided adjacency matrices for improved accuracy under 
constrained parameters. However, standard GCN exhibits 
low-pass filtering characteristics, causing over-smoothing 
and loss of high-frequency motion with deeper layers. 
Subsequent works enhance the capture of pose structure by 
refining graph convolution operations; for instance, Azizi et 
al. [37] propose MöbiusGCN, which uses Möbius 
transformations in the spectral domain to explicitly model 
inter-joint rotations, achieving state-of-the-art accuracy 
with drastically fewer parameters (only 0.042M in its 
lightest version); Zhang [38] introduces GroupGCN, 
decoupling shared aggregation into group convolutions with 
independent adjacency kernels per feature group and cross-
group interaction; Yu et al. [13] present GLA-GCN, an 
adaptive global-local architecture with one branch 
aggregating spatio-temporal features over the entire 
skeleton graph and another refining per-joint features via 
independently 
connected 
layers. 
These 
methods 
demonstrate that dynamically learning or explicitly 
encoding skeletal graph structures (even geometric 
transformations) enhances GCN performance in topological 
prior encoding, but can lead to loss of high-frequency 
geometric details and may not fully integrate long-range 
global dependencies under complex motions. 
2.3  Attention-Based Global Spatio-Temporal 
Modeling 
Attention mechanisms have emerged as powerful tools for 
capturing 
long-range 
dependencies, 
making 
them 
particularly suitable for global spatio-temporal modeling in 
video sequences. These methods directly model interactions 
between arbitrary joints and frames, thereby overcoming the 
inherent limitations of convolutional approaches in terms of 
local receptive fields and enabling effective capture of global 
dependencies over extended temporal ranges. PoseFormer 
[27] represents one of the pioneering works, being the first to 
decouple spatial and temporal attention modules to      
separately encode intra-frame joint relationships and inter-
frame motion dependencies. This approach better captures 
cross-frame global spatio-temporal relationships and has 
provided an important foundation for subsequent research. 
Numerous follow-up studies have further refined 
attention-based frameworks to address challenges such as 
depth ambiguity and long-sequence modeling. For 
example, MHFormer [39] employs a multi-hypothesis 
generation mechanism to improve pose estimation 
accuracy while enhancing model efficiency through 
alternating 
spatio-temporal 
blocks; 
MixSTE 
[16] 
incorporates 
frequency-domain 
representations 
and 
optimizes token designs to expand receptive fields and 
improve robustness; STCFormer [40] adopts Spatio-
Temporal Criss-Cross Attention, achieving global 
interactions 
with 
sub-quadratic 
complexity 
while 
integrating local convolutions to provide richer contextual 
information. Beyond purely attention-based designs, 
several studies have explored attention variants combined 
with Graph Convolutional Networks or convolutional 
backbones to enhance the efficiency of global modeling. 
For instance, GAST-Net [41] integrates graph attention 
with spatio-temporal convolutions, enabling adaptive 
joint weighting to capture global spatial relationships; the  
SaEGC-Net [42] proposes a simplified spatio-temporal 
attention module (SST-Att) embedded within a GCN 
framework, 
effectively 
modeling 
long-range 
dependencies between non-adjacent joints while avoiding 
the quadratic complexity of conventional self-attention 
mechanisms. Despite the significant advantages of 
attention-based methods in global dependency modeling, 
they typically process flattened joint sequences or rely on 
implicit learning, resulting in a lack of explicit encoding 
of physical skeletal connections. This limitation can 
impair the model's understanding of human skeletal 
structure, particularly when handling rapid motions or 
complex actions. Furthermore, the absence of explicit 
structural 
awareness 
may 
hinder 
the 
effective 
preservation of high-frequency geometric details arising 
from rapid local movements, thereby affecting the 
accuracy of depth estimation and subtle motion capture. 
2.4  Hybrid Architecture and Spectral/Frequency-
Domain Enhancement 
Recent efforts combine paradigms to address inductive 
biases and leverage frequency representations for 
efficiency. For instance, Zhao et al. [28] propose 
PoseFormerV2, converting long joint sequences to the 
frequency domain via Discrete Cosine Transform, using 
few low-frequency coefficients to expand receptive fields 
and substantially reduce computation, fusing time- and 
frequency-domain features for better speed-accuracy 
trade-off and noise robustness; Lin et al. [43] introduce 
AMPose, alternately stacking Transformer and GCN 
layers to jointly encode global joint relations and local 

 
 
4
1
2
3
T-1
T-2
T
1
2
3
T-2
T-1
T
...
...
...
...
...
SPE
SBA
TSA
Spatial
HGCN
Temporal
HGCN
...
...
...
...
...
Reshape
Reshape
 Regression 
Head
＋
SBAHGBlock
×N
SBA
TSA
Skeleton-Biased Attention
Temporal Self-Attention
SPE Spatial Position Embedding
s
P
X
( )
0
F
(
1)
i
F
−
x
( )i
A
F
( )i
G
F
( )i
F
(
)
N
F
P

 
Fig. 1  The overall architecture of SBAHGNet. 
bone connectivity; Zhai et al. [44] present HGFreNet, 
combining 
hop-based 
graph 
attention 
blocks 
with 
Transformer encoders and enforcing temporal consistency 
via frequency-domain loss for smoother trajectories. 
Frequency-domain techniques, such as graph wavelets and 
scattering, mitigate GCN low-pass effects and enhance long-
sequence 
efficiency 
by 
preserving 
high-frequency 
components [21, 22]. While these hybrid and frequency-
domain methods advance performance on multiple fronts, 
existing works often face challenges in simultaneously 
achieving strong structural awareness and effective recovery 
of high-frequency geometric details within a lightweight 
framework [45, 46].  
Based on the above analysis, the proposed SBAHGNet 
adopts a dual-branch design for integration: the skeleton-
biased attention branch explicitly embeds topology into 
self-attention to enhance structural awareness; The 
proposed unified and lightweight framework directly 
addresses the two primary limitations commonly observed 
in prior approaches (particularly hybrid methods): 
insufficient explicit structural awareness of the human 
skeleton and inadequate preservation of high-frequency 
geometric 
details. 
This design 
confers 
substantial 
advantages over existing hybrid paradigms in terms of 
model 
simplicity, 
computational 
efficiency, 
and 
performance on fine-grained motion capture.  
3  Method 
3.1  Overall architecture 
As shown in Fig. 1, the overall architecture of the proposed  
SBAHGNet is as follows. The model input is a 2D 
keypoint sequence 
3
B T J
x
R 

, and the model output 
(prediction) is 
3
B T J
P
R 



, where B is the batch size, T 
is the number of frames, and J is the number of joints. The 
last dimension 3 denotes the 2D coordinates plus a 
confidence score. The data processing pipeline is: first, 
the input joint features 𝑥 are linearly projected along the 
last dimension to D channels, yielding features 
B T J D
X
R 

 . 
A 
spatial 
positional 
encoding 
1
s
J D
P
R 

is then added to X (broadcasted over the B 
and T dimensions). The resultant representation (denoted 
as 
( )
0
B T J D
F
R 

) is subsequently input to N cascaded 
SBAHGBlock modules, thereby progressively extracting 
hierarchical 
spatio-temporal 
features, 
producing 
(
)
N
B T J D
F
R 

. Finally, the joint features are mapped to 
a higher dimensional space via linear layers and a 
regression head is used to produce the predicted 3D 
keypoint sequence P

. The present model is trained by 
employing both a position loss function (
3D
L
) and a 
velocity loss function (
P
L). The corresponding formulas 
are given as follows: 
t,
3
,
t 1
1
t,
,
t 2
1
||
||
||
||
T
J
j
D
t j
j
T
J
j
P
t j
j
L
P
P
L
P
P

=
=


=
=
=
−
=

−


                                         (1) 
Where 
1
t
t
t
P
P
P



−

=
−
 denotes 
the 
inter-frame 
differences of the predicted 3D pose sequence output by 

 
 
5
the model, and 
1
t
t
t
P
P
P−

=
−
 represents the inter-frame 
differences of the corresponding ground-truth 3D pose 
sequence. The overall loss function of the model is defined 
as follows: 
3D
P
P
L
L
L


=
+
                                                              (2) 
where ߣ௱ುis a hyperparameter that balances positional 
accuracy and motion smoothness. 
3.2  SBAHGBlock 
The proposed module is composed of three primary 
components: a GCN branch, an attention branch, and an 
adaptive fusion module. Within the GCN branch, a temporal 
High-Frequency Enhanced Graph Convolutional Network 
(HGCN) module and a spatial HGCN module are employed, 
both following the architecture depicted in Figure 2. The 
attention branch consists of a Skeletal-Biased Attention 
(SBA) module and a Temporal Self-Attention (TSA) 
module. Both branches adopt a cascaded architecture, where 
spatial features are extracted prior to temporal features. The 
GCN branch employs a spatio-temporal HGCN module to 
extract and fuse spatio-temporal adjacency relationships, 
thereby enhancing the spatio-temporal representation of 3D 
poses. The attention branch utilizes SBA and TSA to capture 
global information and effectively model long-range 
dependencies in human motion, wherein the skeleton bias 
introduced in SBA facilitates a better understanding of 
spatial dependencies among joints. Finally, features from the 
two branches are integrated via the adaptive fusion module, 
yielding a fused representation that balances different 
information foci. 
3.2.1  Spatio-Temporal HGCN Module 
The spatial HGCN and temporal HGCN are respectively 
used to extract spatial and temporal information from human 
motion, thereby capturing local connectivity relationships 
between joints as well as temporal dependencies during the 
motion process. The difference between the spatial module 
and the temporal module lies in the input to the spatio-
temporal module and the adjacency matrix. The output 
( )i
GS
F
 
of the spatial HGCN and the output 
( )i
G
F
 of the temporal 
HGCN are formulated as shown in Equations (3) and (4), 
respectively. 
(
1)
(
1)
( )
DP(Reshape(
(
(
)))
(
))
(
)
Reshape(LBR(
MHSA(
)))
i
i
S
R
R
i
GS
S
S
G
W A
F
F
GCN F
G
G
−
−

=
+
=
+
V
U
              (3) 
( )
( )
( )
DP(Reshape(
(
(
)))
(
))
(
)
Reshape(LBR(
MHSA(
)))
i
i
T
RGS
RGS
i
G
T
T
G
W A
F
F
GCN F
G
G

=
+
=
+
V
U
              (4) 
Where
(i 1)
(
)
BT
J D
R
F
R
−


represents the output of the previous 
×
1×3 branch
1×5 branch
HGCN
F3
F5
F
Input
Linear 
Projection V
Reshape
Reshape
0.5
1.0
0.5
0.5
1.0
0.5
0.5
1.0
0.5
-0.5
1.0
-0.5
-0.5
1.0
-0.5
-0.5
1.0
-0.5
×
α
＋
0.5
1.0
0.5
0.25
0.25
0.5
1.0
0.5
0.25
0.25
0.5
1.0
0.5
0.25
0.25
-0.5
1.0
-0.5
-0.25
-0.25
-0.5
1.0
-0.5
-0.25
-0.25
-0.5
1.0
-0.5
-0.25
-0.25
×
 β
＋
＋
Linear 
Projection U
DP
MHSA
Reshape
LBR
＋
＋
Reshape
Output
Dropout
MHSA
Multi-Head Self-Attention
LBR
LN+BN+ReLu
DP
MSHFE
A
 
Fig. 2  The overall architecture of HGCN, where the brown dashed 
portion illustrates the structure of the MSHFE module. 
SBAHGBlock after sequence rearrangement. For the 
spatial module, the input is 
(i 1)
R
F
−
 and the output 
( )i
B T J D
GS
F
R 

 denotes the output of the current spatial 
module, with the adjacency matrix constructed based on 
human body topology. For the temporal module, the input 
is 
(i)
(
)
BJ
T D
RGS
F
R


 (i.e., the rearranged
( )i
GS
F
) and the 
output 
( )i
B T J D
G
F
R 

represents the final output of the 
current branch, with the adjacency matrix constructed 
based on inter-frame joint similarity. V and U are two 
trainable weight matrices, 
N
A
A
I

=
+
represents the 
adjacency matrix with self-connections added, 
NI stands 
for the identity matrix, 
( )
W  denotes the processing by 
MSHFE, and LBR refers to the sequential operations of 
layer normalization, batch normalization, and ReLU 
activation. 
For the construction of the adjacency matrix A, in the 
spatial module, the adjacency matrix is predefined based 
on the topological relationships of human joints, with the 
specific connectivity illustrated in Figure 3. In the 
temporal module, we employ a dynamic adjacency 
matrix: first, the input temporal sequence features are L2-
normalized; then, cosine similarity between all pairs of 
time steps is computed via inner product (i.e., cosine 
similarity is calculated for the same joint across different 
frames), forming a similarity matrix; subsequently, for 
each row, the top-K largest values are selected as the 
threshold, and edges with similarity greater than or equal 
to this threshold are set to 1, yielding a sparse binary 
adjacency matrix; finally, symmetric normalization is 
applied for use in graph convolution. By combining the 
dynamic adjacency matrix with the graph convolutional 
network, the model can fully account for temporal 
relationships between different time steps during 
learning, thereby better capturing long-range temporal 
dependencies. The advantage of this approach lies in its 
ability to adaptively construct the adjacency matrix based 
on the input data at each moment, providing greater 
flexibility and adaptability across diverse motion 
scenarios. Figure 4 illustrates an example with 3 temporal  

 
 
6
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
Spatial Adjacency Matrix
 
Fig. 3  Spatial Adjacency Matrix based on human skeletal topology. 
frames, where a K-nearest neighbors (KNN) strategy is 
adopted with K=1, converting the similarity matrix into a 
binary adjacency matrix. 
Traditional graph convolutional networks exhibit an 
inherent smoothing effect during feature aggregation, which 
leads to the loss of high-frequency details in temporal 
sequences, 
such 
as 
motion 
boundaries 
and 
joint 
accelerations [18-20]. To mitigate this issue, we propose 
MSHFE, which is embedded in the aggregation stage of 
graph convolution. This module extracts local dynamic 
features through multi-scale convolution and enhances high-
frequency information using an adaptive mechanism, 
thereby improving the model's expressive capability for 
complex motion sequences. The structure of MSHFE is 
illustrated in Figure 2. 
MSHFE employs a dual-branch parallel architecture, 
utilizing 1×3 and 1×5 depthwise separable convolutions for  
efficient 
multi-scale 
feature 
decomposition. 
The 
convolution stride is 1 for both, with padding of 1 and 2, 
respectively, to ensure that the output sequence length 
matches the input exactly and avoids boundary information 
loss. Each branch consists of low-frequency and high-
frequency sub-paths: 
1) Low-frequency sub-path: 
A predefined smoothing kernel (1×3: [0.5, 1.0, 0.5]; 1×5: 
[0.25, 0.5, 1.0, 0.5, 0.25]) is applied to perform local weighted 
averaging on the input sequence, preserving the structural 
backbone information. 
2) High-frequency sub-path:  
Corresponding difference kernels (1×3: [-0.5, 1.0, -0.5]; 1×5: 
[-0.25, -0.5, 1.0, -0.5, -0.25]) are used to capture local rates of 
change. The high-frequency output is multiplied by learnable 
high-frequency gain factors ߙ and ߚ, which are initialized in 
logarithmic form to ensure positivity and numerical stability. 
These gains are dynamically adjusted during training to 
achieve adaptive amplification of abrupt motions. 
The low-frequency and high-frequency outputs at each 
scale are concatenated along the channel dimension, 
followed by 1×1 convolution, batch normalization, and 
ReLU activation for non-linear fusion within the same scale, 
producing refined representations F3 and F5. Subsequently, 
the module introduces a lightweight cross-scale attention 
mechanism: first, F3 and F5 are averaged and subjected to 
global average pooling to obtain a global context 
descriptor; then, weights are generated through two 
layers of 1×1 convolutions (C → C/2 → 2) followed by 
Softmax normalization. The final output is: 
3
3
5
5
F
F
F


=
+

                                                         (5) 
3.2.2  Spatio-Temporal Attention Module 
The attention branch is responsible for extracting spatio-
temporal features from the input, primarily achieved 
through a two-stage multi-head self-attention mechanism 
that processes spatial information first and temporal 
information second. Specifically, this branch first models 
spatial relationships between joints within each frame, 
followed by capturing long-range dependencies along the 
temporal dimension for the motion trajectory of each 
joint. This sequential design facilitates the separate 
handling of intra-frame structural information and inter-
frame dynamic information. In this paper, a lightweight 
multi-scale skeleton bias is introduced in the spatial 
attention stage of this branch to provide additional 
skeletal topological priors for the attention weights; the 
temporal 
attention 
stage 
retains 
the 
standard 
implementation without any additional operations. The 
formulation of the spatial SBA is as follows: 
s
1
( )
( )
( )
s
(
,
,
)
(
,...,
)
(
)
(
)
O
s
s
h
S
i
i
T
i
s
i
s
k
SBA Q K V
Concat head
head W
Q
K
head
softmax
B V
d
=
=
+
              (6) 
where 
O
S
W is the output projection matrix, h is the number 
of parallel attention heads, 
B T J J
B
R 

 denotes the 
multi-scale skeleton bias (its computation will be 
elaborated in the subsequent section), and 
/
kd
D h
=
 
represents the channel dimension per attention head. To 
compute the query matrix 
( )
s
i
Q
, key matrix 
( )
s
i
K
, and value 
matrix 
( )
s
i
V
, we have 
( )
,
( )
,
( )
,
s
s
s
,
,
i
Q i
i
K i
i
V i
S
s
S
s
S
s
Q
F W
K
F W
V
F W
=
=
=
                      (7) 
where 
BT J D
S
F
R


 is the reshaped version of the output 
(
1)
i
F
− from the previous layer, and 
,
Q i
s
W
,
,
K i
s
W
, 
,
V i
s
W
 are 
the projection matrices. 
The multi-scale skeleton bias B interacts with the input 
features 
(
1)
i
F
− through the association matrix 
s
E
J
s
H
R


 
to capture higher-order relationships between different 
joints. This bias is based on a normalized association 
matrix (allowing fine-tuning during training) and learnable 
scale weights 
s
w , injecting skeletal topological priors into 
the attention mechanism in the form of a dynamic outer 

 
 
7
 
KNN
(K=1)
Cosine Similarity
T=0
T=1
T=2
Temporal Adjacency Matrix
 
Fig. 4  Schematic illustration of the Temporal Adjacency Matrix constructed using the K-Nearest Neighbors (K-NN) algorithm. Connected 
edges are determined by considering the highest similarity of each joint across the entire temporal sequence (e.g., the right ankle in the figure). 
 
（a）
（b）
 
Fig. 5  Initialization of the normalized incidence matrices, where different colors represent distinct hyperedges. (a) Fine-scale normalized 
incidence matrix (13 17

); (b) Coarse-scale normalized incidence matrix ( 6 17

). 
 
product, thereby providing additional positive offsets for 
structurally related joint pairs. The bias is shared across all 
attention heads and broadcast along the head dimension 
before being added to the original dot-product scores. This 
process does not alter the standard attention computation flow 
and serves merely as a lightweight additive term for biasing. 
The specific computation is divided into three steps, with the 
formulations as follows: 
1) Node-to-Hyperedge Aggregation: 
For the s-th scale, the normalized incidence matrix 
s
H is 
utilized to average the joint features within the same 
hyperedge, generating the hyperedge feature 
s
B T E
D
eI
R 


: 
(
1)
i
e
s
I
H
F
−
=

                                                                   (8) 
where 
{1,2}
s
 denotes the scale index,
s
E is the number of 
hyperedges at this scale, and J=17 is the number of joints. 
Each row of the matrix corresponds to a hyperedge e; if joint 
j belongs to hyperedge e, then 
( , )
1/ | |
s
H e j
e
=
, otherwise 0. 
The matrix is initialized and treated as a learnable parameter, 
allowing 
fine-tuning 
during 
training. 
The 
specific 
initializations of the incidence matrices for the two scales are 
shown in Figures 5. 
2) Hyperedge-to-Node Broadcasting: 
The aggregated hyperedge features are broadcast back to 
the individual joints, yielding smoothed node features 
B T J D
e
B
R 

: 
(
1)
T
i
e
s
B
H
F
−
=

                                                                  (9) 
where 
e
B represents the bias information broadcast from 
hyperedge e, capturing similarity relationships between 
joints. 
3) Outer Product and Scale Fusion: 
The outer product of the smoothed features 
e
B and the 
original features 
(
1)
i
F
− is computed, ensuring that joints 
belonging to the same hyperedge share the same broadcast 
component, thereby naturally receiving additional positive 
contributions during similarity computation. Subsequently, 
the results from the two scales are fused using learnable 
weights 
s
w (normalized via Softmax) to obtain the final 
skeleton bias 
B T J J
B
R 

: 
(
1)
1
(
(
) )
s
i
T
s
e
i
B
w B F
−
=
= 
                                                     (10) 
The complete formulation integrating the above three steps 
is given as follows: 

 
 
8
(
1)
(
1)
1
((
(
))(
) )
s
T
i
i
T
s
s
s
i
B
w
H
H F
F
−
−
=
= 
                               (11) 
Subsequently, the output of the spatial SBA is reshaped into 
BJ T D
T
F
R


 to serve as the input to the TSA, which is used 
to extract temporal features of the joints. The computation 
formula for the temporal TSA is similar to that of the spatial 
SBA without the bias B, and is given as follows: 
1
( )
( )
( )
t
(
,
,
)
(
,...,
)
(
)
(
)
O
T
T
T
h
T
i
i
T
i
t
i
t
k
TSA Q
K
V
Concat head
head W
Q
K
head
softmax
V
d
=
=
             (12) 
where 
T
Q , 
T
K , and 
T
V are computed in a similar manner to 
Equation (7) (i.e., obtained through corresponding linear 
projections). 
3.2.3  Adaptive Fusion 
To fully leverage the complementary modeling capabilities of 
the attention branch and the graph convolution branch, we 
incorporate a lightweight adaptive fusion module in each 
SBAHGBlock. Unlike simple averaging or concatenation 
operations, this module dynamically assigns weights from the 
two branches for each joint and each frame. Specifically, after 
both branches complete their respective spatial and temporal 
processing, the output 
( )i
A
F
 from the attention branch and the 
output 
( )i
G
F
 from the graph convolution branch are 
concatenated along the channel dimension to obtain 
2
cat
B T J
D
F
R 

. Subsequently, a linear layer with only 2C+2 
parameters is applied to project and generate the weights. The 
final fused output 
( )i
F
 is computed as follows: 
2
( )
( )
( )
Softmax(
)
[
,0:1]
[
,0: 2]
B T J
f
cat
f
i
i
i
A
G
W F
b
R
F
F
F




=
+

=

+


                     (13) 
where 
f
W and 
fb are the weight matrix and bias vector of 
the projection linear layer, respectively, denotes element-
wise multiplication, and the weight  is automatically 
broadcast along the channel dimension to match the feature 
dimension D. 
 
4  Experiments 
4.1  Datasets and evaluation metrics 
To evaluate the performance of the proposed model, we 
conduct experiments on two widely used benchmark 
datasets for 3D human pose estimation: Human3.6M [47] 
and MPI-INF-3DHP [48]. 
Human3.6M is currently the most widely used benchmark 
dataset for indoor human pose estimation, comprising 
approximately 3.6 million frames captured from 11 subjects 
performing 15 categories of daily activities. Consistent 
with previous studies [16, 39], to ensure comparability 
with existing works, we adopt the standard data split: 
training on subjects 1, 5, 6, 7, and 8, and testing on 
subjects 9 and 11.Evaluation is conducted using two 
standard protocols: Protocol #1 (MPJPE) measures the 
mean per-joint position error (in millimeters) after 
aligning the root joints of the predicted and ground-truth 
poses; Protocol #2 (P-MPJPE) computes the error after 
rigid Procrustes alignment between the predicted pose 
and the ground truth. 
MPI-INF-3DHP includes both indoor and outdoor 
scenes, with its test set covering three types of 
environments: studio with green screen, studio without 
green screen, and outdoor. Following the practices of 
previous works [16, 39], we report the mean per-joint 
position error (MPJPE), the percentage of correct 
keypoints (PCK) with a threshold of 150 mm, and the 
corresponding area under the curve (AUC). These 
metrics collectively provide a comprehensive reflection 
of the method's performance in terms of spatial accuracy 
and robustness in keypoint detection 
4.2  Implementation details 
The proposed SBAHGNet model is implemented using 
PyTorch and trained on a single NVIDIA RTX 3080 Ti 
GPU. Horizontal flipping is applied as data augmentation 
during both training and testing phases,following (Zhu et 
al. 2023; Zhao et al. 2023). During training, the batch size 
is set to 2. The AdamW [49] optimizer is employed for 
network parameter optimization, with training conducted 
for 120 epochs and a weight decay of 0.01. The initial 
learning rate is set to 5e-4, with an exponential decay 
schedule applied using a decay factor of 0.99. 
For experiments on Human3.6M, 2D pose inputs are 
obtained from either the Stacked Hourglass detector [50] 
or the ground-truth 2D poses provided by the dataset. For 
MPI-INF-3DHP, ground-truth 2D poses from the dataset 
are used as input. Other key hyperparameters include a 
feature dimension D of 128, 8 attention heads (h=8) in the 
attention branch, and 2 nearest neighbors in the temporal 
adjacency matrix. 
4.3  Comparison experiment 
Results on Human3.6M. We compare the proposed 
SBAHGNet with other methods on the Human3.6M 
dataset (as shown in Table 1). To ensure a fair 
comparison, only results from models without pre-
training on additional data are included. SBAHGNet 
achieves an MPJPE of 37.2 mm with estimated 2D pose 
inputs and 12.4 mm with ground-truth 2D pose inputs. 
Notably, compared to MotionBERT [52], our method 
utilizes only 43% of its parameter count and 51% of its 
computational resources, while improving accuracy by 

 
 
9
2.0 mm and 5.4 mm, respectively. Furthermore, compared to 
another 
Table 1  Quantitative comparisons on Human3.6M. T: Number of input frames. CE: Estimating center frame only. P1: MPJPE error (mm). P2: 
P-MPJPE error (mm). P1†: P1 error on 2D ground truth. (*) denotes using HRNet[] for 2D pose estimation. The best and second-best scores 
are in bold and underlined, respectively. For detailed per-action results, please refer to Table 3 as shown. 
Method 
T 
CE 
Param 
MACs 
P1↓/P2↓ 
P1†↓ 
MHFormer [39] CVPR’22 
351 
√ 
30.9M 
7.0G 
43.0/34.3 
30.5 
P-STMO [51] ECCV’22 
243 
√ 
6.2M 
0.7G 
42.8/34.4 
29.3 
STCFormer [40] CVPR’23 
243 
× 
4.7M 
19.6G 
41.0/32.0 
21.3 
STCFormer-L [40] CVPR’23 
243 
× 
19.9M 
78.2G 
40.5/31.8 
- 
PoseFormerV2 [28] CVPR’23 
243 
√ 
14.4M 
4.8G 
45.2/35.6 
- 
GLA-GCN [13] ICCV’23 
243 
√ 
1.3M 
1.5G 
44.4/34.8 
21.0 
MotionBERT [52] ICCV’23 
243 
× 
42.3M 
174.8G 
39.2/32.9 
17.8 
HDFormer [53] IJCAI’23 
96 
× 
3.7M 
0.6G 
42.6/33.1 
21.6 
HSTFormer [54] arXiv’23 
81 
× 
22.7M 
1.0G 
42.7/33.7 
27.8 
MotionAGFormer-L [14] WACV’24 
243 
× 
19.0M 
78.3G 
38.4/32.5 
17.3 
TCPFormer [55] AAAI’25 
243 
× 
35.1M 
109.2G 
37.9/31.7 
15.5 
SBAHGNet 
243 
× 
18.3M 
88.9G 
37.2/31.6 
12.4 
 
Table 2  Quantitative comparisons on MPI-INF-3DHP. T: Number of 
input frames. The best and second-best scores are in bold and 
underlined, respectively. (ties are marked accordingly). 
Method 
T PCK↑ AUC↑ P1↓ 
MHFormer [39] CVPR’22 
9 
93.8 
63.3 
58.0 
P-STMO [51] ECCV’22 
81 
97.9 
75.8 
32.2 
STCFormer [40] CVPR’23 
81 
98.7 
83.9 
23.1 
PoseFormerV2 [28] CVP