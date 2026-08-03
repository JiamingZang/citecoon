# PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model

> 2025 · id: W4409368373 · arXiv: 2408.03540 · pdf: https://ojs.aaai.org/index.php/AAAI/article/download/32401/34556 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Transformers have significantly advanced the field of 3D hu-
man pose estimation (HPE). However, existing transformer-
based methods primarily use self-attention mechanisms for
spatio-temporal modeling, leading to a quadratic complex-
ity, unidirectional modeling of spatio-temporal relationships,
and insufficient learning of spatial-temporal correlations. Re-
cently, the Mamba architecture, utilizing the state space
model (SSM), has exhibited superior long-range modeling
capabilities in a variety of vision tasks with linear complexity.
In this paper, we propose PoseMamba, a novel purely SSM-
based approach with linear complexity for 3D human pose es-
timation in monocular video. Specifically, we propose a bidi-
rectional global-local spatio-temporal SSM block that com-
prehensively models human joint relations within individual
frames as well as temporal correlations across frames. Within
this bidirectional global-local spatio-temporal SSM block, we
introduce a reordering strategy to enhance the local modeling
capability of the SSM. This strategy provides a more logi-
cal geometric scanning order and integrates it with the global
SSM, resulting in a combined global-local spatial scan. We
have quantitatively and qualitatively evaluated our approach
using two benchmark datasets: Human3.6M and MPI-INF-
3DHP. Extensive experiments demonstrate that PoseMamba
achieves state-of-the-art performance on both datasets while
maintaining a smaller model size and reducing computational
costs. The code and models will be released.

## introduction
3D human pose estimation from monocular observations is
a fundamental task in computer vision with various real-
world applications (Mehta et al. 2017b; Wiederer et al. 2020;
Czech et al. 2022; Bauer et al. 2023; Munea et al. 2020).
Typically, this involves two separate steps: 2D pose detec-
tion to locate keypoints on the image plane, followed by
2D-to-3D lifting to determine joint positions in 3D space
from 2D keypoints. Recovering accurate 3D pose from 2D
keypoints is challenging due to depth ambiguity and self-
occlusion in monocular data. To address these challenges,
significant advancements in deep learning approaches have
been made, consistently improving performance (Liu et al.
2020; Chen et al. 2020; Zeng et al. 2020; Wang et al. 2020).
Recently, transformers (Vaswani et al. 2017) have demon-
*Corresponding author
Copyright © 2025, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
0
100
200
300
400
500
600
700
MACs/frame (M)
38
39
40
41
42
43
44
45
MPJPE (mm)
MHFormer
MixSTE
P-STMO
Stridedformer
Einfalt
STCFormer
STCFormer-L
PoseFormerV2
GLA-GCN
MotionBERT
HDFormer
HSTFormer
DC-GCT
MotionAGFormer-L
PoseMamba-S
PoseMamba-B
PoseMamba-L
Figure 1: Comparisons of recent 3D human pose estima-
tion techniques on Human3.6M (Ionescu et al. 2013) (lower
is better). MACs/frame represents multiply-accumulate op-
erations for each output frame. Our PoseMamba method
presents various versions and achieves superior results,
while maintaining computational efficiency.
strated significant potential in 3D human pose estimation.
Its self-attention mechanism enables it to efficiently capture
spatio-temporal relationships for this domain. For example,
PoseFormer (Zheng et al. 2021) leverages spatio-temporal
information to estimate more accurate central-frame pose in
video sequence. MHFormer (Li et al. 2022b) learns spatio-
temporal representations of multiple pose hypotheses in an
end-to-end manner. MixSTE (Zhang et al. 2022) proposes an
alternating design using a transformer-based seq2seq model
to capture the coherence between sequences. However, ap-
plying full attention mechanisms to long 2D keypoints se-
quence results in a notable rise in computational require-
ments, due to the quadratic complexity of attention calcula-
tions in both computation and memory. This naturally raises
the question: how can a method be designed to function with
linear complexity while still preserving the advantages of
capturing spatio-temporal information?
We observe recent progress in state space models (Gu and
Dao 2023; Wang et al. 2023; Islam and Bertasius 2022), par-
ticularly with the emergence of the structured state space
squence model (S4) (Gu, Goel, and R´e 2021) as a promis-
ing architecture for sequence modeling. Building upon S4,
Mamba (Gu and Dao 2023) incorporates time-varying pa-
rameters into the SSM, introducing an efficient hardware-
aware algorithm with global receptive fields and linear com-
plexity. Recently, a few concurrent approaches (Zhu et al.
2024; Liu et al. 2024) have focused on 2D vision tasks, such
arXiv:2408.03540v2  [cs.CV]  15 Dec 2024

as classification and segmentation.
Driven by the successes of SSM in 2D image process-
ing, we propose Pose State Space Model (denoted as Pose-
Mamba), which features bidirectional global-local spatial-
temporal modeling with linear complexity. We aim to ex-
plore the potential of SSM in 3D human pose estimation.
Through pilot tests, we have observed that relying solely on
Mamba (Gu and Dao 2023) may not lead to optimal perfor-
mance. We hypothesize that the issue arises from the uni-
directional modeling approach of the standard SSM. To ad-
dress this, we propose a bidirectional global-local spatial-
temporal modeling approach for 3D human pose estima-
tion. Here, global refers to spatial modeling that captures
the full-body pose, while local pertains to spatial model-
ing focused on the limbs and their detailed movements.
Specifically, within this bidirectional global-local spatio-
temporal SSM block, we introduce a reordering strategy
to enhance the local modeling capability of the SSM. This
strategy provides a more logical geometric scanning order
and integrates it with the global SSM, resulting in a com-
bined global-local spatial scan. Experimental results on Hu-
man3.6M and MPI-INF-3DHP demonstrate the effective-
ness of our method. Our PoseMamba surpasses the previous
state-of-the-art (SOTA) methods while having fewer param-
eters and MACs, demonstrating the potential of SSM in 3D
human pose estimation, as shown in Figure 1.
In summary, the main contributions of our work are:
• To the best of our knowledge, we are the first to introduce
a novel bidirectional global-local spatio-temporal mod-
eling approach and logical geometric scanning strategy
within the Mamba framework, PoseMamba, for 3D HPE
under the category of 2D-to-3D lifting.
• We propose bidirectional global-local spatial-temporal
modeling, enabling the PoseMamba to sufficiently learn
global-local spatial-temporal information with linear
complexity, exploiting the human skeleton geometry.
• Efficiency and Flexibility: i) Our PoseMamba is distin-
guished by its lightweight design and faster speed with
fewer parameters compared to previous SOTA meth-
ods, while maintaining promising accuracy. Specifically,
PoseMamba is 2.8× faster than MotionAGFormer and re-
duces 64.7% GPU memory when performing batch in-
ference to achieve 3D pose from 2D pose estimation at
the frame of 243. ii) To accommodate diverse needs, we
provide various versions of PoseMamba, allowing users
to choose a balanced option between accuracy and speed
based on their specific requirements.
• Without bells and whistles, our PoseMamba model
achieves state-of-the-art results on both Human3.6M and
MPI-INF-3DHP datasets.

## method
T
MPJPE↓
MHFormer (Li et al. 2022b)
9
58.0
MixSTE (Zhang et al. 2022)
27
54.9
P-STMO (Shan et al. 2022)
81
32.2
Einfalt et al. (Einfalt, Ludwig, and Lienhart 2023)
81
46.9
STCFormer (Tang et al. 2023)
81
23.1
PoseFormerV2 (Zhao et al. 2023)
81
27.8
GLA-GCN (Yu et al. 2023)
81
27.7
HSTFormer (Qian et al. 2023)
81
41.4
HDFormer (Chen et al. 2023)
96
37.2
MotionAGFormer-XS
27
19.2
MotionAGFormer-S
81
17.1
MotionAGFormer-B
81
18.2
MotionAGFormer-L
81
16.2
PoseMamba-S
27
17.79
PoseMamba-S
81
15.27
PoseMamba-B
81
14.51
model variants compared to others in terms of MPJPE, as il-
lustrated in Table 3, showcasing the excellence of our model.
Ablation Studies
To evaluate the impact and performance of each component
in our model, we evaluate their effectiveness in this section.
Bidirectional Global-Local Spatio-Temporal Modeling
We perform comprehensive experiments to verify the effec-
tiveness of modifying the crucial bidirectional global-local
spatio-temporal modeling in PoseMamba on Human3.6M
using our small variant version, where feature dimensions
are altered to ensure comparable architectural parameters
and MACs for a fair evaluation. As shown in Table 4,
employing unidirectional spatio-temporal modeling results
in a model performance of MPJPE ranging from 43.0 to
43.8 mm, which is comparatively less efficient than the bidi-
rectional spatio-temporal modeling yielding an MPJPE of
42.4 mm. Furthermore, integrated with the local spatial scan
to enhance accurate limb prediction, our final model is
0.6 mm better than bidirectional spatial-temporal modeling,
which indicates the efficacy of our global-local modeling.
Table 4: Ablation study for various spatial-temporal model-
ing with MPJPE on Human3.6M.
Spatial-Temporal Modeling Strategy
T
Params
MACs
MPJPE
Unidirectional Spatial-Temporal 1
243
0.860 M
3.587 G
43.1
Unidirectional Spatial-Temporal 2
243
0.860 M
3.587 G
43.2
Unidirectional Spatial-Temporal 3
243
0.860 M
3.587 G
43.8
Unidirectional Spatial-Temporal 4
243
0.860 M
3.587 G
43.0
Bidirectional Spatial-Temporal
243
0.860 M
3.587 G
42.4
Bidirectional Global-Local Spatial-Temporal
243
0.860 M
3.587 G
41.8
Effect of Loss Function
We explore the contribution of
our loss function using our small variant version in detail.
As shown in Table 5, the MPJPE metric decreases from
43.7 to 43.5 mm after applying the 2D loss and decreases
from 43.5 to 42.1 mm after applying the T-Loss. The result
demonstrates that the T-Loss and 2D loss is an essential loss
to improve accuracy. Finally, after applying the T-Loss, 2D-
loss, and MPJPE loss to our method, the result achieves the
best on the MPJPE metrics 41.8 mm. The results demon-
strate that our loss function is comprehensive for the pro-
posed model regarding accuracy and smoothness.
Table 5: Ablation study for loss function with MPJPE and
PMPJPE on Human3.6M.
Loss
MPJPE↓/PMPJPE↓
MPJPE Loss
43.7/36.5
MPJPE Loss + 2D-Loss
43.5/36.2
MPJPE Loss + T-Loss
42.1/35.1
Ours (MPJPE Loss + T-Loss + 2D-Loss)
41.8/35.0
Parameter Setting Analysis
Table 6 shows how the set-
ting of different hyper-parameters in our method impacts

Table 6: The P1 error comparison by varying num-
ber of PoseMamba blocks and number of channels on
Human3.6M. dm: Number of channels in each Pose-
Mamba block. T is kept 243 in all experiments.
Depth
dm
Param
MACs
P1
12
64
0.516 M
2.2 G
43.0
16
64
0.688 M
2.9 G
42.0
20
64
0.860 M
3.6 G
41.8
24
64
1.031 M
4.3 G
41.5
32
64
1.375 M
5.7 G
41.5
40
64
1.719 M
7.2 G
41.1
48
64
2.062 M
8.6 G
41.1
40
32
0.450 M
1.9 G
43.1
40
64
1.719 M
7.2 G
41.1
40
128
6.714 M
27.9 G
38.1
40
256
26.535 M
109.9 G
37.1
12
256
7.963 M
33.0 G
39.1
20
128
3.358 M
13.9 G
40.8
the performance under Protocol 1 with MPJPE. There are
three main hyper-parameters for the network: the depth of
PoseMamba (N), the dimension of model (dm), and the in-
put sequence length (T). We divide the configurations into
2 groups row-wise, and different values are assigned for
one hyper-parameters while keeping the other two hyper-
parameters fixed to evaluate the impact and choice of each
configuration. In addition to these two sets of experiments,
we have also conducted additional hyperparameter experi-
ments. Based on the results in the table, considering perfor-
mance and efficiency, we choose three variants in Table 2.
Qualitative Analysis
Figure 5 visualizes last spatio-temporal SSM block map of
action (Walking of testset S9). It can be easily observed from
spatial map (left of Figure 5) that our model learns distinct
dependencies between joints. Furthermore, we also visualize
the temporal map (right of Figure 5). The two light-colored
parts have similar poses in adjacent frames, while dark-
colored frame (the middle image in the frame sequence)
has a more distinct pose in adjacent frames. Figure 6 com-
pares PoseMamba-L with recent approaches, which shows
that our PoseMamba achieves more accurate poses than Mo-
tionBERT and MotionAGFormer. Moreover, Figure 7 shows
the qualitative comparison on some wild videos. It is evident
that our method can produce more accurate 3D poses, par-
ticularly in cases the human action is complex and rare.
x
y
[0] Hip
[1] R Hip
[2] R Knee
[3] R Foot
[4] L Hip
[5] L Knee
[6] L Foot
[7] Spine
[8] Thorax
[9] Neck
[10] Head
[11] L Shoulder
[12] L Elbow
[13] L Wrist
[14] R Shoulder
[15] R Elbow
[16] R Wrist
Figure 5: Visualization of SSM map among body joints and
frames.
MotionBERT
MotionAGFormer
PoseMamba(ours)
(a)
(b)
(c)
Figure 6: Qualitative comparisons with MotionBERT and
MotionAGFormer. The gray skeleton is the ground-truth 3D
pose and the blue skeleton is the estimated body.
Input
MotionBERT
MotionAGFormer
PoseMamba
Figure 7: Qualitative comparisons with MotionBERT and
MotionAGFormer on challenging wild videos. Wrong esti-
mations are highlighted by yellow arrows.

## experiments
We evaluate our proposed PoseMamba on two large-
scale 3D human pose estimation datasets, i.e., Hu-
man3.6M (Ionescu et al. 2013) and MPI-INF-3DHP (Mehta
et al. 2017a).
Datasets and Evaluation Metrics
Human3.6M is a commonly used indoor dataset for 3D hu-
man pose estimation. It contains 3.6 million video frames
of 11 subjects performing 15 different daily activities. To
ensure fair evaluation, we follow the standard approach and
train the model using data from subjects 1, 5, 6, 7, and 8, and
then test it on data from subjects 9 and 11. Following the
previous work (Zhu et al. 2023), we use two protocols for
evaluation. The first protocol (referred to as P1) uses Mean
Per Joint Position Error (MPJPE) in millimeters between
the estimated pose and the actual pose, after aligning their
root joints (sacrum). The second protocol (referred to as P2)
measures Procrustes-MPJPE, where the actual pose and the
estimated pose are aligned through a rigid transformation.
MPI-INF-3DHP is another large-scale dataset gathered in
three different settings: green screen, non-green screen, and
outdoor environments. This dataset has 1.3 million frames,
containing a wider range of movements than Human3.6M.
We utilize MPJPE as the evaluation metric.
Implementation Details
Model Variants
We create three model configurations, de-
tailed in Table 2. Our base model, PoseMamba-B, balances
accuracy and computational cost. Other variants are named
based on parameters and computational needs. The selec-
tion of each variant depends on specific application needs,
like real-time processing or precise estimations. The MLP’s
expansion layer is α = 2 for all experiments.
Experimental settings
Our model is developed utilizing
PyTorch and deployed on one NVIDIA RTX 3090 GPU.
Horizontal flipping augmentation is applied for both train-
ing and testing, as outlined in (Zhu et al. 2023; Zhao et al.
2023). During model training, the batch size is configured
with 4 sequences. The optimization of network parameters is
carried out using the AdamW (Loshchilov and Hutter 2017)
optimizer across 120 epochs with a weight decay of 0.01.
The initial learning rate is established at 2e−4 with an expo-
nential learning rate decay schedule, utilizing a decay fac-
tor of 0.99. In our approach, we leverage the Stacked Hour-
glass (Newell, Yang, and Deng 2016) 2D pose detection out-
comes and 2D ground truths sourced from the Human3.6M
and MPI-INF-3DHP datasets, following MotionBERT (Zhu
et al. 2023). In MPI-INF-3DHP, we employ ground truth 2D
detection using a methodology following methods (Zhao
et al. 2023; Tang et al. 2023).
Performance comparison on Human3.6M
We present a comparative analysis of our PoseMamba model
against other models using the Human3.6M dataset. To en-
sure a fair assessment, only the outcomes of models with-
out additional pre-training on supplementary data are con-
sidered. The results, as detailed in Table 1, reveal that
PoseMamba-L achieves a P1 error of 38.1 mm for estimated
2D pose and 15.6 mm for ground truth 2D pose. Notably,
these results are accomplished with only 16% of the com-
putational resources in comparison to the previous SOTA
model, MotionBERT, while exhibiting an enhanced accu-
racy of 1.1 mm and 2.2 mm, respectively. Furthermore, our
model achieves these results using only 36% of the computa-
tional resource compared to another previous SOTA model,
MotionAGFormer (Mehraban, Adeli, and Taati 2024), while
being 0.3 mm and 1.7 mm more accurate, respectively.
Performance comparison on MPI-INF-3DHP
When assessing our approach to the MPI-INF-3DHP
dataset, we adapted our small and base models to accommo-
date 27 and 81 frames to suit the shorter video sequences.
Our method demonstrates superior performance across all

Table 1: Quantitative comparisons on Human3.6M. T: Number of input frames. CE: Estimating center frame only. P1: MPJPE
error (mm). P2: P-MPJPE error (mm). P1†: P1 error on 2D ground truth. (*) denotes using HRNet (Sun et al. 2019) for 2D
pose estimation. The best and second-best scores are in bold and underlined, respectively.

## related_work
State Space Model
We can think of SSM as linear time-
invariant (LTI) system that maps input x(t) ∈RL to output
y(t) ∈RL via hidden state h(t) ∈CN. It can be described
as linear ordinary differential equations (ODEs):
˙h(t) = Ah(t) + Bx(t)
y(t) = Ch(t) + Dx(t)
(1)

Here, ˙h(t) represents the time derivative of the hidden
state vector h(t), A ∈CN×N, B, C ∈CN, and D ∈C1
represent the weighting parameters.
Discretization of SSM
To process discrete sequence in-
puts, continuous-time SSMs must be discretized, typically
accomplished by solving the ODE followed by a simple dis-
cretization technique. Specifically, the analytical solution to
Equation (1) can be represented as:
h(tb) = eA(tb−ta)(h(ta) +
Z tb
ta
B(τ)x(τ)e−A(τ−ta) dτ)
(2)
Subsequently, through sampling with step size ∆(i.e.,
dτ|ti+1
ti
= ∆i), h(tb) can be discretized as:
hb = eA(Pb−1
i=a ∆i)
 
ha +
b−1
X
i=a
Bixie−A(Pi
j=a ∆j)∆i
!
(3)
Notably, this discretization approach is roughly equiva-
lent to the outcome achieved through the zero-order hold
(ZOH) technique(Gu and Dao 2023), commonly found in
SSM-related literature.
To provide a specific example, when b = a + 1, Equa-
tion (3) can be expressed as:
ha+1 = Aaha + Baxa
(4)
Here, Aa = eA∆a corresponds to the ZOH discretiza-
tion result (Gu and Dao 2023), while Ba = Ba∆a essen-
tially represents the first-order Taylor expansion of the ZOH-
derived equivalent.
Selective Scan
The weight matrix B in Equation (2) and
Equation (3), along with C, D, and ∆, is tailored to be
input-dependent to overcome the limitations of LTI SSMs
(Equation (1)) in capturing contextual details (Gu and Dao
2023). However, the introduction of time-varying SSMs
presents a computational challenge because convolutions
with dynamic weights are not supported, making them un-
suitable for this purpose. Nonetheless, deriving the recur-
rence relation of hb in Equation (3) enables efficient compu-
tation. Specifically, if we define eA(∆a+...+∆i−1) as pi
A,a,
its recurrence relation can be expressed as
pi
A,a = eA∆i−1pi−1
A,a
(5)
Regarding the second term of Equation (3), we obtain
pb
B,a = eA(∆a+...+∆b−1)
b−1
X
i=a
Bixie−A(∆a+...+∆i)∆i (6)
Therefore, utilizing the relationships derived in Equa-
tion (5) and Equation (6), the computation of hb
=
pb
A,aha+pb
B,a can be efficiently parallelized using associa-
tive scan algorithms (Martin and Cundy 2017; Smith, War-
rington, and Linderman 2022), which are facilitated by var-
ious contemporary programming libraries.
PoseMamba
As illustrated in Figure 2, our network processes a concate-
nated 2D coordinate array CT,J ∈RT ×J×2 representing J
joints across T frames. The input has a channel size of 2.
Initially, we project the input keypoint sequence CT,J
into a high-dimensional feature PT,J ∈RT ×J×dm with
each joint represented by a feature dimension of dm. Sub-
sequently, we incorporate a spatial and a temporal posi-
tion embedding matrix to preserve positional details across
spatial and temporal domains. The proposed PoseMamba
takes PT,J as input and focuses on capturing global bidi-
rectional spatial-temporal information efficiently through
Mamba blocks with linear complexity. Lastly, we employ
a regression head to combine the encoder’s outputs Z ∈
RT ×J×dm, adjusting the dimension from dm to 3 to derive
the 3D human pose sequence Out ∈RT ×J×3.
2D Keypoints
3D Pose Sequence 
Spatial Embedding
Spatial-Temporal
Mamba Block 𝐿𝐿2
Temporal Embedding
Regression Head
Spatial-Temporal
Mamba Block 𝐿𝐿𝑁𝑁
…
PoseMamba
Layer Norm
Bidirectional Global-Local 
Spatial-Temporal SSM
Layer Norm
MLP
Spatial-Temporal
Mamba Block 
Figure 2: The pipeline of our PoseMamba. We start by us-
ing fully connected layer to project the input keypoint se-
quence, and then embed position and temporal embedding
matrix into sequence. After that, we feed the sequence into
the Mamba blocks.
Spatio-Temporal Encoder
Transformer-Based
Spatio-Temporal
Correlation
Learning
Prior transformer-based studies have primarily
concentrated on utilizing multi-head self-attention mech-
anisms to understand spatio-temporal relationships, as
illustrated in Fig. 3(a). The computation of attention for
the query, key, and value matrices Q, K, V in each head is
expressed as:
Attention(Q, K, V ) = Softmax(QKT
√dm
)V,
(7)
where {Q, K, V } ∈RO×dm, O indicates the number of to-
kens, and dm is the dimension of each token.

Bidirectional Global-Local Spatio-Temporal Modeling
In contrast to prior methods using attention mechanisms
with quadratic computational complexity, we propose a state
space model to encapsulate comprehensive spatio-temporal
information at a linear complexity. Specifically, inspired by
T
S
(a) Self-attention mechanism
T
S
(b)
Bidirectional
Spatio-
Temporal scan mechanism
T
S
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
14
15
16
11
12
13
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
0 0 1 2 3 0 4 5 6 8 11 12 13 8 14 15 16
Global Spatial Scan
Local Spatial Scan
…
Fusion
Global-Local Spatial Scan
(c) Bidirectional Global-Local Spatio-Temporal scan mechanism
Figure 3: Illustration of various spatio-temporal model-
ing mechanisms. (a) Self-attention (Vaswani et al. 2017;
Dosovitskiy et al. 2020). (b) Bidirectional spatio-temporal
scan (Liu et al. 2024). (c) Our proposed bidirectional global-
local spatio-temporal scan mechanism, which leverages the
geometry of the human skeleton to enhance detail.
VMamba (Liu et al. 2024), before inputting the tokens into
the S6 model, we reorganize the tokens in both spatial and
temporal dimensions, specifically forward spatial scan, for-
ward temporal scan, backward spatial scan, and backward
temporal scan, as depicted in Figure 3(b). Subsequently, the
resultant features are merged. This approach enables the
model to obtain comprehensive bidirectional global spatio-
temporal information from bidirectional spatial and tempo-
ral dimensions. Furthermore, the computational complex-
ity remains at linear complexity in contrast to the self-
attention operation with quadratic complexity in transformer
Figure 3(a). To better demonstrate the benefits of bidirec-
tional spatio-temporal modeling, we conduct experiments on
four unidirectional spatio-temporal scan mechanisms, as de-
picted in Figure 4, which demonstrates that relying solely on
Mamba can not achieve optimal performance.
Furthermore, to address the persistent challenge of in-
accurate limb prediction, we introduce a novel reorder-
ing strategy designed to augment the local modeling ca-
pabilities of the state space model. This enhancement is
T
S
Unidirectional Spatio-Temporal 
scan mechanism 1
T
S
Unidirectional Spatio-Temporal
 scan mechanism 2
T
S
Unidirectional Spatio-Temporal 
scan mechanism 4
T
S
Unidirectional Spatio-Temporal
 scan mechanism 3
Figure 4: Illustration of different unidirectional spatio-
temporal scan mechanisms.
achieved by establishing a more rational geometric scan-
ning sequence, which is then seamlessly integrated with the
global SSM framework. This integration facilitates a com-
prehensive global-local spatial scanning approach, as illus-
trated in Figure 3(c). Our proposed strategy not only re-
fines the spatial scanning process but also ensures a harmo-
nious fusion of local details with the broader spatial context,
thereby significantly improving the precision of limb predic-
tions. Specifically, we posit that scanning key points on the
human skeleton from 0 to 16 enables the extraction of global
spatial features. However, our experimental findings indicate
that relying only on global scanning consistently led to inac-
curate limb prediction. Therefore, exploiting the interactions
between body joints, we propose a local scanning approach
to capture local human skeleton details, as detailed in Fig-
ure 3(c). We design a global-local spatial scanning approach
by merging these two scanning sequences. Additionally, by
incorporating temporal scanning, we develop a bidirectional
global-local spatio-temporal mamba block, advancing the
modeling of spatio-temporal features for 3D HPE.
Bidirectional Global-Local Spatio-Temporal Mamba
Block
For each spatio-temporal Mamba block, layer nor-
malization (LN), bidirectional spatio-temporal SSM, depth-
wise convolution (Chollet 2017), and residual connections
are employed. A spatio-temporal Mamba block is shown in
Figure 2, and the output can be summarized as follows:
Z′
l = LN(SSM(σ(DW(LN(Zl−1))))) + Zl−1,
Zl = MLP(LN(Z′
l)) + Z′
l,
(8)
where Zl ∈RT ×J×C is the output of the l-th block. DW
means the depth-wise convolution. Following the DW, a
SiLU (Hendrycks and Gimpel 2016) and SSM are adopted.
Spatio-Temporal Correlation Learning
We employ the
bidirectional global-local spatio-temporal Mamba blocks
to learn spatio-temporal correlations among joints in over
frames. Firstly, we take 2D keypoints sequence as input
CT,J
∈RT ×J×2 and project each keypoint to a high-
dimensional feature PT,J ∈RT ×J×dm with the linear em-
bedding layer. We then embed the spatial position informa-
tion with a positional matrix Espos ∈RJ×dm. Each joint
token p

## conclusion
We present PoseMamba, a novel SSM-based approach for
3D human pose estimation, which has a bidirectional global-
local spatio-temporal mamba block to comprehensively
model the human joint relations within each frame as well as
the temporal correlations across frames. In the bidirectional
global-local spatio-temporal mamba block, we propose a re-
ordering strategy to enhance SSM’s local modeling ability
by providing a more logical geometric scanning order and
fusing it with global SSM to get global-local spatial scan.
Experimental results demonstrate that PoseMamba outper-
forms the existing counterparts on both datasets while sig-
nificantly reducing parameters and MACs. As a newcomer
to 3D human pose estimation, PoseMamba is a promising
option for constructing 3D vision foundation models, and
we hope it can offer a new perspective for the field.
Acknowledgements. This work was supported by the Na-
tional Natural Science Foundation of China under Grant
62406120 and the Guangxi Science and Technology Project
(GuiKe-AB21196034).