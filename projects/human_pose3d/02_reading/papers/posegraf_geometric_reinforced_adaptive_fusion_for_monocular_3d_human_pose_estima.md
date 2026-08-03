# Posegraf: Geometric-Reinforced Adaptive Fusion for Monocular 3d Human Pose Estimation

> 2025 · id: W4412612582 · arXiv: 2506.14596 · 来源: arxiv
> 注：此为抽取文本，公式/图表可能失真；关键细节以原 PDF 为准。

## abstract
Existing monocular 3D pose estimation methods primarily rely on joint positional features, while 
overlooking intrinsic directional and angular correlations within the skeleton. As a result, they often 
produce implausible poses under joint occlusions or rapid motion changes. To address these challenges, 
we propose the PoseGRAF framework. We first construct a dual graph convolutional structure that 
separately processes joint and bone graphs, effectively capturing their local dependencies. A Cross-
Attention module is then introduced to model interdependencies between bone directions and joint 
features. Building upon this, a dynamic fusion module is designed to adaptively integrate both feature 
types by leveraging the relational dependencies between joints and bones. An improved Transformer 
encoder is further incorporated in a residual manner to generate the final output. Experimental results on 
the Human3.6M and MPI-INF-3DHP datasets show that our method exceeds state-of-the-art approaches. 
Additional evaluations on in-the-wild videos further validate its generalizability. The code is publicly 
available at https://github.com/iCityLab/PoseGRAF.

## introduction
Monocular 3D human pose estimation is a fundamental task in computer vision that aims to predict 
human body poses in 3D space from a single RGB image. It serves as a key enabling technology for a 
wide range of applications, including motion analysis[1], human-computer interaction[2], [3] and 
virtual/augmented reality[4] Existing approaches are predominantly divided into two technical 
paradigms: direct regression of 3D poses from RGB inputs[5], [6] and 2D-to-3D lifting based on detected 
2D keypoints[7], [8], [9]. Direct regression methods typically adopt end-to-end convolutional neural 
networks (CNNs) to estimate 3D poses. In contrast, 2D-to-3D lifting methods first detect 2D keypoints 
from input images and then infer 3D joint locations. Benefiting from well-established 2D pose detectors, 
these methods often achieve superior accuracy in practice. Despite the progress, existing 2D-to-3D lifting 
approaches still face two key limitations: (1) they rely heavily on 2D joint coordinates, which overlooks 
the underlying structural relationships between joints [10]. (2) they fail to effectively integrate geometric 
constraints such as bone directions and joint angles [11], [12]. In particular, conventional methods tend 
to treat bone directions and joint angles as independent limit conditions without modeling their intrinsic 
correlation, leading to inaccurate pose predictions under complex motion patterns. To address these 
challenges, we propose PoseGRAF, a novel framework for 3D human pose estimation that integrates 
geometry-aware graph representation with adaptive feature fusion. PoseGRAF explicitly captures joint 

angle relationships on the skeleton graph to enhance the representation of bone directions. Specifically, 
we propose a dual-graph approach to model bone direction relationships, consisting of: (i) a weighted 
graph, where nodes represent bones and edge weights encode angles between adjacent bones; and (ii) an 
unweighted graph, where nodes also represent bones and edges indicate binary connectivity (1 for 
connected, 0 for not connected). Based on this, we design a geometry-enhanced joint embedding method, 
which integrates Cross-Attention and Joint GCN to extract joint features and employ Bone Direction 
GCN to integratively encode bone direction and angle information. Furthermore, we design an attention-
based dynamic feature fusion module that adaptively fuses positional and geometric features, and co-
constructs a residual structure with an improved Transformer encoder. The proposed architecture 
alleviates unreasonable pose predictions during fast or intricate motions. Extensive experiments on two 
benchmark datasets, Human3.6M [13] and MPI-INF-3DHP [14] demonstrate that our method 
outperforms state-of-the-art approaches across multiple metrics, validating its effectiveness and 
robustness. The main contributions of this work can be summarized as follows: 
 (1) We design a geometry-enhanced graph to explicitly model the relationships between bone 
directions and their connections, overcoming the limitations of traditional joint graphs in angle 
representation. A graph convolution module is designed to effectively capture the spatial correlation of 
bone directions. 
 (2) The proposed attention-based dynamic feature fusion module adaptively integrates joint 
position and bone direction features. 
 (3) Comprehensive evaluations conducted on the Human3.6M and MPI-INF 3DHP datasets 
demonstrate that the proposed method achieves superior performance compared to existing state-of-the-
art approaches.

## method
As shown in Fig. 2(a), we propose PoseGRAF, a novel 3D human pose estimation model based on 
Graph Convolutional Networks (GCN) and Transformer, designed to enhance 3D human pose estimation 
performance by leveraging advanced graph-based and attention mechanisms to capture the intricate 
relationships within human skeletal structures. 
 
Fig.2. (a) Overview of the proposed framework. 
 denotes the concatenation of bone directional features and 
joint features. (b) Transformer encoder module: 
 represents the relative distance matrix of human body 
topology, 
 indicates feature embeddings processed by Cross-Attention, 
 corresponds to embeddings from 
Dynamic Fusion. (c) Bone-Directional graph convolution module. (d) Dynamic fusion module. 
3.1  Overview of the network 
The PoseGRAF framework consists of five modules: Joint GCN, Bone GCN, Cross-Attention, 
Dynamic Fusion, and Transformer Encoder. We begin by extracting 2D keypoints from the input image 
using the CPN detector[30], followed by constructing both directed weighted and undirected graphs to 
represent skeletal structures. The Bone GCN extracts directional and angular relational features from the 

skeletal graph. In parallel, the Joint GCN module aggregates features from adjacent nodes to model local 
spatial dependencies among joints. Next, the Cross-Attention mechanism allows joint features to attend 
to bone direction features, enhancing the joint representations by incorporating relevant directional 
information. The Dynamic Fusion module then adaptively integrates the refined joint features with the 
bone direction features. These fused features are processed by an improved Transformer Encoder, 
embedded within a residual structure, to generate the final feature representations. Finally, a regression 
head linearly projects these features to three-dimensional space, enabling accurate estimation of the 3D 
human pose from the 2D input. 
3.2 Graph Convolutional Networks 
PoseGRAF employs a dual-stream graph convolutional network Architecture comprising Joint 
GCN and Bone-Direction GCN. The former aims to model local spatial dependencies between human 
joints, while the latter employs joint angles to generate weighted representations of geometric 
correlations in bone directions. 
Joint GCN. We represent joint features as a graph 𝐺𝐽= (𝑉𝐽, 𝐴𝐽), where the vertex set 𝑉𝐽 contains 
𝑁 joints and the edge connections are defined by an adjacency matrix 𝐴𝐽∈{0,1}𝑁×𝑁 . Specifically, 
𝐴𝐽
(𝑖,𝑗) = 1 if joints 𝑖 and 𝑗 share a physical connection, otherwise 𝐴𝐽
(𝑖,𝑗) = 0 . Let 𝑋𝐽
(𝑙) denote the latent 
representation of pose data at the 𝑙−𝑡ℎ layer，The joint feature representation is updated through graph 
convolution-based neighbor aggregation, formulated as: 
𝑋𝐽
(𝑙+1) =  𝜎(𝐷̃𝐽
−1/2𝐴̃𝐽𝐷̃𝐽
−1/2𝑋𝐽
𝑙𝛩𝐽) 
(1) 
Where 𝐴̃𝐽= 𝐴𝐽+ 𝐼𝑁 denotes the self−loop augmented adjacency matrix. 𝐷̃𝐽
−1/2 represents the 
normalized node degree diagonal matrix of 𝐴̃𝐽.𝛩𝐽∈ℝ𝐷×𝐷 is a trainable weight matrix. 
Bone Direction GCN. As shown in Fig. 2(c), this module constructs two geometrically enhanced 
graphs: a directed weighted bone graph and a directed unweighted bone graph. The directed weighted 
bone graph is denoted as 𝐺𝐵𝑊= (𝑉𝐵, 𝑊𝐵), where the vertex set 𝑉𝐵 contains M  bone nodes, constructed 
as illustrated in Fig. 1. The feature 𝑣𝐵
𝑝 of a bone node 𝑥𝐵
𝑝 is computed as follows: 
𝑥𝐵
𝑝= 
𝑥𝐽
𝑖−𝑥𝐽
𝑗
‖𝑥𝐽
𝑖−𝑥𝐽
𝑗‖
 
(2) 
where 𝑥𝐽
𝑖 and 𝑥𝐽
𝑗 represent the features of the source joint and target joint of 𝑣𝐵
𝑝 ,respectively. The 
edge weight between bone nodes 𝑣𝐵
𝑝 and 𝑣𝐵
𝑞 is computed as follows: 
𝑤𝐵
(𝑝,𝑞) = { arccos (
𝑥𝐵
𝑝∙𝑥𝐵
𝑞
‖𝑥𝐵
𝑝‖ ‖𝑥𝐵
𝑝‖) ，   𝑖𝑓  𝑣𝐵
𝑝 𝑎𝑛𝑑 𝑣𝐵
𝑝 𝑠ℎ𝑎𝑟𝑒 𝑎 𝑗𝑜𝑖𝑛𝑡 
 0,                                           𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒                                   
 
(3) 
The directed unweighted bone graph is denoted as 𝐺𝐵𝐴= (𝑉𝐵, 𝐴𝐵), where 𝐴𝐵
(𝑝,𝑞) = 1, if bone nodes 
𝑣𝐵
𝑝 and 𝑣𝐵
𝑞 share a common joint, otherwise 𝐴𝐵
(𝑝,𝑞) = 0. Serving as inputs to the Bone Direction GCN 
module,𝐺𝐵𝐴 and 𝐺𝐵𝑊 undergo feature extraction through two separate graph convolutional layers. These 
layers update the representations of bone nodes in the subsequent layer by aggregating information from 
neighboring bone nodes and angular relationships, formulated as follows: 
  𝑋̅𝑊
(𝑙+1) = 𝜎(𝐷̃𝐵
−1/2𝑊̃𝐵𝐷̃𝐵
−1/2𝑋𝐵
𝑙Θ𝑊) 
(4) 
𝑋̅𝐴
(𝑙+1) = 𝜎(𝐷̃𝐵
−1/2𝐴̃𝐵𝐷̃𝐵
−1/2𝑋𝐵
𝑙Θ𝐴) 

                                                𝑋𝐵
(𝑙+1) = 𝑋̅𝑊
(𝑙+1) ⊕𝑋̅𝐴
(𝑙+1) 
Where 𝑊̃𝐵 denotes the angular−weighted adjacency matrix and 𝐴̃𝐵 represents the original bone 
connectivity matrix. 𝐷̃𝐵
−1/2 corresponds to the normalized bone node degree diagonal matrix. Θ𝑊, Θ𝐴 ∈
ℝ𝐷×𝐷 are two independent learnable parameter matrices. The outputs 𝑋̅𝑊
(𝑙+1) and 𝑋̅𝐴
(𝑙+1) from the two 
graph convolutional layers are aggregated through summation to generate the updated node 
representation features for the l+1 layer. The activation function 𝜎(•) employs LeakyReLU to mitigate 
gradient vanishing while preserving feature sparsity, with a negative slope α=0.01.
3.3  Cross-Attention 
This module is designed to capture the intrinsic correlations between human bone directions and 
joints. We concatenate joint features with bone direction features as follows:  
𝑋= [𝑋𝐽
1; 𝑋𝐽
2; … ; 𝑋𝐽
𝑁; 𝑋𝐵
1; 𝑋𝐵
2; … ; 𝑋𝐵
𝑀] 
(5) 
This module takes 𝑋∈ℝ((𝑁+𝑀)×𝐷)  as input, where 𝐷  denotes the embedding dimension. The 
module first computes correlation scores between joints and bone directions using the following 
formulation:  
𝐴̂ℎ;𝑘= [𝑎̂ℎ;𝑘
𝑁+1; 𝑎̂ℎ;𝑘
𝑁+2; … ; 𝑎̂ℎ;𝑘
𝑁+𝑀] ∈ℝ𝑀×𝑁 
(6) 
Here, 𝑎̂ℎ;𝑘
𝑁+𝑖 denotes the attention score vector between the 𝑖−𝑡ℎ joint and all bone directions in the 
ℎ−𝑡ℎ head, reflecting the interactions between joints and bone edges. Following [31], we employ 
Exponential Moving Average (EMA) to aggregate multi-head attention mechanisms across layers. 
𝐴̅ℎ;𝑘= 𝛽∙𝐴̅ℎ;𝑘−1 + (1 −𝛽) ∙𝐴̅ℎ;𝑘 
(7) 
where β=0.99. The final layer’s 𝐴̅ℎ;𝑘 is then employed to aggregate attention vectors from different 
heads across joints and bone directions, yielding the final visual token correlation scores: 
𝑆= 1
𝐻𝑀 ∑∑𝑎̅ℎ;𝑘
(𝑁+𝑖)
𝑀
𝑖=1
𝐻
ℎ=1
 
(8) 
where 𝑎̅ℎ;𝑘
𝑁+𝑖 represents the 𝑖−𝑡ℎ column of matrix 𝐴̅ℎ;𝑘, with 𝐻 denoting the number of attention 
heads. After cross-attention processing, the output features are partitioned into bone direction features 
𝑋𝐵𝐶∈ℝM×D and joint features 𝑋𝐽𝐶∈ℝN×D, enabling independent processing by subsequent modules. 
This design not only preserves the structural information of features but also explicitly models multi-
scale dependencies between key joints and bone directions through the multi-head attention mechanism. 
By incorporating attention mechanisms, this module significantly enhances the model’s capability to 
capture relationships between critical joints and bone directions in human poses, thereby providing richer 
and more precise feature representations for downstream pose estimation tasks. 
3.4  Dynamic_Fusion 
Inspired by An et al.[32], we propose an attention-based dynamic feature fusion mechanism. This 
mechanism effectively fuses joint feature embeddings with bone direction embeddings, as illustrated in 
Fig. 2(d). The Feature Selection function Filter implements adaptive key joint selection through a 
learnable threshold parameter 𝜇 , balancing computational efficiency with precision. Where 𝑆ℎ𝑖𝑔ℎ 
denotes attention scores, based on which the 𝑡𝑜𝑝−𝜇  joint features with the strongest skeletal 
correlations are extracted. This process is a dynamic feature gating operation to learn feature subset 

selection. The graph reconstruction function Reconstruction restores global joint features from individual 
joint node features according to All the Bone Direction Features, with its detailed implementation 
described in Algorithm 1 (lines 5-7). This function achieves feature reconstruction through iterative 
topological diffusion. Specifically, starting from a single selected key joint feature 𝑋𝐽𝐶
(𝑖) as propagation 
seeds, a Breadth-First Search (BFS) is performed based on the topological structure of human skeletal 
graph 𝐺𝐽 . While in incorporating bone direction features 𝑋𝐵  during traversal. The process is 
mathematically formulated as: 
𝒥𝐵
𝑖= 𝐵𝐹𝑆(𝐺𝐽, 𝑋𝐽𝐶
(𝑖), 𝑋𝐵) 
(9) 
𝒥𝐵
𝑖represents the global joint feature obtained from the joint feature 𝑋𝐽𝐶
(𝑖). The BFS process takes as 
input and encodes joint features based on bone direction features 𝑋𝐵. The final joint descriptor features 
are obtained through aggregation and residual connections. This process is formulated as in Eq. (10): 
𝑋𝐷𝐹= ∑
ℐ𝐵
𝑖
𝜇−1
𝑖=0
 +𝑋𝐽𝐶 
(10) 
Where the 𝑋𝐽𝐶 preserves original joint information to prevent gradient vanishing. By integrating 
bone direction features with joint features obtained through attention mechanisms, this module enhances 
the spatial representational capacity of each joint. Through this approach, the model can accurately 
capture geometric relationships between joints and prioritize key points via attention mechanisms.  
Algorithm 1: Dynamic Fusion 
Input: 𝑋𝐽𝐶, 𝑋𝐵, 𝑆ℎ𝑖𝑔ℎ, 𝐺𝐽 
O

## experiments
4.1  Datasets and evaluation metrics 
This section presents comprehensive  studies on two real-world 3D human pose estimation 
benchmark datasets to systematically validate the superiority of the proposed model. 

Human3.6M Datasets: As the most representative benchmark in 3D human pose estimation, the 
Human3.6M Dataset [13] provides 3.6 million frames of multi-view motion captured data captured by 
four synchronized cameras at a 50 Hz sampling rate, covering 15 categories of daily activities performed 
by 11 subjects in indoor scenes. Following the standard experimental protocol, we adopt data from 
subjects (S1, S5, S6, S7, S8) for model training, and evaluate performance on two subjects (S9, S11). 
Two mainstream evaluation metrics are employed: Protocol 1 (MPJPE) measures absolute errors by 
computing the Euclidean distance (in millimeters) between predicted and ground-truth 3D joint 
coordinates; Protocol 2 (P-MPJPE) calculates relative errors after aligning predictions with ground truth 
via Procrustes analysis.  
MPI-INF-3DHP Dataset: The MPI-INF-3DHP Dataset [14] is a more challenging 3D human pose 
estimation benchmark, capturing 1.3 million frames of diverse poses from 8 subjects in indoor/outdoor 
hybrid scenes using 14 cameras. Aligned with settings in [11], [9], and [14], we utilize the Percentage of 
Correct Keypoints (PCK) under a 150 mm radius and the Area Under the Curve (AUC) as evaluation 
metrics. 
Table 2. presents experimental comparisons on the Human3.6M dataset using ground-truth 2D poses as network 
input. The symbol (*) indicates models utilizing temporal information. Best results are highlighted in bold. 
MPJPE(mm)( ↓) 
Dir. 
Disc. 
Eat. 
Greet. 
Phone. 
Photo. 
Pose. 
Purch. 
Sit. 
SitD. 
Smoke. 
Wait. 
WalkD. 
Walk 
WalkT. 
Avg. 
Liu et al [41] 
36.8 
40.3 
33.0 
36.3 
37.5 
45.0 
39.7 
34.9 
40.3 
47.7 
37.4 
38.5 
38.6 
29.6 
32.0 
37.8 
SRNet [7] 
35.9 
36.7 
29.3 
34.5 
36.0 
42.8 
37.7 
31.7 
40.1 
44.3 
35.8 
37.2 
36.2 
33.7 
34.0 
36.4 
PoseGTAC [44] 
37.2 
42.2 
32.6 
38.6 
38.0 
44.0 
40.7 
35.2 
41.0 
45.5 
38.2 
39.5 
38.2 
29.8 
33.0 
38.2 
GraphSH [35] 
35.8 
38.1 
31.0 
35.3 
35.8 
43.2 
37.3 
31.7 
38.4 
45.5 
35.4 
36.7 
36.8 
27.9 
30.7 
35.8 
GraFormer [36] 
32.0 
38.0 
30.4 
34.4 
34.7 
43.3 
35.2 
31.4 
38.0 
46.2 
34.2 
35.7 
36.1 
27.4 
30.6 
35.2 
PHGANet [45] 
32.4 
36.5 
30.1 
33.3 
36.3 
43.5 
36.1 
30.5 
37.5 
45.3 
33.8 
35.1 
35.3 
27.5 
30.2 
34.9 
DGFormer [8] 
31.5 
34.3 
28.2 
32.2 
31.3 
36.8 
37.0 
29.4 
34.9 
37.8 
31.8 
32.5 
33.0 
26.7 
28.9 
32.4 
GraphMLP [40] 
32.2 
38.2 
29.3 
33.4 
33.5 
38.1 
38.2 
31.7 
37.3 
38.5 
34.2 
36.1 
35.5 
28.0 
29.3 
34.2 
Ours 
30.9 
35.5 
27.2 
31.6 
31.7 
36.3 
36.4 
30.3 
36.6 
35.0 
31.4 
34.3 
32.5 
25.6 
26.2 
32.1 
4.2  Implementation details 
 Our method is implemented using PyTorch on a single NVIDIA RTX 3090 GPU. Core architectural 
parameters are configured as follows: the Transformer encoder comprises L = 6 stacked layers, each self-
attention layer contains h = 8 attention heads, and the feature embedding dimension is d = 512. During 
training, horizontal flipping data augmentation is applied to enhance model robustness, while the same 
strategy is synchronized in the test phase for result ensembling. The optimization process employs the 
Adam optimizer with an initial learning rate of 0.001 and an exponential decay scheduler (decay rate γ 
= 0.96), and is trained for 40 epochs. For 2D pose detection, both Human3.6M and MPI-INF-3DHP 
datasets utilize the Cascaded Pyramid Network (CPN) [30] as the base detector to ensure reliable 2D 
input features. 
4.3  Comparsion with state-of-the art 
Result On Human3.6M: As shown in Tables 1 and 2, when using 2D poses detected by CPN as 
input, our model outperforms existing methods on both MPJPE (48.1 mm) and P-MPJPE (38.3 mm) 
metrics. Compared to state-of-the-art graph transformer approaches, PoseGRAF achieves a reduction of 
10.6 mm in MPJPE over GraFormer[36] and 1.1 mm over GraphMLP[40]. Notably, PoseGRAF 
demonstrates superior 3D pose prediction accuracy in complex motion scenarios such as Phoning and 
Walking. Quantitative analysis reveals that the fusion of geometric features—joint positions, bone 

directions, and joint angles—significantly improves pose estimation accuracy, effectively enhancing 
geometric consistency between predictions and ground-truth annotations.  
Result on MPI-INF-3DHP: We further validate the generalization capability of our model 
PoseGRAF using the MPI-INF-3DHP dataset, which contains diverse pose variations. The model trained 
on Human3.6M is directly applied to regress 3D pose coordinates. As shown in Table 3, our method 
achieves state-of-the-art performance on both PCK and AUC metrics. These results demonstrate that the 
proposed model exhibits strong generalization and effectively adapts to unseen data. 
Table. 3. Results on MPI-INF-3DHP

## related_work
2.1  3D human pose estimation 
In recent years, deep learning has significantly advanced the field of monocular 3D human pose 
estimation. Existing methods can be broadly categorized into two paradigms. The first is direct regression 
approaches[5], [6], [15] which utilize end-to-end CNN architectures to directly predict 3D joint positions 
or reconstruct human meshes from raw RGB images. While these methods leverage rich visual 
information, they are often sensitive to environmental variations and computationally expensive, making 
them less suitable for real-time processing in dynamic scenes. The second category adopts a two-
stage ’image-to-2D-to-3D’ pipeline. Chen et al. [16], perform matching and retrieval from a predefined 
3D. pose library, achieving computational efficiency but limited by pose diversity. Martinez et al. [17] 
propose a fully connected residual network that regresses 3D joint positions from 2D keypoints, 
significantly improving accuracy and benefiting from reliable feature support provided by a detector 
pretrained on a large‑scale 2D dataset. Subsequent improvements, including hierarchical joint prediction 
[18], keypoint refinement[12], and viewpoint-invariant constraints[19], further enhanced model 
performance. Although current data augmentation techniques[20] have made significant progress in 

predictive accuracy, their generalization ability to complex real-world scenarios remains insufficient. 
2.2  Graph-Based Learning Methods 
Graph Convolutional Networks (GCNs) have demonstrated strong performance in monocular 2D-
to-3D pose lifting tasks by modeling the topological structure of the human skeleton, where joints are 
treated as nodes and bones as edges. GCNs can effectively capture spatial dependencies through graph-
based convolutional operations. In existing works, Ci et al. [21] proposed a locally connected network to 
enhance feature representation, SemGCN[22] incorporated joint semantic relationships to refine 
predictions, and MGCN[23] introduced weight modulation to improve accuracy. However, these 
approaches rely on static adjacency matrices to define edge weights, making it difficult to model dynamic 
skeletal interactions. To address this, Zhou et al. [24] proposed Hyperformer, which leverages hypergraph 
self-attention (HyperSA) to embed skeletal structures into a Transformer framework. While this 
improves skeletal action recognition, it still falls short in modeling high-order interactions such as bone 
direction dynamics.. 
2.3  Skeletal Geometry-Aware Methods 
Traditional 2D-to-3D pose estimation methods typically regress 3D coordinates directly from 2D 
joint coordinates. Ma et al.[25] integrated bone length constraints within the GCN framework to mitigate 
depth ambiguity, while Azizi et al. [26] encoded poses through inter-segment angles to achieve finer 
skeletal representations. Hu et al. [27] employed a directional graph approach to explicitly model joint-
bone relationships, and Yu et al. [28] optimized estimations through GCN-based global-local feature 
integration. Though these methods emphasize the importance of bone directions and angles, most treat 
such constraints as auxiliary signals rather than directly incorporating them into graph structures. Sun et 
al. [10] regressed joint relative displacements through skeletal representations, and Kanazawa et al. [29] 
incorporated skeletal constraints in 3D mesh reconstruction, yet neither dynamically leveraged angle 
information. We propose a novel approach: constructing a weighted graph using skeletal orientation 
angles as edge weights, applied to Graph Convolutional Networks (GCN) processing to achieve higher-
accuracy dynamic modeling. 
Fig.1. (a) Dance pose. (b) 2D joint graph. (c) Directed weighted bone graph. (d) indicates angles between adjacent 
bone directions.

## conclusion
This paper proposes PoseGRAF, a geometry-enhanced graph learning framework, addressing the 
limitations of monocular 3D human pose estimation caused by over-reliance on 2D joint coordinates and 
insufficient utilization of bone directions and joint angles. By constructing a geometry-enhanced graph 
structure that unifies the encoding of bone directions and joint angles, integrating graph convolution 
modules to capture skeletal spatial correlations, and introducing an attention-driven dynamic feature 
fusion mechanism to adaptively consolidate global joint positions with local geometric features, our 
method effectively mitigates prediction bias in occluded scenarios. On the Human3.6M and MPI-INF-
3DHP datasets, PoseGRAF significantly outperforms state-of-the-art methods and demonstrates robust 
generalization capabilities in in-the-wild video testing. Future work will explore temporal kinematic 
constraints and lightweight deployment to advance real-time applications such as medical rehabilitation.  

Reference
[1] Du, S., Yuan, Z., & Ikenaga, T. (2024). *Kinematics-aware spatial-temporal feature 
transform for 3D human pose estimation*. Pattern Recognition, 150, 110316.  
[2] Islam, M. M., Nooruddin, S., Karray, F., & Muhammad, G. (2023). Multi-level feature fusion 
for multimodal human activity recognition in Internet of Healthcare Things. Information 
Fusion, 94, 17–31.  
[3] Xu, C., He, J., Zhang, X., Yao, C., & Tseng, P. H. (2018). Geometrical kinematic modeling 
on human motion using method of multi-sensor fusion. Information Fusion, 41, 243–254.   
[4] Tripathi, A., Prathosh, A. P., Muthukrishnan, S. P., & Kumar, L. (2023). SurfMyoAiR: A 
surface electromyography-based framework for airwriting recognition. IEEE Transactions 
on Instrumentation and Measurement, 72, 1–12.  
[5] Xu, X., Liu, L., & Yan, S. (2023). SMPlER: Taming Transformers for Monocular 3D Human 
Shape and Pose Estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 
46(5), 3275–3289.  
[6] Moon, G., Chang, J. Y., & Lee, K. M. (2019). *Camera Distance-Aware Top-Down Approach 
for 3D Multi-Person Pose Estimation from a Single RGB Image*. In Proceedings of the 
IEEE/CVF International Conference on Computer Vision (pp. 10133–10142).   
[7] Zeng, A., Sun, X., Huang, F., Liu, M., Xu, Q., & Lin, S. (2020). *SRNet: Improving 
generalization in 3D human pose estimation with a split-and-recombine approach*. 
In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 
2020, Proceedings, Part XIV 16 (pp. 507–523).   
[8] Chen, Z., Dai, J., Bai, J., & Pan, J. (2024). DGFormer: Dynamic graph transformer for 3D 
human pose estimation. Pattern Recognition, 152, 110446.   
[9] Zheng, C., Zhu, S., Mendieta, M., Yang, T., Chen, C., & Ding, Z. (2021). 3D Human Pose 
Estimation with Spatial and Temporal Transformers. In Proceedings of the IEEE/CVF 
International Conference on Computer Vision (pp. 11656–11665).  
[10] Sun, X., Shang, J., Liang, S., & Wei, Y. (2017). Compositional Human Pose Regression. 
In Proceedings of the IEEE International Conference on Computer Vision (pp. 2602–2611).   
[11] Chen, T., Fang, C., Shen, X., Zhu, Y., Chen, Z., & Luo, J. (2021). *Anatomy-aware 3D 
human pose estimation with bone-based pose decomposition*. IEEE Transactions on 
Circuits and Systems for Video Technology, 32(1), 198–209.   
[12] Xu, J., Yu, Z., Ni, B., Yang, J., Yang, X., & Zhang, W. (2020). Deep kinematics analysis for 
monocular 3D human pose estimation. In Proceedings of the IEEE/CVF Conference on 
Computer Vision and Pattern Recognition (pp. 899–908).   
[13] Ionescu, C., Papava, D., Olaru, V., & Sminchisescu, C. (2013). Human3.6M: Large Scale 
Datasets and Predictive Methods for 3D Human Sensing in Natural Environments. IEEE 
Transactions on Pattern Analysis and Machine Intelligence, 36(7), 1325–1339.   
[14] Mehta, D., Rhodin, H., Casas, D., Fua, P., Sotnychenko, O., Xu, W., & Theobalt, C. 
(2017). Monocular 3D Human Pose Estimation in the Wild Using Improved CNN 
Supervision. In Proceedings of the International Conference on 3D Vision (3DV) (pp. 506–
516).  
[15] Xu, X., & Loy, C. C. (2021). 3D human texture estimation from a single image with 
transformers. In Proceedings of the IEEE/CVF International Conference on Computer 

Vision (pp. 13849–13858).  
[16] Chen, C. H., & Ramanan, D. (2017). *3D Human Pose Estimation = 2D Pose Estimation + 
Matching*. In Proceedings of the IEEE Conference on Computer Vision and Pattern 
Recognition (pp. 7035–7043).   
[17] Martinez, J., Hossain, R., Romero, J., & Little, J. J. (2017). A simple yet effective baseline 
for 3D human pose estimation. In Proceedings of the IEEE International Conference on 
Computer Vision (pp. 2640–2649).   
[18] Lee, K., Lee, I., & Lee, S. (2018). Propagating LSTM: 3D Pose Estimation Based on Joint 
Interdependency. In Proceedings of the European Conference on Computer Vision 
(ECCV) (pp. 119–135).  
[19] Wei, G., Lan, C., Zeng, W., & Chen, Z. (2019). View Invariant 3D Human Pose Estimation. 
IEEE Transactions on Circuits and Systems for Video Technology, 30(12), 4601–4610.   
[20] Gong, K., Zhang, J., & Feng, J. (2021). PoseAug: A Differentiable Pose Augmentation 
Framework for 3D Human Pose Estimation. In Proceedings of the IEEE/CVF Conference 
on Computer Vision and Pattern Recognition (pp. 8575–8584).   
[21] Ci, H., Wang, C., Ma, X., & Wang, Y. (2019). Optimizing Network Structure for 3D Human 
Pose Estimation. In Proceedings of the IEEE/CVF International Conference on Computer 
Vision (pp. 2262–2271).   
[22] Zhao, L., Peng, X., Tian, Y., Kapadia, M., & Metaxas, D. N. (2019). Semantic Graph 
Convolutional Networks for 3D Human Pose Regression. In Proceedings of the IEEE/CVF 
Conference on Computer Vision and Pattern Recognition (pp. 3425–3435).   
[23] Zou, Z., & Tang, W. (2021). Modulated Graph Convolutional Network for 3D Human Pose 
Estimation. In Proceedings of the IEEE/CVF International Conference on Computer 
Vision (pp. 11477–11487). https://doi.org/10.1109/ICCV48922.2021.01128 
[24] Zhou, Y., Cheng, Z. Q., Li, C., Fang, Y., Geng, Y., Xie, X., & Keuper, M. (2022). Hypergraph 
Transformer for Skeleton-Based Action Recognition. arXiv preprint arXiv:2211.09590.  
[25] Ma, X., Su, J., Wang, C., Ci, H., & Wang, Y. (2021). Context Modeling in 3D Human Pose 
Estimation: A Unified Perspective. In Proceedings of the IEEE/CVF Conference on 
Computer Vision and Pattern Recognition (pp. 6238–6247).  
[26] Azizi, N., Possegger, H., Rodolà, E., & Bischof, H. (2022). 3D Human Pose Estimation 
Using Möbius Graph Convolutional Networks. In Proceedings of the European Conference 
on Computer Vision (ECCV) (pp. 160–178).   
[27] Hu, W., Zhang, C., Zhan, F., Zhang, L., & Wong, T. T. (2021). Conditional Directed Graph 
Convolution for 3D Human Pose Estimation. In Proceedings of the 29th ACM International 
Conference on Multimedia (pp. 602–611).   
[28] Yu, B. X., Zhang, Z., Liu, Y., Zhong, S. H., Liu, Y., & Chen, C. W. (2023). *GLA-GCN: 
Global-Local Adaptive Graph Convolutional Network for 3D Human Pose Estimation from 
Monocular Video*. In Proceedings of the IEEE/CVF International Conference on Computer 
Vision (pp. 8818–8829).   
[29] Kanazawa, A., Black, M. J., Jacobs, D. W., & Malik, J. (2018). End-to-End Recovery of 
Human Shape and Pose. In Proceedings of the IEEE Conference on Computer Vision and 
Pattern Recognition (pp. 7122–7131).   
[30] Chen, Y., Wang, Z., Peng, Y., Zhang, Z., Yu, G., & Sun, J. (2018). Cascaded Pyramid 
Network for Multi-Person Pose Estimation. In Proceedings of the IEEE Conference on 

Computer Vision and Pattern Recognition (pp. 7103–7112).   
[31] Chen, M., Lin, M., Li, K., Shen, Y., Wu, Y., Chao, F., & Ji, R. (2023). CF-ViT: A General 
Coarse-to-Fine Method for Vision Transformer. In Proceedings of the AAAI Conference on 
Artificial Intelligence, 37(6), 7042–7052.   
[32] An, X., Zhao, L., Gong, C., Wang, N., Wang, D., & Yang, J. (2024). Sharpose: Sparse high-
resolution representation for human pose estimation. In Proceedings of the AAAI Conference 
on Artificial Intelligence, 38(2), 691–699.   
[33] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & 
Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing 
Systems, 30.   
[34] Zhang, Z., et al. (2021). Motif-based graph self-supervised learning for molecular property 
prediction. Advances in Neural Information Processing Systems, 34, 15870–15882. 
[35] Xu, T., & Takano, W. (2021). Graph Stacked Hourglass Networks for 3D Human Pose 
Estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern 
Recognition (pp. 16105–16114).   
[36] Zhao, W., Wang, W., & Tian, Y. (2022). *Graformer: Graph-Oriented Transformer for 3D 
Pose Estimation*. In Proceedings of the IEEE/CVF Conference on Computer Vision and 
Pattern Recognition (pp. 20438–20447).   
[37] Zhang, J., Tu, Z., Yang, J., Chen, Y., & Y